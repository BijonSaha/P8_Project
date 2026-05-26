import pyvisa
import numpy as np
import csv
import time
from pathlib import Path

# ── Change these two before every session ─────────────────────────────────────
PARTICIPANT_ID = "P018"           # P01, P02 ... P10
POSITION       = "left_plus3cm"  # see position list below

# ── Available positions ───────────────────────────────────────────────────────
# left_nominal   → left forearm, nominal mark
# right_nominal  → right forearm, nominal mark
# left_plus3cm   → left forearm, 3 cm toward wrist
# left_minus3cm  → left forearm, 3 cm toward elbow
# right_plus3cm  → right forearm, 3 cm toward wrist
# right_minus3cm → right forearm, 3 cm toward elbow

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/Users/bijonsaha/Documents/Aalborg University 2nd Semester/Semester Project/RMGDataset")

# ── Measurement protocol ──────────────────────────────────────────────────────
GESTURES = [
    "fist", "grasp", "open_palm", "pinch", "point",
    "relaxed", "shaka", "thumbs_up", "twist",
    "wrist_down", "wrist_up"
]

N_REPS    = 12
HOLD_TIME = 2.0
REST_TIME = 2.0

# ── Find VNA ──────────────────────────────────────────────────────────────────  ← here
def find_vna():
    rm        = pyvisa.ResourceManager()
    resources = rm.list_resources()

    for resource in resources:
        if "0x0AAD" in resource and "0x019B" in resource:
            print(f"VNA found: {resource}")
            return resource

    print("VNA not found. Check USB connection.")
    return None



# ── VNA settings ──────────────────────────────────────────────────────────────
VISA_ADDRESS  = "USB0::0x0AAD::0x019B::101989::INSTR"
F_START       = 2.5e9
F_STOP        = 3.5e9
N_POINTS      = 300
TX_POWER      = -20
IF_BW         = 1000
S11_THRESHOLD = -10   # dB

S_PARAMS = ["S11", "S22", "S12", "S21"]

# ── Connect ───────────────────────────────────────────────────────────────────
def connect_vna():
    rm  = pyvisa.ResourceManager()
    vna = rm.open_resource(VISA_ADDRESS)
    vna.timeout           = 60000
    vna.write_termination = "\n"
    vna.read_termination  = "\n"
    idn = vna.query("*IDN?")
    print(f"Connected : {idn.strip()}")
    return vna

# ── Configure ─────────────────────────────────────────────────────────────────
def configure_vna(vna):
    print("Configuring VNA...")

    vna.write("*RST")
    vna.query("*OPC?")
    vna.write("*CLS")
    time.sleep(3)

    vna.write(f"SENS1:FREQ:STAR {F_START}")
    vna.query("*OPC?")
    vna.write(f"SENS1:FREQ:STOP {F_STOP}")
    vna.query("*OPC?")
    vna.write(f"SENS1:SWE:POIN {N_POINTS}")
    vna.query("*OPC?")
    vna.write("SENS1:SWE:TYPE LIN")
    vna.query("*OPC?")
    vna.write(f"SOUR1:POW {TX_POWER}")
    vna.query("*OPC?")
    vna.write(f"SENS1:BWID {IF_BW}")
    vna.query("*OPC?")
    vna.write("FORM ASCII")
    vna.query("*OPC?")
    vna.write("SENS1:SWE:MODE CONT")
    vna.query("*OPC?")

    print(f"  Frequency : {F_START/1e9} – {F_STOP/1e9} GHz")
    print(f"  Points    : {N_POINTS}")
    print(f"  TX power  : {TX_POWER} dBm")
    print(f"  IF BW     : {IF_BW} Hz")

# ── Verify settings ───────────────────────────────────────────────────────────
def verify_vna_settings(vna):
    print("\nVerifying VNA settings...")

    actual_start  = float(vna.query("SENS1:FREQ:STAR?"))
    actual_stop   = float(vna.query("SENS1:FREQ:STOP?"))
    actual_points = int(float(vna.query("SENS1:SWE:POIN?")))

    print(f"  Start  : {actual_start/1e9:.4f} GHz "
          f"(expected {F_START/1e9:.4f})")
    print(f"  Stop   : {actual_stop/1e9:.4f} GHz "
          f"(expected {F_STOP/1e9:.4f})")
    print(f"  Points : {actual_points} (expected {N_POINTS})")

    if abs(actual_start - F_START) > 1e6:
        vna.write(f"SENS1:FREQ:STAR {F_START}")
        vna.query("*OPC?")
    if abs(actual_stop - F_STOP) > 1e6:
        vna.write(f"SENS1:FREQ:STOP {F_STOP}")
        vna.query("*OPC?")
    if actual_points != N_POINTS:
        vna.write(f"SENS1:SWE:POIN {N_POINTS}")
        vna.query("*OPC?")

    # re-verify after forcing
    actual_start  = float(vna.query("SENS1:FREQ:STAR?"))
    actual_stop   = float(vna.query("SENS1:FREQ:STOP?"))
    actual_points = int(float(vna.query("SENS1:SWE:POIN?")))

    print(f"\n  Final start  : {actual_start/1e9:.4f} GHz")
    print(f"  Final stop   : {actual_stop/1e9:.4f} GHz")
    print(f"  Final points : {actual_points}")

    if actual_points != N_POINTS:
        print("ABORT: could not set correct point count.")
        return False
    if abs(actual_stop - F_STOP) > 1e6:
        print("ABORT: could not set correct stop frequency.")
        return False

    print("  Settings verified ✓")
    return True

# ── Error checker ─────────────────────────────────────────────────────────────
def check_errors(vna):
    while True:
        err = vna.query("SYST:ERR?")
        if err.strip().startswith("0") or "No error" in err:
            break
        print(f"  VNA ERROR: {err.strip()}")

# ── S11 coupling check ────────────────────────────────────────────────────────
def check_s11_coupling(vna):
    print(f"\nChecking antenna coupling "
          f"(threshold: {S11_THRESHOLD} dB)...")

    vna.write("*CLS")
    vna.write("CALC1:PAR:DEL:ALL")
    vna.query("*OPC?")
    vna.write("CALC1:PAR:SDEF 'CoupCheck', 'S11'")
    vna.query("*OPC?")
    vna.write("CALC1:PAR:SEL 'CoupCheck'")
    vna.query("*OPC?")
    vna.write("DISP:WIND1:STAT ON")
    vna.query("*OPC?")
    vna.write("DISP:WIND1:TRAC1:FEED 'CoupCheck'")
    vna.query("*OPC?")
    vna.write("SENS1:SWE:MODE CONT")
    vna.query("*OPC?")
    vna.write("INIT1:IMM")
    vna.query("*OPC?")
    time.sleep(0.1)

    vna.write("CALC1:FORM MLOG")
    vna.query("*OPC?")
    raw    = vna.query("CALC1:DATA? SDAT")
    values = [float(v) for v in raw.strip().split(",")]
    reals  = values[0::2]
    imags  = values[1::2]

    mags_db = [20 * np.log10(np.sqrt(r**2 + i**2) + 1e-12)
               for r, i in zip(reals, imags)]
    min_s11 = min(mags_db)

    print(f"  Min S11 : {min_s11:.1f} dB")

    if min_s11 <= S11_THRESHOLD:
        print(f"  Coupling : ✓ Good")
        return True
    else:
        print(f"  Coupling : ✗ Poor — recheck armband contact")
        return False

# ── Output folder ─────────────────────────────────────────────────────────────
def make_output_dir(participant, gesture, position):
    path = OUTPUT_DIR / participant / gesture / position
    path.mkdir(parents=True, exist_ok=True)
    return path

# ── Measure one rep ───────────────────────────────────────────────────────────
def measure_one_rep(vna, participant, gesture, position, rep):
    for sp in S_PARAMS:
        port_rx = int(sp[1])
        port_tx = int(sp[2])

        vna.write("*CLS")
        vna.write("CALC1:PAR:DEL:ALL")
        vna.query("*OPC?")
        vna.write(f"CALC1:PAR:SDEF 'Trc1', 'S{port_rx}{port_tx}'")
        vna.query("*OPC?")
        vna.write("CALC1:PAR:SEL 'Trc1'")
        vna.query("*OPC?")
        vna.write("DISP:WIND1:STAT ON")
        vna.query("*OPC?")
        vna.write("DISP:WIND1:TRAC1:FEED 'Trc1'")
        vna.query("*OPC?")
        vna.write("SENS1:SWE:MODE CONT")
        vna.query("*OPC?")
        vna.write("INIT1:IMM")
        vna.query("*OPC?")
        time.sleep(0.1)

        # read frequency
        freq_raw = vna.query("CALC1:DATA:STIM?")
        freqs    = [float(f) for f in freq_raw.strip().split(",")]

        # read complex data
        vna.write("CALC1:FORM MLOG")
        vna.query("*OPC?")
        raw    = vna.query("CALC1:DATA? SDAT")
        values = [float(v) for v in raw.strip().split(",")]
        reals  = values[0::2]
        imags  = values[1::2]

        mags   = [np.sqrt(r**2 + i**2)
                  for r, i in zip(reals, imags)]
        phases = [np.degrees(np.arctan2(i, r))
                  for r, i in zip(reals, imags)]

        if len(freqs) != N_POINTS:
            print(f"  WARNING: {sp} — {len(freqs)} points "
                  f"(expected {N_POINTS})")

        # save CSV
        folder   = make_output_dir(participant, gesture, position)
        filename = (f"{participant}_{gesture}_{position}"
                    f"_{sp}_iter{rep}.csv")
        filepath = folder / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency(Hz)", "Magnitude", "Phase(deg)"])
            for freq, mag, phase in zip(freqs, mags, phases):
                writer.writerow([freq, mag, phase])

    print(f"    ✓ iter{rep:02d} — "
          f"{freqs[0]/1e9:.3f}–{freqs[-1]/1e9:.3f} GHz — "
          f"{len(mags)} pts")

# ── Collection loop ───────────────────────────────────────────────────────────
def collect_session(vna, participant, position):
    total = len(GESTURES) * N_REPS * len(S_PARAMS)
    print(f"\n{'='*60}")
    print(f"Participant : {participant}")
    print(f"Position    : {position}")
    print(f"Gestures    : {len(GESTURES)}")
    print(f"Reps        : {N_REPS}")
    print(f"Total CSVs  : {total}")
    print(f"{'='*60}")

    for gesture in GESTURES:
        print(f"\n  Gesture: {gesture.upper()}")
        input(f"  Press ENTER when ready...")

        for rep in range(N_REPS):
            print(f"    Rep {rep+1:02d}/{N_REPS} — hold gesture...",
                  end=" ", flush=True)
            time.sleep(HOLD_TIME)
            measure_one_rep(vna, participant, gesture, position, rep)
            check_errors(vna)
            if rep < N_REPS - 1:
                time.sleep(REST_TIME)

        print(f"  ✓ {gesture} complete.")

    print(f"\n✓ Session complete — {position} done.")

# ── Verification ──────────────────────────────────────────────────────────────
def verify_output(participant, position):
    print(f"\n{'='*60}")
    print("Output verification:")

    csv_files = list(
        (OUTPUT_DIR / participant).rglob(f"*{position}*.csv")
    )
    expected = len(GESTURES) * N_REPS * len(S_PARAMS)

    print(f"  Position : {position}")
    print(f"  Expected : {expected} CSVs")
    print(f"  Found    : {len(csv_files)} CSVs")

    if csv_files:
        sample = csv_files[0]
        with open(sample) as f:
            rows = list(csv.reader(f))
        print(f"  Sample   : {sample.name}")
        print(f"  Rows     : {len(rows)-1} (expected {N_POINTS})")
        print(f"  First    : {rows[1]}")
        print(f"  Last     : {rows[-1]}")

    status = "✓ COMPLETE" if len(csv_files) == expected \
             else f"✗ INCOMPLETE — {expected-len(csv_files)} missing"
    print(f"  Status   : {status}")
    print(f"{'='*60}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RMG Data Collection")
    print(f"{'='*60}")
    print(f"Participant : {PARTICIPANT_ID}")
    print(f"Position    : {POSITION}")
    print(f"Gestures    : {len(GESTURES)}")
    print(f"Reps        : {N_REPS}")
    print(f"S-params    : {S_PARAMS}")
    print(f"Total CSVs  : {len(GESTURES) * N_REPS * len(S_PARAMS)}")
    print(f"{'='*60}")

    vna = connect_vna()
    configure_vna(vna)

    if not verify_vna_settings(vna):
        vna.close()
        exit()

    coupling_ok = check_s11_coupling(vna)
    if not coupling_ok:
        proceed = input("Coupling poor. Proceed anyway? (y/n): ")
        if proceed.lower() != "y":
            vna.close()
            exit()

    collect_session(vna, PARTICIPANT_ID, POSITION)

    vna.write("SENS1:SWE:MODE CONT")
    vna.query("*OPC?")
    vna.close()

    verify_output(PARTICIPANT_ID, POSITION)

    print(f"\nData saved to:")
    print(f"  {OUTPUT_DIR / PARTICIPANT_ID}")