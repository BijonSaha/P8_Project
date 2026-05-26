"""
RMG Delta Plot Generator
=========================
Generates delta (differential from relaxed baseline) plots for N stratified
samples across all 3 participant groups.

Output structure:
  Results/delta_plots/
    Participant_1/
      right_nominal/
        fist/
          P01_right_nominal_iter3_delta_heatmap.png
          P01_right_nominal_iter3_delta_freq.png
        ...
    Participant_2/ ...
    Participant_3/ ...

Two plot types per sample:
  delta_heatmap — RdBu diverging heatmap of ΔMagnitude + ΔPhase (all S-params)
  delta_freq    — line plot of ΔS12 and ΔS21 magnitude vs frequency
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent.parent / "RMGDataset"
OUTPUT_DIR   = Path(__file__).parent.parent / "Results" / "delta_plots"

N_SAMPLES    = 198   # 11 gestures × 6 positions × 3 groups = 198 cells → 1 per cell
RANDOM_SEED  = 42
DPI          = 150          # lighter than original 300 for faster generation

S_PARAM_ORDER = ["S11", "S12", "S21", "S22"]

ALL_GESTURES = [
    "fist", "grasp", "open_palm", "pinch", "point",
    "relaxed", "shaka", "thumbs_up", "twist", "wrist_down", "wrist_up",
]

# ── PARTICIPANT → GROUP MAPPING ───────────────────────────────────────────────
# Built once from the folder tree; avoids hardcoding P-ID ranges.
def build_group_map(dataset_root: Path) -> dict[str, str]:
    mapping = {}
    for group_dir in sorted(dataset_root.iterdir()):
        if not group_dir.is_dir() or not group_dir.name.startswith("Participant_"):
            continue
        for p_dir in sorted(group_dir.iterdir()):
            if p_dir.is_dir() and p_dir.name.startswith("P") and p_dir.name[1:].isdigit():
                mapping[p_dir.name] = group_dir.name
    return mapping


# ── 1. CSV LOADING ─────────────────────────────────────────────────────────────
def load_csv(filepath: Path):
    """Linear magnitude → dB. Returns (freq_hz, mag_db, phase_deg)."""
    df     = pd.read_csv(filepath)
    mag_db = 20.0 * np.log10(np.clip(df["Magnitude"].values, 1e-12, None))
    return df["Frequency(Hz)"].values, mag_db, df["Phase(deg)"].values


def load_one_iteration(gesture_dir: Path, participant: str,
                       s_params: list, iteration: int):
    """
    Load one gesture iteration. CSV files live directly in gesture_dir.
    Naming: {participant}_{gesture}_{position}_{s_param}_iter{N}.csv
    Returns (freq[n_freq], mag[n_sp, n_freq], phase[n_sp, n_freq]) or Nones.
    """
    gesture  = gesture_dir.name
    position = gesture_dir.parent.name

    mag = phase = freq = None
    for si, sp in enumerate(s_params):
        fname = f"{participant}_{gesture}_{position}_{sp}_iter{iteration}.csv"
        fpath = gesture_dir / fname
        if not fpath.exists():
            return None, None, None
        f, m, p = load_csv(fpath)
        if mag is None:
            n_freq = len(m)
            mag    = np.full((len(s_params), n_freq), np.nan)
            phase  = np.full((len(s_params), n_freq), np.nan)
            freq   = f
        mag[si]   = m
        phase[si] = p
    return freq, mag, phase


def load_relaxed_baseline(dataset_root: Path, participant: str,
                          position: str, s_params: list):
    """
    Mean of all available relaxed iterations for this participant + position.
    Returns (mean_mag[n_sp, n_freq], mean_phase[n_sp, n_freq], freq[n_freq]).
    """
    relaxed_dirs = [
        d for d in dataset_root.rglob("relaxed")
        if d.is_dir()
        and any(part == participant for part in d.parts)
        and d.parent.name == position
    ]
    if not relaxed_dirs:
        return None, None, None

    relaxed_dir = relaxed_dirs[0]
    all_mags, all_phases, freq_ref = [], [], None

    for it in range(12):
        freq, mag, phase = load_one_iteration(relaxed_dir, participant, s_params, it)
        if mag is not None:
            all_mags.append(mag)
            all_phases.append(phase)
            if freq_ref is None:
                freq_ref = freq

    if not all_mags:
        return None, None, None

    mean_mag   = np.mean(all_mags, axis=0)
    phases_rad = np.deg2rad(np.array(all_phases))
    mean_phase = np.rad2deg(np.angle(np.mean(np.exp(1j * phases_rad), axis=0)))
    return mean_mag, mean_phase, freq_ref


# ── 2. DISCOVER SAMPLES ────────────────────────────────────────────────────────
def discover_all_samples(dataset_root: Path, gestures: list,
                         group_map: dict) -> list[dict]:
    all_samples = []
    for gesture in gestures:
        for gesture_dir in sorted(dataset_root.rglob(gesture)):
            if not gesture_dir.is_dir():
                continue
            position    = gesture_dir.parent.name
            participant = next(
                (p for p in gesture_dir.parts
                 if p.startswith("P") and p[1:].isdigit()),
                None,
            )
            if not participant:
                continue
            csv_files = sorted(gesture_dir.glob("*.csv"))
            if not csv_files:
                continue
            parsed = [_parse_fname(f) for f in csv_files]
            parsed = [p for p in parsed if p]
            if not parsed:
                continue
            s_params_found = sorted(set(p["s_param"] for p in parsed))
            s_params       = [sp for sp in S_PARAM_ORDER if sp in s_params_found]
            if not s_params:
                continue
            for it in sorted(set(p["iteration"] for p in parsed)):
                all_samples.append({
                    "gesture":     gesture,
                    "participant": participant,
                    "position":    position,
                    "iteration":   it,
                    "s_params":    s_params,
                    "gesture_dir": gesture_dir,
                    "group":       group_map.get(participant, "Unknown"),
                })
    print(f"  Discovered {len(all_samples)} total samples")
    return all_samples


def _parse_fname(filepath: Path) -> dict | None:
    parts = filepath.stem.split("_")
    try:
        return {
            "s_param":   parts[-2].upper(),
            "iteration": int(parts[-1].replace("iter", "")),
        }
    except (IndexError, ValueError):
        return None


# ── 3. STRATIFIED SAMPLING ────────────────────────────────────────────────────
def stratified_sample(all_samples: list, n_total: int, seed: int = 42) -> list[dict]:
    """
    Three-level stratified sampling: gesture × position × group.
    Each (gesture, position, group) cell gets floor(n_total / n_cells) samples,
    with remainder distributed evenly. Guarantees at least 1 per cell when
    n_total >= n_cells (198 for 11 × 6 × 3).
    """
    rng       = random.Random(seed)
    gestures  = sorted(set(s["gesture"]  for s in all_samples))
    positions = sorted(set(s["position"] for s in all_samples))
    groups    = sorted(set(s["group"]    for s in all_samples))

    # Build (gesture, position, group) → sample pool
    by_cell: dict[tuple, list] = defaultdict(list)
    for s in all_samples:
        by_cell[(s["gesture"], s["position"], s["group"])].append(s)

    cells    = sorted(by_cell.keys())
    n_cells  = len(cells)
    base     = n_total // n_cells
    n_extras = n_total  % n_cells
    extra_cells = set(rng.sample(cells, n_extras))

    selected = []
    for cell in cells:
        pool   = by_cell[cell]
        n_pick = base + (1 if cell in extra_cells else 0)
        n_pick = min(n_pick, len(pool))
        if n_pick <= 0:
            continue
        selected.extend(rng.sample(pool, n_pick))

    rng.shuffle(selected)

    g_counts   = Counter(s["gesture"]  for s in selected)
    pos_counts = Counter(s["position"] for s in selected)
    grp_counts = Counter(s["group"]    for s in selected)

    print(f"\n  Gesture distribution:  "
          f"{min(g_counts.values())}–{max(g_counts.values())} per gesture")
    print(f"  Position distribution: "
          f"{min(pos_counts.values())}–{max(pos_counts.values())} per position")
    print(f"  Group distribution:    "
          f"{min(grp_counts.values())}–{max(grp_counts.values())} per group")
    print(f"  Total selected: {len(selected)}  (cells: {n_cells})")
    return selected


# ── 4. DELTA COMPUTATION ──────────────────────────────────────────────────────
def compute_delta(mag_iter, phase_iter, base_mag, base_phase):
    delta_mag   = mag_iter - base_mag
    raw_diff    = phase_iter - base_phase
    delta_phase = (raw_diff + 180) % 360 - 180
    return delta_mag, delta_phase


# ── 5a. DELTA HEATMAP PLOT ────────────────────────────────────────────────────
def plot_delta_heatmap(delta_mag, delta_phase, s_params,
                       freq_hz, title, save_path):
    n_sp     = len(s_params)
    freq_ghz = freq_hz / 1e9
    n_freq   = delta_mag.shape[1]

    abs_max = float(np.percentile(np.abs(delta_mag), 99))
    abs_max = max(abs_max, 0.5)

    cmap_mag   = plt.cm.RdBu_r
    cmap_phase = plt.cm.coolwarm

    rgba_mag   = np.zeros((n_sp, n_freq, 4))
    rgba_phase = np.zeros((n_sp, n_freq, 4))
    mag_labels = []

    for i, sp in enumerate(s_params):
        norm_mag      = np.clip((delta_mag[i]   + abs_max) / (2 * abs_max), 0, 1)
        norm_phase    = np.clip((delta_phase[i] + 180)     / 360,           0, 1)
        rgba_mag[i]   = cmap_mag(norm_mag)
        rgba_phase[i] = cmap_phase(norm_phase)
        mag_labels.append(f"{sp}\n[±{abs_max:.1f} dB]")

    fig, axes = plt.subplots(
        1, 4, figsize=(14, 2.2 * n_sp),
        gridspec_kw={"width_ratios": [12, 0.5, 12, 0.5], "wspace": 0.06},
    )
    ax_mag, ax_cb1, ax_phase, ax_cb2 = axes
    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.02)

    extent = [freq_ghz[0], freq_ghz[-1], n_sp - 0.5, -0.5]

    ax_mag.imshow(rgba_mag, aspect="auto", interpolation="nearest", extent=extent)
    ax_mag.set_title("ΔMagnitude vs Relaxed  (RdBu)\nRed = more  |  Blue = less",
                     fontsize=8, pad=4)
    ax_mag.set_xlabel("Frequency (GHz)", fontsize=9)
    ax_mag.set_ylabel("S-parameter", fontsize=9)
    ax_mag.set_yticks(range(n_sp))
    ax_mag.set_yticklabels(mag_labels, fontsize=8)
    for row in range(1, n_sp):
        ax_mag.axhline(row - 0.5, color="white", lw=0.8, alpha=0.6)
    ax_mag.text(0.99, 0.01, "white = no change",
                transform=ax_mag.transAxes, fontsize=6.5,
                ha="right", va="bottom", color="grey")

    sm1 = plt.cm.ScalarMappable(cmap=cmap_mag,
                                  norm=plt.Normalize(-abs_max, abs_max))
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, cax=ax_cb1)
    cb1.set_label("ΔdB", fontsize=7)
    cb1.set_ticks([-abs_max, 0, abs_max])
    cb1.set_ticklabels([f"-{abs_max:.1f}", "0", f"+{abs_max:.1f}"], fontsize=6)

    ax_phase.imshow(rgba_phase, aspect="auto", interpolation="nearest", extent=extent)
    ax_phase.set_title("ΔPhase vs Relaxed  (coolwarm)\nRed = leads  |  Blue = lags",
                       fontsize=8, pad=4)
    ax_phase.set_xlabel("Frequency (GHz)", fontsize=9)
    ax_phase.set_yticks(range(n_sp))
    ax_phase.set_yticklabels(s_params, fontsize=8)
    for row in range(1, n_sp):
        ax_phase.axhline(row - 0.5, color="white", lw=0.8, alpha=0.6)

    sm2 = plt.cm.ScalarMappable(cmap=cmap_phase, norm=plt.Normalize(-180, 180))
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, cax=ax_cb2)
    cb2.set_label("Δ°", fontsize=7)
    cb2.set_ticks([-180, -90, 0, 90, 180])
    cb2.set_ticklabels(["-180°", "-90°", "0°", "90°", "180°"], fontsize=6)

    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ── 5b. DELTA FREQUENCY PLOT ──────────────────────────────────────────────────
def plot_delta_freq(delta_mag, s_params, freq_hz, title, save_path):
    """
    Line plot of ΔS12 and ΔS21 magnitude (dB) vs frequency.
    Horizontal dashed line at y=0 marks the relaxed baseline.
    """
    freq_ghz = freq_hz / 1e9

    # Extract S12 and S21 indices
    s12_idx = s_params.index("S12") if "S12" in s_params else None
    s21_idx = s_params.index("S21") if "S21" in s_params else None

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(title, fontsize=9, fontweight="bold")

    if s12_idx is not None:
        ax.plot(freq_ghz, delta_mag[s12_idx],
                color="#1f77b4", lw=1.8, label="ΔS12")
    if s21_idx is not None:
        ax.plot(freq_ghz, delta_mag[s21_idx],
                color="#ff7f0e", lw=1.8, label="ΔS21")

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6, label="Relaxed (0 dB)")

    # Clip y-axis so noise spikes don't collapse the scale
    all_vals = []
    if s12_idx is not None:
        all_vals.append(delta_mag[s12_idx])
    if s21_idx is not None:
        all_vals.append(delta_mag[s21_idx])
    if all_vals:
        combined = np.concatenate(all_vals)
        lo = max(np.percentile(combined, 1),  -30)
        hi = min(np.percentile(combined, 99),  30)
        pad = max((hi - lo) * 0.1, 0.5)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Frequency (GHz)", fontsize=10)
    ax.set_ylabel("ΔMagnitude (dB)", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ── 6. MAIN GENERATION LOOP ───────────────────────────────────────────────────
def generate(dataset_root: Path, samples: list[dict],
             output_dir: Path) -> list[dict]:
    index          = []
    n              = len(samples)
    failed         = 0
    baseline_cache = {}

    for i, sample in enumerate(samples):
        gesture     = sample["gesture"]
        participant = sample["participant"]
        position    = sample["position"]
        iteration   = sample["iteration"]
        s_params    = sample["s_params"]
        gesture_dir = sample["gesture_dir"]
        group       = sample["group"]

        # ── Relaxed baseline (cached per participant + position) ──────
        cache_key = (participant, position)
        if cache_key not in baseline_cache:
            base_mag, base_phase, _ = load_relaxed_baseline(
                dataset_root, participant, position, s_params
            )
            if base_mag is None:
                print(f"  [{i+1:>4}/{n}] [SKIP] no relaxed baseline: "
                      f"{participant}/{position}")
                failed += 1
                continue
            baseline_cache[cache_key] = (base_mag, base_phase)
        base_mag, base_phase = baseline_cache[cache_key]

        # ── Load gesture iteration ────────────────────────────────────
        freq, mag_iter, phase_iter = load_one_iteration(
            gesture_dir, participant, s_params, iteration
        )
        if mag_iter is None:
            print(f"  [{i+1:>4}/{n}] [SKIP] missing files: "
                  f"{gesture}/{participant}/{position}/iter{iteration}")
            failed += 1
            continue

        # ── Compute delta ─────────────────────────────────────────────
        delta_mag, delta_phase = compute_delta(
            mag_iter, phase_iter, base_mag, base_phase
        )

        # ── Output folders: separate heatmaps / freq_plots trees ─────
        heatmap_dir = output_dir / "heatmaps"  / group / position / gesture
        freq_dir    = output_dir / "freq_plots" / group / position / gesture
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        freq_dir.mkdir(parents=True, exist_ok=True)

        stem  = f"{participant}_{position}_iter{iteration}"
        title = f"{participant}  |  {position}  |  iter {iteration}  |  Δ vs relaxed baseline"

        # ── Heatmap ───────────────────────────────────────────────────
        heatmap_path = heatmap_dir / f"{stem}_delta_heatmap.png"
        plot_delta_heatmap(
            delta_mag, delta_phase, s_params, freq, title, heatmap_path
        )

        # ── Frequency line plot ───────────────────────────────────────
        freq_path = freq_dir / f"{stem}_delta_freq.png"
        plot_delta_freq(delta_mag, s_params, freq, title, freq_path)

        index.append({
            "heatmap_path": str(heatmap_path),
            "freq_path":    str(freq_path),
            "gesture":      gesture,
            "participant":  participant,
            "group":        group,
            "position":     position,
            "iteration":    iteration,
        })
        print(f"  [{i+1:>4}/{n}]  {gesture:<14}  {participant:<6}  "
              f"{position:<20}  iter{iteration}")

    print(f"\n  Generated : {len(index)} samples  ({failed} skipped)")
    return index


# ── 7. INDEX CSV ──────────────────────────────────────────────────────────────
def save_index(index: list[dict], output_dir: Path):
    path = output_dir / "sample_index.csv"
    fields = ["gesture", "participant", "group", "position",
              "iteration", "heatmap_path", "freq_path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index)
    print(f"  Index → {path}")

    dist = Counter(r["gesture"] for r in index)
    print("\n  Gesture distribution:")
    for g in ALL_GESTURES:
        print(f"    {g:<14}: {dist.get(g, 0)}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMG Delta Plot Generator")
    parser.add_argument("--all", action="store_true",
                        help="Generate plots for the full dataset (all 2376 samples)")
    parser.add_argument("--n", type=int, default=N_SAMPLES,
                        help=f"Number of stratified samples (default: {N_SAMPLES})")
    args = parser.parse_args()

    group_map = build_group_map(DATASET_ROOT)
    print(f"Participant groups: {group_map}")

    print("\n── Step 1: Discover samples ──")
    all_samples = discover_all_samples(DATASET_ROOT, ALL_GESTURES, group_map)

    if args.all:
        selected = all_samples
        print(f"\n── Full dataset mode: {len(selected)} samples ──")
    else:
        print(f"\n── Step 2: Stratified sampling ({args.n} total) ──")
        selected = stratified_sample(all_samples, args.n, seed=RANDOM_SEED)
        print(f"  Selected: {len(selected)} samples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n── Generating delta plots → {OUTPUT_DIR} ──")
    index = generate(DATASET_ROOT, selected, OUTPUT_DIR)

    print("\n── Saving index ──")
    save_index(index, OUTPUT_DIR)

    print(f"\n[DONE] → {OUTPUT_DIR.resolve()}")
