# P8_Project
# Radiomyography (RMG) — VLM and CNN Gesture Classification

P8 Master's Thesis, Electronic Systems, Aalborg University (AAU)  
Supervisor: Ming Shen

---

## Overview

This repository contains the full implementation for gesture classification using RF S-parameter measurements captured with a 4-port R&S ZNA VNA at 3 GHz. Two classification pipelines are implemented:

1. **VLM-based in-context learning** — Qwen2.5-VL-7B and Gemma-3-12B classifying delta heatmaps and combined representations without training.
2. **Supervised multi-task CNN** — 1D convolutional network trained on the spectral tensor for 11-class gesture recognition and 3-class tremor detection.

---

## Repository Structure
```
rmg-vlm-cnn/
│
├── rmg_delta_generator.py   # Delta heatmap generation(RdBu_r / coolwarm)
├── rmg_gemma3.py            # Gemma-3-12B inference pipeline
├── rmg_delta_colab_vlm.py   # qwen2.5vl:7B inference pipeline
├── OneGesture.py            # S-parameter data gathering through VNA.
│
├── figures/
│   └── reference/           # Reference heatmaps — Participant 1, nominal position
│       ├── fist_iter0.png
│       ├── grasp_iter0.png
│       └── ...              # All 11 gesture reference images
...
└── README.md
```
## Dataset

- **Participants:** 3 (1 reference, 2 test)
- **Positions:** 6 per participant (right/left arm × nominal, −3 cm, +3 cm)
- **Gestures:** 11 classes — fist, grasp, open_palm, pinch, point, relaxed, shaka, thumbs_up, twist, wrist_down, wrist_up
- **Repetitions:** 12 per gesture per position
- **S-parameters:** S11, S12, S21, S22 at 300 frequency points (2.5–3.5 GHz)

**Dataset access:** Available via OneDrive — [Request access here](https://aaudk-my.sharepoint.com/:f:/r/personal/om73gw_student_aau_dk/Documents/RMGDataset?csf=1&web=1&e=Ld8c5D))

> Place the downloaded dataset in a `data/` folder at the repository root before running any scripts.

---

## Pipeline 1 — VLM Classification

**Input representations:**
- **Delta mode:** ΔS-parameter heatmaps (gesture − relaxed baseline), rendered with `RdBu_r` (magnitude) and `coolwarm` (phase) colourmaps
- **Combined mode:** Delta heatmaps + ΔS12/ΔS21 frequency line traces

**Reference set:** 1 image per gesture class, selected by minimum Euclidean distance to the per-class mean magnitude matrix — drawn from Participant 1 at the nominal position.

**Requirements:**
- [Ollama](https://ollama.com) installed locally
- Qwen2.5-VL-7B model pulled

```bash
ollama pull qwen2.5-vl:7b
python VLMTest.py              # Qwen delta mode
python rmg_gemma3.py           # Gemma-3-12B delta mode
python rmg_delta_colab_vlm.py  # Combined mode
```

---

## Pipeline 2 — Multi-Task CNN

**Input:** Spectral tensor of shape `(8, 300)` — magnitude and phase of S11, S12, S21, S22 across 300 frequency points.

**Architecture:**
- Dual-stem `Conv1d(4→8, k=7)` for modality separation (magnitude/phase)
- 3-stage residual backbone with Squeeze-and-Excitation (SE) blocks
- Channel expansion: 16 → 16 → 32 → 64
- Two heads: 11-class gesture recognition, 3-class tremor detection

**Training:** PyTorch, NVIDIA A100, AdamW, focal loss, phased curriculum (30-epoch gesture-only warmup), fixed seed 42.

---

## Requirements
```
python >= 3.10
torch
numpy
pandas
matplotlib
scikit-learn
ollama
Pillow
pathlib
```
Install:
```bash
pip install -r requirements.txt
```

---

## Results

| Pipeline | Representation | Nominal | −3 cm | +3 cm |
|---|---|---|---|---|
| Qwen2.5-VL-7B | Delta | 9.0% | 11.0% | 10.0% |
| Qwen2.5-VL-7B | Combined | 11.0% | 11.0% | 7.0% |
| Gemma-3-12B | Delta | 12.0% | 11.0% | 9.0% |
| Gemma-3-12B | Combined | 10.0% | 15.0% | 13.0% |
| Random baseline | — | 10.0% | 10.0% | 10.0% |
| CNN (multi-task) | Spectral tensor | TBD | TBD | TBD |

---

## Citation


@mastersthesis{rmg_aau_2026,
  author     = {Bijon and Riccardo}
  title      = {Radiomyography-Based Gesture Classification Using VLM
                In-Context Learning and Multi-Task CNN},
  school     = {Aalborg University},
  department = {Electronic Systems},
  year       = {2026}
}

