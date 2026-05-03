# Byzantine-Robust Federated Learning for ICS Intrusion Detection

> **Paper:** *Byzantine-Robust Federated Learning for Industrial Control System Intrusion Detection: A Comprehensive Framework with Explainable AI*
> **Authors:** Mohamed Massaoudi, Katherine R. Davis, Maymouna Ez Eddin
> **Affiliation:** Texas A&M University · Tarleton State University

---

## Overview

This repository provides the complete, reproducible implementation of the paper. The framework addresses a critical gap in industrial cybersecurity: existing Byzantine-robust federated learning (FL) defenses have never been systematically evaluated on real Modbus/ICS traffic.

**Key contributions:**
- First Byzantine-robust FL framework evaluated specifically on Modbus ICS traffic (6,690 samples, 17 features)
- Seven aggregation strategies benchmarked under five attack types
- Novel **Bulyan-Adaptive** method with per-round norm clipping achieving 100% accuracy across all attack configurations
- SHAP-based explainability with Pearson *r* > 0.85 consistency across federated rounds

---

## Quick Start

```bash
git clone https://github.com/mmassaoudi/byzantine-robust-fl-ics.git
cd byzantine-robust-fl-ics
pip install -r requirements.txt
python main.py
```

No manual configuration required. All outputs (figures + `results_summary.json`) are written to the same directory.

---

## Repository Structure

```
.
├── main.py               # Single-file reproducible implementation
├── requirements.txt      # Pinned dependencies (Python 3.9+)
└── results_summary.json  # Pre-computed experiment results
```

---

## Methods

| Aggregation | Byzantine Tolerance | Key Property |
|---|---|---|
| FedAvg | None | Baseline; vulnerable to all attacks |
| Trimmed Mean | Partial | Removes coordinate-wise extremes |
| Median | ~50% | High breakdown point |
| Krum | f < N/2 | Nearest-neighbour selection |
| Multi-Krum | f < N/2 | Averages top-k Krum candidates |
| Bulyan | f < N/4 | Krum selection + trimmed mean |
| **Bulyan-Adaptive** | **f < N/4** | **+ adaptive norm clipping (proposed)** |

### Attack Models

| Attack | Description |
|---|---|
| Label Flipping | Sign inversion with amplification (α = 1.5) |
| Gaussian Noise | High-variance injection (σ = 5.0) |
| Backdoor | Parameter amplification (β = 10.0) |
| Same-Value | Constant replacement (c = 50.0) |
| Zero Attack | Gradient suppression |

---

## Results Summary

Accuracy (%) under 20% Byzantine ratio, IID setting, 30 rounds:

| Method | No Attack | Label Flip | Gaussian | Backdoor | Same-Value | Zero |
|---|---|---|---|---|---|---|
| FedAvg | 100.0 | 100.0 | **37.7** | 95.9 | **40.0** | 100.0 |
| Median | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Bulyan | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **Bulyan-Adaptive** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |

At 40% Byzantine ratio, FedAvg collapses to 40.0% while Bulyan-Adaptive holds at 100%.

---

## Model Architecture

```
Input(17) -> Conv1D(32) -> MaxPool -> Conv1D(64) -> MaxPool
          -> Self-Attention -> FC(128) -> Dropout(0.3)
          -> FC(64) -> Dropout(0.3) -> Output(2)
```
Total parameters: **113,538**. Inference latency: **< 10 ms** (36 KB quantized footprint).

---

## Reproducibility

- Random seed fixed at `SEED = 42`
- Device forced to CPU for identical results across machines
- Synthetic dataset generated deterministically if real CSV is unavailable
- All dependency versions pinned in `requirements.txt`

---

## Citation

If you use this code, please cite:

```bibtex
@article{massaoudi2026byzantine,
  title   = {Byzantine-Robust Federated Learning for Industrial Control System
             Intrusion Detection: A Comprehensive Framework with Explainable AI},
  author  = {Massaoudi, Mohamed and Davis, Katherine R. and Ez Eddin, Maymouna},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026}
}
```

---

## License

This project is released for academic research purposes.
