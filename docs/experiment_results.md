# Experiment Results

This document records completed clean-data baselines. Model checkpoints, epoch histories, and full confusion matrices live in the ignored `results/` directory; this file is the version-controlled summary.

## 1. Raw Conv1D Baseline

**Status:** Complete clean-data baseline. No NSTDB noise was used.

| Item | Value |
|---|---|
| Architecture | One-lead `Conv1D -> Conv1D -> Conv1D -> max pooling -> fully connected` |
| Input | Per-beat z-scored, 256-sample fixed ECG window |
| Classes | `N`, `S`, `V` |
| Split | 18 record DS1 train, 4 DS1 validation, 22 DS2 test records |
| Training beats | 41,988 |
| Validation beats | 8,590 |
| Test beats | 49,298 |
| Training | AdamW, tempered inverse-frequency class weights, 40 epochs |
| Checkpoint selection | Highest validation macro-F1, epoch 26 |

| Held-out DS2 metric | Result |
|---|---:|
| Accuracy | 89.53% |
| Macro-F1 | 55.23% |
| `N` F1 | 95.40% |
| `S` F1 | 7.19% |
| `V` F1 | 63.10% |

**Interpretation:** Accuracy is inflated by the dominant normal class. The poor `S` result indicates that fixed morphology alone does not capture enough rhythm context for supraventricular beats.

Artifacts: `results/raw_cnn/full_clean_record_disjoint/`.

## 2. Zhang et al. Regular CNN Reproduction

**Status:** Complete clean-data reproduction. No NSTDB noise was used.

| Item | Value |
|---|---|
| Architecture | Zhang raw/RR residual-attention CNN, without the adversarial subject-ID branch |
| Input | Two ECG leads, dynamic beat segmentation, 128-point resampling, mean removal, pre-RR and near-pre-RR feature rows |
| Classes | `N`, `S`, `V`, `F`, `Q` |
| Split | Standard DS1 training; seeded random 20% DS1 validation; untouched DS2 test |
| DS1 examples after preprocessing | 50,993 |
| DS2 examples after preprocessing | 49,682 |
| Training | Adam, initial learning rate 0.001, ReduceLROnPlateau, early stopping |
| Checkpoint selection | Lowest validation cross-entropy, epoch 28 |

| Held-out DS2 metric | Result |
|---|---:|
| Accuracy | 94.53% |
| `N` F1 | 97.14% |
| `S` F1 | 73.88% |
| `V` F1 | 92.08% |
| `F` F1 | 10.55% |
| `Q` F1 | 0.00% |

The published regular-CNN reference reports 93.9% overall accuracy, 69.1% `S` sensitivity, and 89.9% `V` sensitivity. This run is close in overall accuracy, with `S` sensitivity 76.6% and `V` sensitivity 89.7%. `F` and `Q` remain unreliable because the DS1 set contains only 414 and 8 examples after preprocessing.

**Reproduction caveat:** Zhang et al. do not publish source code or attention-module internals. This implementation uses documented channel-and-temporal attention and current official MIT-BIH annotations, whose accepted-beat total is slightly higher than the paper's printed table.

Reference: Zhang et al., [Interpatient ECG Heartbeat Classification with an Adversarial Convolutional Neural Network](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/), 2021.

Artifacts: `results/zhang_regular_cnn/zhang_regular_clean/`.

## 3. Dindin-Style Betti Curves

**Status:** Complete clean-data PH-only and fusion experiment. No NSTDB noise was used.

Planned controlled comparison on the same DS1-to-DS2 protocol:

| Variant | Input |
|---|---|
| PH-only | Dindin-style sublevel and upper-level H0 Betti curves -> small 1D CNN |
| Fusion | Zhang raw/RR embedding + Betti-curve embedding -> classifier |

The PH construction uses GUDHI and follows Dindin et al.'s use of sublevel and upper-level filtrations. It deliberately excludes their other channels, autoencoder, FFT features, and multi-database cross-validation so the incremental value of the Betti branch can be measured directly.

| Item | Value |
|---|---|
| Split | Standard DS1 training; seeded random 20% DS1 validation; untouched DS2 test |
| PH input | Lead-I sequence spanning the preceding, central, and following annotated R peaks; resampled to 256 samples |
| PH representation | GUDHI H0 sublevel and upper-level Betti curves, 128 bins each |
| Paired DS1 examples | 50,977 |
| Paired DS2 examples | 49,668 |
| Training | Adam, initial learning rate 0.001, ReduceLROnPlateau, validation-loss early stopping |

| Held-out DS2 metric | PH-only | Raw + Betti fusion |
|---|---:|---:|
| Best epoch | 100 | 15 |
| Accuracy | 89.30% | **96.77%** |
| Macro-F1, all five classes | 30.53% | 54.28% |
| `N` F1 | 94.25% | **98.29%** |
| `S` F1 | 0.21% | **76.50%** |
| `V` F1 | 57.28% | **94.26%** |
| `F` F1 | 0.88% | 2.37% |
| `Q` F1 | 0.00% | 0.00% |

**Interpretation:** Betti curves alone do not carry enough morphology or rhythm information to be a competitive five-class classifier. The end-to-end fusion does improve the clean `N`, `S`, and `V` results over the prior raw Zhang run (94.53% accuracy, `S` F1 73.88%, `V` F1 92.08%). This is encouraging but is not yet a noise-robustness conclusion. The PH input spans three beats, so it introduces local rhythm context in addition to topology; the next step must test the same paired clean/noisy examples across NSTDB conditions.

Artifacts: `results/dindin_betti/dindin_betti_clean/`. Cached PH inputs: `data/tda/betti_curves/`.

## 4. Matched Raw-Only Versus Fusion Ablation

**Status:** Complete three-seed clean-data ablation. No NSTDB noise was used.

This removes the main ambiguity in the initial fusion comparison: both variants use the identical three-beat-filtered DS1 and DS2 examples. The raw-only model receives only the Zhang two-lead ECG/RR input; the fusion model receives the same raw input plus the precomputed Dindin-style Betti curves. Each seed changes both the DS1 train/validation partition and neural-network initialization.

| Held-out DS2 metric | Raw-only mean +/- SD | Fusion mean +/- SD | Fusion change |
|---|---:|---:|---:|
| Accuracy | 92.57% +/- 1.36% | **95.61% +/- 0.94%** | +3.04 points |
| `N` F1 | 95.90% +/- 0.80% | **97.66% +/- 0.54%** | +1.76 points |
| `S` F1 | 72.76% +/- 3.66% | **75.20% +/- 5.72%** | +2.44 points |
| `V` F1 | 89.47% +/- 3.04% | **91.60% +/- 2.26%** | +2.13 points |

Seeds: `42`, `43`, `44`. The fusion accuracy was higher in all three matched runs. `S` F1 improved on average but has higher variation, so it should not be described as a settled improvement until evaluated under NSTDB noise and with more seeds.

Generate the chart from saved JSON metrics:

```bash
./.venv/bin/python scripts/visualization/plot_ablation_results.py \
  --run-dir results/dindin_betti/paired_ablation \
  --seeds 42 43 44
```

Chart: `results/dindin_betti/paired_ablation/paired_ablation_metrics.png`.

Reference: Dindin, Umeda, and Chazal, [Topological Data Analysis for Arrhythmia Detection through Modular Neural Networks](https://arxiv.org/abs/1906.05795), 2020.
