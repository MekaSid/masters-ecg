# Full DS2 NSTDB Robustness Results: Seed 42

This is the completed first seed of the final full-data protocol. Raw-only,
PH-only, and fusion checkpoints were trained with the matched seed-42 DS1
train/validation split. Every model received the same generated noisy ECG
realization for each DS2 beat and condition.

## Scope

- Test data: all 49,668 eligible three-beat MIT-BIH DS2 contexts.
- Class counts: N 44,218; S 1,836; V 3,219; F 388; Q 7.
- Noise: NSTDB baseline wander (BW), muscle artifact (MA), electrode motion
  (EM), and equal-power combined `BW + MA + EM`.
- SNR: 24, 18, 12, 6, 0, and -6 dB. Lower SNR means stronger corruption.
- Models: Zhang raw ECG/RR CNN; Dindin-style PH-only Betti CNN; raw/RR + PH
  fusion CNN.

The complete condition-level accuracy, macro-F1, per-class precision, F1,
recall, and confusion matrices are in local ignored artifacts:

- `results/noise_robustness/full_ds2_seed42/noise_robustness_metrics.json`
- `results/noise_robustness/full_ds2_seed42/noise_robustness_summary.csv`

## Accuracy / Macro-F1

Each cell is accuracy / macro-F1, in percent.

| Noise | SNR | Raw-only | PH-only | Fusion |
| --- | ---: | ---: | ---: | ---: |
| Clean | -- | 94.00 / 52.24 | 89.30 / 30.53 | 94.84 / 51.53 |
| BW | 24 | 94.00 / 52.18 | 89.42 / 30.64 | 94.87 / 51.61 |
| BW | 18 | 94.04 / 52.24 | 89.75 / 30.75 | 94.84 / 51.47 |
| BW | 12 | 94.23 / 52.31 | 89.86 / 30.57 | 94.73 / 51.54 |
| BW | 6 | 94.32 / 52.36 | 89.29 / 29.32 | 94.56 / 51.29 |
| BW | 0 | 93.67 / 50.95 | 88.31 / 26.78 | 93.89 / 49.26 |
| BW | -6 | 91.79 / 45.78 | 87.74 / 23.94 | 92.17 / 43.38 |
| MA | 24 | 94.10 / 52.29 | 90.16 / 31.02 | 95.03 / 51.57 |
| MA | 18 | 94.25 / 52.36 | 90.69 / 31.23 | 95.16 / 51.48 |
| MA | 12 | 94.70 / 52.52 | 90.72 / 30.03 | 95.39 / 51.47 |
| MA | 6 | 94.95 / 51.68 | 90.02 / 26.61 | 95.00 / 49.78 |
| MA | 0 | 93.72 / 47.18 | 89.45 / 23.08 | 93.04 / 42.61 |
| MA | -6 | 91.16 / 35.91 | 88.92 / 20.39 | 90.46 / 30.21 |
| EM | 24 | 94.02 / 52.27 | 89.88 / 30.83 | 94.88 / 51.59 |
| EM | 18 | 94.13 / 52.26 | 90.17 / 30.98 | 94.72 / 51.23 |
| EM | 12 | 94.06 / 51.94 | 90.10 / 30.80 | 94.57 / 50.97 |
| EM | 6 | 93.64 / 50.22 | 89.23 / 29.93 | 93.90 / 49.21 |
| EM | 0 | 90.79 / 43.68 | 87.36 / 26.06 | 91.75 / 41.30 |
| EM | -6 | 85.89 / 30.93 | 86.67 / 21.62 | 88.91 / 29.01 |
| Mix | 24 | 94.03 / 52.18 | 89.92 / 30.96 | 94.82 / 51.50 |
| Mix | 18 | 93.98 / 52.25 | 90.46 / 31.29 | 94.87 / 51.63 |
| Mix | 12 | 94.20 / 52.08 | 90.52 / 31.15 | 94.71 / 51.28 |
| Mix | 6 | 93.99 / 50.91 | 89.86 / 29.83 | 94.15 / 49.31 |
| Mix | 0 | 91.79 / 44.99 | 88.52 / 25.75 | 92.48 / 43.06 |
| Mix | -6 | 87.94 / 32.98 | 87.97 / 21.38 | 89.68 / 30.09 |

## Selected Per-Class Results

F and Q remain too rare for reliable inference; all per-class results remain
available in the JSON above. The clinically useful N/S/V comparison is:

| Condition | Model | N F1 / recall | S F1 / recall | V F1 / recall |
| --- | --- | ---: | ---: | ---: |
| Clean | Raw-only | 96.7 / 95.5 | 72.2 / 76.5 | 92.1 / 94.8 |
| Clean | PH-only | 94.3 / 95.6 | 0.2 / 0.1 | 57.3 / 65.1 |
| Clean | Fusion | 97.2 / 96.5 | 68.6 / 77.2 | 91.0 / 93.0 |
| Mix -6 dB | Raw-only | 94.0 / 94.0 | 24.3 / 14.2 | 45.1 / 57.9 |
| Mix -6 dB | PH-only | 93.6 / 98.2 | 0.2 / 0.1 | 13.1 / 8.6 |
| Mix -6 dB | Fusion | 94.7 / 97.8 | 13.2 / 7.2 | 42.1 / 35.7 |

## Interim Reading

- Fusion has higher accuracy than raw-only in the clean condition and at every
  combined-noise SNR in this seed, but its macro-F1 is lower because S/V
  performance degrades more sharply at severe noise.
- PH-only is not a viable standalone five-class classifier in this setup. It
  retains high N accuracy because N dominates DS2, while S recall is near zero.
- The strongest failure mode is severe EM and combined noise, particularly for
  S and V. Accuracy alone masks this because N dominates the test set.

These are seed-42 results only. The same report will be produced for seeds 43
and 44, then aggregated with mean and standard deviation before conclusions.
