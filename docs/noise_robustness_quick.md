# Quick NSTDB Robustness Evaluation

Run date: 2026-09-17

This is a reproducible **screening evaluation**, not the final thesis result. It
evaluates two clean-trained saved checkpoints on a deterministic class-stratified
subset of the patient-disjoint MIT-BIH DS2 test records:

- Raw-only: saved Zhang regular CNN.
- Fusion: saved Zhang raw ECG/RR encoder plus the Dindin-style Betti-curve PH branch.

## Protocol

- Test set: up to 300 examples from each AAMI class N/S/V/F and all 7 available
  Q examples, for 1,207 total beats. This deliberately balanced set is unlike
  natural DS2, which is dominated by N beats, so its overall accuracy must not
  be compared directly with the earlier full-DS2 clean accuracy.
- Corruptions: NSTDB baseline wander (`bw`), muscle artifact (`ma`), electrode
  motion (`em`), and `mix` (`bw + ma + em`). For `mix`, the summed noise is
  scaled jointly to the listed total SNR.
- SNR levels: 24, 12, 0, and -6 dB.
- Raw model: both ECG morphology channels are corrupted. RR-ratio feature maps
  remain clean, because the benchmark injects morphology noise but does not
  perturb expert R-peak annotations.
- Fusion PH branch: noisy lead-I three-beat context -> resample to 256 ->
  min-max normalization -> GUDHI direct H0 sublevel/upper-level persistence ->
  two 128-bin Betti curves.

Full machine-readable metrics, including confusion matrices and per-class
precision/recall/F1, are written to ignored local files:

- `results/noise_robustness/quick_ds2/noise_robustness_metrics.json`
- `results/noise_robustness/quick_ds2/noise_robustness_summary.csv`

## Aggregate Results

Accuracy / macro-F1, in percent. The clean row is the appropriate baseline for
this class-balanced subset.

| Condition | SNR dB | Raw-only | Fusion (raw + PH) |
| --- | ---: | ---: | ---: |
| clean | -- | 69.10 / 54.13 | 65.04 / 47.40 |
| bw | 24 | 68.93 / 53.82 | 64.79 / 47.20 |
| bw | 12 | 68.52 / 53.20 | 64.37 / 46.87 |
| bw | 0 | 65.37 / 49.19 | 60.98 / 44.33 |
| bw | -6 | 58.16 / 45.69 | 53.77 / 38.29 |
| ma | 24 | 69.18 / 54.09 | 64.54 / 46.88 |
| ma | 12 | 67.27 / 51.47 | 63.30 / 45.94 |
| ma | 0 | 56.84 / 41.61 | 52.11 / 37.88 |
| ma | -6 | 45.40 / 31.94 | 36.45 / 23.41 |
| em | 24 | 68.35 / 53.14 | 64.62 / 46.84 |
| em | 12 | 67.85 / 52.56 | 63.55 / 46.51 |
| em | 0 | 55.18 / 39.38 | 50.62 / 36.18 |
| em | -6 | 42.09 / 25.19 | 36.70 / 21.40 |
| mix | 24 | 68.68 / 53.67 | 64.87 / 47.16 |
| mix | 12 | 67.52 / 52.12 | 64.04 / 46.61 |
| mix | 0 | 54.60 / 38.51 | 50.62 / 36.04 |
| mix | -6 | 42.09 / 26.45 | 38.44 / 23.37 |

## Interpretation

- The expected degradation pattern appears: 24/12 dB has limited impact,
  whereas 0/-6 dB is substantially harder. Muscle artifact and electrode
  motion were more damaging than baseline wander at low SNR.
- In this preliminary checkpoint comparison, the PH fusion model does **not**
  improve aggregate noise robustness. It declines from 65.04% clean accuracy
  to 38.44% with mixed -6 dB noise; raw-only declines from 69.10% to 42.09%.
- This is not yet a controlled architecture conclusion: the raw-only and fusion
  checkpoints came from separately trained runs. The existing matched
  clean-data ablation should be extended to matched noise training/evaluation
  seeds before making a thesis claim about PH robustness.

## Re-run

```bash
./.venv/bin/python scripts/evaluation/evaluate_noise_robustness.py
```

Use `--max-per-class 0` to process all eligible DS2 examples. That is a more
complete but slower evaluation.

## Matched Seed-45 Follow-Up

After the initial screening, raw-only and fusion were retrained using the same
DS1 split seed (`45`), training schedule, validation-loss checkpoint selection,
and clean training examples. Their full clean DS2 accuracies were 92.69%
(raw-only) and 94.18% (fusion). The same quick class-balanced NSTDB evaluation
then produced the results below.

| Condition | SNR dB | Matched raw-only | Matched fusion |
| --- | ---: | ---: | ---: |
| clean | -- | 59.32 / 46.13 | 64.37 / 46.99 |
| bw | 24 | 59.49 / 45.84 | 64.37 / 46.83 |
| bw | 12 | 59.98 / 45.99 | 63.55 / 45.79 |
| bw | 0 | 59.82 / 45.49 | 63.05 / 45.15 |
| bw | -6 | 53.02 / 38.72 | 56.75 / 40.75 |
| ma | 24 | 59.40 / 45.80 | 64.21 / 46.63 |
| ma | 12 | 60.23 / 46.53 | 63.63 / 45.70 |
| ma | 0 | 50.21 / 36.34 | 55.01 / 38.74 |
| ma | -6 | 38.61 / 26.95 | 41.67 / 28.28 |
| em | 24 | 60.07 / 46.14 | 64.04 / 46.35 |
| em | 12 | 59.49 / 45.41 | 64.13 / 46.29 |
| em | 0 | 51.62 / 36.58 | 53.69 / 36.70 |
| em | -6 | 40.51 / 25.26 | 39.44 / 23.06 |
| mix | 24 | 59.49 / 45.28 | 64.13 / 46.33 |
| mix | 12 | 60.23 / 45.95 | 64.54 / 47.03 |
| mix | 0 | 52.53 / 37.51 | 54.43 / 37.76 |
| mix | -6 | 40.27 / 25.31 | 39.02 / 22.16 |

Values are accuracy / macro-F1, in percent. Full metrics are local and ignored:

- `results/noise_robustness/matched_seed45_quick_ds2/noise_robustness_metrics.json`
- `results/noise_robustness/matched_seed45_quick_ds2/noise_robustness_summary.csv`

This controlled single-seed result supports a limited conclusion: fusion has a
better clean and moderate-noise operating point, but it is not uniformly more
robust at severe electrode motion or severe combined noise. Multiple matched
seeds and full DS2 evaluation are still required before making a final claim.
