# Full DS2 NSTDB Robustness: Three-Seed Aggregate

This report aggregates three matched clean-training seeds (42, 43, 44) evaluated
on all 49,668 eligible three-beat MIT-BIH DS2 contexts. Each model receives the
same generated NSTDB-corrupted ECG realization for every beat, noise condition,
and SNR within a seed.

Models:

- **Raw-only:** Zhang two-lead ECG + RR feature CNN.
- **PH-only:** direct 1D H0 sublevel and upper-level PH -> two Betti curves -> CNN.
- **Fusion:** raw/RR encoder plus the PH Betti-curve encoder.

Conditions are baseline wander (BW), muscle artifact (MA), electrode motion
(EM), and a combined equal-power BW + MA + EM mixture. SNR decreases from 24
dB (light noise) to -6 dB (severe noise).

## Aggregate Outputs

Local full metrics and generated figures are intentionally Git-ignored:

- `results/noise_robustness/aggregate_seeds_42_44/aggregate_metrics_mean_sd.csv`
- `results/noise_robustness/aggregate_seeds_42_44/accuracy_vs_snr_mean_sd.png`
- `results/noise_robustness/aggregate_seeds_42_44/macro_f1_vs_snr_mean_sd.png`
- `results/noise_robustness/aggregate_seeds_42_44/per_class_recall_vs_snr_{bw,ma,em,mix}_mean_sd.png`

The reported standard deviation is the sample SD across the three seeds, not a
confidence interval or statistical significance test.

## Accuracy and Macro-F1

Values are mean +/- SD, in percent.

| Condition | Raw-only | PH-only | Fusion |
| --- | ---: | ---: | ---: |
| Clean accuracy | 92.57 +/- 1.36 | 89.16 +/- 0.21 | **95.61 +/- 0.94** |
| Clean macro-F1 | 51.79 +/- 1.31 | 30.48 +/- 0.31 | **54.06 +/- 2.20** |
| BW -6 dB accuracy | 91.86 +/- 1.16 | 87.62 +/- 0.25 | **93.19 +/- 0.92** |
| BW -6 dB macro-F1 | 45.32 +/- 1.79 | 24.11 +/- 0.21 | **46.31 +/- 2.53** |
| MA -6 dB accuracy | 90.50 +/- 0.58 | 88.94 +/- 0.05 | **90.84 +/- 0.40** |
| MA -6 dB macro-F1 | **33.31 +/- 2.45** | 20.62 +/- 0.22 | 32.35 +/- 3.05 |
| EM -6 dB accuracy | 87.51 +/- 1.60 | 86.11 +/- 0.95 | **89.83 +/- 0.94** |
| EM -6 dB macro-F1 | 30.82 +/- 0.54 | 21.78 +/- 0.41 | **31.20 +/- 2.16** |
| Mix 0 dB accuracy | 91.94 +/- 0.24 | 88.53 +/- 0.26 | **93.56 +/- 0.94** |
| Mix 0 dB macro-F1 | 43.66 +/- 1.22 | 26.28 +/- 0.64 | **46.10 +/- 2.65** |
| Mix -6 dB accuracy | 88.91 +/- 1.00 | 87.62 +/- 0.55 | **90.44 +/- 0.78** |
| Mix -6 dB macro-F1 | 32.22 +/- 0.87 | 21.67 +/- 0.37 | **32.56 +/- 2.26** |

## Interpretation

- **Fusion is consistently best for overall accuracy.** It improves clean
  accuracy by 3.04 points over raw-only and retains a 1.53-point advantage
  under combined -6 dB noise.
- **The PH branch is not useful by itself.** PH-only has about 89% accuracy
  because N dominates DS2, but its clean S recall is only 0.1% and its clean V
  recall is 66.2%. Its macro-F1 is about 30%, far below raw-only and fusion.
- **Light noise is not the decisive regime.** At 24 to 12 dB, all models are
  close to their clean operating point. Meaningful degradation appears at 0
  dB and becomes substantial at -6 dB.
- **EM and combined noise are the hardest overall conditions.** Fusion remains
  highest in accuracy, but severe noise disproportionately harms S and V,
  lowering macro-F1. At mixed -6 dB, mean S recall is only 10.3% for fusion and
  10.2% for raw-only; mean V recall is 41.9% for fusion and 50.9% for raw-only.
- **Macro-F1 changes the conclusion in one case.** Fusion wins MA -6 dB
  accuracy by 0.34 points but raw-only has a 0.96-point macro-F1 advantage.
  Thus an accuracy-only claim of superior robustness would be incomplete.
- **F and Q are not interpretable as robust estimates.** DS2 contains only 388
  F beats and 7 Q beats. Report their metrics, but do not make a comparative
  robustness claim from them.

## Current Conclusion

The controlled three-seed result supports: the raw + PH fusion improves clean
classification and preserves the strongest overall accuracy under NSTDB noise.
It does **not** support a broad claim that PH uniformly protects minority-class
classification at severe noise. The next required analysis is to relate these
classification curves to clean-to-noisy Betti and persistence-diagram stability
measures, particularly for S and V under MA, EM, and combined noise.
