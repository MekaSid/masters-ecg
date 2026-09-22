# Dindin Recreation Visuals

Generate an explanatory recreation of the direct-1D persistent-homology branch:

```bash
./.venv/bin/python scripts/dindin_recreation/generate_visualizations.py --snr-db 12
```

The default uses an `A` beat from MIT-BIH record 100 and generates clean, NSTDB
baseline-wander, muscle-artifact, and electrode-motion images at 12 dB.

The primary `comparison_{bw,ma,em}_12db.png` figures place clean and noisy
versions side by side: raw ECG context, persistence barcodes, and Betti curves.

Each figure contains:

1. The previous, central labeled, and following ECG beats from lead I.
2. A sublevel-filtration snapshot, where the blue regions are samples with
   normalized amplitude less than or equal to the displayed threshold.
3. GUDHI H0 sublevel and upper-level persistence barcodes. Upper-level PH is
   implemented as sublevel PH on the negated ECG.
4. The final `2 x 128` Betti tensor. Each point is the number of H0 barcode
   intervals alive at one filtration threshold; the horizontal axis is voltage
   threshold, not time.

The saved `recreation_metadata.json` records the deterministic NSTDB segment
starts and achieved SNRs. Output is written to ignored `results/dindin_recreation/`.
