# Codex Reference Notes

Last updated: September 3, 2026

## Project Context

This repository studies robustness of ECG arrhythmia classification under realistic noise using:

- MIT-BIH Arrhythmia Database (`mitdb`) for clean labeled ECG beats
- MIT-BIH Noise Stress Test Database (`nstdb`) for realistic noise

Planned comparison:

- Raw ECG model
- Persistent homology / TDA model
- Raw ECG + TDA fusion model

Current milestone is only the shared data, preprocessing, noise, and TDA pipeline. Full model training is intentionally deferred.

## MIT-BIH Summary

MIT-BIH is used as the clean source dataset.

It contains:

- ECG waveform recordings over time
- one or more ECG channels per record
- expert annotation sample positions
- annotation symbols for beats/events

Typical task:

`ECG signal -> extract beat around annotated R-peak -> classify heartbeat type`

In this project, MIT-BIH provides:

- clean ECG waveforms
- beat locations
- original annotation symbols
- mapped beat classes such as `N`, `S`, `V`, `F`, `Q`

Example from local record `100`:

- total samples: `650000`
- channels in header: `MLII`, `V5`
- sampling rate: `360 Hz`
- annotation count: `2274`

Important detail:

- not every annotation is a heartbeat class
- early symbols in record `100` include markers such as `"+"`
- label mapping and filtering must remain centralized

## NSTDB Summary

NSTDB is not the labeled arrhythmia dataset.

It is a standalone noise dataset used to corrupt clean MIT-BIH beats in a controlled way.

Noise types:

- `bw` = baseline wander
- `ma` = muscle artifact
- `em` = electrode motion

Use in this project:

`clean MIT-BIH beat + NSTDB noise segment + chosen SNR -> noisy beat`

NSTDB is used to stress-test robustness, not to provide labels.

## Stress-Test Conditions

Each noisy condition is defined by:

- one noise type: `bw`, `ma`, or `em`
- one SNR level: `24`, `18`, `12`, `6`, `0`, `-6 dB`

So each clean beat can produce:

- `18` noisy versions total
- `3` noise types x `6` SNR levels

Higher SNR means less noise. Lower SNR means more noise.

Practical interpretation:

- `24 dB` = mild noise
- `18 dB` = noticeable noise
- `12 dB` = moderate noise
- `6 dB` = strong noise
- `0 dB` = noise power roughly equal to signal power
- `-6 dB` = noise stronger than signal

## How Noise Is Applied

NSTDB corruption is created synthetically in code, not by reading a pre-corrupted MIT-BIH dataset.

Pipeline:

1. Load a clean beat from MIT-BIH.
2. Load one NSTDB noise recording.
3. Sample a noise segment of the same length as the beat.
4. Scale the noise segment to match the target SNR.
5. Add the scaled noise to the clean beat.

Formula:

`noisy_signal = clean_signal + scaled_noise`

This keeps the label fixed while changing only signal quality, which is necessary for robustness experiments and clean-vs-noisy TDA comparisons.

Implementation reference:

- [src/data/noise.py](/Users/sid/Desktop/masters_final/src/data/noise.py:1)

## PhysioNet File Types

Core WFDB file types:

- `.dat`: raw waveform samples
- `.hea`: header/metadata needed to interpret the waveform
- `.atr`: annotation file with beat/event sample positions and symbols

Other mirrored files that may appear:

- `.xws`: extra PhysioNet-side metadata/workspace file; not required for this pipeline
- `index.html`, `robots.txt`: mirror artifacts from recursive website download; not required for this pipeline

For this project the important files are:

- MIT-BIH: `.dat`, `.hea`, `.atr`
- NSTDB: `.dat`, `.hea`

## Local Download Status

Confirmed present locally as of September 3, 2026:

MIT-BIH project data:

- `data/raw/mitdb/100.dat`
- `data/raw/mitdb/100.hea`
- `data/raw/mitdb/100.atr`

NSTDB project data:

- `data/raw/nstdb/bw.dat`
- `data/raw/nstdb/bw.hea`
- `data/raw/nstdb/ma.dat`
- `data/raw/nstdb/ma.hea`
- `data/raw/nstdb/em.dat`
- `data/raw/nstdb/em.hea`

Additional recursive mirror path created later:

- `data/physionet.org/files/mitdb/1.0.0/`

That mirrored directory was started with `wget` and then intentionally interrupted after confirming it worked, so it is incomplete.

## Download History

What was attempted:

1. The project downloader in `scripts/download_data.py` using `wfdb`
2. That hit a transient PhysioNet `502 Bad Gateway` on `100.dat`
3. Direct file downloads were then used to populate `data/raw/mitdb` and `data/raw/nstdb`
4. Later, the website’s recommended recursive `wget` command was also tested successfully

Exact website-style command that was tested:

```bash
cd data
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
```

Important behavior:

- this mirrors the website layout under `data/physionet.org/files/mitdb/1.0.0/`
- it downloads more than the minimal pipeline files
- it pulls the full database recursively unless interrupted

## Code Assumptions

Current code expects main raw project data under:

- `data/raw/mitdb/`
- `data/raw/nstdb/`

The recursive `wget` mirror path is separate unless the code/config is updated to point there.

Current beat extraction assumption:

- fixed window around each annotated beat
- default config uses `128` pre-samples and `128` post-samples

## Short Explanation To Reuse

MIT-BIH gives clean labeled heartbeat data.

NSTDB gives realistic noise recordings.

The experiment takes a clean MIT-BIH beat, adds one NSTDB noise type at one chosen SNR level, and then measures how raw-signal and TDA-based methods degrade as noise increases.
