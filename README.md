# ECG TDA Robustness

Initial codebase for studying how persistent-homology-based ECG arrhythmia classification behaves under realistic noise from the MIT-BIH Noise Stress Test Database.

## Research Goal

The eventual comparison is:

- Raw ECG model
- Persistent Homology / TDA model
- Raw ECG + Persistent Homology fusion model

All approaches are intended to train on clean MIT-BIH Arrhythmia Database beats and be evaluated under increasing NSTDB noise.

This milestone implements the shared data, preprocessing, noise, and TDA pipeline needed before model training.

The TDA implementation is currently consolidated onto `gudhi` for:

- Vietoris-Rips persistent homology
- 1D sublevel-set persistent homology
- persistence images
- diagram distances

## Datasets

- MIT-BIH Arrhythmia Database (`mitdb`)
- MIT-BIH Noise Stress Test Database (`nstdb`)

Data access is handled with `wfdb` and stored locally under `data/raw/`.

## Repository Layout

```text
configs/         YAML configuration for data and TDA parameters
data/            Raw, processed, and TDA-derived artifacts
notebooks/       Lightweight exploration notebooks
scripts/         CLI entrypoints for download, preprocessing, noise, TDA, and smoke tests
src/             Reusable project code
tests/           Lightweight unit tests for core pipeline steps
results/         Figures and diagnostic outputs
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Data

```bash
python3 scripts/download_data.py
```

This downloads:

- `mitdb`, defaulting to record `100` for the first smoke test
- `nstdb` noise records `bw`, `ma`, and `em`

## Preprocess MIT-BIH Beats

```bash
python3 scripts/preprocess_mitdb.py --records 100
```

This extracts fixed-length heartbeat windows around annotated beats, preserves original symbols, maps labels into AAMI-style classes, and saves:

- beat metadata to `data/processed/clean/*.csv`
- waveform arrays to `data/processed/clean/*.npz`

## Visualize an ECG Window

```bash
./.venv/bin/python scripts/visualize_ecg.py --record 100 --channel 0 --start-seconds 0 --duration-seconds 10
```

This saves a labeled PNG under `results/waveforms/`. The graph uses physical ECG amplitude in mV, labels the time axis in seconds, and marks annotated beats from the matching `.atr` file.

Generate ten different examples sampled across the recording:

```bash
./.venv/bin/python scripts/generate_waveform_examples.py --record 100 --count 10 --duration-seconds 5
```

The individual PNGs are saved under `results/waveforms/record_100_examples/`.

### Annotation Symbols in Waveform Plots

The letter above each red marker is the original MIT-BIH expert annotation, not a prediction. Common symbols are `N` (normal), `A` (atrial premature), `V` (premature ventricular), `F` (fusion), and `/` (paced). The preprocessing step preserves these detailed labels and maps them into broader AAMI-style classes in `src/data/labels.py`.

## Run TDA Pipeline

```bash
python3 scripts/compute_tda.py --record 100 --beat-index 0
```

This computes:

- delay embeddings
- H0/H1 persistence diagrams
- persistence images

## Train the Raw ECG Conv1D Baseline

```bash
./.venv/bin/python scripts/download_data.py --mitdb-records 101 106 108 109 --skip-nstdb
./.venv/bin/python scripts/train_raw_ecg.py
```

The baseline architecture is `ECG beat -> Conv1D -> Conv1D -> Conv1D -> max pooling -> fully connected classifier`.

The initial record-disjoint pilot split is configured in `configs/training.yaml`:

- Train: records `100`, `101`, `106`
- Validation: record `109`
- Test: record `108`
- Target classes: `N` (normal), `S` (supraventricular), `V` (ventricular)

The script saves `raw_ecg_conv1d.pt`, `metrics.json`, and `history.csv` under `results/raw_cnn/`. This small split is intended to validate the training path only; a formal experiment should use a larger standard patient-disjoint MIT-BIH split because these few records have highly uneven class distributions.

Alternative direct 1D topology baseline:

```bash
python3 scripts/compute_tda.py --record 100 --beat-index 0 --method sublevel
```

## End-to-End Smoke Test

```bash
python3 scripts/test_pipeline.py --record 100 --noise-type ma --snr-db 12
```

The smoke test:

1. Loads MIT-BIH record `100`
2. Loads annotations
3. Extracts one valid heartbeat
4. Maps its label
5. Loads NSTDB noise
6. Generates a noisy version at the requested SNR
7. Computes delay embeddings for clean and noisy beats
8. Computes H0/H1 persistence diagrams
9. Generates persistence images
10. Computes bottleneck and Wasserstein distances
11. Saves diagnostic plots under `results/`

## Configuration

Main settings live in:

- `configs/data.yaml`
- `configs/tda.yaml`

These control beat windowing, dataset paths, record splits, noise settings, embedding parameters, and persistence image settings.

## Label Mapping

Detailed MIT-BIH beat symbols are mapped centrally in `src/data/labels.py` into AAMI-style classes:

- `N`: `N`, `L`, `R`, `e`, `j`
- `S`: `A`, `a`, `J`, `S`
- `V`: `V`, `E`
- `F`: `F`
- `Q`: `/`, `f`, `Q`, `|`, `~`, `!`, `+`, `[`, `]`, `"`, `x`

The original annotation symbol is retained in processed metadata alongside the mapped class.

## Current Status

Implemented:

- Repository structure
- Data download and loading
- Beat extraction and label mapping
- Record-based split configuration
- NSTDB noise loading and SNR-based corruption
- Delay embedding and persistent homology computation
- Alternative sublevel-set persistent homology baseline
- Persistence image generation
- Persistence summary statistics
- Diagram distance utilities
- End-to-end smoke-test script
- Lightweight tests

Not implemented yet:

- TDA classifier training
- Fusion model training
- Formal train/validation/test experiment runner

## Notes

- Splitting is record-driven by configuration; beats are not randomly split by default.
- Persistence image fitting is currently local to the input diagrams for smoke testing. For formal experiments it should be fit on training data only and reused for validation/test data.
- The primary TDA path is `Takens delay embedding -> Vietoris-Rips PH -> H0/H1 -> persistence image/statistics`.
- The secondary TDA path is `1D sublevel-set PH -> H0 -> persistence image/statistics`.
- Large datasets and generated artifacts are excluded from Git via `.gitignore`.
