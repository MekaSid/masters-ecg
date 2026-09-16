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
configs/         YAML configuration: data/TDA at root, model runs in models/
data/            Raw, processed, and TDA-derived artifacts
notebooks/       Lightweight exploration notebooks
scripts/         CLI entrypoints grouped into data/, tda/, training/, visualization/
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
python3 scripts/data/download_data.py
```

This downloads:

- `mitdb`, defaulting to record `100` for the first smoke test
- `nstdb` noise records `bw`, `ma`, and `em`

## Preprocess MIT-BIH Beats

```bash
python3 scripts/data/preprocess_mitdb.py --records 100
```

This extracts fixed-length heartbeat windows around annotated beats, preserves original symbols, maps labels into AAMI-style classes, and saves:

- beat metadata to `data/processed/clean/*.csv`
- waveform arrays to `data/processed/clean/*.npz`

## Visualize an ECG Window

```bash
./.venv/bin/python scripts/visualization/visualize_ecg.py --record 100 --channel 0 --start-seconds 0 --duration-seconds 10
```

This saves a labeled PNG under `results/waveforms/`. The graph uses physical ECG amplitude in mV, labels the time axis in seconds, and marks annotated beats from the matching `.atr` file.

Generate ten different examples sampled across the recording:

```bash
./.venv/bin/python scripts/visualization/generate_waveform_examples.py --record 100 --count 10 --duration-seconds 5
```

The individual PNGs are saved under `results/waveforms/record_100_examples/`.

## Visualize NSTDB Noise

```bash
./.venv/bin/python scripts/visualization/visualize_noise.py --noise-type ma --channel 0 --start-seconds 0 --duration-seconds 10
```

This saves a labeled raw-noise PNG under `results/noise/`. Replace `ma` with `bw` for baseline wander or `em` for electrode-motion noise.

### Annotation Symbols in Waveform Plots

The letter above each red marker is the original MIT-BIH expert annotation, not a prediction. Common symbols are `N` (normal), `A` (atrial premature), `V` (premature ventricular), `F` (fusion), and `/` (paced). The preprocessing step preserves these detailed labels and maps them into broader AAMI-style classes in `src/data/labels.py`.

## Run TDA Pipeline

```bash
python3 scripts/tda/compute_tda.py --record 100 --beat-index 0
```

This computes:

- delay embeddings
- H0/H1 persistence diagrams
- persistence images

## Train the Raw ECG Conv1D Baseline

```bash
./.venv/bin/python scripts/training/train_raw_ecg.py
```

The baseline architecture is `ECG beat -> Conv1D -> Conv1D -> Conv1D -> max pooling -> fully connected classifier`.

The record-disjoint split is configured in `configs/models/training.yaml`:

- Train: 18 standard DS1 records
- Validation: 4 held-out DS1 records
- Test: 22 standard DS2 records
- Target classes: `N` (normal), `S` (supraventricular), `V` (ventricular)

The script saves `raw_ecg_conv1d.pt`, `metrics.json`, and `history.csv` under `results/raw_cnn/full_clean_record_disjoint/`.

## Reproduce Zhang et al. Regular CNN

The separate Zhang et al. (2021) reproduction uses the paper's DS1-to-DS2 protocol, dynamic two-lead heartbeat windows, 128-point resampling, mean removal, pre-RR and near-pre-RR ratio feature rows, and the seven-convolution residual-attention encoder. It intentionally excludes the paper's adversarial subject-ID branch.

```bash
./.venv/bin/python scripts/training/train_zhang_regular_cnn.py --run-name zhang_regular_clean
```

Configuration is in `configs/models/zhang_regular_cnn.yaml`. Training uses Adam, validation-loss learning-rate reduction, and validation-loss early stopping. Artifacts are saved under `results/zhang_regular_cnn/zhang_regular_clean/`.

This is a close reproduction rather than a bit-for-bit replication: the publication does not provide source code or attention-module internals, so the implementation uses a documented channel-and-temporal attention block. It also retains all supported beats in the current official MIT-BIH annotations; this produces slightly more beats than the paper's printed class-count table. The paper is [Zhang et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8181174/).

## Train Dindin-Style Betti Models

This controlled PH experiment computes GUDHI H0 sublevel and upper-level persistence on a three-beat lead-I sequence, converts both barcodes into 128-bin Betti curves, and trains PH-only and raw-plus-PH fusion models.

```bash
./.venv/bin/python scripts/training/train_dindin_betti_models.py --run-name dindin_betti_clean
```

The first run caches paired DS1/DS2 Betti curves under `data/tda/betti_curves/`; later runs reuse the cache. Use `--models ph_only` or `--models fusion` to train one branch, and `--rebuild-cache` to regenerate representations. Configuration is in `configs/models/dindin_betti_fusion.yaml`; results are saved under `results/dindin_betti/`.

Run the matched raw-only versus fusion ablation across seeds and generate its PNG summary:

```bash
./.venv/bin/python scripts/training/train_dindin_betti_models.py \
  --models raw_only fusion --seeds 42 43 44 --run-name paired_ablation
./.venv/bin/python scripts/visualization/plot_ablation_results.py \
  --run-dir results/dindin_betti/paired_ablation --seeds 42 43 44
```

This is a controlled adaptation, not a full reproduction of Dindin et al.: their published architecture also includes filtering, autoencoders, FFT, handcrafted features, other databases, and patient-fold cross-validation. The PH-only and fusion branches here isolate the contribution of Betti curves to the Zhang raw/RR model. [Dindin et al., 2020](https://arxiv.org/abs/1906.05795).

## Generate ECG-to-PH Gallery

Generate ten diverse, annotated ECG-to-PH comparisons from the locally downloaded MIT-BIH records:

```bash
./.venv/bin/python scripts/visualization/generate_ph_gallery.py --count 10
```

Each PNG places the raw three-beat ECG context beside the Dindin-style GUDHI H0 persistence barcodes and the sublevel/upper-level Betti curves used by the PH CNN. The default gallery prioritizes distinct non-`N` annotation symbols. Output PNGs and `selection.csv` are saved under `results/ph_gallery/`.

Alternative direct 1D topology baseline:

```bash
python3 scripts/tda/compute_tda.py --record 100 --beat-index 0 --method sublevel
```

## End-to-End Smoke Test

```bash
python3 scripts/tda/test_pipeline.py --record 100 --noise-type ma --snr-db 12
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
