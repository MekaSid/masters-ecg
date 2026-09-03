from __future__ import annotations

import argparse

import numpy as np

from src.data.beat_extraction import BeatWindow, extract_beats
from src.data.load_ecg import load_annotations, load_noise_record, load_record
from src.data.noise import NoiseAugmentor
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a noisy ECG beat example.")
    parser.add_argument("--record", default="100", help="MIT-BIH record ID.")
    parser.add_argument("--beat-index", type=int, default=0, help="Index within extracted beats.")
    parser.add_argument("--noise-type", default="ma", choices=["bw", "ma", "em"], help="NSTDB noise type.")
    parser.add_argument("--snr-db", type=float, default=12.0, help="Target SNR in dB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_configs()["data"]
    beat_cfg = config["data"]["beat_window"]
    window = BeatWindow(pre_samples=beat_cfg["pre_samples"], post_samples=beat_cfg["post_samples"])

    mitdb_dir = REPO_ROOT / config["paths"]["mitdb_dir"]
    nstdb_dir = REPO_ROOT / config["paths"]["nstdb_dir"]
    output_dir = ensure_dir(REPO_ROOT / config["paths"]["processed_noisy_dir"])

    record = load_record(mitdb_dir, args.record)
    ann = load_annotations(mitdb_dir, args.record)
    beats = extract_beats(record.signal, ann.samples, ann.symbols, args.record, window)
    beat = beats[args.beat_index]

    noise_record = load_noise_record(nstdb_dir, args.noise_type)
    augmentor = NoiseAugmentor(seed=config["noise"]["seed"])
    noisy = augmentor.add_noise(beat.waveform, noise_record.signal, args.noise_type, args.snr_db)

    output_path = output_dir / f"{args.record}_beat{args.beat_index}_{args.noise_type}_{int(args.snr_db)}db.npz"
    np.savez_compressed(
        output_path,
        clean=noisy.clean,
        noisy=noisy.noisy,
        noise=noisy.noise,
        snr_db_target=noisy.snr_db_target,
        snr_db_achieved=noisy.snr_db_achieved,
    )
    print(f"Saved noisy beat to {output_path}. Achieved SNR: {noisy.snr_db_achieved:.2f} dB")


if __name__ == "__main__":
    main()
