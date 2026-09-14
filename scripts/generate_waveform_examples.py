from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_ecg import load_annotations, load_record
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.plotting import save_annotated_ecg_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple labeled ECG waveform PNG examples.")
    parser.add_argument("--record", default="100", help="MIT-BIH record ID.")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based ECG channel index.")
    parser.add_argument("--count", type=int, default=10, help="Number of PNG windows to generate.")
    parser.add_argument("--duration-seconds", type=float, default=5.0, help="Duration of each waveform window.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1.")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive.")

    config = load_project_configs()["data"]
    record = load_record(PROJECT_ROOT / config["paths"]["mitdb_dir"], args.record, channel=args.channel)
    annotations = load_annotations(PROJECT_ROOT / config["paths"]["mitdb_dir"], args.record)
    duration_samples = round(args.duration_seconds * record.fs)
    if duration_samples > len(record.signal):
        raise ValueError("Requested duration exceeds the recording length.")

    output_dir = args.output_dir or ensure_dir(
        PROJECT_ROOT / config["paths"]["results_dir"] / "waveforms" / f"record_{args.record}_examples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start_samples = np.linspace(0, len(record.signal) - duration_samples, num=args.count, dtype=int)

    for example_index, start_sample in enumerate(start_samples, start=1):
        end_sample = start_sample + duration_samples
        start_seconds = start_sample / record.fs
        output_path = output_dir / (
            f"example_{example_index:02d}_mitdb_{args.record}_{record.sig_name[args.channel]}_{start_seconds:.1f}s.png"
        )
        save_annotated_ecg_window(
            signal=record.signal[start_sample:end_sample],
            sampling_rate=record.fs,
            channel_name=record.sig_name[args.channel],
            record_id=args.record,
            start_sample=start_sample,
            annotation_samples=annotations.samples,
            annotation_symbols=annotations.symbols,
            output_path=output_path,
        )
        print(f"Saved example {example_index}/{args.count}: {output_path}")


if __name__ == "__main__":
    main()
