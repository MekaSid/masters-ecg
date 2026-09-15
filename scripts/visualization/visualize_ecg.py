from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_ecg import load_annotations, load_record
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.plotting import save_annotated_ecg_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a labeled PNG of a window from a MIT-BIH ECG record.")
    parser.add_argument("--record", default="100", help="MIT-BIH record ID.")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based ECG channel index.")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Window start time in seconds.")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Window duration in seconds.")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_seconds < 0:
        raise ValueError("--start-seconds must be non-negative.")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive.")

    config = load_project_configs()["data"]
    mitdb_dir = PROJECT_ROOT / config["paths"]["mitdb_dir"]
    record = load_record(mitdb_dir, args.record, channel=args.channel)
    annotations = load_annotations(mitdb_dir, args.record)

    start_sample = round(args.start_seconds * record.fs)
    duration_samples = round(args.duration_seconds * record.fs)
    end_sample = start_sample + duration_samples
    if end_sample > len(record.signal):
        raise ValueError(
            f"Requested window ends at sample {end_sample}, but record {args.record} contains {len(record.signal)} samples."
        )

    if args.output is None:
        output_dir = ensure_dir(PROJECT_ROOT / config["paths"]["results_dir"] / "waveforms")
        output_path = output_dir / (
            f"mitdb_{args.record}_{record.sig_name[args.channel]}_"
            f"{args.start_seconds:g}s_{args.duration_seconds:g}s.png"
        )
    else:
        output_path = args.output

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
    print(f"Saved labeled ECG waveform PNG: {output_path}")


if __name__ == "__main__":
    main()
