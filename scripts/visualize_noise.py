from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_ecg import load_noise_record
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT, ensure_dir
from src.utils.plotting import save_noise_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a labeled PNG from an NSTDB noise recording.")
    parser.add_argument("--noise-type", choices=["bw", "ma", "em"], default="ma", help="NSTDB noise recording.")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based noise channel index.")
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
    noise = load_noise_record(PROJECT_ROOT / config["paths"]["nstdb_dir"], args.noise_type, channel=args.channel)
    start_sample = round(args.start_seconds * noise.fs)
    end_sample = start_sample + round(args.duration_seconds * noise.fs)
    if end_sample > len(noise.signal):
        raise ValueError(f"Requested window exceeds the {args.noise_type} recording length.")

    if args.output is None:
        output_dir = ensure_dir(PROJECT_ROOT / config["paths"]["results_dir"] / "noise")
        output_path = output_dir / f"nstdb_{args.noise_type}_{noise.sig_name[args.channel]}_{args.start_seconds:g}s_{args.duration_seconds:g}s.png"
    else:
        output_path = args.output

    save_noise_window(
        signal=noise.signal[start_sample:end_sample],
        sampling_rate=noise.fs,
        noise_type=args.noise_type,
        channel_name=noise.sig_name[args.channel],
        start_sample=start_sample,
        output_path=output_path,
    )
    print(f"Saved labeled NSTDB noise PNG: {output_path}")


if __name__ == "__main__":
    main()
