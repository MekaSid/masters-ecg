from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.beat_extraction import BeatWindow, extract_beats, save_beats
from src.data.load_ecg import load_annotations, load_record
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract beat-level samples from MIT-BIH records.")
    parser.add_argument("--records", nargs="+", default=["100"], help="MIT-BIH record IDs to preprocess.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_configs()["data"]
    beat_cfg = config["data"]["beat_window"]
    window = BeatWindow(pre_samples=beat_cfg["pre_samples"], post_samples=beat_cfg["post_samples"])

    mitdb_dir = PROJECT_ROOT / config["paths"]["mitdb_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_clean_dir"]
    unknown_class = config["labels"]["unknown_class"]

    for record_id in args.records:
        record = load_record(mitdb_dir, record_id)
        ann = load_annotations(mitdb_dir, record_id)
        beats = extract_beats(
            signal=record.signal,
            annotation_samples=ann.samples,
            annotation_symbols=ann.symbols,
            record_id=record_id,
            window=window,
            unknown_class=unknown_class,
        )
        save_beats(
            beats,
            metadata_path=processed_dir / f"{record_id}_beats.csv",
            waveforms_path=processed_dir / f"{record_id}_beats.npz",
        )
        print(f"Saved {len(beats)} beats for record {record_id}.")


if __name__ == "__main__":
    main()
