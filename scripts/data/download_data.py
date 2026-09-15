from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.download import download_physionet_database
from src.utils.config import load_project_configs
from src.utils.paths import REPO_ROOT as PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia and NSTDB data.")
    parser.add_argument("--mitdb-records", nargs="*", default=["100"], help="MIT-BIH record IDs to download.")
    parser.add_argument("--nstdb-records", nargs="*", default=["bw", "ma", "em"], help="NSTDB noise record IDs.")
    parser.add_argument("--skip-nstdb", action="store_true", help="Download only MIT-BIH records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_configs()["data"]
    paths = config["paths"]

    mitdb_dir = PROJECT_ROOT / paths["mitdb_dir"]
    nstdb_dir = PROJECT_ROOT / paths["nstdb_dir"]

    download_physionet_database("mitdb", mitdb_dir, args.mitdb_records)
    if not args.skip_nstdb:
        download_physionet_database("nstdb", nstdb_dir, args.nstdb_records)


if __name__ == "__main__":
    main()
