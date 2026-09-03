from __future__ import annotations

from pathlib import Path
from typing import Iterable

import wfdb


def download_physionet_database(
    db_name: str,
    output_dir: Path,
    records: Iterable[str] | None = None,
) -> None:
    """Download a PhysioNet WFDB database or selected records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(db_name, dl_dir=str(output_dir), records=list(records) if records else None)


def download_default_datasets(mitdb_dir: Path, nstdb_dir: Path) -> None:
    """Download the minimal datasets required for the first milestone."""
    download_physionet_database("mitdb", mitdb_dir, records=["100"])
    download_physionet_database("nstdb", nstdb_dir, records=["bw", "ma", "em"])
