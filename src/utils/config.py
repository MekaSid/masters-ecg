from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIGS_DIR


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_project_configs() -> dict[str, Any]:
    data_cfg = load_yaml(CONFIGS_DIR / "data.yaml")
    tda_cfg = load_yaml(CONFIGS_DIR / "tda.yaml")
    return {"data": data_cfg, "tda": tda_cfg}
