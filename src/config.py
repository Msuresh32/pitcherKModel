from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    for key in ["raw_dir", "processed_dir", "exports_dir", "odds_dir"]:
        Path(data_cfg[key]).mkdir(parents=True, exist_ok=True)
    Path(data_cfg["processed_dir"], "models").mkdir(parents=True, exist_ok=True)
