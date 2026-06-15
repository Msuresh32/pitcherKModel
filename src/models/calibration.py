from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import TARGETS


def build_calibration(
    predictions: pd.DataFrame,
    residual_std_multiplier: float = 1.25,
    edge_shrink_factor: float = 0.5,
) -> dict[str, Any]:
    markets: dict[str, Any] = {}
    for target in TARGETS:
        pred_col = f"{target}_projection"
        if pred_col not in predictions.columns:
            continue
        residual = predictions[target] - predictions[pred_col]
        markets[target] = {
            "bias": float(residual.mean()),
            "mae": float(residual.abs().mean()),
            "rmse": float((residual.pow(2).mean()) ** 0.5),
            "residual_std": float(residual.std(ddof=0)),
            "conservative_std": float(residual.std(ddof=0) * residual_std_multiplier),
            "rows": int(residual.notna().sum()),
        }
    return {
        "edge_shrink_factor": float(edge_shrink_factor),
        "residual_std_multiplier": float(residual_std_multiplier),
        "markets": markets,
    }


def save_calibration(calibration: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return path


def load_calibration(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def residual_std_from_calibration(
    config: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, float]:
    residual_std = dict(config["betting"]["default_residual_std"])
    for market, values in calibration.get("markets", {}).items():
        if values.get("conservative_std"):
            residual_std[market] = float(values["conservative_std"])
    return residual_std


def edge_shrink_from_calibration(config: dict[str, Any], calibration: dict[str, Any]) -> float:
    return float(
        calibration.get(
            "edge_shrink_factor",
            config.get("betting", {}).get("edge_shrink_factor", 1.0),
        )
    )


def bias_corrections_from_calibration(
    config: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, float]:
    """Return per-market bias correction values (mean actual - mean prediction).

    Only populated when config betting.apply_bias_correction is true.
    Adding bias to a projection shifts it toward the observed mean.
    """
    if not config.get("betting", {}).get("apply_bias_correction", False):
        return {}
    corrections: dict[str, float] = {}
    for market, values in calibration.get("markets", {}).items():
        b = values.get("bias")
        if b is not None:
            corrections[market] = float(b)
    return corrections
