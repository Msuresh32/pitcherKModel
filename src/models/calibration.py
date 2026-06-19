from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

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


def build_probability_calibration(
    scored: pd.DataFrame,
    min_rows: int = 200,
    method: str = "logit",
    regularization_c: float = 1.0,
) -> dict[str, Any]:
    """Fit serializable calibrators for over probabilities.

    The calibrator maps the model's over_probability to the empirical chance
    that the market goes over. It is intentionally stored as plain JSON instead
    of a pickled sklearn object so calibration.json remains portable.
    """
    if method not in {"logit", "isotonic"}:
        raise ValueError("method must be 'logit' or 'isotonic'")

    try:
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
        else:
            from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {}

    markets: dict[str, Any] = {}
    for target in TARGETS:
        if target not in scored.columns:
            continue
        sub = scored[scored.get("market", target) == target].copy()
        if sub.empty or "over_probability" not in sub.columns or "line" not in sub.columns:
            continue
        sub = sub.dropna(subset=["over_probability", "line", target])
        if len(sub) < min_rows:
            continue

        x = pd.to_numeric(sub["over_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
        y = (sub[target] > sub["line"]).astype(float)
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        if len(x) < min_rows or y.nunique() < 2:
            continue

        if method == "isotonic":
            model = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
            model.fit(x.to_numpy(), y.to_numpy())
            calibrated = pd.Series(model.predict(x.to_numpy()), index=x.index)
            spec = {
                "method": "isotonic",
                "x_thresholds": [float(v) for v in model.X_thresholds_],
                "y_thresholds": [float(v) for v in model.y_thresholds_],
            }
        else:
            x_clipped = x.clip(1e-6, 1 - 1e-6)
            logits = np.log(x_clipped / (1 - x_clipped)).to_numpy().reshape(-1, 1)
            model = LogisticRegression(C=regularization_c, solver="lbfgs")
            model.fit(logits, y.to_numpy())
            calibrated = pd.Series(model.predict_proba(logits)[:, 1], index=x.index)
            spec = {
                "method": "logit",
                "coefficient": float(model.coef_[0][0]),
                "intercept": float(model.intercept_[0]),
                "regularization_c": float(regularization_c),
            }
        raw_brier = float(((x - y) ** 2).mean())
        calibrated_brier = float(((calibrated - y) ** 2).mean())

        markets[target] = {
            "source_rows": int(len(x)),
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "mean_raw_over_probability": float(x.mean()),
            "actual_over_rate": float(y.mean()),
            **spec,
        }
    return markets


def probability_calibrators_from_calibration(
    calibration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return per-market probability calibrator specs from calibration.json."""
    out: dict[str, dict[str, Any]] = {}
    for market, values in calibration.get("markets", {}).items():
        spec = values.get("probability_calibration")
        if not spec:
            continue
        if spec.get("method") == "isotonic" and spec.get("x_thresholds") and spec.get("y_thresholds"):
            out[market] = spec
        elif spec.get("method") == "logit" and "coefficient" in spec and "intercept" in spec:
            out[market] = spec
    return out


def apply_probability_calibrator(probability: float, calibrator: dict[str, Any] | None) -> float:
    if calibrator is None or pd.isna(probability):
        return probability
    if calibrator.get("method") != "isotonic":
        if calibrator.get("method") != "logit":
            return probability
        p = min(max(float(probability), 1e-6), 1 - 1e-6)
        z = np.log(p / (1 - p))
        linear = float(calibrator["coefficient"]) * z + float(calibrator["intercept"])
        calibrated = 1 / (1 + np.exp(-linear))
        return min(max(float(calibrated), 1e-6), 1 - 1e-6)
    else:
        x = np.asarray(calibrator.get("x_thresholds", []), dtype=float)
        y = np.asarray(calibrator.get("y_thresholds", []), dtype=float)
        if len(x) == 0 or len(x) != len(y):
            return probability
        calibrated = float(np.interp(float(probability), x, y, left=y[0], right=y[-1]))
    return min(max(calibrated, 1e-6), 1 - 1e-6)
