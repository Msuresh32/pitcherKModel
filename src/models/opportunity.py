from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
from src.models.train import _make_model

OPPORTUNITY_TARGETS = ["innings_pitched", "pitches", "batters_faced"]


def available_opportunity_targets(df: pd.DataFrame) -> list[str]:
    targets = []
    for target in OPPORTUNITY_TARGETS:
        if target not in df.columns:
            continue
        values = pd.to_numeric(df[target], errors="coerce")
        if values.notna().any():
            targets.append(target)
    return targets


def train_opportunity_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_dir: str | Path,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    # Opportunity targets (IP, pitches, batters faced) are continuous — always use RF
    # regardless of the configured model_type for the main prop models
    _opp_model_type = "random_forest"
    x_raw = train_df[feature_cols]
    x = (
        x_raw
        .replace([np.inf, -np.inf], np.nan)
        .fillna(x_raw.median(numeric_only=True).fillna(0.0))
        .fillna(0.0)
    )

    for target in available_opportunity_targets(train_df):
        y = pd.to_numeric(train_df[target], errors="coerce")
        mask = y.notna()
        if mask.sum() < 100:
            continue
        model = _make_model(_opp_model_type, random_state)
        model.fit(x.loc[mask], y.loc[mask])
        preds = model.predict(x.loc[mask])
        metrics[target] = {
            "mae": float(mean_absolute_error(y.loc[mask], preds)),
            "rmse": float(np.sqrt(mean_squared_error(y.loc[mask], preds))),
            "rows": int(mask.sum()),
        }
        joblib.dump(
            {"model": model, "feature_cols": feature_cols, "target": target},
            model_dir / f"expected_{target}.joblib",
        )

    return metrics


def load_opportunity_models(model_dir: str | Path) -> dict[str, dict[str, Any]]:
    model_dir = Path(model_dir)
    models = {}
    for target in OPPORTUNITY_TARGETS:
        path = model_dir / f"expected_{target}.joblib"
        if path.exists():
            models[target] = joblib.load(path)
    return models


def add_expected_opportunity_features(
    df: pd.DataFrame,
    models: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added = []
    for target, bundle in models.items():
        cols = bundle["feature_cols"]
        x = (
            out[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(out[cols].median(numeric_only=True).fillna(0.0))
            .fillna(0.0)
        )
        col = f"expected_{target}"
        out[col] = bundle["model"].predict(x)
        added.append(col)
    return out, added
