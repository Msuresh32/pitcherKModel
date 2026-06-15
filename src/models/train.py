from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.schema import TARGETS


def _make_model(model_type: str, random_state: int, alpha: float | None = None):
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install xgboost or set training.model_type=random_forest") from exc
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )

    if model_type == "xgboost_poisson":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install xgboost") from exc
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="count:poisson",
            random_state=random_state,
        )

    if model_type == "poisson":
        return PoissonRegressor(max_iter=2000, alpha=alpha if alpha is not None else 1e-4)

    if model_type == "tweedie":
        return TweedieRegressor(power=1.5, max_iter=2000, alpha=alpha if alpha is not None else 1e-4)

    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=8,
        random_state=random_state,
        n_jobs=-1,
    )


def _is_count_model(model_type: str) -> bool:
    return model_type in {"poisson", "tweedie", "xgboost_poisson"}


def _train_ensemble(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_dir: Path,
    random_state: int,
    per_target_alpha: dict[str, float] | None,
) -> dict[str, dict[str, Any]]:
    """Train Poisson GLM + XGBoost Poisson for each target, blend 60/40."""
    blend_weights = [0.6, 0.4]
    metrics: dict[str, dict[str, Any]] = {}

    x = (
        train_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_df[feature_cols].median(numeric_only=True).fillna(0.0))
        .fillna(0.0)
    )
    for target in TARGETS:
        y = train_df[target].astype(float).clip(lower=0)
        target_alpha = (per_target_alpha or {}).get(target, 1e-4)

        poisson_model = _make_model("poisson", random_state, alpha=target_alpha)
        poisson_pipeline = Pipeline([("scaler", StandardScaler()), ("model", poisson_model)])
        poisson_pipeline.fit(x, y)

        xgb_model = _make_model("xgboost_poisson", random_state)
        xgb_pipeline = Pipeline([("scaler", StandardScaler()), ("model", xgb_model)])
        xgb_pipeline.fit(x, y)

        pois_preds = np.maximum(poisson_pipeline.predict(x), 0)
        xgb_preds = np.maximum(xgb_pipeline.predict(x), 0)
        preds = blend_weights[0] * pois_preds + blend_weights[1] * xgb_preds

        metrics[target] = {
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "rows": int(len(train_df)),
        }
        joblib.dump(
            {
                "ensemble": True,
                "poisson_pipeline": poisson_pipeline,
                "xgb_pipeline": xgb_pipeline,
                "feature_cols": feature_cols,
                "target": target,
                "model_type": "ensemble",
                "blend_weights": blend_weights,
            },
            model_dir / f"{target}.joblib",
        )

    return metrics


def train_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_dir: str | Path,
    model_type: str = "random_forest",
    random_state: int = 42,
    alpha: float | None = None,
    per_target_alpha: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "ensemble":
        return _train_ensemble(train_df, feature_cols, model_dir, random_state, per_target_alpha)

    metrics: dict[str, dict[str, Any]] = {}

    x = (
        train_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_df[feature_cols].median(numeric_only=True).fillna(0.0))
        .fillna(0.0)
    )
    for target in TARGETS:
        y = train_df[target].astype(float)
        if _is_count_model(model_type):
            y = y.clip(lower=0)
        target_alpha = (per_target_alpha or {}).get(target, alpha)
        model = _make_model(model_type, random_state, alpha=target_alpha)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipeline.fit(x, y)
        preds = pipeline.predict(x)
        if _is_count_model(model_type):
            preds = np.maximum(preds, 0)
        metrics[target] = {
            "mae": float(mean_absolute_error(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "rows": int(len(train_df)),
        }
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_cols": feature_cols,
                "target": target,
                "model_type": model_type,
            },
            model_dir / f"{target}.joblib",
        )

    return metrics


def load_models(model_dir: str | Path) -> dict[str, dict[str, Any]]:
    model_dir = Path(model_dir)
    models = {}
    for target in TARGETS:
        path = model_dir / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model: {path}")
        models[target] = joblib.load(path)
    return models


def predict_targets(df: pd.DataFrame, models: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    for target, bundle in models.items():
        feature_cols = bundle["feature_cols"]
        x = out[feature_cols]
        if bundle.get("ensemble"):
            w = bundle.get("blend_weights", [0.6, 0.4])
            pois_preds = np.maximum(bundle["poisson_pipeline"].predict(x), 0)
            xgb_preds = np.maximum(bundle["xgb_pipeline"].predict(x), 0)
            preds = w[0] * pois_preds + w[1] * xgb_preds
        else:
            preds = bundle["pipeline"].predict(x)
            model_type = bundle.get("model_type", "random_forest")
            if _is_count_model(model_type):
                preds = np.maximum(preds, 0)
        out[f"{target}_projection"] = preds
    return out


def temporal_cross_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 4,
    model_type: str = "random_forest",
    random_state: int = 42,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Walk-forward (expanding window) cross-validation.

    Each fold trains on all data before the fold window and tests on the fold window.
    Returns out-of-sample MAE and RMSE per fold per target.
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    # Ensemble CV uses Poisson as a fast approximation of the blended model
    cv_model_type = "poisson" if model_type == "ensemble" else model_type
    n = len(df)
    # Reserve at least n//(n_splits+1) rows per fold
    fold_size = max(n // (n_splits + 1), 50)

    results = []
    for fold in range(n_splits):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            break

        train_split = df.iloc[:train_end]
        test_split = df.iloc[test_start:test_end]

        # Compute fill values from this fold's training data; fallback to 0 for all-NaN columns
        fold_fills = train_split[feature_cols].median(numeric_only=True).fillna(0.0)

        x_train = (
            train_split[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fold_fills)
            .fillna(0.0)
        )
        x_test = (
            test_split[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fold_fills)
            .fillna(0.0)
        )

        for target in TARGETS:
            y_train = train_split[target].astype(float)
            y_test = test_split[target].astype(float)
            if _is_count_model(cv_model_type):
                y_train = y_train.clip(lower=0)

            model = _make_model(cv_model_type, random_state, alpha=alpha)
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipeline.fit(x_train, y_train)
            preds = pipeline.predict(x_test)
            if _is_count_model(cv_model_type):
                preds = np.maximum(preds, 0)

            results.append(
                {
                    "fold": fold,
                    "target": target,
                    "train_rows": len(train_split),
                    "test_rows": len(test_split),
                    "train_date_max": str(train_split["game_date"].max().date()),
                    "test_date_min": str(test_split["game_date"].min().date()),
                    "test_date_max": str(test_split["game_date"].max().date()),
                    "mae": float(mean_absolute_error(y_test, preds)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                }
            )

    return pd.DataFrame(results)


def select_top_features(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    top_k: int,
    random_state: int = 42,
) -> list[str]:
    """Rank features by mean Random Forest importance across all targets.

    Returns the top_k feature names. Using RF importance (not permutation importance)
    is fast and good enough for pruning clearly low-signal features before GLM fitting.
    """
    if top_k >= len(feature_cols):
        return feature_cols

    x = (
        train_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_df[feature_cols].median(numeric_only=True).fillna(0.0))
        .fillna(0.0)
    )

    importances = np.zeros(len(feature_cols))
    for target in TARGETS:
        y = train_df[target].astype(float).clip(lower=0)
        rf = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=8, random_state=random_state, n_jobs=-1
        )
        rf.fit(x, y)
        importances += rf.feature_importances_

    importances /= len(TARGETS)
    ranked = sorted(zip(feature_cols, importances), key=lambda t: t[1], reverse=True)
    selected = [name for name, _ in ranked[:top_k]]
    print(f"Feature selection: kept {top_k} of {len(feature_cols)} features.")
    return selected


def save_fill_values(fill_values: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy scalars to plain floats for JSON serialisation
    clean = {k: (float(v) if pd.notna(v) else 0.0) for k, v in fill_values.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def load_fill_values(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
