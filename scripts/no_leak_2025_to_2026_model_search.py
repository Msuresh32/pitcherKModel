from __future__ import annotations

from dataclasses import dataclass
from math import erf, floor, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EDGE_FILE = ROOT / "data" / "processed_noopp_wf2025_ext" / "bt_noopp_oos_edges.csv"
OUT_DIR = ROOT / "reports" / "no_leak_2025_to_2026"


IDENTIFIER_COLS = {
    "game_date",
    "game_pk",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "market",
    "player_name",
    "fetched_at",
    "over_bookmaker",
    "under_bookmaker",
}
ACTUAL_OUTCOME_COLS = {
    "strikeouts",
    "walks",
    "hits_allowed",
    "innings_pitched",
    "pitches",
    "strikes",
    "batters_faced",
}
BETTING_OUTPUT_PREFIXES = (
    "raw_",
    "over_",
    "under_",
    "fair_",
)
BETTING_OUTPUT_COLS = {
    "line",
    "best_side",
    "ev",
    "edge_pct",
    "kelly_fraction",
    "strikeouts_projection",
    "walks_projection",
    "hits_allowed_projection",
}


def american_to_decimal(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(odds > 0, 1 + odds / 100.0, 1 + 100.0 / odds.abs()), index=odds.index)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def is_feature_col(col: str, numeric_cols: set[str]) -> bool:
    if col not in numeric_cols:
        return False
    if col in IDENTIFIER_COLS or col in ACTUAL_OUTCOME_COLS or col in BETTING_OUTPUT_COLS:
        return False
    if col == "league_k":
        return False
    if col.startswith("expected_"):
        return False
    if col.endswith("_projection"):
        return False
    if col.endswith("_probability"):
        return False
    if col.endswith("_ev"):
        return False
    if col.endswith("_odds"):
        return False
    if col in {"over_odds", "under_odds"}:
        return False
    if any(col.startswith(prefix) for prefix in BETTING_OUTPUT_PREFIXES):
        return False
    return True


def add_betting_frame(df: pd.DataFrame, projection: np.ndarray, residual_std: float, label: str) -> pd.DataFrame:
    out = df.copy()
    out["projection"] = np.maximum(projection, 0.05)
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["strikeouts"] = pd.to_numeric(out["strikeouts"], errors="coerce")
    out["over_odds"] = pd.to_numeric(out["over_odds"], errors="coerce")
    out["under_odds"] = pd.to_numeric(out["under_odds"], errors="coerce")
    k = np.floor(out["line"].to_numpy(dtype=float)).astype(int)
    out["over_probability"] = poisson.sf(k, out["projection"].to_numpy(dtype=float))
    out["under_probability"] = 1.0 - out["over_probability"]
    over_dec = american_to_decimal(out["over_odds"])
    under_dec = american_to_decimal(out["under_odds"])
    out["over_ev"] = out["over_probability"] * (over_dec - 1.0) - (1.0 - out["over_probability"])
    out["under_ev"] = out["under_probability"] * (under_dec - 1.0) - (1.0 - out["under_probability"])
    out["side"] = np.where(out["over_ev"] >= out["under_ev"], "over", "under")
    out["edge_pct"] = np.where(out["side"] == "over", out["over_ev"], out["under_ev"]) * 100.0
    out["signed_gap"] = out["projection"] - out["line"]
    out["abs_gap"] = out["signed_gap"].abs()
    out["edge_gap_product"] = out["edge_pct"] * out["abs_gap"]
    out["norm_gap"] = out["abs_gap"] / out["line"].clip(lower=0.5)
    out["bet_odds"] = np.where(out["side"] == "over", out["over_odds"], out["under_odds"])
    out["decimal_odds"] = american_to_decimal(pd.Series(out["bet_odds"], index=out.index))
    out["won"] = np.where(out["side"] == "over", out["strikeouts"] > out["line"], out["strikeouts"] < out["line"])
    out["push"] = out["strikeouts"] == out["line"]
    out["profit_unit"] = np.where(out["push"], 0.0, np.where(out["won"], out["decimal_odds"] - 1.0, -1.0))
    out["model"] = label
    out["residual_std_train"] = residual_std
    return out


def dedupe_bets(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_date", "pitcher_id", "market", "line", "side"]
    return (
        df.sort_values(["edge_pct", "abs_gap"], ascending=[False, False])
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )


def summarize(df: pd.DataFrame) -> dict:
    d = df.copy()
    n = len(d)
    if n == 0:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "profit_units": 0.0,
        }
    wins = int((d["won"] & ~d["push"]).sum())
    losses = int((~d["won"] & ~d["push"]).sum())
    x = d["profit_unit"].to_numpy(dtype=float)
    se = np.nanstd(x, ddof=1) / sqrt(n) if n > 1 else np.nan
    z = np.nanmean(x) / se if se and np.isfinite(se) and se > 0 else np.nan
    ci = bootstrap_ci(x)
    return {
        "bets": n,
        "wins": wins,
        "losses": losses,
        "pushes": int(d["push"].sum()),
        "win_rate": wins / (wins + losses) if wins + losses else np.nan,
        "roi": float(np.nanmean(x)),
        "profit_units": float(np.nansum(x)),
        "roi_ci_lo": ci[0],
        "roi_ci_mid": ci[1],
        "roi_ci_hi": ci[2],
        "p_roi_gt_0": 1 - norm_cdf(z) if np.isfinite(z) else np.nan,
        "avg_edge_pct": float(d["edge_pct"].mean()),
        "avg_abs_gap": float(d["abs_gap"].mean()),
        "avg_odds": float(pd.to_numeric(d["bet_odds"], errors="coerce").mean()),
        "overs": int(d["side"].eq("over").sum()),
        "unders": int(d["side"].eq("under").sum()),
        "brier": brier(d),
        "log_loss": log_loss_score(d),
    }


def bootstrap_ci(values: np.ndarray, n_iter: int = 3000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_iter, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.5, 0.975]))


def brier(df: pd.DataFrame) -> float:
    y = (df["strikeouts"] > df["line"]).astype(float)
    p = pd.to_numeric(df["over_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    return float(((p - y) ** 2).mean())


def log_loss_score(df: pd.DataFrame) -> float:
    y = (df["strikeouts"] > df["line"]).astype(float)
    p = pd.to_numeric(df["over_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    return float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())


def filter_strategy(df: pd.DataFrame, edge_min: float, eg_min: float, side: str = "all") -> pd.DataFrame:
    q = df[(df["edge_pct"] >= edge_min) & (df["edge_gap_product"] >= eg_min)].copy()
    if side == "over":
        q = q[q["side"] == "over"]
    elif side == "under":
        q = q[q["side"] == "under"]
    return dedupe_bets(q)


@dataclass
class FittedModel:
    name: str
    estimator: object | None
    features: list[str]
    residual_std: float
    train_mae: float
    train_rmse: float


def top_corr_features(train: pd.DataFrame, features: list[str], k: int) -> list[str]:
    y = pd.to_numeric(train["strikeouts"], errors="coerce")
    scores = []
    for col in features:
        x = pd.to_numeric(train[col], errors="coerce")
        if x.notna().sum() < 50 or x.nunique(dropna=True) <= 1:
            continue
        corr = x.corr(y)
        if pd.notna(corr):
            scores.append((abs(float(corr)), col))
    scores.sort(reverse=True)
    return [col for _, col in scores[:k]]


def fit_model(name: str, train: pd.DataFrame, features: list[str], estimator) -> FittedModel:
    x = train[features]
    y = train["strikeouts"].astype(float).clip(lower=0)
    if estimator is None:
        pred = train[name].astype(float).clip(lower=0.05).to_numpy()
        residual = y.to_numpy() - pred
        return FittedModel(name=name, estimator=None, features=[name], residual_std=float(np.std(residual)), train_mae=float(np.mean(np.abs(residual))), train_rmse=float(np.sqrt(np.mean(residual**2))))
    estimator.fit(x, y)
    pred = np.maximum(estimator.predict(x), 0.05)
    residual = y.to_numpy() - pred
    return FittedModel(
        name=name,
        estimator=estimator,
        features=features,
        residual_std=float(np.std(residual)),
        train_mae=float(mean_absolute_error(y, pred)),
        train_rmse=float(np.sqrt(mean_squared_error(y, pred))),
    )


def predict_model(model: FittedModel, df: pd.DataFrame) -> np.ndarray:
    if model.estimator is None:
        return df[model.name].astype(float).clip(lower=0.05).to_numpy()
    return np.maximum(model.estimator.predict(df[model.features]), 0.05)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(EDGE_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[raw["market"] == "strikeouts"].copy()
    numeric_cols = set(raw.select_dtypes(include=[np.number]).columns)
    all_features = [col for col in raw.columns if is_feature_col(col, numeric_cols)]

    # One training row per pitcher-game to prevent alt-line multiplicity from
    # changing the projection model fit.
    train_games = (
        raw[raw["game_date"].dt.year == 2025]
        .sort_values("game_date")
        .drop_duplicates(["game_date", "pitcher_id"], keep="first")
        .reset_index(drop=True)
    )
    train_inner = train_games[train_games["game_date"] < "2025-07-01"].copy()
    val_inner = raw[(raw["game_date"].dt.year == 2025) & (raw["game_date"] >= "2025-07-01")].copy()
    test_2026 = raw[raw["game_date"].dt.year == 2026].copy()

    top60 = top_corr_features(train_inner, all_features, 60)
    top120 = top_corr_features(train_inner, all_features, 120)

    model_specs = [
        ("p_strikeouts_roll20", None, ["p_strikeouts_roll20"]),
        ("p_strikeouts_career_avg_prior", None, ["p_strikeouts_career_avg_prior"]),
        (
            "poisson_top60_a1",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", PoissonRegressor(alpha=1.0, max_iter=2000))]),
            top60,
        ),
        (
            "poisson_top120_a2",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", PoissonRegressor(alpha=2.0, max_iter=2000))]),
            top120,
        ),
        (
            "ridge_top120",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=50.0))]),
            top120,
        ),
        (
            "elastic_top120",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.15, max_iter=5000))]),
            top120,
        ),
        (
            "hgb_top120",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, l2_regularization=1.0, min_samples_leaf=30, random_state=42))]),
            top120,
        ),
        (
            "rf_top120",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=42))]),
            top120,
        ),
    ]

    validation_rows = []
    test_rows = []
    prediction_scores = []
    candidate_predictions = {}

    for name, estimator, features in model_specs:
        if not features:
            continue
        fitted_inner = fit_model(name, train_inner, features, estimator)
        val_pred = predict_model(fitted_inner, val_inner)
        val_bets = add_betting_frame(val_inner, val_pred, fitted_inner.residual_std, name)
        candidate_predictions[(name, "val")] = val_bets

        # Hyperparameter/filter evaluation is only on 2025 validation.
        for edge in [0, 3, 5, 7, 10, 12, 15]:
            for eg in [0, 3, 5, 6, 8, 10, 12]:
                for side in ["all", "over", "under"]:
                    q = filter_strategy(val_bets, edge, eg, side)
                    if len(q) < 40:
                        continue
                    row = summarize(q)
                    row.update({"model": name, "edge_min": edge, "eg_min": eg, "side_filter": side, "sample": "2025_val"})
                    validation_rows.append(row)

        # Final frozen model: train on all 2025 pitcher-games, test 2026.
        fitted_final = fit_model(name, train_games, features, estimator)
        pred2026 = predict_model(fitted_final, test_2026)
        test_bets = add_betting_frame(test_2026, pred2026, fitted_final.residual_std, name)
        candidate_predictions[(name, "2026")] = test_bets
        prediction_scores.append(
            {
                "model": name,
                "train_rows": len(train_games),
                "features": len(features),
                "train_mae": fitted_final.train_mae,
                "train_rmse": fitted_final.train_rmse,
                "test2026_mae": mean_absolute_error(test_2026["strikeouts"], pred2026),
                "test2026_rmse": sqrt(mean_squared_error(test_2026["strikeouts"], pred2026)),
            }
        )

    val_df = pd.DataFrame(validation_rows)
    val_df.to_csv(OUT_DIR / "validation_2025_filter_grid.csv", index=False)
    pd.DataFrame(prediction_scores).to_csv(OUT_DIR / "prediction_scores.csv", index=False)

    # Select only configurations that would have looked acceptable using 2025 validation.
    eligible = val_df[
        (val_df["bets"] >= 100)
        & (val_df["roi"] > 0.03)
        & (val_df["roi_ci_lo"] > -0.02)
        & (val_df["p_roi_gt_0"] < 0.10)
    ].copy()
    eligible = eligible.sort_values(["profit_units", "bets"], ascending=[False, False])
    eligible.to_csv(OUT_DIR / "eligible_from_2025_validation.csv", index=False)

    selected_specs = eligible.head(50)[["model", "edge_min", "eg_min", "side_filter"]].drop_duplicates()
    for _, spec in selected_specs.iterrows():
        bets = candidate_predictions[(spec["model"], "2026")]
        q = filter_strategy(bets, float(spec["edge_min"]), float(spec["eg_min"]), str(spec["side_filter"]))
        row = summarize(q)
        row.update(spec.to_dict())
        row["sample"] = "2026_test"
        test_rows.append(row)
        if not q.empty:
            q["edge_min"] = spec["edge_min"]
            q["eg_min"] = spec["eg_min"]
            q["side_filter"] = spec["side_filter"]
            q.to_csv(OUT_DIR / f"bets_2026_{spec['model']}_e{spec['edge_min']}_eg{spec['eg_min']}_{spec['side_filter']}.csv", index=False)

    test_df = pd.DataFrame(test_rows).sort_values(["roi", "bets"], ascending=[False, False])
    test_df.to_csv(OUT_DIR / "test_2026_for_eligible_2025_configs.csv", index=False)

    # Month and side breakdown for the top 2026-tested candidates.
    breakdown_rows = []
    for _, spec in test_df.head(10).iterrows():
        bets = candidate_predictions[(spec["model"], "2026")]
        q = filter_strategy(bets, float(spec["edge_min"]), float(spec["eg_min"]), str(spec["side_filter"]))
        q["month"] = pd.to_datetime(q["game_date"]).dt.to_period("M").astype(str)
        for month, group in q.groupby("month"):
            row = summarize(group)
            row.update({"model": spec["model"], "edge_min": spec["edge_min"], "eg_min": spec["eg_min"], "side_filter": spec["side_filter"], "breakdown": month})
            breakdown_rows.append(row)
        for side, group in q.groupby("side"):
            row = summarize(group)
            row.update({"model": spec["model"], "edge_min": spec["edge_min"], "eg_min": spec["eg_min"], "side_filter": spec["side_filter"], "breakdown": side})
            breakdown_rows.append(row)
    pd.DataFrame(breakdown_rows).to_csv(OUT_DIR / "top_candidate_2026_breakdowns.csv", index=False)

    print(f"Wrote no-leak model search outputs to {OUT_DIR}")
    print("Prediction scores:")
    print(pd.DataFrame(prediction_scores).round(4).to_string(index=False))
    print("\nTop eligible 2025 configs tested on 2026:")
    print(test_df.head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
