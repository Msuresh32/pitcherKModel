import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.loaders import (
    filter_date_range,
    load_batter_game_logs,
    load_game_context_logs,
    load_park_factors,
    load_pitcher_game_logs,
    load_statcast_pitcher_daily,
    load_statcast_batter_pitch_type_daily,
    load_team_batting_game_logs,
)
from src.features.build_features import build_training_features
from src.odds.pricing import add_betting_columns
from scripts.train_strikeout_bet_selector import FEATURE_CANDIDATES


def _regressor(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _selector(model_type: str, random_state: int) -> Pipeline:
    if model_type == "logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _american_to_decimal(odds: pd.Series) -> pd.Series:
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / odds.abs())


def _prep_odds(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if Path(path).exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No odds files found.")
    odds = pd.concat(frames, ignore_index=True, sort=False)
    odds = odds[odds["market"] == "strikeouts"].copy()
    odds = odds.dropna(subset=["pitcher_id", "line", "over_odds", "under_odds"])
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    odds["pitcher_id"] = pd.to_numeric(odds["pitcher_id"], errors="coerce").astype("Int64").astype(str)
    odds = odds.sort_values(
        [col for col in ["game_date", "pitcher_id", "historical_snapshot", "fetched_at"] if col in odds]
    )
    return odds.drop_duplicates(["game_date", "pitcher_id"], keep="last")


def _train_projection_model(train_df: pd.DataFrame, feature_cols: list[str], random_state: int):
    opportunity = _regressor(random_state)
    opportunity.fit(train_df[feature_cols], train_df["innings_pitched"].astype(float))
    train_aug = train_df.copy()
    train_aug["expected_innings_pitched"] = opportunity.predict(train_df[feature_cols])
    prop_features = feature_cols + ["expected_innings_pitched"]

    model = _regressor(random_state)
    model.fit(train_aug[prop_features], train_aug["strikeouts"].astype(float))
    train_pred = model.predict(train_aug[prop_features])
    residual_std = float((train_aug["strikeouts"].astype(float) - train_pred).std(ddof=0) * 1.25)
    return opportunity, model, prop_features, residual_std


def _candidate_bets(
    featured: pd.DataFrame,
    feature_cols: list[str],
    odds: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    train = filter_date_range(featured, train_start, train_end)
    test = filter_date_range(featured, test_start, test_end)
    opportunity, model, prop_features, residual_std = _train_projection_model(
        train, feature_cols, random_state
    )
    test = test.copy()
    test["expected_innings_pitched"] = opportunity.predict(test[feature_cols])
    test["strikeouts_projection"] = model.predict(test[prop_features])

    merged = test.merge(
        odds,
        on=["game_date", "pitcher_id"],
        how="inner",
        suffixes=("", "_odds"),
    )
    if merged.empty:
        return merged, {"mae": np.nan, "rows": 0}

    scored = add_betting_columns(
        merged,
        market="strikeouts",
        residual_std=residual_std,
        max_kelly_fraction=0.05,
        edge_shrink_factor=0.5,
    )
    scored["best_side"] = np.where(scored["over_ev"] >= scored["under_ev"], "over", "under")
    scored["bet_odds"] = np.where(scored["best_side"] == "over", scored["over_odds"], scored["under_odds"])
    scored["decimal_odds"] = _american_to_decimal(scored["bet_odds"].astype(float))
    scored["won"] = np.where(
        scored["best_side"] == "over",
        scored["strikeouts"] > scored["line"],
        scored["strikeouts"] < scored["line"],
    )
    scored["push"] = scored["strikeouts"] == scored["line"]
    scored["unit_profit"] = np.where(
        scored["push"],
        0.0,
        np.where(scored["won"], scored["decimal_odds"] - 1, -1.0),
    )
    scored["projection_gap"] = (scored["strikeouts_projection"] - scored["line"]).abs()
    scored["projection_signed_gap"] = scored["strikeouts_projection"] - scored["line"]
    scored["is_over"] = (scored["best_side"] == "over").astype(int)
    scored["is_under"] = (scored["best_side"] == "under").astype(int)
    scored["won_int"] = scored["won"].astype(int)
    candidates = scored[scored["edge_pct"] >= 5].copy()

    metrics = {
        "mae": float(mean_absolute_error(scored["strikeouts"], scored["strikeouts_projection"])),
        "rows": int(len(scored)),
        "candidate_bets": int(len(candidates)),
    }
    return candidates, metrics


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [col for col in FEATURE_CANDIDATES if col in df.columns]
    cols += ["projection_signed_gap", "is_over", "is_under"]
    return [col for col in cols if col in df.columns]


def _summarize(df: pd.DataFrame, label: str, mask: pd.Series) -> dict:
    group = df[mask].sort_values("game_date").copy()
    if group.empty:
        return {"strategy": label, "bets": 0}
    group["flat_profit"] = group["unit_profit"] * 100
    cumulative = group["flat_profit"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "strategy": label,
        "bets": int(len(group)),
        "wins": int(group["won"].sum()),
        "losses": int((~group["won"] & ~group["push"]).sum()),
        "win_rate": float(group["won"].mean()),
        "profit_100_flat": float(group["flat_profit"].sum()),
        "roi": float(group["flat_profit"].sum() / (len(group) * 100)),
        "max_drawdown": float(drawdown.min()),
        "avg_selector_prob": float(group["selector_win_probability"].mean())
        if "selector_win_probability" in group
        else np.nan,
        "avg_edge_pct": float(group["edge_pct"].mean()),
        "avg_projection_gap": float(group["projection_gap"].mean()),
    }


def _selector_eval(train_candidates: pd.DataFrame, test_candidates: pd.DataFrame, model_type: str, random_state: int):
    train = train_candidates.copy()
    test = test_candidates.copy()
    cols = _feature_cols(train)
    model = _selector(model_type, random_state)
    model.fit(train[cols], train["won_int"])
    train["selector_win_probability"] = model.predict_proba(train[cols])[:, 1]
    test["selector_win_probability"] = model.predict_proba(test[cols])[:, 1]
    rows = [
        _summarize(test, "all edge>=5 candidates", test["selector_win_probability"] >= 0),
    ]
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        rows.append(
            _summarize(
                test,
                f"selector_prob >= {threshold:.2f}",
                test["selector_win_probability"] >= threshold,
            )
        )
    for count in [25, 50, 100]:
        top = test.sort_values("selector_win_probability", ascending=False).head(count)
        rows.append(_summarize(test, f"top {count} selector bets", test.index.isin(top.index)))
    return test, rows, {
        "train_auc": float(roc_auc_score(train["won_int"], train["selector_win_probability"])),
        "test_auc": float(roc_auc_score(test["won_int"], test["selector_win_probability"])),
        "features": ",".join(cols),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward strikeout projection and bet selector.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--odds",
        default="data/odds/historical_pitcher_props.csv,data/odds/historical_pitcher_strikeouts_6h_2026.csv",
    )
    parser.add_argument("--model-type", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--output-prefix", default="data/processed/walk_forward_strikeout_selector")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_batting = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    game_context = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_logs = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    statcast_batter_pitch_types = load_statcast_batter_pitch_type_daily(
        config["data"]["statcast_batter_pitch_type_daily_file"]
    )
    park_factors = load_park_factors(config["data"]["park_factors_file"])
    featured, feature_cols, _ = build_training_features(
        logs,
        rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        team_batting_logs=team_batting,
        game_context_logs=game_context,
        batter_game_logs=batter_logs,
        statcast_pitcher_daily=statcast,
        statcast_batter_pitch_type_daily=statcast_batter_pitch_types,
        park_factors=park_factors,
    )
    odds = _prep_odds([part.strip() for part in args.odds.split(",") if part.strip()])

    folds = [
        ("2023", "2022-01-01", "2022-12-31", "2023-05-03", "2023-12-31"),
        ("2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2025", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        ("2026", "2022-01-01", "2025-12-31", "2026-01-01", "2026-05-31"),
    ]
    candidates_by_year = {}
    projection_metrics = []
    for year, train_start, train_end, test_start, test_end in folds:
        candidates, metrics = _candidate_bets(
            featured,
            feature_cols,
            odds,
            train_start,
            train_end,
            test_start,
            test_end,
            args.random_state,
        )
        candidates["eval_year"] = year
        candidates_by_year[year] = candidates
        projection_metrics.append({"eval_year": year, **metrics})

    rows = []
    scored_tests = []
    meta_rows = []
    selector_tests = [
        ("2024", ["2023"]),
        ("2025", ["2023", "2024"]),
        ("2026", ["2023", "2024", "2025"]),
    ]
    for test_year, train_years in selector_tests:
        train_candidates = pd.concat([candidates_by_year[year] for year in train_years], ignore_index=True)
        test_candidates = candidates_by_year[test_year].copy()
        if train_candidates.empty or test_candidates.empty:
            continue
        scored, summaries, meta = _selector_eval(
            train_candidates, test_candidates, args.model_type, args.random_state
        )
        scored["selector_test_year"] = test_year
        scored_tests.append(scored)
        for summary in summaries:
            rows.append({"eval_year": test_year, "selector_train_years": ",".join(train_years), **summary})
        meta_rows.append({"eval_year": test_year, "selector_train_years": ",".join(train_years), **meta})

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(candidates_by_year.values(), ignore_index=True).to_csv(
        prefix.with_name(prefix.name + "_candidates.csv"), index=False
    )
    if scored_tests:
        pd.concat(scored_tests, ignore_index=True).to_csv(
            prefix.with_name(prefix.name + "_scored_tests.csv"), index=False
        )
    pd.DataFrame(rows).to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    pd.DataFrame(meta_rows).to_csv(prefix.with_name(prefix.name + "_selector_meta.csv"), index=False)
    pd.DataFrame(projection_metrics).to_csv(
        prefix.with_name(prefix.name + "_projection_metrics.csv"), index=False
    )

    print(f"Walk-forward summary saved to {prefix.with_name(prefix.name + '_summary.csv')}")
    print(pd.DataFrame(projection_metrics).to_string(index=False))
    print(
        pd.DataFrame(rows)
        .sort_values(["eval_year", "roi"], ascending=[True, False])
        .groupby("eval_year")
        .head(8)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
