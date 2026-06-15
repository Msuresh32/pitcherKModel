import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
from src.models.opportunity import add_expected_opportunity_features, load_opportunity_models
from src.odds.pricing import add_betting_columns


def _american_to_decimal(odds: pd.Series) -> pd.Series:
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / odds.abs())


def _prep_odds(path: str, market: str = "strikeouts") -> pd.DataFrame:
    odds = pd.read_csv(path)
    odds = odds[odds["market"] == market].copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    odds = odds.dropna(subset=["pitcher_id", "line", "over_odds", "under_odds"])
    odds["pitcher_id"] = pd.to_numeric(odds["pitcher_id"], errors="coerce").astype("Int64").astype(str)
    sort_cols = [col for col in ["game_date", "pitcher_id", "historical_snapshot", "fetched_at"] if col in odds]
    odds = odds.sort_values(sort_cols)
    odds = odds.drop_duplicates(["game_date", "pitcher_id"], keep="last")
    odds["over_decimal"] = _american_to_decimal(odds["over_odds"].astype(float))
    odds["under_decimal"] = _american_to_decimal(odds["under_odds"].astype(float))
    odds["over_implied"] = 1 / odds["over_decimal"]
    odds["under_implied"] = 1 / odds["under_decimal"]
    vig = odds["over_implied"] + odds["under_implied"]
    odds["novig_over_probability"] = odds["over_implied"] / vig
    odds["novig_under_probability"] = odds["under_implied"] / vig
    odds["market_mid_projection"] = odds["line"]
    return odds[
        [
            "game_date",
            "pitcher_id",
            "line",
            "over_odds",
            "under_odds",
            "over_implied",
            "under_implied",
            "novig_over_probability",
            "novig_under_probability",
            "market_mid_projection",
        ]
    ]


def _make_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
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


def _metrics(y: pd.Series, pred: np.ndarray, label: str) -> dict:
    err = y - pred
    return {
        "model": label,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "bias_actual_minus_pred": float(err.mean()),
        "rows": int(len(y)),
    }


def _bet_summary(df: pd.DataFrame, projection_col: str, label: str, residual_std: float) -> dict:
    scored = df.copy()
    scored["strikeouts_projection"] = scored[projection_col]
    scored = add_betting_columns(
        scored,
        market="strikeouts",
        residual_std=residual_std,
        max_kelly_fraction=0.05,
        edge_shrink_factor=0.5,
    )
    scored["projection_gap"] = (scored["strikeouts_projection"] - scored["line"]).abs()
    bets = scored[(scored["edge_pct"] >= 5) & (scored["projection_gap"] >= 0.5)].copy()
    if bets.empty:
        return {"model": label, "bets": 0}
    bets["bet_odds"] = np.where(bets["best_side"] == "over", bets["over_odds"], bets["under_odds"])
    decimal = _american_to_decimal(bets["bet_odds"].astype(float))
    bets["won"] = np.where(
        bets["best_side"] == "over",
        bets["strikeouts"] > bets["line"],
        bets["strikeouts"] < bets["line"],
    )
    bets["push"] = bets["strikeouts"] == bets["line"]
    bets["unit_profit"] = np.where(
        bets["push"],
        0.0,
        np.where(bets["won"], decimal - 1, -1.0),
    )
    return {
        "model": label,
        "bets": int(len(bets)),
        "wins": int(bets["won"].sum()),
        "losses": int((~bets["won"] & ~bets["push"]).sum()),
        "win_rate": float(bets["won"].mean()),
        "unit_profit": float(bets["unit_profit"].sum()),
        "unit_roi": float(bets["unit_profit"].mean()),
        "avg_edge_pct": float(bets["edge_pct"].mean()),
        "avg_projection_gap": float(bets["projection_gap"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strikeout models with and without odds features.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--odds", default="data/odds/historical_pitcher_props.csv")
    parser.add_argument("--train-start", default="2023-05-03")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--output-prefix", default="data/processed/market_aware_strikeouts")
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
    model_dir = Path(config["data"]["processed_dir"]) / "models"
    opportunity_models = load_opportunity_models(model_dir)
    if opportunity_models:
        featured, _ = add_expected_opportunity_features(featured, opportunity_models)
        feature_cols = list(feature_cols) + [
            col
            for col in ["expected_innings_pitched", "expected_pitches", "expected_batters_faced"]
            if col in featured.columns and col not in feature_cols
        ]

    odds = _prep_odds(args.odds)
    df = featured.merge(odds, on=["game_date", "pitcher_id"], how="inner")
    train = filter_date_range(df, args.train_start, args.train_end)
    test = filter_date_range(df, args.test_start, args.test_end)
    if train.empty or test.empty:
        raise ValueError(f"Not enough merged odds/features rows. train={len(train)}, test={len(test)}")

    odds_features = [
        "line",
        "over_odds",
        "under_odds",
        "over_implied",
        "under_implied",
        "novig_over_probability",
        "novig_under_probability",
        "market_mid_projection",
    ]
    base_cols = [col for col in feature_cols if col in train.columns]
    market_cols = base_cols + odds_features

    y_train = train["strikeouts"].astype(float)
    y_test = test["strikeouts"].astype(float)

    base_model = _make_model(config["training"]["random_state"])
    market_model = _make_model(config["training"]["random_state"])
    base_model.fit(train[base_cols], y_train)
    market_model.fit(train[market_cols], y_train)

    base_train_pred = base_model.predict(train[base_cols])
    market_train_pred = market_model.predict(train[market_cols])
    base_test_pred = base_model.predict(test[base_cols])
    market_test_pred = market_model.predict(test[market_cols])

    train_residual_std = float((y_train - base_train_pred).std(ddof=0) * 1.25)
    market_train_residual_std = float((y_train - market_train_pred).std(ddof=0) * 1.25)

    predictions = test.copy()
    predictions["base_projection"] = base_test_pred
    predictions["market_aware_projection"] = market_test_pred

    metrics = pd.DataFrame(
        [
            _metrics(y_test, base_test_pred, "base"),
            _metrics(y_test, market_test_pred, "market_aware"),
        ]
    )
    betting = pd.DataFrame(
        [
            _bet_summary(predictions, "base_projection", "base", train_residual_std),
            _bet_summary(
                predictions,
                "market_aware_projection",
                "market_aware",
                market_train_residual_std,
            ),
        ]
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prefix.with_name(prefix.name + "_predictions.csv"), index=False)
    metrics.to_csv(prefix.with_name(prefix.name + "_metrics.csv"), index=False)
    betting.to_csv(prefix.with_name(prefix.name + "_betting.csv"), index=False)

    print(f"Predictions saved to {prefix.with_name(prefix.name + '_predictions.csv')}")
    print(f"Metrics saved to {prefix.with_name(prefix.name + '_metrics.csv')}")
    print(f"Betting comparison saved to {prefix.with_name(prefix.name + '_betting.csv')}")
    print(metrics.to_string(index=False))
    print(betting.to_string(index=False))


if __name__ == "__main__":
    main()
