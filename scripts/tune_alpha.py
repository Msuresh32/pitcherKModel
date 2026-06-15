"""Grid-search Poisson regularization strength (alpha) using walk-forward CV.

Runs temporal_cross_validate for each alpha candidate and reports OOS MAE/RMSE
per market. Use the results to set training.alpha (global) or
training.per_target_alpha (per-market) in config.yaml.

Usage:
    python scripts/tune_alpha.py
    python scripts/tune_alpha.py --alphas 0.01 0.1 0.5 1.0 5.0 10.0
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

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
from src.models.train import load_fill_values, temporal_cross_validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search Poisson alpha via walk-forward CV.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Alpha values to evaluate.",
    )
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--output", default="data/processed/alpha_tuning.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    model_type = config["training"]["model_type"]

    if model_type not in ("poisson", "tweedie"):
        print(f"Alpha tuning only applies to poisson/tweedie. Current model_type={model_type}. Exiting.")
        return

    # ------------------------------------------------------------------
    # Build features once
    # ------------------------------------------------------------------
    logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_batting = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    game_context = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_logs = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    statcast_batter = load_statcast_batter_pitch_type_daily(
        config["data"]["statcast_batter_pitch_type_daily_file"]
    )
    park_factors = load_park_factors(config["data"]["park_factors_file"])

    model_dir = Path(config["data"]["processed_dir"]) / "models"
    fill_values = load_fill_values(model_dir / "fill_values.json")

    featured, feature_cols, _ = build_training_features(
        logs,
        rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        team_batting_logs=team_batting,
        game_context_logs=game_context,
        batter_game_logs=batter_logs,
        statcast_pitcher_daily=statcast,
        statcast_batter_pitch_type_daily=statcast_batter,
        park_factors=park_factors,
        fill_values=fill_values,
    )

    opp_models = load_opportunity_models(model_dir)
    if opp_models:
        featured, opp_cols = add_expected_opportunity_features(featured, opp_models)
        feature_cols = feature_cols + opp_cols

    train_df = filter_date_range(
        featured,
        config["training"]["train_start"],
        config["training"]["train_end"],
    )

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    results = []
    for alpha in args.alphas:
        print(f"Evaluating alpha={alpha} ...")
        cv_df = temporal_cross_validate(
            train_df,
            feature_cols=feature_cols,
            n_splits=args.cv_splits,
            model_type=model_type,
            random_state=config["training"]["random_state"],
            alpha=alpha,
        )
        summary = cv_df.groupby("target")[["mae", "rmse"]].mean()
        for target, row in summary.iterrows():
            results.append(
                {
                    "alpha": alpha,
                    "target": target,
                    "cv_mae": round(row["mae"], 4),
                    "cv_rmse": round(row["rmse"], 4),
                }
            )

    result_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\nAlpha tuning results (lower MAE = better):\n")
    pivot = result_df.pivot_table(
        index="alpha", columns="target", values="cv_mae"
    ).round(4)
    print(pivot.to_string())

    # Recommend best alpha per market
    print("\nRecommended alpha per market:")
    for target in result_df["target"].unique():
        sub = result_df[result_df["target"] == target]
        best = sub.loc[sub["cv_mae"].idxmin()]
        print(f"  {target:16s}: alpha={best['alpha']}  (CV MAE={best['cv_mae']:.4f})")

    overall_best = result_df.groupby("alpha")["cv_mae"].mean()
    best_overall_alpha = overall_best.idxmin()
    print(f"\nBest single alpha (minimises mean MAE across all markets): {best_overall_alpha}")
    print(f"\nResults saved to {out_path}")
    print("\nTo apply: add to config.yaml under training:")
    print(f"  alpha: {best_overall_alpha}")
    print("  # Or per-market:")
    for target in result_df["target"].unique():
        sub = result_df[result_df["target"] == target]
        best = sub.loc[sub["cv_mae"].idxmin()]
        print(f"  # {target}: {best['alpha']}")


if __name__ == "__main__":
    main()
