import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.fangraphs_source import merge_fangraphs_prior_season
from src.data.loaders import (
    filter_date_range,
    load_batter_game_logs,
    load_fangraphs_stats,
    load_game_context_logs,
    load_park_factors,
    load_pitcher_game_logs,
    load_statcast_pitcher_daily,
    load_statcast_batter_pitch_type_daily,
    load_statcast_pitcher_advanced,
    load_statcast_batter_discipline,
    load_team_batting_game_logs,
)
from src.features.build_features import build_training_features
from src.models.opportunity import (
    add_expected_opportunity_features,
    load_opportunity_models,
    train_opportunity_models,
)
from src.models.train import (
    save_fill_values,
    select_top_features,
    temporal_cross_validate,
    train_models,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=4,
        help="Number of walk-forward CV folds. Set to 0 to skip CV.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_batting = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    game_context = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_logs = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    statcast_batter_pitch_types = load_statcast_batter_pitch_type_daily(
        config["data"]["statcast_batter_pitch_type_daily_file"]
    )
    park_factors = load_park_factors(config["data"]["park_factors_file"])

    fangraphs_path = config["data"].get("fangraphs_file", "")
    fangraphs = load_fangraphs_stats(fangraphs_path) if fangraphs_path else pd.DataFrame()

    adv_pitcher_path = config["data"].get("statcast_pitcher_advanced_file", "")
    statcast_pitcher_adv = load_statcast_pitcher_advanced(adv_pitcher_path) if adv_pitcher_path else pd.DataFrame()

    bat_disc_path = config["data"].get("statcast_batter_discipline_file", "")
    statcast_bat_disc = load_statcast_batter_discipline(bat_disc_path) if bat_disc_path else pd.DataFrame()

    if not statcast_pitcher_adv.empty:
        print(f"Loaded {len(statcast_pitcher_adv)} advanced pitcher rows")
    if not statcast_bat_disc.empty:
        print(f"Loaded {len(statcast_bat_disc)} batter discipline rows")

    # ------------------------------------------------------------------
    # Build features (fill_values computed from ALL data at this stage;
    # train-period fill_values are saved separately below)
    # ------------------------------------------------------------------
    featured, feature_cols, fill_values_all = build_training_features(
        logs,
        rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        team_batting_logs=team_batting,
        game_context_logs=game_context,
        batter_game_logs=batter_logs,
        statcast_pitcher_daily=statcast,
        statcast_batter_pitch_type_daily=statcast_batter_pitch_types,
        park_factors=park_factors,
        statcast_pitcher_advanced=statcast_pitcher_adv if not statcast_pitcher_adv.empty else None,
        statcast_batter_discipline=statcast_bat_disc if not statcast_bat_disc.empty else None,
    )

    # Merge prior-season FanGraphs stats (safe: uses season-1 join)
    if not fangraphs.empty:
        featured, fg_cols = merge_fangraphs_prior_season(featured, fangraphs)
        feature_cols = feature_cols + fg_cols
        print(f"Added {len(fg_cols)} FanGraphs feature columns: {fg_cols}")

    train_df = filter_date_range(
        featured,
        config["training"]["train_start"],
        config["training"]["train_end"],
    )

    # Compute fill_values from training period only and save for inference
    fill_values_train = train_df[feature_cols].median(numeric_only=True).to_dict()

    model_dir = Path(config["data"]["processed_dir"]) / "models"

    # ------------------------------------------------------------------
    # Feature selection (RF importance-based pruning before Poisson GLM)
    # ------------------------------------------------------------------
    top_k = config["features"].get("top_k_features")
    if top_k and config["training"]["model_type"] in ("poisson", "tweedie", "ensemble"):
        print(f"\nSelecting top {top_k} features by Random Forest importance...")
        feature_cols = select_top_features(
            train_df,
            feature_cols=feature_cols,
            top_k=top_k,
            random_state=config["training"]["random_state"],
        )

    # ------------------------------------------------------------------
    # Walk-forward cross-validation (OOS metrics before final training)
    # ------------------------------------------------------------------
    per_target_alpha = config["training"].get("per_target_alpha", {})
    global_alpha = config["training"].get("alpha")

    if args.cv_splits > 0:
        print(f"\nRunning {args.cv_splits}-fold walk-forward cross-validation...")
        # CV uses mean alpha across markets for simplicity
        cv_alpha = (
            sum(per_target_alpha.values()) / len(per_target_alpha)
            if per_target_alpha
            else global_alpha
        )
        cv_df = temporal_cross_validate(
            train_df,
            feature_cols=feature_cols,
            n_splits=args.cv_splits,
            model_type=config["training"]["model_type"],
            random_state=config["training"]["random_state"],
            alpha=cv_alpha,
        )
        cv_path = Path(config["data"]["processed_dir"]) / "cv_metrics.csv"
        cv_df.to_csv(cv_path, index=False)

        # Print summary
        summary = cv_df.groupby("target")[["mae", "rmse"]].mean().round(4)
        print("\nOOS CV summary (mean across folds):")
        print(summary.to_string())
        print(f"CV metrics saved to {cv_path}\n")

    # ------------------------------------------------------------------
    # Opportunity models (innings pitched, pitches, batters faced)
    # ------------------------------------------------------------------
    opportunity_metrics = train_opportunity_models(
        train_df=train_df,
        feature_cols=feature_cols,
        model_dir=model_dir,
        model_type=config["training"]["model_type"],
        random_state=config["training"]["random_state"],
    )
    opportunity_models = load_opportunity_models(model_dir)
    featured, opportunity_cols = add_expected_opportunity_features(featured, opportunity_models)
    train_df = filter_date_range(
        featured,
        config["training"]["train_start"],
        config["training"]["train_end"],
    )
    feature_cols = feature_cols + opportunity_cols

    # ------------------------------------------------------------------
    # Final model training
    # ------------------------------------------------------------------
    metrics = train_models(
        train_df=train_df,
        feature_cols=feature_cols,
        model_dir=model_dir,
        model_type=config["training"]["model_type"],
        random_state=config["training"]["random_state"],
        alpha=global_alpha,
        per_target_alpha=per_target_alpha,
    )

    # Update fill_values to include opportunity feature columns (predicted, rarely NaN)
    fill_values_train.update(
        {col: float(train_df[col].median()) for col in opportunity_cols if col in train_df.columns}
    )
    save_fill_values(fill_values_train, model_dir / "fill_values.json")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics_path = Path(config["data"]["processed_dir"]) / "train_metrics.csv"
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("market").reset_index().to_csv(
        metrics_path, index=False
    )
    opportunity_metrics_path = (
        Path(config["data"]["processed_dir"]) / "opportunity_train_metrics.csv"
    )
    pd.DataFrame.from_dict(opportunity_metrics, orient="index").rename_axis(
        "target"
    ).reset_index().to_csv(opportunity_metrics_path, index=False)

    print(f"Trained models saved to {model_dir}")
    print(f"Fill values saved to {model_dir / 'fill_values.json'}")
    print(f"Training metrics saved to {metrics_path}")
    print(f"Opportunity metrics saved to {opportunity_metrics_path}")

    # Print in-sample vs OOS comparison if CV ran
    if args.cv_splits > 0:
        print("\nIn-sample vs OOS comparison:")
        for target, m in metrics.items():
            print(f"  {target}: train MAE={m['mae']:.4f}  |  see cv_metrics.csv for OOS")


if __name__ == "__main__":
    main()
