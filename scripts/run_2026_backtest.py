"""
run_2026_backtest.py — Run new Statcast model inference on 2026 games.

Extracts 2026 odds from bt_poisson_2026_full_edges.csv, runs feature building
+ model inference on 2026 pitcher-game data using the trained V4 model, then
outputs predictions + edges to data/processed_poisson_wf2025/.
"""
from __future__ import annotations

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
from src.models.calibration import bias_corrections_from_calibration, load_calibration
from src.models.opportunity import add_expected_opportunity_features, load_opportunity_models
from src.models.train import load_fill_values, load_models, predict_targets
from src.backtesting.backtest import attach_odds_and_edges, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_v4_production.yaml")
    parser.add_argument("--start",  default="2026-03-26")
    parser.add_argument("--end",    default="2026-06-30")
    parser.add_argument("--output-prefix", default="bt_2026_new_model")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    proc_dir = Path(config["data"]["processed_dir"])
    model_dir = proc_dir / "models"

    # ── Extract 2026 odds from the existing edges file ──────────────────────
    src_edges = Path("data/processed_poisson/bt_poisson_2026_full_edges.csv")
    print(f"Extracting 2026 odds from {src_edges}...")
    raw_edges = pd.read_csv(src_edges, low_memory=False,
                            usecols=["game_date","pitcher_id","market",
                                     "line","over_odds","under_odds",
                                     "over_bookmaker","under_bookmaker"])
    # Rename bookmaker cols for _best_lines_from_historical
    raw_edges = raw_edges.rename(columns={"over_bookmaker": "bookmaker"})
    raw_edges["game_date"] = pd.to_datetime(raw_edges["game_date"])
    raw_edges = raw_edges[(raw_edges["game_date"] >= args.start) &
                          (raw_edges["game_date"] <= args.end)]
    odds_path = proc_dir / "2026_odds_extract.csv"
    raw_edges.to_csv(odds_path, index=False)
    print(f"  Saved {len(raw_edges)} odds rows -> {odds_path}")

    # ── Load raw data ────────────────────────────────────────────────────────
    print("Loading raw data...")
    logs        = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_bat    = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    game_ctx    = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_logs = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast    = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    sc_batter   = load_statcast_batter_pitch_type_daily(
        config["data"]["statcast_batter_pitch_type_daily_file"])
    park_factors = load_park_factors(config["data"]["park_factors_file"])

    fg_path = config["data"].get("fangraphs_file", "")
    fangraphs = load_fangraphs_stats(fg_path) if fg_path else pd.DataFrame()

    adv_path = config["data"].get("statcast_pitcher_advanced_file", "")
    sc_adv   = load_statcast_pitcher_advanced(adv_path) if adv_path else pd.DataFrame()

    disc_path = config["data"].get("statcast_batter_discipline_file", "")
    sc_disc   = load_statcast_batter_discipline(disc_path) if disc_path else pd.DataFrame()

    fill_values = load_fill_values(model_dir / "fill_values.json")

    # ── Build features ───────────────────────────────────────────────────────
    print("Building features (this takes ~2-3 min)...")
    featured, feature_cols, _ = build_training_features(
        logs,
        rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        team_batting_logs=team_bat,
        game_context_logs=game_ctx,
        batter_game_logs=batter_logs,
        statcast_pitcher_daily=statcast,
        statcast_batter_pitch_type_daily=sc_batter,
        park_factors=park_factors,
        fill_values=fill_values,
        statcast_pitcher_advanced=sc_adv if not sc_adv.empty else None,
        statcast_batter_discipline=sc_disc if not sc_disc.empty else None,
    )
    if not fangraphs.empty:
        featured, fg_cols = merge_fangraphs_prior_season(featured, fangraphs)
        feature_cols = feature_cols + fg_cols
    print(f"  Features built: {len(featured)} rows, {len(feature_cols)} features")

    # ── Load opportunity models ───────────────────────────────────────────────
    opp_models = load_opportunity_models(model_dir)
    if opp_models:
        featured, new_opp_cols = add_expected_opportunity_features(
            featured, opp_models, fill_values=fill_values
        )
        feature_cols = feature_cols + [c for c in new_opp_cols if c not in feature_cols]

    # ── Filter to 2026 eval window ────────────────────────────────────────────
    predictions_all = filter_date_range(featured, args.start, args.end)
    print(f"  2026 eval rows: {len(predictions_all)}")

    # ── Load models + predict ─────────────────────────────────────────────────
    print("Loading models and predicting...")
    models = load_models(model_dir)

    existing_calibration = load_calibration(proc_dir / "calibration.json")
    bias_corrections = bias_corrections_from_calibration(config, existing_calibration)
    if bias_corrections:
        print(f"  Bias corrections: {bias_corrections}")

    predictions_all = predict_targets(predictions_all, models)

    # Apply bias corrections manually
    if bias_corrections:
        for target, correction in bias_corrections.items():
            col = f"{target}_projection"
            if col in predictions_all.columns:
                predictions_all[col] = predictions_all[col] + correction

    # Save raw predictions
    pred_out = proc_dir / f"{args.output_prefix}_predictions.csv"
    predictions_all.to_csv(pred_out, index=False)
    print(f"  Saved predictions -> {pred_out}")

    # ── Accuracy on 2026 ──────────────────────────────────────────────────────
    scores = score_predictions(predictions_all)
    print("\nModel accuracy on 2026:")
    print(scores[["market","mae","rmse","rows"]].to_string(index=False))

    # ── Attach odds + edges ───────────────────────────────────────────────────
    print("\nAttaching odds and computing edges...")
    from scripts.backtest import _best_lines_from_historical
    odds = pd.read_csv(odds_path)
    odds = _best_lines_from_historical(odds)
    print(f"  Odds rows after dedup: {len(odds)}")

    from src.models.calibration import probability_calibrators_from_calibration
    prob_cals = probability_calibrators_from_calibration(existing_calibration)

    edges = attach_odds_and_edges(
        predictions_all,
        odds,
        residual_std=config["betting"]["default_residual_std"],
        max_kelly_fraction=float(config["betting"]["max_kelly_fraction"]),
        edge_shrink_factor=float(config["betting"].get("edge_shrink_factor", 1.0)),
        distribution=config["betting"]["market_distribution"],
        bias_corrections=bias_corrections,
        disabled_markets=config["betting"].get("disabled_markets"),
        probability_calibrators=prob_cals,
    )
    edges_out = proc_dir / f"{args.output_prefix}_edges.csv"
    edges.to_csv(edges_out, index=False)
    print(f"  Saved edges -> {edges_out} ({len(edges)} rows)")

    # ── Quick summary ─────────────────────────────────────────────────────────
    if "strikeouts" in edges.columns and "edge_gap_product" in edges.columns:
        min_egp   = float(config["betting"].get("min_edge_gap_product", 6.0))
        min_prob  = float(config["betting"].get("min_model_prob", 0.0))

        eligible = edges[
            (edges["market"] == "strikeouts") &
            (edges["edge_gap_product"].notna()) &
            (edges["edge_gap_product"] >= min_egp)
        ].copy()

        if min_prob > 0 and "over_probability" in eligible.columns:
            prob = eligible.apply(
                lambda r: r.get("over_probability", 0) if r.get("best_side") == "over"
                          else r.get("under_probability", 0), axis=1
            )
            eligible = eligible[pd.to_numeric(prob, errors="coerce").fillna(0) >= min_prob]

        print(f"\nEligible bets (EGP>={min_egp}"
              + (f", prob>={min_prob:.0%}" if min_prob else "") + f"): {len(eligible)}")
        if len(eligible) > 0 and "strikeouts" in eligible.columns:
            resolved = eligible.dropna(subset=["strikeouts"])
            if len(resolved) > 0:
                from src.backtesting.backtest import summarize_bets
                bankroll = float(config["betting"].get("bankroll", 1000))
                min_eg   = float(config["betting"].get("min_edge_gap_product", 6.0))
                summary = summarize_bets(resolved, bankroll=bankroll,
                                         min_edge_pct=0.0,
                                         min_edge_gap_product=min_eg)
                if not summary.empty:
                    print(summary[["bets","win_rate","roi","profit"]].to_string(index=False))


if __name__ == "__main__":
    main()
