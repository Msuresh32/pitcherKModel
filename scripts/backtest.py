import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.backtest import (
    attach_odds_and_edges,
    compute_clv,
    compute_edge_bucket_stats,
    compute_sequential_pnl,
    score_predictions,
    score_probability_calibration,
    summarize_bets,
)
from src.config import ensure_directories, load_config
from src.data.fangraphs_source import merge_fangraphs_prior_season
from src.data.loaders import (
    filter_date_range,
    load_batter_game_logs,
    load_fangraphs_stats,
    load_game_context_logs,
    load_odds,
    load_park_factors,
    load_pitcher_game_logs,
    load_statcast_pitcher_daily,
    load_statcast_batter_pitch_type_daily,
    load_statcast_pitcher_advanced,
    load_statcast_batter_discipline,
    load_team_batting_game_logs,
)
from src.features.build_features import build_training_features
from src.models.calibration import (
    bias_corrections_from_calibration,
    build_calibration,
    build_probability_calibration,
    load_calibration,
    probability_calibrators_from_calibration,
    save_calibration,
)
from src.models.opportunity import add_expected_opportunity_features, load_opportunity_models
from src.models.train import load_fill_values, load_models, predict_targets

def _best_lines_from_historical(odds: pd.DataFrame) -> pd.DataFrame:
    """Reduce a multi-bookmaker historical odds DataFrame to one best-line row
    per (game_date, pitcher_id, market, line) — same logic as best_current_lines."""
    required = {"game_date", "pitcher_id", "market", "line", "over_odds", "under_odds"}
    missing = required - set(odds.columns)
    if missing:
        return pd.DataFrame()

    odds = odds.dropna(subset=["pitcher_id", "line"]).copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    # Normalize pitcher_id: CSV may store as float (e.g. 663460.0) due to NaN rows
    odds["pitcher_id"] = (
        pd.to_numeric(odds["pitcher_id"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .reindex(odds.index, fill_value="")
    )
    odds = odds[odds["pitcher_id"] != ""]
    odds["over_odds"] = pd.to_numeric(odds["over_odds"], errors="coerce")
    odds["under_odds"] = pd.to_numeric(odds["under_odds"], errors="coerce")

    rows = []
    for keys, group in odds.groupby(
        ["game_date", "pitcher_id", "market", "line"], dropna=False
    ):
        over_row = group.sort_values("over_odds", ascending=False, na_position="last").iloc[0]
        under_row = group.sort_values("under_odds", ascending=False, na_position="last").iloc[0]
        rows.append(
            {
                "game_date": keys[0],
                "pitcher_id": keys[1],
                "market": keys[2],
                "line": keys[3],
                "over_odds": over_row["over_odds"],
                "under_odds": under_row["under_odds"],
                "over_bookmaker": over_row.get("bookmaker", ""),
                "under_bookmaker": under_row.get("bookmaker", ""),
                "player_name": over_row.get("player_name", ""),
                "fetched_at": over_row.get("fetched_at", ""),
            }
        )
    return pd.DataFrame(rows)


_DISTRIBUTION_FOR_MODEL = {
    "poisson": "poisson",
    "tweedie": "poisson",
    "xgboost_poisson": "poisson",
    "random_forest": "normal",
    "xgboost": "normal",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--save-calibration",
        action="store_true",
        help="Save calibration.json from this run.",
    )
    parser.add_argument(
        "--odds",
        default=None,
        help=(
            "Path to historical odds CSV (e.g. from fetch_historical_odds.py). "
            "If it contains a snapshot_type column, 'open' rows are used for edge/EV "
            "and 'close' rows are used for CLV automatically."
        ),
    )
    parser.add_argument("--closing-odds", default=None, help="Separate CSV with closing-line odds for CLV.")
    parser.add_argument(
        "--blend",
        default=None,
        help="Override ensemble blend as 'glm_weight,xgb_weight' (e.g. '0.8,0.2'). "
             "Allows testing blend ratios without retraining.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override model directory (default: processed_dir/models). Useful for A/B testing saved model snapshots.",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help="Skip feature engineering + prediction and load pre-computed predictions CSV directly. "
             "Speeds up grid searches dramatically when only betting params change.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="Override betting.min_edge_pct without modifying config.yaml.",
    )
    parser.add_argument(
        "--edge-shrink",
        type=float,
        default=None,
        help="Override betting.edge_shrink_factor without modifying config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Command-line betting overrides (avoids config.yaml race conditions in grid searches)
    if args.min_edge is not None:
        config["betting"]["min_edge_pct"] = args.min_edge
    if args.edge_shrink is not None:
        config["betting"]["edge_shrink_factor"] = args.edge_shrink
    ensure_directories(config)

    start = args.start or config["training"]["backtest_start"]
    end   = args.end   or config["training"]["backtest_end"]
    output_prefix = args.output_prefix or "backtest"

    # ------------------------------------------------------------------
    # Fast path: load pre-computed predictions (skip feature eng + model)
    # ------------------------------------------------------------------
    if args.predictions_file:
        logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
        model_dir = Path(args.model_dir) if args.model_dir else Path(config["data"]["processed_dir"]) / "models"
        predictions = pd.read_csv(args.predictions_file)
        predictions["game_date"] = pd.to_datetime(predictions["game_date"])
        if "pitcher_id" in predictions.columns:
            predictions["pitcher_id"] = predictions["pitcher_id"].astype(str)
        # Filter to the requested date range (the file may cover a broader period)
        predictions = filter_date_range(predictions, start, end)
        print(f"Loaded {len(predictions)} pre-computed predictions from {args.predictions_file}")
    else:
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

        # ------------------------------------------------------------------
        # Load saved fill_values from training run
        # ------------------------------------------------------------------
        model_dir = Path(args.model_dir) if args.model_dir else Path(config["data"]["processed_dir"]) / "models"
        fill_values = load_fill_values(model_dir / "fill_values.json")

        # ------------------------------------------------------------------
        # Build features
        # ------------------------------------------------------------------
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
            fill_values=fill_values,
            statcast_pitcher_advanced=statcast_pitcher_adv if not statcast_pitcher_adv.empty else None,
            statcast_batter_discipline=statcast_bat_disc if not statcast_bat_disc.empty else None,
        )

        if not fangraphs.empty:
            featured, fg_cols = merge_fangraphs_prior_season(featured, fangraphs)
            feature_cols = feature_cols + fg_cols

    if not args.predictions_file:
        opportunity_models = load_opportunity_models(model_dir)
        if opportunity_models:
            featured, opp_cols = add_expected_opportunity_features(featured, opportunity_models)
            feature_cols = feature_cols + opp_cols

        backtest_df = filter_date_range(featured, start, end)

        # ------------------------------------------------------------------
        # Predict
        # ------------------------------------------------------------------
        if args.model_dir:
            model_dir = Path(args.model_dir)
        models = load_models(model_dir)

        # Override blend weights if --blend was supplied
        if args.blend:
            try:
                w = [float(x) for x in args.blend.split(",")]
                assert len(w) == 2
                for bundle in models.values():
                    if bundle.get("ensemble"):
                        bundle["blend_weights"] = w
                print(f"Blend override: {w[0]:.2f} GLM / {w[1]:.2f} XGB")
            except Exception as e:
                print(f"Warning: could not parse --blend '{args.blend}': {e}")

        predictions = predict_targets(backtest_df, models)

        pred_path = Path(config["data"]["processed_dir"]) / f"{output_prefix}_predictions.csv"
        predictions.to_csv(pred_path, index=False)

    # Per-market distribution from config (overrides model-type default)
    config_dist = config["betting"].get("market_distribution", {})
    if config_dist:
        distribution = {
            mkt: config_dist.get(mkt, "normal")
            for mkt in ["strikeouts", "walks", "hits_allowed"]
        }
    else:
        distribution = "normal"

    disabled_markets = config["betting"].get("disabled_markets", [])

    # ------------------------------------------------------------------
    # Point-estimate scores (MAE / RMSE)
    # ------------------------------------------------------------------
    scores = score_predictions(predictions)
    score_path = Path(config["data"]["processed_dir"]) / f"{output_prefix}_scores.csv"
    scores.to_csv(score_path, index=False)
    print("\nBacktest MAE / RMSE:")
    print(scores.to_string(index=False))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    calibration = build_calibration(
        predictions,
        residual_std_multiplier=config["betting"]["residual_std_multiplier"],
        edge_shrink_factor=config["betting"]["edge_shrink_factor"],
    )
    calibration_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    if args.save_calibration or args.output_prefix is None:
        save_calibration(calibration, calibration_path)

    # ------------------------------------------------------------------
    # Odds + betting columns
    # ------------------------------------------------------------------
    # Priority: --odds arg > config odds_file
    hist_odds_path = args.odds or config["data"].get("closing_odds_file", "")
    close_odds_df = pd.DataFrame()

    if hist_odds_path and Path(hist_odds_path).exists():
        raw_hist = pd.read_csv(hist_odds_path)
        raw_hist["game_date"] = pd.to_datetime(raw_hist["game_date"])
        raw_hist["pitcher_id"] = raw_hist["pitcher_id"].astype(str)

        if "snapshot_type" in raw_hist.columns:
            # Split open (entry) and close (CLV) automatically
            open_rows = raw_hist[raw_hist["snapshot_type"] == "open"].copy()
            close_odds_df = raw_hist[raw_hist["snapshot_type"] == "close"].copy()
        else:
            open_rows = raw_hist.copy()

        # Deduplicate to best line per pitcher/market/game (best over and under across books)
        odds = _best_lines_from_historical(open_rows)
        print(f"Loaded {len(odds)} open-snapshot lines from {hist_odds_path}")
    else:
        odds = load_odds(config["data"]["odds_file"])

    # Main-line only filter: discard alt lines at extreme odds
    if config["betting"].get("main_line_only", False):
        min_o = config["betting"].get("main_line_min_odds", -160)
        max_o = config["betting"].get("main_line_max_odds", 140)
        before = len(odds)
        odds = odds[
            odds["over_odds"].between(min_o, max_o) | odds["under_odds"].between(min_o, max_o)
        ].copy()
        print(f"Main-line filter: {len(odds)} of {before} lines kept (odds in [{min_o}, {max_o}])")

    calibration_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    existing_calibration = load_calibration(calibration_path)
    bias_corrections = bias_corrections_from_calibration(config, existing_calibration)
    probability_calibrators = probability_calibrators_from_calibration(existing_calibration)
    if bias_corrections:
        print(f"Applying bias corrections: { {k: f'{v:+.3f}' for k, v in bias_corrections.items()} }")
    if probability_calibrators:
        print(f"Applying probability calibrators: {list(probability_calibrators)}")

    # Fit NB dispersion from historical K distribution (method of moments)
    _ks = logs["strikeouts"].dropna().values
    _mu = float(_ks.mean())
    _nb_alpha_k = max(0.0, float((_ks.var() - _mu) / (_mu ** 2))) if _mu > 0 else 0.0
    nb_alpha = {"strikeouts": _nb_alpha_k}
    print(f"NB dispersion alpha (strikeouts): {_nb_alpha_k:.4f}")

    scored = attach_odds_and_edges(
        predictions,
        odds,
        residual_std=config["betting"]["default_residual_std"],
        max_kelly_fraction=config["betting"]["max_kelly_fraction"],
        edge_shrink_factor=config["betting"]["edge_shrink_factor"],
        distribution=distribution,
        bias_corrections=bias_corrections,
        disabled_markets=disabled_markets,
        nb_alpha=nb_alpha,
        probability_calibrators=probability_calibrators,
    )
    edges_path = Path(config["data"]["processed_dir"]) / f"{output_prefix}_edges.csv"
    scored.to_csv(edges_path, index=False)

    has_odds = "market" in scored.columns

    # ------------------------------------------------------------------
    # Probability calibration (Brier, log loss)
    # ------------------------------------------------------------------
    if has_odds:
        prob_scores = score_probability_calibration(scored)
        if not prob_scores.empty:
            prob_path = (
                Path(config["data"]["processed_dir"]) / f"{output_prefix}_prob_calibration.csv"
            )
            prob_scores.to_csv(prob_path, index=False)
            print("\nProbability calibration (Brier / log loss):")
            print(prob_scores.to_string(index=False))
    else:
        print(
            "\nNote: no sportsbook odds matched the backtest period — "
            "run scripts/fetch_historical_odds.py to get 2025 odds for Brier/CLV analysis."
        )

    # Fit isotonic probability calibration and merge into calibration.json
    if has_odds and (args.save_calibration or args.output_prefix is None):
        prob_cal = build_probability_calibration(scored)
        for target, spec in prob_cal.items():
            calibration.setdefault("markets", {}).setdefault(target, {})["probability_calibration"] = spec
        save_calibration(calibration, calibration_path)
        if prob_cal:
            for target, spec in prob_cal.items():
                improvement = spec["raw_brier"] - spec["calibrated_brier"]
                print(
                    f"Probability calibration fitted ({target}): "
                    f"Brier {spec['raw_brier']:.4f} -> {spec['calibrated_brier']:.4f} "
                    f"(improvement: {improvement:.4f})"
                )

    # ------------------------------------------------------------------
    # Edge-bucket segmentation
    # ------------------------------------------------------------------
    if has_odds:
        bucket_stats = compute_edge_bucket_stats(scored)
        if not bucket_stats.empty:
            bucket_path = (
                Path(config["data"]["processed_dir"]) / f"{output_prefix}_edge_buckets.csv"
            )
            bucket_stats.to_csv(bucket_path, index=False)
            print("\nEdge bucket hit rates:")
            print(bucket_stats.to_string(index=False))

    # ------------------------------------------------------------------
    # Qualifying bets: edge in [min, max], deduplicated by best edge per key
    # ------------------------------------------------------------------
    qualifying_bets = pd.DataFrame()
    if has_odds and "edge_pct" in scored.columns:
        min_e = config["betting"]["min_edge_pct"]
        max_e = config["betting"].get("max_edge_pct", float("inf"))
        q = scored[
            (scored["edge_pct"] >= min_e) & (scored["edge_pct"] <= max_e)
        ].copy()

        # Projection gap filter: only bet when model and market are close.
        max_gap = config["betting"].get("max_proj_gap")
        if max_gap and "market" in q.columns and "line" in q.columns:
            proj_vals = q.apply(
                lambda r: r.get(f"{r['market']}_projection", float("nan")), axis=1
            )
            q = q[abs(proj_vals - q["line"]) <= max_gap].copy()

        # Minimum current-season starts: avoids early-season noise when the model
        # is working off stale prior-year rolling features.
        min_starts = config["betting"].get("min_starts_current_season", 0)
        if min_starts > 0 and "pitcher_starts_ytd" in q.columns:
            q = q[q["pitcher_starts_ytd"] >= min_starts].copy()

        bet_key = ["game_date", "pitcher_id", "market", "line", "best_side"]
        if all(c in q.columns for c in bet_key):
            q = q.sort_values("edge_pct", ascending=False).drop_duplicates(subset=bet_key)
        qualifying_bets = q

    # ------------------------------------------------------------------
    # Bet summary (uses capped qualifying_bets)
    # ------------------------------------------------------------------
    bet_summary = summarize_bets(
        qualifying_bets if not qualifying_bets.empty else scored,
        bankroll=config["betting"]["bankroll"],
        min_edge_pct=config["betting"]["min_edge_pct"] if qualifying_bets.empty else 0.0,
    )
    summary_path = Path(config["data"]["processed_dir"]) / f"{output_prefix}_bet_summary.csv"
    bet_summary.to_csv(summary_path, index=False)
    if not bet_summary.empty:
        print("\nBet summary:")
        print(bet_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Sequential P&L with drawdown
    # ------------------------------------------------------------------
    if not qualifying_bets.empty:
        pnl = compute_sequential_pnl(qualifying_bets, bankroll=config["betting"]["bankroll"])
        if not pnl.empty:
            pnl_path = (
                Path(config["data"]["processed_dir"]) / f"{output_prefix}_sequential_pnl.csv"
            )
            pnl.to_csv(pnl_path, index=False)
            print(
                f"\nMax drawdown: ${pnl.attrs.get('max_drawdown', 0):.2f}  "
                f"Sharpe: {pnl.attrs.get('sharpe', 0):.2f}  "
                f"ROI: {pnl.attrs.get('roi', 0):.2%}"
            )

    # ------------------------------------------------------------------
    # CLV
    # close_odds_df already populated if --odds file had snapshot_type column.
    # --closing-odds arg provides a separate file as fallback.
    # ------------------------------------------------------------------
    if close_odds_df.empty:
        closing_odds_path = args.closing_odds or config["data"].get("closing_odds_file", "")
        if closing_odds_path and Path(closing_odds_path).exists():
            close_odds_df = pd.read_csv(closing_odds_path)

    if not close_odds_df.empty and not qualifying_bets.empty:
        # Reduce to best closing line per pitcher/market/game
        if "snapshot_type" in close_odds_df.columns:
            close_odds_df = close_odds_df[close_odds_df["snapshot_type"] == "close"].copy()
        close_best = _best_lines_from_historical(close_odds_df)
        with_clv = compute_clv(qualifying_bets, close_best)
        clv_path = Path(config["data"]["processed_dir"]) / f"{output_prefix}_clv.csv"
        with_clv.to_csv(clv_path, index=False)

        clv_col = with_clv["clv_pct"] if "clv_pct" in with_clv.columns else pd.Series(dtype=float)
        mean_clv = float(clv_col.mean()) if not clv_col.dropna().empty else float("nan")
        clv_coverage = int(clv_col.notna().sum())
        print(f"\nCLV Results ({clv_coverage} bets with closing line matched):")
        print(f"  Mean CLV:     {mean_clv:+.2f}%")

        # Per-market breakdown
        if "market" in with_clv.columns and "clv_pct" in with_clv.columns:
            for mkt in with_clv["market"].unique():
                mkt_clv = with_clv.loc[with_clv["market"] == mkt, "clv_pct"].dropna()
                if not mkt_clv.empty:
                    print(f"  {mkt:16s}: mean CLV = {mkt_clv.mean():+.2f}%  (n={len(mkt_clv)})")
        print(f"CLV analysis saved to {clv_path}")

    if not args.predictions_file:
        print(f"\nBacktest predictions saved to {pred_path}")
    print(f"Backtest scores saved to {score_path}")
    if args.save_calibration or args.output_prefix is None:
        print(f"Calibration saved to {calibration_path}")
    print(f"Backtest edges saved to {edges_path}")
    print(f"Bet summary saved to {summary_path}")


if __name__ == "__main__":
    main()
