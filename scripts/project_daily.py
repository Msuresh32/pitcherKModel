import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.fangraphs_source import merge_fangraphs_prior_season
from src.data.loaders import (
    load_batter_game_logs,
    load_fangraphs_stats,
    load_game_context_logs,
    load_odds,
    load_park_factors,
    load_pitcher_game_logs,
    load_probable_pitchers,
    load_statcast_pitcher_daily,
    load_statcast_batter_pitch_type_daily,
    load_statcast_pitcher_advanced,
    load_statcast_batter_discipline,
    load_team_batting_game_logs,
)
from src.data.schema import TARGETS
from src.export.exporters import export_csv, export_google_sheets, export_pretty_excel
from src.features.build_features import build_daily_features
from src.models.calibration import (
    bias_corrections_from_calibration,
    edge_shrink_from_calibration,
    load_calibration,
    residual_std_from_calibration,
)
from src.models.opportunity import add_expected_opportunity_features, load_opportunity_models
from src.models.train import load_fill_values, load_models, predict_targets
from src.odds.pricing import add_betting_columns

_DISTRIBUTION_FOR_MODEL = {
    "poisson": "poisson",
    "tweedie": "poisson",
    "xgboost_poisson": "poisson",
    "random_forest": "normal",
    "xgboost": "normal",
}


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}%"


def _format_american(value: float) -> str:
    if pd.isna(value):
        return ""
    value = int(round(float(value)))
    return f"+{value}" if value > 0 else str(value)


def _assign_confidence_tier(row: pd.Series) -> str:
    """High/Medium/Low based on pitcher history depth, Statcast availability, and lineup confirmation."""
    games = float(row.get("p_games_prior", 0) or 0)
    has_statcast = pd.notna(row.get("sc_csw_rate_avg_prior")) or pd.notna(
        row.get("sc_swinging_strike_rate_avg_prior")
    )
    confirmed_starters = float(row.get("opp_lineup_confirmed_starters", 0) or 0)
    lineup_ok = confirmed_starters >= 7
    expected_ip = row.get("expected_innings_pitched")
    low_ip_flag = pd.notna(expected_ip) and float(expected_ip) < 4.5

    if low_ip_flag:
        return "Low"
    if games >= 20 and has_statcast and lineup_ok:
        return "High"
    if games >= 10 and (has_statcast or lineup_ok):
        return "Medium"
    return "Low"


def _format_daily_board(picks: pd.DataFrame) -> pd.DataFrame:
    out = picks.copy()

    if "edge_pct" in out:
        sort_cols = ["edge_pct", "market"] + (["pitcher_name"] if "pitcher_name" in out.columns else [])
        sort_asc  = [False, True] + ([True] if "pitcher_name" in out.columns else [])
        out = out.sort_values(sort_cols, ascending=sort_asc, na_position="last")

    # One row per pitcher+market: keep the bookmaker with highest edge
    name_col = "pitcher_name" if "pitcher_name" in out.columns else None
    if name_col and "edge_pct" in out.columns and "market" in out.columns:
        out = (out.sort_values("edge_pct", ascending=False)
                  .drop_duplicates(subset=[name_col, "market"])
                  .sort_values(["edge_pct", "market"], ascending=[False, True])
                  .reset_index(drop=True))

    if "best_side" in out:
        out["hit_probability"] = out.apply(
            lambda row: row["over_probability"]
            if row.get("best_side") == "over"
            else row.get("under_probability"),
            axis=1,
        )
        out["fair_odds"] = out.apply(
            lambda row: row["fair_over_odds"]
            if row.get("best_side") == "over"
            else row.get("fair_under_odds"),
            axis=1,
        )
        out["recommended_odds"] = out.apply(
            lambda row: row["over_odds"] if row.get("best_side") == "over" else row.get("under_odds"),
            axis=1,
        )
        out["bookmaker"] = out.apply(
            lambda row: row.get("over_bookmaker")
            if row.get("best_side") == "over"
            else row.get("under_bookmaker"),
            axis=1,
        )
        out["hit_probability_pct"] = (out["hit_probability"] * 100).map(_format_pct)
        out["fair_odds"] = out["fair_odds"].map(_format_american)
        out["recommended_odds"] = out["recommended_odds"].map(_format_american)

    if "edge_pct" in out:
        out["edge"] = out["edge_pct"].map(_format_pct)
    if "kelly_fraction" in out:
        out["kelly"] = (out["kelly_fraction"] * 100).map(_format_pct)

    preferred_cols = [
        "game_date",
        "pitcher_name",
        "market",
        "best_side",
        "projection",
        "line",
        "recommended_odds",
        "fair_odds",
        "hit_probability_pct",
        "edge",
        "kelly",
        "confidence_tier",
        "expected_innings_pitched",
        "bookmaker",
        "pitcher_id",
        "team",
        "opponent",
        "is_home",
        "over_odds",
        "under_odds",
        "over_bookmaker",
        "under_bookmaker",
        "edge_pct",
        "kelly_fraction",
        "fetched_at",
    ]
    cols = [col for col in preferred_cols if col in out.columns]
    cols += [col for col in out.columns if col not in cols]
    return out[cols].reset_index(drop=True)


def _load_calibrated_residual_std(config: dict) -> dict[str, float]:
    calibration_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    calibration = load_calibration(calibration_path)
    if calibration:
        return residual_std_from_calibration(config, calibration)

    residual_std = dict(config["betting"]["default_residual_std"])
    score_path = Path(config["data"]["processed_dir"]) / "backtest_scores.csv"
    if not score_path.exists():
        return residual_std

    scores = pd.read_csv(score_path)
    if not {"market", "rmse"}.issubset(scores.columns):
        return residual_std

    for _, row in scores.iterrows():
        if pd.notna(row["rmse"]):
            residual_std[str(row["market"])] = float(row["rmse"])
    return residual_std


def _load_edge_shrink_factor(config: dict) -> float:
    calibration_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    calibration = load_calibration(calibration_path)
    return edge_shrink_from_calibration(config, calibration)


def _detect_distribution(models: dict, config: dict) -> dict:
    config_dist = config["betting"].get("market_distribution", {})
    out = {}
    for mkt in ["strikeouts", "walks", "hits_allowed"]:
        if mkt in config_dist:
            out[mkt] = config_dist[mkt]
        else:
            mt = models.get(mkt, {}).get("model_type", "random_forest")
            out[mkt] = _DISTRIBUTION_FOR_MODEL.get(mt, "normal")
    return out


def _daily_market_rows(
    projections: pd.DataFrame,
    odds: pd.DataFrame,
    residual_std: dict[str, float],
    max_kelly_fraction: float,
    edge_shrink_factor: float,
    distribution: dict,
    bias_corrections: dict,
    disabled_markets: list,
) -> pd.DataFrame:
    rows = []
    key_cols = ["game_date", "pitcher_id"]
    base_cols = [
        "game_date",
        "pitcher_id",
        "pitcher_name",
        "team",
        "opponent",
        "is_home",
        "confidence_tier",
    ]
    # Include expected IP if present
    if "expected_innings_pitched" in projections.columns:
        base_cols.append("expected_innings_pitched")

    base_cols = [c for c in base_cols if c in projections.columns]

    disabled = set(disabled_markets)

    if odds.empty:
        for market in TARGETS:
            market_df = projections[base_cols + [f"{market}_projection"]].copy()
            market_df["market"] = market
            market_df["betting_disabled"] = market in disabled
            market_df = market_df.rename(columns={f"{market}_projection": "projection"})
            rows.append(market_df)
        return pd.concat(rows, ignore_index=True, sort=False)

    for market in TARGETS:
        market_odds = odds[odds["market"] == market].copy()
        # Drop cols from odds that already exist in projections (except merge keys) to avoid _x/_y suffixes
        odds_drop = [c for c in market_odds.columns if c in base_cols and c not in key_cols]
        market_odds = market_odds.drop(columns=odds_drop, errors="ignore")
        market_proj = projections[base_cols + [f"{market}_projection"]].copy()
        merged = market_proj.merge(market_odds, on=key_cols, how="left")
        merged["market"] = market
        merged["betting_disabled"] = market in disabled
        dist = distribution.get(market, "normal")
        bias = bias_corrections.get(market, 0.0)
        if market not in disabled:
            merged = add_betting_columns(
                merged.rename(columns={f"{market}_projection": "projection"}).assign(
                    **{f"{market}_projection": lambda x: x["projection"]}
                ),
                market=market,
                residual_std=residual_std[market],
                max_kelly_fraction=max_kelly_fraction,
                edge_shrink_factor=edge_shrink_factor,
                distribution=dist,
                bias_correction=bias,
            )
        else:
            merged = merged.rename(columns={f"{market}_projection": "projection"})
        rows.append(merged)

    return pd.concat(rows, ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--date", default=date.today().isoformat())
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

    # Inject today's confirmed/projected lineup players so the model computes
    # real matchup features instead of using fill-value averages.
    today_lineups_path = config["data"].get("today_lineups_file", "data/raw/today_lineups.csv")
    _today_lineups = Path(today_lineups_path)
    if _today_lineups.exists():
        _tl = pd.read_csv(_today_lineups)
        if not _tl.empty:
            _tl["game_date"] = pd.to_datetime(_tl["game_date"])
            _tl_today = _tl[_tl["game_date"].dt.date.astype(str) == args.date]
            if not _tl_today.empty and batter_logs is not None and not batter_logs.empty:
                batter_logs = pd.concat([batter_logs, _tl_today], ignore_index=True, sort=False)
                print(f"Injected {len(_tl_today)} lineup players for {args.date} into batter features")
            elif not _tl_today.empty:
                batter_logs = _tl_today
                print(f"Using {len(_tl_today)} lineup players for {args.date} as batter features")

    statcast = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    statcast_batter_pitch_types = load_statcast_batter_pitch_type_daily(
        config["data"]["statcast_batter_pitch_type_daily_file"]
    )
    park_factors = load_park_factors(config["data"]["park_factors_file"])
    probable = load_probable_pitchers(config["data"]["probable_pitchers_file"], args.date)
    if probable.empty:
        raise ValueError(f"No probable pitchers found for {args.date}")

    fangraphs_path = config["data"].get("fangraphs_file", "")
    fangraphs = load_fangraphs_stats(fangraphs_path) if fangraphs_path else pd.DataFrame()

    adv_pitcher_path = config["data"].get("statcast_pitcher_advanced_file", "")
    statcast_pitcher_adv = load_statcast_pitcher_advanced(adv_pitcher_path) if adv_pitcher_path else pd.DataFrame()

    bat_disc_path = config["data"].get("statcast_batter_discipline_file", "")
    statcast_bat_disc = load_statcast_batter_discipline(bat_disc_path) if bat_disc_path else pd.DataFrame()

    # Load fill_values from training run for leak-free imputation
    model_dir = Path(config["data"]["processed_dir"]) / "models"
    fill_values = load_fill_values(model_dir / "fill_values.json")

    # ------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------
    features = build_daily_features(
        historical_logs=logs,
        probable_pitchers=probable,
        rolling_windows=config["features"]["rolling_windows"],
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
        features, _ = merge_fangraphs_prior_season(features, fangraphs)

    # ------------------------------------------------------------------
    # Opportunity models + predict
    # ------------------------------------------------------------------
    opportunity_models = load_opportunity_models(model_dir)
    if opportunity_models:
        features, _ = add_expected_opportunity_features(features, opportunity_models)

    models = load_models(model_dir)
    projections = predict_targets(features, models)
    distribution = _detect_distribution(models, config)

    # ------------------------------------------------------------------
    # Confidence tiers (uses p_games_prior, Statcast cols, lineup confirmation)
    # ------------------------------------------------------------------
    projections["confidence_tier"] = projections.apply(_assign_confidence_tier, axis=1)

    # ------------------------------------------------------------------
    # Odds + betting columns
    # ------------------------------------------------------------------
    calibration_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    calibration = load_calibration(calibration_path)
    bias_corrections = bias_corrections_from_calibration(config, calibration)
    disabled_markets = config["betting"].get("disabled_markets", [])

    odds = load_odds(config["data"]["odds_file"])
    odds = odds[odds["game_date"] == pd.to_datetime(args.date)].copy()

    picks = _daily_market_rows(
        projections,
        odds,
        residual_std=_load_calibrated_residual_std(config),
        max_kelly_fraction=config["betting"]["max_kelly_fraction"],
        edge_shrink_factor=_load_edge_shrink_factor(config),
        distribution=distribution,
        bias_corrections=bias_corrections,
        disabled_markets=disabled_markets,
    )
    picks = _format_daily_board(picks)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    out_path = Path(config["data"]["exports_dir"]) / f"daily_pitcher_props_{args.date}.csv"
    export_csv(picks, out_path)
    excel_path = Path(config["data"]["exports_dir"]) / f"daily_pitcher_props_{args.date}.xlsx"
    try:
        export_pretty_excel(picks, excel_path)
    except PermissionError:
        timestamp = datetime.now().strftime("%H%M%S")
        excel_path = (
            Path(config["data"]["exports_dir"])
            / f"daily_pitcher_props_{args.date}_{timestamp}.xlsx"
        )
        export_pretty_excel(picks, excel_path)

    if config["export"]["google_sheets"]:
        export_google_sheets(picks, config["export"]["google_sheet_name"])

    # Print summary
    print(f"Daily projections saved to {out_path}")

    print(f"Styled daily board saved to {excel_path}")
    min_e = config["betting"]["min_edge_pct"]
    max_e = config["betting"].get("max_edge_pct", float("inf"))
    flagged = picks[
        picks.get("edge_pct", pd.Series(dtype=float)).between(min_e, max_e, inclusive="both")
    ] if "edge_pct" in picks.columns else pd.DataFrame()

    # Filter: projection must agree with direction (over→proj>line, under→proj<line)
    if not flagged.empty and "strikeouts_projection" in flagged.columns:
        proj_col = flagged["strikeouts_projection"]
        agrees = (
            ((flagged["best_side"] == "over")  & (proj_col > flagged["line"])) |
            ((flagged["best_side"] == "under") & (proj_col < flagged["line"]))
        )
        flagged = flagged[agrees].copy()

    # Deduplicate: one bet per pitcher (highest edge)
    _name_col = "pitcher_name" if "pitcher_name" in flagged.columns else "player_name"
    if not flagged.empty and _name_col in flagged.columns:
        flagged = (flagged.sort_values("edge_pct", ascending=False)
                          .drop_duplicates(subset=[_name_col])
                          .sort_values("edge_pct", ascending=False)
                          .reset_index(drop=True))

    if not flagged.empty:
        print(f"\n{len(flagged)} flagged bets (edge {min_e}%–{max_e}%):")
        for _, row in flagged.iterrows():
            tier = row.get("confidence_tier", "?")
            proj_val = row.get("strikeouts_projection", row.get("projection", float("nan")))
            line_val = row.get("line", float("nan"))
            side = row.get("best_side", "?")
            try:
                raw_gap = proj_val - line_val
                gap_str = f"+{raw_gap:.2f}" if raw_gap >= 0 else f"{raw_gap:.2f}"
            except Exception:
                gap_str = "?"
            odds_val = row.get("over_odds") if side == "over" else row.get("under_odds")
            odds_str = f"{int(odds_val):+d}" if pd.notna(odds_val) else "?"
            pname = row.get("pitcher_name", row.get("player_name", "?"))
            print(
                f"  {pname} | {side} {line_val} "
                f"| proj={proj_val:.2f} gap={gap_str} | edge={row.get('edge_pct', row.get('edge','?')):.1f}% "
                f"odds={odds_str} | [{tier}]"
            )

    # Save canonical picks log (append) so results can be resolved later
    picks_log_path = Path(config["data"]["exports_dir"]) / "picks_log.csv"
    log_cols = ["game_date", "pitcher_name", "best_side", "line", "strikeouts_projection",
                "edge_pct", "over_odds", "under_odds", "confidence_tier"]
    if not flagged.empty:
        log_rows = []
        for _, row in flagged.iterrows():
            proj_val = row.get("strikeouts_projection", row.get("projection", float("nan")))
            side = row.get("best_side", "?")
            odds_val = row.get("over_odds") if side == "over" else row.get("under_odds")
            log_rows.append({
                "game_date": args.date,
                "pitcher_name": row.get("pitcher_name", row.get("player_name", "")),
                "best_side": side,
                "line": row.get("line", float("nan")),
                "strikeouts_projection": proj_val,
                "gap": (proj_val - row.get("line", float("nan"))) if pd.notna(proj_val) else float("nan"),
                "edge_pct": round(float(row.get("edge_pct", 0)), 4),
                "odds_used": odds_val,
                "actual": "",
                "won": "",
            })
        log_df = pd.DataFrame(log_rows)
        if picks_log_path.exists():
            existing = pd.read_csv(picks_log_path, dtype=str)
            # Remove any rows for this date so re-runs don't duplicate
            existing = existing[existing["game_date"] != str(args.date)]
            log_df = pd.concat([existing, log_df.astype(str)], ignore_index=True)
        log_df.to_csv(picks_log_path, index=False)
        print(f"Picks log updated: {picks_log_path}")


if __name__ == "__main__":
    main()
