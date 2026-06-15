import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.backtest import attach_odds_and_edges
from src.config import ensure_directories, load_config
from src.models.calibration import (
    edge_shrink_from_calibration,
    load_calibration,
    residual_std_from_calibration,
)
from src.odds.pricing import american_to_decimal


def _grade_bets(scored: pd.DataFrame, bankroll: float, min_edge_pct: float) -> pd.DataFrame:
    if scored.empty or "edge_pct" not in scored:
        return pd.DataFrame()

    bets = scored[scored["edge_pct"] >= min_edge_pct].copy()
    if bets.empty:
        return bets

    bets["bet_odds"] = np.where(
        bets["best_side"] == "over",
        bets["over_odds"],
        bets["under_odds"],
    )
    actual_by_market = [
        bets.loc[bets["market"] == market, market]
        for market in ("strikeouts", "walks", "hits_allowed")
        if market in bets.columns
    ]
    if not actual_by_market:
        raise ValueError("Could not find actual result columns to grade bets.")
    bets["actual_result"] = pd.concat(actual_by_market).sort_index()
    bets["won"] = np.where(
        bets["best_side"] == "over",
        bets["actual_result"] > bets["line"],
        bets["actual_result"] < bets["line"],
    )
    bets["push"] = bets["actual_result"] == bets["line"]
    bets["decimal_odds"] = bets["bet_odds"].map(american_to_decimal)
    bets["unit_profit"] = np.where(
        bets["push"],
        0.0,
        np.where(bets["won"], bets["decimal_odds"] - 1, -1.0),
    )
    bets["kelly_stake"] = bankroll * bets["kelly_fraction"].fillna(0)
    bets["kelly_profit"] = np.where(
        bets["push"],
        0.0,
        np.where(bets["won"], bets["kelly_stake"] * (bets["decimal_odds"] - 1), -bets["kelly_stake"]),
    )
    return bets


def _summary_rows(bets: pd.DataFrame, group_cols: Optional[list[str]] = None) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    group_cols = group_cols or []
    rows = []
    iterator = bets.groupby(group_cols, dropna=False) if group_cols else [((), bets)]
    for keys, group in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        decided = group[~group["push"]]
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "bets": int(len(group)),
                "pushes": int(group["push"].sum()),
                "wins": int(group["won"].sum()),
                "losses": int((~group["won"] & ~group["push"]).sum()),
                "win_rate_ex_push": float(decided["won"].mean()) if len(decided) else np.nan,
                "unit_profit": float(group["unit_profit"].sum()),
                "unit_roi": float(group["unit_profit"].sum() / len(group)) if len(group) else 0.0,
                "kelly_staked": float(group["kelly_stake"].sum()),
                "kelly_profit": float(group["kelly_profit"].sum()),
                "kelly_roi": float(group["kelly_profit"].sum() / group["kelly_stake"].sum())
                if group["kelly_stake"].sum()
                else 0.0,
                "avg_edge_pct": float(group["edge_pct"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest model picks against historical prop odds.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--predictions", default="data/processed/backtest_predictions.csv")
    parser.add_argument("--odds", default="data/odds/historical_pitcher_props.csv")
    parser.add_argument("--min-edge-pct", type=float, default=None)
    parser.add_argument("--output-prefix", default="data/processed/historical_odds")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    predictions = pd.read_csv(args.predictions)
    odds = pd.read_csv(args.odds)
    if odds.empty:
        raise ValueError(f"No historical odds rows found in {args.odds}")

    calibration_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    calibration = load_calibration(calibration_path)
    residual_std = residual_std_from_calibration(config, calibration)
    edge_shrink_factor = edge_shrink_from_calibration(config, calibration)
    min_edge_pct = (
        args.min_edge_pct
        if args.min_edge_pct is not None
        else float(config["betting"]["min_edge_pct"])
    )

    scored = attach_odds_and_edges(
        predictions,
        odds,
        residual_std=residual_std,
        max_kelly_fraction=float(config["betting"]["max_kelly_fraction"]),
        edge_shrink_factor=edge_shrink_factor,
    )
    bets = _grade_bets(scored, bankroll=float(config["betting"]["bankroll"]), min_edge_pct=min_edge_pct)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(prefix.with_name(prefix.name + "_edges.csv"), index=False)
    bets.to_csv(prefix.with_name(prefix.name + "_bets.csv"), index=False)

    summary = pd.concat(
        [
            _summary_rows(bets),
            _summary_rows(bets, ["market"]),
            _summary_rows(bets, ["bookmaker"]),
            _summary_rows(bets, ["market", "bookmaker"]),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)

    print(f"Historical odds edges saved to {prefix.with_name(prefix.name + '_edges.csv')}")
    print(f"Historical odds bets saved to {prefix.with_name(prefix.name + '_bets.csv')}")
    print(f"Historical odds summary saved to {prefix.with_name(prefix.name + '_summary.csv')}")
    if not summary.empty:
        print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
