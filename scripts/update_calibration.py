"""Dynamic weekly bias recalibration.

Run this every 7-14 days during the season.  It loads the most recent
N days of actual game results vs model projections, recomputes the bias
correction for strikeouts, and patches calibration.json in-place.

This adapts to K-rate drift at the start of each season without
requiring a full retrain.

Usage:
    python scripts/update_calibration.py
    python scripts/update_calibration.py --window 21
    python scripts/update_calibration.py --window 21 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.models.calibration import load_calibration, save_calibration

DEFAULT_WINDOW_DAYS = 21
MIN_SAMPLES         = 25


def load_recent_predictions(config: dict, window_days: int) -> pd.DataFrame:
    """Load the most recently saved backtest predictions and filter to the last window_days."""
    pred_path = Path(config["data"]["processed_dir"]) / "backtest_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions file found at {pred_path}. Run scripts/backtest.py first.")

    preds = pd.read_csv(pred_path)
    preds["game_date"] = pd.to_datetime(preds["game_date"])
    cutoff = preds["game_date"].max() - pd.Timedelta(days=window_days)
    recent = preds[preds["game_date"] > cutoff].copy()
    return recent


def compute_rolling_bias(preds: pd.DataFrame, target: str, window_days: int) -> dict:
    """Compute rolling bias from recent actual vs projected values."""
    proj_col = f"{target}_projection"
    if target not in preds.columns or proj_col not in preds.columns:
        return {}

    sub = preds[[target, proj_col, "game_date"]].dropna()
    if len(sub) < MIN_SAMPLES:
        return {"skipped": True, "reason": f"only {len(sub)} samples (need {MIN_SAMPLES})"}

    residuals = sub[target] - sub[proj_col]
    bias      = float(residuals.mean())
    mae       = float(residuals.abs().mean())
    n         = len(sub)
    date_range = f"{sub['game_date'].min().date()} to {sub['game_date'].max().date()}"

    return {
        "bias":          bias,
        "mae":           mae,
        "rows":          n,
        "window_days":   window_days,
        "date_range":    date_range,
        "updated_at":    date.today().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--window",  type=int, default=DEFAULT_WINDOW_DAYS,
                        help="Rolling window in days (default 21)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print new biases without writing calibration.json")
    args = parser.parse_args()

    config      = load_config(args.config)
    cal_path    = Path(config["data"]["processed_dir"]) / "calibration.json"
    calibration = load_calibration(cal_path)

    if not calibration:
        print("No existing calibration.json found — run scripts/backtest.py --save-calibration first.")
        return

    try:
        recent = load_recent_predictions(config, args.window)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Loaded {len(recent)} predictions | "
          f"date range: {recent['game_date'].min().date()} → {recent['game_date'].max().date()}")
    print(f"Rolling window: {args.window} days  |  min samples: {MIN_SAMPLES}\n")

    updated = False
    from src.data.schema import TARGETS
    for target in TARGETS:
        new_stats = compute_rolling_bias(recent, target, args.window)

        if not new_stats or new_stats.get("skipped"):
            reason = new_stats.get("reason", "no data") if new_stats else "no data"
            print(f"  {target:15s}  SKIPPED — {reason}")
            continue

        old_bias = calibration.get("markets", {}).get(target, {}).get("bias", 0.0)
        new_bias = new_stats["bias"]
        delta    = new_bias - old_bias

        print(f"  {target:15s}  "
              f"old bias={old_bias:+.4f}  new bias={new_bias:+.4f}  "
              f"delta={delta:+.4f}  "
              f"n={new_stats['rows']}  range={new_stats['date_range']}")

        if not args.dry_run:
            if target not in calibration.setdefault("markets", {}):
                calibration["markets"][target] = {}
            calibration["markets"][target].update({
                "bias":        new_bias,
                "mae":         new_stats["mae"],
                "rows":        new_stats["rows"],
                "window_days": args.window,
                "updated_at":  new_stats["updated_at"],
            })
            updated = True

    if updated:
        save_calibration(calibration, cal_path)
        print(f"\nCalibration updated → {cal_path}")
        print("Run scripts/backtest.py (or scripts/project_daily.py) to use new biases.")
    elif args.dry_run:
        print("\n[dry-run] No changes written.")
    else:
        print("\nNo updates made.")


if __name__ == "__main__":
    main()
