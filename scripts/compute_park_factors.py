"""Compute park strikeout factors from pitcher logs and game context.

Uses prior-year K/IP at each venue relative to league average.
The loader applies them as: factor_year = game_date.year - 1.

Output: data/raw/park_factors.csv

Run once before training:
  python scripts/compute_park_factors.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.config import load_config

CONFIG = "config/config_v4_production.yaml"
MIN_GAMES = 40  # minimum games to include a venue (excludes special events)


def main():
    config = load_config(CONFIG)

    logs = pd.read_csv(config["data"]["pitcher_logs_file"], low_memory=False)
    ctx  = pd.read_csv(config["data"]["game_context_logs_file"], low_memory=False)

    logs["game_date"] = pd.to_datetime(logs["game_date"])
    ctx["game_date"]  = pd.to_datetime(ctx["game_date"])

    merged = logs.merge(
        ctx[["game_date", "game_pk", "venue_id", "venue_name"]],
        on=["game_date", "game_pk"],
        how="left",
    )

    merged["year"]            = merged["game_date"].dt.year
    merged["innings_pitched"] = pd.to_numeric(merged["innings_pitched"], errors="coerce")
    merged["strikeouts"]      = pd.to_numeric(merged["strikeouts"], errors="coerce")
    merged = merged.dropna(subset=["venue_id", "innings_pitched", "strikeouts"])
    merged = merged[merged["innings_pitched"] > 0]

    # League average K/IP per year
    lg = (
        merged.groupby("year")
        .apply(lambda x: x["strikeouts"].sum() / x["innings_pitched"].sum())
        .reset_index(name="lg_k_per_ip")
    )

    # Per-venue-year K/IP
    by_vy = (
        merged.groupby(["venue_id", "venue_name", "year"])
        .agg(k=("strikeouts", "sum"), ip=("innings_pitched", "sum"), games=("game_pk", "nunique"))
        .reset_index()
    )
    by_vy = by_vy.merge(lg, on="year")

    # Only regular MLB parks
    by_vy = by_vy[by_vy["games"] >= MIN_GAMES].copy()

    by_vy["park_so_factor"] = (by_vy["k"] / by_vy["ip"]) / by_vy["lg_k_per_ip"]
    by_vy["factor_year"]    = by_vy["year"]

    out = by_vy[["factor_year", "venue_id", "venue_name", "park_so_factor"]].copy()

    # Fill other required columns with 1.0 (neutral — not computed)
    for col in ["park_runs_factor", "park_hits_factor", "park_bb_factor",
                "park_hr_factor", "park_1b_factor", "park_2b_factor", "park_3b_factor"]:
        out[col] = 1.0

    out = out.sort_values(["venue_id", "factor_year"]).reset_index(drop=True)

    out_path = Path(config["data"]["park_factors_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved {len(out)} rows to {out_path}")
    print(f"  Years: {sorted(out.factor_year.unique())}")
    print(f"  Venues: {out.venue_id.nunique()}")
    print(f"  park_so_factor range: {out.park_so_factor.min():.3f} to {out.park_so_factor.max():.3f}")
    top5 = out.sort_values("park_so_factor", ascending=False).head(5)
    print(f"  Top K parks (latest year):\n{top5[['factor_year','venue_name','park_so_factor']].to_string(index=False)}")


if __name__ == "__main__":
    main()
