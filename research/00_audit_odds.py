"""Quick check of the historical odds file as it accumulates."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

ODDS_FILE = Path("data/odds/historical/pitcher_strikeouts_2025_2026.csv")


def no_vig_prob(over_odds, under_odds):
    """Return (p_over_novig, p_under_novig, total_vig) from American odds."""
    def imp(o):
        o = float(o)
        return -o / (-o + 100) if o < 0 else 100 / (100 + o)
    p_o = imp(over_odds)
    p_u = imp(under_odds)
    total = p_o + p_u
    return p_o / total, p_u / total, total - 1.0


def main():
    if not ODDS_FILE.exists():
        print(f"Not found: {ODDS_FILE}")
        return

    df = pd.read_csv(ODDS_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    print("=== ODDS FILE AUDIT ===")
    print(f"Total rows:     {len(df):,}")
    print(f"Date range:     {df.game_date.min().date()} to {df.game_date.max().date()}")
    print(f"Snapshot types: {df.snapshot_type.value_counts().to_dict()}")
    print(f"Bookmakers:     {sorted(df.bookmaker.dropna().unique())}")
    print(f"Lines offered:  {sorted(df.line.dropna().unique())}")
    print(f"Markets:        {df.market.dropna().unique().tolist()}")

    # Rows with both sides
    both = df.dropna(subset=["over_odds", "under_odds"])
    print(f"\nRows with both sides: {len(both):,}")

    # No-vig prices
    both = both.copy()
    both[["p_over", "p_under", "vig"]] = both.apply(
        lambda r: pd.Series(no_vig_prob(r.over_odds, r.under_odds)), axis=1
    )
    print(f"Avg vig:        {both.vig.mean():.3f}")
    print(f"Vig range:      {both.vig.min():.3f} to {both.vig.max():.3f}")
    print(f"\nNo-vig P(over) distribution:")
    print(both.p_over.describe().round(3))

    # Unique pitcher-game-lines (close snapshot only)
    close = df[df.snapshot_type == "close"]
    print(f"\nClose-snapshot pitcher-line rows: {len(close):,}")
    by_day = close.groupby("game_date")["player_name"].nunique()
    print(f"Avg pitchers/day: {by_day.mean():.1f}")
    print(f"Days with data:   {by_day.shape[0]}")


if __name__ == "__main__":
    main()
