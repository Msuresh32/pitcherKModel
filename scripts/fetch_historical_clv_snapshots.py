"""
Fetch historical closing odds for backtest CLV computation.

For each qualifying bet in the backtest, we have entry odds from the existing
~4h-before-game snapshot (historical_pitcher_props_plus_2026_6h.csv).
This script fetches the CLOSING odds (15min before first pitch) for each
unique event that had a qualifying bet, then computes proper CLV.

Usage:
    python scripts/fetch_historical_clv_snapshots.py           # fetch + compute
    python scripts/fetch_historical_clv_snapshots.py --compute-only  # skip fetch, just recompute

API cost: 1 call per unique event (game). ~276 calls for the full backtest period.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import (
    fetch_historical_event_odds,
    get_api_key,
    normalize_event_odds,
)

HIST_ODDS_FILE  = Path("data/odds/historical_pitcher_props_plus_2026_6h.csv")
CLOSING_FILE    = Path("data/odds/historical_closing_odds.csv")
CLV_OUTPUT      = Path("data/processed/backtest_clv_historical.csv")
MIN_EDGE_PCT    = 10.0   # match the best config


def american_to_decimal(odds) -> float:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(o):
        return float("nan")
    return (100 / abs(o) + 1) if o < 0 else (o / 100 + 1)


def closing_snapshot_time(commence_time: str, minutes_before: int = 15) -> str:
    ts = pd.to_datetime(commence_time, utc=True)
    return (ts - pd.Timedelta(minutes=minutes_before)).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_all_closing_odds(events_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch closing odds for each unique event. Skips events already in CLOSING_FILE."""
    api_key = get_api_key()

    # Load already-fetched events to avoid re-fetching
    already_fetched = set()
    if CLOSING_FILE.exists():
        existing = pd.read_csv(CLOSING_FILE)
        already_fetched = set(existing["event_id"].dropna().unique())
        print(f"Already have closing odds for {len(already_fetched)} events.")

    unique_events = events_df[["event_id", "commence_time"]].drop_duplicates("event_id")
    to_fetch = unique_events[~unique_events["event_id"].isin(already_fetched)]
    print(f"Need to fetch closing odds for {len(to_fetch)} events.")

    new_rows = []
    for i, (_, row) in enumerate(to_fetch.iterrows()):
        event_id = row["event_id"]
        commence_time = row["commence_time"]
        snapshot_time = closing_snapshot_time(commence_time, minutes_before=15)

        try:
            event_data, _, _ = fetch_historical_event_odds(
                api_key=api_key,
                event_id=event_id,
                snapshot_date=snapshot_time,
                markets=["pitcher_strikeouts"],
            )
            if event_data:
                frame = normalize_event_odds(event_data, fetched_at=snapshot_time)
                if not frame.empty:
                    new_rows.append(frame)
        except Exception as e:
            print(f"  [{i+1}/{len(to_fetch)}] ERROR {event_id}: {e}")
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(to_fetch)}] fetched {event_id[:12]}... ({snapshot_time[:10]})")

        # Be polite to the API — 3 calls/second max
        time.sleep(0.35)

    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        # Append to existing file
        if CLOSING_FILE.exists():
            existing = pd.read_csv(CLOSING_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        CLOSING_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(CLOSING_FILE, index=False)
        print(f"Closing odds saved: {len(combined)} rows in {CLOSING_FILE}")
    else:
        print("No new closing odds fetched.")

    return pd.read_csv(CLOSING_FILE) if CLOSING_FILE.exists() else pd.DataFrame()


def compute_clv(entry_df: pd.DataFrame, closing_df: pd.DataFrame, qualifying_bets: pd.DataFrame) -> pd.DataFrame:
    """
    For each qualifying bet:
      entry odds  = 4h-before snapshot (from historical_pitcher_props_plus_2026_6h.csv)
      closing odds = 15min-before-game (fetched above)
      CLV% = (entry_decimal / close_decimal - 1) * 100
    """
    # Best entry price per (event_id, market, player_name, line)
    def best_odds(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        rows = []
        for keys, grp in df.groupby(["event_id", "market", "player_name", "line"], dropna=False):
            over_row  = grp.sort_values("over_odds",  ascending=False, na_position="last").iloc[0]
            under_row = grp.sort_values("under_odds", ascending=False, na_position="last").iloc[0]
            rows.append({
                "event_id":   keys[0],
                "market":     keys[1],
                "player_name": keys[2],
                "line":       keys[3],
                "over_odds":  over_row["over_odds"],
                "under_odds": under_row["under_odds"],
            })
        return pd.DataFrame(rows)

    entry_best   = best_odds(entry_df)
    closing_best = best_odds(closing_df)

    results = []
    for _, bet in qualifying_bets.iterrows():
        pid    = bet.get("pitcher_id")
        name   = str(bet.get("pitcher_name", "")).strip()
        mkt    = str(bet.get("market", "strikeouts"))
        line   = float(bet.get("line", float("nan")))
        side   = str(bet.get("best_side", ""))
        eid    = str(bet.get("event_id", ""))
        actual = bet.get("strikeouts", float("nan"))

        # Outcome
        if pd.notna(actual):
            if side == "over":
                won = 1 if actual > line else 0
            else:
                won = 1 if actual <= line else 0
        else:
            won = float("nan")

        # Entry odds (use pitcher_name for matching since event_id + player_name is most reliable)
        e_rows = entry_best[
            (entry_best["event_id"] == eid) &
            (entry_best["market"] == mkt) &
            (entry_best["line"] == line)
        ]
        c_rows = closing_best[
            (closing_best["event_id"] == eid) &
            (closing_best["market"] == mkt) &
            (closing_best["line"] == line)
        ]

        entry_odds = closing_odds = clv_pct = float("nan")
        if not e_rows.empty:
            entry_odds = e_rows.iloc[0]["over_odds"] if side == "over" else e_rows.iloc[0]["under_odds"]
        if not c_rows.empty:
            closing_odds = c_rows.iloc[0]["over_odds"] if side == "over" else c_rows.iloc[0]["under_odds"]

        if pd.notna(entry_odds) and pd.notna(closing_odds):
            e_dec = american_to_decimal(entry_odds)
            c_dec = american_to_decimal(closing_odds)
            if not np.isnan(e_dec) and not np.isnan(c_dec) and c_dec > 1:
                clv_pct = (e_dec / c_dec - 1) * 100

        results.append({
            "game_date":     bet.get("game_date"),
            "pitcher_name":  bet.get("pitcher_name"),
            "pitcher_id":    pid,
            "market":        mkt,
            "line":          line,
            "best_side":     side,
            "edge_pct":      bet.get("edge_pct"),
            "event_id":      eid,
            "entry_odds":    entry_odds,
            "closing_odds":  closing_odds,
            "clv_pct":       clv_pct,
            "actual":        actual,
            "won":           won,
        })

    return pd.DataFrame(results)


def print_summary(clv_df: pd.DataFrame) -> None:
    valid = clv_df.dropna(subset=["clv_pct"])
    print(f"\n{'='*55}")
    print(f"HISTORICAL BACKTEST CLV  ({len(valid)} bets with closing lines)")
    print(f"{'='*55}")
    if valid.empty:
        print("No matched closing lines found.")
        return

    mean_clv    = valid["clv_pct"].mean()
    median_clv  = valid["clv_pct"].median()
    pct_positive = (valid["clv_pct"] > 0).mean()
    print(f"Mean CLV:           {mean_clv:+.3f}%")
    print(f"Median CLV:         {median_clv:+.3f}%")
    print(f"Beat closing line:  {pct_positive:.1%}  ({(valid['clv_pct']>0).sum()}/{len(valid)})")
    print(f"Std dev:            {valid['clv_pct'].std():.3f}%")

    # t-stat: is mean CLV significantly != 0?
    import math
    n = len(valid)
    se = valid["clv_pct"].std() / math.sqrt(n)
    t = mean_clv / se if se > 0 else float("nan")
    print(f"t-stat (vs 0):      {t:.2f}  (|t|>2 = statistically significant)")

    # CLV vs actual win rate
    settled = valid.dropna(subset=["won"])
    if not settled.empty:
        settled = settled.copy()
        settled["won_bool"] = settled["won"].astype(float) == 1.0
        print(f"\nOf {len(settled)} settled bets:")
        print(f"  Win rate:           {settled['won_bool'].mean():.1%}")
        for label, mask in [("CLV > 0", settled["clv_pct"] > 0), ("CLV <= 0", settled["clv_pct"] <= 0)]:
            sub = settled[mask]
            if not sub.empty:
                print(f"  Win rate ({label}):  {sub['won_bool'].mean():.1%}  (n={len(sub)})")

    # By edge bucket
    print(f"\nCLV by edge bucket:")
    bins = [0, 12, 15, 20, 100]
    labels = ["10-12%", "12-15%", "15-20%", "20%+"]
    valid2 = valid.copy()
    valid2["edge_bucket"] = pd.cut(valid2["edge_pct"].astype(float), bins=bins, labels=labels, right=False)
    for bucket, grp in valid2.groupby("edge_bucket", observed=True):
        print(f"  [{bucket}] n={len(grp)}, mean CLV={grp['clv_pct'].mean():+.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-only", action="store_true",
                        help="Skip API fetching — just recompute CLV from already-fetched data.")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_PCT)
    args = parser.parse_args()

    # Load historical entry odds
    print(f"Loading entry odds from {HIST_ODDS_FILE}...")
    entry_df = pd.read_csv(HIST_ODDS_FILE)
    entry_2026 = entry_df[entry_df["game_date"] >= "2026-01-01"].copy()

    # Load qualifying bets from backtest
    print("Loading qualifying bets from backtest_edges.csv...")
    bets = pd.read_csv("data/processed/backtest_edges.csv")
    qual = bets[
        (bets["market"] == "strikeouts") &
        (bets["edge_pct"] >= args.min_edge)
    ].copy()
    print(f"Qualifying bets (edge>={args.min_edge}%): {len(qual)}")

    # Match event_ids from entry file
    keys = entry_2026[["pitcher_id", "game_date", "market", "event_id", "commence_time", "line"]].drop_duplicates()
    qual = qual.merge(keys, on=["pitcher_id", "game_date", "market", "line"], how="left")
    matched = qual["event_id"].notna().sum()
    print(f"Matched to event_id: {matched}/{len(qual)}")
    print(f"Unique events to fetch closing odds for: {qual['event_id'].nunique()}")

    if not args.compute_only:
        closing_df = fetch_all_closing_odds(qual[["event_id", "commence_time"]].dropna())
    else:
        if not CLOSING_FILE.exists():
            print(f"No closing file found at {CLOSING_FILE}. Run without --compute-only first.")
            return
        closing_df = pd.read_csv(CLOSING_FILE)
        print(f"Loaded {len(closing_df)} closing odds rows from {CLOSING_FILE}")

    print("\nComputing CLV...")
    clv_df = compute_clv(entry_2026, closing_df, qual)
    CLV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clv_df.to_csv(CLV_OUTPUT, index=False)
    print(f"CLV results saved to {CLV_OUTPUT}")

    print_summary(clv_df)


if __name__ == "__main__":
    main()
