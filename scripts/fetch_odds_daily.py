"""Fetch pitcher prop odds for today.

Saves two outputs:
  1. data/odds/pitcher_props.csv          — live file used by project_daily.py (morning only)
  2. data/odds/snapshots/{date}_{snapshot}.csv — timestamped snapshot for CLV tracking

--snapshot morning  (default) : 7am pick-time odds
--snapshot closing             : pre-game closing odds (~6:45pm ET)
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import fetch_pitcher_prop_odds

PROBS_FILE   = Path("data/raw/probable_pitchers.csv")
LIVE_OUTPUT  = Path("data/odds/pitcher_props.csv")
SNAPSHOT_DIR = Path("data/odds/snapshots")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--snapshot",
        choices=["morning", "closing", "nightly"],
        default="morning",
        help="Label this fetch: morning=7am open, closing=pre-game, nightly=11pm early lines.",
    )
    args = parser.parse_args()

    target = args.date
    probs = pd.read_csv(PROBS_FILE)
    probs_today = probs[probs["game_date"].astype(str).str[:10] == target]
    print(f"Pitchers for {target}: {len(probs_today)} | snapshot={args.snapshot}")

    result = fetch_pitcher_prop_odds(probs_today, target_date=target, markets=["pitcher_strikeouts"])
    df = result[0] if isinstance(result, tuple) else result
    print(f"Odds rows fetched: {len(df)}")

    if df.empty:
        print("No odds found.")
        return

    # Always save timestamped snapshot for CLV tracking
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / f"{target}_{args.snapshot}.csv"
    df.to_csv(snapshot_path, index=False)
    print(f"Snapshot saved: {snapshot_path}")

    # Morning and nightly fetches update the live file used by project_daily.py
    if args.snapshot in ("morning", "nightly"):
        LIVE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LIVE_OUTPUT, index=False)
        print(f"Live odds updated: {LIVE_OUTPUT}")

    print(df[["pitcher_name", "line", "over_odds", "under_odds"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
