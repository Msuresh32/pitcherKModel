"""Fetch pitcher prop odds for today and save to data/odds/pitcher_props.csv."""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import fetch_pitcher_prop_odds

PROBS_FILE = Path("data/raw/probable_pitchers.csv")
OUTPUT     = Path("data/odds/pitcher_props.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    target = args.date
    probs = pd.read_csv(PROBS_FILE)
    probs_today = probs[probs["game_date"].astype(str).str[:10] == target]
    print(f"Pitchers for {target}: {len(probs_today)}")

    result = fetch_pitcher_prop_odds(probs_today, target_date=target, markets=["pitcher_strikeouts"])
    df = result[0] if isinstance(result, tuple) else result
    print(f"Odds rows fetched: {len(df)}")

    if not df.empty:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT, index=False)
        print(f"Saved to {OUTPUT}")
        print(df[["pitcher_name", "line", "over_odds", "under_odds"]].drop_duplicates().to_string(index=False))
    else:
        print("No odds found.")


if __name__ == "__main__":
    main()
