"""Fetch probable pitchers for today and append to probable_pitchers.csv."""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mlb_source import fetch_probable_pitchers

OUTPUT = Path("data/raw/probable_pitchers.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    target = args.date
    new_df = fetch_probable_pitchers(target)
    print(f"Fetched {len(new_df)} probable pitchers for {target}")
    if new_df.empty:
        print("No probables found.")
        return

    new_df["game_date"] = new_df["game_date"].astype(str).str[:10]

    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        existing["game_date"] = existing["game_date"].astype(str).str[:10]
        existing = existing[existing["game_date"] != target]
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()

    combined = combined.sort_values(["game_date", "team"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"Saved {len(combined)} total rows  ({len(new_df)} for {target})")


if __name__ == "__main__":
    main()
