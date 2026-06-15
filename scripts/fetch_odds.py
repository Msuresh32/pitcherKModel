import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.loaders import load_probable_pitchers
from src.odds.odds_api import DEFAULT_MARKETS, best_current_lines, fetch_pitcher_prop_odds, append_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MLB pitcher prop odds from The Odds API.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", default=None)
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    parser.add_argument("--snapshots-output", default="data/odds/odds_snapshots.csv")
    parser.add_argument("--current-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    probable = load_probable_pitchers(config["data"]["probable_pitchers_file"], args.date)
    if probable.empty:
        raise ValueError(f"No probable pitchers found for {args.date}. Fetch probables first.")

    markets = [market.strip() for market in args.markets.split(",") if market.strip()]
    odds, headers = fetch_pitcher_prop_odds(
        probable_pitchers=probable,
        target_date=args.date,
        regions=args.regions,
        markets=markets,
        bookmakers=args.bookmakers,
    )
    if odds.empty:
        print("No pitcher prop odds returned.")
        return

    odds = odds[odds["game_date"] == args.date].copy()
    append_csv(odds, args.snapshots_output)

    current_output = Path(args.current_output or config["data"]["odds_file"])
    current = best_current_lines(odds)
    current_output.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(current_output, index=False)

    print(f"Saved {len(odds)} odds snapshot rows to {args.snapshots_output}")
    print(f"Saved {len(current)} current best-line rows to {current_output}")
    if "x-requests-remaining" in headers:
        print(f"Odds API requests remaining: {headers['x-requests-remaining']}")


if __name__ == "__main__":
    main()
