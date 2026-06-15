import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.loaders import load_probable_pitchers
from src.odds.scrapers import SCRAPER_REGISTRY, UNSUPPORTED_SPORTSBOOKS, ScraperUnavailable
from src.odds.scrapers.base import append_csv, best_scraped_lines, map_scraped_odds_to_probables


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape current daily MLB pitcher prop odds.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--sportsbooks", default="draftkings")
    parser.add_argument("--snapshots-output", default="data/odds/odds_snapshots.csv")
    parser.add_argument("--current-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    probable = load_probable_pitchers(config["data"]["probable_pitchers_file"], args.date)
    if probable.empty:
        raise ValueError(f"No probable pitchers found for {args.date}. Fetch probables first.")

    requested = [book.strip().lower() for book in args.sportsbooks.split(",") if book.strip()]
    frames = []
    for sportsbook in requested:
        if sportsbook in UNSUPPORTED_SPORTSBOOKS:
            print(
                f"{sportsbook}: adapter scaffold exists, but live parsing is not implemented yet. "
                "Skipping for now."
            )
            continue
        scraper_cls = SCRAPER_REGISTRY.get(sportsbook)
        if not scraper_cls:
            print(f"{sportsbook}: no scraper registered. Skipping.")
            continue

        scraper = scraper_cls()
        try:
            odds = scraper.fetch_pitcher_props(args.date)
        except ScraperUnavailable as exc:
            print(f"{sportsbook}: scraper unavailable: {exc}")
            continue

        if odds.empty:
            print(f"{sportsbook}: no pitcher prop odds found for {args.date}")
            continue
        frames.append(odds)
        print(f"{sportsbook}: scraped {len(odds)} pitcher prop rows")

    if not frames:
        print("No odds rows scraped.")
        return

    odds = pd.concat(frames, ignore_index=True, sort=False)
    odds = map_scraped_odds_to_probables(odds, probable)
    append_csv(odds, args.snapshots_output)

    current_output = Path(args.current_output or config["data"]["odds_file"])
    current = best_scraped_lines(odds)
    current_output.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(current_output, index=False)

    mapped = int(current["pitcher_id"].notna().sum()) if "pitcher_id" in current else 0
    print(f"Saved {len(odds)} scraped odds snapshot rows to {args.snapshots_output}")
    print(f"Saved {len(current)} current best-line rows to {current_output}")
    print(f"Mapped best-line rows to probable pitchers: {mapped}/{len(current)}")


if __name__ == "__main__":
    main()
