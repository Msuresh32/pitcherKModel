"""Fetch FanGraphs pitcher season stats via pybaseball and save to CSV.

Usage:
    python scripts/fetch_fangraphs.py --start-year 2022 --end-year 2025
    python scripts/fetch_fangraphs.py  # uses config defaults
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.fangraphs_source import fetch_fangraphs_pitcher_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FanGraphs pitcher season stats.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First season to fetch (defaults to training.train_start year from config).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last season to fetch (defaults to current year).",
    )
    parser.add_argument(
        "--qual",
        type=int,
        default=0,
        help="Minimum IP qualifier. 0 = all pitchers (default).",
    )
    parser.add_argument("--output", default=None, help="Output CSV path (overrides config).")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    start_year = args.start_year or int(config["training"]["train_start"][:4])
    end_year = args.end_year or date.today().year

    output_path = Path(
        args.output or config["data"].get("fangraphs_file", "data/raw/fangraphs_pitcher_stats.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching FanGraphs pitcher stats for {start_year}–{end_year} (qual={args.qual})...")
    df = fetch_fangraphs_pitcher_stats(start_year, end_year, qual=args.qual)

    if df.empty:
        print("No data returned. Check pybaseball installation and network access.")
        return

    unmapped = df["pitcher_id"].isna().sum()
    if unmapped > 0:
        print(
            f"Warning: {unmapped}/{len(df)} rows could not be mapped to MLBAM pitcher_id. "
            "These rows will not match game logs at inference time."
        )

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Seasons: {sorted(df['season'].unique().tolist())}")
    print(f"Pitchers with MLBAM ID: {df['pitcher_id'].notna().sum()}")


if __name__ == "__main__":
    main()
