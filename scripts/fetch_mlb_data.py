import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.mlb_source import (
    fetch_batter_game_logs,
    fetch_game_context_logs,
    fetch_pitcher_game_logs,
    fetch_probable_pitchers,
    fetch_team_batting_and_context_logs,
    fetch_team_batting_game_logs,
)


def _merge_existing_csv(new_df, output: Path, key_cols: list[str]) -> pd.DataFrame:
    if output.exists():
        existing = pd.read_csv(output)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()
    keys = [col for col in key_cols if col in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values([col for col in ["game_date", "game_pk", "team", "pitcher_id", "batter_id"] if col in combined.columns])


def fetch_logs(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_directories(config)
    output = Path(args.output or config["data"]["pitcher_logs_file"])

    df = fetch_pitcher_game_logs(args.start, args.end, progress_every=args.progress_every)
    df = _merge_existing_csv(df, output, ["game_pk", "pitcher_id", "team", "opponent"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} pitcher game log rows to {output}")


def fetch_batting(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_directories(config)
    output = Path(args.output or config["data"]["team_batting_logs_file"])

    df = fetch_team_batting_game_logs(args.start, args.end, progress_every=args.progress_every)
    df = _merge_existing_csv(df, output, ["game_pk", "team"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} team batting game log rows to {output}")


def fetch_batters(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_directories(config)
    output = Path(args.output or config["data"]["batter_game_logs_file"])

    df = fetch_batter_game_logs(
        args.start,
        args.end,
        progress_every=args.progress_every,
        max_workers=args.max_workers,
    )
    df = _merge_existing_csv(df, output, ["game_pk", "batter_id", "team"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} batter game log rows to {output}")


def fetch_context(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_directories(config)
    output = Path(args.output or config["data"]["game_context_logs_file"])

    df = fetch_game_context_logs(args.start, args.end, progress_every=args.progress_every)
    df = _merge_existing_csv(df, output, ["game_pk", "pitcher_id", "team", "opponent"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} game context rows to {output}")


def fetch_extras(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_directories(config)
    batting_output = Path(args.batting_output or config["data"]["team_batting_logs_file"])
    context_output = Path(args.context_output or config["data"]["game_context_logs_file"])

    batting_df, context_df = fetch_team_batting_and_context_logs(
        args.start,
        args.end,
        progress_every=args.progress_every,
        max_workers=args.max_workers,
    )
    batting_df = _merge_existing_csv(batting_df, batting_output, ["game_pk", "team"])
    context_df = _merge_existing_csv(context_df, context_output, ["game_pk", "pitcher_id", "team", "opponent"])
    batting_output.parent.mkdir(parents=True, exist_ok=True)
    context_output.parent.mkdir(parents=True, exist_ok=True)
    batting_df.to_csv(batting_output, index=False)
    context_df.to_csv(context_output, index=False)
    print(f"Saved {len(batting_df)} team batting game log rows to {batting_output}")
    print(f"Saved {len(context_df)} game context rows to {context_output}")


def fetch_probables(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_directories(config)
    output = Path(args.output or config["data"]["probable_pitchers_file"])

    df = fetch_probable_pitchers(args.date)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} probable pitcher rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MLB pitcher data for the MVP model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    logs = subparsers.add_parser("logs", help="Fetch starter game logs from MLB boxscores.")
    logs.add_argument("--config", default="config/config.yaml")
    logs.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    logs.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    logs.add_argument("--output", default=None)
    logs.add_argument("--progress-every", type=int, default=50)
    logs.set_defaults(func=fetch_logs)

    batting = subparsers.add_parser(
        "batting",
        help="Fetch team batting game logs from MLB boxscores.",
    )
    batting.add_argument("--config", default="config/config.yaml")
    batting.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    batting.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    batting.add_argument("--output", default=None)
    batting.add_argument("--progress-every", type=int, default=50)
    batting.set_defaults(func=fetch_batting)

    batters = subparsers.add_parser(
        "batters",
        help="Fetch batter-level game logs from MLB live feeds.",
    )
    batters.add_argument("--config", default="config/config.yaml")
    batters.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    batters.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    batters.add_argument("--output", default=None)
    batters.add_argument("--progress-every", type=int, default=100)
    batters.add_argument("--max-workers", type=int, default=12)
    batters.set_defaults(func=fetch_batters)

    context = subparsers.add_parser(
        "context",
        help="Fetch venue, weather, umpire, handedness, and lineup context.",
    )
    context.add_argument("--config", default="config/config.yaml")
    context.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    context.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    context.add_argument("--output", default=None)
    context.add_argument("--progress-every", type=int, default=50)
    context.set_defaults(func=fetch_context)

    extras = subparsers.add_parser(
        "extras",
        help="Fetch team batting and game context logs together using parallel live-feed calls.",
    )
    extras.add_argument("--config", default="config/config.yaml")
    extras.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    extras.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    extras.add_argument("--batting-output", default=None)
    extras.add_argument("--context-output", default=None)
    extras.add_argument("--progress-every", type=int, default=100)
    extras.add_argument("--max-workers", type=int, default=12)
    extras.set_defaults(func=fetch_extras)

    probables = subparsers.add_parser("probables", help="Fetch probable pitchers for one date.")
    probables.add_argument("--config", default="config/config.yaml")
    probables.add_argument("--date", required=True, help="Game date, YYYY-MM-DD.")
    probables.add_argument("--output", default=None)
    probables.set_defaults(func=fetch_probables)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
