"""Fetch historical MLB pitcher prop odds with open and closing line snapshots.

Two snapshots are fetched per game:
  open  — `--open-hours-before` hours before first pitch (default 4h)
  close — `--closing-hours-before` hours before first pitch (default 0.05h ≈ 3 min)

Both are written to the same output CSV with a `snapshot_type` column ("open" / "close").
Pass the close rows to scripts/backtest.py --closing-odds for CLV computation.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.loaders import load_pitcher_game_logs
from src.odds.odds_api import (
    DEFAULT_MARKETS,
    append_csv,
    fetch_historical_event_odds,
    fetch_historical_events,
    game_snapshot_time,
    get_api_key,
    map_odds_to_pitcher_logs,
    normalize_event_odds,
)

HISTORICAL_PROPS_START = pd.Timestamp("2023-05-03T05:30:00Z")


def _date_range(start: str, end: str) -> list[pd.Timestamp]:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    days = []
    cursor = start_date
    while cursor <= end_date:
        days.append(pd.Timestamp(cursor))
        cursor += timedelta(days=1)
    return days


def _day_bounds(day: pd.Timestamp) -> tuple[str, str]:
    start = day.tz_localize("UTC")
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def _event_discovery_snapshot(day: pd.Timestamp) -> str:
    snapshot = day.tz_localize("UTC") + pd.Timedelta(hours=16)
    if snapshot < HISTORICAL_PROPS_START:
        snapshot = HISTORICAL_PROPS_START
    return snapshot.isoformat().replace("+00:00", "Z")


def _clamp_snapshot(snapshot_str: str) -> str:
    """Ensure snapshot is not before the historical props data start."""
    ts = pd.to_datetime(snapshot_str, utc=True)
    if ts < HISTORICAL_PROPS_START:
        return HISTORICAL_PROPS_START.isoformat().replace("+00:00", "Z")
    return snapshot_str


def _fetch_snapshot(
    api_key: str,
    event: dict,
    hours_before: float,
    snapshot_type: str,
    regions: str,
    markets: list[str],
    bookmakers: Optional[str],
) -> Optional[pd.DataFrame]:
    snapshot = game_snapshot_time(event["commence_time"], hours_before)
    snapshot = _clamp_snapshot(snapshot)
    try:
        event_odds, _, payload = fetch_historical_event_odds(
            api_key=api_key,
            event_id=event["id"],
            snapshot_date=snapshot,
            regions=regions,
            markets=markets,
            bookmakers=bookmakers,
        )
    except RuntimeError as exc:
        print(
            f"  Skipping {snapshot_type} snapshot for event {event.get('id')} "
            f"({event.get('away_team')} @ {event.get('home_team')}): {exc}"
        )
        return None

    actual_snapshot = payload.get("timestamp", snapshot)
    frame = normalize_event_odds(event_odds, fetched_at=actual_snapshot)
    if frame.empty:
        return None

    frame["snapshot_type"] = snapshot_type
    frame["requested_snapshot"] = snapshot
    frame["historical_snapshot"] = actual_snapshot
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical MLB pitcher prop odds (open + close snapshots)."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--start", required=True, help="YYYY-MM-DD. Player props available after 2023-05-03."
    )
    parser.add_argument("--end", required=True, help="YYYY-MM-DD.")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", default=None)
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    parser.add_argument(
        "--open-hours-before",
        type=float,
        default=4.0,
        help="Hours before game time for the 'open' snapshot (default 4h).",
    )
    parser.add_argument(
        "--closing-hours-before",
        type=float,
        default=0.05,
        help="Hours before game time for the 'close' snapshot (default 0.05h ≈ 3 min).",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Only fetch the open snapshot (halves API credit usage).",
    )
    parser.add_argument("--output", default="data/odds/historical_pitcher_props.csv")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dates already fully present in the output CSV.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    api_key = get_api_key()
    pitcher_logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    days = _date_range(args.start, args.end)
    if args.max_days:
        days = days[: args.max_days]

    output = Path(args.output)
    existing_days: set[str] = set()
    if args.resume and output.exists():
        existing = pd.read_csv(output, usecols=["game_date"])
        existing_days = set(existing["game_date"].dropna().astype(str).unique())

    total_rows = 0
    for day in days:
        day_key = str(day.date())
        if day_key in existing_days:
            print(f"Skipping {day.date()} (already in {output}).")
            continue

        if day.tz_localize("UTC") < HISTORICAL_PROPS_START.normalize():
            print(f"Skipping {day.date()} (player props begin after 2023-05-03).")
            continue

        commence_from, commence_to = _day_bounds(day)
        discovery_snapshot = _event_discovery_snapshot(day)
        events, _, _ = fetch_historical_events(
            api_key,
            snapshot_date=discovery_snapshot,
            commence_time_from=commence_from,
            commence_time_to=commence_to,
        )
        print(f"{day.date()}: found {len(events)} MLB events")
        if args.dry_run:
            continue

        day_rows: list[pd.DataFrame] = []
        for event in events:
            # --- Open snapshot ---
            open_frame = _fetch_snapshot(
                api_key=api_key,
                event=event,
                hours_before=args.open_hours_before,
                snapshot_type="open",
                regions=args.regions,
                markets=markets,
                bookmakers=args.bookmakers,
            )
            if open_frame is not None:
                day_rows.append(open_frame)

            # --- Close snapshot (skip if --open-only) ---
            if not args.open_only:
                close_frame = _fetch_snapshot(
                    api_key=api_key,
                    event=event,
                    hours_before=args.closing_hours_before,
                    snapshot_type="close",
                    regions=args.regions,
                    markets=markets,
                    bookmakers=args.bookmakers,
                )
                if close_frame is not None:
                    day_rows.append(close_frame)

        if not day_rows:
            print(f"{day.date()}: no rows returned")
            continue

        day_odds = pd.concat(day_rows, ignore_index=True, sort=False)
        day_odds = map_odds_to_pitcher_logs(day_odds, pitcher_logs)
        if day_odds.empty:
            print(f"{day.date()}: no matched pitcher prop odds")
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        day_odds.to_csv(output, mode="a", header=not output.exists(), index=False)
        total_rows += len(day_odds)

        snap_col = day_odds["snapshot_type"] if "snapshot_type" in day_odds.columns else pd.Series(dtype=str)
        open_count = int((snap_col == "open").sum())
        close_count = int((snap_col == "close").sum())
        print(
            f"{day.date()}: appended {len(day_odds)} rows "
            f"({open_count} open, {close_count} close) -> {output}"
        )

    if args.dry_run:
        print("Dry run complete. No odds were fetched.")
        return

    print(f"\nSaved {total_rows} historical pitcher prop rows to {output}")
    if not args.open_only:
        print(
            "Tip: to compute CLV, pass the 'close' rows to:\n"
            "  python scripts/backtest.py --closing-odds <path_to_close_only_csv>"
        )


if __name__ == "__main__":
    main()
