"""
Capture closing odds for games starting within the next window.
Run this every 30 minutes — it fetches odds only for games whose
start time is 20–90 minutes from now, and saves to the closing snapshot.

Usage:
    python scripts/fetch_pregame_odds.py              # default: 20-90min window
    python scripts/fetch_pregame_odds.py --window 60  # 0-60min window
    python scripts/fetch_pregame_odds.py --date 2026-06-20  # specific date
"""
import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import (
    get_api_key,
    fetch_events,
    fetch_event_odds,
    normalize_event_odds,
)

SNAPSHOT_DIR  = Path("data/odds/snapshots")
PROBS_FILE    = Path("data/raw/probable_pitchers.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--window", type=int, default=90,
                        help="Capture odds for games starting within this many minutes (default: 90)")
    parser.add_argument("--min-before", type=int, default=20,
                        help="Minimum minutes before game time to capture (default: 20)")
    args = parser.parse_args()

    target  = args.date
    api_key = get_api_key()
    now     = datetime.now(timezone.utc)

    print(f"Fetching pre-game closing odds for {target} at {now.strftime('%H:%M UTC')}")
    print(f"Window: games starting in {args.min_before}–{args.window} minutes")

    # Fetch live events
    time_from = f"{target}T00:00:00Z"
    time_to   = f"{pd.to_datetime(target).date() + pd.Timedelta(days=1)}T09:00:00Z"
    events, _ = fetch_events(api_key, commence_time_from=time_from, commence_time_to=time_to)

    # Filter to games starting in the window
    target_events = []
    for event in events:
        ct = pd.to_datetime(event.get("commence_time"), utc=True)
        mins_to_start = (ct - now).total_seconds() / 60
        if args.min_before <= mins_to_start <= args.window:
            target_events.append((event, mins_to_start, ct))

    if not target_events:
        print(f"No games starting in {args.min_before}–{args.window} min window. Exiting.")
        return

    print(f"Found {len(target_events)} game(s) in window:")
    for ev, mins, ct in target_events:
        print(f"  {ev['away_team']} @ {ev['home_team']} in {mins:.0f}min ({ct.strftime('%H:%M UTC')})")

    fetched_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    all_frames = []

    for event, mins, ct in target_events:
        try:
            event_odds, _ = fetch_event_odds(
                api_key,
                event_id=event["id"],
                regions="us",
                markets=["pitcher_strikeouts"],
            )
            frame = normalize_event_odds(event_odds, fetched_at)
            if not frame.empty:
                frame["game_date"] = target
                all_frames.append(frame)
                pitchers = frame["player_name"].unique()
                print(f"  Got {len(frame)} rows for {event['away_team']} @ {event['home_team']}: {list(pitchers)[:3]}")
        except Exception as exc:
            print(f"  Error {event['away_team']} @ {event['home_team']}: {exc}")
        time.sleep(0.3)

    if not all_frames:
        print("No odds retrieved.")
        return

    result = pd.concat(all_frames, ignore_index=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Append to closing snapshot (multiple runs throughout the day)
    out_path = SNAPSHOT_DIR / f"{target}_closing.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Deduplicate by player_name + line + bookmaker + commence_time
        result = pd.concat([existing, result], ignore_index=True)
        key_cols = [c for c in ["player_name", "line", "bookmaker", "commence_time"] if c in result.columns]
        result = result.drop_duplicates(key_cols, keep="last")

    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} rows → {out_path}")


if __name__ == "__main__":
    main()
