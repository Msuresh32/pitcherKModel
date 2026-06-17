"""
Backfill morning + closing odds snapshots for June 9-16 picks.

For each date with picks:
  1. Fetch all events at 11:00 UTC (7am ET) — morning snapshot
  2. For each event, fetch morning odds and check if any of our picks' pitchers appear
  3. For matching events, fetch closing odds at commence_time - 15 min
  4. Save both to data/odds/snapshots/{date}_morning.csv and {date}_closing.csv

API calls: ~13 events/date × 8 dates × 2 snapshots = ~210 calls
"""
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import (
    fetch_historical_event_odds,
    fetch_historical_events,
    get_api_key,
    normalize_event_odds,
)

PICKS_LOG    = Path("data/exports/picks_log.csv")
SNAPSHOT_DIR = Path("data/odds/snapshots")
BACKFILL_DATES = [
    "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
    "2026-06-13", "2026-06-14", "2026-06-15", "2026-06-16",
]


def normalize(name: str) -> str:
    ascii_name = unicodedata.normalize("NFD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", ascii_name.lower())).strip()


def morning_snapshot_time(date_str: str) -> str:
    return f"{date_str}T11:00:00Z"


def closing_snapshot_time(commence_time: str, minutes_before: int = 15) -> str:
    ts = pd.to_datetime(commence_time, utc=True)
    return (ts - pd.Timedelta(minutes=minutes_before)).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_date(api_key: str, date_str: str, target_names: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch morning + closing snapshots for one date. Returns (morning_df, closing_df)."""
    morning_time = morning_snapshot_time(date_str)
    time_from = f"{date_str}T00:00:00Z"
    time_to   = f"{date_str}T23:59:59Z"

    print(f"  Fetching events for {date_str}...")
    events, _, _ = fetch_historical_events(
        api_key, snapshot_date=morning_time,
        commence_time_from=time_from, commence_time_to=time_to
    )
    print(f"  Found {len(events)} events")
    time.sleep(0.35)

    morning_frames = []
    closing_frames = []
    matched_events = 0

    for event in events:
        event_id      = event["id"]
        commence_time = event.get("commence_time", "")

        # Fetch morning odds for this event
        try:
            morning_data, _, _ = fetch_historical_event_odds(
                api_key=api_key,
                event_id=event_id,
                snapshot_date=morning_time,
                markets=["pitcher_strikeouts"],
            )
        except Exception as e:
            print(f"    WARN morning odds {event_id[:8]}: {e}")
            time.sleep(0.35)
            continue
        time.sleep(0.35)

        if not morning_data:
            continue

        morning_df = normalize_event_odds(morning_data, fetched_at=morning_time)
        if morning_df.empty:
            continue

        # Check if any of our target pitchers appear in this event's odds
        event_names = {normalize(n) for n in morning_df["player_name"].dropna()}
        hits = target_names & event_names
        if not hits:
            continue

        matched_events += 1
        morning_frames.append(morning_df)

        # Fetch closing odds for this matched event
        if commence_time:
            close_time = closing_snapshot_time(commence_time)
            try:
                close_data, _, _ = fetch_historical_event_odds(
                    api_key=api_key,
                    event_id=event_id,
                    snapshot_date=close_time,
                    markets=["pitcher_strikeouts"],
                )
                time.sleep(0.35)
                if close_data:
                    close_df = normalize_event_odds(close_data, fetched_at=close_time)
                    if not close_df.empty:
                        closing_frames.append(close_df)
            except Exception as e:
                print(f"    WARN closing odds {event_id[:8]}: {e}")
                time.sleep(0.35)

    print(f"  Matched {matched_events} events with picks")
    morning_out = pd.concat(morning_frames, ignore_index=True) if morning_frames else pd.DataFrame()
    closing_out = pd.concat(closing_frames, ignore_index=True) if closing_frames else pd.DataFrame()
    return morning_out, closing_out


def main() -> None:
    picks = pd.read_csv(PICKS_LOG, dtype=str)
    api_key = get_api_key()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for date_str in BACKFILL_DATES:
        morning_path = SNAPSHOT_DIR / f"{date_str}_morning.csv"
        closing_path = SNAPSHOT_DIR / f"{date_str}_closing.csv"

        if morning_path.exists() and closing_path.exists():
            print(f"{date_str}: snapshots already exist, skipping")
            continue

        day_picks = picks[picks["game_date"] == date_str]
        if day_picks.empty:
            print(f"{date_str}: no picks, skipping")
            continue

        target_names = {normalize(n) for n in day_picks["pitcher_name"].dropna()}
        print(f"\n{date_str}: {len(day_picks)} picks — {target_names}")

        morning_df, closing_df = fetch_date(api_key, date_str, target_names)

        if not morning_df.empty:
            morning_df.to_csv(morning_path, index=False)
            print(f"  Saved morning: {len(morning_df)} rows -> {morning_path.name}")
        else:
            print(f"  No morning odds found")

        if not closing_df.empty:
            closing_df.to_csv(closing_path, index=False)
            print(f"  Saved closing: {len(closing_df)} rows -> {closing_path.name}")
        else:
            print(f"  No closing odds found")

    # Run CLV computation
    print("\n\nRunning CLV computation...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/compute_live_clv.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        for line in result.stderr.splitlines():
            if "Warning" not in line and "UserWarning" not in line and "RuntimeWarning" not in line:
                print(line)


if __name__ == "__main__":
    main()
