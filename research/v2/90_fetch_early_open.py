"""Fetch an EARLY open snapshot (T-12h, fallback T-6h) for 2025+2026 games.

For each event: try the T-12h snapshot; if the pitcher_strikeouts market is
absent/empty there, retry at T-6h. Rows are tagged snapshot_type='open' plus
open_hours_used (12.0 or 6.0). Close snapshots are NOT refetched — CLV reuses
the existing T-3min closes in pitcher_strikeouts_2025_2026.csv.

Output: data/odds/historical/pitcher_strikeouts_early_open_2025_2026.csv
Resumable by day (--resume). ~10 credits per event request.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.data.loaders import load_pitcher_game_logs
from src.odds.odds_api import (
    fetch_historical_event_odds,
    fetch_historical_events,
    game_snapshot_time,
    get_api_key,
    map_odds_to_pitcher_logs,
    normalize_event_odds,
)

HISTORICAL_PROPS_START = pd.Timestamp("2023-05-03T05:30:00Z")
OUTPUT = Path("data/odds/historical/pitcher_strikeouts_early_open_2025_2026.csv")
MARKETS = ["pitcher_strikeouts"]

RANGES = [("2025-03-27", "2025-11-01"), ("2026-03-26", "2026-07-10")]


def _clamp(s: str) -> str:
    ts = pd.to_datetime(s, utc=True)
    if ts < HISTORICAL_PROPS_START:
        return HISTORICAL_PROPS_START.isoformat().replace("+00:00", "Z")
    return s


def fetch_at(api_key, event, hours):
    snapshot = _clamp(game_snapshot_time(event["commence_time"], hours))
    try:
        event_odds, _, payload = fetch_historical_event_odds(
            api_key=api_key, event_id=event["id"], snapshot_date=snapshot,
            regions="us", markets=MARKETS, bookmakers=None)
    except RuntimeError as exc:
        print(f"  skip {event.get('id')} @T-{hours}h: {exc}", flush=True)
        return None
    actual = payload.get("timestamp", snapshot)
    frame = normalize_event_odds(event_odds, fetched_at=actual)
    if frame.empty:
        return None
    frame["snapshot_type"] = "open"
    frame["requested_snapshot"] = snapshot
    frame["historical_snapshot"] = actual
    frame["open_hours_used"] = float(hours)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()

    config = load_config("config/config_v4_production.yaml")
    api_key = get_api_key()
    logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])

    days = []
    for start, end in RANGES:
        days += list(pd.date_range(start, end, freq="D"))
    if args.max_days:
        days = days[: args.max_days]

    done = set()
    if args.resume and OUTPUT.exists():
        done = set(pd.read_csv(OUTPUT, usecols=["game_date"])["game_date"]
                   .dropna().astype(str).unique())

    total = 0
    for day in days:
        key = str(day.date())
        if key in done:
            continue
        start = day.tz_localize("UTC")
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        disc = _clamp((start + pd.Timedelta(hours=16)).isoformat().replace("+00:00", "Z"))
        try:
            events, _, _ = fetch_historical_events(
                api_key, snapshot_date=disc,
                commence_time_from=start.isoformat().replace("+00:00", "Z"),
                commence_time_to=end.isoformat().replace("+00:00", "Z"))
        except RuntimeError as exc:
            print(f"{key}: discovery failed: {exc}", flush=True)
            continue
        rows, n12, n6 = [], 0, 0
        for ev in events:
            fr = fetch_at(api_key, ev, 12.0)
            if fr is not None:
                n12 += 1
            else:
                fr = fetch_at(api_key, ev, 6.0)
                if fr is not None:
                    n6 += 1
            if fr is not None:
                rows.append(fr)
        if not rows:
            print(f"{key}: {len(events)} events, no odds", flush=True)
            continue
        day_odds = pd.concat(rows, ignore_index=True, sort=False)
        day_odds = map_odds_to_pitcher_logs(day_odds, logs)
        if day_odds.empty:
            print(f"{key}: no matched rows", flush=True)
            continue
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        day_odds.to_csv(OUTPUT, mode="a", header=not OUTPUT.exists(), index=False)
        total += len(day_odds)
        print(f"{key}: {len(events)} events -> {len(day_odds)} rows "
              f"(T-12: {n12}, T-6 fallback: {n6})", flush=True)
    print(f"done, {total} rows appended", flush=True)


if __name__ == "__main__":
    main()
