"""
Backfill closing odds for a past date using the Odds API historical endpoint.
Fetches odds at 30 minutes before each game's start time.

Usage:
    python scripts/fetch_closing_odds_historical.py --date 2026-06-19
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.odds.odds_api import (
    get_api_key,
    fetch_historical_events,
    fetch_historical_event_odds,
    normalize_event_odds,
    game_snapshot_time,
)

SNAPSHOT_DIR = Path("data/odds/snapshots")
PICKS_LOG    = Path("data/exports/picks_log.csv")


def _american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds >= 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _best_odds(snap: pd.DataFrame, pitcher_name: str, line: float, side: str):
    col = f"{side}_odds"
    if col not in snap.columns:
        return None
    name_col = "pitcher_name" if "pitcher_name" in snap.columns else "player_name"
    last = pitcher_name.strip().split()[-1].lower()
    rows = snap[snap[name_col].str.lower().str.contains(last, na=False)]
    rows = rows[pd.to_numeric(rows["line"], errors="coerce") == float(line)]
    vals = pd.to_numeric(rows[col], errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else None


def update_picks_clv(target: str, snapshot: pd.DataFrame) -> None:
    if not PICKS_LOG.exists():
        return
    picks = pd.read_csv(PICKS_LOG, dtype=str)
    picks["game_date"] = picks["game_date"].astype(str).str[:10]
    for col in ("closing_odds", "clv_pct"):
        if col not in picks.columns:
            picks[col] = ""
    mask = picks["game_date"] == target
    updated = 0
    for idx in picks[mask].index:
        row = picks.loc[idx]
        close = _best_odds(snapshot, str(row["pitcher_name"]),
                           float(row["line"]), str(row["best_side"]))
        if close is None:
            continue
        picks.loc[idx, "closing_odds"] = str(close)
        open_o = row.get("opening_odds", "")
        if str(open_o) not in ("", "nan"):
            p_open  = _american_to_implied(float(open_o))
            p_close = _american_to_implied(close)
            picks.loc[idx, "clv_pct"] = str(round((p_close - p_open) * 100, 2))
        updated += 1
    picks.to_csv(PICKS_LOG, index=False)
    print(f"Updated {updated} picks with closing odds + CLV%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=(date.today() - timedelta(days=1)).isoformat())
    parser.add_argument("--minutes-before", type=int, default=30,
                        help="Fetch odds this many minutes before game time (default: 30)")
    args = parser.parse_args()

    target      = args.date
    mins_before = args.minutes_before
    api_key     = get_api_key()

    # Load picks for this date to know which pitchers we care about
    pitcher_names = set()
    if PICKS_LOG.exists():
        log = pd.read_csv(PICKS_LOG, dtype=str)
        log["game_date"] = log["game_date"].astype(str).str[:10]
        day_picks = log[log["game_date"] == target]
        if not day_picks.empty:
            pitcher_names = set(day_picks["pitcher_name"].str.lower().str.strip())

    print(f"Fetching historical closing odds for {target} ({mins_before}min before game)")
    print(f"Target pitchers: {len(pitcher_names)} from picks_log")

    # Fetch historical events for the date
    time_from = f"{target}T00:00:00Z"
    time_to   = f"{pd.to_datetime(target).date() + pd.Timedelta(days=1)}T09:00:00Z"

    events, _, _ = fetch_historical_events(
        api_key,
        snapshot_date=f"{target}T12:00:00Z",
        commence_time_from=time_from,
        commence_time_to=time_to,
    )
    print(f"Found {len(events)} events on {target}")

    all_frames = []
    for event in events:
        commence = event.get("commence_time", "")
        if not commence:
            continue

        snapshot_ts = game_snapshot_time(commence, mins_before / 60)
        event_id    = event["id"]
        home        = event.get("home_team", "")
        away        = event.get("away_team", "")

        try:
            event_odds, _, _ = fetch_historical_event_odds(
                api_key,
                event_id=event_id,
                snapshot_date=snapshot_ts,
                regions="us",
                markets=["pitcher_strikeouts"],
            )
            frame = normalize_event_odds(event_odds, snapshot_ts)
            if not frame.empty:
                frame["game_date"] = target
                all_frames.append(frame)
                # Report which pitchers we got
                pitchers_in_frame = frame["player_name"].unique()
                matched = [p for p in pitchers_in_frame
                           if any(part in p.lower() for part in
                                  [n.split()[-1] for n in pitcher_names])]
                if matched:
                    print(f"  {away} @ {home}: got odds for {matched}")
        except Exception as exc:
            print(f"  Skipping {away} @ {home} ({event_id}): {exc}")

        time.sleep(0.3)  # be polite to the API

    if not all_frames:
        print("No odds retrieved.")
        return

    result = pd.concat(all_frames, ignore_index=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{target}_closing.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows to {out_path}")

    # Immediately compute CLV% for the date's picks
    update_picks_clv(target, result)

    # Report coverage for our picks
    if pitcher_names:
        got = set(result["player_name"].str.lower().str.strip())
        found = [p for p in pitcher_names if any(p.split()[-1] in g for g in got)]
        missing = [p for p in pitcher_names if p not in found]
        print(f"Pick coverage: {len(found)}/{len(pitcher_names)} found")
        if missing:
            print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
