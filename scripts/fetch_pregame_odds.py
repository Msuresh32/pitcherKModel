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

SNAPSHOT_DIR = Path("data/odds/snapshots")
PICKS_LOG    = Path("data/exports/picks_log.csv")


def _american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds >= 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _strip_accents(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def _best_odds(snap: pd.DataFrame, pitcher_name: str, line: float, side: str):
    """Return best (most favorable) American odds for the side from a snapshot."""
    col = f"{side}_odds"
    if col not in snap.columns:
        return None
    name_col = "pitcher_name" if "pitcher_name" in snap.columns else "player_name"
    last = _strip_accents(pitcher_name.strip().split()[-1])
    rows = snap[snap[name_col].map(_strip_accents).str.contains(last, na=False)]
    rows = rows[pd.to_numeric(rows["line"], errors="coerce") == float(line)]
    vals = pd.to_numeric(rows[col], errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else None


def update_picks_clv(target: str, snapshot: pd.DataFrame) -> None:
    """Update picks_log.csv with closing_odds and clv_pct for today's picks."""
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
    print(f"Saved {len(result)} rows to {out_path}")

    # Immediately compute CLV% for today's picks
    update_picks_clv(target, result)


if __name__ == "__main__":
    main()
