"""Fetch per-(pitcher, game_date) CONTACT QUALITY ALLOWED aggregates from
Statcast pitch-level data (pybaseball, cached).

Columns: bbe, avg_ev_allowed, max_ev_allowed, hardhit_rate, barrel_rate,
gb_rate, ld_rate, fb_rate, sweet_spot_rate, xba_con_allowed, xwoba_con_allowed.

Output: data/raw/statcast_pitcher_contact_daily.csv (appended, deduped).
"""
from __future__ import annotations
import argparse, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path("data/raw/statcast_pitcher_contact_daily.csv")


def aggregate_contact(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    # batted ball events = rows with a bb_type
    bbe = df[df["bb_type"].notna()].copy()
    if bbe.empty:
        return pd.DataFrame()
    bbe["hardhit"] = (pd.to_numeric(bbe["launch_speed"], errors="coerce") >= 95).astype(float)
    bbe["barrel"] = (pd.to_numeric(bbe["launch_speed_angle"], errors="coerce") == 6).astype(float)
    la = pd.to_numeric(bbe["launch_angle"], errors="coerce")
    bbe["sweet"] = ((la >= 8) & (la <= 32)).astype(float)
    bbe["gb"] = (bbe["bb_type"] == "ground_ball").astype(float)
    bbe["ld"] = (bbe["bb_type"] == "line_drive").astype(float)
    bbe["fb"] = (bbe["bb_type"] == "fly_ball").astype(float)
    bbe["ev"] = pd.to_numeric(bbe["launch_speed"], errors="coerce")
    bbe["xba"] = pd.to_numeric(bbe["estimated_ba_using_speedangle"], errors="coerce")
    bbe["xwoba"] = pd.to_numeric(bbe["estimated_woba_using_speedangle"], errors="coerce")
    g = bbe.groupby(["game_date", "pitcher"]).agg(
        bbe=("ev", "size"),
        avg_ev_allowed=("ev", "mean"),
        max_ev_allowed=("ev", "max"),
        hardhit_rate=("hardhit", "mean"),
        barrel_rate=("barrel", "mean"),
        gb_rate=("gb", "mean"),
        ld_rate=("ld", "mean"),
        fb_rate=("fb", "mean"),
        sweet_spot_rate=("sweet", "mean"),
        xba_con_allowed=("xba", "mean"),
        xwoba_con_allowed=("xwoba", "mean"),
    ).reset_index().rename(columns={"pitcher": "pitcher_id"})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--chunk-days", type=int, default=14)
    args = ap.parse_args()

    from pybaseball import cache, statcast
    cache.enable()

    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()
    cursor = start
    frames = []
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=args.chunk_days - 1), end)
        print(f"chunk {cursor} .. {chunk_end}", flush=True)
        try:
            raw = statcast(start_dt=cursor.isoformat(), end_dt=chunk_end.isoformat())
            agg = aggregate_contact(raw)
            if len(agg):
                frames.append(agg)
                print(f"  {len(agg)} pitcher-days", flush=True)
        except Exception as exc:
            print(f"  WARNING skipped: {exc}", flush=True)
        cursor = chunk_end + timedelta(days=1)

    if not frames:
        print("no data")
        return
    df = pd.concat(frames, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        old = pd.read_csv(OUTPUT)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["game_date", "pitcher_id"], keep="last")
    df.to_csv(OUTPUT, index=False)
    print(f"saved {len(df):,} rows -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
