import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.statcast_source import (
    fetch_statcast_pitcher_daily,
    fetch_statcast_catcher_framing_daily,
    save_statcast_pitcher_daily,
    save_statcast_catcher_framing_daily,
)


def _date_chunks(start_date: str, end_date: str, days: int) -> list[tuple[str, str]]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pitcher daily Statcast aggregates.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument(
        "--framing",
        action="store_true",
        help="Also fetch and save per-catcher take-pitch called-strike rate (framing proxy).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    output = args.output or config["data"]["statcast_pitcher_daily_file"]

    if args.chunk_days <= 0:
        path = save_statcast_pitcher_daily(args.start, args.end, output)
        if args.framing:
            framing_output = config["data"].get("catcher_framing_file", "data/raw/statcast_catcher_framing_daily.csv")
            save_statcast_catcher_framing_daily(args.start, args.end, framing_output)
            print(f"Saved catcher framing data to {framing_output}")
    else:
        pitcher_frames = []
        framing_frames = []
        for chunk_start, chunk_end in _date_chunks(args.start, args.end, args.chunk_days):
            print(f"Fetching Statcast {chunk_start} to {chunk_end}")
            try:
                pitcher_frame = fetch_statcast_pitcher_daily(chunk_start, chunk_end)
                if not pitcher_frame.empty:
                    pitcher_frames.append(pitcher_frame)
            except Exception as exc:
                print(f"  WARNING: skipped {chunk_start}–{chunk_end} pitcher data: {exc}")
            if args.framing:
                try:
                    framing_frame = fetch_statcast_catcher_framing_daily(chunk_start, chunk_end)
                    if not framing_frame.empty:
                        framing_frames.append(framing_frame)
                except Exception as exc:
                    print(f"  WARNING: skipped {chunk_start}–{chunk_end} framing data: {exc}")

        # ── Pitcher daily ──
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat(pitcher_frames, ignore_index=True, sort=False) if pitcher_frames else pd.DataFrame()
        if not df.empty and path.exists():
            existing = pd.read_csv(path)
            key_cols = [c for c in ["game_date", "pitcher_id"] if c in df.columns and c in existing.columns]
            if key_cols:
                combined = pd.concat([existing, df], ignore_index=True, sort=False)
                df = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
            else:
                df = pd.concat([existing, df], ignore_index=True, sort=False).drop_duplicates()
        df.to_csv(path, index=False)

        # ── Catcher framing ──
        if args.framing and framing_frames:
            framing_output = config["data"].get("catcher_framing_file", "data/raw/statcast_catcher_framing_daily.csv")
            framing_path = Path(framing_output)
            framing_path.parent.mkdir(parents=True, exist_ok=True)
            cf = pd.concat(framing_frames, ignore_index=True, sort=False)
            if framing_path.exists():
                existing_cf = pd.read_csv(framing_path)
                combined_cf = pd.concat([existing_cf, cf], ignore_index=True, sort=False)
                cf = combined_cf.drop_duplicates(subset=["game_date", "catcher_id"], keep="last").reset_index(drop=True)
            cf.to_csv(framing_path, index=False)
            print(f"Saved catcher framing data to {framing_path}")

    print(f"Saved Statcast pitcher daily data to {path}")


if __name__ == "__main__":
    main()
