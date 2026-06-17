import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.statcast_source import fetch_statcast_pitcher_daily, save_statcast_pitcher_daily


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
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    output = args.output or config["data"]["statcast_pitcher_daily_file"]
    if args.chunk_days <= 0:
        path = save_statcast_pitcher_daily(args.start, args.end, output)
    else:
        frames = []
        for chunk_start, chunk_end in _date_chunks(args.start, args.end, args.chunk_days):
            print(f"Fetching Statcast {chunk_start} to {chunk_end}")
            frame = fetch_statcast_pitcher_daily(chunk_start, chunk_end)
            if not frame.empty:
                frames.append(frame)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if not df.empty and path.exists():
            existing = pd.read_csv(path)
            key_cols = [c for c in ["game_date", "pitcher_id", "pitcher"] if c in df.columns and c in existing.columns]
            if key_cols:
                combined = pd.concat([existing, df], ignore_index=True, sort=False)
                df = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
            else:
                df = pd.concat([existing, df], ignore_index=True, sort=False).drop_duplicates()
        df.to_csv(path, index=False)
    print(f"Saved Statcast pitcher daily data to {path}")


if __name__ == "__main__":
    main()
