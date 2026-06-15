import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.statcast_source import fetch_statcast_batter_pitch_type_daily


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


def _merge_existing(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    if output.exists():
        existing = pd.read_csv(output)
        df = pd.concat([existing, df], ignore_index=True, sort=False)
    if df.empty:
        return df
    return (
        df.drop_duplicates(["game_date", "batter_id"], keep="last")
        .sort_values(["batter_id", "game_date"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch batter pitch-type Statcast aggregates.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunk-days", type=int, default=31)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    output = Path(args.output or config["data"]["statcast_batter_pitch_type_daily_file"])
    frames = []
    for chunk_start, chunk_end in _date_chunks(args.start, args.end, args.chunk_days):
        print(f"Fetching Statcast batter pitch types {chunk_start} to {chunk_end}")
        frame = fetch_statcast_batter_pitch_type_daily(chunk_start, chunk_end)
        if not frame.empty:
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    df = _merge_existing(df, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} batter pitch-type Statcast rows to {output}")


if __name__ == "__main__":
    main()
