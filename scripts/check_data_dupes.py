"""Check raw data files for duplicate rows and fix them."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_and_fix(path: str, dedup_key: list[str], label: str) -> None:
    df = pd.read_csv(path)
    total = len(df)
    unique = df.drop_duplicates(dedup_key).shape[0]
    print(f"\n{label}")
    print(f"  Total rows:    {total:,}")
    print(f"  Unique rows:   {unique:,}")
    print(f"  Duplicate rows:{total - unique:,}  ({(total-unique)/total*100:.1f}%)")

    if total > unique:
        dupes = df[df.duplicated(dedup_key, keep=False)]
        sample = dupes.sort_values(dedup_key).head(6)
        print(f"  Sample duplicates:")
        print(sample[dedup_key[:3]].to_string(index=False))

        # Fix: keep first occurrence
        fixed = df.drop_duplicates(dedup_key, keep="first")
        fixed.to_csv(path, index=False)
        print(f"  FIXED: deduplicated to {len(fixed):,} rows -> saved to {path}")
    else:
        print(f"  No duplicates found.")


def main():
    check_and_fix(
        "data/raw/pitcher_game_logs.csv",
        ["game_date", "pitcher_id", "game_pk"],
        "Pitcher game logs",
    )
    check_and_fix(
        "data/raw/team_batting_game_logs.csv",
        ["game_date", "team", "game_pk"],
        "Team batting game logs",
    )
    check_and_fix(
        "data/raw/batter_game_logs.csv",
        ["game_date", "batter_id", "game_pk"],
        "Batter game logs",
    )
    check_and_fix(
        "data/raw/statcast_pitcher_daily.csv",
        ["game_date", "pitcher_id"],
        "Statcast pitcher daily",
    )
    check_and_fix(
        "data/raw/game_context_logs.csv",
        ["game_date", "pitcher_id", "game_pk"],
        "Game context logs",
    )


if __name__ == "__main__":
    main()
