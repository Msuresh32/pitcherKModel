"""
Consolidate all closing-odds snapshot files + june_2026_odds.csv into a single
file that backtest.py's CLV engine can match against (needs pitcher_id).

Output: data/odds/closing_odds_master.csv
Columns: game_date, pitcher_id, player_name, market, line, over_odds, under_odds,
          snapshot_type (always "close")

Usage:
    python scripts/build_closing_odds_master.py
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SNAPSHOT_DIR = Path("data/odds/snapshots")
LOGS_FILE    = Path("data/raw/pitcher_game_logs.csv")
JUNE_ODDS    = Path("data/odds/june_2026_odds.csv")
OUT_FILE     = Path("data/odds/closing_odds_master.csv")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii").lower().strip()


def build_name_to_id(logs: pd.DataFrame) -> dict:
    """Build {normalized_last_name: pitcher_id} lookup from game logs."""
    mapping = {}
    for _, row in logs[["pitcher_name", "pitcher_id"]].drop_duplicates().iterrows():
        name = str(row["pitcher_name"])
        pid  = str(row["pitcher_id"])
        if pd.notna(row["pitcher_name"]) and pd.notna(row["pitcher_id"]):
            mapping[_norm(name)] = pid
            # Also index by last name only for fuzzy matching
            last = _norm(name).split()[-1] if _norm(name) else ""
            if last not in mapping:
                mapping[last] = pid
    return mapping


def attach_pitcher_id(df: pd.DataFrame, name_col: str, name_to_id: dict) -> pd.DataFrame:
    df = df.copy()
    ids = []
    for name in df[name_col]:
        n = _norm(str(name))
        pid = name_to_id.get(n)
        if pid is None:
            last = n.split()[-1] if n else ""
            pid = name_to_id.get(last, "")
        ids.append(pid or "")
    df["pitcher_id"] = ids
    return df


def load_snapshot_closing(path: Path, date_str: str) -> pd.DataFrame:
    """Load a *_closing.csv snapshot and normalize to master format."""
    df = pd.read_csv(path)
    name_col = "player_name" if "player_name" in df.columns else "pitcher_name"
    if name_col not in df.columns:
        return pd.DataFrame()

    needed = [name_col, "line", "over_odds", "under_odds"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[needed].copy()
    df = df.rename(columns={name_col: "player_name"})
    df["game_date"]     = date_str
    df["market"]        = "strikeouts"
    df["snapshot_type"] = "close"
    df["over_odds"]     = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"]    = pd.to_numeric(df["under_odds"], errors="coerce")
    df["line"]          = pd.to_numeric(df["line"], errors="coerce")
    return df.dropna(subset=["line"])


def main():
    print("Loading pitcher game logs for name->id mapping...")
    logs = pd.read_csv(LOGS_FILE)
    name_to_id = build_name_to_id(logs)
    print(f"  {len(name_to_id)} name entries in lookup")

    frames = []

    # ── Part 1: june_2026_odds.csv (June 1, has pitcher_id already) ──────────
    if JUNE_ODDS.exists():
        raw = pd.read_csv(JUNE_ODDS)
        raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date.astype(str)
        close = raw[raw["snapshot_type"] == "close"].copy() if "snapshot_type" in raw.columns else raw.copy()
        if "player_name" not in close.columns and "pitcher_name" in close.columns:
            close = close.rename(columns={"pitcher_name": "player_name"})
        needed = ["game_date", "pitcher_id", "player_name", "market", "line", "over_odds", "under_odds"]
        close = close[[c for c in needed if c in close.columns]]
        close["snapshot_type"] = "close"
        close["pitcher_id"] = close["pitcher_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        frames.append(close)
        print(f"  June 1 from june_2026_odds.csv: {len(close)} rows")

    # ── Part 2: per-date _closing.csv snapshots ───────────────────────────────
    for snap in sorted(SNAPSHOT_DIR.glob("*_closing.csv")):
        date_str = snap.name.replace("_closing.csv", "")
        df = load_snapshot_closing(snap, date_str)
        if df.empty:
            continue
        df = attach_pitcher_id(df, "player_name", name_to_id)
        kept = df[df["pitcher_id"] != ""]
        print(f"  {date_str} snapshot: {len(df)} rows, {len(kept)} with pitcher_id matched")
        frames.append(df)

    if not frames:
        print("No data found.")
        return

    master = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep one row per (game_date, pitcher_id, market, line, bookmaker)
    # Since snapshots don't have bookmaker, use player_name as tie-breaker
    master = master.drop_duplicates(subset=["game_date", "player_name", "market", "line"])
    master = master.sort_values(["game_date", "player_name", "line"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(master)} closing-odds rows to {OUT_FILE}")
    print(f"Date coverage: {master['game_date'].min()} to {master['game_date'].max()}")
    matched = (master["pitcher_id"] != "").sum()
    print(f"Pitcher ID matched: {matched}/{len(master)} ({100*matched/len(master):.1f}%)")


if __name__ == "__main__":
    main()
