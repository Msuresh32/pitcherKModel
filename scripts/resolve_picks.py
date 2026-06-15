"""
Resolve previous day's picks: fetch game logs, update picks_log.csv with
actual K totals and W/L, then rebuild the 2026 backtest.

Usage:
    python scripts/resolve_picks.py              # resolves yesterday
    python scripts/resolve_picks.py --date 2026-06-12  # resolves specific date
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mlb_source import fetch_pitcher_game_logs

LOGS_FILE   = Path("data/raw/pitcher_game_logs.csv")
PICKS_LOG   = Path("data/exports/picks_log.csv")
BT_2026     = Path("data/exports/2026_backtest_extended.csv")


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return (1 + odds / 100) if odds >= 0 else (1 + 100 / abs(odds))


def _merge_logs(new_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["game_pk", "pitcher_id", "team", "opponent"]
    if LOGS_FILE.exists():
        existing = pd.read_csv(LOGS_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()
    keys = [c for c in key_cols if c in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values(
        [c for c in ["game_date", "game_pk", "pitcher_id"] if c in combined.columns]
    )


def lookup_k(logs: pd.DataFrame, game_date: str, pitcher_name: str) -> float:
    """Look up actual strikeouts; also tries game_date+1 for UTC offset."""
    from datetime import datetime
    for delta in range(2):
        d = (datetime.strptime(game_date, "%Y-%m-%d") + timedelta(days=delta)).strftime("%Y-%m-%d")
        rows = logs[
            (logs["game_date"] == d) &
            (logs["pitcher_name"].str.lower() == pitcher_name.strip().lower())
        ]
        if not rows.empty:
            return float(rows.iloc[0]["strikeouts"])
        last = pitcher_name.strip().split()[-1].lower()
        fuzzy = logs[
            (logs["game_date"] == d) &
            (logs["pitcher_name"].str.lower().str.contains(last))
        ]
        if len(fuzzy) == 1:
            return float(fuzzy.iloc[0]["strikeouts"])
    return np.nan


def rebuild_backtest(picks_log: pd.DataFrame) -> None:
    bt = pd.read_csv(BT_2026)
    bt["game_date"] = bt["game_date"].astype(str).str[:10]
    bt_old = bt[bt["game_date"] < "2026-06-09"].copy()
    bt_old = bt_old[pd.to_numeric(bt_old["edge_pct"], errors="coerce") >= 7.0]

    needed = ["game_date", "pitcher_name", "best_side", "line",
              "strikeouts_projection", "gap", "edge_pct", "odds_used", "actual", "won"]
    log_sub = picks_log.copy()
    for col in ["line", "strikeouts_projection", "gap", "edge_pct", "odds_used", "actual"]:
        log_sub[col] = pd.to_numeric(log_sub[col], errors="coerce")

    for col in needed:
        if col not in log_sub.columns:
            log_sub[col] = np.nan

    combined = pd.concat([bt_old, log_sub[needed]], ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)
    combined.to_csv(BT_2026, index=False)

    settled = combined[combined["won"].astype(str).isin(["0", "1", "0.0", "1.0"])]
    wins = settled[settled["won"].astype(str).isin(["1", "1.0"])]
    total_pnl = sum(
        ((american_to_decimal(float(r["odds_used"])) - 1) * 100
         if str(r["won"]) in ("1", "1.0") else -100)
        for _, r in settled.iterrows()
        if pd.notna(r["odds_used"])
    )
    roi = total_pnl / (len(settled) * 100) * 100 if settled.shape[0] > 0 else 0
    print(f"Backtest: {len(combined)} picks  ({len(settled)} settled)  "
          f"{len(wins)}W/{len(settled)-len(wins)}L  "
          f"P&L=${total_pnl:+.0f}  ROI={roi:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve prior-day picks")
    parser.add_argument("--date", default=None,
                        help="Date to resolve (default: yesterday, YYYY-MM-DD)")
    args = parser.parse_args()

    resolve_date = args.date or (date.today() - timedelta(days=1)).isoformat()
    print(f"Resolving picks for {resolve_date}...")

    # 1. Fetch / update game logs for that date
    print(f"  Fetching game logs for {resolve_date}...")
    new_logs = fetch_pitcher_game_logs(resolve_date, resolve_date)
    merged   = _merge_logs(new_logs)
    merged.to_csv(LOGS_FILE, index=False)
    date_rows = merged[merged["game_date"] == resolve_date]
    print(f"  {len(date_rows)} pitcher rows for {resolve_date} in logs")

    # 2. Load picks_log and resolve the target date
    if not PICKS_LOG.exists():
        print("No picks_log.csv found — nothing to resolve.")
        return

    logs_df = pd.read_csv(LOGS_FILE)
    logs_df["game_date"] = logs_df["game_date"].astype(str).str[:10]

    picks = pd.read_csv(PICKS_LOG, dtype=str)
    picks["game_date"] = picks["game_date"].astype(str).str[:10]

    mask = picks["game_date"] == resolve_date
    if not mask.any():
        print(f"  No picks found for {resolve_date} in picks_log.")
        return

    updated = 0
    for idx in picks[mask].index:
        row = picks.loc[idx]
        if str(row.get("won", "")) in ("0", "1"):
            continue  # already resolved
        actual = lookup_k(logs_df, resolve_date, str(row["pitcher_name"]))
        if pd.notna(actual):
            line = float(row["line"])
            side = str(row["best_side"])
            won  = 1 if (side == "over" and actual > line) or \
                        (side == "under" and actual < line) else 0
            picks.loc[idx, "actual"] = str(actual)
            picks.loc[idx, "won"]    = str(won)
            updated += 1

    picks.to_csv(PICKS_LOG, index=False)
    resolved = picks[(picks["game_date"] == resolve_date) & picks["won"].astype(str).isin(["0", "1"])]
    wins = resolved[resolved["won"] == "1"]
    print(f"  Resolved {updated} new picks  "
          f"({len(wins)}W/{len(resolved)-len(wins)}L for {resolve_date})")

    # 3. Rebuild 2026 backtest
    rebuild_backtest(picks)


if __name__ == "__main__":
    main()
