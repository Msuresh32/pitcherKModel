"""
Compute Closing Line Value (CLV) from real morning vs closing snapshots.

Usage:
    python scripts/compute_live_clv.py                  # all dates in picks_log
    python scripts/compute_live_clv.py --date 2026-06-17

How it works:
  - Morning snapshot (data/odds/snapshots/{date}_morning.csv): odds at pick time (~7am ET)
  - Closing snapshot (data/odds/snapshots/{date}_closing.csv): odds just before games (~6:45pm ET)
  - CLV% = (entry_decimal / closing_decimal - 1) * 100
    Positive = you got better odds than the market ended up at (you beat the closer)
    Negative = market moved against you (you were on the wrong side)

A consistently positive CLV over 50+ bets is strong evidence of genuine edge.
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PICKS_LOG    = Path("data/exports/picks_log.csv")
SNAPSHOT_DIR = Path("data/odds/snapshots")
OUTPUT       = Path("data/processed/live_clv.csv")


def _norm_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", ascii_name.lower())).strip()


def american_to_decimal(odds) -> float:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(o):
        return float("nan")
    return (100 / abs(o) + 1) if o < 0 else (o / 100 + 1)


def best_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-bookmaker odds to best available over/under per player + market + line.

    Always includes player_name (normalized) for name-based fallback matching, plus
    pitcher_id if the snapshot has it.
    """
    if df.empty:
        return df
    df = df.copy()
    df["_norm_name"] = df["player_name"].apply(_norm_name)
    has_pid = "pitcher_id" in df.columns
    id_col  = "pitcher_id" if has_pid else "_norm_name"
    rows = []
    for keys, grp in df.groupby([id_col, "market", "line"], dropna=False):
        over_row  = grp.sort_values("over_odds",  ascending=False, na_position="last").iloc[0]
        under_row = grp.sort_values("under_odds", ascending=False, na_position="last").iloc[0]
        row = {
            "_norm_name": grp["_norm_name"].iloc[0],
            "market":     keys[1],
            "line":       keys[2],
            "over_odds":  over_row["over_odds"],
            "under_odds": under_row["under_odds"],
        }
        if has_pid:
            row["pitcher_id"] = keys[0]
        rows.append(row)
    return pd.DataFrame(rows)


def load_snapshot(date_str: str, snapshot: str) -> pd.DataFrame:
    path = SNAPSHOT_DIR / f"{date_str}_{snapshot}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize pitcher_id to string for safe joining
    if "pitcher_id" in df.columns:
        df["pitcher_id"] = df["pitcher_id"].astype(str)
    return df


def compute_clv_for_date(picks: pd.DataFrame, date_str: str) -> pd.DataFrame:
    morning = load_snapshot(date_str, "morning")
    closing = load_snapshot(date_str, "closing")

    if morning.empty:
        print(f"  {date_str}: no morning snapshot — skipping")
        return pd.DataFrame()
    if closing.empty:
        print(f"  {date_str}: no closing snapshot yet — skipping")
        return pd.DataFrame()

    morning_best = best_lines(morning)
    closing_best = best_lines(closing)

    results = []
    for _, pick in picks.iterrows():
        raw_pid  = pick.get("pitcher_id", "")
        pid_valid = raw_pid not in (None, "", "nan", float("nan")) and not (isinstance(raw_pid, float) and np.isnan(raw_pid))
        pid   = str(raw_pid) if pid_valid else None
        raw_mkt = pick.get("market", "")
        mkt   = str(raw_mkt) if raw_mkt not in (None, "", "nan") and not (isinstance(raw_mkt, float) and np.isnan(raw_mkt)) else "strikeouts"
        line  = float(pick.get("line", float("nan")))
        side  = str(pick.get("best_side", ""))
        name_norm = _norm_name(pick.get("pitcher_name", ""))

        # Match by pitcher_id if available, else by normalized name
        has_pid_col = "pitcher_id" in morning_best.columns
        if pid and has_pid_col:
            m_row = morning_best[
                (morning_best["pitcher_id"].astype(str) == pid) &
                (morning_best["market"] == mkt) &
                (morning_best["line"] == line)
            ]
            c_row = closing_best[
                (closing_best["pitcher_id"].astype(str) == pid) &
                (closing_best["market"] == mkt) &
                (closing_best["line"] == line)
            ]
        else:
            m_row = morning_best[
                (morning_best["_norm_name"] == name_norm) &
                (morning_best["market"] == mkt) &
                (morning_best["line"] == line)
            ]
            c_row = closing_best[
                (closing_best["_norm_name"] == name_norm) &
                (closing_best["market"] == mkt) &
                (closing_best["line"] == line)
            ]

        if m_row.empty or c_row.empty:
            entry_odds = closing_odds = float("nan")
            clv = float("nan")
        else:
            entry_odds  = m_row.iloc[0]["over_odds"]  if side == "over"  else m_row.iloc[0]["under_odds"]
            closing_odds = c_row.iloc[0]["over_odds"] if side == "over"  else c_row.iloc[0]["under_odds"]
            entry_dec  = american_to_decimal(entry_odds)
            close_dec  = american_to_decimal(closing_odds)
            clv = (entry_dec / close_dec - 1) * 100 if not (np.isnan(entry_dec) or np.isnan(close_dec) or close_dec <= 1) else float("nan")

        results.append({
            "game_date":    date_str,
            "pitcher_name": pick.get("pitcher_name", ""),
            "pitcher_id":   pid,
            "market":       mkt,
            "line":         line,
            "best_side":    side,
            "entry_odds":   entry_odds,
            "closing_odds": closing_odds,
            "clv_pct":      clv,
            "edge_pct":     pick.get("edge_pct", float("nan")),
            "actual":       pick.get("actual", ""),
            "won":          pick.get("won", ""),
        })

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Specific date (YYYY-MM-DD) or all if omitted")
    args = parser.parse_args()

    if not PICKS_LOG.exists():
        print(f"No picks log found at {PICKS_LOG}")
        return

    picks = pd.read_csv(PICKS_LOG, dtype=str)
    if "pitcher_id" not in picks.columns:
        print("Warning: picks_log missing pitcher_id — CLV matching will be approximate (name-based).")
        picks["pitcher_id"] = picks.get("pitcher_name", "")

    if args.date:
        dates = [args.date]
    else:
        dates = sorted(picks["game_date"].dropna().unique())

    all_clv = []
    for d in dates:
        day_picks = picks[picks["game_date"] == d].copy()
        if day_picks.empty:
            continue
        result = compute_clv_for_date(day_picks, d)
        if not result.empty:
            all_clv.append(result)

    if not all_clv:
        print("No CLV data available yet (need both morning and closing snapshots).")
        return

    clv_df = pd.concat(all_clv, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clv_df.to_csv(OUTPUT, index=False)

    # Summary
    valid = clv_df.dropna(subset=["clv_pct"])
    print(f"\n{'='*50}")
    print(f"LIVE CLV SUMMARY  ({len(valid)} bets with closing lines)")
    print(f"{'='*50}")
    if valid.empty:
        print("No matched bets yet.")
        return

    mean_clv = valid["clv_pct"].mean()
    positive = (valid["clv_pct"] > 0).sum()
    print(f"Mean CLV:          {mean_clv:+.2f}%")
    print(f"Beat closing line: {positive}/{len(valid)} ({positive/len(valid):.1%})")
    print(f"Median CLV:        {valid['clv_pct'].median():+.2f}%")
    print(f"Std dev:           {valid['clv_pct'].std():.2f}%")

    # CLV vs win rate
    if "won" in valid.columns:
        won_valid = valid[valid["won"].notna() & (valid["won"] != "")]
        if not won_valid.empty:
            won_valid = won_valid.copy()
            won_valid["won_bool"] = won_valid["won"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])
            print(f"\nOf {len(won_valid)} settled bets:")
            print(f"  Win rate:          {won_valid['won_bool'].mean():.1%}")
            for label, mask in [("CLV > 0", won_valid["clv_pct"] > 0), ("CLV <= 0", won_valid["clv_pct"] <= 0)]:
                sub = won_valid[mask]
                if not sub.empty:
                    print(f"  Win rate ({label}): {sub['won_bool'].mean():.1%}  (n={len(sub)})")

    # Per-date breakdown
    print(f"\nPer-date breakdown:")
    for d, grp in valid.groupby("game_date"):
        print(f"  {d}: {len(grp)} bets, mean CLV {grp['clv_pct'].mean():+.2f}%")

    print(f"\nFull results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
