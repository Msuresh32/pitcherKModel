"""Verify CLV calculation end-to-end: duplicates, formula, and snapshot timing."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def to_decimal(american):
    american = float(american)
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def main():
    clv = pd.read_csv("data/processed/backtest_clv.csv")
    hist = pd.read_csv("data/odds/historical_pitcher_props_2025.csv")
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    hist["pitcher_id"] = (
        pd.to_numeric(hist["pitcher_id"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .reindex(hist.index, fill_value="")
    )

    # ── 1. Duplicate check ───────────────────────────────────────────────────
    key = ["game_date", "pitcher_id", "market", "line", "best_side"]
    dupe_mask = clv.duplicated(subset=key, keep=False)
    print("=== Duplicate Check ===")
    print(f"Total CLV rows:      {len(clv)}")
    print(f"Duplicate rows:      {dupe_mask.sum()}")
    print(f"Unique bets:         {clv.drop_duplicates(subset=key).shape[0]}")
    print()

    # ── 2. Trace one specific bet manually ───────────────────────────────────
    # Pick: Matthew Liberatore, 2025-04-01, strikeouts, 4.5, over
    lib = hist[
        hist["pitcher_name"].str.contains("Liberatore", na=False)
        & (hist["game_date"].dt.date.astype(str) == "2025-04-01")
        & (hist["market"] == "strikeouts")
        & (hist["line"] == 4.5)
    ].copy()

    print("=== Raw snapshots: Liberatore 2025-04-01 strikeouts 4.5 ===")
    print(
        lib[
            ["snapshot_type", "historical_snapshot", "bookmaker", "over_odds", "under_odds"]
        ].to_string(index=False)
    )
    print()

    open_best = lib[lib["snapshot_type"] == "open"]["over_odds"].max()
    close_best = lib[lib["snapshot_type"] == "close"]["over_odds"].max()

    entry_dec = to_decimal(open_best)
    close_dec = to_decimal(close_best)
    manual_clv = (entry_dec / close_dec - 1) * 100

    print("=== Manual CLV calculation ===")
    print(f"Best open  over_odds: {open_best:+.0f}  -> decimal {entry_dec:.4f}")
    print(f"Best close over_odds: {close_best:+.0f}  -> decimal {close_dec:.4f}")
    print(f"CLV = ({entry_dec:.4f} / {close_dec:.4f} - 1) * 100 = {manual_clv:+.4f}%")

    stored = clv[
        clv["pitcher_name"].str.contains("Liberatore", na=False)
        & (clv["game_date"] == "2025-04-01")
        & (clv["market"] == "strikeouts")
        & (clv["line"] == 4.5)
    ]["clv_pct"].values
    print(f"Stored CLV in file:   {stored}")
    match = np.isclose(stored, manual_clv, atol=0.01).all()
    print(f"Match: {'YES' if match else 'NO — MISMATCH'}")
    print()

    # ── 3. Snapshot timing check ─────────────────────────────────────────────
    print("=== Snapshot timing ===")
    commence = lib["commence_time"].iloc[0]
    open_snap = lib[lib["snapshot_type"] == "open"]["historical_snapshot"].iloc[0]
    close_snap = lib[lib["snapshot_type"] == "close"]["historical_snapshot"].iloc[0]

    commence_ts = pd.to_datetime(commence, utc=True)
    open_ts = pd.to_datetime(open_snap, utc=True)
    close_ts = pd.to_datetime(close_snap, utc=True)

    hours_open = (commence_ts - open_ts).total_seconds() / 3600
    hours_close = (commence_ts - close_ts).total_seconds() / 3600

    print(f"Game commence time:       {commence}")
    print(f"Open snapshot time:       {open_snap}  ({hours_open:.2f}h before game)")
    print(f"Close snapshot time:      {close_snap}  ({hours_close:.2f}h before game)")
    timing_ok = hours_open > 1.0 and 0 <= hours_close < 0.5
    print(f"Timing looks correct:     {'YES' if timing_ok else 'WARNING — check snapshots'}")
    print()

    # ── 4. Summary stats on CLV ─────────────────────────────────────────────
    print("=== CLV summary (deduplicated) ===")
    deduped = clv.drop_duplicates(subset=key)
    print(f"Deduplicated bets:   {len(deduped)}")
    for mkt in deduped["market"].dropna().unique():
        sub = deduped[deduped["market"] == mkt]["clv_pct"].dropna()
        print(f"  {mkt:16s}: mean={sub.mean():+.3f}%  median={sub.median():+.3f}%  n={len(sub)}")
    overall = deduped["clv_pct"].dropna()
    print(f"  {'OVERALL':16s}: mean={overall.mean():+.3f}%  median={overall.median():+.3f}%  n={len(overall)}")


if __name__ == "__main__":
    main()
