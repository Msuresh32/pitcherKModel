"""Find the optimal |projection - line| gap threshold for betting."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def roi_fn(grp):
    if grp.empty: return np.nan
    odds = np.where(grp["best_side"]=="over", grp["over_odds"], grp["under_odds"])
    dec = np.where(odds > 0, 1+odds/100, 1+100/np.abs(np.where(odds==0,1,odds)))
    return float(np.where(grp["won"].astype(bool), dec-1, -1.0).mean())


def resolve_outcome(row):
    mkt = row["market"]
    actual = row.get(mkt)
    if pd.isna(actual): return np.nan
    return 1 if (actual > row["line"] if row["best_side"]=="over" else actual < row["line"]) else 0


def main():
    edges = pd.read_csv("data/processed/backtest_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])

    q = edges[(edges["edge_pct"] >= 2.0) & (edges["edge_pct"] <= 15.0)].copy()
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])

    sk = q[q["market"]=="strikeouts"].copy()
    sk["abs_gap"] = abs(sk["strikeouts_projection"] - sk["line"])

    print("=== Strikeouts: ROI by max projection gap (edge 2-15%) ===")
    print(f"{'MaxGap':>8} {'N':>6} {'WinRate':>8} {'ROI':>8}")
    for max_gap in [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 999]:
        sub = sk[sk["abs_gap"] <= max_gap]
        if len(sub) < 20: continue
        label = f"<= {max_gap}" if max_gap < 999 else "all"
        print(f"  {label:>6}  {len(sub):>6}  {sub['won'].mean():>8.3f}  {roi_fn(sub):>8.3f}")

    print("\n=== Strikeouts: ROI by max gap + min edge 5% ===")
    sk5 = sk[sk["edge_pct"] >= 5.0]
    for max_gap in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 999]:
        sub = sk5[sk5["abs_gap"] <= max_gap]
        if len(sub) < 20: continue
        label = f"<= {max_gap}" if max_gap < 999 else "all"
        print(f"  {label:>6}  {len(sub):>6}  {sub['won'].mean():>8.3f}  {roi_fn(sub):>8.3f}")

    # Also check under vs over separately at near-line
    print("\n=== Near-line (gap <= 0.8) by side ===")
    near = sk[sk["abs_gap"] <= 0.8]
    for side in ["over", "under"]:
        sub = near[near["best_side"]==side]
        print(f"  {side}: n={len(sub)}  win={sub['won'].mean():.3f}  roi={roi_fn(sub):.3f}")

    # Best combo: gap + edge + side
    print("\n=== Best combos: gap<=0.8, edge 3-10% ===")
    combo = sk[(sk["abs_gap"] <= 0.8) & (sk["edge_pct"] >= 3.0) & (sk["edge_pct"] <= 10.0)]
    print(f"  n={len(combo)}  win={combo['won'].mean():.3f}  roi={roi_fn(combo):.3f}")
    for side in ["over","under"]:
        s = combo[combo["best_side"]==side]
        if len(s) > 20:
            print(f"    {side}: n={len(s)}  win={s['won'].mean():.3f}  roi={roi_fn(s):.3f}")

    # CLV for near-line bets
    try:
        clv = pd.read_csv("data/processed/backtest_clv.csv")
        clv["game_date"] = pd.to_datetime(clv["game_date"])
        key = ["game_date","pitcher_id","market","line","best_side"]
        near_clv = near.merge(clv[key+["clv_pct"]], on=key, how="left").dropna(subset=["clv_pct"])
        if len(near_clv) > 0:
            print(f"\n=== CLV for near-line bets (gap<=0.8) ===")
            print(f"  n={len(near_clv)}  mean_CLV={near_clv['clv_pct'].mean():.3f}")
            pos = near_clv[near_clv["clv_pct"] > 0]
            neg = near_clv[near_clv["clv_pct"] <= 0]
            print(f"  positive CLV (n={len(pos)}): win={pos['won'].mean():.3f}  roi={roi_fn(pos):.3f}")
            print(f"  negative CLV (n={len(neg)}): win={neg['won'].mean():.3f}  roi={roi_fn(neg):.3f}")
    except Exception as e:
        print(f"CLV join failed: {e}")


if __name__ == "__main__":
    main()
