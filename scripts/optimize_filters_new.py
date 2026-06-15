"""Find optimal edge/gap filters for the new high-accuracy Statcast model."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def resolve_outcome(row):
    actual = row.get("strikeouts")
    if pd.isna(actual): return np.nan
    return 1 if (actual > row["line"] if row["best_side"]=="over" else actual < row["line"]) else 0


def roi_fn(grp):
    if grp.empty: return np.nan
    odds = np.where(grp["best_side"]=="over", grp["over_odds"], grp["under_odds"])
    dec = np.where(odds>0, 1+odds/100, 1+100/np.abs(np.where(odds==0,1,odds)))
    return float(np.where(grp["won"].astype(bool), dec-1, -1.0).mean())


def load_and_score(edges_path):
    e = pd.read_csv(edges_path)
    e["game_date"] = pd.to_datetime(e["game_date"])
    e = e[e["market"]=="strikeouts"].copy()
    e["won"] = e.apply(resolve_outcome, axis=1)
    e = e.dropna(subset=["won","edge_pct"])
    e["abs_gap"] = abs(e["strikeouts_projection"] - e["line"])
    return e


def scan(e, label):
    print(f"\n=== {label} ===")
    print(f"{'min_edge':>10} {'max_gap':>8} {'N':>6} {'Win%':>7} {'ROI':>8}")
    best_roi = -99; best_cfg = None
    for min_e in [0, 3, 5, 7, 10, 12, 15]:
        for max_gap in [0.5, 0.8, 1.0, 1.5, 2.0, 99]:
            sub = e[(e["edge_pct"] >= min_e) & (e["abs_gap"] <= max_gap)]
            if len(sub) < 30: continue
            r = roi_fn(sub) * 100
            label_g = f"<={max_gap}" if max_gap < 99 else "all"
            if len(sub) >= 50:
                print(f"  edge>={min_e}%  gap{label_g:>5}  {len(sub):>6}  {sub['won'].mean()*100:>6.1f}%  {r:>7.2f}%")
            if r > best_roi and len(sub) >= 50:
                best_roi = r; best_cfg = (min_e, max_gap, len(sub), sub['won'].mean(), r)
    if best_cfg:
        print(f"\n  BEST: edge>={best_cfg[0]}%  gap<={best_cfg[1]}  n={best_cfg[2]}  win={best_cfg[3]*100:.1f}%  roi={best_cfg[4]:.2f}%")
    return best_cfg


# Load both years
e25 = load_and_score("data/processed/backtest_final_2025_edges.csv")
e26 = load_and_score("data/processed/backtest_final_2026_edges.csv")

cfg25 = scan(e25, "2025 in-sample")
cfg26 = scan(e26, "2026 OOS")

# Find config that works well in BOTH years
print("\n=== Configs that are positive in BOTH 2025 and 2026 ===")
print(f"{'min_edge':>10} {'max_gap':>8} {'N25':>5} {'ROI25':>8} {'N26':>5} {'ROI26':>8}")
for min_e in [5, 7, 10, 12]:
    for max_gap in [1.0, 1.5, 2.0, 99]:
        sub25 = e25[(e25["edge_pct"] >= min_e) & (e25["abs_gap"] <= max_gap)]
        sub26 = e26[(e26["edge_pct"] >= min_e) & (e26["abs_gap"] <= max_gap)]
        if len(sub25) < 30 or len(sub26) < 20: continue
        r25 = roi_fn(sub25) * 100
        r26 = roi_fn(sub26) * 100
        if r25 > 0 and r26 > 0:
            label_g = f"<={max_gap}" if max_gap < 99 else "all"
            print(f"  edge>={min_e}%  gap{label_g:>6}  {len(sub25):>5}  {r25:>7.2f}%  {len(sub26):>5}  {r26:>7.2f}%  ***")
        elif r26 > 5:
            label_g = f"<={max_gap}" if max_gap < 99 else "all"
            print(f"  edge>={min_e}%  gap{label_g:>6}  {len(sub25):>5}  {r25:>7.2f}%  {len(sub26):>5}  {r26:>7.2f}%")
