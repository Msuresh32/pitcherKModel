import pandas as pd, numpy as np
from pathlib import Path

def american_to_decimal(o):
    return o / 100 + 1 if o > 0 else 100 / abs(o) + 1

def load(path, edge_min=0):
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    df["won"] = df.apply(lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                          else (r["strikeouts"] < r["line"]), axis=1)
    df["entry_odds"] = df.apply(lambda r: r["over_odds"] if r["best_side"] == "over"
                                 else r["under_odds"], axis=1)
    df["decimal"] = df["entry_odds"].apply(american_to_decimal)
    df["bep"]     = 1 / df["decimal"]
    df["profit"]  = df.apply(lambda r: r["decimal"] - 1 if r["won"] else -1.0, axis=1)
    df = (df.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))
    return df[df["edge_pct"] >= edge_min]

# 2025 threshold selection
bets25 = load("data/processed_2024/thresh_sel_2025_dk_edges.csv")

# 2026 walk-forward
dfs = []
for f, d in [("wf2026_p1_mar_apr_edges.csv","data/processed"),
             ("wf2026_p2_may_edges.csv","data/processed_apr2026"),
             ("wf2026_p3_jun_edges.csv","data/processed")]:
    if Path(f"{d}/{f}").exists():
        dfs.append(load(f"{d}/{f}"))
bets26 = (pd.concat(dfs, ignore_index=True)
            .sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))

BANDS = [(15,20,"15-20%"), (20,25,"20-25%"), (25,999,"25%+"), (15,999,"15%+ (all)")]

SEP = "=" * 65

for label, bets in [("2025 THRESHOLD SELECTION", bets25), ("2026 WALK-FORWARD (OOS)", bets26)]:
    print(f"\n{SEP}")
    print(label)
    print(SEP)
    print(f"  {'Band':<12} {'N':>5}  {'Win%':>6}  {'BEP':>6}  {'Win-BEP':>8}  {'ROI':>7}  {'Sharpe':>7}")
    print("  " + "-"*58)
    for lo, hi, name in BANDS:
        sub = bets[(bets["edge_pct"] >= lo) & (bets["edge_pct"] < hi)]
        if len(sub) < 5:
            print(f"  {name:<12} {len(sub):>5}  <5 bets")
            continue
        n      = len(sub)
        win    = sub["won"].mean()
        bep    = sub["bep"].mean()
        roi    = sub["profit"].mean()
        sharpe = roi / sub["profit"].std() * n**0.5 if sub["profit"].std() > 0 else 0
        over_n = (sub["best_side"] == "over").sum()
        under_n= (sub["best_side"] == "under").sum()
        print(f"  {name:<12} {n:>5}  {win:>6.1%}  {bep:>6.1%}  {win-bep:>+8.1%}  {roi:>+7.1%}  {sharpe:>7.2f}")
    print()

    # Monthly breakdown for full 15%+ pool
    sub = bets[bets["edge_pct"] >= 15].copy()
    sub["game_date"] = pd.to_datetime(sub["game_date"])
    sub["month"] = sub["game_date"].dt.to_period("M")
    print(f"  Monthly (edge>=15%):")
    print(f"  {'Month':<10} {'N':>5}  {'Win%':>6}  {'ROI':>7}  {'Over':>5}  {'Under':>6}")
    print("  " + "-"*46)
    for m, g in sub.groupby("month"):
        roi = g["profit"].mean()
        ov  = (g["best_side"]=="over").sum()
        un  = (g["best_side"]=="under").sum()
        print(f"  {str(m):<10} {len(g):>5}  {g['won'].mean():>6.1%}  {roi:>+7.1%}  {ov:>5}  {un:>6}")
