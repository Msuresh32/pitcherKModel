import pandas as pd
import numpy as np

def analyze(path, label, min_edge=15):
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    def won(r): return (r["strikeouts"] > r["line"]) if r["best_side"]=="over" else (r["strikeouts"] < r["line"])
    def pay(r):
        o = r["over_odds"] if r["best_side"]=="over" else r["under_odds"]
        return o/100 if o>0 else 100/abs(o)
    df["won"] = df.apply(won, axis=1)
    df["pay"] = df.apply(pay, axis=1)
    df["profit"] = df.apply(lambda r: r["pay"] if r["won"] else -1.0, axis=1)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"] = df["game_date"].dt.to_period("M")
    dedup = (df.sort_values("edge_pct", ascending=False)
               .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
               .reset_index(drop=True))
    sub = dedup[dedup["edge_pct"] >= min_edge]
    print(f"\n{'='*60}")
    print(f"  {label}  (edge >= {min_edge}%)")
    print(f"{'='*60}")
    n = len(sub)
    sharpe = sub["profit"].mean()/sub["profit"].std()*n**0.5
    print(f"  Total:  {n} bets  |  {sub['won'].mean():.1%} win  |  {sub['profit'].mean():+.1%} ROI  |  Sharpe {sharpe:.2f}")
    print()
    print(f"  {'Month':<10} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
    print(f"  {'-'*33}")
    cumprofit = 0
    for m, g in sub.groupby("month"):
        cumprofit += g["profit"].sum()
        print(f"  {str(m):<10} {len(g):>5}  {g['won'].mean():>6.1%}  {g['profit'].mean():>+7.1%}")
    print()
    print(f"  Cumulative profit: {cumprofit:+.1f} units")
    return sub

print("\nWALK-FORWARD VALIDATION — FULL PICTURE")
print("Threshold frozen at edge >= 15% using 2025 data only")
print("="*60)

s25 = analyze("data/processed_2024/thresh_sel_2025_dk_edges.csv",
              "2025 — THRESHOLD SELECTION (2024-only model)", min_edge=15)
s26 = analyze("data/processed/frozen_thresh_2026_edges.csv",
              "2026 — FINAL TEST (old model, frozen threshold)", min_edge=15)

# Combined
all_bets = pd.concat([s25, s26], ignore_index=True)
print(f"\n{'='*60}")
print(f"  COMBINED 2025 + 2026  (edge >= 15%)")
print(f"{'='*60}")
n = len(all_bets)
sharpe = all_bets["profit"].mean()/all_bets["profit"].std()*n**0.5
print(f"  {n} bets  |  {all_bets['won'].mean():.1%} win  |  {all_bets['profit'].mean():+.1%} ROI  |  Sharpe {sharpe:.2f}")
