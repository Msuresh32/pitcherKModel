import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/full2026_dk_b70_30_e10_edges.csv")
df = df[df["market"] == "strikeouts"].copy()

def bet_won(row):
    if row["best_side"] == "over":
        return int(row["strikeouts"] > row["line"])
    else:
        return int(row["strikeouts"] < row["line"])

def payout(row):
    odds = row["over_odds"] if row["best_side"] == "over" else row["under_odds"]
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

df["won"] = df.apply(bet_won, axis=1)
df["payout"] = df.apply(payout, axis=1)
df["profit"] = df.apply(lambda r: r["payout"] if r["won"] else -1.0, axis=1)
df["gap"] = df["strikeouts_projection"] - df["line"]
df["game_date"] = pd.to_datetime(df["game_date"])
df["month"] = df["game_date"].dt.to_period("M")

# Deduplicate
dedup = (df.sort_values("edge_pct", ascending=False)
           .drop_duplicates(subset=["game_date", "pitcher_name", "line", "best_side"])
           .reset_index(drop=True))

print("=" * 55)
print("FULL 2026 OOS - OLD MODEL (DK lines, Mar26-Jun16)")
print("=" * 55)
print()

# Threshold sweep
print("Edge threshold sweep:")
print(f"{'MinEdge':>9} {'Bets':>5} {'Win%':>7} {'ROI':>8} {'Sharpe':>8}")
print("-" * 42)
for t in [0, 5, 10, 12, 15, 20, 25]:
    sub = dedup[dedup["edge_pct"] >= t]
    if len(sub) == 0:
        continue
    sharpe = sub["profit"].mean() / sub["profit"].std() * (len(sub)**0.5) if sub["profit"].std() > 0 else 0
    print(f"{t:>8}%  {len(sub):>5}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}  {sharpe:>8.2f}")

print()

# Monthly breakdown (edge >= 10%)
d10 = dedup[dedup["edge_pct"] >= 10]
print("Monthly (edge >= 10%):")
print(f"{'Month':<10} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for m, g in d10.groupby("month"):
    print(f"{str(m):<10} {len(g):>5}  {g['won'].mean():>6.1%}  {g['profit'].mean():>+7.1%}")

print()

# Over vs under
d10_ov = d10[d10["best_side"] == "over"]
d10_un = d10[d10["best_side"] == "under"]
print("Over vs Under (edge >= 10%):")
print(f"  OVER : {len(d10_ov):>3} bets  {d10_ov['won'].mean():>6.1%} win  {d10_ov['profit'].mean():>+7.1%} ROI")
print(f"  UNDER: {len(d10_un):>3} bets  {d10_un['won'].mean():>6.1%} win  {d10_un['profit'].mean():>+7.1%} ROI")

print()

# Gap tier
print("Gap tiers (edge >= 10%):")
print(f"{'Gap':>15} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
print("-" * 40)
bins = [("gap < 0 (under)", d10[d10["gap"] < 0]),
        ("0.0-0.5", d10[(d10["gap"] >= 0) & (d10["gap"] < 0.5)]),
        ("0.5-1.0", d10[(d10["gap"] >= 0.5) & (d10["gap"] < 1.0)]),
        ("1.0-1.5", d10[(d10["gap"] >= 1.0) & (d10["gap"] < 1.5)]),
        ("1.5+", d10[d10["gap"] >= 1.5])]
for label, sub in bins:
    if len(sub) == 0: continue
    print(f"{label:>15}  {len(sub):>5}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")
