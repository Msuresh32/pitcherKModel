import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/full2026_b70_30_e10_edges.csv")
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

# Raw (with duplicates)
raw = df[df["edge_pct"] >= 10]
print(f"Raw (edge >= 10%, with dupes): {len(raw)} bets, {raw['won'].mean():.1%} win, {raw['profit'].mean():+.1%} ROI")

# Deduplicated
dedup = (df.sort_values("edge_pct", ascending=False)
           .drop_duplicates(subset=["game_date", "pitcher_name", "line", "best_side"])
           .sort_values(["game_date", "pitcher_name"])
           .reset_index(drop=True))

d10 = dedup[dedup["edge_pct"] >= 10]
print(f"Dedup (edge >= 10%):           {len(d10)} bets, {d10['won'].mean():.1%} win, {d10['profit'].mean():+.1%} ROI")
print()

# Monthly breakdown
d10["game_date"] = pd.to_datetime(d10["game_date"])
d10["month"] = d10["game_date"].dt.to_period("M")
print(f"{'Month':<10} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for m, g in d10.groupby("month"):
    print(f"{str(m):<10} {len(g):>5}  {g['won'].mean():>6.1%}  {g['profit'].mean():>+7.1%}")

print()
# Edge threshold sweep on dedup
print("Cumulative thresholds (dedup):")
print(f"{'MinEdge':>9} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for t in [0, 5, 10, 12, 15, 20, 25]:
    sub = dedup[dedup["edge_pct"] >= t]
    print(f"{t:>8}%  {len(sub):>6}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")
