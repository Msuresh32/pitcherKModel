"""Gap analysis on 2025 backtest data."""
import pandas as pd

df = pd.read_csv("data/processed/gap_2025_edges.csv")
df = df[df["market"] == "strikeouts"].copy()
df = df.dropna(subset=["line", "best_side", "edge_pct", "over_odds", "under_odds", "strikeouts"])

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

print(f"Total 2025 strikeout rows: {len(df)}")
print()

def gap_band(g):
    if g < 0:     return "A: gap < 0 (under)"
    elif g < 0.5: return "B: 0.0-0.5"
    elif g < 1.0: return "C: 0.5-1.0"
    elif g < 1.5: return "D: 1.0-1.5"
    else:         return "E: 1.5+"

df["gband"] = df["gap"].apply(gap_band)

print("2025 -- All edge levels:")
print(f"{'Gap band':<28} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 55)
for gb in sorted(df["gband"].unique()):
    sub = df[df["gband"] == gb]
    print(f"{gb[3:]:<28} {len(sub):>6}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")

print()
df5 = df[df["edge_pct"] >= 5].copy()
df5["gband"] = df5["gap"].apply(gap_band)

print("2025 -- Edge >= 5% only:")
print(f"{'Gap band':<28} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 55)
for gb in sorted(df5["gband"].unique()):
    sub = df5[df5["gband"] == gb]
    print(f"{gb[3:]:<28} {len(sub):>6}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")

print()
print("2025 -- Cumulative min gap (edge >= 5%):")
print(f"{'Min gap':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 38)
for thresh in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    sub = df5[df5["gap"] >= thresh]
    if len(sub) == 0:
        continue
    print(f"{thresh:>9.2f}  {len(sub):>6}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")

print()
print("Note: 2025 is partially in-sample (model trained on 2025 data).")
print("Patterns are directionally valid but win rates are inflated vs true OOS.")
