"""One-off analysis of edge bandwidth performance from edge_full_edges.csv."""
import pandas as pd

df = pd.read_csv("data/processed/edge_full_edges.csv")
df = df[df["market"] == "strikeouts"].copy()
df = df.dropna(subset=["line", "best_side", "edge_pct", "over_odds", "under_odds", "strikeouts"])

def bet_won(row):
    if row["best_side"] == "over":
        return int(row["strikeouts"] > row["line"])
    else:
        return int(row["strikeouts"] < row["line"])

def payout(row):
    odds = row["over_odds"] if row["best_side"] == "over" else row["under_odds"]
    if odds > 0:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)

df["won"] = df.apply(bet_won, axis=1)
df["payout"] = df.apply(payout, axis=1)
df["profit"] = df.apply(lambda r: r["payout"] if r["won"] else -1.0, axis=1)

def edge_band(e):
    if e < 0:    return "A: < 0%"
    elif e < 5:  return "B: 0-5%"
    elif e < 10: return "C: 5-10%"
    elif e < 15: return "D: 10-15%"
    elif e < 20: return "E: 15-20%"
    elif e < 25: return "F: 20-25%"
    elif e < 35: return "G: 25-35%"
    else:        return "H: 35%+"

df["band"] = df["edge_pct"].apply(edge_band)

print(f"Total rows: {len(df)}")
print()
print(f"{'Band':<12} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'AvgEdge':>9} {'AvgOdds':>9}")
print("-" * 58)
for band in sorted(df["band"].unique()):
    sub = df[df["band"] == band]
    bets = len(sub)
    win_pct = sub["won"].mean()
    roi = sub["profit"].mean()
    avg_edge = sub["edge_pct"].mean()
    avg_odds = sub.apply(lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1).mean()
    print(f"{band[3:]:<12} {bets:>6}  {win_pct:>6.1%}  {roi:>+7.1%}  {avg_edge:>8.1f}%  {avg_odds:>+8.0f}")

print()
# Cumulative ROI: what if you only bet edge >= threshold?
print("Cumulative (edge >= threshold):")
print(f"{'Min edge':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Sharpe':>8}")
print("-" * 45)
for threshold in [0, 3, 5, 7, 10, 12, 15, 20, 25]:
    sub = df[df["edge_pct"] >= threshold]
    if len(sub) == 0:
        continue
    bets = len(sub)
    win_pct = sub["won"].mean()
    roi = sub["profit"].mean()
    sharpe = sub["profit"].mean() / sub["profit"].std() * (bets ** 0.5) if sub["profit"].std() > 0 else 0
    print(f"{threshold:>9}%  {bets:>6}  {win_pct:>6.1%}  {roi:>+7.1%}  {sharpe:>8.2f}")

print()
# Projection gap analysis for edge >= 0
df2 = df[df["edge_pct"] >= 0].copy()
df2["gap"] = df2["strikeouts_projection"] - df2["line"]

def gap_band(g):
    if g < 0:    return "A: gap < 0 (model below line)"
    elif g < 0.5: return "B: 0.0-0.5"
    elif g < 1.0: return "C: 0.5-1.0"
    elif g < 1.5: return "D: 1.0-1.5"
    else:         return "E: 1.5+"

df2["gband"] = df2["gap"].apply(gap_band)
print("Projection gap (model - line) breakdown:")
print(f"{'Gap band':<30} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 55)
for gb in sorted(df2["gband"].unique()):
    sub = df2[df2["gband"] == gb]
    print(f"{gb[3:]:<30} {len(sub):>6}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")

print()
# Best combos: edge >= 5% AND gap >= 0.5
combos = [
    ("edge>=5, gap>=0", df[(df["edge_pct"] >= 5)]),
    ("edge>=5, gap>=0.5", df[(df["edge_pct"] >= 5) & ((df["strikeouts_projection"] - df["line"]) >= 0.5)]),
    ("edge>=10, gap>=0.5", df[(df["edge_pct"] >= 10) & ((df["strikeouts_projection"] - df["line"]) >= 0.5)]),
    ("edge>=5, gap>=1.0", df[(df["edge_pct"] >= 5) & ((df["strikeouts_projection"] - df["line"]) >= 1.0)]),
    ("edge>=3-15%, gap>=0.5", df[(df["edge_pct"] >= 3) & (df["edge_pct"] < 15) & ((df["strikeouts_projection"] - df["line"]) >= 0.5)]),
    ("edge>=3-15%", df[(df["edge_pct"] >= 3) & (df["edge_pct"] < 15)]),
]
print("Combined filters:")
print(f"{'Filter':<35} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
print("-" * 60)
for label, sub in combos:
    if len(sub) == 0:
        print(f"{label:<35} {'0':>6}  {'N/A':>6}  {'N/A':>8}")
    else:
        print(f"{label:<35} {len(sub):>6}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}")
