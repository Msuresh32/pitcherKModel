"""Honest assessment: over vs under split, and sanity checks on 2026 OOS results."""
import pandas as pd
import numpy as np

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
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

df["won"] = df.apply(bet_won, axis=1)
df["payout"] = df.apply(payout, axis=1)
df["profit"] = df.apply(lambda r: r["payout"] if r["won"] else -1.0, axis=1)
df["gap"] = df["strikeouts_projection"] - df["line"]

# ---- 1. Over vs Under split (all edge >= 0) ----
pos = df[df["edge_pct"] >= 0]
print("=" * 55)
print("OVER vs UNDER (edge >= 0, Jan-Jun 16 2026 OOS)")
print("=" * 55)
for side in ["over", "under"]:
    sub = pos[pos["best_side"] == side]
    roi = sub["profit"].mean()
    print(f"  {side.upper():6s}: {len(sub):>4} bets  {sub['won'].mean():>6.1%} win  {roi:>+7.1%} ROI  avg odds {sub.apply(lambda r: r['over_odds'] if r['best_side']=='over' else r['under_odds'], axis=1).mean():>+.0f}")

print()

# ---- 2. Over vs Under at production edge >= 12% ----
prod = df[df["edge_pct"] >= 12]
print("OVER vs UNDER (edge >= 12%, production threshold)")
print("-" * 55)
for side in ["over", "under"]:
    sub = prod[prod["best_side"] == side]
    if len(sub) == 0:
        continue
    roi = sub["profit"].mean()
    print(f"  {side.upper():6s}: {len(sub):>4} bets  {sub['won'].mean():>6.1%} win  {roi:>+7.1%} ROI")

print()

# ---- 3. Monthly ROI (sanity check -- is it consistent or lucky months?) ----
df["game_date"] = pd.to_datetime(df["game_date"])
df["month"] = df["game_date"].dt.to_period("M")
pos2 = df[df["edge_pct"] >= 0].copy()
print("MONTHLY BREAKDOWN (edge >= 0) -- is it consistent?")
print(f"{'Month':<10} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
print("-" * 35)
for month, grp in pos2.groupby("month"):
    print(f"{str(month):<10} {len(grp):>5}  {grp['won'].mean():>6.1%}  {grp['profit'].mean():>+7.1%}")

print()

# ---- 4. Are results driven by a few big wins? ----
print("DISTRIBUTION CHECK (edge >= 0)")
profits = pos2["profit"].values
print(f"  Mean profit per bet:   {profits.mean():>+.4f}")
print(f"  Median profit per bet: {np.median(profits):>+.4f}")
print(f"  Std dev:               {profits.std():>+.4f}")
print(f"  % bets won:            {(profits > 0).mean():>6.1%}")
print(f"  Avg win payout:        {profits[profits > 0].mean():>+.4f}")
print(f"  Avg loss:              {profits[profits < 0].mean():>+.4f}")
wins = profits[profits > 0]
losses = profits[profits < 0]
print(f"  Win/loss ratio:        {wins.mean() / abs(losses.mean()):.3f}")
top10 = np.array(sorted(profits)[-10:])
print(f"  Top 10 biggest wins contribution: {top10.sum() / profits.sum():.1%} of total profit")

print()

# ---- 5. Odds distribution -- are we beating juice? ----
print("ODDS DISTRIBUTION (edge >= 0)")
all_odds = pos2.apply(lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1)
print(f"  Avg odds:  {all_odds.mean():>+.1f}")
print(f"  % at negative odds (paying juice): {(all_odds < 0).mean():>6.1%}")
print(f"  % at +100 or better:               {(all_odds >= 100).mean():>6.1%}")
print(f"  Implied break-even win rate at avg odds: "
      f"{(abs(all_odds[all_odds<0]) / (abs(all_odds[all_odds<0]) + 100)).mean():.1%} (neg odds bets)")
