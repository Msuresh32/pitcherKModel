import pandas as pd
import numpy as np

df = pd.read_csv("data/processed_2024/thresh_sel_2025_dk_edges.csv")
df = df[df["market"] == "strikeouts"].copy()

def bet_won(r):
    return (r["strikeouts"] > r["line"]) if r["best_side"] == "over" else (r["strikeouts"] < r["line"])
def payout(r):
    o = r["over_odds"] if r["best_side"] == "over" else r["under_odds"]
    return o / 100.0 if o > 0 else 100.0 / abs(o)

df["won"]    = df.apply(bet_won, axis=1)
df["payout"] = df.apply(payout, axis=1)
df["profit"] = df.apply(lambda r: r["payout"] if r["won"] else -1.0, axis=1)

dedup = (df.sort_values("edge_pct", ascending=False)
           .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
           .reset_index(drop=True))

print("2025 THRESHOLD SELECTION (2024-only model, DK lines, truly OOS)")
print("="*60)
print(f"{'Threshold':>10}  {'Bets':>5}  {'Win%':>7}  {'ROI':>8}  {'Sharpe':>8}")
print("-"*48)

best_t, best_sharpe = None, -99
for t in [0, 5, 7, 10, 12, 15, 18, 20, 25]:
    sub = dedup[dedup["edge_pct"] >= t]
    if len(sub) < 50: continue
    sharpe = sub["profit"].mean() / sub["profit"].std() * len(sub)**0.5
    flag = ""
    # Track best with >= 200 bets (robust)
    if len(sub) >= 200 and sharpe > best_sharpe:
        best_sharpe, best_t = sharpe, t
    print(f"edge >= {t:>2}%  {len(sub):>5}  {sub['won'].mean():>6.1%}  {sub['profit'].mean():>+7.1%}  {sharpe:>8.2f}")

print()
print(f">>> SELECTED THRESHOLD: edge >= {best_t}%  (best Sharpe with >= 200 bets)")
print(f"    2025 OOS performance: {len(dedup[dedup['edge_pct']>=best_t])} bets, "
      f"{dedup[dedup['edge_pct']>=best_t]['won'].mean():.1%} win, "
      f"{dedup[dedup['edge_pct']>=best_t]['profit'].mean():+.1%} ROI")
print()
print("This threshold is now FROZEN. Applying to 2026 OOS below.")
