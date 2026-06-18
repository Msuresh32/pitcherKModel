import pandas as pd, numpy as np
from scipy import stats

def load(pfx, d):
    df = pd.read_csv(f"{d}/{pfx}_edges.csv")
    df = df[df["market"] == "strikeouts"].copy()
    df["won"] = df.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                  else (r["strikeouts"] < r["line"]), axis=1)
    df["pay"] = df.apply(
        lambda r: (r["over_odds"]/100 if r["over_odds"] > 0 else 100/abs(r["over_odds"]))
                  if r["best_side"] == "over"
                  else (r["under_odds"]/100 if r["under_odds"] > 0 else 100/abs(r["under_odds"])), axis=1)
    df["profit"] = df.apply(lambda r: r["pay"] if r["won"] else -1.0, axis=1)
    df = (df.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))
    return df[df["edge_pct"] >= 15].copy()

periods = [
    ("Mar 26 - Apr 30  [Dec-2025 model]", "wf2026_p1_mar_apr", "data/processed"),
    ("May 01 - May 31  [Apr-2026 model]", "wf2026_p2_may",     "data/processed_apr2026"),
    ("Jun 01 - Jun 16  [May-2026 model]", "wf2026_p3_jun",     "data/processed"),
]

all_dfs = []
print()
print("MONTHLY WALK-FORWARD 2026")
print("Each period uses a model trained BEFORE it started")
print("=" * 65)

for label, pfx, d in periods:
    df = load(pfx, d)
    n   = len(df)
    roi = df["profit"].mean()
    sh  = roi / df["profit"].std() * n**0.5

    clv_df  = pd.read_csv(f"{d}/{pfx}_clv.csv")
    clv_col = clv_df["clv_pct"].dropna() if "clv_pct" in clv_df.columns else pd.Series(dtype=float)
    clv_str = f"{clv_col.mean():+.2f}%" if len(clv_col) > 0 else "N/A"

    print(f"\n  {label}")
    print(f"  {n} bets | {df['won'].mean():.1%} win | {roi:+.1%} ROI | Sharpe {sh:.2f} | CLV {clv_str}")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"]     = df["game_date"].dt.to_period("M")
    for m, g in df.groupby("month"):
        print(f"    {str(m)}  {len(g):>3} bets  {g['won'].mean():.1%} win  {g['profit'].mean():+.1%} ROI")
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
n   = len(combined)
roi = combined["profit"].mean()
sh  = roi / combined["profit"].std() * n**0.5

np.random.seed(42)
odds_vals = combined.apply(
    lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1)
implied = [abs(o)/(abs(o)+100) if o < 0 else 100/(o+100) for o in odds_vals]
bep   = np.mean(implied)
_, p2 = stats.ttest_1samp(combined["profit"].values, 0)
p_b   = stats.binom_test(int(combined["won"].sum()), n, bep, alternative="greater")
boot  = [np.random.choice(combined["profit"].values, n, replace=True).mean()
         for _ in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

print()
print("=" * 65)
print("COMBINED WALK-FORWARD 2026  (edge >= 15%, DK lines)")
print("=" * 65)
print(f"  Bets:      {n}")
print(f"  Win rate:  {combined['won'].mean():.1%}")
print(f"  ROI:       {roi:+.1%}")
print(f"  Sharpe:    {sh:.2f}")
print(f"  p (t-test): {p2/2:.4f}  |  p (binomial): {p_b:.4f}")
print(f"  95% CI:    [{lo:+.2%},  {hi:+.2%}]")
print(f"  Sig at 5%: {'YES' if p2/2 < 0.05 else 'NO'}")
print()

combined["game_date"] = pd.to_datetime(combined["game_date"])
combined["month"]     = combined["game_date"].dt.to_period("M")
print("  Monthly breakdown:")
for m, g in combined.groupby("month"):
    print(f"    {str(m)}  {len(g):>3} bets  {g['won'].mean():.1%} win  {g['profit'].mean():+.1%} ROI")
