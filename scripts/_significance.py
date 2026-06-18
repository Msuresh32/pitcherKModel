import pandas as pd
import numpy as np
from scipy import stats

def load_dedup(path, min_edge=15):
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    def won(r): return (r["strikeouts"] > r["line"]) if r["best_side"]=="over" else (r["strikeouts"] < r["line"])
    def pay(r):
        o = r["over_odds"] if r["best_side"]=="over" else r["under_odds"]
        return o/100 if o>0 else 100/abs(o)
    df["won"]    = df.apply(won, axis=1)
    df["profit"] = df.apply(lambda r: pay(r) if r["won"] else -1.0, axis=1)
    df["odds"]   = df.apply(lambda r: r["over_odds"] if r["best_side"]=="over" else r["under_odds"], axis=1)
    dedup = (df.sort_values("edge_pct", ascending=False)
               .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
               .reset_index(drop=True))
    return dedup[dedup["edge_pct"] >= min_edge].copy()

s25 = load_dedup("data/processed_2024/thresh_sel_2025_dk_edges.csv", 15)
s26 = load_dedup("data/processed/frozen_thresh_2026_edges.csv", 15)
all_bets = pd.concat([s25, s26], ignore_index=True)

np.random.seed(42)

def run_tests(df, label):
    profits = df["profit"].values
    wins    = df["won"].values
    odds    = df["odds"].values
    n       = len(profits)
    win_rate = wins.mean()
    mean_roi = profits.mean()

    # Break-even win rate given avg odds
    neg_odds = odds[odds < 0]
    pos_odds = odds[odds >= 0]
    implied = []
    for o in odds:
        if o < 0: implied.append(abs(o) / (abs(o) + 100))
        else:     implied.append(100 / (o + 100))
    avg_breakeven = np.mean(implied)

    # 1. t-test: mean profit > 0
    t_stat, p_ttest = stats.ttest_1samp(profits, 0)
    p_ttest_one = p_ttest / 2  # one-tailed

    # 2. Binomial test: win rate > break-even rate
    wins_count = int(wins.sum())
    p_binom = stats.binom_test(wins_count, n, avg_breakeven, alternative="greater")

    # 3. Bootstrap 95% CI for ROI
    boot_means = [np.random.choice(profits, n, replace=True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # 4. Z-score
    se = profits.std() / n**0.5
    z  = mean_roi / se

    # 5. Sharpe
    sharpe = mean_roi / profits.std() * n**0.5

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")
    print(f"  Win rate:          {win_rate:.3f}  ({win_rate:.1%})")
    print(f"  Avg break-even:    {avg_breakeven:.3f}  ({avg_breakeven:.1%})")
    print(f"  Mean ROI per bet:  {mean_roi:+.4f}  ({mean_roi:+.2%})")
    print(f"  Std dev per bet:   {profits.std():.4f}")
    print()
    print(f"  ── Statistical Tests ──")
    print(f"  t-test (ROI > 0):     t={t_stat:+.3f},  p={p_ttest_one:.4f}  {'*** SIGNIFICANT' if p_ttest_one < 0.05 else '(not sig at 5%)'}")
    print(f"  Binomial (win > BEP): p={p_binom:.4f}             {'*** SIGNIFICANT' if p_binom < 0.05 else '(not sig at 5%)'}")
    print(f"  Z-score (ROI/SE):     z={z:+.3f}")
    print()
    print(f"  Bootstrap 95% CI for ROI:  [{ci_lo:+.2%},  {ci_hi:+.2%}]")
    print(f"  (If CI excludes 0, edge is statistically real)")
    print()
    print(f"  Sharpe ratio:  {sharpe:.2f}  (>1.5 = strong, >2.0 = very strong)")

    # Rough sample size for 95% confidence at observed edge
    # n needed = (z_alpha * sigma / effect)^2
    z_95 = 1.645
    needed = (z_95 * profits.std() / mean_roi) ** 2
    print(f"  Bets needed for 95% confidence at this edge: ~{int(needed)}")
    return {"p_t": p_ttest_one, "p_b": p_binom, "ci_lo": ci_lo, "ci_hi": ci_hi}

r25  = run_tests(s25,       "2025 — Threshold selection period")
r26  = run_tests(s26,       "2026 — Blind test (frozen threshold)")
rall = run_tests(all_bets,  "COMBINED 2025 + 2026")

print("\n\n  VERDICT SUMMARY")
print("  " + "="*50)
for label, r in [("2025 alone", r25), ("2026 alone", r26), ("Combined", rall)]:
    sig = "SIGNIFICANT" if r["p_t"] < 0.05 and r["p_b"] < 0.05 else \
          "BORDERLINE"  if r["p_t"] < 0.10 or  r["p_b"] < 0.10 else \
          "NOT SIG"
    print(f"  {label:<16}: t-test p={r['p_t']:.4f}  binom p={r['p_b']:.4f}  "
          f"CI=[{r['ci_lo']:+.1%},{r['ci_hi']:+.1%}]  → {sig}")
