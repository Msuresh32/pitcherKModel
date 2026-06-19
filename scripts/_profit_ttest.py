"""
Profit t-test: proper significance test for ROI with variable odds.

For each bet:
  profit = (decimal_odds - 1) if won, else -1.0  (per unit staked)

H0: E[profit] = 0  (no edge)
H1: E[profit] > 0  (positive edge)

This handles variable odds correctly — no assumption of flat -110 break-even.
"""
import sys, io
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def implied_prob(odds: float) -> float:
    o = float(odds)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)


def devig_pair(over_o, under_o):
    if pd.isna(over_o) or pd.isna(under_o):
        return np.nan, np.nan
    ip_o = implied_prob(over_o)
    ip_u = implied_prob(under_o)
    d = ip_o + ip_u
    if d <= 0:
        return np.nan, np.nan
    return ip_o / d, ip_u / d


def load_bets(edge_min=0):
    sources = [
        ("thresh_sel_2025_dk_edges.csv", "data/processed_2024"),
        ("wf2026_p1_mar_apr_edges.csv",  "data/processed"),
        ("wf2026_p2_may_edges.csv",       "data/processed_apr2026"),
        ("wf2026_p3_jun_edges.csv",       "data/processed"),
    ]
    dfs = []
    for fname, d in sources:
        p = Path(d) / fname
        if p.exists():
            df = pd.read_csv(p)
            df["_source"] = fname
            dfs.append(df[df["market"] == "strikeouts"].copy())
        else:
            print(f"  [missing] {p}")

    if not dfs:
        print("No bet files found.")
        return pd.DataFrame()

    bets = pd.concat(dfs, ignore_index=True)
    bets = (bets.sort_values("edge_pct", ascending=False)
                .drop_duplicates(subset=["game_date", "pitcher_name", "line", "best_side"])
                .reset_index(drop=True))
    bets = bets[bets["edge_pct"] >= edge_min].copy()
    bets["game_date"] = pd.to_datetime(bets["game_date"])

    # Actual bet odds
    bets["bet_odds"] = np.where(
        bets["best_side"] == "over",
        bets["over_odds"],
        bets["under_odds"],
    )
    bets["decimal_odds"] = bets["bet_odds"].apply(american_to_decimal)

    # Break-even probability per bet (no-vig entry side)
    nv = bets.apply(lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    bets["nv_over"] = nv[0]
    bets["nv_under"] = nv[1]
    bets["breakeven_prob"] = np.where(
        bets["best_side"] == "over",
        bets["nv_over"],
        bets["nv_under"],
    )

    # Win/loss
    bets["won"] = np.where(
        bets["best_side"] == "over",
        bets["strikeouts"] > bets["line"],
        bets["strikeouts"] < bets["line"],
    )

    # Profit per unit staked
    bets["profit"] = np.where(
        bets["won"],
        bets["decimal_odds"] - 1,
        -1.0,
    )

    return bets.dropna(subset=["profit", "won", "decimal_odds"])


def run_analysis(bets: pd.DataFrame, label: str):
    if bets.empty:
        print(f"\n{label}: no bets")
        return

    n = len(bets)
    wins = int(bets["won"].sum())
    win_rate = wins / n

    # Weighted break-even rate (average of per-bet break-even probs)
    avg_breakeven = bets["breakeven_prob"].mean()

    mean_profit = bets["profit"].mean()
    std_profit = bets["profit"].std(ddof=1)
    se = std_profit / np.sqrt(n)
    t_stat = mean_profit / se
    p_val = stats.t.sf(t_stat, df=n - 1)  # one-tailed

    # Expected profit under H0 = 0, so ROI test is same as profit test
    roi = mean_profit  # per unit = ROI

    # Weighted break-even win test (generalized binomial)
    # Under H0: each bet wins with prob p_i (break-even)
    expected_wins = bets["breakeven_prob"].sum()
    var_wins = (bets["breakeven_prob"] * (1 - bets["breakeven_prob"])).sum()
    z_wins = (wins - expected_wins) / np.sqrt(var_wins)
    p_wins = stats.norm.sf(z_wins)  # one-tailed

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  Bets (n):              {n}")
    print(f"  Wins:                  {wins}  ({win_rate:.1%})")
    print(f"  Avg break-even prob:   {avg_breakeven:.3%}  (weighted by actual odds)")
    print(f"  Expected wins (H0):    {expected_wins:.1f}")
    print(f"  Excess wins:           {wins - expected_wins:+.1f}")
    print()
    print(f"  Mean profit/unit:      {mean_profit:+.4f}  ({roi:+.2%} ROI)")
    print(f"  Std profit/unit:       {std_profit:.4f}")
    print(f"  Std error:             {se:.4f}")
    print()
    print(f"  --- Profit t-test (H0: E[profit]=0) ---")
    print(f"  t-stat:                {t_stat:.3f}")
    print(f"  p-value (one-tailed):  {p_val:.2e}")
    sig = "*** (p<0.001)" if p_val < 0.001 else "** (p<0.01)" if p_val < 0.01 else "* (p<0.05)" if p_val < 0.05 else "ns"
    print(f"  Significance:          {sig}")
    print()
    print(f"  --- Generalized binomial test (H0: win at break-even rate) ---")
    print(f"  z-stat:                {z_wins:.3f}")
    print(f"  p-value (one-tailed):  {p_wins:.2e}")
    sig2 = "*** (p<0.001)" if p_wins < 0.001 else "** (p<0.01)" if p_wins < 0.01 else "* (p<0.05)" if p_wins < 0.05 else "ns"
    print(f"  Significance:          {sig2}")
    print()

    # Odds distribution
    print(f"  --- Odds distribution ---")
    bins = [(-999,-150), (-150,-120), (-120,-105), (-105,-90), (-90, 0), (0, 999)]
    labels = ["<-150", "-150 to -120", "-120 to -105", "-105 to -90", "-90 to 0", "positive"]
    for (lo, hi), lbl in zip(bins, labels):
        sub = bets[(bets["bet_odds"] >= lo) & (bets["bet_odds"] < hi)]
        if len(sub) > 0:
            wr = sub["won"].mean()
            be = sub["breakeven_prob"].mean()
            print(f"    {lbl:<20}: n={len(sub):>4}  WR={wr:.1%}  BE={be:.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

print("Loading all bets...")
all_bets = load_bets(edge_min=0)
print(f"  Total bets loaded (edge >= 0%):  {len(all_bets)}")

bets_edge15 = all_bets[all_bets["edge_pct"] >= 15].copy()
print(f"  Bets at edge >= 15%:             {len(bets_edge15)}")

# Split by period
is_2025 = all_bets["game_date"].dt.year == 2025
is_2026 = all_bets["game_date"].dt.year == 2026

run_analysis(all_bets[all_bets["edge_pct"] >= 15], "EDGE >= 15% — ALL PERIODS COMBINED")
run_analysis(all_bets[is_2025 & (all_bets["edge_pct"] >= 15)], "EDGE >= 15% — 2025 ONLY")
run_analysis(all_bets[is_2026 & (all_bets["edge_pct"] >= 15)], "EDGE >= 15% — 2026 ONLY (true OOS)")
run_analysis(all_bets[all_bets["edge_pct"] >= 0], "ALL BETS (edge >= 0%) — REFERENCE")
