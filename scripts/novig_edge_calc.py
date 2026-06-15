"""Calculate edge improvement from getting better prices on an exchange (NoVig).

Two scenarios modelled:
  A) 'X cents better' — add X to the American odds for the side we bet
     e.g. current +116 → +121 at +5 cents
  B) NoVig (vig-free) — remove the sportsbook margin from both sides,
     bet at true implied probability. This is what NoVig exchange gives you.

All ROI figures are flat-stake per bet.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def american_to_decimal(odds):
    o = float(odds)
    if np.isnan(o): return np.nan
    return (1 + o / 100) if o >= 100 else (1 + 100 / abs(o))


def add_cents_american(odds, cents):
    """Move American odds `cents` units in the bettor's favour (higher payout)."""
    o = float(odds)
    if o >= 100:
        return o + cents
    # Negative odds: moving toward zero means better payout
    # e.g. -115 + 5 = -110, -103 + 5 crosses par → +102
    improved = o + cents
    if o < -100 and improved > -100:
        # crossed the -100 / +100 gap (skip the gap: -100 = +100 = even money)
        overshoot = improved + 100       # how far past par (e.g. -98 → 2)
        improved  = 100 + overshoot      # +102
    return improved


def novig_decimal(over_odds, under_odds, side):
    """Return the no-vig decimal odds for `side` given both sides' American odds."""
    d_over  = american_to_decimal(over_odds)
    d_under = american_to_decimal(under_odds)
    if any(np.isnan(x) or x <= 1 for x in [d_over, d_under]):
        return np.nan
    p_over  = 1 / d_over
    p_under = 1 / d_under
    total   = p_over + p_under
    if side == "over":
        return total / p_over       # = 1 / (p_over / total)
    else:
        return total / p_under


def resolve_outcome(row):
    actual = row.get("strikeouts")
    if pd.isna(actual): return np.nan
    return 1 if (actual > row["line"] if row["best_side"] == "over" else actual < row["line"]) else 0


def load_filtered():
    edges = pd.read_csv("data/processed/backtest_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])
    q = edges[(edges["edge_pct"] >= 3.0) & (edges["edge_pct"] <= 10.0)].copy()
    q = q[q["market"] == "strikeouts"].copy()
    q["abs_gap"] = abs(q["strikeouts_projection"] - q["line"])
    q = q[q["abs_gap"] <= 0.8].copy()
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])
    # Current odds (American) for the side we bet
    q["current_odds_am"] = np.where(
        q["best_side"] == "over", q["over_odds"], q["under_odds"]
    )
    q["current_dec"] = q["current_odds_am"].apply(american_to_decimal)
    return q.sort_values("game_date").reset_index(drop=True)


def compute_scenario(q, label, dec_series):
    """Given a series of improved decimal odds, compute ROI + stats."""
    dec   = dec_series.values
    won   = q["won"].values
    profit = np.where(won.astype(bool), dec - 1, -1.0)
    n          = len(q)
    win_rate   = won.mean()
    roi        = profit.mean()
    breakeven  = (1.0 / dec).mean()
    z          = (win_rate - breakeven) / np.sqrt(breakeven * (1 - breakeven) / n)
    p          = float(stats.norm.sf(z))
    rng        = np.random.default_rng(42)
    boot_means = [profit[rng.integers(0, n, n)].mean() for _ in range(8_000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    units_yr = roi * n
    return dict(
        label=label, n=n, win_rate=win_rate,
        roi_pct=roi * 100, breakeven_pct=breakeven * 100,
        z=z, p=p, sig="YES" if p < 0.05 else "no",
        ci_lo=ci_lo * 100, ci_hi=ci_hi * 100,
        units_yr=units_yr,
    )


def main():
    q = load_filtered()
    n = len(q)

    print("=" * 72)
    print("  NOVIG EXCHANGE — PRICE IMPROVEMENT EDGE CALCULATOR")
    print("=" * 72)

    avg_am  = q["current_odds_am"].mean()
    avg_dec = q["current_dec"].mean()
    print(f"\nQualifying bets: {n}  |  Win rate: {q['won'].mean():.3f}")
    print(f"Current avg odds: {avg_am:+.1f} American  |  decimal {avg_dec:.3f}")
    print(f"Odds distribution (American):  "
          f"p25={q['current_odds_am'].quantile(0.25):+.0f}  "
          f"p50={q['current_odds_am'].quantile(0.50):+.0f}  "
          f"p75={q['current_odds_am'].quantile(0.75):+.0f}")
    pct_neg = (q["current_odds_am"] < 0).mean()
    print(f"% of bets at negative odds: {pct_neg:.1%}  "
          f"(positive: {1-pct_neg:.1%})")

    # ── Scenario A: X cents better American odds ──────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  SCENARIO A: Getting X cents better American odds")
    print(f"{'─'*72}")
    print(f"  (e.g. current +116 → +121 at +5 cents)")
    print()
    print(f"  {'Scenario':>12}  {'ROI':>8}  {'Win%':>6}  {'Breakeven':>9}  "
          f"{'95% CI':>20}  {'p-val':>7}  {'Sig?':>5}  {'Units/yr':>9}")
    print(f"  {'-'*90}")

    for cents in [0, 5, 10, 15, 25]:
        improved_am  = q["current_odds_am"].apply(lambda o: add_cents_american(o, cents))
        improved_dec = improved_am.apply(american_to_decimal)
        label = f"+{cents}c" if cents > 0 else "baseline"
        r = compute_scenario(q, label, improved_dec)
        ci_str = f"[{r['ci_lo']:+.1f}%, {r['ci_hi']:+.1f}%]"
        flag = "  <-- books" if cents == 0 else ""
        print(
            f"  {r['label']:>12}  {r['roi_pct']:>+7.2f}%  {r['win_rate']*100:>5.1f}%  "
            f"  {r['breakeven_pct']:>8.2f}%  {ci_str:>20}  "
            f"{r['p']:>7.4f}  {r['sig']:>5}  {r['units_yr']:>8.1f}u{flag}"
        )

    # ── Scenario B: NoVig — remove sportsbook margin ──────────────────────────
    print(f"\n{'─'*72}")
    print(f"  SCENARIO B: NoVig exchange — full vig removal")
    print(f"{'─'*72}")
    print(f"  (bet at the fair implied probability; no sportsbook margin)")

    # Compute per-bet no-vig decimal odds
    novig_dec = q.apply(
        lambda r: novig_decimal(r["over_odds"], r["under_odds"], r["best_side"]),
        axis=1
    )
    valid_novig = novig_dec.notna().mean()
    q_nv  = q[novig_dec.notna()].copy()
    nv_dec = novig_dec[novig_dec.notna()]

    # Average vig removed
    current_dec_valid = q_nv["current_dec"]
    avg_vig_removed   = (nv_dec.values - current_dec_valid.values).mean()
    avg_am_equiv      = avg_vig_removed * 100   # rough American cents equivalent

    print(f"\n  NoVig coverage: {valid_novig:.1%} of bets")
    print(f"  Avg vig removed: {avg_vig_removed:.4f} decimal  "
          f"(~{avg_am_equiv:.1f} American cents equiv.)")
    print()

    r_nv = compute_scenario(q_nv, "no-vig", nv_dec)
    r_bk = compute_scenario(q_nv, "baseline", q_nv["current_dec"])

    for r in [r_bk, r_nv]:
        ci_str = f"[{r['ci_lo']:+.1f}%, {r['ci_hi']:+.1f}%]"
        print(f"  {r['label']:>12}  ROI={r['roi_pct']:>+7.2f}%  "
              f"breakeven={r['breakeven_pct']:.2f}%  "
              f"95% CI {ci_str}  p={r['p']:.4f}  {r['sig']}")

    # ── Find minimum cents for significance ───────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  At what price improvement does edge become statistically significant?")
    print(f"{'─'*72}")
    for cents in range(0, 31):
        imp_am  = q["current_odds_am"].apply(lambda o: add_cents_american(o, cents))
        imp_dec = imp_am.apply(american_to_decimal)
        r = compute_scenario(q, str(cents), imp_dec)
        if r["p"] < 0.05:
            print(f"  +{cents} cents: p={r['p']:.4f}  ROI={r['roi_pct']:+.2f}%  "
                  f"CI=[{r['ci_lo']:+.1f}%, {r['ci_hi']:+.1f}%]  --> SIGNIFICANT")
            break

    # ── With positive CLV filter ──────────────────────────────────────────────
    clv_path = Path("data/processed/backtest_clv.csv")
    if clv_path.exists():
        clv_df = pd.read_csv(clv_path)
        clv_df["game_date"] = pd.to_datetime(clv_df["game_date"])
        key = ["game_date","pitcher_id","market","line","best_side"]
        qc  = q.merge(clv_df[key+["clv_pct"]], on=key, how="left")
        pos = qc[qc["clv_pct"] > 0].copy()
        neg = qc[qc["clv_pct"] <= 0].dropna(subset=["clv_pct"]).copy()

        print(f"\n{'─'*72}")
        print(f"  Combined: price improvement  +  positive CLV filter")
        print(f"  (38.7% of bets — these are the ones where market agreed with you)")
        print(f"{'─'*72}")
        print(f"\n  {'Scenario':>12}  {'N':>5}  {'ROI':>8}  {'Win%':>6}  "
              f"{'95% CI':>20}  {'p-val':>7}  {'Sig?'}")
        print(f"  {'-'*75}")
        for cents in [0, 5, 10, 15, 25]:
            for grp, lbl in [(pos, "pos CLV"), (neg, "neg CLV")]:
                imp_am  = grp["current_odds_am"].apply(lambda o: add_cents_american(o, cents))
                imp_dec = imp_am.apply(american_to_decimal)
                r = compute_scenario(grp, f"+{cents}c {lbl}", imp_dec)
                ci_str = f"[{r['ci_lo']:+.1f}%, {r['ci_hi']:+.1f}%]"
                print(f"  {r['label']:>20}  {r['n']:>5}  {r['roi_pct']:>+7.2f}%  "
                      f"{r['win_rate']*100:>5.1f}%  {ci_str:>20}  "
                      f"{r['p']:>7.4f}  {r['sig']}")

    # ── Vig reference ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  REFERENCE: vig at common odds levels")
    print(f"{'─'*72}")
    print(f"  {'Line':>12}  {'Vig':>6}  {'Fair odds':>10}  {'Cents saved'}")
    examples = [(-115, +105), (-110, -110), (-108, -102), (-105, -105)]
    for ov, un in examples:
        d_ov = american_to_decimal(ov); d_un = american_to_decimal(un)
        vig  = (1/d_ov + 1/d_un - 1) * 100
        fair_ov_dec = (1/d_ov + 1/d_un) / (1/d_ov)
        fair_am = (fair_ov_dec - 1) * 100 if fair_ov_dec >= 2 else -(100 / (fair_ov_dec - 1))
        saved = fair_ov_dec - d_ov
        print(f"  {ov:+4d} / {un:+4d}   {vig:>5.2f}%   {fair_am:>+9.1f}  {saved*100:>+8.1f} cents")


if __name__ == "__main__":
    main()
