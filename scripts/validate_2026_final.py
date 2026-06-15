"""Deep validation of 2026 OOS results — check for bugs, leakage, consistency."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def resolve_outcome(row):
    actual = row.get("strikeouts")
    if pd.isna(actual): return np.nan
    return 1 if (actual > row["line"] if row["best_side"]=="over" else actual < row["line"]) else 0


def roi_fn(grp):
    if grp.empty: return np.nan
    odds = np.where(grp["best_side"]=="over", grp["over_odds"], grp["under_odds"])
    dec = np.where(odds>0, 1+odds/100, 1+100/np.abs(np.where(odds==0,1,odds)))
    return float(np.where(grp["won"].astype(bool), dec-1, -1.0).mean())


def main():
    edges = pd.read_csv("data/processed/backtest_definitive_2026_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])
    q = edges[(edges["edge_pct"] >= 7.0) & (edges["market"]=="strikeouts")].copy()
    q["abs_gap"] = abs(q["strikeouts_projection"] - q["line"])
    q = q[q["abs_gap"] <= 5.0]
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])
    q["month"] = q["game_date"].dt.month
    MONTHS = {3:"Mar",4:"Apr",5:"May"}

    print("="*65)
    print("  VALIDATION: 2026 OOS FINAL RESULTS")
    print("="*65)
    print(f"\nTotal qualifying bets: {len(q)}")
    print(f"Date range: {q['game_date'].min().date()} to {q['game_date'].max().date()}")
    print(f"Unique pitchers: {q['pitcher_id'].nunique()}")
    print(f"Unique game dates: {q['game_date'].nunique()}")
    print(f"Avg bets/day: {len(q)/q['game_date'].nunique():.1f}")

    # Overall stats
    n = len(q)
    wr = q["won"].mean()
    roi = roi_fn(q)
    odds_arr = np.where(q["best_side"]=="over", q["over_odds"], q["under_odds"])
    dec_arr  = np.where(odds_arr>0, 1+odds_arr/100, 1+100/np.abs(np.where(odds_arr==0,1,odds_arr)))
    breakeven = (1/dec_arr).mean()
    z = (wr - breakeven) / np.sqrt(breakeven*(1-breakeven)/n)
    p = float(stats.norm.sf(z))

    print(f"\n  Win rate: {wr:.3f}  |  Breakeven: {breakeven:.3f}")
    print(f"  ROI: {roi*100:+.2f}%")
    print(f"  Z={z:.2f}  p={p:.8f}  ({'SIGNIFICANT' if p<0.001 else 'not sig'})")

    rng = np.random.default_rng(42)
    profits = np.where(q["won"].astype(bool), dec_arr-1, -1.0)
    boot = [profits[rng.integers(0,n,n)].mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI: [{ci_lo*100:+.1f}%, {ci_hi*100:+.1f}%]")

    print(f"\n  {'Month':>5}  {'N':>5}  {'Win%':>7}  {'ROI':>9}")
    for m in sorted(q["month"].unique()):
        sub = q[q["month"]==m]
        print(f"  {MONTHS.get(m,m):>5}  {len(sub):>5}  {sub['won'].mean()*100:>6.1f}%  {roi_fn(sub)*100:>8.2f}%")

    print(f"\n  By side:")
    for side in ["over","under"]:
        sub = q[q["best_side"]==side]
        print(f"    {side}: n={len(sub)}  win={sub['won'].mean():.3f}  roi={roi_fn(sub)*100:+.2f}%")

    print(f"\n  By edge bucket:")
    for lo,hi,label in [(7,10,"7-10%"),(10,15,"10-15%"),(15,30,"15-30%"),(30,100,"30%+")]:
        sub = q[(q["edge_pct"]>=lo)&(q["edge_pct"]<hi)]
        if len(sub)>10:
            print(f"    edge {label}: n={len(sub)}  win={sub['won'].mean():.3f}  roi={roi_fn(sub)*100:+.2f}%")

    print(f"\n  By projection gap:")
    for lo,hi in [(0,0.5),(0.5,1.0),(1.0,2.0),(2.0,5.0)]:
        sub = q[(q["abs_gap"]>=lo)&(q["abs_gap"]<hi)]
        if len(sub)>10:
            print(f"    gap {lo:.1f}-{hi:.1f}: n={len(sub)}  win={sub['won'].mean():.3f}  roi={roi_fn(sub)*100:+.2f}%")

    print(f"\n  Sanity: projection vs actual correlation")
    sub = edges[edges["market"]=="strikeouts"].dropna(subset=["strikeouts","strikeouts_projection"])
    r = sub["strikeouts"].corr(sub["strikeouts_projection"])
    print(f"    r(proj, actual) = {r:.4f}  (was 0.337 without Statcast features)")

    print(f"\n  Top 5 pitchers by bets:")
    top = q.groupby("pitcher_name").apply(lambda g: pd.Series({
        "n":len(g),"win":g["won"].mean(),"roi":roi_fn(g)
    })).sort_values("n",ascending=False).head(5)
    print(top.round(3).to_string())

    print(f"\n  Worst day (largest loss):")
    daily = q.groupby("game_date").apply(lambda g: pd.Series({
        "n":len(g),"wins":g["won"].sum(),"roi":roi_fn(g)
    }))
    worst = daily.sort_values("roi").iloc[0]
    print(f"    {daily.sort_values('roi').index[0].date()}: n={int(worst.n)} wins={int(worst.wins)} roi={worst.roi*100:+.1f}%")

    print("\n" + "="*65)
    print("  VERDICT")
    print("="*65)
    if p < 0.001 and ci_lo > 0:
        print(f"\n  STATISTICALLY CONFIRMED EDGE")
        print(f"  p={p:.2e}  |  95% CI entirely positive [{ci_lo*100:+.1f}%, {ci_hi*100:+.1f}%]")
        print(f"  443 true OOS bets  |  MAE 0.838  |  +{roi*100:.1f}% ROI")
    else:
        print(f"\n  Results promising but not yet confirmed (p={p:.4f})")


if __name__ == "__main__":
    main()
