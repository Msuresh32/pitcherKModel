"""Validate backtest results: check monthly consistency and robustness."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def roi_fn(grp):
    if grp.empty: return np.nan
    odds = np.where(grp["best_side"]=="over", grp["over_odds"], grp["under_odds"])
    dec = np.where(odds > 0, 1+odds/100, 1+100/np.abs(np.where(odds==0,1,odds)))
    return float(np.where(grp["won"].astype(bool), dec-1, -1.0).mean())


def resolve_outcome(row):
    mkt = row["market"]
    actual = row.get(mkt)
    if pd.isna(actual): return np.nan
    return 1 if (actual > row["line"] if row["best_side"]=="over" else actual < row["line"]) else 0


def main():
    edges = pd.read_csv("data/processed/backtest_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])

    # Apply same filters as the backtest
    q = edges[(edges["edge_pct"] >= 3.0) & (edges["edge_pct"] <= 10.0)].copy()
    q = q[q["market"] == "strikeouts"].copy()

    # Gap filter
    q["abs_gap"] = abs(q["strikeouts_projection"] - q["line"])
    q = q[q["abs_gap"] <= 0.8].copy()

    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])
    q["month"] = q["game_date"].dt.month

    print(f"Total qualifying bets: {len(q)}")
    print(f"Overall: win={q['won'].mean():.3f}  roi={roi_fn(q):.3f}")

    print("\n=== Monthly breakdown ===")
    MONTHS = {4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep'}
    monthly = q.groupby("month").apply(
        lambda g: pd.Series({"n":len(g),"win":g["won"].mean(),"roi":roi_fn(g)})
    ).round(3)
    monthly.index = monthly.index.map(lambda m: MONTHS.get(m, str(m)))
    print(monthly.to_string())

    print("\n=== By best_side ===")
    for side in ["over","under"]:
        sub = q[q["best_side"]==side]
        print(f"  {side}: n={len(sub)}  win={sub['won'].mean():.3f}  roi={roi_fn(sub):.3f}")

    print("\n=== Cumulative P&L by month ===")
    q_sorted = q.sort_values("game_date").copy()
    odds_arr = np.where(q_sorted["best_side"]=="over", q_sorted["over_odds"], q_sorted["under_odds"])
    dec_arr = np.where(odds_arr > 0, 1+odds_arr/100, 1+100/np.abs(np.where(odds_arr==0,1,odds_arr)))
    q_sorted["profit"] = np.where(q_sorted["won"].astype(bool), dec_arr - 1, -1.0)
    q_sorted["cum_profit"] = q_sorted["profit"].cumsum()
    by_month = q_sorted.groupby("month").agg(
        n=("profit","count"),
        monthly_profit=("profit","sum"),
        cum_end=("cum_profit","last"),
    ).round(3)
    by_month.index = by_month.index.map(lambda m: MONTHS.get(m, str(m)))
    print(by_month.to_string())

    print("\n=== Sharpe (monthly returns) ===")
    daily_roi = q_sorted.groupby("game_date")["profit"].sum() / len(q)
    sharpe = float(daily_roi.mean() / daily_roi.std() * np.sqrt(162)) if daily_roi.std() > 0 else 0.0
    print(f"  Sharpe: {sharpe:.3f}")

    print("\n=== Bet distribution ===")
    print(f"  Mean edge:       {q['edge_pct'].mean():.2f}%")
    print(f"  Mean abs gap:    {q['abs_gap'].mean():.3f}")
    print(f"  Mean line:       {q['line'].mean():.2f}")
    print(f"  Mean projection: {q['strikeouts_projection'].mean():.2f}")
    print(f"  Unique pitchers: {q['pitcher_id'].nunique()}")
    print(f"  Unique game dates: {q['game_date'].nunique()}")
    print(f"  Avg bets/day:    {len(q)/q['game_date'].nunique():.1f}")

    # CLV breakdown for these filtered bets
    clv_path = Path("data/processed/backtest_clv.csv")
    if clv_path.exists():
        clv = pd.read_csv(clv_path)
        clv["game_date"] = pd.to_datetime(clv["game_date"])
        key = ["game_date","pitcher_id","market","line","best_side"]
        q2 = q.merge(clv[key+["clv_pct"]], on=key, how="left")
        with_clv = q2.dropna(subset=["clv_pct"])
        print(f"\n=== CLV on filtered bets (n={len(with_clv)}) ===")
        print(f"  Mean CLV: {with_clv['clv_pct'].mean():.3f}%")
        pos = with_clv[with_clv["clv_pct"] > 0]
        neg = with_clv[with_clv["clv_pct"] <= 0]
        print(f"  Positive CLV (n={len(pos)}): win={pos['won'].mean():.3f}  roi={roi_fn(pos):.3f}")
        print(f"  Negative CLV (n={len(neg)}): win={neg['won'].mean():.3f}  roi={roi_fn(neg):.3f}")
        print(f"  CLV>0 rate: {(with_clv['clv_pct']>0).mean():.3f}")


if __name__ == "__main__":
    main()
