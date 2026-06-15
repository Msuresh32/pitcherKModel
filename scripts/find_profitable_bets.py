"""Identify characteristics of profitable vs unprofitable bets in backtest."""
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
    clv = pd.read_csv("data/processed/backtest_clv.csv")
    clv["game_date"] = pd.to_datetime(clv["game_date"])

    q = edges[(edges["edge_pct"] >= 2.0) & (edges["edge_pct"] <= 15.0)].copy()
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])

    key = ["game_date","pitcher_id","market","line","best_side"]
    merged = q.merge(clv[key+["clv_pct"]], on=key, how="left")

    sk = merged[merged["market"]=="strikeouts"].copy()

    # Is our projected K rate deviation from career norm predictive?
    career_col = "p_strikeouts_career_avg_prior"
    recent_col = "p_strikeouts_roll5"
    if career_col in sk.columns and recent_col in sk.columns:
        sk["recent_vs_career"] = sk[recent_col] - sk[career_col]
        sk["declining"] = sk["recent_vs_career"] < -0.3
        print("=== Recent K vs career K — declining pitchers ===")
        for dec_flag, label in [(True,"declining"),(False,"stable/rising")]:
            sub = sk[sk["declining"]==dec_flag]
            print(f"  {label} (n={len(sub)}): win={sub['won'].mean():.3f}  roi={roi_fn(sub):.3f}")
            # Their CLV?
            sub_clv = sub.dropna(subset=["clv_pct"])
            if len(sub_clv) > 0:
                print(f"    CLV: mean={sub_clv['clv_pct'].mean():.3f}  pos_rate={( sub_clv['clv_pct']>0).mean():.3f}")

    # Velocity slope analysis
    velo_col = "sc_velocity_slope_roll8"
    if velo_col in sk.columns:
        sk["velo_declining"] = sk[velo_col] < -0.5
        print("\n=== Velocity trend — declining (slope < -0.5 mph/start) ===")
        for dec_flag, label in [(True,"velo declining"),(False,"velo stable")]:
            sub = sk[sk["velo_declining"]==dec_flag]
            if len(sub) > 30:
                print(f"  {label} (n={len(sub)}): win={sub['won'].mean():.3f}  roi={roi_fn(sub):.3f}")

    # Games where projection is FAR from the line — high conviction
    sk["abs_gap"] = abs(sk["strikeouts_projection"] - sk["line"])
    sk["high_conviction"] = sk["abs_gap"] > 1.0
    print("\n=== High conviction bets (|proj - line| > 1) ===")
    for hc, label in [(True,"high conviction"),(False,"near line")]:
        sub = sk[sk["high_conviction"]==hc]
        if len(sub) > 30:
            print(f"  {label} (n={len(sub)}): win={sub['won'].mean():.3f}  roi={roi_fn(sub):.3f}")
            sub_clv = sub.dropna(subset=["clv_pct"])
            if len(sub_clv) > 0:
                print(f"    CLV: mean={sub_clv['clv_pct'].mean():.3f}  pos_rate={( sub_clv['clv_pct']>0).mean():.3f}")

    # The actual best performers — describe them
    print("\n=== Top 20 most profitable pitchers (strikeouts bets) ===")
    pitcher_roi = sk.groupby("pitcher_name").apply(lambda g: pd.Series({
        "n": len(g),
        "win": g["won"].mean(),
        "roi": roi_fn(g),
        "avg_k_rate": g["p_k_rate_roll5"].mean() if "p_k_rate_roll5" in g else None,
        "avg_clv": g["clv_pct"].mean() if "clv_pct" in g.columns else None,
    }))
    pitcher_roi = pitcher_roi[pitcher_roi["n"] >= 10].sort_values("roi", ascending=False)
    print(pitcher_roi.head(20).round(3).to_string())

    print("\n=== Bottom 10 least profitable pitchers ===")
    print(pitcher_roi.tail(10).round(3).to_string())

    # Highlight: pitchers with positive CLV
    print("\n=== CLV > 1% bets: what do they look like? ===")
    pos_clv = sk[sk["clv_pct"] > 1].dropna(subset=["clv_pct"])
    neg_clv = sk[sk["clv_pct"] < -1].dropna(subset=["clv_pct"])
    feat_compare = ["edge_pct","over_probability","strikeouts_projection","line","days_rest",
                    "p_k_rate_roll5","sc_csw_rate_roll5","opp_batting_k_rate_roll5"]
    feat_compare = [f for f in feat_compare if f in sk.columns]
    print(f"\nPositive CLV bets (n={len(pos_clv)}) vs Negative CLV (n={len(neg_clv)}):")
    for col in feat_compare:
        pos_mean = pos_clv[col].mean()
        neg_mean = neg_clv[col].mean()
        diff = pos_mean - neg_mean
        print(f"  {col:45s}  pos={pos_mean:.3f}  neg={neg_mean:.3f}  diff={diff:+.3f}")


if __name__ == "__main__":
    main()
