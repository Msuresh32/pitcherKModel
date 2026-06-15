"""Deep-dive analysis: find which bet segments have positive ROI / CLV."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def roi_fn(grp):
    if grp.empty:
        return np.nan
    odds = np.where(grp["best_side"] == "over", grp["over_odds"], grp["under_odds"])
    dec = np.where(odds > 0, 1 + odds / 100, 1 + 100 / np.abs(np.where(odds == 0, 1, odds)))
    profit = np.where(grp["won"].astype(bool), dec - 1, -1.0)
    return float(profit.mean())


def resolve_outcome(row):
    mkt = row["market"]
    actual = row.get(mkt)
    if pd.isna(actual):
        return np.nan
    if row["best_side"] == "over":
        return 1 if actual > row["line"] else 0
    else:
        return 1 if actual < row["line"] else 0


def main():
    edges = pd.read_csv("data/processed/backtest_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])

    q = edges[(edges["edge_pct"] >= 2.0) & (edges["edge_pct"] <= 15.0)].copy()
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])
    print(f"Qualifying bets with outcomes: {len(q)}")

    # --- Projection vs actual correlation ---
    print("\n=== Projection-actual correlation ===")
    for mkt in ["strikeouts", "walks"]:
        sub = edges.dropna(subset=[mkt, f"{mkt}_projection"])
        r = sub[mkt].corr(sub[f"{mkt}_projection"])
        print(f"  {mkt}: r={r:.3f}")

    # --- High-K pitcher strikeouts breakdown ---
    sk = q[q["market"] == "strikeouts"].copy()
    k_col = "p_k_rate_roll5"
    if k_col in sk.columns:
        med = sk[k_col].median()
        print(f"\n=== Strikeouts by K-rate (median={med:.3f}) ===")
        for hi, label in [(True, "high-K"), (False, "low-K")]:
            sub = sk[sk[k_col] > med] if hi else sk[sk[k_col] <= med]
            for side in ["over", "under"]:
                s2 = sub[sub["best_side"] == side]
                if len(s2) > 30:
                    print(f"  {label} {side}: n={len(s2)}  win={s2['won'].mean():.3f}  roi={roi_fn(s2):.3f}")

    # --- Over_probability vs actual win rate ---
    print("\n=== Over_probability calibration ===")
    for mkt in ["strikeouts", "walks"]:
        sub = q[q["market"] == mkt]
        bins = np.arange(0.35, 0.70, 0.05)
        sub2 = sub.copy()
        sub2["p_bin"] = pd.cut(sub2["over_probability"], bins=bins)
        print(f"  {mkt}:")
        print(
            sub2.groupby("p_bin")["won"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "actual_win_rate"})
            .round(3)
            .to_string()
        )

    # --- CLV-based filtering ---
    clv_path = Path("data/processed/backtest_clv.csv")
    if clv_path.exists():
        clv = pd.read_csv(clv_path)
        clv["game_date"] = pd.to_datetime(clv["game_date"])
        key = ["game_date", "pitcher_id", "market", "line", "best_side"]
        merged = q.merge(clv[key + ["clv_pct"]], on=key, how="left")

        print("\n=== Win rate by CLV sign ===")
        for mkt in ["strikeouts", "walks"]:
            sub = merged[merged["market"] == mkt].dropna(subset=["clv_pct"])
            pos = sub[sub["clv_pct"] > 0]
            neg = sub[sub["clv_pct"] <= 0]
            print(f"  {mkt} positive CLV (n={len(pos)}): win={pos['won'].mean():.3f}  roi={roi_fn(pos):.3f}")
            print(f"  {mkt} negative CLV (n={len(neg)}): win={neg['won'].mean():.3f}  roi={roi_fn(neg):.3f}")

        print("\n=== Win rate by CLV threshold ===")
        for mkt in ["strikeouts", "walks"]:
            sub = merged[merged["market"] == mkt].dropna(subset=["clv_pct"])
            for clv_min in [0, 0.5, 1.0, 2.0]:
                filt = sub[sub["clv_pct"] >= clv_min]
                if len(filt) > 20:
                    print(
                        f"  {mkt} CLV>={clv_min}%: n={len(filt)}  "
                        f"win={filt['won'].mean():.3f}  roi={roi_fn(filt):.3f}"
                    )

    # --- Days rest analysis ---
    print("\n=== By days rest ===")
    for mkt in ["strikeouts", "walks"]:
        sub = q[q["market"] == mkt].copy()
        sub["rest_bin"] = pd.cut(sub["days_rest"], bins=[0, 4, 5, 6, 20], labels=["<4d", "4d", "5d", "6d+"])
        print(
            sub.groupby("rest_bin")["won"]
            .agg(lambda s: f"win={s.mean():.3f} n={len(s)}")
            .rename(f"{mkt} by rest")
            .to_string()
        )

    # --- Projection size vs line ---
    print("\n=== By projection gap (proj - line) ===")
    for mkt in ["strikeouts", "walks"]:
        sub = q[q["market"] == mkt].copy()
        proj_col = f"{mkt}_projection"
        if proj_col in sub.columns:
            sub["gap"] = sub[proj_col] - sub["line"]
            sub["gap_bin"] = pd.cut(sub["gap"], bins=5)
            r = sub.groupby("gap_bin")["won"].agg(["mean", "count"]).round(3)
            print(f"  {mkt}:")
            print(r.to_string())


if __name__ == "__main__":
    main()
