"""
Monthly expanding-window walk-forward for 2026.

Period 1: Mar 26 – Apr 30  → model trained through Dec 2025 (models_backup_2025)
Period 2: May 1  – May 31  → model trained through Apr 2026 (processed_apr2026/models)
Period 3: Jun 1  – Jun 16  → model trained through May 2026 (current models)

Each period uses predictions from the model that existed BEFORE that period started.
Threshold = edge >= 15% (frozen from 2025 selection period).
Odds = DraftKings only (full_2026_odds_dk.csv).
"""
import subprocess, sys, pandas as pd, numpy as np
from pathlib import Path
from scipy import stats

ODDS      = "data/odds/full_2026_odds_dk.csv"
MIN_EDGE  = 15.0
PREFIX    = "wf2026"

PERIODS = [
    {
        "label":       "Mar 26 – Apr 30  (Dec-2025 model)",
        "pred_file":   "data/processed/exp_pred_bk_20260101_20260616_b70_30_predictions.csv",
        "start":       "2026-03-26",
        "end":         "2026-04-30",
        "output_pfx":  f"{PREFIX}_p1_mar_apr",
        "output_dir":  "data/processed",
        "config":      "config/config.yaml",
    },
    {
        "label":       "May 1  – May 31  (Apr-2026 model)",
        "pred_file":   None,
        "start":       "2026-05-01",
        "end":         "2026-05-31",
        "output_pfx":  f"{PREFIX}_p2_may",
        "output_dir":  "data/processed_apr2026",
        "config":      "config/config_apr2026.yaml",
    },
    {
        "label":       "Jun 1  – Jun 16  (May-2026 model)",
        "pred_file":   "data/processed/exp_pred_new_20260601_20260616_b70_30_predictions.csv",
        "start":       "2026-06-01",
        "end":         "2026-06-16",
        "output_pfx":  f"{PREFIX}_p3_jun",
        "output_dir":  "data/processed",
        "config":      "config/config.yaml",
    },
]


def run_backtest(period):
    cmd = [sys.executable, "scripts/backtest.py",
           "--config",  period["config"],
           "--start",   period["start"],
           "--end",     period["end"],
           "--odds",    ODDS,
           "--min-edge", str(MIN_EDGE),
           "--output-prefix", period["output_pfx"]]
    if period["pred_file"]:
        cmd += ["--predictions-file", period["pred_file"]]
    print(f"\nRunning: {' '.join(cmd[-6:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout
    print(tail)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1000:])
        raise RuntimeError(f"Backtest failed for {period['label']}")


def load_edges(output_pfx, output_dir="data/processed"):
    path = f"{output_dir}/{output_pfx}_edges.csv"
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    df["won"]    = df.apply(lambda r: (r["strikeouts"] > r["line"])
                            if r["best_side"]=="over" else (r["strikeouts"] < r["line"]), axis=1)
    df["payout"] = df.apply(lambda r:
        (r["over_odds"]/100 if r["over_odds"]>0 else 100/abs(r["over_odds"]))
        if r["best_side"]=="over"
        else (r["under_odds"]/100 if r["under_odds"]>0 else 100/abs(r["under_odds"])), axis=1)
    df["profit"] = df.apply(lambda r: r["payout"] if r["won"] else -1.0, axis=1)
    df["gap"]    = df["strikeouts_projection"] - df["line"]
    df = (df.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))
    return df[df["edge_pct"] >= MIN_EDGE].copy()


def print_results(label, df, clv_df=None):
    n      = len(df)
    wins   = df["won"].sum()
    roi    = df["profit"].mean()
    sharpe = roi / df["profit"].std() * n**0.5 if df["profit"].std() > 0 else 0

    clv_mean = np.nan
    if clv_df is not None and "clv_pct" in clv_df.columns:
        c = clv_df["clv_pct"].dropna()
        clv_mean = c.mean() if len(c) > 0 else np.nan

    print(f"  {label}")
    print(f"    Bets: {n:>4}  Win: {wins/n:.1%}  ROI: {roi:+.1%}  "
          f"Sharpe: {sharpe:.2f}  CLV: {f'{clv_mean:+.2f}%' if not np.isnan(clv_mean) else 'N/A'}")

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"]     = df["game_date"].dt.to_period("M")
    for m, g in df.groupby("month"):
        print(f"      {str(m):>8}  {len(g):>3} bets  "
              f"{g['won'].mean():.1%} win  {g['profit'].mean():+.1%} ROI")
    return {"n": n, "win": wins/n, "roi": roi, "sharpe": sharpe, "df": df}


if __name__ == "__main__":
    print("=" * 65)
    print("  MONTHLY WALK-FORWARD VALIDATION — 2026")
    print("  Each month's model was trained BEFORE that month began")
    print("=" * 65)

    # Run backtests for each period
    for p in PERIODS:
        edges_path = Path(f"data/processed/{p['output_pfx']}_edges.csv")
        if edges_path.exists():
            print(f"\n[skip] {p['label']} — edges already exist")
        else:
            run_backtest(p)

    # Load and combine
    print("\n" + "=" * 65)
    print("  RESULTS BY PERIOD")
    print("=" * 65)

    dfs, stats_list = [], []
    for p in PERIODS:
        df  = load_edges(p["output_pfx"], p["output_dir"])
        clv = pd.read_csv(f"{p['output_dir']}/{p['output_pfx']}_clv.csv")
        r   = print_results(p["label"], df, clv)
        stats_list.append(r)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    print("\n" + "=" * 65)
    print("  COMBINED WALK-FORWARD 2026")
    print("=" * 65)
    n      = len(combined)
    roi    = combined["profit"].mean()
    sharpe = roi / combined["profit"].std() * n**0.5

    # Significance
    np.random.seed(42)
    implied = []
    for _, bet in combined.iterrows():
        o = bet["over_odds"] if bet["best_side"]=="over" else bet["under_odds"]
        implied.append(abs(o)/(abs(o)+100) if o<0 else 100/(o+100))
    bep   = np.mean(implied)
    t_s, p2 = stats.ttest_1samp(combined["profit"].values, 0)
    p_t   = p2 / 2
    p_b   = stats.binom_test(int(combined["won"].sum()), n, bep, alternative="greater")
    boot  = [np.random.choice(combined["profit"].values, n, replace=True).mean()
             for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"  Bets:      {n}")
    print(f"  Win rate:  {combined['won'].mean():.1%}")
    print(f"  ROI:       {roi:+.1%}")
    print(f"  Sharpe:    {sharpe:.2f}")
    print(f"  p-value:   {p_t:.4f} (t-test)  {p_b:.4f} (binomial)")
    print(f"  95% CI:    [{lo:+.2%}, {hi:+.2%}]")
    print(f"  Sig at 5%: {'YES' if p_t < 0.05 else 'NO'}")

    print("\n  Monthly breakdown (combined):")
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined["month"]     = combined["game_date"].dt.to_period("M")
    for m, g in combined.groupby("month"):
        print(f"    {str(m):>8}  {len(g):>3} bets  "
              f"{g['won'].mean():.1%} win  {g['profit'].mean():+.1%} ROI")

    # Save combined for reference
    combined.to_csv("data/processed/walkforward_2026_combined.csv", index=False)
    print("\n  Saved -> data/processed/walkforward_2026_combined.csv")
