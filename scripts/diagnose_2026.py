"""Diagnose 2026 out-of-sample performance vs 2025."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def roi_fn(grp):
    if grp.empty: return np.nan
    odds = np.where(grp["best_side"]=="over", grp["over_odds"], grp["under_odds"])
    dec = np.where(odds > 0, 1+odds/100, 1+100/np.abs(np.where(odds==0,1,odds)))
    return float(np.where(grp["won"].astype(bool), dec-1, -1.0).mean())


def resolve_outcome(row, mkt="strikeouts"):
    actual = row.get(mkt)
    if pd.isna(actual): return np.nan
    return 1 if (actual > row["line"] if row["best_side"]=="over" else actual < row["line"]) else 0


def load_year(edges_path, year_label):
    edges = pd.read_csv(edges_path)
    edges["game_date"] = pd.to_datetime(edges["game_date"])
    q = edges[(edges["edge_pct"] >= 3.0) & (edges["edge_pct"] <= 10.0)].copy()
    q = q[q["market"] == "strikeouts"].copy()
    q["abs_gap"] = abs(q["strikeouts_projection"] - q["line"])
    q = q[q["abs_gap"] <= 0.8].copy()
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])
    q["month"] = q["game_date"].dt.month
    q["year"] = year_label
    return q


def main():
    e25 = pd.read_csv("data/processed/backtest_edges.csv")
    e26 = pd.read_csv("data/processed/backtest_2026_edges.csv")
    e25["game_date"] = pd.to_datetime(e25["game_date"])
    e26["game_date"] = pd.to_datetime(e26["game_date"])

    # ── 1. Calibration comparison ─────────────────────────────────────────────
    print("=" * 62)
    print("  DIAGNOSIS: 2025 vs 2026 OUT-OF-SAMPLE")
    print("=" * 62)

    for label, df in [("2025", e25), ("2026", e26)]:
        sk = df[df["market"]=="strikeouts"].dropna(subset=["strikeouts","strikeouts_projection"])
        bias = (sk["strikeouts"] - sk["strikeouts_projection"]).mean()
        print(f"\n{label} calibration:")
        print(f"  Mean actual K:      {sk['strikeouts'].mean():.3f}")
        print(f"  Mean projected K:   {sk['strikeouts_projection'].mean():.3f}")
        print(f"  Mean error (actual-proj): {bias:+.3f}  "
              f"({'model over-predicts' if bias < 0 else 'model under-predicts'})")
        print(f"  Projection-actual r: {sk['strikeouts'].corr(sk['strikeouts_projection']):.3f}")

    # ── 2. Monthly breakdown 2026 ─────────────────────────────────────────────
    q26 = load_year("data/processed/backtest_2026_edges.csv", "2026")
    MONTHS = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep"}
    print(f"\n── 2026 Monthly breakdown (qualifying bets) ──────────────────")
    print(f"  {'Month':>5}  {'N':>5}  {'Win%':>7}  {'ROI':>8}")
    for m in sorted(q26["month"].unique()):
        sub = q26[q26["month"]==m]
        print(f"  {MONTHS.get(m,m):>5}  {len(sub):>5}  {sub['won'].mean()*100:>6.1f}%  {roi_fn(sub)*100:>7.2f}%")

    # ── 3. What does 2026 look like WITHOUT the 2025 bias correction? ─────────
    print(f"\n── Effect of bias correction in 2026 ────────────────────────")
    e26_sk = e26[e26["market"]=="strikeouts"].copy()
    # The bias correction applied was -0.074 (subtracts from projection)
    # Raw projection = projection + 0.074 (undo the correction)
    bias_correction_applied = -0.074
    e26_sk["raw_projection"] = e26_sk["strikeouts_projection"] - bias_correction_applied
    actual_mean = e26_sk["strikeouts"].mean()
    proj_corrected = e26_sk["strikeouts_projection"].mean()
    proj_raw = e26_sk["raw_projection"].mean()
    print(f"  Mean actual 2026 K:          {actual_mean:.3f}")
    print(f"  Mean projection (corrected): {proj_corrected:.3f}  (bias applied: {bias_correction_applied:+.3f})")
    print(f"  Mean projection (raw):       {proj_raw:.3f}")
    print(f"  2026 true bias needed:       {actual_mean - proj_raw:+.3f}")
    print(f"  --> 2025 bias correction is WRONG direction for 2026" if (actual_mean - proj_raw) > 0 else
          f"  --> 2025 bias correction direction OK")

    # ── 4. 2026 qualifying bets breakdown ──────────────────────────────────────
    print(f"\n── 2026 qualifying bets breakdown ───────────────────────────")
    q26_full = e26[(e26["market"]=="strikeouts")].copy()
    q26_full["won"] = q26_full.apply(resolve_outcome, axis=1)
    q26_full = q26_full.dropna(subset=["won"])
    q26_full["abs_gap"] = abs(q26_full["strikeouts_projection"] - q26_full["line"])

    for gap_max in [0.5, 0.8, 1.0, 1.5]:
        for edge_min, edge_max in [(3,10), (2,15), (0,5)]:
            sub = q26_full[
                (q26_full["edge_pct"] >= edge_min) &
                (q26_full["edge_pct"] <= edge_max) &
                (q26_full["abs_gap"] <= gap_max)
            ]
            if len(sub) < 20: continue
            print(f"  gap<={gap_max} edge {edge_min}-{edge_max}%:  "
                  f"n={len(sub):4d}  win={sub['won'].mean():.3f}  roi={roi_fn(sub)*100:+.2f}%")

    # ── 5. 2026 projection accuracy vs 2025 ───────────────────────────────────
    print(f"\n── 2026 prediction accuracy vs 2025 ─────────────────────────")
    for label, df in [("2025 full", e25), ("2026 Apr-May", e26)]:
        sk = df[df["market"]=="strikeouts"].dropna(subset=["strikeouts","strikeouts_projection"])
        mae = abs(sk["strikeouts"] - sk["strikeouts_projection"]).mean()
        r = sk["strikeouts"].corr(sk["strikeouts_projection"])
        print(f"  {label:14s}  n={len(sk):5d}  MAE={mae:.3f}  r={r:.3f}")

    # ── 6. Line distribution shift ────────────────────────────────────────────
    print(f"\n── Line distribution shift ──────────────────────────────────")
    for label, df in [("2025", e25), ("2026", e26)]:
        sk = df[df["market"]=="strikeouts"].dropna(subset=["line"])
        print(f"  {label}: mean line={sk['line'].mean():.2f}  "
              f"p25={sk['line'].quantile(0.25):.1f}  "
              f"p50={sk['line'].median():.1f}  "
              f"p75={sk['line'].quantile(0.75):.1f}")


if __name__ == "__main__":
    main()
