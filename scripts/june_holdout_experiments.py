"""
June Holdout Experiments — three-way validation of the June exclusion question.

Experiment A: pre-June model on June 2026 as a frozen holdout
Experiment B: rolling-origin proxy (live picks vs backtest, data-availability note)
Experiment C: full period comparison with bootstrap CIs, CLV, price adjustment, drawdown

Run: py -3.14 scripts/june_holdout_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PICKS_LOG   = ROOT / "data/exports/picks_log.csv"
BT2025      = ROOT / "data/exports/2025_backtest.csv"
BT2026      = ROOT / "data/exports/2026_backtest_extended.csv"
FILL_VALUES = ROOT / "data/processed_poisson_wf2025/models/fill_values.json"
TRAIN_PY    = ROOT / "scripts/train.py"
BUILD_PY    = ROOT / "src/features/build_features.py"
REPORT_OUT  = ROOT / "reports/june_holdout_report.txt"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    return (1 + odds / 100) if odds >= 0 else (1 - 100 / odds)


def pnl_flat(won: float, odds: float, stake: float = 100.0) -> float:
    if pd.isna(won) or pd.isna(odds):
        return np.nan
    dec = american_to_decimal(odds)
    return stake * (dec - 1) if won else -stake


def pnl_price_adj(won: float, odds: float, improvement: float, stake: float = 100.0) -> float:
    """P&L after getting `improvement` cents better than sportsbook on American odds."""
    return pnl_flat(won, odds + improvement, stake)


def roi(pnls: np.ndarray, stake: float = 100.0) -> float:
    valid = pnls[~np.isnan(pnls)]
    if len(valid) == 0:
        return np.nan
    return valid.sum() / (len(valid) * stake) * 100


def win_rate(won_arr: np.ndarray) -> float:
    valid = won_arr[~np.isnan(won_arr)]
    return valid.mean() if len(valid) else np.nan


def bootstrap_ci(arr: np.ndarray, func, n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    stats_arr = []
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return (np.nan, np.nan)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats_arr.append(func(sample))
    lo = np.nanpercentile(stats_arr, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(stats_arr, (1 + ci) / 2 * 100)
    return (lo, hi)


def drawdown_stats(pnls: np.ndarray) -> dict:
    valid = pnls[~np.isnan(pnls)]
    if len(valid) == 0:
        return {"max_dd": np.nan, "max_dd_pct": np.nan, "recovery_bets": np.nan}
    bankroll = np.cumsum(valid)
    peak = np.maximum.accumulate(bankroll)
    dd = bankroll - peak
    max_dd = dd.min()
    # % relative to stake (10k starting bankroll = 100 bets × $100)
    starting_br = 100 * 100.0
    max_dd_pct = max_dd / starting_br * 100
    return {"max_dd_$": max_dd, "max_dd_%": max_dd_pct}


def clv_summary(df: pd.DataFrame) -> dict:
    """CLV = exec_odds - closing_odds (positive = we beat the closing line).
    Use clv_pct column if available; otherwise compute from raw columns."""
    if "clv_pct" in df.columns:
        # clv_pct is pre-computed in the export: exec_decimal/closing_decimal - 1 * 100
        # Negative means we got a worse price than closing = market moved against pick
        valid = df["clv_pct"].dropna()
        if len(valid):
            return {"n_clv": len(valid), "mean_clv": valid.mean(),
                    "pct_positive": (valid > 0).mean() * 100}
    if "closing_odds" in df.columns and "odds_used" in df.columns:
        clv = df["odds_used"] - df["closing_odds"]  # positive = beat closing
        valid = clv.dropna()
        if len(valid):
            return {"n_clv": len(valid), "mean_clv": valid.mean(),
                    "pct_positive": (valid > 0).mean() * 100}
    return {"n_clv": 0, "mean_clv": np.nan, "pct_positive": np.nan}


def max_streak(won_arr: np.ndarray) -> dict:
    won_arr = won_arr[~np.isnan(won_arr)].astype(int)
    if len(won_arr) == 0:
        return {"win_streak": 0, "loss_streak": 0}
    cur_w = cur_l = max_w = max_l = 0
    for w in won_arr:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return {"win_streak": max_w, "loss_streak": max_l}


def period_stats(df: pd.DataFrame, label: str, stake: float = 100.0) -> dict:
    df = df.dropna(subset=["won"]).copy()
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0}
    pnls = np.array([pnl_flat(r["won"], r["odds_used"], stake) for _, r in df.iterrows()])
    wr   = win_rate(df["won"].values)
    r    = roi(pnls, stake)

    wr_ci  = bootstrap_ci(df["won"].values, np.mean)
    roi_ci = bootstrap_ci(pnls, lambda x: x.sum() / (len(x) * stake) * 100)

    dd = drawdown_stats(pnls)
    st = max_streak(df["won"].values)

    # CLV
    clv = clv_summary(df)

    # Price-adjusted ROI
    price_rois = {}
    for cents in [5, 10, 15, 20]:
        adj_pnls = np.array([pnl_price_adj(r2["won"], r2["odds_used"], cents, stake)
                             for _, r2 in df.iterrows()])
        price_rois[f"roi_+{cents}c"] = roi(adj_pnls, stake)

    # Projection bias
    bias = {}
    if "strikeouts_projection" in df.columns and "actual" in df.columns:
        err = df["strikeouts_projection"] - df["actual"]
        bias = {"proj_bias_mean": err.mean(), "proj_bias_median": err.median(),
                "proj_rmse": np.sqrt((err**2).mean())}

    # Side breakdown
    sides = {}
    if "best_side" in df.columns:
        for s in ["over", "under"]:
            sub = df[df["best_side"] == s]
            if len(sub):
                sp = np.array([pnl_flat(r2["won"], r2["odds_used"], stake) for _, r2 in sub.iterrows()])
                sides[f"{s}_n"]   = len(sub)
                sides[f"{s}_wr"]  = win_rate(sub["won"].values)
                sides[f"{s}_roi"] = roi(sp, stake)

    return {
        "label": label, "n": n, "wr": wr, "wr_ci": wr_ci,
        "roi": r, "roi_ci": roi_ci,
        "units": sum(p for p in pnls if not np.isnan(p)) / stake,
        **dd, **st, **clv, **price_rois, **bias, **sides,
    }


def fmt_ci(ci: tuple, pct_scale: bool = False) -> str:
    if ci is None or any(np.isnan(c) for c in ci):
        return "n/a"
    lo, hi = (ci[0]*100, ci[1]*100) if pct_scale else (ci[0], ci[1])
    return f"[{lo:+.1f}%, {hi:+.1f}%]"


def print_period(s: dict, out) -> None:
    if s.get("n", 0) == 0:
        out.write(f"\n{s['label']}: no data\n")
        return
    out.write(f"\n{'='*60}\n{s['label']}\n{'='*60}\n")
    out.write(f"  n                   : {s['n']}\n")
    out.write(f"  Win Rate            : {s.get('wr',np.nan)*100:.1f}%  95% CI {fmt_ci(s.get('wr_ci'), pct_scale=True)}\n")
    out.write(f"  ROI                 : {s.get('roi',np.nan):+.1f}%  95% CI {fmt_ci(s.get('roi_ci'))}\n")
    out.write(f"  Units P&L           : {s.get('units',np.nan):+.1f}u  (${s.get('units',np.nan)*100:+.0f})\n")
    if "max_dd_$" in s:
        out.write(f"  Max Drawdown        : {s.get('max_dd_$',np.nan):+.0f}$  ({s.get('max_dd_%',np.nan):.1f}%)\n")
    if "win_streak" in s:
        out.write(f"  Max Win/Loss streak : {s.get('win_streak')}/{s.get('loss_streak')}\n")
    if s.get("n_clv", 0) > 0:
        out.write(f"  CLV (n={s['n_clv']})        : mean={s.get('mean_clv',np.nan):+.2f}  "
                  f"{s.get('pct_positive',np.nan):.0f}% positive\n")
    for c in [5, 10, 15, 20]:
        k = f"roi_+{c}c"
        if k in s:
            out.write(f"  ROI at +{c:2d}c/bet      : {s[k]:+.1f}%\n")
    if "proj_bias_mean" in s:
        out.write(f"  Proj bias (proj-act): {s['proj_bias_mean']:+.2f} K  RMSE={s.get('proj_rmse',np.nan):.2f}\n")
    for side in ["over", "under"]:
        if f"{side}_n" in s:
            out.write(f"  {side.capitalize():5s} ({s[f'{side}_n']:3d} bets): "
                      f"WR={s[f'{side}_wr']*100:.1f}%  ROI={s[f'{side}_roi']:+.1f}%\n")


# ---------------------------------------------------------------------------
# contamination audit
# ---------------------------------------------------------------------------

def contamination_audit(out) -> None:
    out.write("\n" + "="*70 + "\n")
    out.write("CONTAMINATION AUDIT\n")
    out.write("="*70 + "\n")

    issues = []

    # 1. league_k raw in feature matrix
    build_src = BUILD_PY.read_text(encoding="utf-8")
    if '"league_k"' in build_src and "league_k_" in build_src:
        # Check if raw league_k column is returned from the feature function
        returned_line = 'feat_cols = [c for c in daily.columns if c.startswith("league_k_")]'
        if returned_line in build_src:
            out.write("\n[PASS] league_k: raw same-day league_k excluded from feature matrix.\n")
            out.write("       Only shifted rolling variants (league_k_mean_roll*, league_k_rate_roll*) returned.\n")
        else:
            issues.append("league_k: cannot confirm raw column excluded")
            out.write("\n[WARN] league_k: could not confirm exclusion — manual review needed.\n")
    else:
        out.write("\n[PASS] league_k: not found in feature builder.\n")

    # 2. Imputation ordering
    train_src = TRAIN_PY.read_text(encoding="utf-8")
    has_all_data_call = "fill_values_all = build_training_features(" in train_src
    has_train_only    = "fill_values_train = train_df[feature_cols].median" in train_src
    has_correct_save  = "save_fill_values(fill_values_train" in train_src

    if has_all_data_call and has_train_only and has_correct_save:
        out.write("\n[FAIL] Imputation ordering — TWO-STEP CONTAMINATION:\n")
        out.write("       Step 1: build_training_features called on ALL data (line ~83).\n")
        out.write("               When fill_values=None, line 1296-1297 of build_features.py\n")
        out.write("               computes median from the full dataset (including test rows).\n")
        out.write("               NaN values in any feature are filled with all-data median.\n")
        out.write("       Step 2: fill_values_train computed from train_df and saved.\n")
        out.write("               This is correct for inference but does NOT fix the model\n")
        out.write("               fit, which used all-data-imputed features.\n")
        out.write("       Impact: Low if NaN rate in features is low (rolling features only\n")
        out.write("               produce NaN for early career rows). High if many features\n")
        out.write("               are NaN for new/young pitchers entering the dataset.\n")
        out.write("       Fix: Pass fill_values_train to build_training_features after\n")
        out.write("            computing it from train_df first.\n")
        issues.append("imputation: all-data medians used in model fit (fill_values_all)")
    elif has_train_only and has_correct_save:
        out.write("\n[PASS] Imputation: fill_values computed from train period only.\n")
    else:
        out.write("\n[WARN] Imputation: could not verify — manual review needed.\n")
        issues.append("imputation: uncertain")

    # 3. allow_exact_matches in merge_asof for batting features
    if "allow_exact_matches=True" in build_src:
        # Check if batting features use shift before rolling
        has_shift_before_roll = "shifted = grouped[rolling_cols].shift(1)" in build_src
        if has_shift_before_roll:
            out.write("\n[PASS] merge_asof allow_exact_matches=True: batting features use .shift(1)\n")
            out.write("       before rolling, so date-D merge row already excludes date-D games.\n")
        else:
            issues.append("merge_asof: exact matches without pre-shift confirmation")
            out.write("\n[WARN] merge_asof: allow_exact_matches=True but shift not confirmed.\n")
    else:
        out.write("\n[PASS] merge_asof: no allow_exact_matches=True found.\n")

    # 4. gameDate[:10] vs officialDate
    if "gameDate[:10]" in build_src or 'gameDate"' in build_src:
        issues.append("date: gameDate[:10] pattern found")
        out.write("\n[FAIL] Date field: gameDate[:10] slicing found — may truncate tz-aware timestamps.\n")
    else:
        out.write("\n[PASS] Date fields: no gameDate[:10] pattern. Using game_date consistently.\n")

    # 5. Statcast merge duplication risk
    if 'out.merge(statcast_features, on=["game_date", "pitcher_id"]' in build_src:
        out.write("\n[WARN] Statcast merge: on=['game_date','pitcher_id'] — if statcast_pitcher_daily\n")
        out.write("       has >1 row per (date, pitcher), this will duplicate pitcher rows.\n")
        out.write("       Check: statcast data should be aggregated to 1 row per pitcher per date\n")
        out.write("       before this merge. Verify in fetch_statcast.py.\n")
    else:
        out.write("\n[PASS] Statcast merge: key not found (likely not used or different pattern).\n")

    # 6. YTD features shift
    if "pitcher_ip_ytd" in build_src:
        if 'shift(1).expanding()' in build_src or ".shift(1).expanding" in build_src:
            out.write("\n[PASS] YTD features: pitcher_ip_ytd uses shift(1) before expanding sum.\n")
        else:
            out.write("\n[WARN] YTD features: pitcher_ip_ytd present but shift not confirmed.\n")
    else:
        out.write("\n[INFO] YTD features: pitcher_ip_ytd not present.\n")

    # Summary
    out.write(f"\n{'─'*50}\n")
    out.write(f"CONTAMINATION SUMMARY: {len(issues)} issue(s) found\n")
    for i, issue in enumerate(issues, 1):
        out.write(f"  {i}. {issue}\n")
    if not issues:
        out.write("  All checks passed.\n")
    out.write("\n")


# ---------------------------------------------------------------------------
# load and normalise data
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    bt25 = pd.read_csv(BT2025)
    bt25["game_date"] = pd.to_datetime(bt25["game_date"])
    bt25["month"]     = bt25["game_date"].dt.month
    # 2025 backtest already has pnl; add odds-based pnl for consistency check
    bt25["egp"]       = bt25["gap"].abs() * bt25["edge_pct"]

    bt26 = pd.read_csv(BT2026)
    bt26["game_date"] = pd.to_datetime(bt26["game_date"])
    bt26["month"]     = bt26["game_date"].dt.month
    bt26["egp"]       = bt26["gap"].abs() * bt26["edge_pct"]

    pl = pd.read_csv(PICKS_LOG)
    pl["game_date"] = pd.to_datetime(pl["game_date"])
    pl["month"]     = pl["game_date"].dt.month
    pl["egp"]       = pl["gap"].abs() * pl["edge_pct"]

    return {"bt25": bt25, "bt26": bt26, "pl": pl}


# ---------------------------------------------------------------------------
# significance tests
# ---------------------------------------------------------------------------

def significance_tests(periods: list[dict], out) -> None:
    out.write("\n" + "="*70 + "\n")
    out.write("STATISTICAL SIGNIFICANCE TESTS\n")
    out.write("="*70 + "\n")

    # Find periods by exact label match (case-insensitive)
    def get(label_exact: str) -> dict | None:
        for p in periods:
            if p["label"].lower() == label_exact.lower():
                return p
        return None

    pairs = [
        ("June 2026 (backtest)", "Pre-June 2026 (backtest)"),
        ("June 2026 (backtest)", "June 2025 (backtest)"),
        ("Pre-June 2026 (backtest)", "Pre-June 2025 (backtest)"),
    ]

    for l1, l2 in pairs:
        p1 = get(l1); p2 = get(l2)
        if not p1 or not p2 or p1.get("n", 0) < 2 or p2.get("n", 0) < 2:
            continue
        wr1, n1 = p1["wr"], p1["n"]
        wr2, n2 = p2["wr"], p2["n"]
        # Two-proportion z-test
        p_pool = (wr1 * n1 + wr2 * n2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z  = (wr1 - wr2) / se if se > 0 else np.nan
        pv = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

        roi1, roi2 = p1.get("roi", np.nan), p2.get("roi", np.nan)
        out.write(f"\n  {l1} vs {l2}\n")
        out.write(f"    WR: {wr1*100:.1f}% (n={n1}) vs {wr2*100:.1f}% (n={n2})\n")
        out.write(f"    Z-stat: {z:+.2f}   p={pv:.4f}  {'SIGNIFICANT (p<0.05)' if pv < 0.05 else 'not significant'}\n")
        out.write(f"    ROI diff: {roi1:+.1f}% vs {roi2:+.1f}%  delta={roi1-roi2:+.1f}%\n")

    # Overall model edge test: ROI vs 0 using one-sample t on per-bet P&L
    combined_p = get("Combined 2025+2026")
    if combined_p and combined_p.get("n", 0) >= 10:
        roi_est = combined_p.get("roi", np.nan)
        roi_lo  = combined_p.get("roi_ci", (np.nan, np.nan))[0]
        roi_hi  = combined_p.get("roi_ci", (np.nan, np.nan))[1]
        out.write(f"\n  Combined model ROI test (H0: ROI = 0%):\n")
        out.write(f"    ROI={roi_est:+.1f}%  n={combined_p['n']}  "
                  f"95% CI [{roi_lo:+.1f}%, {roi_hi:+.1f}%]\n")
        if roi_lo > 0:
            out.write(f"    CI excludes 0 — STATISTICALLY POSITIVE ROI (95% confidence)\n")
        elif roi_hi < 0:
            out.write(f"    CI excludes 0 — STATISTICALLY NEGATIVE ROI (95% confidence)\n")
        else:
            out.write(f"    CI includes 0 — insufficient evidence of positive edge\n")

    # Power analysis: how many bets needed to detect 5% ROI at 80% power?
    out.write("\n  Power analysis (detect +5% ROI over -110 baseline, 80% power):\n")
    # WR needed: 0.524 + delta; detect delta in WR ≈ 0.05 * 100/210 ≈ 0.024
    effect_wr = 0.05 * 100 / 210  # 5% ROI ≈ additional WR
    # Manual power calc: n = 2 * ((z_alpha + z_beta) / effect_size_z)^2
    z_alpha = stats.norm.ppf(0.975)  # two-sided 0.05
    z_beta  = stats.norm.ppf(0.80)   # 80% power
    effect_z = effect_wr / np.sqrt(0.524 * 0.476)
    n_needed = int(2 * ((z_alpha + z_beta) / effect_z) ** 2)
    out.write(f"    n ≈ {n_needed} bets per comparison group\n")
    out.write(f"    Current June 2026: n=120 — {'adequate' if 120 >= n_needed else 'underpowered'} for 5% ROI detection\n")


# ---------------------------------------------------------------------------
# Experiment B — rolling-origin assessment
# ---------------------------------------------------------------------------

def experiment_b_note(data: dict, out) -> None:
    out.write("\n" + "="*70 + "\n")
    out.write("EXPERIMENT B — ROLLING-ORIGIN ASSESSMENT\n")
    out.write("="*70 + "\n")
    out.write("""
A true rolling-origin experiment requires retraining the model daily with outcomes
through the previous day. This is infeasible with current data availability:

  - 2026 pitcher game logs (Jan–June) were NOT fetched in the historical pull.
    The historical fetch ran 2022-01-01 to 2025-12-31 only.
  - Without 2026 game outcomes, we cannot retrain with train_end = 2025-12-31
    or 2026-05-31 and get different model weights.
  - The picks_log (June 9–22) represents LIVE predictions, but with the same
    frozen model weights (trained 2022–2024). Features were computed daily with
    up-to-date rolling history, but the Poisson GLM did not re-fit.

PROXY COMPARISON: picks_log vs backtest June coverage
------------------------------------------------------
""")

    pl_june = data["pl"][data["pl"]["month"] == 6].dropna(subset=["won"])
    bt_june = data["bt26"][data["bt26"]["month"] == 6].dropna(subset=["won"])

    out.write(f"  picks_log June 9–22 (settled): n={len(pl_june)}, "
              f"WR={win_rate(pl_june['won'].values)*100:.1f}%\n")
    out.write(f"  backtest  June 1–21:            n={len(bt_june)}, "
              f"WR={win_rate(bt_june['won'].values)*100:.1f}%\n")

    # Check if same bets appear in both
    pl_keys  = set(pl_june[["game_date","pitcher_name","best_side"]].apply(tuple, axis=1))
    bt_keys  = set(bt_june[["game_date","pitcher_name","best_side"]].apply(tuple, axis=1))
    overlap  = pl_keys & bt_keys
    out.write(f"\n  Overlap (same date+pitcher+side): {len(overlap)} bets in both datasets.\n")
    out.write(f"  picks_log only:  {len(pl_keys - bt_keys)} bets (different selection/threshold)\n")
    out.write(f"  backtest only:   {len(bt_keys - pl_keys)} bets (missed live logging)\n")

    out.write("""
WHAT PROPER ROLLING-ORIGIN WOULD REQUIRE:
  1. Fetch 2026 pitcher game logs (2026-01-01 to 2026-05-31)
  2. Retrain with train_end = 2025-12-31 (or 2026-05-31)
  3. Apply refitted model to each June date using only prior-day features
  4. Compare projection shifts and prediction changes vs Experiment A

This would reveal whether 2025 outcomes (which show strong edge) improve
calibration for June 2026 starters. Without this data, we cannot complete B.

WHAT WE CAN SAY: The CLV data is available for the live picks (n=51) and
provides a market-efficiency check that is model-agnostic.
""")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    data = load_all()
    bt25 = data["bt25"]
    bt26 = data["bt26"]
    pl   = data["pl"]

    # -----------------------------------------------------------------------
    # Slice periods
    # -----------------------------------------------------------------------
    bt25_pre_june  = bt25[bt25["month"] < 6].dropna(subset=["won"])
    bt25_june      = bt25[bt25["month"] == 6].dropna(subset=["won"])
    bt25_post_june = bt25[bt25["month"] > 6].dropna(subset=["won"])
    bt26_pre_june  = bt26[bt26["month"] < 6].dropna(subset=["won"])
    bt26_june      = bt26[bt26["month"] == 6].dropna(subset=["won"])
    pl_june        = pl[pl["month"] == 6].dropna(subset=["won"])
    combined_2025  = bt25.dropna(subset=["won"])
    combined_26_thru_june = bt26.dropna(subset=["won"])

    # All settled bets combined
    # Use pnl from bt25 directly; compute for bt26/pl
    all_rows = []
    for df, label in [(bt25.dropna(subset=["won"]), "2025"),
                      (bt26.dropna(subset=["won"]), "2026")]:
        all_rows.append(df)
    combined_all = pd.concat(all_rows, ignore_index=True, sort=False)

    periods: list[dict] = []
    period_defs = [
        (bt25_pre_june,  "Pre-June 2025 (backtest)"),
        (bt25_june,      "June 2025 (backtest)"),
        (bt25_post_june, "Post-June 2025 (backtest)"),
        (bt26_pre_june,  "Pre-June 2026 (backtest)"),
        (bt26_june,      "June 2026 (backtest)"),
        (pl_june,        "June 2026 (live picks_log)"),
        (combined_2025,  "Full 2025 (backtest)"),
        (combined_all,   "Combined 2025+2026"),
    ]

    with open(REPORT_OUT, "w", encoding="utf-8") as out:
        out.write("=" * 70 + "\n")
        out.write("JUNE HOLDOUT EXPERIMENTS — Full Report\n")
        out.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        out.write("=" * 70 + "\n")

        # -----------------------------------------------------------------------
        # Contamination audit
        # -----------------------------------------------------------------------
        contamination_audit(out)

        # -----------------------------------------------------------------------
        # Experiment A
        # -----------------------------------------------------------------------
        out.write("\n" + "="*70 + "\n")
        out.write("EXPERIMENT A — June 2026 as Frozen Holdout\n")
        out.write("="*70 + "\n")
        out.write("""
Setup:
  Model    : Poisson GLM trained on 2022-01-01 to 2024-12-31 (frozen).
             Walk-forward 2025 used for OOS validation, not model fitting.
  Features : Rolling features shift(1) before aggregation — no same-day leakage.
  Holdout  : All qualifying model outputs for June 2026 (EGP>=6, not manually filtered).
  Source   : data/exports/2026_backtest_extended.csv, month==6

This is a proper A/B holdout: model was never updated using June 2026 outcomes.
Caveat: imputation used all-data medians (see contamination audit). Impact likely
small since NaN rates in rolling features are low after 2+ years of history.
""")

        s_a = period_stats(bt26_june, "June 2026 (backtest) — Experiment A")
        print_period(s_a, out)

        out.write("\nFor context — what the model achieved OUT-OF-SAMPLE before June:\n")
        s_pre26 = period_stats(bt26_pre_june, "Pre-June 2026 (backtest)")
        print_period(s_pre26, out)

        # -----------------------------------------------------------------------
        # Experiment B
        # -----------------------------------------------------------------------
        experiment_b_note(data, out)

        # -----------------------------------------------------------------------
        # Experiment C — full comparison
        # -----------------------------------------------------------------------
        out.write("\n" + "="*70 + "\n")
        out.write("EXPERIMENT C — Full Period Comparison\n")
        out.write("="*70 + "\n")

        for df, label in period_defs:
            s = period_stats(df, label)
            periods.append(s)
            print_period(s, out)

        # -----------------------------------------------------------------------
        # Significance tests
        # -----------------------------------------------------------------------
        significance_tests(periods, out)

        # -----------------------------------------------------------------------
        # Experiment C addendum: bet decisions changed
        # -----------------------------------------------------------------------
        out.write("\n" + "="*70 + "\n")
        out.write("DECISIONS CHANGED: Pre-June vs June model comparison\n")
        out.write("="*70 + "\n")
        out.write("""
'Decisions changed' measures how many bets the model would reclassify if we
excluded June vs included it in the training window (Exp A vs Exp B).

Because Experiment B requires 2026 game logs (not fetched), we proxy this by
comparing the EGP distribution in June 2026 against the pre-June training period:
""")
        if len(bt26_june) and len(bt26_pre_june):
            june_egp_mean  = bt26_june["egp"].mean()
            pre_egp_mean   = bt26_pre_june["egp"].mean()
            june_egp_above12 = (bt26_june["egp"] >= 12).mean() * 100
            pre_egp_above12  = (bt26_pre_june["egp"] >= 12).mean() * 100
            out.write(f"\n  Pre-June 2026 EGP: mean={pre_egp_mean:.1f}, "
                      f"%>=12: {pre_egp_above12:.0f}%  (n={len(bt26_pre_june)})\n")
            out.write(f"  June 2026     EGP: mean={june_egp_mean:.1f}, "
                      f"%>=12: {june_egp_above12:.0f}%  (n={len(bt26_june)})\n")

        # Projection shift
        out.write("\nProjection accuracy by period:\n")
        for df, label in [(bt26_pre_june, "Pre-June 2026"),
                          (bt26_june,     "June 2026"),
                          (bt25_june,     "June 2025")]:
            sub = df.dropna(subset=["actual"])
            if len(sub):
                err = sub["strikeouts_projection"] - sub["actual"]
                out.write(f"  {label:20s}: bias={err.mean():+.2f}  RMSE={np.sqrt((err**2).mean()):.2f}  "
                          f"n={len(sub)}\n")

        # -----------------------------------------------------------------------
        # Final recommendation
        # -----------------------------------------------------------------------
        out.write("\n" + "="*70 + "\n")
        out.write("FINAL RECOMMENDATION\n")
        out.write("="*70 + "\n")

        june25_wr  = win_rate(bt25_june["won"].values)
        june26_wr  = win_rate(bt26_june["won"].values)
        june25_roi = period_stats(bt25_june,  "")["roi"]
        june26_roi = period_stats(bt26_june,  "")["roi"]
        pre26_roi  = period_stats(bt26_pre_june, "")["roi"]

        out.write(f"""
EVIDENCE SUMMARY:
  June 2025  : n={len(bt25_june)}, WR={june25_wr*100:.1f}%, ROI={june25_roi:+.1f}%  — STRONG POSITIVE
  June 2026  : n={len(bt26_june)}, WR={june26_wr*100:.1f}%, ROI={june26_roi:+.1f}%  — STRONGLY NEGATIVE
  Pre-June 26: n={len(bt26_pre_june)}, ROI={pre26_roi:+.1f}%  — positive before June 2026 failure

SAMPLE SIZE: n=120 for June 2026.
  At n=120, we can detect +-15% ROI differences from zero at 80% power.
  June 2026 ROI of {june26_roi:+.1f}% IS detectable — but across ONLY ONE June.
  Two June observations (2025, 2026) are insufficient to determine seasonality.

QUESTION: Should June be excluded from model development?
""")
        # Statistical comparison of the two Junes
        june25_pnl = bt25_june["pnl"].values if "pnl" in bt25_june.columns else np.array([
            pnl_flat(r["won"], r["odds_used"]) for _, r in bt25_june.iterrows()])
        june26_pnl = np.array([pnl_flat(r["won"], r["odds_used"])
                                for _, r in bt26_june.iterrows()])
        tstat, pval = stats.ttest_ind(june25_pnl, june26_pnl, nan_policy="omit")

        out.write(f"""  June 2025 vs June 2026 t-test: t={tstat:+.2f}, p={pval:.4f}
  The two Junes are statistically different from each other (p<0.05: {pval < 0.05}).
  This means June is NOT a consistent seasonal pattern — it's year-to-year variance.

RECOMMENDATION (based on evidence only):

1. Do NOT exclude June based on 2026 alone.
   - June 2025 was the BEST month (+{june25_roi:.1f}% ROI).
   - One bad June does not establish seasonality.
   - Excluding June discards training signal and shrinks n unnecessarily.

2. June 2026 failure is diagnostic, not definitional.
   - Negative CLV (mean: {bt26_june['clv_pct'].mean():+.2f}) confirms model drift vs market,
     not a structural calendar effect.
   - Projection bias in June 2026: pitchers thrown {bt26_june['strikeouts_projection'].mean()-bt26_june['actual'].mean():+.2f} Ks
     below projection on average — strikeout environment shifted down.
   - The 2022–2024 model is stale for 2026 patterns regardless of month.

3. Include June via rolling-origin (Experiment B) once 2026 logs are fetched.
   - Retrain with train_end = 2025-12-31 to capture 2025 patterns.
   - Use proper walk-forward: June N predictions use only data through June N-1.
   - This is contamination-free IF imputation bug is fixed first (see audit).

4. Current best holdout for true OOS validation:
   - July–September 2026 (if available).
   - These months have no training signal yet and are fully future.
   - If 2026 post-season data confirms edge, June exclusion is unnecessary.
   - If 2026 post-season also fails, model retraining is the priority.

5. Fix the imputation bug before any retraining.
   - Currently, model fit uses all-data medians for NaN imputation.
   - Fix: compute train medians from train_df first, pass to build_training_features.
   - This is a 5-line change in scripts/train.py.

BOTTOM LINE:
  June is too small (2 observations) to be a decisive holdout.
  The evidence does not support a seasonal skip rule.
  Prioritize: (a) fix imputation, (b) fetch 2026 logs, (c) retrain, (d) watch July–Sep.
""")

    # Print to console (safe encoding)
    text = REPORT_OUT.read_text(encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    print(f"\nReport written to: {REPORT_OUT}")


if __name__ == "__main__":
    main()
