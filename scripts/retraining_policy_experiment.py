"""
Retraining Policy Experiment — walk-forward comparison of training cadences.

Evaluates 7 candidate policies on out-of-sample pitcher strikeout projection
accuracy for 2025-2026. Anti-leakage is enforced: every prediction uses only
data that was available before the prediction date.

Run:
    py -3.14 scripts/retraining_policy_experiment.py --build-features
    py -3.14 scripts/retraining_policy_experiment.py [--use-cache]

Outputs:
    reports/retraining_policy/policy_comparison.csv
    reports/retraining_policy/audit_table.csv
    reports/retraining_policy/report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loaders import (
    filter_date_range,
    load_batter_game_logs,
    load_game_context_logs,
    load_pitcher_game_logs,
    load_statcast_pitcher_daily,
    load_team_batting_game_logs,
    load_park_factors,
)
from src.features.build_features import build_training_features

CACHE_DIR   = ROOT / "data" / "wf_cache"
REPORT_DIR  = ROOT / "reports" / "retraining_policy"
CONFIG_PATH = ROOT / "config" / "config_v4_production.yaml"

# Features from the production model (non-opportunity)
PROD_FEATURES_PATH = ROOT / "data" / "processed_poisson_wf2025" / "models" / "strikeouts.joblib"


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    name: str
    label: str
    cadence_days: int | None      # None = never retrain (frozen)
    window_days: int | None       # None = expanding from fixed start
    fixed_train_start: str = "2022-01-01"
    fixed_train_end: str | None = None  # for frozen policy

    def folds(self, eval_start: pd.Timestamp, eval_end: pd.Timestamp,
              data_start: pd.Timestamp) -> list[dict]:
        """Return list of fold dicts: {fold, train_start, train_end, eval_start, eval_end}."""
        folds = []

        if self.cadence_days is None:
            # Frozen: one fit, evaluate the whole period
            folds.append({
                "fold": 0,
                "train_start": pd.Timestamp(self.fixed_train_start),
                "train_end":   pd.Timestamp(self.fixed_train_end),
                "eval_start":  eval_start,
                "eval_end":    eval_end,
                "policy":      self.name,
            })
            return folds

        # Rolling retraining: add a new fit every `cadence_days`
        # First retrain happens before eval_start
        # Each fold covers cadence_days of evaluation
        retrain_date = eval_start
        fold_idx = 0
        while retrain_date <= eval_end:
            # Train on everything up to (but not including) retrain_date
            train_end = retrain_date - pd.Timedelta(days=1)
            if self.window_days:
                train_start = train_end - pd.Timedelta(days=self.window_days)
                train_start = max(train_start, data_start)
            else:
                train_start = pd.Timestamp(self.fixed_train_start)

            # Must have at least 365 days of training data
            if (train_end - train_start).days < 365:
                retrain_date += pd.Timedelta(days=self.cadence_days)
                fold_idx += 1
                continue

            fold_eval_end = min(
                retrain_date + pd.Timedelta(days=self.cadence_days - 1),
                eval_end,
            )
            folds.append({
                "fold":        fold_idx,
                "train_start": train_start,
                "train_end":   train_end,
                "eval_start":  retrain_date,
                "eval_end":    fold_eval_end,
                "policy":      self.name,
            })
            retrain_date += pd.Timedelta(days=self.cadence_days)
            fold_idx += 1

        return folds


POLICIES = [
    Policy("frozen",             "Frozen (2022-2024)",       cadence_days=None,  window_days=None,
           fixed_train_end="2024-12-31"),
    Policy("annual_expanding",   "Annual (expanding)",        cadence_days=365,   window_days=None),
    Policy("rolling_3yr",        "Rolling 3-year window",     cadence_days=30,    window_days=365*3),
    Policy("rolling_2yr",        "Rolling 2-year window",     cadence_days=30,    window_days=365*2),
    Policy("expanding_monthly",  "Monthly (expanding)",       cadence_days=30,    window_days=None),
    Policy("expanding_biweekly", "Bi-weekly (expanding)",     cadence_days=14,    window_days=None),
    Policy("expanding_weekly",   "Weekly (expanding)",        cadence_days=7,     window_days=None),
]


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def load_prod_features() -> list[str]:
    """Load the 103 feature names from the production model (non-opportunity subset)."""
    import joblib
    m = joblib.load(PROD_FEATURES_PATH)
    feature_cols = m["feature_cols"]
    # Exclude opportunity features (predicted, not raw) for walk-forward
    opp_cols = {"expected_innings_pitched", "expected_pitches", "expected_batters_faced"}
    return [f for f in feature_cols if f not in opp_cols]


def build_and_cache_features(config: dict) -> pd.DataFrame:
    """Build raw feature matrix (no imputation) and cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "raw_features.parquet"

    if cache_path.exists():
        print(f"Loading cached feature matrix from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Building feature matrix (this takes 10-20 minutes)...")
    t0 = time.time()

    logs         = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_batting = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    game_context = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_logs  = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast     = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    park_factors = load_park_factors(config["data"]["park_factors_file"])

    print(f"  Loaded {len(logs):,} pitcher-game-log rows")

    featured, _, _ = build_training_features(
        logs,
        rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        team_batting_logs=team_batting,
        game_context_logs=game_context,
        batter_game_logs=batter_logs,
        statcast_pitcher_daily=statcast,
        park_factors=park_factors,
        return_before_impute=True,  # defer imputation — fixes leakage bug
    )

    print(f"  Feature matrix: {featured.shape}  ({time.time()-t0:.0f}s)")
    featured.to_parquet(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return featured


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_poisson(X_train: pd.DataFrame, y_train: pd.Series,
                feature_cols: list[str], alpha: float = 0.1) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=alpha, max_iter=500, solver="lbfgs")),
    ])
    X = X_train[feature_cols].values
    pipe.fit(X, y_train.values)
    return pipe


def predict_poisson(pipe: Pipeline, X_test: pd.DataFrame,
                    feature_cols: list[str]) -> np.ndarray:
    X = X_test[feature_cols].values
    return pipe.predict(X)


def impute_fold(df_raw: pd.DataFrame, feature_cols: list[str],
                train_mask: pd.Series, fill_values_override: dict | None = None) -> pd.DataFrame:
    """Impute feature_cols using medians from training rows only."""
    df = df_raw.copy()
    if fill_values_override:
        fill = pd.Series(fill_values_override).reindex(feature_cols, fill_value=0.0)
    else:
        train_medians = df.loc[train_mask, feature_cols].median(numeric_only=True).fillna(0.0)
        fill = train_medians.reindex(feature_cols, fill_value=0.0)
    df[feature_cols] = df[feature_cols].fillna(fill)
    return df


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def bias(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(pred - actual))


def poisson_log_likelihood(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean log-likelihood under Poisson(lambda=pred) for observed actual."""
    lam = np.maximum(pred, 1e-6)
    ll = actual * np.log(lam) - lam  # ignoring log(actual!) constant
    return float(np.mean(ll))


def calibration_score(pred: np.ndarray, actual: np.ndarray, n_bins: int = 5) -> float:
    """Mean absolute calibration error across quantile bins."""
    from scipy.stats import poisson as sp_poisson
    # Compute actual over-rate vs model-predicted over-rate at each observation's projected K
    # Use integer line = round(projection to nearest 0.5 increment)
    errors = []
    lines = np.round(pred * 2) / 2  # round to nearest 0.5
    unique_lines = np.unique(lines)
    for ln in unique_lines:
        mask = lines == ln
        if mask.sum() < 5:
            continue
        model_prob = 1 - sp_poisson.cdf(ln, pred[mask])
        actual_hit = (actual[mask] > ln).astype(float)
        errors.append(abs(model_prob.mean() - actual_hit.mean()))
    return float(np.mean(errors)) if errors else np.nan


def bootstrap_ci_rmse(pred: np.ndarray, actual: np.ndarray,
                      n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    rmse_vals = []
    n = len(pred)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rmse_vals.append(rmse(pred[idx], actual[idx]))
    lo = np.percentile(rmse_vals, (1 - ci) / 2 * 100)
    hi = np.percentile(rmse_vals, (1 + ci) / 2 * 100)
    return (lo, hi)


def binomial_pvalue(wins: int, n: int, p0: float = 0.524) -> float:
    """Two-sided binomial test vs break-even win rate p0."""
    result = stats.binomtest(wins, n, p0, alternative="two-sided")
    return result.pvalue


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Max drawdown as % of starting bankroll (100 * n bets)."""
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    n = len(pnl_series)
    return float(dd.min() / (100 * n) * 100)


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------

def run_policy(policy: Policy, df_raw: pd.DataFrame, feature_cols: list[str],
               eval_start: pd.Timestamp, eval_end: pd.Timestamp,
               data_start: pd.Timestamp, target_col: str = "strikeouts",
               alpha: float = 0.1) -> tuple[pd.DataFrame, list[dict]]:
    """Run one policy's walk-forward. Returns (predictions_df, audit_rows)."""
    folds = policy.folds(eval_start, eval_end, data_start)
    if not folds:
        print(f"  {policy.name}: no valid folds")
        return pd.DataFrame(), []

    all_preds = []
    audit_rows = []
    t_policy = time.time()

    for fold in folds:
        train_start = fold["train_start"]
        train_end   = fold["train_end"]
        f_eval_s    = fold["eval_start"]
        f_eval_e    = fold["eval_end"]

        # Anti-leakage check
        assert train_end < f_eval_s, f"LEAKAGE in fold {fold['fold']}: train_end={train_end} >= eval_start={f_eval_s}"

        # Build masks
        train_mask = (df_raw["game_date"] >= train_start) & (df_raw["game_date"] <= train_end)
        eval_mask  = (df_raw["game_date"] >= f_eval_s)    & (df_raw["game_date"] <= f_eval_e)

        # Must have both training and evaluation data
        if train_mask.sum() < 100 or eval_mask.sum() < 5:
            continue

        # Impute using train-period medians only
        df_imp = impute_fold(df_raw, feature_cols, train_mask)

        X_train = df_imp.loc[train_mask, feature_cols]
        y_train = df_imp.loc[train_mask, target_col]
        X_eval  = df_imp.loc[eval_mask,  feature_cols]
        y_eval  = df_imp.loc[eval_mask,  target_col]

        # Remove rows where target is NaN
        valid_train = y_train.notna()
        valid_eval  = y_eval.notna()
        if valid_train.sum() < 50 or valid_eval.sum() < 5:
            continue

        t0 = time.time()
        try:
            model = fit_poisson(X_train[valid_train], y_train[valid_train], feature_cols, alpha)
        except Exception as e:
            print(f"    fold {fold['fold']} fit failed: {e}")
            continue
        fit_time = time.time() - t0

        preds = predict_poisson(model, X_eval, feature_cols)

        # Build fold predictions DataFrame
        fold_df = df_imp.loc[eval_mask].copy()
        fold_df["prediction"] = preds
        fold_df["policy"] = policy.name
        fold_df["fold"] = fold["fold"]
        fold_df["train_start"] = train_start
        fold_df["train_end"] = train_end
        fold_df["train_days"] = (train_end - train_start).days
        fold_df["model_age_at_eval"] = (f_eval_s - train_end).days
        all_preds.append(fold_df)

        audit_rows.append({
            "policy": policy.name,
            "fold": fold["fold"],
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "eval_start": f_eval_s.date(),
            "eval_end": f_eval_e.date(),
            "n_train": int(valid_train.sum()),
            "n_eval": int(valid_eval.sum()),
            "train_end_lt_eval_start": train_end < f_eval_s,
            "fit_seconds": round(fit_time, 1),
        })

    elapsed = time.time() - t_policy
    n_folds = len(audit_rows)
    print(f"  {policy.name}: {n_folds} folds, {elapsed:.0f}s")

    if not all_preds:
        return pd.DataFrame(), audit_rows

    return pd.concat(all_preds, ignore_index=True), audit_rows


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(preds_df: pd.DataFrame, policy: Policy,
                    backtest_df: pd.DataFrame | None = None) -> dict:
    """Compute all required metrics for a policy's predictions."""
    df = preds_df.dropna(subset=["prediction", "strikeouts"])
    if len(df) == 0:
        return {"policy": policy.name, "n": 0}

    pred  = df["prediction"].values
    actual = df["strikeouts"].values

    r = rmse(pred, actual)
    m = mae(pred, actual)
    b = bias(pred, actual)
    ll = poisson_log_likelihood(pred, actual)
    cal = calibration_score(pred, actual)
    ci  = bootstrap_ci_rmse(pred, actual)

    result = {
        "policy":     policy.name,
        "label":      policy.label,
        "cadence":    policy.cadence_days,
        "window":     policy.window_days,
        "n_predictions": len(df),
        "rmse":       round(r, 4),
        "rmse_ci_lo": round(ci[0], 4),
        "rmse_ci_hi": round(ci[1], 4),
        "mae":        round(m, 4),
        "bias":       round(b, 4),
        "poisson_ll": round(ll, 4),
        "calibration_mae": round(cal, 4) if not np.isnan(cal) else None,
        "n_folds":    df["fold"].nunique(),
    }

    # Monthly breakdown
    df["month_key"] = df["game_date"].dt.to_period("M")
    monthly = df.groupby("month_key").apply(
        lambda g: pd.Series({
            "n": len(g),
            "rmse": rmse(g["prediction"].values, g["strikeouts"].values),
            "bias": bias(g["prediction"].values, g["strikeouts"].values),
        })
    ).reset_index()
    result["monthly"] = monthly.to_dict(orient="records")

    # Model age analysis
    if "model_age_at_eval" in df.columns:
        age_corr = np.corrcoef(df["model_age_at_eval"],
                               (df["prediction"] - df["strikeouts"]).abs())[0, 1]
        result["age_rmse_corr"] = round(float(age_corr), 4)
        # Bin by age quartile
        df["age_bin"] = pd.qcut(df["model_age_at_eval"], q=4, labels=False, duplicates="drop")
        age_rmse = df.groupby("age_bin").apply(
            lambda g: pd.Series({
                "age_days": g["model_age_at_eval"].median(),
                "n": len(g),
                "rmse": rmse(g["prediction"].values, g["strikeouts"].values),
                "bias": bias(g["prediction"].values, g["strikeouts"].values),
            })
        ).reset_index()
        result["age_rmse"] = age_rmse.to_dict(orient="records")

    # Betting performance (if backtest data available)
    if backtest_df is not None and len(backtest_df) > 0:
        bet_stats = compute_bet_metrics(df, backtest_df)
        result.update(bet_stats)

    return result


def compute_bet_metrics(preds_df: pd.DataFrame, backtest_df: pd.DataFrame) -> dict:
    """Join walk-forward predictions with backtest odds and evaluate betting P&L."""
    # Normalize names for join
    preds_df = preds_df.copy()
    preds_df["game_date"] = pd.to_datetime(preds_df["game_date"])
    bt = backtest_df.copy()
    bt["game_date"] = pd.to_datetime(bt["game_date"])

    # Merge: for each backtest bet, get the walk-forward projected K count
    merged = bt.merge(
        preds_df[["game_date", "pitcher_id", "prediction"]].dropna(),
        on=["game_date", "pitcher_id"],
        how="inner",
    )
    if len(merged) == 0:
        return {}

    # Recompute edge using walk-forward projection
    from scipy.stats import poisson as sp_poisson
    def recompute_bet(row):
        lam = max(row["prediction"], 0.01)
        line = row["line"]
        side = row["best_side"]
        odds = row["odds_used"]
        if pd.isna(line) or pd.isna(odds) or pd.isna(side):
            return np.nan, np.nan, np.nan

        model_prob = (1 - sp_poisson.cdf(line, lam)) if side == "over" else sp_poisson.cdf(line - 0.5, lam)
        imp_prob   = abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
        edge_pct   = (model_prob - imp_prob) / imp_prob * 100
        gap        = (lam - line) if side == "over" else (line - lam)
        egp        = abs(gap) * edge_pct
        return edge_pct, egp, model_prob

    merged[["wf_edge_pct", "wf_egp", "wf_prob"]] = merged.apply(
        lambda r: pd.Series(recompute_bet(r)), axis=1)

    # Apply same EGP filter as production (>= 6)
    bets = merged[merged["wf_egp"] >= 6].dropna(subset=["won"])

    if len(bets) == 0:
        return {"bet_n": 0}

    won = bets["won"].values
    odds = bets["odds_used"].values

    def pnl_flat(w, o, stake=100):
        dec = (1 + o/100) if o >= 0 else (1 - 100/o)
        return stake * (dec - 1) if w else -stake

    pnls = np.array([pnl_flat(w, o) for w, o in zip(won, odds)])

    wr  = won.mean()
    roi = pnls.sum() / (len(pnls) * 100) * 100
    n   = len(bets)
    units = pnls.sum() / 100

    # Decisions changed vs original model
    orig_bets = backtest_df.dropna(subset=["won"])
    orig_keys = set(zip(orig_bets["game_date"], orig_bets["pitcher_id"], orig_bets["best_side"]))
    wf_keys   = set(zip(bets["game_date"], bets["pitcher_id"], bets["best_side"]))
    added   = len(wf_keys - orig_keys)
    removed = len(orig_keys - wf_keys)
    changed = added + removed

    # Price-adjusted ROI
    price_adj = {}
    for cents in [5, 10, 15, 20]:
        adj_pnls = np.array([pnl_flat(w, o + cents) for w, o in zip(won, odds)])
        price_adj[f"roi_+{cents}c"] = round(adj_pnls.sum() / (len(adj_pnls) * 100) * 100, 2)

    # CLV if available
    clv_stats = {}
    if "clv_pct" in bets.columns:
        clv = bets["clv_pct"].dropna()
        if len(clv):
            clv_stats = {"clv_mean": round(clv.mean(), 3), "clv_pct_positive": round((clv > 0).mean() * 100, 1)}

    # Bootstrap CI on ROI
    rng = np.random.default_rng(42)
    roi_boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(pnls), len(pnls))
        roi_boots.append(pnls[idx].sum() / (len(pnls) * 100) * 100)
    roi_ci = (np.percentile(roi_boots, 2.5), np.percentile(roi_boots, 97.5))

    # Binomial p-value
    bp = binomial_pvalue(int(won.sum()), n, p0=0.524) if n > 0 else np.nan

    # Max drawdown
    dd = max_drawdown(pnls)

    # Side breakdown
    sides = {}
    for s in ["over", "under"]:
        sub = bets[bets["best_side"] == s]
        if len(sub):
            sp = np.array([pnl_flat(w, o) for w, o in zip(sub["won"].values, sub["odds_used"].values)])
            sides[f"bet_{s}_n"]   = len(sub)
            sides[f"bet_{s}_wr"]  = round(sub["won"].mean(), 3)
            sides[f"bet_{s}_roi"] = round(sp.sum() / (len(sp) * 100) * 100, 1)

    return {
        "bet_n":         n,
        "bet_wr":        round(wr, 3),
        "bet_roi":       round(roi, 2),
        "bet_roi_ci":    (round(roi_ci[0], 1), round(roi_ci[1], 1)),
        "bet_units":     round(units, 1),
        "bet_binom_p":   round(bp, 4),
        "bet_max_dd_pct": round(dd, 1),
        "bet_decisions_changed": changed,
        "bet_added":    added,
        "bet_removed":  removed,
        **price_adj,
        **clv_stats,
        **sides,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(all_results: list[dict], audit_df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("RETRAINING POLICY EXPERIMENT — Full Report\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 72 + "\n\n")

        # Anti-leakage audit
        f.write("ANTI-LEAKAGE AUDIT\n")
        f.write("-" * 72 + "\n")
        violations = audit_df[~audit_df["train_end_lt_eval_start"]]
        if len(violations):
            f.write(f"VIOLATIONS FOUND: {len(violations)}\n")
            f.write(violations.to_string())
        else:
            f.write(f"PASS: All {len(audit_df)} folds satisfy train_end < eval_start.\n")
            f.write(f"      Total folds audited: {len(audit_df)}\n")
            f.write(f"      Policies covered: {audit_df['policy'].nunique()}\n")
        f.write("\n")

        # Projection accuracy summary
        f.write("=" * 72 + "\n")
        f.write("PROJECTION ACCURACY (RMSE on actual K counts)\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Policy':<30} {'n':>6} {'RMSE':>7} {'95% CI':>18} {'MAE':>7} {'Bias':>7} {'PoissonLL':>10} {'CalMAE':>8}\n")
        f.write("-" * 72 + "\n")
        sorted_results = sorted(all_results, key=lambda x: x.get("rmse", 99))
        for r in sorted_results:
            if r.get("n_predictions", 0) == 0:
                continue
            ci = r.get("rmse_ci_lo", np.nan), r.get("rmse_ci_hi", np.nan)
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            f.write(f"  {r['label']:<28} {r['n_predictions']:>6} {r['rmse']:>7.4f} {ci_str:>18} "
                    f"{r['mae']:>7.4f} {r['bias']:>+7.4f} {r['poisson_ll']:>10.4f} "
                    f"{r.get('calibration_mae', 'n/a'):>8}\n")
        f.write("\n")

        # Model age analysis
        f.write("=" * 72 + "\n")
        f.write("MODEL STALENESS: AGE vs RMSE CORRELATION\n")
        f.write("=" * 72 + "\n")
        for r in sorted_results:
            corr = r.get("age_rmse_corr")
            if corr is not None:
                f.write(f"  {r['label']:<35}: Pearson r = {corr:+.3f} (age_days vs |error|)\n")
        f.write("  Note: positive r = errors grow with model age (staleness)\n\n")

        # Age-binned RMSE for frozen model
        for r in all_results:
            if r["policy"] == "frozen" and "age_rmse" in r:
                f.write("Frozen model RMSE by model age quartile:\n")
                f.write(f"  {'Age (days)':>12} {'n':>6} {'RMSE':>7} {'Bias':>7}\n")
                for row in r["age_rmse"]:
                    f.write(f"  {row['age_days']:>12.0f} {row['n']:>6.0f} {row['rmse']:>7.4f} {row['bias']:>+7.4f}\n")
                f.write("\n")

        # Betting performance
        has_bet = any("bet_n" in r for r in all_results)
        if has_bet:
            f.write("=" * 72 + "\n")
            f.write("BETTING PERFORMANCE (joined with backtest odds)\n")
            f.write("=" * 72 + "\n")
            f.write(f"{'Policy':<30} {'n':>5} {'WR':>6} {'ROI':>8} {'95% CI':>18} {'binom_p':>8} {'MaxDD':>7}\n")
            f.write("-" * 72 + "\n")
            for r in sorted(all_results, key=lambda x: -x.get("bet_roi", -999)):
                if r.get("bet_n", 0) == 0:
                    continue
                ci = r.get("bet_roi_ci", (np.nan, np.nan))
                ci_str = f"[{ci[0]:+.1f}%, {ci[1]:+.1f}%]"
                f.write(f"  {r['label']:<28} {r['bet_n']:>5} {r['bet_wr']*100:>5.1f}% "
                        f"{r['bet_roi']:>+7.1f}% {ci_str:>18} {r['bet_binom_p']:>8.4f} "
                        f"{r['bet_max_dd_pct']:>+6.1f}%\n")
            f.write("\n")

            # Price-adjusted ROI
            f.write("Betting ROI with price improvement (exchange fills):\n")
            f.write(f"{'Policy':<30} {'@SB':>8} {'+5c':>8} {'+10c':>8} {'+15c':>8} {'+20c':>8}\n")
            f.write("-" * 72 + "\n")
            for r in sorted(all_results, key=lambda x: -x.get("bet_roi", -999)):
                if r.get("bet_n", 0) == 0:
                    continue
                f.write(f"  {r['label']:<28} {r['bet_roi']:>+7.1f}% "
                        f"{r.get('roi_+5c', 0):>+7.1f}% "
                        f"{r.get('roi_+10c', 0):>+7.1f}% "
                        f"{r.get('roi_+15c', 0):>+7.1f}% "
                        f"{r.get('roi_+20c', 0):>+7.1f}%\n")
            f.write("\n")

            # Decisions changed
            f.write("Bet selection vs original model:\n")
            for r in all_results:
                if "bet_decisions_changed" not in r:
                    continue
                f.write(f"  {r['label']:<35}: {r['bet_decisions_changed']:>4} changed "
                        f"(+{r['bet_added']} added, -{r['bet_removed']} removed)\n")
            f.write("\n")

        # Monthly breakdown for each policy
        f.write("=" * 72 + "\n")
        f.write("MONTHLY RMSE BREAKDOWN\n")
        f.write("=" * 72 + "\n")
        for r in all_results:
            if "monthly" not in r or not r["monthly"]:
                continue
            f.write(f"\n  {r['label']}:\n")
            f.write(f"    {'Month':>8} {'n':>5} {'RMSE':>7} {'Bias':>7}\n")
            for row in sorted(r["monthly"], key=lambda x: str(x.get("month_key", ""))):
                f.write(f"    {str(row.get('month_key', '')):>8} {row['n']:>5} {row['rmse']:>7.4f} {row['bias']:>+7.4f}\n")

        # Final recommendation
        f.write("\n\n" + "=" * 72 + "\n")
        f.write("PRODUCTION POLICY RECOMMENDATION\n")
        f.write("=" * 72 + "\n")
        best_rmse = sorted_results[0] if sorted_results else {}
        f.write(f"""
Based on projection accuracy (RMSE), the best policy is:
  {best_rmse.get('label', 'N/A')}  (RMSE = {best_rmse.get('rmse', 'N/A')})

The frozen model (trained once on 2022-2024) is the baseline.
Any policy with meaningfully lower RMSE justifies its computational overhead.

RMSE improvement threshold: 0.05 Ks/start is meaningful for betting edge.
Below that, the improvement likely does not change bet selection or ROI.

See full report details above for per-month and per-age breakdown.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-features", action="store_true",
                        help="Rebuild feature matrix (ignore cache)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached feature matrix (skip rebuild)")
    parser.add_argument("--eval-start", default="2025-03-01")
    parser.add_argument("--eval-end",   default="2026-06-27")
    parser.add_argument("--config",     default=str(CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Clear cache if requested
    cache_path = CACHE_DIR / "raw_features.parquet"
    if args.build_features and cache_path.exists():
        cache_path.unlink()
        print("Cleared feature cache.")

    # -------------------------------------------------------------------------
    # Load / build feature matrix
    # -------------------------------------------------------------------------
    print("\n=== PHASE 1: Feature Matrix ===")
    df_raw = build_and_cache_features(config)
    df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])

    eval_start  = pd.Timestamp(args.eval_start)
    eval_end    = pd.Timestamp(args.eval_end)
    data_start  = df_raw["game_date"].min()

    print(f"Data: {data_start.date()} to {df_raw['game_date'].max().date()}")
    print(f"Eval: {eval_start.date()} to {eval_end.date()}")
    print(f"Rows in eval window: {((df_raw['game_date'] >= eval_start) & (df_raw['game_date'] <= eval_end)).sum()}")

    # Get feature names from production model
    feature_cols = load_prod_features()
    # Only keep features that exist in the raw matrix
    feature_cols = [f for f in feature_cols if f in df_raw.columns]
    print(f"Features used: {len(feature_cols)} (of {len(load_prod_features())} production features)")

    # -------------------------------------------------------------------------
    # Load backtest data for betting evaluation
    # -------------------------------------------------------------------------
    print("\n=== PHASE 2: Loading Backtest Data ===")
    bt_paths = [
        ROOT / "data/exports/2025_backtest.csv",
        ROOT / "data/exports/2026_backtest_extended.csv",
    ]
    bt_dfs = []
    for p in bt_paths:
        if p.exists():
            bt = pd.read_csv(p)
            bt["game_date"] = pd.to_datetime(bt["game_date"])
            # Need pitcher_id for merge
            if "pitcher_id" not in bt.columns:
                # Join from df_raw on game_date + pitcher_name
                name_id_map = df_raw[["game_date","pitcher_name","pitcher_id"]].drop_duplicates()
                bt = bt.merge(name_id_map, on=["game_date","pitcher_name"], how="left")
            bt_dfs.append(bt)
            print(f"  Loaded {len(bt)} bets from {p.name}")

    backtest_df = pd.concat(bt_dfs, ignore_index=True, sort=False) if bt_dfs else pd.DataFrame()
    backtest_df = backtest_df[
        (backtest_df["game_date"] >= eval_start) &
        (backtest_df["game_date"] <= eval_end)
    ].copy()
    print(f"  Backtest bets in eval window: {len(backtest_df)}")

    # -------------------------------------------------------------------------
    # Run walk-forward for each policy
    # -------------------------------------------------------------------------
    print("\n=== PHASE 3: Walk-Forward Evaluation ===")
    all_results = []
    all_audit   = []

    for policy in POLICIES:
        print(f"\nPolicy: {policy.label}")
        pred_df, audit_rows = run_policy(
            policy, df_raw, feature_cols,
            eval_start, eval_end, data_start,
            alpha=0.1,
        )
        all_audit.extend(audit_rows)

        if len(pred_df):
            # Save per-policy predictions
            pred_path = REPORT_DIR / f"preds_{policy.name}.parquet"
            pred_df.to_parquet(pred_path, index=False)

        metrics = compute_metrics(
            pred_df, policy,
            backtest_df=backtest_df if len(backtest_df) else None,
        )
        all_results.append(metrics)
        print(f"  RMSE={metrics.get('rmse','N/A')}  n={metrics.get('n_predictions',0)}")

    # -------------------------------------------------------------------------
    # Save outputs and write report
    # -------------------------------------------------------------------------
    print("\n=== PHASE 4: Writing Report ===")
    audit_df = pd.DataFrame(all_audit)
    audit_df.to_csv(REPORT_DIR / "audit_table.csv", index=False)
    print(f"Audit table: {REPORT_DIR/'audit_table.csv'} ({len(audit_df)} rows)")

    # Save summary metrics
    summary_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_DIR / "policy_comparison.csv", index=False)
    print(f"Summary: {REPORT_DIR/'policy_comparison.csv'}")

    report_path = REPORT_DIR / "report.txt"
    write_report(all_results, audit_df, report_path)

    # Print to console
    text = report_path.read_text(encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
