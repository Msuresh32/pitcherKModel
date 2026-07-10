"""Phase 4: Out-of-sample evaluation on June 1 – July 7, 2026.

Loads the best model config selected in 02_model_search.py,
retrains on ALL pre-OOS data, then evaluates on the reserved
OOS holdout period. Never touches thresholds during OOS eval.

Outputs:
  research/oos_results/oos_bets.csv     — bet-level log
  research/oos_results/oos_metrics.json — aggregated metrics
  research/oos_results/oos_report.txt   — human-readable summary
"""
from __future__ import annotations
import sys, json, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DATASET_FILE = Path("research/dataset.parquet")
MODEL_DIR    = Path("research/model_results")
OUT_DIR      = Path("research/oos_results")

OOS_START    = pd.Timestamp("2026-06-01")
VAL_START    = pd.Timestamp("2026-04-01")

# These must match 02_model_search.py exactly
CORE_FEATURES = [
    "p_strikeouts_roll5", "p_strikeouts_roll10",
    "p_k_rate_roll5", "p_k_rate_roll10",
    "p_k_per_ip_roll5", "p_k_per_ip_roll10",
    "p_strikeouts_trimmed_roll5", "p_strikeouts_trimmed_roll10",
    "sc_swinging_strike_rate_roll5", "sc_swinging_strike_rate_roll10",
    "sc_csw_rate_roll5", "sc_csw_rate_roll10",
    "sc_called_strike_rate_roll5",
    "sc_zone_rate_roll5",
    "sc_fastball_pct_roll5", "sc_slider_pct_roll5", "sc_breaking_pct_roll5",
    "sc_avg_release_speed_roll5",
    "p_innings_pitched_roll5", "p_innings_pitched_roll10",
    "p_batters_faced_roll5",
    "p_deep_start_rate_6ip_roll5", "p_short_start_rate_under5ip_roll5",
    "p_strikeouts_career_avg_prior", "p_k_rate_career_prior",
    "opp_batting_k_rate_roll5", "opp_batting_k_rate_roll20",
    "opp_k_rate_trend",
    "opp_lineup_same_hand_batters", "opp_lineup_opposite_hand_batters",
    "opp_lineup_k_rate_roll7", "opp_lineup_k_rate_roll14",
    "opp_lineup_k_rate_roll7_vs_hand",
    "umpire_csr_avg_prior", "umpire_csr_excess_prior",
    "catcher_csr_avg_prior",
    "temperature",
    "park_so_factor",
    "days_rest", "days_into_season", "month",
    "line",
    "p_over_open",
    "n_books_open",
]


def get_feature_cols(dataset: pd.DataFrame) -> list[str]:
    return [f for f in CORE_FEATURES if f in dataset.columns]


# ---------------------------------------------------------------------------
# Model trainers (duplicated from 02 to avoid circular import)
# ---------------------------------------------------------------------------

def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        return pipe

    elif name == "poisson":
        from sklearn.linear_model import PoissonRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X_reg = X_train.drop(columns=["line", "p_over_open", "n_books_open"], errors="ignore")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.1, max_iter=500)),
        ])
        pipe.fit(X_reg, y_train)
        return pipe

    elif name == "xgboost":
        from xgboost import XGBClassifier
        from sklearn.calibration import CalibratedClassifierCV
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0,
        )
        cal = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
        cal.fit(X_train, y_train)
        return cal

    elif name == "lightgbm":
        import lightgbm as lgb
        from sklearn.calibration import CalibratedClassifierCV
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=-1,
        )
        cal = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
        cal.fit(X_train, y_train)
        return cal

    raise ValueError(f"Unknown model: {name}")


def predict(name: str, model, X: pd.DataFrame, line: pd.Series) -> pd.Series:
    if name == "poisson":
        from scipy.stats import poisson
        X_reg = X.drop(columns=["line", "p_over_open", "n_books_open"], errors="ignore")
        mu = pd.Series(np.maximum(model.predict(X_reg), 0.01), index=X.index)
        return pd.Series(
            [1 - poisson.cdf(int(l), mu_i) for l, mu_i in zip(line, mu)],
            index=X.index,
        )
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def american_to_pnl(odds: float) -> float:
    """Return profit per 1 unit bet (e.g. +200 → 2.0, -110 → 0.909)."""
    if odds >= 0:
        return odds / 100
    return 100 / abs(odds)


def evaluate_bets(bets: pd.DataFrame) -> dict:
    if len(bets) == 0:
        return {"n_bets": 0}

    bets = bets.copy()
    bets["pnl"] = bets.apply(
        lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1
    )

    n = len(bets)
    wins = int(bets["won"].sum())
    units = bets["pnl"].sum()
    roi = units / n

    # Bootstrap CI
    boot_rois = [bets["pnl"].sample(n, replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot_rois, [5, 95])

    # Binomial test vs break-even win rate
    avg_odds = bets["best_odds"].mean()
    if avg_odds < 0:
        breakeven_wr = abs(avg_odds) / (abs(avg_odds) + 100)
    else:
        breakeven_wr = 100 / (100 + avg_odds)
    binom = scipy_stats.binomtest(wins, n, breakeven_wr, alternative="greater")

    # Max drawdown
    cumsum = bets["pnl"].cumsum()
    max_dd = (cumsum - cumsum.cummax()).min()

    # CLV
    clv_cols = {
        "over": "clv_over",
        "under": "clv_over",  # negative for under
    }
    clv_vals = []
    for _, row in bets.iterrows():
        if "clv_over" in bets.columns:
            c = row["clv_over"] if row["bet_side"] == "over" else -row["clv_over"]
            clv_vals.append(c)

    return {
        "n_bets": n,
        "n_over": int((bets.bet_side == "over").sum()),
        "n_under": int((bets.bet_side == "under").sum()),
        "wins": wins,
        "win_rate": wins / n,
        "roi": roi,
        "units": units,
        "roi_ci_lo_90": lo,
        "roi_ci_hi_90": hi,
        "max_drawdown": max_dd,
        "avg_edge": float(bets["edge"].mean()),
        "mean_clv": float(np.mean(clv_vals)) if clv_vals else None,
        "binom_p_value": float(binom.pvalue),
        "breakeven_wr": breakeven_wr,
    }


def make_bets(
    preds: pd.Series,
    oos_data: pd.DataFrame,
    edge_threshold: float,
    min_prob: float,
) -> pd.DataFrame:
    df = oos_data.copy()
    df["p_model"] = preds.values
    df = df.dropna(subset=["p_model", "p_over_close"])

    over = df[
        (df["p_model"] - df["p_over_close"] >= edge_threshold) &
        (df["p_model"] >= min_prob) &
        df["best_over_odds_close"].notna()
    ].copy()
    over["bet_side"] = "over"
    over["won"] = over["outcome_over"] == 1
    over["best_odds"] = over["best_over_odds_close"]
    over["edge"] = over["p_model"] - over["p_over_close"]

    under = df[
        (df["p_over_close"] - df["p_model"] >= edge_threshold) &
        ((1 - df["p_model"]) >= min_prob) &
        df["best_under_odds_close"].notna()
    ].copy()
    under["bet_side"] = "under"
    under["won"] = under["outcome_over"] == 0
    under["best_odds"] = under["best_under_odds_close"]
    under["edge"] = under["p_over_close"] - under["p_model"]

    all_bets = pd.concat([over, under], ignore_index=True).sort_values("game_date")
    return all_bets


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------

def remove_top_n_wins(bets: pd.DataFrame, n: int = 5) -> dict:
    """Metric after removing top-n winning bets (largest single-bet pnl)."""
    if len(bets) == 0 or "pnl" not in bets.columns:
        return {}
    b = bets.copy()
    if "pnl" not in b.columns:
        b["pnl"] = b.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)
    top_idx = b[b["won"]].nlargest(min(n, len(b[b["won"]])), "pnl").index
    b2 = b.drop(index=top_idx)
    return {f"roi_minus_top{n}_wins": b2["pnl"].mean() if len(b2) else None}


def random_baseline(oos_data: pd.DataFrame, n_bets_match: int, n_trials: int = 1000) -> dict:
    """Compare to random betting with same number of bets."""
    if n_bets_match == 0:
        return {}
    rois = []
    for _ in range(n_trials):
        sample = oos_data.sample(min(n_bets_match, len(oos_data)), replace=True)
        random_over = np.random.rand(len(sample)) > 0.5
        side_won = np.where(random_over, sample["outcome_over"] == 1, sample["outcome_over"] == 0)
        pnl = np.where(side_won, 100 / 110, -1.0)  # ~-110 odds
        rois.append(pnl.mean())
    return {
        "random_baseline_roi_mean": np.mean(rois),
        "random_baseline_roi_p5": np.percentile(rois, 5),
        "random_baseline_roi_p95": np.percentile(rois, 95),
    }


def monthly_splits(bets: pd.DataFrame) -> pd.DataFrame:
    """ROI by month."""
    if len(bets) == 0:
        return pd.DataFrame()
    bets = bets.copy()
    if "pnl" not in bets.columns:
        bets["pnl"] = bets.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)
    bets["month"] = pd.to_datetime(bets["game_date"]).dt.to_period("M")
    return bets.groupby("month").agg(
        n_bets=("pnl", "count"),
        roi=("pnl", "mean"),
        units=("pnl", "sum"),
        win_rate=("won", "mean"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATASET_FILE.exists():
        print(f"ERROR: Run 01_build_dataset.py first.")
        return

    config_file = MODEL_DIR / "best_config.json"
    if not config_file.exists():
        print(f"ERROR: Run 02_model_search.py first. {config_file} not found.")
        return

    with open(config_file) as f:
        best_cfg = json.load(f)

    model_name     = best_cfg["best_model"]
    edge_threshold = best_cfg["best_edge_threshold"]
    min_prob       = best_cfg["best_min_prob"]

    print(f"=== OOS Evaluation ===")
    print(f"Model: {model_name}  edge_threshold={edge_threshold}  min_prob={min_prob}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])

    feat_cols = get_feature_cols(dataset)

    # Train on ALL pre-OOS data
    pre_oos = dataset[dataset["game_date"] < OOS_START].copy()
    oos     = dataset[dataset["game_date"] >= OOS_START].copy()

    print(f"Pre-OOS train rows: {len(pre_oos):,}")
    print(f"OOS rows:           {len(oos):,}")
    print(f"OOS date range:     {oos.game_date.min().date()} to {oos.game_date.max().date()}")

    X_train = pre_oos[feat_cols].fillna(0)
    y_train = pre_oos["outcome_over"]

    print(f"\nTraining {model_name} on {len(X_train):,} rows...")
    model = train_model(model_name, X_train, y_train)

    X_oos = oos[feat_cols].fillna(0)
    oos_preds = predict(model_name, model, X_oos, oos["line"])
    oos_preds.index = oos.index

    # Record model probability for all OOS rows
    oos = oos.copy()
    oos["p_model"] = oos_preds

    # Make bets
    all_bets = make_bets(oos_preds, oos, edge_threshold, min_prob)
    metrics  = evaluate_bets(all_bets)

    # Add pnl column for robustness checks
    if len(all_bets):
        all_bets["pnl"] = all_bets.apply(
            lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1
        )

    # Robustness
    rob = {}
    rob.update(remove_top_n_wins(all_bets, 5))
    rob.update(random_baseline(oos, metrics.get("n_bets", 0)))

    monthly = monthly_splits(all_bets)

    # Calibration check: mean p_model vs actual over rate by decile
    oos_notnull = oos.dropna(subset=["p_model", "p_over_close"])
    oos_notnull = oos_notnull.copy()
    oos_notnull["decile"] = pd.qcut(oos_notnull["p_model"], 10, labels=False, duplicates="drop")
    calib = (
        oos_notnull.groupby("decile")
        .agg(mean_pred=("p_model", "mean"), actual_rate=("outcome_over", "mean"), n=("outcome_over", "count"))
        .reset_index()
    )

    # Output
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(all_bets):
        keep_cols = [
            c for c in [
                "game_date", "pitcher_id", "pitcher_name", "line",
                "bet_side", "won", "best_odds", "edge",
                "p_model", "p_over_open", "p_over_close", "p_market",
                "clv_over", "outcome_over", "actual_ks", "pnl",
            ] if c in all_bets.columns
        ]
        all_bets[keep_cols].to_csv(OUT_DIR / "oos_bets.csv", index=False)

    monthly.to_csv(OUT_DIR / "monthly_splits.csv", index=False)
    calib.to_csv(OUT_DIR / "calibration.csv", index=False)
    oos[["game_date", "pitcher_id", "p_model", "p_over_close", "outcome_over", "line"]].to_csv(
        OUT_DIR / "oos_all_predictions.csv", index=False
    )

    full_metrics = {**metrics, **rob, "config": best_cfg}
    with open(OUT_DIR / "oos_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)

    # Human-readable report
    report = _format_report(metrics, rob, monthly, calib, best_cfg, len(oos))
    with open(OUT_DIR / "oos_report.txt", "w") as f:
        f.write(report)
    print("\n" + report)


def _format_report(metrics, rob, monthly, calib, cfg, n_total_oos) -> str:
    n = metrics.get("n_bets", 0)
    lines = [
        "=" * 60,
        "OOS EVALUATION REPORT",
        f"Period: 2026-06-01 to 2026-07-07",
        f"Model:  {cfg.get('best_model', '?')}",
        f"Edge threshold: {cfg.get('best_edge_threshold', '?')}",
        f"Min prob:       {cfg.get('best_min_prob', '?')}",
        "=" * 60,
        "",
        f"Total OOS pitcher-game-lines: {n_total_oos:,}",
        f"Bets placed (met threshold):  {n}",
    ]

    if n == 0:
        lines.append("\nNo bets met threshold — model generated no OOS signals.")
        return "\n".join(lines)

    lines += [
        f"  Over bets:  {metrics.get('n_over', '?')}",
        f"  Under bets: {metrics.get('n_under', '?')}",
        "",
        "BETTING PERFORMANCE",
        f"  Win rate:     {metrics.get('win_rate', 0):.1%}  (breakeven: {metrics.get('breakeven_wr', 0):.1%})",
        f"  ROI:          {metrics.get('roi', 0):+.3f} per bet",
        f"  ROI 90% CI:   [{metrics.get('roi_ci_lo_90', 0):+.3f}, {metrics.get('roi_ci_hi_90', 0):+.3f}]",
        f"  Total units:  {metrics.get('units', 0):+.2f}",
        f"  Max drawdown: {metrics.get('max_drawdown', 0):.2f} units",
        f"  Avg edge:     {metrics.get('avg_edge', 0):+.3f}",
        f"  Mean CLV:     {metrics.get('mean_clv') or 'N/A'}",
        f"  Binom p-val:  {metrics.get('binom_p_value', 1):.4f}",
        "",
        "ROBUSTNESS",
        f"  ROI - top 5 wins: {rob.get('roi_minus_top5_wins', 'N/A')}",
        f"  Random baseline ROI (mean): {rob.get('random_baseline_roi_mean', 'N/A')}",
        f"  Random baseline 90% CI: [{rob.get('random_baseline_roi_p5', 'N/A')}, {rob.get('random_baseline_roi_p95', 'N/A')}]",
    ]

    if not monthly.empty:
        lines += ["", "MONTHLY SPLITS"]
        for _, row in monthly.iterrows():
            lines.append(
                f"  {row['month']}: {int(row['n_bets'])} bets  ROI={row['roi']:+.3f}  WR={row['win_rate']:.1%}"
            )

    if not calib.empty:
        lines += ["", "CALIBRATION (model prob decile vs actual rate)"]
        for _, row in calib.iterrows():
            lines.append(f"  pred={row['mean_pred']:.3f}  actual={row['actual_rate']:.3f}  n={int(row['n'])}")

    lines += [
        "",
        "VERDICT",
    ]
    roi = metrics.get("roi", 0)
    p   = metrics.get("binom_p_value", 1)
    clv = metrics.get("mean_clv")
    ci_lo = metrics.get("roi_ci_lo_90", 0)
    if roi > 0 and p < 0.10 and ci_lo > -0.02 and (clv is None or clv > 0):
        verdict = "BORDERLINE POSITIVE — edge is small, CIs are wide. Not conclusive at this sample size."
    elif roi > 0.03 and p < 0.05:
        verdict = "POSITIVE — model shows measurable edge on OOS holdout. Worth further investigation."
    elif roi < -0.05 or p > 0.40:
        verdict = "NEGATIVE — no evidence of edge. Do not deploy."
    else:
        verdict = "INCONCLUSIVE — sample too small to distinguish from variance."

    lines.append(f"  {verdict}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    main()
