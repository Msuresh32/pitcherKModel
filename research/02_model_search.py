"""Phase 3+4: Model search and walk-forward backtest.

Trains multiple models to predict P(over) for each (pitcher, game, line).
Evaluates against market no-vig probability via walk-forward CV.

Splits:
  Train:      All data before 2026-06-01 (used for walk-forward CV)
  Validation: 2026-04-01 to 2026-05-31 (model selection, threshold tuning)
  OOS:        2026-06-01 to 2026-07-07 (never touched during model selection)

Models tested:
  1. Poisson GLM → P(over) via CDF
  2. Negative Binomial GLM → P(over) via CDF
  3. XGBoost direct P(over) classifier
  4. LightGBM direct P(over) classifier
  5. Calibrated ensemble
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DATASET_FILE = Path("research/dataset.parquet")
META_FILE    = Path("research/dataset_meta.json")
OUT_DIR      = Path("research/model_results")

OOS_START    = pd.Timestamp("2026-06-01")
VAL_START    = pd.Timestamp("2026-04-01")
MIN_TRAIN_ROWS = 200   # minimum training rows before any validation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kelly_fraction(p_model: float, p_market: float, odds_american: float, f: float = 0.25) -> float:
    """Quarter-Kelly stake as fraction of bankroll."""
    if odds_american >= 0:
        b = odds_american / 100
    else:
        b = 100 / abs(odds_american)
    edge = p_model * (1 + b) - 1
    if edge <= 0:
        return 0.0
    return f * (edge / b)


def roi_from_bets(bets: pd.DataFrame) -> dict:
    """Compute standard betting metrics from a bets DataFrame."""
    if len(bets) == 0:
        return {}

    def pnl(row):
        if row.won:
            odds = row.get("best_odds", -110)
            if odds >= 0:
                return odds / 100
            else:
                return 100 / abs(odds)
        return -1.0

    bets = bets.copy()
    bets["pnl"] = bets.apply(pnl, axis=1)
    n = len(bets)
    wins = int(bets["won"].sum())
    total_pnl = bets["pnl"].sum()

    # Bootstrap CI on ROI
    boot_rois = [
        bets["pnl"].sample(n, replace=True).mean()
        for _ in range(1000)
    ]
    lo, hi = np.percentile(boot_rois, [5, 95])

    # Max drawdown
    cumsum = bets["pnl"].cumsum()
    roll_max = cumsum.cummax()
    drawdown = (cumsum - roll_max).min()

    return {
        "n_bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "roi": total_pnl / n,
        "units": total_pnl,
        "roi_ci_lo": lo,
        "roi_ci_hi": hi,
        "max_drawdown": drawdown,
    }


def no_vig_edge(p_model: float, p_market: float) -> float:
    return p_model - p_market


# ---------------------------------------------------------------------------
# Feature set definition — only pre-game available features
# ---------------------------------------------------------------------------

CORE_FEATURES = [
    # Rolling pitcher K rate (main signal)
    "p_strikeouts_roll5", "p_strikeouts_roll10",
    "p_k_rate_roll5", "p_k_rate_roll10",
    "p_k_per_ip_roll5", "p_k_per_ip_roll10",
    "p_strikeouts_trimmed_roll5", "p_strikeouts_trimmed_roll10",
    # Statcast stuff quality
    "sc_swinging_strike_rate_roll5", "sc_swinging_strike_rate_roll10",
    "sc_csw_rate_roll5", "sc_csw_rate_roll10",
    "sc_called_strike_rate_roll5",
    "sc_zone_rate_roll5",
    # Pitch mix
    "sc_fastball_pct_roll5", "sc_slider_pct_roll5", "sc_breaking_pct_roll5",
    # Release speed
    "sc_avg_release_speed_roll5",
    # Innings & workload
    "p_innings_pitched_roll5", "p_innings_pitched_roll10",
    "p_batters_faced_roll5",
    "p_deep_start_rate_6ip_roll5", "p_short_start_rate_under5ip_roll5",
    # Career priors
    "p_strikeouts_career_avg_prior", "p_k_rate_career_prior",
    # Opponent K vulnerability (actual column names from build_features)
    "opp_batting_k_rate_roll5", "opp_batting_k_rate_roll20",
    "opp_k_rate_trend",
    # Handedness matchup + lineup K rate
    "opp_lineup_same_hand_batters", "opp_lineup_opposite_hand_batters",
    "opp_lineup_k_rate_roll7", "opp_lineup_k_rate_roll14",
    "opp_lineup_k_rate_roll7_vs_hand",
    # Context
    "umpire_csr_avg_prior", "umpire_csr_excess_prior",
    "catcher_csr_avg_prior",
    "temperature",
    "park_so_factor",  # absent if park_factors.csv missing — skipped automatically
    # Research-specific (from add_research_features and odds join)
    "days_rest", "days_into_season", "month",
    # Line anchor (critical: what are we predicting against?)
    "line",
    # Market prior (open price as feature — reflects consensus pregame knowledge)
    "p_over_open",
    "n_books_open",
]


def get_feature_cols(dataset: pd.DataFrame) -> list[str]:
    """Return CORE_FEATURES that actually exist in dataset."""
    available = set(dataset.columns)
    used = [f for f in CORE_FEATURES if f in available]
    missing = [f for f in CORE_FEATURES if f not in available]
    if missing:
        print(f"  Missing features (will skip): {missing}")
    print(f"  Using {len(used)} / {len(CORE_FEATURES)} core features")
    return used


# ---------------------------------------------------------------------------
# Model 1: Poisson mean → P(over)
# ---------------------------------------------------------------------------

def train_poisson(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Drop line from regression target (line is not a predictor for K count)
    X_reg = X_train.drop(columns=["line", "p_over_open", "n_books_open"], errors="ignore")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=0.1, max_iter=500)),
    ])
    pipe.fit(X_reg, y_train)
    return pipe


def predict_poisson_prob(model, X: pd.DataFrame, line: pd.Series) -> pd.Series:
    """P(actual > line) using fitted Poisson mean."""
    from scipy.stats import poisson
    X_reg = X.drop(columns=["line", "p_over_open", "n_books_open"], errors="ignore")
    mu = pd.Series(np.maximum(model.predict(X_reg), 0.01), index=X.index)
    probs = pd.Series(
        [1 - poisson.cdf(int(l), mu_i) for l, mu_i in zip(line, mu)],
        index=X.index,
    )
    return probs


# ---------------------------------------------------------------------------
# Model 2: XGBoost direct P(over) classifier
# ---------------------------------------------------------------------------

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  xgboost not installed, skipping")
        return None

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    # Calibrate with isotonic regression to get proper probabilities
    cal = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


# ---------------------------------------------------------------------------
# Model 3: LightGBM direct P(over) classifier
# ---------------------------------------------------------------------------

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm not installed, skipping")
        return None

    from sklearn.calibration import CalibratedClassifierCV

    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    cal = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


# ---------------------------------------------------------------------------
# Model 4: Logistic regression (baseline)
# ---------------------------------------------------------------------------

def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    predictions: pd.Series,
    dataset: pd.DataFrame,
    edge_threshold: float = 0.03,
    min_prob: float = 0.52,
) -> dict:
    """Evaluate betting performance from model predictions vs market."""
    df = dataset.copy()
    df["p_model"] = predictions.values

    # Use closing market price for comparison
    df["p_market"] = df["p_over_close"]
    df = df.dropna(subset=["p_model", "p_market", "p_over_close"])

    # Over bets: model says more likely to go over than market
    over_bets = df[
        (df["p_model"] - df["p_market"] >= edge_threshold) &
        (df["p_model"] >= min_prob) &
        df["best_over_odds_close"].notna()
    ].copy()
    over_bets["bet_side"] = "over"
    over_bets["won"] = over_bets["outcome_over"] == 1
    over_bets["best_odds"] = over_bets["best_over_odds_close"]
    over_bets["edge"] = over_bets["p_model"] - over_bets["p_market"]

    # Under bets: model says more likely to go under
    under_bets = df[
        (df["p_market"] - df["p_model"] >= edge_threshold) &
        ((1 - df["p_model"]) >= min_prob) &
        df["best_under_odds_close"].notna()
    ].copy()
    under_bets["bet_side"] = "under"
    under_bets["won"] = under_bets["outcome_over"] == 0
    under_bets["best_odds"] = under_bets["best_under_odds_close"]
    under_bets["edge"] = under_bets["p_market"] - under_bets["p_model"]

    all_bets = pd.concat([over_bets, under_bets], ignore_index=True)

    if len(all_bets) == 0:
        return {"n_bets": 0}

    metrics = roi_from_bets(all_bets)

    # CLV
    if "clv_over" in all_bets.columns:
        over_clv  = all_bets.loc[all_bets.bet_side == "over",  "clv_over"].mean()
        under_clv = (-all_bets.loc[all_bets.bet_side == "under", "clv_over"]).mean()
        all_clv = pd.concat([
            all_bets.loc[all_bets.bet_side == "over",  "clv_over"],
            -all_bets.loc[all_bets.bet_side == "under", "clv_over"],
        ])
        metrics["mean_clv"] = float(all_clv.mean())

    metrics["over_bets"]  = len(over_bets)
    metrics["under_bets"] = len(under_bets)
    metrics["bets_df"] = all_bets

    return metrics


def walk_forward_evaluate(
    dataset: pd.DataFrame,
    feat_cols: list[str],
    model_fn,
    predict_fn,
    edge_threshold: float = 0.03,
    min_prob: float = 0.52,
    fold_months: int = 2,
) -> tuple[dict, pd.DataFrame]:
    """Walk-forward evaluation on validation period (pre-OOS)."""

    val_data = dataset[
        (dataset["game_date"] >= VAL_START) &
        (dataset["game_date"] < OOS_START)
    ].copy()

    if len(val_data) == 0:
        return {"n_bets": 0}, pd.DataFrame()

    # Sort by date; train on everything before each fold
    val_data = val_data.sort_values("game_date")

    # Train on all pre-validation data
    train_data = dataset[dataset["game_date"] < VAL_START].copy()

    if len(train_data) < MIN_TRAIN_ROWS:
        print(f"  Insufficient training data: {len(train_data)} rows")
        return {"n_bets": 0}, pd.DataFrame()

    X_train = train_data[feat_cols].fillna(0)
    y_train = train_data["outcome_over"]

    print(f"  Training on {len(X_train):,} rows...")
    model = model_fn(X_train, y_train)
    if model is None:
        return {"n_bets": 0}, pd.DataFrame()

    X_val = val_data[feat_cols].fillna(0)
    preds = predict_fn(model, X_val, val_data["line"])

    metrics = evaluate_model(preds, val_data, edge_threshold, min_prob)
    all_bets = metrics.pop("bets_df", pd.DataFrame())

    return metrics, all_bets


# ---------------------------------------------------------------------------
# Threshold grid search (on validation, NOT OOS)
# ---------------------------------------------------------------------------

def grid_search_thresholds(
    val_preds: pd.Series,
    val_data: pd.DataFrame,
) -> pd.DataFrame:
    """Search edge and min_prob thresholds on validation period."""
    results = []
    for edge_thresh in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        for min_p in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            m = evaluate_model(val_preds, val_data, edge_thresh, min_p)
            bets_df = m.pop("bets_df", None)
            results.append({
                "edge_threshold": edge_thresh,
                "min_prob": min_p,
                **m,
            })
    df = pd.DataFrame(results)
    return df.sort_values("roi", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATASET_FILE.exists():
        print(f"ERROR: Run 01_build_dataset.py first. {DATASET_FILE} not found.")
        return

    print("=== Loading dataset ===")
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    print(f"Total rows: {len(dataset):,}")
    print(f"Train period: {dataset[dataset.game_date < VAL_START].game_date.max().date()} (end)")
    print(f"Val period:   {VAL_START.date()} to {(OOS_START - pd.Timedelta(days=1)).date()}")
    print(f"OOS period:   {OOS_START.date()} to {dataset.game_date.max().date()}")

    feat_cols = get_feature_cols(dataset)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic": (train_logistic, lambda m, X, L: pd.Series(m.predict_proba(X)[:, 1], index=X.index)),
        "poisson": (train_poisson, predict_poisson_prob),
        "xgboost": (train_xgboost, lambda m, X, L: pd.Series(m.predict_proba(X)[:, 1], index=X.index)),
        "lightgbm": (train_lightgbm, lambda m, X, L: pd.Series(m.predict_proba(X)[:, 1], index=X.index)),
    }

    val_results = {}
    val_predictions = {}

    for name, (train_fn, pred_fn) in models.items():
        print(f"\n=== Model: {name} ===")
        metrics, bets = walk_forward_evaluate(
            dataset, feat_cols, train_fn, pred_fn,
            edge_threshold=0.04, min_prob=0.54,
        )
        val_results[name] = metrics
        if not bets.empty:
            bets.to_csv(OUT_DIR / f"val_bets_{name}.csv", index=False)
        print(f"  Validation: {metrics}")

    # Save validation summary
    with open(OUT_DIR / "val_summary.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, pd.DataFrame)}
             for k, v in val_results.items()},
            f, indent=2, default=str,
        )

    # Grid search on best model
    best_model_name = max(
        {k: v for k, v in val_results.items() if v.get("n_bets", 0) >= 20},
        key=lambda k: val_results[k].get("roi", -1),
        default=None,
    )
    if best_model_name:
        print(f"\n=== Best model: {best_model_name} ===")
        print(f"Validation metrics: {val_results[best_model_name]}")

        # Retrain on all pre-OOS data for grid search
        train_fn, pred_fn = models[best_model_name]
        pre_oos = dataset[dataset["game_date"] < OOS_START].copy()
        X_pre = pre_oos[feat_cols].fillna(0)
        y_pre = pre_oos["outcome_over"]
        best_model = train_fn(X_pre, y_pre)

        val_set = pre_oos[pre_oos["game_date"] >= VAL_START].copy()
        X_val = val_set[feat_cols].fillna(0)
        val_preds = pred_fn(best_model, X_val, val_set["line"])
        val_preds.index = val_set.index

        print("\n=== Grid search on validation ===")
        grid = grid_search_thresholds(val_preds, val_set)
        grid.to_csv(OUT_DIR / "threshold_grid.csv", index=False)
        print(grid.head(10).to_string())

        # Save best model name and best threshold for OOS eval
        best_thresh_row = grid[grid["n_bets"] >= 30].iloc[0] if len(grid[grid["n_bets"] >= 30]) else grid.iloc[0]
        config_out = {
            "best_model": best_model_name,
            "best_edge_threshold": float(best_thresh_row["edge_threshold"]),
            "best_min_prob": float(best_thresh_row["min_prob"]),
        }
        with open(OUT_DIR / "best_config.json", "w") as f:
            json.dump(config_out, f, indent=2)
        print(f"\nBest config saved: {config_out}")
    else:
        print("\nNo model met minimum bet count on validation. Check data.")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
