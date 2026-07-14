"""V2 Phase 2: walk-forward model search.

Folds:
  A: train 2022       -> test 2023   (synthetic lines, prediction metrics)
  B: train 2022-2023  -> test 2024   (synthetic lines, prediction metrics)
  C: train 2022-2024  -> 2025 preds  (real odds rows; betting metrics downstream)

Calibrators and the stacker are fit ONLY on fold A+B out-of-sample predictions,
then applied to 2025. 2026 is never touched here.

Output:
  research/v2/preds_2025.parquet   - per (pitcher,game,line) model probabilities
  research/v2/fold_metrics.csv     - fold A/B prediction metrics per model
  research/v2/models/              - fitted models for the final forward test
"""
from __future__ import annotations
import sys, json, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

OUT = Path("research/v2")
MODELS_DIR = OUT / "models"
SYN_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
SEED = 42


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_data():
    feats = pd.read_parquet(OUT / "features_full.parquet")
    feats["game_date"] = pd.to_datetime(feats["game_date"])
    meta = json.loads((OUT / "meta.json").read_text())
    feat_cols = [c for c in meta["feature_cols"] if c in feats.columns]
    # numeric coercion, residual NaN -> 0
    for c in feat_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats[feat_cols] = feats[feat_cols].fillna(0.0)
    feats["strikeouts"] = pd.to_numeric(feats["strikeouts"], errors="coerce")
    feats = feats.dropna(subset=["strikeouts"])
    feats["year"] = feats.game_date.dt.year

    bets = pd.read_parquet(OUT / "bets.parquet")
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    bets["year"] = bets.game_date.dt.year
    for c in feat_cols:
        if c in bets.columns:
            bets[c] = pd.to_numeric(bets[c], errors="coerce")
    bets[feat_cols] = bets[feat_cols].fillna(0.0)
    # half-lines only (clean win/loss, no pushes)
    bets = bets[(bets["line"] % 1) == 0.5].copy()
    return feats, bets, feat_cols


def expand_lines(games: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Game rows -> game x synthetic-line rows with binary target."""
    reps = []
    for L in SYN_LINES:
        g = games.copy()
        g["line"] = L
        g["y_over"] = (g["strikeouts"] > L).astype(int)
        reps.append(g)
    out = pd.concat(reps, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Count models -> P(over) via NB CDF
# ---------------------------------------------------------------------------

def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments NB2 dispersion alpha: Var = mu + alpha*mu^2."""
    mu = np.maximum(mu, 1e-6)
    alpha = np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2)
    return float(max(alpha, 0.0))


def prob_over_nb(mu: np.ndarray, alpha: float, line: np.ndarray) -> np.ndarray:
    """P(K > line) under NB2(mu, alpha); Poisson if alpha ~ 0."""
    mu = np.maximum(mu, 0.01)
    k = np.floor(line).astype(int)
    if alpha < 1e-6:
        return 1.0 - poisson.cdf(k, mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    return 1.0 - nbinom.cdf(k, r, p)


class CountModel:
    """Wraps a regressor predicting K mean; converts to P(over) with NB tail."""

    def __init__(self, name, est):
        self.name, self.est, self.alpha = name, est, 0.0

    def fit(self, X, y):
        self.est.fit(X, y)
        mu = np.maximum(self.est.predict(X), 0.01)
        self.alpha = fit_dispersion(y.values, mu)
        return self

    def predict_mu(self, X):
        return np.maximum(self.est.predict(X), 0.01)

    def predict_prob(self, X, line):
        return prob_over_nb(self.predict_mu(X), self.alpha, np.asarray(line))


def make_count_models():
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    models = {
        "poisson_glm": Pipeline([
            ("sc", StandardScaler()),
            ("m", PoissonRegressor(alpha=1.0, max_iter=1000)),
        ]),
        "xgb_poisson": XGBRegressor(
            objective="count:poisson", n_estimators=500, max_depth=4,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=20,
            random_state=SEED, verbosity=0, n_jobs=-1),
        "lgb_poisson": LGBMRegressor(
            objective="poisson", n_estimators=600, num_leaves=31,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=2.0, min_child_samples=40,
            random_state=SEED, verbosity=-1, n_jobs=-1),
        "hgb_poisson": HistGradientBoostingRegressor(
            loss="poisson", max_iter=500, max_depth=None, max_leaf_nodes=31,
            learning_rate=0.03, l2_regularization=1.0, min_samples_leaf=40,
            random_state=SEED),
    }
    try:
        from catboost import CatBoostRegressor
        models["cat_poisson"] = CatBoostRegressor(
            loss_function="Poisson", iterations=800, depth=5,
            learning_rate=0.03, l2_leaf_reg=5.0, random_seed=SEED,
            verbose=False, allow_writing_files=False)
    except ImportError:
        pass
    return {k: CountModel(k, v) for k, v in models.items()}


# ---------------------------------------------------------------------------
# Classifier models on game x line rows
# ---------------------------------------------------------------------------

class ClfModel:
    def __init__(self, name, est):
        self.name, self.est = name, est

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict_prob(self, X, line=None):
        return self.est.predict_proba(X)[:, 1]


def make_clf_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier, ExtraTreesClassifier)
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    models = {
        "logistic": Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(C=0.05, max_iter=2000, random_state=SEED)),
        ]),
        "xgb_clf": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=40, eval_metric="logloss",
            random_state=SEED, verbosity=0, n_jobs=-1),
        "lgb_clf": LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
            min_child_samples=80, random_state=SEED, verbosity=-1, n_jobs=-1),
        "hgb_clf": HistGradientBoostingClassifier(
            max_iter=500, max_leaf_nodes=31, learning_rate=0.03,
            l2_regularization=1.0, min_samples_leaf=80, random_state=SEED),
        "rf_clf": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=40,
            max_features=0.3, random_state=SEED, n_jobs=-1),
        "et_clf": ExtraTreesClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=40,
            max_features=0.3, random_state=SEED, n_jobs=-1),
    }
    try:
        from catboost import CatBoostClassifier
        models["cat_clf"] = CatBoostClassifier(
            iterations=800, depth=5, learning_rate=0.03, l2_leaf_reg=5.0,
            random_seed=SEED, verbose=False, allow_writing_files=False)
    except ImportError:
        pass
    return {k: ClfModel(k, v) for k, v in models.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pred_metrics(y, p, tag=""):
    from sklearn.metrics import log_loss, brier_score_loss
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # calibration slope/intercept via logistic fit on logit(p)
    from sklearn.linear_model import LogisticRegression
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "cal_slope": float(lr.coef_[0][0]),
        "cal_intercept": float(lr.intercept_[0]),
        "base_rate": float(np.mean(y)),
        "mean_pred": float(np.mean(p)),
        "n": len(y),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    feats, bets, feat_cols = load_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    count_feat_cols = [c for c in feat_cols if c != "line"]
    clf_feat_cols = count_feat_cols + ["line"]

    folds = [
        ("A", [2022], 2023),
        ("B", [2022, 2023], 2024),
    ]

    count_models = make_count_models()
    clf_models = make_clf_models()
    all_names = list(count_models) + list(clf_models)

    fold_rows = []
    # store fold OOS preds for calibrator/stacker training
    oos_pred_store = {name: [] for name in all_names}
    oos_target_store = []
    oos_line_store = []

    for fold_name, train_years, test_year in folds:
        tr_g = feats[feats.year.isin(train_years)]
        te_g = feats[feats.year == test_year]
        tr_x = expand_lines(tr_g, feat_cols)
        te_x = expand_lines(te_g, feat_cols)
        print(f"\n== Fold {fold_name}: train {train_years} ({len(tr_g)} games) "
              f"-> test {test_year} ({len(te_g)} games) ==", flush=True)

        oos_target_store.append(te_x["y_over"].values)
        oos_line_store.append(te_x["line"].values)

        for name, cm in make_count_models().items():
            t = time.time()
            cm.fit(tr_g[count_feat_cols], tr_g["strikeouts"])
            mu = cm.predict_mu(te_g[count_feat_cols])
            rmse = float(np.sqrt(np.mean((te_g["strikeouts"].values - mu) ** 2)))
            p = cm.predict_prob(te_x[count_feat_cols], te_x["line"])
            m = pred_metrics(te_x["y_over"].values, p)
            m.update({"fold": fold_name, "model": name, "k_rmse": rmse,
                      "alpha": cm.alpha, "secs": round(time.time() - t, 1)})
            fold_rows.append(m)
            oos_pred_store[name].append(p)
            print(f"  {name:14s} rmse={rmse:.3f} logloss={m['logloss']:.4f} "
                  f"slope={m['cal_slope']:.2f} ({m['secs']}s)", flush=True)

        for name, cf in make_clf_models().items():
            t = time.time()
            cf.fit(tr_x[clf_feat_cols], tr_x["y_over"])
            p = cf.predict_prob(te_x[clf_feat_cols])
            m = pred_metrics(te_x["y_over"].values, p)
            m.update({"fold": fold_name, "model": name, "k_rmse": np.nan,
                      "alpha": np.nan, "secs": round(time.time() - t, 1)})
            fold_rows.append(m)
            oos_pred_store[name].append(p)
            print(f"  {name:14s} logloss={m['logloss']:.4f} "
                  f"slope={m['cal_slope']:.2f} ({m['secs']}s)", flush=True)

    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)

    # ---- Calibrators fit on fold A+B OOS preds ----
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    y_oos = np.concatenate(oos_target_store)
    calibrators = {}
    for name in all_names:
        p_oos = np.clip(np.concatenate(oos_pred_store[name]), 1e-6, 1 - 1e-6)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(p_oos, y_oos)
        z = np.log(p_oos / (1 - p_oos)).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_oos)
        calibrators[name] = {"iso": iso, "platt": platt}

    # ---- Stacker: logistic on base-model logits (fit on fold OOS) ----
    stack_base = [n for n in all_names]
    Z_oos = np.column_stack([
        np.log(np.clip(np.concatenate(oos_pred_store[n]), 1e-6, 1 - 1e-6) /
               (1 - np.clip(np.concatenate(oos_pred_store[n]), 1e-6, 1 - 1e-6)))
        for n in stack_base
    ])
    stacker = LogisticRegression(C=0.5, max_iter=2000).fit(Z_oos, y_oos)
    print("\nStacker weights:", dict(zip(stack_base, np.round(stacker.coef_[0], 3))), flush=True)

    # ---- Fold C: train 2022-2024, predict 2025 real odds rows ----
    print("\n== Fold C: train 2022-2024 -> predict 2025 odds rows ==", flush=True)
    tr_g = feats[feats.year <= 2024]
    tr_x = expand_lines(tr_g, feat_cols)
    b25 = bets[bets.year == 2025].copy()

    fitted = {}
    for name, cm in count_models.items():
        t = time.time()
        cm.fit(tr_g[count_feat_cols], tr_g["strikeouts"])
        fitted[name] = cm
        b25[f"p_{name}"] = cm.predict_prob(b25[count_feat_cols], b25["line"])
        b25[f"mu_{name}"] = cm.predict_mu(b25[count_feat_cols])
        print(f"  {name} fitted ({time.time()-t:.0f}s)", flush=True)
    for name, cf in clf_models.items():
        t = time.time()
        cf.fit(tr_x[clf_feat_cols], tr_x["y_over"])
        fitted[name] = cf
        b25[f"p_{name}"] = cf.predict_prob(b25[clf_feat_cols])
        print(f"  {name} fitted ({time.time()-t:.0f}s)", flush=True)

    # calibrated variants
    for name in all_names:
        p = np.clip(b25[f"p_{name}"].values, 1e-6, 1 - 1e-6)
        b25[f"p_{name}_iso"] = calibrators[name]["iso"].predict(p)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        b25[f"p_{name}_platt"] = calibrators[name]["platt"].predict_proba(z)[:, 1]

    # ensembles
    Z25 = np.column_stack([
        np.log(np.clip(b25[f"p_{n}"].values, 1e-6, 1 - 1e-6) /
               (1 - np.clip(b25[f"p_{n}"].values, 1e-6, 1 - 1e-6)))
        for n in stack_base
    ])
    b25["p_stack"] = stacker.predict_proba(Z25)[:, 1]
    count_names = list(count_models)
    b25["p_mean_count"] = b25[[f"p_{n}_iso" for n in count_names]].mean(axis=1)
    b25["p_mean_all"] = b25[[f"p_{n}_iso" for n in all_names]].mean(axis=1)

    keep = [c for c in b25.columns if not c.startswith(("sc_", "opp_", "p_strikeouts_",
            "p_walks", "p_hits", "p_innings", "p_pitches", "p_strikes", "p_batters",
            "p_k_", "p_bb_", "p_deep", "p_short", "p_high", "p_low", "p_bf_",
            "p_strike_rate", "umpire_", "catcher_", "venue_", "park_", "lineup_",
            "batter_", "league_", "bullpen_", "pitcher_ip", "pitcher_starts",
            "fg_", "prior_", "starter_"))]
    b25[keep].to_parquet(OUT / "preds_2025.parquet", index=False)

    with open(MODELS_DIR / "fitted.pkl", "wb") as f:
        pickle.dump({"fitted": fitted, "calibrators": calibrators,
                     "stacker": stacker, "stack_base": stack_base,
                     "count_names": count_names, "all_names": all_names,
                     "feat_cols": feat_cols}, f)

    print(f"\nSaved preds_2025.parquet ({len(b25)} rows) and models. "
          f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
