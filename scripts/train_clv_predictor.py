"""Train and evaluate a CLV-direction predictor for strikeouts.

The predictor answers: "Given what we observe at bet time (T-4h), will the
closing line move in our favour?"  If yes (predicted CLV > 0), we bet.
If no, we skip — even if our edge estimate is positive.

Training split: April-June 2025 (entry data available)
Test split:     July-September 2025

Saves the fitted model to data/processed/models/clv_predictor_strikeouts.joblib
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    return 1 if (actual > row["line"] if row["best_side"] == "over" else actual < row["line"]) else 0


def main():
    # --- Load data ---
    edges = pd.read_csv("data/processed/backtest_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])
    clv = pd.read_csv("data/processed/backtest_clv.csv")
    clv["game_date"] = pd.to_datetime(clv["game_date"])

    q = edges[(edges["edge_pct"] >= 2.0) & (edges["edge_pct"] <= 15.0)].copy()
    q["won"] = q.apply(resolve_outcome, axis=1)
    q = q.dropna(subset=["won"])

    key = ["game_date", "pitcher_id", "market", "line", "best_side"]
    merged = q.merge(clv[key + ["clv_pct"]], on=key, how="left")
    merged = merged.dropna(subset=["clv_pct"])
    merged["clv_positive"] = (merged["clv_pct"] > 0).astype(int)

    sk = merged[merged["market"] == "strikeouts"].copy()
    sk["month"] = sk["game_date"].dt.month
    print(f"Total strikeouts bets with CLV: {len(sk)}")
    print(f"CLV positive rate: {sk['clv_positive'].mean():.3f}")

    # --- Features observable at bet time ---
    feature_candidates = [
        "edge_pct",
        "over_probability",
        "strikeouts_projection",
        "line",
        "days_rest",
        "is_home",
        "pitcher_throws_left",
        "p_k_rate_roll5",
        "p_k_per_ip_roll5",
        "p_strikeouts_roll5",
        "p_strikeouts_roll10",
        "p_k_minus_bb_roll5",
        "sc_csw_rate_roll5",
        "sc_swinging_strike_rate_roll5",
        "sc_velocity_slope_roll8",
        "opp_batting_k_rate_roll5",
        "opp_lineup_k_rate_vs_starter_hand_prior",
        "p_log_expected_bf",
        "park_so_factor",
        "temperature",
    ]
    feat_cols = [f for f in feature_candidates if f in sk.columns]
    print(f"Using {len(feat_cols)} features: {feat_cols}")

    # Add projection-minus-line gap
    if "strikeouts_projection" in sk.columns:
        sk["proj_gap"] = sk["strikeouts_projection"] - sk["line"]
        feat_cols.append("proj_gap")

    # --- Train/test split by month ---
    train = sk[sk["month"] <= 6].copy()
    test = sk[sk["month"] >= 7].copy()
    print(f"\nTrain (Apr-Jun): {len(train)} bets  |  Test (Jul-Sep): {len(test)} bets")

    x_train = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(train[feat_cols].median()).fillna(0)
    x_test = test[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(train[feat_cols].median()).fillna(0)
    y_train = train["clv_positive"]
    y_test = test["clv_positive"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])
    model.fit(x_train, y_train)

    prob_test = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, prob_test)
    print(f"\nTest AUC: {auc:.4f}")
    print(classification_report(y_test, (prob_test > 0.5).astype(int)))

    # --- Evaluate betting performance by predicted CLV threshold ---
    test = test.copy()
    test["pred_clv_prob"] = prob_test
    print("\n=== Betting results by predicted CLV probability threshold ===")
    for thresh in [0.45, 0.50, 0.55, 0.60]:
        filt = test[test["pred_clv_prob"] >= thresh]
        if len(filt) < 20:
            continue
        print(
            f"  pred_CLV >= {thresh}: n={len(filt)}  "
            f"actual_CLV_pos={filt['clv_positive'].mean():.3f}  "
            f"win_rate={filt['won'].mean():.3f}  "
            f"roi={roi_fn(filt):.3f}"
        )

    print("\n=== Baseline (no CLV filter) ===")
    print(
        f"  All: n={len(test)}  win_rate={test['won'].mean():.3f}  roi={roi_fn(test):.3f}"
    )

    # --- Feature importance ---
    coefs = model.named_steps["clf"].coef_[0]
    feat_importance = sorted(zip(feat_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\n=== Top 10 features predicting positive CLV ===")
    for fname, coef in feat_importance[:10]:
        direction = "+" if coef > 0 else "-"
        print(f"  {fname:45s}  {direction}{abs(coef):.3f}")

    # --- Save model ---
    model_path = Path("data/processed/models/clv_predictor_strikeouts.joblib")
    fill_vals = train[feat_cols].median().fillna(0).to_dict()
    joblib.dump(
        {
            "pipeline": model,
            "feature_cols": feat_cols,
            "fill_values": fill_vals,
            "threshold": 0.50,
        },
        model_path,
    )
    print(f"\nCLV predictor saved to {model_path}")


if __name__ == "__main__":
    main()
