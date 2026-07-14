"""V2 Phase 5: feature importance, stability across seasons, SHAP.

Uses ONLY 2022-2025 data (no 2026). Importance is computed out-of-sample:
train 2022-2023, permute on 2024; and train 2022-2024, permute on 2025 rows.
Stability = rank agreement of importance between the two OOS years.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("research/v2")
SEED = 42


def main():
    feats = pd.read_parquet(OUT / "features_full.parquet")
    feats["game_date"] = pd.to_datetime(feats["game_date"])
    feats["year"] = feats.game_date.dt.year
    meta = json.loads((OUT / "meta.json").read_text())
    feat_cols = [c for c in meta["feature_cols"] if c in feats.columns and c != "line"]
    for c in feat_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats[feat_cols] = feats[feat_cols].fillna(0.0)
    feats["strikeouts"] = pd.to_numeric(feats["strikeouts"], errors="coerce")
    feats = feats.dropna(subset=["strikeouts"])
    feats = feats[feats.year <= 2025]

    from lightgbm import LGBMRegressor

    def fit_model(train):
        m = LGBMRegressor(objective="poisson", n_estimators=600, num_leaves=31,
                          learning_rate=0.03, subsample=0.8, colsample_bytree=0.6,
                          reg_alpha=0.5, reg_lambda=2.0, min_child_samples=40,
                          random_state=SEED, verbosity=-1, n_jobs=-1)
        m.fit(train[feat_cols], train["strikeouts"])
        return m

    def perm_importance(model, test, n_rep=5):
        rng = np.random.default_rng(SEED)
        base = test["strikeouts"].values
        mu0 = np.maximum(model.predict(test[feat_cols]), 0.01)
        # poisson deviance as loss
        def dev(mu):
            return 2 * np.mean(np.where(base > 0, base * np.log(base / mu), 0) - (base - mu))
        loss0 = dev(mu0)
        rows = []
        X = test[feat_cols].reset_index(drop=True)
        for c in feat_cols:
            deltas = []
            for _ in range(n_rep):
                Xp = X.copy()
                Xp[c] = rng.permutation(Xp[c].values)
                mu = np.maximum(model.predict(Xp), 0.01)
                deltas.append(dev(mu) - loss0)
            rows.append({"feature": c, "perm_delta": float(np.mean(deltas))})
        return pd.DataFrame(rows)

    results = {}
    for tag, train_years, test_year in [("oos2024", [2022, 2023], 2024),
                                        ("oos2025", [2022, 2023, 2024], 2025)]:
        print(f"== {tag}: train {train_years} -> permute {test_year} ==", flush=True)
        m = fit_model(feats[feats.year.isin(train_years)])
        te = feats[feats.year == test_year]
        imp = perm_importance(m, te)
        imp = imp.sort_values("perm_delta", ascending=False).reset_index(drop=True)
        imp["rank"] = np.arange(1, len(imp) + 1)
        results[tag] = imp
        print(imp.head(25).to_string(index=False), flush=True)

    merged = results["oos2024"].merge(results["oos2025"], on="feature",
                                      suffixes=("_24", "_25"))
    rho = merged[["perm_delta_24", "perm_delta_25"]].corr(method="spearman").iloc[0, 1]
    print(f"\nImportance rank stability (Spearman 2024 vs 2025): {rho:.3f}", flush=True)
    merged.to_csv(OUT / "feature_importance.csv", index=False)

    # positive in BOTH years = stable useful; negative in both = candidate drop
    merged["both_pos"] = (merged.perm_delta_24 > 0) & (merged.perm_delta_25 > 0)
    merged["both_neg"] = (merged.perm_delta_24 <= 0) & (merged.perm_delta_25 <= 0)
    print(f"Features helpful both years: {merged.both_pos.sum()} / {len(merged)}", flush=True)
    print(f"Features useless/harmful both years: {merged.both_neg.sum()}", flush=True)

    # SHAP on final train (2022-2024) sample
    try:
        import shap
        m = fit_model(feats[feats.year <= 2024])
        samp = feats[feats.year == 2025][feat_cols].sample(
            min(2000, (feats.year == 2025).sum()), random_state=SEED)
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(samp)
        mean_abs = pd.Series(np.abs(sv).mean(0), index=feat_cols).sort_values(ascending=False)
        mean_abs.to_csv(OUT / "shap_mean_abs.csv")
        print("\nTop 25 SHAP mean|values| (2025 rows, model trained 2022-24):", flush=True)
        print(mean_abs.head(25).to_string(), flush=True)
    except Exception as e:
        print(f"SHAP skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
