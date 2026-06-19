"""
Quantifies the lineup timing leak:
1. Which lineup features made the top-100 and their importances
2. Correlation of lineup features to actual strikeout outcomes
3. Backtest comparison: with vs without lineup features
"""
import joblib, json, pandas as pd, numpy as np
from pathlib import Path

SEP = "=" * 65

# ── 1. FEATURE IMPORTANCES ────────────────────────────────────────
print(f"\n{SEP}")
print("1. FEATURE IMPORTANCES — lineup features in top-100")
print(SEP)

model_obj = joblib.load("data/processed/models/strikeouts.joblib")
print("Model type:", type(model_obj))

# Could be a dict with model + metadata
if isinstance(model_obj, dict):
    model     = model_obj.get("model") or model_obj.get("estimator")
    feat_names = model_obj.get("feature_names") or model_obj.get("selected_features") or model_obj.get("features")
    print("Dict keys:", list(model_obj.keys()))
else:
    model = model_obj
    feat_names = None

# Try to extract importances
if hasattr(model, "feature_importances_"):
    imps = model.feature_importances_
elif hasattr(model, "estimators_"):       # Voting/ensemble
    imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
elif hasattr(model, "named_steps"):       # Pipeline
    for name, step in model.named_steps.items():
        if hasattr(step, "feature_importances_"):
            imps = step.feature_importances_; break
    else:
        imps = None
else:
    imps = None

if feat_names and imps is not None and len(feat_names) == len(imps):
    fi = pd.Series(imps, index=feat_names).sort_values(ascending=False)
    lineup_fi = fi[fi.index.str.startswith("opp_lineup")]
    print(f"\n  Total features: {len(fi)}")
    print(f"  Lineup features present: {len(lineup_fi)}")
    if len(lineup_fi) > 0:
        print(f"\n  Lineup feature importances:")
        for feat, imp in lineup_fi.items():
            rank = list(fi.index).index(feat) + 1
            print(f"    rank={rank:>4}  imp={imp:.5f}  {feat}")
        total_lineup_imp = lineup_fi.sum()
        print(f"\n  Total lineup importance: {total_lineup_imp:.4f}  ({total_lineup_imp/fi.sum()*100:.1f}% of model)")
    else:
        print("  No lineup features in model (may have been deselected)")
    print(f"\n  Top 20 features overall:")
    for feat, imp in fi.head(20).items():
        tag = " ← LINEUP" if feat.startswith("opp_lineup") else ""
        print(f"    {imp:.5f}  {feat}{tag}")
elif feat_names:
    print(f"  Feature names found ({len(feat_names)}), but no importances")
    lineup_feats = [f for f in feat_names if "lineup" in f]
    print(f"  Lineup features selected: {len(lineup_feats)}")
    for f in lineup_feats: print(f"    {f}")
else:
    print("  Could not extract feature names/importances from model object")
    print("  Trying train_metrics.csv...")
    tm = Path("data/processed/train_metrics.csv")
    if tm.exists():
        df = pd.read_csv(tm)
        print(df.head(10))

# ── 2. CORRELATION TEST ───────────────────────────────────────────
print(f"\n{SEP}")
print("2. LINEUP FEATURE CORRELATIONS TO STRIKEOUTS")
print(SEP)

pred_file = Path("data/processed/exp_pred_bk_20260101_20260616_b70_30_predictions.csv")
if pred_file.exists():
    preds = pd.read_csv(pred_file)
    lineup_cols = [c for c in preds.columns if c.startswith("opp_lineup")]
    print(f"  Lineup columns in predictions: {len(lineup_cols)}")
    if "strikeouts" in preds.columns:
        print(f"\n  {'Feature':<50} {'corr to K':>10}")
        print("  " + "-"*62)
        for col in lineup_cols:
            if preds[col].std() > 0:
                c = preds["strikeouts"].corr(preds[col])
                print(f"  {col:<50} {c:>+10.4f}")

# ── 3. BACKTEST: WITH vs WITHOUT LINEUP FEATURES ─────────────────
print(f"\n{SEP}")
print("3. BACKTEST COMPARISON — lineup features in/out of top-100")
print(SEP)
print("  (Checking if lineup features were even selected by RF selector)")

# Check train_metrics for feature selection info
for metrics_path in ["data/processed/train_metrics.csv",
                     "data/processed_2024/train_metrics.csv"]:
    mp = Path(metrics_path)
    if mp.exists():
        df = pd.read_csv(mp)
        print(f"\n  {metrics_path}:")
        print(df.to_string(index=False))

# ── 4. LIVE vs TRAINING — how lineup data is fetched ──────────────
print(f"\n{SEP}")
print("4. LIVE PREDICTION — lineup source check")
print(SEP)

import ast
proj_daily = Path("scripts/project_daily.py")
if proj_daily.exists():
    src = proj_daily.read_text()
    # Find lineup-related lines
    for i, line in enumerate(src.split("\n"), 1):
        if "lineup" in line.lower() and not line.strip().startswith("#"):
            print(f"  line {i:>4}: {line.rstrip()}")
