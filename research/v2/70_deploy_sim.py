"""Deployment simulation: annual-retrain policy applied to 2026.

Identical architecture + hyperparameters + FROZEN filter (final_config.json).
Count models retrained on 2022-2025 (as the production annual retrain would
have done before 2026 opening day). Isotonic calibrators fit on walk-forward
OOS predictions only: 2023 (trained on 22), 2024 (trained on 22-23), and 2025
real-line preds (trained on 22-24, from preds_2025.parquet).

ZERO 2026 information enters training, calibration, or filter choice.
Pre-declared single run for the final report; no iteration on its output.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = __import__("30_strategy")
MS = __import__("20_model_search")

OUT = Path("research/v2")
FORWARD_END = pd.Timestamp("2026-07-10")


def main():
    cfg = json.loads((OUT / "final_config.json").read_text())
    feats, bets, feat_cols = MS.load_data()
    count_feat_cols = [c for c in feat_cols if c != "line"]

    # --- calibration training data: WF OOS preds from 2023/2024 synthetic ---
    folds = [([2022], 2023), ([2022, 2023], 2024)]
    oos_p = {n: [] for n in
             ["poisson_glm", "xgb_poisson", "lgb_poisson", "hgb_poisson", "cat_poisson"]}
    oos_y = []
    for train_years, test_year in folds:
        tr = feats[feats.year.isin(train_years)]
        te = feats[feats.year == test_year]
        te_x = MS.expand_lines(te, feat_cols)
        oos_y.append(te_x["y_over"].values)
        for name, cm in MS.make_count_models().items():
            cm.fit(tr[count_feat_cols], tr["strikeouts"])
            oos_p[name].append(cm.predict_prob(te_x[count_feat_cols], te_x["line"]))
        print(f"fold {train_years}->{test_year} done", flush=True)

    # --- plus 2025 real-line OOS preds from the 22-24 model (already computed) ---
    p25 = pd.read_parquet(OUT / "preds_2025.parquet")
    p25 = p25[p25["outcome_push"] == 0]
    oos_y.append(p25["outcome_over"].values)
    for name in oos_p:
        oos_p[name].append(p25[f"p_{name}"].values)

    y = np.concatenate(oos_y)
    from sklearn.isotonic import IsotonicRegression
    calibrators = {}
    for name in oos_p:
        p = np.clip(np.concatenate(oos_p[name]), 1e-6, 1 - 1e-6)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(p, y)
        calibrators[name] = iso

    # --- retrain on 2022-2025, predict 2026 ---
    tr = feats[feats.year <= 2025]
    b26 = bets[(bets.year == 2026) & (bets.game_date <= FORWARD_END)].copy()
    b26 = b26[b26["outcome_push"] == 0]
    print(f"train games 2022-2025: {len(tr):,} | 2026 rows: {len(b26):,}", flush=True)

    iso_cols = []
    for name, cm in MS.make_count_models().items():
        cm.fit(tr[count_feat_cols], tr["strikeouts"])
        p = cm.predict_prob(b26[count_feat_cols], b26["line"])
        b26[f"p_{name}_iso"] = calibrators[name].predict(np.clip(p, 1e-6, 1 - 1e-6))
        iso_cols.append(f"p_{name}_iso")
        print(f"  {name} retrained", flush=True)
    b26["p_mean_count"] = b26[iso_cols].mean(axis=1)

    bx = S.make_bets(b26, "p_mean_count", min_edge=cfg["min_edge"],
                     sides=cfg["sides"], min_books=cfg["min_books"])
    m = S.bet_metrics(bx)
    print("\n=== DEPLOYMENT SIM: retrained-through-2025, frozen filter, 2026 ===",
          flush=True)
    for k, v in m.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}",
              flush=True)
    if len(bx):
        bx["month"] = bx.game_date.dt.to_period("M").astype(str)
        print(bx.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
              roi=("pnl", "mean"), clv=("clv", "mean")).round(4).to_string(), flush=True)
        bx.to_csv(OUT / "deploy_sim_2026_bets.csv", index=False)

    # calibration check on all 2026 rows
    d = b26.dropna(subset=["p_mean_count"]).copy()
    d["bucket"] = pd.qcut(d["p_mean_count"], 10, duplicates="drop")
    print("\nCalibration (all 2026 rows, retrained model):", flush=True)
    print(d.groupby("bucket", observed=True).agg(
        pred=("p_mean_count", "mean"), actual=("outcome_over", "mean"),
        n=("p_mean_count", "size")).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
