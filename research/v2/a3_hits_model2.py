"""Hits model V2: retrain with hits-specific features, compare vs baseline.

Baseline (research/v2/96-98): fold A/B logloss 0.515/0.501; 2025 frozen filter
(UNDER-only, edge>=0.04): n=660 ROI +8.8% [+2.0,+15.3] CLV +1.01pp.

This script: same protocol on features_hits.parquet (407 base + new cols).
2026 NOT touched. Stores 2025 preds + fold metrics + per-game mus/vintages
for a later one-shot 2026 confirmation (only if 2025 materially improves).
"""
from __future__ import annotations
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
MS = __import__("20_model_search")
S = __import__("30_strategy")
H = __import__("97_hits_strategy_2025")

OUT = Path("research/v2")
SYN_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
TARGET = "hits_allowed"
COUNT_NAMES = ["poisson_glm", "xgb_poisson", "lgb_poisson", "hgb_poisson", "cat_poisson"]
VINTAGES_2026 = [pd.Timestamp(x) for x in
                 ["2026-01-01", "2026-05-01", "2026-06-01", "2026-07-01"]]
FORWARD_END = pd.Timestamp("2026-07-11")


def expand_lines_t(games, feat_cols, target):
    reps = []
    for L in SYN_LINES:
        g = games.copy()
        g["line"] = L
        g["y_over"] = (g[target] > L).astype(int)
        reps.append(g)
    return pd.concat(reps, ignore_index=True)


def main():
    t0 = time.time()
    feats = pd.read_parquet(OUT / "features_hits.parquet")
    feats["game_date"] = pd.to_datetime(feats["game_date"])
    meta = json.loads((OUT / "meta.json").read_text())
    hmeta = json.loads((OUT / "hits_feature_meta.json").read_text())
    feat_cols = [c for c in meta["feature_cols"] if c in feats.columns and c != "line"]
    feat_cols += [c for c in hmeta["new_cols"] if c in feats.columns]
    for c in feat_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats[feat_cols] = feats[feat_cols].fillna(0.0)

    logs = pd.read_csv("data/raw/pitcher_game_logs.csv")
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs["pitcher_id"] = pd.to_numeric(logs["pitcher_id"], errors="coerce")
    logs = logs.drop_duplicates(subset=["game_date", "pitcher_id"])
    feats["pitcher_id"] = pd.to_numeric(feats["pitcher_id"], errors="coerce")
    if TARGET not in feats.columns:
        feats = feats.merge(logs[["game_date", "pitcher_id", TARGET]],
                            on=["game_date", "pitcher_id"], how="left")
    feats[TARGET] = pd.to_numeric(feats[TARGET], errors="coerce")
    feats = feats.dropna(subset=[TARGET])
    feats["year"] = feats.game_date.dt.year
    print(f"{len(feats):,} games | {len(feat_cols)} features "
          f"({len(hmeta['new_cols'])} new)", flush=True)

    # ---- folds A/B ----
    oos_p = {n: [] for n in COUNT_NAMES}
    oos_y = []
    for fold, train_years, test_year in [("A", [2022], 2023),
                                         ("B", [2022, 2023], 2024)]:
        tr = feats[feats.year.isin(train_years)]
        te = feats[feats.year == test_year]
        te_x = expand_lines_t(te, feat_cols, TARGET)
        oos_y.append(te_x["y_over"].values)
        for name, cm in MS.make_count_models().items():
            cm.fit(tr[feat_cols], tr[TARGET])
            mu = cm.predict_mu(te[feat_cols])
            rmse = float(np.sqrt(np.mean((te[TARGET].values - mu) ** 2)))
            p = cm.predict_prob(te_x[feat_cols], te_x["line"])
            m = MS.pred_metrics(te_x["y_over"].values, p)
            oos_p[name].append(p)
            print(f"fold {fold} {name:13s} rmse={rmse:.3f} "
                  f"logloss={m['logloss']:.4f} slope={m['cal_slope']:.2f}", flush=True)

    from sklearn.isotonic import IsotonicRegression
    y_cal = np.concatenate(oos_y)
    iso = {}
    for n in COUNT_NAMES:
        p = np.clip(np.concatenate(oos_p[n]), 1e-6, 1 - 1e-6)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        ir.fit(p, y_cal)
        iso[n] = ir

    # ---- 2025 eval with FROZEN filter ----
    mkt = H.load_market()
    tr = feats[feats.year <= 2024]
    models, alphas = {}, {}
    for name, cm in MS.make_count_models().items():
        cm.fit(tr[feat_cols], tr[TARGET])
        models[name] = cm
        alphas[("2025", name)] = cm.alpha
    f25 = feats[feats.year == 2025].copy()
    j = f25.merge(mkt, on=["game_date", "pitcher_id"], how="inner")
    j = j[(j.line % 1) == 0.5]
    j = j[j.game_date.dt.year == 2025]
    j["outcome_over"] = (j[TARGET] > j.line).astype(float)
    j["outcome_push"] = 0.0
    for n in COUNT_NAMES:
        p = MS.prob_over_nb(models[n].predict_mu(j[feat_cols]),
                            alphas[("2025", n)], j["line"].values)
        j[f"p_{n}"] = iso[n].predict(np.clip(p, 1e-6, 1 - 1e-6))
    j["p_mean_count"] = j[[f"p_{n}" for n in COUNT_NAMES]].mean(axis=1)

    print("\n=== 2025 frozen filter (UNDER-only, edge>=0.04) — V2 features ===", flush=True)
    b = S.make_bets(j, "p_mean_count", min_edge=0.04, sides="under")
    m = S.bet_metrics(b)
    for k, v in m.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}", flush=True)
    print("  BASELINE: n=660 roi=+0.0875 [+0.021,+0.152] clv=+0.0101", flush=True)
    b["month"] = b.game_date.dt.to_period("M").astype(str)
    print(b.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
          roi=("pnl", "mean"), clv=("clv", "mean")).round(4).to_string(), flush=True)
    j.to_parquet(OUT / "hits2_preds_2025.parquet", index=False)

    # ---- store 2026 walk-forward mus for potential one-shot confirmation ----
    mu_rows = []
    for ci, cutoff in enumerate(VINTAGES_2026):
        nxt = VINTAGES_2026[ci + 1] if ci + 1 < len(VINTAGES_2026) else \
            FORWARD_END + pd.Timedelta(days=1)
        rows26 = feats[(feats.year == 2026) & (feats.game_date < nxt)]
        if ci > 0:
            rows26 = rows26[rows26.game_date >= cutoff]
        if len(rows26) == 0:
            continue
        trv = feats[feats.game_date < cutoff]
        vmodels = {}
        for name, cm in MS.make_count_models().items():
            cm.fit(trv[feat_cols], trv[TARGET])
            vmodels[name] = cm
            alphas[(str(cutoff.date()), name)] = cm.alpha
        rec = rows26[["game_date", "pitcher_id", TARGET]].copy()
        rec["vintage"] = str(cutoff.date())
        for name, cm in vmodels.items():
            rec[f"mu_{name}"] = cm.predict_mu(rows26[feat_cols])
        mu_rows.append(rec)
    pd.concat(mu_rows, ignore_index=True).to_parquet(OUT / "hits2_mu_2026.parquet",
                                                     index=False)
    with open(OUT / "hits2_models.pkl", "wb") as f:
        pickle.dump({"iso": iso, "alphas": alphas, "feat_cols": feat_cols}, f)
    print(f"\nstored 2026 mus (NOT evaluated) | total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
