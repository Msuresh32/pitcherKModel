"""Hits-allowed model: walk-forward count-model search, mirroring the K pipeline.

Splits (same mandate as Ks):
  Fold A: train 2022      -> test 2023   (synthetic lines 3.5-8.5)
  Fold B: train 2022-23   -> test 2024
  Fold C: train 2022-24   -> store 2025 per-game mus (odds joined later)
  2026:   monthly expanding vintages (walk-forward), per-game mus stored

Outputs:
  research/v2/hits_fold_metrics.csv
  research/v2/hits_mu.parquet          (per pitcher-game mu for 5 models + vintage)
  research/v2/hits_models.pkl          (iso calibrators + alphas)
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

OUT = Path("research/v2")
SYN_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
TARGET = "hits_allowed"
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
    feats = pd.read_parquet(OUT / "features_full.parquet")
    feats["game_date"] = pd.to_datetime(feats["game_date"])
    meta = json.loads((OUT / "meta.json").read_text())
    feat_cols = [c for c in meta["feature_cols"] if c in feats.columns and c != "line"]
    for c in feat_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats[feat_cols] = feats[feat_cols].fillna(0.0)

    # target: hits_allowed from raw logs
    logs = pd.read_csv("data/raw/pitcher_game_logs.csv")
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs["pitcher_id"] = pd.to_numeric(logs["pitcher_id"], errors="coerce")
    logs = logs.drop_duplicates(subset=["game_date", "pitcher_id"])
    feats["pitcher_id"] = pd.to_numeric(feats["pitcher_id"], errors="coerce")
    feats = feats.merge(logs[["game_date", "pitcher_id", TARGET]],
                        on=["game_date", "pitcher_id"], how="left",
                        suffixes=("", "_log"))
    tcol = TARGET if TARGET in feats.columns else f"{TARGET}_log"
    feats[TARGET] = pd.to_numeric(feats[tcol], errors="coerce")
    feats = feats.dropna(subset=[TARGET])
    feats["year"] = feats.game_date.dt.year
    print(f"{len(feats):,} games with {TARGET} | mean={feats[TARGET].mean():.2f} "
          f"var={feats[TARGET].var():.2f}", flush=True)

    # ---- folds A/B ----
    fold_rows = []
    oos_p = {n: [] for n in ["poisson_glm", "xgb_poisson", "lgb_poisson",
                             "hgb_poisson", "cat_poisson"]}
    oos_y = []
    for fold, train_years, test_year in [("A", [2022], 2023),
                                         ("B", [2022, 2023], 2024)]:
        tr = feats[feats.year.isin(train_years)]
        te = feats[feats.year == test_year]
        te_x = expand_lines_t(te, feat_cols, TARGET)
        oos_y.append(te_x["y_over"].values)
        print(f"== fold {fold}: {len(tr)} -> {len(te)} ==", flush=True)
        for name, cm in MS.make_count_models().items():
            t = time.time()
            cm.fit(tr[feat_cols], tr[TARGET])
            mu = cm.predict_mu(te[feat_cols])
            rmse = float(np.sqrt(np.mean((te[TARGET].values - mu) ** 2)))
            p = cm.predict_prob(te_x[feat_cols], te_x["line"])
            m = MS.pred_metrics(te_x["y_over"].values, p)
            m.update({"fold": fold, "model": name, "rmse": rmse, "alpha": cm.alpha})
            fold_rows.append(m)
            oos_p[name].append(p)
            print(f"  {name:14s} rmse={rmse:.3f} logloss={m['logloss']:.4f} "
                  f"slope={m['cal_slope']:.2f} alpha={cm.alpha:.3f} "
                  f"({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(fold_rows).to_csv(OUT / "hits_fold_metrics.csv", index=False)

    from sklearn.isotonic import IsotonicRegression
    y_cal = np.concatenate(oos_y)
    iso = {}
    for n in oos_p:
        p = np.clip(np.concatenate(oos_p[n]), 1e-6, 1 - 1e-6)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        ir.fit(p, y_cal)
        iso[n] = ir

    # ---- per-game mus: 2025 (22-24 vintage) + 2026 monthly vintages ----
    mu_rows = []
    alphas = {}

    def store(rows_df, models, vintage_tag):
        rec = rows_df[["game_date", "pitcher_id", TARGET]].copy()
        rec["vintage"] = vintage_tag
        for name, cm in models.items():
            rec[f"mu_{name}"] = cm.predict_mu(rows_df[feat_cols])
        mu_rows.append(rec)

    print("== fold C: train 2022-24 -> 2025 mus ==", flush=True)
    tr = feats[feats.year <= 2024]
    models = {}
    for name, cm in MS.make_count_models().items():
        cm.fit(tr[feat_cols], tr[TARGET])
        models[name] = cm
        alphas[("2025", name)] = cm.alpha
    store(feats[feats.year == 2025], models, "2025")

    for ci, cutoff in enumerate(VINTAGES_2026):
        nxt = VINTAGES_2026[ci + 1] if ci + 1 < len(VINTAGES_2026) else \
            FORWARD_END + pd.Timedelta(days=1)
        rows26 = feats[(feats.game_date >= max(cutoff, pd.Timestamp("2026-01-01"))) &
                       (feats.game_date < nxt) & (feats.year == 2026)]
        if ci == 0:
            rows26 = feats[(feats.year == 2026) & (feats.game_date < nxt)]
        if len(rows26) == 0:
            continue
        tr = feats[feats.game_date < cutoff]
        models = {}
        for name, cm in MS.make_count_models().items():
            cm.fit(tr[feat_cols], tr[TARGET])
            models[name] = cm
            alphas[(str(cutoff.date()), name)] = cm.alpha
        store(rows26, models, str(cutoff.date()))
        print(f"vintage {cutoff.date()}: {len(rows26)} rows", flush=True)

    mus = pd.concat(mu_rows, ignore_index=True)
    mus.to_parquet(OUT / "hits_mu.parquet", index=False)
    with open(OUT / "hits_models.pkl", "wb") as f:
        pickle.dump({"iso": iso, "alphas": alphas, "feat_cols": feat_cols}, f)
    print(f"stored {len(mus):,} mu rows | total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
