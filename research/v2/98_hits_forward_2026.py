"""Hits-allowed: ONE-SHOT 2026 walk-forward with the frozen filter.

FROZEN (from 2025 only, research/v2/97 + robustness):
  model p_mean_count (5 iso-calibrated count models), UNDER-only, edge >= 0.04
  vs morning-open consensus, best open price, one bet per pitcher-game.
2026 process: monthly expanding vintages (already stored in hits_mu.parquet)
+ weekly trailing-90d Platt recalibration on settled real-line rows (H1 analog).
Run once; no iteration on its output.
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = __import__("30_strategy")
MS = __import__("20_model_search")
H = __import__("97_hits_strategy_2025")

OUT = Path("research/v2")
COUNT_NAMES = ["poisson_glm", "xgb_poisson", "lgb_poisson", "hgb_poisson", "cat_poisson"]
CFG = {"sides": "under", "min_edge": 0.04}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    mkt = H.load_market()
    mus = pd.read_parquet(OUT / "hits_mu.parquet")
    mus["game_date"] = pd.to_datetime(mus["game_date"])
    with open(OUT / "hits_models.pkl", "rb") as f:
        bundle = pickle.load(f)
    iso, alphas = bundle["iso"], bundle["alphas"]

    m26 = mus[mus.vintage != "2025"].copy()
    j = m26.merge(mkt, on=["game_date", "pitcher_id"], how="inner")
    j = j[(j.line % 1) == 0.5]
    j = j[j.game_date.dt.year == 2026]
    j["outcome_over"] = (j.hits_allowed > j.line).astype(float)
    j["outcome_push"] = 0.0
    print(f"2026 joined rows: {len(j):,} "
          f"({j.game_date.min().date()} .. {j.game_date.max().date()})", flush=True)

    for n in COUNT_NAMES:
        p = np.empty(len(j))
        for v, g in j.groupby("vintage"):
            pp = MS.prob_over_nb(g[f"mu_{n}"].values, alphas[(v, n)], g["line"].values)
            p[j.index.get_indexer(g.index)] = pp
        j[f"p_{n}"] = iso[n].predict(np.clip(p, 1e-6, 1 - 1e-6))
    j["p_h0"] = j[[f"p_{n}" for n in COUNT_NAMES]].mean(axis=1)

    # weekly trailing Platt on settled real-line rows (2025 rows seed the window)
    p25 = pd.read_parquet(OUT / "hits_preds_2025.parquet")
    p25["game_date"] = pd.to_datetime(p25["game_date"])
    hist = pd.concat([
        p25[["game_date", "p_mean_count", "outcome_over"]].rename(
            columns={"p_mean_count": "p_h0"}),
        j[["game_date", "p_h0", "outcome_over"]],
    ], ignore_index=True)

    from sklearn.linear_model import LogisticRegression
    j = j.sort_values("game_date").reset_index(drop=True)
    j["p_h1"] = j["p_h0"]
    mondays = pd.date_range("2026-04-20", j.game_date.max(), freq="W-MON")
    for wk in mondays:
        trail = hist[(hist.game_date < wk) &
                     (hist.game_date >= wk - pd.Timedelta(days=90))]
        cur = (j.game_date >= wk) & (j.game_date < wk + pd.Timedelta(days=7))
        if len(trail) < 400 or cur.sum() == 0:
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(logit(trail["p_h0"]).reshape(-1, 1), trail["outcome_over"].astype(int))
        j.loc[cur, "p_h1"] = lr.predict_proba(
            logit(j.loc[cur, "p_h0"]).reshape(-1, 1))[:, 1]

    for head in ["p_h0", "p_h1"]:
        bets = S.make_bets(j, head, min_edge=CFG["min_edge"], sides=CFG["sides"])
        m = S.bet_metrics(bets)
        print(f"\n=== 2026 WF ({head}, UNDER-only, edge>=0.04) ===", flush=True)
        for k, v in m.items():
            print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}",
                  flush=True)
        if len(bets):
            bets["month"] = bets.game_date.dt.to_period("M").astype(str)
            print(bets.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
                  roi=("pnl", "mean"), clv=("clv", "mean")).round(4).to_string(), flush=True)
            if head == "p_h1":
                bets.to_csv(OUT / "hits_forward_2026_bets.csv", index=False)

    # edge->CLV structure 2026
    dd = j.dropna(subset=["clv_over"]).copy()
    dd["e"] = dd.p_h1 - dd.p_over_open
    dd["eb"] = pd.qcut(dd["e"], 6, duplicates="drop")
    print("\nedge -> CLV structure (2026):", flush=True)
    print(dd.groupby("eb", observed=True).agg(n=("clv_over", "size"),
          clv_over=("clv_over", "mean"), over_rate=("outcome_over", "mean"))
          .round(4).to_string(), flush=True)
    j.to_parquet(OUT / "hits_preds_2026.parquet", index=False)


if __name__ == "__main__":
    main()
