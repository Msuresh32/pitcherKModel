"""Hits-allowed: 2025-ONLY validation + filter search. 2026 never touched here.

Joins 2025 per-game mus to the hits odds (open = T-12h/T-6h morning board,
close = T-3min for CLV). Bets executed at best OPEN price, edge vs open
consensus no-vig, one bet per pitcher-game, half-lines only.
"""
from __future__ import annotations
import sys, json, pickle
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
ODDS = Path("data/odds/historical/pitcher_hits_allowed_2025_2026.csv")
COUNT_NAMES = ["poisson_glm", "xgb_poisson", "lgb_poisson", "hgb_poisson", "cat_poisson"]


def load_market():
    df = pd.read_csv(ODDS, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")
    df = df.dropna(subset=["over_odds", "under_odds", "line", "pitcher_id"])
    po = np.where(df.over_odds < 0, -df.over_odds/(-df.over_odds+100), 100/(100+df.over_odds))
    pu = np.where(df.under_odds < 0, -df.under_odds/(-df.under_odds+100), 100/(100+df.under_odds))
    tot = po + pu
    keep = (tot > 0.98) & (tot < 1.20)
    df = df[keep]
    df["p_over_novig"] = (po[keep]) / tot[keep]

    def consensus(sub, tag):
        return sub.groupby(["game_date", "pitcher_id", "line"]).agg(
            **{f"p_over_{tag}": ("p_over_novig", "mean"),
               f"n_books_{tag}": ("bookmaker", "nunique"),
               f"best_over_odds_{tag}": ("over_odds", "max"),
               f"best_under_odds_{tag}": ("under_odds", "max")}).reset_index()

    o = consensus(df[df.snapshot_type == "open"], "open")
    c = consensus(df[df.snapshot_type == "close"], "close")
    mkt = o.merge(c, on=["game_date", "pitcher_id", "line"], how="left")
    mkt["clv_over"] = mkt["p_over_close"] - mkt["p_over_open"]
    return mkt


def main():
    mkt = load_market()
    print(f"market rows: {len(mkt):,} "
          f"({mkt.game_date.min().date()} .. {mkt.game_date.max().date()})", flush=True)

    mus = pd.read_parquet(OUT / "hits_mu.parquet")
    mus["game_date"] = pd.to_datetime(mus["game_date"])
    with open(OUT / "hits_models.pkl", "rb") as f:
        bundle = pickle.load(f)
    iso, alphas = bundle["iso"], bundle["alphas"]

    m25 = mus[mus.vintage == "2025"].copy()
    j = m25.merge(mkt, on=["game_date", "pitcher_id"], how="inner")
    j = j[(j.line % 1) == 0.5]
    j["outcome_over"] = (j.hits_allowed > j.line).astype(float)
    j["outcome_push"] = 0.0
    print(f"2025 joined rows: {len(j):,} "
          f"({j.game_date.dt.year.min()}..{j.game_date.dt.year.max()})", flush=True)
    j = j[j.game_date.dt.year == 2025]

    # P(over) per model via NB tail + iso -> ensemble mean
    for n in COUNT_NAMES:
        p = MS.prob_over_nb(j[f"mu_{n}"].values, alphas[("2025", n)], j["line"].values)
        j[f"p_{n}"] = iso[n].predict(np.clip(p, 1e-6, 1 - 1e-6))
    j["p_mean_count"] = j[[f"p_{n}" for n in COUNT_NAMES]].mean(axis=1)

    # calibration check
    d = j.copy()
    d["bucket"] = pd.qcut(d.p_mean_count, 10, duplicates="drop")
    print("\n2025 calibration (all rows):", flush=True)
    print(d.groupby("bucket", observed=True).agg(pred=("p_mean_count", "mean"),
          actual=("outcome_over", "mean"), n=("outcome_over", "size")).round(3).to_string(),
          flush=True)

    # ---- filter grid (2025 only) ----
    rows = []
    for edge in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        for sides in ["both", "over", "under"]:
            b = S.make_bets(j, "p_mean_count", min_edge=edge, sides=sides)
            m = S.bet_metrics(b)
            m.update({"edge": edge, "sides": sides})
            rows.append(m)
    g = pd.DataFrame(rows)
    g = g[g.n > 0]
    cols = ["edge", "sides", "n", "wr", "roi", "roi_lo90", "roi_hi90",
            "clv_mean", "clv_pos_pct", "profit_factor", "over_pct", "avg_odds"]
    print("\n=== 2025 filter grid ===", flush=True)
    print(g[cols].round(4).to_string(index=False), flush=True)
    g.to_csv(OUT / "hits_grid_2025.csv", index=False)

    # edge -> CLV structure
    dd = j.dropna(subset=["clv_over"]).copy()
    dd["e"] = dd.p_mean_count - dd.p_over_open
    dd["eb"] = pd.qcut(dd["e"], 8, duplicates="drop")
    print("\nedge -> CLV structure (2025):", flush=True)
    print(dd.groupby("eb", observed=True).agg(n=("clv_over", "size"),
          clv_over=("clv_over", "mean"), over_rate=("outcome_over", "mean"),
          mkt=("p_over_open", "mean")).round(4).to_string(), flush=True)

    j.to_parquet(OUT / "hits_preds_2025.parquet", index=False)
    print("\nSaved hits_preds_2025.parquet + hits_grid_2025.csv", flush=True)


if __name__ == "__main__":
    main()
