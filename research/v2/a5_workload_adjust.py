"""Workload adjustment: fold market-implied IP (from pitcher_outs lines) into
the K model via a 3-parameter logistic residual layer.

  IP_mkt      : invert outs line + no-vig price -> implied mean outs / 3
                (normal tail, sigma_outs estimated from 2022-24 actuals)
  IP_base     : pitcher rolling IP (roll10, fallback roll5) from features
  p_adj       = sigma(a + b*logit(p_model) + c*log(IP_mkt/IP_base))

Layer fit on 2025 ONLY (the tuning year). 2026 = single adopt/reject check
with coefficients frozen (mandate rule: adopt only if 2025 improves AND 2026
does not degrade).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = __import__("30_strategy")

OUT = Path("research/v2")
OUTS = Path("data/odds/historical/pitcher_outs_open_2025_2026.csv")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def market_ip():
    df = pd.read_csv(OUTS, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")
    df = df.dropna(subset=["line", "over_odds", "under_odds", "pitcher_id"])
    po = np.where(df.over_odds < 0, -df.over_odds/(-df.over_odds+100), 100/(100+df.over_odds))
    pu = np.where(df.under_odds < 0, -df.under_odds/(-df.under_odds+100), 100/(100+df.under_odds))
    df["p_over"] = po / (po + pu)

    # sigma of outs around expectation, from 2022-24 actual outs distribution
    logs = pd.read_csv("data/raw/pitcher_game_logs.csv")
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs[(logs.game_date < "2025-01-01") & (logs.game_date >= "2022-01-01")]
    outs = pd.to_numeric(logs["innings_pitched"], errors="coerce") * 3
    sigma = float(outs.std())

    # implied mean outs: P(X > line) = p  ->  mu = line + z(p)*sigma  (normal)
    df["mu_outs"] = df["line"] + norm.ppf(np.clip(df["p_over"], 0.02, 0.98)) * sigma
    g = (df.groupby(["game_date", "pitcher_id"])
           .agg(ip_mkt=("mu_outs", "mean"), n_outs_books=("bookmaker", "nunique"))
           .reset_index())
    g["ip_mkt"] = g["ip_mkt"] / 3.0
    print(f"outs sigma={sigma:.2f} | implied-IP rows: {len(g):,} "
          f"({g.game_date.min().date()} .. {g.game_date.max().date()})", flush=True)
    return g


def prep(pred_file, pcol, ip):
    d = pd.read_parquet(OUT / pred_file)
    d["game_date"] = pd.to_datetime(d["game_date"])
    d = d[d.outcome_push == 0]
    d["pitcher_id"] = pd.to_numeric(d["pitcher_id"], errors="coerce")
    ipcols = [c for c in ["p_innings_pitched_roll10", "p_innings_pitched_roll5"]
              if c in d.columns]
    if not ipcols:
        feats = pd.read_parquet(OUT / "features_full.parquet")
        feats["game_date"] = pd.to_datetime(feats["game_date"])
        feats["pitcher_id"] = pd.to_numeric(feats["pitcher_id"], errors="coerce")
        ipcols = [c for c in ["p_innings_pitched_roll10", "p_innings_pitched_roll5"]
                  if c in feats.columns]
        d = d.merge(feats[["game_date", "pitcher_id"] + ipcols].drop_duplicates(
            subset=["game_date", "pitcher_id"]), on=["game_date", "pitcher_id"], how="left")
    d["ip_base"] = d[ipcols[0]].fillna(d[ipcols[-1]])
    d = d.merge(ip, on=["game_date", "pitcher_id"], how="left")
    d["log_ip_ratio"] = np.log(
        d["ip_mkt"].clip(1.0, 9.0) / d["ip_base"].clip(1.0, 9.0))
    d["has_outs"] = d["ip_mkt"].notna()
    d["log_ip_ratio"] = d["log_ip_ratio"].fillna(0.0)
    d["p0"] = d[pcol]
    return d


def frozen_eval(d, pcol, label):
    b = S.make_bets(d, pcol, min_edge=0.08)
    m = S.bet_metrics(b)
    print(f"  {label:36s} n={m.get('n',0)} roi={m.get('roi',float('nan')):+.4f} "
          f"[{m.get('roi_lo90',float('nan')):+.3f},{m.get('roi_hi90',float('nan')):+.3f}] "
          f"clv={m.get('clv_mean',float('nan')):+.4f} "
          f"clv+%={m.get('clv_pos_pct',float('nan')):.3f}", flush=True)
    return m


def main():
    ip = market_ip()

    # ---------- 2025: fit layer, evaluate ----------
    d25 = prep("preds_2025.parquet", "p_mean_count", ip)
    print(f"2025 rows: {len(d25):,} | outs-line coverage: {d25.has_outs.mean():.1%}",
          flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    X = np.column_stack([logit(d25["p0"]), d25["log_ip_ratio"]])
    y = d25["outcome_over"].astype(int).values
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    a, (b_, c_) = float(lr.intercept_[0]), lr.coef_[0]
    print(f"layer: a={a:+.3f} b(logit p)={b_:+.3f} c(log IP ratio)={c_:+.3f} "
          f"(expect c>0)", flush=True)
    d25["p_adj"] = lr.predict_proba(X)[:, 1]
    print(f"logloss 2025: base={log_loss(y, d25['p0'].clip(1e-6,1-1e-6)):.5f} "
          f"adj={log_loss(y, d25['p_adj']):.5f}", flush=True)
    print("2025 frozen filter (edge>=0.08 both sides):", flush=True)
    frozen_eval(d25, "p0", "base (p_mean_count)")
    frozen_eval(d25, "p_adj", "workload-adjusted")
    sub = d25[d25.has_outs]
    print(f"  -- rows WITH outs line only ({len(sub):,}):", flush=True)
    frozen_eval(sub, "p0", "base")
    frozen_eval(sub, "p_adj", "workload-adjusted")

    # ---------- 2026: single adopt/reject check, coefficients frozen ----------
    d26 = prep("adaptive_preds_2026.parquet", "p_h1", ip)
    print(f"\n2026 rows: {len(d26):,} | outs coverage: {d26.has_outs.mean():.1%}",
          flush=True)
    X26 = np.column_stack([logit(d26["p0"]), d26["log_ip_ratio"]])
    d26["p_adj"] = lr.predict_proba(X26)[:, 1]
    y26 = d26["outcome_over"].astype(int).values
    print(f"logloss 2026: base={log_loss(y26, d26['p0'].clip(1e-6,1-1e-6)):.5f} "
          f"adj={log_loss(y26, d26['p_adj']):.5f}", flush=True)
    print("2026 WF frozen filter:", flush=True)
    frozen_eval(d26, "p0", "base (p_h1)")
    frozen_eval(d26, "p_adj", "workload-adjusted")
    d26.to_parquet(OUT / "k_workload_preds_2026.parquet", index=False)
    d25.to_parquet(OUT / "k_workload_preds_2025.parquet", index=False)


if __name__ == "__main__":
    main()
