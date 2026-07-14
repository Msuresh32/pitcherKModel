"""V2 Phase 8: adaptive deployment candidates, walk-forward through 2026.

2026 re-partition (user-authorized 2026-07-11):
  Mar 26 - May 31 2026 : adaptation window (may be inspected for head selection)
  Jun 1  - Jul 10 2026 : FINAL GATE - predictions saved here but metrics only
                         computed by 81_final_gate.py, run ONCE.

All predictions are walk-forward: every model/calibrator/blend used on date D
is fit only on data with game_date < D (2026 data enters as it settles).

Heads:
  H0: count ensemble, monthly expanding retrain, pre-2026 iso calibration
  H1: H0 + trailing-90d Platt recalibration (weekly refresh, settled rows only)
  H2: market-blend logistic on [logit(p_market_open), logit(p_H1)],
      fit on 2025 + settled 2026, weekly refresh. Edge threshold chosen by
      matching 2025 bet volume (~1352), NOT by ROI.

Outputs: research/v2/adaptive_preds_2026.parquet (all heads, all 2026 rows)
         research/v2/adaptive_adaptation_report.txt (Mar-May metrics only)
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
ADAPT_END = pd.Timestamp("2026-05-31")   # inclusive; after this = final gate
GATE_END = pd.Timestamp("2026-07-10")
RETRAIN_CUTOFFS = [pd.Timestamp(x) for x in
                   ["2026-01-01", "2026-05-01", "2026-06-01", "2026-07-01"]]
COUNT_NAMES = ["poisson_glm", "xgb_poisson", "lgb_poisson", "hgb_poisson", "cat_poisson"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    feats, bets, feat_cols = MS.load_data()
    count_feat_cols = [c for c in feat_cols if c != "line"]

    # ---------- calibration base: pre-2026 WF OOS preds ----------
    print("Fitting pre-2026 iso calibrators from WF folds...", flush=True)
    oos_p = {n: [] for n in COUNT_NAMES}
    oos_y = []
    for train_years, test_year in [([2022], 2023), ([2022, 2023], 2024)]:
        tr = feats[feats.year.isin(train_years)]
        te_x = MS.expand_lines(feats[feats.year == test_year], feat_cols)
        oos_y.append(te_x["y_over"].values)
        for name, cm in MS.make_count_models().items():
            cm.fit(tr[count_feat_cols], tr["strikeouts"])
            oos_p[name].append(cm.predict_prob(te_x[count_feat_cols], te_x["line"]))
    p25 = pd.read_parquet(OUT / "preds_2025.parquet")
    p25 = p25[p25["outcome_push"] == 0]
    oos_y.append(p25["outcome_over"].values)
    for n in COUNT_NAMES:
        oos_p[n].append(p25[f"p_{n}"].values)
    y_cal = np.concatenate(oos_y)

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    iso = {}
    for n in COUNT_NAMES:
        p = np.clip(np.concatenate(oos_p[n]), 1e-6, 1 - 1e-6)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        ir.fit(p, y_cal)
        iso[n] = ir

    # ---------- monthly expanding retrains, walk-forward 2026 preds ----------
    b26 = bets[(bets.year == 2026) & (bets.game_date <= GATE_END)].copy()
    b26 = b26[b26["outcome_push"] == 0].sort_values("game_date").reset_index(drop=True)
    print(f"2026 rows: {len(b26):,}", flush=True)

    b26["p_h0"] = np.nan
    # Persist the five calibrated component probabilities so downstream
    # conviction scoring can measure ensemble disagreement without refitting.
    for name in COUNT_NAMES:
        b26[f"p_h0_{name}"] = np.nan
    vintages = {}
    for ci, cutoff in enumerate(RETRAIN_CUTOFFS):
        nxt = RETRAIN_CUTOFFS[ci + 1] if ci + 1 < len(RETRAIN_CUTOFFS) else \
            GATE_END + pd.Timedelta(days=1)
        mask = (b26.game_date >= cutoff) & (b26.game_date < nxt) \
            if ci > 0 else (b26.game_date < nxt)
        if mask.sum() == 0:
            continue
        tr = feats[feats.game_date < cutoff]
        print(f"vintage {cutoff.date()}: train n={len(tr):,} -> {mask.sum()} rows",
              flush=True)
        probs = []
        for name, cm in MS.make_count_models().items():
            cm.fit(tr[count_feat_cols], tr["strikeouts"])
            p = cm.predict_prob(b26.loc[mask, count_feat_cols], b26.loc[mask, "line"])
            p_cal = iso[name].predict(np.clip(p, 1e-6, 1 - 1e-6))
            probs.append(p_cal)
            b26.loc[mask, f"p_h0_{name}"] = p_cal
        b26.loc[mask, "p_h0"] = np.mean(probs, axis=0)

    # ---------- H1: trailing-90d Platt on settled rows, weekly refresh ----------
    # recalibrates p_h0 using rows with game_date < refresh date
    b26["p_h1"] = b26["p_h0"]
    hist = pd.concat([
        p25[["game_date", "outcome_over"]].assign(
            p=p25[[f"p_{n}" for n in COUNT_NAMES]].apply(
                lambda r: np.mean([iso[n].predict([np.clip(r[f"p_{n}"], 1e-6, 1-1e-6)])[0]
                                   for n in COUNT_NAMES]), axis=1)),
    ]) if False else None  # 2025 base handled via pre-2026 iso; trailing uses 2026 only
    mondays = pd.date_range("2026-04-20", GATE_END, freq="W-MON")
    for wk in mondays:
        trail = b26[(b26.game_date < wk) &
                    (b26.game_date >= wk - pd.Timedelta(days=90))]
        cur = (b26.game_date >= wk) & (b26.game_date < wk + pd.Timedelta(days=7))
        if len(trail) < 400 or cur.sum() == 0:
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(logit(trail["p_h0"]).reshape(-1, 1), trail["outcome_over"].astype(int))
        b26.loc[cur, "p_h1"] = lr.predict_proba(
            logit(b26.loc[cur, "p_h0"]).reshape(-1, 1))[:, 1]

    # ---------- H2: market blend, fit 2025 + settled 2026, weekly ----------
    p25v = p25.dropna(subset=["p_over_open"]).copy()
    p25v["p_base"] = p25v[[f"p_{n}" for n in COUNT_NAMES]].copy().apply(
        lambda r: np.mean([iso[n].predict(
            [np.clip(r[f"p_{n}"], 1e-6, 1 - 1e-6)])[0] for n in COUNT_NAMES]), axis=1)
    Z25 = np.column_stack([logit(p25v["p_over_open"]), logit(p25v["p_base"])])
    y25 = p25v["outcome_over"].astype(int).values

    b26["p_h2"] = np.nan
    d26 = b26.dropna(subset=["p_over_open"]).copy()
    refresh = [pd.Timestamp("2026-03-01")] + list(mondays)
    for i, wk in enumerate(refresh):
        nxt = refresh[i + 1] if i + 1 < len(refresh) else GATE_END + pd.Timedelta(days=1)
        cur_idx = d26[(d26.game_date >= wk) & (d26.game_date < nxt)].index
        if len(cur_idx) == 0:
            continue
        past = d26[d26.game_date < wk]
        Z = Z25
        y = y25
        if len(past) > 100:
            Zp = np.column_stack([logit(past["p_over_open"]), logit(past["p_h1"])])
            Z = np.vstack([Z25, Zp])
            y = np.concatenate([y25, past["outcome_over"].astype(int).values])
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(Z, y)
        Zc = np.column_stack([logit(d26.loc[cur_idx, "p_over_open"]),
                              logit(d26.loc[cur_idx, "p_h1"])])
        b26.loc[cur_idx, "p_h2"] = lr.predict_proba(Zc)[:, 1]
        if wk in (refresh[0], refresh[len(refresh)//2], refresh[-1]):
            print(f"  H2 {wk.date()}: coefs mkt={lr.coef_[0][0]:.3f} "
                  f"model={lr.coef_[0][1]:.3f}", flush=True)

    # H2 threshold: match 2025 volume (~n of frozen config) using 2025 in-sample blend
    lr25 = LogisticRegression(C=1.0, max_iter=2000).fit(Z25, y25)
    p25v["p_h2"] = lr25.predict_proba(Z25)[:, 1]
    target_n = 1352
    cand = np.arange(0.01, 0.09, 0.005)
    best_t, best_gap = 0.03, 10**9
    for t in cand:
        n = len(S.make_bets(p25v.assign(**{c: p25v[c] for c in p25v.columns}),
                            "p_h2", min_edge=t))
        if abs(n - target_n) < best_gap:
            best_t, best_gap = float(t), abs(n - target_n)
    print(f"H2 edge threshold matched to 2025 volume: {best_t:.3f}", flush=True)

    b26.to_parquet(OUT / "adaptive_preds_2026.parquet", index=False)
    json.dump({"h2_edge_threshold": best_t},
              open(OUT / "adaptive_h2_threshold.json", "w"))

    # ---------- adaptation-window metrics ONLY (Mar-May) ----------
    ad = b26[b26.game_date <= ADAPT_END].copy()
    lines = []
    def rep(s):
        print(s, flush=True); lines.append(s)
    rep(f"=== ADAPTATION WINDOW (2026-03-26 .. {ADAPT_END.date()}) ===")
    for head, thr in [("p_h0", 0.08), ("p_h1", 0.08), ("p_h2", best_t)]:
        bx = S.make_bets(ad, head, min_edge=thr)
        m = S.bet_metrics(bx)
        if m.get("n", 0) == 0:
            rep(f"{head}: no bets"); continue
        rep(f"{head} thr={thr:.3f}: n={m['n']} wr={m['wr']:.3f} roi={m['roi']:+.4f} "
            f"[{m['roi_lo90']:+.3f},{m['roi_hi90']:+.3f}] clv={m['clv_mean']:+.4f} "
            f"clv+%={m['clv_pos_pct']:.3f} over%={m['over_pct']:.2f}")
        bx["month"] = bx.game_date.dt.to_period("M").astype(str)
        rep(bx.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
            roi=("pnl", "mean"), clv=("clv", "mean")).round(4).to_string())
        # edge -> CLV structure on all adaptation rows
        dd = ad.dropna(subset=[head, "p_over_open", "clv_over"]).copy()
        dd["e"] = dd[head] - dd["p_over_open"]
        dd["eb"] = pd.qcut(dd["e"], 6, duplicates="drop")
        rep(dd.groupby("eb", observed=True).agg(n=("clv_over", "size"),
            clv_over=("clv_over", "mean"), over_rate=("outcome_over", "mean"))
            .round(4).to_string())
    (OUT / "adaptive_adaptation_report.txt").write_text("\n".join(lines))
    print("\nSaved adaptive_preds_2026.parquet; gate metrics NOT computed here.",
          flush=True)


if __name__ == "__main__":
    main()
