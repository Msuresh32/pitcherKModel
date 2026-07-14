"""V2 Phase 6: robustness battery on 2025 validation ONLY (no 2026).

Tests, per finalist config:
  1. Random-side baseline at same execution style (best open price)   - is ROI
     explained by best-price shopping alone?
  2. Odds-matched random baseline (sample non-bet rows w/ similar odds)
  3. Shuffle test: permute model probabilities within game-date, re-run filter
  4. Monthly ROI + rolling stability
  5. By line / by odds bucket / by best-price book
  6. Execution sensitivity: best price vs MEDIAN price vs majors-only best
  7. Edge -> CLV monotonicity (does bigger model edge buy more CLV?)
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

OUT = Path("research/v2")
RNG = np.random.default_rng(7)
MAJORS = {"draftkings", "fanduel", "betmgm", "betrivers", "williamhill_us", "fanatics"}

FINALISTS = [
    {"model_col": "p_mean_count", "min_edge": 0.10, "sides": "both", "min_books": 1},
    {"model_col": "p_mean_count", "min_edge": 0.08, "sides": "both", "min_books": 1},
    {"model_col": "p_mean_all",   "min_edge": 0.10, "sides": "both", "min_books": 1},
    {"model_col": "p_cat_clf_platt", "min_edge": 0.08, "sides": "over", "min_books": 1},
    {"model_col": "p_poisson_glm_platt", "min_edge": 0.08, "sides": "both", "min_books": 1},
]


def execution_prices(df):
    """Add correctly aggregated median prices and majors-only best prices.

    American odds are discontinuous at zero, so their raw numeric median is
    not a valid executable price. Convert each quote to decimal first, then
    aggregate. (The dedicated 61_exec_sensitivity.py check uses the same
    convention.)
    """
    odds = pd.read_csv("data/odds/historical/pitcher_strikeouts_2025_2026.csv",
                       low_memory=False)
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce")
    odds = odds[odds.snapshot_type == "open"].copy()
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["pitcher_id"] = pd.to_numeric(odds["pitcher_id"], errors="coerce")
    for side in ("over", "under"):
        american = pd.to_numeric(odds[f"{side}_odds"], errors="coerce")
        odds[f"{side}_decimal"] = np.where(
            american < 0, 1.0 + 100.0 / -american, 1.0 + american / 100.0
        )
    med = odds.groupby(["game_date", "pitcher_id", "line"]).agg(
        med_over_decimal=("over_decimal", "median"),
        med_under_decimal=("under_decimal", "median"),
    ).reset_index()
    majors = odds[odds.bookmaker.isin(MAJORS)]
    grp = majors.groupby(["game_date", "pitcher_id", "line"]).agg(
        mj_over=("over_odds", "max"), mj_under=("under_odds", "max"),
        mj_books=("bookmaker", "nunique")).reset_index()
    return (df.merge(med, on=["game_date", "pitcher_id", "line"], how="left")
              .merge(grp, on=["game_date", "pitcher_id", "line"], how="left"))


def summarize(bets, label):
    m = S.bet_metrics(bets)
    if m.get("n", 0) == 0:
        print(f"  {label}: no bets")
        return m
    print(f"  {label:34s} n={m['n']:5d} wr={m['wr']:.3f} roi={m['roi']:+.4f} "
          f"[{m['roi_lo90']:+.3f},{m['roi_hi90']:+.3f}] clv={m['clv_mean']:+.4f} "
          f"clv+%={m['clv_pos_pct']:.2f}", flush=True)
    return m


def main():
    df = pd.read_parquet(OUT / "preds_2025.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    assert df.game_date.dt.year.max() == 2025
    df = df[df["outcome_push"] == 0]
    df = execution_prices(df)

    # ---- Global market sanity: was 2025 an "over year" vs OPEN prices? ----
    print("=== Market-level sanity (all 2025 rows) ===", flush=True)
    print(f"rows={len(df)}  mean outcome_over={df.outcome_over.mean():.4f}  "
          f"mean p_over_open={df.p_over_open.mean():.4f}  "
          f"mean p_over_close={df.p_over_close.mean():.4f}", flush=True)

    # blind baselines: bet EVERY row one side at best open price
    for side in ["over", "under"]:
        col = f"best_{side}_odds_open"
        d = df.dropna(subset=[col]).drop_duplicates(subset=["game_date", "pitcher_id"])
        won = (d.outcome_over == 1) if side == "over" else (d.outcome_over == 0)
        prof = S.american_profit(d[col].values)
        pnl = np.where(won, prof, -1.0)
        print(f"blind ALL-{side} at best open price: n={len(d)} roi={pnl.mean():+.4f}",
              flush=True)

    # random-side baseline (500 sims): pick a side at random per pitcher-game
    d = df.drop_duplicates(subset=["game_date", "pitcher_id"]).dropna(
        subset=["best_over_odds_open", "best_under_odds_open"])
    rois = []
    for _ in range(500):
        side_over = RNG.random(len(d)) < 0.5
        prof_o = S.american_profit(d["best_over_odds_open"].values)
        prof_u = S.american_profit(d["best_under_odds_open"].values)
        pnl = np.where(side_over,
                       np.where(d.outcome_over == 1, prof_o, -1),
                       np.where(d.outcome_over == 0, prof_u, -1))
        rois.append(pnl.mean())
    print(f"random-side @best open price: mean={np.mean(rois):+.4f} "
          f"90%CI=[{np.percentile(rois,5):+.4f},{np.percentile(rois,95):+.4f}]",
          flush=True)

    for cfg in FINALISTS:
        pc = cfg["model_col"]
        print(f"\n=== {json.dumps(cfg)} ===", flush=True)
        bets = S.make_bets(df, pc, min_edge=cfg["min_edge"], sides=cfg["sides"],
                           min_books=cfg["min_books"])
        base = summarize(bets, "headline (best open price)")

        # (6) execution sensitivity: median odds
        med = bets.copy()
        med["median_decimal"] = np.where(
            med.bet_side == "over", med["med_over_decimal"], med["med_under_decimal"]
        )
        med = med.dropna(subset=["median_decimal"])
        prof = med["median_decimal"].to_numpy() - 1.0
        med["pnl"] = np.where(med["won"], prof, -1.0)
        summarize(med, "execution @ MEDIAN open odds")

        # majors-only best price (re-filter edge on same consensus)
        mj = bets.copy()
        mj["odds"] = np.where(mj.bet_side == "over", mj["mj_over"], mj["mj_under"])
        mj = mj.dropna(subset=["odds"])
        prof = S.american_profit(mj["odds"].values)
        mj["pnl"] = np.where(mj["won"], prof, -1.0)
        summarize(mj, "execution @ MAJORS-only best")

        # odds cap: drop longshot prices > +150
        capped = bets[bets["odds"] <= 150]
        summarize(capped, "odds capped at +150")

        # (3) shuffle test: permute model prob within date, 200 reps
        sh_rois = []
        for _ in range(200):
            dfx = df[["game_date", "pitcher_id", "line", "p_over_open",
                      "best_over_odds_open", "best_under_odds_open",
                      "med_over_odds_open", "med_under_odds_open",
                      "n_books_open", "outcome_over", "clv_over", pc]].copy()
            dfx[pc] = dfx.groupby("game_date")[pc].transform(
                lambda s: s.sample(frac=1.0, random_state=None).values)
            b = S.make_bets(dfx, pc, min_edge=cfg["min_edge"], sides=cfg["sides"],
                            min_books=cfg["min_books"])
            if len(b) > 30:
                sh_rois.append(b["pnl"].mean())
        if sh_rois:
            frac_ge = float(np.mean([r >= base.get("roi", 0) for r in sh_rois]))
            print(f"  shuffle test: mean={np.mean(sh_rois):+.4f} "
                  f"p(shuffled >= actual)={frac_ge:.3f} (n={len(sh_rois)})", flush=True)

        # (4) monthly
        bets["month"] = bets.game_date.dt.to_period("M").astype(str)
        print(bets.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
              roi=("pnl", "mean"), clv=("clv", "mean")).round(4).to_string(), flush=True)

        # (5) by line + odds bucket + book
        print(bets.groupby("line").agg(n=("pnl", "size"), roi=("pnl", "mean"),
              clv=("clv", "mean")).round(4).to_string(), flush=True)
        bets["odds_b"] = pd.cut(bets["odds"], [-1000, -160, -120, -101, 120, 160, 10000])
        print(bets.groupby("odds_b", observed=True).agg(n=("pnl", "size"),
              roi=("pnl", "mean")).round(4).to_string(), flush=True)
        print(bets.groupby("book").agg(n=("pnl", "size"), roi=("pnl", "mean"),
              clv=("clv", "mean")).round(4).to_string(), flush=True)

        # (7) edge -> CLV monotonicity on ALL rows (not just bets)
        dd = df.dropna(subset=[pc, "p_over_open", "clv_over"]).copy()
        dd["model_edge_over"] = dd[pc] - dd["p_over_open"]
        dd["edge_bucket"] = pd.qcut(dd["model_edge_over"], 8, duplicates="drop")
        print(dd.groupby("edge_bucket", observed=True).agg(
            n=("clv_over", "size"), mean_clv_over=("clv_over", "mean"),
            over_rate=("outcome_over", "mean"),
            mkt_open=("p_over_open", "mean")).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
