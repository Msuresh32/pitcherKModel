"""V2 Phase 3: bet filter search + robustness on 2025 validation ONLY.

2026 is never loaded here.

Simulation rules (deployable-realistic):
  - Decide at OPEN: edge = p_model - consensus_novig_open (over) or the mirror (under)
  - Execute at best OPEN price across books
  - One bet per pitcher-game: the line with max |edge| (dedup)
  - Flat 1u stakes; quarter-Kelly reported as secondary
  - CLV = no-vig close prob minus no-vig open prob, in bet direction
  - Bootstrap CIs are BLOCK bootstraps over game dates (bets same day correlate)
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
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Bet construction + metrics
# ---------------------------------------------------------------------------

def american_profit(odds: np.ndarray) -> np.ndarray:
    """Profit per 1u stake if the bet wins."""
    o = np.asarray(odds, dtype=float)
    return np.where(o >= 0, o / 100.0, 100.0 / np.abs(o))


def make_bets(df: pd.DataFrame, pcol: str,
              min_edge: float = 0.05,
              min_ev: float = 0.0,
              sides: str = "both",
              min_books: int = 1,
              odds_min: float = -10000, odds_max: float = 10000,
              lines: tuple = (3.5, 9.5),
              dedup: bool = True) -> pd.DataFrame:
    d = df.dropna(subset=[pcol, "p_over_open"]).copy()
    d = d[(d["line"] >= lines[0]) & (d["line"] <= lines[1])]
    d = d[d["n_books_open"] >= min_books]

    p = d[pcol].clip(0.01, 0.99)
    edge_over = p - d["p_over_open"]

    over = d[edge_over >= min_edge].copy()
    over["bet_side"] = "over"
    over["p_bet"] = p[edge_over >= min_edge]
    over["edge"] = edge_over[edge_over >= min_edge]
    over["odds"] = over["best_over_odds_open"]
    over["book"] = over.get("best_over_book_open")
    over["won"] = (over["outcome_over"] == 1)
    over["clv"] = over["clv_over"]

    under = d[-edge_over >= min_edge].copy()
    under["bet_side"] = "under"
    under["p_bet"] = (1 - p)[-edge_over >= min_edge]
    under["edge"] = -edge_over[-edge_over >= min_edge]
    under["odds"] = under["best_under_odds_open"]
    under["book"] = under.get("best_under_book_open")
    under["won"] = (under["outcome_over"] == 0)
    under["clv"] = -under["clv_over"]

    bets = pd.concat([over, under], ignore_index=True)
    if sides != "both":
        bets = bets[bets["bet_side"] == sides]
    bets = bets.dropna(subset=["odds"])

    prof = american_profit(bets["odds"].values)
    bets["ev"] = bets["p_bet"] * prof - (1 - bets["p_bet"])
    bets = bets[(bets["odds"] >= odds_min) & (bets["odds"] <= odds_max)]
    bets = bets[bets["ev"] >= min_ev]

    if dedup and len(bets):
        bets = (bets.sort_values("edge", ascending=False)
                    .drop_duplicates(subset=["game_date", "pitcher_id"], keep="first"))

    if len(bets):
        prof = american_profit(bets["odds"].values)
        bets["pnl"] = np.where(bets["won"], prof, -1.0)
        # quarter-Kelly
        b = prof
        kf = np.clip(0.25 * (bets["p_bet"] * (1 + b) - 1) / b, 0, 0.05)
        bets["pnl_kelly"] = np.where(bets["won"], kf * b, -kf)
    return bets.sort_values("game_date").reset_index(drop=True)


def block_bootstrap_ci(bets: pd.DataFrame, n_boot: int = 2000, col: str = "pnl"):
    """Bootstrap over game dates (cluster bootstrap)."""
    if len(bets) == 0:
        return (np.nan, np.nan)
    groups = {k: v[col].values for k, v in bets.groupby("game_date")}
    keys = list(groups)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        vals = np.concatenate([groups[keys[j]] for j in pick])
        means[i] = vals.mean()
    return tuple(np.percentile(means, [5, 95]))


def bet_metrics(bets: pd.DataFrame) -> dict:
    if len(bets) == 0:
        return {"n": 0}
    pnl = bets["pnl"].values
    wins = bets["won"].sum()
    gross_w = pnl[pnl > 0].sum()
    gross_l = -pnl[pnl < 0].sum()
    cum = np.cumsum(pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    lo, hi = block_bootstrap_ci(bets)
    daily = bets.groupby("game_date")["pnl"].sum()
    sharpe_daily = float(daily.mean() / daily.std() * np.sqrt(180)) if daily.std() > 0 else np.nan
    return {
        "n": len(bets),
        "wr": float(wins / len(bets)),
        "roi": float(pnl.mean()),
        "roi_lo90": float(lo), "roi_hi90": float(hi),
        "units": float(pnl.sum()),
        "avg_odds": float(bets["odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "avg_ev": float(bets["ev"].mean()),
        "clv_mean": float(bets["clv"].mean()),
        "clv_pos_pct": float((bets["clv"] > 0).mean()),
        "sharpe_ann": sharpe_daily,
        "max_dd": dd,
        "profit_factor": float(gross_w / gross_l) if gross_l > 0 else np.inf,
        "kelly_roi": float(bets["pnl_kelly"].mean()),
        "over_pct": float((bets["bet_side"] == "over").mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_parquet(OUT / "preds_2025.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    assert df.game_date.dt.year.max() == 2025, "2026 leaked into strategy search!"
    df = df[df["outcome_push"] == 0]

    import re
    MODEL_PAT = re.compile(
        r"^p_(poisson_glm|xgb_poisson|lgb_poisson|hgb_poisson|cat_poisson|"
        r"logistic|xgb_clf|lgb_clf|hgb_clf|rf_clf|et_clf|cat_clf|"
        r"stack|mean_count|mean_all)(_iso|_platt)?$")
    pcols = [c for c in df.columns if MODEL_PAT.match(c)]
    print(f"{len(df):,} rows 2025 | candidate prob columns: {len(pcols)}", flush=True)

    # ---- Step 1: screen all model variants at fixed default filter ----
    screen = []
    for pc in pcols:
        bets = make_bets(df, pc, min_edge=0.05)
        m = bet_metrics(bets)
        m["model"] = pc
        screen.append(m)
    screen_df = pd.DataFrame(screen).sort_values("roi", ascending=False)
    screen_df.to_csv(OUT / "screen_2025.csv", index=False)
    cols = ["model", "n", "wr", "roi", "roi_lo90", "roi_hi90", "clv_mean",
            "clv_pos_pct", "sharpe_ann", "profit_factor", "over_pct"]
    print("\n=== 2025 screen (edge>=0.05, both sides, dedup) ===", flush=True)
    print(screen_df[cols].to_string(index=False), flush=True)

    # ---- Step 2: shortlist = top by CLV (market-respect) AND roi_lo90 ----
    ok = screen_df[screen_df["n"] >= 300]
    short = set(ok.nlargest(4, "clv_mean")["model"]) | set(ok.nlargest(4, "roi_lo90")["model"])
    print(f"\nShortlist: {sorted(short)}", flush=True)

    # ---- Step 3: filter grid on shortlist ----
    rows = []
    for pc in sorted(short):
        for edge in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            for sides in ["both", "over", "under"]:
                for min_books in [1, 3]:
                    bets = make_bets(df, pc, min_edge=edge, sides=sides,
                                     min_books=min_books)
                    m = bet_metrics(bets)
                    m.update({"model": pc, "min_edge": edge, "sides": sides,
                              "min_books": min_books})
                    rows.append(m)
    grid = pd.DataFrame(rows)
    grid = grid[grid["n"] >= 100].copy()
    # objective: LCB of ROI, but require scale
    grid["score"] = grid["roi_lo90"] + 0.02 * np.log(grid["n"] / 500)
    grid = grid.sort_values("score", ascending=False)
    grid.to_csv(OUT / "grid_2025.csv", index=False)
    print("\n=== Top filter configs by score (LCB-based) ===", flush=True)
    gcols = ["model", "min_edge", "sides", "min_books"] + cols[1:]
    print(grid[gcols].head(20).to_string(index=False), flush=True)

    print("\nSaved screen_2025.csv and grid_2025.csv", flush=True)


if __name__ == "__main__":
    main()
