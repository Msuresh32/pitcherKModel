"""Execution sensitivity for the finalist config (2025 only).

Correct median test (decimal odds), DK/FD-only, exclude-BetRivers,
n_books distribution.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = __import__("30_strategy")
OUT = Path("research/v2")

CFGS = [
    {"model_col": "p_mean_count", "min_edge": 0.08, "sides": "both", "min_books": 1},
    {"model_col": "p_mean_count", "min_edge": 0.10, "sides": "both", "min_books": 1},
]


def to_decimal(american):
    o = np.asarray(american, dtype=float)
    return np.where(o >= 0, 1 + o / 100.0, 1 + 100.0 / np.abs(o))


def exec_roi(bets, over_odds_col, under_odds_col, label, decimal=False):
    b = bets.copy()
    b["o"] = np.where(b.bet_side == "over", b[over_odds_col], b[under_odds_col])
    b = b.dropna(subset=["o"])
    prof = (b["o"].values - 1.0) if decimal else S.american_profit(b["o"].values)
    pnl = np.where(b["won"], prof, -1.0)
    print(f"  {label:36s} n={len(b):5d} roi={pnl.mean():+.4f} "
          f"units={pnl.sum():+.1f}", flush=True)


def main():
    df = pd.read_parquet(OUT / "preds_2025.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["outcome_push"] == 0]

    odds = pd.read_csv("data/odds/historical/pitcher_strikeouts_2025_2026.csv",
                       low_memory=False)
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce")
    odds = odds[odds.snapshot_type == "open"].copy()
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["pitcher_id"] = pd.to_numeric(odds["pitcher_id"], errors="coerce")
    odds["dec_over"] = to_decimal(odds["over_odds"])
    odds["dec_under"] = to_decimal(odds["under_odds"])

    def agg_books(sub, tag):
        g = sub.groupby(["game_date", "pitcher_id", "line"]).agg(
            **{f"{tag}_dec_over_med": ("dec_over", "median"),
               f"{tag}_dec_under_med": ("dec_under", "median"),
               f"{tag}_over_best": ("over_odds", "max"),
               f"{tag}_under_best": ("under_odds", "max")}).reset_index()
        return g

    all_med = agg_books(odds, "all")
    dkfd = agg_books(odds[odds.bookmaker.isin({"draftkings", "fanduel"})], "dkfd")
    norivers = agg_books(odds[odds.bookmaker != "betrivers"], "nrv")

    for g in [all_med, dkfd, norivers]:
        df = df.merge(g, on=["game_date", "pitcher_id", "line"], how="left")

    print("n_books_open distribution (2025 rows):", flush=True)
    print(df.n_books_open.value_counts().sort_index().to_string(), flush=True)

    for cfg in CFGS:
        print(f"\n=== {cfg} ===", flush=True)
        bets = S.make_bets(df, cfg["model_col"], min_edge=cfg["min_edge"],
                           sides=cfg["sides"], min_books=cfg["min_books"])
        m = S.bet_metrics(bets)
        print(f"  headline best-price: n={m['n']} roi={m['roi']:+.4f} "
              f"[{m['roi_lo90']:+.3f},{m['roi_hi90']:+.3f}]", flush=True)
        exec_roi(bets, "all_dec_over_med", "all_dec_under_med",
                 "MEDIAN book (decimal, correct)", decimal=True)
        exec_roi(bets, "dkfd_over_best", "dkfd_under_best", "DK/FD only best")
        exec_roi(bets, "nrv_over_best", "nrv_under_best", "best EXCLUDING BetRivers")


if __name__ == "__main__":
    main()
