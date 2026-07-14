"""Re-evaluate the frozen system against the EARLY open (T-12h / T-6h fallback).

Market reference, execution price and CLV baseline switch to the early-open
snapshot; model predictions are the stored ones (2025: p_mean_count from
preds_2025.parquet; 2026: walk-forward p_h1 from adaptive_preds_2026.parquet).
No model retraining, no filter changes: edge >= 0.08, both sides, best price
at the early snapshot, one bet per pitcher-game, flat 1u. CLV = existing
T-3min close consensus minus early-open consensus, in bet direction.
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
EARLY = Path("data/odds/historical/pitcher_strikeouts_early_open_2025_2026.csv")
BASE = Path("data/odds/historical/pitcher_strikeouts_2025_2026.csv")


def load_snap(path, snap_filter=None):
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if snap_filter:
        df = df[df.snapshot_type == snap_filter]
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")
    df = df.dropna(subset=["over_odds", "under_odds", "line", "pitcher_id"])
    po = np.where(df.over_odds < 0, -df.over_odds/(-df.over_odds+100), 100/(100+df.over_odds))
    pu = np.where(df.under_odds < 0, -df.under_odds/(-df.under_odds+100), 100/(100+df.under_odds))
    tot = po + pu
    df = df[(tot > 0.98) & (tot < 1.20)]
    df["p_over_novig"] = po[(tot > 0.98) & (tot < 1.20)] / tot[(tot > 0.98) & (tot < 1.20)]
    return df


def consensus(df, tag):
    g = df.groupby(["game_date", "pitcher_id", "line"]).agg(
        **{f"p_over_{tag}": ("p_over_novig", "mean"),
           f"n_books_{tag}": ("bookmaker", "nunique"),
           f"best_over_odds_{tag}": ("over_odds", "max"),
           f"best_under_odds_{tag}": ("under_odds", "max")}).reset_index()
    return g


def main():
    early = load_snap(EARLY)
    if "open_hours_used" in early.columns:
        print("open_hours_used distribution:")
        print(early.groupby([early.game_date.dt.year, "open_hours_used"]).size().to_string())
    e = consensus(early, "open")
    close = consensus(load_snap(BASE, "close"), "close")
    mkt = e.merge(close, on=["game_date", "pitcher_id", "line"], how="left")
    mkt["clv_over"] = mkt["p_over_close"] - mkt["p_over_open"]

    results = {}
    for year, pred_file, pcol in [(2025, "preds_2025.parquet", "p_mean_count"),
                                  (2026, "adaptive_preds_2026.parquet", "p_h1")]:
        preds = pd.read_parquet(OUT / pred_file)
        preds["game_date"] = pd.to_datetime(preds["game_date"])
        preds = preds[preds.outcome_push == 0]
        keep = ["game_date", "pitcher_id", "line", pcol, "outcome_over",
                "actual_ks", "outcome_push"]
        keep = [c for c in keep if c in preds.columns]
        pr = preds[keep].copy()
        pr["pitcher_id"] = pd.to_numeric(pr["pitcher_id"], errors="coerce")

        j = pr.merge(mkt, on=["game_date", "pitcher_id", "line"], how="inner")
        j = j[(j.line % 1) == 0.5]
        print(f"\n=== {year}: matched {len(j):,} of {len(pr):,} model rows "
              f"({len(j)/len(pr):.1%}) at the early open ===")

        bets = S.make_bets(j, pcol, min_edge=0.08)
        m = S.bet_metrics(bets)
        print(f"ALL   n={m.get('n',0)} wr={m.get('wr',float('nan')):.3f} "
              f"roi={m.get('roi',float('nan')):+.4f} "
              f"[{m.get('roi_lo90',float('nan')):+.3f},{m.get('roi_hi90',float('nan')):+.3f}] "
              f"clv={m.get('clv_mean',float('nan')):+.4f} "
              f"clv+%={m.get('clv_pos_pct',float('nan')):.3f} "
              f"avg_odds={m.get('avg_odds',float('nan')):+.1f}")
        conv = bets[bets.edge >= 0.15]
        if len(conv):
            mc = S.bet_metrics(conv)
            print(f"CONV  n={mc['n']} roi={mc['roi']:+.4f} "
                  f"[{mc['roi_lo90']:+.3f},{mc['roi_hi90']:+.3f}] clv={mc['clv_mean']:+.4f}")
        bets["month"] = bets.game_date.dt.to_period("M").astype(str)
        print(bets.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
              roi=("pnl", "mean"), clv=("clv", "mean")).round(4).to_string())
        bets.to_csv(OUT / f"early_open_bets_{year}.csv", index=False)
        results[year] = m

    print("\n=== T-4h baseline for comparison ===")
    base_parts = []
    for year, pred_file, pcol in [(2025, "preds_2025.parquet", "p_mean_count"),
                                  (2026, "adaptive_preds_2026.parquet", "p_h1")]:
        d = pd.read_parquet(OUT / pred_file)
        b = S.make_bets(d[d.outcome_push == 0], pcol, min_edge=0.08)
        m = S.bet_metrics(b)
        base_parts.append(
            f"{year}: roi={m['roi']:+.4f} n={m['n']} clv={m['clv_mean']:+.4f}"
        )
    print(" | ".join(base_parts))


if __name__ == "__main__":
    main()
