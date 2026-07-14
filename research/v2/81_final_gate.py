"""FINAL GATE - run ONCE. Selected head: H1 (pre-registered 2026-07-11).

Window: 2026-06-01 .. 2026-07-10. Deployable iff:
  ROI > 0, OR (ROI > -3% AND clv_mean > +0.003 with intact edge->CLV structure).
Also reports combined Mar-Jul walk-forward performance.
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

HEAD, THR = "p_h1", 0.08
GATE_START = pd.Timestamp("2026-06-01")


def show(df, label):
    bx = S.make_bets(df, HEAD, min_edge=THR)
    m = S.bet_metrics(bx)
    print(f"\n=== {label} ===", flush=True)
    if m.get("n", 0) == 0:
        print("no bets"); return None, None
    for k, v in m.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}")
    bx["month"] = bx.game_date.dt.to_period("M").astype(str)
    print(bx.groupby("month").agg(n=("pnl", "size"), wr=("won", "mean"),
          roi=("pnl", "mean"), units=("pnl", "sum"), clv=("clv", "mean"))
          .round(4).to_string(), flush=True)
    return bx, m


def main():
    b26 = pd.read_parquet(OUT / "adaptive_preds_2026.parquet")
    b26["game_date"] = pd.to_datetime(b26["game_date"])

    gate = b26[b26.game_date >= GATE_START].copy()
    bx, m = show(gate, "FINAL GATE Jun 1 - Jul 10 2026 (H1, edge>=0.08)")

    # edge->CLV structure in gate window
    dd = gate.dropna(subset=[HEAD, "p_over_open", "clv_over"]).copy()
    dd["e"] = dd[HEAD] - dd["p_over_open"]
    dd["eb"] = pd.qcut(dd["e"], 5, duplicates="drop")
    print("\nedge->CLV structure (gate window, all rows):")
    print(dd.groupby("eb", observed=True).agg(n=("clv_over", "size"),
          clv_over=("clv_over", "mean"), over_rate=("outcome_over", "mean"),
          mkt=("p_over_open", "mean")).round(4).to_string(), flush=True)

    if m:
        roi, clv = m["roi"], m["clv_mean"]
        deployable = (roi > 0) or (roi > -0.03 and clv > 0.003)
        print(f"\nGATE CRITERIA: roi={roi:+.4f} clv={clv:+.4f} -> "
              f"{'PASS - DEPLOYABLE (shadow-scale)' if deployable else 'FAIL - NOT DEPLOYABLE'}",
              flush=True)
        if bx is not None:
            bx.to_csv(OUT / "final_gate_bets.csv", index=False)

    # combined full 2026 walk-forward (context)
    show(b26, "COMBINED 2026 walk-forward Mar 26 - Jul 10 (H1)")


if __name__ == "__main__":
    main()
