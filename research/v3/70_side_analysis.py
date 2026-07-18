"""Side-of-market analysis: is the under-side edge real in BOTH years?

Motivation (declared before running): every saved ledger shows unders
outperforming overs. This has a structural prior — recreational money
concentrates on overs, so books shade over prices — which makes it a
hypothesis worth testing across both seasons, not a 2026-only artifact.

This script does NOT change any routed rule. It measures, with date-block
bootstrap CIs, the under-only and over-only cuts of the existing frozen
ledgers. If the under cut is positive in both years with CLV support, it
becomes a *tracked paper cohort* (derivable from the daily cards' side
column) with a pre-registered promotion gate — never a live rule today.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "research" / "v2"
V3 = ROOT / "research" / "v3"
sys.path.insert(0, str(V2))
S = importlib.import_module("30_strategy")

LEDGERS = [
    # label, source, build
    ("2025 T-4h raw8pp", "build_2025_t4", None),
    ("2025 T-12h LCB", V3 / "upgrade_bets_2025_t12.csv", None),
    ("2026 T-4h H0 raw8pp", V3 / "h0_bets_2026_t4.csv", None),
    ("2026 T-4h H3 raw8pp", V3 / "upgrade_bets_2026_t4.csv", None),
    ("2026 T-12h H0 raw8pp", V3 / "h0_bets_2026_t12.csv", None),
    ("2026 T-12h LCB", V3 / "upgrade_bets_2026_t12.csv", None),
]


def load(label, src):
    if src == "build_2025_t4":
        p25 = pd.read_parquet(V2 / "preds_2025.parquet")
        p25["game_date"] = pd.to_datetime(p25.game_date)
        p25 = p25[p25.outcome_push == 0]
        return S.make_bets(p25, "p_mean_count", min_edge=0.08)
    df = pd.read_csv(src)
    df["game_date"] = pd.to_datetime(df.game_date)
    return df


def cut(label, side, bets):
    b = bets if side == "all" else bets[bets.bet_side == side]
    if not len(b):
        return None
    m = S.bet_metrics(b)
    return {"ledger": label, "side": side, "n": int(m["n"]),
            "roi": round(m["roi"], 4),
            "roi_lo90": round(m["roi_lo90"], 4),
            "roi_hi90": round(m["roi_hi90"], 4),
            "clv_mean_pp": round(m["clv_mean"] * 100, 2),
            "clv_pos_pct": round(m["clv_pos_pct"], 3)}


def main() -> None:
    rows = []
    for label, src, _ in LEDGERS:
        bets = load(label, src)
        for side in ("all", "over", "under"):
            r = cut(label, side, bets)
            if r:
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(V3 / "side_analysis.csv", index=False)
    print(out.to_string(index=False))

    unders = out[out.side == "under"]
    print("\nUnder-only summary: positive ROI in "
          f"{int((unders.roi > 0).sum())}/{len(unders)} ledgers; "
          f"positive CLV in {int((unders.clv_mean_pp > 0).sum())}/{len(unders)}; "
          f"lo90>0 in {int((unders.roi_lo90 > 0).sum())}/{len(unders)}")


if __name__ == "__main__":
    main()
