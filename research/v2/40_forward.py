"""V2 Phase 4: ONE-SHOT forward test on 2026 (opening day .. Jul 10).

Run this ONCE, after research/v2/final_config.json is frozen from the 2025-only
strategy search. No parameter may be changed based on this script's output.

Applies the frozen model + filter to 2026 rows and reports full metrics.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("research/v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
strategy = __import__("30_strategy")
# models were pickled from 20_model_search run as __main__; restore class refs
_ms = __import__("20_model_search")
import __main__ as _mn
_mn.CountModel = _ms.CountModel
_mn.ClfModel = _ms.ClfModel

FORWARD_END = pd.Timestamp("2026-07-10")


def main():
    cfg = json.loads((OUT / "final_config.json").read_text())
    print("FROZEN CONFIG:", json.dumps(cfg, indent=2), flush=True)

    with open(OUT / "models" / "fitted.pkl", "rb") as f:
        bundle = pickle.load(f)
    fitted = bundle["fitted"]
    calibrators = bundle["calibrators"]
    stacker = bundle["stacker"]
    stack_base = bundle["stack_base"]
    count_names = bundle["count_names"]
    all_names = bundle["all_names"]
    feat_cols = bundle["feat_cols"]

    bets = pd.read_parquet(OUT / "bets.parquet")
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    b26 = bets[(bets.game_date.dt.year == 2026) & (bets.game_date <= FORWARD_END)].copy()
    b26 = b26[(b26["line"] % 1) == 0.5]
    b26 = b26[b26["outcome_push"] == 0]
    for c in feat_cols:
        if c in b26.columns:
            b26[c] = pd.to_numeric(b26[c], errors="coerce")
    b26[feat_cols] = b26[feat_cols].fillna(0.0)
    print(f"2026 rows: {len(b26):,} "
          f"({b26.game_date.min().date()} .. {b26.game_date.max().date()})", flush=True)

    count_feat_cols = [c for c in feat_cols if c != "line"]
    clf_feat_cols = count_feat_cols + ["line"]

    # base model predictions
    for name in all_names:
        m = fitted[name]
        if name in count_names:
            b26[f"p_{name}"] = m.predict_prob(b26[count_feat_cols], b26["line"])
        else:
            b26[f"p_{name}"] = m.predict_prob(b26[clf_feat_cols])
        p = np.clip(b26[f"p_{name}"].values, 1e-6, 1 - 1e-6)
        b26[f"p_{name}_iso"] = calibrators[name]["iso"].predict(p)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        b26[f"p_{name}_platt"] = calibrators[name]["platt"].predict_proba(z)[:, 1]

    Z = np.column_stack([
        np.log(np.clip(b26[f"p_{n}"].values, 1e-6, 1 - 1e-6) /
               (1 - np.clip(b26[f"p_{n}"].values, 1e-6, 1 - 1e-6)))
        for n in stack_base
    ])
    b26["p_stack"] = stacker.predict_proba(Z)[:, 1]
    b26["p_mean_count"] = b26[[f"p_{n}_iso" for n in count_names]].mean(axis=1)
    b26["p_mean_all"] = b26[[f"p_{n}_iso" for n in all_names]].mean(axis=1)

    # frozen strategy
    bets26 = strategy.make_bets(
        b26, cfg["model_col"],
        min_edge=cfg["min_edge"], min_ev=cfg.get("min_ev", 0.0),
        sides=cfg.get("sides", "both"), min_books=cfg.get("min_books", 1),
        odds_min=cfg.get("odds_min", -10000), odds_max=cfg.get("odds_max", 10000),
        lines=tuple(cfg.get("lines", (3.5, 9.5))),
    )
    m = strategy.bet_metrics(bets26)
    print("\n=== 2026 FORWARD TEST ===", flush=True)
    for k, v in m.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}", flush=True)

    # monthly
    if len(bets26):
        bets26["month"] = bets26.game_date.dt.to_period("M").astype(str)
        print("\nMonthly:", flush=True)
        print(bets26.groupby("month").agg(
            n=("pnl", "size"), wr=("won", "mean"), roi=("pnl", "mean"),
            units=("pnl", "sum"), clv=("clv", "mean")).to_string(), flush=True)
        print("\nBy line:", flush=True)
        print(bets26.groupby("line").agg(
            n=("pnl", "size"), wr=("won", "mean"), roi=("pnl", "mean"),
            clv=("clv", "mean")).to_string(), flush=True)
        print("\nBy book:", flush=True)
        print(bets26.groupby("book").agg(
            n=("pnl", "size"), wr=("won", "mean"), roi=("pnl", "mean")).to_string(), flush=True)
        print("\nBy side:", flush=True)
        print(bets26.groupby("bet_side").agg(
            n=("pnl", "size"), wr=("won", "mean"), roi=("pnl", "mean"),
            clv=("clv", "mean")).to_string(), flush=True)
        bets26.to_csv(OUT / "forward_2026_bets.csv", index=False)

    # calibration of the frozen model on ALL 2026 rows (not just bets)
    pc = cfg["model_col"]
    d = b26.dropna(subset=[pc]).copy()
    d["bucket"] = pd.qcut(d[pc], 10, duplicates="drop")
    cal = d.groupby("bucket", observed=True).agg(
        pred=(pc, "mean"), actual=("outcome_over", "mean"), n=(pc, "size"))
    print("\nCalibration (all 2026 rows):", flush=True)
    print(cal.to_string(), flush=True)

    pd.Series(m).to_json(OUT / "forward_2026_metrics.json")
    print("\nSaved forward_2026_bets.csv / forward_2026_metrics.json", flush=True)


if __name__ == "__main__":
    main()
