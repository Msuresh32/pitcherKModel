"""Bankroll simulation + staking optimization on the 2026 walk-forward ledger.

Rules compared (stakes as % of CURRENT bankroll, same-day bets sized off the
day's opening bankroll, settled together = realistic compounding):
  flat-f          : f% on every bet
  tiered-f        : f% base tier, 2f% conviction (edge >= 15pp)
  shrunk-kelly    : p_eff = p_mkt + lambda*(p_model - p_mkt);
                    f_i = mult * kelly(p_eff, b), capped at cap%
Selection: maximize median log-growth s.t. P(max drawdown > 40%) < 10%,
day-block bootstrap (2,000 resamples of the season's days).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
N_SIM = 2000


def load_bets():
    b = pd.read_csv("reports/v2_walkforward_2026_bets.csv")
    b["b"] = np.where(b.odds >= 0, b.odds / 100.0, 100.0 / np.abs(b.odds))
    b["p_mkt_side"] = b.p_mkt
    b["p_model_side"] = b.p_model
    return b


def day_groups(b):
    return [g[["b", "won", "p_model_side", "p_mkt_side", "tier"]].values
            for _, g in b.groupby("date")]


def stakes_for_day(day, rule):
    b_, won, pm, pk, tier = day[:, 0].astype(float), day[:, 1], \
        day[:, 2].astype(float), day[:, 3].astype(float), day[:, 4]
    kind = rule["kind"]
    if kind == "flat":
        return np.full(len(day), rule["f"])
    if kind == "tiered":
        conv = (pm - pk) >= 0.15
        return np.where(conv, 2 * rule["f"], rule["f"])
    if kind == "kelly":
        p_eff = pk + rule["lam"] * (pm - pk)
        f = (p_eff * (1 + b_) - 1) / b_
        f = np.clip(rule["mult"] * f, 0, rule["cap"])
        return f
    raise ValueError(kind)


def simulate(groups, rule, n_sim=N_SIM):
    nd = len(groups)
    term = np.empty(n_sim)
    maxdd = np.empty(n_sim)
    for s in range(n_sim):
        order = RNG.integers(0, nd, nd)
        bank, peak, dd = 1.0, 1.0, 0.0
        for gi in order:
            day = groups[gi]
            f = stakes_for_day(day, rule)
            won = day[:, 1].astype(bool)
            b_ = day[:, 0].astype(float)
            pnl = np.where(won, f * b_, -f).sum()
            bank *= max(1e-9, 1 + pnl)
            peak = max(peak, bank)
            dd = max(dd, 1 - bank / peak)
        term[s] = bank
        maxdd[s] = dd
    return {
        "median_terminal": float(np.median(term)),
        "p5_terminal": float(np.percentile(term, 5)),
        "median_log_growth": float(np.median(np.log(term))),
        "p_dd_gt30": float((maxdd > 0.30).mean()),
        "p_dd_gt40": float((maxdd > 0.40).mean()),
        "median_maxdd": float(np.median(maxdd)),
    }


def main():
    b = load_bets()
    groups = day_groups(b)
    print(f"{len(b)} bets over {len(groups)} days (2026 WF, T-4h ledger)\n", flush=True)

    rules = []
    for f in [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]:
        rules.append({"kind": "flat", "f": f, "name": f"flat {f*100:.2f}%"})
    for f in [0.0025, 0.005, 0.0075, 0.01, 0.015]:
        rules.append({"kind": "tiered", "f": f,
                      "name": f"tiered {f*100:.2f}%/{2*f*100:.2f}%"})
    for lam in [0.15, 0.25, 0.40]:
        for mult in [0.25, 0.5]:
            for cap in [0.02, 0.04]:
                rules.append({"kind": "kelly", "lam": lam, "mult": mult, "cap": cap,
                              "name": f"kelly lam={lam} mult={mult} cap={cap*100:.0f}%"})

    rows = []
    for r in rules:
        m = simulate(groups, r)
        m["name"] = r["name"]
        rows.append(m)
        print(f"{r['name']:32s} med_term={m['median_terminal']:.3f} "
              f"p5={m['p5_terminal']:.3f} medDD={m['median_maxdd']:.1%} "
              f"P(DD>30%)={m['p_dd_gt30']:.1%} P(DD>40%)={m['p_dd_gt40']:.1%}", flush=True)

    df = pd.DataFrame(rows)
    ok = df[df.p_dd_gt40 < 0.10]
    best = ok.sort_values("median_log_growth", ascending=False).head(5)
    print("\n=== Best rules (median growth s.t. P(maxDD>40%)<10%) ===", flush=True)
    print(best[["name", "median_terminal", "p5_terminal", "median_maxdd",
                "p_dd_gt30", "p_dd_gt40"]].round(3).to_string(index=False), flush=True)
    df.to_csv("research/v2/bankroll_sim_results.csv", index=False)


if __name__ == "__main__":
    main()
