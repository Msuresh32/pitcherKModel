"""Settle every past daily paper card (K + hits-allowed) against final box scores.

Recomputes the full ledger on every run (idempotent: late boxscores or data
repairs self-heal on the next pass). Results for a card date D are looked up in
pitcher_game_logs under D or D+1, because the fetcher dates games by UTC
timestamp, which pushes starts of 8 PM ET or later onto the next calendar day.
Starters never pitch on consecutive days, so the two-day window is unambiguous.

Output: reports/daily/v2_results_ledger.json  (consumed by a6_build_artifact)
"""
from __future__ import annotations
import json, re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "reports" / "daily"
RAW = ROOT / "data" / "raw"


def profit(price: float, won: bool) -> float:
    if not won:
        return -1.0
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def settle(pitcher: str, day: str, logs: pd.DataFrame, stat: str):
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    g = logs[(logs.pitcher_name == pitcher) & (logs.d.isin([day, nxt]))]
    if not len(g):
        return None
    g = g.sort_values("d")  # prefer the card date itself over the UTC spillover
    return float(g[stat].iloc[0])


def outcome(actual: float | None, line: float, side: str, price: float):
    if actual is None:
        return "no data", None
    if actual == line:
        return "PUSH", 0.0
    won = actual > line if side == "OVER" else actual < line
    return ("WIN" if won else "LOSS"), round(profit(price, won), 3)


def main() -> None:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    logs = pd.read_csv(RAW / "pitcher_game_logs.csv", low_memory=False)
    logs["d"] = logs.game_date.astype(str).str[:10]

    k_rows, hit_rows = [], []
    for f in sorted(DAILY.glob("v2_plays_*.csv")):
        day = re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1)
        if day >= today:
            continue  # tonight's card settles tomorrow
        for _, p in pd.read_csv(f).iterrows():
            if p.get("signal") not in ("OVER", "UNDER"):
                continue
            ks = settle(p.pitcher, day, logs, "strikeouts")
            res, pnl = outcome(ks, p.line, p.signal, p.price)
            timing = p.get("odds_timing")
            k_rows.append({
                "date": day, "pitcher": p.pitcher, "tier": p.get("tier", ""),
                "timing": timing if isinstance(timing, str) else None,
                "side": p.signal, "line": float(p.line), "price": int(p.price),
                "actual_ks": None if ks is None else int(ks),
                "result": res, "pnl": pnl})
        hf = DAILY / f"v2_hits_paper_{day}.csv"
        if hf.exists():
            for _, p in pd.read_csv(hf).iterrows():
                h = settle(p.pitcher, day, logs, "hits_allowed")
                res, pnl = outcome(h, p.line, "UNDER", p.price)
                hit_rows.append({
                    "date": day, "pitcher": p.pitcher, "line": float(p.line),
                    "price": int(p.price), "edge": round(float(p.edge), 3),
                    "actual_hits": None if h is None else int(h),
                    "result": res, "pnl": pnl})

    k_rows.sort(key=lambda r: r["date"], reverse=True)
    hit_rows.sort(key=lambda r: r["date"], reverse=True)
    out = {"updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "k": k_rows, "hits": hit_rows}
    (DAILY / "v2_results_ledger.json").write_text(
        json.dumps(out, allow_nan=False), encoding="utf-8")
    ks = [r for r in k_rows if r["pnl"] is not None]
    hs = [r for r in hit_rows if r["pnl"] is not None]
    print(f"results ledger: {len(ks)} K plays settled ({sum(r['pnl'] for r in ks):+.2f}u), "
          f"{len(hs)} hits plays settled ({sum(r['pnl'] for r in hs):+.2f}u)")


if __name__ == "__main__":
    main()
