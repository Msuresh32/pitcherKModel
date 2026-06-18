from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sharp_benchmark import consensus_from_rows

RAW = BASE / "scratch/audits/optimized_unders_20260615_16_sharp_raw.csv"
KALSHI = BASE / "scratch/audits/kalshi_kprop_fast_close_coverage.csv"
OUT = BASE / "scratch/audits/optimized_unders_20260615_16_sharp_grade.csv"
SUMMARY = BASE / "scratch/audits/optimized_unders_20260615_16_sharp_grade_summary.csv"

TARGETS = [
    ("J.T. Ginn", "2026-06-15", 4.5, "UNDER", 43.0),
    ("Jared Jones", "2026-06-15", 4.5, "UNDER", 42.0),
    ("Ryan Gusto", "2026-06-15", 3.5, "UNDER", 40.0),
    ("Ryne Nelson", "2026-06-15", 4.5, "UNDER", 43.0),
    ("Shota Imanaga", "2026-06-15", 5.5, "UNDER", 43.0),
    ("Andre Pallante", "2026-06-16", 3.5, "UNDER", 46.0),
    ("Drew Rasmussen", "2026-06-16", 4.5, "UNDER", 39.0),
    ("Justin Wrobleski", "2026-06-16", 3.5, "UNDER", 40.0),
    ("Kodai Senga", "2026-06-16", 4.5, "UNDER", 41.0),
    ("Robert Gasser", "2026-06-16", 4.5, "UNDER", 40.0),
]

def norm(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def last_key(x: object) -> str:
    n = norm(x)
    return n.split("_")[-1] if n else ""


def american_to_prob(o: object) -> float:
    try:
        o = float(o)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return np.nan


def no_vig_under(over_odds: float, under_odds: float) -> float:
    op = american_to_prob(over_odds)
    up = american_to_prob(under_odds)
    if not np.isfinite(op) or not np.isfinite(up) or op + up <= 0:
        return np.nan
    return up / (op + up) * 100.0


def ticker_date(ticker: str) -> str:
    m = re.search(r"KXMLBKS-(\d{2})([A-Z]{3})(\d{2})", str(ticker))
    if not m:
        return ""
    yy, mon, dd = m.groups()
    months = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    return f"20{yy}-{months[mon]:02d}-{int(dd):02d}"


def paired_price(group: pd.DataFrame) -> tuple[float, float, float, str] | None:
    """Return over, under, no-vig under, market for best two-sided pair."""
    for market_name in ["pitcher_strikeouts", "pitcher_strikeouts_alternate"]:
        m = group[group["market"].eq(market_name)]
        over = m[m["side"].eq("OVER")]
        under = m[m["side"].eq("UNDER")]
        if not over.empty and not under.empty:
            oo = float(over.iloc[-1]["american_odds"])
            uo = float(under.iloc[-1]["american_odds"])
            return oo, uo, no_vig_under(oo, uo), market_name
    over = group[group["side"].eq("OVER")]
    under = group[group["side"].eq("UNDER")]
    if over.empty or under.empty:
        return None
    oo = float(over.iloc[-1]["american_odds"])
    uo = float(under.iloc[-1]["american_odds"])
    return oo, uo, no_vig_under(oo, uo), "mixed"


def choose_close(raw: pd.DataFrame, pitcher: str, line: float) -> dict[str, object]:
    cand = raw[raw["clean_name"].eq(norm(pitcher))].copy()
    if cand.empty:
        cand = raw[raw["last_key"].eq(last_key(pitcher))].copy()
    if cand.empty:
        return {"close_source": "missing", "match_type": "missing"}

    pairs: list[dict[str, object]] = []
    for (book, close_line), g in cand.groupby(["bookmaker", "line"], dropna=False):
        pair = paired_price(g)
        if pair is None:
            continue
        over_odds, under_odds, under_cents, market_name = pair
        if not np.isfinite(under_cents):
            continue
        pairs.append(
            {
                "bookmaker": str(book),
                "matched_line": float(close_line),
                "hook_diff": float(close_line) - float(line),
                "over_close": int(over_odds),
                "under_close": int(under_odds),
                "sharp_under_cents": float(under_cents),
                "market": market_name,
                "last_update": str(g["last_update"].dropna().iloc[-1]) if g["last_update"].notna().any() else "",
            }
        )
    if not pairs:
        return {"close_source": "missing", "match_type": "missing"}

    pair_df = pd.DataFrame(pairs)

    def from_pool(pool: pd.DataFrame, match_type: str) -> dict[str, object] | None:
        if pool.empty:
            return None
        choice = consensus_from_rows(pool, "sharp_under_cents")
        if not np.isfinite(choice["value"]):
            return None
        rec = pool.iloc[0].to_dict()
        rec["over_close"] = np.nan
        rec["under_close"] = np.nan
        rec["sharp_under_cents"] = choice["value"]
        rec["market"] = "de_vigged_book_consensus"
        rec["last_update"] = ""
        rec["close_source"] = choice["source"]
        rec["sharp_books"] = choice["books"]
        rec["sharp_book_count"] = choice["book_count"]
        rec["sharp_confidence"] = choice["confidence"]
        rec["match_type"] = match_type
        return rec

    # First use exact-line Pin+FD+BetOnline no-vig consensus.
    exact = pair_df[np.isclose(pair_df["matched_line"], float(line))]
    rec = from_pool(exact, "exact")
    if rec is not None:
        return rec

    # If no exact two-sided close exists, use the nearest line consensus.
    pair_df["_abs_hook"] = pair_df["hook_diff"].abs()
    nearest_line = pair_df.sort_values(["_abs_hook", "matched_line"]).iloc[0]["matched_line"]
    g = pair_df[np.isclose(pair_df["matched_line"], nearest_line)]
    rec = from_pool(g, "exact" if np.isclose(nearest_line, float(line)) else "nearest")
    return rec if rec is not None else {"close_source": "missing", "match_type": "missing"}


def main() -> None:
    raw = pd.read_csv(RAW)
    raw["clean_name"] = raw["pitcher_name"].map(norm)
    raw["last_key"] = raw["pitcher_name"].map(last_key)
    raw["bookmaker"] = raw["bookmaker"].astype(str).str.lower()
    raw["side"] = raw["side"].astype(str).str.upper()
    raw["line"] = pd.to_numeric(raw["line"], errors="coerce")

    kalshi = pd.read_csv(KALSHI)
    kalshi = kalshi.assign(_date=kalshi["ticker"].map(ticker_date), _clean=kalshi["pitcher"].map(norm))

    rows = []
    for pitcher, date, line, side, fill in TARGETS:
        rec = {
            "pitcher": pitcher,
            "date": date,
            "line": line,
            "side": side,
            "vwap_fill_cents": fill,
        }
        close = choose_close(raw, pitcher, line)
        rec.update(close)
        if np.isfinite(rec.get("sharp_under_cents", np.nan)):
            rec["sharp_side_close_cents"] = rec["sharp_under_cents"]
            rec["sharp_clv_pp"] = rec["sharp_side_close_cents"] - fill
        else:
            rec["sharp_side_close_cents"] = np.nan
            rec["sharp_clv_pp"] = np.nan

        km = kalshi[
            kalshi["_date"].eq(date)
            & kalshi["_clean"].eq(norm(pitcher))
            & np.isclose(kalshi["line"], line)
            & kalshi["net_side"].eq(side)
        ]
        if not km.empty:
            k = km.iloc[0]
            rec["contracts"] = float(k["total_contracts"])
            rec["kalshi_close_cents"] = float(k["close_cents"])
            rec["kalshi_clv_pp"] = float(k["close_cents"]) - fill
            rec["execution_style"] = str(k["execution_style"])
        else:
            rec["contracts"] = np.nan
            rec["kalshi_close_cents"] = np.nan
            rec["kalshi_clv_pp"] = np.nan
            rec["execution_style"] = ""
        rec["benchmark_delta_pp"] = rec["sharp_clv_pp"] - rec["kalshi_clv_pp"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    covered = out[out["sharp_clv_pp"].notna()].copy()
    summary = pd.DataFrame(
        [
            {
                "target_count": len(out),
                "under_count": int(out["side"].eq("UNDER").sum()),
                "sharp_coverage": int(out["sharp_clv_pp"].notna().sum()),
                "missing_sharp": int(out["sharp_clv_pp"].isna().sum()),
                "exact_line_matches": int(out["match_type"].eq("exact").sum()),
                "nearest_line_matches": int(out["match_type"].eq("nearest").sum()),
                "mean_sharp_clv_pp": float(covered["sharp_clv_pp"].mean()),
                "contract_weighted_sharp_clv_pp": float(np.average(covered["sharp_clv_pp"], weights=covered["contracts"])),
                "positive_sharp_clv_pct": float((covered["sharp_clv_pp"] > 0).mean() * 100),
                "mean_kalshi_clv_pp_same_10": float(covered["kalshi_clv_pp"].mean()),
                "contract_weighted_kalshi_clv_pp_same_10": float(np.average(covered["kalshi_clv_pp"], weights=covered["contracts"])),
                "positive_kalshi_clv_pct_same_10": float((covered["kalshi_clv_pp"] > 0).mean() * 100),
                "mean_benchmark_delta_pp": float(covered["benchmark_delta_pp"].mean()),
                "contract_weighted_benchmark_delta_pp": float(np.average(covered["benchmark_delta_pp"], weights=covered["contracts"])),
            }
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    summary.to_csv(SUMMARY, index=False)

    display_cols = [
        "pitcher",
        "date",
        "line",
        "side",
        "vwap_fill_cents",
        "close_source",
        "match_type",
        "matched_line",
        "hook_diff",
        "over_close",
        "under_close",
        "sharp_side_close_cents",
        "sharp_clv_pp",
        "kalshi_close_cents",
        "kalshi_clv_pp",
        "benchmark_delta_pp",
        "contracts",
    ]
    print("RULE_COMPLIANT_UNDERS", int(out["side"].eq("UNDER").sum()))
    print("SHARP_COVERAGE", f"{int(out['sharp_clv_pp'].notna().sum())}/{len(out)}")
    print("BOOK_SOURCE_COUNTS")
    print(out["close_source"].value_counts(dropna=False).to_string())
    print("MATCH_TYPE_COUNTS")
    print(out["match_type"].value_counts(dropna=False).to_string())
    print("SUMMARY")
    print(summary.round(6).to_string(index=False))
    print("TABLE")
    print(out[display_cols].round(6).to_string(index=False))
    print("OUT", OUT)
    print("SUMMARY_OUT", SUMMARY)


if __name__ == "__main__":
    main()
