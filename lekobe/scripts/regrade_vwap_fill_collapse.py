from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
FILLS = BASE / "outputs" / "kalshi_kprop_fills.csv"
CLOSES = BASE / "scratch" / "audits" / "kalshi_kprop_fast_close_coverage.csv"
OUT = BASE / "scratch" / "audits" / "kalshi_kprop_vwap_regrade.csv"
DIFF_OUT = BASE / "scratch" / "audits" / "kalshi_kprop_vwap_vs_simple_diffs.csv"
SUMMARY_OUT = BASE / "scratch" / "audits" / "kalshi_kprop_vwap_regrade_summary.csv"


def event_date_from_ticker(ticker: str) -> str:
    match = re.search(r"KXMLBKS-(\d{2})([A-Z]{3})(\d{2})", str(ticker))
    if not match:
        return ""
    yy, mon, dd = match.groups()
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


def signed_no_qty(row: pd.Series) -> float:
    action = str(row["action"]).lower()
    side = str(row["side"]).lower()
    qty = float(row["count_fp"])
    if action == "buy" and side == "no":
        return qty
    if action == "sell" and side == "yes":
        return qty
    if action == "buy" and side == "yes":
        return -qty
    if action == "sell" and side == "no":
        return -qty
    return 0.0


def collapse_positions(fills: pd.DataFrame) -> pd.DataFrame:
    fills = fills.copy()
    fills["count_fp"] = pd.to_numeric(fills["count_fp"], errors="coerce").fillna(0.0)
    fills["yes_price_cents"] = pd.to_numeric(fills["yes_price_cents"], errors="coerce")
    fills["no_price_cents"] = pd.to_numeric(fills["no_price_cents"], errors="coerce")
    fills["fill_price_cents"] = pd.to_numeric(fills["fill_price_cents"], errors="coerce")
    fills["is_taker_bool"] = fills["is_taker"].astype(str).str.lower().eq("true")
    fills["_signed_no_qty"] = fills.apply(signed_no_qty, axis=1)

    rows = []
    for ticker, group in fills.groupby("ticker", sort=True):
        net_no_qty = group["_signed_no_qty"].sum()
        if net_no_qty > 0:
            side = "UNDER"
            price_col = "no_price_cents"
        elif net_no_qty < 0:
            side = "OVER"
            price_col = "yes_price_cents"
        else:
            side = "FLAT"
            price_col = "fill_price_cents"

        weights = group["count_fp"].abs()
        total_contracts = abs(net_no_qty)
        gross_contracts = weights.sum()
        if gross_contracts <= 0:
            continue

        simple_mean = group[price_col].mean()
        vwap = np.average(group[price_col], weights=weights)
        taker_frac = np.average(group["is_taker_bool"].astype(float), weights=weights)

        rows.append(
            {
                "ticker": ticker,
                "event_date": event_date_from_ticker(ticker),
                "pitcher": group["pitcher"].dropna().iloc[0],
                "line": float(group["line"].dropna().iloc[0]),
                "side": side,
                "total_contracts": float(total_contracts),
                "gross_contracts": float(gross_contracts),
                "simple_mean_fill_cents": float(simple_mean),
                "vwap_fill_cents": float(vwap),
                "simple_minus_vwap_cents": float(simple_mean - vwap),
                "abs_gap_cents": float(abs(simple_mean - vwap)),
                "gap_gt_1c": bool(abs(simple_mean - vwap) > 1.0),
                "maker_taker": "taker" if taker_frac > 0.5 else "maker",
                "taker_contract_frac": float(taker_frac),
                "fill_count": int(len(group)),
                "maker_contracts": float(group.loc[~group["is_taker_bool"], "count_fp"].sum()),
                "taker_contracts": float(group.loc[group["is_taker_bool"], "count_fp"].sum()),
                "dollar_stake": float(total_contracts * vwap / 100.0),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["scope", "event_date", "grade_source"], dropna=False):
        scope, event_date, grade_source = keys
        rows.append(
            {
                "scope": scope,
                "event_date": event_date,
                "grade_source": grade_source,
                "positions": len(group),
                "contracts": group["total_contracts"].sum(),
                "dollar_stake": group["dollar_stake"].sum(),
                "mean_clv_vwap_pp": group["clv_vwap_pp"].mean(),
                "positive_clv_vwap_pct": (group["clv_vwap_pp"] > 0).mean() * 100,
                "mean_clv_simple_pp": group["clv_simple_pp"].mean(),
                "positive_clv_simple_pct": (group["clv_simple_pp"] > 0).mean() * 100,
                "vwap_minus_simple_clv_pp": (
                    group["clv_vwap_pp"].mean() - group["clv_simple_pp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    fills = pd.read_csv(FILLS)
    fills = fills[fills["ticker"].astype(str).str.startswith("KXMLBKS")].copy()
    total_fill_contracts = pd.to_numeric(fills["count_fp"], errors="coerce").sum()

    positions = collapse_positions(fills)
    closes = pd.read_csv(CLOSES)[["ticker", "close_source", "grade_source", "close_cents"]]
    out = positions.merge(closes, on="ticker", how="left")
    out["clv_vwap_pp"] = out["close_cents"] - out["vwap_fill_cents"]
    out["clv_simple_pp"] = out["close_cents"] - out["simple_mean_fill_cents"]
    out["clv_change_vwap_minus_simple_pp"] = out["clv_vwap_pp"] - out["clv_simple_pp"]

    diffs = out[out["abs_gap_cents"] > 0.0001].copy()
    summary_parts = []
    by_slate = out.copy()
    by_slate["scope"] = "slate"
    summary_parts.append(by_slate)
    overall = out.copy()
    overall["scope"] = "overall"
    overall["event_date"] = "ALL"
    summary_parts.append(overall)
    summary = summarize(pd.concat(summary_parts, ignore_index=True))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    diffs.to_csv(DIFF_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"TOTAL_KXMLBKS_FILL_ROWS {len(fills)}")
    print(f"TOTAL_FILL_CONTRACTS {total_fill_contracts:.2f}")
    print(f"POSITIONS {len(out)}")
    print(f"TOTAL_NET_CONTRACTS {out['total_contracts'].sum():.2f}")
    print(f"TOTAL_DOLLAR_STAKE {out['dollar_stake'].sum():.2f}")
    print(f"POSITIONS_SIMPLE_NE_VWAP {len(diffs)}")
    print(f"POSITIONS_GAP_GT_1C {int(out['gap_gt_1c'].sum())}")
    print("OVERALL")
    overall_summary = summary[
        (summary["scope"] == "overall") & (summary["event_date"] == "ALL")
    ].sort_values("grade_source")
    print(overall_summary.to_string(index=False))
    print("BY_SLATE")
    by_slate_summary = summary[summary["scope"] == "slate"].sort_values(
        ["event_date", "grade_source"]
    )
    print(by_slate_summary.to_string(index=False))
    print("DIFFS")
    diff_cols = [
        "event_date",
        "pitcher",
        "line",
        "side",
        "simple_mean_fill_cents",
        "vwap_fill_cents",
        "simple_minus_vwap_cents",
        "gap_gt_1c",
        "total_contracts",
        "dollar_stake",
        "maker_taker",
        "close_source",
        "clv_simple_pp",
        "clv_vwap_pp",
    ]
    if diffs.empty:
        print("NONE")
    else:
        print(diffs.sort_values("abs_gap_cents", ascending=False)[diff_cols].to_string(index=False))
    print(f"OUT {OUT}")
    print(f"DIFF_OUT {DIFF_OUT}")
    print(f"SUMMARY_OUT {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
