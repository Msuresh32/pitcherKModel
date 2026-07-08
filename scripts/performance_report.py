"""
performance_report.py — V4 live performance tracker

Reads data/logs/live_bets_log.csv and produces:
  - Overall P&L, ROI, win rate, CLV
  - V2_core vs V4_extra breakdown  ← the key research question
  - Over vs under breakdown
  - Monthly performance
  - Pitcher-level leaderboard
  - Edge_gap_product bucket breakdown
  - Exchange pricing impact analysis
  - Writes reports/performance/live_performance_summary.md

Usage:
    python scripts/performance_report.py
    python scripts/performance_report.py --since 2026-07-01
    python scripts/performance_report.py --bucket V4_extra   # filter to one bucket
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG_FILE    = Path("data/logs/live_bets_log.csv")
REPORT_FILE = Path("reports/performance/live_performance_summary.md")

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _amer_to_dec(odds: float) -> float:
    odds = float(odds)
    return (1 + odds / 100) if odds >= 0 else (1 + 100 / abs(odds))


def _load_log(since: str | None = None, bucket_filter: str | None = None) -> pd.DataFrame:
    if not LOG_FILE.exists():
        print(f"[performance_report] {LOG_FILE} not found — no data yet.")
        sys.exit(0)

    df = pd.read_csv(LOG_FILE, dtype=str)
    if df.empty:
        print("[performance_report] Log is empty — no bets yet.")
        sys.exit(0)

    # Parse numeric columns
    for col in ("edge_pct", "abs_proj_gap", "edge_gap_product",
                "sportsbook_odds", "price_improvement_cents",
                "stake", "result", "profit", "closing_odds", "clv_cents",
                "model_prob", "market_prob", "line"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if since:
        df = df[df["date"] >= pd.Timestamp(since)].copy()

    if bucket_filter:
        df = df[df["strategy_bucket"].astype(str).str.strip() == bucket_filter].copy()

    return df


def _settled(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have a recorded result AND stake."""
    has_result = df["result"].notna()
    has_stake  = df["stake"].notna()
    return df[has_result & has_stake].copy()


def _roi(settled: pd.DataFrame) -> float:
    total_staked = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    return float(total_profit / total_staked) if total_staked > 0 else float("nan")


def _win_rate(settled: pd.DataFrame) -> float:
    if settled.empty:
        return float("nan")
    # Won = profit > 0
    won = settled["profit"] > 0
    return float(won.mean())


def _mean_clv(settled: pd.DataFrame) -> tuple[float, int]:
    """Return (mean_clv_cents, coverage_count)."""
    clv = settled["clv_cents"].dropna()
    return (float(clv.mean()) if not clv.empty else float("nan"), len(clv))


def _summary_row(label: str, grp: pd.DataFrame) -> dict:
    s = _settled(grp)
    n_all      = len(grp)
    n_settled  = len(s)
    if n_settled == 0:
        return {
            "label": label, "n_all": n_all, "n_settled": 0,
            "win_rate": "—", "roi": "—", "total_stake": "—",
            "total_profit": "—", "mean_clv": "—",
        }
    wr     = _win_rate(s)
    roi    = _roi(s)
    stake  = s["stake"].sum()
    profit = s["profit"].sum()
    clv_m, clv_n = _mean_clv(s)
    return {
        "label":        label,
        "n_all":        n_all,
        "n_settled":    n_settled,
        "win_rate":     f"{wr:.1%}",
        "roi":          f"{roi:+.2%}" if not np.isnan(roi) else "—",
        "total_stake":  f"${stake:.0f}",
        "total_profit": f"${profit:+.0f}",
        "mean_clv":     f"{clv_m:+.1f}c ({clv_n})" if not np.isnan(clv_m) else f"— ({clv_n})",
    }


def _table_from_rows(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_No data._"
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    body   = []
    for r in rows:
        cells = [str(r.get(c, "—")) for c in cols]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


# ── Section builders ──────────────────────────────────────────────────────────

def _section_overall(df: pd.DataFrame) -> tuple[str, dict]:
    s = _settled(df)
    lines = ["## Overall Performance", ""]

    if s.empty:
        lines.append("_No settled bets with stake recorded yet._")
        return "\n".join(lines), {}

    wr          = _win_rate(s)
    roi         = _roi(s)
    total_stake = s["stake"].sum()
    total_profit= s["profit"].sum()
    clv_m, clv_n= _mean_clv(s)
    n_open      = len(df[df["result"].isna()])

    # Drawdown
    s_sorted    = s.sort_values("date")
    cum_profit  = s_sorted["profit"].cumsum()
    peak        = cum_profit.cummax()
    max_dd      = (cum_profit - peak).min()

    has_impr  = df["price_improvement_cents"].notna()
    mean_impr = df.loc[has_impr, "price_improvement_cents"].mean() if has_impr.any() else float("nan")

    lines += [
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Total logged | {len(df)} ({n_open} open) |",
        f"| Settled (with stake) | {len(s)} |",
        f"| Win rate | {wr:.1%} |",
        f"| ROI | {roi:+.2%} |",
        f"| Total staked | ${total_stake:,.0f} |",
        f"| Total profit | ${total_profit:+,.0f} |",
        f"| Max drawdown | ${max_dd:+,.0f} |",
        f"| Mean CLV | {clv_m:+.1f}c ({clv_n} bets) |" if not np.isnan(clv_m) else f"| Mean CLV | — ({clv_n} bets) |",
        f"| Mean price improvement | {mean_impr:+.1f}c |" if not np.isnan(mean_impr) else "| Mean price improvement | — |",
    ]

    stats = {"roi": roi, "wr": wr, "profit": total_profit, "stake": total_stake}
    return "\n".join(lines), stats


def _section_buckets(df: pd.DataFrame) -> str:
    """V2_core vs V4_extra — the core research question."""
    lines = [
        "## Strategy Bucket Comparison",
        "",
        "> **Key research question**: Is V4_extra (6 ≤ edge×gap < 12) profitable?",
        "> Does exchange pricing make it worth taking even if raw sportsbook ROI is lower?",
        "",
    ]

    cols = ["label", "n_all", "n_settled", "win_rate", "roi",
            "total_stake", "total_profit", "mean_clv"]
    col_hdrs = ["Bucket", "Candidates", "Settled", "Win%", "ROI",
                "Total Staked", "Total P&L", "Mean CLV"]

    rows = []
    for bucket in ["V2_core", "V4_extra", "ALL"]:
        grp = df if bucket == "ALL" else df[df["strategy_bucket"].astype(str) == bucket]
        rows.append(_summary_row(bucket, grp))

    lines.append(_table_from_rows(rows, cols))
    lines.append("")
    return "\n".join(lines)


def _section_direction(df: pd.DataFrame) -> str:
    lines = ["## Over vs Under Performance", ""]
    cols = ["label", "n_all", "n_settled", "win_rate", "roi", "total_profit", "mean_clv"]
    col_hdrs = ["Side", "Candidates", "Settled", "Win%", "ROI", "P&L", "CLV"]
    rows = []
    for side in ["over", "under", "ALL"]:
        grp = df if side == "ALL" else df[df["side"].astype(str).str.lower() == side]
        rows.append(_summary_row(side, grp))
    lines.append(_table_from_rows(rows, cols))
    lines.append("")
    return "\n".join(lines)


def _section_monthly(df: pd.DataFrame) -> str:
    lines = ["## Monthly Performance", ""]
    if "date" not in df.columns or df["date"].isna().all():
        return "\n".join(lines + ["_No date data._", ""])

    df = df.copy()
    df["month"] = df["date"].dt.month
    df["month_label"] = df["month"].map(lambda m: MONTH_NAMES.get(int(m), str(m)))

    cols = ["label", "n_all", "n_settled", "win_rate", "roi", "total_profit", "mean_clv"]
    rows = []
    for m in sorted(df["month"].dropna().unique().astype(int)):
        grp = df[df["month"] == m]
        rows.append(_summary_row(MONTH_NAMES.get(m, str(m)), grp))
    lines.append(_table_from_rows(rows, cols))
    lines.append("")
    return "\n".join(lines)


def _section_pitchers(df: pd.DataFrame, top_n: int = 20) -> str:
    lines = ["## Pitcher-Level Performance", ""]
    s = _settled(df)
    if s.empty:
        return "\n".join(lines + ["_No settled data._", ""])

    pitcher_col = "pitcher" if "pitcher" in s.columns else "pitcher_name"
    grouped = s.groupby(pitcher_col)

    rows = []
    for name, grp in grouped:
        total_profit = grp["profit"].sum()
        roi          = _roi(grp)
        wr           = _win_rate(grp)
        clv_m, clv_n = _mean_clv(grp)
        rows.append({
            "pitcher":  name,
            "n":        len(grp),
            "win_rate": f"{wr:.0%}",
            "roi":      f"{roi:+.2%}" if not np.isnan(roi) else "—",
            "profit":   f"${total_profit:+.0f}",
            "clv":      f"{clv_m:+.1f}c" if not np.isnan(clv_m) else "—",
            "_sort":    total_profit,
        })

    rows.sort(key=lambda r: r["_sort"], reverse=True)
    cols_display = ["pitcher", "n", "win_rate", "roi", "profit", "clv"]

    lines.append(f"_Top {min(top_n, len(rows))} by total profit (settled bets)_")
    lines.append("")
    lines.append(_table_from_rows(rows[:top_n], cols_display))
    lines.append("")

    if len(rows) > top_n:
        lines.append("_Bottom 5 by total profit:_")
        lines.append("")
        lines.append(_table_from_rows(rows[-5:], cols_display))
        lines.append("")

    return "\n".join(lines)


def _section_eg_buckets(df: pd.DataFrame) -> str:
    """ROI broken down by edge_gap_product bucket."""
    lines = ["## Edge×Gap Product Buckets", ""]
    if "edge_gap_product" not in df.columns or df["edge_gap_product"].isna().all():
        return "\n".join(lines + ["_edge_gap_product not available._", ""])

    bins   = [0, 6, 8, 10, 12, 15, 18, 20, 25, 9999]
    labels = ["<6", "6–8", "8–10", "10–12", "12–15", "15–18", "18–20", "20–25", "25+"]
    df = df.copy()
    df["eg_bucket"] = pd.cut(df["edge_gap_product"], bins=bins, labels=labels, right=False)

    cols = ["label", "n_all", "n_settled", "win_rate", "roi", "total_profit", "mean_clv"]
    rows = []
    for lbl in labels:
        grp = df[df["eg_bucket"].astype(str) == lbl]
        if len(grp) == 0:
            continue
        rows.append(_summary_row(f"eg {lbl}", grp))

    lines.append(_table_from_rows(rows, cols))
    lines.append("")
    lines.append(
        "> V4 research: efficient frontier peaks at 5–6. "
        "V2_core is 12+. Monitor both to track whether extra volume pays off."
    )
    lines.append("")
    return "\n".join(lines)


def _section_clv(df: pd.DataFrame) -> str:
    """CLV analysis — is the model beating the closing line?"""
    lines = ["## Closing Line Value (CLV)", ""]
    s = _settled(df)
    clv_data = s[s["clv_cents"].notna()]

    if clv_data.empty:
        lines.append("_No CLV data yet. Add --closing-odds via update_results.py after games close._")
        return "\n".join(lines + [""])

    clv = clv_data["clv_cents"]
    lines += [
        f"- Coverage: {len(clv_data)}/{len(s)} settled bets",
        f"- Mean CLV: **{clv.mean():+.2f}c**",
        f"- Median CLV: {clv.median():+.1f}c",
        f"- Positive CLV bets: {(clv > 0).sum()} ({(clv > 0).mean():.0%})",
        f"- CLV std: {clv.std():.1f}c",
        "",
        "> Positive mean CLV confirms the model is identifying genuine market inefficiencies,",
        "> not just variance. It is a leading indicator of long-run profitability.",
        "",
    ]

    # CLV by bucket
    for bucket in ["V2_core", "V4_extra"]:
        grp = clv_data[clv_data["strategy_bucket"].astype(str) == bucket]
        if grp.empty:
            continue
        clv_m, clv_n = _mean_clv(grp)
        lines.append(f"- **{bucket}** CLV: {clv_m:+.2f}c ({clv_n} bets)")
    lines.append("")

    return "\n".join(lines)


# ── Full report ───────────────────────────────────────────────────────────────

def _build_report(df: pd.DataFrame, since: str | None) -> str:
    since_str = f" (since {since})" if since else ""
    parts = [
        f"# V4 Live Performance Report{since_str}",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_Source: {LOG_FILE}_",
        "",
    ]

    overall_section, stats = _section_overall(df)
    parts.append(overall_section)
    parts.append("")
    parts.append(_section_buckets(df))
    parts.append(_section_direction(df))
    parts.append(_section_monthly(df))
    parts.append(_section_eg_buckets(df))
    parts.append(_section_pitchers(df))
    parts.append(_section_clv(df))

    return "\n".join(parts)


def _print_quick_summary(df: pd.DataFrame) -> None:
    s = _settled(df)
    print()
    print("=" * 65)
    print("  V4 PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Total logged: {len(df)}  |  Settled: {len(s)}  |  Open: {len(df)-len(s)}")

    if s.empty:
        print("  No settled bets with stake recorded.")
        return

    wr     = _win_rate(s)
    roi    = _roi(s)
    profit = s["profit"].sum()
    stake  = s["stake"].sum()
    clv_m, clv_n = _mean_clv(s)

    print(f"\n  Win rate:     {wr:.1%}")
    print(f"  ROI:          {roi:+.2%}")
    print(f"  Total staked: ${stake:,.0f}")
    print(f"  Total profit: ${profit:+,.0f}")
    if not np.isnan(clv_m):
        print(f"  Mean CLV:     {clv_m:+.2f}c ({clv_n} bets)")

    print()
    print(f"  {'Bucket':<12}  {'N':>4}  {'Win%':>6}  {'ROI':>8}  {'P&L':>9}")
    for bucket in ["V2_core", "V4_extra"]:
        grp = s[s["strategy_bucket"].astype(str) == bucket]
        if grp.empty:
            print(f"  {bucket:<12}  {'0':>4}  {'—':>6}  {'—':>8}  {'—':>9}")
            continue
        grp_roi    = _roi(grp)
        grp_wr     = _win_rate(grp)
        grp_profit = grp["profit"].sum()
        print(f"  {bucket:<12}  {len(grp):>4}  {grp_wr:.0%}  {grp_roi:>+8.2%}  ${grp_profit:>+8.0f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="V4 performance report")
    parser.add_argument("--since",  default=None,
                        help="Include only bets from this date onward (YYYY-MM-DD)")
    parser.add_argument("--bucket", default=None,
                        choices=["V2_core", "V4_extra"],
                        help="Filter to a single strategy bucket")
    parser.add_argument("--no-write", action="store_true",
                        help="Skip writing the markdown report file")
    args = parser.parse_args()

    df = _load_log(since=args.since, bucket_filter=args.bucket)
    _print_quick_summary(df)

    if not args.no_write:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        report = _build_report(df, args.since)
        REPORT_FILE.write_text(report, encoding="utf-8")
        print(f"[performance_report] Full report written to {REPORT_FILE}")
    else:
        print("[performance_report] --no-write: skipping file output")


if __name__ == "__main__":
    main()
