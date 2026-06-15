import argparse
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _max_drawdown(values: pd.Series) -> float:
    drawdown = values - values.cummax()
    return float(drawdown.min()) if len(drawdown) else 0.0


def _curve_points(values: pd.Series) -> str:
    if values.empty:
        return ""
    spread = max(float(values.max() - values.min()), 1.0)
    return " ".join(
        f"{i * 100 / max(len(values) - 1, 1):.2f},{100 - ((value - values.min()) / spread * 100):.2f}"
        for i, value in enumerate(values)
    )


def _bars(frame: pd.DataFrame, label_col: str, value_col: str) -> str:
    if frame.empty:
        return ""
    max_abs = max(float(frame[value_col].abs().max()), 1.0)
    rows = []
    for _, row in frame.iterrows():
        value = float(row[value_col])
        cls = "pos" if value >= 0 else "neg"
        width = abs(value) / max_abs * 100
        rows.append(
            f"<div class='bar-row'><span>{html.escape(str(row[label_col]))}</span>"
            f"<div class='bar-track'><div class='bar {cls}' style='width:{width:.1f}%'></div></div>"
            f"<b>${value:+,.0f}</b></div>"
        )
    return "\n".join(rows)


def _dashboard(picks: pd.DataFrame, summary: pd.DataFrame, output_csv: str) -> str:
    total = summary[summary["eval_year"] == "ALL"].iloc[0]
    yearly_rows = "\n".join(
        "<tr>"
        f"<td>{r.eval_year}</td><td>{int(r.bets)}</td><td>{int(r.wins)}-{int(r.losses)}</td>"
        f"<td>{r.win_rate:.1%}</td><td>${r.profit_100_flat:,.0f}</td><td>{r.roi:.1%}</td>"
        f"<td>${r.max_drawdown:,.0f}</td>"
        "</tr>"
        for _, r in summary[summary["eval_year"] != "ALL"].iterrows()
    )
    by_year = picks.groupby("eval_year", as_index=False).agg(profit=("flat_profit", "sum"))
    by_side = picks.groupby("best_side", as_index=False).agg(profit=("flat_profit", "sum"))
    top = picks.sort_values("flat_profit", ascending=False).head(12)
    worst = picks.sort_values("flat_profit").head(12)

    def rows(frame: pd.DataFrame) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{r.date}</td><td>{r.eval_year}</td><td>{html.escape(str(r.pitcher_name))}</td>"
            f"<td>{r.best_side}</td><td>{r.strikeouts_projection:.2f}</td><td>{r.line:g}</td>"
            f"<td>{r.bet_odds:+.0f}</td><td>{r.strikeouts:g}</td>"
            f"<td>{r.selector_win_probability:.2f}</td><td>{'Y' if r.won else 'N'}</td>"
            f"<td>${r.flat_profit:,.0f}</td>"
            "</tr>"
            for _, r in frame.iterrows()
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Walk-Forward Logistic Selector Picks</title>
  <style>
    body {{ margin:0; font-family: Segoe UI, Arial, sans-serif; color:#172033; background:#f6f8fb; }}
    header {{ padding:28px 36px; background:#101827; color:white; }}
    h1 {{ margin:0 0 6px; font-size:30px; }}
    h2 {{ margin-top:0; }}
    main {{ padding:24px 36px 42px; }}
    .sub {{ color:#cbd5e1; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:14px; margin-bottom:18px; }}
    .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }}
    .card {{ background:white; border:1px solid #e5e7eb; border-radius:8px; padding:16px; box-shadow:0 1px 2px #0001; }}
    .metric {{ font-size:25px; font-weight:750; margin-top:6px; }}
    .good {{ color:#047857; }} .bad {{ color:#b91c1c; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:8px 9px; border-bottom:1px solid #e5e7eb; text-align:left; }}
    th {{ background:#f9fafb; color:#4b5563; }}
    svg {{ width:100%; height:250px; background:#fbfdff; border:1px solid #e5e7eb; border-radius:8px; }}
    polyline {{ fill:none; stroke:#2563eb; stroke-width:2.5; }}
    .bar-row {{ display:grid; grid-template-columns:82px 1fr 90px; gap:8px; align-items:center; margin:10px 0; }}
    .bar-track {{ height:12px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
    .bar {{ height:100%; }} .pos {{ background:#10b981; }} .neg {{ background:#ef4444; }}
    .note {{ color:#64748b; font-size:13px; }}
  </style>
</head>
<body>
  <header>
    <h1>Walk-Forward Logistic Selector Picks</h1>
    <div class="sub">Top selector picks by year, $100 flat stake, no same-year selector training leakage</div>
  </header>
  <main>
    <div class="grid">
      <div class="card"><div>Total Picks</div><div class="metric">{int(total.bets)}</div></div>
      <div class="card"><div>Record</div><div class="metric">{int(total.wins)}-{int(total.losses)}</div></div>
      <div class="card"><div>Profit</div><div class="metric {'good' if total.profit_100_flat >= 0 else 'bad'}">${total.profit_100_flat:,.0f}</div></div>
      <div class="card"><div>ROI</div><div class="metric {'good' if total.roi >= 0 else 'bad'}">{total.roi:.1%}</div></div>
    </div>
    <div class="card">
      <h2>By Year</h2>
      <table><thead><tr><th>Year</th><th>Bets</th><th>Record</th><th>Win Rate</th><th>Profit</th><th>ROI</th><th>Max DD</th></tr></thead><tbody>{yearly_rows}</tbody></table>
    </div>
    <div class="two">
      <div class="card"><h2>Rolling Bankroll</h2><svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="{_curve_points(picks['rolling_bankroll'])}" /></svg></div>
      <div class="card"><h2>Profit Splits</h2><h3>By Year</h3>{_bars(by_year, 'eval_year', 'profit')}<h3>By Side</h3>{_bars(by_side, 'best_side', 'profit')}</div>
    </div>
    <div class="two">
      <div class="card"><h2>Best 12 Picks</h2><table><thead><tr><th>Date</th><th>Year</th><th>Pitcher</th><th>Side</th><th>Proj</th><th>Line</th><th>Odds</th><th>Actual</th><th>Selector</th><th>Won</th><th>Profit</th></tr></thead><tbody>{rows(top)}</tbody></table></div>
      <div class="card"><h2>Worst 12 Picks</h2><table><thead><tr><th>Date</th><th>Year</th><th>Pitcher</th><th>Side</th><th>Proj</th><th>Line</th><th>Odds</th><th>Actual</th><th>Selector</th><th>Won</th><th>Profit</th></tr></thead><tbody>{rows(worst)}</tbody></table></div>
    </div>
    <p class="note">Full picks exported to {html.escape(output_csv)}.</p>
  </main>
</body>
</html>"""


def _summarize(frame: pd.DataFrame, year: str) -> dict:
    if frame.empty:
        return {"eval_year": year, "bets": 0}
    cumulative = frame["flat_profit"].cumsum()
    return {
        "eval_year": year,
        "bets": int(len(frame)),
        "wins": int(frame["won"].sum()),
        "losses": int((~frame["won"] & ~frame["push"]).sum()),
        "win_rate": float(frame["won"].mean()),
        "profit_100_flat": float(frame["flat_profit"].sum()),
        "roi": float(frame["flat_profit"].sum() / (len(frame) * 100)),
        "max_drawdown": _max_drawdown(cumulative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export readable walk-forward selector picks.")
    parser.add_argument("--scored", default="data/processed/walk_forward_strikeout_selector_logistic_scored_tests.csv")
    parser.add_argument("--top-n-per-year", type=int, default=100)
    parser.add_argument("--start-bankroll", type=float, default=10000.0)
    parser.add_argument("--output-csv", default="data/processed/walk_forward_logistic_selector_picks.csv")
    parser.add_argument("--output-summary", default="data/processed/walk_forward_logistic_selector_picks_summary.csv")
    parser.add_argument("--output-html", default="data/exports/walk_forward_logistic_selector_picks.html")
    args = parser.parse_args()

    scored = pd.read_csv(args.scored)
    scored["game_date"] = pd.to_datetime(scored["game_date"])
    picks = (
        scored.sort_values(["eval_year", "selector_win_probability"], ascending=[True, False])
        .groupby("eval_year", group_keys=False)
        .head(args.top_n_per_year)
        .sort_values(["game_date", "pitcher_id"])
        .copy()
    )
    picks["date"] = picks["game_date"].dt.date.astype(str)
    picks["pitcher_name"] = picks.get("pitcher_name_x", picks.get("player_name", ""))
    picks["flat_stake"] = 100.0
    picks["flat_profit"] = picks["unit_profit"] * picks["flat_stake"]
    picks["rolling_bankroll"] = args.start_bankroll + picks["flat_profit"].cumsum()
    picks["projection_gap"] = (picks["strikeouts_projection"] - picks["line"]).abs()
    picks["prediction_side"] = picks["best_side"]

    cols = [
        "eval_year",
        "date",
        "pitcher_id",
        "pitcher_name",
        "best_side",
        "strikeouts_projection",
        "line",
        "projection_gap",
        "bet_odds",
        "strikeouts",
        "won",
        "edge_pct",
        "selector_win_probability",
        "flat_stake",
        "flat_profit",
        "rolling_bankroll",
        "opp_lineup_k_rate_prior",
        "expected_innings_pitched",
        "sc_swinging_strike_rate_roll5",
        "sc_csw_rate_roll5",
    ]
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    output_html = Path(args.output_html)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    picks[[c for c in cols if c in picks.columns]].to_csv(output_csv, index=False)
    summary_rows = [_summarize(group.sort_values("game_date"), year) for year, group in picks.groupby("eval_year")]
    summary_rows.append(_summarize(picks.sort_values("game_date"), "ALL"))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_summary, index=False)
    output_html.write_text(_dashboard(picks, summary, str(output_csv)), encoding="utf-8")

    print(f"Selector picks saved to {output_csv}")
    print(f"Selector summary saved to {output_summary}")
    print(f"Selector dashboard saved to {output_html}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
