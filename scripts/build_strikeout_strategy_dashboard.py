import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _american_decimal(odds: pd.Series) -> pd.Series:
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / odds.abs())


def _max_drawdown(series: pd.Series) -> float:
    dd = series - series.cummax()
    return float(dd.min()) if len(dd) else 0.0


def _bankroll_path(df: pd.DataFrame, start_bankroll: float, flat_stake: float, qk_fraction: float) -> pd.DataFrame:
    out = df.sort_values(["game_date", "pitcher_id"]).copy()
    out["flat_stake"] = float(flat_stake)
    out["flat_profit"] = out["unit_profit"] * out["flat_stake"]
    out["rolling_flat_bankroll"] = start_bankroll + out["flat_profit"].cumsum()

    qk_bankroll = float(start_bankroll)
    qk_stakes = []
    qk_profits = []
    qk_bankrolls = []
    for _, row in out.iterrows():
        stake = qk_bankroll * float(row.get("kelly_fraction", 0)) * qk_fraction
        stake = max(min(stake, qk_bankroll), 0)
        profit = stake * float(row["unit_profit"])
        qk_bankroll += profit
        qk_stakes.append(stake)
        qk_profits.append(profit)
        qk_bankrolls.append(qk_bankroll)

    out["qk_stake"] = qk_stakes
    out["qk_profit"] = qk_profits
    out["rolling_qk_bankroll"] = qk_bankrolls
    return out


def _summary(df: pd.DataFrame, start_bankroll: float) -> dict:
    flat_profit = float(df["flat_profit"].sum())
    qk_profit = float(df["qk_profit"].sum())
    return {
        "bets": int(len(df)),
        "wins": int(df["won"].sum()),
        "losses": int((~df["won"] & ~df["push"]).sum()),
        "pushes": int(df["push"].sum()),
        "win_rate": float(df["won"].mean()),
        "flat_profit": flat_profit,
        "flat_roi": flat_profit / float(df["flat_stake"].sum()),
        "flat_ending_bankroll": start_bankroll + flat_profit,
        "flat_max_drawdown": _max_drawdown(df["rolling_flat_bankroll"]),
        "qk_profit": qk_profit,
        "qk_roi": qk_profit / float(df["qk_stake"].sum()) if df["qk_stake"].sum() else 0.0,
        "qk_ending_bankroll": float(df["rolling_qk_bankroll"].iloc[-1]),
        "qk_max_drawdown": _max_drawdown(df["rolling_qk_bankroll"]),
        "avg_edge_pct": float(df["edge_pct"].mean()),
        "avg_projection_gap": float(df["projection_gap"].mean()),
        "avg_odds": float(df["bet_odds"].mean()),
    }


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


def _dashboard(df: pd.DataFrame, summary: dict, output_csv: str, qk_fraction: float, title: str) -> str:
    by_side = df.groupby("best_side", as_index=False).agg(
        bets=("flat_profit", "size"), profit=("flat_profit", "sum")
    )
    by_month = (
        df.assign(month=df["game_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(bets=("flat_profit", "size"), profit=("flat_profit", "sum"))
    )
    top = df.sort_values("flat_profit", ascending=False).head(12)
    worst = df.sort_values("flat_profit").head(12)

    def rows(frame: pd.DataFrame) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{r.date}</td><td>{html.escape(str(r.pitcher_name))}</td><td>{r.best_side}</td>"
            f"<td>{r.projection:.2f}</td><td>{r.line:g}</td><td>{r.bet_odds:+.0f}</td>"
            f"<td>{r.actual_result:g}</td><td>{r.projection_gap:.2f}</td>"
            f"<td>{'Y' if r.won else 'N'}</td><td>${r.flat_profit:,.0f}</td>"
            "</tr>"
            for _, r in frame.iterrows()
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
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
    .qk polyline {{ stroke:#059669; }}
    .bar-row {{ display:grid; grid-template-columns:82px 1fr 90px; gap:8px; align-items:center; margin:10px 0; }}
    .bar-track {{ height:12px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
    .bar {{ height:100%; }} .pos {{ background:#10b981; }} .neg {{ background:#ef4444; }}
    .note {{ color:#64748b; font-size:13px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="sub">Criteria: strikeouts only, edge >= 5%, projection gap >= 0.50 Ks</div>
  </header>
  <main>
    <div class="grid">
      <div class="card"><div>Bets</div><div class="metric">{summary['bets']}</div></div>
      <div class="card"><div>Record</div><div class="metric">{summary['wins']}-{summary['losses']}</div></div>
      <div class="card"><div>Flat $100 Profit</div><div class="metric {'good' if summary['flat_profit'] >= 0 else 'bad'}">${summary['flat_profit']:,.0f}</div></div>
      <div class="card"><div>Quarter Kelly Profit</div><div class="metric {'good' if summary['qk_profit'] >= 0 else 'bad'}">${summary['qk_profit']:,.0f}</div></div>
    </div>
    <div class="card">
      <h2>Results</h2>
      <table><thead><tr><th>Win Rate</th><th>Flat ROI</th><th>Flat Ending Bankroll</th><th>Flat Max DD</th><th>QK ROI</th><th>QK Ending Bankroll</th><th>QK Max DD</th><th>Avg Edge</th><th>Avg Gap</th></tr></thead>
      <tbody><tr><td>{summary['win_rate']:.1%}</td><td>{summary['flat_roi']:.1%}</td><td>${summary['flat_ending_bankroll']:,.0f}</td><td>${summary['flat_max_drawdown']:,.0f}</td><td>{summary['qk_roi']:.1%}</td><td>${summary['qk_ending_bankroll']:,.0f}</td><td>${summary['qk_max_drawdown']:,.0f}</td><td>{summary['avg_edge_pct']:.1f}%</td><td>{summary['avg_projection_gap']:.2f}</td></tr></tbody></table>
    </div>
    <div class="two">
      <div class="card"><h2>Flat Bankroll</h2><svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="{_curve_points(df['rolling_flat_bankroll'])}" /></svg></div>
      <div class="card qk"><h2>Quarter Kelly Bankroll</h2><svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="{_curve_points(df['rolling_qk_bankroll'])}" /></svg><p class="note">QK stake = current bankroll x capped Kelly fraction x {qk_fraction:.2f}</p></div>
    </div>
    <div class="two">
      <div class="card"><h2>Flat Profit By Side</h2>{_bars(by_side, 'best_side', 'profit')}</div>
      <div class="card"><h2>Flat Profit By Month</h2>{_bars(by_month, 'month', 'profit')}</div>
    </div>
    <div class="two">
      <div class="card"><h2>Best 12 Bets</h2><table><thead><tr><th>Date</th><th>Pitcher</th><th>Side</th><th>Proj</th><th>Line</th><th>Odds</th><th>Actual</th><th>Gap</th><th>Won</th><th>Profit</th></tr></thead><tbody>{rows(top)}</tbody></table></div>
      <div class="card"><h2>Worst 12 Bets</h2><table><thead><tr><th>Date</th><th>Pitcher</th><th>Side</th><th>Proj</th><th>Line</th><th>Odds</th><th>Actual</th><th>Gap</th><th>Won</th><th>Profit</th></tr></thead><tbody>{rows(worst)}</tbody></table></div>
    </div>
    <p class="note">Full plays exported to {html.escape(output_csv)}.</p>
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard for the best strikeout betting filter.")
    parser.add_argument("--bets", default="data/processed/historical_odds_edge5_bets.csv")
    parser.add_argument("--start-bankroll", type=float, default=10000.0)
    parser.add_argument("--flat-stake", type=float, default=100.0)
    parser.add_argument("--qk-fraction", type=float, default=0.25)
    parser.add_argument("--min-edge", type=float, default=5.0)
    parser.add_argument("--min-projection-gap", type=float, default=0.50)
    parser.add_argument("--output-csv", default="data/processed/strikeout_strategy_gap50_plays_2025.csv")
    parser.add_argument("--output-html", default="data/exports/strikeout_strategy_gap50_dashboard_2025.html")
    parser.add_argument("--title", default="2025 Strikeout Strategy Dashboard")
    args = parser.parse_args()

    df = pd.read_csv(args.bets)
    df = df[df["market"] == "strikeouts"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "pitcher_id", "edge_pct"], ascending=[True, True, False])
    df = df.drop_duplicates(["game_date", "pitcher_id"], keep="first")
    df["projection"] = df["strikeouts_projection"]
    df["projection_gap"] = (df["projection"] - df["line"]).abs()
    df = df[(df["edge_pct"] >= args.min_edge) & (df["projection_gap"] >= args.min_projection_gap)].copy()
    if df.empty:
        raise ValueError("No bets matched the requested strategy filters.")

    df["date"] = df["game_date"].dt.date.astype(str)
    df["pitcher_name"] = df.get("pitcher_name_x", df.get("player_name", ""))
    df["prediction_side"] = np.where(df["projection"] > df["line"], "over", "under")
    df["actual_side"] = np.where(df["actual_result"] > df["line"], "over", "under")
    df["prediction_correct"] = df["prediction_side"] == df["actual_side"]
    df["decimal_odds_calc"] = _american_decimal(df["bet_odds"])
    df = _bankroll_path(df, args.start_bankroll, args.flat_stake, args.qk_fraction)
    summary = _summary(df, args.start_bankroll)

    cols = [
        "date",
        "pitcher_id",
        "pitcher_name",
        "best_side",
        "prediction_side",
        "projection",
        "line",
        "projection_gap",
        "bet_odds",
        "actual_result",
        "actual_side",
        "prediction_correct",
        "won",
        "edge_pct",
        "over_probability",
        "under_probability",
        "fair_over_odds",
        "fair_under_odds",
        "flat_stake",
        "flat_profit",
        "rolling_flat_bankroll",
        "kelly_fraction",
        "qk_stake",
        "qk_profit",
        "rolling_qk_bankroll",
    ]

    output_csv = Path(args.output_csv)
    output_html = Path(args.output_html)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in cols if c in df.columns]].to_csv(output_csv, index=False)
    output_html.write_text(
        _dashboard(df, summary, str(output_csv), args.qk_fraction, args.title),
        encoding="utf-8",
    )

    print(f"Strategy plays saved to {output_csv}")
    print(f"Strategy dashboard saved to {output_html}")
    print(
        pd.DataFrame([summary])[
            [
                "bets",
                "wins",
                "losses",
                "win_rate",
                "flat_profit",
                "flat_roi",
                "qk_profit",
                "qk_roi",
                "flat_max_drawdown",
                "qk_max_drawdown",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
