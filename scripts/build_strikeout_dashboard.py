import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_threshold_bets(path: Path, threshold: int, start: str, end: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing bets file: {path}")

    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[
        (df["market"] == "strikeouts")
        & (df["game_date"] >= pd.to_datetime(start))
        & (df["game_date"] <= pd.to_datetime(end))
    ].copy()
    if df.empty:
        return df

    df = df.sort_values(["game_date", "pitcher_id", "edge_pct"], ascending=[True, True, False])
    df = df.drop_duplicates(["game_date", "pitcher_id"], keep="first")
    df["threshold"] = threshold
    df["date"] = df["game_date"].dt.date.astype(str)
    df["pitcher_name"] = df.get("pitcher_name_x", df.get("player_name", ""))
    df["projection"] = df["strikeouts_projection"]
    df["prediction_side"] = np.where(df["projection"] > df["line"], "over", "under")
    df["actual_side"] = np.where(df["actual_result"] > df["line"], "over", "under")
    df["prediction_correct"] = df["prediction_side"] == df["actual_side"]
    df["bet_correct"] = df["won"]
    return df


def _max_drawdown(cumulative_profit: pd.Series) -> float:
    running_max = cumulative_profit.cummax()
    drawdown = cumulative_profit - running_max
    return float(drawdown.min()) if len(drawdown) else 0.0


def _summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    ordered = df.sort_values("game_date").copy()
    ordered["cum_profit"] = ordered["flat_profit"].cumsum()
    ordered["scaled_cum_profit"] = ordered["scaled_profit"].cumsum()
    decided = ordered[~ordered["push"]]
    wins = int(ordered["won"].sum())
    pushes = int(ordered["push"].sum())
    losses = int((~ordered["won"] & ~ordered["push"]).sum())
    total_staked = float(ordered["stake"].sum())
    profit = float(ordered["flat_profit"].sum())
    scaled_staked = float(ordered["scaled_stake"].sum())
    scaled_profit = float(ordered["scaled_profit"].sum())
    decimal = np.where(
        ordered["bet_odds"] > 0,
        1 + ordered["bet_odds"] / 100,
        1 + 100 / ordered["bet_odds"].abs(),
    )

    return {
        "bets": int(len(ordered)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(decided["won"].mean()) if len(decided) else np.nan,
        "prediction_accuracy": float(ordered["prediction_correct"].mean()),
        "profit": profit,
        "roi": profit / total_staked if total_staked else 0.0,
        "total_staked": total_staked,
        "scaled_staked": scaled_staked,
        "scaled_profit": scaled_profit,
        "scaled_roi": scaled_profit / scaled_staked if scaled_staked else 0.0,
        "avg_edge": float(ordered["edge_pct"].mean()),
        "avg_scaled_stake": float(ordered["scaled_stake"].mean()),
        "avg_odds": float(ordered["bet_odds"].mean()),
        "break_even": float((1 / decimal).mean()),
        "max_drawdown": _max_drawdown(ordered["cum_profit"]),
        "scaled_max_drawdown": _max_drawdown(ordered["scaled_cum_profit"]),
    }


def _daily_curve(df: pd.DataFrame, profit_col: str) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)
        .agg(bets=(profit_col, "size"), profit=(profit_col, "sum"))
        .sort_values("date")
    )
    daily["cum_profit"] = daily["profit"].cumsum()
    return daily


def _curve_points(daily: pd.DataFrame) -> str:
    if daily.empty:
        return ""
    min_profit = daily["cum_profit"].min()
    max_profit = daily["cum_profit"].max()
    spread = max(max_profit - min_profit, 1)
    return " ".join(
        f"{i * 100 / max(len(daily) - 1, 1):.2f},{100 - ((row.cum_profit - min_profit) / spread * 100):.2f}"
        for i, row in daily.reset_index(drop=True).iterrows()
    )


def _bars(values: pd.DataFrame, label_col: str, value_col: str) -> str:
    if values.empty:
        return ""
    max_abs = max(float(values[value_col].abs().max()), 1.0)
    parts = []
    for _, row in values.iterrows():
        value = float(row[value_col])
        width = abs(value) / max_abs * 100
        cls = "pos" if value >= 0 else "neg"
        parts.append(
            f"<div class='bar-row'><span>{html.escape(str(row[label_col]))}</span>"
            f"<div class='bar-track'><div class='bar {cls}' style='width:{width:.1f}%'></div></div>"
            f"<b>${value:+,.0f}</b></div>"
        )
    return "\n".join(parts)


def _html_dashboard(thresholds: pd.DataFrame, bets: pd.DataFrame, output_csv: str) -> str:
    best = thresholds.sort_values("roi", ascending=False).iloc[0]
    chosen = bets[bets["threshold"] == int(best["threshold"])].sort_values("game_date").copy()
    flat_daily = _daily_curve(chosen, "flat_profit")
    scaled_daily = _daily_curve(chosen, "scaled_profit")
    side = chosen.groupby("best_side", as_index=False).agg(
        bets=("flat_profit", "size"),
        profit=("flat_profit", "sum"),
    )
    monthly = (
        chosen.assign(month=chosen["game_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(bets=("flat_profit", "size"), profit=("flat_profit", "sum"))
    )
    top = chosen.sort_values("flat_profit", ascending=False).head(10)
    worst = chosen.sort_values("flat_profit", ascending=True).head(10)

    threshold_rows = "\n".join(
        "<tr>"
        f"<td>{int(r.threshold)}%</td><td>{int(r.bets)}</td><td>{r.win_rate:.1%}</td>"
        f"<td>{r.prediction_accuracy:.1%}</td><td>${r.profit:,.0f}</td><td>{r.roi:.1%}</td>"
        f"<td>${r.scaled_profit:,.0f}</td><td>{r.scaled_roi:.1%}</td>"
        f"<td>${r.max_drawdown:,.0f}</td><td>{r.avg_edge:.1f}%</td>"
        "</tr>"
        for _, r in thresholds.iterrows()
    )

    def bet_rows(frame: pd.DataFrame) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{r.date}</td><td>{html.escape(str(r.pitcher_name))}</td><td>{r.best_side}</td>"
            f"<td>{r.projection:.2f}</td><td>{r.line:g}</td><td>{r.bet_odds:+.0f}</td>"
            f"<td>{r.actual_result:g}</td><td>{'Y' if r.prediction_correct else 'N'}</td>"
            f"<td>{r.edge_pct:.1f}%</td><td>${r.flat_profit:,.0f}</td>"
            "</tr>"
            for _, r in frame.iterrows()
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>2025 Strikeout Props Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f6f8fb; color: #172033; }}
    header {{ background: #111827; color: white; padding: 28px 36px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    h2 {{ margin-top: 0; }}
    .sub {{ color: #cbd5e1; }}
    main {{ padding: 24px 36px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin-bottom: 22px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px #0001; }}
    .metric {{ font-size: 26px; font-weight: 750; margin-top: 6px; }}
    .good {{ color: #047857; }}
    .bad {{ color: #b91c1c; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
    th {{ background: #f9fafb; color: #4b5563; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }}
    svg {{ width: 100%; height: 240px; background: #fbfdff; border: 1px solid #e5e7eb; border-radius: 8px; }}
    polyline {{ fill: none; stroke: #2563eb; stroke-width: 2.5; }}
    .scaled polyline {{ stroke: #059669; }}
    .bar-row {{ display: grid; grid-template-columns: 82px 1fr 90px; align-items: center; gap: 8px; margin: 10px 0; }}
    .bar-track {{ height: 12px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 100%; }}
    .pos {{ background: #10b981; }}
    .neg {{ background: #ef4444; }}
    .note {{ color: #64748b; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>2025 Strikeout Props Betting Dashboard</h1>
    <div class="sub">Strikeouts only, 5%+ edge thresholds, vs DraftKings historical pitcher strikeout lines</div>
  </header>
  <main>
    <div class="grid">
      <div class="card"><div>Best Threshold</div><div class="metric">{int(best.threshold)}% edge</div></div>
      <div class="card"><div>Flat $100 Profit</div><div class="metric {'good' if best.profit >= 0 else 'bad'}">${best.profit:,.0f}</div></div>
      <div class="card"><div>Scaled Profit</div><div class="metric {'good' if best.scaled_profit >= 0 else 'bad'}">${best.scaled_profit:,.0f}</div></div>
      <div class="card"><div>Record</div><div class="metric">{int(best.wins)}-{int(best.losses)}</div></div>
    </div>

    <div class="card">
      <h2>Threshold Comparison</h2>
      <table><thead><tr><th>Min Edge</th><th>Bets</th><th>Win Rate</th><th>Prediction Accuracy</th><th>Flat Profit</th><th>Flat ROI</th><th>Scaled Profit</th><th>Scaled ROI</th><th>Flat Max DD</th><th>Avg Edge</th></tr></thead>
      <tbody>{threshold_rows}</tbody></table>
    </div>

    <div class="two">
      <div class="card">
        <h2>Flat $100 Equity Curve: {int(best.threshold)}% Edge</h2>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="{_curve_points(flat_daily)}" /></svg>
        <p class="note">Flat stake is always $100 per qualifying bet.</p>
      </div>
      <div class="card scaled">
        <h2>Scaled Stake Equity Curve</h2>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="{_curve_points(scaled_daily)}" /></svg>
        <p class="note">Scaled stake uses capped Kelly fraction against a $10,000 bankroll.</p>
      </div>
    </div>

    <div class="two">
      <div class="card"><h2>Flat Profit By Side</h2>{_bars(side, "best_side", "profit")}</div>
      <div class="card"><h2>Flat Profit By Month</h2>{_bars(monthly, "month", "profit")}</div>
    </div>

    <div class="two">
      <div class="card"><h2>Best 10 Bets</h2><table><thead><tr><th>Date</th><th>Pitcher</th><th>Side</th><th>Proj</th><th>Line</th><th>Odds</th><th>Actual</th><th>Correct</th><th>Edge</th><th>Profit</th></tr></thead><tbody>{bet_rows(top)}</tbody></table></div>
      <div class="card"><h2>Worst 10 Bets</h2><table><thead><tr><th>Date</th><th>Pitcher</th><th>Side</th><th>Proj</th><th>Line</th><th>Odds</th><th>Actual</th><th>Correct</th><th>Edge</th><th>Profit</th></tr></thead><tbody>{bet_rows(worst)}</tbody></table></div>
    </div>

    <p class="note">Detailed plays exported to {html.escape(output_csv)}. This measures betting performance against the historical line snapshot, not CLV.</p>
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2025 strikeout betting dashboard.")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--thresholds", default="5,8,10")
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--bankroll", type=float, default=10000.0)
    parser.add_argument("--output-html", default="data/exports/strikeout_betting_dashboard_2025.html")
    parser.add_argument("--output-bets", default="data/processed/strikeout_full_plays_2025.csv")
    parser.add_argument("--output-summary", default="data/processed/strikeout_flat_summary_2025.csv")
    args = parser.parse_args()

    all_bets = []
    summaries = []
    for threshold in [int(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        path = Path(f"data/processed/historical_odds_edge{threshold}_bets.csv")
        bets = _load_threshold_bets(path, threshold, args.start, args.end)
        if bets.empty:
            continue
        bets["stake"] = float(args.stake)
        bets["flat_profit"] = bets["unit_profit"] * bets["stake"]
        bets["scaled_stake"] = float(args.bankroll) * bets["kelly_fraction"].fillna(0)
        bets["scaled_profit"] = bets["unit_profit"] * bets["scaled_stake"]
        all_bets.append(bets)
        summary = _summarize(bets)
        summary["threshold"] = threshold
        summaries.append(summary)

    if not all_bets:
        raise ValueError("No strikeout bets found. Run the historical odds backtests first.")

    bets_out = pd.concat(all_bets, ignore_index=True, sort=False)
    bets_out = bets_out.sort_values(["threshold", "game_date", "pitcher_id"]).copy()
    bets_out["rolling_bankroll"] = (
        float(args.bankroll) + bets_out.groupby("threshold")["flat_profit"].cumsum()
    )
    bets_out["rolling_scaled_bankroll"] = (
        float(args.bankroll) + bets_out.groupby("threshold")["scaled_profit"].cumsum()
    )
    summary_out = pd.DataFrame(summaries).sort_values("threshold")

    output_bets = Path(args.output_bets)
    output_summary = Path(args.output_summary)
    output_html = Path(args.output_html)
    output_bets.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    play_cols = [
        "threshold",
        "date",
        "pitcher_id",
        "pitcher_name",
        "best_side",
        "prediction_side",
        "projection",
        "line",
        "bet_odds",
        "actual_result",
        "actual_side",
        "prediction_correct",
        "bet_correct",
        "edge_pct",
        "over_probability",
        "under_probability",
        "fair_over_odds",
        "fair_under_odds",
        "stake",
        "flat_profit",
        "scaled_stake",
        "scaled_profit",
        "rolling_bankroll",
        "rolling_scaled_bankroll",
    ]
    export_cols = [col for col in play_cols if col in bets_out.columns]
    bets_out[export_cols].to_csv(output_bets, index=False)
    summary_out.to_csv(output_summary, index=False)
    output_html.write_text(_html_dashboard(summary_out, bets_out, str(output_bets)), encoding="utf-8")

    print(f"Strikeout plays saved to {output_bets}")
    print(f"Strikeout summary saved to {output_summary}")
    print(f"Dashboard saved to {output_html}")
    print(
        summary_out[
            [
                "threshold",
                "bets",
                "wins",
                "losses",
                "profit",
                "roi",
                "scaled_profit",
                "scaled_roi",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
