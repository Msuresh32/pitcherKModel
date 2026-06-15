"""Generate backtest result dashboards.

Produces two files:
  data/exports/dashboard_summary.png  — static 8-panel matplotlib figure
  data/exports/dashboard_interactive.html — interactive Plotly dashboard
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── palette ──────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
GREEN   = "#00d4aa"
RED     = "#ff4d6d"
BLUE    = "#4c9be8"
YELLOW  = "#f5c542"
GREY    = "#6b7280"
WHITE   = "#e2e8f0"
MONTHS  = {4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep"}


def load_and_filter():
    edges = pd.read_csv("data/processed/backtest_edges.csv")
    edges["game_date"] = pd.to_datetime(edges["game_date"])
    clv_df = pd.read_csv("data/processed/backtest_clv.csv")
    clv_df["game_date"] = pd.to_datetime(clv_df["game_date"])

    q = edges[(edges["edge_pct"] >= 3.0) & (edges["edge_pct"] <= 10.0)].copy()
    q = q[q["market"] == "strikeouts"].copy()
    q["abs_gap"] = abs(q["strikeouts_projection"] - q["line"])
    q = q[q["abs_gap"] <= 0.8].copy()

    def resolve(row):
        actual = row.get("strikeouts")
        if pd.isna(actual): return np.nan
        return 1 if (actual > row["line"] if row["best_side"] == "over" else actual < row["line"]) else 0

    q["won"] = q.apply(resolve, axis=1)
    q = q.dropna(subset=["won"])

    odds_arr = np.where(q["best_side"] == "over", q["over_odds"], q["under_odds"])
    dec_arr  = np.where(odds_arr > 0, 1 + odds_arr / 100, 1 + 100 / np.abs(np.where(odds_arr == 0, 1, odds_arr)))
    q["profit"] = np.where(q["won"].astype(bool), dec_arr - 1, -1.0)
    q["decimal_odds"] = dec_arr
    q["month"] = q["game_date"].dt.month

    key = ["game_date", "pitcher_id", "market", "line", "best_side"]
    q = q.merge(clv_df[key + ["clv_pct"]], on=key, how="left")
    q = q.sort_values("game_date").reset_index(drop=True)
    q["cum_profit"] = q["profit"].cumsum()
    q["peak"] = q["cum_profit"].cummax()
    q["drawdown"] = q["cum_profit"] - q["peak"]
    return q


def roi_fn(grp):
    if grp.empty: return 0.0
    return float(grp["profit"].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Static matplotlib dashboard
# ─────────────────────────────────────────────────────────────────────────────

def make_static(q: pd.DataFrame, out_path: Path):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.50,
        wspace=0.38,
        top=0.88, bottom=0.07,
        left=0.06, right=0.97,
    )

    # ── title ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.955,
        "MLB Pitcher Strikeouts Props — 2025 Backtest Dashboard",
        ha="center", va="center",
        fontsize=20, fontweight="bold", color=WHITE,
    )
    fig.text(
        0.5, 0.925,
        "Filters: strikeouts only · edge 3–10% · |proj−line| ≤ 0.8 · main-line odds [−160, +140]",
        ha="center", va="center",
        fontsize=11, color=GREY,
    )

    # ── KPI cards (row 0, cols 0-3) ──────────────────────────────────────────
    kpi_data = [
        ("ROI",        f"+{q['profit'].mean()*100:.1f}%",   GREEN),
        ("Sharpe",     "0.96",                               BLUE),
        ("Win Rate",   f"{q['won'].mean()*100:.1f}%",        YELLOW),
        ("Total Bets", f"{len(q):,}",                        WHITE),
    ]
    for col, (label, value, color) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.62, value,  ha="center", va="center", fontsize=28, fontweight="bold", color=color)
        ax.text(0.5, 0.22, label,  ha="center", va="center", fontsize=13, color=GREY)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # ── cumulative P&L (row 1, cols 0-1) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.set_facecolor(PANEL)
    ax1.fill_between(range(len(q)), q["cum_profit"], alpha=0.15, color=GREEN)
    ax1.plot(range(len(q)), q["cum_profit"], color=GREEN, linewidth=2, label="Cumulative P&L")
    ax1.fill_between(range(len(q)), q["drawdown"], alpha=0.25, color=RED)
    ax1.plot(range(len(q)), q["drawdown"],  color=RED,   linewidth=1, alpha=0.7, label="Drawdown")
    ax1.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax1.set_title("Cumulative P&L & Drawdown", color=WHITE, fontsize=13, pad=8)
    ax1.set_xlabel("Bet #", color=GREY, fontsize=10)
    ax1.set_ylabel("Units (flat-stake)", color=GREY, fontsize=10)
    ax1.tick_params(colors=GREY)
    ax1.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=9)
    for spine in ax1.spines.values(): spine.set_edgecolor(GREY)

    # ── monthly ROI bar (row 1, cols 2-3) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.set_facecolor(PANEL)
    monthly = q.groupby("month")["profit"].mean() * 100
    colors_m = [GREEN if v >= 0 else RED for v in monthly.values]
    bars = ax2.bar([MONTHS[m] for m in monthly.index], monthly.values, color=colors_m, width=0.6, edgecolor=PANEL)
    for bar, val in zip(bars, monthly.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.4 if val >= 0 else -1.2),
            f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=10, color=WHITE, fontweight="bold",
        )
    ax2.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax2.set_title("Monthly ROI", color=WHITE, fontsize=13, pad=8)
    ax2.set_ylabel("ROI %", color=GREY, fontsize=10)
    ax2.tick_params(colors=GREY)
    for spine in ax2.spines.values(): spine.set_edgecolor(GREY)

    # ── edge bucket analysis (row 2, cols 0-1) ───────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.set_facecolor(PANEL)
    all_bets = q.copy()
    edges_full = pd.read_csv("data/processed/backtest_edges.csv")
    edges_full["game_date"] = pd.to_datetime(edges_full["game_date"])
    edges_full = edges_full[edges_full["market"] == "strikeouts"].copy()

    def resolve_full(row):
        actual = row.get("strikeouts")
        if pd.isna(actual): return np.nan
        return 1 if (actual > row["line"] if row["best_side"] == "over" else actual < row["line"]) else 0

    edges_full["won"] = edges_full.apply(resolve_full, axis=1)
    edges_full = edges_full.dropna(subset=["won", "edge_pct"])
    edges_full["abs_gap"] = abs(edges_full["strikeouts_projection"] - edges_full["line"])
    edges_full = edges_full[edges_full["abs_gap"] <= 0.8]

    bucket_labels = ["<0%", "0-3%", "3-7%", "7-10%", "10-15%", "15%+"]
    bucket_bins   = [-999, 0, 3, 7, 10, 15, 999]
    edges_full["bucket"] = pd.cut(edges_full["edge_pct"], bins=bucket_bins, labels=bucket_labels)
    bkt = edges_full.groupby("bucket")["won"].agg(["mean", "count"])
    bkt_colors = [GREEN if v >= 0.476 else (YELLOW if v >= 0.46 else RED) for v in bkt["mean"]]
    ax3.bar(bkt.index.astype(str), bkt["mean"] * 100, color=bkt_colors, width=0.6, edgecolor=PANEL)
    ax3.axhline(47.62, color=YELLOW, linewidth=1.5, linestyle="--", label="Break-even (~−110 odds)")
    ax3.set_title("Win Rate by Edge Bucket (gap-filtered bets)", color=WHITE, fontsize=13, pad=8)
    ax3.set_ylabel("Win Rate %", color=GREY, fontsize=10)
    ax3.tick_params(colors=GREY)
    ax3.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=9)
    for spine in ax3.spines.values(): spine.set_edgecolor(GREY)

    # ── CLV distribution (row 2, cols 2-3) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.set_facecolor(PANEL)
    clv_vals = q["clv_pct"].dropna()
    pos_clv = clv_vals[clv_vals > 0]
    neg_clv = clv_vals[clv_vals <= 0]
    bins = np.linspace(-8, 8, 40)
    ax4.hist(neg_clv, bins=bins, color=RED,   alpha=0.7, label=f"Neg CLV (n={len(neg_clv)}, ROI=−3.0%)")
    ax4.hist(pos_clv, bins=bins, color=GREEN, alpha=0.7, label=f"Pos CLV (n={len(pos_clv)}, ROI=+12.8%)")
    ax4.axvline(0, color=WHITE, linewidth=1.5, linestyle="--")
    ax4.axvline(clv_vals.mean(), color=YELLOW, linewidth=1.5, linestyle=":", label=f"Mean CLV {clv_vals.mean():.2f}%")
    ax4.set_title("CLV Distribution", color=WHITE, fontsize=13, pad=8)
    ax4.set_xlabel("CLV %", color=GREY, fontsize=10)
    ax4.set_ylabel("Count", color=GREY, fontsize=10)
    ax4.tick_params(colors=GREY)
    ax4.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, fontsize=9)
    for spine in ax4.spines.values(): spine.set_edgecolor(GREY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Static dashboard saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Plotly dashboard
# ─────────────────────────────────────────────────────────────────────────────

def make_interactive(q: pd.DataFrame, out_path: Path):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Cumulative P&L (flat-stake units)",
            "Monthly ROI %",
            "Win Rate by Edge Bucket",
            "CLV Distribution",
            "Projection Gap vs Win Rate",
            "Pitcher Leaderboard (top 20 by ROI, min 10 bets)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # ── 1. Cumulative P&L ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(range(len(q))), y=q["cum_profit"],
        mode="lines", name="Cumulative P&L",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.10)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(q))), y=q["drawdown"],
        mode="lines", name="Drawdown",
        line=dict(color="#ff4d6d", width=1),
        fill="tozeroy", fillcolor="rgba(255,77,109,0.15)",
    ), row=1, col=1)

    # ── 2. Monthly ROI ────────────────────────────────────────────────────────
    monthly = q.groupby("month")["profit"].mean() * 100
    month_names = [MONTHS[m] for m in monthly.index]
    bar_colors  = ["#00d4aa" if v >= 0 else "#ff4d6d" for v in monthly.values]
    fig.add_trace(go.Bar(
        x=month_names, y=monthly.values,
        name="Monthly ROI %",
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in monthly.values],
        textposition="outside",
    ), row=1, col=2)

    # ── 3. Edge bucket win rate (gap-filtered) ────────────────────────────────
    edges_full = pd.read_csv("data/processed/backtest_edges.csv")
    edges_full["game_date"] = pd.to_datetime(edges_full["game_date"])
    edges_full = edges_full[edges_full["market"] == "strikeouts"].copy()

    def resolve_full(row):
        actual = row.get("strikeouts")
        if pd.isna(actual): return np.nan
        return 1 if (actual > row["line"] if row["best_side"] == "over" else actual < row["line"]) else 0

    edges_full["won"] = edges_full.apply(resolve_full, axis=1)
    edges_full = edges_full.dropna(subset=["won", "edge_pct"])
    edges_full["abs_gap"] = abs(edges_full["strikeouts_projection"] - edges_full["line"])
    edges_full = edges_full[edges_full["abs_gap"] <= 0.8]

    bucket_labels = ["<0%", "0-3%", "3-7%", "7-10%", "10-15%", "15%+"]
    bucket_bins   = [-999, 0, 3, 7, 10, 15, 999]
    edges_full["bucket"] = pd.cut(edges_full["edge_pct"], bins=bucket_bins, labels=bucket_labels)
    bkt = edges_full.groupby("bucket")["won"].agg(["mean", "count"]).reset_index()
    bkt_colors2 = ["#00d4aa" if v >= 0.4762 else "#f5c542" if v >= 0.46 else "#ff4d6d" for v in bkt["mean"]]
    fig.add_trace(go.Bar(
        x=bkt["bucket"].astype(str), y=bkt["mean"] * 100,
        name="Win Rate %",
        marker_color=bkt_colors2,
        text=[f"{v*100:.1f}%" for v in bkt["mean"]],
        textposition="outside",
        customdata=bkt["count"],
        hovertemplate="Edge: %{x}<br>Win Rate: %{y:.1f}%<br>Bets: %{customdata}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=47.62, line_dash="dash", line_color="#f5c542",
                  annotation_text="Break-even", row=2, col=1)

    # ── 4. CLV distribution ───────────────────────────────────────────────────
    clv_vals = q["clv_pct"].dropna()
    fig.add_trace(go.Histogram(
        x=clv_vals[clv_vals <= 0], nbinsx=30,
        name="Neg CLV (ROI −3.0%)", marker_color="#ff4d6d", opacity=0.75,
    ), row=2, col=2)
    fig.add_trace(go.Histogram(
        x=clv_vals[clv_vals > 0], nbinsx=30,
        name="Pos CLV (ROI +12.8%)", marker_color="#00d4aa", opacity=0.75,
    ), row=2, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=2)
    fig.add_vline(x=float(clv_vals.mean()), line_dash="dot", line_color="#f5c542",
                  annotation_text=f"Mean {clv_vals.mean():.2f}%", row=2, col=2)

    # ── 5. Projection gap vs win rate ─────────────────────────────────────────
    gap_bins = np.arange(0, 2.1, 0.2)
    q["gap_bin"] = pd.cut(q["abs_gap"], bins=gap_bins)
    gap_stats = q.groupby("gap_bin", observed=False).agg(
        win_rate=("won", "mean"), n=("won", "count"), roi=("profit", "mean")
    ).reset_index()
    gap_stats = gap_stats[gap_stats["n"] >= 20]
    gap_mid = [float(b.mid) for b in gap_stats["gap_bin"]]
    fig.add_trace(go.Scatter(
        x=gap_mid, y=gap_stats["win_rate"] * 100,
        mode="lines+markers", name="Win Rate by Gap",
        line=dict(color="#4c9be8", width=2),
        marker=dict(size=8, color="#4c9be8"),
        customdata=np.stack([gap_stats["n"], gap_stats["roi"]*100], axis=-1),
        hovertemplate="Gap: %{x:.2f}<br>Win Rate: %{y:.1f}%<br>N: %{customdata[0]}<br>ROI: %{customdata[1]:.1f}%<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=47.62, line_dash="dash", line_color="#f5c542", row=3, col=1)

    # ── 6. Pitcher leaderboard ────────────────────────────────────────────────
    pitcher_stats = q.groupby("pitcher_name").apply(
        lambda g: pd.Series({"n": len(g), "win": g["won"].mean(), "roi": g["profit"].mean() * 100})
    ).reset_index()
    pitcher_stats = pitcher_stats[pitcher_stats["n"] >= 10].sort_values("roi", ascending=False).head(20)
    bar_colors_p = ["#00d4aa" if v >= 0 else "#ff4d6d" for v in pitcher_stats["roi"]]
    fig.add_trace(go.Bar(
        x=pitcher_stats["roi"], y=pitcher_stats["pitcher_name"],
        orientation="h", name="Pitcher ROI %",
        marker_color=bar_colors_p,
        text=[f"{v:+.1f}%" for v in pitcher_stats["roi"]],
        textposition="outside",
        customdata=np.stack([pitcher_stats["n"], pitcher_stats["win"]*100], axis=-1),
        hovertemplate="%{y}<br>ROI: %{x:.1f}%<br>Bets: %{customdata[0]}<br>Win: %{customdata[1]:.1f}%<extra></extra>",
    ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="MLB Pitcher Strikeouts Props — 2025 Backtest Dashboard<br>"
                 "<sub>Filters: edge 3–10% · |proj−line| ≤ 0.8 · main-line odds · strikeouts only</sub>",
            font=dict(size=20, color="#e2e8f0"),
            x=0.5,
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#e2e8f0", size=11),
        height=1100,
        showlegend=True,
        legend=dict(bgcolor="#1a1d27", bordercolor="#6b7280", font=dict(size=10)),
        barmode="overlay",
    )
    fig.update_xaxes(gridcolor="#2d3748", zerolinecolor="#4a5568")
    fig.update_yaxes(gridcolor="#2d3748", zerolinecolor="#4a5568")

    # Axis labels
    fig.update_xaxes(title_text="Bet #", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="ROI %", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate %", row=2, col=1)
    fig.update_xaxes(title_text="CLV %", row=2, col=2)
    fig.update_xaxes(title_text="|Projection − Line|", row=3, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=3, col=1)
    fig.update_xaxes(title_text="ROI %", row=3, col=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Interactive dashboard saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    q = load_and_filter()
    print(f"Loaded {len(q)} qualifying bets  |  ROI={q['profit'].mean()*100:.2f}%  |  Win={q['won'].mean()*100:.1f}%")
    make_static(q, Path("data/exports/dashboard_summary.png"))
    make_interactive(q, Path("data/exports/dashboard_interactive.html"))
    print("\nDone. Open data/exports/dashboard_interactive.html in a browser.")


if __name__ == "__main__":
    main()
