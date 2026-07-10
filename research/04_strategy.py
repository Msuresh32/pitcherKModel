"""Phase 5+6: Strategy optimization and robustness report.

Runs threshold grid search on VALIDATION data only,
then generates a full validation + OOS combined strategy report.

This script is ANALYSIS ONLY — no model is trained here.
It reads outputs from 02 and 03.

Outputs:
  research/strategy/strategy_report.txt  — full strategy analysis
  research/strategy/val_threshold_grid.csv
  research/strategy/edge_by_line.csv     — edge by specific K line (5.5, 6.5, 7.5, etc)
  research/strategy/edge_by_odds.csv     — edge by odds bucket
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATASET_FILE = Path("research/dataset.parquet")
MODEL_DIR    = Path("research/model_results")
OOS_DIR      = Path("research/oos_results")
OUT_DIR      = Path("research/strategy")

OOS_START    = pd.Timestamp("2026-06-01")
VAL_START    = pd.Timestamp("2026-04-01")


def american_to_pnl(odds: float) -> float:
    if odds >= 0:
        return odds / 100
    return 100 / abs(odds)


def load_oos_bets() -> pd.DataFrame:
    path = OOS_DIR / "oos_bets.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "pnl" not in df.columns and "won" in df.columns and "best_odds" in df.columns:
        df["pnl"] = df.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)
    return df


def load_val_bets() -> dict[str, pd.DataFrame]:
    """Load validation bet files from 02_model_search.py."""
    out = {}
    for path in MODEL_DIR.glob("val_bets_*.csv"):
        name = path.stem.replace("val_bets_", "")
        df = pd.read_csv(path)
        df["game_date"] = pd.to_datetime(df["game_date"])
        if "pnl" not in df.columns and "won" in df.columns:
            df["pnl"] = df.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)
        out[name] = df
    return out


def summarize_bets(bets: pd.DataFrame, label: str) -> dict:
    if len(bets) == 0:
        return {"label": label, "n_bets": 0}
    bets = bets.copy()
    if "pnl" not in bets.columns:
        bets["pnl"] = bets.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)
    n = len(bets)
    wins = int(bets["won"].sum())
    units = bets["pnl"].sum()
    roi = units / n
    boot_rois = [bets["pnl"].sample(n, replace=True).mean() for _ in range(1000)]
    lo, hi = np.percentile(boot_rois, [5, 95])
    return {
        "label": label,
        "n_bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "roi": roi,
        "units": units,
        "roi_ci_lo": lo,
        "roi_ci_hi": hi,
    }


def edge_by_segment(bets: pd.DataFrame, col: str, bins=None, labels=None) -> pd.DataFrame:
    """Break down ROI by a column (e.g., line, odds bucket)."""
    if len(bets) == 0 or col not in bets.columns:
        return pd.DataFrame()
    bets = bets.copy()
    if "pnl" not in bets.columns:
        bets["pnl"] = bets.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)

    if bins is not None:
        bets[f"_{col}_bin"] = pd.cut(bets[col], bins=bins, labels=labels)
        group_col = f"_{col}_bin"
    else:
        group_col = col

    return (
        bets.groupby(group_col)
        .agg(n_bets=("pnl", "count"), roi=("pnl", "mean"), units=("pnl", "sum"), win_rate=("won", "mean"))
        .reset_index()
        .rename(columns={group_col: col})
    )


def run_line_analysis(bets: pd.DataFrame, label: str) -> pd.DataFrame:
    """ROI by K line (5.5, 6.5, 7.5, etc.)."""
    return edge_by_segment(bets, "line")


def run_odds_analysis(bets: pd.DataFrame, label: str) -> pd.DataFrame:
    """ROI by odds bucket."""
    if "best_odds" not in bets.columns:
        return pd.DataFrame()
    bins   = [-300, -160, -130, -110, -90, 100, 300]
    labels = ["<-160", "-160:-130", "-130:-110", "-110:-90", "-90:+100", "+100:+300"]
    return edge_by_segment(bets, "best_odds", bins=bins, labels=labels)


def run_edge_bucket_analysis(bets: pd.DataFrame) -> pd.DataFrame:
    """ROI by edge size bucket."""
    if "edge" not in bets.columns:
        return pd.DataFrame()
    bins   = [0, 0.03, 0.05, 0.07, 0.10, 0.15, 1.0]
    labels = ["0-3%", "3-5%", "5-7%", "7-10%", "10-15%", ">15%"]
    return edge_by_segment(bets, "edge", bins=bins, labels=labels)


def over_under_split(bets: pd.DataFrame) -> pd.DataFrame:
    """ROI by bet side."""
    if "bet_side" not in bets.columns:
        return pd.DataFrame()
    bets = bets.copy()
    if "pnl" not in bets.columns:
        bets["pnl"] = bets.apply(lambda r: american_to_pnl(r.best_odds) if r.won else -1.0, axis=1)
    return (
        bets.groupby("bet_side")
        .agg(n_bets=("pnl", "count"), roi=("pnl", "mean"), win_rate=("won", "mean"))
        .reset_index()
    )


def clv_analysis(bets: pd.DataFrame) -> dict:
    """CLV: positive CLV suggests model is ahead of market open."""
    if "clv_over" not in bets.columns:
        return {}
    bets = bets.copy()
    bets["clv_signed"] = np.where(
        bets["bet_side"] == "over", bets["clv_over"], -bets["clv_over"]
    )
    pos = (bets["clv_signed"] > 0).mean()
    return {
        "mean_clv": bets["clv_signed"].mean(),
        "pct_positive_clv": pos,
        "clv_corr_with_pnl": bets[["clv_signed", "pnl"]].corr().iloc[0, 1]
        if "pnl" in bets.columns else None,
    }


def format_df(df: pd.DataFrame, float_cols: list = None, pct_cols: list = None) -> str:
    if df.empty:
        return "  (no data)"
    df = df.copy()
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A")
    if pct_cols:
        for c in pct_cols:
            if c in df.columns:
                df[c] = df[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    return df.to_string(index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Strategy Analysis ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load OOS bets from phase 3
    oos_bets = load_oos_bets()
    val_bets_by_model = load_val_bets()

    # Load threshold grid from phase 2
    grid_path = MODEL_DIR / "threshold_grid.csv"
    grid = pd.read_csv(grid_path) if grid_path.exists() else pd.DataFrame()

    # Load best config
    cfg_path = MODEL_DIR / "best_config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}

    best_model = cfg.get("best_model", "unknown")
    val_bets = val_bets_by_model.get(best_model, pd.DataFrame())

    report_lines = [
        "=" * 65,
        "STRATEGY ANALYSIS REPORT",
        f"Best model: {best_model}",
        f"Edge threshold: {cfg.get('best_edge_threshold', '?')}",
        f"Min prob: {cfg.get('best_min_prob', '?')}",
        "=" * 65,
        "",
    ]

    # --- Threshold grid ---
    if not grid.empty:
        report_lines += [
            "VALIDATION THRESHOLD GRID (top 10 by ROI, min 30 bets)",
            format_df(
                grid[grid["n_bets"] >= 30].head(10)[
                    ["edge_threshold", "min_prob", "n_bets", "roi", "roi_ci_lo", "roi_ci_hi", "win_rate"]
                ],
                float_cols=["roi", "roi_ci_lo", "roi_ci_hi"],
                pct_cols=["win_rate"],
            ),
            "",
        ]
        grid.to_csv(OUT_DIR / "val_threshold_grid.csv", index=False)

    # --- Validation summary ---
    if not val_bets.empty:
        val_summary = summarize_bets(val_bets, "Validation")
        report_lines += [
            "VALIDATION PERIOD (2026-04-01 to 2026-05-31)",
            f"  N bets:   {val_summary['n_bets']}",
            f"  Win rate: {val_summary.get('win_rate', 0):.1%}",
            f"  ROI:      {val_summary.get('roi', 0):+.3f}",
            f"  90% CI:   [{val_summary.get('roi_ci_lo', 0):+.3f}, {val_summary.get('roi_ci_hi', 0):+.3f}]",
            f"  Units:    {val_summary.get('units', 0):+.2f}",
            "",
        ]

        val_line = run_line_analysis(val_bets, "val")
        val_odds = run_odds_analysis(val_bets, "val")
        val_edge = run_edge_bucket_analysis(val_bets)

        report_lines += [
            "Val: ROI by K line",
            format_df(val_line, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
            "Val: ROI by odds bucket",
            format_df(val_odds, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
            "Val: ROI by edge size",
            format_df(val_edge, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
        ]

    # --- OOS summary ---
    if not oos_bets.empty:
        oos_summary = summarize_bets(oos_bets, "OOS")

        report_lines += [
            "OOS PERIOD (2026-06-01 to 2026-07-07)",
            f"  N bets:   {oos_summary['n_bets']}",
            f"  Win rate: {oos_summary.get('win_rate', 0):.1%}",
            f"  ROI:      {oos_summary.get('roi', 0):+.3f}",
            f"  90% CI:   [{oos_summary.get('roi_ci_lo', 0):+.3f}, {oos_summary.get('roi_ci_hi', 0):+.3f}]",
            f"  Units:    {oos_summary.get('units', 0):+.2f}",
            "",
        ]

        oos_line = run_line_analysis(oos_bets, "oos")
        oos_odds = run_odds_analysis(oos_bets, "oos")
        oos_edge = run_edge_bucket_analysis(oos_bets)
        oos_side = over_under_split(oos_bets)
        clv_stats = clv_analysis(oos_bets)

        report_lines += [
            "OOS: ROI by K line",
            format_df(oos_line, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
            "OOS: ROI by odds bucket",
            format_df(oos_odds, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
            "OOS: ROI by edge size",
            format_df(oos_edge, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
            "OOS: Over vs Under",
            format_df(oos_side, float_cols=["roi"], pct_cols=["win_rate"]),
            "",
            "OOS: CLV Analysis",
        ]
        for k, v in clv_stats.items():
            report_lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        oos_line.to_csv(OUT_DIR / "edge_by_line.csv", index=False)
        oos_odds.to_csv(OUT_DIR / "edge_by_odds.csv", index=False)
        oos_edge.to_csv(OUT_DIR / "edge_by_edge_bucket.csv", index=False)

        report_lines += [
            "",
            "FINAL STRATEGY RECOMMENDATION",
        ]
        roi = oos_summary.get("roi", 0)
        n   = oos_summary.get("n_bets", 0)
        clv = clv_stats.get("mean_clv")

        if n < 50:
            rec = (
                "Sample size too small (< 50 bets OOS) to draw conclusions. "
                "Track live for at least 200 bets before scaling."
            )
        elif roi > 0.04 and (clv is None or clv > 0):
            rec = (
                f"Positive ROI ({roi:+.3f}) with positive CLV. "
                "Consider paper trading at half-Kelly for 60 days before real deployment."
            )
        elif roi > 0:
            rec = (
                f"Marginally positive ROI ({roi:+.3f}) but CIs include zero. "
                "Do not deploy. Extend sample period."
            )
        else:
            rec = (
                f"Negative ROI ({roi:+.3f}). Do not deploy. "
                "Return to model search or feature engineering."
            )

        report_lines.append(f"  {rec}")
    else:
        report_lines.append("No OOS bets found. Run 03_oos_eval.py first.")

    report_lines.append("\n" + "=" * 65)
    report = "\n".join(report_lines)

    out_path = OUT_DIR / "strategy_report.txt"
    out_path.write_text(report)
    print(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
