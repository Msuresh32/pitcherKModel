"""
generate_backtest_excel.py — Comprehensive backtest report with new Statcast model.

Sheets:
  1. Summary        — headline metrics, edge verdict
  2. All_Bets       — bet-by-bet detail (2025 + 2026)
  3. By_Month       — WR / ROI / n / avg_edge by calendar month
  4. By_EGP_Bucket  — V2_core vs V4_extra
  5. By_Side        — over vs under
  6. Early_vs_Late  — Apr-Jun vs Jul-Sep
  7. Prob_Floor     — sensitivity to min_model_prob threshold
  8. Calibration    — model prob bin vs actual win rate
  9. Sequential_PnL — cumulative P&L over time
  10. Model_Accuracy — RMSE / MAE by month and season

Usage:
    py -3.14 scripts/generate_backtest_excel.py --config config/config_v4_production.yaml
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.odds.pricing import add_betting_columns, american_to_decimal

# ── constants ─────────────────────────────────────────────────────────────────

MONTH_NAME = {4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct"}

# ── helpers ───────────────────────────────────────────────────────────────────

def amer_to_implied(odds: float) -> float:
    o = float(odds)
    return 100 / (o + 100) if o >= 0 else abs(o) / (abs(o) + 100)


def bet_profit(odds: float, stake: float = 100.0) -> float:
    dec = american_to_decimal(odds)
    return stake * (dec - 1) if not np.isnan(dec) else np.nan


def compute_win(row: pd.Series) -> float:
    actual = row.get("strikeouts")
    line   = row.get("line")
    side   = row.get("best_side")
    if pd.isna(actual) or pd.isna(line) or pd.isna(side):
        return np.nan
    actual, line = float(actual), float(line)
    if actual == line:
        return np.nan  # push
    if side == "over":
        return 1.0 if actual > line else 0.0
    return 1.0 if actual < line else 0.0


def win_rate_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = wins / n
    margin = z * np.sqrt(p * (1 - p) / n)
    return max(0.0, p - margin), min(1.0, p + margin)


def metrics_for(bets: pd.DataFrame, stake: float = 100.0) -> dict:
    resolved = bets[bets["win"].notna()].copy()
    n = len(resolved)
    if n == 0:
        return {"n": 0, "WR": np.nan, "WR_lo": np.nan, "WR_hi": np.nan,
                "ROI": np.nan, "profit": np.nan,
                "avg_edge": np.nan, "avg_clv": np.nan,
                "avg_model_prob": np.nan, "avg_egp": np.nan,
                "sharpe": np.nan, "max_dd": np.nan}
    wins = int(resolved["win"].sum())
    wr   = wins / n
    lo, hi = win_rate_ci(wins, n)
    profit   = resolved["profit"].sum()
    roi      = profit / (n * stake)
    avg_edge = resolved["edge_vs_book"].mean()
    # clv_pct is stored in percentage-point units (e.g. 5.0 = +5% CLV); divide to get fraction
    avg_clv  = (resolved["clv_pct"].mean() / 100.0) if "clv_pct" in resolved.columns else np.nan
    avg_prob = pd.to_numeric(resolved["model_prob"], errors="coerce").mean()
    avg_egp  = resolved["edge_gap_product"].mean()

    # Sharpe (daily P&L, annualised by 162-game season)
    daily  = resolved.groupby("game_date")["profit"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(162)) if daily.std() > 0 else np.nan

    # Max drawdown on sequential bets
    cum      = resolved["profit"].cumsum()
    roll_max = cum.cummax()
    dd       = (cum - roll_max).min()

    return {"n": n, "WR": wr, "WR_lo": lo, "WR_hi": hi,
            "ROI": roi, "profit": profit,
            "avg_edge": avg_edge, "avg_clv": avg_clv,
            "avg_model_prob": avg_prob, "avg_egp": avg_egp,
            "sharpe": sharpe, "max_dd": dd}


def bucket(eg: float, v2: float = 12.0) -> str:
    return "V2_core" if eg >= v2 else "V4_extra"


# ── load data ─────────────────────────────────────────────────────────────────

def load_2025_universe(config: dict) -> pd.DataFrame:
    """Load 2025 real odds + new model projections. Dedup to one row/pitcher-date."""
    proc = config["data"]["processed_dir"]

    pred_path = Path(proc) / "backtest_predictions.csv"
    preds = pd.read_csv(pred_path, low_memory=False,
                        usecols=["game_date", "pitcher_id", "pitcher_name",
                                 "strikeouts", "strikeouts_projection",
                                 "expected_innings_pitched"])
    preds["game_date"]  = pd.to_datetime(preds["game_date"]).dt.strftime("%Y-%m-%d")
    preds["pitcher_id"] = preds["pitcher_id"].astype(str)
    # keep only 2025; dedup to one row per pitcher-date (keep first/highest projection)
    preds = preds[preds["game_date"].str.startswith("2025")].copy()
    preds = preds.sort_values("strikeouts_projection", ascending=False).drop_duplicates(
        subset=["game_date", "pitcher_id"]
    ).reset_index(drop=True)

    odds_path = Path(proc) / "bt_poisson_2025_edges.csv"
    if not odds_path.exists():
        print(f"  SKIP: {odds_path} not found")
        return pd.DataFrame()

    avail     = pd.read_csv(odds_path, nrows=0).columns.tolist()
    odds_cols = ["game_date", "pitcher_id", "team", "opponent", "market",
                 "line", "over_odds", "under_odds", "over_bookmaker", "under_bookmaker"]
    odds = pd.read_csv(odds_path, low_memory=False,
                       usecols=[c for c in odds_cols if c in avail])
    odds["game_date"]  = pd.to_datetime(odds["game_date"]).dt.strftime("%Y-%m-%d")
    odds["pitcher_id"] = odds["pitcher_id"].astype(str)
    odds = odds[odds["market"] == "strikeouts"].copy()

    # Dedup: best generous-odds bookmaker per pitcher-date
    odds["_maxodds"] = odds[["over_odds", "under_odds"]].max(axis=1)
    odds = (odds.sort_values("_maxodds", ascending=False)
                .drop_duplicates(subset=["game_date", "pitcher_id"])
                .drop(columns=["_maxodds"])
                .reset_index(drop=True))

    # CLV
    clv_path = Path(proc) / "bt_pois_2025_e12_clv.csv"
    if clv_path.exists():
        clv = pd.read_csv(clv_path, low_memory=False,
                          usecols=["game_date", "pitcher_id", "clv_pct"])
        clv["game_date"]  = pd.to_datetime(clv["game_date"]).dt.strftime("%Y-%m-%d")
        clv["pitcher_id"] = clv["pitcher_id"].astype(str)
        clv = clv.drop_duplicates(subset=["game_date", "pitcher_id"])
        odds = odds.merge(clv, on=["game_date", "pitcher_id"], how="left")

    merged = odds.merge(
        preds[["game_date", "pitcher_id", "pitcher_name", "strikeouts",
               "strikeouts_projection", "expected_innings_pitched"]],
        on=["game_date", "pitcher_id"], how="inner",
    )
    merged["year"]  = pd.to_datetime(merged["game_date"]).dt.year
    merged["month"] = pd.to_datetime(merged["game_date"]).dt.month
    print(f"  2025: {len(merged)} pitcher-date bets "
          f"(odds: {len(odds)}, preds: {len(preds)})")
    return merged


def load_2026_universe(config: dict) -> pd.DataFrame:
    """Load 2026 real odds with NEW model projections from run_2026_backtest.py output."""
    proc = config["data"]["processed_dir"]

    # Prefer new-model edges (generated by run_2026_backtest.py)
    new_edges_path = Path(proc) / "bt_2026_new_model_edges.csv"
    old_edges_path = Path("data/processed_poisson/bt_poisson_2026_full_edges.csv")

    if new_edges_path.exists():
        edges_path = new_edges_path
        using_new = True
    elif old_edges_path.exists():
        edges_path = old_edges_path
        using_new = False
    else:
        print("  SKIP: no 2026 edges file found")
        return pd.DataFrame()

    need = ["game_date", "pitcher_id", "pitcher_name", "team", "opponent", "market",
            "line", "over_odds", "under_odds", "over_bookmaker", "under_bookmaker",
            "strikeouts", "strikeouts_projection", "expected_innings_pitched"]
    avail = pd.read_csv(edges_path, nrows=0).columns.tolist()
    odds  = pd.read_csv(edges_path, low_memory=False,
                        usecols=[c for c in need if c in avail])
    odds["game_date"]  = pd.to_datetime(odds["game_date"]).dt.strftime("%Y-%m-%d")
    odds["pitcher_id"] = odds["pitcher_id"].astype(str)
    odds = odds[odds["market"] == "strikeouts"].copy()

    odds["_maxodds"] = odds[["over_odds", "under_odds"]].max(axis=1)
    odds = (odds.sort_values("_maxodds", ascending=False)
                .drop_duplicates(subset=["game_date", "pitcher_id"])
                .drop(columns=["_maxodds"])
                .reset_index(drop=True))

    odds["year"]  = pd.to_datetime(odds["game_date"]).dt.year
    odds["month"] = pd.to_datetime(odds["game_date"]).dt.month
    label = "new model projections" if using_new else "OLD model projections (fallback)"
    print(f"  2026: {len(odds)} pitcher-date bets ({label})")
    return odds


# ── price & filter ─────────────────────────────────────────────────────────────

def compute_pricing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Vectorised pricing for the full universe."""
    bet    = config["betting"]
    dist   = bet["market_distribution"].get("strikeouts", "negative_binomial")
    std    = float(bet["default_residual_std"]["strikeouts"])
    shrink = float(bet.get("edge_shrink_factor", 0.7))
    kelly  = float(bet.get("max_kelly_fraction", 0.05))
    return add_betting_columns(
        df.copy(),
        market="strikeouts",
        residual_std=std,
        max_kelly_fraction=kelly,
        edge_shrink_factor=shrink,
        distribution=dist,
        bias_correction=0.0,
        nb_alpha=None,
    )


def apply_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    bet      = config["betting"]
    min_egp  = float(bet.get("min_edge_gap_product", 6.0))
    min_prob = float(bet.get("min_model_prob", 0.0))
    overs_only = bool(bet.get("overs_only", False))
    max_bet_odds = bet.get("max_bet_odds", None)  # e.g. -150 means only bet ≤ -150

    out = df[df["edge_gap_product"].notna() & (df["edge_gap_product"] >= min_egp)].copy()
    out = out[
        ((out["best_side"] == "over")  & (out["strikeouts_projection"] > out["line"])) |
        ((out["best_side"] == "under") & (out["strikeouts_projection"] < out["line"]))
    ].copy()
    if overs_only:
        out = out[out["best_side"] == "over"].copy()
    if min_prob > 0:
        prob = out.apply(
            lambda r: r.get("over_probability", 0.0) if r.get("best_side") == "over"
                      else r.get("under_probability", 0.0), axis=1
        )
        out = out[pd.to_numeric(prob, errors="coerce").fillna(0) >= min_prob].copy()
    if max_bet_odds is not None:
        # Determine the odds for the side we'd bet (over_odds or under_odds)
        bet_odds = out.apply(
            lambda r: r.get("over_odds") if r.get("best_side") == "over"
                      else r.get("under_odds"), axis=1
        )
        out = out[pd.to_numeric(bet_odds, errors="coerce") <= float(max_bet_odds)].copy()
    return out.reset_index(drop=True)


def finalize(df: pd.DataFrame, stake: float = 100.0) -> pd.DataFrame:
    out = df.copy()
    out["model_prob"] = out.apply(
        lambda r: r.get("over_probability") if r.get("best_side") == "over"
                  else r.get("under_probability"), axis=1
    )
    out["book_odds"] = out.apply(
        lambda r: r.get("over_odds") if r.get("best_side") == "over"
                  else r.get("under_odds"), axis=1
    )
    out["book_implied"] = out["book_odds"].apply(
        lambda o: amer_to_implied(o) if pd.notna(o) else np.nan
    )
    out["edge_vs_book"] = (pd.to_numeric(out["model_prob"], errors="coerce")
                           - out["book_implied"])
    out["win"]    = out.apply(compute_win, axis=1)
    out["profit"] = out.apply(
        lambda r: (bet_profit(r["book_odds"], stake) if r["win"] == 1.0
                   else (-stake if r["win"] == 0.0 else 0.0)), axis=1
    )
    out["strategy_bucket"] = out["edge_gap_product"].apply(
        lambda eg: bucket(float(eg))
    )
    out["period"]     = out["month"].apply(
        lambda m: "Early (Apr-Jun)" if m in (4,5,6)
                  else ("Late (Jul-Sep)" if m in (7,8,9) else "Other")
    )
    out["month_name"] = out["month"].map(MONTH_NAME)
    return out


# ── Excel builder ──────────────────────────────────────────────────────────────

def _write_table(ws, df: pd.DataFrame, hdr_fmt, start_row: int = 0) -> None:
    for c, col_name in enumerate(df.columns):
        ws.write(start_row, c, str(col_name), hdr_fmt)
    for r, (_, row) in enumerate(df.iterrows()):
        for c, val in enumerate(row):
            v = val.item() if hasattr(val, "item") else val
            if isinstance(v, float) and np.isnan(v):
                ws.write(start_row + 1 + r, c, "")
            else:
                ws.write(start_row + 1 + r, c, v)


def build_excel(
    filtered: pd.DataFrame,
    unfiltered: pd.DataFrame,
    preds_all: pd.DataFrame,
    config: dict,
    out_path: Path,
    stake: float = 100.0,
) -> None:
    import xlsxwriter  # type: ignore

    wb  = xlsxwriter.Workbook(str(out_path))
    hdr = wb.add_format({"bold": True, "bg_color": "#1F3864",
                         "font_color": "white", "border": 1})
    bold = wb.add_format({"bold": True})
    pct  = wb.add_format({"num_format": "0.0%"})
    pct2 = wb.add_format({"num_format": "0.00%"})
    dlr  = wb.add_format({"num_format": "$#,##0"})
    num1 = wb.add_format({"num_format": "0.0"})
    num2 = wb.add_format({"num_format": "0.00"})

    # ── Sheet 1: Summary ────────────────────────────────────────────────────
    ws = wb.add_worksheet("Summary")
    ws.set_column(0, 0, 40)
    ws.set_column(1, 13, 16)

    ws.write(0, 0, "MLB Pitcher K Model — Backtest Report (New Statcast Model)", bold)
    ws.write(1, 0, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    ws.write(2, 0,
             f"V4 filters: EGP >= {config['betting']['min_edge_gap_product']}, "
             f"model_prob >= {config['betting'].get('min_model_prob', 0):.0%}, "
             "direction agreement")

    segments = [
        ("ALL (EGP>=6, no prob floor)",     unfiltered),
        ("FILTERED (EGP>=6 + prob>=65%)",   filtered),
        ("2025 — filtered",                 filtered[filtered["year"] == 2025]),
        ("2026 — filtered",                 filtered[filtered["year"] == 2026]),
        ("Early Apr-Jun — filtered",        filtered[filtered["month"].isin([4,5,6])]),
        ("Late Jul-Sep — filtered",         filtered[filtered["month"].isin([7,8,9])]),
        ("V2_core (EGP>=12) — filtered",    filtered[filtered["strategy_bucket"] == "V2_core"]),
        ("V4_extra (6<=EGP<12) — filtered", filtered[filtered["strategy_bucket"] == "V4_extra"]),
        ("Overs only — filtered",           filtered[filtered["best_side"] == "over"]),
        ("Unders only — filtered",          filtered[filtered["best_side"] == "under"]),
    ]

    col_hdrs = ["Segment", "n", "Win Rate", "CI low", "CI high",
                "ROI", "Profit ($100/bet)", "Avg Edge vs Book",
                "Avg CLV%", "Avg Model Prob", "Avg EGP", "Sharpe", "Max Drawdown"]
    for c, h in enumerate(col_hdrs):
        ws.write(4, c, h, hdr)

    fmts = [None, None, pct, pct, pct, pct, dlr, pct2, pct2, pct, num1, num2, dlr]
    for r, (label, sub) in enumerate(segments):
        m = metrics_for(sub, stake)
        vals = [label, m["n"], m["WR"], m["WR_lo"], m["WR_hi"], m["ROI"],
                m["profit"], m["avg_edge"], m["avg_clv"],
                m["avg_model_prob"], m["avg_egp"], m["sharpe"], m["max_dd"]]
        for c, (val, fmt) in enumerate(zip(vals, fmts)):
            v = val.item() if hasattr(val, "item") else val
            if isinstance(v, float) and np.isnan(v):
                ws.write(5 + r, c, "—")
            elif fmt:
                ws.write(5 + r, c, v, fmt)
            else:
                ws.write(5 + r, c, v)

    # Model accuracy box
    ws.write(17, 0, "MODEL ACCURACY (OOS walk-forward, new Statcast model)", bold)
    ws.write(18, 0, "Metric", hdr); ws.write(18, 1, "Value", hdr)
    if "strikeouts" in preds_all.columns and "strikeouts_projection" in preds_all.columns:
        err = (preds_all["strikeouts_projection"] - preds_all["strikeouts"]).dropna()
        for i, (label, val) in enumerate([
            ("RMSE (strikeouts)", np.sqrt((err**2).mean())),
            ("MAE  (strikeouts)", err.abs().mean()),
            ("Bias (mean error)", err.mean()),
        ]):
            ws.write(19 + i, 0, label)
            ws.write(19 + i, 1, round(float(val), 4), num2)

    # ── Sheet 2: All Bets ───────────────────────────────────────────────────
    ws2 = wb.add_worksheet("All_Bets")
    show = ["game_date", "pitcher_name", "team", "opponent",
            "line", "strikeouts_projection", "abs_proj_gap",
            "best_side", "book_odds", "model_prob", "book_implied",
            "edge_vs_book", "edge_pct", "edge_gap_product",
            "strategy_bucket", "strikeouts", "win", "profit",
            "clv_pct", "year", "month_name", "period"]
    show = [c for c in show if c in filtered.columns]
    ws2.set_column(0, len(show), 14)
    _write_table(ws2, filtered[show], hdr)

    # ── Sheet 3: By Month ───────────────────────────────────────────────────
    ws3 = wb.add_worksheet("By_Month")
    rows_m = []
    for (yr, mo), grp in filtered.groupby(["year", "month"]):
        m = metrics_for(grp, stake)
        m.update({"year": int(yr), "month_num": int(mo),
                  "month_name": MONTH_NAME.get(mo, str(mo))})
        rows_m.append(m)
    df_m = pd.DataFrame(rows_m).sort_values(["year","month_num"])
    out_m = ["year","month_name","n","WR","WR_lo","WR_hi",
             "ROI","profit","avg_edge","avg_clv","avg_model_prob","avg_egp","sharpe"]
    _write_table(ws3, df_m[[c for c in out_m if c in df_m.columns]], hdr)
    ws3.set_column(0, 13, 14)

    # ── Sheet 4: By EGP Bucket ──────────────────────────────────────────────
    ws4 = wb.add_worksheet("By_EGP_Bucket")
    rows_b = []
    for label, sub in [
        ("No floor — EGP>=6",              unfiltered),
        ("No floor — V2_core (EGP>=12)",   unfiltered[unfiltered["strategy_bucket"]=="V2_core"]),
        ("Prob>=65% — EGP>=6",             filtered),
        ("Prob>=65% — V2_core (EGP>=12)",  filtered[filtered["strategy_bucket"]=="V2_core"]),
        ("Prob>=65% — V4_extra (6-12)",    filtered[filtered["strategy_bucket"]=="V4_extra"]),
    ]:
        m = metrics_for(sub, stake); m["Segment"] = label; rows_b.append(m)
    df_b = pd.DataFrame(rows_b)
    _write_table(ws4, df_b[["Segment","n","WR","WR_lo","WR_hi",
                              "ROI","profit","avg_edge","avg_clv","avg_model_prob","avg_egp"]], hdr)
    ws4.set_column(0, 0, 35); ws4.set_column(1, 11, 14)

    # ── Sheet 5: By Side ────────────────────────────────────────────────────
    ws5 = wb.add_worksheet("By_Side")
    rows_s = []
    for (yr, side), grp in filtered.groupby(["year","best_side"]):
        m = metrics_for(grp, stake); m["year"] = int(yr); m["side"] = side; rows_s.append(m)
    for side, grp in filtered.groupby("best_side"):
        m = metrics_for(grp, stake); m["year"] = "All"; m["side"] = side; rows_s.append(m)
    df_s = pd.DataFrame(rows_s)
    _write_table(ws5, df_s[["year","side","n","WR","WR_lo","WR_hi",
                              "ROI","profit","avg_edge","avg_clv"]], hdr)
    ws5.set_column(0, 10, 14)

    # ── Sheet 6: Early vs Late ──────────────────────────────────────────────
    ws6 = wb.add_worksheet("Early_vs_Late")
    rows_p = []
    for (yr, period), grp in filtered.groupby(["year","period"]):
        m = metrics_for(grp, stake); m["year"] = int(yr); m["period"] = period; rows_p.append(m)
    df_p = pd.DataFrame(rows_p)
    _write_table(ws6, df_p[["year","period","n","WR","WR_lo","WR_hi",
                              "ROI","profit","avg_edge","avg_model_prob"]], hdr)
    ws6.set_column(0, 10, 18)

    # ── Sheet 7: Prob Floor Sensitivity ────────────────────────────────────
    ws7 = wb.add_worksheet("Prob_Floor")
    thresholds = [0.55, 0.58, 0.60, 0.63, 0.65, 0.67, 0.70, 0.73, 0.75]
    rows_t = []
    for t in thresholds:
        prob_ser = unfiltered.apply(
            lambda r: r.get("over_probability", 0.0) if r.get("best_side") == "over"
                      else r.get("under_probability", 0.0), axis=1
        )
        sub_all = unfiltered[pd.to_numeric(prob_ser, errors="coerce").fillna(0) >= t]
        m = metrics_for(sub_all, stake); m["min_prob"] = t; m["year"] = "All"
        rows_t.append(m)
        for yr in sorted(unfiltered["year"].unique()):
            sub_yr = sub_all[sub_all["year"] == yr]
            m2 = metrics_for(sub_yr, stake); m2["min_prob"] = t; m2["year"] = int(yr)
            rows_t.append(m2)
    df_t = pd.DataFrame(rows_t)
    _write_table(ws7, df_t[["min_prob","year","n","WR","WR_lo","WR_hi",
                              "ROI","profit","avg_edge","avg_model_prob"]], hdr)
    ws7.set_column(0, 10, 14)

    # ── Sheet 8: Calibration ────────────────────────────────────────────────
    ws8 = wb.add_worksheet("Calibration")
    resolved = unfiltered[unfiltered["win"].notna()].copy()
    resolved["model_prob_num"] = pd.to_numeric(resolved["model_prob"], errors="coerce")
    bins = [0.50, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70, 0.75, 0.80, 1.01]
    resolved["prob_bin"] = pd.cut(resolved["model_prob_num"], bins=bins, right=False)
    rows_c = []
    for b, grp in resolved.groupby("prob_bin", observed=True):
        n = len(grp); wins = int(grp["win"].sum())
        rows_c.append({
            "prob_bin":        str(b),
            "avg_model_prob":  round(grp["model_prob_num"].mean(), 4),
            "actual_win_rate": round(wins / n, 4) if n > 0 else np.nan,
            "n": n,
            "overconfidence":  round(grp["model_prob_num"].mean() - wins/n, 4) if n > 0 else np.nan,
        })
    _write_table(ws8, pd.DataFrame(rows_c), hdr)
    ws8.set_column(0, 5, 18)

    # ── Sheet 9: Sequential P&L ─────────────────────────────────────────────
    ws9 = wb.add_worksheet("Sequential_PnL")
    pnl = (filtered[filtered["win"].notna()]
           .sort_values(["game_date","pitcher_name"])
           .reset_index(drop=True))
    pnl["bet_num"]    = pnl.index + 1
    pnl["cum_profit"] = pnl["profit"].cumsum()
    pnl_cols = ["bet_num","game_date","pitcher_name","best_side",
                "line","strikeouts_projection","strikeouts",
                "book_odds","win","profit","cum_profit"]
    pnl_cols = [c for c in pnl_cols if c in pnl.columns]
    _write_table(ws9, pnl[pnl_cols], hdr)
    ws9.set_column(0, len(pnl_cols), 14)

    # ── Sheet 10: Odds Bucket Analysis ──────────────────────────────────────
    ws10 = wb.add_worksheet("Odds_Brackets")
    rows_o = []
    brackets = [
        ("Heavier than -150",   -999, -150),
        ("-150 to -131",        -150, -130),
        ("-130 to -116",        -130, -115),
        ("-115 to pick'em",     -115,    1),
        ("Plus money (+100+)",     1,  300),
    ]
    for yr_label, sub_df in [("All years", filtered),
                              ("2025",      filtered[filtered["year"]==2025]),
                              ("2026",      filtered[filtered["year"]==2026])]:
        sub_res = sub_df[sub_df["win"].notna()].copy()
        for label, lo, hi in brackets:
            sub = sub_res[(sub_res["book_odds"] > lo) & (sub_res["book_odds"] <= hi)]
            if len(sub) == 0:
                continue
            wr  = sub["win"].mean()
            roi = sub["profit"].sum() / (len(sub) * stake)
            avg_odds = sub["book_odds"].mean()
            beven = abs(avg_odds) / (abs(avg_odds) + 100) if avg_odds < 0 else 100 / (avg_odds + 100) if avg_odds > 0 else 0.5
            rows_o.append({
                "year": yr_label, "odds_bracket": label, "n": len(sub),
                "WR": wr, "breakeven_WR": beven,
                "WR_vs_beven": wr - beven,
                "ROI": roi, "profit": sub["profit"].sum(), "avg_odds": avg_odds,
            })
    df_o = pd.DataFrame(rows_o)
    _write_table(ws10, df_o, hdr)
    ws10.set_column(0, 0, 16); ws10.set_column(1, 1, 22); ws10.set_column(2, 9, 14)
    ws10.write(len(df_o)+2, 0,
               "Key finding: model's alpha concentrates in heavy-juice bets "
               "(market agrees with model direction). Favorable-odds bets underperform — "
               "suggests adding market_consensus filter.", bold)

    # ── Sheet 11: Model Accuracy ─────────────────────────────────────────────
    ws11 = wb.add_worksheet("Model_Accuracy")
    acc = preds_all[["game_date","year","month","strikeouts","strikeouts_projection"]].dropna().copy()
    acc["error"] = acc["strikeouts_projection"] - acc["strikeouts"]
    rows_a = []
    for (yr, mo), grp in acc.groupby(["year","month"]):
        if len(grp) < 3:
            continue
        rows_a.append({
            "year": int(yr), "month_num": int(mo),
            "month_name": MONTH_NAME.get(mo, str(mo)),
            "n": len(grp),
            "RMSE": round(np.sqrt((grp["error"]**2).mean()), 4),
            "MAE":  round(grp["error"].abs().mean(), 4),
            "bias": round(grp["error"].mean(), 4),
        })
    df_a = pd.DataFrame(rows_a).sort_values(["year","month_num"])
    _write_table(ws11, df_a[["year","month_name","n","RMSE","MAE","bias"]], hdr)
    ws11.set_column(0, 6, 14)

    wb.close()
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_v4_production.yaml")
    parser.add_argument("--out",    default="reports/backtest_new_model.xlsx")
    parser.add_argument("--stake",  type=float, default=100.0)
    args = parser.parse_args()

    config = load_config(args.config)
    stake  = args.stake

    print("Loading 2025 universe...")
    u25 = load_2025_universe(config)
    print("Loading 2026 universe...")
    u26 = load_2026_universe(config)

    raw = pd.concat([u25, u26], ignore_index=True) if not u26.empty else u25
    print(f"Total: {len(raw)} pitcher-date rows across both seasons")

    pred_path = Path(config["data"]["processed_dir"]) / "backtest_predictions.csv"
    preds_all = pd.read_csv(pred_path, low_memory=False,
                            usecols=["game_date","pitcher_id","strikeouts","strikeouts_projection"])
    preds_all["game_date"] = pd.to_datetime(preds_all["game_date"])
    preds_all["year"]      = preds_all["game_date"].dt.year
    preds_all["month"]     = preds_all["game_date"].dt.month
    preds_all["game_date"] = preds_all["game_date"].dt.strftime("%Y-%m-%d")

    print("Computing pricing (vectorised)...")
    priced = compute_pricing(raw, config)

    min_egp = float(config["betting"].get("min_edge_gap_product", 6.0))
    unfiltered = priced[
        priced["edge_gap_product"].notna() & (priced["edge_gap_product"] >= min_egp) &
        (
            ((priced["best_side"] == "over")  & (priced["strikeouts_projection"] > priced["line"])) |
            ((priced["best_side"] == "under") & (priced["strikeouts_projection"] < priced["line"]))
        )
    ].copy()
    unfiltered = finalize(unfiltered, stake)
    print(f"Unfiltered (EGP>={min_egp}, no prob floor): {len(unfiltered)} bets")

    filtered = apply_filters(priced, config)
    filtered = finalize(filtered, stake)
    min_prob = config["betting"].get("min_model_prob", 0)
    print(f"Filtered (all V4 incl. prob>={min_prob:.0%}): {len(filtered)} bets")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print("Building Excel workbook...")
    build_excel(filtered, unfiltered, preds_all, config, Path(args.out), stake)

    print()
    print("=" * 65)
    print("  HEADLINE METRICS  ($100/bet flat stake)")
    print("=" * 65)
    for label, sub in [
        ("ALL BETS (EGP>=6, no floor)",      unfiltered),
        ("FILTERED (EGP>=6 + prob>=65%)",    filtered),
        ("  2025 filtered",                  filtered[filtered["year"]==2025]),
        ("  2026 filtered",                  filtered[filtered["year"]==2026]),
        ("  Early Apr-Jun (filtered)",       filtered[filtered["month"].isin([4,5,6])]),
        ("  Late Jul-Sep (filtered)",        filtered[filtered["month"].isin([7,8,9])]),
        ("  V2_core EGP>=12 (filtered)",     filtered[filtered["strategy_bucket"]=="V2_core"]),
        ("  Overs (filtered)",               filtered[filtered["best_side"]=="over"]),
        ("  Unders (filtered)",              filtered[filtered["best_side"]=="under"]),
    ]:
        m = metrics_for(sub, stake)
        clv  = f"  CLV={m['avg_clv']:.2%}" if not np.isnan(m.get("avg_clv", np.nan)) else ""
        wr   = f"{m['WR']:.1%}" if not np.isnan(m["WR"]) else "  —  "
        roi  = f"{m['ROI']:+.1%}" if not np.isnan(m["ROI"]) else "  —  "
        shp  = f"  Sharpe={m['sharpe']:.2f}" if not np.isnan(m.get("sharpe", np.nan)) else ""
        print(f"  {label:<40} n={m['n']:4d}  WR={wr}  ROI={roi}{clv}{shp}")
    print("=" * 65)


if __name__ == "__main__":
    main()
