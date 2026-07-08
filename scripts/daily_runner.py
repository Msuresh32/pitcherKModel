"""
daily_runner.py — V4 production decision-support pipeline

Runs the full daily workflow:
  1. Invoke project_daily.py to produce model projections
  2. Load all candidates from the exported CSV
  3. Apply V4 filters and tag strategy buckets (V2_core / V4_extra)
  4. Write data/daily/{date}_candidates.csv
  5. Append new rows to data/logs/live_bets_log.csv
  6. Write reports/daily/{date}_summary.md
  7. Print console summary

After reviewing candidates, place bets manually on NoVig (limit orders).
Record your stake and fill price via:
    python scripts/update_results.py --pitcher "Name" --stake 100 --price-improvement 10

Usage:
    python scripts/daily_runner.py
    python scripts/daily_runner.py --date 2026-07-01
    python scripts/daily_runner.py --skip-model   # reuse existing model output
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config

# ── Output paths ──────────────────────────────────────────────────────────────
LOGS_DIR      = Path("data/logs")
DAILY_DIR     = Path("data/daily")
REPORTS_DAILY = Path("reports/daily")
REPORTS_PERF  = Path("reports/performance")
LOG_FILE      = LOGS_DIR / "live_bets_log.csv"

LOG_COLUMNS = [
    "date", "timestamp", "game_id", "pitcher", "team", "opponent",
    "market", "line", "side", "model_projection", "model_prob",
    "market_prob", "edge_pct", "abs_proj_gap", "edge_gap_product",
    "strategy_bucket", "sportsbook", "sportsbook_odds",
    "stake", "price_improvement_cents",
    "result", "profit", "closing_odds", "clv_cents", "notes",
]


def _ensure_dirs() -> None:
    for d in (LOGS_DIR, DAILY_DIR, REPORTS_DAILY, REPORTS_PERF):
        d.mkdir(parents=True, exist_ok=True)


def _amer_to_dec(odds: float) -> float:
    odds = float(odds)
    return (1 + odds / 100) if odds >= 0 else (1 + 100 / abs(odds))


def _amer_to_implied(odds: float) -> float:
    dec = _amer_to_dec(odds)
    return 1 / dec if dec > 0 else float("nan")


def _tag_bucket(eg: float, v2_thresh: float, v4_thresh: float) -> str:
    if eg >= v2_thresh:
        return "V2_core"
    if eg >= v4_thresh:
        return "V4_extra"
    return "Filtered"


def _load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype=str)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _save_log(df: pd.DataFrame) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_FILE, index=False)


# ── Step 1: run the model ─────────────────────────────────────────────────────

def _run_model(config_path: str, target_date: str) -> None:
    cmd = [sys.executable, "scripts/project_daily.py",
           "--config", config_path, "--date", target_date]
    print(f"[daily_runner] Running model: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"project_daily.py exited with code {result.returncode}. "
            "Check the output above for errors."
        )


# ── Step 2: load model candidates ────────────────────────────────────────────

def _load_model_candidates(target_date: str) -> pd.DataFrame:
    candidates = [
        Path("data/exports") / f"daily_pitcher_props_{target_date}.csv",
        Path("data/exports") / f"daily_pitcher_props_{target_date}_.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"No daily picks file found for {target_date}. "
        "Expected: data/exports/daily_pitcher_props_{date}.csv"
    )


# ── Step 3: apply V4 filters and tag buckets ──────────────────────────────────

def _apply_v4_filters(picks: pd.DataFrame, target_date: str, config: dict) -> pd.DataFrame:
    bet         = config["betting"]
    min_egp     = float(bet.get("min_edge_gap_product", 6.0))
    skip_months = bet.get("skip_months", [])
    overs_only  = bool(bet.get("overs_only", False))

    month = pd.to_datetime(target_date).month
    if month in (skip_months or []):
        print(f"[V4 filter] Month {month} is in skip_months — no bets today.")
        return pd.DataFrame(columns=picks.columns)

    df = picks.copy()
    for col in ("edge_pct", "edge_gap_product", "abs_proj_gap", "line",
                "over_odds", "under_odds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "edge_gap_product" not in df.columns or df["edge_gap_product"].isna().all():
        proj_col = "strikeouts_projection" if "strikeouts_projection" in df.columns else "projection"
        if proj_col in df.columns and "line" in df.columns:
            df["abs_proj_gap"] = (df[proj_col] - df["line"]).abs()
            df["edge_gap_product"] = df["edge_pct"] * df["abs_proj_gap"]

    df = df[df["edge_gap_product"].notna() & (df["edge_gap_product"] >= min_egp)].copy()

    proj_col = "strikeouts_projection" if "strikeouts_projection" in df.columns else "projection"
    if proj_col in df.columns and "line" in df.columns and "best_side" in df.columns:
        agrees = (
            ((df["best_side"] == "over")  & (df[proj_col] > df["line"])) |
            ((df["best_side"] == "under") & (df[proj_col] < df["line"]))
        )
        df = df[agrees].copy()

    if overs_only and not df.empty:
        df = df[df["best_side"] == "over"].copy()

    if "betting_disabled" in df.columns:
        df = df[~df["betting_disabled"]].copy()

    min_model_prob = float(bet.get("min_model_prob", 0.0))
    if min_model_prob > 0 and not df.empty:
        if "over_probability" in df.columns and "under_probability" in df.columns:
            prob = df.apply(
                lambda r: r.get("over_probability", 0) if r.get("best_side") == "over"
                          else r.get("under_probability", 0),
                axis=1,
            )
            df = df[pd.to_numeric(prob, errors="coerce").fillna(0) >= min_model_prob].copy()

    max_bet_odds = bet.get("max_bet_odds")
    if max_bet_odds is not None and not df.empty:
        bet_odds = df.apply(
            lambda r: r.get("over_odds") if r.get("best_side") == "over"
                      else r.get("under_odds"),
            axis=1,
        )
        n_before = len(df)
        df = df[pd.to_numeric(bet_odds, errors="coerce") <= float(max_bet_odds)].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[V4 filter] max_bet_odds={int(max_bet_odds)}: dropped {n_dropped} non-qualifying bet(s).")

    return df.reset_index(drop=True)


# ── Step 4: build log rows ────────────────────────────────────────────────────

def _build_log_rows(flagged: pd.DataFrame, target_date: str, config: dict) -> pd.DataFrame:
    bet       = config["betting"]
    v2_thresh = float(bet.get("v2_core_threshold", 12.0))
    v4_thresh = float(bet.get("min_edge_gap_product", 6.0))
    ts        = datetime.now().isoformat(timespec="seconds")

    rows = []
    for _, row in flagged.iterrows():
        side     = str(row.get("best_side", ""))
        odds_col = "over_odds" if side == "over" else "under_odds"
        book_col = "over_bookmaker" if side == "over" else "under_bookmaker"
        sb_odds  = row.get(odds_col, np.nan)
        proj_col = "strikeouts_projection" if "strikeouts_projection" in row.index else "projection"
        proj     = row.get(proj_col, np.nan)

        prob_col   = "over_probability" if side == "over" else "under_probability"
        model_prob = row.get(prob_col, np.nan)
        market_prob = _amer_to_implied(float(sb_odds)) if pd.notna(sb_odds) else np.nan

        eg = float(row.get("edge_gap_product", 0))

        rows.append({
            "date":                   target_date,
            "timestamp":              ts,
            "game_id":                row.get("pitcher_id", ""),
            "pitcher":                row.get("pitcher_name", row.get("player_name", "")),
            "team":                   row.get("team", ""),
            "opponent":               row.get("opponent", ""),
            "market":                 row.get("market", "strikeouts"),
            "line":                   row.get("line", ""),
            "side":                   side,
            "model_projection":       proj,
            "model_prob":             round(float(model_prob), 4) if pd.notna(model_prob) else "",
            "market_prob":            round(float(market_prob), 4) if pd.notna(market_prob) else "",
            "edge_pct":               round(float(row.get("edge_pct", 0)), 4),
            "abs_proj_gap":           round(float(row.get("abs_proj_gap", 0)), 4),
            "edge_gap_product":       round(eg, 4),
            "strategy_bucket":        _tag_bucket(eg, v2_thresh, v4_thresh),
            "sportsbook":             row.get(book_col, ""),
            "sportsbook_odds":        sb_odds,
            "stake":                  "",
            "price_improvement_cents": "",
            "result":                 "",
            "profit":                 "",
            "closing_odds":           "",
            "clv_cents":              "",
            "notes":                  "",
        })

    return pd.DataFrame(rows, columns=LOG_COLUMNS)


# ── Step 5: append to log ─────────────────────────────────────────────────────

def _append_to_log(new_rows: pd.DataFrame, target_date: str) -> int:
    if new_rows.empty:
        return 0
    existing = _load_log()
    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"].astype(str) != str(target_date)].copy()
    combined = pd.concat([existing, new_rows.astype(str)], ignore_index=True)
    _save_log(combined)
    return len(new_rows)


# ── Step 6: write daily candidates file ──────────────────────────────────────

def _write_daily_file(flagged: pd.DataFrame, target_date: str) -> None:
    path = DAILY_DIR / f"{target_date}_candidates.csv"
    flagged.to_csv(path, index=False)
    print(f"[daily_runner] Candidates: {path} ({len(flagged)} rows)")


# ── Step 7: summary ───────────────────────────────────────────────────────────

def _build_table(flagged: pd.DataFrame, v2_thresh: float, v4_thresh: float) -> str:
    proj_col = "strikeouts_projection" if "strikeouts_projection" in flagged.columns else "projection"
    headers = ["Pitcher", "Side", "Line", "Proj", "EGP", "Edge%", "Bucket", "SB Odds"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in flagged.sort_values("edge_gap_product", ascending=False).iterrows():
        side     = str(row.get("best_side", ""))
        odds_col = "over_odds" if side == "over" else "under_odds"
        sb_odds  = row.get(odds_col, "")
        eg       = float(row.get("edge_gap_product", 0))
        try:
            odds_str = f"{int(float(sb_odds)):+d}"
        except (ValueError, TypeError):
            odds_str = "?"
        try:
            proj_str = f"{float(row.get(proj_col, '')):.1f}"
        except (ValueError, TypeError):
            proj_str = "?"
        lines.append(
            f"| {row.get('pitcher_name', row.get('player_name',''))}"
            f" | {side} | {float(row.get('line',0)):.1f}"
            f" | {proj_str} | {eg:.2f} | {float(row.get('edge_pct',0)):.2f}"
            f" | {_tag_bucket(eg, v2_thresh, v4_thresh)} | {odds_str} |"
        )
    return "\n".join(lines)


def _write_summary(flagged: pd.DataFrame, target_date: str, config: dict, n_logged: int) -> Path:
    bet       = config["betting"]
    v2_thresh = float(bet.get("v2_core_threshold", 12.0))
    v4_thresh = float(bet.get("min_edge_gap_product", 6.0))

    v2_n = len(flagged[flagged["strategy_bucket"] == "V2_core"]) if "strategy_bucket" in flagged.columns else 0
    v4_n = len(flagged[flagged["strategy_bucket"] == "V4_extra"]) if "strategy_bucket" in flagged.columns else 0

    lines = [
        f"# Daily Model Summary — {target_date}",
        "",
        "## Configuration",
        f"- Min edge×gap: {v4_thresh}",
        f"- V2_core threshold: {v2_thresh}",
        f"- Min model prob: {float(bet.get('min_model_prob', 0.0)):.0%}",
        f"- Max bet odds: {bet.get('max_bet_odds', 'none')}",
        f"- Skip months: {bet.get('skip_months', [])}",
        f"- Overs only: {bet.get('overs_only', False)}",
        f"- Flat stake: ${float(bet.get('flat_stake', 100)):.0f}",
        "",
        "## Candidates",
        f"- Total (edge×gap ≥ {v4_thresh}): **{len(flagged)}**",
        f"- V2_core (eg ≥ {v2_thresh}): {v2_n}",
        f"- V4_extra ({v4_thresh} ≤ eg < {v2_thresh}): {v4_n}",
        f"- Rows logged: {n_logged}",
        "",
        "## All Candidates",
        "",
        _build_table(flagged, v2_thresh, v4_thresh),
        "",
        "## Next Steps",
        "1. Review candidates above.",
        "2. Place limit orders on NoVig at your target price.",
        "3. When filled, record stake + price improvement:",
        "   `python scripts/update_results.py --pitcher \"Name\" --stake 100 --price-improvement 10`",
        "4. Next morning, settle results:",
        "   `python scripts/update_results.py`",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
    ]

    path = REPORTS_DAILY / f"{target_date}_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_console_summary(flagged: pd.DataFrame, target_date: str, config: dict) -> None:
    bet       = config["betting"]
    v2_thresh = float(bet.get("v2_core_threshold", 12.0))
    v4_thresh = float(bet.get("min_edge_gap_product", 6.0))
    proj_col  = "strikeouts_projection" if "strikeouts_projection" in flagged.columns else "projection"

    print()
    print("=" * 65)
    print(f"  V4 CANDIDATES — {target_date}")
    print("=" * 65)

    if flagged.empty:
        print("  No candidates today.")
        print()
        return

    print(f"\n  {'Pitcher':<22} {'Side':<6} {'Line':>5} {'Proj':>5} {'EGP':>6}  {'Bucket':<10} {'SB Odds':>8}")
    print("  " + "-" * 70)

    for _, row in flagged.sort_values("edge_gap_product", ascending=False).iterrows():
        side     = str(row.get("best_side", ""))
        odds_col = "over_odds" if side == "over" else "under_odds"
        sb_odds  = row.get(odds_col, "")
        eg       = float(row.get("edge_gap_product", 0))
        try:
            sb_str = f"{int(float(sb_odds)):+d}"
        except (ValueError, TypeError):
            sb_str = "?"
        try:
            proj_str = f"{float(row.get(proj_col, '')):.1f}"
        except (ValueError, TypeError):
            proj_str = "?"

        print(
            f"  {str(row.get('pitcher_name', row.get('player_name',''))):<22} "
            f"{side:<6} {float(row.get('line', 0)):>5.1f} {proj_str:>5} "
            f"{eg:>6.1f}  {_tag_bucket(eg, v2_thresh, v4_thresh):<10} {sb_str:>8}"
        )

    v2_n = sum(1 for _, r in flagged.iterrows()
               if _tag_bucket(float(r.get("edge_gap_product", 0)), v2_thresh, v4_thresh) == "V2_core")
    v4_n = len(flagged) - v2_n

    print()
    print(f"  V2_core (eg >= {v2_thresh:.0f}): {v2_n}   V4_extra: {v4_n}   Total: {len(flagged)}")
    print()
    print("  Place limit orders on NoVig, then record fills:")
    print("  py -3.14 scripts/update_results.py --pitcher \"Name\" --stake 100 --price-improvement 10")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V4 daily decision-support pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",     default="config/config_v4_production.yaml")
    parser.add_argument("--date",       default=date.today().isoformat())
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip running project_daily.py (use existing output)")
    args = parser.parse_args()

    _ensure_dirs()
    config      = load_config(args.config)
    target_date = args.date

    if not args.skip_model:
        _run_model(args.config, target_date)
    else:
        print(f"[daily_runner] --skip-model: using existing picks for {target_date}")

    picks = _load_model_candidates(target_date)
    print(f"[daily_runner] Loaded {len(picks)} rows from model output")

    flagged = _apply_v4_filters(picks, target_date, config)
    print(f"[daily_runner] After V4 filters: {len(flagged)} candidates")

    if flagged.empty:
        print("[daily_runner] No candidates today.")
        _print_console_summary(flagged, target_date, config)
        return

    bet       = config["betting"]
    v2_thresh = float(bet.get("v2_core_threshold", 12.0))
    v4_thresh = float(bet.get("min_edge_gap_product", 6.0))
    flagged["strategy_bucket"] = flagged["edge_gap_product"].apply(
        lambda eg: _tag_bucket(float(eg), v2_thresh, v4_thresh)
    )

    def _sb_odds(row):
        side = str(row.get("best_side", ""))
        return row.get("over_odds" if side == "over" else "under_odds", "")

    def _sb_name(row):
        side = str(row.get("best_side", ""))
        return row.get("over_bookmaker" if side == "over" else "under_bookmaker", "")

    flagged["sportsbook_odds"] = flagged.apply(_sb_odds, axis=1)
    flagged["sportsbook"]      = flagged.apply(_sb_name, axis=1)

    log_rows = _build_log_rows(flagged, target_date, config)
    _write_daily_file(flagged, target_date)
    n_logged = _append_to_log(log_rows, target_date)
    print(f"[daily_runner] Logged {n_logged} row(s) to {LOG_FILE}")

    report_path = _write_summary(flagged, target_date, config, n_logged)
    print(f"[daily_runner] Summary: {report_path}")

    _print_console_summary(flagged, target_date, config)


if __name__ == "__main__":
    main()
