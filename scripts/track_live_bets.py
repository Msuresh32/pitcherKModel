"""Live bet tracker for NoVig exchange.

Log each bet as you place it, then record results as games finish.
Tracks ROI, CLV, Sharpe, drawdown in real time.

Usage:
    python scripts/track_live_bets.py log     --date 2026-06-02 --pitcher "Gerrit Cole" \
        --market strikeouts --line 7.5 --side over --odds +108 --stake 25
    python scripts/track_live_bets.py result  --date 2026-06-02 --pitcher "Gerrit Cole" \
        --market strikeouts --actual 9
    python scripts/track_live_bets.py summary
    python scripts/track_live_bets.py summary --since 2026-04-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BETS_FILE = Path("data/live/live_bets.csv")

COLUMNS = [
    "logged_at", "game_date", "pitcher_name", "market", "line", "side",
    "odds", "stake", "model_projection", "model_edge_pct",
    "actual_result", "won", "profit", "clv_pct", "notes",
]


def _load() -> pd.DataFrame:
    if not BETS_FILE.exists():
        BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(BETS_FILE, index=False)
        return df
    return pd.read_csv(BETS_FILE)


def _save(df: pd.DataFrame) -> None:
    BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BETS_FILE, index=False)


def american_to_decimal(odds: float) -> float:
    return (1 + odds / 100) if odds >= 100 else (1 + 100 / abs(odds))


def cmd_log(args) -> None:
    df = _load()
    odds   = float(args.odds)
    stake  = float(args.stake)
    new_row = {
        "logged_at":        pd.Timestamp.now().isoformat(timespec="seconds"),
        "game_date":        args.date,
        "pitcher_name":     args.pitcher,
        "market":           args.market,
        "line":             float(args.line),
        "side":             args.side.lower(),
        "odds":             odds,
        "stake":            stake,
        "model_projection": float(args.projection) if args.projection else None,
        "model_edge_pct":   float(args.edge) if args.edge else None,
        "actual_result":    None,
        "won":              None,
        "profit":           None,
        "clv_pct":          float(args.clv) if args.clv else None,
        "notes":            args.notes or "",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save(df)
    print(f"Logged: {args.pitcher} | {args.market} {args.side} {args.line} @ {args.odds:+.0f} | stake ${stake:.2f}")


def cmd_result(args) -> None:
    df = _load()
    actual = float(args.actual)
    mask = (
        (df["game_date"].astype(str) == str(args.date)) &
        (df["pitcher_name"].str.contains(args.pitcher, case=False, na=False)) &
        (df["market"] == args.market) &
        (df["actual_result"].isna())
    )
    if not mask.any():
        print(f"No matching open bet found for {args.pitcher} {args.market} on {args.date}")
        return

    idx = df[mask].index[-1]
    row = df.loc[idx]
    line = float(row["line"])
    side = str(row["side"])
    odds = float(row["odds"])
    stake = float(row["stake"])

    won = (actual > line) if side == "over" else (actual < line)
    dec = american_to_decimal(odds)
    profit = stake * (dec - 1) if won else -stake

    df.loc[idx, "actual_result"] = actual
    df.loc[idx, "won"]           = int(won)
    df.loc[idx, "profit"]        = round(profit, 2)
    _save(df)

    result_str = "WON" if won else "LOST"
    print(f"Result: {row['pitcher_name']} {row['market']} {side} {line} | "
          f"actual={actual} | {result_str} | profit=${profit:+.2f}")


def cmd_clv(args) -> None:
    """Record closing-line value after game closes."""
    df = _load()
    mask = (
        (df["game_date"].astype(str) == str(args.date)) &
        (df["pitcher_name"].str.contains(args.pitcher, case=False, na=False)) &
        (df["market"] == args.market)
    )
    if not mask.any():
        print("No matching bet found.")
        return
    idx = df[mask].index[-1]
    entry_dec = american_to_decimal(float(df.loc[idx, "odds"]))
    close_dec = american_to_decimal(float(args.close_odds))
    clv = (entry_dec / close_dec - 1) * 100
    df.loc[idx, "clv_pct"] = round(clv, 3)
    _save(df)
    print(f"CLV recorded: entry={df.loc[idx,'odds']:+.0f} close={args.close_odds:+.0f} CLV={clv:+.2f}%")


def cmd_summary(args) -> None:
    df = _load()
    if df.empty:
        print("No bets logged yet.")
        return

    df["game_date"] = pd.to_datetime(df["game_date"])
    if args.since:
        df = df[df["game_date"] >= pd.Timestamp(args.since)]

    settled = df[df["won"].notna()].copy()
    open_bets = df[df["won"].isna()]

    print("=" * 60)
    print("  LIVE BET TRACKER SUMMARY")
    print("=" * 60)
    print(f"\n  Total logged: {len(df)}  |  Settled: {len(settled)}  |  Open: {len(open_bets)}")

    if settled.empty:
        print("  No settled bets yet.")
        return

    total_stake  = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    roi          = total_profit / total_stake if total_stake > 0 else 0.0
    win_rate     = settled["won"].mean()

    print(f"\n  Win rate:     {win_rate:.1%}")
    print(f"  Total staked: ${total_stake:.2f}")
    print(f"  Total profit: ${total_profit:+.2f}")
    print(f"  ROI:          {roi:+.2%}")

    # Drawdown
    settled_sorted = settled.sort_values("game_date")
    cum_profit  = settled_sorted["profit"].cumsum()
    peak        = cum_profit.cummax()
    drawdown    = cum_profit - peak
    max_dd      = drawdown.min()
    print(f"  Max drawdown: ${max_dd:.2f}")

    # CLV
    clv_vals = settled["clv_pct"].dropna()
    if len(clv_vals) > 0:
        print(f"\n  CLV coverage: {len(clv_vals)}/{len(settled)} bets")
        print(f"  Mean CLV:     {clv_vals.mean():+.2f}%")
        pos_clv = settled[settled["clv_pct"] > 0]
        neg_clv = settled[settled["clv_pct"] <= 0]
        if len(pos_clv) > 0:
            pos_roi = pos_clv["profit"].sum() / pos_clv["stake"].sum()
            print(f"  Pos CLV bets: {len(pos_clv)}  ROI={pos_roi:+.2%}")
        if len(neg_clv) > 0:
            neg_roi = neg_clv["profit"].sum() / neg_clv["stake"].sum()
            print(f"  Neg CLV bets: {len(neg_clv)}  ROI={neg_roi:+.2%}")

    # By market
    print(f"\n  {'Market':>12}  {'N':>4}  {'Win%':>6}  {'ROI':>8}")
    for mkt, grp in settled.groupby("market"):
        mkt_roi = grp["profit"].sum() / grp["stake"].sum() if grp["stake"].sum() > 0 else 0
        print(f"  {mkt:>12}  {len(grp):>4}  {grp['won'].mean()*100:>5.1f}%  {mkt_roi:>+7.2%}")

    # Recent bets table
    print(f"\n  Recent settled bets:")
    cols = ["game_date", "pitcher_name", "market", "side", "line", "odds", "stake", "actual_result", "profit", "clv_pct"]
    cols = [c for c in cols if c in settled.columns]
    print(settled.sort_values("game_date", ascending=False).head(10)[cols].to_string(index=False))

    if not open_bets.empty:
        print(f"\n  Open bets ({len(open_bets)}):")
        open_cols = ["game_date", "pitcher_name", "market", "side", "line", "odds", "stake"]
        print(open_bets[open_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live bet tracker")
    sub = parser.add_subparsers(dest="command")

    # log
    p_log = sub.add_parser("log", help="Log a new bet")
    p_log.add_argument("--date",       required=True)
    p_log.add_argument("--pitcher",    required=True)
    p_log.add_argument("--market",     required=True, choices=["strikeouts", "walks", "hits_allowed"])
    p_log.add_argument("--line",       required=True, type=float)
    p_log.add_argument("--side",       required=True, choices=["over", "under"])
    p_log.add_argument("--odds",       required=True, type=float, help="American odds e.g. +108 or -112")
    p_log.add_argument("--stake",      required=True, type=float, help="Dollar amount")
    p_log.add_argument("--projection", type=float, default=None, help="Model projection")
    p_log.add_argument("--edge",       type=float, default=None, help="Model edge %")
    p_log.add_argument("--clv",        type=float, default=None, help="CLV % if already known")
    p_log.add_argument("--notes",      default="")

    # result
    p_res = sub.add_parser("result", help="Record game result")
    p_res.add_argument("--date",     required=True)
    p_res.add_argument("--pitcher",  required=True)
    p_res.add_argument("--market",   required=True)
    p_res.add_argument("--actual",   required=True, type=float, help="Actual stat value")

    # clv
    p_clv = sub.add_parser("clv", help="Record closing line value")
    p_clv.add_argument("--date",        required=True)
    p_clv.add_argument("--pitcher",     required=True)
    p_clv.add_argument("--market",      required=True)
    p_clv.add_argument("--close-odds",  required=True, type=float)

    # summary
    p_sum = sub.add_parser("summary", help="Show performance summary")
    p_sum.add_argument("--since", default=None, help="Filter bets since this date")

    args = parser.parse_args()

    if args.command == "log":
        cmd_log(args)
    elif args.command == "result":
        cmd_result(args)
    elif args.command == "clv":
        cmd_clv(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
