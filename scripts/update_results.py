"""
update_results.py — Settle open bets in live_bets_log.csv

Fetches actual game outcomes and fills in result, profit, closing_odds,
and CLV for every unsettled row matching the target date.

Usage:
    # Settle yesterday's bets (default):
    python scripts/update_results.py

    # Settle a specific date:
    python scripts/update_results.py --date 2026-07-01

    # Record closing odds only (no result lookup):
    python scripts/update_results.py --date 2026-07-01 --closing-only

    # Manually set result for one bet (bypasses game-log lookup):
    python scripts/update_results.py --date 2026-07-01 --pitcher "Gerrit Cole" --actual 8

    # Record stake + NoVig fill price after placing:
    python scripts/update_results.py --date 2026-07-01 --pitcher "Gerrit Cole" --stake 100 --price-improvement 10

CLV methodology:
    clv_cents = entry_odds_american - closing_odds_american
    Positive = you beat the closing line (positive EV signal).
    Example: bet +108, close +104 → clv_cents = +4 (beat close by 4 cents).
    Example: bet -107, close -110 → clv_cents = +3 (beat close by 3 cents).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mlb_source import fetch_pitcher_game_logs

LOG_FILE     = Path("data/logs/live_bets_log.csv")
LOGS_FILE    = Path("data/raw/pitcher_game_logs.csv")
SNAPSHOT_DIR = Path("data/odds/snapshots")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _amer_to_dec(odds: float) -> float:
    odds = float(odds)
    return (1 + odds / 100) if odds >= 0 else (1 + 100 / abs(odds))


def _load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        print(f"[update_results] {LOG_FILE} not found — nothing to update.")
        sys.exit(0)
    return pd.read_csv(LOG_FILE, dtype=str)


def _save_log(df: pd.DataFrame) -> None:
    df.to_csv(LOG_FILE, index=False)
    print(f"[update_results] Saved {len(df)} rows to {LOG_FILE}")


def _normalise_name(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii").lower()


def _lookup_actual(game_logs: pd.DataFrame, game_date: str, pitcher_name: str) -> float:
    """Look up actual strikeouts from game logs. Tries exact match then last-name fuzzy."""
    for delta in range(2):  # try same day and +1 day for UTC offset
        d = (pd.Timestamp(game_date) + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
        rows = game_logs[
            (game_logs["game_date"].astype(str).str[:10] == d) &
            (game_logs["pitcher_name"].str.lower() == pitcher_name.strip().lower())
        ]
        if not rows.empty:
            return float(rows.iloc[0]["strikeouts"])

        # Fuzzy: last name only
        last = pitcher_name.strip().split()[-1].lower()
        fuzzy = game_logs[
            (game_logs["game_date"].astype(str).str[:10] == d) &
            (game_logs["pitcher_name"].str.lower().str.contains(last, na=False))
        ]
        if len(fuzzy) == 1:
            return float(fuzzy.iloc[0]["strikeouts"])
    return np.nan


def _lookup_closing_odds(
    game_date: str,
    pitcher_name: str,
    line: float,
    side: str,
) -> float | None:
    """
    Look for closing odds in data/odds/snapshots/{date}_closing.csv.
    Returns American odds or None if not found.
    """
    path = SNAPSHOT_DIR / f"{game_date}_closing.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    col = f"{side}_odds"
    if col not in df.columns:
        return None

    name_col = "pitcher_name" if "pitcher_name" in df.columns else "player_name"
    if name_col not in df.columns:
        return None

    last = pitcher_name.strip().split()[-1].lower()
    mask = (
        df[name_col].str.lower().str.contains(last, na=False) &
        (pd.to_numeric(df.get("line", pd.Series()), errors="coerce") == float(line))
    )
    matches = df[mask]
    if matches.empty:
        return None

    vals = pd.to_numeric(matches[col], errors="coerce").dropna()
    return float(vals.iloc[0]) if not vals.empty else None


# ── Core update logic ─────────────────────────────────────────────────────────

def _settle_row(
    row: pd.Series,
    game_logs: pd.DataFrame,
    actual_override: float | None = None,
    closing_override: float | None = None,
) -> dict:
    """
    Return a dict of columns to update for this row.
    Only fills fields that are currently blank.
    """
    updates: dict = {}
    game_date    = str(row.get("date", ""))
    pitcher_name = str(row.get("pitcher", ""))
    side         = str(row.get("side", ""))
    market       = str(row.get("market", "strikeouts"))

    try:
        line = float(row.get("line", ""))
    except (ValueError, TypeError):
        return updates

    # Entry odds = sportsbook odds + any price improvement recorded
    sb_odds = row.get("sportsbook_odds", "")
    impr    = row.get("price_improvement_cents", "")
    try:
        entry_odds = float(sb_odds) + float(impr) if str(impr).strip() not in ("", "nan") else float(sb_odds)
    except (ValueError, TypeError):
        entry_odds = np.nan

    # ── Actual result ──────────────────────────────────────────────────────
    result_blank = str(row.get("result", "")).strip() in ("", "nan")
    if result_blank:
        actual = actual_override if actual_override is not None else _lookup_actual(
            game_logs, game_date, pitcher_name
        )
        if pd.notna(actual) and market == "strikeouts":
            won = (actual > line) if side == "over" else (actual < line)

            # Stake might already be set
            try:
                stake = float(row.get("stake", ""))
            except (ValueError, TypeError):
                stake = np.nan

            if pd.notna(stake) and pd.notna(entry_odds):
                dec = _amer_to_dec(entry_odds)
                profit = round(stake * (dec - 1) if won else -stake, 2)
                updates["profit"] = str(profit)

            updates["result"] = str(int(actual))
            updates["won_internal"] = "1" if won else "0"  # not a log column, just for CLV

    # ── Closing odds and CLV ───────────────────────────────────────────────
    closing_blank = str(row.get("closing_odds", "")).strip() in ("", "nan")
    if closing_blank:
        closing = closing_override if closing_override is not None else _lookup_closing_odds(
            game_date, pitcher_name, line, side
        )
        if closing is not None:
            updates["closing_odds"] = str(closing)
            if pd.notna(entry_odds):
                # clv_cents = entry_american - closing_american
                # Positive = beat the closing line (positive EV signal)
                clv = int(round(entry_odds - closing))
                updates["clv_cents"] = str(clv)

    return updates


def _update_date(
    log: pd.DataFrame,
    resolve_date: str,
    game_logs: pd.DataFrame,
    actual_override: float | None = None,
    closing_override: float | None = None,
    closing_only: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Settle all open rows for resolve_date. Returns (updated_log, n_updated)."""
    mask = log["date"].astype(str) == resolve_date
    if not mask.any():
        print(f"[update_results] No rows found for {resolve_date}")
        return log, 0

    n_updated = 0
    for idx in log[mask].index:
        row = log.loc[idx]

        # Skip already-settled rows unless closing_only
        result_set = str(row.get("result", "")).strip() not in ("", "nan")
        if result_set and not closing_only:
            continue

        updates = _settle_row(
            row, game_logs,
            actual_override=actual_override if not closing_only else None,
            closing_override=closing_override,
        )
        if not updates:
            continue

        # Remove internal scratch keys before writing
        updates.pop("won_internal", None)

        for col, val in updates.items():
            if col in log.columns:
                log.at[idx, col] = val

        n_updated += 1

    return log, n_updated


def _stake_update(
    log: pd.DataFrame,
    target_date: str,
    pitcher_name: str,
    stake: float,
    market: str = "strikeouts",
    price_improvement: float | None = None,
) -> pd.DataFrame:
    """Set stake (and optionally price improvement) for a specific bet."""
    last = pitcher_name.strip().split()[-1].lower()
    mask = (
        (log["date"].astype(str) == target_date) &
        (log["pitcher"].str.lower().str.contains(last, na=False)) &
        (log["market"].astype(str).str.lower() == market.lower())
    )
    if not mask.any():
        print(f"[update_results] No matching bet found for {pitcher_name} {market} on {target_date}")
        return log
    idx = log[mask].index[-1]
    log.at[idx, "stake"] = str(stake)
    if price_improvement is not None:
        log.at[idx, "price_improvement_cents"] = str(int(price_improvement))
    print(f"[update_results] Set stake=${stake:.2f} for {log.at[idx,'pitcher']} {market} on {target_date}"
          + (f" (+{int(price_improvement)}c improvement)" if price_improvement is not None else ""))

    # Recompute profit if result already known
    result = str(log.at[idx, "result"]).strip()
    if result not in ("", "nan"):
        sb_odds = log.at[idx, "sportsbook_odds"]
        impr    = log.at[idx, "price_improvement_cents"]
        side    = str(log.at[idx, "side"])
        line    = float(log.at[idx, "line"])
        try:
            entry_odds = float(sb_odds) + float(impr) if str(impr).strip() not in ("", "nan") else float(sb_odds)
        except (ValueError, TypeError):
            entry_odds = np.nan

        if pd.notna(entry_odds):
            actual = float(result)
            won = (actual > line) if side == "over" else (actual < line)
            dec = _amer_to_dec(entry_odds)
            profit = round(stake * (dec - 1) if won else -stake, 2)
            log.at[idx, "profit"] = str(profit)

    return log


# ── Performance summary ───────────────────────────────────────────────────────

def _print_date_summary(log: pd.DataFrame, resolve_date: str) -> None:
    mask = log["date"].astype(str) == resolve_date
    day_rows = log[mask].copy()
    if day_rows.empty:
        return

    # Settled
    settled = day_rows[day_rows["result"].astype(str).str.strip().isin(
        [str(i) for i in range(100)]  # any numeric result
    )]

    print(f"\n  {resolve_date} — {len(day_rows)} bets logged, {len(settled)} settled")

    for _, row in day_rows.sort_values("edge_gap_product", errors="ignore").iterrows():
        pitcher = str(row.get("pitcher", ""))
        side    = str(row.get("side", ""))
        line    = row.get("line", "")
        result  = str(row.get("result", "—")).strip()
        bucket  = str(row.get("strategy_bucket", ""))
        profit  = str(row.get("profit", "")).strip()
        clv     = str(row.get("clv_cents", "")).strip()
        try:
            profit_str = f"${float(profit):+.2f}"
        except (ValueError, TypeError):
            profit_str = "—"
        try:
            clv_str = f"{int(float(clv)):+d}c CLV"
        except (ValueError, TypeError):
            clv_str = ""

        print(f"    {pitcher:<22} {side:<6} {line}  result={result:<3} {profit_str:>9}  [{bucket}]  {clv_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Settle open bets in live_bets_log.csv")
    parser.add_argument("--date",         default=None,
                        help="Date to settle (default: yesterday, YYYY-MM-DD)")
    parser.add_argument("--pitcher",      default=None,
                        help="Pitcher name for manual update (partial match OK)")
    parser.add_argument("--actual",       type=float, default=None,
                        help="Actual strikeout total for manual entry")
    parser.add_argument("--stake",             type=float, default=None,
                        help="Record stake for a specific bet (use with --pitcher)")
    parser.add_argument("--price-improvement", type=float, default=None,
                        help="Cents better than sportsbook you got filled at on NoVig (e.g. 10 for +10c)")
    parser.add_argument("--market",            default="strikeouts",
                        help="Market for --pitcher/--stake lookup (default: strikeouts)")
    parser.add_argument("--closing-odds", type=float, default=None,
                        help="Closing odds (American) for manual CLV entry")
    parser.add_argument("--closing-only", action="store_true",
                        help="Only update closing odds / CLV; skip result lookup")
    args = parser.parse_args()

    resolve_date = args.date or (date.today() - timedelta(days=1)).isoformat()
    print(f"[update_results] Resolving {resolve_date}...")

    log = _load_log()

    # ── Stake update (standalone) ──────────────────────────────────────────
    if args.stake is not None and args.pitcher is not None:
        log = _stake_update(log, resolve_date, args.pitcher, args.stake, args.market,
                            price_improvement=args.price_improvement)
        _save_log(log)
        _print_date_summary(log, resolve_date)
        return

    # ── Fetch game logs ────────────────────────────────────────────────────
    game_logs = pd.DataFrame()
    if not args.closing_only:
        if LOGS_FILE.exists():
            game_logs = pd.read_csv(LOGS_FILE)
            game_logs["game_date"] = game_logs["game_date"].astype(str).str[:10]
            print(f"[update_results] Loaded {len(game_logs)} rows from {LOGS_FILE}")
        else:
            print(f"[update_results] {LOGS_FILE} not found — fetching from MLB API...")
            try:
                game_logs = fetch_pitcher_game_logs(resolve_date, resolve_date)
                if not game_logs.empty:
                    game_logs.to_csv(LOGS_FILE, index=False)
                    print(f"[update_results] Fetched and saved {len(game_logs)} rows")
            except Exception as exc:
                print(f"[update_results] Could not fetch game logs: {exc}")
                print("  Provide --actual manually or ensure pitcher_game_logs.csv is up to date.")

    # ── Manual single-pitcher update ───────────────────────────────────────
    if args.pitcher is not None and args.actual is not None:
        mask = (
            (log["date"].astype(str) == resolve_date) &
            (log["pitcher"].str.lower().str.contains(
                args.pitcher.strip().split()[-1].lower(), na=False
            )) &
            (log["market"].astype(str).str.lower() == args.market.lower())
        )
        if not mask.any():
            print(f"[update_results] No matching bet for '{args.pitcher}' on {resolve_date}")
            sys.exit(1)
        idx = log[mask].index[-1]
        row = log.loc[idx]
        side = str(row.get("side", ""))
        try:
            line = float(row.get("line", ""))
        except (ValueError, TypeError):
            print("[update_results] Could not parse line value")
            sys.exit(1)

        actual = args.actual
        won    = (actual > line) if side == "over" else (actual < line)
        log.at[idx, "result"] = str(int(actual))

        # Compute profit if stake is known
        try:
            stake = float(log.at[idx, "stake"])
        except (ValueError, TypeError):
            stake = np.nan

        sb_odds = log.at[idx, "sportsbook_odds"]
        impr    = log.at[idx, "price_improvement_cents"]
        try:
            entry_odds = float(sb_odds) + float(impr) if str(impr).strip() not in ("", "nan") else float(sb_odds)
        except (ValueError, TypeError):
            entry_odds = np.nan

        if pd.notna(stake) and pd.notna(entry_odds):
            dec    = _amer_to_dec(entry_odds)
            profit = round(stake * (dec - 1) if won else -stake, 2)
            log.at[idx, "profit"] = str(profit)
            print(f"[update_results] {row['pitcher']} | actual={actual} | "
                  f"{'WON' if won else 'LOST'} | profit=${profit:+.2f}")
        else:
            print(f"[update_results] {row['pitcher']} | actual={actual} | "
                  f"{'WON' if won else 'LOST'} (stake or odds missing — profit not computed)")

        # Closing odds
        if args.closing_odds is not None and pd.notna(entry_odds):
            log.at[idx, "closing_odds"] = str(args.closing_odds)
            clv = int(round(entry_odds - args.closing_odds))
            log.at[idx, "clv_cents"]   = str(clv)
            print(f"[update_results] CLV = {clv:+d}c (entry={entry_odds:+.0f}, close={args.closing_odds:+.0f})")

        _save_log(log)
        return

    # ── Batch update for a date ────────────────────────────────────────────
    log, n_updated = _update_date(
        log, resolve_date, game_logs,
        actual_override=args.actual,
        closing_override=args.closing_odds,
        closing_only=args.closing_only,
    )
    _save_log(log)
    print(f"[update_results] Updated {n_updated} rows for {resolve_date}")
    _print_date_summary(log, resolve_date)


if __name__ == "__main__":
    main()
