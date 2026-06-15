from __future__ import annotations

from math import floor, log

import numpy as np
import pandas as pd

from src.data.schema import TARGETS
from src.odds.pricing import add_betting_columns, american_to_decimal


# ---------------------------------------------------------------------------
# Prediction scoring (point estimates)
# ---------------------------------------------------------------------------

def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        pred_col = f"{target}_projection"
        if pred_col not in predictions:
            continue
        err = predictions[target] - predictions[pred_col]
        rows.append(
            {
                "market": target,
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "rows": int(err.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probability scoring (requires odds lines to determine over/under threshold)
# ---------------------------------------------------------------------------

def _brier(prob: float, outcome: int) -> float:
    return (float(prob) - float(outcome)) ** 2


def _log_loss_single(prob: float, outcome: int, eps: float = 1e-7) -> float:
    p = max(min(float(prob), 1 - eps), eps)
    return -(float(outcome) * log(p) + (1 - float(outcome)) * log(1 - p))


def score_probability_calibration(scored: pd.DataFrame) -> pd.DataFrame:
    """Compute Brier score and log loss per market.

    Requires 'over_probability', 'line', 'market', and the actual stat column.
    Only rows where we have a real sportsbook line are included.
    """
    rows = []
    for market in TARGETS:
        sub = scored[scored["market"] == market].copy()
        if sub.empty or "over_probability" not in sub.columns:
            continue
        sub = sub.dropna(subset=["over_probability", "line", market])
        if sub.empty:
            continue
        # Outcome: 1 if actual > line (over wins)
        sub["_outcome"] = (sub[market] > sub["line"]).astype(int)
        brier = sub.apply(lambda r: _brier(r["over_probability"], r["_outcome"]), axis=1).mean()
        ll = sub.apply(lambda r: _log_loss_single(r["over_probability"], r["_outcome"]), axis=1).mean()
        hit_rate = float(sub["_outcome"].mean())
        mean_prob = float(sub["over_probability"].mean())
        rows.append(
            {
                "market": market,
                "brier_score": float(brier),
                "log_loss": float(ll),
                "mean_over_probability": mean_prob,
                "actual_over_rate": hit_rate,
                "calibration_gap": round(mean_prob - hit_rate, 4),
                "rows": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Edge-bucket segmentation
# ---------------------------------------------------------------------------

def compute_edge_bucket_stats(scored: pd.DataFrame) -> pd.DataFrame:
    """Break rows with actual outcomes into edge % buckets and show hit rate + ROI."""
    if "edge_pct" not in scored.columns:
        return pd.DataFrame()

    buckets = [
        ("negative", -999, 0),
        ("0-3%", 0, 3),
        ("3-7%", 3, 7),
        ("7-12%", 7, 12),
        ("12%+", 12, 999),
    ]
    rows = []
    for market in TARGETS:
        sub = scored[scored["market"] == market].copy()
        if sub.empty:
            continue
        # Resolve actual result column
        actual_col = market
        if actual_col not in sub.columns:
            continue
        sub = sub.dropna(subset=["edge_pct", "line", actual_col, "best_side"])
        if sub.empty:
            continue
        sub["_won"] = np.where(
            sub["best_side"] == "over",
            sub[actual_col] > sub["line"],
            sub[actual_col] < sub["line"],
        )
        odds_col = np.where(sub["best_side"] == "over", sub.get("over_odds", np.nan), sub.get("under_odds", np.nan))
        decimal = np.where(odds_col > 0, 1 + odds_col / 100, 1 + 100 / np.abs(np.where(odds_col == 0, 1, odds_col)))
        sub["_profit"] = np.where(sub["_won"], decimal - 1, -1.0)

        for label, lo, hi in buckets:
            bucket = sub[(sub["edge_pct"] >= lo) & (sub["edge_pct"] < hi)]
            if bucket.empty:
                continue
            rows.append(
                {
                    "market": market,
                    "edge_bucket": label,
                    "bets": int(len(bucket)),
                    "win_rate": float(bucket["_won"].mean()),
                    "roi": float(bucket["_profit"].mean()),
                    "mean_edge_pct": float(bucket["edge_pct"].mean()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sequential P&L, drawdown, Sharpe
# ---------------------------------------------------------------------------

def compute_sequential_pnl(
    bets: pd.DataFrame,
    bankroll: float,
) -> pd.DataFrame:
    """Compute bet-by-bet P&L sorted by game_date.

    Returns a row-per-bet DataFrame with running bankroll, max drawdown, and Sharpe ratio
    appended as summary rows at the bottom.
    """
    if bets.empty or "game_date" not in bets.columns:
        return pd.DataFrame()

    b = bets.sort_values("game_date").copy()
    b = b.dropna(subset=["kelly_fraction", "best_side"])

    # Resolve actual result
    actual_vals = []
    for _, row in b.iterrows():
        mkt = row.get("market")
        if mkt and mkt in row.index and pd.notna(row[mkt]):
            actual_vals.append(row[mkt])
        else:
            actual_vals.append(np.nan)
    b["_actual"] = actual_vals

    b = b.dropna(subset=["_actual", "line"])
    if b.empty:
        return pd.DataFrame()

    odds_series = np.where(
        b["best_side"] == "over",
        pd.to_numeric(b.get("over_odds", pd.Series(dtype=float)), errors="coerce"),
        pd.to_numeric(b.get("under_odds", pd.Series(dtype=float)), errors="coerce"),
    )
    dec = np.where(odds_series > 0, 1 + odds_series / 100, 1 + 100 / np.abs(np.where(odds_series == 0, 1, odds_series)))

    won = np.where(
        b["best_side"] == "over",
        b["_actual"] > b["line"],
        b["_actual"] < b["line"],
    ).astype(bool)

    stake = bankroll * b["kelly_fraction"].values
    profit = np.where(won, stake * (dec - 1), -stake)

    b["stake"] = stake
    b["won"] = won
    b["profit"] = profit
    b["cumulative_profit"] = profit.cumsum()
    b["running_bankroll"] = bankroll + b["cumulative_profit"]
    b["peak_bankroll"] = b["running_bankroll"].cummax()
    b["drawdown"] = b["running_bankroll"] - b["peak_bankroll"]

    max_drawdown = float(b["drawdown"].min())
    total_staked = float(stake.sum())
    total_profit = float(profit.sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    # Daily Sharpe (annualised assuming ~162 game days/year)
    daily_returns = b.groupby("game_date")["profit"].sum() / bankroll
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(162))
        if daily_returns.std() > 0
        else 0.0
    )

    summary_cols = [
        "game_date", "market", "best_side", "line", "stake", "won",
        "profit", "cumulative_profit", "running_bankroll", "drawdown",
    ]
    out = b[[c for c in summary_cols if c in b.columns]].copy()
    out.attrs["max_drawdown"] = max_drawdown
    out.attrs["roi"] = roi
    out.attrs["sharpe"] = sharpe
    out.attrs["total_bets"] = int(len(b))
    out.attrs["win_rate"] = float(won.mean())
    return out


# ---------------------------------------------------------------------------
# CLV computation
# ---------------------------------------------------------------------------

def compute_clv(
    bets: pd.DataFrame,
    close_odds: pd.DataFrame,
) -> pd.DataFrame:
    """Attach closing line value to a bets DataFrame.

    close_odds must have columns: game_date, pitcher_id, market, over_odds, under_odds.
    CLV% = (entry_decimal / close_decimal - 1) * 100.
    Positive CLV means you got a better number than the closing line.
    """
    if close_odds.empty or bets.empty:
        return bets

    # Match on line too — a bet at 5.5 should only compare to the closing 5.5 line
    key = ["game_date", "pitcher_id", "market", "line"]
    close = close_odds[
        [c for c in key if c in close_odds.columns] + ["over_odds", "under_odds"]
    ].copy()
    close = close.rename(columns={"over_odds": "_close_over", "under_odds": "_close_under"})

    merged = bets.merge(close, on=[c for c in key if c in close.columns], how="left")

    def _clv_pct(row: pd.Series) -> float:
        side = row.get("best_side")
        entry_odds = row.get("over_odds") if side == "over" else row.get("under_odds")
        close_val = row.get("_close_over") if side == "over" else row.get("_close_under")
        if pd.isna(entry_odds) or pd.isna(close_val):
            return np.nan
        entry_dec = american_to_decimal(entry_odds)
        close_dec = american_to_decimal(close_val)
        if pd.isna(close_dec) or close_dec <= 1:
            return np.nan
        return (entry_dec / close_dec - 1) * 100

    merged["clv_pct"] = merged.apply(_clv_pct, axis=1)
    return merged.drop(columns=["_close_over", "_close_under"], errors="ignore")


# ---------------------------------------------------------------------------
# Odds attachment and bet summary (updated)
# ---------------------------------------------------------------------------

def attach_odds_and_edges(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    residual_std: dict[str, float],
    max_kelly_fraction: float,
    edge_shrink_factor: float = 1.0,
    distribution: str | dict = "normal",
    bias_corrections: dict[str, float] | None = None,
    disabled_markets: list[str] | None = None,
) -> pd.DataFrame:
    if odds.empty:
        return predictions

    disabled = set(disabled_markets or [])
    pieces = []
    key_cols = ["game_date", "pitcher_id"]
    for market in TARGETS:
        if market in disabled:
            continue
        market_odds = odds[odds["market"] == market].copy()
        if market_odds.empty:
            continue
        merged = predictions.merge(market_odds, on=key_cols, how="inner")
        if merged.empty:
            continue
        merged["market"] = market
        dist = distribution.get(market, "normal") if isinstance(distribution, dict) else distribution
        bias = (bias_corrections or {}).get(market, 0.0)
        pieces.append(
            add_betting_columns(
                merged,
                market=market,
                residual_std=residual_std[market],
                max_kelly_fraction=max_kelly_fraction,
                edge_shrink_factor=edge_shrink_factor,
                distribution=dist,
                bias_correction=bias,
            )
        )

    if not pieces:
        return predictions
    return pd.concat(pieces, ignore_index=True, sort=False)


def summarize_bets(
    scored: pd.DataFrame,
    bankroll: float,
    min_edge_pct: float,
) -> pd.DataFrame:
    if "edge_pct" not in scored:
        return pd.DataFrame()
    bets = scored[scored["edge_pct"] >= min_edge_pct].copy()
    if bets.empty:
        return pd.DataFrame()

    # Resolve actual result per market row
    actual_by_market = [
        bets.loc[bets["market"] == market, market]
        for market in TARGETS
        if market in bets.columns
    ]
    if not actual_by_market:
        return pd.DataFrame()
    bets["actual_result"] = pd.concat(actual_by_market).sort_index()
    actual = np.where(
        bets["best_side"] == "over",
        bets["actual_result"] > bets["line"],
        bets["actual_result"] < bets["line"],
    )
    bets["stake"] = bankroll * bets["kelly_fraction"]
    bets["won"] = actual
    odds_col = np.where(bets["best_side"] == "over", bets["over_odds"], bets["under_odds"])
    decimal = np.where(odds_col > 0, 1 + odds_col / 100, 1 + 100 / np.abs(odds_col))
    bets["profit"] = np.where(bets["won"], bets["stake"] * (decimal - 1), -bets["stake"])

    total_staked = float(bets["stake"].sum())
    total_profit = float(bets["profit"].sum())

    # Drawdown
    bets_sorted = bets.sort_values("game_date") if "game_date" in bets.columns else bets
    cum_profit = bets_sorted["profit"].cumsum()
    running_bk = bankroll + cum_profit
    peak = running_bk.cummax()
    drawdown = running_bk - peak
    max_drawdown = float(drawdown.min())

    # Sharpe (annualised over ~162 game days)
    if "game_date" in bets_sorted.columns:
        daily_ret = bets_sorted.groupby("game_date")["profit"].sum() / bankroll
        sharpe = (
            float(daily_ret.mean() / daily_ret.std() * np.sqrt(162))
            if daily_ret.std() > 0
            else 0.0
        )
    else:
        sharpe = 0.0

    return pd.DataFrame(
        [
            {
                "bets": int(len(bets)),
                "win_rate": float(bets["won"].mean()),
                "staked": total_staked,
                "profit": total_profit,
                "roi": total_profit / total_staked if total_staked > 0 else 0.0,
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown / bankroll,
                "sharpe": sharpe,
            }
        ]
    )
