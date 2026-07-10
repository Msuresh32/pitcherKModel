"""Phase 1+2: Build the clean research dataset.

Joins pitcher game logs + Statcast + context features + historical odds.
For each (pitcher, game, line), produces:
  - Pre-game features (shift-1 rolling stats, context, opponent)
  - Market no-vig P(over) at that line
  - Actual outcome: did pitcher throw > line Ks?
  - CLV: opening vs closing no-vig probability

Output: research/dataset.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loaders import (
    load_batter_game_logs,
    load_game_context_logs,
    load_park_factors,
    load_pitcher_game_logs,
    load_statcast_catcher_framing_daily,
    load_statcast_pitcher_catcher_daily,
    load_statcast_pitcher_daily,
    load_team_batting_game_logs,
)
from src.features.build_features import build_training_features

CONFIG = "config/config_v4_production.yaml"
ODDS_FILE = Path("data/odds/historical/pitcher_strikeouts_2025_2026.csv")
OUT_FILE = Path("research/dataset.parquet")
OOS_START = pd.Timestamp("2026-06-01")

# Books considered "sharp" for consensus pricing (use all if not available)
SHARP_BOOKS = {"pinnacle", "betfair", "draftkings", "fanduel"}


# ---------------------------------------------------------------------------
# Step 1: Load and validate historical odds
# ---------------------------------------------------------------------------

def _implied(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(o < 0, -o / (-o + 100), 100 / (100 + o)),
        index=odds.index,
    )


def load_odds(path: Path) -> pd.DataFrame:
    print(f"Loading odds from {path}...")
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")

    # Keep only rows where both sides exist
    df = df.dropna(subset=["over_odds", "under_odds", "line", "pitcher_id"])

    # Implied probs and no-vig
    df["p_over_raw"] = _implied(df["over_odds"])
    df["p_under_raw"] = _implied(df["under_odds"])
    df["total_implied"] = df["p_over_raw"] + df["p_under_raw"]
    df["vig"] = df["total_implied"] - 1.0
    df["p_over_novig"] = df["p_over_raw"] / df["total_implied"]

    # Sanity: remove rows with impossible vig (data errors)
    df = df[(df["vig"] > -0.02) & (df["vig"] < 0.20)]

    print(f"  Loaded {len(df):,} rows | vig mean={df.vig.mean():.3f}")
    return df


def consensus_novig(odds: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """
    For each (game_date, pitcher_id, line), compute consensus no-vig P(over)
    by averaging across all bookmakers at the given snapshot type.
    Returns columns: game_date, pitcher_id, line, p_over_novig_{snapshot}, n_books_{snapshot}
    """
    sub = odds[odds["snapshot_type"] == snapshot].copy()
    grp = (
        sub.groupby(["game_date", "pitcher_id", "line"], observed=True)
        .agg(
            p_over_novig=(  "p_over_novig", "mean"),
            n_books=(       "bookmaker",    "nunique"),
            best_over_odds=("over_odds",    "max"),
            best_under_odds=("under_odds",  "max"),
        )
        .reset_index()
        .rename(columns={
            "p_over_novig":   f"p_over_{snapshot}",
            "n_books":        f"n_books_{snapshot}",
            "best_over_odds": f"best_over_odds_{snapshot}",
            "best_under_odds":f"best_under_odds_{snapshot}",
        })
    )
    return grp


# ---------------------------------------------------------------------------
# Step 2: Build pitcher feature matrix using existing clean pipeline
# ---------------------------------------------------------------------------

def build_features(config: dict) -> pd.DataFrame:
    print("Building feature matrix...")
    logs       = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_bat   = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    ctx        = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_log = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast   = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    framing    = load_statcast_catcher_framing_daily(
                     config["data"].get("catcher_framing_file", ""))
    pc_map     = load_statcast_pitcher_catcher_daily(
                     config["data"].get("pitcher_catcher_file", ""))
    park       = load_park_factors(config["data"].get("park_factors_file", ""))

    # Build without imputation so we can apply train-only fill values
    featured, feat_cols, _ = build_training_features(
        logs,
        rolling_windows=config["features"]["rolling_windows"],
        min_history_games=config["training"]["min_history_games"],
        min_starter_ip=config["training"].get("min_starter_ip"),
        team_batting_logs=team_bat,
        game_context_logs=ctx,
        batter_game_logs=batter_log,
        statcast_pitcher_daily=statcast,
        park_factors=park,
        catcher_framing_daily=framing if not framing.empty else None,
        pitcher_catcher_map=pc_map if not pc_map.empty else None,
        return_before_impute=True,  # return NaN so we can apply train-only medians
    )

    # Compute fill values from training period ONLY (pre-OOS) to avoid leakage
    train_mask = featured["game_date"] < OOS_START
    train_raw = featured[train_mask].copy()
    fill_values = {
        col: float(train_raw[col].median())
        for col in feat_cols
        if col in train_raw.columns and train_raw[col].notna().any()
    }

    # Apply fill values to full dataset
    for col, val in fill_values.items():
        if col in featured.columns:
            featured[col] = featured[col].fillna(val)

    print(f"  Feature matrix: {len(featured):,} rows × {len(feat_cols)} features")
    return featured, feat_cols, fill_values


# ---------------------------------------------------------------------------
# Step 3: Add pitcher-specific hand-crafted features not in existing pipeline
# ---------------------------------------------------------------------------

def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features not produced by build_training_features."""
    df = df.copy()

    # Season phase (days into season from Apr 1)
    # days_rest is already in the feature matrix from build_training_features
    april1 = pd.to_datetime(df["game_date"].dt.year.astype(str) + "-04-01")
    df["days_into_season"] = (df["game_date"] - april1).dt.days.clip(0, 200)

    # Month (proxy for in-season fatigue and pitch mix trends)
    df["month"] = df["game_date"].dt.month  # 4=Apr ... 9=Sep

    return df


# ---------------------------------------------------------------------------
# Step 4: Join features + odds + compute outcomes
# ---------------------------------------------------------------------------

def build_research_dataset(features: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one row per (pitcher, game, line).
    Target: actual_Ks > line (binary).
    """
    print("Building research dataset...")

    # Consensus no-vig for open and close snapshots
    open_prices  = consensus_novig(odds, "open")
    close_prices = consensus_novig(odds, "close")

    # Merge open and close on same key
    market = open_prices.merge(
        close_prices,
        on=["game_date", "pitcher_id", "line"],
        how="outer",
    )

    # Cast keys
    features = features.copy()
    features["game_date"] = pd.to_datetime(features["game_date"])
    features["pitcher_id"] = pd.to_numeric(features["pitcher_id"], errors="coerce")
    market["game_date"]   = pd.to_datetime(market["game_date"])
    market["pitcher_id"]  = pd.to_numeric(market["pitcher_id"], errors="coerce")

    # Join on (game_date, pitcher_id)
    dataset = market.merge(
        features,
        on=["game_date", "pitcher_id"],
        how="inner",
    )
    print(f"  After join: {len(dataset):,} rows")

    # Filter to starter-relevant lines only (3.5–9.5)
    # Lines below 3.5 are typically for relievers; above 9.5 are near-meaningless bets
    dataset = dataset[(dataset["line"] >= 3.5) & (dataset["line"] <= 9.5)].copy()

    # Outcome: actual Ks vs line
    dataset["actual_ks"] = pd.to_numeric(dataset["strikeouts"], errors="coerce")
    dataset["outcome_over"] = (dataset["actual_ks"] > dataset["line"]).astype(float)
    dataset["outcome_push"] = (dataset["actual_ks"] == dataset["line"]).astype(float)
    # Exclude pushes from training/evaluation
    dataset = dataset[dataset["outcome_push"] == 0].copy()

    # CLV: did closing line move in our direction vs opening?
    # positive CLV (from over perspective) = close P(over) > open P(over)
    dataset["clv_over"] = dataset["p_over_close"] - dataset["p_over_open"]

    print(f"  Final dataset: {len(dataset):,} rows (pushes excluded)")
    print(f"  Date range: {dataset.game_date.min().date()} to {dataset.game_date.max().date()}")
    print(f"  Lines: {sorted(dataset.line.dropna().unique())}")
    print(f"  Actual over rate: {dataset.outcome_over.mean():.3f}")
    print(f"  Market P(over) mean [close]: {dataset.p_over_close.mean():.3f}")

    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not ODDS_FILE.exists():
        print(f"ERROR: Odds file not found: {ODDS_FILE}")
        print("Run scripts/fetch_historical_odds.py first.")
        return

    config = load_config(CONFIG)
    odds = load_odds(ODDS_FILE)

    features, feat_cols, fill_values = build_features(config)
    features = add_research_features(features)

    dataset = build_research_dataset(features, odds)

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved dataset to {OUT_FILE}")
    print(f"Feature columns: {len(feat_cols)}")

    # Save feature list and fill values for downstream use
    import json
    meta = {"feature_cols": feat_cols, "fill_values": {k: v for k, v in fill_values.items()}}
    Path("research/dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved research/dataset_meta.json")


if __name__ == "__main__":
    main()
