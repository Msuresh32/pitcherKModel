"""V2 research pipeline — Phase 1: build clean datasets.

Splits (mandated, DO NOT CHANGE):
  Train:        2022-2024
  Validation:   2025 season
  Forward test: 2026 opening day .. 2026-07-10 (untouched until the end)

Leakage rules enforced here:
  - Imputation fill values computed from TRAIN YEARS (2022-2024) ONLY.
  - Odds join uses OPEN snapshot for anything decision-relevant.
    CLOSE snapshot is retained ONLY for CLV scoring.
  - One row per (pitcher, game, line); pushes flagged, kept but excluded downstream.

Outputs:
  research/v2/features_full.parquet  - full feature matrix 2022-2026 (per pitcher-game)
  research/v2/bets.parquet           - odds-joined rows 2025-2026 (per pitcher-game-line)
  research/v2/meta.json              - feature list + fill values
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
OUT_DIR = Path("research/v2")

TRAIN_END = pd.Timestamp("2025-01-01")   # fill values use data strictly before this


def _implied(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(
        np.where(o < 0, -o / (-o + 100), 100 / (100 + o)),
        index=odds.index,
    )


def load_odds(path: Path) -> pd.DataFrame:
    print(f"Loading odds from {path}...", flush=True)
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")
    df = df.dropna(subset=["over_odds", "under_odds", "line", "pitcher_id"])

    df["p_over_raw"] = _implied(df["over_odds"])
    df["p_under_raw"] = _implied(df["under_odds"])
    df["total_implied"] = df["p_over_raw"] + df["p_under_raw"]
    df["vig"] = df["total_implied"] - 1.0
    df["p_over_novig"] = df["p_over_raw"] / df["total_implied"]
    df = df[(df["vig"] > -0.02) & (df["vig"] < 0.20)]
    print(f"  {len(df):,} odds rows | vig mean={df.vig.mean():.3f}", flush=True)
    return df


def consensus(odds: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    sub = odds[odds["snapshot_type"] == snapshot].copy()
    grp = (
        sub.groupby(["game_date", "pitcher_id", "line"], observed=True)
        .agg(
            p_over=("p_over_novig", "mean"),
            n_books=("bookmaker", "nunique"),
            best_over_odds=("over_odds", "max"),
            best_under_odds=("under_odds", "max"),
            med_over_odds=("over_odds", "median"),
            med_under_odds=("under_odds", "median"),
        )
        .reset_index()
    )
    grp.columns = ["game_date", "pitcher_id", "line"] + [
        f"{c}_{snapshot}" for c in
        ["p_over", "n_books", "best_over_odds", "best_under_odds",
         "med_over_odds", "med_under_odds"]
    ]
    return grp


def best_book_at_open(odds: pd.DataFrame) -> pd.DataFrame:
    """Which book offered the best over/under price at open (for by-book reporting)."""
    sub = odds[odds["snapshot_type"] == "open"].copy()
    io = sub.groupby(["game_date", "pitcher_id", "line"])["over_odds"].idxmax()
    iu = sub.groupby(["game_date", "pitcher_id", "line"])["under_odds"].idxmax()
    bo = sub.loc[io, ["game_date", "pitcher_id", "line", "bookmaker"]].rename(
        columns={"bookmaker": "best_over_book_open"})
    bu = sub.loc[iu, ["game_date", "pitcher_id", "line", "bookmaker"]].rename(
        columns={"bookmaker": "best_under_book_open"})
    return bo.merge(bu, on=["game_date", "pitcher_id", "line"], how="outer")


def main():
    config = load_config(CONFIG)

    print("Building feature matrix (this can take several minutes)...", flush=True)
    logs = load_pitcher_game_logs(config["data"]["pitcher_logs_file"])
    team_bat = load_team_batting_game_logs(config["data"]["team_batting_logs_file"])
    ctx = load_game_context_logs(config["data"]["game_context_logs_file"])
    batter_log = load_batter_game_logs(config["data"]["batter_game_logs_file"])
    statcast = load_statcast_pitcher_daily(config["data"]["statcast_pitcher_daily_file"])
    framing = load_statcast_catcher_framing_daily(config["data"].get("catcher_framing_file", ""))
    pc_map = load_statcast_pitcher_catcher_daily(config["data"].get("pitcher_catcher_file", ""))

    # DATA QUALITY FIX (2026-07-11): statcast_pitcher_daily.csv and the framing
    # file contain every row exactly twice (append bug in the fetch task).
    # Duplicated rows corrupt all shift(1)-rolling windows downstream.
    for _name, _df in [("statcast", statcast), ("framing", framing), ("pc_map", pc_map)]:
        before = len(_df)
        _df.drop_duplicates(inplace=True)
        if len(_df) < before:
            print(f"  DEDUP {_name}: {before:,} -> {len(_df):,} rows", flush=True)
    park = load_park_factors(config["data"].get("park_factors_file", ""))

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
        return_before_impute=True,
    )
    featured["game_date"] = pd.to_datetime(featured["game_date"])

    # ---- Train-only imputation (2022-2024) ----
    train_mask = featured["game_date"] < TRAIN_END
    train_raw = featured[train_mask]
    fill_values = {
        c: float(train_raw[c].median())
        for c in feat_cols
        if c in train_raw.columns and train_raw[c].notna().any()
    }
    # Track missingness BEFORE imputation (candidate model feature, honest pre-game info)
    featured["n_missing_feats"] = featured[feat_cols].isna().sum(axis=1)
    for c, v in fill_values.items():
        featured[c] = featured[c].fillna(v)

    # Season-phase extras
    april1 = pd.to_datetime(featured["game_date"].dt.year.astype(str) + "-04-01")
    featured["days_into_season"] = (featured["game_date"] - april1).dt.days.clip(0, 200)
    featured["month"] = featured["game_date"].dt.month
    extra = ["days_into_season", "month", "n_missing_feats"]
    feat_cols = feat_cols + [c for c in extra if c not in feat_cols]

    print(f"  Feature matrix: {len(featured):,} rows x {len(feat_cols)} features", flush=True)
    print(featured.groupby(featured.game_date.dt.year).size(), flush=True)

    keep_cols = list(dict.fromkeys(
        ["game_date", "pitcher_id", "pitcher_name", "team", "opponent", "is_home",
         "strikeouts", "innings_pitched", "batters_faced"] + feat_cols
    ))
    keep_cols = [c for c in keep_cols if c in featured.columns]
    featured[keep_cols].to_parquet(OUT_DIR / "features_full.parquet", index=False)

    # ---- Odds join ----
    odds = load_odds(ODDS_FILE)
    open_p = consensus(odds, "open")
    close_p = consensus(odds, "close")
    market = open_p.merge(close_p, on=["game_date", "pitcher_id", "line"], how="outer")
    books = best_book_at_open(odds)
    market = market.merge(books, on=["game_date", "pitcher_id", "line"], how="left")

    feats = featured[keep_cols].copy()
    feats["pitcher_id"] = pd.to_numeric(feats["pitcher_id"], errors="coerce")
    market["game_date"] = pd.to_datetime(market["game_date"])

    bets = market.merge(feats, on=["game_date", "pitcher_id"], how="inner")
    bets = bets[(bets["line"] >= 3.5) & (bets["line"] <= 9.5)].copy()
    bets["actual_ks"] = pd.to_numeric(bets["strikeouts"], errors="coerce")
    bets = bets.dropna(subset=["actual_ks"])
    bets["outcome_over"] = (bets["actual_ks"] > bets["line"]).astype(float)
    bets["outcome_push"] = (bets["actual_ks"] == bets["line"]).astype(float)
    bets["clv_over"] = bets["p_over_close"] - bets["p_over_open"]

    print(f"  Bets dataset: {len(bets):,} rows "
          f"({bets.game_date.min().date()} .. {bets.game_date.max().date()})", flush=True)
    print(bets.groupby(bets.game_date.dt.year).size(), flush=True)
    bets.to_parquet(OUT_DIR / "bets.parquet", index=False)

    meta = {"feature_cols": feat_cols, "fill_values": fill_values,
            "train_end_for_impute": str(TRAIN_END.date())}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
