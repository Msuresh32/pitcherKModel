from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import (
    FANGRAPHS_COLUMNS,
    OPTIONAL_LOG_COLUMNS,
    PROBABLE_PITCHER_COLUMNS,
    REQUIRED_BATTER_GAME_COLUMNS,
    REQUIRED_GAME_CONTEXT_COLUMNS,
    REQUIRED_LOG_COLUMNS,
    REQUIRED_ODDS_COLUMNS,
    REQUIRED_PARK_FACTOR_COLUMNS,
    REQUIRED_STATCAST_PITCHER_DAILY_COLUMNS,
    REQUIRED_STATCAST_BATTER_PITCH_TYPE_DAILY_COLUMNS,
    REQUIRED_TEAM_BATTING_COLUMNS,
)


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_pitcher_game_logs(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    _require_columns(df, REQUIRED_LOG_COLUMNS, "pitcher game logs")
    for col in OPTIONAL_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["is_home"] = df["is_home"].astype(int)
    return df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def load_team_batting_game_logs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_TEAM_BATTING_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_TEAM_BATTING_COLUMNS, "team batting game logs")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["team"] = df["team"].astype(str)
    df["opponent"] = df["opponent"].astype(str)
    df["is_home"] = df["is_home"].astype(int)
    return df.sort_values(["team", "game_date"]).reset_index(drop=True)


def load_batter_game_logs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_BATTER_GAME_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_BATTER_GAME_COLUMNS, "batter game logs")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["batter_id"] = df["batter_id"].astype(str)
    df["team"] = df["team"].astype(str)
    df["opponent"] = df["opponent"].astype(str)
    df["is_home"] = df["is_home"].astype(int)
    return df.sort_values(["batter_id", "game_date"]).reset_index(drop=True)


def load_statcast_pitcher_daily(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_STATCAST_PITCHER_DAILY_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_STATCAST_PITCHER_DAILY_COLUMNS, "statcast pitcher daily")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    return df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def load_statcast_batter_pitch_type_daily(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_STATCAST_BATTER_PITCH_TYPE_DAILY_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_STATCAST_BATTER_PITCH_TYPE_DAILY_COLUMNS, "statcast batter pitch type daily")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["batter_id"] = df["batter_id"].astype(str)
    return df.sort_values(["batter_id", "game_date"]).reset_index(drop=True)


def load_park_factors(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_PARK_FACTOR_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_PARK_FACTOR_COLUMNS, "park factors")
    df["factor_year"] = df["factor_year"].astype(int)
    df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce")
    return df.sort_values(["venue_id", "factor_year"]).reset_index(drop=True)


def load_game_context_logs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_GAME_CONTEXT_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_GAME_CONTEXT_COLUMNS, "game context logs")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["team"] = df["team"].astype(str)
    df["opponent"] = df["opponent"].astype(str)
    return df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def load_probable_pitchers(path: str | Path, game_date: str | None = None) -> pd.DataFrame:
    df = _read_csv(path)
    _require_columns(df, PROBABLE_PITCHER_COLUMNS, "probable pitchers")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["is_home"] = df["is_home"].astype(int)
    if game_date:
        df = df[df["game_date"] == pd.to_datetime(game_date)]
    return df.reset_index(drop=True)


def load_odds(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_ODDS_COLUMNS)
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_ODDS_COLUMNS, "odds")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64").astype(str)
    return df


def load_fangraphs_stats(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FANGRAPHS_COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["season"] = df["season"].astype(int)
    return df


def load_statcast_pitcher_advanced(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    return df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def load_statcast_batter_discipline(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["batter_id"] = df["batter_id"].astype(str)
    return df.sort_values(["batter_id", "game_date"]).reset_index(drop=True)


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df[(df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)].copy()
