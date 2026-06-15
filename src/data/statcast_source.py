from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import (
    REQUIRED_STATCAST_BATTER_PITCH_TYPE_DAILY_COLUMNS,
    REQUIRED_STATCAST_PITCHER_DAILY_COLUMNS,
)

FASTBALLS = {"FF", "FA", "FC", "SI", "FS"}
SLIDERS = {"SL", "ST"}
BREAKING = {"CU", "KC", "CS", "SV", "SL", "ST"}
OFFSPEED = {"CH", "FS", "FO", "SC"}
CALLED_STRIKES = {"called_strike"}
SWINGING_STRIKES = {"swinging_strike", "swinging_strike_blocked"}
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}


def _rate(mask: pd.Series, denom: pd.Series) -> float:
    count = float(mask.sum())
    total = float(denom.count())
    return count / total if total else 0.0


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _aggregate_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_STATCAST_PITCHER_DAILY_COLUMNS)

    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df["pitcher_id"] = df["pitcher"].astype(str)
    df["release_speed"] = pd.to_numeric(df["release_speed"], errors="coerce")
    df["zone"] = pd.to_numeric(df["zone"], errors="coerce")

    rows = []
    for (game_date, pitcher_id), group in df.groupby(["game_date", "pitcher_id"]):
        descriptions = group["description"].astype(str)
        pitch_types = group["pitch_type"].astype(str)
        rows.append(
            {
                "game_date": game_date,
                "pitcher_id": pitcher_id,
                "statcast_pitches": int(len(group)),
                "avg_release_speed": _safe_float(group["release_speed"].mean()),
                "max_release_speed": _safe_float(group["release_speed"].max()),
                "called_strike_rate": _rate(descriptions.isin(CALLED_STRIKES), descriptions),
                "swinging_strike_rate": _rate(descriptions.isin(SWINGING_STRIKES), descriptions),
                "csw_rate": _rate(
                    descriptions.isin(CALLED_STRIKES | SWINGING_STRIKES),
                    descriptions,
                ),
                "zone_rate": _rate(group["zone"].between(1, 9), group["zone"]),
                "fastball_pct": _rate(pitch_types.isin(FASTBALLS), pitch_types),
                "slider_pct": _rate(pitch_types.isin(SLIDERS), pitch_types),
                "breaking_pct": _rate(pitch_types.isin(BREAKING), pitch_types),
                "offspeed_pct": _rate(pitch_types.isin(OFFSPEED), pitch_types),
            }
        )

    return pd.DataFrame(rows).sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def _pitch_family_masks(pitch_types: pd.Series) -> dict[str, pd.Series]:
    return {
        "fastball": pitch_types.isin(FASTBALLS),
        "slider": pitch_types.isin(SLIDERS),
        "breaking": pitch_types.isin(BREAKING),
        "offspeed": pitch_types.isin(OFFSPEED),
    }


def _aggregate_batter_pitch_types(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_STATCAST_BATTER_PITCH_TYPE_DAILY_COLUMNS)

    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df["batter_id"] = df["batter"].astype(str)
    rows = []
    for (game_date, batter_id), group in df.groupby(["game_date", "batter_id"]):
        descriptions = group["description"].astype(str)
        pitch_types = group["pitch_type"].astype(str)
        family_masks = _pitch_family_masks(pitch_types)
        row = {
            "game_date": game_date,
            "batter_id": batter_id,
            "statcast_batter_pitches": int(len(group)),
        }
        for family, mask in family_masks.items():
            family_descriptions = descriptions[mask]
            row[f"{family}_pitches"] = int(mask.sum())
            row[f"{family}_swings"] = int(family_descriptions.isin(SWING_DESCRIPTIONS).sum())
            row[f"{family}_whiffs"] = int(family_descriptions.isin(SWINGING_STRIKES).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["batter_id", "game_date"]).reset_index(drop=True)


def fetch_statcast_pitcher_daily(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        from pybaseball import cache
        from pybaseball import statcast
    except ImportError as exc:
        raise ImportError("Install pybaseball to fetch Statcast data.") from exc

    cache.enable()
    raw = statcast(start_dt=start_date, end_dt=end_date)
    return _aggregate_statcast(raw)


def fetch_statcast_batter_pitch_type_daily(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        from pybaseball import cache
        from pybaseball import statcast
    except ImportError as exc:
        raise ImportError("Install pybaseball to fetch Statcast data.") from exc

    cache.enable()
    raw = statcast(start_dt=start_date, end_dt=end_date)
    return _aggregate_batter_pitch_types(raw)


def save_statcast_pitcher_daily(start_date: str, end_date: str, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_statcast_pitcher_daily(start_date, end_date)
    df.to_csv(output, index=False)
    return output


def save_statcast_batter_pitch_type_daily(start_date: str, end_date: str, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_statcast_batter_pitch_type_daily(start_date, end_date)
    df.to_csv(output, index=False)
    return output
