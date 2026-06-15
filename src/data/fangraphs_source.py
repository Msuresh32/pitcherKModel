from __future__ import annotations

import pandas as pd

# FanGraphs season-level columns we care about, renamed to fg_* prefix to avoid collisions
_FG_RENAME = {
    "FIP": "fg_fip",
    "xFIP": "fg_xfip",
    "SIERA": "fg_siera",
    "ERA": "fg_era",
    "WHIP": "fg_whip",
    "K/9": "fg_k_per9",
    "BB/9": "fg_bb_per9",
    "K%": "fg_k_pct",
    "BB%": "fg_bb_pct",
    "K-BB%": "fg_k_minus_bb_pct",
    "SwStr%": "fg_swstr_pct",
    "CSW%": "fg_csw_pct",
    "GB%": "fg_gb_pct",
    "HR/FB": "fg_hr_per_fb",
    "BABIP": "fg_babip",
    "LOB%": "fg_lob_pct",
    "WAR": "fg_war",
    "IP": "fg_ip",
    "GS": "fg_gs",
}

FG_FEATURE_COLS = list(_FG_RENAME.values())


def fetch_fangraphs_pitcher_stats(
    start_year: int,
    end_year: int,
    qual: int = 0,
) -> pd.DataFrame:
    """Fetch FanGraphs pitcher season stats via pybaseball.

    qual=0 means no minimum IP qualifier (include all pitchers).
    Returns a DataFrame with pitcher_id (MLBAM), season, and fg_* feature columns.
    """
    try:
        import pybaseball
    except ImportError as exc:
        raise ImportError("Install pybaseball: pip install pybaseball") from exc

    pybaseball.cache.enable()

    frames = []
    for year in range(start_year, end_year + 1):
        try:
            df = pybaseball.pitching_stats(year, year, qual=qual)
        except Exception as exc:
            print(f"Warning: could not fetch FanGraphs stats for {year}: {exc}")
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df["season"] = year
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _map_to_mlbam(combined)
    combined = _rename_and_clean(combined)
    return combined


def _map_to_mlbam(fg_df: pd.DataFrame) -> pd.DataFrame:
    """Add MLBAM pitcher_id by reverse-looking up FanGraphs IDfg."""
    if "IDfg" not in fg_df.columns:
        fg_df["pitcher_id"] = pd.NA
        return fg_df

    try:
        from pybaseball import playerid_reverse_lookup
    except ImportError:
        fg_df["pitcher_id"] = pd.NA
        return fg_df

    fg_ids = fg_df["IDfg"].dropna().astype(int).unique().tolist()
    if not fg_ids:
        fg_df["pitcher_id"] = pd.NA
        return fg_df

    try:
        id_map = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
    except Exception as exc:
        print(f"Warning: playerid_reverse_lookup failed: {exc}")
        fg_df["pitcher_id"] = pd.NA
        return fg_df

    id_map = id_map[["key_fangraphs", "key_mlbam"]].dropna().copy()
    id_map["key_fangraphs"] = id_map["key_fangraphs"].astype(int)
    id_map["key_mlbam"] = id_map["key_mlbam"].astype(str)

    out = fg_df.copy()
    out["IDfg"] = out["IDfg"].astype(int)
    out = out.merge(
        id_map.rename(columns={"key_fangraphs": "IDfg", "key_mlbam": "pitcher_id"}),
        on="IDfg",
        how="left",
    )
    return out


def _rename_and_clean(fg_df: pd.DataFrame) -> pd.DataFrame:
    rename = {k: v for k, v in _FG_RENAME.items() if k in fg_df.columns}
    out = fg_df.rename(columns=rename)

    keep = ["pitcher_id", "season", "Name"] + [v for v in rename.values() if v in out.columns]
    out = out[[c for c in keep if c in out.columns]].copy()

    # Pct columns from FanGraphs come as strings like "22.3 %" — normalise to 0-1 float
    for col in out.columns:
        if col.endswith("_pct") or col.endswith("_per_fb"):
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")
            # If values look like 22.3 (percent form), convert to 0.223
            if out[col].dropna().max() > 1.5:
                out[col] = out[col] / 100

    out["pitcher_id"] = out["pitcher_id"].astype(str)
    return out.drop_duplicates(subset=["pitcher_id", "season"])


def merge_fangraphs_prior_season(
    df: pd.DataFrame,
    fg_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Join prior-season FanGraphs stats to a pitcher DataFrame by pitcher_id + year.

    Uses season = game_year - 1 to avoid leakage (prior season stats only).
    Returns (merged_df, list_of_added_feature_columns).
    """
    if fg_stats is None or fg_stats.empty:
        return df, []

    fg = fg_stats.copy()
    fg["join_season"] = fg["season"] + 1  # shift forward: 2023 stats join to 2024 games

    out = df.copy()
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    out["join_season"] = pd.to_datetime(out["game_date"]).dt.year

    fg_feature_cols = [c for c in FG_FEATURE_COLS if c in fg.columns]
    if not fg_feature_cols:
        return df, []

    join_cols = ["pitcher_id", "join_season"] + fg_feature_cols
    out = out.merge(fg[join_cols], on=["pitcher_id", "join_season"], how="left")
    out = out.drop(columns=["join_season"])

    # Impute missing FanGraphs values with overall median
    for col in fg_feature_cols:
        if col in out.columns:
            median_val = out[col].median()
            out[col] = out[col].fillna(median_val if pd.notna(median_val) else 0.0)

    return out, fg_feature_cols
