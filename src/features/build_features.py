from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TARGETS


CONTEXT_FEATURES = [
    "venue_id",
    "temperature",
    "wind_speed_mph",
    "pitcher_throws_left",
    "pitcher_throws_right",
    "home_plate_umpire_id",
    "opp_lineup_left_batters",
    "opp_lineup_right_batters",
    "opp_lineup_switch_batters",
    "opp_lineup_same_hand_batters",
    "opp_lineup_opposite_hand_batters",
]

PARK_FACTOR_FEATURES = [
    "park_runs_factor",
    "park_hits_factor",
    "park_bb_factor",
    "park_so_factor",
    "park_hr_factor",
    "park_1b_factor",
    "park_2b_factor",
    "park_3b_factor",
]

BASE_FEATURES = ["is_home", "days_rest"]


def _available_numeric(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    cols = []
    for col in candidates:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            df[col] = numeric
            cols.append(col)
    return cols


def _rolling_by_pitcher(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["pitcher_id", "game_date"]).copy()
    grouped = df.groupby("pitcher_id", group_keys=False)

    previous_game = grouped["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - previous_game).dt.days.fillna(5).clip(0, 14)
    df["p_games_prior"] = grouped.cumcount()

    rolling_cols = TARGETS + ["innings_pitched"]
    opportunity_cols = _available_numeric(df, ["pitches", "strikes", "batters_faced"])
    rolling_cols += opportunity_cols

    for window in windows:
        shifted = grouped[rolling_cols].shift(1)
        rolled = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        rolled_std = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=2).std()
        rolled_std = rolled_std.reset_index(level=0, drop=True)
        rolled_max = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=1).max()
        rolled_max = rolled_max.reset_index(level=0, drop=True)
        rolled_min = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=1).min()
        rolled_min = rolled_min.reset_index(level=0, drop=True)
        rolled_count = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=1).count()
        rolled_count = rolled_count.reset_index(level=0, drop=True)
        for col in rolling_cols:
            df[f"p_{col}_roll{window}"] = rolled[col]
            df[f"p_{col}_std_roll{window}"] = rolled_std[col]
        # Trimmed rolling mean for strikeouts: excludes the single highest game per window.
        # Reduces upward bias from one-off dominant starts (e.g. a 9K outlier biasing a 10-start avg).
        if "strikeouts" in rolling_cols:
            k_sum = rolled["strikeouts"] * rolled_count["strikeouts"]
            k_max = rolled_max["strikeouts"]
            k_count = rolled_count["strikeouts"]
            denom = (k_count - 1).replace(0, np.nan)
            trimmed = (k_sum - k_max) / denom
            df[f"p_strikeouts_trimmed_roll{window}"] = trimmed.where(k_count >= 2, rolled["strikeouts"])
        innings = rolled["innings_pitched"].replace(0, np.nan)
        df[f"p_k_per_ip_roll{window}"] = rolled["strikeouts"] / innings
        df[f"p_bb_per_ip_roll{window}"] = rolled["walks"] / innings
        df[f"p_hits_per_ip_roll{window}"] = rolled["hits_allowed"] / innings
        df[f"p_innings_pitched_max_roll{window}"] = rolled_max["innings_pitched"]
        df[f"p_innings_pitched_min_roll{window}"] = rolled_min["innings_pitched"]
        df[f"p_deep_start_rate_6ip_roll{window}"] = (
            grouped["innings_pitched"]
            .shift(1)
            .ge(6)
            .groupby(df["pitcher_id"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"p_short_start_rate_under5ip_roll{window}"] = (
            grouped["innings_pitched"]
            .shift(1)
            .lt(5)
            .groupby(df["pitcher_id"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        if "batters_faced" in rolled:
            bf = rolled["batters_faced"].replace(0, np.nan)
            df[f"p_k_rate_roll{window}"] = rolled["strikeouts"] / bf
            df[f"p_bb_rate_roll{window}"] = rolled["walks"] / bf
            df[f"p_hits_per_bf_roll{window}"] = rolled["hits_allowed"] / bf
            df[f"p_bf_per_ip_roll{window}"] = rolled["batters_faced"] / innings
        if {"strikes", "pitches"}.issubset(rolled.columns):
            pitches = rolled["pitches"].replace(0, np.nan)
            df[f"p_strike_rate_roll{window}"] = rolled["strikes"] / pitches
            df[f"p_pitches_max_roll{window}"] = rolled_max["pitches"]
            df[f"p_pitches_min_roll{window}"] = rolled_min["pitches"]
            df[f"p_pitches_range_roll{window}"] = rolled_max["pitches"] - rolled_min["pitches"]
            df[f"p_high_pitch_rate_90_roll{window}"] = (
                grouped["pitches"]
                .shift(1)
                .ge(90)
                .groupby(df["pitcher_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"p_high_pitch_rate_100_roll{window}"] = (
                grouped["pitches"]
                .shift(1)
                .ge(100)
                .groupby(df["pitcher_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"p_low_pitch_rate_under80_roll{window}"] = (
                grouped["pitches"]
                .shift(1)
                .lt(80)
                .groupby(df["pitcher_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    shifted_totals = grouped[rolling_cols].shift(1)
    expanding = shifted_totals.groupby(df["pitcher_id"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    for col in rolling_cols:
        df[f"p_{col}_career_avg_prior"] = expanding[col]
    innings = expanding["innings_pitched"].replace(0, np.nan)
    df["p_k_per_ip_career_prior"] = expanding["strikeouts"] / innings
    df["p_bb_per_ip_career_prior"] = expanding["walks"] / innings
    df["p_hits_per_ip_career_prior"] = expanding["hits_allowed"] / innings
    if "batters_faced" in expanding:
        bf = expanding["batters_faced"].replace(0, np.nan)
        df["p_k_rate_career_prior"] = expanding["strikeouts"] / bf
        df["p_bb_rate_career_prior"] = expanding["walks"] / bf
        df["p_hits_per_bf_career_prior"] = expanding["hits_allowed"] / bf
        df["p_bf_per_ip_career_prior"] = expanding["batters_faced"] / innings
    if {"strikes", "pitches"}.issubset(expanding.columns):
        pitches = expanding["pitches"].replace(0, np.nan)
        df["p_strike_rate_career_prior"] = expanding["strikes"] / pitches
        prior_pitches = grouped["pitches"].shift(1)
        df["p_high_pitch_rate_90_career_prior"] = (
            prior_pitches.ge(90).groupby(df["pitcher_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["p_high_pitch_rate_100_career_prior"] = (
            prior_pitches.ge(100).groupby(df["pitcher_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["p_low_pitch_rate_under80_career_prior"] = (
            prior_pitches.lt(80).groupby(df["pitcher_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        )

    prior_innings = grouped["innings_pitched"].shift(1)
    df["p_deep_start_rate_6ip_career_prior"] = (
        prior_innings.ge(6).groupby(df["pitcher_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["p_short_start_rate_under5ip_career_prior"] = (
        prior_innings.lt(5).groupby(df["pitcher_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )

    return df


def _opponent_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["opponent", "game_date"]).copy()
    grouped = df.groupby("opponent", group_keys=False)

    for window in windows:
        shifted = grouped[TARGETS].shift(1)
        rolled = shifted.groupby(df["opponent"]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        for col in TARGETS:
            df[f"opp_pitcher_{col}_roll{window}"] = rolled[col]

    shifted_targets = grouped[TARGETS].shift(1)
    expanding = shifted_targets.groupby(df["opponent"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    for col in TARGETS:
        df[f"opp_pitcher_{col}_avg_prior"] = expanding[col]

    return df


def _team_batting_rolling_features(team_batting_logs: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    if team_batting_logs is None or team_batting_logs.empty:
        return pd.DataFrame()

    batting = team_batting_logs.copy()
    batting["game_date"] = pd.to_datetime(batting["game_date"])
    batting["team"] = batting["team"].astype(str)
    batting = batting.sort_values(["team", "game_date"])
    grouped = batting.groupby("team", group_keys=False)

    rolling_cols = ["runs", "hits", "walks", "strikeouts", "plate_appearances"]
    for window in windows:
        shifted = grouped[rolling_cols].shift(1)
        rolled = shifted.groupby(batting["team"]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        pa = rolled["plate_appearances"].replace(0, np.nan)
        batting[f"opp_batting_runs_roll{window}"] = rolled["runs"]
        batting[f"opp_batting_hits_roll{window}"] = rolled["hits"]
        batting[f"opp_batting_pa_roll{window}"] = rolled["plate_appearances"]
        batting[f"opp_batting_k_rate_roll{window}"] = rolled["strikeouts"] / pa
        batting[f"opp_batting_bb_rate_roll{window}"] = rolled["walks"] / pa
        batting[f"opp_batting_hit_rate_roll{window}"] = rolled["hits"] / pa

    shifted = grouped[rolling_cols].shift(1)
    expanding = shifted.groupby(batting["team"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    pa = expanding["plate_appearances"].replace(0, np.nan)
    batting["opp_batting_runs_avg_prior"] = expanding["runs"]
    batting["opp_batting_hits_avg_prior"] = expanding["hits"]
    batting["opp_batting_pa_avg_prior"] = expanding["plate_appearances"]
    batting["opp_batting_k_rate_prior"] = expanding["strikeouts"] / pa
    batting["opp_batting_bb_rate_prior"] = expanding["walks"] / pa
    batting["opp_batting_hit_rate_prior"] = expanding["hits"] / pa

    feature_cols = [col for col in batting.columns if col.startswith("opp_batting_")]
    return batting[["game_date", "team"] + feature_cols].rename(columns={"team": "opponent"})


def _merge_opposing_starter_hand(
    batter_game_logs: pd.DataFrame,
    game_context_logs: pd.DataFrame | None,
) -> pd.DataFrame:
    batters = batter_game_logs.copy()
    if game_context_logs is None or game_context_logs.empty:
        batters["starter_throws_left"] = np.nan
        batters["starter_throws_right"] = np.nan
        return batters

    context = game_context_logs[
        ["game_pk", "opponent", "pitcher_throws_left", "pitcher_throws_right"]
    ].copy()
    context = context.rename(
        columns={
            "opponent": "team",
            "pitcher_throws_left": "starter_throws_left",
            "pitcher_throws_right": "starter_throws_right",
        }
    )
    context["team"] = context["team"].astype(str)
    batters["team"] = batters["team"].astype(str)
    return batters.merge(context, on=["game_pk", "team"], how="left")


def _batter_prior_features(
    batter_game_logs: pd.DataFrame,
    game_context_logs: pd.DataFrame | None = None,
    statcast_batter_pitch_type_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if batter_game_logs is None or batter_game_logs.empty:
        return pd.DataFrame()

    batters = _merge_opposing_starter_hand(batter_game_logs, game_context_logs)
    batters["game_date"] = pd.to_datetime(batters["game_date"])
    batters["batter_id"] = batters["batter_id"].astype(str)
    batters["team"] = batters["team"].astype(str)
    batters["bat_side"] = batters["bat_side"].astype(str)
    batters = batters.sort_values(["batter_id", "game_date"])
    grouped = batters.groupby("batter_id", group_keys=False)
    cols = ["plate_appearances", "hits", "walks", "strikeouts"]
    shifted = grouped[cols].shift(1)
    expanding = shifted.groupby(batters["batter_id"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    pa = expanding["plate_appearances"].replace(0, np.nan)

    batters["batter_pa_avg_prior"] = expanding["plate_appearances"]
    batters["batter_k_rate_prior"] = expanding["strikeouts"] / pa
    batters["batter_bb_rate_prior"] = expanding["walks"] / pa
    batters["batter_hit_rate_prior"] = expanding["hits"] / pa

    for side, flag_col in [("lhp", "starter_throws_left"), ("rhp", "starter_throws_right")]:
        faced_side = pd.to_numeric(batters[flag_col], errors="coerce").fillna(0).astype(bool)
        side_pa = batters["plate_appearances"].where(faced_side, 0)
        side_k = batters["strikeouts"].where(faced_side, 0)
        side_bb = batters["walks"].where(faced_side, 0)
        side_hits = batters["hits"].where(faced_side, 0)

        prior_pa = side_pa.groupby(batters["batter_id"]).shift(1).groupby(batters["batter_id"]).cumsum()
        prior_k = side_k.groupby(batters["batter_id"]).shift(1).groupby(batters["batter_id"]).cumsum()
        prior_bb = side_bb.groupby(batters["batter_id"]).shift(1).groupby(batters["batter_id"]).cumsum()
        prior_hits = side_hits.groupby(batters["batter_id"]).shift(1).groupby(batters["batter_id"]).cumsum()
        prior_pa = prior_pa.replace(0, np.nan)
        batters[f"batter_k_rate_vs_{side}_prior"] = prior_k / prior_pa
        batters[f"batter_bb_rate_vs_{side}_prior"] = prior_bb / prior_pa
        batters[f"batter_hit_rate_vs_{side}_prior"] = prior_hits / prior_pa

    if statcast_batter_pitch_type_daily is not None and not statcast_batter_pitch_type_daily.empty:
        stat = statcast_batter_pitch_type_daily.copy()
        stat["game_date"] = pd.to_datetime(stat["game_date"])
        stat["batter_id"] = stat["batter_id"].astype(str)
        stat = stat.sort_values(["batter_id", "game_date"])
        count_cols = [
            col
            for col in stat.columns
            if col.endswith("_pitches") or col.endswith("_swings") or col.endswith("_whiffs")
        ]
        prior = stat[["game_date", "batter_id"]].copy()
        grouped_stat = stat.groupby("batter_id", group_keys=False)
        for col in count_cols:
            prior[f"prior_{col}"] = grouped_stat[col].shift(1).groupby(stat["batter_id"]).cumsum()
        for family in ["fastball", "slider", "breaking", "offspeed"]:
            pitches = prior[f"prior_{family}_pitches"].replace(0, np.nan)
            swings = prior[f"prior_{family}_swings"].replace(0, np.nan)
            prior[f"batter_{family}_swing_rate_prior"] = prior[f"prior_{family}_swings"] / pitches
            prior[f"batter_{family}_whiff_per_pitch_prior"] = prior[f"prior_{family}_whiffs"] / pitches
            prior[f"batter_{family}_whiff_per_swing_prior"] = prior[f"prior_{family}_whiffs"] / swings
        rate_cols = [
            col
            for col in prior.columns
            if col.startswith("batter_") and col.endswith("_prior")
        ]
        batters = batters.merge(prior[["game_date", "batter_id"] + rate_cols], on=["game_date", "batter_id"], how="left")

    return_cols = [
        "game_date",
        "game_pk",
        "batter_id",
        "team",
        "bat_side",
        "batting_order",
        "starter_throws_left",
        "starter_throws_right",
        "batter_pa_avg_prior",
        "batter_k_rate_prior",
        "batter_bb_rate_prior",
        "batter_hit_rate_prior",
        "batter_k_rate_vs_lhp_prior",
        "batter_bb_rate_vs_lhp_prior",
        "batter_hit_rate_vs_lhp_prior",
        "batter_k_rate_vs_rhp_prior",
        "batter_bb_rate_vs_rhp_prior",
        "batter_hit_rate_vs_rhp_prior",
        "batter_fastball_swing_rate_prior",
        "batter_fastball_whiff_per_pitch_prior",
        "batter_fastball_whiff_per_swing_prior",
        "batter_slider_swing_rate_prior",
        "batter_slider_whiff_per_pitch_prior",
        "batter_slider_whiff_per_swing_prior",
        "batter_breaking_swing_rate_prior",
        "batter_breaking_whiff_per_pitch_prior",
        "batter_breaking_whiff_per_swing_prior",
        "batter_offspeed_swing_rate_prior",
        "batter_offspeed_whiff_per_pitch_prior",
        "batter_offspeed_whiff_per_swing_prior",
    ]
    for col in return_cols:
        if col not in batters.columns:
            batters[col] = np.nan
    return batters[return_cols]


def _weighted_average(group: pd.DataFrame, col: str) -> float:
    valid = group[col].notna() & group["lineup_weight"].notna()
    if not valid.any():
        return np.nan
    weights = group.loc[valid, "lineup_weight"]
    return float(np.average(group.loc[valid, col], weights=weights))


def _lineup_features_from_batters(
    batter_game_logs: pd.DataFrame,
    game_context_logs: pd.DataFrame | None = None,
    statcast_batter_pitch_type_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    batter_features = _batter_prior_features(
        batter_game_logs,
        game_context_logs,
        statcast_batter_pitch_type_daily,
    )
    if batter_features.empty:
        return pd.DataFrame()

    starters = batter_features[
        batter_features["batting_order"].notna()
        & (batter_features["batting_order"] <= 900)
    ].copy()
    if starters.empty:
        return pd.DataFrame()

    starters["batting_slot"] = (starters["batting_order"] // 100).astype(int)
    true_starters = starters[(starters["batting_order"] % 100 == 0) & starters["batting_slot"].between(1, 9)]
    if not true_starters.empty:
        starters = true_starters.copy()
    starters = starters.sort_values(["game_date", "game_pk", "team", "batting_slot"])
    starters = starters.drop_duplicates(["game_date", "game_pk", "team", "batting_slot"], keep="first")

    order_weights = {
        1: 0.13,
        2: 0.13,
        3: 0.12,
        4: 0.12,
        5: 0.11,
        6: 0.11,
        7: 0.10,
        8: 0.09,
        9: 0.09,
    }
    starters["lineup_weight"] = starters["batting_slot"].map(order_weights)
    starters["batter_k_rate_vs_starter_hand_prior"] = np.where(
        pd.to_numeric(starters["starter_throws_left"], errors="coerce").fillna(0).astype(bool),
        starters["batter_k_rate_vs_lhp_prior"],
        starters["batter_k_rate_vs_rhp_prior"],
    )
    starters["batter_bb_rate_vs_starter_hand_prior"] = np.where(
        pd.to_numeric(starters["starter_throws_left"], errors="coerce").fillna(0).astype(bool),
        starters["batter_bb_rate_vs_lhp_prior"],
        starters["batter_bb_rate_vs_rhp_prior"],
    )
    starters["batter_hit_rate_vs_starter_hand_prior"] = np.where(
        pd.to_numeric(starters["starter_throws_left"], errors="coerce").fillna(0).astype(bool),
        starters["batter_hit_rate_vs_lhp_prior"],
        starters["batter_hit_rate_vs_rhp_prior"],
    )
    for col in [
        "batter_k_rate_vs_starter_hand_prior",
        "batter_bb_rate_vs_starter_hand_prior",
        "batter_hit_rate_vs_starter_hand_prior",
    ]:
        fallback = col.replace("_vs_starter_hand", "")
        starters[col] = starters[col].fillna(starters[fallback])

    rows = []
    for keys, group in starters.groupby(["game_date", "team"], sort=False):
        top6 = group[group["batting_slot"] <= 6]
        bottom3 = group[group["batting_slot"] >= 7]
        rows.append(
            {
                "game_date": keys[0],
                "opponent": str(keys[1]),
                "opp_lineup_batter_pa_avg_prior": group["batter_pa_avg_prior"].mean(),
                "opp_lineup_k_rate_prior": group["batter_k_rate_prior"].mean(),
                "opp_lineup_bb_rate_prior": group["batter_bb_rate_prior"].mean(),
                "opp_lineup_hit_rate_prior": group["batter_hit_rate_prior"].mean(),
                "opp_lineup_weighted_k_rate_prior": _weighted_average(group, "batter_k_rate_prior"),
                "opp_lineup_weighted_bb_rate_prior": _weighted_average(group, "batter_bb_rate_prior"),
                "opp_lineup_weighted_hit_rate_prior": _weighted_average(group, "batter_hit_rate_prior"),
                "opp_lineup_k_rate_vs_starter_hand_prior": group[
                    "batter_k_rate_vs_starter_hand_prior"
                ].mean(),
                "opp_lineup_weighted_k_rate_vs_starter_hand_prior": _weighted_average(
                    group, "batter_k_rate_vs_starter_hand_prior"
                ),
                "opp_lineup_weighted_bb_rate_vs_starter_hand_prior": _weighted_average(
                    group, "batter_bb_rate_vs_starter_hand_prior"
                ),
                "opp_lineup_weighted_hit_rate_vs_starter_hand_prior": _weighted_average(
                    group, "batter_hit_rate_vs_starter_hand_prior"
                ),
                "opp_lineup_top6_k_rate_prior": top6["batter_k_rate_prior"].mean(),
                "opp_lineup_bottom3_k_rate_prior": bottom3["batter_k_rate_prior"].mean(),
                "opp_lineup_top6_k_rate_vs_starter_hand_prior": top6[
                    "batter_k_rate_vs_starter_hand_prior"
                ].mean(),
                "opp_lineup_bottom3_k_rate_vs_starter_hand_prior": bottom3[
                    "batter_k_rate_vs_starter_hand_prior"
                ].mean(),
                "opp_lineup_weighted_fastball_whiff_per_pitch_prior": _weighted_average(
                    group, "batter_fastball_whiff_per_pitch_prior"
                ),
                "opp_lineup_weighted_slider_whiff_per_pitch_prior": _weighted_average(
                    group, "batter_slider_whiff_per_pitch_prior"
                ),
                "opp_lineup_weighted_breaking_whiff_per_pitch_prior": _weighted_average(
                    group, "batter_breaking_whiff_per_pitch_prior"
                ),
                "opp_lineup_weighted_offspeed_whiff_per_pitch_prior": _weighted_average(
                    group, "batter_offspeed_whiff_per_pitch_prior"
                ),
                "opp_lineup_weighted_fastball_whiff_per_swing_prior": _weighted_average(
                    group, "batter_fastball_whiff_per_swing_prior"
                ),
                "opp_lineup_weighted_slider_whiff_per_swing_prior": _weighted_average(
                    group, "batter_slider_whiff_per_swing_prior"
                ),
                "opp_lineup_weighted_breaking_whiff_per_swing_prior": _weighted_average(
                    group, "batter_breaking_whiff_per_swing_prior"
                ),
                "opp_lineup_weighted_offspeed_whiff_per_swing_prior": _weighted_average(
                    group, "batter_offspeed_whiff_per_swing_prior"
                ),
                "opp_lineup_confirmed_starters": int(len(group)),
                "opp_lineup_left_batters_from_batters": int((group["bat_side"] == "L").sum()),
                "opp_lineup_right_batters_from_batters": int((group["bat_side"] == "R").sum()),
                "opp_lineup_switch_batters_from_batters": int((group["bat_side"] == "S").sum()),
            }
        )
    return pd.DataFrame(rows)


def _merge_lineup_batter_features(
    pitcher_df: pd.DataFrame,
    batter_game_logs: pd.DataFrame | None,
    game_context_logs: pd.DataFrame | None = None,
    statcast_batter_pitch_type_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    lineup = _lineup_features_from_batters(
        batter_game_logs,
        game_context_logs,
        statcast_batter_pitch_type_daily,
    )
    if lineup.empty:
        return pitcher_df

    out = pitcher_df.copy()
    out["opponent"] = out["opponent"].astype(str)
    lineup["opponent"] = lineup["opponent"].astype(str)
    return out.merge(lineup, on=["game_date", "opponent"], how="left")


def _statcast_prior_features(
    statcast_pitcher_daily: pd.DataFrame | None,
    windows: list[int],
) -> pd.DataFrame:
    if statcast_pitcher_daily is None or statcast_pitcher_daily.empty:
        return pd.DataFrame()

    statcast = statcast_pitcher_daily.copy()
    statcast["game_date"] = pd.to_datetime(statcast["game_date"])
    statcast["pitcher_id"] = statcast["pitcher_id"].astype(str)
    statcast = statcast.sort_values(["pitcher_id", "game_date"])
    grouped = statcast.groupby("pitcher_id", group_keys=False)
    stat_cols = [
        "statcast_pitches",
        "avg_release_speed",
        "max_release_speed",
        "called_strike_rate",
        "swinging_strike_rate",
        "csw_rate",
        "zone_rate",
        "fastball_pct",
        "slider_pct",
        "breaking_pct",
        "offspeed_pct",
    ]
    stat_cols = [col for col in stat_cols if col in statcast.columns]

    for window in windows:
        shifted = grouped[stat_cols].shift(1)
        rolled = shifted.groupby(statcast["pitcher_id"]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        for col in stat_cols:
            statcast[f"sc_{col}_roll{window}"] = rolled[col]

    shifted = grouped[stat_cols].shift(1)
    expanding = shifted.groupby(statcast["pitcher_id"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    for col in stat_cols:
        statcast[f"sc_{col}_avg_prior"] = expanding[col]

    # Velocity trend: linear slope of avg_release_speed over last 8 starts.
    # Negative = pitcher losing velocity (fatigue/injury risk); positive = gaining.
    if "avg_release_speed" in statcast.columns:
        _shifted_speed = grouped["avg_release_speed"].shift(1)

        def _velo_slope(s: pd.Series) -> float:
            valid = s.dropna()
            if len(valid) < 3:
                return np.nan
            return float(np.polyfit(np.arange(len(valid)), valid.values, 1)[0])

        statcast["sc_velocity_slope_roll8"] = (
            _shifted_speed.groupby(statcast["pitcher_id"])
            .rolling(8, min_periods=3)
            .apply(_velo_slope, raw=False)
            .reset_index(level=0, drop=True)
        )

    feature_cols = [col for col in statcast.columns if col.startswith("sc_")]
    return statcast[["game_date", "pitcher_id"] + feature_cols]


def _merge_statcast_features(
    pitcher_df: pd.DataFrame,
    statcast_pitcher_daily: pd.DataFrame | None,
    windows: list[int],
) -> pd.DataFrame:
    statcast_features = _statcast_prior_features(statcast_pitcher_daily, windows)
    if statcast_features.empty:
        return pitcher_df
    out = pitcher_df.copy()
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    return out.merge(statcast_features, on=["game_date", "pitcher_id"], how="left")


def _statcast_advanced_pitcher_features(
    pitcher_advanced: pd.DataFrame | None,
    windows: list[int],
) -> pd.DataFrame:
    """Rolling features from pitch-level Statcast aggregates.

    Covers per-pitch-family whiff rates, velocities, spin rates, movement,
    arm angle, first-pitch strike rate, 2-strike K rate, times-through-order K rate.
    These are the primary 'stuff quality' features missing from surface-level Statcast.
    """
    if pitcher_advanced is None or pitcher_advanced.empty:
        return pd.DataFrame()

    df = pitcher_advanced.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df = df.sort_values(["pitcher_id", "game_date"])
    grouped = df.groupby("pitcher_id", group_keys=False)

    # All numeric adv_ columns are candidates for rolling
    adv_cols = [c for c in df.columns if c.startswith("adv_") and c not in ("adv_max_tto",)]
    adv_cols = [c for c in adv_cols if pd.api.types.is_numeric_dtype(df[c])]

    for window in windows:
        shifted = grouped[adv_cols].shift(1)
        rolled = shifted.groupby(df["pitcher_id"]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        for col in adv_cols:
            df[f"{col}_roll{window}"] = rolled[col]

    shifted = grouped[adv_cols].shift(1)
    expanding = shifted.groupby(df["pitcher_id"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    for col in adv_cols:
        df[f"{col}_avg_prior"] = expanding[col]

    # Velocity slope per pitch family (5-start trend)
    def _slope(s: pd.Series) -> float:
        valid = s.dropna()
        if len(valid) < 3:
            return np.nan
        return float(np.polyfit(np.arange(len(valid)), valid.values, 1)[0])

    for fam in ("ff", "sl", "cu", "ch"):
        vcol = f"adv_{fam}_velo"
        if vcol in df.columns:
            shifted_v = grouped[vcol].shift(1)
            df[f"adv_{fam}_velo_slope5"] = (
                shifted_v.groupby(df["pitcher_id"])
                .rolling(5, min_periods=3)
                .apply(_slope, raw=False)
                .reset_index(level=0, drop=True)
            )

    # Whiff-rate slope per pitch family
    for fam in ("ff", "sl", "cu", "ch"):
        wcol = f"adv_{fam}_whiff_rate"
        if wcol in df.columns:
            shifted_w = grouped[wcol].shift(1)
            df[f"adv_{fam}_whiff_slope5"] = (
                shifted_w.groupby(df["pitcher_id"])
                .rolling(5, min_periods=3)
                .apply(_slope, raw=False)
                .reset_index(level=0, drop=True)
            )

    # Only return LAGGED features (roll/avg_prior/slope) — raw adv_ columns
    # are the current game's actual pitch outcomes and would cause data leakage.
    feat_cols = [
        c for c in df.columns
        if c.startswith("adv_") and (
            "_roll" in c or "_avg_prior" in c or "_slope" in c
        )
    ]
    return df[["game_date", "pitcher_id"] + feat_cols]


def _statcast_batter_discipline_features(
    batter_discipline: pd.DataFrame | None,
    windows: list[int],
) -> pd.DataFrame:
    """Rolling plate discipline features per batter from pitch-level Statcast.

    Covers overall O-swing%, Z-contact%, whiff rate, and per-pitch-family
    whiff/chase rates. These drive the batter vulnerability features used
    in lineup matchup scoring.
    """
    if batter_discipline is None or batter_discipline.empty:
        return pd.DataFrame()

    df = batter_discipline.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["batter_id"] = df["batter_id"].astype(str)
    df = df.sort_values(["batter_id", "game_date"])
    grouped = df.groupby("batter_id", group_keys=False)

    bat_cols = [c for c in df.columns if c.startswith("bat_")
                and pd.api.types.is_numeric_dtype(df[c])]

    for window in windows:
        shifted = grouped[bat_cols].shift(1)
        rolled = shifted.groupby(df["batter_id"]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        for col in bat_cols:
            df[f"{col}_roll{window}"] = rolled[col]

    shifted = grouped[bat_cols].shift(1)
    expanding = shifted.groupby(df["batter_id"]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    for col in bat_cols:
        df[f"{col}_avg_prior"] = expanding[col]

    # Only return LAGGED features — raw bat_ columns are current-game outcomes (leakage).
    feat_cols = [
        c for c in df.columns
        if c.startswith("bat_") and ("_roll" in c or "_avg_prior" in c)
    ]
    return df[["game_date", "batter_id"] + feat_cols]


def _arsenal_lineup_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-features: pitcher's pitch-family effectiveness × lineup vulnerability.

    These capture the key matchup signal:
      - If a pitcher's slider has high whiff rate AND the lineup chases sliders,
        K total should be elevated above what historical averages predict.
      - If a pitcher's fastball is losing velocity AND lineup punishes slow fastballs,
        K total should be depressed.
    """
    out = df.copy()

    for fam in ("ff", "sl", "cu", "ch"):
        p_whiff_col  = f"adv_{fam}_whiff_rate_roll5"
        p_usage_col  = f"adv_{fam}_pct_roll5"
        l_whiff_col  = f"lineup_{fam}_whiff_prior"   # from batter discipline merge
        l_osw_col    = f"lineup_{fam}_o_swing_prior"

        # Pitcher stuff score for this family = usage × whiff_rate
        if p_whiff_col in out.columns and p_usage_col in out.columns:
            out[f"matchup_{fam}_stuff_score"] = out[p_whiff_col] * out[p_usage_col]

        # Matchup index = pitcher's stuff × lineup's vulnerability
        if p_whiff_col in out.columns and l_whiff_col in out.columns:
            out[f"matchup_{fam}_k_index"] = out[p_whiff_col] * out[l_whiff_col]

        if p_usage_col in out.columns and l_whiff_col in out.columns:
            out[f"matchup_{fam}_exposure"] = out[p_usage_col] * out[l_whiff_col]

    # Overall stuff score (weighted by usage across all families)
    stuff_scores = [f"matchup_{fam}_stuff_score" for fam in ("ff","sl","cu","ch")
                    if f"matchup_{fam}_stuff_score" in out.columns]
    if stuff_scores:
        out["matchup_overall_stuff"] = out[stuff_scores].mean(axis=1)

    # Lineup overall vulnerability composite
    vuln_cols = [f"lineup_{fam}_whiff_prior" for fam in ("ff","sl","cu","ch")
                 if f"lineup_{fam}_whiff_prior" in out.columns]
    if vuln_cols:
        out["lineup_overall_vulnerability"] = out[vuln_cols].mean(axis=1)

    return out


def _add_pitch_type_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    families = ["fastball", "slider", "breaking", "offspeed"]
    for window in [3, 5, 10]:
        pitch_mix_cols = {
            family: f"sc_{family}_pct_roll{window}"
            for family in families
            if f"sc_{family}_pct_roll{window}" in out.columns
        }
        if not pitch_mix_cols:
            continue
        for rate_type in ["whiff_per_pitch", "whiff_per_swing"]:
            pieces = []
            for family, mix_col in pitch_mix_cols.items():
                lineup_col = f"opp_lineup_weighted_{family}_{rate_type}_prior"
                if lineup_col in out.columns:
                    pieces.append(out[mix_col] * out[lineup_col])
            if pieces:
                out[f"opp_lineup_pitch_mix_{rate_type}_roll{window}"] = sum(pieces)
    return out


def _merge_park_factor_features(
    pitcher_df: pd.DataFrame,
    park_factors: pd.DataFrame | None,
) -> pd.DataFrame:
    if park_factors is None or park_factors.empty:
        return pitcher_df
    if "venue_id" not in pitcher_df.columns:
        return pitcher_df

    out = pitcher_df.copy()
    out["venue_id"] = pd.to_numeric(out["venue_id"], errors="coerce")
    out["factor_year"] = pd.to_datetime(out["game_date"]).dt.year - 1

    factors = park_factors.copy()
    factors["venue_id"] = pd.to_numeric(factors["venue_id"], errors="coerce")
    factors["factor_year"] = factors["factor_year"].astype(int)
    cols = ["venue_id", "factor_year"] + [
        col for col in PARK_FACTOR_FEATURES if col in factors.columns
    ]
    return out.merge(factors[cols], on=["venue_id", "factor_year"], how="left")


def _merge_opponent_batting_features(
    pitcher_df: pd.DataFrame,
    team_batting_logs: pd.DataFrame | None,
    windows: list[int],
) -> pd.DataFrame:
    batting_features = _team_batting_rolling_features(team_batting_logs, windows)
    if batting_features.empty:
        return pitcher_df

    pieces = []
    pitcher_df = pitcher_df.copy()
    pitcher_df["opponent"] = pitcher_df["opponent"].astype(str)
    batting_features["opponent"] = batting_features["opponent"].astype(str)

    for opponent, group in pitcher_df.groupby("opponent", sort=False):
        opponent_batting = batting_features[batting_features["opponent"] == opponent]
        if opponent_batting.empty:
            pieces.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("game_date"),
            opponent_batting.sort_values("game_date"),
            on="game_date",
            by="opponent",
            direction="backward",
            allow_exact_matches=True,
        )
        pieces.append(merged)

    return pd.concat(pieces, ignore_index=True, sort=False)


def _merge_game_context_features(
    pitcher_df: pd.DataFrame,
    game_context_logs: pd.DataFrame | None,
) -> pd.DataFrame:
    if game_context_logs is None or game_context_logs.empty:
        return pitcher_df
    if "game_pk" not in pitcher_df.columns:
        return pitcher_df

    context = game_context_logs.copy()
    context["pitcher_id"] = context["pitcher_id"].astype(str)
    for col in ["venue_id", "home_plate_umpire_id"]:
        context[col] = pd.to_numeric(context[col], errors="coerce")

    merge_cols = ["game_pk", "pitcher_id"]
    available_features = [col for col in CONTEXT_FEATURES if col in context.columns]
    return pitcher_df.merge(
        context[merge_cols + available_features],
        on=merge_cols,
        how="left",
    )


def _prior_environment_features(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
    windows: list[int],
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    cols = ["game_date", group_col] + TARGETS
    if "game_pk" in df.columns:
        cols.insert(1, "game_pk")

    env = df[cols].dropna(subset=[group_col]).copy()
    if env.empty:
        return pd.DataFrame()

    group_keys = [group_col, "game_date"]
    if "game_pk" in env.columns:
        group_keys.insert(1, "game_pk")

    env = env.groupby(group_keys, as_index=False)[TARGETS].mean()
    env = env.sort_values([group_col, "game_date"])
    grouped = env.groupby(group_col, group_keys=False)

    for window in windows:
        shifted = grouped[TARGETS].shift(1)
        rolled = shifted.groupby(env[group_col]).rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        for target in TARGETS:
            env[f"{prefix}_{target}_roll{window}"] = rolled[target]

    shifted = grouped[TARGETS].shift(1)
    expanding = shifted.groupby(env[group_col]).expanding(min_periods=1).mean()
    expanding = expanding.reset_index(level=0, drop=True)
    for target in TARGETS:
        env[f"{prefix}_{target}_avg_prior"] = expanding[target]

    feature_cols = [
        col for col in env.columns if col.startswith(f"{prefix}_") and col != group_col
    ]
    return env[["game_date", group_col] + feature_cols]


def _merge_prior_environment_features(
    pitcher_df: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    out = pitcher_df.copy()
    for group_col, prefix in [
        ("venue_id", "venue"),
        ("home_plate_umpire_id", "umpire"),
    ]:
        env_features = _prior_environment_features(out, group_col, prefix, windows)
        if env_features.empty:
            continue
        out = out.merge(env_features, on=["game_date", group_col], how="left")
    return out


def _add_composite_pitching_metrics(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add FIP-like and K-BB% composite features derived from existing rolling stats.

    These are arithmetic combinations of already-computed rolling features so they
    introduce no new data leakage risk and require no extra rolling computation.

    Added features per window w:
      p_fip_no_hr_roll{w}      = 3*BB/IP - 2*K/IP  (FIP component excluding HR)
      p_k_minus_bb_roll{w}     = K/BF - BB/BF       (K-BB%, defense-independent quality)
      p_contact_rate_roll{w}   = 1 - K/BF - BB/BF   (balls in play rate)
      p_k9_roll{w}             = K/IP * 9            (K/9 innings)
      p_bb9_roll{w}            = BB/IP * 9           (BB/9 innings)

    Career-average equivalents are also added.
    """
    out = df.copy()
    for w in windows:
        k_ip = f"p_k_per_ip_roll{w}"
        bb_ip = f"p_bb_per_ip_roll{w}"
        k_bf = f"p_k_rate_roll{w}"
        bb_bf = f"p_bb_rate_roll{w}"

        if k_ip in out.columns and bb_ip in out.columns:
            out[f"p_fip_no_hr_roll{w}"] = 3 * out[bb_ip] - 2 * out[k_ip]
            out[f"p_k9_roll{w}"] = out[k_ip] * 9
            out[f"p_bb9_roll{w}"] = out[bb_ip] * 9

        if k_bf in out.columns and bb_bf in out.columns:
            out[f"p_k_minus_bb_roll{w}"] = out[k_bf] - out[bb_bf]
            out[f"p_contact_rate_roll{w}"] = 1.0 - out[k_bf] - out[bb_bf]

    # Career-average equivalents
    k_ip_c = "p_k_per_ip_career_prior"
    bb_ip_c = "p_bb_per_ip_career_prior"
    k_bf_c = "p_k_rate_career_prior"
    bb_bf_c = "p_bb_rate_career_prior"

    if k_ip_c in out.columns and bb_ip_c in out.columns:
        out["p_fip_no_hr_career"] = 3 * out[bb_ip_c] - 2 * out[k_ip_c]
        out["p_k9_career"] = out[k_ip_c] * 9
        out["p_bb9_career"] = out[bb_ip_c] * 9

    if k_bf_c in out.columns and bb_bf_c in out.columns:
        out["p_k_minus_bb_career"] = out[k_bf_c] - out[bb_bf_c]
        out["p_contact_rate_career"] = 1.0 - out[k_bf_c] - out[bb_bf_c]

    # Exposure proxy for proper Poisson scaling: log(expected BF).
    # Uses rolling 5-game BF average as the pre-game exposure estimate.
    if "p_batters_faced_roll5" in out.columns:
        out["p_log_expected_bf"] = np.log1p(out["p_batters_faced_roll5"].clip(lower=0))
    elif "expected_batters_faced" in out.columns:
        out["p_log_expected_bf"] = np.log1p(out["expected_batters_faced"].clip(lower=0))

    return out


def _league_krate_drift_features(logs: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling league-wide K rate to capture season-level drift.

    When K rates shift league-wide (e.g. early-season ramp-up, rule changes),
    pitcher-specific rolling features lag behind. This feature gives the model
    a real-time read on the current K environment.

    Returns a DataFrame keyed by game_date with columns:
      league_k_rate_roll{w}   — mean K/BF across all starters in last w game-days
      league_k_mean_roll{w}   — mean raw Ks per start
    """
    if logs is None or logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")

    # Daily league averages (one row per unique game date)
    daily = (
        df.groupby("game_date")
        .agg(
            league_k=("strikeouts", "mean"),
            league_bf=("batters_faced", "mean") if "batters_faced" in df.columns else ("strikeouts", "count"),
        )
        .reset_index()
        .sort_values("game_date")
    )

    has_bf = "batters_faced" in df.columns
    for w in windows:
        # Shift 1 to avoid leakage then roll
        daily[f"league_k_mean_roll{w}"] = (
            daily["league_k"].shift(1).rolling(w, min_periods=1).mean()
        )
        if has_bf:
            daily[f"league_k_rate_roll{w}"] = (
                daily["league_bf"].shift(1).rolling(w, min_periods=1)
                .apply(lambda x: x.mean(), raw=True)
            )
            # Simpler: k/bf ratio rolling
            daily[f"league_k_rate_roll{w}"] = (
                (daily["league_k"] / daily["league_bf"].replace(0, np.nan))
                .shift(1).rolling(w, min_periods=1).mean()
            )

    feat_cols = [c for c in daily.columns if c.startswith("league_k")]
    return daily[["game_date"] + feat_cols]


def _pitcher_vs_team_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute each pitcher's historical K/BB/H rates specifically against each opponent team.

    Uses an expanding window (all prior games vs that team) to stay leak-free.
    Returns a DataFrame keyed by (game_date, pitcher_id, opponent) with columns:
      pvt_k_rate_vs_opp, pvt_bb_rate_vs_opp, pvt_hits_per_bf_vs_opp, pvt_games_vs_opp
    """
    if logs is None or logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["opponent"] = df["opponent"].astype(str)
    df["batters_faced"] = pd.to_numeric(df.get("batters_faced", pd.Series(dtype=float)), errors="coerce")
    df = df.sort_values(["pitcher_id", "opponent", "game_date"])

    rows = []
    grouped = df.groupby(["pitcher_id", "opponent"], group_keys=False)
    for (pid, opp), grp in grouped:
        grp = grp.sort_values("game_date").copy()
        cum_k = grp["strikeouts"].shift(1).expanding(min_periods=1).sum()
        cum_bb = grp["walks"].shift(1).expanding(min_periods=1).sum()
        cum_h = grp["hits_allowed"].shift(1).expanding(min_periods=1).sum()
        cum_bf = grp["batters_faced"].shift(1).expanding(min_periods=1).sum()
        cum_n = grp["game_date"].shift(1).expanding(min_periods=1).count()

        safe_bf = cum_bf.replace(0, np.nan)
        for idx, row in grp.iterrows():
            rows.append({
                "game_date": row["game_date"],
                "pitcher_id": pid,
                "opponent": opp,
                "pvt_k_rate_vs_opp": float(cum_k[idx] / safe_bf[idx]) if pd.notna(safe_bf[idx]) else np.nan,
                "pvt_bb_rate_vs_opp": float(cum_bb[idx] / safe_bf[idx]) if pd.notna(safe_bf[idx]) else np.nan,
                "pvt_hits_per_bf_vs_opp": float(cum_h[idx] / safe_bf[idx]) if pd.notna(safe_bf[idx]) else np.nan,
                "pvt_games_vs_opp": float(cum_n[idx]),
            })

    result = pd.DataFrame(rows)
    result["pvt_games_vs_opp"] = result["pvt_games_vs_opp"].fillna(0)
    return result


def _bullpen_workload_features(logs: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling relief-pitcher innings per team from game logs.

    Returns a DataFrame keyed by (game_date, team) with columns
    opp_bullpen_ip_roll{w} — the sum of reliever innings in the prior w games.
    Merge onto the pitcher df by joining on opponent == team.
    """
    if logs is None or logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["innings_pitched"] = pd.to_numeric(df["innings_pitched"], errors="coerce").fillna(0)

    group_key = ["game_date", "game_pk", "team"] if "game_pk" in df.columns else ["game_date", "team"]
    starter_idx = df.groupby(group_key)["innings_pitched"].idxmax()
    df["_is_starter"] = df.index.isin(set(starter_idx.dropna().values))

    relievers = df[~df["_is_starter"]]
    if relievers.empty:
        return pd.DataFrame()

    bullpen = (
        relievers.groupby(["game_date", "team"])["innings_pitched"]
        .sum()
        .reset_index()
        .rename(columns={"innings_pitched": "bullpen_ip"})
        .sort_values(["team", "game_date"])
    )

    grouped = bullpen.groupby("team", group_keys=False)
    for window in windows:
        shifted = grouped["bullpen_ip"].shift(1)
        rolled = (
            shifted.groupby(bullpen["team"])
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        bullpen[f"opp_bullpen_ip_roll{window}"] = rolled

    feat_cols = [c for c in bullpen.columns if c.startswith("opp_bullpen_ip_roll")]
    return bullpen[["game_date", "team"] + feat_cols]


def _add_situational_features(df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Add situational features not in the standard rolling stats.

    - pitcher_ip_ytd: innings pitched year-to-date (fatigue proxy)
    - pitcher_starts_ytd: starts in the current season
    - is_day_game: 1 if game_date has a day game indicator from context
    - days_into_season: calendar days since April 1 of that year (ramp-up proxy)
    """
    out = df.copy()

    # Season IP/starts accumulation
    lg = logs.copy()
    lg["game_date"] = pd.to_datetime(lg["game_date"])
    lg["pitcher_id"] = lg["pitcher_id"].astype(str)
    lg = lg.sort_values(["pitcher_id", "game_date"])
    lg["season"] = lg["game_date"].dt.year
    grouped = lg.groupby(["pitcher_id", "season"], group_keys=False)
    lg["pitcher_ip_ytd"]     = grouped["innings_pitched"].apply(lambda s: s.shift(1).expanding().sum())
    lg["pitcher_starts_ytd"] = grouped["innings_pitched"].apply(lambda s: s.shift(1).expanding().count())

    ip_merge = lg[["game_date", "pitcher_id", "pitcher_ip_ytd", "pitcher_starts_ytd"]]
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    out = out.merge(ip_merge, on=["game_date", "pitcher_id"], how="left")

    # Days into season (captures early-season ramp-up effect)
    out["_season_start"] = pd.to_datetime(out["game_date"].dt.year.astype(str) + "-04-01")
    out["days_into_season"] = (out["game_date"] - out["_season_start"]).dt.days.clip(lower=0)
    out = out.drop(columns=["_season_start"])

    return out


def build_training_features(
    logs: pd.DataFrame,
    rolling_windows: list[int],
    min_history_games: int = 3,
    team_batting_logs: pd.DataFrame | None = None,
    game_context_logs: pd.DataFrame | None = None,
    batter_game_logs: pd.DataFrame | None = None,
    statcast_pitcher_daily: pd.DataFrame | None = None,
    statcast_batter_pitch_type_daily: pd.DataFrame | None = None,
    park_factors: pd.DataFrame | None = None,
    fill_values: dict | None = None,
    statcast_pitcher_advanced: pd.DataFrame | None = None,
    statcast_batter_discipline: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    df = logs.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["team"] = df["team"].astype(str)
    df["opponent"] = df["opponent"].astype(str)
    df = _rolling_by_pitcher(df, rolling_windows)
    df = _opponent_features(df, rolling_windows)
    df = _merge_opponent_batting_features(df, team_batting_logs, rolling_windows)
    df = _merge_game_context_features(df, game_context_logs)
    df = _merge_lineup_batter_features(
        df,
        batter_game_logs,
        game_context_logs,
        statcast_batter_pitch_type_daily,
    )
    df = _merge_statcast_features(df, statcast_pitcher_daily, rolling_windows)
    df = _add_pitch_type_matchup_features(df)
    df = _merge_park_factor_features(df, park_factors)
    df = _merge_prior_environment_features(df, rolling_windows)
    df = _add_composite_pitching_metrics(df, rolling_windows)

    # Bullpen workload: opponent's relief pitcher IP over last N games
    bullpen = _bullpen_workload_features(logs, rolling_windows)
    if not bullpen.empty:
        bullpen = bullpen.rename(columns={"team": "opponent"})
        bullpen["opponent"] = bullpen["opponent"].astype(str)
        df["opponent"] = df["opponent"].astype(str)
        df = df.merge(bullpen, on=["game_date", "opponent"], how="left")

    # League-wide K-rate drift: captures season-level shifts in run environment
    league_drift = _league_krate_drift_features(logs, rolling_windows)
    if not league_drift.empty:
        df = df.merge(league_drift, on="game_date", how="left")

    # Pitcher-vs-team matchup: historical K/BB/H rates against this specific opponent
    pvt = _pitcher_vs_team_features(logs)
    if not pvt.empty:
        pvt["pitcher_id"] = pvt["pitcher_id"].astype(str)
        pvt["opponent"] = pvt["opponent"].astype(str)
        df = df.merge(pvt, on=["game_date", "pitcher_id", "opponent"], how="left")

    # Advanced Statcast: pitch-type whiff, spin, arm angle, TTO K rates
    if statcast_pitcher_advanced is not None and not statcast_pitcher_advanced.empty:
        adv_features = _statcast_advanced_pitcher_features(statcast_pitcher_advanced, rolling_windows)
        if not adv_features.empty:
            adv_features["pitcher_id"] = adv_features["pitcher_id"].astype(str)
            df = df.merge(adv_features, on=["game_date", "pitcher_id"], how="left")

    # Advanced batter discipline: per-pitch-family whiff, O-swing per batter → lineup agg
    if statcast_batter_discipline is not None and not statcast_batter_discipline.empty:
        bat_disc = _statcast_batter_discipline_features(statcast_batter_discipline, rolling_windows)
        if not bat_disc.empty and "game_pk" in df.columns and batter_game_logs is not None:
            # Aggregate to lineup level: join batter discipline to batter game logs,
            # then aggregate per (game_date, game_pk, opponent)
            bat_disc["batter_id"] = bat_disc["batter_id"].astype(str)
            bl = batter_game_logs.copy()
            bl["batter_id"] = bl["batter_id"].astype(str)
            bl["game_date"] = pd.to_datetime(bl["game_date"])

            # Use most-recent prior-game discipline stats per batter per game
            bl_disc = bl[["game_date","game_pk","batter_id","team","batting_order"]].merge(
                bat_disc, on=["game_date","batter_id"], how="left"
            )

            # Lineup aggregate per (game_date, game_pk, team)
            disc_rate_cols = [c for c in bat_disc.columns
                              if c.startswith("bat_") and ("avg_prior" in c or "roll5" in c)
                              and any(fam in c for fam in ("ff","sl","cu","ch","whiff","o_swing","k_rate"))]

            lineup_agg_rows = []
            for (gd, gpk, team), grp in bl_disc.groupby(["game_date","game_pk","team"]):
                row = {"game_date": gd, "game_pk": gpk, "opponent": str(team)}
                for col in disc_rate_cols:
                    row[f"lineup_{col}"] = grp[col].mean()
                lineup_agg_rows.append(row)

            if lineup_agg_rows:
                lineup_disc = pd.DataFrame(lineup_agg_rows)
                lineup_disc["opponent"] = lineup_disc["opponent"].astype(str)
                df["opponent"] = df["opponent"].astype(str)
                df = df.merge(lineup_disc, on=["game_date","game_pk","opponent"], how="left",
                              suffixes=("","_lineup_disc"))

    # Arsenal × lineup cross-features (matchup K index)
    df = _arsenal_lineup_matchup_features(df)

    # Situational features (YTD IP/starts, days into season)
    df = _add_situational_features(df, logs)

    df["pitcher_game_number"] = df.groupby("pitcher_id").cumcount() + 1
    df = df[df["pitcher_game_number"] > min_history_games].copy()

    feature_cols = BASE_FEATURES + [
        col
        for col in df.columns
        if col.startswith("p_")
        or col.startswith("opp_pitcher_")
        or col.startswith("opp_batting_")
        or col.startswith("opp_lineup_")
        or col.startswith("opp_bullpen_")
        or col.startswith("pvt_")
        or col.startswith("league_k")
        or col.startswith("adv_")
        or col.startswith("bat_")
        or col.startswith("lineup_bat_")
        or col.startswith("lineup_")
        or col.startswith("matchup_")
        or col.startswith("venue_")
        or col.startswith("umpire_")
        or col.startswith("sc_")
        or col in ("pitcher_ip_ytd", "pitcher_starts_ytd", "days_into_season")
        or col in PARK_FACTOR_FEATURES
        or col in CONTEXT_FEATURES
    ]
    # deduplicate while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if c not in seen and not seen.add(c)]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = df[feature_cols].median(numeric_only=True).to_dict()
    fill_series = pd.Series(fill_values).reindex(feature_cols, fill_value=0.0)
    df[feature_cols] = df[feature_cols].fillna(fill_series)
    return df.reset_index(drop=True), feature_cols, fill_values


def build_daily_features(
    historical_logs: pd.DataFrame,
    probable_pitchers: pd.DataFrame,
    rolling_windows: list[int],
    team_batting_logs: pd.DataFrame | None = None,
    game_context_logs: pd.DataFrame | None = None,
    batter_game_logs: pd.DataFrame | None = None,
    statcast_pitcher_daily: pd.DataFrame | None = None,
    statcast_batter_pitch_type_daily: pd.DataFrame | None = None,
    park_factors: pd.DataFrame | None = None,
    fill_values: dict | None = None,
    statcast_pitcher_advanced: pd.DataFrame | None = None,
    statcast_batter_discipline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    hist = historical_logs.copy()
    prob = probable_pitchers.copy()
    for frame in [hist, prob]:
        frame["pitcher_id"] = frame["pitcher_id"].astype(str)
        frame["team"] = frame["team"].astype(str)
        frame["opponent"] = frame["opponent"].astype(str)
    for target in TARGETS + ["innings_pitched"]:
        if target not in prob.columns:
            prob[target] = np.nan

    combined = pd.concat([hist, prob], ignore_index=True, sort=False)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    featured, feature_cols, _ = build_training_features(
        combined,
        rolling_windows=rolling_windows,
        min_history_games=0,
        team_batting_logs=team_batting_logs,
        game_context_logs=game_context_logs,
        batter_game_logs=batter_game_logs,
        statcast_pitcher_daily=statcast_pitcher_daily,
        statcast_batter_pitch_type_daily=statcast_batter_pitch_type_daily,
        park_factors=park_factors,
        fill_values=fill_values,
        statcast_pitcher_advanced=statcast_pitcher_advanced,
        statcast_batter_discipline=statcast_batter_discipline,
    )
    daily_keys = prob[["game_date", "pitcher_id", "team", "opponent"]]
    out = featured.merge(
        daily_keys.assign(_daily_row=1),
        on=["game_date", "pitcher_id", "team", "opponent"],
        how="inner",
    )
    return out[out["_daily_row"] == 1].drop(columns=["_daily_row"]).reset_index(drop=True)


def feature_columns_from_frame(df: pd.DataFrame) -> list[str]:
    seen: set = set()
    cols = BASE_FEATURES + [
        col
        for col in df.columns
        if col.startswith("p_")
        or col.startswith("opp_pitcher_")
        or col.startswith("opp_batting_")
        or col.startswith("opp_lineup_")
        or col.startswith("opp_bullpen_")
        or col.startswith("pvt_")
        or col.startswith("league_k")
        or col.startswith("adv_")
        or col.startswith("bat_")
        or col.startswith("lineup_bat_")
        or col.startswith("lineup_")
        or col.startswith("matchup_")
        or col.startswith("venue_")
        or col.startswith("umpire_")
        or col.startswith("sc_")
        or col in ("pitcher_ip_ytd", "pitcher_starts_ytd", "days_into_season")
        or col in PARK_FACTOR_FEATURES
        or col in CONTEXT_FEATURES
    ]
    return [c for c in cols if c not in seen and not seen.add(c)]
