"""
Leakage Fix Snippets — MLB Pitcher K Model
==========================================
Date: 2026-07-02
Audit file: reports/leakage_audit_report.txt

This file contains corrected code snippets for all confirmed leakage instances.
DO NOT apply directly — a human must review each fix and integrate into the source.

Each fix is labelled with the file and approximate line number it replaces.
"""

# =============================================================================
# FIX 1: build_features.py ~line 838-845
# _merge_opponent_batting_features: allow_exact_matches=True → False
# =============================================================================
# BEFORE (LEAKY):
# ---------------
# merged = pd.merge_asof(
#     group.sort_values("game_date"),
#     opponent_batting.sort_values("game_date"),
#     on="game_date",
#     by="opponent",
#     direction="backward",
#     allow_exact_matches=True,   # <-- BUG: includes same-day batting stats
# )

# AFTER (FIXED):
def _merge_opponent_batting_features_FIXED(
    pitcher_df, team_batting_logs, windows
):
    """Fixed version: allow_exact_matches=False prevents same-day doubleheader
    contamination where a team's Game-1 batting stats leak into Game-2 pitcher features."""
    from src.features.build_features import _team_batting_rolling_features
    import pandas as pd

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
            allow_exact_matches=False,  # FIXED: no same-day contamination
        )
        pieces.append(merged)

    return pd.concat(pieces, ignore_index=True, sort=False)


# =============================================================================
# FIX 2: build_features.py ~line 918-931
# _merge_prior_environment_features: exact merge → merge_asof for venue/umpire features
# =============================================================================
# BEFORE (LEAKY):
# ---------------
# def _merge_prior_environment_features(pitcher_df, windows):
#     out = pitcher_df.copy()
#     for group_col, prefix in [("venue_id", "venue"), ("home_plate_umpire_id", "umpire")]:
#         env_features = _prior_environment_features(out, group_col, prefix, windows)
#         if env_features.empty:
#             continue
#         out = out.merge(env_features, on=["game_date", group_col], how="left")  # BUG: exact date match
#     return out

# AFTER (FIXED):
def _merge_prior_environment_features_FIXED(pitcher_df, windows):
    """Fixed version: uses merge_asof with allow_exact_matches=False so that venue/umpire
    features for a game on date T only use environment statistics from strictly before T.

    The _prior_environment_features function aggregates pitcher outcomes by venue/umpire
    and date, then shifts. If two games occur at the same venue on the same date (split
    doubleheader or same park), an exact merge creates cross-contamination. merge_asof
    with allow_exact_matches=False avoids this."""
    import pandas as pd
    from src.features.build_features import _prior_environment_features

    out = pitcher_df.copy()
    for group_col, prefix in [
        ("venue_id", "venue"),
        ("home_plate_umpire_id", "umpire"),
    ]:
        env_features = _prior_environment_features(out, group_col, prefix, windows)
        if env_features.empty:
            continue

        # Use merge_asof instead of exact merge to avoid same-day cross-contamination
        env_features = env_features.sort_values("game_date")
        feat_cols = [c for c in env_features.columns if c not in ("game_date", group_col)]
        out[group_col] = pd.to_numeric(out[group_col], errors="coerce")
        env_features[group_col] = pd.to_numeric(env_features[group_col], errors="coerce")

        pieces = []
        for grp_val, grp in out.groupby(group_col, dropna=False):
            env_grp = env_features[env_features[group_col] == grp_val].sort_values("game_date")
            if env_grp.empty:
                pieces.append(grp)
                continue
            merged_piece = pd.merge_asof(
                grp.sort_values("game_date"),
                env_grp[["game_date", group_col] + feat_cols],
                on="game_date",
                by=group_col,
                direction="backward",
                allow_exact_matches=False,  # FIXED: exclude same-date environment outcomes
            )
            pieces.append(merged_piece)

        if pieces:
            out = pd.concat(pieces, ignore_index=True, sort=False)

    return out


# =============================================================================
# FIX 3: src/data/fangraphs_source.py ~line 160-164
# merge_fangraphs_prior_season: use train-period medians for imputation
# =============================================================================
# BEFORE (LEAKY):
# ---------------
# for col in fg_feature_cols:
#     if col in out.columns:
#         median_val = out[col].median()  # uses ALL rows including future data
#         out[col] = out[col].fillna(median_val if pd.notna(median_val) else 0.0)

# AFTER (FIXED):
def merge_fangraphs_prior_season_FIXED(df, fg_stats, train_end_date=None):
    """Fixed version: imputes missing FanGraphs values from train-period-only median.

    Args:
        df: pitcher game log DataFrame with game_date column
        fg_stats: FanGraphs season stats DataFrame
        train_end_date: str or datetime, last date of training period. If provided,
                        imputation uses only rows where game_date <= train_end_date.
                        If None, falls back to full-dataset median (still leaky — pass
                        train_end_date for strict no-leakage).
    """
    import pandas as pd
    from src.data.fangraphs_source import _FG_RENAME, FG_FEATURE_COLS

    if fg_stats is None or fg_stats.empty:
        return df, []

    fg = fg_stats.copy()
    fg["join_season"] = fg["season"] + 1  # prior-season join: 2023 → 2024 games

    out = df.copy()
    out["pitcher_id"] = out["pitcher_id"].astype(str)
    out["join_season"] = pd.to_datetime(out["game_date"]).dt.year

    fg_feature_cols = [c for c in FG_FEATURE_COLS if c in fg.columns]
    if not fg_feature_cols:
        return df, []

    join_cols = ["pitcher_id", "join_season"] + fg_feature_cols
    out = out.merge(fg[join_cols], on=["pitcher_id", "join_season"], how="left")
    out = out.drop(columns=["join_season"])

    # Compute imputation medians from TRAIN PERIOD ONLY
    if train_end_date is not None:
        train_mask = pd.to_datetime(out["game_date"]) <= pd.to_datetime(train_end_date)
        impute_source = out.loc[train_mask]
    else:
        # Fallback: still uses full dataset — not ideal, but preserved for backward compat
        impute_source = out

    for col in fg_feature_cols:
        if col in out.columns:
            median_val = impute_source[col].median()
            out[col] = out[col].fillna(median_val if pd.notna(median_val) else 0.0)

    return out, fg_feature_cols


# Caller change required in scripts/train.py ~line 101:
# BEFORE:
#     featured_raw, fg_cols = merge_fangraphs_prior_season(featured_raw, fangraphs)
# AFTER:
#     featured_raw, fg_cols = merge_fangraphs_prior_season_FIXED(
#         featured_raw, fangraphs,
#         train_end_date=config["training"]["train_end"]
#     )


# =============================================================================
# FIX 4: src/models/opportunity.py ~line 80-97
# add_expected_opportunity_features: use train-period fill_values for NaN imputation
# =============================================================================
# BEFORE (LEAKY):
# ---------------
# def add_expected_opportunity_features(df, models):
#     out = df.copy()
#     for target, bundle in models.items():
#         cols = bundle["feature_cols"]
#         x = (
#             out[cols]
#             .replace([np.inf, -np.inf], np.nan)
#             .fillna(out[cols].median(numeric_only=True).fillna(0.0))  # BUG: full-dataset median
#             .fillna(0.0)
#         )
#         ...

# AFTER (FIXED):
def add_expected_opportunity_features_FIXED(df, models, fill_values=None):
    """Fixed version: accepts fill_values from the training period to avoid full-dataset
    median imputation contaminating the NaN cells in opportunity model inputs.

    Args:
        df: feature DataFrame (all rows, train + test)
        models: loaded opportunity model bundles
        fill_values: dict of {col: median_value} computed from training period ONLY.
                     If None, falls back to full-dataset median (leaky path).
    """
    import pandas as pd
    import numpy as np

    out = df.copy()
    added = []
    for target, bundle in models.items():
        cols = bundle["feature_cols"]

        if fill_values is not None:
            # Use train-period medians for imputation — CORRECT
            fills = pd.Series(fill_values).reindex(cols, fill_value=0.0)
        else:
            # Fallback: full-dataset median (still leaky if called pre-split)
            fills = out[cols].median(numeric_only=True).fillna(0.0)

        x = (
            out[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fills)
            .fillna(0.0)
        )
        col = f"expected_{target}"
        out[col] = bundle["model"].predict(x)
        added.append(col)
    return out, added


# Caller change required in scripts/train.py ~line 177:
# BEFORE:
#     featured, opportunity_cols = add_expected_opportunity_features(featured, opportunity_models)
# AFTER:
#     featured, opportunity_cols = add_expected_opportunity_features_FIXED(
#         featured, opportunity_models, fill_values=fill_values_train
#     )

# Caller change required in scripts/backtest.py ~line 232:
# BEFORE:
#     featured, opp_cols = add_expected_opportunity_features(featured, opportunity_models)
# AFTER:
#     featured, opp_cols = add_expected_opportunity_features_FIXED(
#         featured, opportunity_models, fill_values=fill_values
#     )
# (fill_values is already loaded from disk at line 205 in backtest.py)


# =============================================================================
# FIX 5 (minor): build_features.py ~line 1022-1029
# Remove dead code in _league_krate_drift_features
# =============================================================================
# BEFORE (has dead code):
# ---------------
# if has_bf:
#     daily[f"league_k_rate_roll{w}"] = (
#         daily["league_bf"].shift(1).rolling(w, min_periods=1)
#         .apply(lambda x: x.mean(), raw=True)          # <-- computed then immediately overwritten
#     )
#     # Simpler: k/bf ratio rolling
#     daily[f"league_k_rate_roll{w}"] = (
#         (daily["league_k"] / daily["league_bf"].replace(0, np.nan))
#         .shift(1).rolling(w, min_periods=1).mean()
#     )

# AFTER (dead code removed):
def _league_krate_drift_features_FIXED_inner(daily, has_bf, w):
    """Inner loop body with dead code removed for clarity."""
    import numpy as np
    # shift(1) excludes same-day league average: today's games are unknown pre-game.
    daily[f"league_k_mean_roll{w}"] = (
        daily["league_k"].shift(1).rolling(w, min_periods=1).mean()
    )
    if has_bf:
        # League K rate = K/BF ratio, shifted and rolled to exclude same-day data.
        daily[f"league_k_rate_roll{w}"] = (
            (daily["league_k"] / daily["league_bf"].replace(0, np.nan))
            .shift(1).rolling(w, min_periods=1).mean()
        )


# =============================================================================
# FIX 6 (minor): scripts/backtest.py ~line 338-340
# NB dispersion fit from all game logs → training period only
# =============================================================================
# BEFORE (LEAKY):
# ---------------
# _ks = logs["strikeouts"].dropna().values   # all game logs including backtest period
# _mu = float(_ks.mean())
# _nb_alpha_k = max(0.0, float((_ks.var() - _mu) / (_mu ** 2))) if _mu > 0 else 0.0

# AFTER (FIXED):
def compute_nb_dispersion_FIXED(logs, train_start, train_end):
    """Compute NB dispersion from training period only to avoid test-set leakage."""
    from src.data.loaders import filter_date_range
    train_logs = filter_date_range(logs, train_start, train_end)
    _ks = train_logs["strikeouts"].dropna().values
    _mu = float(_ks.mean())
    _nb_alpha_k = max(0.0, float((_ks.var() - _mu) / (_mu ** 2))) if _mu > 0 else 0.0
    return {"strikeouts": _nb_alpha_k}

# Caller change in scripts/backtest.py:
# BEFORE:
#     _ks = logs["strikeouts"].dropna().values
#     _mu = float(_ks.mean())
#     _nb_alpha_k = max(0.0, float((_ks.var() - _mu) / (_mu ** 2))) if _mu > 0 else 0.0
#     nb_alpha = {"strikeouts": _nb_alpha_k}
# AFTER:
#     nb_alpha = compute_nb_dispersion_FIXED(
#         logs,
#         config["training"]["train_start"],
#         config["training"]["train_end"]
#     )
