# Pitcher Strikeouts Research Track

This repo started as a three-market pitcher prop MVP. The active research lane is now
pitcher strikeouts, especially finding fewer, sharper bets instead of betting every
model edge.

## Current Goal

Build and test a strikeout-only workflow that can:

- Project pitcher strikeouts.
- Compare projections to market lines.
- Score whether a bet is worth taking.
- Walk-forward test the selector by season.
- Produce daily projection and paper-trading files.

## Active Workflow Files

Use these first. Most other generated CSVs are useful history, but these are the main
strikeout workflow pieces.

```text
config/config.yaml
src/features/build_features.py
src/models/train.py
src/backtesting/backtest.py
src/odds/

scripts/fetch_mlb_data.py
scripts/fetch_statcast.py
scripts/fetch_statcast_batter_pitch_types.py
scripts/fetch_park_factors.py
scripts/fetch_historical_odds.py
scripts/walk_forward_strikeout_selector.py
scripts/build_walk_forward_selector_report.py
scripts/project_daily.py
scripts/scrape_daily_odds.py
```

## Important Data Files

```text
data/raw/pitcher_game_logs.csv
data/raw/team_batting_game_logs.csv
data/raw/batter_game_logs.csv
data/raw/game_context_logs.csv
data/raw/statcast_pitcher_daily.csv
data/raw/statcast_batter_pitch_type_daily.csv
data/raw/park_factors.csv

data/odds/historical_pitcher_props.csv
data/odds/historical_pitcher_strikeouts_6h_2026.csv
data/odds/pitcher_props.csv

data/processed/walk_forward_logistic_lineup_v2_selector_picks.csv
data/processed/walk_forward_logistic_lineup_v2_selector_picks_summary.csv
data/exports/walk_forward_logistic_lineup_v2_selector_picks.html
```

## Current Model Shape

There are two related pieces:

1. Strikeout projection model
   - Predicts the number of strikeouts.
   - Uses historical pitcher, opponent, lineup, Statcast, park, context, and workload
     features.

2. Bet selector model
   - Predicts whether an already-identified strikeout bet is likely to win.
   - Trained walk-forward so each test season only uses earlier seasons.
   - Current preferred baseline is logistic regression selector, because it is less
     likely to memorize noise than a flexible tree selector.

## Feature Inventory

The feature set is intentionally broad, but most columns fall into a few logical
families.

### Pitcher Form

Rolling and prior-career pitcher performance:

```text
p_strikeouts_roll3 / roll5 / roll10
p_walks_roll3 / roll5 / roll10
p_hits_allowed_roll3 / roll5 / roll10
p_innings_pitched_roll3 / roll5 / roll10
p_k_per_ip_roll3 / roll5 / roll10
p_bb_per_ip_roll3 / roll5 / roll10
p_hits_per_ip_roll3 / roll5 / roll10
p_k_rate_roll3 / roll5 / roll10
p_bb_rate_roll3 / roll5 / roll10
p_hits_per_bf_roll3 / roll5 / roll10
p_*_career_prior
days_rest
is_home
```

### Pitch Count And Leash

Workload stability and manager trust signals:

```text
p_pitches_roll3 / roll5 / roll10
p_strikes_roll3 / roll5 / roll10
p_batters_faced_roll3 / roll5 / roll10
p_pitches_std_roll3 / roll5 / roll10
p_innings_pitched_std_roll3 / roll5 / roll10
p_pitches_max_roll3 / roll5 / roll10
p_pitches_min_roll3 / roll5 / roll10
p_pitches_range_roll3 / roll5 / roll10
p_high_pitch_rate_90_roll3 / roll5 / roll10
p_high_pitch_rate_100_roll3 / roll5 / roll10
p_low_pitch_rate_under80_roll3 / roll5 / roll10
p_deep_start_rate_6ip_roll3 / roll5 / roll10
p_short_start_rate_under5ip_roll3 / roll5 / roll10
career-prior versions of the same workload ideas
```

### Opponent Team Form

Opponent team batting and pitcher environment:

```text
opp_batting_runs_roll3 / roll5 / roll10
opp_batting_hits_roll3 / roll5 / roll10
opp_batting_walks_roll3 / roll5 / roll10
opp_batting_strikeouts_roll3 / roll5 / roll10
opp_batting_plate_appearances_roll3 / roll5 / roll10
opp_batting_k_rate_roll3 / roll5 / roll10
opp_batting_bb_rate_roll3 / roll5 / roll10
opp_batting_hit_rate_roll3 / roll5 / roll10
opp_batting_*_prior
opp_pitcher_strikeouts_roll3 / roll5 / roll10
opp_pitcher_walks_roll3 / roll5 / roll10
opp_pitcher_hits_allowed_roll3 / roll5 / roll10
opp_pitcher_innings_pitched_roll3 / roll5 / roll10
opp_pitcher_*_avg_prior
```

### Opposing Lineup

Confirmed or projected lineup quality, handedness, and batting-order weighting:

```text
opp_lineup_k_rate_prior
opp_lineup_bb_rate_prior
opp_lineup_hit_rate_prior
opp_lineup_weighted_k_rate_prior
opp_lineup_weighted_bb_rate_prior
opp_lineup_weighted_hit_rate_prior
opp_lineup_k_rate_vs_starter_hand_prior
opp_lineup_weighted_k_rate_vs_starter_hand_prior
opp_lineup_weighted_bb_rate_vs_starter_hand_prior
opp_lineup_weighted_hit_rate_vs_starter_hand_prior
opp_lineup_top6_k_rate_prior
opp_lineup_bottom3_k_rate_prior
opp_lineup_top6_k_rate_vs_starter_hand_prior
opp_lineup_bottom3_k_rate_vs_starter_hand_prior
opp_lineup_confirmed_starters
opp_lineup_left_batters
opp_lineup_right_batters
opp_lineup_switch_batters
opp_lineup_same_hand_batters
opp_lineup_opposite_hand_batters
```

### Pitcher Statcast

Pitcher skill and pitch-mix signals:

```text
avg_release_speed
csw_rate
swinging_strike_rate
zone_rate
fastball_pct
slider_pct
breaking_pct
offspeed_pct
rolling and prior versions of the above
```

### Pitch-Type Matchup

How the opposing lineup profiles against the pitcher's pitch mix:

```text
opp_lineup_weighted_fastball_whiff_per_pitch_prior
opp_lineup_weighted_slider_whiff_per_pitch_prior
opp_lineup_weighted_breaking_whiff_per_pitch_prior
opp_lineup_weighted_offspeed_whiff_per_pitch_prior
opp_lineup_weighted_fastball_whiff_per_swing_prior
opp_lineup_weighted_slider_whiff_per_swing_prior
opp_lineup_weighted_breaking_whiff_per_swing_prior
opp_lineup_weighted_offspeed_whiff_per_swing_prior
opp_lineup_pitch_mix_whiff_per_pitch_roll3 / roll5 / roll10
opp_lineup_pitch_mix_whiff_per_swing_roll3 / roll5 / roll10
```

### Game Context

Venue, weather, umpire, and handedness:

```text
venue_id
temperature
wind_speed
pitcher_hand
home_plate_umpire
venue_strikeouts_roll5
venue_strikeouts_avg_prior
umpire_strikeouts_roll5
umpire_strikeouts_avg_prior
```

### Park Factors

Baseball Savant park-factor features:

```text
park_runs_factor
park_hits_factor
park_bb_factor
park_so_factor
park_hr_factor
park_1b_factor
park_2b_factor
park_3b_factor
```

### Market And Selector Features

Used by betting/backtest scripts, especially the selector:

```text
line
over_odds
under_odds
best_side
bet_odds
projection_gap
projection_signed_gap
edge_pct
fair_probability
selector_win_probability
```

## What To Run Next

If pitch count/leash is only populated for 2026, backfill older seasons with the
current merge-safe logs fetch:

```bash
python scripts\fetch_mlb_data.py logs --config config\config.yaml --start 2022-01-01 --end 2025-12-31
```

Fetch batter pitch-type matchup data if that file is not already populated:

```bash
python scripts\fetch_statcast_batter_pitch_types.py --config config\config.yaml --start 2022-01-01 --end 2026-05-31
```

Then rerun the strikeout walk-forward selector:

```bash
python scripts\walk_forward_strikeout_selector.py --model-type logistic --output-prefix data\processed\walk_forward_strikeout_selector_logistic_pitchtype_leash_v1
```

Build the report:

```bash
python scripts\build_walk_forward_selector_report.py --scored data\processed\walk_forward_strikeout_selector_logistic_pitchtype_leash_v1_scored_tests.csv --top-n-per-year 100 --output-csv data\processed\walk_forward_logistic_pitchtype_leash_v1_selector_picks.csv --output-summary data\processed\walk_forward_logistic_pitchtype_leash_v1_selector_picks_summary.csv --output-html data\exports\walk_forward_logistic_pitchtype_leash_v1_selector_picks.html
```

## My Opinion On Feature Count

We have a lot of features, but not randomly many. Most are useful baseball concepts:
pitcher ability, workload, opposing lineup strikeout tendency, handedness, pitch mix,
park, umpire, and market price.

The risk is not the raw number of features by itself. The bigger risk is that many
rolling-window columns are highly correlated, and a flexible model can learn noise
from a small historical betting sample. That is why the walk-forward selector matters.

For the current stage, the feature set is good enough for serious testing. I would not
add another pile of broad features yet. The next best step is to run ablations:

```text
baseline pitcher/opponent only
+ lineup
+ pitch count/leash
+ Statcast pitcher
+ pitch-type matchup
+ park/umpire/context
+ market/selector features
```

If a group does not improve walk-forward ROI, hit rate, calibration, or year-to-year
stability, remove it or shrink its influence. The goal is not the biggest model; it is
the most stable signal.
