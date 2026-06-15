TARGETS = ["strikeouts", "walks", "hits_allowed"]

REQUIRED_LOG_COLUMNS = [
    "game_date",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "is_home",
    "strikeouts",
    "walks",
    "hits_allowed",
    "innings_pitched",
]

OPTIONAL_LOG_COLUMNS = [
    "game_pk",
    "pitches",
    "strikes",
    "batters_faced",
]

REQUIRED_TEAM_BATTING_COLUMNS = [
    "game_date",
    "game_pk",
    "team",
    "opponent",
    "is_home",
    "runs",
    "hits",
    "walks",
    "strikeouts",
    "at_bats",
    "plate_appearances",
]

REQUIRED_BATTER_GAME_COLUMNS = [
    "game_date",
    "game_pk",
    "batter_id",
    "batter_name",
    "team",
    "opponent",
    "is_home",
    "bat_side",
    "batting_order",
    "at_bats",
    "plate_appearances",
    "hits",
    "walks",
    "strikeouts",
]

REQUIRED_STATCAST_PITCHER_DAILY_COLUMNS = [
    "game_date",
    "pitcher_id",
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

REQUIRED_STATCAST_BATTER_PITCH_TYPE_DAILY_COLUMNS = [
    "game_date",
    "batter_id",
    "statcast_batter_pitches",
    "fastball_pitches",
    "fastball_swings",
    "fastball_whiffs",
    "slider_pitches",
    "slider_swings",
    "slider_whiffs",
    "breaking_pitches",
    "breaking_swings",
    "breaking_whiffs",
    "offspeed_pitches",
    "offspeed_swings",
    "offspeed_whiffs",
]

REQUIRED_PARK_FACTOR_COLUMNS = [
    "factor_year",
    "venue_id",
    "venue_name",
    "park_runs_factor",
    "park_hits_factor",
    "park_bb_factor",
    "park_so_factor",
    "park_hr_factor",
    "park_1b_factor",
    "park_2b_factor",
    "park_3b_factor",
]

REQUIRED_GAME_CONTEXT_COLUMNS = [
    "game_date",
    "game_pk",
    "pitcher_id",
    "team",
    "opponent",
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

REQUIRED_ODDS_COLUMNS = [
    "game_date",
    "pitcher_id",
    "market",
    "line",
    "over_odds",
    "under_odds",
]

PROBABLE_PITCHER_COLUMNS = [
    "game_date",
    "pitcher_id",
    "pitcher_name",
    "team",
    "opponent",
    "is_home",
]

FANGRAPHS_COLUMNS = [
    "pitcher_id",
    "season",
    "Name",
    "fg_fip",
    "fg_xfip",
    "fg_siera",
    "fg_era",
    "fg_whip",
    "fg_k_per9",
    "fg_bb_per9",
    "fg_k_pct",
    "fg_bb_pct",
    "fg_k_minus_bb_pct",
    "fg_swstr_pct",
    "fg_csw_pct",
    "fg_gb_pct",
    "fg_hr_per_fb",
    "fg_babip",
    "fg_lob_pct",
    "fg_war",
    "fg_ip",
    "fg_gs",
]
