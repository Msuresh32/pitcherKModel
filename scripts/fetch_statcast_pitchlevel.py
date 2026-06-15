"""Fetch Statcast pitch-level data and aggregate to per-game pitcher/batter summaries.

Aggregates on the fly to avoid storing hundreds of millions of raw rows.
Saves two files:
  data/raw/statcast_pitcher_advanced_daily.csv   — per pitcher per game
  data/raw/statcast_batter_discipline_daily.csv  — per batter per game

Usage:
    python scripts/fetch_statcast_pitchlevel.py --start 2023-03-01 --end 2026-05-31
    python scripts/fetch_statcast_pitchlevel.py --start 2026-05-01 --end 2026-05-31 --append
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

PITCHER_OUT = Path("data/raw/statcast_pitcher_advanced_daily.csv")
BATTER_OUT  = Path("data/raw/statcast_batter_discipline_daily.csv")

# Pitch family mapping
FF_TYPES  = {"FF", "SI", "FC"}        # fastball family
SL_TYPES  = {"SL", "ST", "SV"}        # slider / sweeper
CU_TYPES  = {"CU", "KC", "CS", "EP"}  # curveball
CH_TYPES  = {"CH", "FS", "FO", "SC"}  # changeup / splitter
FAMILIES  = {"ff": FF_TYPES, "sl": SL_TYPES, "cu": CU_TYPES, "ch": CH_TYPES}

SWING_DESCS   = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "foul_bunt", "missed_bunt", "hit_into_play", "hit_into_play_no_out",
    "hit_into_play_score",
}
WHIFF_DESCS   = {"swinging_strike", "swinging_strike_blocked"}
STRIKE_DESCS  = {"swinging_strike", "swinging_strike_blocked", "called_strike",
                  "foul", "foul_tip", "foul_bunt"}


def _family(pt: str) -> str:
    if pt in FF_TYPES:  return "ff"
    if pt in SL_TYPES:  return "sl"
    if pt in CU_TYPES:  return "cu"
    if pt in CH_TYPES:  return "ch"
    return "other"


def _prep_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"]       = df["pitch_type"].fillna("XX").apply(_family)
    df["is_swing"]     = df["description"].isin(SWING_DESCS)
    df["is_whiff"]     = df["description"].isin(WHIFF_DESCS)
    df["is_cs"]        = df["description"] == "called_strike"
    df["in_zone"]      = df["zone"].between(1, 9)
    df["out_zone"]     = ~df["in_zone"]
    df["is_o_swing"]   = df["is_swing"] & df["out_zone"]
    df["is_z_swing"]   = df["is_swing"] & df["in_zone"]
    df["is_z_contact"] = df["is_z_swing"] & ~df["is_whiff"]
    df["is_strike"]    = df["description"].isin(STRIKE_DESCS)
    return df


def _agg_pitcher(df: pd.DataFrame) -> pd.DataFrame:
    df = _prep_flags(df)
    rows = []

    for (gdate, pid), g in df.groupby(["game_date", "pitcher"]):
        row: dict = {"game_date": gdate, "pitcher_id": str(int(pid))}
        n = len(g)
        swings = g["is_swing"].sum()

        # Overall
        row["adv_total_pitches"]     = n
        row["adv_whiff_rate"]        = g["is_whiff"].sum() / swings if swings else np.nan
        row["adv_csw_rate"]          = (g["is_whiff"].sum() + g["is_cs"].sum()) / n if n else np.nan
        o_zone = g["out_zone"].sum()
        row["adv_o_swing_rate"]      = g["is_o_swing"].sum() / o_zone if o_zone else np.nan
        row["adv_zone_rate"]         = g["in_zone"].mean()
        z_swings = g["is_z_swing"].sum()
        row["adv_z_contact_rate"]    = g["is_z_contact"].sum() / z_swings if z_swings else np.nan

        # First-pitch strike %
        fp = g[g["pitch_number"] == 1]
        row["adv_f_strike_rate"]     = (fp["type"] == "S").mean() if len(fp) else np.nan

        # K rate in 2-strike counts (PA-level)
        ts = g[g["strikes"] == 2]
        ts_pas = ts["at_bat_number"].nunique()
        ts_ks  = (ts["events"] == "strikeout").sum()
        row["adv_k_per_2strike"]     = ts_ks / ts_pas if ts_pas else np.nan

        # Times through order
        if "n_thruorder_pitcher" in g.columns:
            row["adv_max_tto"]       = int(g["n_thruorder_pitcher"].max())
            # K rate by TTO bucket
            for tto in [1, 2, 3]:
                sub = g[g["n_thruorder_pitcher"] == tto]
                sub_k  = (sub["events"] == "strikeout").sum()
                sub_pa = sub["at_bat_number"].nunique()
                row[f"adv_k_rate_tto{tto}"] = sub_k / sub_pa if sub_pa else np.nan

        # Arm angle
        if "arm_angle" in g.columns:
            row["adv_arm_angle"]     = g["arm_angle"].mean()
            row["adv_arm_angle_std"] = g["arm_angle"].std()

        # Release extension
        if "release_extension" in g.columns:
            row["adv_release_ext"]   = g["release_extension"].mean()

        # Per pitch family
        for fam, ftypes in FAMILIES.items():
            fg = g[g["family"] == fam]
            row[f"adv_{fam}_count"]    = len(fg)
            row[f"adv_{fam}_pct"]      = len(fg) / n if n else 0
            if len(fg) == 0:
                for col in ("whiff_rate","velo","eff_speed","spin_rate",
                            "pfx_x","pfx_z","o_swing","in_zone_rate"):
                    row[f"adv_{fam}_{col}"] = np.nan
                continue
            fg_swings = fg["is_swing"].sum()
            row[f"adv_{fam}_whiff_rate"]   = fg["is_whiff"].sum() / fg_swings if fg_swings else np.nan
            row[f"adv_{fam}_velo"]         = fg["release_speed"].mean()
            row[f"adv_{fam}_eff_speed"]    = fg["effective_speed"].mean() if "effective_speed" in fg.columns else np.nan
            row[f"adv_{fam}_spin_rate"]    = fg["release_spin_rate"].mean()
            row[f"adv_{fam}_pfx_x"]        = fg["pfx_x"].mean()
            row[f"adv_{fam}_pfx_z"]        = fg["pfx_z"].mean()
            row[f"adv_{fam}_o_swing"]      = fg["is_o_swing"].sum() / fg["out_zone"].sum() if fg["out_zone"].sum() else np.nan
            row[f"adv_{fam}_in_zone_rate"] = fg["in_zone"].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def _agg_batter(df: pd.DataFrame) -> pd.DataFrame:
    df = _prep_flags(df)
    rows = []

    for (gdate, bid), g in df.groupby(["game_date", "batter"]):
        row: dict = {"game_date": gdate, "batter_id": str(int(bid))}
        n      = len(g)
        swings = g["is_swing"].sum()

        # Overall plate discipline
        row["bat_total_pitches"]   = n
        row["bat_whiff_rate"]      = g["is_whiff"].sum() / swings if swings else np.nan
        o_zone = g["out_zone"].sum()
        row["bat_o_swing_rate"]    = g["is_o_swing"].sum() / o_zone if o_zone else np.nan
        z_swings = g["is_z_swing"].sum()
        row["bat_z_contact_rate"]  = g["is_z_contact"].sum() / z_swings if z_swings else np.nan
        row["bat_zone_rate"]       = g["in_zone"].mean()

        # K rate for this game
        pas = g["at_bat_number"].nunique()
        ks  = (g["events"] == "strikeout").sum()
        row["bat_k_rate"]          = ks / pas if pas else np.nan

        # Bat speed / swing length (newer Statcast fields)
        if "bat_speed" in g.columns:
            row["bat_speed_mean"]  = g["bat_speed"].dropna().mean()
        if "swing_length" in g.columns:
            row["bat_swing_len"]   = g["swing_length"].dropna().mean()

        # Per pitch family plate discipline
        for fam in FAMILIES:
            fg = g[g["family"] == fam]
            if len(fg) == 0:
                row[f"bat_{fam}_whiff_rate"]  = np.nan
                row[f"bat_{fam}_o_swing"]     = np.nan
                continue
            fg_swings = fg["is_swing"].sum()
            row[f"bat_{fam}_whiff_rate"]  = fg["is_whiff"].sum() / fg_swings if fg_swings else np.nan
            o_z = fg["out_zone"].sum()
            row[f"bat_{fam}_o_swing"]     = fg["is_o_swing"].sum() / o_z if o_z else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def fetch_and_aggregate(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch statcast for a date range and return (pitcher_agg, batter_agg)."""
    from pybaseball import statcast

    print(f"  Fetching {start} to {end}...")
    df = statcast(start, end)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Keep only regular-season, standard pitches
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    df = df.dropna(subset=["pitcher", "batter", "pitch_type"])
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)

    pitcher_agg = _agg_pitcher(df)
    batter_agg  = _agg_batter(df)
    return pitcher_agg, batter_agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  default="2023-03-01")
    parser.add_argument("--end",    default="2026-05-31")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing files instead of overwriting")
    parser.add_argument("--chunk-days", type=int, default=15,
                        help="Days per API call (default 15)")
    args = parser.parse_args()

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(args.start, args.end, freq=f"{args.chunk_days}D")
    chunks = list(zip(dates, list(dates[1:]) + [pd.Timestamp(args.end) + pd.Timedelta(days=1)]))

    all_pitcher: list[pd.DataFrame] = []
    all_batter:  list[pd.DataFrame] = []

    for i, (start, end) in enumerate(chunks):
        s = start.strftime("%Y-%m-%d")
        e = (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if s > args.end:
            break
        e = min(e, args.end)

        for attempt in range(3):
            try:
                p, b = fetch_and_aggregate(s, e)
                if not p.empty:
                    all_pitcher.append(p)
                if not b.empty:
                    all_batter.append(b)
                print(f"  [{i+1}/{len(chunks)}] {s}→{e}: {len(p)} pitcher-game rows, {len(b)} batter-game rows")
                break
            except Exception as ex:
                print(f"  Attempt {attempt+1} failed: {ex}")
                time.sleep(10 * (attempt + 1))

    if not all_pitcher:
        print("No data fetched.")
        return

    pitcher_df = pd.concat(all_pitcher, ignore_index=True)
    batter_df  = pd.concat(all_batter,  ignore_index=True)

    # Deduplicate
    pitcher_df = pitcher_df.drop_duplicates(["game_date", "pitcher_id"])
    batter_df  = batter_df.drop_duplicates(["game_date", "batter_id"])

    mode = "a" if args.append and PITCHER_OUT.exists() else "w"
    header = not (args.append and PITCHER_OUT.exists())

    pitcher_df.to_csv(PITCHER_OUT, mode=mode, header=header, index=False)
    batter_df.to_csv(BATTER_OUT,   mode=mode,
                     header=not (args.append and BATTER_OUT.exists()), index=False)

    print(f"\nSaved {len(pitcher_df)} pitcher-game rows to {PITCHER_OUT}")
    print(f"Saved {len(batter_df)} batter-game rows  to {BATTER_OUT}")


if __name__ == "__main__":
    main()
