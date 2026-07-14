"""Hits-specific features, all strictly pre-game (shift(1) everywhere).

Adds to features_full.parquet:
  DEFENSE (pitcher's own team, behind him):
    def_hits_per_bip_roll15/roll50  - team hits-allowed per ball-in-play
    team_oaa_prior                  - Savant outs-above-average, prior season
  OPPONENT OFFENSE:
    opp_off_babip_roll15/roll50     - opponent hits per BIP (their batters)
  PITCHER BIP PROFILE:
    p_hits_per_bip_roll5/roll20, p_hits_per_bip_career
    p_bip_per_bf_roll10             - contact allowed rate
  CONTACT QUALITY ALLOWED (statcast_pitcher_contact_daily.csv, if present):
    hc_{avg_ev_allowed,hardhit_rate,barrel_rate,gb_rate,ld_rate,
        sweet_spot_rate,xba_con_allowed,xwoba_con_allowed}_roll{3,10}

Output: research/v2/features_hits.parquet + hits_feature_meta.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT = Path("research/v2")
CONTACT = Path("data/raw/statcast_pitcher_contact_daily.csv")

CONTACT_COLS = ["avg_ev_allowed", "hardhit_rate", "barrel_rate", "gb_rate",
                "ld_rate", "sweet_spot_rate", "xba_con_allowed", "xwoba_con_allowed"]


def team_date_roll(df, group_col, num_col, den_col, windows, prefix):
    """Aggregate to (group, date), shift(1), rolling sum ratio."""
    g = (df.groupby([group_col, "game_date"], as_index=False)[[num_col, den_col]].sum()
           .sort_values([group_col, "game_date"]))
    out_cols = []
    grouped = g.groupby(group_col, group_keys=False)
    for w in windows:
        num = grouped[num_col].apply(lambda s: s.shift(1).rolling(w, min_periods=5).sum())
        den = grouped[den_col].apply(lambda s: s.shift(1).rolling(w, min_periods=5).sum())
        col = f"{prefix}_roll{w}"
        g[col] = (num / den.replace(0, np.nan)).values
        out_cols.append(col)
    return g[[group_col, "game_date"] + out_cols]


def main():
    feats = pd.read_parquet(OUT / "features_full.parquet")
    feats["game_date"] = pd.to_datetime(feats["game_date"])
    feats["pitcher_id"] = pd.to_numeric(feats["pitcher_id"], errors="coerce")
    feats["team"] = feats["team"].astype(str)
    feats["opponent"] = feats["opponent"].astype(str)
    new_cols = []

    logs = pd.read_csv("data/raw/pitcher_game_logs.csv")
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs.drop_duplicates(subset=["game_pk", "pitcher_id"])
    logs["team"] = logs["team"].astype(str)
    for c in ["hits_allowed", "strikeouts", "walks", "batters_faced"]:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    logs["bip"] = (logs.batters_faced - logs.strikeouts - logs.walks).clip(lower=0)

    # ---- team defense: hits allowed per BIP by the pitcher's TEAM ----
    d = team_date_roll(logs, "team", "hits_allowed", "bip", [15, 50], "def_hits_per_bip")
    feats = feats.merge(d, on=["team", "game_date"], how="left")
    new_cols += [c for c in d.columns if c.startswith("def_")]

    # ---- team OAA prior season ----
    oaa = pd.read_csv("data/raw/team_oaa_by_season.csv")
    oaa = oaa[["team_id", "season", "outs_above_average"]].copy()
    oaa["team"] = oaa["team_id"].astype(str)
    oaa["oaa_year"] = oaa["season"] + 1  # applies to the following season
    feats["oaa_year"] = feats.game_date.dt.year
    feats = feats.merge(oaa[["team", "oaa_year", "outs_above_average"]]
                        .rename(columns={"outs_above_average": "team_oaa_prior"}),
                        on=["team", "oaa_year"], how="left")
    feats = feats.drop(columns=["oaa_year"])
    new_cols.append("team_oaa_prior")

    # ---- opponent offense BABIP-ish ----
    tb = pd.read_csv("data/raw/team_batting_game_logs.csv")
    tb["game_date"] = pd.to_datetime(tb["game_date"])
    tb["team"] = tb["team"].astype(str)
    for c in ["hits", "strikeouts", "walks", "plate_appearances"]:
        tb[c] = pd.to_numeric(tb[c], errors="coerce")
    tb["bip"] = (tb.plate_appearances - tb.strikeouts - tb.walks).clip(lower=0)
    ob = team_date_roll(tb, "team", "hits", "bip", [15, 50], "opp_off_babip")
    ob = ob.rename(columns={"team": "opponent"})
    feats = feats.merge(ob, on=["opponent", "game_date"], how="left")
    new_cols += [c for c in ob.columns if c.startswith("opp_off_babip")]

    # ---- pitcher BIP profile ----
    pl = logs.sort_values(["pitcher_id", "game_date"]).copy()
    pl["pitcher_id"] = pd.to_numeric(pl["pitcher_id"], errors="coerce")
    grouped = pl.groupby("pitcher_id", group_keys=False)
    for w, tag in [(5, "roll5"), (20, "roll20")]:
        h = grouped["hits_allowed"].apply(lambda s: s.shift(1).rolling(w, min_periods=3).sum())
        b = grouped["bip"].apply(lambda s: s.shift(1).rolling(w, min_periods=3).sum())
        pl[f"p_hits_per_bip_{tag}"] = (h / b.replace(0, np.nan)).values
    ch = grouped["hits_allowed"].apply(lambda s: s.shift(1).expanding(min_periods=5).sum())
    cb = grouped["bip"].apply(lambda s: s.shift(1).expanding(min_periods=5).sum())
    pl["p_hits_per_bip_career"] = (ch / cb.replace(0, np.nan)).values
    bip10 = grouped["bip"].apply(lambda s: s.shift(1).rolling(10, min_periods=3).sum())
    bf10 = grouped["batters_faced"].apply(lambda s: s.shift(1).rolling(10, min_periods=3).sum())
    pl["p_bip_per_bf_roll10"] = (bip10 / bf10.replace(0, np.nan)).values
    pcols = ["p_hits_per_bip_roll5", "p_hits_per_bip_roll20",
             "p_hits_per_bip_career", "p_bip_per_bf_roll10"]
    feats = feats.merge(pl[["pitcher_id", "game_date"] + pcols]
                        .drop_duplicates(subset=["pitcher_id", "game_date"]),
                        on=["pitcher_id", "game_date"], how="left")
    new_cols += pcols

    # ---- contact quality allowed (rolling over pitcher's games) ----
    if CONTACT.exists():
        cc = pd.read_csv(CONTACT).drop_duplicates(subset=["game_date", "pitcher_id"])
        cc["game_date"] = pd.to_datetime(cc["game_date"])
        cc["pitcher_id"] = pd.to_numeric(cc["pitcher_id"], errors="coerce")
        cc = cc.sort_values(["pitcher_id", "game_date"])
        gg = cc.groupby("pitcher_id", group_keys=False)
        added = []
        for col in CONTACT_COLS:
            for w in [3, 10]:
                cc[f"hc_{col}_roll{w}"] = gg[col].apply(
                    lambda s: s.shift(1).rolling(w, min_periods=2).mean()).values
                added.append(f"hc_{col}_roll{w}")
        feats = feats.merge(cc[["pitcher_id", "game_date"] + added],
                            on=["pitcher_id", "game_date"], how="left")
        new_cols += added
        print(f"contact features merged: {len(added)} cols, "
              f"coverage={feats[added[0]].notna().mean():.1%}", flush=True)
    else:
        print("contact file not present yet — skipped", flush=True)

    # train-only imputation for the new columns (2022-2024 medians)
    train_mask = feats.game_date < "2025-01-01"
    fills = {}
    for c in new_cols:
        v = float(feats.loc[train_mask, c].median())
        fills[c] = v
        feats[c] = feats[c].fillna(v)

    feats.to_parquet(OUT / "features_hits.parquet", index=False)
    json.dump({"new_cols": new_cols, "fills": fills},
              open(OUT / "hits_feature_meta.json", "w"), indent=1)
    print(f"saved features_hits.parquet with {len(new_cols)} new features:", flush=True)
    print(new_cols, flush=True)


if __name__ == "__main__":
    main()
