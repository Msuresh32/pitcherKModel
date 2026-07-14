"""Assemble the INTERNAL half of the qualitative-validation context for each
flagged play (see qual_validation_prompt.md for the analyst instructions).

Usage: SCORE_DATE=YYYY-MM-DD py -3.14 research/v2/a7_qual_brief.py
Reads  reports/daily/v2_plays_<date>.csv (+ props file) and the raw datasets.
Writes reports/daily/v2_qual_context_<date>.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATE = os.environ.get("SCORE_DATE", pd.Timestamp.now().strftime("%Y-%m-%d"))
DAILY = Path("reports/daily")


def main():
    plays_f = DAILY / f"v2_plays_{DATE}.csv"
    if not plays_f.exists():
        raise SystemExit(f"no plays file for {DATE}")
    plays = pd.read_csv(plays_f)
    props = pd.read_csv(DAILY / f"v2_props_{DATE}.csv")

    logs = pd.read_csv("data/raw/pitcher_game_logs.csv")
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs.drop_duplicates(subset=["game_pk", "pitcher_id"])
    sc = pd.read_csv("data/raw/statcast_pitcher_daily.csv").drop_duplicates()
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    probs = pd.read_csv("data/raw/probable_pitchers.csv")
    probs = probs[probs.game_date.astype(str).str[:10] == DATE]

    ctx = []
    for _, p in plays.iterrows():
        pr = props[props.pitcher == p.pitcher].head(1)
        pid = float(pr.pitcher_id.iloc[0]) if len(pr) and "pitcher_id" in pr else None
        entry = {
            "pitcher": p.pitcher, "line": p.line, "signal": p.signal,
            "tier": p.tier, "price": p.get("price"), "book": p.get("book"),
            "proj_ks": p.get("proj_ks"), "edge": p.get("edge_over"),
            "n_books": p.get("n_books"),
            "stake_status": "PAPER (deployment guard active)"
            if p.get("stake_usd", 0) == 0 else f"${p.get('stake_usd'):.0f}",
        }
        if pid is not None:
            g = logs[logs.pitcher_id == pid].sort_values("game_date").tail(30)
            if len(g):
                last5 = g.tail(5)
                entry["recent_form"] = {
                    "ks_last5": [int(x) for x in last5.strikeouts],
                    "ip_last5": [float(x) for x in last5.innings_pitched],
                    "pitches_last5": [int(x) for x in last5.pitches.fillna(0)],
                    "days_since_last_start": int(
                        (pd.Timestamp(DATE) - g.game_date.max()).days),
                }
                entry["workload"] = {
                    "avg_pitches_l10": float(g.tail(10).pitches.mean()),
                    "short_starts_under5ip_l10": int(
                        (g.tail(10).innings_pitched < 5).sum()),
                }
            s = sc[sc.pitcher_id == pid].sort_values("game_date").tail(10)
            if len(s) >= 4:
                entry["velocity"] = {
                    "avg_velo_l3": round(float(s.tail(3).avg_release_speed.mean()), 2),
                    "avg_velo_prev7": round(float(s.head(len(s) - 3)
                                                  .avg_release_speed.mean()), 2),
                    "csw_l3": round(float(s.tail(3).csw_rate.mean()), 3),
                    "swstr_l3": round(float(s.tail(3).swinging_strike_rate.mean()), 3),
                }
        my_prob = probs[probs.pitcher_name == p.pitcher]
        if len(my_prob):
            entry["opponent_team_id"] = int(my_prob.opponent.iloc[0])
            entry["is_home"] = int(my_prob.is_home.iloc[0])
        ctx.append(entry)

    out = DAILY / f"v2_qual_context_{DATE}.json"
    out.write_text(json.dumps({"date": DATE, "plays": ctx}, indent=1))
    print(f"wrote {out} ({len(ctx)} plays)")


if __name__ == "__main__":
    main()
