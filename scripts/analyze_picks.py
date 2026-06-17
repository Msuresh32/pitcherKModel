"""
Post-projection analysis for flagged picks.
Prints recent form, opponent K rate, outlier check, and a verdict for each play.
Usage: python scripts/analyze_picks.py --date 2026-06-15
"""
import argparse
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LEAGUE_AVG_K_RATE = 0.246


def load_data(config_path="config/config.yaml"):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    pitcher_logs = pd.read_csv(config["data"]["pitcher_logs_file"])
    pitcher_logs["game_date"] = pd.to_datetime(pitcher_logs["game_date"])
    batter_logs  = pd.read_csv(config["data"]["batter_game_logs_file"])
    lineups      = pd.read_csv(config["data"].get("today_lineups_file", "data/raw/today_lineups.csv"))
    return pitcher_logs, batter_logs, lineups


def recent_form(pitcher_logs, pitcher_name, line, n=10):
    """Return last-n starts summary and hit rate at the given line."""
    mask = pitcher_logs["pitcher_name"].str.lower().str.strip() == pitcher_name.lower().strip()
    logs = pitcher_logs[mask].sort_values("game_date").tail(n).copy()
    if logs.empty:
        return None
    ks   = logs["strikeouts"].tolist()
    avg5 = round(np.mean(ks[-5:]), 2) if len(ks) >= 5 else round(np.mean(ks), 2)
    avg10 = round(np.mean(ks), 2)
    hit_rate = round(sum(1 for k in ks if k > line) / len(ks), 2)
    # Outlier flag: does removing the single best game drop avg by 0.5+?
    if len(ks) >= 5:
        without_best = np.mean(sorted(ks)[:-1])
        outlier_inflated = (avg10 - without_best) >= 0.6
    else:
        outlier_inflated = False
    return {
        "starts": len(ks),
        "recent_ks": ks[-5:],
        "avg5": avg5,
        "avg10": avg10,
        "hit_rate": hit_rate,
        "outlier_inflated": outlier_inflated,
        "best_game": max(ks),
    }


def opp_k_rate(batter_logs, lineups, target_date, opp_team_id):
    """Return lineup avg K rate for the opposing team."""
    today = lineups[(lineups["game_date"] == target_date) & (lineups["team"] == opp_team_id)]
    if today.empty:
        return None, []
    ids = today["batter_id"].tolist()
    stats = (
        batter_logs[batter_logs["batter_id"].isin(ids)]
        .groupby("batter_id")
        .agg(pa=("plate_appearances", "sum"), k=("strikeouts", "sum"), name=("batter_name", "last"))
        .reset_index()
    )
    stats = stats[stats["pa"] >= 30]
    if stats.empty:
        return None, []
    stats["k_rate"] = stats["k"] / stats["pa"]
    return round(stats["k_rate"].mean(), 3), stats.sort_values("k_rate", ascending=False)["name"].tolist()[:3]


def verdict(proj, line, side, form, opp_rate, model_edge):
    """Return LIKE / LEAN / PASS with a one-line reason."""
    reasons = []
    flags   = []

    # 1. Recent hit rate at the line
    if form:
        be_hit_rate = 0.45  # rough break-even for typical +odds
        if form["hit_rate"] >= 0.55:
            reasons.append(f"hits the line {int(form['hit_rate']*100)}% of last {form['starts']} starts")
        elif form["hit_rate"] <= 0.35:
            flags.append(f"only hits line {int(form['hit_rate']*100)}% recently")

        # 2. Outlier inflation
        if form["outlier_inflated"]:
            flags.append(f"best game ({form['best_game']}K) is inflating rolling avg")

        # 3. Recent avg vs projection
        gap = proj - form["avg5"]
        if gap >= 1.2:
            flags.append(f"model projects {proj:.1f} but avg last 5 is {form['avg5']}")
        elif gap <= 0.3:
            reasons.append(f"projection ({proj:.1f}K) aligns with recent avg ({form['avg5']}K)")

    # 4. Opponent K rate
    if opp_rate is not None:
        vs_league = opp_rate - LEAGUE_AVG_K_RATE
        if side == "over" and opp_rate < 0.215:
            flags.append(f"opp K rate {opp_rate:.3f} - one of the harder lineups to K")
        elif side == "over" and opp_rate >= 0.260:
            reasons.append(f"opp K rate {opp_rate:.3f} - above league avg, good matchup")
        elif side == "under" and opp_rate >= 0.260:
            reasons.append(f"opp K rate {opp_rate:.3f} - strikeout-prone lineup makes under risky")
        elif side == "under" and opp_rate < 0.215:
            reasons.append(f"opp K rate {opp_rate:.3f} - tough lineup to K supports under")

    # 5. Edge size sanity
    if model_edge >= 30:
        flags.append(f"edge {model_edge:.0f}% -- verify line is available / not a stale alt-line")

    # Verdict
    n_flags = len(flags)
    n_good  = len(reasons)
    if n_flags == 0 and n_good >= 1:
        v = "LIKE"
    elif n_flags >= 2:
        v = "PASS"
    else:
        v = "LEAN"

    note = "; ".join(reasons + ["!! " + f for f in flags]) or "no strong signal"
    return v, note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--picks-log", default="data/exports/picks_log.csv")
    parser.add_argument("--props-csv", default=None)
    args = parser.parse_args()

    pitcher_logs, batter_logs, lineups = load_data()

    # Load today's flagged picks from picks_log
    log = pd.read_csv(args.picks_log)
    today = log[log["game_date"] == args.date].copy()
    if today.empty:
        print(f"No picks found for {args.date}")
        return

    # Also load the full props CSV for opponent team_id lookup
    props_path = args.props_csv or f"data/exports/daily_pitcher_props_{args.date}.csv"
    props = pd.read_csv(props_path) if Path(props_path).exists() else pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  PICK ANALYSIS  {args.date}")
    print(f"{'='*60}")

    for _, pick in today.iterrows():
        name   = pick["pitcher_name"]
        side   = pick["best_side"]
        line   = float(pick["line"]) if pd.notna(pick.get("line")) else None
        proj   = float(pick["strikeouts_projection"]) if pd.notna(pick.get("strikeouts_projection")) else None
        edge   = float(pick["edge_pct"]) * 100 if pd.notna(pick.get("edge_pct")) and float(pick["edge_pct"]) < 2 else float(pick.get("edge_pct", 0))
        odds   = pick.get("odds_used", "?")

        # Look up opponent team id
        opp_id = None
        if not props.empty:
            row = props[props["pitcher_name"].str.lower().str.strip() == name.lower().strip()]
            if not row.empty:
                opp_id = int(row.iloc[0]["opponent"]) if pd.notna(row.iloc[0].get("opponent")) else None

        form     = recent_form(pitcher_logs, name, line) if line else None
        opp_rate, top_ks = opp_k_rate(batter_logs, lineups, args.date, opp_id) if opp_id else (None, [])
        v, note  = verdict(proj, line, side, form, opp_rate, edge)

        verdict_icon = {"LIKE": "++", "LEAN": "~", "PASS": "XX"}.get(v, "?")

        print(f"\n{verdict_icon} {v}  |  {name}  {side.upper()} {line}  |  proj={proj:.2f}  edge={edge:.1f}%  odds={odds}")
        if form:
            print(f"   Form (last {form['starts']}): {form['recent_ks']}  avg5={form['avg5']}  hit-rate@{line}={int(form['hit_rate']*100)}%")
        if opp_rate is not None:
            league_diff = opp_rate - LEAGUE_AVG_K_RATE
            diff_str = f"{'+'if league_diff>=0 else ''}{league_diff:.3f} vs league"
            print(f"   Opp K rate: {opp_rate:.3f} ({diff_str})")
        print(f"   Note: {note}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
