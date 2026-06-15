"""Fetch probable pitchers and projected/confirmed lineups from the MLB Stats API.

Run once or twice before first pitch. Writes two files:
  - probable_pitchers.csv  — used by project_daily.py for pitcher features
  - today_lineups.csv      — lineup players (batter_id + batting_order) injected
                             into batter_game_logs so the model computes real
                             lineup matchup features instead of using fill values.

Usage:
    python scripts/fetch_pregame_lineups.py
    python scripts/fetch_pregame_lineups.py --date 2025-05-01
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.schema import PROBABLE_PITCHER_COLUMNS

_MLB_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}&hydrate=probablePitcher,lineups,team"
)

_TEAM_ID_TO_ABB: dict[int, str] = {}


def _get_team_abbreviations() -> dict[int, str]:
    """Cache MLB team ID to abbreviation mapping."""
    global _TEAM_ID_TO_ABB
    if _TEAM_ID_TO_ABB:
        return _TEAM_ID_TO_ABB
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=15
        )
        resp.raise_for_status()
        for t in resp.json().get("teams", []):
            _TEAM_ID_TO_ABB[t["id"]] = t.get("abbreviation", str(t["id"]))
    except Exception as exc:
        print(f"Warning: could not fetch team abbreviations ({exc})")
    return _TEAM_ID_TO_ABB


def _fetch_schedule(game_date: str) -> list[dict]:
    url = _MLB_SCHEDULE_URL.format(date=game_date)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    games = []
    for date_entry in data.get("dates", []):
        games.extend(date_entry.get("games", []))
    return games


def _extract_pitcher_rows(games: list[dict], game_date: str) -> list[dict]:
    """Parse MLB API game objects into probable-pitcher rows."""
    team_abb = _get_team_abbreviations()
    rows = []

    for game in games:
        game_pk = game.get("gamePk")
        teams = game.get("teams", {})

        for side, opp_side in [("home", "away"), ("away", "home")]:
            team_data = teams.get(side, {})
            opp_data = teams.get(opp_side, {})
            probable = team_data.get("probablePitcher")
            if not probable:
                continue

            pitcher_id = str(probable.get("id", ""))
            pitcher_name = probable.get("fullName", "")
            if not pitcher_id:
                continue

            team_id = team_data.get("team", {}).get("id")
            opp_id = opp_data.get("team", {}).get("id")
            team = team_abb.get(team_id, str(team_id))
            opponent = team_abb.get(opp_id, str(opp_id))
            is_home = 1 if side == "home" else 0

            rows.append(
                {
                    "game_date": game_date,
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": pitcher_name,
                    "team": team,
                    "opponent": opponent,
                    "is_home": is_home,
                }
            )

    return rows


def _extract_lineup_counts(games: list[dict]) -> dict[int, dict]:
    """Return {game_pk: {team_abb: confirmed_starters_count}} from lineup data."""
    team_abb = _get_team_abbreviations()
    lineup_counts: dict[int, dict] = {}

    for game in games:
        game_pk = game.get("gamePk")
        lineups = game.get("lineups", {})
        if not lineups:
            continue

        counts: dict[str, int] = {}
        for side, key in [("home", "homePlayers"), ("away", "awayPlayers")]:
            players = lineups.get(key, [])
            starters = [p for p in players if p.get("battingOrder", 0) and p["battingOrder"] <= 900]
            team_id = game["teams"].get(side, {}).get("team", {}).get("id")
            abb = team_abb.get(team_id, str(team_id))
            counts[abb] = len(starters)

        lineup_counts[game_pk] = counts

    return lineup_counts


def _extract_lineup_players(games: list[dict], game_date: str) -> pd.DataFrame:
    """Extract individual lineup batters from the MLB API response.

    Returns a DataFrame in batter_game_logs format with today's lineup players
    but zero stats — the model uses only their historical prior stats.
    """
    team_abb = _get_team_abbreviations()
    rows = []

    for game in games:
        game_pk = game.get("gamePk")
        lineups = game.get("lineups", {})
        if not lineups:
            continue

        for side, players_key in [("home", "homePlayers"), ("away", "awayPlayers")]:
            players = lineups.get(players_key, [])
            team_id = game["teams"].get(side, {}).get("team", {}).get("id")
            opp_side = "away" if side == "home" else "home"
            opp_id = game["teams"].get(opp_side, {}).get("team", {}).get("id")
            team = team_abb.get(team_id, str(team_id))

            for player in players:
                batting_order = player.get("battingOrder", 0)
                if not batting_order or batting_order > 900:
                    continue
                batter_id = str(player.get("id", ""))
                if not batter_id:
                    continue
                bat_side = player.get("batSide", {}).get("code", "") if isinstance(player.get("batSide"), dict) else ""
                rows.append({
                    "game_date": game_date,
                    "game_pk": game_pk,
                    "batter_id": batter_id,
                    "team": team,
                    "bat_side": bat_side,
                    "batting_order": batting_order,
                    "plate_appearances": 0,
                    "hits": 0,
                    "walks": 0,
                    "strikeouts": 0,
                })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def fetch_pregame_lineups(game_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch probable pitchers and lineup players for a date.

    Returns:
        pitchers_df  — probable pitchers with opp_lineup_confirmed_starters
        lineups_df   — individual lineup batters in batter_game_logs format
    """
    print(f"Fetching MLB schedule + lineups for {game_date}...")
    games = _fetch_schedule(game_date)
    print(f"  Found {len(games)} games")

    rows = _extract_pitcher_rows(games, game_date)
    if not rows:
        print("  No probable pitchers found")
        return pd.DataFrame(columns=PROBABLE_PITCHER_COLUMNS), pd.DataFrame()

    pitchers_df = pd.DataFrame(rows)

    lineup_counts = _extract_lineup_counts(games)
    confirmed = []
    for _, row in pitchers_df.iterrows():
        gpk = row.get("game_pk")
        opp = row.get("opponent", "")
        count = lineup_counts.get(gpk, {}).get(opp, 0) if gpk else 0
        confirmed.append(count)
    pitchers_df["opp_lineup_confirmed_starters"] = confirmed

    for col in PROBABLE_PITCHER_COLUMNS:
        if col not in pitchers_df.columns:
            pitchers_df[col] = None

    lineups_df = _extract_lineup_players(games, game_date)
    total_players = len(lineups_df)
    games_with_lineups = lineups_df["game_pk"].nunique() if not lineups_df.empty else 0
    print(f"  Lineup players: {total_players} batters across {games_with_lineups} games")

    return pitchers_df, lineups_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pre-game lineups from MLB Stats API")
    parser.add_argument("--date", default=date.today().isoformat(), help="Game date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None, help="Probable pitchers CSV path")
    parser.add_argument("--lineups-output", default=None, help="Today's lineup players CSV path")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing probable_pitchers.csv instead of overwriting",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = Path(args.output or config["data"]["probable_pitchers_file"])
    lineups_path = Path(args.lineups_output or config["data"].get("today_lineups_file", "data/raw/today_lineups.csv"))

    pitchers_df, lineups_df = fetch_pregame_lineups(args.date)
    if pitchers_df.empty:
        print("No data fetched — exiting without writing")
        return

    # --- Write probable pitchers ---
    if args.merge and out_path.exists():
        existing = pd.read_csv(out_path)
        existing["game_date"] = pd.to_datetime(existing["game_date"]).dt.date.astype(str)
        existing = existing[existing["game_date"] != args.date]
        pitchers_df["game_date"] = pd.to_datetime(pitchers_df["game_date"]).dt.date.astype(str)
        combined = pd.concat([existing, pitchers_df], ignore_index=True, sort=False)
        combined.sort_values(["game_date", "team"], inplace=True)
        combined.to_csv(out_path, index=False)
        print(f"Merged {len(pitchers_df)} pitcher rows into {out_path} ({len(combined)} total)")
    else:
        pitchers_df.to_csv(out_path, index=False)
        print(f"Saved {len(pitchers_df)} pitcher rows to {out_path}")

    # --- Write today's lineup players ---
    if not lineups_df.empty:
        lineups_path.parent.mkdir(parents=True, exist_ok=True)
        lineups_df.to_csv(lineups_path, index=False)
        print(f"Saved {len(lineups_df)} lineup player rows to {lineups_path}")
    else:
        print("No lineup players available yet (lineups not yet posted for today)")

    # Print lineup confirmation summary
    if "opp_lineup_confirmed_starters" in pitchers_df.columns:
        confirmed_games = pitchers_df[pitchers_df["opp_lineup_confirmed_starters"] >= 7]
        print(
            f"\nLineup confirmation: {len(confirmed_games)}/{len(pitchers_df)} pitchers have "
            f"opponent lineup confirmed (>=7 starters)"
        )
        for _, row in pitchers_df.iterrows():
            c = row.get("opp_lineup_confirmed_starters", 0)
            flag = "CONFIRMED" if c >= 7 else f"partial ({c})"
            print(
                f"  {row['pitcher_name']:25s} vs {row['opponent']}  [{flag}]"
            )


if __name__ == "__main__":
    main()
