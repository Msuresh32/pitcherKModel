from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from src.data.schema import (
    OPTIONAL_LOG_COLUMNS,
    PROBABLE_PITCHER_COLUMNS,
    REQUIRED_BATTER_GAME_COLUMNS,
    REQUIRED_GAME_CONTEXT_COLUMNS,
    REQUIRED_LOG_COLUMNS,
    REQUIRED_TEAM_BATTING_COLUMNS,
)

MLB_STATS_API_BASE_URL = "https://statsapi.mlb.com/api/v1"
MLB_STATS_API_V11_BASE_URL = "https://statsapi.mlb.com/api/v1.1"


def _get_json(url: str, timeout: int = 30, retries: int = 3, backoff_seconds: float = 1.5) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, URLError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    raise last_error


def _api_url(path: str, params: dict | None = None) -> str:
    url = f"{MLB_STATS_API_BASE_URL}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    return url


def _api_v11_url(path: str, params: dict | None = None) -> str:
    url = f"{MLB_STATS_API_V11_BASE_URL}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    return url


def _date_chunks(start_date: str, end_date: str, days: int = 31) -> list[tuple[str, str]]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _innings_to_float(value: str | int | float | None) -> float:
    if value in (None, ""):
        return 0.0
    text = str(value)
    if "." not in text:
        return float(text)
    whole, outs = text.split(".", 1)
    return float(whole) + float(outs[:1] or 0) / 3


def _completed_regular_season_games(schedule_payload: dict) -> list[dict]:
    games = []
    for date_block in schedule_payload.get("dates", []):
        for game in date_block.get("games", []):
            if game.get("status", {}).get("abstractGameState") == "Final":
                games.append(game)
    return games


def _team_label(team: dict) -> str:
    return str(
        team.get("id")
        or team.get("abbreviation")
        or team.get("teamName")
        or team.get("name")
        or ""
    )


def fetch_games(start_date: str, end_date: str) -> list[dict]:
    games: list[dict] = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date):
        payload = _get_json(
            _api_url(
                "/schedule",
                {
                    "sportId": 1,
                    "gameTypes": "R",
                    "startDate": chunk_start,
                    "endDate": chunk_end,
                },
            )
        )
        for game in _completed_regular_season_games(payload):
            games.append({"game_pk": game["gamePk"], "game_date": game["gameDate"][:10]})

    unique = {(game["game_pk"], game["game_date"]): game for game in games}
    return sorted(unique.values(), key=lambda game: (game["game_date"], game["game_pk"]))


def _starter_row_from_boxscore(
    boxscore: dict,
    game_pk: int,
    game_date: str,
    side: str,
) -> dict | None:
    team_box = boxscore["teams"][side]
    pitcher_ids = team_box.get("pitchers", [])
    if not pitcher_ids:
        return None

    starter_id = pitcher_ids[0]
    player = team_box["players"].get(f"ID{starter_id}")
    if not player:
        return None

    pitching = player.get("stats", {}).get("pitching", {})
    opponent_side = "home" if side == "away" else "away"
    team = _team_label(team_box["team"])
    opponent = _team_label(boxscore["teams"][opponent_side]["team"])

    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "pitcher_id": str(starter_id),
        "pitcher_name": player["person"]["fullName"],
        "team": team,
        "opponent": opponent,
        "is_home": 1 if side == "home" else 0,
        "strikeouts": int(pitching.get("strikeOuts", 0)),
        "walks": int(pitching.get("baseOnBalls", 0)),
        "hits_allowed": int(pitching.get("hits", 0)),
        "innings_pitched": _innings_to_float(pitching.get("inningsPitched", 0)),
        "pitches": _int_stat(pitching, "numberOfPitches"),
        "strikes": _int_stat(pitching, "strikes"),
        "batters_faced": _int_stat(pitching, "battersFaced"),
    }


def _int_stat(stats: dict, key: str) -> int:
    value = stats.get(key, 0)
    if value in (None, ""):
        return 0
    return int(value)


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _wind_speed_mph(wind: str | None) -> float | None:
    if not wind:
        return None
    first = str(wind).split(" ", 1)[0]
    return _float_or_none(first)


def _home_plate_umpire(feed: dict) -> tuple[str | None, str | None]:
    officials = feed.get("liveData", {}).get("boxscore", {}).get("officials", [])
    for official in officials:
        if official.get("officialType") == "Home Plate":
            person = official.get("official", {})
            umpire_id = person.get("id")
            return (str(umpire_id) if umpire_id else None, person.get("fullName"))
    return None, None


def _player_hand(feed: dict, player_id: int, hand_type: str) -> str:
    player = feed.get("gameData", {}).get("players", {}).get(f"ID{player_id}", {})
    hand = player.get(hand_type, {}).get("code")
    return str(hand or "").upper()


def _lineup_batter_ids(team_box: dict) -> list[int]:
    batter_ids = []
    for batter_id in team_box.get("batters", []):
        player = team_box.get("players", {}).get(f"ID{batter_id}", {})
        batting_order = str(player.get("battingOrder", ""))
        if batting_order and batting_order.isdigit() and int(batting_order) <= 900:
            batter_ids.append(batter_id)
    return batter_ids[:9] if batter_ids else team_box.get("batters", [])[:9]


def _opponent_lineup_hand_counts(feed: dict, side: str, pitcher_hand: str) -> dict:
    opponent_side = "home" if side == "away" else "away"
    opponent_box = feed["liveData"]["boxscore"]["teams"][opponent_side]
    counts = {"L": 0, "R": 0, "S": 0, "same": 0, "opposite": 0}
    for batter_id in _lineup_batter_ids(opponent_box):
        bat_side = _player_hand(feed, batter_id, "batSide")
        if bat_side in counts:
            counts[bat_side] += 1
        if pitcher_hand in {"L", "R"}:
            if bat_side == pitcher_hand:
                counts["same"] += 1
            elif bat_side in {"L", "R", "S"}:
                counts["opposite"] += 1
    return counts


def _game_context_row_from_feed(feed: dict, game_pk: int, game_date: str, side: str) -> dict | None:
    team_box = feed["liveData"]["boxscore"]["teams"][side]
    pitcher_ids = team_box.get("pitchers", [])
    if not pitcher_ids:
        return None

    starter_id = pitcher_ids[0]
    opponent_side = "home" if side == "away" else "away"
    pitcher_hand = _player_hand(feed, starter_id, "pitchHand")
    lineup_counts = _opponent_lineup_hand_counts(feed, side, pitcher_hand)
    weather = feed.get("gameData", {}).get("weather", {})
    venue = feed.get("gameData", {}).get("venue", {})
    umpire_id, umpire_name = _home_plate_umpire(feed)

    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "pitcher_id": str(starter_id),
        "team": _team_label(team_box["team"]),
        "opponent": _team_label(feed["liveData"]["boxscore"]["teams"][opponent_side]["team"]),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "temperature": _float_or_none(weather.get("temp")),
        "wind_speed_mph": _wind_speed_mph(weather.get("wind")),
        "weather_condition": weather.get("condition"),
        "pitcher_hand": pitcher_hand,
        "pitcher_throws_left": 1 if pitcher_hand == "L" else 0,
        "pitcher_throws_right": 1 if pitcher_hand == "R" else 0,
        "home_plate_umpire_id": umpire_id,
        "home_plate_umpire_name": umpire_name,
        "opp_lineup_left_batters": lineup_counts["L"],
        "opp_lineup_right_batters": lineup_counts["R"],
        "opp_lineup_switch_batters": lineup_counts["S"],
        "opp_lineup_same_hand_batters": lineup_counts["same"],
        "opp_lineup_opposite_hand_batters": lineup_counts["opposite"],
    }


def _team_batting_row_from_feed(feed: dict, game_pk: int, game_date: str, side: str) -> dict:
    team_box = feed["liveData"]["boxscore"]["teams"][side]
    opponent_side = "home" if side == "away" else "away"
    batting = team_box.get("teamStats", {}).get("batting", {})
    at_bats = _int_stat(batting, "atBats")
    walks = _int_stat(batting, "baseOnBalls")
    hbp = _int_stat(batting, "hitByPitch")
    sac_flies = _int_stat(batting, "sacFlies")
    plate_appearances = _int_stat(batting, "plateAppearances")
    if plate_appearances == 0:
        plate_appearances = at_bats + walks + hbp + sac_flies

    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "team": _team_label(team_box["team"]),
        "opponent": _team_label(feed["liveData"]["boxscore"]["teams"][opponent_side]["team"]),
        "is_home": 1 if side == "home" else 0,
        "runs": _int_stat(batting, "runs"),
        "hits": _int_stat(batting, "hits"),
        "walks": walks,
        "strikeouts": _int_stat(batting, "strikeOuts"),
        "at_bats": at_bats,
        "plate_appearances": plate_appearances,
    }


def _batter_rows_from_feed(feed: dict, game_pk: int, game_date: str, side: str) -> list[dict]:
    team_box = feed["liveData"]["boxscore"]["teams"][side]
    opponent_side = "home" if side == "away" else "away"
    rows = []
    for batter_id in team_box.get("batters", []):
        player = team_box.get("players", {}).get(f"ID{batter_id}", {})
        batting = player.get("stats", {}).get("batting", {})
        batting_order = str(player.get("battingOrder", ""))
        if not batting and not batting_order:
            continue

        at_bats = _int_stat(batting, "atBats")
        walks = _int_stat(batting, "baseOnBalls")
        hbp = _int_stat(batting, "hitByPitch")
        sac_flies = _int_stat(batting, "sacFlies")
        plate_appearances = _int_stat(batting, "plateAppearances")
        if plate_appearances == 0:
            plate_appearances = at_bats + walks + hbp + sac_flies

        rows.append(
            {
                "game_date": game_date,
                "game_pk": game_pk,
                "batter_id": str(batter_id),
                "batter_name": player.get("person", {}).get("fullName"),
                "team": _team_label(team_box["team"]),
                "opponent": _team_label(feed["liveData"]["boxscore"]["teams"][opponent_side]["team"]),
                "is_home": 1 if side == "home" else 0,
                "bat_side": _player_hand(feed, batter_id, "batSide"),
                "batting_order": int(batting_order) if batting_order.isdigit() else None,
                "at_bats": at_bats,
                "plate_appearances": plate_appearances,
                "hits": _int_stat(batting, "hits"),
                "walks": walks,
                "strikeouts": _int_stat(batting, "strikeOuts"),
            }
        )
    return rows


def _extras_for_game(game: dict) -> tuple[list[dict], list[dict]]:
    game_pk = game["game_pk"]
    feed = _get_json(_api_v11_url(f"/game/{game_pk}/feed/live"))
    batting_rows = []
    context_rows = []
    for side in ["away", "home"]:
        batting_rows.append(_team_batting_row_from_feed(feed, game_pk, game["game_date"], side))
        context_row = _game_context_row_from_feed(feed, game_pk, game["game_date"], side)
        if context_row:
            context_rows.append(context_row)
    return batting_rows, context_rows


def _batter_logs_for_game(game: dict) -> list[dict]:
    game_pk = game["game_pk"]
    feed = _get_json(_api_v11_url(f"/game/{game_pk}/feed/live"))
    rows = []
    for side in ["away", "home"]:
        rows.extend(_batter_rows_from_feed(feed, game_pk, game["game_date"], side))
    return rows


def _team_batting_row_from_boxscore(
    boxscore: dict,
    game_pk: int,
    game_date: str,
    side: str,
) -> dict:
    team_box = boxscore["teams"][side]
    opponent_side = "home" if side == "away" else "away"
    batting = team_box.get("teamStats", {}).get("batting", {})
    at_bats = _int_stat(batting, "atBats")
    walks = _int_stat(batting, "baseOnBalls")
    hbp = _int_stat(batting, "hitByPitch")
    sac_flies = _int_stat(batting, "sacFlies")
    plate_appearances = _int_stat(batting, "plateAppearances")
    if plate_appearances == 0:
        plate_appearances = at_bats + walks + hbp + sac_flies

    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "team": _team_label(team_box["team"]),
        "opponent": _team_label(boxscore["teams"][opponent_side]["team"]),
        "is_home": 1 if side == "home" else 0,
        "runs": _int_stat(batting, "runs"),
        "hits": _int_stat(batting, "hits"),
        "walks": walks,
        "strikeouts": _int_stat(batting, "strikeOuts"),
        "at_bats": at_bats,
        "plate_appearances": plate_appearances,
    }


def fetch_pitcher_game_logs(
    start_date: str,
    end_date: str,
    progress_every: int | None = None,
) -> pd.DataFrame:
    rows = []
    games = fetch_games(start_date, end_date)
    for index, game in enumerate(games, start=1):
        if progress_every and (index == 1 or index % progress_every == 0):
            print(f"Fetching boxscore {index}/{len(games)}")
        game_pk = game["game_pk"]
        boxscore = _get_json(_api_url(f"/game/{game_pk}/boxscore"))
        for side in ["away", "home"]:
            row = _starter_row_from_boxscore(boxscore, game_pk, game["game_date"], side)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_LOG_COLUMNS + OPTIONAL_LOG_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    return df.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def fetch_team_batting_game_logs(
    start_date: str,
    end_date: str,
    progress_every: int | None = None,
) -> pd.DataFrame:
    rows = []
    games = fetch_games(start_date, end_date)
    for index, game in enumerate(games, start=1):
        if progress_every and (index == 1 or index % progress_every == 0):
            print(f"Fetching boxscore {index}/{len(games)}")
        game_pk = game["game_pk"]
        boxscore = _get_json(_api_url(f"/game/{game_pk}/boxscore"))
        for side in ["away", "home"]:
            rows.append(_team_batting_row_from_boxscore(boxscore, game_pk, game["game_date"], side))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_TEAM_BATTING_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    return df.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def fetch_batter_game_logs(
    start_date: str,
    end_date: str,
    progress_every: int | None = None,
    max_workers: int = 12,
) -> pd.DataFrame:
    games = fetch_games(start_date, end_date)
    rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_batter_logs_for_game, game) for game in games]
        for future in as_completed(futures):
            rows.extend(future.result())
            completed += 1
            if progress_every and (completed == 1 or completed % progress_every == 0):
                print(f"Fetched batter logs {completed}/{len(games)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_BATTER_GAME_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    return df.sort_values(["game_date", "game_pk", "team", "batting_order"]).reset_index(drop=True)


def fetch_game_context_logs(
    start_date: str,
    end_date: str,
    progress_every: int | None = None,
) -> pd.DataFrame:
    rows = []
    games = fetch_games(start_date, end_date)
    for index, game in enumerate(games, start=1):
        if progress_every and (index == 1 or index % progress_every == 0):
            print(f"Fetching live feed {index}/{len(games)}")
        game_pk = game["game_pk"]
        feed = _get_json(_api_v11_url(f"/game/{game_pk}/feed/live"))
        for side in ["away", "home"]:
            row = _game_context_row_from_feed(feed, game_pk, game["game_date"], side)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_GAME_CONTEXT_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    return df.sort_values(["game_date", "game_pk", "team"]).reset_index(drop=True)


def fetch_team_batting_and_context_logs(
    start_date: str,
    end_date: str,
    progress_every: int | None = None,
    max_workers: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = fetch_games(start_date, end_date)
    batting_rows = []
    context_rows = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extras_for_game, game) for game in games]
        for future in as_completed(futures):
            game_batting, game_context = future.result()
            batting_rows.extend(game_batting)
            context_rows.extend(game_context)
            completed += 1
            if progress_every and (completed == 1 or completed % progress_every == 0):
                print(f"Fetched extras {completed}/{len(games)}")

    batting_df = pd.DataFrame(batting_rows)
    if batting_df.empty:
        batting_df = pd.DataFrame(columns=REQUIRED_TEAM_BATTING_COLUMNS)
    else:
        batting_df["game_date"] = pd.to_datetime(batting_df["game_date"]).dt.date.astype(str)
        batting_df = batting_df.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)

    context_df = pd.DataFrame(context_rows)
    if context_df.empty:
        context_df = pd.DataFrame(columns=REQUIRED_GAME_CONTEXT_COLUMNS)
    else:
        context_df["game_date"] = pd.to_datetime(context_df["game_date"]).dt.date.astype(str)
        context_df = context_df.sort_values(["game_date", "game_pk", "team"]).reset_index(drop=True)

    return batting_df, context_df


def fetch_probable_pitchers(game_date: str) -> pd.DataFrame:
    payload = _get_json(
        _api_url(
            "/schedule",
            {
                "sportId": 1,
                "gameTypes": "R",
                "date": game_date,
                "hydrate": "probablePitcher(note)",
            },
        )
    )
    rows = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            for side in ["away", "home"]:
                team_info = teams.get(side, {})
                opponent_side = "home" if side == "away" else "away"
                probable = team_info.get("probablePitcher")
                if not probable:
                    continue
                rows.append(
                    {
                        "game_date": pd.to_datetime(game["gameDate"], utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d"),
                        "pitcher_id": str(probable["id"]),
                        "pitcher_name": probable["fullName"],
                        "team": _team_label(team_info["team"]),
                        "opponent": _team_label(teams[opponent_side]["team"]),
                        "is_home": 1 if side == "home" else 0,
                    }
                )
    return pd.DataFrame(rows, columns=PROBABLE_PITCHER_COLUMNS)


def save_source_template(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=PROBABLE_PITCHER_COLUMNS).to_csv(path, index=False)
