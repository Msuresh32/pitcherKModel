from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"

MARKET_MAP = {
    "pitcher_strikeouts": "strikeouts",
    "pitcher_walks": "walks",
    "pitcher_hits_allowed": "hits_allowed",
}

DEFAULT_MARKETS = list(MARKET_MAP.keys())


def load_env(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    load_env()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError("Missing ODDS_API_KEY. Add it to .env first.")
    return api_key


def _get_json(url: str, timeout: int = 30) -> tuple[dict | list, dict[str, str]]:
    try:
        with urlopen(url, timeout=timeout) as response:
            headers = {k.lower(): v for k, v in response.headers.items()}
            return json.loads(response.read().decode("utf-8")), headers
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        body = re.sub(r"apiKey=[^&]+", "apiKey=REDACTED", body)
        if exc.code == 401:
            message = (
                "The Odds API returned 401 Unauthorized. Check that ODDS_API_KEY in .env "
                "is the active key for the account/plan and has no extra characters."
            )
        elif exc.code == 402:
            message = "The Odds API returned 402 Payment Required. The request may need a paid plan or credits."
        elif exc.code == 429:
            message = "The Odds API returned 429 Too Many Requests. You may be out of credits or rate limited."
        else:
            message = f"The Odds API returned HTTP {exc.code}."
        raise RuntimeError(f"{message} Response body: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach The Odds API: {exc}") from exc


def _api_url(path: str, params: dict) -> str:
    return f"{ODDS_API_BASE_URL}{path}?{urlencode(params)}"


def _normalize_name(name: str) -> str:
    # Transliterate accented chars (é→e, ú→u, etc.) before stripping non-ASCII
    ascii_name = unicodedata.normalize("NFD", str(name)).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9 ]+", "", ascii_name.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def fetch_events(
    api_key: str,
    sport_key: str = SPORT_KEY,
    date_format: str = "iso",
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
) -> tuple[list[dict], dict[str, str]]:
    params: dict = {"apiKey": api_key, "dateFormat": date_format}
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    url = _api_url(f"/sports/{sport_key}/events", params)
    payload, headers = _get_json(url)
    return list(payload), headers


def fetch_event_odds(
    api_key: str,
    event_id: str,
    sport_key: str = SPORT_KEY,
    regions: str = "us",
    markets: list[str] | None = None,
    bookmakers: str | None = None,
) -> tuple[dict, dict[str, str]]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets or DEFAULT_MARKETS),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    url = _api_url(f"/sports/{sport_key}/events/{event_id}/odds", params)
    return _get_json(url)


def fetch_historical_events(
    api_key: str,
    snapshot_date: str,
    sport_key: str = SPORT_KEY,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
) -> tuple[list[dict], dict[str, str], dict]:
    params = {
        "apiKey": api_key,
        "date": snapshot_date,
        "dateFormat": "iso",
    }
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    payload, headers = _get_json(_api_url(f"/historical/sports/{sport_key}/events", params))
    return list(payload.get("data", [])), headers, payload


def fetch_historical_event_odds(
    api_key: str,
    event_id: str,
    snapshot_date: str,
    sport_key: str = SPORT_KEY,
    regions: str = "us",
    markets: list[str] | None = None,
    bookmakers: str | None = None,
) -> tuple[dict, dict[str, str], dict]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets or DEFAULT_MARKETS),
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_date,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    payload, headers = _get_json(
        _api_url(f"/historical/sports/{sport_key}/events/{event_id}/odds", params)
    )
    return payload.get("data", {}), headers, payload


def normalize_event_odds(event_odds: dict, fetched_at: str) -> pd.DataFrame:
    rows = []
    event_id = event_odds.get("id")
    commence_time = event_odds.get("commence_time")
    home_team = event_odds.get("home_team")
    away_team = event_odds.get("away_team")

    for bookmaker in event_odds.get("bookmakers", []):
        bookmaker_key = bookmaker.get("key")
        bookmaker_title = bookmaker.get("title")
        last_update = bookmaker.get("last_update")
        grouped: dict[tuple[str, str, float], dict] = {}

        for market in bookmaker.get("markets", []):
            api_market = market.get("key")
            internal_market = MARKET_MAP.get(api_market)
            if not internal_market:
                continue
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description") or outcome.get("player") or ""
                side = str(outcome.get("name", "")).lower()
                line = outcome.get("point")
                price = outcome.get("price")
                if not player_name or side not in {"over", "under"} or line is None:
                    continue
                key = (internal_market, player_name, float(line))
                grouped.setdefault(
                    key,
                    {
                        "fetched_at": fetched_at,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker_key,
                        "bookmaker_title": bookmaker_title,
                        "bookmaker_last_update": last_update,
                        "market": internal_market,
                        "api_market": api_market,
                        "player_name": player_name,
                        "line": float(line),
                        "over_odds": pd.NA,
                        "under_odds": pd.NA,
                    },
                )
                grouped[key][f"{side}_odds"] = price

        rows.extend(grouped.values())

    return pd.DataFrame(rows)


def map_odds_to_probables(odds: pd.DataFrame, probable_pitchers: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds

    prob = probable_pitchers.copy()
    prob["name_key"] = prob["pitcher_name"].map(_normalize_name)
    prob = prob[["game_date", "pitcher_id", "pitcher_name", "team", "opponent", "name_key"]]

    out = odds.copy()
    out["name_key"] = out["player_name"].map(_normalize_name)
    out = out.merge(prob, on="name_key", how="left")
    out["game_date"] = (
        pd.to_datetime(out["commence_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    return out.drop(columns=["name_key"])


def map_odds_to_pitcher_logs(odds: pd.DataFrame, pitcher_logs: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds

    pitchers = pitcher_logs[
        ["game_date", "pitcher_id", "pitcher_name", "team", "opponent"]
    ].drop_duplicates()
    pitchers = pitchers.copy()
    pitchers["game_date"] = pd.to_datetime(pitchers["game_date"]).dt.date.astype(str)
    pitchers["name_key"] = pitchers["pitcher_name"].map(_normalize_name)

    out = odds.copy()
    out["game_date"] = (
        pd.to_datetime(out["commence_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    out["name_key"] = out["player_name"].map(_normalize_name)
    out = out.merge(
        pitchers,
        on=["game_date", "name_key"],
        how="left",
        suffixes=("", "_matched"),
    )
    return out.drop(columns=["name_key"])


def game_snapshot_time(commence_time: str, hours_before: float) -> str:
    ts = pd.to_datetime(commence_time, utc=True)
    return (ts - pd.Timedelta(hours=hours_before)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_pitcher_prop_odds(
    probable_pitchers: pd.DataFrame,
    target_date: str | None = None,
    regions: str = "us",
    markets: list[str] | None = None,
    bookmakers: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    api_key = get_api_key()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # Pass explicit UTC date range so evening games (which roll into next UTC day) are included
    time_from = f"{target_date}T00:00:00Z" if target_date else None
    time_to   = f"{pd.to_datetime(target_date).date() + pd.Timedelta(days=1)}T09:00:00Z" if target_date else None
    events, event_headers = fetch_events(api_key, commence_time_from=time_from, commence_time_to=time_to)
    if target_date:
        target = pd.to_datetime(target_date).date()
        events = [
            event
            for event in events
            if abs((pd.to_datetime(event["commence_time"], utc=True).date() - target).days) <= 1
        ]

    frames = []
    latest_headers = event_headers
    for event in events:
        event_odds, headers = fetch_event_odds(
            api_key=api_key,
            event_id=event["id"],
            regions=regions,
            markets=markets or DEFAULT_MARKETS,
            bookmakers=bookmakers,
        )
        latest_headers = headers
        frame = normalize_event_odds(event_odds, fetched_at)
        if not frame.empty:
            frames.append(frame)

    odds = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    odds = map_odds_to_probables(odds, probable_pitchers)
    return odds, latest_headers


def append_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)
    return path


def best_current_lines(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds
    mapped = odds.dropna(subset=["pitcher_id", "line"]).copy()
    if mapped.empty:
        return mapped

    rows = []
    for keys, group in mapped.groupby(["game_date", "pitcher_id", "market", "line"], dropna=False):
        over_row = group.sort_values("over_odds", ascending=False, na_position="last").iloc[0]
        under_row = group.sort_values("under_odds", ascending=False, na_position="last").iloc[0]
        rows.append(
            {
                "game_date": keys[0],
                "pitcher_id": keys[1],
                "market": keys[2],
                "line": keys[3],
                "over_odds": over_row["over_odds"],
                "under_odds": under_row["under_odds"],
                "over_bookmaker": over_row["bookmaker"],
                "under_bookmaker": under_row["bookmaker"],
                "player_name": over_row["player_name"],
                "fetched_at": over_row["fetched_at"],
            }
        )

    return pd.DataFrame(rows).sort_values(["game_date", "pitcher_id", "market", "line"])
