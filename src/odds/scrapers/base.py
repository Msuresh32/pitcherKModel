from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


class ScraperUnavailable(RuntimeError):
    pass


@dataclass
class OddsRow:
    game_date: str
    sportsbook: str
    bookmaker: str
    bookmaker_title: str
    market: str
    player_name: str
    line: float
    over_odds: float | int | None
    under_odds: float | int | None
    event_id: str | None = None
    commence_time: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    fetched_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SportsbookScraper(ABC):
    sportsbook: str
    bookmaker_title: str

    @abstractmethod
    def fetch_pitcher_props(self, target_date: str) -> pd.DataFrame:
        raise NotImplementedError


def fetched_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json_url(url: str, timeout: int = 30) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            import json

            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ScraperUnavailable(f"HTTP {exc.code} from {url}") from exc
    except URLError as exc:
        raise ScraperUnavailable(f"Could not reach {url}: {exc}") from exc


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", "", str(name).lower())
    return re.sub(r"\s+", " ", normalized).strip()


def parse_american_odds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("−", "-")
    match = re.search(r"([+-]?\d+)", text)
    if not match:
        return None
    return float(match.group(1))


def append_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def map_scraped_odds_to_probables(odds: pd.DataFrame, probable_pitchers: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        return odds

    prob = probable_pitchers.copy()
    prob["game_date"] = pd.to_datetime(prob["game_date"]).dt.date.astype(str)
    prob["name_key"] = prob["pitcher_name"].map(normalize_name)
    prob = prob[["game_date", "pitcher_id", "pitcher_name", "team", "opponent", "name_key"]]

    out = odds.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date.astype(str)
    out["name_key"] = out["player_name"].map(normalize_name)
    out = out.merge(
        prob,
        on=["game_date", "name_key"],
        how="left",
        suffixes=("", "_matched"),
    )
    return out.drop(columns=["name_key"])


def best_scraped_lines(odds: pd.DataFrame) -> pd.DataFrame:
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
                "over_bookmaker": over_row.get("bookmaker", over_row.get("sportsbook")),
                "under_bookmaker": under_row.get("bookmaker", under_row.get("sportsbook")),
                "player_name": over_row["player_name"],
                "pitcher_name": over_row.get("pitcher_name", over_row["player_name"]),
                "fetched_at": over_row.get("fetched_at"),
            }
        )
    return pd.DataFrame(rows).sort_values(["game_date", "pitcher_id", "market", "line"])
