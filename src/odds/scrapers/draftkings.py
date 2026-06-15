from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable

import pandas as pd

from src.odds.scrapers.base import (
    OddsRow,
    ScraperUnavailable,
    SportsbookScraper,
    fetched_at_utc,
    parse_american_odds,
    read_json_url,
)


MARKET_NAME_MAP = {
    "pitcher strikeouts": "strikeouts",
    "pitcher strikeout": "strikeouts",
    "player strikeouts": "strikeouts",
    "strikeouts": "strikeouts",
    "pitcher walks": "walks",
    "pitcher walk": "walks",
    "walks allowed": "walks",
    "pitcher hits allowed": "hits_allowed",
    "hits allowed": "hits_allowed",
}


class DraftKingsScraper(SportsbookScraper):
    sportsbook = "draftkings"
    bookmaker_title = "DraftKings"

    # MLB event group ID has historically been 84240. DraftKings can change
    # category/offer-group structure, so this adapter parses the whole payload.
    event_group_id = "84240"
    url = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{event_group_id}?format=json"
    page_url = "https://sportsbook.draftkings.com/leagues/baseball/mlb"

    def fetch_pitcher_props(self, target_date: str) -> pd.DataFrame:
        try:
            payload = read_json_url(self.url.format(event_group_id=self.event_group_id))
        except ScraperUnavailable:
            payload = self._fetch_payload_with_browser()
        rows = self._parse_payload(payload, target_date=target_date)
        return pd.DataFrame([row.to_dict() for row in rows])

    def _fetch_payload_with_browser(self) -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ScraperUnavailable(
                "DraftKings returned 403 and Playwright is not installed. "
                "Run `pip install playwright` and `python -m playwright install chromium`."
            ) from exc

        captured: list[dict[str, Any]] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1365, "height": 900},
            )
            page = context.new_page()

            def capture_response(response: Any) -> None:
                url = response.url
                if "eventgroups" not in url or self.event_group_id not in url:
                    return
                try:
                    text = response.text()
                    captured.append(json.loads(text))
                except Exception:
                    return

            page.on("response", capture_response)
            page.goto(self.page_url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(8000)
            context.close()
            browser.close()

        if not captured:
            raise ScraperUnavailable(
                "DraftKings blocked the direct endpoint and no event-group JSON was captured "
                "from the browser fallback."
            )
        return captured[-1]

    def _parse_payload(self, payload: dict[str, Any], target_date: str) -> list[OddsRow]:
        fetched_at = fetched_at_utc()
        events = self._events_by_id(payload)
        groups: dict[tuple[str, str, str, float, str], dict[str, Any]] = defaultdict(dict)

        for offer in self._walk_dicts(payload):
            market = self._market_from_offer(offer)
            if not market:
                continue

            event_id = str(
                offer.get("eventId")
                or offer.get("event_id")
                or offer.get("eventGroupId")
                or offer.get("providerEventId")
                or ""
            )
            event = events.get(event_id, {})
            game_date = self._game_date(event, offer)
            if game_date != target_date:
                continue

            outcomes = self._extract_outcomes(offer)
            for outcome in outcomes:
                side = self._side(outcome)
                if side not in {"over", "under"}:
                    continue
                player_name = self._player_name(outcome, offer)
                line = self._line(outcome, offer)
                odds = self._odds(outcome)
                if not player_name or line is None or odds is None:
                    continue

                key = (game_date, event_id, market, float(line), player_name)
                groups[key].update(
                    {
                        "game_date": game_date,
                        "sportsbook": self.sportsbook,
                        "bookmaker": self.sportsbook,
                        "bookmaker_title": self.bookmaker_title,
                        "market": market,
                        "player_name": player_name,
                        "line": float(line),
                        "event_id": event_id or None,
                        "commence_time": event.get("startDate") or event.get("startTime"),
                        "home_team": event.get("teamName2") or event.get("homeTeamName"),
                        "away_team": event.get("teamName1") or event.get("awayTeamName"),
                        "fetched_at": fetched_at,
                    }
                )
                groups[key][f"{side}_odds"] = odds

        rows = []
        for values in groups.values():
            rows.append(
                OddsRow(
                    over_odds=values.get("over_odds"),
                    under_odds=values.get("under_odds"),
                    **{k: v for k, v in values.items() if k not in {"over_odds", "under_odds"}},
                )
            )
        return rows

    def _events_by_id(self, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        events = {}
        for item in self._walk_dicts(payload):
            if "eventId" in item and ("startDate" in item or "teamName1" in item or "teamName2" in item):
                events[str(item["eventId"])] = item
            if "id" in item and ("startDate" in item or "teamName1" in item or "teamName2" in item):
                events[str(item["id"])] = item
        return events

    def _market_from_offer(self, offer: dict[str, Any]) -> str | None:
        text_parts = []
        for key in ("label", "name", "marketName", "marketTypeName", "subcategoryName", "categoryName"):
            value = offer.get(key)
            if value:
                text_parts.append(str(value))
        text = " ".join(text_parts).lower()
        for phrase, market in MARKET_NAME_MAP.items():
            if phrase in text:
                return market
        return None

    def _extract_outcomes(self, offer: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("outcomes", "selections", "offers"):
            value = offer.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                # Nested offer lists are common; keep walking until we reach selections.
                if any(self._side(item) in {"over", "under"} for item in value):
                    return value
        return []

    def _side(self, outcome: dict[str, Any]) -> str | None:
        text = " ".join(
            str(outcome.get(key, ""))
            for key in ("label", "name", "outcomeLabel", "selectionName", "displayLabel")
        ).lower()
        if re.search(r"\bover\b|\bo\b", text):
            return "over"
        if re.search(r"\bunder\b|\bu\b", text):
            return "under"
        return None

    def _player_name(self, outcome: dict[str, Any], offer: dict[str, Any]) -> str | None:
        for source in (outcome, offer):
            for key in ("participant", "participantName", "playerName", "nameIdentifier", "description"):
                value = source.get(key)
                if value and not self._looks_like_side_or_market(str(value)):
                    return str(value).strip()

        label = str(outcome.get("label") or outcome.get("name") or offer.get("label") or "")
        label = re.sub(r"\b(over|under)\b", "", label, flags=re.I)
        label = re.sub(r"\d+(\.\d+)?", "", label).strip(" -:")
        return label or None

    def _looks_like_side_or_market(self, value: str) -> bool:
        text = value.lower()
        return text in {"over", "under"} or any(phrase in text for phrase in MARKET_NAME_MAP)

    def _line(self, outcome: dict[str, Any], offer: dict[str, Any]) -> float | None:
        for source in (outcome, offer):
            for key in ("line", "points", "point", "handicap"):
                value = source.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        text = " ".join(str(outcome.get(key, "")) for key in ("label", "name", "displayLabel"))
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else None

    def _odds(self, outcome: dict[str, Any]) -> float | None:
        for key in ("oddsAmerican", "americanOdds", "displayOdds", "odds"):
            odds = parse_american_odds(outcome.get(key))
            if odds is not None:
                return odds
        return None

    def _game_date(self, event: dict[str, Any], offer: dict[str, Any]) -> str | None:
        value = event.get("startDate") or event.get("startTime") or offer.get("startDate")
        if not value:
            return None
        return pd.to_datetime(value, utc=True).tz_convert(None).date().isoformat()

    def _walk_dicts(self, value: Any) -> Iterable[dict[str, Any]]:
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._walk_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._walk_dicts(child)
