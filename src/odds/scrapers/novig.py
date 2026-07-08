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
    "strikeouts thrown": "strikeouts",
    "pitcher strikeouts": "strikeouts",
    "strikeouts": "strikeouts",
    "walks allowed": "walks",
    "pitcher walks": "walks",
    "hits allowed": "hits_allowed",
    "pitcher hits allowed": "hits_allowed",
}

MARKET_TYPE_MAP = {
    "STRIKEOUTS_OVER_UNDER": "strikeouts",
    "PITCHER_STRIKEOUTS": "strikeouts",
    "STRIKEOUTS": "strikeouts",
    "PITCHER_WALKS": "walks",
    "WALKS_OVER_UNDER": "walks",
    "HITS_ALLOWED": "hits_allowed",
}

BATCH_URL_FRAGMENT = "/nbx/v1/markets/book/batch"
HOME_PAGE_URL     = "https://api.novig.us/recs/v1/home/page?displayUnseededMarkets=false"
EVENT_BASE_URL    = "https://novig.com/event-markets/"
AUTH_STATE_FILE   = "data/auth/novig_state.json"


class NovigScraper(SportsbookScraper):
    sportsbook     = "novig"
    bookmaker_title = "NoVig"

    def fetch_pitcher_props(self, target_date: str) -> pd.DataFrame:
        event_ids = self._discover_mlb_event_ids()
        if not event_ids:
            raise ScraperUnavailable("No MLB events found on NoVig home page.")

        payloads = self._fetch_payloads_with_browser(event_ids)
        rows = self._parse_payloads(payloads, target_date=target_date)
        df = pd.DataFrame([row.to_dict() for row in rows])
        return df

    # ── Event discovery (public REST, no browser) ─────────────────────────────

    def _discover_mlb_event_ids(self) -> list[str]:
        data = read_json_url(HOME_PAGE_URL)
        seen: dict[str, None] = {}
        for comp in data.get("components", []):
            if comp.get("league") == "MLB" and comp.get("eventId"):
                seen[comp["eventId"]] = None
        return list(seen.keys())

    # ── Browser: navigate to each event, capture pitcher-props batch ──────────

    def _fetch_payloads_with_browser(self, event_ids: list[str]) -> list[dict[str, Any]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ScraperUnavailable(
                "NoVig requires a browser session. Run `pip install playwright` "
                "and `python -m playwright install chromium`."
            ) from exc

        import os
        auth_path = AUTH_STATE_FILE
        has_auth = os.path.exists(auth_path)
        if not has_auth:
            raise ScraperUnavailable(
                f"NoVig session not found at {auth_path}. "
                "Run `python scripts/novig_login.py` once to save your login session."
            )

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
                viewport={"width": 1440, "height": 900},
                storage_state=auth_path,
            )
            page = context.new_page()

            def capture_response(response: Any) -> None:
                if BATCH_URL_FRAGMENT not in response.url:
                    return
                try:
                    captured.append(json.loads(response.text()))
                except Exception:
                    return

            page.on("response", capture_response)

            for event_id in event_ids:
                url = EVENT_BASE_URL + event_id
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    self._open_pitcher_props_tab(page)
                    page.wait_for_timeout(4000)
                    print(f"[novig] Captured {len(captured)} batch payloads after {event_id[:8]}...")
                except Exception as exc:
                    print(f"[novig] Error navigating to {event_id[:8]}...: {exc}")
                    continue

            context.close()
            browser.close()

        if not captured:
            raise ScraperUnavailable(
                "No NoVig batch-odds responses were captured. "
                "The site may require login or the Pitcher Props tab selector changed."
            )
        return captured

    def _open_pitcher_props_tab(self, page: Any) -> None:
        for selector in (
            "text=Pitcher Props",
            "button:has-text('Pitcher Props')",
            "[role=tab]:has-text('Pitcher Props')",
        ):
            try:
                locator = page.locator(selector).first
                if locator.is_visible(timeout=3000):
                    locator.click(timeout=3000)
                    return
            except Exception:
                continue

    # ── Parser ────────────────────────────────────────────────────────────────

    def _parse_payloads(self, payloads: list[Any], target_date: str) -> list[OddsRow]:
        fetched_at = fetched_at_utc()
        groups: dict[tuple[str, str, float], dict[str, Any]] = defaultdict(dict)

        for payload in payloads:
            # Each payload is a list of {market: {...}, ladders: {...}} objects
            entries = payload if isinstance(payload, list) else []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                market_obj = entry.get("market", {})
                ladders    = entry.get("ladders", {})

                market = self._market_type(market_obj)
                if not market:
                    continue

                player_name = self._player_name(market_obj)
                if not player_name:
                    continue

                line = float(market_obj.get("strike", 0) or 0)

                for outcome in market_obj.get("outcomes", []):
                    outcome_id  = outcome.get("id", "")
                    side        = self._side_from_description(outcome.get("description", ""))
                    if side not in {"over", "under"}:
                        continue

                    best_price = self._best_price_from_ladder(ladders.get(outcome_id, {}))
                    if best_price is None:
                        continue

                    american = self._price_to_american(best_price)

                    key = (player_name, market, line)
                    groups[key].update(
                        {
                            "game_date":       target_date,
                            "sportsbook":      self.sportsbook,
                            "bookmaker":       self.sportsbook,
                            "bookmaker_title": self.bookmaker_title,
                            "market":          market,
                            "player_name":     player_name,
                            "line":            line,
                            "event_id":        market_obj.get("id"),
                            "fetched_at":      fetched_at,
                        }
                    )
                    groups[key][f"{side}_odds"] = american

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

    def _market_type(self, market_obj: dict[str, Any]) -> str | None:
        mtype = str(market_obj.get("type", "")).upper()
        if mtype in MARKET_TYPE_MAP:
            return MARKET_TYPE_MAP[mtype]

        description = str(market_obj.get("description", "")).lower()
        for phrase, market in MARKET_NAME_MAP.items():
            if phrase in description:
                return market
        return None

    def _player_name(self, market_obj: dict[str, Any]) -> str | None:
        comp = market_obj.get("competitor", {})
        if comp and comp.get("name"):
            return str(comp["name"]).strip()
        return None

    def _side_from_description(self, description: str) -> str | None:
        text = description.lower().strip()
        if text in {"over", "yes", "o"}:
            return "over"
        if text in {"under", "no", "u"}:
            return "under"
        if "over" in text:
            return "over"
        if "under" in text:
            return "under"
        return None

    def _best_price_from_ladder(self, ladder: dict[str, Any]) -> float | None:
        """Return best available ask price (lowest ask = cheapest to buy this outcome).

        NoVig ladders have bids and asks keyed by outcomeId.
        asks[0] is the best (lowest) ask: what you'd pay to take this side.
        If no asks, fall back to best (highest) bid as a proxy for fair value.
        """
        asks = ladder.get("asks", [])
        if asks:
            prices = [float(a["price"]) for a in asks if "price" in a]
            if prices:
                return min(prices)

        bids = ladder.get("bids", [])
        if bids:
            prices = [float(b["price"]) for b in bids if "price" in b]
            if prices:
                return max(prices)

        return None

    @staticmethod
    def _price_to_american(price: float) -> float:
        """Convert NoVig decimal probability to American odds.

        price = implied probability (e.g. 0.45 = 45%).
        decimal_odds = 1 / price.
        American: if decimal >= 2: (decimal - 1) * 100; else: -100 / (decimal - 1).
        """
        if price <= 0 or price >= 1:
            return float("nan")
        decimal = 1.0 / price
        if decimal >= 2.0:
            return round((decimal - 1.0) * 100, 1)
        else:
            return round(-100.0 / (decimal - 1.0), 1)
