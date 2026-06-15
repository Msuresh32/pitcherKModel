from src.odds.scrapers.base import ScraperUnavailable, SportsbookScraper
from src.odds.scrapers.draftkings import DraftKingsScraper

SCRAPER_REGISTRY = {
    "draftkings": DraftKingsScraper,
}

UNSUPPORTED_SPORTSBOOKS = {"fanduel", "betmgm", "caesars", "pinnacle"}

__all__ = [
    "DraftKingsScraper",
    "SCRAPER_REGISTRY",
    "ScraperUnavailable",
    "SportsbookScraper",
    "UNSUPPORTED_SPORTSBOOKS",
]
