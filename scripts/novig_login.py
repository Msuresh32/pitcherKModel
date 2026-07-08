"""
novig_login.py — One-time login session saver for NoVig.

Launches a headed (visible) Chromium browser so you can log into NoVig manually.
After you're logged in and can see odds, press Enter in this terminal.
The session (cookies + localStorage) is saved to data/auth/novig_state.json.
All subsequent scraper runs load that saved session automatically (no re-login).

Usage:
    python scripts/novig_login.py
"""
import sys
from pathlib import Path

AUTH_FILE = Path("data/auth/novig_state.json")
LOGIN_URL = "https://novig.com/"


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")
        sys.exit(1)

    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NoVig Login Session Saver")
    print("=" * 60)
    print()
    print("A browser window will open. Log into your NoVig account.")
    print("Once you can see odds (e.g. open any game's Pitcher Props),")
    print("come back here and press Enter.")
    print()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)

        input("Browser is open. Log in, verify you can see odds, then press Enter here: ")

        context.storage_state(path=str(AUTH_FILE))
        print(f"\nSession saved to {AUTH_FILE}")
        print("The scraper will now use this session automatically.")

        context.close()
        browser.close()


if __name__ == "__main__":
    main()
