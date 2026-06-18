import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scratch"))

from fetch_historical_pitcher_k_clv import american_to_prob
from mlb_identity import normalize_name


SPORT = "baseball_mlb"
MARKET = "pitcher_strikeouts"
BASE_URL = "https://api.the-odds-api.com/v4"
OUT_DIR = BASE_DIR / "scratch" / "scanners"


def prob_to_american(prob: float) -> str:
    if pd.isna(prob) or prob <= 0 or prob >= 1:
        return ""
    if prob >= 0.5:
        return str(int(round(-100 * prob / (1 - prob))))
    return f"+{int(round(100 * (1 - prob) / prob))}"


def prob_label(prob: float) -> str:
    if pd.isna(prob):
        return ""
    return f"{prob * 100:.1f}c ({prob_to_american(prob)})"


def get_api_key() -> str:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is missing.")
    return api_key


def get_json(session: requests.Session, url: str, params: dict) -> dict:
    for attempt in range(4):
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(1.5 + attempt * 2)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"Odds API error {response.status_code}: {response.text[:300]}")
        return response.json()
    raise RuntimeError(f"Odds API unavailable after retries: {url}")


def fetch_events(session: requests.Session, api_key: str) -> list[dict]:
    return get_json(session, f"{BASE_URL}/sports/{SPORT}/events", {"apiKey": api_key})


def fetch_event_odds(session: requests.Session, api_key: str, event_id: str) -> dict:
    return get_json(
        session,
        f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
        {
            "apiKey": api_key,
            "regions": "us,us2,uk,eu,au",
            "markets": MARKET,
            "oddsFormat": "american",
        },
    )


def extract_rows(event: dict, odds_payload: dict) -> list[dict]:
    rows = []
    for bookmaker in odds_payload.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            if market.get("key") != MARKET:
                continue
            for outcome in market.get("outcomes", []) or []:
                pitcher_name = outcome.get("description") or outcome.get("name")
                side = str(outcome.get("name", "")).upper()
                point = outcome.get("point")
                price = outcome.get("price")
                if not pitcher_name or side not in {"OVER", "UNDER"} or point is None or price is None:
                    continue
                rows.append(
                    {
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "away_team": event.get("away_team"),
                        "home_team": event.get("home_team"),
                        "bookmaker": bookmaker.get("key"),
                        "book_title": bookmaker.get("title"),
                        "last_update": market.get("last_update"),
                        "pitcher": pitcher_name,
                        "clean_name": normalize_name(pitcher_name),
                        "side": side,
                        "line": float(point),
                        "american_odds": int(price),
                        "implied_prob": american_to_prob(int(price)),
                    }
                )
    return rows


def pivot_market(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    pivot = (
        rows.pivot_table(
            index=[
                "event_id",
                "commence_time",
                "away_team",
                "home_team",
                "bookmaker",
                "book_title",
                "clean_name",
                "pitcher",
                "line",
            ],
            columns="side",
            values="american_odds",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"OVER": "over_odds", "UNDER": "under_odds"})
    )
    for col in ["over_odds", "under_odds"]:
        if col not in pivot:
            pivot[col] = np.nan
    pivot["over_imp"] = pivot["over_odds"].map(lambda x: american_to_prob(x) if pd.notna(x) else np.nan)
    pivot["under_imp"] = pivot["under_odds"].map(lambda x: american_to_prob(x) if pd.notna(x) else np.nan)
    denom = pivot["over_imp"] + pivot["under_imp"]
    pivot["no_vig_over"] = pivot["over_imp"] / denom.replace(0, np.nan)
    pivot["no_vig_under"] = 1.0 - pivot["no_vig_over"]
    return pivot.dropna(subset=["no_vig_over", "no_vig_under"])


def choose_rows(market: pd.DataFrame, min_consensus_books: int) -> pd.DataFrame:
    key = ["event_id", "clean_name", "line"]
    consensus = (
        market.groupby(key)
        .agg(
            consensus_over=("no_vig_over", "mean"),
            consensus_under=("no_vig_under", "mean"),
            book_coverage=("bookmaker", "nunique"),
            books=("bookmaker", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
    )
    pin = market[market["bookmaker"].eq("pinnacle")][
        key + ["over_odds", "under_odds", "no_vig_over", "no_vig_under"]
    ].rename(
        columns={
            "over_odds": "pin_over_odds",
            "under_odds": "pin_under_odds",
            "no_vig_over": "pin_over",
            "no_vig_under": "pin_under",
        }
    )
    base = (
        market.sort_values(["bookmaker"])
        .drop_duplicates(key)
        [["event_id", "commence_time", "away_team", "home_team", "clean_name", "pitcher", "line"]]
    )
    out = base.merge(consensus, on=key, how="left").merge(pin, on=key, how="left")
    out["source"] = np.where(out["pin_under"].notna(), "Pinnacle", "Consensus")
    out["sharp_over"] = out["pin_over"].fillna(out["consensus_over"])
    out["sharp_under"] = out["pin_under"].fillna(out["consensus_under"])
    out["trigger_over_odds"] = out["pin_over_odds"]
    out["valid_benchmark"] = out["pin_under"].notna() | (out["book_coverage"] >= min_consensus_books)
    return out[out["valid_benchmark"]].copy()


def scan(args: argparse.Namespace) -> pd.DataFrame:
    api_key = get_api_key()
    session = requests.Session()
    events = fetch_events(session, api_key)
    if args.slate_date:
        events = [e for e in events if str(e.get("commence_time", ""))[:10] in {args.slate_date, args.next_utc_date}]
    all_rows = []
    for event in events:
        payload = fetch_event_odds(session, api_key, event["id"])
        all_rows.extend(extract_rows(event, payload))
    raw = pd.DataFrame(all_rows)
    if raw.empty:
        return pd.DataFrame()
    market = pivot_market(raw)
    chosen = choose_rows(market, args.min_consensus_books)
    qualified = chosen[(chosen["sharp_over"] >= args.trigger_over_prob)].copy()
    if qualified.empty:
        return qualified
    qualified["priority"] = (qualified["sharp_over"] >= args.priority_over_prob) | (
        qualified["trigger_over_odds"].fillna(999) <= args.priority_over_american
    )
    qualified["edge_buffer"] = np.where(qualified["source"].eq("Pinnacle"), args.pinnacle_edge_buffer, args.consensus_edge_buffer)
    qualified["max_bet_prob"] = (qualified["sharp_under"] + qualified["edge_buffer"]).clip(0.01, 0.99)
    qualified["sharp_fair_under"] = qualified["sharp_under"].map(prob_label)
    qualified["max_bet_price"] = qualified["max_bet_prob"].map(prob_label)
    qualified["over_no_vig_prob"] = qualified["sharp_over"]
    qualified["over_price"] = qualified["trigger_over_odds"].map(lambda x: "" if pd.isna(x) else str(int(x)))
    qualified.loc[qualified["over_price"].eq(""), "over_price"] = "consensus"
    qualified["matchup"] = qualified["away_team"] + " @ " + qualified["home_team"]
    cols = [
        "priority",
        "pitcher",
        "matchup",
        "commence_time",
        "line",
        "over_price",
        "over_no_vig_prob",
        "source",
        "sharp_fair_under",
        "max_bet_price",
        "book_coverage",
    ]
    return qualified[cols].sort_values(["priority", "over_no_vig_prob", "book_coverage"], ascending=[False, False, False])


def main() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    next_utc_date = datetime.now(UTC).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Scan live MLB K props for the validated fade-heavy-over under pocket.")
    parser.add_argument("--slate-date", default=today)
    parser.add_argument("--next-utc-date", default=next_utc_date)
    parser.add_argument("--min-consensus-books", type=int, default=5)
    parser.add_argument("--trigger-over-prob", type=float, default=0.55)
    parser.add_argument("--priority-over-prob", type=float, default=0.58)
    parser.add_argument("--priority-over-american", type=int, default=-150)
    parser.add_argument("--pinnacle-edge-buffer", type=float, default=0.025)
    parser.add_argument("--consensus-edge-buffer", type=float, default=0.015)
    parser.add_argument("--out", default=str(OUT_DIR / "heavy_over_under_scanner_latest.csv"))
    args = parser.parse_args()

    rows = scan(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False)
    if rows.empty:
        print("No qualifying unders.")
        print(f"out={args.out}")
        return
    display = rows.copy()
    display["over_no_vig_prob"] = (display["over_no_vig_prob"] * 100).map(lambda x: f"{x:.1f}%")
    display["priority"] = display["priority"].map(lambda x: "YES" if x else "NO")
    display["line"] = display["line"].map(lambda x: f"{x:.1f}")
    print(display.to_string(index=False))
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
