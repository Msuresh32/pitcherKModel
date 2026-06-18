import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from mlb_identity import normalize_name, normalize_team


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
DEFAULT_MARKETS = "pitcher_strikeouts,pitcher_strikeouts_alternate"


def american_to_prob(odds):
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def parse_utc_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_api_key():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is missing from .env")
    return api_key


def get_json(session, url, params, timeout=30):
    response = session.get(url, params=params, timeout=timeout)
    credit_headers = {
        key.lower(): value
        for key, value in response.headers.items()
        if key.lower().startswith("x-requests-")
    }
    if response.status_code != 200:
        raise RuntimeError(f"Odds API {response.status_code}: {response.text[:500]}")
    return response.json(), credit_headers


def date_range(start_date, end_date):
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    while current <= end:
        yield current
        current += timedelta(days=1)


def choose_snapshot_time(event, snapshot_mode, event_scan_hour_utc, minutes_before):
    if snapshot_mode == "closing":
        return iso_z(parse_utc_timestamp(event["commence_time"]) - timedelta(minutes=minutes_before))
    game_date = parse_utc_timestamp(event["commence_time"]).date()
    return f"{game_date:%Y-%m-%d}T{event_scan_hour_utc:02d}:00:00Z"


def fetch_events(session, api_key, game_date, event_scan_hour_utc):
    snapshot = f"{game_date:%Y-%m-%d}T{event_scan_hour_utc:02d}:00:00Z"
    payload, headers = get_json(
        session,
        f"{BASE_URL}/historical/sports/{SPORT}/events",
        {"apiKey": api_key, "date": snapshot},
    )
    slate_start = datetime(game_date.year, game_date.month, game_date.day, tzinfo=timezone.utc)
    slate_end = slate_start + timedelta(hours=36)
    events = [
        event
        for event in payload.get("data", [])
        if slate_start <= parse_utc_timestamp(event["commence_time"]) < slate_end
    ]
    return events, headers


def fetch_event_odds(session, api_key, event, args):
    snapshot_time = choose_snapshot_time(
        event,
        args.snapshot_mode,
        args.event_scan_hour_utc,
        args.minutes_before,
    )
    payload, headers = get_json(
        session,
        f"{BASE_URL}/historical/sports/{SPORT}/events/{event['id']}/odds",
        {
            "apiKey": api_key,
            "regions": args.regions,
            "bookmakers": args.bookmaker,
            "markets": args.markets,
            "oddsFormat": "american",
            "date": snapshot_time,
        },
    )
    return payload.get("data", {}), headers, snapshot_time


def extract_rows(event, odds_payload, snapshot_time, slate_date):
    rows = []
    home_team = normalize_team(event.get("home_team"))
    away_team = normalize_team(event.get("away_team"))
    game_date = slate_date.isoformat()

    for bookmaker in odds_payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            last_update = market.get("last_update")
            for outcome in market.get("outcomes", []):
                pitcher_name = outcome.get("description") or outcome.get("name")
                side = outcome.get("name")
                line = outcome.get("point")
                price = outcome.get("price")
                if not pitcher_name or side not in {"Over", "Under"} or line is None or price is None:
                    continue
                rows.append(
                    {
                        "game_date": game_date,
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "snapshot_time": snapshot_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker.get("key"),
                        "market": market_key,
                        "last_update": last_update,
                        "pitcher_name": pitcher_name,
                        "clean_name": normalize_name(pitcher_name),
                        "side": side.upper(),
                        "line": float(line),
                        "american_odds": int(price),
                        "implied_prob": american_to_prob(price),
                    }
                )
    return rows


def build_mainlines(raw_df):
    if raw_df.empty:
        return pd.DataFrame()

    main = raw_df[raw_df["market"] == "pitcher_strikeouts"].copy()
    if main.empty:
        return pd.DataFrame()

    pivot = (
        main.pivot_table(
            index=[
                "game_date",
                "event_id",
                "commence_time",
                "snapshot_time",
                "home_team",
                "away_team",
                "bookmaker",
                "pitcher_name",
                "clean_name",
                "line",
            ],
            columns="side",
            values="american_odds",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"OVER": "over_odds", "UNDER": "under_odds"})
    )
    for column in ["over_odds", "under_odds"]:
        if column not in pivot:
            pivot[column] = pd.NA
    pivot["over_implied_prob"] = pivot["over_odds"].dropna().map(american_to_prob)
    pivot["under_implied_prob"] = pivot["under_odds"].dropna().map(american_to_prob)
    prob_sum = pivot["over_implied_prob"] + pivot["under_implied_prob"]
    pivot["market_no_vig_over"] = pivot["over_implied_prob"] / prob_sum
    return pivot


def append_or_write(df, output_path, append, dedupe_columns):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if append and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset=dedupe_columns, keep="last")
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest historical DraftKings MLB pitcher strikeout odds from The Odds API.")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument("--bookmaker", default="draftkings")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--markets", default=DEFAULT_MARKETS)
    parser.add_argument("--snapshot-mode", choices=["closing", "fixed"], default="closing")
    parser.add_argument("--minutes-before", type=int, default=10)
    parser.add_argument("--event-scan-hour-utc", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--flush-each-date", action="store_true")
    parser.add_argument("--quiet-events", action="store_true")
    parser.add_argument("--raw-out", default="data/historical_dk_pitcher_k_odds_raw.csv")
    parser.add_argument("--main-out", default="data/historical_dk_pitcher_k_mainlines.csv")
    args = parser.parse_args()

    api_key = get_api_key()
    session = requests.Session()
    raw_rows = []
    event_count = 0
    last_headers = {}
    total_credits = 0
    total_raw_rows = 0
    total_mainline_rows = 0

    for game_date in date_range(args.start, args.end):
        events, headers = fetch_events(session, api_key, game_date, args.event_scan_hour_utc)
        last_headers = headers or last_headers
        total_credits += int(headers.get("x-requests-last", 0) or 0)
        print(f"{game_date}: events={len(events)}")

        for event in events:
            if args.max_events is not None and event_count >= args.max_events:
                break
            event_count += 1
            try:
                odds_payload, headers, snapshot_time = fetch_event_odds(session, api_key, event, args)
            except RuntimeError as exc:
                print(f"  skip {event.get('away_team')} @ {event.get('home_team')}: {exc}")
                continue

            last_headers = headers or last_headers
            total_credits += int(headers.get("x-requests-last", 0) or 0)
            rows = extract_rows(event, odds_payload, snapshot_time, game_date)
            raw_rows.extend(rows)
            total_raw_rows += len(rows)
            pitchers = len({row["clean_name"] for row in rows})
            markets = sorted({row["market"] for row in rows})
            if not args.quiet_events:
                print(f"  {event.get('away_team')} @ {event.get('home_team')}: rows={len(rows)} pitchers={pitchers} markets={markets}")
            if args.sleep:
                time.sleep(args.sleep)

        if args.flush_each_date and raw_rows and not args.dry_run:
            raw_df = pd.DataFrame(raw_rows)
            raw_df = append_or_write(
                raw_df,
                args.raw_out,
                True,
                ["event_id", "snapshot_time", "bookmaker", "market", "clean_name", "side", "line"],
            )
            main_df = build_mainlines(raw_df)
            if not main_df.empty:
                main_file_df = append_or_write(
                    main_df,
                    args.main_out,
                    True,
                    ["event_id", "snapshot_time", "bookmaker", "clean_name", "line"],
                )
                total_mainline_rows = len(main_file_df)
            print(f"  checkpoint wrote through {game_date}: raw_rows_file={len(raw_df)}")
            raw_rows = []

        if args.max_events is not None and event_count >= args.max_events:
            break

    raw_df = pd.DataFrame(raw_rows)
    main_df = build_mainlines(raw_df)
    if not args.flush_each_date:
        total_mainline_rows = len(main_df)

    print("")
    print(f"events_fetched={event_count}")
    print(f"raw_rows={total_raw_rows}")
    print(f"mainline_rows={total_mainline_rows}")
    print(f"credits_used_this_run={total_credits}")
    if last_headers:
        print(f"requests_remaining={last_headers.get('x-requests-remaining', 'unknown')}")

    if args.dry_run:
        print("dry_run=true; skipped CSV writes")
        if not raw_df.empty:
            preview_cols = ["game_date", "pitcher_name", "market", "side", "line", "american_odds"]
            print(raw_df[preview_cols].head(20).to_string(index=False))
        return

    if raw_df.empty:
        print("No rows returned; skipped CSV writes.")
        return

    raw_df = append_or_write(
        raw_df,
        args.raw_out,
        args.append,
        ["event_id", "snapshot_time", "bookmaker", "market", "clean_name", "side", "line"],
    )
    main_df = build_mainlines(raw_df)
    main_df = append_or_write(
        main_df,
        args.main_out,
        args.append,
        ["event_id", "snapshot_time", "bookmaker", "clean_name", "line"],
    )
    print(f"wrote_raw={args.raw_out} rows={len(raw_df)}")
    print(f"wrote_mainlines={args.main_out} rows={len(main_df)}")


if __name__ == "__main__":
    main()
