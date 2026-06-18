import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from ingest_historical_dk_pitcher_k_odds import american_to_prob, get_json, parse_utc_timestamp
from mlb_identity import normalize_name, normalize_team


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
DEFAULT_MARKETS = "pitcher_strikeouts"


def date_range(start_date, end_date):
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    while current <= end:
        yield current
        current += timedelta(days=1)


def iso_z(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_api_key():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY is missing.")
    return api_key


def fetch_events(session, api_key, slate_date, event_scan_hour_utc):
    snapshot = f"{slate_date:%Y-%m-%d}T{event_scan_hour_utc:02d}:00:00Z"
    payload, headers = get_json(
        session,
        f"{BASE_URL}/historical/sports/{SPORT}/events",
        {"apiKey": api_key, "date": snapshot},
    )
    slate_start = datetime(slate_date.year, slate_date.month, slate_date.day, tzinfo=timezone.utc)
    slate_end = slate_start + timedelta(hours=36)
    events = [
        event
        for event in payload.get("data", [])
        if slate_start <= parse_utc_timestamp(event["commence_time"]) < slate_end
    ]
    return events, headers


def snapshot_specs(slate_date, event, fixed_hours_utc, closing_minutes_before, include_close=True):
    commence = parse_utc_timestamp(event["commence_time"])
    specs = []
    for hour in fixed_hours_utc:
        snap = datetime(slate_date.year, slate_date.month, slate_date.day, hour, tzinfo=timezone.utc)
        label = f"h{hour:02d}"
        if snap < commence - timedelta(minutes=5):
            specs.append((label, snap))
    if include_close:
        close_snap = commence - timedelta(minutes=closing_minutes_before)
        specs.append((f"close_{closing_minutes_before}m", close_snap))
    deduped = []
    seen = set()
    for label, snap in specs:
        key = iso_z(snap)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, snap))
    return deduped


def fetch_event_odds(session, api_key, event, snapshot_time, args):
    payload, headers = get_json(
        session,
        f"{BASE_URL}/historical/sports/{SPORT}/events/{event['id']}/odds",
        {
            "apiKey": api_key,
            "regions": args.regions,
            "bookmakers": args.bookmaker,
            "markets": args.markets,
            "oddsFormat": "american",
            "date": iso_z(snapshot_time),
        },
    )
    return payload.get("data", {}), headers


def extract_rows(slate_date, event, odds_payload, snapshot_label, snapshot_time, allowed_markets=None):
    rows = []
    allowed_markets = set(allowed_markets or [DEFAULT_MARKETS])
    home_team = normalize_team(event.get("home_team"))
    away_team = normalize_team(event.get("away_team"))
    for bookmaker in odds_payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") not in allowed_markets:
                continue
            for outcome in market.get("outcomes", []):
                pitcher_name = outcome.get("description") or outcome.get("name")
                side = outcome.get("name")
                line = outcome.get("point")
                price = outcome.get("price")
                if not pitcher_name or side not in {"Over", "Under"} or line is None or price is None:
                    continue
                rows.append(
                    {
                        "slate_date": slate_date.isoformat(),
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "snapshot_label": snapshot_label,
                        "snapshot_time": iso_z(snapshot_time),
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker.get("key"),
                        "market": market.get("key"),
                        "last_update": market.get("last_update"),
                        "pitcher_name": pitcher_name,
                        "clean_name": normalize_name(pitcher_name),
                        "side": side.upper(),
                        "line": float(line),
                        "american_odds": int(price),
                        "implied_prob": american_to_prob(price),
                    }
                )
    return rows


def pivot_snapshot(raw):
    if raw.empty:
        return pd.DataFrame()
    pivot = (
        raw.pivot_table(
            index=[
                "slate_date",
                "event_id",
                "commence_time",
                "snapshot_label",
                "snapshot_time",
                "home_team",
                "away_team",
            "bookmaker",
            "market",
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
    for col in ["over_odds", "under_odds"]:
        if col not in pivot:
            pivot[col] = np.nan
    pivot["over_implied_prob"] = pivot["over_odds"].apply(american_to_prob)
    pivot["under_implied_prob"] = pivot["under_odds"].apply(american_to_prob)
    denom = pivot["over_implied_prob"] + pivot["under_implied_prob"]
    pivot["no_vig_over"] = pivot["over_implied_prob"] / denom.replace(0, np.nan)
    pivot["no_vig_under"] = 1.0 - pivot["no_vig_over"]
    return pivot


def build_clv_pairs(main):
    if main.empty:
        return pd.DataFrame()
    close = main[main["snapshot_label"].str.startswith("close_")].copy()
    entries = main[~main["snapshot_label"].str.startswith("close_")].copy()
    if close.empty or entries.empty:
        return pd.DataFrame()

    key = ["slate_date", "event_id", "clean_name", "market", "line"]
    close_cols = key + [
        "snapshot_time",
        "over_odds",
        "under_odds",
        "no_vig_over",
        "no_vig_under",
    ]
    paired = entries.merge(
        close[close_cols].rename(
            columns={
                "snapshot_time": "closing_snapshot_time",
                "over_odds": "closing_over_odds",
                "under_odds": "closing_under_odds",
                "no_vig_over": "closing_no_vig_over",
                "no_vig_under": "closing_no_vig_under",
            }
        ),
        on=key,
        how="left",
    )
    paired["same_line_close_available"] = paired["closing_no_vig_over"].notna()
    paired["over_clv_prob"] = paired["closing_no_vig_over"] - paired["no_vig_over"]
    paired["under_clv_prob"] = paired["closing_no_vig_under"] - paired["no_vig_under"]

    close_line = close.sort_values("snapshot_time").drop_duplicates(
        ["slate_date", "event_id", "clean_name", "market"],
        keep="last",
    )[["slate_date", "event_id", "clean_name", "market", "line"]].rename(columns={"line": "closing_main_line"})
    paired = paired.merge(close_line, on=["slate_date", "event_id", "clean_name", "market"], how="left")
    paired["line_move"] = paired["closing_main_line"] - paired["line"]
    return paired


def summarize_clv(pairs):
    if pairs.empty:
        return pd.DataFrame()
    rows = []
    for label, group in pairs.groupby("snapshot_label"):
        same = group[group["same_line_close_available"]].copy()
        rows.append(
            {
                "snapshot_label": label,
                "rows": len(group),
                "same_line_rows": len(same),
                "avg_abs_line_move": group["line_move"].abs().mean(),
                "avg_over_clv_prob": same["over_clv_prob"].mean() if not same.empty else np.nan,
                "avg_under_clv_prob": same["under_clv_prob"].mean() if not same.empty else np.nan,
                "over_beat_close_rate": (same["over_clv_prob"] > 0).mean() if not same.empty else np.nan,
                "under_beat_close_rate": (same["under_clv_prob"] > 0).mean() if not same.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch historical DK pitcher-K multi-snapshot prices for CLV testing.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bookmaker", default="draftkings")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--markets", default=DEFAULT_MARKETS)
    parser.add_argument("--fixed-hours-utc", default="16,19,22")
    parser.add_argument("--closing-minutes-before", type=int, default=10)
    parser.add_argument("--no-close", action="store_true")
    parser.add_argument("--event-scan-hour-utc", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--out-prefix", default=str(BASE_DIR / "scratch" / "clv" / "dk_pitcher_k_clv"))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    fixed_hours = [int(x.strip()) for x in args.fixed_hours_utc.split(",") if x.strip()]
    allowed_markets = [x.strip() for x in args.markets.split(",") if x.strip()]
    api_key = get_api_key()
    session = requests.Session()
    rows = []
    event_count = 0
    credit_total = 0
    last_headers = {}

    for slate_date in date_range(args.start, args.end):
        events, headers = fetch_events(session, api_key, slate_date, args.event_scan_hour_utc)
        credit_total += int(headers.get("x-requests-last", 0) or 0)
        last_headers = headers or last_headers
        if not args.quiet:
            print(f"{slate_date}: events={len(events)}")
        for event in events:
            if args.max_events is not None and event_count >= args.max_events:
                break
            event_count += 1
            specs = snapshot_specs(
                slate_date,
                event,
                fixed_hours,
                args.closing_minutes_before,
                include_close=not args.no_close,
            )
            for label, snapshot_time in specs:
                try:
                    odds_payload, headers = fetch_event_odds(session, api_key, event, snapshot_time, args)
                except RuntimeError as exc:
                    print(f"  skip {label} {event.get('away_team')} @ {event.get('home_team')}: {exc}")
                    continue
                credit_total += int(headers.get("x-requests-last", 0) or 0)
                last_headers = headers or last_headers
                new_rows = extract_rows(slate_date, event, odds_payload, label, snapshot_time, allowed_markets)
                rows.extend(new_rows)
                if not args.quiet:
                    print(f"  {label} {event.get('away_team')} @ {event.get('home_team')}: rows={len(new_rows)}")
                if args.sleep:
                    time.sleep(args.sleep)
        if args.max_events is not None and event_count >= args.max_events:
            break

    raw = pd.DataFrame(rows)
    main = pivot_snapshot(raw)
    pairs = build_clv_pairs(main)
    summary = summarize_clv(pairs)

    print("")
    print(f"events_fetched={event_count}")
    print(f"raw_rows={len(raw)}")
    print(f"main_rows={len(main)}")
    print(f"clv_pairs={len(pairs)}")
    print(f"credits_used_this_run={credit_total}")
    if last_headers:
        print(f"requests_remaining={last_headers.get('x-requests-remaining', 'unknown')}")
    if not summary.empty:
        print(summary.to_string(index=False))

    if args.dry_run:
        print("dry_run=true; skipped CSV writes")
        return

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(f"{prefix}_raw.csv", index=False)
    main.to_csv(f"{prefix}_main.csv", index=False)
    pairs.to_csv(f"{prefix}_pairs.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    print(f"wrote {prefix}_raw.csv")
    print(f"wrote {prefix}_main.csv")
    print(f"wrote {prefix}_pairs.csv")
    print(f"wrote {prefix}_summary.csv")


if __name__ == "__main__":
    main()
