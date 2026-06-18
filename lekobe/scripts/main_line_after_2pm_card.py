from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import os
import re
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from kalshi_auth import sign_kalshi_request
from mlb_identity import normalize_name

load_dotenv(BASE / ".env")

API_KEY = os.getenv("THE_ODDS_API_KEY")
SPORT = "baseball_mlb"
ODDS_BASE = "https://api.the-odds-api.com/v4"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
TZ = ZoneInfo("America/Los_Angeles")
TODAY = datetime.now(TZ).date()
CUTOFF = datetime.combine(TODAY, time(14, 0), tzinfo=TZ)
OUT = BASE / "scratch/audits/after_2pm_main_line_qualifiers.csv"
AUDIT = BASE / "scratch/audits/after_2pm_main_line_all_pitchers.csv"


def american_to_prob(odds) -> float | None:
    try:
        odds = float(odds)
    except Exception:
        return None
    if odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def prob_to_american(prob: float) -> int | None:
    if prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def devig_pair(over_odds, under_odds) -> dict | None:
    op = american_to_prob(over_odds)
    up = american_to_prob(under_odds)
    if op is None or up is None or op + up <= 0:
        return None
    return {"over": op / (op + up), "under": up / (op + up)}


def get_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def signed_get(path: str, params: dict | None = None) -> dict:
    headers = sign_kalshi_request("GET", "/trade-api/v2" + path)
    response = requests.get(KALSHI_BASE + path, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def cents(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return int(round(numeric * 100)) if numeric <= 1 else int(round(numeric))


def event_date_from_ticker(ticker: str) -> str:
    match = re.search(r"KXMLBKS-(\d{2})([A-Z]{3})(\d{2})", str(ticker))
    if not match:
        return ""
    yy, mon, day = match.groups()
    months = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    return f"20{yy}-{months[mon]:02d}-{int(day):02d}"


def parse_kalshi_market(market: dict) -> dict:
    ticker = str(market.get("ticker") or "")
    text = " ".join(
        str(market.get(key) or "")
        for key in ["title", "subtitle", "yes_sub_title", "no_sub_title", "rules_primary"]
    )
    match = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ .'-]+?)\s*:\s*(\d+)\+", text, flags=re.I)
    pitcher = ""
    target = None
    if match:
        pitcher = match.group(1).strip()
        target = int(match.group(2))
    if target is None:
        suffix = re.search(r"-(\d+)$", ticker)
        target = int(suffix.group(1)) if suffix else None
    return {
        "ticker": ticker,
        "event_date": event_date_from_ticker(ticker),
        "pitcher": pitcher,
        "clean_name": normalize_name(pitcher) if pitcher else "",
        "line": float(target) - 0.5 if target is not None else math.nan,
        "title": market.get("title", ""),
        "no_ask": cents(market.get("no_ask_dollars", market.get("no_ask"))),
        "yes_ask": cents(market.get("yes_ask_dollars", market.get("yes_ask"))),
        "status": market.get("status", ""),
    }


def fetch_kalshi_markets() -> pd.DataFrame:
    rows = []
    cursor = ""
    while True:
        params = {"limit": 1000, "series_ticker": "KXMLBKS"}
        if cursor:
            params["cursor"] = cursor
        payload = signed_get("/markets", params=params)
        rows.extend(parse_kalshi_market(m) for m in payload.get("markets", []) or [])
        cursor = payload.get("cursor") or ""
        if not cursor:
            break
    return pd.DataFrame(rows)


def collect_pairs(payload: dict, event: dict, dt_local: datetime) -> pd.DataFrame:
    rows = []
    game = f"{event.get('away_team')} @ {event.get('home_team')}"
    for book in payload.get("bookmakers", []) or []:
        bkey = book.get("key")
        for market in book.get("markets", []) or []:
            if market.get("key") not in {"pitcher_strikeouts", "pitcher_strikeouts_alternate"}:
                continue
            grouped = {}
            for outcome in market.get("outcomes", []) or []:
                pitcher = outcome.get("description") or outcome.get("participant") or ""
                line = outcome.get("point")
                if not pitcher or line is None:
                    continue
                key = (normalize_name(pitcher), pitcher, float(line), market.get("key"))
                grouped.setdefault(key, {})[str(outcome.get("name")).lower()] = outcome.get("price")
            for (clean, pitcher, line, market_key), sides in grouped.items():
                if "over" not in sides or "under" not in sides:
                    continue
                fair = devig_pair(sides["over"], sides["under"])
                if fair is None:
                    continue
                rows.append(
                    {
                        "event_sort": dt_local.isoformat(),
                        "event_start_pt": dt_local.strftime("%-I:%M %p"),
                        "game": game,
                        "bookmaker": bkey,
                        "market_key": market_key,
                        "clean_name": clean,
                        "pitcher": pitcher,
                        "line": float(line),
                        "over_odds": int(sides["over"]),
                        "under_odds": int(sides["under"]),
                        "fair_over_prob": fair["over"],
                        "fair_under_prob": fair["under"],
                    }
                )
    return pd.DataFrame(rows)


def line_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in raw.groupby(["event_sort", "event_start_pt", "game", "clean_name", "pitcher", "line"]):
        event_sort, event_start, game, clean, pitcher, line = keys
        dedup = group.sort_values("market_key").drop_duplicates("bookmaker", keep="first")
        med_over = float(dedup["over_odds"].median())
        med_under = float(dedup["under_odds"].median())
        mean_under_fair = float(dedup["fair_under_prob"].mean() * 100.0)
        main_books = int(dedup["market_key"].eq("pitcher_strikeouts").sum())
        alt_extreme = bool(
            (dedup["over_odds"].le(-200) | dedup["under_odds"].le(-200) | dedup["over_odds"].ge(160) | dedup["under_odds"].ge(160)).mean()
            >= 0.5
        )
        centered = bool(med_over > -200 and med_under > -200 and med_over < 160 and med_under < 160)
        rows.append(
            {
                "event_sort": event_sort,
                "event_start_pt": event_start,
                "game": game,
                "clean_name": clean,
                "pitcher": pitcher,
                "line": float(line),
                "book_count": int(dedup["bookmaker"].nunique()),
                "main_market_books": main_books,
                "market_source": "main" if main_books > 0 else "alternate_only",
                "over_price": int(round(med_over)),
                "under_price": int(round(med_under)),
                "sharp_fair_no_cents": mean_under_fair,
                "balance_score": abs(mean_under_fair - 50.0),
                "centered": centered,
                "alt_extreme": alt_extreme,
            }
        )
    return pd.DataFrame(rows)


def choose_main_lines(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in summary.groupby(["event_sort", "game", "clean_name", "pitcher"], dropna=False):
        candidates = group[(group["centered"]) & (~group["alt_extreme"])].copy()
        if candidates.empty:
            candidates = group.copy()
        candidates["_main_sort"] = (candidates["main_market_books"] <= 0).astype(int)
        candidates = candidates.sort_values(
            ["_main_sort", "main_market_books", "book_count", "balance_score"],
            ascending=[True, False, False, True],
        )
        row = candidates.iloc[0].drop(labels=["_main_sort"]).to_dict()
        row["no_main_over_le_140"] = bool(row["over_price"] > -140)
        rows.append(row)
    return pd.DataFrame(rows)


def attach_kalshi(qualifiers: pd.DataFrame, kalshi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kalshi = kalshi[(kalshi["event_date"].eq(str(TODAY))) & (kalshi["status"].isin(["active", "initialized"]))].copy()
    for _, row in qualifiers.iterrows():
        clean = row["clean_name"]
        line = float(row["line"])
        matches = kalshi[(kalshi["clean_name"].eq(clean)) & (kalshi["line"].eq(line))].copy()
        if matches.empty:
            last = clean.split("_")[-1]
            matches = kalshi[
                ((kalshi["clean_name"].str.endswith("_" + last)) | (kalshi["clean_name"].eq(last)))
                & (kalshi["line"].eq(line))
            ].copy()
        rec = row.to_dict()
        if matches.empty:
            rec.update({"kalshi_ticker": "", "kalshi_title": "", "live_no_ask": math.nan, "fillable": False})
        else:
            matches["_ask_sort"] = pd.to_numeric(matches["no_ask"], errors="coerce").fillna(999)
            match = matches.sort_values(["_ask_sort", "ticker"]).iloc[0]
            no_ask = match["no_ask"]
            rec.update(
                {
                    "kalshi_ticker": match["ticker"],
                    "kalshi_title": match["title"],
                    "live_no_ask": no_ask,
                    "fillable": bool(pd.notna(no_ask) and int(no_ask) <= int(rec["ceiling_cents"])),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    if not API_KEY:
        raise SystemExit("THE_ODDS_API_KEY missing")
    events = get_json(f"{ODDS_BASE}/sports/{SPORT}/events", {"apiKey": API_KEY})
    raw_frames = []
    games = 0
    for event in events:
        dt_local = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).astimezone(TZ)
        if dt_local.date() != TODAY or dt_local <= CUTOFF:
            continue
        games += 1
        payload = get_json(
            f"{ODDS_BASE}/sports/{SPORT}/events/{event['id']}/odds",
            {
                "apiKey": API_KEY,
                "regions": "us,us2,eu,au",
                "markets": "pitcher_strikeouts,pitcher_strikeouts_alternate",
                "oddsFormat": "american",
            },
        )
        pairs = collect_pairs(payload, event, dt_local)
        if not pairs.empty:
            raw_frames.append(pairs)
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    summary = line_summary(raw)
    main = choose_main_lines(summary)
    main = main.sort_values(["event_sort", "game", "pitcher"])
    main.to_csv(AUDIT, index=False)

    q = main[main["over_price"].le(-140)].copy()
    q["post_at_cents"] = q["sharp_fair_no_cents"].map(lambda x: math.floor(float(x)))
    q["ceiling_cents"] = q["sharp_fair_no_cents"].map(lambda x: math.floor(float(x) + 2.0))
    q["fair_no_american"] = q["sharp_fair_no_cents"].map(lambda x: prob_to_american(float(x) / 100.0))
    q["post_at_american"] = q["post_at_cents"].map(lambda x: prob_to_american(float(x) / 100.0))
    q["ceiling_american"] = q["ceiling_cents"].map(lambda x: prob_to_american(float(x) / 100.0))
    q["clean_multi_book"] = q["book_count"].ge(3)
    q = attach_kalshi(q, fetch_kalshi_markets()) if not q.empty else q
    q = q.sort_values(["clean_multi_book", "event_sort", "over_price"], ascending=[False, True, True])
    q.to_csv(OUT, index=False)

    print("AS_OF", datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("GAMES_AFTER_2PM_PT", games)
    print("PITCHERS_WITH_MAIN_LINE", len(main))
    print("MAIN_LINE_QUALIFIERS_OVER_LE_NEG140", len(q))
    print("CLEAN_MULTI_BOOK", int(q["clean_multi_book"].sum()) if not q.empty else 0)
    print("ONE_OR_TWO_BOOK", int((~q["clean_multi_book"]).sum()) if not q.empty else 0)
    print("ALT_LINES_EXCLUDED", int(len(summary) - len(main)))
    print("NO_MAIN_OVER_LE_NEG140", int(main["no_main_over_le_140"].sum()))
    cols = [
        "event_start_pt",
        "game",
        "pitcher",
        "line",
        "over_price",
        "under_price",
        "book_count",
        "market_source",
        "sharp_fair_no_cents",
        "fair_no_american",
        "post_at_cents",
        "post_at_american",
        "ceiling_cents",
        "ceiling_american",
        "kalshi_ticker",
        "live_no_ask",
        "fillable",
        "clean_multi_book",
    ]
    if q.empty:
        print("NO_QUALIFIERS")
    else:
        print(q[cols].round({"sharp_fair_no_cents": 2}).to_string(index=False))
    print("OUT", OUT)
    print("AUDIT", AUDIT)


if __name__ == "__main__":
    main()
