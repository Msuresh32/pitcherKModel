from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(SCRIPT_DIR))

from kalshi_auth import sign_kalshi_request
from mlb_identity import normalize_name
from sharp_benchmark import book_family, consensus_from_rows

load_dotenv(BASE / ".env")

API_KEY = os.getenv("THE_ODDS_API_KEY")
ODDS_BASE = "https://api.the-odds-api.com/v4"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SPORT = "baseball_mlb"
TZ = ZoneInfo("America/Los_Angeles")
GRADE_DATE = "2026-06-17"
CARD_DATE = "2026-06-18"
FILLS = BASE / "outputs/kalshi_kprop_fills.csv"
PREV_LEDGER = BASE / "scratch/audits/full_record_since_2026_06_15.csv"
GRADE_OUT = BASE / "scratch/audits/jun17_sharp_grade.csv"
LEDGER_OUT = BASE / "scratch/audits/running_optimized_kprop_ledger.csv"
CARD_OUT = BASE / "scratch/audits/jun18_main_line_card.csv"
CARD_AUDIT = BASE / "scratch/audits/jun18_main_line_all_pitchers.csv"
JUN15_16_SHARP = BASE / "scratch/audits/optimized_unders_20260615_16_sharp_grade.csv"

PUBLIC_FAVORITES = {
    "shohei_ohtani",
    "carlos_rodon",
    "george_kirby",
    "max_scherzer",
    "gavin_williams",
    "kyle_bradish",
    "robbie_ray",
}


def signed_get(path: str, params: dict | None = None) -> dict:
    headers = sign_kalshi_request("GET", "/trade-api/v2" + path)
    response = requests.get(KALSHI_BASE + path, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def odds_get(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def american_to_prob(odds) -> float:
    try:
        odds = float(odds)
    except Exception:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return np.nan


def prob_to_american(prob: float) -> int | None:
    if not np.isfinite(prob) or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100.0 * prob / (1.0 - prob)))
    return int(round(100.0 * (1.0 - prob) / prob))


def devig_under(over_odds, under_odds) -> float:
    op = american_to_prob(over_odds)
    up = american_to_prob(under_odds)
    if not np.isfinite(op) or not np.isfinite(up) or op + up <= 0:
        return np.nan
    return up / (op + up) * 100.0


def cents(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return int(round(numeric * 100.0)) if numeric <= 1 else int(round(numeric))


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
        "status": market.get("status", ""),
        "result": market.get("result", ""),
        "occurrence_datetime": market.get("occurrence_datetime") or market.get("close_time"),
        "yes_ask": cents(market.get("yes_ask_dollars", market.get("yes_ask"))),
        "no_ask": cents(market.get("no_ask_dollars", market.get("no_ask"))),
    }


def fetch_kalshi_markets() -> pd.DataFrame:
    rows = []
    cursor = ""
    while True:
        params = {"limit": 1000, "series_ticker": "KXMLBKS"}
        if cursor:
            params["cursor"] = cursor
        payload = signed_get("/markets", params=params)
        rows.extend(parse_kalshi_market(market) for market in payload.get("markets", []) or [])
        cursor = payload.get("cursor") or ""
        if not cursor:
            break
    return pd.DataFrame(rows)


def market_meta(ticker: str) -> dict:
    return signed_get(f"/markets/{ticker}").get("market") or {}


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def kalshi_close_no(ticker: str, occurrence_override: datetime | None = None) -> float:
    meta = market_meta(ticker)
    occurrence = occurrence_override or parse_dt(meta.get("occurrence_datetime") or meta.get("close_time") or meta.get("expected_expiration_time"))
    if occurrence is None:
        return np.nan
    end = occurrence - timedelta(minutes=10)
    start = end - timedelta(hours=12)
    try:
        data = signed_get(
            f"/series/KXMLBKS/markets/{ticker}/candlesticks",
            {
                "start_ts": int(start.timestamp()),
                "end_ts": int(end.timestamp()),
                "period_interval": 1,
                "include_latest_before_start": "true",
            },
        )
    except Exception:
        return np.nan
    candles = data.get("candlesticks") or []
    yes_vals = []
    for candle in candles:
        price = candle.get("price") or {}
        value = price.get("close_dollars") or price.get("previous_dollars")
        if value is not None:
            try:
                yes_vals.append(float(value) * 100.0)
            except Exception:
                pass
    if not yes_vals:
        return np.nan
    return 100.0 - yes_vals[-1]


def collect_odds_pairs(payload: dict, event: dict, dt_local: datetime | None = None) -> pd.DataFrame:
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
                under = devig_under(sides["over"], sides["under"])
                if not np.isfinite(under):
                    continue
                rec = {
                    "game": game,
                    "bookmaker": bkey,
                    "market_key": market_key,
                    "clean_name": clean,
                    "pitcher": pitcher,
                    "line": float(line),
                    "over_odds": int(sides["over"]),
                    "under_odds": int(sides["under"]),
                    "sharp_fair_no_cents": under,
                }
                if dt_local is not None:
                    rec["event_sort"] = dt_local.isoformat()
                    rec["event_start_pt"] = dt_local.strftime("%-I:%M %p")
                rows.append(rec)
    return pd.DataFrame(rows)


def historical_close_rows(slate_date: str) -> pd.DataFrame:
    snapshot = f"{slate_date}T16:00:00Z"
    events_payload = odds_get(
        f"{ODDS_BASE}/historical/sports/{SPORT}/events",
        {"apiKey": API_KEY, "date": snapshot},
    )
    rows = []
    start = datetime.fromisoformat(slate_date + "T00:00:00+00:00")
    end = start + timedelta(hours=36)
    for event in events_payload.get("data", []) or []:
        commence = parse_dt(event.get("commence_time"))
        if commence is None or not (start <= commence < end):
            continue
        close_snap = commence - timedelta(minutes=10)
        try:
            payload = odds_get(
                f"{ODDS_BASE}/historical/sports/{SPORT}/events/{event['id']}/odds",
                {
                    "apiKey": API_KEY,
                    "regions": "us,us2,eu,au",
                    "markets": "pitcher_strikeouts,pitcher_strikeouts_alternate",
                    "oddsFormat": "american",
                    "date": close_snap.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            ).get("data", {})
        except Exception:
            continue
        pairs = collect_odds_pairs(payload, event)
        if not pairs.empty:
            pairs["commence_time"] = event.get("commence_time")
            pairs["close_snapshot"] = close_snap.strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append(pairs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def choose_sharp_close(close_rows: pd.DataFrame, pitcher: str, line: float) -> dict:
    clean = normalize_name(pitcher)
    cand = close_rows[close_rows["clean_name"].eq(clean)].copy()
    if cand.empty:
        last = clean.split("_")[-1]
        cand = close_rows[(close_rows["clean_name"].str.endswith("_" + last)) | (close_rows["clean_name"].eq(last))].copy()
    if cand.empty:
        return {
            "sharp_source": "missing",
            "sharp_books": "",
            "sharp_book_count": 0,
            "sharp_confidence": "missing",
            "match_type": "missing",
            "sharp_close_no_cents": np.nan,
        }
    cand["abs_hook"] = (cand["line"] - float(line)).abs()
    exact = cand[cand["abs_hook"].le(0.001)].copy()
    pool = exact if not exact.empty else cand[cand["abs_hook"].eq(cand["abs_hook"].min())].copy()
    match_type = "exact" if not exact.empty else "nearest"
    choice = consensus_from_rows(pool, "sharp_fair_no_cents", sort_cols=["market_key"])
    single_book = choice["book_count"] == 1
    hit = pool[pool["bookmaker"].map(book_family).isin(str(choice["books"]).split(","))]
    row = (hit if not hit.empty else pool).sort_values("market_key").iloc[0]
    return {
        "sharp_source": choice["source"],
        "sharp_books": choice["books"],
        "sharp_book_count": choice["book_count"],
        "sharp_confidence": choice["confidence"],
        "match_type": match_type,
        "matched_line": float(pool["line"].iloc[0]),
        "hook_diff": float(pool["line"].iloc[0]) - float(line),
        "close_over_odds": int(row["over_odds"]) if single_book and np.isfinite(row["over_odds"]) else np.nan,
        "close_under_odds": int(row["under_odds"]) if single_book and np.isfinite(row["under_odds"]) else np.nan,
        "sharp_close_no_cents": choice["value"],
        "commence_time": row.get("commence_time", ""),
    }


def recompute_entry_before_commence(ticker: str, side: str, commence_time: str) -> dict:
    fills = pd.read_csv(FILLS)
    group = fills[fills["ticker"].eq(ticker)].copy()
    if group.empty:
        return {"contracts_pregame": np.nan, "vwap_entry_cents": np.nan, "pregame_maker_taker": "", "pregame_taker_contract_frac": np.nan, "post_start_contracts": np.nan, "vwap_all_cents": np.nan}
    group["count_fp"] = pd.to_numeric(group["count_fp"], errors="coerce").fillna(0.0)
    group["no_price_cents"] = pd.to_numeric(group["no_price_cents"], errors="coerce")
    group["yes_price_cents"] = pd.to_numeric(group["yes_price_cents"], errors="coerce")
    group["is_taker_bool"] = group["is_taker"].astype(str).str.lower().eq("true")
    price_col = "no_price_cents" if side == "UNDER" else "yes_price_cents"
    all_contracts = float(group["count_fp"].sum())
    all_vwap = float(np.average(group[price_col], weights=group["count_fp"])) if all_contracts else np.nan
    commence = parse_dt(commence_time)
    if commence is None:
        pre = group
    else:
        fill_ts = pd.to_datetime(group["created_time"], utc=True, errors="coerce")
        pre = group[fill_ts <= commence]
    pre_contracts = float(pre["count_fp"].sum())
    pre_vwap = float(np.average(pre[price_col], weights=pre["count_fp"])) if pre_contracts else np.nan
    pre_taker = float(np.average(pre["is_taker_bool"].astype(float), weights=pre["count_fp"])) if pre_contracts else np.nan
    return {
        "contracts_pregame": pre_contracts,
        "vwap_entry_cents": pre_vwap,
        "pregame_maker_taker": "taker" if pre_taker > 0.5 else "maker",
        "pregame_taker_contract_frac": pre_taker,
        "post_start_contracts": all_contracts - pre_contracts,
        "vwap_all_cents": all_vwap,
    }


def actual_ks_map(slate_date: str) -> dict:
    schedule = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": slate_date},
        timeout=30,
    )
    schedule.raise_for_status()
    out = {}
    for date_block in schedule.json().get("dates", []) or []:
        for game in date_block.get("games", []) or []:
            game_pk = game.get("gamePk")
            if not game_pk:
                continue
            box = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=30)
            if box.status_code != 200:
                continue
            data = box.json()
            for side in ["home", "away"]:
                players = data.get("teams", {}).get(side, {}).get("players", {}) or {}
                for player in players.values():
                    name = player.get("person", {}).get("fullName", "")
                    pitching = player.get("stats", {}).get("pitching", {})
                    if name and "strikeOuts" in pitching:
                        out[normalize_name(name)] = int(pitching.get("strikeOuts") or 0)
    return out


def collapse_jun17_fills(kalshi_markets: pd.DataFrame) -> pd.DataFrame:
    fills = pd.read_csv(FILLS)
    fills = fills[fills["ticker"].astype(str).str.startswith("KXMLBKS-26JUN17")].copy()
    fills["count_fp"] = pd.to_numeric(fills["count_fp"], errors="coerce").fillna(0.0)
    fills["no_price_cents"] = pd.to_numeric(fills["no_price_cents"], errors="coerce")
    fills["yes_price_cents"] = pd.to_numeric(fills["yes_price_cents"], errors="coerce")
    fills["is_taker_bool"] = fills["is_taker"].astype(str).str.lower().eq("true")
    kmeta = kalshi_markets.set_index("ticker").to_dict("index")
    rows = []
    for ticker, group in fills.groupby("ticker"):
        side = "UNDER" if (group["side"].astype(str).str.lower().eq("no")).mean() >= 0.5 else "OVER"
        price_col = "no_price_cents" if side == "UNDER" else "yes_price_cents"
        contracts = float(group["count_fp"].sum())
        vwap_all = float(np.average(group[price_col], weights=group["count_fp"])) if contracts else np.nan
        meta = kmeta.get(ticker, {})
        occurrence = parse_dt(meta.get("occurrence_datetime"))
        fill_ts = pd.to_datetime(group["created_time"], utc=True, errors="coerce")
        pre_mask = pd.Series(True, index=group.index)
        if occurrence is not None:
            pre_mask = fill_ts <= occurrence
        pre = group[pre_mask]
        pre_contracts = float(pre["count_fp"].sum())
        vwap_pre = float(np.average(pre[price_col], weights=pre["count_fp"])) if pre_contracts else np.nan
        taker_frac = float(np.average(group["is_taker_bool"].astype(float), weights=group["count_fp"])) if contracts else np.nan
        pre_taker_frac = float(np.average(pre["is_taker_bool"].astype(float), weights=pre["count_fp"])) if pre_contracts else np.nan
        result = str(group["result"].dropna().iloc[-1]).lower() if group["result"].notna().any() else str(meta.get("result", "")).lower()
        clean = normalize_name(group["pitcher"].dropna().iloc[0])
        rows.append(
            {
                "date": GRADE_DATE,
                "ticker": ticker,
                "pitcher": group["pitcher"].dropna().iloc[0],
                "clean_name": clean,
                "line": float(group["line"].dropna().iloc[0]),
                "side": side,
                "contracts_all": contracts,
                "vwap_all_cents": vwap_all,
                "contracts_pregame": pre_contracts,
                "vwap_entry_cents": vwap_pre,
                "post_start_contracts": contracts - pre_contracts,
                "maker_taker": "taker" if taker_frac > 0.5 else "maker",
                "pregame_maker_taker": "taker" if pre_taker_frac > 0.5 else "maker",
                "taker_contract_frac": taker_frac,
                "pregame_taker_contract_frac": pre_taker_frac,
                "kalshi_result": result,
                "market_title": meta.get("title", group["market_title"].dropna().iloc[0] if group["market_title"].notna().any() else ""),
            }
        )
    return pd.DataFrame(rows)


def line_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in raw.groupby(["event_sort", "event_start_pt", "game", "clean_name", "pitcher", "line"], dropna=False):
        event_sort, event_start, game, clean, pitcher, line = keys
        dedup = group.sort_values("market_key").drop_duplicates("bookmaker", keep="first")
        med_over = float(dedup["over_odds"].median())
        med_under = float(dedup["under_odds"].median())
        choice = consensus_from_rows(dedup, "sharp_fair_no_cents", sort_cols=["market_key"])
        fair_no = float(choice["value"])
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
                "sharp_fair_no_cents": fair_no,
                "sharp_source": choice["source"],
                "sharp_books": choice["books"],
                "sharp_book_count": choice["book_count"],
                "sharp_confidence": choice["confidence"],
                "balance_score": abs(fair_no - 50.0),
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
        rows.append(candidates.iloc[0].drop(labels=["_main_sort"]).to_dict())
    return pd.DataFrame(rows)


def live_main_line_card(kalshi_markets: pd.DataFrame) -> pd.DataFrame:
    events = odds_get(f"{ODDS_BASE}/sports/{SPORT}/events", {"apiKey": API_KEY})
    raw_frames = []
    for event in events:
        dt_local = parse_dt(event.get("commence_time")).astimezone(TZ)
        if dt_local.date().isoformat() != CARD_DATE:
            continue
        payload = odds_get(
            f"{ODDS_BASE}/sports/{SPORT}/events/{event['id']}/odds",
            {
                "apiKey": API_KEY,
                "regions": "us,us2,eu,au",
                "markets": "pitcher_strikeouts,pitcher_strikeouts_alternate",
                "oddsFormat": "american",
            },
        )
        pairs = collect_odds_pairs(payload, event, dt_local)
        if not pairs.empty:
            raw_frames.append(pairs)
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    summary = line_summary(raw)
    main = choose_main_lines(summary)
    main.to_csv(CARD_AUDIT, index=False)
    q = main[main["over_price"].le(-140)].copy()
    if q.empty:
        return q
    q["post_at_cents"] = q["sharp_fair_no_cents"].map(lambda x: math.floor(float(x)))
    q["ceiling_cents"] = q["sharp_fair_no_cents"].map(lambda x: math.floor(float(x) + 2.0))
    q["fair_no_american"] = q["sharp_fair_no_cents"].map(lambda x: prob_to_american(float(x) / 100.0))
    q["clean_multi_book"] = q["book_count"].ge(3)
    q["mechanism"] = q["clean_name"].map(lambda x: "public-favorite" if x in PUBLIC_FAVORITES else "weak/line-driven")
    q["orientation_sum"] = 100.0
    rows = []
    km = kalshi_markets[kalshi_markets["event_date"].eq(CARD_DATE)].copy()
    for _, row in q.iterrows():
        matches = km[(km["clean_name"].eq(row["clean_name"])) & (np.isclose(km["line"], row["line"]))].copy()
        if matches.empty:
            last = str(row["clean_name"]).split("_")[-1]
            matches = km[
                ((km["clean_name"].str.endswith("_" + last)) | (km["clean_name"].eq(last)))
                & (np.isclose(km["line"], row["line"]))
            ].copy()
        rec = row.to_dict()
        if matches.empty:
            rec.update({"kalshi_ticker": "", "kalshi_title": "", "live_no_ask": np.nan, "fillable": False})
        else:
            matches["_ask_sort"] = pd.to_numeric(matches["no_ask"], errors="coerce").fillna(999)
            hit = matches.sort_values(["_ask_sort", "ticker"]).iloc[0]
            rec.update(
                {
                    "kalshi_ticker": hit["ticker"],
                    "kalshi_title": hit["title"],
                    "live_no_ask": hit["no_ask"],
                    "fillable": bool(pd.notna(hit["no_ask"]) and int(hit["no_ask"]) <= int(row["ceiling_cents"])),
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["clean_multi_book", "event_sort", "over_price"], ascending=[False, True, True])


def build_ledger(today_grade: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if JUN15_16_SHARP.exists():
        prev = pd.read_csv(JUN15_16_SHARP)
        for _, r in prev.iterrows():
            rows.append(
                {
                    "date": r["date"],
                    "pitcher": r["pitcher"],
                    "line": r["line"],
                    "side": r["side"],
                    "entry_cents": r["vwap_fill_cents"],
                    "sharp_close_no_cents": r.get("sharp_side_close_cents", np.nan),
                    "sharp_clv_pp": r.get("sharp_clv_pp", np.nan),
                    "match_type": r.get("match_type", ""),
                    "kalshi_clv_pp": r.get("kalshi_clv_pp", np.nan),
                    "result": "",
                    "actual_ks": np.nan,
                }
            )
    for _, r in today_grade.iterrows():
        rows.append(
            {
                "date": r["date"],
                "pitcher": r["pitcher"],
                "line": r["line"],
                "side": r["side"],
                "entry_cents": r["vwap_entry_cents"],
                "sharp_close_no_cents": r["sharp_close_no_cents"],
                "sharp_clv_pp": r["sharp_clv_pp"],
                "match_type": r["match_type"],
                "kalshi_clv_pp": r["kalshi_clv_pp"],
                "result": r["result"],
                "actual_ks": r["actual_ks"],
            }
        )
    ledger = pd.DataFrame(rows)
    ledger.to_csv(LEDGER_OUT, index=False)
    return ledger


def main() -> None:
    if not API_KEY:
        raise SystemExit("THE_ODDS_API_KEY missing")
    kalshi_markets = fetch_kalshi_markets()
    positions = collapse_jun17_fills(kalshi_markets)
    close_rows = historical_close_rows(GRADE_DATE)
    actual = actual_ks_map(GRADE_DATE)
    grade_rows = []
    for _, r in positions.iterrows():
        sharp = choose_sharp_close(close_rows, r["pitcher"], float(r["line"]))
        entry = recompute_entry_before_commence(r["ticker"], r["side"], sharp.get("commence_time", ""))
        for key, value in entry.items():
            r[key] = value
        commence = parse_dt(sharp.get("commence_time", ""))
        kclose = kalshi_close_no(r["ticker"], commence)
        actual_k = actual.get(r["clean_name"], np.nan)
        won = np.nan
        if np.isfinite(actual_k):
            won = bool(actual_k < float(r["line"])) if r["side"] == "UNDER" else bool(actual_k > float(r["line"]))
        result = "W" if won is True else "L" if won is False else "PENDING"
        rec = r.to_dict()
        rec.update(sharp)
        rec["kalshi_close_no_cents"] = kclose
        rec["sharp_clv_pp"] = rec["sharp_close_no_cents"] - rec["vwap_entry_cents"] if np.isfinite(rec["sharp_close_no_cents"]) else np.nan
        rec["kalshi_clv_pp"] = kclose - rec["vwap_entry_cents"] if np.isfinite(kclose) else np.nan
        rec["all_fill_sharp_clv_pp"] = rec["sharp_close_no_cents"] - rec["vwap_all_cents"] if np.isfinite(rec["sharp_close_no_cents"]) else np.nan
        rec["actual_ks"] = actual_k
        rec["result"] = result
        rec["non_pregame_note"] = "post-start fills excluded from entry VWAP" if rec["post_start_contracts"] > 0 else ""
        grade_rows.append(rec)
    grade = pd.DataFrame(grade_rows).sort_values(["pitcher", "line"])
    grade.to_csv(GRADE_OUT, index=False)

    ledger = build_ledger(grade)
    exact = grade[grade["match_type"].eq("exact")]
    blended = grade[grade["sharp_clv_pp"].notna()]
    record = grade["result"].value_counts().to_dict()
    ledger_exact = ledger[ledger["match_type"].eq("exact") & ledger["sharp_clv_pp"].notna()]
    ledger_blended = ledger[ledger["sharp_clv_pp"].notna()]

    card = live_main_line_card(kalshi_markets)
    card.to_csv(CARD_OUT, index=False)

    print("TODAY_GRADE_DATE", GRADE_DATE)
    print("FILLED_POSITIONS", len(grade))
    print("SHARP_EXACT", len(exact), "SHARP_BLENDED", len(blended))
    print("TODAY_MEAN_SHARP_CLV_EXACT", round(exact["sharp_clv_pp"].mean(), 4) if len(exact) else "NA")
    print("TODAY_MEAN_SHARP_CLV_BLENDED", round(blended["sharp_clv_pp"].mean(), 4) if len(blended) else "NA")
    print("TODAY_POSITIVE_SHARP_PCT", round((blended["sharp_clv_pp"] > 0).mean() * 100, 2) if len(blended) else "NA")
    print("TODAY_RECORD", f"{int(record.get('W', 0))}-{int(record.get('L', 0))}")
    ohtani = grade[grade["clean_name"].eq("shohei_ohtani")]
    if not ohtani.empty:
        row = ohtani.iloc[0]
        print("OHTANI_ENTRY", round(row["vwap_entry_cents"], 4), "SHARP_CLOSE", round(row["sharp_close_no_cents"], 4), "SHARP_CLV", round(row["sharp_clv_pp"], 4), "ALL_FILL_VWAP", round(row["vwap_all_cents"], 4), "POST_START_CONTRACTS", round(row["post_start_contracts"], 2))
    print("TODAY_TABLE")
    today_cols = [
        "pitcher",
        "line",
        "side",
        "contracts_pregame",
        "vwap_entry_cents",
        "sharp_source",
        "sharp_books",
        "sharp_book_count",
        "sharp_confidence",
        "match_type",
        "matched_line",
        "close_over_odds",
        "close_under_odds",
        "sharp_close_no_cents",
        "sharp_clv_pp",
        "kalshi_close_no_cents",
        "kalshi_clv_pp",
        "pregame_maker_taker",
        "actual_ks",
        "result",
        "post_start_contracts",
        "non_pregame_note",
    ]
    print(grade[today_cols].round(4).to_string(index=False))
    print("LEDGER_TOTAL_OPTIMIZED_BETS", len(ledger))
    print("LEDGER_SHARP_EXACT_N", len(ledger_exact))
    print("LEDGER_SHARP_BLENDED_N", len(ledger_blended))
    print("LEDGER_MEAN_SHARP_CLV_EXACT", round(ledger_exact["sharp_clv_pp"].mean(), 4) if len(ledger_exact) else "NA")
    print("LEDGER_MEAN_SHARP_CLV_BLENDED", round(ledger_blended["sharp_clv_pp"].mean(), 4) if len(ledger_blended) else "NA")
    print("LEDGER_POSITIVE_SHARP_PCT", round((ledger_blended["sharp_clv_pp"] > 0).mean() * 100, 2) if len(ledger_blended) else "NA")
    print("LEDGER_RECORD", f"{int((ledger['result']=='W').sum())}-{int((ledger['result']=='L').sum())}", "NOT_EVALUATED")
    print("FILLS_TO_35", max(0, 35 - len(ledger)))
    print("CARD_DATE", CARD_DATE)
    print("CARD_ROWS", len(card))
    if not card.empty:
        print("CARD_CLEAN_MULTI_BOOK", int(card["clean_multi_book"].sum()))
        card_cols = [
            "event_start_pt",
            "game",
            "pitcher",
            "line",
            "over_price",
            "book_count",
            "sharp_fair_no_cents",
            "fair_no_american",
            "post_at_cents",
            "ceiling_cents",
            "kalshi_ticker",
            "live_no_ask",
            "fillable",
            "mechanism",
            "clean_multi_book",
            "orientation_sum",
        ]
        print("CARD_TABLE")
        print(card[card_cols].round({"sharp_fair_no_cents": 2}).to_string(index=False))
    else:
        print("CARD_TABLE EMPTY")
    print("ALT_LINES_EXCLUDED_CARD", (CARD_AUDIT.exists()))
    print("GRADE_OUT", GRADE_OUT)
    print("LEDGER_OUT", LEDGER_OUT)
    print("CARD_OUT", CARD_OUT)
    print("CARD_AUDIT", CARD_AUDIT)


if __name__ == "__main__":
    main()
