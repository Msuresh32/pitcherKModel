import argparse
import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from kalshi_auth import sign_kalshi_request


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
K_PROP_PREFIX = "KXMLBKS"
DEFAULT_OUTPUT = "outputs/kalshi_kprop_fills.csv"


def dollars_to_cents(value):
    if value in (None, ""):
        return ""
    try:
        return int((Decimal(str(value)) * Decimal("100")).to_integral_value())
    except (InvalidOperation, TypeError, ValueError):
        return ""


def side_fill_price_cents(fill):
    side = str(fill.get("side") or fill.get("outcome_side") or "").lower()
    if side == "yes":
        return dollars_to_cents(fill.get("yes_price_dollars"))
    if side == "no":
        return dollars_to_cents(fill.get("no_price_dollars"))
    return ""


def signed_get(path, params=None):
    sign_path = f"/trade-api/v2{path}"
    headers = sign_kalshi_request("GET", sign_path)
    response = requests.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_fills(limit=1000, max_pages=5):
    fills = []
    cursor = ""
    for _ in range(max_pages):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = signed_get("/portfolio/fills", params=params)
        fills.extend(data.get("fills", []))
        cursor = data.get("cursor") or ""
        if not cursor:
            break
    return fills


def fetch_market(ticker):
    try:
        return signed_get(f"/markets/{ticker}").get("market") or {}
    except requests.HTTPError:
        return {}


def parse_from_ticker(ticker):
    target_match = re.search(r"-(\d+)$", ticker)
    target_ks = int(target_match.group(1)) if target_match else None
    line = target_ks - 0.5 if target_ks is not None else ""

    pitcher = ""
    slug = ticker.rsplit("-", 2)[-2] if target_match and "-" in ticker else ""
    if len(slug) > 3:
        player_slug = re.sub(r"\d+$", "", slug[3:])
        match = re.match(r"([A-Z])([A-Z]+)$", player_slug)
        if match:
            pitcher = f"{match.group(1)} {match.group(2).title()}"

    return pitcher, line, target_ks


def parse_market_details(ticker, market):
    ticker_pitcher, ticker_line, ticker_target = parse_from_ticker(ticker)
    pitcher = ticker_pitcher
    line = ticker_line
    target_ks = ticker_target

    for key in ("no_sub_title", "yes_sub_title", "subtitle", "title", "rules_primary"):
        text = str(market.get(key) or "")
        match = re.search(
            r"(?:If\s+)?([A-Za-zÀ-ÖØ-öø-ÿ .'-]+?)\s*(?::|records?|to record|with|,)?\s*(\d+)\+?\s+strikeouts?",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"([A-Za-zÀ-ÖØ-öø-ÿ .'-]+?)\s*:\s*(\d+)\+",
                text,
                flags=re.IGNORECASE,
            )
        if match:
            pitcher = match.group(1).strip()
            target_ks = int(match.group(2))
            line = target_ks - 0.5
            break

    floor_strike = market.get("floor_strike")
    if floor_strike not in (None, ""):
        try:
            line = float(floor_strike)
            target_ks = int(line + 0.5)
        except (TypeError, ValueError):
            pass

    return pitcher, line, target_ks


def normalize_kprop_fill(fill, market):
    ticker = fill.get("ticker") or fill.get("market_ticker") or ""
    pitcher, line, target_ks = parse_market_details(ticker, market)
    yes_cents = dollars_to_cents(fill.get("yes_price_dollars"))
    no_cents = dollars_to_cents(fill.get("no_price_dollars"))

    return {
        "created_time": fill.get("created_time", ""),
        "ticker": ticker,
        "pitcher": pitcher,
        "line": line,
        "target_ks": target_ks if target_ks is not None else "",
        "action": fill.get("action", ""),
        "side": fill.get("side", ""),
        "outcome_side": fill.get("outcome_side", ""),
        "book_side": fill.get("book_side", ""),
        "is_taker": fill.get("is_taker", ""),
        "count_fp": fill.get("count_fp", ""),
        "fill_price_cents": side_fill_price_cents(fill),
        "yes_price_cents": yes_cents,
        "no_price_cents": no_cents,
        "yes_price_dollars": fill.get("yes_price_dollars", ""),
        "no_price_dollars": fill.get("no_price_dollars", ""),
        "fee_cost": fill.get("fee_cost", ""),
        "order_id": fill.get("order_id", ""),
        "fill_id": fill.get("fill_id", ""),
        "trade_id": fill.get("trade_id", ""),
        "ts": fill.get("ts", ""),
        "event_ticker": market.get("event_ticker", ""),
        "market_title": market.get("title", ""),
        "market_subtitle": market.get("subtitle", ""),
        "yes_sub_title": market.get("yes_sub_title", ""),
        "no_sub_title": market.get("no_sub_title", ""),
        "floor_strike": market.get("floor_strike", ""),
        "result": market.get("result", ""),
    }


def write_csv(rows, output_path):
    fieldnames = [
        "created_time",
        "ticker",
        "pitcher",
        "line",
        "target_ks",
        "action",
        "side",
        "outcome_side",
        "book_side",
        "is_taker",
        "count_fp",
        "fill_price_cents",
        "yes_price_cents",
        "no_price_cents",
        "yes_price_dollars",
        "no_price_dollars",
        "fee_cost",
        "order_id",
        "fill_id",
        "trade_id",
        "ts",
        "event_ticker",
        "market_title",
        "market_subtitle",
        "yes_sub_title",
        "no_sub_title",
        "floor_strike",
        "result",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-enrich", action="store_true", help="skip per-market metadata calls")
    args = parser.parse_args()

    fills = fetch_fills(limit=args.limit, max_pages=args.max_pages)
    kprop_fills = [
        f for f in fills
        if str(f.get("ticker") or f.get("market_ticker") or "").startswith(K_PROP_PREFIX)
    ]

    market_cache = {}
    rows = []
    for fill in kprop_fills:
        ticker = fill.get("ticker") or fill.get("market_ticker") or ""
        if args.no_enrich:
            market = {}
        else:
            if ticker not in market_cache:
                market_cache[ticker] = fetch_market(ticker)
            market = market_cache[ticker]
        rows.append(normalize_kprop_fill(fill, market))

    output = write_csv(rows, args.output)

    print(f"TOTAL_FILLS_FETCHED: {len(fills)}")
    print(f"KXMLBKS_FILLS: {len(rows)}")
    print(f"UNIQUE_KXMLBKS_TICKERS: {len({row['ticker'] for row in rows})}")
    print(f"CSV_PATH: {output}")
    print()
    for row in rows:
        print(
            row["created_time"],
            "|", row["ticker"],
            "|", row["pitcher"],
            "| line", row["line"],
            "|", row["action"], row["side"],
            "|", row["fill_price_cents"], "c",
            "| is_taker", row["is_taker"],
        )


if __name__ == "__main__":
    main()
