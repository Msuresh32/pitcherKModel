"""
Fetch Pinnacle closing odds for our bet dates via The Odds API.
Caches results to data/odds/pinnacle_k_close_cache.csv to avoid re-fetching.
"""
import sys, time, datetime, requests
import pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from src.odds.odds_api import get_api_key

CACHE = Path("data/odds/pinnacle_k_close_cache.csv")
BASE  = "https://api.the-odds-api.com/v4"
API_KEY = get_api_key()

def american_to_prob(o):
    o = float(o)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)

def devig_pair(over_o, under_o):
    if pd.isna(over_o) or pd.isna(under_o):
        return np.nan, np.nan
    ip_o = american_to_prob(over_o)
    ip_u = american_to_prob(under_o)
    d = ip_o + ip_u
    if d <= 0:
        return np.nan, np.nan
    return ip_o / d, ip_u / d

# Load bet dates
def load_bet_dates():
    from pathlib import Path
    dfs = []
    for path, d in [
        ("thresh_sel_2025_dk_edges.csv",  "data/processed_2024"),
        ("wf2026_p1_mar_apr_edges.csv",   "data/processed"),
        ("wf2026_p2_may_edges.csv",        "data/processed_apr2026"),
        ("wf2026_p3_jun_edges.csv",        "data/processed"),
    ]:
        p = Path(d) / path
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df[df["market"] == "strikeouts"][["game_date"]].copy())
    all_bets = pd.concat(dfs)
    all_bets["game_date"] = pd.to_datetime(all_bets["game_date"])
    return sorted(all_bets["game_date"].dt.date.unique())

# Check cache
if CACHE.exists():
    cached = pd.read_csv(CACHE)
    cached["game_date"] = pd.to_datetime(cached["game_date"])
    cached_dates = set(cached["game_date"].dt.date.astype(str))
    print(f"Cache: {len(cached)} rows across {len(cached_dates)} dates")
else:
    cached = pd.DataFrame()
    cached_dates = set()

bet_dates = load_bet_dates()
needed = [d for d in bet_dates if str(d) not in cached_dates]
print(f"Bet dates: {len(bet_dates)}  |  Need to fetch: {len(needed)}")

if not needed:
    print("Nothing to fetch — all dates cached.")
    sys.exit(0)

session = requests.Session()
new_rows = []
total_credits = 0

for i, date in enumerate(needed):
    date_str = str(date)
    snap = f"{date_str}T17:00:00Z"  # 17:00 UTC = 1pm ET — captures all same-day games

    # Step 1: get events for this date
    try:
        r = session.get(f"{BASE}/historical/sports/baseball_mlb/events",
                        params={"apiKey": API_KEY, "date": snap}, timeout=20)
        r.raise_for_status()
        data = r.json()
        events = data.get("data", [])
        total_credits += int(r.headers.get("x-requests-last", 1))
        remaining = r.headers.get("x-requests-remaining", "?")
    except Exception as e:
        print(f"  [{i+1}/{len(needed)}] {date_str} EVENTS ERROR: {e}")
        continue

    # Filter to games on this calendar date (allow next-day UTC games too for late starts)
    d = date
    day_events = [
        ev for ev in events
        if (d.isoformat() in ev.get("commence_time","") or
            (datetime.date.fromisoformat(ev["commence_time"][:10]) == d))
    ]

    date_rows = 0
    for ev in day_events:
        commence = pd.to_datetime(ev["commence_time"], utc=True)
        close_time = (commence - pd.Timedelta(minutes=8)).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            r2 = session.get(
                f"{BASE}/historical/sports/baseball_mlb/events/{ev['id']}/odds",
                params={
                    "apiKey":      API_KEY,
                    "markets":     "pitcher_strikeouts",
                    "bookmakers":  "pinnacle",
                    "oddsFormat":  "american",
                    "date":        close_time,
                }, timeout=20)
            r2.raise_for_status()
            odata = r2.json().get("data", {})
            total_credits += int(r2.headers.get("x-requests-last", 1))
            remaining = r2.headers.get("x-requests-remaining", "?")
        except Exception as e:
            continue

        for bm in odata.get("bookmakers", []):
            if bm.get("key") != "pinnacle":
                continue
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "pitcher_strikeouts":
                    continue
                by_player = {}
                for oc in mkt.get("outcomes", []):
                    player = oc.get("description") or oc.get("name", "")
                    side   = oc.get("name", "").lower()
                    line   = oc.get("point")
                    price  = oc.get("price")
                    if side not in ("over", "under") or line is None or price is None:
                        continue
                    key = (player, float(line))
                    if key not in by_player:
                        by_player[key] = {}
                    by_player[key][side] = price

                for (player, line), sides in by_player.items():
                    if "over" not in sides or "under" not in sides:
                        continue
                    new_rows.append({
                        "game_date":   date_str,
                        "bookmaker":   "pinnacle",
                        "player_name": player,
                        "line":        line,
                        "over_odds":   sides["over"],
                        "under_odds":  sides["under"],
                    })
                    date_rows += 1

        time.sleep(0.08)

    if (i + 1) % 25 == 0 or i == len(needed) - 1:
        print(f"  [{i+1:>3}/{len(needed)}] {date_str}  rows_today={date_rows}  "
              f"total_new={len(new_rows)}  credits_used={total_credits}  remaining={remaining}")

# Save
if new_rows:
    new_df = pd.DataFrame(new_rows)
    new_df["game_date"] = pd.to_datetime(new_df["game_date"])
    combined = pd.concat([cached, new_df], ignore_index=True) if len(cached) else new_df
    combined.to_csv(CACHE, index=False)
    print(f"\nSaved {len(combined)} total rows ({len(new_rows)} new) to {CACHE}")
else:
    print("\nNo new rows fetched.")
    combined = cached

# Quick stats
if len(combined):
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined[["nv_over","nv_under"]] = combined.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    print(f"Pinnacle rows: {len(combined)}  |  "
          f"Dates: {combined['game_date'].nunique()}  |  "
          f"Players: {combined['player_name'].nunique()}")
    print(f"Date range: {combined['game_date'].min().date()} to {combined['game_date'].max().date()}")
print(f"\nTotal API credits used this run: {total_credits}")
