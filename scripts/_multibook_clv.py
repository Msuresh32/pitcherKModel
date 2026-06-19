"""
Multi-book de-vigged CLV analysis.
- Runs local books immediately: DK, BetOnline, FanDuel, BetRivers, BetMGM
- Fetches Pinnacle close via API (caches to data/odds/pinnacle_close_cache.csv)
- Weighted average using inverse Brier score weights from calibration data
"""
import sys, time
import pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from src.odds.odds_api import get_api_key

# ── Calibration weights (inverse Brier, from Kobe's ranking) ──────
# Lower Brier = sharper. We weight by 1/Brier normalized.
BOOK_BRIER = {
    "betrivers":   0.2125,
    "fanduel":     0.2474,
    "betonlineag": 0.2476,
    "draftkings":  0.2478,
    "betmgm":      0.2484,
    "pinnacle":    0.2484,  # same as betmgm per calibration
}
raw_weights = {b: 1 / v for b, v in BOOK_BRIER.items()}
total_w = sum(raw_weights.values())
WEIGHTS = {b: v / total_w for b, v in raw_weights.items()}

CACHE = Path("data/odds/pinnacle_close_cache.csv")

# ── Helpers ────────────────────────────────────────────────────────
def american_to_prob(o):
    o = float(o)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)

def devig_pair(over_o, under_o):
    if pd.isna(over_o) or pd.isna(under_o): return np.nan, np.nan
    ip_o = american_to_prob(over_o)
    ip_u = american_to_prob(under_o)
    d = ip_o + ip_u
    if d <= 0: return np.nan, np.nan
    return ip_o / d, ip_u / d

def load_bets(paths, edge_min=15):
    dfs = []
    for path, d in paths:
        p = Path(d) / path
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df[df["market"] == "strikeouts"].copy())
    bets = pd.concat(dfs, ignore_index=True)
    bets = (bets.sort_values("edge_pct", ascending=False)
                .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
                .reset_index(drop=True))
    bets = bets[bets["edge_pct"] >= edge_min].copy()
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    bets["won"] = bets.apply(lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                              else (r["strikeouts"] < r["line"]), axis=1)
    bets[["nv_entry_over","nv_entry_under"]] = bets.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    bets["nv_entry_side"] = bets.apply(
        lambda r: r["nv_entry_over"] if r["best_side"] == "over" else r["nv_entry_under"], axis=1)
    return bets

def build_close_index(odds_path, books):
    odds = pd.read_csv(odds_path)
    odds = odds[(odds["snapshot_type"] == "close") & (odds["bookmaker"].isin(books))].copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    odds["fetched_at"] = pd.to_datetime(odds["fetched_at"], errors="coerce")
    odds = (odds.sort_values("fetched_at")
                .groupby(["game_date","bookmaker","player_name","line"])
                .last().reset_index())
    odds[["nv_over","nv_under"]] = odds.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return odds

def compute_clv(bets, close_df, book):
    sub = close_df[close_df["bookmaker"] == book][
        ["game_date","player_name","line","nv_over","nv_under"]].copy()
    m = bets.merge(sub, left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"], how="left")
    matched = m.dropna(subset=["nv_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    return matched

# ── Fetch Pinnacle via API ─────────────────────────────────────────
def fetch_pinnacle_close(bets_all):
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        cached["game_date"] = pd.to_datetime(cached["game_date"])
        print(f"  Pinnacle cache: {len(cached)} rows ({cached['game_date'].nunique()} dates)")
    else:
        cached = pd.DataFrame()

    cached_dates = set(cached["game_date"].dt.date.astype(str)) if len(cached) else set()
    needed_dates = sorted(set(bets_all["game_date"].dt.date.astype(str)) - cached_dates)
    print(f"  Dates needed from API: {len(needed_dates)}")

    if not needed_dates:
        return cached

    api_key = get_api_key()
    import requests
    new_rows = []

    for i, date_str in enumerate(needed_dates):
        # Fetch events for this date
        snap = f"{date_str}T14:00:00Z"
        try:
            resp = requests.get(
                "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events",
                params={"apiKey": api_key, "date": snap}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            events = data.get("data", [])
            remaining = resp.headers.get("x-requests-remaining","?")
        except Exception as e:
            print(f"  [{i+1}/{len(needed_dates)}] {date_str} events error: {e}")
            continue

        # Filter to this date's games
        import datetime
        d = datetime.date.fromisoformat(date_str)
        day_events = [ev for ev in events if d.isoformat() in ev.get("commence_time","")]

        for ev in day_events:
            commence = ev["commence_time"]
            close_time = (pd.to_datetime(commence, utc=True) - pd.Timedelta(minutes=8)
                         ).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                odds_resp = requests.get(
                    f"https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/{ev['id']}/odds",
                    params={"apiKey": api_key, "markets": "pitcher_strikeouts",
                            "bookmakers": "pinnacle", "oddsFormat": "american",
                            "date": close_time}, timeout=20)
                odds_resp.raise_for_status()
                odata = odds_resp.json().get("data", {})
                remaining = odds_resp.headers.get("x-requests-remaining","?")
            except Exception as e:
                continue

            for bm in odata.get("bookmakers", []):
                if bm.get("key") != "pinnacle": continue
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "pitcher_strikeouts": continue
                    outcomes = mkt.get("outcomes", [])
                    by_player = {}
                    for oc in outcomes:
                        player = oc.get("description") or oc.get("name","")
                        side   = oc.get("name","").lower()
                        line   = oc.get("point")
                        price  = oc.get("price")
                        if side not in ("over","under"): continue
                        key = (player, line)
                        if key not in by_player: by_player[key] = {}
                        by_player[key][side] = price
                    for (player, line), sides in by_player.items():
                        if "over" not in sides or "under" not in sides: continue
                        new_rows.append({
                            "game_date": date_str,
                            "bookmaker": "pinnacle",
                            "player_name": player,
                            "line": line,
                            "over_odds": sides["over"],
                            "under_odds": sides["under"],
                        })

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(needed_dates)}] done  remaining={remaining}")
        time.sleep(0.15)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df["game_date"] = pd.to_datetime(new_df["game_date"])
        combined = pd.concat([cached, new_df], ignore_index=True) if len(cached) else new_df
        combined.to_csv(CACHE, index=False)
        print(f"  Saved {len(new_rows)} new Pinnacle rows to cache")
        result = combined
    else:
        result = cached

    result[["nv_over","nv_under"]] = result.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    result["bookmaker"] = "pinnacle"
    return result

# ── Main analysis ─────────────────────────────────────────────────
SEP = "=" * 70

bets25 = load_bets([("thresh_sel_2025_dk_edges.csv","data/processed_2024")])
bets26 = load_bets([
    ("wf2026_p1_mar_apr_edges.csv","data/processed"),
    ("wf2026_p2_may_edges.csv","data/processed_apr2026"),
    ("wf2026_p3_jun_edges.csv","data/processed"),
])
bets_all = pd.concat([bets25, bets26], ignore_index=True)
bets_all = (bets_all.sort_values("edge_pct", ascending=False)
                    .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
                    .reset_index(drop=True))
print(f"Total bets: 2025={len(bets25)}  2026={len(bets26)}")

# Local books
print("\nBuilding local close index...")
local_close = pd.concat([
    build_close_index("data/odds/historical_pitcher_props_2025.csv",
                      ["draftkings","betonlineag","fanduel","betrivers","betmgm"]),
    build_close_index("data/odds/full_2026_odds.csv",
                      ["draftkings","betonlineag","fanduel","betrivers","betmgm"]),
], ignore_index=True)

# Fetch Pinnacle
print("\nFetching Pinnacle close data...")
pin_close = fetch_pinnacle_close(bets_all)
if len(pin_close) > 0:
    all_close = pd.concat([local_close, pin_close], ignore_index=True)
else:
    all_close = local_close

LOCAL_BOOKS = ["draftkings","betonlineag","fanduel","betrivers","betmgm","pinnacle"]

for year_label, bets in [("2025 (in-sample threshold)", bets25),
                          ("2026 (OOS walk-forward)",    bets26),
                          ("COMBINED",                   bets_all)]:
    print(f"\n{SEP}")
    print(f"DE-VIGGED CLV — {year_label}  (edge>=15%,  n={len(bets)})")
    print(SEP)
    print(f"  {'Book':<14} {'Matched':>8}  {'Match%':>7}  {'CLV pp':>8}  "
          f"{'t-stat':>7}  {'%Pos':>6}  {'Win%':>6}  {'Weight':>7}")
    print("  " + "-"*72)

    weighted_clv_total = 0.0
    weighted_n = 0
    book_results = {}

    for book in LOCAL_BOOKS:
        matched = compute_clv(bets, all_close, book)
        if len(matched) < 20:
            continue
        mean_clv = matched["clv_pp"].mean()
        se       = matched["clv_pp"].std() / len(matched)**0.5
        t        = mean_clv / se if se > 0 else 0
        pct_pos  = (matched["clv_pp"] > 0).mean()
        win      = matched["won"].mean()
        w        = WEIGHTS.get(book, 0.15)
        book_results[book] = (len(matched), mean_clv, t, pct_pos, win, w)
        print(f"  {book:<14} {len(matched):>8}  {len(matched)/len(bets):>7.1%}  "
              f"{mean_clv:>+7.3f}pp  {t:>7.2f}  {pct_pos:>6.1%}  {win:>6.1%}  {w:>7.3f}")

    # Weighted average (weight by book weight × n_matched)
    if book_results:
        numerator = sum(r[1] * r[5] * r[0] for r in book_results.values())
        denominator = sum(r[5] * r[0] for r in book_results.values())
        wtd_clv = numerator / denominator if denominator > 0 else np.nan
        # Simple average too
        simple_avg = np.mean([r[1] for r in book_results.values()])
        print(f"\n  {'SIMPLE AVG':<14} {'':>8}  {'':>7}  {simple_avg:>+7.3f}pp")
        print(f"  {'WEIGHTED AVG':<14} {'':>8}  {'':>7}  {wtd_clv:>+7.3f}pp  "
              f"(weighted by calibration × n)")
