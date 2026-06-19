"""
Multi-book de-vigged CLV — ALL bets (edge >= 0), not just the 15% subset.
Uses local close data (DK, BOL, FD, BetRivers) + Pinnacle from cache.
Shows results by edge band so we can see where CLV concentrates.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")

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

def load_bets(edge_min=0):
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
            dfs.append(df[df["market"] == "strikeouts"].copy())
    bets = pd.concat(dfs, ignore_index=True)
    bets = (bets.sort_values("edge_pct", ascending=False)
                .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
                .reset_index(drop=True))
    bets = bets[bets["edge_pct"] >= edge_min].copy()
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    bets["won"] = bets.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
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

def load_pinnacle_cache():
    cache = Path("data/odds/pinnacle_close_cache.csv")
    if not cache.exists():
        print("  WARNING: Pinnacle cache not found")
        return pd.DataFrame()
    df = pd.read_csv(cache)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["bookmaker"] = "pinnacle"
    df[["nv_over","nv_under"]] = df.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return df

def compute_clv(bets, close_df, book):
    sub = close_df[close_df["bookmaker"] == book][
        ["game_date","player_name","line","nv_over","nv_under"]].copy()
    m = bets.merge(sub,
                   left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"],
                   how="left")
    matched = m.dropna(subset=["nv_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    return matched

# Calibration weights — inverse Brier, normalized
BOOK_BRIER = {
    "betrivers":   0.2125,
    "fanduel":     0.2474,
    "betonlineag": 0.2476,
    "draftkings":  0.2478,
    "pinnacle":    0.2484,
}
raw_w = {b: 1 / v for b, v in BOOK_BRIER.items()}
total_w = sum(raw_w.values())
WEIGHTS = {b: v / total_w for b, v in raw_w.items()}

BOOKS = ["draftkings", "betonlineag", "fanduel", "betrivers", "pinnacle"]

BANDS = [
    (0,  5,  "0-5%"),
    (5,  10, "5-10%"),
    (10, 15, "10-15%"),
    (15, 20, "15-20%"),
    (20, 25, "20-25%"),
    (25, 999,"25%+"),
    (0,  999,"ALL (0%+)"),
    (15, 999,"15%+ (bet)"),
]

# Load everything
print("Loading bets...")
bets25 = load_bets(0)
bets25_year = bets25[bets25["game_date"].dt.year == 2025]
bets26_year = bets25[bets25["game_date"].dt.year == 2026]
print(f"  All bets: n={len(bets25)}  |  2025: {len(bets25_year)}  |  2026: {len(bets26_year)}")

print("Building close index...")
local_close = pd.concat([
    build_close_index("data/odds/historical_pitcher_props_2025.csv",
                      ["draftkings","betonlineag","fanduel","betrivers"]),
    build_close_index("data/odds/full_2026_odds.csv",
                      ["draftkings","betonlineag","fanduel","betrivers"]),
], ignore_index=True)
pin_close = load_pinnacle_cache()
all_close = pd.concat([local_close, pin_close], ignore_index=True)
print(f"  Close rows: {len(all_close)}  |  Books: {dict(all_close['bookmaker'].value_counts())}")

SEP = "=" * 76

for year_label, bets in [
    (f"2025 in-sample      n={len(bets25_year)}", bets25_year),
    (f"2026 OOS            n={len(bets26_year)}", bets26_year),
    (f"COMBINED            n={len(bets25)}",       bets25),
]:
    print(f"\n{SEP}")
    print(f"DE-VIGGED CLV  --  {year_label}  (ALL edges)")
    print(SEP)
    print(f"  {'Book':<14} {'N':>5}  {'Match':>6}  {'Match%':>7}  {'CLV pp':>8}"
          f"  {'t':>6}  {'%Pos':>6}  {'Win%':>6}  {'Wt':>6}")
    print("  " + "-" * 74)

    book_results = {}
    for book in BOOKS:
        matched = compute_clv(bets, all_close, book)
        n = len(bets)
        nm = len(matched)
        if nm < 20:
            print(f"  {book:<14} {n:>5}  {nm:>6}  {'<20':>7}")
            continue
        mean_clv = matched["clv_pp"].mean()
        se = matched["clv_pp"].std() / nm ** 0.5
        t = mean_clv / se if se > 0 else 0
        pct_pos = (matched["clv_pp"] > 0).mean()
        win = matched["won"].mean()
        w = WEIGHTS[book]
        book_results[book] = (nm, mean_clv, t, pct_pos, win, w)
        print(f"  {book:<14} {n:>5}  {nm:>6}  {nm/n:>7.1%}  {mean_clv:>+7.3f}pp"
              f"  {t:>6.2f}  {pct_pos:>6.1%}  {win:>6.1%}  {w:>6.3f}")

    if book_results:
        numerator   = sum(r[1] * r[5] * r[0] for r in book_results.values())
        denominator = sum(r[5] * r[0] for r in book_results.values())
        wtd = numerator / denominator if denominator > 0 else float("nan")
        simple = np.mean([r[1] for r in book_results.values()])
        print()
        print(f"  {'SIMPLE AVG':<14}  {simple:>+7.3f}pp")
        print(f"  {'CALIB-WT AVG':<14}  {wtd:>+7.3f}pp  (inv-Brier × n weighted)")

    # Band breakdown using DK close
    print(f"\n  Edge-band breakdown  (DK close, all bets)")
    print(f"  {'Band':<10} {'N':>5}  {'CLV pp':>8}  {'t':>6}  {'%Pos':>6}  {'Win%':>6}  {'ROI':>7}")
    print("  " + "-" * 56)
    dk_matched = compute_clv(bets, all_close, "draftkings")
    for lo, hi, name in BANDS:
        sub = dk_matched[(dk_matched["edge_pct"] >= lo) & (dk_matched["edge_pct"] < hi)]
        if len(sub) < 10:
            continue
        m  = sub["clv_pp"].mean()
        se = sub["clv_pp"].std() / len(sub) ** 0.5
        t  = m / se if se > 0 else 0
        pp = (sub["clv_pp"] > 0).mean()
        # ROI
        def american_to_dec(o):
            o = float(o)
            return o / 100 + 1 if o > 0 else 100 / abs(o) + 1
        sub = sub.copy()
        sub["entry_odds"] = sub.apply(
            lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1)
        sub["decimal"] = sub["entry_odds"].apply(american_to_dec)
        sub["profit"] = sub.apply(lambda r: r["decimal"] - 1 if r["won"] else -1.0, axis=1)
        roi = sub["profit"].mean()
        print(f"  {name:<10} {len(sub):>5}  {m:>+7.3f}pp  {t:>6.2f}  {pp:>6.1%}"
              f"  {sub['won'].mean():>6.1%}  {roi:>+7.1%}")

    # Pinnacle band breakdown for comparison
    print(f"\n  Edge-band breakdown  (Pinnacle close)")
    print(f"  {'Band':<10} {'N':>5}  {'CLV pp':>8}  {'t':>6}  {'%Pos':>6}  {'Win%':>6}")
    print("  " + "-" * 48)
    pin_matched = compute_clv(bets, all_close, "pinnacle")
    for lo, hi, name in BANDS:
        sub = pin_matched[(pin_matched["edge_pct"] >= lo) & (pin_matched["edge_pct"] < hi)]
        if len(sub) < 10:
            continue
        m  = sub["clv_pp"].mean()
        se = sub["clv_pp"].std() / len(sub) ** 0.5
        t  = m / se if se > 0 else 0
        pp = (sub["clv_pp"] > 0).mean()
        print(f"  {name:<10} {len(sub):>5}  {m:>+7.3f}pp  {t:>6.2f}  {pp:>6.1%}"
              f"  {sub['won'].mean():>6.1%}")
