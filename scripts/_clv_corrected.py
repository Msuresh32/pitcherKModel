"""
Corrected multi-book CLV analysis.
Bug fix: filter to only both-sided BOL rows before deduplication.
GroupBy.last() on all-books data was mixing NaN rows for BOL alternate lines.
"""
import sys, io
import pandas as pd, numpy as np
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")

def implied(o):
    o = float(o)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)

def devig_pair(over_o, under_o):
    if pd.isna(over_o) or pd.isna(under_o): return np.nan, np.nan
    ip_o = implied(over_o); ip_u = implied(under_o)
    d = ip_o + ip_u
    if d <= 0: return np.nan, np.nan
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
    bets[["nv_entry_over","nv_entry_under"]] = bets.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    bets["nv_entry_side"] = bets.apply(
        lambda r: r["nv_entry_over"] if r["best_side"] == "over" else r["nv_entry_under"], axis=1)
    bets["won"] = bets.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                  else (r["strikeouts"] < r["line"]) if pd.notna(r["strikeouts"]) else np.nan, axis=1)
    return bets

def build_close_correct(odds_df, books):
    """
    Correct deduplication: filter to only both-sided rows first to avoid
    GroupBy.last() mixing NaN and non-NaN rows from different positions.
    """
    o = odds_df[(odds_df["snapshot_type"] == "close") & (odds_df["bookmaker"].isin(books))].copy()
    # Keep only rows with valid both-sided odds (drop alternates)
    o = o[o["over_odds"].notna() & o["under_odds"].notna()].copy()
    o["game_date"] = pd.to_datetime(o["game_date"])
    o["fetched_at"] = pd.to_datetime(o["fetched_at"], errors="coerce")
    o = (o.sort_values("fetched_at")
          .groupby(["game_date","bookmaker","player_name","line"])
          .last().reset_index())
    o[["nv_over","nv_under"]] = o.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return o

def get_clv(bets, close, book):
    sub = close[close["bookmaker"] == book][["game_date","player_name","line","nv_over","nv_under"]].copy()
    m = bets.merge(sub, left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"], how="left")
    matched = m.dropna(subset=["nv_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    return matched

# Load
print("Loading data...")
odds25 = pd.read_csv("data/odds/historical_pitcher_props_2025.csv")
odds26 = pd.read_csv("data/odds/full_2026_odds.csv")

LOCAL_BOOKS = ["betonlineag", "draftkings", "fanduel", "betrivers"]
close = pd.concat([
    build_close_correct(odds25, LOCAL_BOOKS),
    build_close_correct(odds26, LOCAL_BOOKS),
], ignore_index=True)

# Calibration weights (inverse Brier from walk-forward validation)
WEIGHTS = {
    "betrivers":   0.2125,
    "fanduel":     0.2474,
    "betonlineag": 0.2476,
    "draftkings":  0.2478,
}

for edge_min in [15, 0]:
    bets = load_bets(edge_min)
    label = f"edge >= {edge_min}%"
    print(f"\n{'='*60}")
    print(f"CLV ANALYSIS ({label}, n={len(bets)})")
    print(f"{'='*60}")

    results = {}
    for book in LOCAL_BOOKS:
        clv = get_clv(bets, close, book)
        if len(clv) == 0:
            print(f"  {book:<14}: NO MATCHES")
            continue
        mean = clv["clv_pp"].mean()
        std = clv["clv_pp"].std()
        t = mean / (std / len(clv)**0.5)
        results[book] = {"mean": mean, "t": t, "n": len(clv)}
        print(f"  {book:<14}: {mean:>+.3f}pp  t={t:>5.2f}  n={len(clv)}")

    # Also load Pinnacle
    pin_cache = Path("data/odds/pinnacle_close_cache.csv")
    if pin_cache.exists():
        pin = pd.read_csv(pin_cache)
        pin["game_date"] = pd.to_datetime(pin["game_date"])
        # Pinnacle cache: one row per game_date/player/line (already deduplicated at fetch time)
        pin = pin[pin["over_odds"].notna() & pin["under_odds"].notna()].copy()
        pin = (pin.drop_duplicates(subset=["game_date","player_name","line"])
                  .reset_index(drop=True))
        pin[["nv_over","nv_under"]] = pin.apply(
            lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
        pin["bookmaker"] = "pinnacle"
        pin_sub = pin[["game_date","player_name","line","nv_over","nv_under","bookmaker"]].copy()
        m = bets.merge(pin_sub, left_on=["game_date","pitcher_name","line"],
                       right_on=["game_date","player_name","line"], how="left")
        pin_matched = m.dropna(subset=["nv_over"]).copy()
        pin_matched["nv_close_side"] = pin_matched.apply(
            lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
        pin_matched["clv_pp"] = (pin_matched["nv_close_side"] - pin_matched["nv_entry_side"]) * 100
        mean_p = pin_matched["clv_pp"].mean()
        t_p = mean_p / (pin_matched["clv_pp"].std() / len(pin_matched)**0.5)
        results["pinnacle"] = {"mean": mean_p, "t": t_p, "n": len(pin_matched)}
        print(f"  {'pinnacle':<14}: {mean_p:>+.3f}pp  t={t_p:>5.2f}  n={len(pin_matched)}")

    # Weighted average (calibration weights)
    pin_weight = 0.2484
    all_weights = {**WEIGHTS, "pinnacle": pin_weight}
    total_w = sum(all_weights[b] for b in results)
    wavg = sum(all_weights[b] * results[b]["mean"] for b in results) / total_w
    print(f"\n  Calibration-weighted avg: {wavg:>+.3f}pp  (books: {list(results.keys())})")

# Edge band breakdown (edge>=0 only)
print(f"\n{'='*60}")
print(f"CLV BY EDGE BAND (DK, corrected)")
print(f"{'='*60}")
bets_all = load_bets(0)
dk_close = close[close["bookmaker"]=="draftkings"]
dk_all = get_clv(bets_all, close, "draftkings")
BANDS = [
    (0,  5,  "0-5%"), (5,  10, "5-10%"), (10, 15, "10-15%"),
    (15, 20, "15-20%"), (20, 25, "20-25%"), (25, 999,"25%+"),
    (0,  999,"ALL (0%+)"), (15, 999,"15%+ (bet)"),
]
print(f"{'Band':<15} {'n':>5} {'CLV':>8} {'t':>6} {'WinRate':>8}")
for lo, hi, name in BANDS:
    sub = dk_all[(dk_all["edge_pct"]>=lo) & (dk_all["edge_pct"]<hi)]
    if len(sub) < 5: continue
    mean = sub["clv_pp"].mean()
    t = mean / (sub["clv_pp"].std() / len(sub)**0.5)
    won = sub["won"].dropna()
    wr = won.mean() if len(won) > 0 else float("nan")
    print(f"{name:<15} {len(sub):>5} {mean:>+7.3f}pp {t:>6.2f} {wr:>8.1%}")

# Kobe's claim: Pinnacle corrected should be +0.31pp
print(f"\n{'='*60}")
print(f"PINNACLE CLV (vs Kobe's +0.31pp claim)")
print(f"{'='*60}")
if "pinnacle" in results:
    r = results["pinnacle"]
    print(f"  Our corrected Pinnacle CLV: {r['mean']:>+.3f}pp  (t={r['t']:.2f}, n={r['n']})")
    print(f"  Kobe's claim:               +0.31pp  (t=2.52, n=918)")
    print(f"  Gap: {r['mean'] - 0.31:>+.3f}pp")
    print(f"\n  Note: if Kobe used 918 bets, his set differs from ours (n={r['n']}).")
    print(f"  Possible causes: different entry odds source, different date range,")
    print(f"  different player-name matching tolerance.")
