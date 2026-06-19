"""
Multi-book de-vigged CLV analysis using local data only.
Books: DraftKings, BetOnline, FanDuel, BetRivers
"""
import sys
import pandas as pd
import numpy as np
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

def load_bets(paths, edge_min=15):
    dfs = []
    from pathlib import Path
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
}
raw_w = {b: 1 / v for b, v in BOOK_BRIER.items()}
total_w = sum(raw_w.values())
WEIGHTS = {b: v / total_w for b, v in raw_w.items()}

# Load bets
bets25 = load_bets([("thresh_sel_2025_dk_edges.csv", "data/processed_2024")])
bets26 = load_bets([
    ("wf2026_p1_mar_apr_edges.csv", "data/processed"),
    ("wf2026_p2_may_edges.csv",     "data/processed_apr2026"),
    ("wf2026_p3_jun_edges.csv",     "data/processed"),
])
bets_all = (pd.concat([bets25, bets26], ignore_index=True)
              .sort_values("edge_pct", ascending=False)
              .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
              .reset_index(drop=True))
print(f"Bets: 2025={len(bets25)}  2026={len(bets26)}  total={len(bets_all)}")

# Build close index
print("Building local close index...")
local_close = pd.concat([
    build_close_index("data/odds/historical_pitcher_props_2025.csv",
                      list(BOOK_BRIER.keys())),
    build_close_index("data/odds/full_2026_odds.csv",
                      list(BOOK_BRIER.keys())),
], ignore_index=True)
print(f"Local close index: {len(local_close)} rows")
print("Books:", dict(local_close["bookmaker"].value_counts()))

BOOKS = ["draftkings", "betonlineag", "fanduel", "betrivers"]
SEP = "=" * 74

for year_label, bets in [
    (f"2025 in-sample  n={len(bets25)}", bets25),
    (f"2026 OOS walk-forward  n={len(bets26)}", bets26),
    (f"COMBINED  n={len(bets_all)}", bets_all),
]:
    print(f"\n{SEP}")
    print(f"DE-VIGGED CLV  --  {year_label}  (edge>=15%)")
    print(SEP)
    print(f"  {'Book':<14} {'N':>5}  {'Match':>6}  {'Match%':>7}  {'CLV pp':>8}  {'t':>6}  {'%Pos':>6}  {'Win%':>6}  {'Wt':>6}")
    print("  " + "-" * 72)

    book_results = {}
    for book in BOOKS:
        matched = compute_clv(bets, local_close, book)
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
        print(f"  {'CALIB-WT AVG':<14}  {wtd:>+7.3f}pp  (inv-Brier * n weighted)")
        # Kobe comparison
        print(f"\n  Note: Kobe's consensus CLV (Pinnacle+FD+BOL) = -2.38pp (n=918)")

    # Band breakdown using DK close
    print(f"\n  DK close  |  band breakdown  (edge>=15%)")
    print(f"  {'Band':<10} {'N':>5}  {'CLV pp':>8}  {'t':>6}  {'%Pos':>6}  {'Win%':>6}")
    print("  " + "-" * 46)
    dk_matched = compute_clv(bets, local_close, "draftkings")
    BANDS = [(15,20,"15-20%"),(20,25,"20-25%"),(25,999,"25%+"),(15,999,"15%+")]
    for lo, hi, name in BANDS:
        sub = dk_matched[(dk_matched["edge_pct"] >= lo) & (dk_matched["edge_pct"] < hi)]
        if len(sub) < 10:
            continue
        m  = sub["clv_pp"].mean()
        se = sub["clv_pp"].std() / len(sub) ** 0.5
        t  = m / se if se > 0 else 0
        pp = (sub["clv_pp"] > 0).mean()
        print(f"  {name:<10} {len(sub):>5}  {m:>+7.3f}pp  {t:>6.2f}  {pp:>6.1%}  {sub['won'].mean():>6.1%}")

print(f"\n{SEP}")
print("Pinnacle: NOT in local data - would need API fetch to add to weighted avg")
print("Without Pinnacle, above weighted avg uses 4 books: DK + BOL + FD + BetRivers")
