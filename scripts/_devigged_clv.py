"""
Proper de-vigged probability-point CLV using Kobe's exact methodology:
  1. De-vig entry: no_vig = implied / (implied_over + implied_under)
  2. De-vig close: same formula, same book
  3. CLV_pp = close_no_vig_side - entry_no_vig_side

Runs on 2025 threshold selection and 2026 walk-forward bets.
Uses DK close and BetOnline close separately so we can see both.
"""
import pandas as pd, numpy as np
from pathlib import Path

def american_to_prob(o):
    o = float(o)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)

def american_to_decimal(o):
    o = float(o)
    return o / 100 + 1 if o > 0 else 100 / abs(o) + 1

def devig_pair(over_odds, under_odds):
    """Returns (no_vig_over, no_vig_under). Returns NaN if either leg missing."""
    if pd.isna(over_odds) or pd.isna(under_odds):
        return np.nan, np.nan
    ip_o = american_to_prob(over_odds)
    ip_u = american_to_prob(under_odds)
    denom = ip_o + ip_u
    if denom <= 0:
        return np.nan, np.nan
    return ip_o / denom, ip_u / denom

SEP = "=" * 65

# ── Load bets ─────────────────────────────────────────────────────
def load_bets(paths, edge_min=15):
    dfs = []
    for path, d in paths:
        p = Path(d) / path
        if p.exists():
            df = pd.read_csv(p)
            df = df[df["market"] == "strikeouts"].copy()
            dfs.append(df)
    bets = pd.concat(dfs, ignore_index=True)
    bets = (bets.sort_values("edge_pct", ascending=False)
                .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
                .reset_index(drop=True))
    bets = bets[bets["edge_pct"] >= edge_min].copy()
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    bets["won"] = bets.apply(lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                              else (r["strikeouts"] < r["line"]), axis=1)
    # De-vig entry
    bets[["nv_entry_over","nv_entry_under"]] = bets.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    bets["nv_entry_side"] = bets.apply(
        lambda r: r["nv_entry_over"] if r["best_side"] == "over" else r["nv_entry_under"], axis=1)
    return bets

bets25 = load_bets([("thresh_sel_2025_dk_edges.csv","data/processed_2024")])
bets26 = load_bets([
    ("wf2026_p1_mar_apr_edges.csv","data/processed"),
    ("wf2026_p2_may_edges.csv","data/processed_apr2026"),
    ("wf2026_p3_jun_edges.csv","data/processed"),
])
print(f"2025 bets: {len(bets25)}   2026 bets: {len(bets26)}")

# ── Load close snapshots ──────────────────────────────────────────
def build_close_index(odds_path, books):
    odds = pd.read_csv(odds_path)
    odds = odds[odds["snapshot_type"] == "close"].copy()
    odds = odds[odds["bookmaker"].isin(books)]
    odds["game_date"] = pd.to_datetime(
        odds["commence_time"], utc=True, errors="coerce"
    ).dt.tz_localize(None).dt.normalize()
    odds["fetched_at"] = pd.to_datetime(odds["fetched_at"], errors="coerce")
    # Keep most-recent close snapshot per book/player/line/date
    odds = (odds.sort_values("fetched_at")
                .groupby(["game_date","bookmaker","player_name","line"])
                .last()
                .reset_index())
    # De-vig
    odds[["nv_close_over","nv_close_under"]] = odds.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return odds

close25 = build_close_index("data/odds/historical_pitcher_props_2025.csv",
                             ["draftkings","betonlineag"])
close26 = build_close_index("data/odds/full_2026_odds.csv",
                             ["draftkings","betonlineag"])
close_all = pd.concat([close25, close26], ignore_index=True)

# ── Merge and compute de-vigged CLV ──────────────────────────────
def compute_clv(bets, close_df, book):
    book_close = close_df[close_df["bookmaker"] == book].copy()
    book_close = book_close[["game_date","player_name","line",
                               "nv_close_over","nv_close_under"]].copy()

    merged = bets.merge(book_close,
                        left_on=["game_date","pitcher_name","line"],
                        right_on=["game_date","player_name","line"],
                        how="left")
    matched = merged.dropna(subset=["nv_close_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_close_over"] if r["best_side"] == "over"
                  else r["nv_close_under"], axis=1)
    # CLV in probability points (Kobe's exact formula)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    return matched

# ── Analysis ──────────────────────────────────────────────────────
BANDS = [(15,20,"15-20%"),(20,25,"20-25%"),(25,999,"25%+"),(15,999,"15%+")]

for year_label, bets in [("2025", bets25), ("2026 OOS", bets26)]:
    print(f"\n{SEP}")
    print(f"DE-VIGGED CLV  —  {year_label}  (edge>=15%)")
    print(SEP)
    print(f"  {'Book':<12} {'N':>5}  {'Matched':>8}  {'Match%':>7}  "
          f"{'Mean CLV pp':>12}  {'t-stat':>7}  {'%Pos':>6}  {'Win%':>6}")
    print("  " + "-"*72)

    for book in ["draftkings","betonlineag"]:
        matched = compute_clv(bets, close_all, book)
        n_total = len(bets)
        n_match = len(matched)
        if n_match < 20:
            print(f"  {book:<12} —  <20 matched")
            continue
        mean_clv = matched["clv_pp"].mean()
        se       = matched["clv_pp"].std() / n_match**0.5
        t        = mean_clv / se if se > 0 else 0
        pct_pos  = (matched["clv_pp"] > 0).mean()
        win      = matched["won"].mean()
        print(f"  {book:<12} {n_total:>5}  {n_match:>8}  {n_match/n_total:>7.1%}  "
              f"  {mean_clv:>+10.3f}pp  {t:>7.2f}  {pct_pos:>6.1%}  {win:>6.1%}")

    print()

    # Band breakdown (DK close only)
    dk_matched = compute_clv(bets, close_all, "draftkings")
    if len(dk_matched) < 20:
        continue
    print(f"  DK de-vigged CLV by edge band:")
    print(f"  {'Band':<10} {'N':>5}  {'CLV pp':>8}  {'t':>6}  {'%Pos':>6}  {'Win%':>6}")
    print("  " + "-"*46)
    for lo, hi, name in BANDS:
        sub = dk_matched[(dk_matched["edge_pct"] >= lo) & (dk_matched["edge_pct"] < hi)]
        if len(sub) < 10:
            continue
        m  = sub["clv_pp"].mean()
        se = sub["clv_pp"].std() / len(sub)**0.5
        t  = m / se if se > 0 else 0
        pp = (sub["clv_pp"] > 0).mean()
        print(f"  {name:<10} {len(sub):>5}  {m:>+7.3f}pp  {t:>6.2f}  {pp:>6.1%}  "
              f"{sub['won'].mean():>6.1%}")

    # Over/under split
    print(f"\n  DK de-vigged CLV by side (15%+):")
    e15 = dk_matched[dk_matched["edge_pct"] >= 15]
    for side in ["over","under"]:
        sub = e15[e15["best_side"] == side]
        if len(sub) < 10: continue
        m  = sub["clv_pp"].mean()
        se = sub["clv_pp"].std() / len(sub)**0.5
        t  = m / se if se > 0 else 0
        print(f"  {side:<8}  n={len(sub):>4}  CLV={m:>+7.3f}pp  t={t:>6.2f}  "
              f"win={sub['won'].mean():.1%}")

    # Monthly (DK, 15%+)
    print(f"\n  DK de-vigged CLV by month (15%+):")
    e15 = dk_matched[dk_matched["edge_pct"] >= 15].copy()
    e15["month"] = e15["game_date"].dt.to_period("M")
    print(f"  {'Month':<10} {'N':>5}  {'CLV pp':>8}  {'Win%':>6}")
    for m, g in e15.groupby("month"):
        mc = g["clv_pp"].mean()
        print(f"  {str(m):<10} {len(g):>5}  {mc:>+7.3f}pp  {g['won'].mean():>6.1%}")
