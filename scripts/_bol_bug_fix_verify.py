"""
Verify the GroupBy.last() NaN-mixing bug and confirm the fix.
The bug: pandas 1.4.2 GroupBy.last() returns last non-NaN per column,
so rows with (400, NaN) AFTER (-114, -114) give over=400, under=-114 (WRONG).
Fix: use .nth(-1) or tail(1) instead of .last() to get the actual last row.
"""
import sys, io
import pandas as pd, numpy as np
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def implied(o):
    o = float(o)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)

def devig_pair(over_o, under_o):
    if pd.isna(over_o) or pd.isna(under_o): return np.nan, np.nan
    ip_o = implied(over_o); ip_u = implied(under_o)
    d = ip_o + ip_u
    if d <= 0: return np.nan, np.nan
    return ip_o / d, ip_u / d

# ── 1. Reproduce the exact bug ────────────────────────────────────────────────
print("=" * 60)
print("1. REPRODUCE: GroupBy.last() NaN-mixing bug")
print("=" * 60)
df_bug = pd.DataFrame({'key': ['A','A'], 'over': [-114.0, 400.0], 'under': [-114.0, np.nan]})
print(f"\nInput (main-line first, alternate second):")
print(df_bug.to_string(index=False))
res_last = df_bug.groupby('key').last()
print(f"\nGroupBy.last() result: over={res_last['over'].values[0]}, under={res_last['under'].values[0]}")
if res_last['over'].values[0] == 400.0:
    print("  BUG CONFIRMED: over=400 taken from alternate row (row 2)")
    nv_o, nv_u = devig_pair(res_last['over'].values[0], res_last['under'].values[0])
    print(f"  devig_pair(400, -114) -> nv_under={nv_u:.4f}  (WRONG, expected ~0.5)")

# The correct fix: use .nth(-1) to get actual last row
res_nth = df_bug.sort_values('key').groupby('key', as_index=False).nth(-1)
print(f"\nth(-1) result: over={res_nth['over'].values[0]}, under={res_nth['under'].values[0]}")
nv_o, nv_u = devig_pair(res_nth['over'].values[0], res_nth['under'].values[0])
print(f"  devig_pair({res_nth['over'].values[0]}, {res_nth['under'].values[0]}) -> nv_under={nv_u:.4f}  (CORRECT: 0.5)")

# Alternative fix: keep over-only rows ONLY if no both-sided row exists
print("\n" + "=" * 60)
print("2. VERIFY: Paul Skenes 6.5 2026-04-13 raw data order in CSV")
print("=" * 60)
odds26 = pd.read_csv("data/odds/full_2026_odds.csv")
bol26 = odds26[(odds26["bookmaker"]=="betonlineag") & (odds26["snapshot_type"]=="close")].copy()
bol26["game_date"] = pd.to_datetime(bol26["game_date"])
bol26["fetched_at"] = pd.to_datetime(bol26["fetched_at"], errors="coerce")

case = bol26[(bol26["player_name"].str.contains("Skenes", na=False)) &
             (bol26["game_date"].dt.date == pd.Timestamp("2026-04-13").date()) &
             (bol26["line"] == 6.5)].copy()
print(f"\nRaw rows in CSV order:")
print(case[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))

# Simulate sort_values (stable, same timestamp):
sorted_c = case.sort_values("fetched_at")
print(f"\nAfter sort_values('fetched_at') (stable, same ts):")
print(sorted_c[["player_name","line","over_odds","under_odds"]].to_string(index=False))

# GroupBy.last() result:
gb_last = sorted_c.groupby(["game_date","player_name","line"]).last().reset_index()
row = gb_last.iloc[0]
print(f"\nGroupBy.last() result: over={row['over_odds']}, under={row['under_odds']}")
nv_o, nv_u = devig_pair(row["over_odds"], row["under_odds"])
print(f"devig_pair -> nv_under={nv_u:.4f}")

# nth(-1) result:
gb_nth = sorted_c.groupby(["game_date","player_name","line"], as_index=False).nth(-1)
row_nth = gb_nth.iloc[0]
print(f"\nnth(-1) result: over={row_nth['over_odds']}, under={row_nth['under_odds']}")
nv_o2, nv_u2 = devig_pair(row_nth["over_odds"], row_nth["under_odds"])
print(f"devig_pair -> nv_under={nv_u2:.4f}")

print("\n" + "=" * 60)
print("3. COUNT: How many BOL groups have over-only row AFTER both-sided row?")
print("=" * 60)

odds25 = pd.read_csv("data/odds/historical_pitcher_props_2025.csv")
bol_all = pd.concat([
    odds25[(odds25["bookmaker"]=="betonlineag") & (odds25["snapshot_type"]=="close")],
    odds26
], ignore_index=True)
bol_all["game_date"] = pd.to_datetime(bol_all["game_date"])
bol_all["fetched_at"] = pd.to_datetime(bol_all["fetched_at"], errors="coerce")

# For each group, check if alternates (no under_odds) appear AFTER main lines
bol_sort = bol_all.sort_values("fetched_at")
grp_sizes = bol_sort.groupby(["game_date","player_name","line"]).size()
multi_groups = grp_sizes[grp_sizes > 1].index.tolist()
print(f"Groups with >1 row: {len(multi_groups)}")

buggy = 0
total_multi = 0
for (gd, pn, ln), grp in bol_sort.groupby(["game_date","player_name","line"]):
    if len(grp) <= 1: continue
    total_multi += 1
    rows = grp.values  # already sorted by fetched_at
    # Check last row: does it have NaN under_odds?
    last = grp.iloc[-1]
    second_last = grp.iloc[-2]
    if pd.isna(last["under_odds"]) and pd.notna(second_last["under_odds"]):
        buggy += 1

print(f"Multi-row groups: {total_multi}")
print(f"Groups where LAST row is over-only (bug-prone): {buggy}")
print(f"  -> These are cases where GroupBy.last() mixes rows from diff rows")

# Now show the actual correction: use nth(-1) approach
print("\n" + "=" * 60)
print("4. FIX: Correct deduplication using nth(-1) vs last()")
print("=" * 60)

def build_close_fixed(odds_df, books):
    """Fixed: use nth(-1) to get actual last row, not last non-NaN per column."""
    o = odds_df[(odds_df["snapshot_type"] == "close") & (odds_df["bookmaker"].isin(books))].copy()
    o["game_date"] = pd.to_datetime(o["game_date"])
    o["fetched_at"] = pd.to_datetime(o["fetched_at"], errors="coerce")
    o = (o.sort_values("fetched_at")
          .groupby(["game_date","bookmaker","player_name","line"], as_index=False)
          .nth(-1))
    o[["nv_over","nv_under"]] = o.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return o

def build_close_buggy(odds_df, books):
    """Buggy: GroupBy.last() mixes columns."""
    o = odds_df[(odds_df["snapshot_type"] == "close") & (odds_df["bookmaker"].isin(books))].copy()
    o["game_date"] = pd.to_datetime(o["game_date"])
    o["fetched_at"] = pd.to_datetime(o["fetched_at"], errors="coerce")
    o = (o.sort_values("fetched_at")
          .groupby(["game_date","bookmaker","player_name","line"])
          .last().reset_index())
    o[["nv_over","nv_under"]] = o.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return o

books = ["betonlineag", "draftkings", "fanduel", "betrivers"]
close_buggy = pd.concat([
    build_close_buggy(odds25, books),
    build_close_buggy(odds26, books),
], ignore_index=True)
close_fixed = pd.concat([
    build_close_fixed(odds25, books),
    build_close_fixed(odds26, books),
], ignore_index=True)

# Check the Skenes case
for label, cl in [("BUGGY (last)", close_buggy), ("FIXED (nth(-1))", close_fixed)]:
    bol_sub = cl[(cl["bookmaker"]=="betonlineag") &
                 cl["player_name"].str.contains("Skenes", na=False) &
                 (cl["game_date"].dt.date == pd.Timestamp("2026-04-13").date()) &
                 (cl["line"]==6.5)]
    if len(bol_sub):
        r = bol_sub.iloc[0]
        print(f"\n{label}: over={r['over_odds']}, under={r['under_odds']} -> nv_under={r['nv_under']:.4f}")
    else:
        print(f"\n{label}: NO MATCH")

# Now run full CLV comparison
print("\n" + "=" * 60)
print("5. FULL CLV COMPARISON: Buggy vs Fixed")
print("=" * 60)

def load_bets(edge_min=15):
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
    return bets

def get_clv(bets, close, book):
    sub = close[close["bookmaker"] == book][["game_date","player_name","line","nv_over","nv_under"]].copy()
    m = bets.merge(sub, left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"], how="left")
    matched = m.dropna(subset=["nv_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    return matched

bets = load_bets(15)
print(f"\nTotal bets (edge>=15%): {len(bets)}")

for label, close in [("BUGGY (last)", close_buggy), ("FIXED (nth(-1))", close_fixed)]:
    print(f"\n  --- {label} ---")
    for book in ["betonlineag", "draftkings", "fanduel", "betrivers"]:
        clv = get_clv(bets, close, book)
        if len(clv) == 0:
            print(f"  {book}: no matches")
            continue
        mean = clv["clv_pp"].mean()
        t = mean / (clv["clv_pp"].std() / len(clv)**0.5)
        print(f"  {book:<14}: {mean:>+.3f}pp  t={t:>5.2f}  n={len(clv)}")
