"""
Trace exactly why BOL shows +27pp CLV for specific bets.
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

# Load 2026 BOL raw data
odds26 = pd.read_csv("data/odds/full_2026_odds.csv")
bol26 = odds26[(odds26["bookmaker"]=="betonlineag") & (odds26["snapshot_type"]=="close")].copy()
bol26["game_date"] = pd.to_datetime(bol26["game_date"])
bol26["fetched_at"] = pd.to_datetime(bol26["fetched_at"], errors="coerce")

# Test case: Paul Skenes 6.5, 2026-04-13
case = bol26[(bol26["player_name"].str.contains("Skenes", na=False)) &
             (bol26["game_date"].dt.date == pd.Timestamp("2026-04-13").date()) &
             (bol26["line"] == 6.5)].copy()

print("=== Paul Skenes 6.5, 2026-04-13 -- ALL BOL close rows ===")
print(case[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))

# Simulate build_close_index groupby
sorted_case = case.sort_values("fetched_at")
print("\nAfter sort_values('fetched_at'):")
print(sorted_case[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))

# What groupby last picks
gb = sorted_case.groupby(["game_date","player_name","line"])
last_rows = gb.last().reset_index()
print("\nAfter groupby.last():")
print(last_rows[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))
row = last_rows.iloc[0]
nv_o, nv_u = devig_pair(row["over_odds"], row["under_odds"])
print(f"\n  nv_over={nv_o}  nv_under={nv_u}")
print(f"  Entry nv_under=0.4514 (DK: over -139, under +109)")
print(f"  Expected CLV = {(nv_u - 0.4514)*100 if nv_u else 'N/A (NaN)':.2f}pp")

# Now check what sort_values with NaN fetched_at does
print("\n=== NaN fetched_at check ===")
print("NaN fetched_at rows:", case["fetched_at"].isna().sum())
print("fetched_at values:", case["fetched_at"].tolist())
print("sort NaN position: pandas sort_values puts NaN at end by default (na_position='last')")

# Build full BOL close index and check
all26 = pd.read_csv("data/odds/full_2026_odds.csv")
bol_close = all26[(all26["bookmaker"]=="betonlineag") & (all26["snapshot_type"]=="close")].copy()
bol_close["game_date"] = pd.to_datetime(bol_close["game_date"])
bol_close["fetched_at"] = pd.to_datetime(bol_close["fetched_at"], errors="coerce")
bol_dedup = (bol_close.sort_values("fetched_at")
             .groupby(["game_date","player_name","line"])
             .last().reset_index())
bol_dedup[["nv_over","nv_under"]] = bol_dedup.apply(
    lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)

skenes_dedup = bol_dedup[(bol_dedup["player_name"].str.contains("Skenes", na=False)) &
                          (bol_dedup["game_date"].dt.date == pd.Timestamp("2026-04-13").date())]
print("\n=== Deduplicated BOL for all Skenes lines 2026-04-13 ===")
print(skenes_dedup[["player_name","line","over_odds","under_odds","nv_over","nv_under"]].to_string(index=False))

# Check if there's a 6.5 row with valid nv_under
sk_6 = skenes_dedup[skenes_dedup["line"]==6.5]
print("\nSkenes 6.5 deduped row:")
print(sk_6[["over_odds","under_odds","nv_over","nv_under"]].to_string(index=False))

# Manual CLV
if len(sk_6) and not pd.isna(sk_6.iloc[0]["nv_under"]):
    nv_u = sk_6.iloc[0]["nv_under"]
    entry_nv_u = 0.451361
    print(f"\n  Manual CLV = ({nv_u:.6f} - {entry_nv_u:.6f}) * 100 = {(nv_u-entry_nv_u)*100:.3f}pp")
else:
    print("\n  No valid BOL close for Skenes 6.5 (NaN) -- this bet would be DROPPED from CLV")

# Also check across all 2026 bets: how many have "doubled" BOL rows (same player/line/date, diff odds)?
print("\n=== How many (player, date, line) groups in BOL have multiple rows? ===")
grp_sizes = bol_close.groupby(["game_date","player_name","line"]).size()
multi = grp_sizes[grp_sizes > 1]
print(f"  Groups with >1 row: {len(multi)} / {len(grp_sizes)}")
print(f"  Max rows in one group: {grp_sizes.max()}")

# For the extreme CLV cases, check the actual BOL close
extreme_cases = [
    ("Tarik Skubal", "2026-04-12", 7.5),
    ("Cole Ragans",  "2026-04-14", 6.5),
]
for name, date, line in extreme_cases:
    rows = bol_dedup[(bol_dedup["player_name"].str.contains(name.split()[-1], na=False)) &
                     (bol_dedup["game_date"].dt.date == pd.Timestamp(date).date()) &
                     (bol_dedup["line"] == line)]
    print(f"\n  {name} {date} line={line}:")
    if len(rows):
        print(rows[["player_name","line","over_odds","under_odds","nv_over","nv_under"]].to_string(index=False))
    else:
        print("  NOT FOUND")
