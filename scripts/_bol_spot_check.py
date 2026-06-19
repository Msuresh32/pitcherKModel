"""
Spot-check the extreme BOL CLV cases — look at actual BOL rows for those bets.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
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

# Load raw BOL close data
print("Loading raw BOL close data...")
odds25 = pd.read_csv("data/odds/historical_pitcher_props_2025.csv")
odds26 = pd.read_csv("data/odds/full_2026_odds.csv")
bol25 = odds25[(odds25["bookmaker"]=="betonlineag") & (odds25["snapshot_type"]=="close")].copy()
bol26 = odds26[(odds26["bookmaker"]=="betonlineag") & (odds26["snapshot_type"]=="close")].copy()
bol_all = pd.concat([bol25, bol26], ignore_index=True)
bol_all["game_date"] = pd.to_datetime(bol_all["game_date"])
bol_all["fetched_at"] = pd.to_datetime(bol_all["fetched_at"], errors="coerce")

# Case 1: Paul Skenes 6.5 under 2026-04-13
print("\n=== Paul Skenes, 6.5, 2026-04-13 (ALL BOL rows) ===")
case1 = bol_all[(bol_all["player_name"].str.contains("Skenes", na=False)) &
                (bol_all["game_date"] == "2026-04-13")]
print(case1[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))

# Case 2: Tarik Skubal 7.5 under 2026-04-12
print("\n=== Tarik Skubal, 7.5, 2026-04-12 (ALL BOL rows) ===")
case2 = bol_all[(bol_all["player_name"].str.contains("Skubal", na=False)) &
                (bol_all["game_date"] == "2026-04-12")]
print(case2[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))

# Case 3: Cole Ragans 6.5 under 2026-04-14
print("\n=== Cole Ragans, 6.5, 2026-04-14 (ALL BOL rows) ===")
case3 = bol_all[(bol_all["player_name"].str.contains("Ragans", na=False)) &
                (bol_all["game_date"] == "2026-04-14")]
print(case3[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))

# Now show what build_close_index returns for these cases (groupby last)
print("\n=== What build_close_index returns for these cases ===")
bol_deduped = (bol_all.sort_values("fetched_at")
               .groupby(["game_date","player_name","line"])
               .last().reset_index())

for name, date, line in [("Paul Skenes", "2026-04-13", 6.5),
                          ("Tarik Skubal", "2026-04-12", 7.5),
                          ("Cole Ragans",  "2026-04-14", 6.5)]:
    row = bol_deduped[(bol_deduped["player_name"].str.contains(name.split()[-1], na=False)) &
                      (bol_deduped["game_date"].astype(str).str[:10] == date) &
                      (bol_deduped["line"] == line)]
    if len(row) == 0:
        print(f"\n  {name} {date} {line}: NO MATCH FOUND in deduped")
    else:
        print(f"\n  {name} {date} {line}:")
        print(row[["player_name","line","over_odds","under_odds","fetched_at"]].to_string(index=False))
        for _, r in row.iterrows():
            nv_over, nv_under = devig_pair(r["over_odds"], r["under_odds"])
            print(f"    → nv_over={nv_over:.4f}  nv_under={nv_under:.4f}")

# Also check: what is the DK close for these same cases?
dk_all = pd.concat([
    odds25[(odds25["bookmaker"]=="draftkings") & (odds25["snapshot_type"]=="close")],
    odds26[(odds26["bookmaker"]=="draftkings") & (odds26["snapshot_type"]=="close")],
], ignore_index=True)
dk_all["game_date"] = pd.to_datetime(dk_all["game_date"])

print("\n=== DK close for same cases ===")
for name, date, line in [("Paul Skenes", "2026-04-13", 6.5),
                          ("Tarik Skubal", "2026-04-12", 7.5),
                          ("Cole Ragans",  "2026-04-14", 6.5)]:
    row = dk_all[(dk_all["player_name"].str.contains(name.split()[-1], na=False)) &
                 (dk_all["game_date"].astype(str).str[:10] == date) &
                 (dk_all["line"] == line)]
    if len(row):
        print(f"\n  {name} DK close: {row[['player_name','line','over_odds','under_odds']].to_string(index=False)}")
        r = row.iloc[0]
        nv_o, nv_u = devig_pair(r["over_odds"], r["under_odds"])
        print(f"    → nv_over={nv_o:.4f}  nv_under={nv_u:.4f}")

# Summary: compute CLV manually for Skenes case
print("\n=== MANUAL CLV CALCULATION — Paul Skenes under 6.5 ===")
# Entry from bets file
import pathlib
bets = pd.read_csv("data/processed/wf2026_p1_mar_apr_edges.csv")
bets = bets[bets["market"]=="strikeouts"].copy()
skenes = bets[(bets["pitcher_name"].str.contains("Skenes", na=False)) &
              (bets["game_date"]=="2026-04-13") &
              (bets["best_side"]=="under")]
print("Entry bet:", skenes[["pitcher_name","game_date","line","best_side","over_odds","under_odds","edge_pct"]].to_string(index=False))
for _, r in skenes.iterrows():
    nv_o, nv_u = devig_pair(r["over_odds"], r["under_odds"])
    print(f"  Entry nv_under = {nv_u:.4f}")
