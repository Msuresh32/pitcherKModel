"""
Verify Kobe's two specific claims:
1. BetOnline +6.18pp is a vig artifact (raw close not de-vigged)
2. Corrected Pinnacle CLV should be ~+0.31pp not +1.62pp
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
    if pd.isna(over_o) or pd.isna(under_o):
        return np.nan, np.nan
    ip_o = implied(over_o)
    ip_u = implied(under_o)
    d = ip_o + ip_u
    if d <= 0:
        return np.nan, np.nan
    return ip_o / d, ip_u / d

# ── 1. BetOnline close data quality ──────────────────────────────
print("=" * 60)
print("1. BETONLINE CLOSE DATA QUALITY CHECK")
print("=" * 60)

for fname, year in [("historical_pitcher_props_2025.csv", 2025),
                    ("full_2026_odds.csv", 2026)]:
    odds = pd.read_csv(f"data/odds/{fname}")
    bol = odds[(odds["bookmaker"] == "betonlineag") &
               (odds["snapshot_type"] == "close")].copy()
    dk  = odds[(odds["bookmaker"] == "draftkings") &
               (odds["snapshot_type"] == "close")].copy()

    bol_both = bol[bol["over_odds"].notna() & bol["under_odds"].notna()].copy()
    bol_over_only = bol[bol["over_odds"].notna() & bol["under_odds"].isna()]
    bol_both["vig"] = bol_both.apply(
        lambda r: implied(r["over_odds"]) + implied(r["under_odds"]) - 1, axis=1)

    dk_both = dk[dk["over_odds"].notna() & dk["under_odds"].notna()].copy()
    dk_both["vig"] = dk_both.apply(
        lambda r: implied(r["over_odds"]) + implied(r["under_odds"]) - 1, axis=1)

    print(f"\n  {year} BetOnline close rows: {len(bol)}")
    print(f"    both over+under: {len(bol_both)} ({len(bol_both)/len(bol):.1%})")
    print(f"    over only (no under): {len(bol_over_only)} ({len(bol_over_only)/len(bol):.1%})")
    print(f"    BOL vig  mean={bol_both['vig'].mean()*100:.2f}%  "
          f"median={bol_both['vig'].median()*100:.2f}%  "
          f">5%_vig={( bol_both['vig']>0.05).sum()}")
    print(f"    DK  vig  mean={dk_both['vig'].mean()*100:.2f}%  "
          f"median={dk_both['vig'].median()*100:.2f}%  "
          f">5%_vig={(dk_both['vig']>0.05).sum()}")

    # Sample BOL rows with both sides
    print(f"\n  Sample BOL both-sided rows ({year}):")
    sample = bol_both[["player_name","line","over_odds","under_odds","vig"]].head(5)
    print(sample.to_string(index=False))

# ── 2. Entry odds: what book are they from? ───────────────────────
print("\n" + "=" * 60)
print("2. ENTRY ODDS SOURCE CHECK")
print("=" * 60)
bets = pd.read_csv("data/processed_2024/thresh_sel_2025_dk_edges.csv")
bets = bets[bets["market"] == "strikeouts"].copy()
print(f"  2025 edges CSV — sample columns with odds:")
cols = [c for c in ["pitcher_name","game_date","line","best_side","over_odds",
                     "under_odds","best_odds","edge_pct"] if c in bets.columns]
print(bets[cols].head(8).to_string(index=False))
print(f"\n  over_odds stats: min={bets['over_odds'].min():.0f}  "
      f"max={bets['over_odds'].max():.0f}  "
      f"mean={bets['over_odds'].mean():.0f}")
print(f"  Suspiciously high over_odds (>+300, not typical K line): "
      f"{(bets['over_odds']>300).sum()}")

# ── 3. Diagnose BOL CLV directly ─────────────────────────────────
print("\n" + "=" * 60)
print("3. DIAGNOSE BOL CLV — WHY +6pp?")
print("=" * 60)

def load_bets_full(edge_min=15):
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
    bets["entry_implied_raw"] = bets.apply(
        lambda r: implied(r["over_odds"]) if r["best_side"] == "over"
                  else implied(r["under_odds"]), axis=1)
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
    # Also raw implied (vigged)
    odds["raw_over_implied"] = odds["over_odds"].apply(lambda x: implied(x) if pd.notna(x) else np.nan)
    odds["raw_under_implied"] = odds["under_odds"].apply(lambda x: implied(x) if pd.notna(x) else np.nan)
    return odds

bets = load_bets_full(15)
local_close = pd.concat([
    build_close_index("data/odds/historical_pitcher_props_2025.csv", ["betonlineag","draftkings"]),
    build_close_index("data/odds/full_2026_odds.csv", ["betonlineag","draftkings"]),
], ignore_index=True)

# BOL matched bets
bol_close = local_close[local_close["bookmaker"] == "betonlineag"][
    ["game_date","player_name","line","nv_over","nv_under",
     "raw_over_implied","raw_under_implied"]].copy()
m = bets.merge(bol_close,
               left_on=["game_date","pitcher_name","line"],
               right_on=["game_date","player_name","line"], how="left")
matched = m.dropna(subset=["nv_over"]).copy()
matched["nv_close_side"] = matched.apply(
    lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
matched["raw_close_side"] = matched.apply(
    lambda r: r["raw_over_implied"] if r["best_side"] == "over" else r["raw_under_implied"], axis=1)

# De-vigged CLV vs raw (vigged) CLV
matched["clv_devigged"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
matched["clv_raw_close"] = (matched["raw_close_side"] - matched["nv_entry_side"]) * 100
# Also: vigged entry vs devigged close (Kobe's original error methodology)
matched["clv_vigged_entry"] = (matched["nv_close_side"] - matched["entry_implied_raw"]) * 100

print(f"  BOL matched bets: {len(matched)}")
print(f"\n  CLV methods comparison:")
print(f"    De-vigged entry vs de-vigged close (CORRECT):  "
      f"{matched['clv_devigged'].mean():>+.3f}pp  "
      f"t={matched['clv_devigged'].mean()/(matched['clv_devigged'].std()/len(matched)**0.5):.2f}")
print(f"    De-vigged entry vs RAW close (vig artifact):    "
      f"{matched['clv_raw_close'].mean():>+.3f}pp")
print(f"    Vigged entry vs de-vigged close (Kobe's error): "
      f"{matched['clv_vigged_entry'].mean():>+.3f}pp")

print(f"\n  Entry vig check:")
print(f"    mean entry implied (raw):    {matched['entry_implied_raw'].mean():.4f}")
print(f"    mean entry implied (devigged): {matched['nv_entry_side'].mean():.4f}")
print(f"    diff (should be >0 if properly devigged): "
      f"{(matched['nv_entry_side'] - matched['entry_implied_raw']).mean()*100:+.4f}pp")

print(f"\n  BOL close vig check:")
bol_check = local_close[local_close["bookmaker"] == "betonlineag"].copy()
bol_check["vig"] = bol_check.apply(
    lambda r: (r["raw_over_implied"] + r["raw_under_implied"] - 1)
    if pd.notna(r["raw_over_implied"]) and pd.notna(r["raw_under_implied"])
    else np.nan, axis=1)
print(f"    BOL close rows with both sides: {bol_check['vig'].notna().sum()}")
print(f"    BOL close vig mean: {bol_check['vig'].mean()*100:.2f}%")
print(f"    DK close vig mean: ", end="")
dk_check = local_close[local_close["bookmaker"] == "draftkings"].copy()
dk_check["vig"] = dk_check.apply(
    lambda r: (r["raw_over_implied"] + r["raw_under_implied"] - 1)
    if pd.notna(r["raw_over_implied"]) and pd.notna(r["raw_under_implied"])
    else np.nan, axis=1)
print(f"{dk_check['vig'].mean()*100:.2f}%")

# ── 4. Scatter: BOL CLV vs DK CLV per bet ────────────────────────
print("\n" + "=" * 60)
print("4. BOL vs DK CLV CROSS-CHECK")
print("=" * 60)
dk_close = local_close[local_close["bookmaker"] == "draftkings"][
    ["game_date","player_name","line","nv_over","nv_under"]].copy()
m2 = bets.merge(dk_close.rename(columns={"nv_over":"dk_nv_over","nv_under":"dk_nv_under"}),
                left_on=["game_date","pitcher_name","line"],
                right_on=["game_date","player_name","line"], how="left")
m2 = m2.merge(bol_close[["game_date","player_name","line","nv_over","nv_under"]].rename(
    columns={"nv_over":"bol_nv_over","nv_under":"bol_nv_under","player_name":"player_name_bol"}),
              left_on=["game_date","pitcher_name","line"],
              right_on=["game_date","player_name_bol","line"], how="left")

both_matched = m2.dropna(subset=["dk_nv_over","bol_nv_over"]).copy()
both_matched["dk_close_side"]  = both_matched.apply(
    lambda r: r["dk_nv_over"] if r["best_side"] == "over" else r["dk_nv_under"], axis=1)
both_matched["bol_close_side"] = both_matched.apply(
    lambda r: r["bol_nv_over"] if r["best_side"] == "over" else r["bol_nv_under"], axis=1)
both_matched["dk_clv"]  = (both_matched["dk_close_side"]  - both_matched["nv_entry_side"]) * 100
both_matched["bol_clv"] = (both_matched["bol_close_side"] - both_matched["nv_entry_side"]) * 100
both_matched["bol_minus_dk"] = both_matched["bol_clv"] - both_matched["dk_clv"]

print(f"  Bets with BOTH DK and BOL close matched: {len(both_matched)}")
print(f"  DK CLV on this subset:  {both_matched['dk_clv'].mean():>+.3f}pp")
print(f"  BOL CLV on this subset: {both_matched['bol_clv'].mean():>+.3f}pp")
print(f"  BOL - DK gap:           {both_matched['bol_minus_dk'].mean():>+.3f}pp")
print(f"\n  Distribution of BOL-DK gap:")
print(f"    <-5pp: {(both_matched['bol_minus_dk']<-5).sum()}")
print(f"    -5 to 0pp: {((both_matched['bol_minus_dk']>=-5) & (both_matched['bol_minus_dk']<0)).sum()}")
print(f"    0 to +5pp: {((both_matched['bol_minus_dk']>=0) & (both_matched['bol_minus_dk']<5)).sum()}")
print(f"    >+5pp: {(both_matched['bol_minus_dk']>=5).sum()}")

print(f"\n  Sample rows (BOL-DK gap > 8pp, suspicious):")
big_gap = both_matched[both_matched["bol_minus_dk"] > 8][
    ["game_date","pitcher_name","line","best_side","over_odds","under_odds",
     "dk_clv","bol_clv","bol_minus_dk"]].head(8)
print(big_gap.to_string(index=False))
