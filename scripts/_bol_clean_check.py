"""
Clean BOL CLV check — correct methodology, verify the +6.18pp claim.
Also: why does Kobe get +0.31pp on Pinnacle while we get +1.62pp?
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
    bets["won"] = bets.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                  else (r["strikeouts"] < r["line"]), axis=1)
    bets[["nv_entry_over","nv_entry_under"]] = bets.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    bets["nv_entry_side"] = bets.apply(
        lambda r: r["nv_entry_over"] if r["best_side"] == "over" else r["nv_entry_under"], axis=1)
    return bets

def build_close(odds_path, books):
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

bets = load_bets(15)
print(f"Bets (edge>=15%): {len(bets)}")

close = pd.concat([
    build_close("data/odds/historical_pitcher_props_2025.csv", ["betonlineag","draftkings"]),
    build_close("data/odds/full_2026_odds.csv", ["betonlineag","draftkings"]),
], ignore_index=True)

def get_clv(bets, close, book):
    sub = close[close["bookmaker"] == book][["game_date","player_name","line","nv_over","nv_under"]].copy()
    m = bets.merge(sub, left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"], how="left")
    matched = m.dropna(subset=["nv_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    return matched

dk_clv  = get_clv(bets, close, "draftkings")
bol_clv = get_clv(bets, close, "betonlineag")

print(f"\nDK  matched: {len(dk_clv)},  mean CLV: {dk_clv['clv_pp'].mean():+.3f}pp")
print(f"BOL matched: {len(bol_clv)},  mean CLV: {bol_clv['clv_pp'].mean():+.3f}pp")
print(f"\nBOL CLV distribution:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  p{p:>2}: {bol_clv['clv_pp'].quantile(p/100):>+.2f}pp")

print(f"\nBOL CLV > 15pp: {(bol_clv['clv_pp']>15).sum()} bets")
print(f"BOL CLV > 10pp: {(bol_clv['clv_pp']>10).sum()} bets")
print(f"BOL CLV > 5pp:  {(bol_clv['clv_pp']>5).sum()} bets")
print(f"BOL CLV < 0pp:  {(bol_clv['clv_pp']<0).sum()} bets")

# Spot-check the extreme cases
print("\n=== SPOT CHECK: Specific extreme bets (per _bol_deep_check) ===")
cases = [
    ("Paul Skenes",   "2026-04-13", 6.5, "under"),
    ("Tarik Skubal",  "2026-04-12", 7.5, "under"),
    ("Cole Ragans",   "2026-04-14", 6.5, "under"),
    ("Tyler Glasnow", "2025-04-27", 6.5, "under"),
    ("Hunter Greene", "2025-05-28", 5.5, "under"),
]
for name, date, line, side in cases:
    # DK
    dk_row = dk_clv[(dk_clv["pitcher_name"].str.contains(name.split()[-1], na=False)) &
                    (dk_clv["game_date"].astype(str).str[:10] == date) &
                    (dk_clv["line"] == line) & (dk_clv["best_side"] == side)]
    bol_row = bol_clv[(bol_clv["pitcher_name"].str.contains(name.split()[-1], na=False)) &
                      (bol_clv["game_date"].astype(str).str[:10] == date) &
                      (bol_clv["line"] == line) & (bol_clv["best_side"] == side)]
    dk_val  = dk_row["clv_pp"].values[0] if len(dk_row) else float("nan")
    bol_val = bol_row["clv_pp"].values[0] if len(bol_row) else float("nan")
    bol_nv  = bol_row["nv_close_side"].values[0] if len(bol_row) else float("nan")
    entry   = bol_row["nv_entry_side"].values[0] if len(bol_row) else (dk_row["nv_entry_side"].values[0] if len(dk_row) else float("nan"))
    print(f"  {name} {date} {line} {side}: dk={dk_val:>+.2f}pp  bol={bol_val:>+.2f}pp  "
          f"(entry_nv={entry:.4f}  bol_close_nv={bol_nv:.4f})")

# Mean CLV if we exclude outliers
print("\n=== BOL CLV ROBUSTNESS ===")
full_mean = bol_clv["clv_pp"].mean()
no_outliers = bol_clv[bol_clv["clv_pp"].abs() <= 15]["clv_pp"].mean()
median = bol_clv["clv_pp"].median()
print(f"  Full mean:           {full_mean:>+.3f}pp  (n={len(bol_clv)})")
print(f"  Excluding |CLV|>15:  {no_outliers:>+.3f}pp  (n={(bol_clv['clv_pp'].abs()<=15).sum()})")
print(f"  Median:              {median:>+.3f}pp")

# What book does BOL actually look like? Low/high lines?
print("\n=== BOL CLOSE ODDS STRUCTURE ===")
bol_sub = close[close["bookmaker"] == "betonlineag"].dropna(subset=["nv_over"]).copy()
print(f"BOL close rows with valid both-sided: {len(bol_sub)}")
bol_sub["vig"] = bol_sub.apply(
    lambda r: implied(r["over_odds"]) + implied(r["under_odds"]) - 1, axis=1)
print(f"BOL vig: mean={bol_sub['vig'].mean()*100:.2f}%  median={bol_sub['vig'].median()*100:.2f}%")
print(f"over_odds mean: {bol_sub['over_odds'].mean():.0f}")
print(f"Sample BOL main-line rows:")
print(bol_sub[["player_name","line","over_odds","under_odds","nv_over","nv_under"]].head(10).to_string(index=False))
