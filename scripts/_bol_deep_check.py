"""
Deep dive into BOL CLV anomaly.
1. What are the actual BOL close timestamps vs DK close timestamps?
2. Are the extreme BOL CLV outliers legitimate line moves?
3. What is BOL CLV with outliers trimmed?
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

# Load all odds with timestamps
def build_close_raw(odds_path, books):
    odds = pd.read_csv(odds_path)
    odds = odds[(odds["snapshot_type"] == "close") & (odds["bookmaker"].isin(books))].copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"])
    odds["fetched_at"] = pd.to_datetime(odds["fetched_at"], errors="coerce")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"], utc=True, errors="coerce")
    odds["fetched_at_utc"] = pd.to_datetime(odds["fetched_at"], utc=True, errors="coerce")
    # Minutes before game
    odds["mins_before_game"] = (
        (odds["commence_time"] - odds["fetched_at_utc"]) / pd.Timedelta(minutes=1)
    )
    # Keep last close per book/player/line/date
    best = (odds.sort_values("fetched_at")
               .groupby(["game_date","bookmaker","player_name","line"])
               .last().reset_index())
    best[["nv_over","nv_under"]] = best.apply(
        lambda r: pd.Series(devig_pair(r["over_odds"], r["under_odds"])), axis=1)
    return best

print("Loading data...")
bets = load_bets(15)

close25 = build_close_raw("data/odds/historical_pitcher_props_2025.csv", ["betonlineag","draftkings"])
close26 = build_close_raw("data/odds/full_2026_odds.csv", ["betonlineag","draftkings"])
close_all = pd.concat([close25, close26], ignore_index=True)

# ── 1. Snapshot timing comparison ─────────────────────────────────
print("\n" + "=" * 60)
print("1. CLOSE SNAPSHOT TIMING (mins before game)")
print("=" * 60)
for book in ["draftkings", "betonlineag"]:
    sub = close_all[close_all["bookmaker"] == book]
    sub = sub[sub["mins_before_game"].notna() & (sub["mins_before_game"] > 0) & (sub["mins_before_game"] < 500)]
    print(f"\n  {book}:")
    print(f"    mean:   {sub['mins_before_game'].mean():.1f} min before game")
    print(f"    median: {sub['mins_before_game'].median():.1f} min before game")
    print(f"    p10:    {sub['mins_before_game'].quantile(0.10):.1f}")
    print(f"    p90:    {sub['mins_before_game'].quantile(0.90):.1f}")
    print(f"    <30min: {(sub['mins_before_game']<30).mean():.1%}")
    print(f"    <8min:  {(sub['mins_before_game']<8).mean():.1%}")

# ── 2. BOL vs DK CLV — per-bet cross-check ────────────────────────
print("\n" + "=" * 60)
print("2. BOL vs DK CLV — DISTRIBUTION OF GAPS")
print("=" * 60)

def compute_clv_matched(bets, close_df, book):
    sub = close_df[close_df["bookmaker"] == book][
        ["game_date","player_name","line","nv_over","nv_under","mins_before_game"]].copy()
    m = bets.merge(sub,
                   left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"], how="left")
    matched = m.dropna(subset=["nv_over"]).copy()
    matched["nv_close_side"] = matched.apply(
        lambda r: r["nv_over"] if r["best_side"] == "over" else r["nv_under"], axis=1)
    matched["clv_pp"] = (matched["nv_close_side"] - matched["nv_entry_side"]) * 100
    matched["won"] = matched.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                  else (r["strikeouts"] < r["line"]), axis=1)
    return matched

dk_m  = compute_clv_matched(bets, close_all, "draftkings")
bol_m = compute_clv_matched(bets, close_all, "betonlineag")

# Merge both on same bets
both = dk_m[["game_date","pitcher_name","line","best_side","edge_pct",
             "nv_entry_side","clv_pp","won"]].rename(
    columns={"clv_pp":"dk_clv"}).merge(
    bol_m[["game_date","pitcher_name","line","best_side",
           "clv_pp","mins_before_game"]].rename(columns={"clv_pp":"bol_clv"}),
    on=["game_date","pitcher_name","line","best_side"], how="inner")

both["gap"] = both["bol_clv"] - both["dk_clv"]
print(f"\n  Bets with both DK and BOL close: {len(both)}")
print(f"  DK CLV:   {both['dk_clv'].mean():+.3f}pp  (std={both['dk_clv'].std():.2f})")
print(f"  BOL CLV:  {both['bol_clv'].mean():+.3f}pp  (std={both['bol_clv'].std():.2f})")
print(f"  BOL-DK gap mean: {both['gap'].mean():+.3f}pp")

print(f"\n  Gap distribution:")
for lo, hi, label in [(-999,-10,"< -10pp"),(-10,-5,"-10 to -5"),(-5,0,"-5 to 0"),
                       (0,5,"0 to +5"),(5,10,"+5 to +10"),(10,999,"> +10pp")]:
    n = ((both["gap"]>=lo) & (both["gap"]<hi)).sum()
    print(f"    {label:>12}: {n:>4} ({n/len(both):>5.1%})")

# Percentile of BOL CLV
print(f"\n  BOL CLV percentiles:")
for p in [5, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:>2}: {both['bol_clv'].quantile(p/100):>+.2f}pp")

# ── 3. Trimmed BOL CLV ────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. BOL CLV — WITH OUTLIER TREATMENT")
print("=" * 60)

# Full BOL sample (not just where DK also matches)
bol_full = bol_m.copy()
for label, sub in [("All BOL matched (n=%d)" % len(bol_full), bol_full),
                   ("BOL matched, DK also matched (n=%d)" % len(both), both.rename(columns={"bol_clv":"clv_pp"}))]:
    clv = sub["clv_pp"]
    mean = clv.mean()
    t = mean / (clv.std() / len(clv)**0.5)
    print(f"\n  {label}")
    print(f"    Full mean:       {mean:>+.3f}pp  t={t:.2f}")
    # Winsorize at 1%/99%
    lo99, hi99 = clv.quantile(0.01), clv.quantile(0.99)
    w = clv.clip(lo99, hi99)
    print(f"    Winsorized 1/99: {w.mean():>+.3f}pp  (clipped to [{lo99:.1f}, {hi99:.1f}])")
    # Trim top/bottom 5%
    t5 = clv[(clv >= clv.quantile(0.05)) & (clv <= clv.quantile(0.95))]
    print(f"    Trimmed 5%:      {t5.mean():>+.3f}pp  (n={len(t5)})")
    # Median
    print(f"    Median:          {clv.median():>+.3f}pp")

# ── 4. Are extreme BOL bets legitimate? ───────────────────────────
print("\n" + "=" * 60)
print("4. BOL EXTREME CLV BETS — ACTUAL OUTCOMES")
print("=" * 60)
extreme = both[both["bol_clv"] > 10].copy()
print(f"\n  Bets with BOL CLV > 10pp: {len(extreme)}")
if len(extreme) > 0:
    print(f"  Win rate: {extreme['won'].mean():.1%}  (expected BEP ~47%)")
    print(f"  Mean BOL CLV: {extreme['bol_clv'].mean():+.2f}pp")
    print(f"  Mean DK CLV:  {extreme['dk_clv'].mean():+.2f}pp")
    print(f"  Mean entry nv: {extreme['nv_entry_side'].mean():.3f}")
    print(f"\n  Sample extreme BOL bets:")
    print(extreme[["game_date","pitcher_name","line","best_side",
                   "nv_entry_side","dk_clv","bol_clv","won","mins_before_game"]].head(10).to_string(index=False))

normal = both[both["bol_clv"] <= 10].copy()
print(f"\n  Bets with BOL CLV <= 10pp: {len(normal)}")
print(f"  Win rate: {normal['won'].mean():.1%}")
print(f"  Mean BOL CLV: {normal['bol_clv'].mean():+.2f}pp")
print(f"  Mean DK CLV:  {normal['dk_clv'].mean():+.2f}pp")

# ── 5. BOL timing: is close actually close? ───────────────────────
print("\n" + "=" * 60)
print("5. BOL SNAPSHOT TIMING ON MATCHED BETS")
print("=" * 60)
both_timing = both[both["mins_before_game"].notna()]
print(f"  Bets with timing data: {len(both_timing)}")
if len(both_timing) > 0:
    print(f"  Mean mins before game: {both_timing['mins_before_game'].mean():.1f}")
    print(f"  Median: {both_timing['mins_before_game'].median():.1f}")
    over_2h = (both_timing["mins_before_game"] > 120).sum()
    print(f"  >2 hours before game: {over_2h} ({over_2h/len(both_timing):.1%}) -- NOT truly 'close'")
    print(f"  <30 min before game: {(both_timing['mins_before_game']<30).sum()}")
