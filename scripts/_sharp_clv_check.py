"""
Investigate the sharp CLV result:
1. Compute BetOnline-based sharp CLV on our local 2026 data
2. Check whether the 47-bet 2026 sample is a coverage artifact
3. Break down win rate by coverage period
"""
import pandas as pd, numpy as np
from pathlib import Path

SEP = "=" * 65

# ── Load walk-forward 2026 bets ────────────────────────────────────
dfs = []
for f, d in [("wf2026_p1_mar_apr_edges.csv", "data/processed"),
             ("wf2026_p2_may_edges.csv",      "data/processed_apr2026"),
             ("wf2026_p3_jun_edges.csv",       "data/processed")]:
    p = Path(d) / f
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["market"] == "strikeouts"].copy()
        dfs.append(df)

wf = pd.concat(dfs, ignore_index=True)
wf = (wf.sort_values("edge_pct", ascending=False)
        .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
        .reset_index(drop=True))
wf["game_date"] = pd.to_datetime(wf["game_date"])
wf["won"] = wf.apply(
    lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
              else (r["strikeouts"] < r["line"]), axis=1)

e15 = wf[wf["edge_pct"] >= 15].copy()
print(f"\n{SEP}")
print("2026 WALK-FORWARD — Win rate by month (edge>=15%)")
print(SEP)
e15["month"] = e15["game_date"].dt.to_period("M")
for m, g in e15.groupby("month"):
    print(f"  {m}  n={len(g):>4}  win={g['won'].mean():.1%}  "
          f"side_mix: {g['best_side'].value_counts().to_dict()}")

# ── BetOnline coverage cutoff ──────────────────────────────────────
BOL_CUTOFF = pd.Timestamp("2026-05-22")
before = e15[e15["game_date"] <= BOL_CUTOFF]
after  = e15[e15["game_date"] >  BOL_CUTOFF]

print(f"\n  Mar–May 22 (matches BOL coverage window): n={len(before)}, win={before['won'].mean():.1%}")
print(f"  May 23–Jun (outside BOL coverage):        n={len(after)},  win={after['won'].mean():.1%}")

# ── Load BetOnline close & compute sharp CLV ──────────────────────
print(f"\n{SEP}")
print("BetOnline Sharp CLV (best local proxy for Pinnacle)")
print(SEP)

odds = pd.read_csv("data/odds/full_2026_odds.csv")
bol = odds[(odds["bookmaker"] == "betonlineag") & (odds["snapshot_type"] == "close")].copy()
bol["game_date"] = (pd.to_datetime(bol["commence_time"], utc=True, errors="coerce")
                      .dt.tz_localize(None).dt.normalize())

# Aggregate to one row per player/line/date (take most recent BOL close)
bol["fetched_at"] = pd.to_datetime(bol["fetched_at"], errors="coerce")
bol_agg = (bol.sort_values("fetched_at")
              .groupby(["game_date","player_name","line"])
              .last()
              .reset_index()[["game_date","player_name","line","over_odds","under_odds"]])
bol_agg.columns = ["game_date","player_name","line","bol_over","bol_under"]

# Merge with bets (pitcher_name -> player_name)
merged = e15.merge(bol_agg,
                   left_on=["game_date","pitcher_name","line"],
                   right_on=["game_date","player_name","line"],
                   how="left")

matched = merged.dropna(subset=["bol_over"])
unmatched = merged[merged["bol_over"].isna()]

print(f"  Edge>=15% bets:      {len(e15)}")
print(f"  Matched to BOL:      {len(matched)}  ({len(matched)/len(e15):.1%})")
print(f"  Unmatched (no BOL):  {len(unmatched)}")

if len(matched) > 0:
    # Compute CLV
    def american_to_decimal(o):
        return o / 100 + 1 if o > 0 else 100 / abs(o) + 1

    matched = matched.copy()
    matched["entry_odds"] = matched.apply(
        lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1)
    matched["close_odds"]  = matched.apply(
        lambda r: r["bol_over"] if r["best_side"] == "over" else r["bol_under"], axis=1)

    matched["entry_dec"]  = matched["entry_odds"].apply(american_to_decimal)
    matched["close_dec"]  = matched["close_odds"].apply(american_to_decimal)
    matched["bol_clv_pct"] = (matched["entry_dec"] / matched["close_dec"] - 1) * 100

    n = len(matched)
    mean_clv = matched["bol_clv_pct"].mean()
    se = matched["bol_clv_pct"].std() / n**0.5
    t = mean_clv / se

    print(f"\n  BOL Sharp CLV:  mean={mean_clv:+.3f}%  t={t:.2f}  n={n}")
    print(f"  % positive CLV: {(matched['bol_clv_pct'] > 0).mean():.1%}")
    print(f"  Matched win rate: {matched['won'].mean():.1%}")
    print(f"\n  DK CLV comparison on same matched subset:")

    # Also compute DK CLV for same matched subset
    matched["dk_entry"] = matched.apply(
        lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1)

    # Try to get DK close from 2026 full odds
    dk_close_raw = odds[(odds["bookmaker"] == "draftkings") & (odds["snapshot_type"] == "close")].copy()
    if len(dk_close_raw) > 0:
        dk_close_raw["game_date"] = (pd.to_datetime(dk_close_raw["commence_time"], utc=True, errors="coerce")
                                       .dt.tz_localize(None).dt.normalize())
        dk_close_raw["fetched_at"] = pd.to_datetime(dk_close_raw["fetched_at"], errors="coerce")
        dk_agg = (dk_close_raw.sort_values("fetched_at")
                              .groupby(["game_date","player_name","line"])
                              .last()
                              .reset_index()[["game_date","player_name","line","over_odds","under_odds"]])
        dk_agg.columns = ["game_date","player_name","line","dk_close_over","dk_close_under"]

        matched2 = matched.merge(dk_agg,
                                 left_on=["game_date","pitcher_name","line"],
                                 right_on=["game_date","player_name","line"],
                                 how="left")
        both = matched2.dropna(subset=["dk_close_over"])
        if len(both) > 0:
            both = both.copy()
            both["dk_close_odds"] = both.apply(
                lambda r: r["dk_close_over"] if r["best_side"] == "over" else r["dk_close_under"], axis=1)
            both["dk_close_dec"] = both["dk_close_odds"].apply(american_to_decimal)
            both["dk_entry_dec"] = both["entry_odds"].apply(american_to_decimal)
            both["dk_clv_pct"]   = (both["dk_entry_dec"] / both["dk_close_dec"] - 1) * 100

            print(f"  Subset with both DK+BOL close (n={len(both)}):")
            print(f"    DK CLV:  {both['dk_clv_pct'].mean():+.3f}%")
            print(f"    BOL CLV: {both['bol_clv_pct'].mean():+.3f}%")
            print(f"    DK - BOL divergence: {both['dk_clv_pct'].mean() - both['bol_clv_pct'].mean():+.3f} pp")

# ── Unmatched bets: are they different? ───────────────────────────
print(f"\n{SEP}")
print("Selection bias check — matched vs unmatched bets")
print(SEP)
if len(unmatched) > 0 and len(matched) > 0:
    for label, sub in [("Matched   (has BOL)", matched), ("Unmatched (no BOL)", unmatched)]:
        print(f"  {label}: n={len(sub):>4}  win={sub['won'].mean():.1%}  "
              f"mean_edge={sub['edge_pct'].mean():.1f}%  "
              f"over_pct={( sub['best_side']=='over').mean():.1%}")
