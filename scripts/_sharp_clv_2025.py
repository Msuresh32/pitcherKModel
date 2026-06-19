"""
Compute BetOnline-based sharp CLV for 2025 threshold selection bets.
Compare to the -2.58 pp Pinnacle result from the other system.
"""
import pandas as pd, numpy as np
from pathlib import Path

def american_to_decimal(o):
    return o / 100 + 1 if o > 0 else 100 / abs(o) + 1

# Load 2025 bets
bets = pd.read_csv("data/processed_2024/thresh_sel_2025_dk_edges.csv")
bets = bets[bets["market"] == "strikeouts"].copy()
bets = (bets.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))
bets = bets[bets["edge_pct"] >= 15].copy()
bets["game_date"] = pd.to_datetime(bets["game_date"])
bets["won"] = bets.apply(
    lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
              else (r["strikeouts"] < r["line"]), axis=1)
bets["entry_odds"] = bets.apply(
    lambda r: r["over_odds"] if r["best_side"] == "over" else r["under_odds"], axis=1)
print(f"2025 edge>=15% bets: {len(bets)}")

# Load 2025 multi-book close data
odds25 = pd.read_csv("data/odds/historical_pitcher_props_2025.csv")
# Check if odds/under cols present
if "over_odds" not in odds25.columns:
    print("No over_odds in 2025 odds — checking columns")
    print(list(odds25.columns))
else:
    # Get BetOnline close
    bol_close = odds25[(odds25["bookmaker"] == "betonlineag") & (odds25["snapshot_type"] == "close")].copy()
    bol_close["game_date"] = (pd.to_datetime(bol_close["commence_time"], utc=True, errors="coerce")
                               .dt.tz_localize(None).dt.normalize())
    bol_close["fetched_at"] = pd.to_datetime(bol_close["fetched_at"], errors="coerce")
    bol_agg = (bol_close.sort_values("fetched_at")
                        .groupby(["game_date","player_name","line"])
                        .last()
                        .reset_index()[["game_date","player_name","line","over_odds","under_odds"]])
    bol_agg.columns = ["game_date","player_name","line","bol_over","bol_under"]
    print(f"BOL close rows (2025): {len(bol_agg)}")
    print(f"BOL date range: {bol_agg['game_date'].min()} -> {bol_agg['game_date'].max()}")

    merged = bets.merge(bol_agg,
                        left_on=["game_date","pitcher_name","line"],
                        right_on=["game_date","player_name","line"],
                        how="left")
    matched = merged.dropna(subset=["bol_over"]).copy()
    unmatched = merged[merged["bol_over"].isna()]
    print(f"\nMatched to BOL: {len(matched)}/{len(bets)} ({len(matched)/len(bets):.1%})")

    if len(matched) > 0:
        matched["close_odds"] = matched.apply(
            lambda r: r["bol_over"] if r["best_side"] == "over" else r["bol_under"], axis=1)
        matched["entry_dec"] = matched["entry_odds"].apply(american_to_decimal)
        matched["close_dec"] = matched["close_odds"].apply(american_to_decimal)
        matched["bol_clv"] = (matched["entry_dec"] / matched["close_dec"] - 1) * 100

        n = len(matched)
        mean_clv = matched["bol_clv"].mean()
        se = matched["bol_clv"].std() / n**0.5
        t = mean_clv / se
        print(f"\n2025 BetOnline Sharp CLV:")
        print(f"  n={n}  mean={mean_clv:+.3f}%  t={t:.2f}  %positive={(matched['bol_clv']>0).mean():.1%}")
        print(f"  Win rate (matched subset): {matched['won'].mean():.1%}")

        # Also check DK CLV on same subset using existing DK close
        dk_patch = pd.read_csv("data/odds/hist_2025_dk_close_patch.csv", low_memory=False) if Path("data/odds/hist_2025_dk_close_patch.csv").exists() else pd.DataFrame()
        if len(dk_patch) > 0:
            print(f"\nDK close patch rows: {len(dk_patch)}, cols: {list(dk_patch.columns[:10])}")

        # Per-side breakdown
        print(f"\n  By side:")
        for side in ["over","under"]:
            sub = matched[matched["best_side"] == side]
            if len(sub) == 0: continue
            mc = sub["bol_clv"].mean()
            print(f"    {side}: n={len(sub)}, BOL CLV={mc:+.3f}%, win={sub['won'].mean():.1%}")

        # Edge band
        print(f"\n  By edge band:")
        for lo, hi in [(15,20),(20,25),(25,100)]:
            sub = matched[(matched["edge_pct"]>=lo) & (matched["edge_pct"]<hi)]
            if len(sub) < 5: continue
            mc = sub["bol_clv"].mean()
            lab = f"{lo}-{hi if hi<100 else ''}%".replace("-100%","+")
            print(f"    {lab}: n={len(sub)}, BOL CLV={mc:+.3f}%, win={sub['won'].mean():.1%}")

        # Month breakdown
        print(f"\n  By month:")
        matched["month"] = matched["game_date"].dt.to_period("M")
        for m, g in matched.groupby("month"):
            mc = g["bol_clv"].mean()
            print(f"    {m}: n={len(g)}, BOL CLV={mc:+.3f}%, win={g['won'].mean():.1%}")
