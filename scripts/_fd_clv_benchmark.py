"""
Compute FanDuel close CLV on 2025 edge>=15% bets.
FD tracks Pinnacle closely on player props — this tests whether
the positive BOL CLV or negative Pinnacle CLV is the more accurate signal.
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

# Load 2025 multi-book close
odds25 = pd.read_csv("data/odds/historical_pitcher_props_2025.csv")

results = {}
for book in ["fanduel", "betonlineag", "draftkings", "betrivers", "betmgm"]:
    close = odds25[(odds25["bookmaker"] == book) & (odds25["snapshot_type"] == "close")].copy()
    if len(close) < 100:
        continue
    close["game_date"] = (pd.to_datetime(close["commence_time"], utc=True, errors="coerce")
                           .dt.tz_localize(None).dt.normalize())
    close["fetched_at"] = pd.to_datetime(close["fetched_at"], errors="coerce")
    agg = (close.sort_values("fetched_at")
                .groupby(["game_date","player_name","line"])
                .last()
                .reset_index()[["game_date","player_name","line","over_odds","under_odds"]])
    agg.columns = ["game_date","player_name","line","cls_over","cls_under"]

    merged = bets.merge(agg, left_on=["game_date","pitcher_name","line"],
                        right_on=["game_date","player_name","line"], how="left")
    matched = merged.dropna(subset=["cls_over"]).copy()

    if len(matched) < 50:
        results[book] = (len(matched), None, None, None)
        continue

    matched["close_odds"] = matched.apply(
        lambda r: r["cls_over"] if r["best_side"] == "over" else r["cls_under"], axis=1)
    matched["entry_dec"] = matched["entry_odds"].apply(american_to_decimal)
    matched["close_dec"] = matched["close_odds"].apply(american_to_decimal)
    matched["clv"] = (matched["entry_dec"] / matched["close_dec"] - 1) * 100

    n = len(matched)
    mean_clv = matched["clv"].mean()
    se = matched["clv"].std() / n**0.5
    t = mean_clv / se
    pct_pos = (matched["clv"] > 0).mean()
    results[book] = (n, mean_clv, t, pct_pos)

print(f"\n{'Book':<16} {'N':>5}  {'Mean CLV':>9}  {'t-stat':>7}  {'%Positive':>9}")
print("-" * 52)
for book, (n, clv, t, pp) in results.items():
    if clv is None:
        print(f"  {book:<14} {n:>5}  {'<50 matched':>9}")
    else:
        print(f"  {book:<14} {n:>5}  {clv:>+8.3f}%  {t:>7.2f}  {pp:>9.1%}")

# Cross-book correlation: does FD move with BOL or against it?
print(f"\nCross-book correlation (do books agree on direction of moves?)")
print("(Positive = books move together; negative = books diverge)")

bol_close = odds25[(odds25["bookmaker"] == "betonlineag") & (odds25["snapshot_type"] == "close")].copy()
fd_close  = odds25[(odds25["bookmaker"] == "fanduel")     & (odds25["snapshot_type"] == "close")].copy()
dk_close  = odds25[(odds25["bookmaker"] == "draftkings")  & (odds25["snapshot_type"] == "close")].copy()

for src_name, src_df in [("betonlineag", bol_close), ("fanduel", fd_close)]:
    src_df = src_df.copy()
    src_df["game_date"] = (pd.to_datetime(src_df["commence_time"], utc=True, errors="coerce")
                            .dt.tz_localize(None).dt.normalize())
    src_df["fetched_at"] = pd.to_datetime(src_df["fetched_at"], errors="coerce")
    src_agg = (src_df.sort_values("fetched_at")
                     .groupby(["game_date","player_name","line"])
                     .last()
                     .reset_index()[["game_date","player_name","line","over_odds","under_odds"]])
    src_agg.columns = ["game_date","player_name","line",f"{src_name[:3]}_over",f"{src_name[:3]}_under"]

    # merge dk close with this book
    dk_c = dk_close.copy()
    dk_c["game_date"] = (pd.to_datetime(dk_c["commence_time"], utc=True, errors="coerce")
                          .dt.tz_localize(None).dt.normalize())
    dk_c["fetched_at"] = pd.to_datetime(dk_c["fetched_at"], errors="coerce")
    dk_agg = (dk_c.sort_values("fetched_at")
                  .groupby(["game_date","player_name","line"])
                  .last()
                  .reset_index()[["game_date","player_name","line","over_odds","under_odds"]])
    dk_agg.columns = ["game_date","player_name","line","dk_over","dk_under"]

    both = dk_agg.merge(src_agg, on=["game_date","player_name","line"])
    both = both.dropna(subset=["dk_over",f"{src_name[:3]}_over"])
    both = both[(both["dk_over"].between(-500,500)) & (both[f"{src_name[:3]}_over"].between(-500,500))]
    if len(both) > 20:
        corr = both["dk_over"].corr(both[f"{src_name[:3]}_over"])
        mean_diff = (both["dk_over"] - both[f"{src_name[:3]}_over"]).mean()
        print(f"  DK vs {src_name:<12}: n={len(both)}, corr={corr:.3f}, mean_diff={mean_diff:+.2f} cents")
