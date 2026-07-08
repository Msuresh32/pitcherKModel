"""Analyze 2026 new-model performance from bt_2026_new_model_edges.csv."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.config import load_config
from scripts.generate_backtest_excel import (
    compute_pricing, apply_filters, finalize, metrics_for, MONTH_NAME
)

config = load_config("config/config_v4_production.yaml")

# Load 2026 new-model edges
edges = pd.read_csv(
    "data/processed_poisson_wf2025/bt_2026_new_model_edges.csv",
    low_memory=False
)
edges["game_date"] = pd.to_datetime(edges["game_date"]).dt.strftime("%Y-%m-%d")
edges["pitcher_id"] = edges["pitcher_id"].astype(str)
edges["year"]  = 2026
edges["month"] = pd.to_datetime(edges["game_date"]).dt.month

# Filter to strikeouts market only, dedup to one row per pitcher-date (best odds)
edges = edges[edges["market"] == "strikeouts"].copy()
edges["_maxodds"] = edges[["over_odds","under_odds"]].max(axis=1)
edges = (edges.sort_values("_maxodds", ascending=False)
              .drop_duplicates(subset=["game_date","pitcher_id"])
              .drop(columns=["_maxodds"])
              .reset_index(drop=True))
print(f"2026 deduped rows: {len(edges)}")

# Rename strikeouts_projection if needed (edges file has it already)
if "strikeouts_projection" not in edges.columns:
    print("ERROR: no strikeouts_projection column found")
    sys.exit(1)

# Apply V4 filters directly (edges file already has edge_gap_product + over/under_probability)
min_egp  = float(config["betting"].get("min_edge_gap_product", 6.0))
min_prob = float(config["betting"].get("min_model_prob", 0.0))

unfiltered = edges[
    edges["edge_gap_product"].notna() & (edges["edge_gap_product"] >= min_egp) &
    (
        ((edges["best_side"] == "over")  & (edges["strikeouts_projection"] > edges["line"])) |
        ((edges["best_side"] == "under") & (edges["strikeouts_projection"] < edges["line"]))
    )
].copy()

filtered = unfiltered.copy()
if min_prob > 0:
    prob = filtered.apply(
        lambda r: r.get("over_probability", 0) if r.get("best_side") == "over"
                  else r.get("under_probability", 0), axis=1
    )
    filtered = filtered[pd.to_numeric(prob, errors="coerce").fillna(0) >= min_prob].copy()

unfiltered = finalize(unfiltered)
filtered   = finalize(filtered)

print(f"Unfiltered (EGP>={min_egp}): {len(unfiltered)}")
print(f"Filtered (prob>={min_prob:.0%}): {len(filtered)}")

stake = float(config["betting"].get("flat_stake", 100))

# --- Headline ---
print()
print("=" * 65)
print("  2026 NEW MODEL PERFORMANCE  ($100/bet flat stake)")
print("=" * 65)
for label, sub in [
    ("ALL BETS (EGP>=6, no floor)",      unfiltered),
    ("FILTERED (EGP>=6 + prob>=65%)",    filtered),
    ("  Mar-Jun (filtered)",             filtered[filtered["month"].isin([3,4,5,6])]),
    ("  April (filtered)",               filtered[filtered["month"]==4]),
    ("  May   (filtered)",               filtered[filtered["month"]==5]),
    ("  June  (filtered)",               filtered[filtered["month"]==6]),
    ("  V2_core EGP>=12 (filtered)",     filtered[filtered["edge_gap_product"]>=12]),
    ("  Overs (filtered)",               filtered[filtered["best_side"]=="over"]),
    ("  Unders (filtered)",              filtered[filtered["best_side"]=="under"]),
]:
    m = metrics_for(sub, stake)
    wr  = f"{m['WR']:.1%}" if not np.isnan(m["WR"]) else "  -  "
    roi = f"{m['ROI']:+.1%}" if not np.isnan(m["ROI"]) else "  -  "
    shp = f"  Sharpe={m['sharpe']:.2f}" if not np.isnan(m.get("sharpe", np.nan)) else ""
    ci  = f"  [{m['WR_lo']:.1%}, {m['WR_hi']:.1%}]" if not np.isnan(m.get("WR_lo", np.nan)) else ""
    print(f"  {label:<40} n={m['n']:4d}  WR={wr}{ci}  ROI={roi}{shp}")

# --- Odds bracket (is the same heavy-juice pattern there?) ---
res = filtered[filtered["win"].notna()].copy()
if len(res) > 0:
    print()
    print("Odds brackets (filtered 2026 bets):")
    for lo, hi, label in [
        (-999, -150, "Heavier than -150"),
        (-150, -130, "-150 to -131"),
        (-130, -115, "-130 to -116"),
        (-115,  300, "-115 and better"),
    ]:
        sub = res[(res["book_odds"] > lo) & (res["book_odds"] <= hi)]
        if len(sub) == 0:
            continue
        wr  = sub["win"].mean()
        roi = sub["profit"].sum() / (len(sub) * stake)
        avg_o = sub["book_odds"].mean()
        beven = abs(avg_o) / (abs(avg_o) + 100) if avg_o < 0 else 100 / (avg_o + 100)
        print(f"  {label}: n={len(sub):3d}  WR={wr:.1%}  ROI={roi:+.1%}  avg_odds={avg_o:.0f}  breakeven={beven:.1%}")

# --- Monthly ---
print()
print("By month (filtered 2026):")
for mo, grp in filtered.groupby("month"):
    m = metrics_for(grp, stake)
    wr  = f"{m['WR']:.1%}" if not np.isnan(m["WR"]) else "  -  "
    roi = f"{m['ROI']:+.1%}" if not np.isnan(m["ROI"]) else "  -  "
    print(f"  {MONTH_NAME.get(mo, str(mo))}: n={m['n']:4d}  WR={wr}  ROI={roi}")

# --- Model accuracy ---
preds = pd.read_csv(
    "data/processed_poisson_wf2025/bt_2026_new_model_predictions.csv",
    low_memory=False,
    usecols=["game_date","strikeouts","strikeouts_projection"]
)
preds = preds.dropna(subset=["strikeouts","strikeouts_projection"])
err = preds["strikeouts_projection"] - preds["strikeouts"]
print()
print(f"2026 model accuracy: RMSE={np.sqrt((err**2).mean()):.4f}  MAE={err.abs().mean():.4f}  "
      f"Bias={err.mean():.4f}  n={len(preds)}")
print("(2025 model RMSE was 2.2932 — improvement={:.1%})".format(
    (2.2932 - np.sqrt((err**2).mean())) / 2.2932
))
