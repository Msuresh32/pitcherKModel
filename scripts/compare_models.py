"""
Compare ensemble vs Poisson-only model projections for a given date.

Usage:
    python scripts/compare_models.py --date 2026-06-21
"""
import argparse
import sys
from pathlib import Path
from datetime import date

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    d = args.date
    exports = Path("data/exports")

    ensemble_path = exports / f"daily_pitcher_props_{d}.csv"
    poisson_path  = exports / f"daily_pitcher_props_{d}_poisson.csv"

    if not ensemble_path.exists():
        print(f"Ensemble output not found: {ensemble_path}")
        return
    if not poisson_path.exists():
        print(f"Poisson output not found: {poisson_path}")
        return

    ens = pd.read_csv(ensemble_path)
    poi = pd.read_csv(poisson_path)

    # Filter to strikeouts market only
    ens_k = ens[ens["market"] == "strikeouts"].copy()
    poi_k = poi[poi["market"] == "strikeouts"].copy()

    # Merge on pitcher_name
    merged = ens_k[["pitcher_name", "best_side", "line", "projection", "edge_pct", "recommended_odds"]].merge(
        poi_k[["pitcher_name", "best_side", "line", "projection", "edge_pct", "recommended_odds"]],
        on=["pitcher_name", "line"],
        suffixes=("_ens", "_poi"),
        how="outer",
    )

    merged["proj_diff"] = (merged["projection_poi"] - merged["projection_ens"]).round(2)
    merged["edge_diff"] = (merged["edge_pct_poi"] - merged["edge_pct_ens"]).round(2)

    # Agreement: both models pick the same side with edge > 12%
    min_edge = 12.0
    merged["ens_pick"]  = merged["best_side_ens"].where(merged["edge_pct_ens"] >= min_edge)
    merged["poi_pick"]  = merged["best_side_poi"].where(merged["edge_pct_poi"] >= min_edge)
    merged["agreement"] = merged["ens_pick"] == merged["poi_pick"]

    cols = ["pitcher_name", "line",
            "projection_ens", "projection_poi", "proj_diff",
            "edge_pct_ens", "edge_pct_poi", "edge_diff",
            "ens_pick", "poi_pick", "agreement"]
    merged = merged[[c for c in cols if c in merged.columns]].sort_values("edge_pct_poi", ascending=False)

    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON — {d}")
    print(f"{'='*80}")
    print(merged.to_string(index=False))

    # Summary
    ens_picks  = merged[merged["edge_pct_ens"] >= min_edge]
    poi_picks  = merged[merged["edge_pct_poi"] >= min_edge]
    both_picks = merged[merged["agreement"] == True]
    print(f"\n--- Summary ---")
    print(f"Ensemble picks (edge >= {min_edge}%):   {len(ens_picks)}")
    print(f"Poisson picks  (edge >= {min_edge}%):   {len(poi_picks)}")
    print(f"Both agree on same side:               {len(both_picks)}")
    print(f"\nHigh-conviction (both agree, poi edge >= 15%):")
    high = both_picks[both_picks["edge_pct_poi"] >= 15.0]
    if not high.empty:
        print(high[["pitcher_name", "line", "poi_pick", "projection_poi", "edge_pct_poi"]].to_string(index=False))
    else:
        print("  None")


if __name__ == "__main__":
    main()
