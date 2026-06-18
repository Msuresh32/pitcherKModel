"""
Step 2 of proper walk-forward validation:
- Model trained on 2023-2024 only (config_2024only.yaml)
- Backtest on 2025 (truly OOS for this model)
- Pick the threshold that performs best on 2025
- That threshold is then frozen for the 2026 test
"""
import pandas as pd
import numpy as np
import subprocess, sys
from pathlib import Path

EDGES_2025 = "data/processed_2024/threshold_sel_2025_edges.csv"
ODDS_2025  = "data/odds/historical_pitcher_props_plus_2026_6h.csv"

def run_backtest_2025():
    """Run backtest on 2025 with 2024-only model. Uses DK lines from 6h file."""
    print("Running 2025 backtest with 2024-only model...")
    cmd = [
        sys.executable, "scripts/backtest.py",
        "--config", "config/config_2024only.yaml",
        "--start", "2025-03-25",
        "--end",   "2025-09-30",
        "--odds",  ODDS_2025,
        "--min-edge", "0",
        "--output-prefix", "threshold_sel_2025",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError("Backtest failed")
    print("2025 backtest complete.")


def find_best_threshold():
    print(f"\nLoading edges: {EDGES_2025}")
    df = pd.read_csv(EDGES_2025)
    df = df[df["market"] == "strikeouts"].copy()
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['game_date'].min()} - {df['game_date'].max()}")

    def bet_won(row):
        return (row["strikeouts"] > row["line"]) if row["best_side"] == "over" \
               else (row["strikeouts"] < row["line"])

    def payout(row):
        odds = row["over_odds"] if row["best_side"] == "over" else row["under_odds"]
        return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

    df["won"]    = df.apply(bet_won, axis=1)
    df["payout"] = df.apply(payout, axis=1)
    df["profit"] = df.apply(lambda r: r["payout"] if r["won"] else -1.0, axis=1)

    # Filter to DK only
    if "bookmaker" in df.columns:
        dk = df[df["bookmaker"] == "draftkings"].copy()
        print(f"DK-only rows: {len(dk)}")
    else:
        dk = df.copy()
        print("No bookmaker column — using all rows")

    # Deduplicate
    dk = (dk.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date", "pitcher_name", "line", "best_side"])
            .reset_index(drop=True))
    print(f"After dedup: {len(dk)} rows")

    print("\n" + "=" * 55)
    print("2025 THRESHOLD SWEEP (2024-only model, truly OOS)")
    print("=" * 55)
    print(f"{'MinEdge':>9} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Sharpe':>8}")
    print("-" * 45)

    results = {}
    for t in [0, 3, 5, 7, 10, 12, 15, 18, 20, 25]:
        sub = dk[dk["edge_pct"] >= t]
        if len(sub) < 30:
            print(f"{t:>8}%  {'<30 bets — skip':>35}")
            continue
        sharpe = (sub["profit"].mean() / sub["profit"].std() * len(sub)**0.5
                  if sub["profit"].std() > 0 else 0)
        results[t] = {
            "bets": len(sub), "win": sub["won"].mean(),
            "roi": sub["profit"].mean(), "sharpe": sharpe,
        }
        print(f"{t:>8}%  {len(sub):>6}  {sub['won'].mean():>6.1%}  "
              f"{sub['profit'].mean():>+7.1%}  {sharpe:>8.2f}")

    # Best threshold = highest Sharpe with >= 50 bets
    valid = {t: v for t, v in results.items() if v["bets"] >= 50}
    best_t = max(valid, key=lambda t: valid[t]["sharpe"])
    best   = valid[best_t]
    print(f"\n{'='*55}")
    print(f"SELECTED THRESHOLD (highest Sharpe, >= 50 bets): edge >= {best_t}%")
    print(f"  2025 OOS: {best['bets']} bets  {best['win']:.1%} win  "
          f"{best['roi']:+.1%} ROI  Sharpe {best['sharpe']:.2f}")
    print(f"{'='*55}")
    print(f"\nNext step: apply edge >= {best_t}% to the 2026 OOS backtest.")
    print(f"  python scripts/backtest.py --predictions-file data/processed/exp_pred_bk_20260101_20260616_b70_30_predictions.csv "
          f"--odds data/odds/full_2026_odds_dk.csv --start 2026-03-26 --end 2026-06-16 "
          f"--min-edge {best_t} --output-prefix frozen_thresh_2026")
    return best_t


if __name__ == "__main__":
    edges_path = Path(EDGES_2025)
    if not edges_path.exists():
        run_backtest_2025()
    else:
        print(f"Edges file already exists ({edges_path}), skipping backtest.")

    best_threshold = find_best_threshold()
