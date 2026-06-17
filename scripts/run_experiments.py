"""
Systematic experiment runner.

Tests config/blend combinations without retraining (blend override is injected
at inference time via --blend flag added to backtest.py).

Results saved to data/processed/experiment_results.csv
"""
import subprocess
import sys
import re
import csv
from pathlib import Path

RESULTS_DIR = Path("data/processed")


CLOSING_ODDS_FILE = "data/odds/june_2026_odds.csv"
PROCESSED_DIR = Path("data/processed")


def run_backtest(start, end, model_dir=None, blend=None,
                 output_prefix="exp_tmp", save_calibration=False,
                 predictions_file=None, min_edge=None, edge_shrink=None):
    cmd = [
        sys.executable, "scripts/backtest.py",
        "--start", start, "--end", end,
        "--output-prefix", output_prefix,
    ]
    if model_dir:
        cmd += ["--model-dir", model_dir]
    if blend:
        cmd += ["--blend", blend]
    if save_calibration:
        cmd += ["--save-calibration"]
    if predictions_file:
        cmd += ["--predictions-file", predictions_file]
    if min_edge is not None:
        cmd += ["--min-edge", str(min_edge)]
    if edge_shrink is not None:
        cmd += ["--edge-shrink", str(edge_shrink)]
    # Always pass closing odds file for CLV computation
    closing_path = Path(CLOSING_ODDS_FILE)
    if closing_path.exists():
        cmd += ["--closing-odds", str(closing_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.stdout + result.stderr


def parse_results(output: str) -> dict:
    metrics = {"bets": None, "win_rate": None, "roi": None, "sharpe": None, "clv": None}

    m = re.search(r"^\s*(\d+)\s+([\d.]+)\s+[\d.]+\s+[-\d.]+\s+([-\d.]+)\s+", output, re.MULTILINE)
    if m:
        metrics["bets"]     = int(m.group(1))
        metrics["win_rate"] = float(m.group(2))
        metrics["roi"]      = float(m.group(3))

    m2 = re.search(r"Sharpe:\s*([-\d.]+)", output)
    if m2:
        metrics["sharpe"] = float(m2.group(1))

    m3 = re.search(r"ROI:\s*([-\d.]+)%", output)
    if m3:
        metrics["roi"] = float(m3.group(1)) / 100.0

    m4 = re.search(r"Mean CLV:\s*([-+\d.]+)%", output)
    if m4:
        metrics["clv"] = float(m4.group(1))

    return metrics


# -----------------------------------------------------------------------
# Experiment grid
# -----------------------------------------------------------------------
# Each tuple: (name, blend_str, min_edge, edge_shrink)
# blend_str = "glm,xgb"  e.g. "0.7,0.3"
#
# We also vary ensemble blend between the stored (original) model's
# two sub-pipelines at inference time — no retraining needed.

BASE_EXPERIMENTS = [
    # blend variations at default shrink/edge
    ("blend_100_0_s07_e10",  "1.0,0.0",  10.0, 0.7),
    ("blend_90_10_s07_e10",  "0.9,0.1",  10.0, 0.7),
    ("blend_80_20_s07_e10",  "0.8,0.2",  10.0, 0.7),
    ("blend_70_30_s07_e10",  "0.7,0.3",  10.0, 0.7),   # current best
    ("blend_60_40_s07_e10",  "0.6,0.4",  10.0, 0.7),
    ("blend_50_50_s07_e10",  "0.5,0.5",  10.0, 0.7),
    # shrink variations at best blend (70/30)
    ("blend_70_30_s05_e10",  "0.7,0.3",  10.0, 0.5),
    ("blend_70_30_s06_e10",  "0.7,0.3",  10.0, 0.6),
    ("blend_70_30_s08_e10",  "0.7,0.3",  10.0, 0.8),
    # edge variations at best blend + shrink
    ("blend_70_30_s07_e08",  "0.7,0.3",   8.0, 0.7),
    ("blend_70_30_s07_e12",  "0.7,0.3",  12.0, 0.7),
    ("blend_70_30_s07_e15",  "0.7,0.3",  15.0, 0.7),
    # conservative high-conviction combos
    ("blend_80_20_s06_e12",  "0.8,0.2",  12.0, 0.6),
    ("blend_90_10_s06_e12",  "0.9,0.1",  12.0, 0.6),
    ("blend_80_20_s07_e12",  "0.8,0.2",  12.0, 0.7),
    ("blend_70_30_s06_e12",  "0.7,0.3",  12.0, 0.6),
    ("blend_100_0_s06_e12",  "1.0,0.0",  12.0, 0.6),
    ("blend_90_10_s07_e15",  "0.9,0.1",  15.0, 0.7),
]


def _blend_slug(blend_str):
    """Turn '0.7,0.3' into 'b70_30' for use in filenames."""
    parts = [str(int(float(x) * 100)) for x in blend_str.split(",")]
    return "b" + "_".join(parts)


def _run_slug(eval_start, eval_end, model_dir):
    """Build a short slug that uniquely identifies this run's model+date scope."""
    s = eval_start.replace("-", "")
    e = eval_end.replace("-", "")
    m = "bk" if (model_dir and "backup" in str(model_dir)) else "new"
    return f"{m}_{s}_{e}"


def main(eval_start="2026-01-01", eval_end="2026-06-16", model_dir=None):
    # Pre-generate predictions for each unique blend (slow, feature-engineering pass)
    # Subsequent experiments with the same blend reuse the saved predictions CSV.
    # Prediction file names are scoped by model+date so concurrent runs don't collide.
    unique_blends = list(dict.fromkeys(b for _, b, _, _ in BASE_EXPERIMENTS))
    blend_pred_files = {}
    run_id = _run_slug(eval_start, eval_end, model_dir)

    results = []
    print("=" * 70)
    print(f"EXPERIMENT GRID  ({eval_start} to {eval_end})")
    if model_dir:
        print(f"  Model dir: {model_dir}")
    print(f"  Run ID: {run_id}")
    print("=" * 70)

    # Step 1: generate predictions for each unique blend
    print(f"\n--- Generating predictions for {len(unique_blends)} unique blend(s) ---")
    for blend_str in unique_blends:
        slug = _blend_slug(blend_str)
        pred_prefix = f"exp_pred_{run_id}_{slug}"
        pred_file = PROCESSED_DIR / f"{pred_prefix}_predictions.csv"

        print(f"\n[predictions] blend={blend_str}  -> {pred_file}")
        out = run_backtest(
            start=eval_start,
            end=eval_end,
            model_dir=model_dir,
            blend=blend_str,
            output_prefix=pred_prefix,
        )
        if not pred_file.exists():
            print(f"  WARNING: predictions file not found after run. Output:\n{out[-1000:]}")
        else:
            blend_pred_files[blend_str] = str(pred_file)
            print(f"  OK ({pred_file.stat().st_size // 1024} KB)")

    # Step 2: run all experiments using the pre-computed predictions
    # Betting params are passed as CLI args — config.yaml is never modified here.
    print(f"\n--- Running {len(BASE_EXPERIMENTS)} betting-param experiments ---")
    for name, blend_str, min_edge, edge_shrink in BASE_EXPERIMENTS:
        print(f"\n>>> {name}   blend={blend_str}  edge>={min_edge}%  shrink={edge_shrink}")

        pred_file = blend_pred_files.get(blend_str)
        output = run_backtest(
            start=eval_start,
            end=eval_end,
            model_dir=model_dir,
            blend=blend_str,
            output_prefix="exp_tmp",
            predictions_file=pred_file,
            min_edge=min_edge,
            edge_shrink=edge_shrink,
        )
        metrics = parse_results(output)

        roi_str  = f"{metrics['roi']:.1%}"  if metrics['roi']  is not None else "N/A"
        clv_str  = f"{metrics['clv']:+.2f}%" if metrics['clv'] is not None else "N/A"
        sh_str   = f"{metrics['sharpe']:.2f}"  if metrics['sharpe'] is not None else "N/A"
        wr_str   = f"{metrics['win_rate']:.1%}" if metrics['win_rate'] is not None else "N/A"

        print(f"    => bets={metrics['bets']}  win%={wr_str}  ROI={roi_str}  Sharpe={sh_str}  CLV={clv_str}")

        results.append({
            "experiment":   name,
            "blend":        blend_str,
            "min_edge_pct": min_edge,
            "edge_shrink":  edge_shrink,
            "bets":         metrics["bets"],
            "win_rate":     metrics["win_rate"],
            "roi":          metrics["roi"],
            "sharpe":       metrics["sharpe"],
            "clv":          metrics["clv"],
        })

    # Save results (run-scoped so multiple runs don't overwrite each other)
    results_path = RESULTS_DIR / f"experiment_results_{run_id}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results:
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {results_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY  (eval: {eval_start} to {eval_end})")
    print("=" * 80)
    print(f"{'Experiment':<28} {'Blend':>9} {'Edge':>5} {'Shrk':>5} {'Bets':>5} {'Win%':>6} {'ROI':>8} {'Sharpe':>7} {'CLV':>7}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (x["roi"] or -999), reverse=True):
        print(
            f"{r['experiment']:<28} {r['blend']:>9} {r['min_edge_pct']:>5.0f} {r['edge_shrink']:>5.1f} "
            f"{r['bets'] or 0:>5} {(r['win_rate'] or 0):>6.1%} {(r['roi'] or 0):>8.1%} "
            f"{(r['sharpe'] or 0):>7.2f} {(r['clv'] or 0):>+7.2f}%"
        )

    if results:
        valid = [r for r in results if r["roi"] is not None]
        if valid:
            best_roi = max(valid, key=lambda x: x["roi"])
            print(f"\nBest ROI:  {best_roi['experiment']}  ({best_roi['roi']:.1%})")
            valid_clv = [r for r in valid if r["clv"] is not None]
            if valid_clv:
                best_clv = max(valid_clv, key=lambda x: x["clv"])
                print(f"Best CLV:  {best_clv['experiment']}  ({best_clv['clv']:+.2f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",     default="2026-01-01")
    p.add_argument("--end",       default="2026-06-16")
    p.add_argument("--model-dir", default=None)
    a = p.parse_args()
    main(eval_start=a.start, eval_end=a.end, model_dir=a.model_dir)
