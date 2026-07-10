"""Run the full research pipeline in order.

Usage:
  python research/run_pipeline.py [--smoke-test]

Steps:
  1. 01_build_dataset.py  — joins features + odds → research/dataset.parquet
  2. 02_model_search.py   — walk-forward model search on train+val → best_config.json
  3. 03_oos_eval.py       — OOS evaluation (Jun 1 - Jul 7 2026)
  4. 04_strategy.py       — threshold analysis + strategy report

--smoke-test: runs step 1 only with whatever odds data is currently available.
"""
from __future__ import annotations
import sys, subprocess, argparse
from pathlib import Path

STEPS = [
    ("01_build_dataset.py",  "Phase 1+2: Build dataset"),
    ("02_model_search.py",   "Phase 3: Model search + validation"),
    ("03_oos_eval.py",       "Phase 4: OOS evaluation"),
    ("04_strategy.py",       "Phase 5+6: Strategy analysis"),
]

RESEARCH_DIR = Path(__file__).parent


def run_step(script: str, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"RUNNING: {label}")
    print(f"Script:  {script}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(RESEARCH_DIR / script)],
        cwd=str(RESEARCH_DIR.parent),
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only step 1 to test pipeline with available data")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from this step number (1-4)")
    args = parser.parse_args()

    steps = STEPS if not args.smoke_test else STEPS[:1]
    if args.from_step > 1:
        steps = steps[args.from_step - 1:]

    print(f"Running {len(steps)} pipeline steps...")
    for i, (script, label) in enumerate(steps, 1):
        ok = run_step(script, label)
        if not ok:
            print(f"\nPipeline aborted at step {i}.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print("Outputs:")
    for path in [
        "research/dataset.parquet",
        "research/model_results/val_summary.json",
        "research/model_results/best_config.json",
        "research/oos_results/oos_report.txt",
        "research/strategy/strategy_report.txt",
    ]:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists} {path}")


if __name__ == "__main__":
    main()
