"""
Rolling recalibration: run the model on recent game log data and update calibration.json.

Usage:
    python scripts/recalibrate.py                    # last 45 days (default)
    python scripts/recalibrate.py --window-days 30   # last 30 days
    python scripts/recalibrate.py --start 2026-05-01 --end 2026-06-16

Run this weekly (or whenever June-style drift is suspected) so the bias correction
reflects the current league environment rather than a frozen backtest period.
"""
import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.models.calibration import load_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=45,
                        help="Number of recent days to use (default 45)")
    parser.add_argument("--start", default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Override end date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    end_date = args.end or (date.today() - timedelta(days=1)).isoformat()
    if args.start:
        start_date = args.start
    else:
        end_dt = date.fromisoformat(end_date)
        start_date = (end_dt - timedelta(days=args.window_days)).isoformat()

    print(f"Recalibrating on window: {start_date} to {end_date}")

    cmd = [
        sys.executable, "scripts/backtest.py",
        "--config", args.config,
        "--start", start_date,
        "--end", end_date,
        "--save-calibration",
        "--output-prefix", "recal_window",
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Backtest failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(1)

    # Print new calibration summary
    config = load_config(args.config)
    cal_path = Path(config["data"]["processed_dir"]) / "calibration.json"
    cal = load_calibration(cal_path)
    if cal:
        print("\nUpdated calibration:")
        for market, vals in cal.get("markets", {}).items():
            print(f"  {market}: bias={vals['bias']:+.3f}, rmse={vals['rmse']:.3f} (n={vals['rows']})")


if __name__ == "__main__":
    main()
