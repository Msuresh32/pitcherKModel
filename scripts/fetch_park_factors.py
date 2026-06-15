import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories, load_config
from src.data.park_factors_source import save_park_factors


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Baseball Savant Statcast park factors.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--rolling", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    output = args.output or config["data"]["park_factors_file"]
    path = save_park_factors(args.start_year, args.end_year, output, rolling=args.rolling)
    print(f"Saved park factors to {path}")


if __name__ == "__main__":
    main()
