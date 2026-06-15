from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.data.schema import REQUIRED_PARK_FACTOR_COLUMNS

BASE_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"


def _fetch_year(year: int, rolling: int = 3) -> pd.DataFrame:
    params = {
        "type": "year",
        "year": year,
        "batSide": "",
        "stat": "index_wOBA",
        "condition": "All",
        "rolling": rolling,
        "parks": "mlb",
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(request, timeout=30).read().decode("utf-8", "ignore")
    match = re.search(r"var data = (.*?);\s*var queryString", html, re.S)
    if not match:
        raise ValueError("Could not find Baseball Savant park-factor data payload.")
    data = json.loads(match.group(1))
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_PARK_FACTOR_COLUMNS)

    out = pd.DataFrame(
        {
            "factor_year": year,
            "venue_id": df["venue_id"],
            "venue_name": df["venue_name"],
            "park_runs_factor": df["index_runs"],
            "park_hits_factor": df["index_hits"],
            "park_bb_factor": df["index_bb"],
            "park_so_factor": df["index_so"],
            "park_hr_factor": df["index_hr"],
            "park_1b_factor": df["index_1b"],
            "park_2b_factor": df["index_2b"],
            "park_3b_factor": df["index_3b"],
        }
    )
    for col in REQUIRED_PARK_FACTOR_COLUMNS:
        if col not in out:
            out[col] = pd.NA
    numeric_cols = [col for col in out.columns if col.startswith("park_")] + ["venue_id"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[REQUIRED_PARK_FACTOR_COLUMNS]


def fetch_park_factors(start_year: int, end_year: int, rolling: int = 3) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching Baseball Savant park factors for {year}")
        frames.append(_fetch_year(year, rolling=rolling))
    return pd.concat(frames, ignore_index=True, sort=False)


def save_park_factors(start_year: int, end_year: int, output: str | Path, rolling: int = 3) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_park_factors(start_year, end_year, rolling=rolling)
    df.to_csv(output, index=False)
    return output
