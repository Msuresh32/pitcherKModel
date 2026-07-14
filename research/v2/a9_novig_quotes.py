"""Pull Novig (zero-vig exchange) pitcher-prop quotes for today and log them.

Purpose: validate the exchange-execution thesis with REAL exchange prices —
the hits model is +2.9% (2026 WF) at idealized no-vig pricing; this logs what
Novig actually offers so the paper track can settle at real fills.

Never breaks the pipeline: any failure just logs and exits 0.
Output: data/daily/novig_quotes_<date>.csv
"""
from __future__ import annotations
import os, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd

DATE = os.environ.get("SCORE_DATE", pd.Timestamp.now().strftime("%Y-%m-%d"))


def main():
    try:
        from src.odds.scrapers.novig import NovigScraper
        df = NovigScraper().fetch_pitcher_props(DATE)
        if df is None or df.empty:
            print(f"novig: no pitcher props for {DATE} (offseason/break or none posted)")
            return
        out = Path(f"data/daily/novig_quotes_{DATE}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        mk = df["market"].value_counts().to_dict() if "market" in df.columns else {}
        print(f"novig: saved {len(df)} rows -> {out} | markets: {mk}")
    except Exception as exc:
        print(f"novig: skipped ({type(exc).__name__}: {exc})")
        traceback.print_exc()


if __name__ == "__main__":
    main()
