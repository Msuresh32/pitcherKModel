import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.config import load_config
from scripts.generate_backtest_excel import (
    load_2025_universe, compute_pricing, apply_filters, finalize
)

config = load_config("config/config_v4_production.yaml")
u25 = load_2025_universe(config)
priced = compute_pricing(u25, config)
filt = apply_filters(priced, config)
filt = finalize(filt, 100)
res = filt[filt["win"].notna()].copy()

print("ROI by odds bracket (2025, prob>=65%):")
brackets = [
    (-201, -150, "-200 to -151"),
    (-150, -130, "-150 to -131"),
    (-130, -115, "-130 to -116"),
    (-115,  150, "-115 and better"),
]
for lo, hi, label in brackets:
    sub = res[(res["book_odds"] > lo) & (res["book_odds"] <= hi)]
    if len(sub) == 0:
        continue
    wr = sub["win"].mean()
    roi = sub["profit"].sum() / (len(sub) * 100)
    avg_odds = sub["book_odds"].mean()
    beven = abs(avg_odds) / (abs(avg_odds) + 100) if avg_odds < 0 else 100 / (avg_odds + 100)
    print(f"  {label}: n={len(sub):3d}  WR={wr:.1%}  ROI={roi:+.1%}  "
          f"avg_odds={avg_odds:.0f}  breakeven={beven:.1%}")

print()
for cut, label in [(-160, "odds >=-160"), (-140, "odds >=-140"), (-120, "odds >=-120")]:
    sub = res[res["book_odds"] >= cut]
    if len(sub) == 0:
        continue
    wr = sub["win"].mean()
    roi = sub["profit"].sum() / (len(sub) * 100)
    print(f"  Filter to {label}: n={len(sub):3d}  WR={wr:.1%}  ROI={roi:+.1%}")
