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
filt = finalize(filt)
resolved = filt[filt["win"].notna()].copy()
clv_vals = resolved["clv_pct"].dropna()
print("Rows with clv_pct:", len(clv_vals), "of", len(resolved))
print("avg_clv raw:", round(clv_vals.mean(), 4))
print("Sample clv_pct:", clv_vals.head(5).values)
ev = resolved["edge_vs_book"].dropna()
print("edge_vs_book avg:", round(ev.mean(), 4))
print("edge_vs_book sample:", ev.head(5).values)
