import pandas as pd
import numpy as np

clv_df = pd.read_csv("data/processed/full2026_dk_b70_30_e10_clv.csv")
clv_df = clv_df[clv_df["market"] == "strikeouts"].copy()

print(f"CLV file shape: {clv_df.shape}")
print(f"CLV column present: {'clv_pct' in clv_df.columns}")
print()

clv = clv_df.dropna(subset=["clv_pct"])
print(f"Bets with CLV data: {len(clv)} of {len(clv_df)}")
print(f"Mean CLV: {clv['clv_pct'].mean():+.2f}%")
print(f"CLV > 0: {(clv['clv_pct'] > 0).sum()}/{len(clv)} ({(clv['clv_pct'] > 0).mean():.1%})")
print()

clv["game_date"] = pd.to_datetime(clv["game_date"])
clv["month"] = clv["game_date"].dt.to_period("M")
print("CLV by month:")
print(f"{'Month':<10} {'Bets':>5} {'Mean CLV':>10} {'CLV>0':>8}")
print("-" * 38)
for m, g in clv.groupby("month"):
    print(f"{str(m):<10} {len(g):>5}  {g['clv_pct'].mean():>+8.2f}%  {(g['clv_pct']>0).mean():>7.1%}")
