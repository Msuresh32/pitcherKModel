import pandas as pd

df = pd.read_csv("data/odds/historical_pitcher_props_plus_2026_6h.csv")
dk = df[(df["bookmaker"] == "draftkings") & 
        (df["game_date"] >= "2025-03-01") & 
        (df["game_date"] <= "2025-09-30")].copy()
# Add snapshot_type so backtest treats it as open
dk["snapshot_type"] = "open"
dk.to_csv("data/odds/hist_2025_dk_open.csv", index=False)
print(f"DK 2025 rows: {len(dk)}")
print(f"Date range: {dk['game_date'].min()} - {dk['game_date'].max()}")
