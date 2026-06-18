import pandas as pd

df = pd.read_csv("data/odds/full_2026_odds.csv")
dk = df[df["bookmaker"] == "draftkings"].copy()
print(f"DraftKings rows: {len(dk)}")
print(f"Date range: {dk['game_date'].min()} - {dk['game_date'].max()}")
print(f"Snapshot types: {dk['snapshot_type'].value_counts().to_dict()}")
dk.to_csv("data/odds/full_2026_odds_dk.csv", index=False)
print("Saved -> data/odds/full_2026_odds_dk.csv")
