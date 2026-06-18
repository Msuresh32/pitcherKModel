import pandas as pd

j1 = pd.read_csv("data/odds/jan_may_2026_odds.csv", nrows=3)
j2 = pd.read_csv("data/odds/june_2026_odds.csv", nrows=3)
print("jan_may columns:", list(j1.columns))
print("june columns:", list(j2.columns))
print("Columns match:", list(j1.columns) == list(j2.columns))

jm = pd.read_csv("data/odds/jan_may_2026_odds.csv")
jn = pd.read_csv("data/odds/june_2026_odds.csv")
print("jan_may rows:", len(jm))
print("june rows:", len(jn))

combined = pd.concat([jm, jn], ignore_index=True)
combined.to_csv("data/odds/full_2026_odds.csv", index=False)
print("Combined rows:", len(combined))
print("Date range:", combined["game_date"].min(), "-", combined["game_date"].max())
print("Saved -> data/odds/full_2026_odds.csv")
