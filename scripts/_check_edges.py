import pandas as pd

df = pd.read_csv("data/processed/full2026_b70_30_e10_edges.csv")
df = df[df["market"] == "strikeouts"].copy()
print("Shape:", df.shape)
print("Date range:", df["game_date"].min(), "-", df["game_date"].max())
print()

# Check for duplicates
dup_cols = ["game_date", "pitcher_name", "line", "best_side"]
dups = df[df.duplicated(subset=dup_cols, keep=False)]
print("Duplicate (date/pitcher/line/side) rows:", len(dups))

# Bookmaker distribution
if "bookmaker" in df.columns:
    print("\nBookmaker counts:")
    print(df["bookmaker"].value_counts().head(10))

# Edge distribution
print("\nEdge distribution:")
print(df["edge_pct"].describe())
