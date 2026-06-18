import pandas as pd

# Original odds file
orig = pd.read_csv("data/odds/historical_pitcher_props_plus_2026_6h.csv", nrows=5000)
print("Original odds file (6h):")
print("  Bookmakers:", orig["bookmaker"].unique() if "bookmaker" in orig.columns else "no bookmaker col")
print("  Cols:", list(orig.columns[:10]))

# New combined odds file
new = pd.read_csv("data/odds/full_2026_odds.csv", nrows=5000)
print("\nNew full_2026_odds:")
print("  Bookmakers:", new["bookmaker"].unique())
print()

# Check line sizes in each
if "line" in orig.columns and "market" in orig.columns:
    orig_sk = orig[orig.get("market", pd.Series()).eq("strikeouts") if "market" in orig.columns else pd.Series([True]*len(orig))]
    print("Original - strikeout line values (sample):", sorted(orig["line"].dropna().unique()[:20]))
    
new_sk = new[new["market"] == "strikeouts"]
print("New - strikeout line values (sample):", sorted(new_sk["line"].dropna().unique()[:20]))
