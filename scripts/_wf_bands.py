import pandas as pd, numpy as np

def load(pfx, d):
    df = pd.read_csv(f"{d}/{pfx}_edges.csv")
    df = df[df["market"] == "strikeouts"].copy()
    df["won"] = df.apply(
        lambda r: (r["strikeouts"] > r["line"]) if r["best_side"] == "over"
                  else (r["strikeouts"] < r["line"]), axis=1)
    df["pay"] = df.apply(
        lambda r: (r["over_odds"]/100 if r["over_odds"] > 0 else 100/abs(r["over_odds"]))
                  if r["best_side"] == "over"
                  else (r["under_odds"]/100 if r["under_odds"] > 0 else 100/abs(r["under_odds"])), axis=1)
    df["profit"] = df.apply(lambda r: r["pay"] if r["won"] else -1.0, axis=1)
    df["gap"]    = df["strikeouts_projection"] - df["line"]
    df = (df.sort_values("edge_pct", ascending=False)
            .drop_duplicates(subset=["game_date","pitcher_name","line","best_side"])
            .reset_index(drop=True))
    return df

# Load all three walk-forward periods (edge >= 0 so we can slice freely)
p1 = load("wf2026_p1_mar_apr", "data/processed");       p1["period"] = "Mar-Apr"
p2 = load("wf2026_p2_may",     "data/processed_apr2026"); p2["period"] = "May"
p3 = load("wf2026_p3_jun",     "data/processed");       p3["period"] = "Jun"
df = pd.concat([p1, p2, p3], ignore_index=True)

def show(label, sub):
    if len(sub) < 5:
        return f"{label:<35} {'<5 bets':>5}"
    sh = sub["profit"].mean()/sub["profit"].std()*len(sub)**0.5 if sub["profit"].std()>0 else 0
    return (f"{label:<35} {len(sub):>5}  {sub['won'].mean():>6.1%}  "
            f"{sub['profit'].mean():>+7.1%}  {sh:>7.2f}")

hdr = f"{'Filter':<35} {'Bets':>5}  {'Win%':>6}  {'ROI':>7}  {'Sharpe':>7}"
div = "-" * 65

# ── 1. EDGE THRESHOLD SWEEP ───────────────────────────────────────
print("\nWALK-FORWARD 2026 — EDGE THRESHOLD SWEEP")
print(hdr); print(div)
for t in [0, 5, 10, 12, 15, 18, 20, 25]:
    sub = df[df["edge_pct"] >= t]
    print(show(f"edge >= {t}%", sub))

# ── 2. GAP BANDS (all edge >= 0) ─────────────────────────────────
print("\nWALK-FORWARD 2026 — PROJECTION GAP BANDS  (edge >= 0)")
print(hdr); print(div)
gap_bins = [
    ("gap < 0  (under bet)",         df[df["gap"] < 0]),
    ("gap 0.0–0.5  (weak over)",     df[(df["gap"] >= 0)   & (df["gap"] < 0.5)]),
    ("gap 0.5–1.0",                  df[(df["gap"] >= 0.5) & (df["gap"] < 1.0)]),
    ("gap 1.0–1.5",                  df[(df["gap"] >= 1.0) & (df["gap"] < 1.5)]),
    ("gap 1.5+",                     df[df["gap"] >= 1.5]),
]
for label, sub in gap_bins:
    print(show(label, sub))

# ── 3. GAP BANDS at edge >= 15% ──────────────────────────────────
print("\nWALK-FORWARD 2026 — GAP BANDS  (edge >= 15%,  frozen threshold)")
print(hdr); print(div)
e15 = df[df["edge_pct"] >= 15]
for label, sub in [
    ("gap < 0  (under)", e15[e15["gap"] < 0]),
    ("gap 0.0–0.5",      e15[(e15["gap"] >= 0)   & (e15["gap"] < 0.5)]),
    ("gap 0.5–1.0",      e15[(e15["gap"] >= 0.5) & (e15["gap"] < 1.0)]),
    ("gap 1.0–1.5",      e15[(e15["gap"] >= 1.0) & (e15["gap"] < 1.5)]),
    ("gap 1.5+",         e15[e15["gap"] >= 1.5]),
]:
    print(show(label, sub))

# ── 4. COMBINED EDGE + GAP FILTERS ───────────────────────────────
print("\nWALK-FORWARD 2026 — COMBINED EDGE + GAP FILTERS")
print(hdr); print(div)
combos = [
    ("edge>=15%  (baseline)",                df[df["edge_pct"] >= 15]),
    ("edge>=15%  + gap>=0.5",                df[(df["edge_pct"] >= 15) & (df["gap"] >= 0.5)]),
    ("edge>=15%  + gap>=1.0",                df[(df["edge_pct"] >= 15) & (df["gap"] >= 1.0)]),
    ("edge>=15%  + gap<0  (unders only)",    df[(df["edge_pct"] >= 15) & (df["gap"] < 0)]),
    ("edge>=20%  (no gap filter)",           df[df["edge_pct"] >= 20]),
    ("edge>=20%  + gap>=0.5",                df[(df["edge_pct"] >= 20) & (df["gap"] >= 0.5)]),
    ("edge>=20%  + gap<0  (unders only)",    df[(df["edge_pct"] >= 20) & (df["gap"] < 0)]),
    ("edge 15-20%  only",                    df[(df["edge_pct"] >= 15) & (df["edge_pct"] < 20)]),
    ("edge>=15%  + skip gap 0–0.5",          df[(df["edge_pct"] >= 15) & ~((df["gap"] >= 0) & (df["gap"] < 0.5))]),
]
for label, sub in combos:
    print(show(label, sub))
