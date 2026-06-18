import pandas as pd, numpy as np

df = pd.read_csv("data/processed_2024/thresh_sel_2025_clv.csv")
df["won"] = df.apply(
    lambda r: (r["strikeouts"] > r["line"]) if r["best_side"]=="over"
              else (r["strikeouts"] < r["line"]), axis=1)
df["game_date"] = pd.to_datetime(df["game_date"])
df["month"] = df["game_date"].dt.to_period("M")

def stats(sub, label):
    valid = sub.dropna(subset=["clv_pct"])
    if len(valid) == 0:
        return f"  {label:<10}  no CLV data"
    n      = len(sub)
    n_clv  = len(valid)
    mean   = valid["clv_pct"].mean()
    beat   = (valid["clv_pct"] > 0).mean()
    win    = sub["won"].mean()
    se     = valid["clv_pct"].std() / n_clv**0.5
    t      = mean / se if se > 0 else 0
    return (f"  {label:<10}  n={n:>4}  matched={n_clv:>4}  "
            f"CLV={mean:>+6.2f}%  beat_close={beat:>5.1%}  win={win:>5.1%}  t={t:>6.2f}")

hdr = ("  Side        n    matched    CLV     beat_close   win    t-stat")
div = "-" * 70

print()
print("2025 CLV — OVER vs UNDER  (edge>=15%, DK close)")
print("=" * 70)
print(hdr); print(div)
for side in ["over", "under", "ALL"]:
    sub = df if side == "ALL" else df[df["best_side"] == side]
    print(stats(sub, side.upper()))

# ── monthly breakdown ──────────────────────────────────────────────────────
for side in ["over", "under"]:
    print()
    label = "OVER" if side == "over" else "UNDER"
    print(f"Monthly {label}  (bets / mean CLV / win rate):")
    sub = df[df["best_side"] == side]
    for m, g in sub.groupby("month"):
        v = g.dropna(subset=["clv_pct"])
        clv_str = f"{v['clv_pct'].mean():+.2f}% (n={len(v)})" if len(v) > 0 else "no match"
        print(f"  {m}  bets={len(g):>3}  CLV={clv_str:<18}  win={g['won'].mean():.1%}")

# ── edge bands by side ─────────────────────────────────────────────────────
print()
print("Edge bands by side:")
print(f"  {'Edge band':<14} {'Side':<8} {'Bets':>5}  {'Mean CLV':>8}  {'Win%':>6}")
print("-" * 52)
bins   = [(15,20), (20,25), (25,100)]
labels = ["15-20%", "20-25%", "25%+"]
for lo, hi in bins:
    for side in ["over", "under"]:
        sub = df[(df["edge_pct"] >= lo) & (df["edge_pct"] < hi) & (df["best_side"] == side)]
        v   = sub.dropna(subset=["clv_pct"])
        band = f"{lo}-{hi if hi<100 else ''}%".replace("-100%", "+")
        if len(sub) < 3:
            print(f"  {band:<14} {side:<8} {len(sub):>5}  {'<5 bets':>8}")
        else:
            clv_m = v["clv_pct"].mean() if len(v) > 0 else np.nan
            print(f"  {band:<14} {side:<8} {len(sub):>5}  {clv_m:>+7.2f}%  {sub['won'].mean():>6.1%}")
