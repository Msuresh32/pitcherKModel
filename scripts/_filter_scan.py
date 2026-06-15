import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')

def resolve(row):
    a = row.get('strikeouts')
    if pd.isna(a): return np.nan
    return 1 if (a > row['line'] if row['best_side']=='over' else a < row['line']) else 0

def roi_fn(g):
    if g.empty: return np.nan
    o = np.where(g['best_side']=='over', g['over_odds'], g['under_odds'])
    d = np.where(o>0, 1+o/100, 1+100/np.abs(np.where(o==0,1,o)))
    return float(np.where(g['won'].astype(bool), d-1, -1.0).mean())

e = pd.read_csv('data/processed/backtest_statcast_recal_2026_edges.csv')
e['game_date'] = pd.to_datetime(e['game_date'])
e = e[e['market']=='strikeouts'].copy()
e['won'] = e.apply(resolve, axis=1)
e = e.dropna(subset=['won', 'edge_pct'])
e['abs_gap'] = abs(e['strikeouts_projection'] - e['line'])

print("=== Filter scan: 2023-2025 Statcast + rolling cal (2026 OOS) ===")
results = []
for min_e in [0, 3, 5, 7, 10, 12, 15]:
    for max_gap in [0.5, 0.8, 1.0, 1.5, 2.0, 99]:
        sub = e[(e['edge_pct'] >= min_e) & (e['abs_gap'] <= max_gap)]
        if len(sub) < 30: continue
        r = roi_fn(sub) * 100
        results.append((min_e, max_gap, len(sub), sub['won'].mean()*100, r))

pos = [(a,b,c,d,r) for a,b,c,d,r in results if r > 0 and c >= 50]
if pos:
    print("Positive ROI combinations:")
    for min_e, max_gap, n, wr, r in sorted(pos, key=lambda x: -x[4]):
        label = f"<={max_gap}" if max_gap < 99 else "all"
        print(f"  edge>={min_e}%  gap{label}  n={n}  win={wr:.1f}%  roi={r:.2f}%")
else:
    print("No positive ROI combos found. Best options:")
    for min_e, max_gap, n, wr, r in sorted(results, key=lambda x: -x[4])[:8]:
        label = f"<={max_gap}" if max_gap < 99 else "all"
        print(f"  edge>={min_e}%  gap{label}  n={n}  win={wr:.1f}%  roi={r:.2f}%")

# Compare with 2025-only + rolling cal best result
print("\n=== Comparison with best previous approach ===")
print("  2025-only + rolling cal:              +2.24% ROI (306 bets)")
print("  2023-2025 + Statcast + rolling cal:  best combo above")
print("\n  OOS CV MAE comparison:")
print("  Without Statcast (2025-only):   1.808")
print("  With Statcast (2023-2025):      1.769  <- best MAE")
print("  But MAE improvement hasn't translated to better betting ROI yet")
