import pandas as pd
import numpy as np

# Load both datasets
backtest = pd.read_csv('data/exports/2026_backtest_extended.csv')
picks_log = pd.read_csv('data/exports/picks_log.csv')

# =============================================================================
# ANALYSIS 1: WEEK-BY-WEEK DEGRADATION
# =============================================================================
print('=' * 80)
print('1. WEEK-BY-WEEK PERFORMANCE DEGRADATION (Backtest)')
print('=' * 80)

backtest['week'] = pd.to_datetime(backtest['game_date']).dt.isocalendar().week
backtest = backtest.sort_values('game_date')

week_list = []
for week, group in backtest.groupby('week', sort=False):
    if len(group) < 3:
        continue
    wr = group['won'].mean()
    wins = int(group['won'].sum())
    n = len(group)
    week_list.append((week, wr))
    print(f'Week {week:2d}: {wins:2d}/{n:2d} ({wr:.1%})')

# =============================================================================
# ANALYSIS 2: EDGE BUCKET BREAKDOWN
# =============================================================================
print('\n' + '=' * 80)
print('2. EDGE BUCKET ANALYSIS (Backtest)')
print('=' * 80)

edge_15_20 = backtest[(backtest['edge_pct'] >= 15) & (backtest['edge_pct'] < 20)]
edge_20_30 = backtest[(backtest['edge_pct'] >= 20) & (backtest['edge_pct'] < 30)]
edge_30plus = backtest[backtest['edge_pct'] >= 30]

if len(edge_15_20) > 0:
    wr = edge_15_20['won'].mean()
    wins = int(edge_15_20['won'].sum())
    print(f'15-20% edge: {wins}/{len(edge_15_20)} ({wr:.1%}) | Expected ~15% | Diff: {(wr-0.15)*100:.1f}pp')

if len(edge_20_30) > 0:
    wr = edge_20_30['won'].mean()
    wins = int(edge_20_30['won'].sum())
    print(f'20-30% edge: {wins}/{len(edge_20_30)} ({wr:.1%}) | Expected ~25% | Diff: {(wr-0.25)*100:.1f}pp')

if len(edge_30plus) > 0:
    wr = edge_30plus['won'].mean()
    wins = int(edge_30plus['won'].sum())
    print(f'30%+ edge:  {wins}/{len(edge_30plus)} ({wr:.1%}) | Expected ~30%+ | Diff: {(wr-0.30)*100:.1f}pp')

# =============================================================================
# ANALYSIS 3: SIDE BREAKDOWN
# =============================================================================
print('\n' + '=' * 80)
print('3. SIDE BREAKDOWN (Backtest)')
print('=' * 80)

over = backtest[backtest['best_side'] == 'over']
under = backtest[backtest['best_side'] == 'under']

print(f'OVER:  {int(over["won"].sum())}/{len(over)} ({over["won"].mean():.1%})')
print(f'UNDER: {int(under["won"].sum())}/{len(under)} ({under["won"].mean():.1%})')

# =============================================================================
# ANALYSIS 4: LINE VALUE BREAKDOWN
# =============================================================================
print('\n' + '=' * 80)
print('4. LINE VALUE BREAKDOWN (Backtest)')
print('=' * 80)

lines = sorted(backtest['line'].dropna().unique())
for line in lines:
    subset = backtest[backtest['line'] == line]
    if len(subset) >= 3:
        wr = subset['won'].mean()
        wins = int(subset['won'].sum())
        print(f'Line {line:3.1f}: {wins:2d}/{len(subset):2d} ({wr:.1%})')

# =============================================================================
# ANALYSIS 5: OUT-OF-SAMPLE BREAKDOWN
# =============================================================================
print('\n' + '=' * 80)
print('5. OUT-OF-SAMPLE SIDE BREAKDOWN (Picks Log)')
print('=' * 80)

picks_log['actual'] = pd.to_numeric(picks_log['actual'], errors='coerce')
picks_log['won'] = pd.to_numeric(picks_log['won'], errors='coerce')

resolved = picks_log[picks_log['actual'].notna()].copy()

over_oos = resolved[resolved['best_side'] == 'over']
under_oos = resolved[resolved['best_side'] == 'under']

print(f'OVER:  {int(over_oos["won"].sum())}/{len(over_oos)} ({over_oos["won"].mean():.1%})')
print(f'UNDER: {int(under_oos["won"].sum())}/{len(under_oos)} ({under_oos["won"].mean():.1%})')

# =============================================================================
# ANALYSIS 6: CLV PERFORMANCE
# =============================================================================
print('\n' + '=' * 80)
print('6. COVERING THE CLOSING LINE (Out-of-Sample)')
print('=' * 80)

picks_log['clv_pct'] = pd.to_numeric(picks_log['clv_pct'], errors='coerce')
clv_data = picks_log[picks_log['clv_pct'].notna()].copy()

print(f'Picks with CLV data: {len(clv_data)}')
print(f'Mean CLV%: {clv_data["clv_pct"].mean():.2f}%')
clv_pos = (clv_data['clv_pct'] > 0).sum()
print(f'Beating closing line: {clv_pos}/{len(clv_data)} ({clv_pos/len(clv_data)*100:.1f}%)')

# Filter positive CLV
pos_clv = clv_data[clv_data['clv_pct'] > 0]
neg_clv = clv_data[clv_data['clv_pct'] <= 0]

if len(pos_clv) > 0:
    pos_clv['won'] = pd.to_numeric(pos_clv['won'], errors='coerce')
    print(f'\nPositive CLV picks: {int(pos_clv["won"].sum())}/{len(pos_clv)} ({pos_clv["won"].mean():.1%})')
if len(neg_clv) > 0:
    neg_clv['won'] = pd.to_numeric(neg_clv['won'], errors='coerce')
    print(f'Negative CLV picks: {int(neg_clv["won"].sum())}/{len(neg_clv)} ({neg_clv["won"].mean():.1%})')

# =============================================================================
# ANALYSIS 7: RECENCY DRIFT
# =============================================================================
print('\n' + '=' * 80)
print('7. RECENCY DRIFT ANALYSIS')
print('=' * 80)

backtest['game_date'] = pd.to_datetime(backtest['game_date'])
backtest_early = backtest[backtest['game_date'] < '2026-05-15']
backtest_late = backtest[backtest['game_date'] >= '2026-05-15']

print(f'Early (< May 15): {int(backtest_early["won"].sum())}/{len(backtest_early)} ({backtest_early["won"].mean():.1%})')
print(f'Late (>= May 15): {int(backtest_late["won"].sum())}/{len(backtest_late)} ({backtest_late["won"].mean():.1%})')

picks_log['game_date'] = pd.to_datetime(picks_log['game_date'])
picks_resolved = picks_log[picks_log['actual'].notna()].copy()

print(f'\nOut-of-sample (Jun 9-21): {int(picks_resolved["won"].sum())}/{len(picks_resolved)} ({picks_resolved["won"].mean():.1%})')

print(f'\nIN-SAMPLE TO OUT-OF-SAMPLE DECLINE: {(backtest["won"].mean() - picks_resolved["won"].mean())*100:.1f} percentage points')

