import pandas as pd
import numpy as np

# Load both datasets
backtest = pd.read_csv('data/exports/2026_backtest_extended.csv')
picks_log = pd.read_csv('data/exports/picks_log.csv')

# Parse dates
backtest['game_date'] = pd.to_datetime(backtest['game_date'])
picks_log['game_date'] = pd.to_datetime(picks_log['game_date'])

print('=' * 80)
print('PROJECTION ERROR & OVERFITTING')
print('=' * 80)

backtest['strikeouts_projection'] = pd.to_numeric(backtest['strikeouts_projection'], errors='coerce')
backtest['actual_ks'] = pd.to_numeric(backtest['actual_ks'], errors='coerce')
backtest_resolved = backtest[backtest['actual_ks'].notna()].copy()
backtest_resolved['error'] = abs(backtest_resolved['actual_ks'] - backtest_resolved['strikeouts_projection'])

picks_log['strikeouts_projection'] = pd.to_numeric(picks_log['strikeouts_projection'], errors='coerce')
picks_log['actual'] = pd.to_numeric(picks_log['actual'], errors='coerce')
oos_resolved = picks_log[picks_log['actual'].notna()].copy()
oos_resolved['error'] = abs(oos_resolved['actual'] - oos_resolved['strikeouts_projection'])

print(f'\nBacktest (in-sample, {len(backtest_resolved)} resolved):')
print(f'  Mean absolute error: {backtest_resolved["error"].mean():.2f} K')
print(f'  Median absolute error: {backtest_resolved["error"].median():.2f} K')
print(f'  Std dev of errors: {backtest_resolved["error"].std():.2f}')

print(f'\nOut-of-sample ({len(oos_resolved)} resolved):')
print(f'  Mean absolute error: {oos_resolved["error"].mean():.2f} K')
print(f'  Median absolute error: {oos_resolved["error"].median():.2f} K')
print(f'  Std dev of errors: {oos_resolved["error"].std():.2f}')

if len(backtest_resolved) > 0 and len(oos_resolved) > 0:
    print(f'\nError degradation (OOS - IS): +{(oos_resolved["error"].mean() - backtest_resolved["error"].mean()):.2f} K')

print('\n' + '=' * 80)
print('EDGE DISTRIBUTION & KELLY SIZING')
print('=' * 80)

backtest['edge_pct'] = pd.to_numeric(backtest['edge_pct'], errors='coerce')
picks_log['edge_pct'] = pd.to_numeric(picks_log['edge_pct'], errors='coerce')

print(f'\nBacktest edge distribution:')
print(f'  Mean: {backtest["edge_pct"].mean():.1f}%')
print(f'  Median: {backtest["edge_pct"].median():.1f}%')
print(f'  Min: {backtest["edge_pct"].min():.1f}%')
print(f'  Max: {backtest["edge_pct"].max():.1f}%')

print(f'\nOut-of-sample edge distribution:')
print(f'  Mean: {picks_log["edge_pct"].mean():.1f}%')
print(f'  Median: {picks_log["edge_pct"].median():.1f}%')

print('\n' + '=' * 80)
print('LINE SHOPPING & CLV ANALYSIS')
print('=' * 80)

picks_log['clv_pct'] = pd.to_numeric(picks_log['clv_pct'], errors='coerce')
clv_data = picks_log[picks_log['clv_pct'].notna()].copy()

print(f'\nCLV data available: {len(clv_data)} picks')
print(f'Mean CLV%: {clv_data["clv_pct"].mean():.2f}%')
print(f'Median CLV%: {clv_data["clv_pct"].median():.2f}%')
print(f'Min CLV%: {clv_data["clv_pct"].min():.2f}%')
print(f'Max CLV%: {clv_data["clv_pct"].max():.2f}%')

clv_pos = (clv_data['clv_pct'] > 0).sum()
print(f'\nPicks beating close: {clv_pos}/{len(clv_data)} ({clv_pos/len(clv_data)*100:.1f}%)')

pos_clv = clv_data[clv_data['clv_pct'] > 0]
neg_clv = clv_data[clv_data['clv_pct'] <= 0]

if len(pos_clv) > 0:
    pos_won = pos_clv[pos_clv['won'] == 1].shape[0]
    print(f'Positive CLV WR: {pos_won}/{len(pos_clv)} ({pos_won/len(pos_clv)*100:.1f}%)')
if len(neg_clv) > 0:
    neg_won = neg_clv[neg_clv['won'] == 1].shape[0]
    print(f'Negative CLV WR: {neg_won}/{len(neg_clv)} ({neg_won/len(neg_clv)*100:.1f}%)')

