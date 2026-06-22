import pandas as pd
import numpy as np

# Load both datasets
backtest = pd.read_csv('data/exports/2026_backtest_extended.csv')
picks_log = pd.read_csv('data/exports/picks_log.csv')

print('=' * 80)
print('BACKTEST DATA (In-Sample)')
print('=' * 80)
print(f'Shape: {backtest.shape}')
print(f'Date range: {backtest["game_date"].min()} to {backtest["game_date"].max()}')
print(f'Total picks: {len(backtest)}')
print(f'Wins: {int(backtest["won"].sum())}')
print(f'Win rate: {backtest["won"].mean():.2%}')

print('\n' + '=' * 80)
print('PICKS LOG DATA (Out-of-Sample)')
print('=' * 80)
print(f'Shape: {picks_log.shape}')
print(f'Date range: {picks_log["game_date"].min()} to {picks_log["game_date"].max()}')

# Count actual results
has_actual = (picks_log['actual'].notna()) & (picks_log['actual'] != '')
print(f'Resolved picks: {has_actual.sum()}')
if has_actual.sum() > 0:
    resolved = picks_log[has_actual].copy()
    resolved['actual'] = pd.to_numeric(resolved['actual'], errors='coerce')
    resolved['won'] = pd.to_numeric(resolved['won'], errors='coerce')
    print(f'Win rate (resolved): {resolved["won"].mean():.2%}')
    
    # CLV analysis
    print('\nCLV Analysis (Out-of-Sample):')
    clv_vals = pd.to_numeric(resolved['clv_pct'], errors='coerce')
    print(f'Mean CLV%: {clv_vals.mean():.2f}%')
    clv_positive = (clv_vals > 0).sum()
    clv_total = clv_vals.notna().sum()
    print(f'Picks beating close: {clv_positive}/{clv_total} ({clv_positive/clv_total*100:.1f}%)')
