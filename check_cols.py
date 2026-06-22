import pandas as pd

backtest = pd.read_csv('data/exports/2026_backtest_extended.csv')
picks_log = pd.read_csv('data/exports/picks_log.csv')

print('Backtest columns:')
print(list(backtest.columns))
print('\nPicks log columns:')
print(list(picks_log.columns))
