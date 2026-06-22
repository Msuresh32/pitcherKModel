import pandas as pd
import numpy as np

# Load both datasets
backtest = pd.read_csv('data/exports/2026_backtest_extended.csv')
picks_log = pd.read_csv('data/exports/picks_log.csv')

# Parse dates
backtest['game_date'] = pd.to_datetime(backtest['game_date'])
picks_log['game_date'] = pd.to_datetime(picks_log['game_date'])

print('=' * 80)
print('SAMPLE SIZE & DAILY PICK FREQUENCY')
print('=' * 80)

# Daily picks
daily_bt = backtest.groupby('game_date').size()
print(f'\nBacktest daily picks:')
print(f'  Mean: {daily_bt.mean():.1f} picks/day')
print(f'  Min: {daily_bt.min()} picks/day')
print(f'  Max: {daily_bt.max()} picks/day')
print(f'  Std: {daily_bt.std():.1f}')

daily_oos = picks_log.groupby('game_date').size()
print(f'\nOut-of-sample daily picks:')
print(f'  Mean: {daily_oos.mean():.1f} picks/day')
print(f'  Min: {daily_oos.min()} picks/day')
print(f'  Max: {daily_oos.max()} picks/day')

print('\n' + '=' * 80)
print('PROJECTION ERROR & OVERFITTING')
print('=' * 80)

backtest['strikeouts_projection'] = pd.to_numeric(backtest['strikeouts_projection'], errors='coerce')
backtest['error'] = abs(backtest[backtest['strikeouts'].notna()]['strikeouts'] - backtest[backtest['strikeouts'].notna()]['strikeouts_projection'])

picks_log['strikeouts_projection'] = pd.to_numeric(picks_log['strikeouts_projection'], errors='coerce')
picks_log['actual'] = pd.to_numeric(picks_log['actual'], errors='coerce')
oos_resolved = picks_log[picks_log['actual'].notna()].copy()
oos_resolved['error'] = abs(oos_resolved['actual'] - oos_resolved['strikeouts_projection'])

print(f'\nBacktest (in-sample):')
print(f'  Mean absolute error: {backtest["error"].mean():.2f} K')
print(f'  Median absolute error: {backtest["error"].median():.2f} K')
print(f'  Std dev of errors: {backtest["error"].std():.2f}')

print(f'\nOut-of-sample:')
print(f'  Mean absolute error: {oos_resolved["error"].mean():.2f} K')
print(f'  Median absolute error: {oos_resolved["error"].median():.2f} K')
print(f'  Std dev of errors: {oos_resolved["error"].std():.2f}')

print(f'\nError degradation: +{(oos_resolved["error"].mean() - backtest["error"].mean()):.2f} K')

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
print(f'  Min: {picks_log["edge_pct"].min():.1f}%')
print(f'  Max: {picks_log["edge_pct"].max():.1f}%')

print('\n' + '=' * 80)
print('LINE SHOPPING & CLV DISTRIBUTION')
print('=' * 80)

picks_log['closing_odds'] = pd.to_numeric(picks_log['closing_odds'], errors='coerce')
picks_log['opening_odds'] = pd.to_numeric(picks_log['opening_odds'], errors='coerce')
picks_log['odds_used'] = pd.to_numeric(picks_log['odds_used'], errors='coerce')

with_opens = picks_log[picks_log['opening_odds'].notna() & picks_log['odds_used'].notna()].copy()
with_opens['line_moved'] = abs(with_opens['odds_used'] - with_opens['opening_odds'])

print(f'\nLine movement analysis (picks with opening odds):')
print(f'  Samples: {len(with_opens)}')
if len(with_opens) > 0:
    print(f'  Mean line move: {with_opens["line_moved"].mean():.0f} cents')
    got_better = (with_opens['odds_used'] > with_opens['opening_odds']).sum()
    print(f'  Got better odds: {got_better}/{len(with_opens)} ({got_better/len(with_opens)*100:.0f}%)')

print('\n' + '=' * 80)
print('RECAP: KEY FINDINGS')
print('=' * 80)

print(f'\n1. IN-SAMPLE WR: 50.64% (198/401 picks)')
print(f'2. OUT-OF-SAMPLE WR: 34.57% (28/81 resolved)')
print(f'3. DECLINE: 16.1 percentage points')
print(f'4. MEAN CLV%: -0.93% (beating close only 30% of the time)')
print(f'5. EDGE EXPECTATIONS NOT MET:')
print(f'   - 15-20% edge picks: 45.9% WR (expected ~15% edge value)')
print(f'   - 20-30% edge picks: 58.2% WR (expected ~25% edge value)')
print(f'   - Actual WR matching 0% implied edge, not the claimed edge')
print(f'6. RECENT (LATE MAY+): 46.6% WR in-sample, 34.6% out-of-sample')
print(f'7. DAILY PICKS: ~5-6 per day average (small sample per day)')

