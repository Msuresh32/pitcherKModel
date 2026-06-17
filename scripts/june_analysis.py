import pandas as pd
import numpy as np

picks = pd.read_csv('data/exports/picks_log.csv')
june = picks[picks['game_date'].str.startswith('2026-06')].copy()
june = june[june['game_date'] <= '2026-06-16']

settled = june[june['actual'].notna() & (june['actual'] != '') & (june['won'] != '')].copy()
settled['won'] = settled['won'].astype(float)
settled['odds_used'] = settled['odds_used'].astype(float)
settled['edge_pct'] = settled['edge_pct'].astype(float)
settled['actual'] = settled['actual'].astype(float)

# Remove obvious duplicates (same date/pitcher/side/line)
settled = settled.drop_duplicates(subset=['game_date','pitcher_name','best_side','line'])

def profit(row):
    if row['won'] == 1:
        o = row['odds_used']
        return (o/100) if o > 0 else (100/abs(o))
    return -1.0

settled['profit_units'] = settled.apply(profit, axis=1)

total = len(settled)
wins = int(settled['won'].sum())
total_profit = settled['profit_units'].sum()
roi = total_profit / total * 100

print("June 1-16 LIVE Results (settled, deduped)")
print("==========================================")
print(f"Bets: {total}  |  Wins: {wins}  |  Win rate: {wins/total:.1%}")
print(f"ROI: {roi:+.1f}%  |  Profit: {total_profit:+.2f} units")
print()

# Edge breakdown
print("By edge tier:")
tiers = [("10-15%", 10, 15), ("15-20%", 15, 20), ("20%+", 20, 999)]
for label, lo, hi in tiers:
    sub = settled[(settled['edge_pct'] >= lo) & (settled['edge_pct'] < hi)]
    if len(sub) == 0:
        continue
    w = int(sub['won'].sum())
    p = sub['profit_units'].sum()
    r = p/len(sub)*100
    print(f"  {label}: {len(sub)} bets  {w}W/{len(sub)-w}L  {w/len(sub):.1%} win rate  ROI {r:+.1f}%")

print()
print("Per-day breakdown:")
for date, grp in settled.groupby('game_date'):
    w = int(grp['won'].sum())
    l = len(grp) - w
    p = grp['profit_units'].sum()
    r = p/len(grp)*100
    print(f"  {date}: {len(grp)} bets  {w}W/{l}L  ROI {r:+.1f}%")

print()
print("Individual bets:")
cols = ['game_date','pitcher_name','best_side','line','edge_pct','odds_used','actual','won','profit_units']
print(settled[cols].to_string(index=False))
