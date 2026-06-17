import pandas as pd

picks = pd.read_csv('data/exports/picks_log.csv')
june = picks[(picks['game_date'] >= '2026-06-09') & (picks['game_date'] <= '2026-06-16')].copy()
settled = june[june['actual'].notna() & (june['actual'] != '') & (june['won'] != '')].copy()
settled = settled.drop_duplicates(subset=['game_date','pitcher_name','best_side','line'])
settled['won'] = settled['won'].astype(float)
settled['edge_pct'] = settled['edge_pct'].astype(float)
settled['actual'] = settled['actual'].astype(float)
settled['strikeouts_projection'] = settled['strikeouts_projection'].astype(float)
clean = settled[(settled['edge_pct'] >= 10) & (settled['edge_pct'] <= 50)].copy()

print("Over vs Under:")
for side, grp in clean.groupby('best_side'):
    w = int(grp['won'].sum())
    print(f"  {side}: {len(grp)} bets  {w}W/{len(grp)-w}L  {w/len(grp):.1%} win rate")

clean['proj_error'] = clean['actual'] - clean['strikeouts_projection']
mean_err = clean['proj_error'].mean()
print(f"\nProjection accuracy (actual - projected): mean {mean_err:+.2f} Ks")
print("(Negative = model over-predicted strikeouts)")

over_losses = clean[(clean['best_side']=='over') & (clean['won']==0)]
if not over_losses.empty:
    avg_proj = over_losses['strikeouts_projection'].mean()
    avg_act  = over_losses['actual'].mean()
    print(f"\nOver bets that lost ({len(over_losses)} bets):")
    print(f"  Avg projected: {avg_proj:.1f} Ks")
    print(f"  Avg actual:    {avg_act:.1f} Ks")
    print(f"  Avg shortfall: {avg_act - avg_proj:+.1f} Ks")

under_losses = clean[(clean['best_side']=='under') & (clean['won']==0)]
if not under_losses.empty:
    avg_proj = under_losses['strikeouts_projection'].mean()
    avg_act  = under_losses['actual'].mean()
    print(f"\nUnder bets that lost ({len(under_losses)} bets):")
    print(f"  Avg projected: {avg_proj:.1f} Ks")
    print(f"  Avg actual:    {avg_act:.1f} Ks")
    print(f"  Avg overshoot: {avg_act - avg_proj:+.1f} Ks")

# League-wide K rate June vs historical
print("\nLeague K context - checking pitcher_game_logs...")
try:
    logs = pd.read_csv('data/raw/pitcher_game_logs.csv')
    logs['game_date'] = pd.to_datetime(logs['game_date'])
    june_logs = logs[(logs['game_date'] >= '2026-06-01') & (logs['game_date'] <= '2026-06-16')]
    may_logs  = logs[(logs['game_date'] >= '2026-05-01') & (logs['game_date'] <= '2026-05-31')]
    if not june_logs.empty and not may_logs.empty:
        print(f"  June 1-16 avg Ks/start: {june_logs['strikeouts'].mean():.2f}")
        print(f"  May avg Ks/start:       {may_logs['strikeouts'].mean():.2f}")
except Exception as e:
    print(f"  Could not load logs: {e}")
