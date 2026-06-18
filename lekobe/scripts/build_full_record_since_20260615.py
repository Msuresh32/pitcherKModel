from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
FILLS=BASE/'outputs/kalshi_kprop_fills.csv'
CLOSE=BASE/'scratch/audits/kalshi_kprop_fast_close_coverage.csv'
HIST=BASE/'historical_bets_ledger.csv'
OUT=BASE/'scratch/audits/full_record_since_2026_06_15.csv'
SUMMARY=BASE/'scratch/audits/full_record_since_2026_06_15_summary.csv'
START='2026-06-15'

def event_date_from_ticker(t):
    import re
    m=re.search(r'KXMLBKS-(\d{2})([A-Z]{3})(\d{2})', str(t))
    if not m: return ''
    yy,mon,dd=m.groups(); months={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    return f'20{yy}-{months[mon]:02d}-{int(dd):02d}'

def collapse_from_fills(f):
    f=f.copy()
    f['count_fp']=pd.to_numeric(f['count_fp'],errors='coerce').fillna(0.0)
    f['yes_price_cents']=pd.to_numeric(f['yes_price_cents'],errors='coerce')
    f['no_price_cents']=pd.to_numeric(f['no_price_cents'],errors='coerce')
    f['is_taker_bool']=f['is_taker'].astype(str).str.lower().eq('true')
    rows=[]
    for ticker,g in f.groupby('ticker'):
        signed_no=[]
        for _,r in g.iterrows():
            action=str(r['action']).lower(); side=str(r['side']).lower(); qty=float(r['count_fp'])
            sign=0
            if action=='buy' and side=='no': sign=1
            elif action=='sell' and side=='yes': sign=1
            elif action=='buy' and side=='yes': sign=-1
            elif action=='sell' and side=='no': sign=-1
            signed_no.append(sign*qty)
        net_no=sum(signed_no)
        net_side='UNDER' if net_no>0 else 'OVER' if net_no<0 else 'FLAT'
        price_col='no_price_cents' if net_side=='UNDER' else 'yes_price_cents' if net_side=='OVER' else 'fill_price_cents'
        qty=g['count_fp']
        avg=float(np.average(g[price_col], weights=qty)) if qty.sum()>0 else np.nan
        taker_frac=float(np.average(g['is_taker_bool'].astype(float),weights=qty)) if qty.sum()>0 else np.nan
        market_result=str(g['result'].dropna().iloc[-1]).lower() if g['result'].notna().any() else ''
        win=None
        if net_side=='UNDER' and market_result in {'yes','no'}: win = market_result=='no'
        elif net_side=='OVER' and market_result in {'yes','no'}: win = market_result=='yes'
        contracts=abs(net_no)
        # Dollar P/L for $1 payout contracts, using avg side price.
        if win is True:
            pnl=contracts*(100-avg)/100
            flat_units=1.0
            result='W'
        elif win is False:
            pnl=-contracts*avg/100
            flat_units=-1.0
            result='L'
        else:
            pnl=np.nan; flat_units=np.nan; result='PENDING'
        rows.append({
            'ticker':ticker,
            'date':event_date_from_ticker(ticker),
            'pitcher':g['pitcher'].dropna().iloc[0],
            'line':float(g['line'].dropna().iloc[0]),
            'side':net_side,
            'avg_fill_cents':avg,
            'contracts':contracts,
            'maker_taker':'taker' if taker_frac>0.5 else 'maker',
            'taker_contract_frac':taker_frac,
            'market_result':market_result,
            'result':result,
            'flat_units':flat_units,
            'contract_pnl_dollars':pnl,
        })
    return pd.DataFrame(rows)

def main():
    fills=pd.read_csv(FILLS)
    base=collapse_from_fills(fills)
    base=base[base['date']>=START].copy()
    close=pd.read_csv(CLOSE)
    close_cols=['ticker','close_source','grade_source','close_cents','clv_pp']
    out=base.merge(close[close_cols],on='ticker',how='left')
    out=out.rename(columns={'close_cents':'sharp_or_kalshi_close_cents'})
    out=out.sort_values(['date','maker_taker','pitcher','line'])
    # Historical ledger contribution check.
    hist=pd.read_csv(HIST)
    hist_after=hist[pd.to_datetime(hist['date_run'],errors='coerce')>=pd.Timestamp(START)]
    resolved=out[out['result'].isin(['W','L'])]
    wins=int((resolved['result']=='W').sum()); losses=int((resolved['result']=='L').sum())
    mean_clv=out['clv_pp'].mean()
    pos_clv=(out['clv_pp']>0).mean()*100 if len(out) else np.nan
    fills=len(out)
    summary=pd.DataFrame([{
        'start_date':START,
        'kalshi_positions':len(out),
        'historical_ledger_rows_added':len(hist_after),
        'resolved_positions':len(resolved),
        'wins':wins,
        'losses':losses,
        'flat_units':resolved['flat_units'].sum(),
        'contract_pnl_dollars':resolved['contract_pnl_dollars'].sum(),
        'mean_clv_pp':mean_clv,
        'positive_clv_pct':pos_clv,
        'fills_to_35':max(0,35-fills),
        'sharp_close_rows':int(out['grade_source'].eq('sharp_close').sum()),
        'kalshi_close_rows':int(out['grade_source'].eq('kalshi_close').sum()),
    }])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT,index=False)
    summary.to_csv(SUMMARY,index=False)
    print('HISTORICAL_LEDGER_DATE_MAX', hist['date_run'].max())
    print('HISTORICAL_LEDGER_ROWS_SINCE_START', len(hist_after))
    print('POSITIONS',len(out))
    print('RESOLVED',len(resolved))
    print('RECORD',f'{wins}-{losses}')
    print('FLAT_UNITS',round(resolved['flat_units'].sum(),2))
    print('CONTRACT_PNL_DOLLARS',round(resolved['contract_pnl_dollars'].sum(),2))
    print('MEAN_CLV_PP',round(mean_clv,4))
    print('POSITIVE_CLV_PCT',round(pos_clv,2))
    print('FILLS_TO_35',max(0,35-fills))
    print('CLOSE_SOURCE_COUNTS')
    print(out['close_source'].value_counts(dropna=False).to_string())
    print('TABLE')
    cols=['pitcher','date','line','side','avg_fill_cents','sharp_or_kalshi_close_cents','clv_pp','maker_taker','result','contracts','contract_pnl_dollars','close_source']
    disp=out[cols].copy()
    for c in ['avg_fill_cents','sharp_or_kalshi_close_cents','clv_pp','contracts','contract_pnl_dollars']:
        disp[c]=disp[c].map(lambda x:'' if pd.isna(x) else f'{x:.2f}')
    print(disp.to_string(index=False))
    print('OUT',OUT)
    print('SUMMARY',SUMMARY)
if __name__=='__main__': main()
