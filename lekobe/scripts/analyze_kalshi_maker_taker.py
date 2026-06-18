from __future__ import annotations

import math
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
FILLS=BASE/'outputs/kalshi_kprop_fills.csv'
RAW=BASE/'scratch/audits/real_kalshi_fill_sharp_raw_rows.csv'
OUT=BASE/'scratch/audits/kalshi_kprop_collapsed_maker_taker.csv'
SUMMARY=BASE/'scratch/audits/kalshi_kprop_maker_taker_summary.csv'

TEAM_PREFIXES={
 'ARI','AZ','ATH','ATL','BAL','BOS','CHC','CIN','CLE','COL','CWS','DET','HOU','KC','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK','PHI','PIT','SD','SEA','SF','STL','TB','TBR','TEX','TOR','WSH'
}

def norm_name(x):
    if pd.isna(x): return ''
    s=str(x).lower()
    # drop accents for common names not needed here, keep ascii-ish tokens
    s=re.sub(r'[^a-z0-9]+','_',s).strip('_')
    return s

def last_name_key(name):
    n=norm_name(name)
    return n.split('_')[-1] if n else ''

def american_to_prob(o):
    if pd.isna(o): return np.nan
    try: o=float(o)
    except Exception: return np.nan
    if o == 0: return np.nan
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def event_date_from_ticker(ticker):
    m=re.search(r'KXMLBKS-(\d{2})([A-Z]{3})(\d{2})', str(ticker))
    if not m: return ''
    yy, mon, dd = m.groups()
    months={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    return f"20{yy}-{months[mon]:02d}-{int(dd):02d}"

def add_raw_local_date(raw):
    t=pd.to_datetime(raw['commence_time'], utc=True, errors='coerce')
    raw=raw.copy()
    raw['event_date_local']=t.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    return raw

def build_sharp_close(raw):
    raw=add_raw_local_date(raw)
    raw=raw[raw['snapshot_label'].eq('close_10m')].copy()
    raw['line']=pd.to_numeric(raw['line'], errors='coerce')
    raw['side']=raw['side'].astype(str).str.upper()
    raw['clean_name_book']=raw['clean_name_book'].fillna(raw.get('pitcher_name_book','')).map(norm_name)
    raw['last_key']=raw['pitcher_name_book'].map(last_name_key)
    rows=[]
    grp_cols=['event_date_local','clean_name_book','last_key','line','event_id']
    for keys,g in raw.groupby(grp_cols, dropna=False):
        event_date, clean, last, line, event_id=keys
        # Pinnacle pair
        def pair_for(df):
            over=df[df['side'].eq('OVER')]['american_odds']
            under=df[df['side'].eq('UNDER')]['american_odds']
            if over.empty or under.empty: return None
            op=american_to_prob(over.iloc[-1]); up=american_to_prob(under.iloc[-1])
            if not np.isfinite(op) or not np.isfinite(up) or op+up<=0: return None
            return up/(op+up)
        pin=g[g['bookmaker'].eq('pinnacle')]
        pin_under=pair_for(pin) if not pin.empty else None
        # consensus: bookmaker-level devig then average under fair
        vals=[]
        for _, bg in g.groupby('bookmaker'):
            v=pair_for(bg)
            if v is not None and np.isfinite(v): vals.append(v)
        cons_under=float(np.mean(vals)) if vals else np.nan
        rows.append({
            'event_date_local':event_date,
            'clean_name_book':clean,
            'last_key':last,
            'line':float(line) if pd.notna(line) else np.nan,
            'event_id':event_id,
            'pinnacle_under_close_cents': pin_under*100 if pin_under is not None else np.nan,
            'consensus_under_close_cents': cons_under*100 if np.isfinite(cons_under) else np.nan,
            'consensus_books': len(vals),
        })
    return pd.DataFrame(rows)

def collapse_fills(fills):
    rows=[]
    fills=fills.copy()
    fills['count_fp']=pd.to_numeric(fills['count_fp'], errors='coerce').fillna(0.0)
    fills['fill_price_cents']=pd.to_numeric(fills['fill_price_cents'], errors='coerce')
    fills['no_price_cents']=pd.to_numeric(fills['no_price_cents'], errors='coerce')
    fills['yes_price_cents']=pd.to_numeric(fills['yes_price_cents'], errors='coerce')
    fills['is_taker_bool']=fills['is_taker'].astype(str).str.lower().eq('true')
    for ticker,g in fills.groupby('ticker'):
        total_abs=g['count_fp'].abs().sum()
        if total_abs <= 0: continue
        # signed no exposure: + for buy no or sell yes; - for buy yes or sell no
        signed_no=[]
        prices=[]
        weights=[]
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
        if net_side=='UNDER':
            side_prices=g['no_price_cents']
        elif net_side=='OVER':
            side_prices=g['yes_price_cents']
        else:
            side_prices=g['fill_price_cents']
        avg_fill=np.average(side_prices, weights=g['count_fp']) if g['count_fp'].sum()>0 else np.nan
        taker_frac=np.average(g['is_taker_bool'].astype(float), weights=g['count_fp']) if g['count_fp'].sum()>0 else np.nan
        rows.append({
            'ticker':ticker,
            'event_date':event_date_from_ticker(ticker),
            'pitcher':g['pitcher'].dropna().iloc[0],
            'clean_name':norm_name(g['pitcher'].dropna().iloc[0]),
            'last_key':last_name_key(g['pitcher'].dropna().iloc[0]),
            'line':float(g['line'].dropna().iloc[0]),
            'net_side':net_side,
            'total_contracts':float(abs(net_no)),
            'gross_contracts':float(g['count_fp'].sum()),
            'avg_fill_cents':float(avg_fill),
            'taker_contract_frac':float(taker_frac),
            'execution_style':'taker_dominant' if taker_frac>0.5 else 'maker_dominant',
            'fill_count':int(len(g)),
            'taker_fills':int(g['is_taker_bool'].sum()),
            'maker_fills':int((~g['is_taker_bool']).sum()),
        })
    return pd.DataFrame(rows)

def attach_close(pos, close):
    out=pos.copy()
    out['sharp_source']='missing'
    out['sharp_under_close_cents']=np.nan
    out['sharp_side_close_cents']=np.nan
    out['consensus_books']=np.nan
    for i,r in out.iterrows():
        cand=close[(close['event_date_local'].eq(r['event_date'])) & (close['line'].eq(r['line']))]
        # exact clean-name match, then last-name fallback
        m=cand[cand['clean_name_book'].eq(r['clean_name'])]
        if m.empty:
            m=cand[cand['last_key'].eq(r['last_key'])]
        if m.empty:
            continue
        # if duplicate, prefer Pinnacle availability then max consensus books
        m=m.assign(_pin=m['pinnacle_under_close_cents'].notna().astype(int))
        row=m.sort_values(['_pin','consensus_books'], ascending=[False,False]).iloc[0]
        source='pinnacle' if pd.notna(row['pinnacle_under_close_cents']) else 'consensus'
        under=row['pinnacle_under_close_cents'] if source=='pinnacle' else row['consensus_under_close_cents']
        side_close=under if r['net_side']=='UNDER' else 100-under if r['net_side']=='OVER' else np.nan
        out.loc[i,'sharp_source']=source
        out.loc[i,'sharp_under_close_cents']=under
        out.loc[i,'sharp_side_close_cents']=side_close
        out.loc[i,'consensus_books']=row['consensus_books']
    out['clv_pp']=out['sharp_side_close_cents']-out['avg_fill_cents']
    return out

def summarize(df):
    rows=[]
    for style,g in df[df['sharp_side_close_cents'].notna()].groupby('execution_style'):
        rows.append({
            'execution_style':style,
            'n':len(g),
            'mean_fill_cents':g['avg_fill_cents'].mean(),
            'mean_sharp_close_cents':g['sharp_side_close_cents'].mean(),
            'mean_clv_pp':g['clv_pp'].mean(),
            'positive_clv_pct':(g['clv_pp']>0).mean()*100,
            'mean_taker_frac':g['taker_contract_frac'].mean()*100,
            'mean_contracts':g['total_contracts'].mean(),
        })
    return pd.DataFrame(rows)

def main():
    fills=pd.read_csv(FILLS)
    raw=pd.read_csv(RAW)
    pos=collapse_fills(fills)
    close=build_sharp_close(raw)
    out=attach_close(pos, close)
    out.to_csv(OUT,index=False)
    summ=summarize(out)
    summ.to_csv(SUMMARY,index=False)
    print('FILL_ROWS', len(fills))
    print('POSITIONS', len(out))
    print('SHARP_CLOSE_COVERAGE', int(out['sharp_side_close_cents'].notna().sum()))
    print('SHARP_CLOSE_COVERAGE_PCT', round(out['sharp_side_close_cents'].notna().mean()*100,2))
    print('PINNACLE_PRIORITY_MATCHES', int(out['sharp_source'].eq('pinnacle').sum()))
    print('CONSENSUS_FALLBACK_MATCHES', int(out['sharp_source'].eq('consensus').sum()))
    print('MISSING_SHARP_CLOSE', int(out['sharp_source'].eq('missing').sum()))
    print('STYLE_ALL')
    print(out.groupby('execution_style').agg(n=('ticker','count'),mean_taker_frac=('taker_contract_frac','mean'),mean_fill=('avg_fill_cents','mean')).to_string())
    print('STYLE_CLV')
    print(summ.to_string(index=False))
    if set(summ['execution_style']) >= {'maker_dominant','taker_dominant'}:
        mm=summ.set_index('execution_style')
        gap=mm.loc['maker_dominant','mean_clv_pp']-mm.loc['taker_dominant','mean_clv_pp']
        print('MAKER_MINUS_TAKER_CLV_GAP_PP', round(gap,4))
    taker=out[out['execution_style'].eq('taker_dominant') & out['sharp_side_close_cents'].notna()]
    maker=out[out['execution_style'].eq('maker_dominant') & out['sharp_side_close_cents'].notna()]
    if len(taker) and len(maker):
        penalty=taker['avg_fill_cents'].mean()-maker['avg_fill_cents'].mean()
        print('TAKER_MINUS_MAKER_MEAN_FILL_CENTS', round(penalty,4))
    print('OUTPUT', OUT)
    print('SUMMARY', SUMMARY)

if __name__=='__main__': main()
