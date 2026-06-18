from __future__ import annotations
import re, json, sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from kalshi_auth import sign_kalshi_request
KBASE='https://api.elections.kalshi.com/trade-api/v2'
POS=BASE/'scratch/audits/kalshi_kprop_collapsed_maker_taker.csv'
RAW=BASE/'scratch/audits/real_kalshi_fill_sharp_raw_rows.csv'
OUT=BASE/'scratch/audits/kalshi_kprop_fast_close_coverage.csv'
SUMMARY=BASE/'scratch/audits/kalshi_kprop_fast_close_summary.csv'
META=BASE/'scratch/audits/kalshi_kprop_fast_meta_cache.json'
CANDLES=BASE/'scratch/audits/kalshi_kprop_fast_candle_cache.json'
PRIORITY=['pinnacle','betonlineag','betonline','draftkings']

def loadj(p): return json.loads(p.read_text()) if p.exists() else {}
def savej(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True))
def norm(x): return re.sub(r'[^a-z0-9]+','_',str(x).lower()).strip('_') if pd.notna(x) else ''
def last(x):
    n=norm(x); return n.split('_')[-1] if n else ''
def ap(o):
    try:o=float(o)
    except Exception:return np.nan
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100) if o<0 else np.nan
def pair(g):
    over=g[g.side.eq('OVER')]; under=g[g.side.eq('UNDER')]
    if over.empty or under.empty: return np.nan
    op=ap(over.american_odds.iloc[-1]); up=ap(under.american_odds.iloc[-1])
    return up/(op+up)*100 if np.isfinite(op) and np.isfinite(up) and op+up>0 else np.nan
def sign_get(path, params=None):
    h=sign_kalshi_request('GET','/trade-api/v2'+path)
    r=requests.get(KBASE+path,params=params,headers=h,timeout=20)
    r.raise_for_status(); return r.json()
def meta(t, cache):
    if t not in cache:
        try: cache[t]=sign_get('/markets/'+t).get('market') or {}
        except Exception as e: cache[t]={'_error':str(e)[:200]}
        print('META',len(cache),t,flush=True)
    return cache[t]

def ticker_event_start_utc(ticker):
    m=re.search(r'KXMLBKS-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})', str(ticker))
    if not m: return None
    yy, mon, dd, hh, mm=m.groups()
    months={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    local=datetime(2000+int(yy), months[mon], int(dd), int(hh), int(mm), tzinfo=ZoneInfo('America/New_York'))
    return local.astimezone(ZoneInfo('UTC'))

def close_time(m):
    t=m.get('close_time') or m.get('occurrence_datetime') or m.get('expected_expiration_time')
    return datetime.fromisoformat(str(t).replace('Z','+00:00')) if t else None
def candle_close(row, mc, cc):
    t=row.ticker; m=meta(t,mc); dt=ticker_event_start_utc(t) or close_time(m)
    if not dt: return np.nan
    end=dt-timedelta(minutes=10); start=end-timedelta(hours=12)
    key=f'{t}|{int(start.timestamp())}|{int(end.timestamp())}'
    if key not in cc:
        try:
            data=sign_get(f'/series/KXMLBKS/markets/{t}/candlesticks',{'start_ts':int(start.timestamp()),'end_ts':int(end.timestamp()),'period_interval':1,'include_latest_before_start':'true'})
            cc[key]=data.get('candlesticks') or []
        except Exception as e: cc[key]={'_error':str(e)[:200]}
        print('CANDLE',len(cc),t,flush=True)
    c=cc[key]
    if isinstance(c,dict) or not c: return np.nan
    vals=[]
    for x in c:
        price=x.get('price') or {}; v=price.get('close_dollars') or price.get('previous_dollars')
        if v is None:
            bid=(x.get('yes_bid') or {}).get('close_dollars'); ask=(x.get('yes_ask') or {}).get('close_dollars')
            if bid is not None and ask is not None: v=(float(bid)+float(ask))/2
            elif bid is not None: v=float(bid)
            elif ask is not None: v=float(ask)
        if v is not None:
            try: vals.append(float(v)*100)
            except Exception: pass
    if not vals: return np.nan
    yes=vals[-1]
    return yes if row.net_side=='OVER' else 100-yes
def sportsbook(row, raw):
    cand=raw[(raw.clean_name_book.eq(row.clean_name)) | (raw.last_key.eq(row.last_key))].copy()
    if cand.empty: return None
    same_date=cand[cand.commence_time.astype(str).str[:10].eq(str(row.event_date))]
    if same_date.empty:
        return None
    cand=same_date
    lines=sorted(cand.line.dropna().unique())
    if not lines: return None
    chosen=row.line if row.line in lines else min(lines,key=lambda x:abs(float(x)-float(row.line)))
    lc=cand[cand.line.eq(chosen)]
    for b in PRIORITY:
        bg=lc[lc.bookmaker.eq(b)] if b!='betonline' else lc[lc.bookmaker.str.contains('betonline',na=False)]
        if not bg.empty:
            v=pair(bg)
            if np.isfinite(v): return b,v,float(chosen),float(chosen)-float(row.line),True
    vals=[]
    for _,bg in lc.groupby('bookmaker'):
        v=pair(bg)
        if np.isfinite(v): vals.append(v)
    if vals: return 'consensus',float(np.mean(vals)),float(chosen),float(chosen)-float(row.line),len(vals)
    return None
def main():
    pos=pd.read_csv(POS); raw=pd.read_csv(RAW)
    pos['clean_name']=pos.pitcher.map(norm); pos['last_key']=pos.pitcher.map(last)
    raw['line']=pd.to_numeric(raw.line,errors='coerce'); raw['side']=raw.side.astype(str).str.upper(); raw['bookmaker']=raw.bookmaker.astype(str).str.lower(); raw['clean_name_book']=raw.clean_name_book.fillna(raw.pitcher_name_book).map(norm); raw['last_key']=raw.pitcher_name_book.map(last)
    mc=loadj(META); cc=loadj(CANDLES); rows=[]
    for i,r in pos.iterrows():
        rec=r.to_dict(); sb=sportsbook(r,raw)
        if sb:
            src,under,line,hook,extra=sb; close=under if r.net_side=='UNDER' else 100-under
            rec.update({'grade_source':'sharp_close','close_source':src,'close_cents':close,'matched_close_line':line,'hook_diff':hook})
        else:
            kc=candle_close(r,mc,cc); rec.update({'grade_source':'kalshi_close' if np.isfinite(kc) else 'missing','close_source':'kalshi_close' if np.isfinite(kc) else 'missing','close_cents':kc,'matched_close_line':np.nan,'hook_diff':np.nan})
        rec['clv_pp']=rec['close_cents']-rec['avg_fill_cents'] if np.isfinite(rec['close_cents']) else np.nan
        rows.append(rec)
    out=pd.DataFrame(rows); out.to_csv(OUT,index=False); savej(META,mc); savej(CANDLES,cc)
    summ=out[out.close_cents.notna()].groupby(['execution_style','grade_source']).agg(n=('ticker','count'),mean_fill=('avg_fill_cents','mean'),mean_close=('close_cents','mean'),mean_clv=('clv_pp','mean'),positive_pct=('clv_pp',lambda s:(s>0).mean()*100)).reset_index(); summ.to_csv(SUMMARY,index=False)
    print('ROWS',len(out)); print('ANY_COVERAGE',out.close_cents.notna().sum()); print('SHARP',out.grade_source.eq('sharp_close').sum()); print('KALSHI',out.grade_source.eq('kalshi_close').sum()); print('MISSING',out.grade_source.eq('missing').sum()); print(out.close_source.value_counts(dropna=False).to_string()); print(summ.to_string(index=False)); print('OUT',OUT); print('SUMMARY',SUMMARY)
if __name__=='__main__': main()
