from __future__ import annotations
import os, re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from mlb_identity import normalize_name
load_dotenv(BASE/'.env')
API_KEY=os.getenv('THE_ODDS_API_KEY')
SPORT='baseball_mlb'
TZ=ZoneInfo(os.getenv('MLB_SLATE_TZ','America/Los_Angeles'))
TODAY=datetime.now(TZ).date()
BASE_URL='https://api.the-odds-api.com/v4'
BOOK_PRIORITY=['pinnacle']
CONSENSUS_EXCLUDE=set()
OUT=BASE/'scratch/audits/today_over_favored_under_posts.csv'

def american_to_prob(o):
    try: o=float(o)
    except Exception: return None
    if o == 0: return None
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def prob_to_american(p):
    if p is None or p<=0 or p>=1: return ''
    return int(round(-100*p/(1-p))) if p>=0.5 else int(round(100*(1-p)/p))

def devig_pair(over_odds, under_odds):
    op=american_to_prob(over_odds); up=american_to_prob(under_odds)
    if op is None or up is None or op+up<=0: return None
    return {'over':op/(op+up), 'under':up/(op+up)}

def floor_cent(x):
    return int(x // 1)

def get_events():
    r=requests.get(f'{BASE_URL}/sports/{SPORT}/events',params={'apiKey':API_KEY},timeout=20)
    r.raise_for_status(); return r.json(), r.headers

def event_date(ev):
    dt=datetime.fromisoformat(ev['commence_time'].replace('Z','+00:00')).astimezone(TZ)
    return dt.date(), dt

def get_event_odds(eid):
    params={
        'apiKey':API_KEY,
        'regions':'us,us2,eu,au',
        'markets':'pitcher_strikeouts,pitcher_strikeouts_alternate',
        'oddsFormat':'american',
    }
    r=requests.get(f'{BASE_URL}/sports/{SPORT}/events/{eid}/odds',params=params,timeout=30)
    r.raise_for_status(); return r.json(), r.headers

def collect_pairs(payload):
    rows=[]
    for bk in payload.get('bookmakers',[]) or []:
        bkey=bk.get('key')
        for mk in bk.get('markets',[]) or []:
            if mk.get('key') not in {'pitcher_strikeouts','pitcher_strikeouts_alternate'}: continue
            by_key={}
            for o in mk.get('outcomes',[]) or []:
                pitcher=o.get('description') or o.get('participant') or ''
                line=o.get('point')
                if not pitcher or line is None: continue
                key=(normalize_name(pitcher), pitcher, float(line), mk.get('key'))
                by_key.setdefault(key,{})[str(o.get('name')).lower()]=o.get('price')
            for (clean,pitcher,line,mkey), sides in by_key.items():
                if 'over' in sides and 'under' in sides:
                    fair=devig_pair(sides['over'], sides['under'])
                    if fair:
                        rows.append({
                            'bookmaker':bkey,
                            'book_title':bk.get('title'),
                            'market_key':mkey,
                            'clean_name':clean,
                            'pitcher':pitcher,
                            'line':line,
                            'over_odds':sides['over'],
                            'under_odds':sides['under'],
                            'fair_over_prob':fair['over'],
                            'fair_under_prob':fair['under'],
                        })
    return pd.DataFrame(rows)

def choose_lines(df):
    out=[]
    if df.empty: return pd.DataFrame()
    group_cols=['clean_name','pitcher','line']
    for _,g in df.groupby(group_cols):
        pin=g[g.bookmaker.eq('pinnacle')]
        if not pin.empty:
            # Prefer mainline over alternate if duplicate.
            row=pin.sort_values('market_key').iloc[0].to_dict()
            row['source']='pinnacle'
            row['source_books']=1
            row['source_over_price']=row['over_odds']
        else:
            vals=[]
            for _,bg in g.groupby('bookmaker'):
                vals.append({
                    'fair_under_prob':bg.fair_under_prob.iloc[0],
                    'fair_over_prob':bg.fair_over_prob.iloc[0],
                    'over_odds':bg.over_odds.iloc[0],
                    'under_odds':bg.under_odds.iloc[0],
                    'bookmaker':bg.bookmaker.iloc[0],
                })
            if not vals: continue
            tmp=pd.DataFrame(vals)
            row=g.iloc[0].to_dict()
            row['source']='consensus'
            row['source_books']=len(tmp)
            row['fair_under_prob']=tmp.fair_under_prob.mean()
            row['fair_over_prob']=tmp.fair_over_prob.mean()
            # Use quoted market price for the rule gate; keep fair_under_prob de-vigged.
            row['source_over_price']=int(round(tmp.over_odds.median()))
            row['over_odds']=row['source_over_price']
            row['under_odds']=prob_to_american(row['fair_under_prob'])
        out.append(row)
    return pd.DataFrame(out)

def main():
    if not API_KEY: raise SystemExit('THE_ODDS_API_KEY missing')
    events, h=get_events()
    now=datetime.now(timezone.utc)
    all_rows=[]; event_count=0; skipped_started=0; skipped_date=0
    for ev in events:
        d, dt_local=event_date(ev)
        if d != TODAY:
            skipped_date += 1; continue
        if datetime.fromisoformat(ev['commence_time'].replace('Z','+00:00')) <= now:
            skipped_started += 1; continue
        event_count += 1
        payload, hh=get_event_odds(ev['id'])
        pairs=collect_pairs(payload)
        if not pairs.empty:
            pairs['event_id']=ev['id']; pairs['commence_time']=ev['commence_time']; pairs['away_team']=ev.get('away_team'); pairs['home_team']=ev.get('home_team')
            all_rows.append(pairs)
    raw=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    chosen=choose_lines(raw)
    if chosen.empty:
        print('NO_ROWS')
        print('event_count',event_count,'skipped_started',skipped_started,'skipped_date',skipped_date)
        return
    # Filter over priced <= -140 using the quoted selected-source price.
    chosen['over_odds_num']=pd.to_numeric(chosen.get('source_over_price', chosen['over_odds']),errors='coerce')
    q=chosen[chosen.over_odds_num <= -140].copy()
    q['sharp_fair_under_cents']=q.fair_under_prob*100
    q['post_at_cents']=q.sharp_fair_under_cents.apply(floor_cent)
    q['ceiling_cents']=(q.sharp_fair_under_cents-2.5).apply(floor_cent)
    q['priority']=q.over_odds_num <= -150
    q=q.sort_values(['priority','over_odds_num','pitcher','line'],ascending=[False,True,True,True])
    cols=['pitcher','line','over_odds_num','source','source_books','sharp_fair_under_cents','post_at_cents','ceiling_cents','commence_time','away_team','home_team']
    OUT.parent.mkdir(parents=True,exist_ok=True)
    q[cols].to_csv(OUT,index=False)
    print('SLATE_DATE',TODAY)
    print('EVENTS_USED',event_count)
    print('PITCHER_LINES_TOTAL',len(chosen))
    print('QUALIFIERS_OVER_LE_NEG140',len(q))
    print('PRIORITY_OVER_LE_NEG150',int(q.priority.sum()))
    print('SOURCE_COUNTS')
    print(q.source.value_counts().to_string() if len(q) else '')
    print('OUTPUT',OUT)
    display=q.copy()
    display['sharp_fair_under_cents']=display['sharp_fair_under_cents'].map(lambda x:f'{x:.2f}')
    print(display[['pitcher','line','over_odds_num','source','source_books','sharp_fair_under_cents','post_at_cents','ceiling_cents','priority']].to_string(index=False))
if __name__=='__main__': main()
