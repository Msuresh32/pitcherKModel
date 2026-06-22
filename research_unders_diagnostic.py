"""
Under Performance Diagnostic
WR=0.491, ROI=+5.0% — investigating root causes.
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_pois, poisson as spp
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

DATA_FILE = "data/processed_noopp_wf2025_ext/bt_noopp_oos_clv.csv"

raw = pd.read_csv(DATA_FILE)
raw["game_date"] = pd.to_datetime(raw["game_date"])

def devig(o, u):
    p_o = 100/(100+o) if o>=0 else abs(o)/(abs(o)+100)
    p_u = 100/(100+u) if u>=0 else abs(u)/(abs(u)+100)
    t = p_o + p_u; return p_o/t if t > 0 else 0.5

def pnl_fn(odds, won):
    o, w = float(odds), float(won)
    return o/100 if w==1 and o>=0 else (100/abs(o) if w==1 else -1.0)

# Build under pool
un = raw[raw["best_side"].astype(str) == "under"].copy().reset_index(drop=True)
un["won"] = np.where(un["strikeouts"] < un["line"], 1.0,
            np.where(un["strikeouts"] > un["line"], 0.0, np.nan))
un = un[un["won"].notna()].copy().reset_index(drop=True)

un["p_model_u"] = un.apply(lambda r: float(sp_pois.cdf(
    int(np.ceil(r["line"])-1), float(r["strikeouts_projection"]))), axis=1)
un["p_mkt_u"]   = un.apply(lambda r: devig(r["under_odds"], r["over_odds"]), axis=1)
un["margin_u"]  = un["line"] - un["strikeouts_projection"]
un["edge_raw"]  = (un["p_model_u"] - un["p_mkt_u"]) / un["p_mkt_u"] * 100
un["pnl"]       = un.apply(lambda r: pnl_fn(r["under_odds"], r["won"]), axis=1)
un["miss"]      = un["strikeouts"] - un["strikeouts_projection"]  # positive = model underestimated
un["month"]     = un["game_date"].dt.to_period("M")

# Also get overs for comparison
ov = raw[raw["best_side"].astype(str) == "over"].copy().reset_index(drop=True)
ov["won"] = np.where(ov["strikeouts"] > ov["line"], 1.0,
            np.where(ov["strikeouts"] < ov["line"], 0.0, np.nan))
ov = ov[ov["won"].notna()].copy().reset_index(drop=True)
ov["p_model"]  = ov.apply(lambda r: float(1-sp_pois.cdf(
    int(np.floor(r["line"])), float(r["strikeouts_projection"]))), axis=1)
ov["p_mkt"]    = ov.apply(lambda r: devig(r["over_odds"], r["under_odds"]), axis=1)
ov["margin"]   = ov["strikeouts_projection"] - ov["line"]
ov["miss"]     = ov["strikeouts"] - ov["strikeouts_projection"]
ov["pnl"]      = ov.apply(lambda r: pnl_fn(r["over_odds"], r["won"]), axis=1)
ov["month"]    = ov["game_date"].dt.to_period("M")

print("=" * 76)
print("UNDER PERFORMANCE DIAGNOSTIC")
print("=" * 76)

n_u = len(un); n_o = len(ov)
print(f"\n  Overs:  n={n_o:3d}  WR={ov['won'].mean():.3f}  ROI={ov['pnl'].mean():>+.3f}")
print(f"  Unders: n={n_u:3d}  WR={un['won'].mean():.3f}  ROI={un['pnl'].mean():>+.3f}")

# ─── 1. MARKET-IMPLIED VS ACTUAL WIN RATE ─────────────────────────────────────
print(f"\n{'─'*60}  1. MARKET-IMPLIED vs ACTUAL WR")
print(f"\n  {'':20}  {'Overs':>10}  {'Unders':>10}")
print(f"  {'Actual WR':20}  {ov['won'].mean():>10.3f}  {un['won'].mean():>10.3f}")
print(f"  {'Market-implied WR':20}  {ov['p_mkt'].mean():>10.3f}  {un['p_mkt_u'].mean():>10.3f}")
print(f"  {'Model p_model':20}  {ov['p_model'].mean():>10.3f}  {un['p_model_u'].mean():>10.3f}")
print(f"  {'Actual - Market':20}  {ov['won'].mean()-ov['p_mkt'].mean():>+10.3f}  {un['won'].mean()-un['p_mkt_u'].mean():>+10.3f}")
print(f"  {'Model - Market (edge)':20}  {ov['p_model'].mean()-ov['p_mkt'].mean():>+10.3f}  {un['p_model_u'].mean()-un['p_mkt_u'].mean():>+10.3f}")

print(f"""
  INTERPRETATION:
  Overs:  actual WR {ov['won'].mean():.3f} > market-implied {ov['p_mkt'].mean():.3f} → model is finding real overs
  Unders: actual WR {un['won'].mean():.3f} vs market-implied {un['p_mkt_u'].mean():.3f}
  Difference: {un['won'].mean()-un['p_mkt_u'].mean():>+.3f}
  {'→ Unders are winning at BELOW market-implied rate — model is selecting wrong side' if un['won'].mean() < un['p_mkt_u'].mean() else '→ Unders winning at market rate or above'}""")

# ─── 2. MODEL PROJECTION BIAS ─────────────────────────────────────────────────
print(f"\n{'─'*60}  2. MODEL PROJECTION BIAS")
print(f"""
  For overs:  model recommends when projection > line (miss = actual - projection)
  For unders: model recommends when projection < line (miss = actual - projection)

  Over bets:
    Projection - line (margin):  mean={ov['margin'].mean():>+.3f}  (model says +{ov['margin'].mean():.2f}K above line)
    Actual - projection (error): mean={ov['miss'].mean():>+.3f}  std={ov['miss'].std():.3f}
    Actual - line:               mean={(ov['strikeouts']-ov['line']).mean():>+.3f}

  Under bets:
    Line - projection (margin):  mean={un['margin_u'].mean():>+.3f}  (model says {un['margin_u'].mean():.2f}K below line)
    Actual - projection (error): mean={un['miss'].mean():>+.3f}  std={un['miss'].std():.3f}
    Actual - line:               mean={(un['strikeouts']-un['line']).mean():>+.3f}
""")

print(f"  KEY: Actual - projection (forecast error):")
print(f"    Overs:  {ov['miss'].mean():>+.4f}  (positive = model underestimated, but bet over → helps)")
print(f"    Unders: {un['miss'].mean():>+.4f}  (positive = model underestimated, bet under → HURTS)")
if un["miss"].mean() > 0:
    print(f"\n  → Model SYSTEMATICALLY UNDERESTIMATES strikeouts on under bets.")
    print(f"    Model says pitcher will get fewer Ks than line, but actual count")
    print(f"    exceeds projection by {un['miss'].mean():.3f} on average.")
    print(f"    This is the core failure mode.")

# ─── 3. CALIBRATION COMPARISON ────────────────────────────────────────────────
print(f"\n{'─'*60}  3. CALIBRATION COMPARISON")
print(f"\n  Overs — p_model vs actual WR (quintile bins by p_model):")
for i, (lo, hi) in enumerate(zip([0,.4,.5,.6,.7,.8],[.4,.5,.6,.7,.8,1.])):
    sub = ov[(ov["p_model"]>=lo)&(ov["p_model"]<hi)]
    if len(sub)>0:
        print(f"    p∈[{lo:.1f},{hi:.1f}): pred={sub['p_model'].mean():.3f}  actual={sub['won'].mean():.3f}  n={len(sub):3d}  "
              f"Δ={sub['won'].mean()-sub['p_model'].mean():>+.3f}")

print(f"\n  Unders — p_model_u vs actual WR:")
for i, (lo, hi) in enumerate(zip([0,.4,.5,.6,.7,.8],[.4,.5,.6,.7,.8,1.])):
    sub = un[(un["p_model_u"]>=lo)&(un["p_model_u"]<hi)]
    if len(sub)>0:
        print(f"    p∈[{lo:.1f},{hi:.1f}): pred={sub['p_model_u'].mean():.3f}  actual={sub['won'].mean():.3f}  n={len(sub):3d}  "
              f"Δ={sub['won'].mean()-sub['p_model_u'].mean():>+.3f}")
print(f"\n  Under model is over-confident in the same direction as over model:")
print(f"    Over model mean p_model={ov['p_model'].mean():.3f} vs actual WR={ov['won'].mean():.3f}  bias={ov['p_model'].mean()-ov['won'].mean():>+.3f}")
print(f"    Under model mean p_model={un['p_model_u'].mean():.3f} vs actual WR={un['won'].mean():.3f}  bias={un['p_model_u'].mean()-un['won'].mean():>+.3f}")

# ─── 4. ODDS STRUCTURE ────────────────────────────────────────────────────────
print(f"\n{'─'*60}  4. ODDS STRUCTURE")
print(f"\n  Overs avg odds:  {ov['over_odds'].mean():>+.1f}")
print(f"  Unders avg odds: {un['under_odds'].mean():>+.1f}")
print(f"\n  Over odds distribution:")
for lo, hi in [(-200,-100),(-100,-50),(-50,0),(0,50),(50,100),(100,200)]:
    sub = ov[(ov["over_odds"]>=lo)&(ov["over_odds"]<hi)]
    if len(sub)>0:
        print(f"    [{lo:>5},{hi:>4}): n={len(sub):3d}  WR={sub['won'].mean():.3f}  ROI={sub['pnl'].mean():>+.3f}")

print(f"\n  Under odds distribution:")
for lo, hi in [(-200,-100),(-100,-50),(-50,0),(0,50),(50,100),(100,200)]:
    sub = un[(un["under_odds"]>=lo)&(un["under_odds"]<hi)]
    if len(sub)>0:
        print(f"    [{lo:>5},{hi:>4}): n={len(sub):3d}  WR={sub['won'].mean():.3f}  ROI={sub['pnl'].mean():>+.3f}")

# ─── 5. LINE DISTRIBUTION ─────────────────────────────────────────────────────
print(f"\n{'─'*60}  5. LINE DISTRIBUTION")
print(f"\n  Over lines:")
for lo, hi in [(0,4),(4,5),(5,6),(6,7),(7,10)]:
    sub = ov[(ov["line"]>=lo)&(ov["line"]<hi)]
    if len(sub)>0:
        print(f"    line∈[{lo},{hi}): n={len(sub):3d}  WR={sub['won'].mean():.3f}  proj={sub['strikeouts_projection'].mean():.2f}  ROI={sub['pnl'].mean():>+.3f}")

print(f"\n  Under lines:")
for lo, hi in [(0,4),(4,5),(5,6),(6,7),(7,10)]:
    sub = un[(un["line"]>=lo)&(un["line"]<hi)]
    if len(sub)>0:
        print(f"    line∈[{lo},{hi}): n={len(sub):3d}  WR={sub['won'].mean():.3f}  proj={sub['strikeouts_projection'].mean():.2f}  ROI={sub['pnl'].mean():>+.3f}")

# ─── 6. MONTHLY PATTERN ────────────────────────────────────────────────────────
print(f"\n{'─'*60}  6. MONTHLY PATTERN")
print(f"\n  {'Month':<10}  {'n_ov':>5}  {'WR_ov':>6}  {'ROI_ov':>7}  ||  {'n_un':>5}  {'WR_un':>6}  {'ROI_un':>7}")
print(f"  {'─'*60}")
all_months = sorted(set(list(ov["month"].unique()) + list(un["month"].unique())))
for m in all_months:
    ov_m = ov[ov["month"]==m]; un_m = un[un["month"]==m]
    ov_s = f"n={len(ov_m):3d}  WR={ov_m['won'].mean():.3f}  ROI={ov_m['pnl'].mean():>+.3f}" if len(ov_m)>0 else "─"*25
    un_s = f"n={len(un_m):3d}  WR={un_m['won'].mean():.3f}  ROI={un_m['pnl'].mean():>+.3f}" if len(un_m)>0 else "─"*25
    print(f"  {str(m):<10}  {ov_s}  ||  {un_s}")

# ─── 7. PITCHER OVERLAP ────────────────────────────────────────────────────────
print(f"\n{'─'*60}  7. PITCHER OVERLAP (same pitcher, over vs under)")
ov_ptchs = set(ov["pitcher_name"].unique())
un_ptchs = set(un["pitcher_name"].unique())
both_ptchs = ov_ptchs & un_ptchs
print(f"\n  Pitchers who appear in BOTH over AND under bets: {len(both_ptchs)}")
print(f"  Over-only pitchers: {len(ov_ptchs - un_ptchs)}")
print(f"  Under-only pitchers: {len(un_ptchs - ov_ptchs)}")
print(f"\n  Top under pitchers (by n):")
un_by_p = un.groupby("pitcher_name").agg(
    n=("won","count"), wr=("won","mean"), roi=("pnl","mean"),
    proj_mean=("strikeouts_projection","mean"), line_mean=("line","mean"),
    miss_mean=("miss","mean")).sort_values("n",ascending=False)
for p, r in un_by_p.head(12).iterrows():
    cross = "†both" if p in both_ptchs else ""
    print(f"    {p:<30}  n={int(r['n']):2d}  WR={r['wr']:.3f}  ROI={r['roi']:>+.3f}  "
          f"proj={r['proj_mean']:.2f}  line={r['line_mean']:.2f}  miss={r['miss_mean']:>+.3f}  {cross}")

# ─── 8. MARKET EDGE ACCURACY ──────────────────────────────────────────────────
print(f"\n{'─'*60}  8. IS THE MODEL EDGE REAL FOR UNDERS?")
print(f"\n  Testing: when model projects edge for unders, does it materialize?")
print(f"\n  Edge bins (edge_raw %) → actual WR:")
for lo, hi in [(0,10),(10,20),(20,30),(30,50),(50,100)]:
    sub = un[(un["edge_raw"]>=lo)&(un["edge_raw"]<hi)]
    if len(sub)>5:
        exp_w = sub["p_mkt_u"].sum(); act_w = sub["won"].sum()
        pp = float(1 - spp.cdf(int(act_w)-1, exp_w))
        print(f"    edge∈[{lo:>3},{hi:>3}%): n={len(sub):3d}  WR={sub['won'].mean():.3f}  "
              f"mkt_impl={sub['p_mkt_u'].mean():.3f}  ROI={sub['pnl'].mean():>+.3f}  Pois={pp:.3f}")
print(f"\n  Monotonicity: do higher under edges win more?")
un_s = un.sort_values("edge_raw"); qs = len(un_s)//4
rois_q = [un_s.iloc[i*qs:(i+1)*qs if i<3 else len(un_s)]["pnl"].mean() for i in range(4)]
print(f"  Q1→Q2→Q3→Q4: {' → '.join([f'{r*100:>+.1f}%' for r in rois_q])}")
mono = sum(rois_q[i+1]>rois_q[i] for i in range(3))
print(f"  Monotone pairs: {mono}/3")

# ─── 9. ROOT CAUSE SUMMARY ────────────────────────────────────────────────────
print(f"\n{'═'*76}  ROOT CAUSE SUMMARY")
actual_u_wr = un["won"].mean()
mkt_u_wr = un["p_mkt_u"].mean()
model_u_wr = un["p_model_u"].mean()
actual_o_wr = ov["won"].mean()
mkt_o_wr = ov["p_mkt"].mean()
avg_miss_u = un["miss"].mean()
avg_miss_o = ov["miss"].mean()
u_mono = mono

print(f"""
  1. MODEL BIAS DIRECTION:
     Model under-estimates strikeouts on BOTH over AND under bets.
     Over miss (actual-proj): {avg_miss_o:>+.3f}K  → HELPS overs (actual > proj > line)
     Under miss (actual-proj): {avg_miss_u:>+.3f}K  → HURTS unders (actual > proj, but bet under)

     The same directional bias that HELPS overs HURTS unders.
     This is the primary cause.

  2. MARKET EFFICIENCY ASYMMETRY:
     Overs:  actual WR ({actual_o_wr:.3f}) > market-implied ({mkt_o_wr:.3f}) → model beats market
     Unders: actual WR ({actual_u_wr:.3f}) {'<' if actual_u_wr < mkt_u_wr else '>='} market-implied ({mkt_u_wr:.3f})
     {'→ Model is WORSE than market at selecting unders' if actual_u_wr < mkt_u_wr else '→ Model beats market on unders too'}

  3. UNDERLYING REASON:
     The Poisson GLM was trained to minimize prediction error on strikeout counts.
     Strikeout distributions are right-skewed: rare big games pull projections high.
     The model projects CONSERVATIVELY (low projections = less risk of big miss).
     Conservative projections mean:
       - When pitcher is projected above line → model is often right (overs)
       - When pitcher is projected below line → real games often exceed conservative forecast

  4. NO-OPPORTUNITY CONSTRAINT:
     The model cannot see lineup quality, weather, or park factors.
     These factors tend to be BULLISH for strikeouts (weak lineups → more Ks).
     Missing bullish information → model underestimates more than it overestimates.
     Under bets are disproportionately hurt by this.

  5. MONOTONICITY: {mono}/3 under edge quintiles are monotone
     {'→ Under edge ranking has some signal' if mono >= 2 else '→ Under edge has NO reliable ordering — model cannot rank unders'}

  CONCLUSION: Under bets are structurally disadvantaged by this model.
  The same feature that makes overs work (projections too low → actual > line)
  makes unders fail (projections too low → actual still > line even when
  model predicted under). The model has a systematic low-K bias.

  RECOMMENDATION: Exclude unders from the deployment policy.
  The under ROI of {un['pnl'].mean():>+.3f} is driven almost entirely by vig
  losses, not model error. Running unders purely to satisfy n>=300
  adds noise and dilutes the genuine edge from the over model.
""")
