"""
Run this script to regenerate dashboard.html with the latest data.
Then open dashboard.html in any browser -- no server needed.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

EXPORTS = Path("data/exports")

# -- helpers ------------------------------------------------------------------
def pnl_calc(won, odds, stake=100):
    if pd.isna(won) or pd.isna(odds):
        return None
    decimal = 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    return round(stake * (decimal - 1) if won == 1 else -stake, 2)

def am_to_dec(odds):
    if pd.isna(odds): return np.nan
    o = float(odds)
    return 1 + o/100 if o > 0 else 1 + 100/abs(o)

def mean_dec(american_list):
    vals = [am_to_dec(o) for o in american_list if pd.notna(o)]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def compute_units_d(edge_pct, gap_abs):
    if (edge_pct or 0) >= 35: return "2.5U"
    if (edge_pct or 0) >= 18 and (gap_abs or 0) >= 0.5: return "2U"
    return "1U"

def composite_score_d(edge_pct, gap_abs):
    return (edge_pct or 0) + 2.0 * (gap_abs or 0)

def dec_to_am(dec):
    if dec is None or pd.isna(dec) or dec <= 1: return None
    return round((dec - 1) * 100) if dec >= 2.0 else round(-100 / (dec - 1))

def min_odds_for_ev(hit_prob, target_ev):
    """Minimum American odds needed to achieve target_ev edge."""
    if not hit_prob or hit_prob <= 0 or hit_prob >= 1: return None
    return dec_to_am((1 + target_ev) / hit_prob)

def edge_at(hit_prob, american_odds):
    if hit_prob is None or american_odds is None: return None
    d = am_to_dec(american_odds)
    return round((hit_prob * (d - 1) - (1 - hit_prob)) * 100, 1)

def build_ladder(hit_prob, entry_odds):
    """Upside shopping ladder: current odds + 4 better-price scenarios."""
    if hit_prob is None or entry_odds is None: return []
    rungs = []
    for shift in [0, 5, 10, 15, 25]:
        adj = entry_odds + shift
        e = edge_at(hit_prob, adj)
        rungs.append({"shift": shift, "odds": int(adj), "edge": e})
    return rungs

def build_clv_index():
    """Returns dict (date_str, player_name, line, side) -> clv_pct float.
    Uses vectorised groupby — O(n) not O(n*k)."""
    dfs = []
    for p in ["data/odds/full_2026_odds.csv", "data/odds/historical_pitcher_props_2025.csv"]:
        if Path(p).exists():
            dfs.append(pd.read_csv(p))
    pin_p = Path("data/odds/pinnacle_close_cache.csv")
    if pin_p.exists():
        pin = pd.read_csv(pin_p)
        pin["bookmaker"] = "pinnacle"
        pin["snapshot_type"] = "close"
        dfs.append(pin)
    if not dfs:
        return {}

    odds = pd.concat(dfs, ignore_index=True)
    odds["game_date"]  = pd.to_datetime(odds["game_date"]).dt.strftime("%Y-%m-%d")
    odds["line"]       = pd.to_numeric(odds["line"],       errors="coerce")
    odds["over_odds"]  = pd.to_numeric(odds["over_odds"],  errors="coerce")
    odds["under_odds"] = pd.to_numeric(odds["under_odds"], errors="coerce")
    odds["line_f"]     = odds["line"].round(1)  # normalise 4.5 vs 4.50

    # ── Pre-build close dict: (date, player, line) → {book: {over_dec, under_dec}} ──
    close_dict = {}  # (date, player, line) → {'pinnacle':{'over':dec,'under':dec}, ...}
    for _, row in odds[odds["snapshot_type"] == "close"].iterrows():
        key  = (row["game_date"], row["player_name"], row["line_f"])
        book = str(row.get("bookmaker",""))
        if key not in close_dict:
            close_dict[key] = {}
        if book not in close_dict[key]:
            close_dict[key][book] = {"over": np.nan, "under": np.nan}
        if pd.notna(row["over_odds"]):
            close_dict[key][book]["over"]  = am_to_dec(row["over_odds"])
        if pd.notna(row["under_odds"]):
            close_dict[key][book]["under"] = am_to_dec(row["under_odds"])

    def close_dec(date, player, line, side):
        key   = (date, player, round(float(line), 1) if pd.notna(line) else None)
        books = close_dict.get(key, {})
        # Pinnacle priority
        pin = books.get("pinnacle", {})
        v   = pin.get(side, np.nan)
        if pd.notna(v) and v > 1: return v
        # Avg of all available books
        vals = [b.get(side, np.nan) for b in books.values()]
        vals = [v for v in vals if pd.notna(v) and v > 1]
        return float(np.mean(vals)) if vals else np.nan

    # ── Open avg: (date, player, line, side) → mean decimal (non-PIN) ──
    op = odds[(odds["snapshot_type"] == "open") & (odds["bookmaker"] != "pinnacle")].copy()

    idx = {}
    for side, col in [("over", "over_odds"), ("under", "under_odds")]:
        grp = (op[op[col].notna()]
               .groupby(["game_date","player_name","line_f"])[col]
               .apply(lambda x: mean_dec(list(x)))
               .reset_index(name="open_dec"))
        for _, row in grp.iterrows():
            d, pl, ln = row["game_date"], row["player_name"], row["line_f"]
            close_d = close_dec(d, pl, ln, side)
            if pd.notna(close_d) and close_d > 1:
                idx[(d, pl, float(ln) if pd.notna(ln) else None, side)] = round(close_d, 6)
    # Also index any close keys not covered by opens
    for (d, pl, ln) in close_dict.keys():
        for side2 in ("over", "under"):
            key = (d, pl, float(ln) if pd.notna(ln) else None, side2)
            if key not in idx:
                c = close_dec(d, pl, ln, side2)
                if pd.notna(c) and c > 1:
                    idx[key] = round(c, 6)
    return idx  # value = close_decimal (not CLV%)

CLOSE_INDEX = build_clv_index()

def get_clv(date_str, player_name, line, side, entry_american_odds):
    """CLV in cents: compares actual entry odds to closing price."""
    key = (date_str, player_name, float(line) if line is not None else None, side)
    close_d = CLOSE_INDEX.get(key)
    if close_d is None or close_d <= 1:
        return None
    entry_d = am_to_dec(entry_american_odds)
    if entry_d is None or pd.isna(entry_d) or entry_d <= 1:
        return None
    return round((entry_d / close_d - 1) * 100, 1)

def gap_label(g):
    g = abs(g)
    if g >= 1.2: return "1.2+ . 59% WR"
    if g >= 0.9: return "0.9-1.2 * . 73% WR"
    if g >= 0.6: return "0.6-0.9 . 55% WR"
    if g >= 0.3: return "0.3-0.6 ! . 44% WR"
    return "0-0.3 . 51% WR"

# -- picks per date ------------------------------------------------------------
picks_by_date = {}
available_dates = sorted(
    [p.stem.replace("daily_pitcher_props_", "") for p in EXPORTS.glob("daily_pitcher_props_*.csv")],
    reverse=True
)
for date_str in available_dates:
    path = EXPORTS / f"daily_pitcher_props_{date_str}.csv"
    df = pd.read_csv(path)
    df = df[df["market"] == "strikeouts"].copy()
    for col in ["strikeouts_projection","line","edge_pct","over_probability","under_probability","over_odds","under_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["pitcher_name"].notna()].copy()
    # normalise projection column name
    if "strikeouts_projection" not in df.columns:
        df["strikeouts_projection"] = pd.to_numeric(df.get("projection"), errors="coerce")
    for col in ["line","edge_pct","over_probability","under_probability","over_odds","under_odds","best_side"]:
        if col not in df.columns:
            df[col] = np.nan
    df["gap"] = df["strikeouts_projection"] - df["line"]
    df["gap_abs"] = df["gap"].abs()
    df["hit_prob"] = np.where(df["best_side"]=="over", df["over_probability"], df["under_probability"])
    df["model_odds"] = np.where(df["best_side"]=="over", df["over_odds"], df["under_odds"])
    df["_score"] = df.apply(lambda r2: composite_score_d(
        float(r2["edge_pct"]) if pd.notna(r2.get("edge_pct")) else 0,
        float(r2["gap_abs"]) if pd.notna(r2.get("gap_abs")) else 0), axis=1)
    df = df.sort_values("_score", ascending=False).drop_duplicates(subset=["pitcher_name"]).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        ep = float(r["edge_pct"]) if pd.notna(r.get("edge_pct")) else 0
        ga = float(r["gap_abs"]) if pd.notna(r.get("gap_abs")) else 0
        hp  = round(float(r["hit_prob"]), 4) if pd.notna(r.get("hit_prob")) else None
        mo  = int(r["model_odds"]) if pd.notna(r.get("model_odds")) else 0
        rows.append({
            "pitcher":   r["pitcher_name"],
            "side":      str(r["best_side"]) + " " + str(r["line"]),
            "proj":      round(float(r["strikeouts_projection"]),2) if pd.notna(r.get("strikeouts_projection")) else None,
            "line":      float(r["line"]) if pd.notna(r.get("line")) else None,
            "gap":       round(float(r["gap"]),2) if pd.notna(r.get("gap")) else None,
            "hitProb":   hp,
            "modelEdge": round(ep, 1),
            "modelOdds": mo,
            "bestSide":  str(r["best_side"]),
            "units":     compute_units_d(ep, ga),
            "makeOdds":  min_odds_for_ev(hp, 0.12),   # 12% EV floor (greedy ask)
            "floorOdds": min_odds_for_ev(hp, 0.15),   # 15% EV floor (calibration floor)
            "ladder":    build_ladder(hp, mo),         # price sensitivity ladder
            "lineupConf": str(r["lineup_confidence"]) if pd.notna(r.get("lineup_confidence")) else "",
        })
    picks_by_date[date_str] = rows

# -- backtest data -------------------------------------------------------------
def load_backtest(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    for col in ["strikeouts_projection","line","edge_pct","odds_used","won","actual","gap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["pitcher_name"].notna()].copy()
    df = df[df["edge_pct"] >= 15.0].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Unit-based stake: 1U=$100, 2U=$200, 2.5U=$250
    df["units_num"] = df.apply(lambda r: (
        2.5 if (float(r["edge_pct"] or 0) >= 35) else
        2.0 if (float(r["edge_pct"] or 0) >= 18 and abs(float(r["gap"] or 0)) >= 0.5) else
        1.0
    ), axis=1)
    df["stake"] = df["units_num"] * 100
    df["pnl"]   = df.apply(lambda r: pnl_calc(r["won"], r["odds_used"], stake=r["stake"]), axis=1)

    resolved = df[df["won"].notna()].copy()
    resolved["cum_pnl"] = resolved["pnl"].cumsum()
    total_wagered = resolved["stake"].sum()

    def cum_at_bump(bump):
        pnls = resolved.apply(lambda r: pnl_calc(r["won"], r["odds_used"] + bump, stake=r["stake"]), axis=1)
        return pnls.cumsum().tolist()

    dates  = [r["game_date"].strftime("%m-%d") for _,r in resolved.iterrows()]
    chart  = [{"d": d, "v": round(float(v),0)} for d,v in zip(dates, resolved["cum_pnl"])]
    chart5 = [{"d": d, "v": round(float(v),0)} for d,v in zip(dates, cum_at_bump(5))]
    chart10= [{"d": d, "v": round(float(v),0)} for d,v in zip(dates, cum_at_bump(10))]
    chart15= [{"d": d, "v": round(float(v),0)} for d,v in zip(dates, cum_at_bump(15))]

    bins   = [0, 0.3, 0.6, 0.9, 1.2, 10]
    blabels= ["0-0.3","0.3-0.6","0.6-0.9","0.9-1.2","1.2+"]
    resolved["gb"] = pd.cut(resolved["gap"].abs(), bins=bins, labels=blabels)
    gap_rows = []
    for b in blabels:
        s = resolved[resolved["gb"]==b]
        if len(s)==0: continue
        n=len(s); w=int(s["won"].sum()); p=s["pnl"].sum(); wgd=s["stake"].sum()
        gap_rows.append({"bucket":b,"n":n,"w":w,"l":n-w,"wr":round(s["won"].mean()*100,1),"pnl":round(p,0),"roi":round(p/wgd*100,1)})

    ebins   = [15,20,100]
    elabels = ["15-20%","20%+"]
    resolved["eb"] = pd.cut(resolved["edge_pct"], bins=ebins, labels=elabels)
    edge_rows = []
    for b in elabels:
        s = resolved[resolved["eb"]==b]
        if len(s)==0: continue
        n=len(s); w=int(s["won"].sum()); p=s["pnl"].sum(); wgd=s["stake"].sum()
        edge_rows.append({"bucket":b,"n":n,"w":w,"l":n-w,"wr":round(s["won"].mean()*100,1),"pnl":round(p,0),"roi":round(p/wgd*100,1)})

    log = []
    for _,r in resolved.sort_values("game_date",ascending=False).iterrows():
        ep  = float(r["edge_pct"]) if pd.notna(r["edge_pct"]) else 0
        ga  = abs(float(r["gap"])) if pd.notna(r["gap"]) else 0
        ds  = r["game_date"].strftime("%Y-%m-%d")
        ln  = float(r["line"]) if pd.notna(r["line"]) else None
        sd  = str(r["best_side"])
        entry_am = float(r["odds_used"]) if pd.notna(r.get("odds_used")) else None
        clv = get_clv(ds, r["pitcher_name"], ln, sd, entry_am) if entry_am is not None else None
        close_key = (ds, r["pitcher_name"], ln, sd)
        close_d = CLOSE_INDEX.get(close_key)
        close_am = dec_to_am(close_d) if close_d else None
        log.append({
            "date": ds,
            "pitcher": r["pitcher_name"],
            "side": sd + " " + str(r["line"]),
            "proj": round(float(r["strikeouts_projection"]),2) if pd.notna(r["strikeouts_projection"]) else None,
            "gap": round(float(r["gap"]),2) if pd.notna(r["gap"]) else None,
            "edge": round(ep, 1),
            "units": compute_units_d(ep, ga),
            "stake": int(r["stake"]),
            "openOdds": int(entry_am) if entry_am is not None else None,
            "closeOdds": int(close_am) if close_am is not None else None,
            "clv": clv,
            "actual": int(r["actual"]) if pd.notna(r["actual"]) else None,
            "won": bool(r["won"]==1),
            "pnl": round(float(r["pnl"]),0) if pd.notna(r["pnl"]) else None,
        })

    total=len(resolved); wins=int(resolved["won"].sum()); total_pnl=resolved["pnl"].sum()

    def scenario_stats(bump):
        p = resolved.apply(lambda r: pnl_calc(r["won"], r["odds_used"] + bump, stake=r["stake"]), axis=1).sum()
        return {"pnl": round(p, 0), "roi": round(p / total_wagered * 100, 1)}

    scenarios = {
        "actual": {"pnl": round(total_pnl, 0), "roi": round(total_pnl / total_wagered * 100, 1)},
        "+5c":    scenario_stats(5),
        "+10c":   scenario_stats(10),
        "+15c":   scenario_stats(15),
    }

    return {
        "total": total, "wins": wins, "losses": total-wins,
        "wr": round(resolved["won"].mean()*100,1),
        "pnl": round(total_pnl,0), "roi": round(total_pnl / total_wagered * 100, 1),
        "range": f"{resolved['game_date'].min().strftime('%b %d')} - {resolved['game_date'].max().strftime('%b %d, %Y')}",
        "chart": chart, "chart5": chart5, "chart10": chart10, "chart15": chart15,
        "scenarios": scenarios,
        "gapRows": gap_rows, "edgeRows": edge_rows, "log": log,
        "clvPosRate": round(sum(1 for r in log if r.get("clv") is not None and r["clv"] >= 0) / max(sum(1 for r in log if r.get("clv") is not None), 1) * 100, 1),
        "avgClv": round(float(np.mean([r["clv"] for r in log if r.get("clv") is not None])), 2) if any(r.get("clv") is not None for r in log) else None,
    }

bt_2026 = load_backtest(EXPORTS / "2026_backtest_extended.csv")
bt_2025 = load_backtest(EXPORTS / "2025_backtest.csv")
bt_all  = {"2026": bt_2026, "2025": bt_2025}
bt_data = bt_2026  # default for backward-compat print at end

# -- build HTML ----------------------------------------------------------------
today_picks = picks_by_date.get(available_dates[0], []) if available_dates else []
ticker_items = [p for p in today_picks if p.get("modelEdge") and p["modelEdge"] >= 15]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVB . Pitcher K Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --black:#000;--panel:#0a0c0a;--panel2:#101410;
  --silver:#d8dde0;--dim:#8a9097;
  --green:#2fd44a;--green2:#16a82f;--glow:rgba(47,212,74,.35);
  --red:#e0483a;--line:rgba(255,255,255,.07);
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--black);color:var(--silver);font-family:'Inter',sans-serif;overflow-x:hidden;line-height:1.6}
::selection{background:var(--green);color:#000}
a{color:inherit;text-decoration:none}
input{font-family:'Inter',sans-serif}

/* ticker */
.ticker{position:sticky;top:0;z-index:100;background:#050705;border-bottom:1px solid var(--line);overflow:hidden;height:34px;display:flex;align-items:center;font-family:'Oswald',sans-serif;font-size:13px;letter-spacing:.5px}
.ticker-track{display:inline-flex;white-space:nowrap;animation:scroll 50s linear infinite}
.ticker-track span{padding:0 26px;color:var(--dim)}
.ticker-track .up{color:var(--green)}
.ticker-track .dn{color:var(--red)}
@keyframes scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}

/* nav */
nav{background:rgba(0,0,0,.7);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:14px 32px;display:flex;align-items:center;justify-content:space-between}
.brand{font-family:'Oswald',sans-serif;font-weight:700;font-size:20px;letter-spacing:1px;color:var(--silver)}
.brand .v{color:var(--green)}
.brand small{font-size:12px;font-weight:400;letter-spacing:3px;color:var(--dim);margin-left:10px}

/* tabs */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--line);padding:0 32px;background:rgba(0,0,0,.4)}
.tab{font-family:'Oswald',sans-serif;font-weight:500;letter-spacing:1px;font-size:14px;padding:14px 22px;color:var(--dim);cursor:pointer;border-bottom:2px solid transparent;transition:color .15s,border-color .15s}
.tab.active{color:var(--green);border-bottom-color:var(--green)}
.tab-content{display:none;padding:32px}
.tab-content.active{display:block}

/* controls row */
.controls{display:flex;align-items:center;gap:16px;margin-bottom:28px;flex-wrap:wrap}
.date-sel{font-family:'Oswald',sans-serif;font-size:14px;font-weight:500;background:var(--panel2);border:1px solid var(--line);color:var(--silver);padding:9px 16px;border-radius:9px;cursor:pointer;appearance:none;-webkit-appearance:none;letter-spacing:.5px}
.date-sel:focus{outline:none;border-color:rgba(47,212,74,.5)}
.toggle{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim);cursor:pointer;user-select:none}
.toggle input{width:14px;height:14px;accent-color:var(--green);cursor:pointer}
.sort-btn{font-family:'Oswald',sans-serif;font-size:12px;letter-spacing:1px;padding:7px 14px;border:1px solid var(--line);background:transparent;color:var(--dim);border-radius:7px;cursor:pointer;transition:color .15s,border-color .15s}
.sort-btn.active{color:var(--green);border-color:rgba(47,212,74,.5)}

/* KPI row */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:16px;overflow:hidden;margin-bottom:28px}
.kpi{background:var(--panel);padding:22px 20px}
.kpi .v{font-family:'Oswald',sans-serif;font-weight:700;font-size:clamp(22px,3vw,32px);color:var(--green);line-height:1}
.kpi .v.s{color:var(--silver)}
.kpi .k{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--dim);margin-top:7px}

/* picks table */
.tbl-wrap{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;overflow:hidden}
.tbl-head{padding:20px 24px 14px;border-bottom:1px solid var(--line);font-family:'Oswald',sans-serif;font-weight:600;font-size:17px;color:var(--silver);letter-spacing:.5px}
.tbl-head small{font-family:'Inter',sans-serif;font-size:12px;font-weight:400;color:var(--dim);margin-left:10px;letter-spacing:0}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--dim);font-weight:600;padding:12px 20px;border-bottom:1px solid var(--line)}
th.r{text-align:right}
td{padding:14px 20px;border-bottom:1px solid var(--line);font-size:14px;vertical-align:middle}
td.r{text-align:right}
tr:last-child td{border-bottom:none}
tr{transition:background .12s}
tr:hover td{background:rgba(255,255,255,.02)}
tr.low-gap td{background:rgba(224,72,58,.07)}
tr.low-gap:hover td{background:rgba(224,72,58,.12)}
td.pitcher{color:var(--silver);font-weight:500;font-family:'Oswald',sans-serif;letter-spacing:.3px}
td.side{color:var(--dim);font-size:13px}
td.proj{font-family:'Oswald',sans-serif;font-weight:600;color:var(--silver)}
td.gap-pos{color:var(--green);font-family:'Oswald',sans-serif;font-weight:600}
td.gap-neg{color:#7fe6a0;font-family:'Oswald',sans-serif;font-weight:600}
td.prob{color:var(--dim);font-size:13px}
td.medge{font-family:'Oswald',sans-serif;font-weight:600}
td.your-odds-cell{min-width:100px}
.odds-inp{width:90px;background:rgba(47,212,74,.06);border:1px solid rgba(47,212,74,.25);color:var(--green);font-family:'Oswald',sans-serif;font-weight:600;font-size:15px;padding:7px 10px;border-radius:8px;text-align:center;transition:border-color .15s}
.odds-inp:focus{outline:none;border-color:var(--green);background:rgba(47,212,74,.1)}
td.your-edge{font-family:'Oswald',sans-serif;font-weight:700;font-size:15px;text-align:right}
.gap-badge{display:inline-block;font-size:11px;padding:3px 9px;border-radius:20px;letter-spacing:.3px;white-space:nowrap}
.gb-star{background:rgba(47,212,74,.15);color:var(--green);border:1px solid rgba(47,212,74,.3)}
.gb-good{background:rgba(47,212,74,.08);color:#7fe6a0;border:1px solid rgba(47,212,74,.15)}
.gb-warn{background:rgba(224,72,58,.08);color:#e0a070;border:1px solid rgba(224,72,58,.2)}
.gb-ok{background:rgba(255,255,255,.04);color:var(--dim);border:1px solid var(--line)}

/* edge colors */
.e-hi{color:#00c853}.e-good{color:var(--green)}.e-ok{color:#fff176}.e-low{color:#ffb74d}.e-neg{color:var(--red)}.e-none{color:var(--dim)}
/* edge bar */
.ebar-wrap{display:inline-block;width:54px;height:5px;background:rgba(255,255,255,.1);border-radius:3px;vertical-align:middle;margin-right:8px;flex-shrink:0}
.ebar{height:100%;border-radius:3px}

/* flagged section */
.flagged{margin-top:28px}
.flagged-title{font-family:'Oswald',sans-serif;font-weight:600;font-size:18px;color:var(--silver);margin-bottom:16px;letter-spacing:.5px}
.play-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:10px;transition:border-color .15s}
.play-card:hover{border-color:rgba(47,212,74,.35)}
.play-card .pname{font-family:'Oswald',sans-serif;font-weight:600;font-size:17px;color:var(--silver)}
.play-card .pside{font-size:13px;color:var(--dim);margin-top:2px}
.play-card .pedge{font-family:'Oswald',sans-serif;font-weight:700;font-size:22px}
.play-card .pdetail{font-size:12px;color:var(--dim);margin-top:2px}

/* backtest */
.bt-kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:16px;overflow:hidden;margin-bottom:28px}
.section-label{font-family:'Oswald',sans-serif;letter-spacing:4px;text-transform:uppercase;font-size:12px;color:var(--green);margin-bottom:12px}
.h2{font-family:'Oswald',sans-serif;font-weight:600;font-size:clamp(22px,3vw,32px);color:var(--silver);margin-bottom:6px;letter-spacing:.5px}
.sub{color:var(--dim);font-size:15px;margin-bottom:24px}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;padding:24px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:28px}

/* chart */
.chart-wrap{background:rgba(0,0,0,.3);border:1px solid var(--line);border-radius:14px;padding:14px 10px 6px;margin-bottom:6px;position:relative}
.kochart{position:relative;width:100%;height:280px;user-select:none;-webkit-user-select:none}
.kochart svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.kochart .grid-line{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke}
.kochart .area{fill:rgba(47,212,74,.1)}
.kochart .line{fill:none;stroke:var(--green);stroke-width:2.5;vector-effect:non-scaling-stroke;stroke-linejoin:round}
.kochart .ax{position:absolute;inset:0;pointer-events:none;font-size:11px;color:var(--dim)}
.kochart .yl{position:absolute;left:4px;transform:translateY(-50%);white-space:nowrap}
.kochart .xl{position:absolute;bottom:0;transform:translateX(-50%);white-space:nowrap}
.kochart .vl{position:absolute;width:0;border-left:1px dashed var(--green);opacity:0;pointer-events:none;transition:opacity .1s}
.kochart .dot{position:absolute;width:10px;height:10px;border-radius:50%;transform:translate(-50%,-50%);opacity:0;pointer-events:none;background:var(--green);border:2px solid var(--panel)}
.kochart .tip{position:absolute;transform:translate(-50%,-110%);background:#0e140f;border:1px solid rgba(47,212,74,.4);border-radius:9px;padding:9px 14px;white-space:nowrap;opacity:0;pointer-events:none;z-index:5;box-shadow:0 8px 22px rgba(0,0,0,.5);min-width:130px}
.kochart .tip-dt{font-size:10px;letter-spacing:.5px;color:var(--dim);text-transform:uppercase;margin-bottom:4px}
.kochart .tip-v{font-family:'Oswald',sans-serif;font-size:16px;font-weight:700;color:var(--green)}

/* scenario toggle */
.scen-btns{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;padding:0 2px}
.scen-btn{font-family:'Oswald',sans-serif;font-size:12px;letter-spacing:.8px;padding:7px 16px;border-radius:8px;cursor:pointer;border:1px solid var(--line);background:transparent;color:var(--dim);transition:all .15s;display:flex;align-items:center;gap:7px}
.scen-btn .dot2{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.scen-btn.active{border-color:currentColor}
.scen-btn.active.s-actual{color:#2fd44a;background:rgba(47,212,74,.1)}
.scen-btn.active.s-5{color:#34d399;background:rgba(52,211,153,.1)}
.scen-btn.active.s-10{color:#60a5fa;background:rgba(96,165,250,.1)}
.scen-btn.active.s-15{color:#a78bfa;background:rgba(167,139,250,.1)}

/* bucket tables */
.btable{width:100%;border-collapse:collapse}
.btable th{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--dim);font-weight:600;padding:10px 16px;border-bottom:1px solid var(--line);text-align:left}
.btable th.r{text-align:right}
.btable td{padding:12px 16px;border-bottom:1px solid var(--line);font-size:14px}
.btable tr:last-child td{border-bottom:none}
.btable td.bname{font-family:'Oswald',sans-serif;color:var(--silver)}
.btable td.r{text-align:right;font-family:'Oswald',sans-serif;font-weight:600}
.btable td.pos{color:var(--green)}
.btable td.neg2{color:var(--red)}

/* log */
.log-wrap{margin-top:28px;overflow-x:auto}
.log-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:10px}
.log-filter{background:var(--panel2);border:1px solid var(--line);color:var(--silver);font-family:'Inter',sans-serif;font-size:13px;padding:7px 12px;border-radius:8px}
.log-filter:focus{outline:none;border-color:rgba(47,212,74,.4)}
.log-table{width:100%;border-collapse:collapse;font-size:13px}
.log-table th{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--dim);font-weight:600;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}
.log-table th.r{text-align:right}
.log-table td{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
.log-table td.r{text-align:right}
.log-table tr:last-child td{border-bottom:none}
.log-table tr:hover td{background:rgba(255,255,255,.02)}
.badge-win{display:inline-block;background:rgba(47,212,74,.15);color:var(--green);border:1px solid rgba(47,212,74,.3);font-family:'Oswald',sans-serif;font-weight:600;font-size:12px;padding:2px 10px;border-radius:20px;letter-spacing:.5px}
.badge-loss{display:inline-block;background:rgba(224,72,58,.12);color:var(--red);border:1px solid rgba(224,72,58,.25);font-family:'Oswald',sans-serif;font-weight:600;font-size:12px;padding:2px 10px;border-radius:20px;letter-spacing:.5px}
.badge-clv-pos{display:inline-block;background:rgba(47,212,74,.12);color:var(--green);border:1px solid rgba(47,212,74,.3);font-family:'Oswald',sans-serif;font-weight:700;font-size:12px;padding:2px 9px;border-radius:6px;letter-spacing:.3px}
.badge-clv-neg{display:inline-block;background:rgba(224,72,58,.12);color:var(--red);border:1px solid rgba(224,72,58,.25);font-family:'Oswald',sans-serif;font-weight:700;font-size:12px;padding:2px 9px;border-radius:6px;letter-spacing:.3px}
.badge-clv-null{display:inline-block;color:var(--dim);font-size:12px;padding:2px 4px}
.ubadge{display:inline-block;font-family:'Oswald',sans-serif;font-size:11px;font-weight:700;padding:2px 9px;border-radius:6px;letter-spacing:.5px}
.ub-25{background:rgba(255,243,205,.18);color:#c8960c;border:1px solid rgba(255,215,0,.35)}
.ub-2{background:rgba(47,212,74,.12);color:var(--green);border:1px solid rgba(47,212,74,.3)}
.ub-1{background:rgba(255,255,255,.04);color:var(--dim);border:1px solid var(--line)}
.ldr-cell{position:relative;cursor:pointer}
.ldr-badge{display:inline-block;font-family:'Oswald',sans-serif;font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;letter-spacing:.3px}
.ldr-pop{display:none;position:absolute;right:0;top:110%;z-index:50;background:#0e140f;border:1px solid rgba(47,212,74,.3);border-radius:10px;padding:10px 14px;white-space:nowrap;min-width:190px;box-shadow:0 8px 24px rgba(0,0,0,.6);font-size:12px}
.ldr-pop table{border-collapse:collapse;width:100%}
.ldr-pop td{padding:3px 8px;font-family:'Oswald',sans-serif;font-size:13px}
.ldr-cell:hover .ldr-pop,.ldr-cell:focus-within .ldr-pop{display:block}

@media(max-width:900px){
  .kpis,.bt-kpis{grid-template-columns:repeat(2,1fr)}
  .two-col{grid-template-columns:1fr}
  .tab-content{padding:20px 16px}
  nav{padding:12px 16px}
  .tabs{padding:0 16px}
  th,td{padding:10px 12px}
}
</style>
</head>
<body>

<div class="ticker"><div class="ticker-track" id="tick"></div></div>

<nav>
  <div class="brand">S<span class="v">V</span>B<small>PITCHER K MODEL</small></div>
  <div style="font-size:13px;color:var(--dim)" id="nav-date"></div>
</nav>

<div class="tabs">
  <div class="tab active" data-tab="picks">Daily Picks</div>
  <div class="tab" data-tab="backtest">Backtest</div>
</div>

<!-- TAB 1: DAILY PICKS --------------------------------------------------- -->
<div class="tab-content active" id="tab-picks">
  <div class="controls">
    <select class="date-sel" id="date-sel"></select>
    <label class="toggle"><input type="checkbox" id="dir-filter" checked> Direction-agreement only</label>
    <label class="toggle"><input type="checkbox" id="edge-filter"> 15%+ edge only</label>
    <button class="sort-btn active" id="sort-edge">Sort: Edge</button>
    <button class="sort-btn" id="sort-gap">Sort: Gap</button>
  </div>
  <div class="kpis" id="picks-kpis"></div>
  <div class="tbl-wrap">
    <div class="tbl-head">Strikeout Projections <small> Edit "Your Odds" -- edge recalculates live</small></div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Pitcher</th><th>Side</th><th class="r">Line</th><th class="r">Proj</th><th class="r">Gap</th>
          <th class="r">P(hit)</th><th class="r" style="min-width:160px">Model Edge</th>
          <th style="text-align:center">Units</th>
          <th style="text-align:center">Your Odds</th><th class="r">Your Edge</th>
          <th class="r">Make</th><th class="r">Floor (15%)</th><th class="r">Worst Line</th>
          <th>Gap Signal</th><th style="text-align:center">Lineup</th>
        </tr></thead>
        <tbody id="picks-body"></tbody>
      </table>
    </div>
  </div>
  <div class="flagged" id="flagged-section"></div>
</div>

<!-- TAB 2: BACKTEST ------------------------------------------------------- -->
<div class="tab-content" id="tab-backtest">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;flex-wrap:wrap">
    <div class="section-label" style="margin-bottom:0">Backtest</div>
    <div style="display:flex;gap:0;border:1px solid var(--line);border-radius:9px;overflow:hidden" id="yr-sel">
      <div class="yr-btn active" data-yr="2026" style="font-family:'Oswald',sans-serif;font-size:13px;letter-spacing:1px;padding:6px 16px;cursor:pointer;background:rgba(47,212,74,.15);color:var(--green);border-right:1px solid var(--line)">2026</div>
      <div class="yr-btn" data-yr="2025" style="font-family:'Oswald',sans-serif;font-size:13px;letter-spacing:1px;padding:6px 16px;cursor:pointer;color:var(--dim)">2025</div>
    </div>
  </div>
  <div class="h2" id="bt-title">Loading...</div>
  <p class="sub">15%+ edge . one bet per pitcher per day . fixed unit sizing (1U/2U/2.5U)</p>
  <div class="bt-kpis" id="bt-kpis"></div>

  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:10px">
      <span style="font-family:'Oswald',sans-serif;font-weight:600;font-size:17px;color:var(--silver);letter-spacing:.5px">Cumulative P&L</span>
      <span style="font-size:12px;color:var(--dim)">$100 flat stake per bet</span>
    </div>
    <div class="chart-wrap">
      <div class="kochart" id="bt-chart">
        <svg preserveAspectRatio="none" viewBox="0 0 1000 300"></svg>
        <div class="ax"></div>
        <div class="vl"></div>
        <div class="dot"></div>
        <div class="tip"><div class="tip-dt"></div><div class="tip-v"></div></div>
      </div>
    </div>
    <div class="scen-btns" id="scen-btns">
      <button class="scen-btn s-actual active" data-scen="actual"><span class="dot2" style="background:#2fd44a"></span>Actual</button>
      <button class="scen-btn s-5"  data-scen="+5c" ><span class="dot2" style="background:#34d399"></span>+5c odds</button>
      <button class="scen-btn s-10" data-scen="+10c"><span class="dot2" style="background:#60a5fa"></span>+10c odds</button>
      <button class="scen-btn s-15" data-scen="+15c"><span class="dot2" style="background:#a78bfa"></span>+15c odds</button>
    </div>
  </div>

  <div class="two-col">
    <div class="card" id="gap-tbl-wrap">
      <div style="font-family:'Oswald',sans-serif;font-weight:600;font-size:17px;color:var(--silver);margin-bottom:18px;letter-spacing:.5px">By Projection Gap</div>
      <table class="btable" id="gap-tbl"></table>
    </div>
    <div class="card" id="edge-tbl-wrap">
      <div style="font-family:'Oswald',sans-serif;font-weight:600;font-size:17px;color:var(--silver);margin-bottom:18px;letter-spacing:.5px">By Edge Bucket</div>
      <table class="btable" id="edge-tbl"></table>
    </div>
  </div>

  <div class="log-wrap">
    <div class="log-head">
      <span style="font-family:'Oswald',sans-serif;font-weight:600;font-size:17px;color:var(--silver);letter-spacing:.5px">Full Pick Log</span>
      <select class="log-filter" id="log-filter">
        <option value="all">All results</option>
        <option value="win">Wins only</option>
        <option value="loss">Losses only</option>
      </select>
    </div>
    <div class="card" style="padding:0;overflow:hidden">
      <div style="overflow-x:auto;max-height:520px;overflow-y:auto">
        <table class="log-table"><thead id="log-thead"></thead><tbody id="log-body"></tbody></table>
      </div>
    </div>
  </div>
</div>

<script>
/* -- embedded data -- */
var PICKS_BY_DATE = """ + json.dumps(picks_by_date, ensure_ascii=True) + """;
var AVAILABLE_DATES = """ + json.dumps(available_dates, ensure_ascii=True) + """;
var BT_ALL = """ + json.dumps(bt_all, ensure_ascii=True) + """;
var BT = BT_ALL['2026'] || BT_ALL['2025'] || {};

/* -- ticker -- */
(function(){
  var today = AVAILABLE_DATES[0];
  var items = (PICKS_BY_DATE[today]||[]).filter(function(p){return p.modelEdge && p.modelEdge>=15;});
  if(!items.length) items = PICKS_BY_DATE[today]||[];
  var html = items.map(function(p){
    var e = p.modelEdge ? (p.modelEdge>0?'+':'')+p.modelEdge.toFixed(1)+'%':'--';
    var cls = p.modelEdge&&p.modelEdge>0?'up':'dn';
    return '<span>'+p.pitcher+' . '+p.side+' <span class="'+cls+'">'+e+'</span></span>';
  }).join('');
  if(!html) html = '<span>No picks loaded</span>';
  var row = html+html;
  document.getElementById('tick').innerHTML = row;
  var d = new Date();
  document.getElementById('nav-date').textContent = d.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'});
})();

/* -- tab switching -- */
document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click',function(){
    document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('active');});
    document.querySelectorAll('.tab-content').forEach(function(x){x.classList.remove('active');});
    t.classList.add('active');
    document.getElementById('tab-'+t.dataset.tab).classList.add('active');
    if(t.dataset.tab==='backtest') renderBacktest();
  });
});

/* -- picks tab -- */
var today = new Date().toISOString().slice(0,10);
var currentDate = AVAILABLE_DATES.indexOf(today) >= 0 ? today : (AVAILABLE_DATES.find(function(d){ return (PICKS_BY_DATE[d]||[]).some(function(p){ return p.modelEdge; }); }) || AVAILABLE_DATES[0] || '');
var sortByGap = false;

// populate date selector
(function(){
  var sel = document.getElementById('date-sel');
  AVAILABLE_DATES.forEach(function(d){
    var opt = document.createElement('option'); opt.value=d; opt.textContent=d; sel.appendChild(opt);
  });
  sel.value = currentDate;
  sel.addEventListener('change',function(){currentDate=this.value;renderPicks();});
})();

document.getElementById('dir-filter').addEventListener('change',renderPicks);
document.getElementById('edge-filter').addEventListener('change',renderPicks);
document.getElementById('sort-edge').addEventListener('click',function(){sortByGap=false;document.getElementById('sort-edge').classList.add('active');document.getElementById('sort-gap').classList.remove('active');renderPicks();});
document.getElementById('sort-gap').addEventListener('click',function(){sortByGap=true;document.getElementById('sort-gap').classList.add('active');document.getElementById('sort-edge').classList.remove('active');renderPicks();});

function calcEdge(prob, odds){
  if(!prob||odds===0||odds===null||odds===undefined||isNaN(odds)) return null;
  var dec = odds>0 ? 1+odds/100 : 1+100/Math.abs(odds);
  return (prob*(dec-1)-(1-prob))*100;
}
function edgeCls(e){
  if(e===null||e===undefined||isNaN(e)) return 'e-none';
  if(e>=15) return 'e-hi'; if(e>=10) return 'e-good'; if(e>=7) return 'e-ok'; if(e>=0) return 'e-low'; return 'e-neg';
}
function fmtEdge(e){return (e===null||isNaN(e))?'--':(e>=0?'+':'')+e.toFixed(1)+'%';}
function gapCls(g){
  var a=Math.abs(g);
  if(a>=0.9&&a<1.2) return 'gb-star';
  if(a>=0.6) return 'gb-good';
  if(a>=0.3) return 'gb-warn';
  return 'gb-ok';
}
function gapLabel(g){
  var a=Math.abs(g);
  if(a>=1.2) return '1.2+ . 59%';
  if(a>=0.9) return '* 0.9-1.2 . 73%';
  if(a>=0.6) return '0.6-0.9 . 55%';
  if(a>=0.3) return '! 0.3-0.6 . 44%';
  return '0-0.3 . 51%';
}

function fmtAm(o){if(o===null||o===undefined) return '--'; return (o>0?'+':'')+o;}
function fmtMake(r){
  var o=r.makeOdds; if(o===null||o===undefined) return '<td class="r" style="color:var(--dim)">--</td>';
  var clr=o>0?'var(--green)':'#ffb74d';
  return '<td class="r" style="color:'+clr+';font-family:Oswald,sans-serif;font-weight:700" title="Min odds for 12% EV">'+fmtAm(o)+'</td>';
}
function fmtFloor(r){
  var o=r.floorOdds; if(o===null||o===undefined) return '<td class="r" style="color:var(--dim)">--</td>';
  var clr='#e0483a';
  return '<td class="r" style="color:'+clr+';font-family:Oswald,sans-serif;font-weight:700">'+fmtAm(o)+'</td>';
}
function fmtLadder(r){
  var base=r.modelOdds; if(base===null||base===undefined) return '<td class="r" style="color:var(--dim)">--</td>';
  var worst=base-20;
  var worstStr=fmtAm(worst);
  // build popup: current + upside shopping scenarios (+5c to +25c better)
  var ld=r.ladder||[];
  var rows=ld.map(function(l){
    var eClr=l.edge>=15?'var(--green)':l.edge>=7?'#ffb74d':'var(--red)';
    var lbl=l.shift===0?'Scraped':'+'+l.shift+'c better';
    return '<tr><td style="color:var(--dim)">'+lbl+'</td>'+
      '<td style="color:var(--silver)">'+fmtAm(l.odds)+'</td>'+
      '<td style="color:'+eClr+';font-weight:700">'+(l.edge!==null?(l.edge>=0?'+':'')+l.edge+'%':'--')+'</td></tr>';
  }).join('');
  // worst row at bottom
  var wEdge=r.hitProb?calcEdge(r.hitProb,worst):null;
  var wClr=wEdge!==null&&wEdge>=15?'var(--green)':wEdge!==null&&wEdge>=7?'#ffb74d':'var(--red)';
  rows+='<tr style="border-top:1px solid rgba(255,255,255,.1)"><td style="color:#e0483a">Worst (-20c)</td>'+
    '<td style="color:#e0483a">'+worstStr+'</td>'+
    '<td style="color:'+wClr+';font-weight:700">'+(wEdge!==null?(wEdge>=0?'+':'')+wEdge.toFixed(1)+'%':'--')+'</td></tr>';
  return '<td class="r ldr-cell">'+
    '<span class="ldr-badge" style="color:#e0483a;font-family:Oswald,sans-serif;font-weight:700;font-size:12px">'+worstStr+' worst</span>'+
    '<div class="ldr-pop"><div style="font-size:10px;letter-spacing:1px;color:var(--dim);margin-bottom:8px;text-transform:uppercase">Price Ladder</div>'+
    '<table><thead><tr>'+
      '<th style="color:var(--dim);font-size:10px;letter-spacing:1px">Scenario</th>'+
      '<th style="color:var(--dim);font-size:10px;letter-spacing:1px">Odds</th>'+
      '<th style="color:var(--dim);font-size:10px;letter-spacing:1px">Edge</th></tr></thead>'+
      '<tbody>'+rows+'</tbody></table></div></td>';
}

function fmtLineupConf(r){
  var c = r.lineupConf||'';
  if(!c) return '<td style="text-align:center;color:var(--dim)">--</td>';
  var bg,clr;
  if(c==='Confirmed'){bg='rgba(47,212,74,.18)';clr='#2fd44a';}
  else if(c==='High'){bg='rgba(52,211,153,.15)';clr='#34d399';}
  else if(c==='Medium'){bg='rgba(251,191,36,.15)';clr='#fbbf24';}
  else if(c==='Low'){bg='rgba(224,72,58,.15)';clr='#e0483a';}
  else{bg='rgba(160,160,160,.12)';clr='#888';}
  return '<td style="text-align:center"><span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;letter-spacing:.5px;background:'+bg+';color:'+clr+'">'+c+'</span></td>';
}

function renderPicks(){
  var rows = (PICKS_BY_DATE[currentDate]||[]).slice();
  var dirFilter = document.getElementById('dir-filter').checked;
  var edgeFilter = document.getElementById('edge-filter').checked;
  if(dirFilter) rows = rows.filter(function(r){
    if(!r.gap) return false;
    return (r.bestSide==='over'&&r.gap>0)||(r.bestSide==='under'&&r.gap<0);
  });
  if(edgeFilter) rows = rows.filter(function(r){return r.modelEdge&&r.modelEdge>=15;});
  if(sortByGap) rows.sort(function(a,b){return Math.abs(b.gap||0)-Math.abs(a.gap||0);});
  else rows.sort(function(a,b){return (b.modelEdge||0)-(a.modelEdge||0);});

  // KPIs
  var flagged = rows.filter(function(r){return r.modelEdge&&r.modelEdge>=15;});
  var bestE = rows.reduce(function(m,r){return r.modelEdge>m?r.modelEdge:m;},0);
  var bigGap = rows.reduce(function(m,r){return Math.abs(r.gap||0)>m?Math.abs(r.gap||0):m;},0);
  var kpis = [
    {v:rows.length,k:'Pitchers'},
    {v:flagged.length,k:'15%+ Edge Picks'},
    {v:bestE?'+'+bestE.toFixed(1)+'%':'--',k:'Best Edge',cls:''},
    {v:bigGap?bigGap.toFixed(2)+'K':'--',k:'Biggest Gap',cls:'s'}
  ];
  document.getElementById('picks-kpis').innerHTML = kpis.map(function(k){
    return '<div class="kpi"><div class="v '+(k.cls||'')+'">'+k.v+'</div><div class="k">'+k.k+'</div></div>';
  }).join('');

  // Table
  var tbody = document.getElementById('picks-body');
  tbody.innerHTML = '';
  rows.forEach(function(r,i){
    var tr = document.createElement('tr');
    var gapStr = r.gap?(r.gap>0?'+':'')+r.gap.toFixed(2):'--';
    var gapTdCls = r.gap?(r.gap>0?'gap-pos':'gap-neg'):'';
    var me = r.modelEdge;
    var meCls = edgeCls(me);
    var meStr = fmtEdge(me);
    var prob = r.hitProb? (r.hitProb*100).toFixed(1)+'%':'--';
    var initOdds = r.modelOdds||0;
    var initEdge = calcEdge(r.hitProb, initOdds);
    var yeCls = edgeCls(initEdge);
    var yeStr = fmtEdge(initEdge);
    var gbCls = r.gap!==null?gapCls(r.gap):'gb-ok';
    var gbLbl = r.gap!==null?gapLabel(r.gap):'--';
    var barW = me!==null ? Math.min(Math.max(me,0)/25*100,100).toFixed(0) : 0;
    var barClr = me>=15?'#00c853':me>=7?'#2fd44a':me>=0?'#ffb74d':'#e0483a';
    var lineStr = r.line!==null&&r.line!==undefined ? r.line : '--';
    var uCls = r.units==='2.5U'?'ub-25':r.units==='2U'?'ub-2':'ub-1';
    var uLbl = r.units||'1U';
    tr.innerHTML =
      '<td class="pitcher">'+r.pitcher+'</td>'+
      '<td class="side">'+r.bestSide+'</td>'+
      '<td class="proj" style="text-align:right">'+lineStr+'</td>'+
      '<td class="proj" style="text-align:right">'+( r.proj?r.proj.toFixed(2):'--')+'</td>'+
      '<td class="'+gapTdCls+'" style="text-align:right">'+gapStr+'</td>'+
      '<td class="prob" style="text-align:right">'+prob+'</td>'+
      '<td class="medge '+meCls+'" style="text-align:right;white-space:nowrap">'+
        '<span class="ebar-wrap"><span class="ebar" style="width:'+barW+'%;background:'+barClr+'"></span></span>'+meStr+'</td>'+
      '<td style="text-align:center"><span class="ubadge '+uCls+'">'+uLbl+'</span></td>'+
      '<td class="your-odds-cell" style="text-align:center"><input class="odds-inp" type="number" value="'+initOdds+'" data-prob="'+(r.hitProb||0)+'" step="5"></td>'+
      '<td class="your-edge '+yeCls+'" data-ye>'+yeStr+'</td>'+
      fmtMake(r)+
      fmtFloor(r)+
      fmtLadder(r)+
      '<td><span class="gap-badge '+gbCls+'">'+gbLbl+'</span></td>'+
      fmtLineupConf(r);
    if(r.gap!==null&&r.gap!==undefined&&Math.abs(r.gap)<0.6) tr.classList.add('low-gap');
    tbody.appendChild(tr);
    tr.querySelector('.odds-inp').addEventListener('input', function(){
      var odds = parseFloat(this.value)||0;
      var prob2 = parseFloat(this.dataset.prob)||0;
      var edge = calcEdge(prob2, odds);
      var td = this.closest('tr').querySelector('[data-ye]');
      td.textContent = fmtEdge(edge);
      td.className = 'your-edge '+edgeCls(edge);
      updateFlagged();
    });
  });

  updateFlagged();
}

function updateFlagged(){
  var rows = document.querySelectorAll('#picks-body tr');
  var qualifying = [];
  rows.forEach(function(tr){
    var inp = tr.querySelector('.odds-inp');
    var yeTd = tr.querySelector('[data-ye]');
    if(!inp||!yeTd) return;
    var odds = parseFloat(inp.value)||0;
    var prob = parseFloat(inp.dataset.prob)||0;
    var edge = calcEdge(prob, odds);
    if(edge!==null && edge>=15){
      var ubEl = tr.querySelector('.ubadge');
      qualifying.push({
        pitcher: tr.querySelector('.pitcher').textContent,
        side: tr.querySelector('.side').textContent,
        proj: tr.querySelector('.proj').textContent,
        gap: tr.querySelector('[class*="gap-"]').textContent,
        edge: edge,
        odds: odds,
        gapBadge: tr.querySelector('.gap-badge').textContent,
        gbCls: tr.querySelector('.gap-badge').className.replace('gap-badge ',''),
        units: ubEl ? ubEl.textContent : '1U',
        uCls: ubEl ? ubEl.className.replace('ubadge ','') : 'ub-1',
      });
    }
  });
  qualifying.sort(function(a,b){return b.edge-a.edge;});
  var sec = document.getElementById('flagged-section');
  if(!qualifying.length){sec.innerHTML='';return;}
  var cards = qualifying.map(function(p){
    var eCls = edgeCls(p.edge);
    var oddsStr = (p.odds>=0?'+':'')+Math.round(p.odds);
    return '<div class="play-card">'+
      '<div><div class="pname">'+p.pitcher+' <span class="ubadge '+(p.uCls||'ub-1')+'" style="font-size:10px;vertical-align:middle;margin-left:6px">'+(p.units||'1U')+'</span></div>'+
      '<div class="pside">'+p.side+' . proj '+p.proj+' . gap '+p.gap+'</div></div>'+
      '<div style="text-align:right"><div class="pedge '+eCls+'">'+fmtEdge(p.edge)+'</div><div class="pdetail">at '+oddsStr+' . <span class="gap-badge '+p.gbCls+'" style="font-size:11px">'+p.gapBadge+'</span></div></div>'+
      '</div>';
  }).join('');
  sec.innerHTML = '<div class="flagged-title">'+qualifying.length+' play'+(qualifying.length>1?'s':'')+' at 15%+ edge (your odds)</div>'+cards;
}

renderPicks();

/* -- year selector -- */
var activeYear = '2026';
document.querySelectorAll('.yr-btn').forEach(function(btn){
  btn.addEventListener('click', function(){
    activeYear = this.dataset.yr;
    BT = BT_ALL[activeYear] || {};
    document.querySelectorAll('.yr-btn').forEach(function(b){
      var on = b.dataset.yr === activeYear;
      b.style.background = on ? 'rgba(47,212,74,.15)' : 'transparent';
      b.style.color = on ? 'var(--green)' : 'var(--dim)';
    });
    btRendered = false;
    renderBacktest();
  });
});

/* -- backtest tab -- */
var btRendered = false;
function renderBacktest(){
  if(btRendered) return;
  btRendered = true;
  document.getElementById('bt-kpis').innerHTML='';
  document.getElementById('gap-tbl').innerHTML='';
  document.getElementById('edge-tbl').innerHTML='';
  document.getElementById('log-body').innerHTML='';
  document.getElementById('log-thead').innerHTML='';
  var chartEl=document.getElementById('bt-chart');
  chartEl.querySelector('svg').innerHTML='';chartEl.querySelector('.ax').innerHTML='';
  if(!BT||!BT.total){document.getElementById('bt-title').textContent='No '+activeYear+' backtest data found.';return;}
  document.getElementById('bt-title').textContent = BT.range+' . $100 flat stake';

  // KPIs
  var pnlCls = BT.pnl>=0?'':'style="color:var(--red)"';
  var clvPosRate = BT.clvPosRate!=null ? BT.clvPosRate : null;
  var avgClv     = BT.avgClv!=null ? BT.avgClv : null;
  var clvV = clvPosRate!=null ? clvPosRate+'% CLV+' : '--';
  var clvSub = avgClv!=null ? 'avg '+(avgClv>=0?'+':'')+avgClv+'c' : '';
  var kpis=[
    {v:BT.total,k:'Total Bets',cls:'s'},
    {v:BT.wins+'W / '+BT.losses+'L',k:'Record',cls:'s'},
    {v:BT.wr+'%',k:'Win Rate'},
    {v:(BT.pnl>=0?'+$':'-$')+Math.abs(BT.pnl).toLocaleString(),k:'Total P&L',st:BT.pnl<0?'color:var(--red)':'',id:'kpi-pnl'},
    {v:(BT.roi>=0?'+':'')+BT.roi+'%',k:'ROI',st:BT.roi<0?'color:var(--red)':'',id:'kpi-roi'},
    {v:clvV,k:'CLV Rate'+(clvSub?' . '+clvSub:''),st:clvPosRate!=null&&clvPosRate>=50?'':'color:var(--red)'},
  ];
  document.getElementById('bt-kpis').innerHTML = kpis.map(function(k){
    return '<div class="kpi"><div class="v '+(k.cls||'')+'" style="'+(k.st||'')+'"'+(k.id?' id="'+k.id+'"':'')+'>'+k.v+'</div><div class="k">'+k.k+'</div></div>';
  }).join('');

  // Chart
  buildChart();

  // Gap table
  var gtbl = document.getElementById('gap-tbl');
  gtbl.innerHTML = '<thead><tr><th>Gap</th><th>Bets</th><th class="r">Win%</th><th class="r">P&L</th><th class="r">ROI</th></tr></thead>';
  var gtbody = document.createElement('tbody');
  (BT.gapRows||[]).forEach(function(r){
    var pos=r.roi>=0; var tr=document.createElement('tr');
    tr.innerHTML='<td class="bname">'+r.bucket+'</td>'+
      '<td style="color:var(--dim)">'+r.n+'</td>'+
      '<td class="r" style="color:var(--silver)">'+r.wr+'%</td>'+
      '<td class="r '+(pos?'pos':'neg2')+'">'+(pos?'+$':'-$')+Math.abs(r.pnl).toLocaleString()+'</td>'+
      '<td class="r '+(pos?'pos':'neg2')+'">'+(r.roi>=0?'+':'')+r.roi+'%</td>';
    gtbody.appendChild(tr);
  });
  gtbl.appendChild(gtbody);

  // Edge table
  var etbl = document.getElementById('edge-tbl');
  etbl.innerHTML = '<thead><tr><th>Edge</th><th>Bets</th><th class="r">Win%</th><th class="r">P&L</th><th class="r">ROI</th></tr></thead>';
  var etbody = document.createElement('tbody');
  (BT.edgeRows||[]).forEach(function(r){
    var pos=r.roi>=0; var tr=document.createElement('tr');
    tr.innerHTML='<td class="bname">'+r.bucket+'</td>'+
      '<td style="color:var(--dim)">'+r.n+'</td>'+
      '<td class="r" style="color:var(--silver)">'+r.wr+'%</td>'+
      '<td class="r '+(pos?'pos':'neg2')+'">'+(pos?'+$':'-$')+Math.abs(r.pnl).toLocaleString()+'</td>'+
      '<td class="r '+(pos?'pos':'neg2')+'">'+(r.roi>=0?'+':'')+r.roi+'%</td>';
    etbody.appendChild(tr);
  });
  etbl.appendChild(etbody);

  // Log
  renderLog('all');
  document.getElementById('log-filter').addEventListener('change',function(){renderLog(this.value);});
}

function renderLog(filter){
  var data = (BT.log||[]).filter(function(r){
    if(filter==='win') return r.won;
    if(filter==='loss') return !r.won;
    return true;
  });
  document.getElementById('log-thead').innerHTML='<tr><th>Date</th><th>Pitcher</th><th>Side</th><th class="r">Proj</th><th class="r">Gap</th><th class="r">Edge</th><th style="text-align:center">Units</th><th class="r">Stake</th><th class="r">Open</th><th class="r">Close</th><th class="r" style="min-width:80px">CLV %</th><th class="r">Actual</th><th>Result</th><th class="r">P&L</th></tr>';
  var tbody = document.getElementById('log-body');
  tbody.innerHTML='';
  data.forEach(function(r){
    var tr=document.createElement('tr');
    var pnlStr = r.pnl!==null?(r.pnl>=0?'<span style="color:var(--green)">+$'+r.pnl.toLocaleString()+'</span>':'<span style="color:var(--red)">-$'+Math.abs(r.pnl).toLocaleString()+'</span>'):'--';
    var uCls = r.units==='2.5U'?'ub-25':r.units==='2U'?'ub-2':'ub-1';
    var clvStr = r.clv!==null&&r.clv!==undefined
      ? '<span class="badge-clv-'+(r.clv>=0?'pos':'neg')+'">'+(r.clv>=0?'+':'')+r.clv.toFixed(1)+'c</span>'
      : '<span class="badge-clv-null">--</span>';
    tr.innerHTML='<td style="color:var(--dim)">'+r.date+'</td>'+
      '<td style="font-weight:500;color:var(--silver)">'+r.pitcher+'</td>'+
      '<td style="color:var(--dim)">'+r.side+'</td>'+
      '<td class="r" style="color:var(--silver)">'+( r.proj?r.proj.toFixed(2):'--')+'</td>'+
      '<td class="r" style="color:var(--dim)">'+( r.gap!==null?(r.gap>0?'+':'')+r.gap.toFixed(2):'--')+'</td>'+
      '<td class="r" style="color:var(--green)">'+( r.edge?'+'+r.edge+'%':'--')+'</td>'+
      '<td style="text-align:center"><span class="ubadge '+uCls+'">'+(r.units||'1U')+'</span></td>'+
      '<td class="r" style="color:var(--dim)">$'+(r.stake||100)+'</td>'+
      '<td class="r" style="color:var(--silver);font-family:Oswald,sans-serif;font-weight:600">'+( r.openOdds!==null&&r.openOdds!==undefined?(r.openOdds>0?'+':'')+r.openOdds:'--')+'</td>'+
      '<td class="r" style="color:var(--dim)">'+( r.closeOdds!==null&&r.closeOdds!==undefined?(r.closeOdds>0?'+':'')+r.closeOdds:'--')+'</td>'+
      '<td class="r">'+clvStr+'</td>'+
      '<td class="r" style="color:var(--silver)">'+( r.actual!==null?r.actual:'--')+'</td>'+
      '<td>'+(r.won?'<span class="badge-win">WIN</span>':'<span class="badge-loss">LOSS</span>')+'</td>'+
      '<td class="r">'+pnlStr+'</td>';
    tbody.appendChild(tr);
  });
}

function buildChart(){
  var pts = BT.chart||[];
  if(!pts.length) return;
  var el=document.getElementById('bt-chart');
  var svg=el.querySelector('svg'),ax=el.querySelector('.ax');
  var tip=el.querySelector('.tip'),vl=el.querySelector('.vl'),dot=el.querySelector('.dot');
  var W=1000,H=300,pL=68,pR=18,pT=16,pB=34;
  var pw=W-pL-pR,ph=H-pT-pB;
  var bumps=[
    {key:'chart15',color:'#a78bfa',label:'+15c',dash:'6,3'},
    {key:'chart10',color:'#60a5fa',label:'+10c',dash:'6,3'},
    {key:'chart5', color:'#34d399',label:'+5c', dash:'6,3'},
    {key:'chart',  color:'#2fd44a',label:'Actual',dash:null},
  ];
  // compute y range across all series
  var allVals=[0];
  bumps.forEach(function(b){(BT[b.key]||[]).forEach(function(p){allVals.push(p.v);});});
  var vmin=Math.min.apply(null,allVals);
  var vmax=Math.max.apply(null,allVals);
  var rng=vmax-vmin||1; vmin-=rng*0.08; vmax+=rng*0.08; rng=vmax-vmin;
  function X(i){return pL+(i/(pts.length-1))*pw;}
  function Y(v){return pT+(1-(v-vmin)/rng)*ph;}
  var NS='http://www.w3.org/2000/svg';
  svg.setAttribute('viewBox','0 0 '+W+' '+H);
  // grid lines
  var gstep = vmax>20000?5000:vmax>10000?5000:vmax>5000?2500:1000;
  var gstart = Math.ceil(vmin/gstep)*gstep;
  for(var gv=gstart;gv<=vmax;gv+=gstep){
    var ln=document.createElementNS(NS,'line');
    ln.setAttribute('x1',pL);ln.setAttribute('x2',W-pR);ln.setAttribute('y1',Y(gv));ln.setAttribute('y2',Y(gv));
    ln.setAttribute('class','grid-line');svg.appendChild(ln);
    var yl=document.createElement('div');yl.className='yl';
    yl.style.top=(Y(gv)/H*100)+'%';
    yl.textContent=gv===0?'$0':(gv>0?'+':'')+( Math.abs(gv)>=1000?(gv/1000).toFixed(0)+'K':gv);
    ax.appendChild(yl);
  }
  // x labels
  var step=Math.ceil(pts.length/6);
  pts.forEach(function(p,i){
    if(i%step!==0&&i!==pts.length-1) return;
    var xl=document.createElement('div');xl.className='xl';
    xl.style.left=(X(i)/W*100)+'%';
    var mm=p.d.split('-');
    var mn={'03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep'};
    xl.textContent=(mn[mm[0]]||mm[0])+' '+parseInt(mm[1]);
    ax.appendChild(xl);
  });
  // draw bump lines (behind actual)
  var lineEls={};
  bumps.forEach(function(b){
    var bpts=BT[b.key]||[];
    if(!bpts.length||b.key==='chart') return;
    var lp='M '+X(0)+' '+Y(bpts[0].v);
    bpts.forEach(function(p,i){if(i>0) lp+=' L '+X(i)+' '+Y(p.v);});
    var bl=document.createElementNS(NS,'path');
    bl.setAttribute('d',lp);bl.setAttribute('fill','none');
    bl.setAttribute('stroke',b.color);bl.setAttribute('stroke-width','1.5');
    bl.setAttribute('stroke-dasharray',b.dash);bl.setAttribute('opacity','0.55');
    bl.setAttribute('vector-effect','non-scaling-stroke');bl.setAttribute('stroke-linejoin','round');
    bl.id='btl-'+b.key;
    svg.appendChild(bl);
    lineEls[b.key]=bl;
  });
  // area fill + actual line
  var aPath='M '+X(0)+' '+Y(Math.max(vmin,0));
  pts.forEach(function(p,i){aPath+=' L '+X(i)+' '+Y(p.v);});
  aPath+=' L '+X(pts.length-1)+' '+Y(Math.max(vmin,0))+' Z';
  var aEl=document.createElementNS(NS,'path');aEl.setAttribute('d',aPath);aEl.setAttribute('class','area');aEl.id='btl-area';svg.appendChild(aEl);
  var lPath='M '+X(0)+' '+Y(pts[0].v);
  pts.forEach(function(p,i){if(i>0) lPath+=' L '+X(i)+' '+Y(p.v);});
  var lEl=document.createElementNS(NS,'path');lEl.setAttribute('d',lPath);lEl.setAttribute('class','line');lEl.id='btl-chart';svg.appendChild(lEl);
  lineEls['chart']=lEl;
  // interaction -- tooltip shows all series values
  function showTip(i){
    var p=pts[i],x=X(i),px=x/W*100,py=Y(p.v)/H*100;
    dot.style.left=px+'%';dot.style.top=py+'%';dot.style.opacity=1;
    vl.style.left=px+'%';vl.style.top=(pT/H*100)+'%';vl.style.height=((H-pB-pT)/H*100)+'%';vl.style.opacity=.5;
    tip.style.left=px+'%';tip.style.top=py+'%';tip.style.opacity=1;
    var mm=p.d.split('-');var mn={'03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep'};
    tip.querySelector('.tip-dt').textContent=(mn[mm[0]]||mm[0])+' '+parseInt(mm[1]);
    var lines=[['Actual',p.v,'#2fd44a']];
    if(BT.chart5&&BT.chart5[i])  lines.push(['+5c', BT.chart5[i].v,'#34d399']);
    if(BT.chart10&&BT.chart10[i]) lines.push(['+10c',BT.chart10[i].v,'#60a5fa']);
    if(BT.chart15&&BT.chart15[i]) lines.push(['+15c',BT.chart15[i].v,'#a78bfa']);
    tip.querySelector('.tip-v').innerHTML=lines.map(function(ln){
      return '<span style="color:'+ln[2]+';display:block;font-size:13px">'
        +ln[0]+': '+(ln[1]>=0?'+$':'-$')+Math.abs(ln[1]).toLocaleString()+'</span>';
    }).join('');
    tip.querySelector('.tip-v').style.color='';
  }
  function hideTip(){dot.style.opacity=0;vl.style.opacity=0;tip.style.opacity=0;}
  function near(cx){
    var r=el.getBoundingClientRect(),rel=(cx-r.left)/r.width*W,best=0,bd=1e9;
    pts.forEach(function(p,i){var d=Math.abs(X(i)-rel);if(d<bd){bd=d;best=i;}});
    return best;
  }
  el.addEventListener('mousemove',function(e){showTip(near(e.clientX));});
  el.addEventListener('mouseleave',hideTip);
  el.addEventListener('touchstart',function(e){showTip(near(e.touches[0].clientX));},{passive:true});
  el.addEventListener('touchmove',function(e){showTip(near(e.touches[0].clientX));},{passive:true});
  el.addEventListener('touchend',hideTip);
  showTip(pts.length-1);setTimeout(hideTip,1600);

  // scenario button logic
  var scenMap={'actual':'chart','+5c':'chart5','+10c':'chart10','+15c':'chart15'};
  var scenColors={'actual':'#2fd44a','+5c':'#34d399','+10c':'#60a5fa','+15c':'#a78bfa'};
  var activeScen='actual';
  function applyScen(scen){
    activeScen=scen;
    var activeKey=scenMap[scen];
    // dim/show lines
    Object.keys(scenMap).forEach(function(s){
      var key=scenMap[s];
      var el2=lineEls[key];
      if(!el2) return;
      if(key==='chart'){
        el2.style.opacity=scen==='actual'?'1':'0.18';
        var ar=document.getElementById('btl-area');
        if(ar) ar.style.opacity=scen==='actual'?'1':'0.18';
      } else {
        el2.style.opacity=s===scen?'1':'0.18';
        el2.setAttribute('stroke-width',s===scen?'2.5':'1.5');
      }
    });
    // update KPI P&L and ROI cells
    var sc=BT.scenarios&&BT.scenarios[scen];
    if(sc){
      var pnlEl=document.getElementById('kpi-pnl');
      var roiEl=document.getElementById('kpi-roi');
      var clr=scenColors[scen];
      if(pnlEl){pnlEl.style.color=clr;pnlEl.textContent=(sc.pnl>=0?'+$':'-$')+Math.abs(sc.pnl).toLocaleString()+(scen!=='actual'?' ('+scen+')':'');}
      if(roiEl){roiEl.style.color=clr;roiEl.textContent=(sc.roi>=0?'+':'')+sc.roi+'%'+(scen!=='actual'?' ('+scen+')':'');}
    }
    // highlight active button
    document.querySelectorAll('.scen-btn').forEach(function(b){
      b.classList.toggle('active',b.dataset.scen===scen);
    });
  }
  document.querySelectorAll('.scen-btn').forEach(function(btn){
    btn.addEventListener('click',function(){applyScen(this.dataset.scen);});
  });
}
</script>
</body>
</html>"""

import shutil

out = Path("dashboard.html")
out.write_text(HTML, encoding="utf-8")

site_copy = Path("site/picks.html")
site_copy.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(out, site_copy)

print(f"Generated {out} and {site_copy}")
print(f"Available dates: {', '.join(available_dates[:5])}{'...' if len(available_dates)>5 else ''}")
if bt_data:
    print(f"Backtest: {bt_data.get('total',0)} bets, {bt_data.get('wins',0)}W/{bt_data.get('losses',0)}L, {bt_data.get('roi',0):+.1f}% ROI")
