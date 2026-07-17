"""Outcome-blind qualitative validation for HITS-ALLOWED paper props.

Mirrors the K validator's rules (research/v2/a7_qual_brief.py /
qual_validation_prompt.md) with hits-specific evidence:
  - opponent ball-in-play profile (hit rate, K rate -> fewer BIP)
  - pitcher contact quality allowed (xBA-on-contact, hard-hit, from
    statcast_pitcher_contact_daily.csv)
  - team defense behind the pitcher (rolling hits-per-BIP, prior-season OAA)
  - workload/leash (short starts, pitch counts -> fewer innings)
  - market thinness + the WATCH-ZONE rule (edge >10pp was anti-signal in the
    2026 walk-forward: mandatory concern, never a support)

All inputs strictly pregame (< score date). Never changes signals or stakes.
Output: reports/daily/v2_hits_qual_<date>.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "reports" / "daily"
RAW = ROOT / "data" / "raw"
NO_CONFIRMATION = "No strong qualitative confirmation found beyond the model edge."


def build(day: str):
    asof = pd.Timestamp(day)
    hits_f = DAILY / f"v2_hits_paper_{day}.csv"
    if not hits_f.exists():
        print(f"no hits paper file for {day}")
        return
    plays = pd.read_csv(hits_f)
    if not len(plays):
        print("no hits plays")
        return

    logs = pd.read_csv(RAW / "pitcher_game_logs.csv", low_memory=False)
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs[logs.game_date < asof].drop_duplicates(["game_pk", "pitcher_id"])
    team = pd.read_csv(RAW / "team_batting_game_logs.csv", low_memory=False)
    team["game_date"] = pd.to_datetime(team["game_date"])
    team = team[team.game_date < asof]
    contact = pd.read_csv(RAW / "statcast_pitcher_contact_daily.csv", low_memory=False)
    contact["game_date"] = pd.to_datetime(contact["game_date"])
    contact = contact[contact.game_date < asof]
    probs = pd.read_csv(RAW / "probable_pitchers.csv")
    probs = probs[probs.game_date.astype(str).str[:10] == day]
    oaa_f = RAW / "team_oaa_by_season.csv"
    oaa = pd.read_csv(oaa_f) if oaa_f.exists() else pd.DataFrame()

    name2id = (logs.drop_duplicates("pitcher_name")
               .set_index("pitcher_name").pitcher_id.to_dict())
    for c in ["hits_allowed", "strikeouts", "walks", "batters_faced",
              "innings_pitched", "pitches"]:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    logs["bip"] = (logs.batters_faced - logs.strikeouts - logs.walks).clip(lower=0)

    cutoff30 = asof - pd.Timedelta(days=30)
    lg30 = team[team.game_date >= cutoff30]
    lg_hit = lg30.hits.sum() / lg30.plate_appearances.sum()
    lg_k = lg30.strikeouts.sum() / lg30.plate_appearances.sum()
    lg_xba = contact[contact.game_date >= cutoff30].xba_con_allowed.mean()

    out = []
    for _, p in plays.iterrows():
        pid = name2id.get(p.pitcher)
        support, concern = [], []
        prow = probs[probs.pitcher_name == p.pitcher]

        # opponent ball-in-play profile
        if len(prow):
            opp = str(prow.opponent.iloc[0])
            t30 = team[(team.team.astype(str) == opp) & (team.game_date >= cutoff30)]
            if t30.plate_appearances.sum() >= 300:
                ohit = t30.hits.sum() / t30.plate_appearances.sum()
                ok = t30.strikeouts.sum() / t30.plate_appearances.sum()
                if ohit < lg_hit - 0.008:
                    support.append(f"Opponent hit rate L30 {ohit:.1%} vs league "
                                   f"{lg_hit:.1%} — a below-average contact offense.")
                elif ohit > lg_hit + 0.008:
                    concern.append(f"Opponent hits at {ohit:.1%} L30 vs league "
                                   f"{lg_hit:.1%} — above-average contact offense.")
                if ok > lg_k + 0.015:
                    support.append(f"Opponent K rate {ok:.1%} L30 (league {lg_k:.1%}) "
                                   f"— strikeouts remove balls in play.")

        if pid is not None:
            g = logs[logs.pitcher_id == pid].sort_values("game_date").tail(10)
            if len(g) >= 5:
                short = int((g.innings_pitched < 5).sum())
                avg_pit = float(g.pitches.mean())
                if short >= 4 or avg_pit < 85:
                    support.append(f"Short leash: {short}/{len(g)} starts under 5 IP, "
                                   f"{avg_pit:.0f} pitches/start — fewer innings cap hits.")
                elif short == 0 and avg_pit > 95:
                    concern.append(f"Deep leash ({avg_pit:.0f} pitches/start, no short "
                                   f"starts in {len(g)}) — more innings, more hit exposure.")
                hpb = g.hits_allowed.sum() / max(g.bip.sum(), 1)
                if hpb > 0.36:
                    concern.append(f"Hits per ball-in-play {hpb:.3f} over his last "
                                   f"{len(g)} starts — contact has been falling in.")
            c10 = contact[contact.pitcher_id == pid].sort_values("game_date").tail(10)
            if len(c10) >= 4 and np.isfinite(lg_xba):
                xba = float(c10.xba_con_allowed.mean())
                if xba < lg_xba - 0.015:
                    support.append(f"Weak contact allowed: xBA-on-contact "
                                   f"{xba:.3f} vs league {lg_xba:.3f} (L10 starts).")
                elif xba > lg_xba + 0.015:
                    concern.append(f"Loud contact allowed: xBA-on-contact "
                                   f"{xba:.3f} vs league {lg_xba:.3f} (L10 starts).")
            # defense behind him
            if len(prow) and len(oaa):
                yr = asof.year - 1
                mine = oaa[(oaa.season == yr) &
                           (oaa.team_id.astype(str) == str(prow.team.iloc[0]))]
                if len(mine):
                    v = float(mine.outs_above_average.iloc[0])
                    if v >= 15:
                        support.append(f"Strong defense behind him: team OAA {v:+.0f} "
                                       f"in {yr}.")
                    elif v <= -15:
                        concern.append(f"Poor defense behind him: team OAA {v:+.0f} "
                                       f"in {yr}.")

        # market + watch zone (mandatory rules)
        if p.n_books <= 1:
            concern.append("Single-book market — thin consensus reference.")
        if p.edge > 0.10:
            concern.append("WATCH ZONE: edge >10pp — the model's largest hits edges "
                           "were anti-signal in the 2026 walk-forward (-9.9% to "
                           "-18.9% ROI despite best-in-book CLV). Extra skepticism required.")

        score = int(np.clip(50 + 12 * len(support) - 12 * len(concern), 0, 100))
        if len(support) >= 2 and len(concern) <= 1:
            verdict = "Moderate qualitative confirmation"
        elif len(support) > len(concern):
            verdict = "Moderate qualitative confirmation" if len(support) >= 2 \
                else "Neutral (model-only edge)"
        elif len(concern) > len(support) + 1:
            verdict = "Qualitative concerns"
        elif concern and support:
            verdict = "Mixed signals"
        else:
            verdict = "Neutral (model-only edge)"
        flag = "SUPPORT" if verdict.startswith(("Strong", "Moderate")) else "NO-SUPPORT"

        html = (f"<h4>Model summary</h4>Projected {p.proj_hits} hits vs line "
                f"{p.line} · edge {p.edge*100:.1f}pp under · {int(p.n_books)} book(s) "
                f"· PAPER (hits system is paper-only pending live fill validation).")
        html += "<h4>Qualitative confirmation</h4>"
        html += ("<ul>" + "".join(f"<li>{s}</li>" for s in support) + "</ul>"
                 if support else f"<em>{NO_CONFIRMATION}</em>")
        html += "<h4>Contradictory evidence</h4>"
        html += ("<ul>" + "".join(f"<li>{c}</li>" for c in concern) + "</ul>"
                 if concern else "<em>None material.</em>")
        html += f'<h4>Verdict</h4><span class="verdict">{verdict}</span>'
        out.append({"pitcher": p.pitcher, "line": float(p.line),
                    "verdict": verdict, "qual_flag": flag,
                    "qual_score": score, "html": html})

    (DAILY / f"v2_hits_qual_{day}.json").write_text(
        json.dumps({"date": day, "plays": out}, indent=1), encoding="utf-8")
    print(f"hits qual: {len(out)} analyses "
          f"({sum(1 for o in out if o['qual_flag']=='SUPPORT')} SUPPORT)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=os.environ.get(
        "SCORE_DATE", pd.Timestamp.now().strftime("%Y-%m-%d")))
    a = ap.parse_args()
    build(a.date)
