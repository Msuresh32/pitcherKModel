"""Rebuild the artifact HTML with the latest daily plays (+ per-play
qualitative analyses when present) spliced into the Today tab.

STDLIB-ONLY (csv/json/re) so it runs both locally and inside the cloud
routine's environment without dependencies.

Inputs (all committed to the repo):
  reports/artifact/template.html
  reports/artifact/report_data.json
  reports/daily/v2_plays_<date>.csv , v2_props_<date>.csv ,
  v2_hits_paper_<date>.csv (optional),
  reports/daily/v2_qual_analysis_<date>.json (optional:
      {"plays": [{"pitcher": ..., "line": ..., "html": "<h4>..."}]})

Output: reports/artifact/backtest_report.html
"""
from __future__ import annotations
import csv, datetime, json, re
from pathlib import Path

ART = Path("reports/artifact")
DAILY = Path("reports/daily")

NUMERIC = {"line", "price", "proj_ks", "edge_over", "n_books", "stake_pct",
           "stake_usd", "stake_usd_conservative", "p_model_over", "p_mkt_over",
           "p_model_under", "p_mkt_under", "edge", "proj_hits", "stake_usd"}


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rec = {}
        for k, v in r.items():
            if v is None or v == "":
                rec[k] = None
            elif k in NUMERIC:
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            else:
                rec[k] = v
        out.append(rec)
    return out


def main():
    dates = sorted(re.findall(r"v2_plays_(\d{4}-\d{2}-\d{2})\.csv",
                              " ".join(p.name for p in DAILY.glob("v2_plays_*.csv"))))
    if not dates:
        raise SystemExit("no daily plays files found")
    day = dates[-1]

    plays = read_csv_rows(DAILY / f"v2_plays_{day}.csv")
    props = read_csv_rows(DAILY / f"v2_props_{day}.csv")
    hits_f = DAILY / f"v2_hits_paper_{day}.csv"
    hits = read_csv_rows(hits_f) if hits_f.exists() else []

    # attach qualitative analyses if the file exists
    qual_f = DAILY / f"v2_qual_analysis_{day}.json"
    if qual_f.exists():
        qual = json.loads(qual_f.read_text(encoding="utf-8")).get("plays", [])
        qmap = {(q.get("pitcher"), float(q.get("line", 0))): q for q in qual}
        for p in plays:
            key = (p.get("pitcher"), float(p.get("line") or 0))
            if key in qmap:
                p["qual"] = qmap[key].get("html", "")
                p["qual_flag"] = qmap[key].get("qual_flag")
        print(f"qual analyses attached: "
              f"{sum(1 for p in plays if p.get('qual'))}/{len(plays)}")

    prop_keep = ["pitcher", "line", "proj_ks", "p_model_over", "p_mkt_over",
                 "edge_over", "n_books", "signal", "tier"]
    props_out = [{k: r.get(k) for k in prop_keep} for r in props
                 if r.get("line") not in (None, "")]

    today = {
        "date": day,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "plays": plays,
        "hits_paper": hits,
        "props": props_out,
    }

    data = json.loads((ART / "report_data.json").read_text(encoding="utf-8"))
    data["today"] = today
    payload = json.dumps(data, allow_nan=False)
    (ART / "report_data.json").write_text(payload, encoding="utf-8")

    tpl = (ART / "template.html").read_text(encoding="utf-8")
    marker = "/*__DATA__*/"
    if marker in tpl:
        html = tpl.replace(marker, payload)
    else:
        raise SystemExit("template missing data marker")
    (ART / "backtest_report.html").write_text(html, encoding="utf-8")
    print(f"built reports/artifact/backtest_report.html for {day} "
          f"({len(plays)} plays, {len(hits)} hits-paper)")


if __name__ == "__main__":
    main()
