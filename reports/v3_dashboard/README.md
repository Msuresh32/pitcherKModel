# K//EDGE V3 Dashboard

Open `index.html` in any modern browser. It is self-contained and does not
need a web server or internet connection.

Views include:

- automatically refreshed morning V3.1 card with exact price, book, routing
  edge, conviction, qualitative verdict, and settlement state
- frozen-forward daily equity, card-by-card grading, qualitative-result splits,
  and automation/notification health
- H0/H1 overview and T-4h/T-12h timing toggle
- 2025 and 2026 equity, monthly results, and stability breakdowns
- fixed edge-band ROI, CLV, confidence intervals, and chronological windows
- outcome-blind conviction grades, corroboration diagnostics, and paper gate
- V3.1 H3 convex ensemble weights and one-SD uncertainty-adjusted edge results
- interactive T-12h price simulator from 5-25 American-odds cents better;
  the slider reprices equity, monthly ROI, 2026 detail, breakdowns, drawdown,
  P&L, and execution-adjusted CLV throughout Overview and Performance
- calibration curves and stable feature importance
- searchable 2026 Bet Explorer combining frozen research rows through July 10
  with every archived post-cutoff paper card, including source, side, band,
  result, conviction, and pending-status filters
- methodology and the active paper-only challenger specification

Rebuild after research artifacts change:

```powershell
python research\v3\40_build_dashboard.py
```

The successful `V2PlaysMorning` Windows task rebuilds and opens this dashboard
at 8:45 AM ET. Optional ntfy phone delivery is implemented but remains disabled
until a destination is explicitly configured and approved. Every delivery is
generated from the same archived `v2_plays_<date>.csv` file displayed here.

The dashboard intentionally retains the paper-only warning. It does not enable
or recommend real-money staking.
