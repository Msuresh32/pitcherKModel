# Side-of-Market Analysis — the T−12 Under Cohort

Registered 2026-07-18 · script: `70_side_analysis.py` · data: `side_analysis.csv`

## Finding

The under side outperforms the over side in **all six** saved ledgers (2025 and
2026, T−4 and T−12, raw and LCB rules), with higher CLV in all six. One cut is
CI-positive in both seasons:

| T−12 LCB unders | n | ROI | 90% date-block CI | CLV | win rate | median odds |
|---|---:|---:|---|---:|---:|---:|
| 2025 | 79 | +29.1% | [+12.2%, +45.5%] | +2.99pp | 68.4% | −115 |
| 2026 | 57 | +22.3% | [+5.5%, +38.4%] | +1.28pp | 64.9% | −124 |

Robustness: profit is breadth-driven (top-5 winners ≈ 1/3 of total), spread
across lines 4.5–7.5 and across months (2025 positive in 5/7 months; 2026
April-heavy — a caveat). Structural prior declared with the hypothesis:
recreational money concentrates on overs, so books shade over prices; the
model's under signals therefore face softer prices. Consistent with the
over/under CLV gap in every ledger.

## Status: paper cohort, not a live rule

The side split was inspected after 2026 outcomes were known — this is
selection-era evidence with small n, and it would be exactly the V3.1 mistake
to deploy it on these numbers. It is therefore a **tracked cohort of the
existing daily card** (no routing change; the cohort is simply the UNDER-side
plays at the t12 route) with a pre-registered gate in `deploy_config.json`:

- **Promote** (small live stakes): ≥80 settled fresh T−12 under plays AND mean
  no-vig CLV ≥ +0.8pp AND date-block bootstrap ROI lower bound > 0.
- **Kill**: mean CLV < 0 after 40 settled fresh plays.
- Fresh window starts 2026-07-19; expected accrual 1–2 plays/day (~8–12 weeks).

If promoted: shrunk half-Kelly (λ=0.25, 2% cap) on ~1–2 bets/day; at the
backtest ROI range that is roughly +2 to +5u/month on flat 1u — modest volume,
but it would be the project's first live-qualified K rule.

## Why nothing was promoted today

Any config "found deployable" by re-searching 2022–Jul 10 2026 would repeat the
V3.1 selection error: enough searches always produce a positive number on data
already seen. Deployability can only come from (a) a frozen cohort passing its
gate on write-once forward data, or (b) execution prices that are +EV under the
market's own closing probabilities (≥ +8c vs the fair close, measured on real
fills). Both tracks are running daily.
