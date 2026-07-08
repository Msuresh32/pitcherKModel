# Final Model June 19-30 Backfill

Window requested: 2026-06-19 through 2026-06-30
Model: `poisson_top120_a2` trained on 2025 pitcher-games only
Rule: `under` only, devig edge >= 7%, edge-gap >= 0

## Result

- Bets: 70
- Wins/Losses/Pushes: 37/33/0
- Win rate: 52.86%
- ROI: -1.65%
- Units: -1.15
- Profit at $100 flat stake: $-115.34
- Average edge: 22.44%
- Average abs gap: 0.78 Ks

## CLV

- CLV matched bets: 50/70
- Average CLV: -1.88%
- Median CLV: -1.49%
- Positive CLV rate: 44.00%

## Prediction Accuracy

- MAE on matched starter-games: 1.966
- RMSE on matched starter-games: 2.476
- Scored starter-games with odds: 201

## Data Coverage

- Open rows fetched: 2466
- Close rows fetched: 2547
- Open best-line rows matched to starter features: 276

## Daily Breakdown

| Date | Bets | W-L-P | ROI | Units | Avg CLV | CLV+ |
|---|---:|---:|---:|---:|---:|---:|
| 2026-06-19 00:00:00 | 3 | 2-1-0 | +35.30% | +1.06 | -0.62% | 66.67% |
| 2026-06-20 00:00:00 | 7 | 4-3-0 | +7.35% | +0.51 | +10.04% | 80.00% |
| 2026-06-21 00:00:00 | 7 | 3-4-0 | -22.15% | -1.55 | -10.95% | 0.00% |
| 2026-06-22 00:00:00 | 6 | 2-4-0 | -39.37% | -2.36 | -5.41% | 33.33% |
| 2026-06-23 00:00:00 | 5 | 3-2-0 | +18.43% | +0.92 | +1.70% | 50.00% |
| 2026-06-24 00:00:00 | 10 | 4-6-0 | -31.15% | -3.11 | +1.90% | 42.86% |
| 2026-06-25 00:00:00 | 6 | 3-3-0 | -3.99% | -0.24 | +0.83% | 80.00% |
| 2026-06-26 00:00:00 | 5 | 3-2-0 | +13.11% | +0.66 | -11.69% | 0.00% |
| 2026-06-27 00:00:00 | 5 | 2-3-0 | -17.33% | -0.87 | +17.32% | 100.00% |
| 2026-06-28 00:00:00 | 10 | 7-3-0 | +31.60% | +3.16 | -7.79% | 16.67% |
| 2026-06-29 00:00:00 | 6 | 4-2-0 | +11.16% | +0.67 | -9.10% | 40.00% |

## Caveats

- The 06/19-06/30 raw feature cache was not present in full when this backfill began.
- Advanced/Statcast/lineup features were carried forward from the last available no-leak research-matrix profile; pitcher rolling K/IP/K9 features were refreshed from newly fetched actual starter logs.
- CLV uses best available entry odds at roughly 4 hours before first pitch versus best available close odds roughly 3 minutes before first pitch.
- Two individual close snapshots returned expired-event 404s from The Odds API; affected bets remain in ROI but can have missing CLV if no same line close was matched.
## Additional CLV Diagnostics

- CLV t-stat versus zero: -0.99.
- Positive-CLV bets: 22 bets, 40.91% win rate, -20.01% ROI, -4.40 units.
- Non-positive-CLV bets: 28 bets, 57.14% win rate, +2.31% ROI, +0.65 units.
- Missing-CLV bets: 20 bets, 60.00% win rate, +13.02% ROI, +2.60 units.

## June 30 Status

The Odds API returned a 2026-06-30 prop slate, but those odds rows did not map to the settled starter logs fetched from MLB Stats API. Name-only scoring found 12 ungraded candidate unders, saved to `final_model_unsettled_2026-06-30_candidates.csv`. They are excluded from ROI and CLV settlement because the slate could not be reconciled safely to actual results.
