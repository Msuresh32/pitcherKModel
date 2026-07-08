# Adversarial Strikeout Model Review

Date: 2026-06-29

## Executive Verdict

No existing model artifact in this repository survives professional deployment scrutiny.

The reason is not weak ROI. The reason is data integrity: the saved models and the V2/V3/V4 backtests use a feature named `league_k` that is same-day actual league strikeouts. That value is not known pregame. It is target leakage.

I patched the feature builder so future training excludes raw `league_k` and keeps only shifted rolling league features such as `league_k_mean_roll3`.

After disqualifying the contaminated artifacts, I trained fresh no-leak models using only 2025 rows from the processed feature matrix and tested only 2026 rows. The best no-leak research candidate is promising:

- Model: regularized Poisson GLM, top 120 no-leak features
- Training: 2025 pitcher-games only
- Test: 2026 rows only, through 2026-06-18 in the repository
- Betting rule: unders only, `edge_pct >= 7`
- 2025 validation: 402 bets, +7.4% ROI, p=0.066, bootstrap CI [-1.9%, +16.6%]
- 2026 test: 613 bets, +10.6% ROI, p=0.0039, bootstrap CI [+3.2%, +18.3%]

That candidate does not satisfy full real-money deployment requirements yet because CLV is unavailable for this no-leak validation. It is suitable for paper/live shadow tracking, not auto-betting.

## Disqualified Evidence

Disqualified:

- `data/processed_oos2025/models/strikeouts.joblib`
- `data/processed_poisson_wf2025/models/strikeouts.joblib`
- `data/processed_ensemble_wf2025/models/strikeouts.joblib`
- `data/processed_poisson/models/strikeouts.joblib`
- V2/V3/V4 ROI claims based on those projections or derived edge files

Why:

- `src/features/build_features.py` created raw `league_k` as same-day mean starter strikeouts.
- `feature_cols` selected all columns starting with `league_k`.
- Binary model inspection confirmed `league_k` inside saved `strikeouts.joblib` feature lists.
- V2 explicitly found `league_k` was the strongest market-error predictor. That was a warning sign, not a discovery.

This is enough to reject all historical performance from those models, even where CLV is positive.

## Patch Applied

Changed [src/features/build_features.py](C:/Users/shado/pitcherKModel/src/features/build_features.py):

- `_league_krate_drift_features` now excludes raw same-day `league_k`.
- feature selection now includes only `league_k_` shifted rolling features.

Existing models must be retrained before production use. Old artifacts remain contaminated.

## Fresh No-Leak Validation

I added two validation scripts:

- [scripts/adversarial_strikeout_validation.py](C:/Users/shado/pitcherKModel/scripts/adversarial_strikeout_validation.py)
- [scripts/no_leak_2025_to_2026_model_search.py](C:/Users/shado/pitcherKModel/scripts/no_leak_2025_to_2026_model_search.py)

Outputs:

- `reports/adversarial_validation/`
- `reports/no_leak_2025_to_2026/`

The no-leak search:

- Uses `data/processed_noopp_wf2025_ext/bt_noopp_oos_edges.csv` as the feature/odds matrix.
- Excludes raw `league_k`.
- Excludes actual outcome columns.
- Excludes old projections, probabilities, edges, odds-derived outputs, and expected-opportunity model outputs.
- Trains projection models only on 2025 pitcher-games.
- Uses July-September 2025 as validation for candidate filters.
- Refits on all 2025 pitcher-games.
- Evaluates 2026 only.

## Model Comparison

Prediction accuracy:

| Model | Features | 2026 MAE | 2026 RMSE |
|---|---:|---:|---:|
| rolling K20 baseline | 1 | 1.811 | 2.263 |
| career K prior | 1 | 1.836 | 2.298 |
| Poisson top60 | 60 | 1.757 | 2.201 |
| Poisson top120 | 120 | 1.736 | 2.172 |
| Ridge top120 | 120 | 1.740 | 2.179 |
| ElasticNet top120 | 120 | 1.738 | 2.178 |
| HGB top120 | 120 | 1.788 | 2.243 |
| RF top120 | 120 | 1.762 | 2.205 |

The simple regularized Poisson GLM is the best prediction model and is also the easiest to maintain.

## Candidate Betting Rules

Eligible configurations were selected from 2025 validation only, then tested on 2026.

| Model | Rule | 2025 Validation | 2026 Test |
|---|---|---:|---:|
| Poisson top120 | under, `edge>=7` | 402 bets, +7.4% ROI | 613 bets, +10.6% ROI |
| ElasticNet top120 | under, `edge>=5` | 526 bets, +6.9% ROI | 699 bets, +8.0% ROI |
| Poisson top120 | under, `edge*gap>=3` | 284 bets, +9.2% ROI | 466 bets, +10.6% ROI |
| Poisson top120 | all sides, `edge>=3` | 1,316 bets, +3.7% ROI | 1,352 bets, +4.5% ROI |

The best balance is Poisson top120, unders only, `edge_pct >= 7`.

It beats the all-side candidate on ROI and statistical strength while preserving a larger sample than the edge-gap under subset.

## 2026 Stability

Poisson top120, under, `edge>=7`:

| Month | Bets | ROI | Win Rate |
|---|---:|---:|---:|
| 2026-03 | 31 | +22.1% | 61.3% |
| 2026-04 | 274 | +18.1% | 59.9% |
| 2026-05 | 185 | +6.1% | 54.6% |
| 2026-06 | 123 | -2.4% | 52.0% |

This model does not depend on one line:

| Line | Bets | ROI |
|---:|---:|---:|
| 3.5 | 29 | +22.4% |
| 4.5 | 160 | +9.6% |
| 5.5 | 252 | +12.2% |
| 6.5 | 131 | +2.9% |
| 7.5 | 38 | +20.7% |

Weakness: June is negative. This is consistent with prior repo notes, but using that as a hard future filter would need more clean evidence.

## CLV

No clean CLV conclusion is possible for the no-leak 2025-trained model.

The repository has CLV artifacts for contaminated models and separate Lekobe sharp-close studies, but I did not find a matched close-line file for the fresh no-leak 2025-to-2026 validation.

This is the biggest reason not to deploy real money yet.

Required before deployment:

- Capture entry odds and close odds for every recommended no-leak pick.
- Track DK close and sharp 3-book consensus close separately.
- Require at least 100 no-leak live/shadow bets with positive mean CLV before staking meaningful money.

## Recommended Status

Best production model today:

- None. Existing deployed artifacts are rejected.

Best research candidate:

- Poisson GLM top120 no-leak features
- Train through 2025 only
- Bet unders only
- `edge_pct >= 7`
- Flat stake only during validation

Runner-up:

- ElasticNet top120
- Bet unders only
- `edge_pct >= 5`
- Slightly lower ROI but larger 2026 sample

Simplest deployable model:

- Poisson GLM top120, under `edge>=7`

Highest ROI among eligible no-leak candidates:

- Poisson GLM top120, under `edge*gap>=3`
- Rejected as primary because it gives up sample size with no 2026 ROI improvement versus `edge>=7`.

Highest confidence:

- Poisson GLM top120, under `edge>=7`
- Best 2026 CI among reasonably large eligible candidates.

## Staking Strategy

Until CLV is proven:

- Paper trade or use minimum token stakes only.
- Do not use Kelly.
- Do not scale from contaminated backtests.

After CLV proof:

- Flat 0.25% to 0.50% bankroll per bet.
- Cap daily exposure at 3% bankroll.
- Re-estimate only after 250 settled no-leak bets.

## Automatic Disable Conditions

Disable betting if any of these occur:

- Mean CLV over last 100 bets <= 0.
- 30-day ROI < -5% and CLV <= 0.
- Any leakage feature reappears in model feature list, especially raw `league_k`.
- Close-line match rate < 80%.
- More than 5% duplicate betting opportunities after dedupe.
- Live Brier score worsens by >0.03 versus 2026 test baseline.
- Monthly drawdown exceeds 12 units per 100 flat-stake bets.
- A sportsbook/market source changes timestamp semantics.

## Live Metrics To Monitor

Track every day:

- Bets
- Win rate
- ROI
- Flat profit
- Average edge
- Average model probability
- Brier score
- Log loss
- Calibration by probability bucket
- Mean CLV
- Median CLV
- Positive CLV %
- CLV by sportsbook
- CLV by line
- CLV by month
- ROI by month
- ROI by line
- ROI by odds bucket
- ROI by pitcher/team/opponent
- Duplicate opportunity count
- Missing close-line count

## Exact Implementation Steps

1. Keep the `league_k` patch.
2. Retrain models from raw data after restoring raw CSVs, with `train_end: 2025-12-31`.
3. Explicitly assert that `league_k`, actual stat columns, old projections, and odds-output columns are absent from `feature_cols`.
4. Produce 2026 predictions from the frozen model.
5. Score only entries available before game start.
6. Select Poisson top120 under `edge_pct >= 7` as the paper-trade candidate.
7. Save every recommendation with entry time, book, odds, line, probability, edge, and model version.
8. Attach close odds after game start.
9. Do not place real stakes until no-leak live CLV is positive on at least 100 bets.

## Final Answer

The old model should not be bet.

The strongest no-leak candidate is credible enough to monitor: Poisson top120, unders only, `edge_pct >= 7`, trained on 2025 and tested on 2026. It shows +10.6% ROI on 613 available 2026 bets with a positive bootstrap interval.

But because CLV is missing for that exact no-leak candidate, it is not yet a professional-grade deployable edge. The correct conclusion is: no real-money model survives today; one promising candidate survives for shadow deployment and CLV validation.
