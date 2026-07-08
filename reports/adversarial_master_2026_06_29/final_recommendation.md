# Final Recommendation Through 2026-06-29

## Executive Verdict

Grade: **B. Small-stakes live test only**.

There is evidence of a possible strikeout-under edge in the clean 2026 sample through 2026-06-18, but the extension through 2026-06-29 is weaker and the full feature cache needed for a canonical post-06/18 point-in-time rebuild is not preserved in the repo. I would not classify this as deployable real-money infrastructure yet.

## Current Model Through 2026-06-29

- Bets: 683
- Win rate: 56.37%
- ROI: +9.32%
- Units: +63.65
- Bootstrap ROI CI: [+1.74%, +16.70%]
- Probability true ROI <= 0 by bootstrap: 0.6%
- Max drawdown: -22.87 units
- Longest losing streak: 7

## CLV

- CLV matched: 309/683
- Average CLV: +1.32%
- Median CLV: +0.87%
- Positive CLV rate: 55.99%

## Direct Answers

- Is there an edge? Possible, but not proven strongly enough for real money.
- Which model should I use? If you must monitor one, use the frozen 2025 `poisson_top120_a2` unders-only rule as a paper/small-token shadow strategy. Do not promote WF yet.
- How confident am I? Low-to-moderate that there is a small edge; high confidence that the repo is not yet production-audit clean through 06/29.
- What can go wrong? Stale feature cache, CLV coverage gaps, best-line optimism, model decay in late June, missing lineup/Statcast timestamps, and variance from under-heavy exposure.
- Should you bet now? No meaningful stakes. Paper trade or token-size only until live CLV and settlement are clean for at least 100 new bets.

## Production Architecture

The right architecture is weekly point-in-time retraining with daily feature/odds refreshes, model snapshots, raw input snapshots, and close-line capture. The current evidence does not justify replacing the frozen model with walk-forward because WF underperformed ROI in the locked comparison.

The simple 75% frozen / 25% weekly-WF blend slightly beat frozen through the complete 06/18 matrix (+65.11 vs +64.81 units), but the improvement is too small to justify added complexity or to call it a new production model. Keep it as a shadow candidate only.

## Tomorrow Morning Checklist

1. Fetch probables, lineups, current odds, and save immutable odds snapshot.
2. Rebuild features using only data through yesterday.
3. Score the frozen candidate and any shadow WF/blend candidates.
4. Log every candidate, including skipped bets and reason.
5. Capture closing odds before first pitch for CLV.
6. Do not bet if projected volume spikes, CLV turns negative over the last 30-50 bets, or June-style drawdown continues.

## Deliverables

- `model_audit_master_summary.csv`
- `valid_model_comparison.csv`
- `best_candidate_bets_through_2026_06_29.csv`
- `clv_audit.csv`
- `robustness_grid.csv`
- `execution_sensitivity.csv`
- `feature_leakage_audit.csv`
- `artifact_inventory.csv`
