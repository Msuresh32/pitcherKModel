# Walk-Forward Training Review

## Research Review

1. Training only on 2025 is likely stale for production. It is excellent for a clean 2026 out-of-sample proof, but it intentionally ignores new pitcher form, role changes, arsenal changes, injuries, lineup context, and league environment updates.
2. An expanding walk-forward model should improve freshness and can improve calibration when the data-generating process drifts. It can also overreact if retrained too often on noisy early-season samples, so the evaluation has to measure calibration, CLV, and month stability, not just ROI.
3. Recommended production cadence: weekly expanding-window retraining, with an emergency/daily refresh only for feature inputs and injuries/lineups. In this repo-sized dataset daily retraining is computationally feasible, but weekly is operationally cleaner and reduces churn from one noisy slate.
4. Computational cost: frozen requires one fit; weekly requires about one fit per week; daily requires one fit per slate. On the current matrix, Poisson top-120 retraining is seconds-level, so the cost is not a blocker. The heavier cost is rebuilding raw Statcast/lineup features, not fitting the GLM.
5. The current repository can support walk-forward modeling at the matrix/backtest level, but production pipeline changes are required: preserve full historical raw feature caches, rebuild features by as-of date, schedule retraining, save model snapshots with cutoffs, and attach open/close odds snapshots for CLV.
6. Correct evaluation: lock the model spec and bet rule, generate predictions sequentially with training data strictly before each game date, compare against the frozen 2025 baseline on the exact same test dates and available markets, and judge ROI, CLV, calibration, prediction error, monthly stability, edge monotonicity, and paired daily P/L differences.

## Primary Comparison

- Test window: 2026-03-26 through 2026-06-18
- Bet rule: unders only, devig edge >= 7%, edge-gap >= 0
- Frozen ROI/units: +10.57%, +64.81 units on 613 bets
- Daily WF ROI/units: +6.51%, +41.12 units on 632 bets
- Weekly WF ROI/units: +6.99%, +44.03 units on 630 bets
- Frozen MAE/RMSE: 1.748/2.189
- Daily WF MAE/RMSE: 1.747/2.182
- Weekly WF MAE/RMSE: 1.746/2.182
- Frozen avg CLV: +1.94% at 42.3% coverage
- Daily WF avg CLV: +1.98% at 39.4% coverage
- Weekly WF avg CLV: +1.97% at 39.4% coverage
- Paired daily P/L diff, daily WF minus frozen: -0.282 units/day, bootstrap CI [-0.541, -0.012], Pr(WF > frozen) 2.0%

## Recommendation

Recommendation from this run: **do not replace solely on this test**.

The professional production architecture should still move to a point-in-time walk-forward framework, because frozen-season training is structurally stale. But replacement of the current betting methodology should require material improvement in the locked comparison, especially CLV and calibration, not only a plausible engineering argument.

## Files

- `walk_forward_summary.csv`
- `walk_forward_monthly.csv`
- `walk_forward_edge_buckets.csv`
- `walk_forward_bets.csv`
- `walk_forward_scored_opportunities.csv`