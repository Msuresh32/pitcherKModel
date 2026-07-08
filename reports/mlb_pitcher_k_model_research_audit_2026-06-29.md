# MLB Pitcher Strikeout Model Research Audit

Date: 2026-06-29

## Executive Summary

The strongest defensible evidence in this repository supports a simple Poisson strikeout model trained through 2024 and validated on 2025 walk-forward data. The best deployable rule from clean 2025 artifacts is:

- Model: Poisson GLM, strikeouts only
- Market: main-line DraftKings-style pitcher strikeout props
- Entry rule: bet when `edge_pct >= 20`
- Optional stricter rule: `edge_pct >= 25` when prioritizing CLV/confidence over volume
- Do not rely on the "full 2026" March-May artifacts as completely out-of-sample evidence, because the checked-in `config_poisson.yaml` trains through 2026-05-31 while those files begin on 2026-03-26.

On clean 2025 walk-forward data, the Poisson edge>=20 rule produced 596 flat-stake bets, +11.9% ROI, +1.00% mean CLV, and a bootstrap 95% ROI interval of roughly +2.9% to +20.6%. That is the most production-plausible signal in the repo.

The clean post-training June 2026 Poisson holdout is positive but small: 96 bets, +12.7% ROI, +1.15% mean CLV, with a normal CI that includes negative ROI. June 2026 is useful directional evidence, not enough by itself to declare the model durable.

## Repository Inventory

Major folders:

- `src/`: production code for feature engineering, model training, calibration, betting math, odds handling, exports.
- `scripts/`: fetchers, trainers, backtests, CLV tools, dashboard builders, diagnostics, and many research one-offs.
- `config/`: model/backtest configs for ensemble, Poisson, 2025 OOS, and 2026 experiments.
- `data/processed_poisson_wf2025/`: cleanest 2025 Poisson walk-forward artifacts.
- `data/processed_ensemble_wf2025/`: 2025 ensemble walk-forward artifacts.
- `data/processed_poisson/`: 2026-era Poisson artifacts, but some are contaminated/in-sample under current config.
- `data/processed_noopp_wf2025_ext/`: no-opportunity OOS edge universe used by under/fade diagnostics.
- `data/exports/` and `docs/picks/`: dashboard/export summaries that do not always agree with the processed-model artifacts.
- `lekobe/`: separate sharp-close, Kalshi, and fade-heavy-over research.

Missing raw production inputs in this checkout:

- `data/raw/pitcher_game_logs.csv`
- `data/raw/team_batting_game_logs.csv`
- `data/raw/batter_game_logs.csv`
- `data/raw/statcast_pitcher_daily.csv`
- `data/raw/statcast_batter_pitch_type_daily.csv`
- `data/raw/park_factors.csv`
- `data/raw/game_context_logs.csv`
- `data/raw/probable_pitchers.csv`
- `data/odds/pitcher_props.csv`
- historical close files referenced by configs

Because those raw inputs are absent, I could audit and recompute results from processed artifacts, but I could not fully reproduce feature generation, retraining, or fresh walk-forward backtests from source.

## Feature And Data Audit

Feature engineering is broad and mostly time-aware:

- Pitcher rolling features: 3/5/10/20 game rolling means, stds, min/max, K/IP, K/BF, BB/IP, pitches, strikes, BF, workload.
- Opponent/team features: opponent pitcher-result history, team batting K/BB/hit rates, lineup handedness, batter prior discipline.
- Statcast/pitch quality: CSW, SwStr, pitch-family matchup features, advanced pitcher/batter discipline if source files exist.
- Context: venue, weather, umpire, rest, home/away, park factors, league K-rate drift, pitcher-vs-team history, bullpen workload, YTD IP/starts.

Leakage posture:

- Most rolling and expanding stats use `shift(1)`, which is correct.
- FanGraphs merge is prior season, which is safe in design.
- Feature fill values are computed from the training period in `scripts/train.py`.
- Raw data absence prevents verifying all joins against actual fetch timestamps.

Data quality issues:

- All-edge files contain multiple line offerings per pitcher-game. Example: 2025 Poisson has 4,707 rows but 3,092 pitcher-games, or 1.52 lines per pitcher-game.
- Duplicate rows exist on `(game_date, pitcher_id, market, line, best_side)` in all-edge files: 643 in 2025 Poisson, 252 in 2026 Poisson. Threshold CLV files are deduped.
- Dashboard/export artifacts disagree with processed artifacts, especially 2026 live/export summaries. The processed model artifacts should be treated as the canonical model evidence.

## Critical Leakage Fix Applied

I patched `scripts/backtest.py`.

Previous behavior: a default backtest could build calibration on the current prediction window, save `calibration.json`, then reload it before scoring bets. That can apply same-period bias/probability calibration to the test window.

New behavior: the script loads any existing calibration before saving the current run's calibration, so a backtest can save calibration for future use without applying it to itself.

Verification: `python -m py_compile scripts\backtest.py`

## Model Comparison

Flat-stake results recalculated from processed artifacts:

| Strategy | Bets | Win Rate | ROI | Mean CLV | CLV N |
|---|---:|---:|---:|---:|---:|
| 2025 Poisson edge>=12 | 1,280 | 50.6% | +6.2% | +0.64% | 1,268 |
| 2025 Poisson edge>=15 | 967 | 50.8% | +7.5% | +0.75% | 956 |
| 2025 Poisson edge>=20 | 596 | 51.8% | +11.9% | +1.00% | 588 |
| 2025 Poisson edge>=25 | 335 | 51.6% | +12.8% | +1.03% | 332 |
| 2025 Ensemble edge>=12 | 1,210 | 49.8% | +5.6% | +0.63% | 1,199 |
| Clean June 2026 Poisson | 96 | 56.3% | +12.7% | +1.15% | 92 |

The ensemble does not beat the simpler Poisson model in the clean 2025 artifacts. The higher-volume Poisson thresholds are therefore preferred unless new raw-data retraining proves otherwise.

## 2026 Evidence Caveat

`config/config_poisson.yaml` has:

- `train_end: 2026-05-31`
- `backtest_start: 2026-06-01`

But `data/processed_poisson/bt_poisson_2026_full_*` spans 2026-03-26 to 2026-06-18. March-May results from those files are not valid complete OOS evidence under the checked-in config.

Treat only June 2026 artifacts as clean for that model snapshot:

- `data/processed_poisson/backtest_poisson_clv.csv`
- 96 bets
- +12.7% flat ROI
- +1.15% mean CLV
- CI approximately -7.6% to +33.0%

## Robustness And Stability

2025 Poisson edge>=20 monthly flat ROI:

| Month | Bets | ROI | Mean CLV |
|---|---:|---:|---:|
| 2025-04 | 103 | +29.8% | -0.08% |
| 2025-05 | 105 | +5.9% | +1.32% |
| 2025-06 | 94 | +5.2% | +1.40% |
| 2025-07 | 79 | -12.1% | +0.06% |
| 2025-08 | 115 | +11.1% | +0.79% |
| 2025-09 | 100 | +26.0% | +2.37% |

Failure pockets:

- July 2025 was negative despite non-negative CLV.
- June 2026 was negative in the contaminated `e20/e25 full` files, but positive in the clean June-only file at a lower/broader threshold.
- Edge>=30 becomes too small in 2025: 225 candidates and CI crosses zero.

Bootstrap ROI intervals:

- 2025 Poisson edge>=20: median +11.8%, 95% CI +2.9% to +20.6%, `P(ROI<=0) ~= 0.4%`.
- Contaminated combined 2025+2026 edge>=20: not production-valid because 2026 March-May is not clean OOS.

## Edge, Gap, And Side Analysis

On 2025 Poisson all-edge data:

- edge>=20: 683 candidate rows, +13.3% ROI before final dedupe/CLV-file filtering.
- edge>=20 and abs gap>=0.50: 440 candidate rows, +17.5% ROI.
- edge>=15 and abs gap>=0.50: 624 candidate rows, +17.6% ROI.

However, the gap filter is less stable across artifact families and can overfit selection. I would not ship a hard gap filter until it is validated in a clean 2026 OOS rerun from raw data.

Side split, 2025 Poisson edge>=20 all-edge data:

| Side | Bets | ROI | Win Rate |
|---|---:|---:|---:|
| Over | 259 | +31.4% | 59.5% |
| Under | 424 | +2.3% | 48.1% |

The strongest 2025 signal is overs, not unders. That conflicts with the Lekobe under/fade research and should be monitored carefully.

## CLV

The Poisson 2025 walk-forward model shows positive CLV across usable thresholds:

- edge>=12: +0.64%
- edge>=15: +0.75%
- edge>=20: +1.00%
- edge>=25: +1.03%
- edge>=30: +1.15%

This is the best reason to believe there may be a real market signal. ROI alone is not enough.

The Lekobe fade-heavy-over rule is separate:

- 964 bets
- +2.9% flat ROI
- +1.64pp mean CLV
- 68.5% positive CLV

But it is not the same production model, and the local reconciliation report found unresolved population differences between Lekobe's set and this model's covered-game universe.

## Recommended Production Model

Production candidate:

- Train through the last fully settled season only.
- Use Poisson GLM, not ensemble, unless a fresh OOS run proves the ensemble beats it.
- Strikeouts only.
- Main-line only.
- Disable walks and hits allowed.
- Use edge shrink factor 0.7.
- Use Poisson distribution for strikeouts.
- Require `edge_pct >= 20`.
- Prefer bets with line 3.5 to 6.5; be cautious on 7.5+ due sparse evidence.
- Track CLV against sharp 3-book close and DK close separately.
- Flat stake or very small fractional Kelly only. Historical Kelly summaries are too aggressive for the uncertainty level.

Expected annualized volume:

- 2025 edge>=20 produced 596 bets from April-September.
- Full-season expectation is roughly 600-800 bets/year depending on odds coverage and main-line availability.

Expected ROI:

- Conservative production expectation: +3% to +8%.
- Backtest point estimate: +11.9% in clean 2025 walk-forward.
- Do not underwrite to the contaminated 2026 March-May +30% style numbers.

Expected CLV:

- +0.5% to +1.0% if the 2025 signal persists.

## Situations Where The Model Should Not Bet

Do not bet when:

- The model was trained on data from the same evaluation period.
- Raw feature inputs or odds timestamps cannot be verified.
- Only stale, post-close, or unmatched line data is available.
- Multiple lines exist and the script has not selected a single main line per pitcher-game.
- CLV tracking is unavailable for more than a few days.
- The edge exists only in a tiny threshold bucket with fewer than about 300 annual bets.
- The slate is early season and current-year workload is unknown, unless separately validated.
- The bet is driven by an under-only/fade-heavy-over rule without resolving the Lekobe population mismatch.

## Remaining Research

Highest-value next steps:

1. Restore raw historical inputs and rerun 2025 and 2026 from source.
2. Build a true 2026 OOS backtest using models trained only through 2025-12-31.
3. Re-run all threshold scans after the calibration leakage patch.
4. Force one main line per pitcher-game before all edge scans.
5. Validate gap>=0.50 as a secondary filter on clean 2026 OOS.
6. Separate DK, Pinnacle, FanDuel, BetOnline, and sharp-consensus CLV.
7. Test whether overs remain the dominant edge after a fresh 2026 OOS rerun.
8. Add a production report that fails closed when raw inputs or close lines are missing.

## Final Verdict

There is enough evidence to continue with a cautious production trial, but not enough to claim a fully proven durable edge across seasons from the checked-in repository alone.

The best defensible model is the 2025 walk-forward Poisson edge>=20 strategy. It has the right profile: hundreds of bets, positive ROI, positive CLV, and a simple deployable rule. The main unresolved risk is that the strongest 2026 artifacts are not clean OOS under the current config, so the next decisive test is a true 2026 OOS rerun from raw data with training ending no later than 2025-12-31.
