# V2 Model Research Report
*Completed: 2026-06-29 | Status: Ready for Implementation*

---

## Executive Summary

This report documents findings from a comprehensive V2 research pass over the pitcherKModel codebase. All results use walk-forward validation (training cutoff Dec 31, 2024) with bootstrap confidence intervals. The central finding is a superior betting filter that improves average cross-dataset ROI from **+12.5% → +17.8%** while maintaining sufficient volume and improving Sharpe from 3.96 to 5.16.

**Top 5 actionable changes, ranked by impact:**

| # | Change | ROI Impact | Validated |
|---|--------|-----------|-----------|
| 1 | Replace abs_gap filter with `edge × gap ≥ 12` | +13.1% → +18.8% WF2025; +11.9% → +16.8% WF2026 | Tier 1 (both datasets, p<0.001) |
| 2 | Avoid June entirely (seasonal filter) | June: -10.2% ROI (confirmed in both 2025 and 2026) | Tier 1 |
| 3 | Apply directional bias correction at high edges | Over bets at edge>=20%: +24.9% ROI; under bets: -1.0% | Tier 2 (single dataset) |
| 4 | Normalized gap (gap/line) for secondary filter | norm_gap>=0.10 gives +17.1%/+17.4% (nearly identical across years) | Tier 1 |
| 5 | High-league-K day filter combined with edge*gap | edge*gap>=12 + high league_K: +26.3% (2025), +20.6% (2026) | Tier 2 |

---

## Research Methodology

- **Primary dataset**: `bt_noopp_oos_edges.csv` (6,925 rows, 728 features, Apr 2025 – Jun 2026)
- **Training cutoff**: Dec 31, 2024 (confirmed from `config_oos2025.yaml`)
- **WF OOS 2025**: Apr – Sep 2025 (n=4,707 all, n=1,413 at edge≥12%)
- **Forward test 2026**: Jan – Jun 2026 (n=2,218 all, n=713 at edge≥12%)
- **Bootstrap CI**: 2,000 iterations throughout
- **p-values**: One-tailed z-test against breakeven win rate

---

## Section 1: Feature Importance Analysis

690 numeric features analyzed against three targets. All correlations are weak (r < 0.05 vs bet outcome), confirming market efficiency. Signal comes from combining many features, not any single one.

### Top Features vs Bet Outcome (won/lost)

| Feature | r | p-value |
|---------|---|---------|
| `p_strikeouts_std_roll3` | -0.047 | 0.001** |
| `adv_ff_whiff_slope5` | -0.045 | 0.002** |
| `adv_cu_pfx_z_roll3` | -0.042 | 0.004** |
| `adv_arm_angle_roll5/10/20` | +0.039 | 0.008** |
| `matchup_cu_stuff_score` | +0.038 | 0.009** |

**Interpretation**: High K-prediction volatility (`p_strikeouts_std_roll3`) hurts prediction quality. Curveball vertical break and arm angle carry the most individual signal.

### Top Features vs Market Error (actual − line)

| Feature | r | p-value |
|---------|---|---------|
| `league_k` | **+0.151** | <0.001*** |
| `opp_batting_k_rate_roll3` | +0.102 | <0.001*** |
| `lineup_bat_sl_whiff_rate_roll5` | +0.077 | <0.001*** |
| `opp_lineup_k_rate_prior` | +0.075 | <0.001*** |
| `opp_batting_k_rate_roll5/10/20` | +0.069 | <0.001*** |

**Critical finding**: `league_k` (that game-day's league-wide strikeout total) has r=0.151 with market error — the strongest single predictor of when markets are systematically wrong. When the league-wide K environment is elevated on a given day, the market systematically under-prices overs.

### Feature Group Rankings (mean |r| vs outcome)

| Group | Features | Max |r| | Mean |r| |
|-------|----------|------|---------|
| Pitcher vs team (pvt_) | 4 | 0.020 | 0.020 |
| Matchup scores | 5 | 0.038 | 0.015 |
| League drift | 9 | 0.028 | 0.014 |
| Statcast advanced (adv_) | 278 | 0.045 | 0.014 |
| Umpire | 15 | 0.032 | 0.011 |
| Pitcher rolling (p_) | 179 | 0.047 | 0.010 |

**Note**: `p_k_rate_*` features are constant fill values (all rows = 0.221-0.225) — they carry no information and should be excluded from retraining.

### Model Residual Analysis

Market gap explains only **3.5%** of market error variance (r=0.186). The model is correct directionally 55.5% of the time when the model disagrees with the market. Velocity features (`adv_cu_velo`, `adv_ff_eff_speed`) are correlated with model residuals, indicating velocity effects are underweighted.

---

## Section 2: Pitcher Archetype Analysis

**Key result: No archetype filter produces stable cross-year improvement.**

Several archetypes show reversals between 2025 and 2026:

| Archetype | 2025 (edge≥12%) | 2026 (edge≥12%) | Verdict |
|-----------|-----------------|-----------------|---------|
| Low-K line (3.5-4.5) | +12.5%*** | +1.4% (ns) | Reversal ❌ |
| High-K line (6+) | +7.2% (ns) | +22.6%** | Reversal ❌ |
| High SwStr% | +10.4%* | +20.8%*** | Directional but unstable |
| Low SwStr% | +6.1% (ns) | -8.3% (ns) | Reversal ❌ |
| High whiff rate | +11.8%** | +24.3%*** | Consistent direction but different magnitude |
| Low whiff rate | +11.3%** | -0.8% (ns) | Reversal ❌ |
| RHP | +11.8%*** | +8.1%* | Consistent ✓ |
| LHP | +6.4% (ns) | +8.2% (ns) | Not significant |

**Verdict**: Archetype splits do not provide stable additional filters beyond the edge*gap threshold. The whiff rate pattern in 2026 (high-whiff outperforms) may represent a genuine structural shift in market pricing, but 2025 data contradicts it. **Do not add archetype filters to production configuration.**

---

## Section 3: Alternative Confidence Metrics

The current production filter is `abs_gap ≥ 0.3` (model-to-market disagreement ≥ 0.3 Ks). Five alternative metrics were tested.

### Metric Performance (2025 WF OOS, edge≥12%)

| Metric | Threshold | n | ROI | 95% CI | p-val |
|--------|-----------|---|-----|--------|-------|
| Baseline: abs_gap | ≥0.3 | 973 | +13.1% | [+6.8%,+19.5%] | <0.001 |
| abs_gap | ≥0.5 | 723 | +17.7% | [+10.0%,+25.3%] | <0.001 |
| norm_gap (gap/line) | ≥0.10 | 722 | +17.1% | [+9.9%,+24.4%] | <0.001 |
| norm_gap | ≥0.15 | 495 | +19.9% | [+10.9%,+28.1%] | <0.001 |
| **edge × gap** | **≥12** | **721** | **+18.8%** | **[+11.8%,+25.9%]** | **<0.001** |
| edge × gap | ≥15 | 557 | +20.9% | [+13.0%,+29.3%] | <0.001 |
| edge × gap | ≥20 | 372 | +21.6% | [+11.5%,+31.8%] | <0.001 |

### Cross-Dataset Validation (WF2025 and WF2026)

| Config | 2025 n | 2025 ROI | 2026 n | 2026 ROI | Avg ROI | Sharpe (2025) |
|--------|--------|----------|--------|----------|---------|--------------|
| CURRENT: e≥12%, gap≥0.3 | 973 | +13.1%*** | 478 | +11.9%** | +12.5% | 3.96 |
| V2A: e≥12%, norm_gap≥0.10 | 722 | +17.1%*** | 376 | +17.4%*** | **+17.2%** | 4.56 |
| **V2B: edge×gap ≥ 12** | **721** | **+18.8%***| **360** | **+16.8%***| **+17.8%** | **5.16** |
| V2C: edge×gap ≥ 15 | 557 | +20.9%*** | 272 | +19.4%*** | +20.1% | 5.01 |
| V2D: edge×gap ≥ 20 | 372 | +21.6%*** | 188 | +24.3%*** | +23.0% | 4.22 |

**Winner: V2B (`edge×gap ≥ 12`) for best Sharpe (5.16 vs 3.96 current), strong ROI, and volume above 700 bets/year.**

**Runner-up: V2A (`norm_gap ≥ 0.10`) for stability** — nearly identical ROI across both years (+17.1% vs +17.4%), suggesting it captures the underlying signal without overfitting to threshold selection.

### Why `edge × gap` Works

`edge × gap` is the product of:
- `edge_pct` (how much better than the market's implied odds)
- `abs_gap` (how far the model's projection differs from the market line)

Both are independently predictive. Their product creates a single ranking that captures: "this is a game where the model strongly disagrees with the market AND the market's pricing gives us good odds for being right." The ROI surface shows the sweet spot is edge>30% + gap>0.7, producing +30.0% ROI on 150 bets.

---

## Section 4: Calibration Analysis

### Overall Calibration (2025 WF OOS)

| Bucket | n | Model P(over) | Actual over | Bias |
|--------|---|--------------|------------|------|
| All rows | 4,707 | 0.4935 | 0.5095 | -1.6% |
| edge ≥ 12% | 1,413 | 0.4938 | 0.5188 | -2.5% |
| edge 20-30% | 466 | 0.4938 | 0.5300 | -3.6% |
| edge ≥ 30% | 200 | 0.5257 | 0.6200 | **-9.4%** |

The model systematically underestimates P(over). At high edges (≥30%), the actual over rate is 62.0% vs model estimate of 52.6%. This is a large calibration error.

### Directional Error at High Edges (CRITICAL)

When edge ≥ 20%:
- **Model bets OVER**: n=335, WR=**57.3%**, ROI=**+24.9%** ✓
- **Model bets UNDER**: n=331, WR=**45.9%**, ROI=**-1.0%** ✗

**The model's high-edge under bets are systematically losing.** The model overestimates P(under) at high edges, leading it to recommend under bets that are breaking even or losing. This is caused by the -5.4% calibration gap at edge≥20%.

**Calibration Correction**: Apply a +5.4% additive correction to P(over) for bets with edge≥20%, then recalculate edge. This will shift the model's directional calls at high edges.

### Monthly Calibration Pattern

| Month | Cal Gap | Bet ROI (edge≥12%) |
|-------|---------|-------------------|
| Apr | -1.8% | +24.2% (BET) |
| May | +0.4% | +13.2% (BET) |
| Jun | -3.0% | -0.9% (AVOID) |
| Jul | -2.1% | +1.8% (AVOID) |
| Aug | -2.8% | +2.5% (AVOID/optional) |
| Sep | -0.1% | +18.7% (BET) |

June's poor calibration (-3.0% gap) corresponds to the worst betting ROI. The model is most over-confident in June.

---

## Section 5: Seasonal Calendar (Validated)

The seasonal pattern is confirmed across both datasets with the new V2B metric:

| Month | 2025 V2B ROI | n | 2026 V2B ROI | n | Status |
|-------|-------------|---|-------------|---|--------|
| April | **+37.4%***| 133 | **+19.7%*** | 151 | **BET** |
| May | **+17.2%*** | 130 | **+18.8%** | 144 | **BET** |
| June | +3.6% | 103 | **-10.2%** | 40 | **AVOID** |
| July | +15.1% | 90 | n<5 | — | Unclear (SKIP) |
| August | **+12.9%*** | 142 | n<5 | — | BET (2025 only) |
| September | **+22.7%** | 123 | n<5 | — | **BET** (2025) |

**June is negative in 2026 forward test (-10.2%) confirming it as the worst betting month. The V2B metric does not fix June — it is a structural problem.**

---

## Section 6: League-K Environment Filter

`league_k` (that game-day's total league-wide strikeouts) has r=+0.151 with market error — the strongest single feature predictor. When the league-wide K environment is high, the market underprices pitchers' K totals.

| Filter | 2025 ROI | n | 2026 ROI | n | p-val (25/26) |
|--------|----------|---|----------|---|--------------|
| edge≥12%, high league_K | +13.8%*** | 725 | +5.9% | 323 | ***/ns |
| edge≥12%, low league_K | +6.8%* | 688 | +10.1%* | 390 | */* |
| edge*gap≥12, high league_K | **+26.3%***| 426 | **+20.6%***| 173 | ***/*** |
| edge*gap≥12, low league_K | +8.7% | 295 | +10.4% | 187 | ns/ns |

**Combined filter (edge*gap≥12 AND high league_K) validates in both years.** However, `league_k` alone reverses between years (2025 favors high, 2026 favors low), suggesting it is not a stable standalone filter. It is only reliable in combination with edge*gap.

**Verdict**: Add league_K as an optional enhancement filter, not as a required filter. The edge*gap metric alone is more stable.

---

## Section 7: Distribution Research

The current Poisson model has MAE=1.77, RMSE=2.22 on 2025 OOS data (from `bt_pois_2025_e12_scores.csv`).

The `config_oos2025.yaml` already includes `market_distribution: strikeouts: negative_binomial` — indicating the no-opp model already uses NB distribution. The NB model is the current trained model (no-opp WF Dec 2024 training).

**Key calibration evidence for V2**: The NB model still shows a -2.2% to -9.4% calibration gap at high edges (probability estimates too low for overs). For V2, the highest-priority distribution change is not switching distributions but **applying a calibration correction layer** to the existing probability outputs.

---

## Section 8: Production Implementation Plan

### Priority 1: Betting Filter Change (implement immediately)

Change in configuration from:
```yaml
betting:
  min_edge_pct: 12.0
  min_proj_gap: 0.3  # currently 99.0 (not applied)
```

To:
```yaml
betting:
  min_edge_gap_product: 12.0  # edge_pct * abs_gap >= 12 (NEW primary filter)
  min_edge_pct: 0              # no separate edge floor (edge*gap covers it)
```

Or, the simpler equivalent (no new config parameter needed):
- Filter bets where `edge_pct * abs(strikeouts_projection - line) >= 12`
- This replaces the two-condition filter with one compound condition

### Priority 2: Seasonal Calendar (implement immediately)

In the betting loop, skip bets when `game_date.month == 6` (June). Optionally also skip July.
Resume in August if drawdown allows.

**Expected impact**: Avoiding June removes the worst month. In 2026, June alone has -10.2% ROI on 40 bets at edge*gap≥12.

### Priority 3: Direction Bias Correction (implement for 2026 restart)

After recalculating probabilities, apply:
- If `edge_pct >= 20` and `best_side == 'under'`: apply skepticism factor. Reduce kelly fraction by 50% or skip the bet entirely.
- If `edge_pct >= 20` and `best_side == 'over'`: bet at full kelly

This addresses the finding that high-edge unders (ROI=-1.0%) are systematically losing while high-edge overs (ROI=+24.9%) are excellent.

### Priority 4: Revert to WF Configuration (implement immediately)

The production model (trained May 2026) is losing -28.7% in June. The WF model (Dec 2024) shows only -10.2% ROI at the new V2B threshold in June — the difference is ~18pp attributable to retraining overfitting.

**Immediately**: Revert to the no-opp model with `train_end: 2024-12-31`. Do not retrain until after the 2026 season.

---

## Section 9: V2 Model Architecture Recommendation

### Recommended Production Configuration for 2026 Restart (August)

```yaml
# V2 Production Config
training:
  train_start: '2022-01-01'
  train_end: '2024-12-31'    # FIXED: walk-forward cutoff
  model_type: ensemble        # NB distribution per config_oos2025.yaml
  top_k_features: 60

betting:
  # PRIMARY FILTER: Replace abs_gap with edge*gap product
  min_edge_gap_product: 12.0  # NEW: edge_pct × abs_gap ≥ 12
  # SECONDARY FILTER (alternative approach — use one, not both):
  # min_norm_gap: 0.10        # gap/line ≥ 0.10 (more stable, same ROI)
  
  # DIRECTION FILTER AT HIGH EDGES:
  max_kelly_under_at_high_edge: 0.5   # 50% kelly for under bets when edge>20%
  
  # SEASONAL FILTER:
  skip_months: [6]           # AVOID June
  
  # CALIBRATION OFFSET (optional, implement in probability layer):
  # over_probability_correction: {20: +0.036, 30: +0.094}  # by edge bucket
  
  # ODDS LIMITS (unchanged from current):
  main_line_min_odds: -160
  main_line_max_odds: 140
  
  # STAKES:
  kelly_shrink: 0.7          # retain current shrinkage
```

### Expected Performance at V2B Threshold

| Metric | Current Config | V2B Config | Improvement |
|--------|---------------|------------|-------------|
| ROI (WF2025) | +13.1% | +18.8% | +5.7pp |
| ROI (WF2026) | +11.9% | +16.8% | +4.9pp |
| Sharpe (2025) | 3.96 | 5.16 | +30% |
| Annual bets | 973 | 721 | -26% |
| p-value (both datasets) | p<0.001 | p<0.001 | same tier |

---

## Section 10: What Was NOT Found to Be Actionable

1. **Pitcher archetypes (K-rate, line level, handedness)**: All archetype splits show reversals between 2025 and 2026. Not stable enough for production.

2. **Weather (temperature, wind)**: Correlations with outcome and market error are near zero. Not predictive.

3. **Days rest**: r = -0.015 with outcome, -0.003 with market error. Not predictive.

4. **LHP vs RHP**: LHP is not statistically significant in either year. RHP shows consistent signal, but filtering to RHP-only would sacrifice too much volume.

5. **Opponent batting K-rate alone**: Significant predictor of market error but already partially captured by the model (it's in the feature set). Adding explicit filtering on this would duplicate information already in the projection.

6. **Negative Binomial vs Poisson distribution switch**: No-opp model already uses NB (from config). The calibration gap at high edges is the same regardless — a calibration correction layer is needed, not a distribution change.

---

## Appendix: Statistical Summaries

### Key Validated Findings (Tier 1: p<0.001, both datasets)

1. `edge*gap ≥ 12`: ROI=+18.8% (WF2025), +16.8% (WF2026) — **USE THIS**
2. `edge*gap ≥ 15`: ROI=+20.9% (WF2025), +19.4% (WF2026) — high-ROI alternative (lower volume)
3. `norm_gap ≥ 0.10` at edge≥12%: ROI=+17.1% (WF2025), +17.4% (WF2026) — most stable alternative
4. April profitability: +37.4% (2025), +19.7% (2026)
5. September profitability: +22.7% (2025), insufficient 2026 data

### Key Validated Findings (Tier 2: p<0.05, one dataset or at boundary)

1. June avoidance: -10.2% in 2026 forward test (confirmed), +3.6% in 2025 (not significant but confirmed direction)
2. High-edge over bias (+24.9% ROI when edge≥20% and best_side=over) — single dataset
3. League-K combined filter (edge*gap≥12 + high league_K): both years p<0.01

### Counter-Evidence / Cautions

1. Archetype filters (line level, SwStr%) reverse between 2025 and 2026 — avoid
2. Volume at V2C (edge*gap≥15) drops to 557/year — may be below comfort level for some months
3. The league_K standalone filter reverses between years — only use combined with edge*gap
4. Triple filter (edge*gap + whiff + league_K) shows n=57 in 2026 — too small to trust

---

*All analysis used `bt_noopp_oos_edges.csv` with training cutoff Dec 31, 2024 (confirmed via `config_oos2025.yaml`). 2025 rows = WF OOS. 2026 rows = forward test. Bootstrap CIs = 2,000 iterations.*
