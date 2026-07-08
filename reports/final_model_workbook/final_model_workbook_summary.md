# Final Model Workbook Summary

Workbook: `C:\Users\shado\pitcherKModel\reports\final_model_workbook\MLB_Pitcher_K_Final_Model_Bet_Ledger.xlsx`
All-bets CSV: `C:\Users\shado\pitcherKModel\reports\final_model_workbook\final_model_all_bets.csv`

## Model Ledger

- Model: `poisson_top120_a2`
- Rule: `under` only, devig edge >= 7%, edge-gap >= 0
- Bet count: 1,015
- Win rate: 56.16%
- Flat-stake ROI: 9.32%
- Units: 94.62
- Profit at $100 flat stake: $9,462.08
- Average odds: -7
- Average margin: 0.45 strikeouts
- Average devig edge: 17.75%

## Bankroll Simulations

- Strategy A (Flat $100 per bet): final bankroll $19,462.08, total return 94.62%, max drawdown $-2,351.87 (-19.17%)
- Strategy B (Flat 1% of current bankroll): final bankroll $24,513.53, total return 145.14%, max drawdown $-3,929.99 (-21.20%)
- Strategy C (Quarter Kelly from model probability): final bankroll $374,006.89, total return 3640.07%, max drawdown $-517,550.60 (-73.59%)
- Strategy D (Half Kelly from model probability): final bankroll $1,054,509.49, total return 10445.09%, max drawdown $-5,993,664.37 (-94.20%)

## Monthly Snapshot

| Month | Bets | Win Rate | ROI | Units | Profit | Ending Bankroll |
|---|---:|---:|---:|---:|---:|---:|
| 2025-07 | 117 | 52.14% | 3.91% | 4.57 | $457.43 | $10,457.43 |
| 2025-08 | 177 | 51.41% | -0.95% | -1.68 | $-167.98 | $10,289.46 |
| 2025-09 | 108 | 64.81% | 24.93% | 26.92 | $2,691.97 | $12,981.42 |
| 2026-03 | 31 | 61.29% | 22.14% | 6.86 | $686.23 | $13,667.66 |
| 2026-04 | 274 | 59.85% | 18.07% | 49.52 | $4,951.70 | $18,619.36 |
| 2026-05 | 185 | 54.59% | 6.13% | 11.34 | $1,133.88 | $19,753.24 |
| 2026-06 | 123 | 52.03% | -2.37% | -2.91 | $-291.17 | $19,462.08 |

## Validation Checks

- Unique Bet IDs: True
- Duplicate bets detected: 0
- Running bankroll reconciles: True
- Profit equals ending bankroll minus starting bankroll: True
- Units reconcile with total profit at $100/unit: True

## Data Notes

- Opening odds, closing odds, and CLV were not available for the clean no-leak candidate ledger and were intentionally left blank.
- 2025 bets are validation bets from the inner no-leak model. 2026 bets are frozen-model test bets.
- The June holdout ends at the latest settled result available in the source data.