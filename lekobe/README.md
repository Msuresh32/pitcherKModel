# Lekobe Pitcher-K Strategy Comparison

This folder contains my comparison package for the MLB pitcher strikeout strategy:

- Rule: on the main pitcher-K line only, when the sportsbook over is priced at or below -140, buy the Kalshi NO contract as the under.
- Execution benchmark: grade fills against the sharp sportsbook close, not against the Kalshi close.
- Main-line filter: exclude alternate lines before applying the over <= -140 rule.
- Fill accounting: collapse Kalshi fills by ticker using contract-weighted VWAP from `count_fp`.
- Maker/taker accounting: classify positions by contract-weighted `is_taker`.

## Contents

- `scripts/read_kalshi_fills.py`: pulls local Kalshi KXMLBKS fills and writes a normalized CSV.
- `scripts/kalshi_auth.py`: local Kalshi API signing helper. It requires `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` from the local environment.
- `scripts/scan_fade_heavy_over_unders.py`, `scripts/pull_today_over_favored_unders.py`, `scripts/main_line_after_2pm_card.py`: scanner/card generation logic.
- `scripts/analyze_kalshi_maker_taker.py`, `scripts/regrade_vwap_fill_collapse.py`: fill collapse and maker/taker diagnostics.
- `scripts/grade_*.py`, `scripts/fast_kprop_close_coverage.py`, `scripts/fetch_historical_pitcher_k_clv.py`: sharp-close grading and ledger construction.
- `ledgers/running_optimized_kprop_ledger.csv`: running sharp-graded execution ledger for the optimized rule.
- `ledgers/jun17_sharp_grade.csv`: June 17 sharp-close grade detail.

No private keys, `.env` files, API tokens, or account credentials are included.
