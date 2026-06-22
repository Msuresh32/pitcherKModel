# Under Bet Analysis — Market Edge by Odds Bucket

**Date:** 2026-06-22  
**Dataset:** 6,208 resolved bets (2025–2026)  
**Strategy tested:** Fading heavy overs (betting the under) at fair devig price

---

## Overall

**ROI: −3.4%** — blindly fading heavy overs at fair devig price does not work across the board.

---

## Breakdown by Over-Odds Bucket

| Over Odds Range | Bets | Win Rate | ROI |
|---|---|---|---|
| −140 to −150 | 409 | 48.7% | **+7.6%** |
| −150 to −160 | 481 | 36.8% | −14.8% |
| −160+ | 5,233 | 30.7% | −3.5% |

The **−140 to −150 range is the only profitable segment (+7.6% ROI)**.  
Once the over reaches −150 or heavier, the market prices it correctly and the under has no edge.

---

## Breakdown by Line

| Line | Bets | ROI |
|---|---|---|
| 3.5 | 2,381 | −1.3% |
| 4.5 | 1,726 | −3.1% |
| 7.5 | 38 | +12.5% *(small sample)* |

---

## Year Split

| Year | Bets | ROI |
|---|---|---|
| 2025 | 4,326 | −6.8% |
| 2026 | 1,882 | **+4.4%** |

The 2025 vs 2026 divergence is notable — worth monitoring as more 2026 data accumulates.

---

## Actionable Takeaway

The **−140 to −150 over range** (409 bets) shows real edge for the under at fair devig.  
At −150 or heavier, the market has it right and you're paying vig into a losing position.

**Filter to apply:** Only take unders when the corresponding over is priced between −140 and −150.

---

## Diagnostic Context (from `research_unders_diagnostic.py`)

Model-level analysis (475 OOS bets, 2025–2026 OOS window) confirmed the same structural finding:

- Negative-odds unders (market already says likely): n=69, WR=68.1%, ROI=**+25.8%**
- Positive-odds unders (model contrarian): n=406, WR=45.8%, ROI=+1.4%

The model only adds value when it *confirms* what the market already believes about a given under.  
Contrarian under picks (model vs market) have no reliable directional skill at current confidence thresholds.

**Core failure mode:** The Poisson GLM has a systematic low-K bias — it underestimates strikeout  
counts by +0.69K on average for under bets. The same bias that gives overs a buffer above the line  
leaves unders with zero cushion (actual ends up +0.06K above line on average).

**Summer deterioration:** June−July ROI drops to −10.9% / −21.1%. Heat, fatigue, and elevated  
real K rates are not captured by the no-opportunity model.
