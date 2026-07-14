# Qualitative Validation Pass — analyst instructions

For every MLB strikeout prop that passes the model's deployment thresholds,
perform a qualitative validation pass.

The goal is NOT to create a narrative for every play. The goal is to determine
whether there are legitimate baseball reasons that independently support the
model's prediction. Only include supporting rationale if it is actually
present. If no meaningful qualitative edge exists, explicitly state:

> "No strong qualitative confirmation found beyond the model edge."

Do NOT force confirmation or cherry-pick statistics.

## Inputs
1. `reports/daily/v2_qual_context_<date>.json` — outcome-blind internal
   context per play (model projection, opponent K/BB profile, pitcher trend,
   workload/pull tendencies, Statcast indicators, park and market coverage).
   Produced by `research/v2/a7_qual_brief.py` using rows strictly before the
   score date. The same script writes a deterministic qualitative score and
   writeup for every qualified play; it never changes the model gate or stake.
2. Web research for what internal data can't see: confirmed lineups, injury /
   IL news, announced umpire, weather (wind/roof), recent velocity readings,
   arsenal-vs-team run values (Savant), opponent K% vs handedness L7/L14/L30.

Verified external findings may be saved to
`reports/daily/v2_qual_external_<date>.json` as:

```json
{"plays":[{"pitcher":"Name","line":5.5,
  "support":["Verified material finding."],
  "contradictory":["Verified material concern."],
  "covers":["weather/roof"],"sources":["https://..."]}]}
```

Absent external fields are explicitly marked unavailable and are never
inferred. The 0-100 qualitative score starts at 50, adds only material support,
and subtracts material concerns. It is separate from model conviction.

## Evaluate (only report what is material)
- **Matchup**: opponent K% vs handedness (L7/L14/L30/season), whiff%, zone
  contact%, chase contact%, swing%, pitches/PA, BB%, ability to extend ABs,
  recent form (wRC+, xwOBA), home/road splits if meaningful.
- **Pitcher profile**: home/road splits, recent velocity changes, K trends,
  pitch-count trends, third-time-through usage, innings limits, bullpen
  rest, team pull tendencies, injury/fatigue/news.
- **Arsenal fit**: pitcher's pitch mix vs opponent performance against those
  pitch types (run values, whiff, contact, zone, chase). Mention only when a
  meaningful matchup edge exists.
- **Batter history**: only with reasonable samples; expected lineup,
  contact-heavy vs high-K bats, L/R balance. Do NOT overvalue BvP.
- **Context**: park, weather, umpire, travel, rest, day/night, last game
  before break, bullpen status, genuine usage incentives.
- **Market**: opening line, current line, steam, books disagreeing. Do NOT
  use line movement itself as evidence the play is good.

## Output format (per play)

### Model Summary
Model projection · sportsbook line · edge · confidence tier · stake status
(NOTE: while `deploy_config.json` says NOT_DEPLOYABLE, all plays are PAPER —
say so explicitly.)

### Qualitative Confirmation
Bulleted findings that materially support the play (or the explicit
no-confirmation line).

### Contradictory Evidence
Any meaningful reasons the play may be weaker than the model suggests.

### Verdict (choose one)
Strong qualitative confirmation · Moderate qualitative confirmation ·
Neutral (model-only edge) · Mixed signals · Qualitative concerns

The qualitative analysis NEVER overrides a qualifying model edge by itself.
It is an independent sanity check, not a source of confirmation bias.
