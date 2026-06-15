# MLB Pitcher Prop Prediction MVP

Baseline Python repo for pitcher strikeouts, walks, and hits allowed props.

**Current research focus:** pitcher strikeouts. The strikeout-only workflow, active
files, feature inventory, and next commands are organized in
[`docs/STRIKEOUTS.md`](docs/STRIKEOUTS.md).

The MVP trains on 2022-2024 MLB pitcher game logs, backtests on 2025, and generates daily projections for probable pitchers. It is intentionally simple: rolling pitcher stats, opponent/team features, one model per prop, optional odds-line backtesting, and CSV export.

## Project Structure

```text
data/
  raw/             # downloaded/source data
  processed/       # model-ready datasets and predictions
  odds/            # optional prop lines / prices
  exports/         # daily picks
notebooks/         # exploration
src/
  data/            # loading and MLB data source adapters
  features/        # feature engineering
  models/          # training and prediction
  backtesting/     # 2025 evaluation and bet simulation
  odds/            # fair odds, EV, Kelly helpers
  export/          # CSV / Google Sheets export hooks
config/            # YAML configs
scripts/           # command-line workflows
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first data source is MLB's public Stats API. The ingestion script pulls schedules, boxscores, starter pitching lines, and probable pitchers into normalized CSVs. If the API is unavailable, place normalized CSVs in `data/raw/` and point config values to those files.

## Expected Data

The modeling pipeline expects pitcher game logs with at least:

```text
game_date, pitcher_id, pitcher_name, team, opponent, is_home,
strikeouts, walks, hits_allowed, innings_pitched
```

Optional odds CSVs should include:

```text
game_date, pitcher_id, market, line, over_odds, under_odds
```

Where `market` is one of:

```text
strikeouts, walks, hits_allowed
```

American odds are expected for `over_odds` and `under_odds`.

## Commands

Fetch starter pitcher game logs for training/backtesting:

```bash
python scripts/fetch_mlb_data.py logs --config config/config.yaml --start 2022-01-01 --end 2025-12-31
```

The `team` and `opponent` fields are MLB team IDs so historical logs and probable-pitcher rows join consistently.
Pitcher logs also include pitch count, strikes, and batters faced when fetched with the current script.

Fetch team batting game logs for opponent-strength features:

```bash
python scripts/fetch_mlb_data.py batting --config config/config.yaml --start 2022-01-01 --end 2025-12-31
```

Fetch game context logs for handedness, park, weather, umpire, and lineup-handedness features:

```bash
python scripts/fetch_mlb_data.py context --config config/config.yaml --start 2022-01-01 --end 2025-12-31
```

Faster combined fetch for both files:

```bash
python scripts/fetch_mlb_data.py extras --config config/config.yaml --start 2022-01-01 --end 2025-12-31
```

Fetch batter-level game logs for lineup aggregate K/BB/hit features:

```bash
python scripts/fetch_mlb_data.py batters --config config/config.yaml --start 2022-01-01 --end 2025-12-31 --max-workers 4
```

Fetch Statcast pitcher daily pitch-quality aggregates:

```bash
python scripts/fetch_statcast.py --config config/config.yaml --start 2022-01-01 --end 2025-12-31
```

Fetch Baseball Savant Statcast park factors:

```bash
python scripts/fetch_park_factors.py --config config/config.yaml --start-year 2021 --end-year 2025 --rolling 3
```

Fetch probable pitchers for a projection date:

```bash
python scripts/fetch_mlb_data.py probables --config config/config.yaml --date 2026-06-01
```

Fetch current pitcher prop odds after probable pitchers are available:

```bash
python scripts/fetch_odds.py --config config/config.yaml --date 2026-06-01
```

This reads `ODDS_API_KEY` from `.env`, appends raw snapshots to `data/odds/odds_snapshots.csv`, and writes current best lines to `data/odds/pitcher_props.csv`.

Scrape current pitcher prop odds from supported sportsbook pages/endpoints:

```bash
python scripts/scrape_daily_odds.py --config config/config.yaml --date 2026-06-01 --sportsbooks draftkings
```

The scraper appends snapshots to `data/odds/odds_snapshots.csv` and overwrites `data/odds/pitcher_props.csv` with the latest best lines, so `project_daily.py` can calculate EV against scraped lines. DraftKings is the first implemented adapter. FanDuel, BetMGM, Caesars, and Pinnacle have adapter slots but need live payload inspection before they should be trusted.

Fetch historical pitcher prop odds for market backtesting:

```bash
python scripts/fetch_historical_odds.py --config config/config.yaml --start 2023-05-03 --end 2025-12-31 --snapshot-hours-before 4 --bookmakers draftkings --resume
```

The Odds API historical player props begin on 2023-05-03, so 2022 pitcher prop lines are not available from this source. Use `--dry-run` first to verify access and event counts without saving odds rows:

```bash
python scripts/fetch_historical_odds.py --config config/config.yaml --start 2025-05-31 --end 2025-05-31 --dry-run
```

Train 2022-2024 models:

```bash
python scripts/train.py --config config/config.yaml
```

Backtest on 2025:

```bash
python scripts/backtest.py --config config/config.yaml
```

Backtest 2025 model edges against fetched historical prop odds:

```bash
python scripts/backtest_historical_odds.py --config config/config.yaml --min-edge-pct 2
```

Generate today's projections:

```bash
python scripts/project_daily.py --config config/config.yaml
```

Generate projections for a specific date:

```bash
python scripts/project_daily.py --config config/config.yaml --date 2026-06-01
```

## Automation

Run the daily projection workflow manually:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_daily.ps1 -Date 2026-06-01
```

Run a full data refresh, retrain, and backtest manually:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_retrain.ps1 -Start 2022-01-01 -End 2025-12-31
```

Recommended schedule:

- Daily morning: `scripts\run_daily.ps1`
- Weekly or monthly: `scripts\run_retrain.ps1`
- Every 5-15 minutes during the MLB slate: run `scripts\scrape_daily_odds.py` to save odds snapshots for CLV tracking.

Windows Task Scheduler example for daily projections:

```powershell
$project = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$project\scripts\run_daily.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00am
Register-ScheduledTask -TaskName "MLB Pitcher Props Daily" -Action $action -Trigger $trigger -Description "Fetch probables and generate MLB pitcher prop projections"
```

For betting and CLV work, daily projections are only half of the automation. The next step is an odds snapshot job that stores current lines repeatedly before first pitch, then compares your bet price to the closing price.

## MVP Model

- Separate model per target:
  - `strikeouts`
  - `walks`
  - `hits_allowed`
- Baseline estimator:
  - `RandomForestRegressor` by default
  - XGBoost can be enabled if installed and selected in config
- Features:
  - pitcher rolling means over 3, 5, and 10 starts
  - rolling innings workload
  - pitch count, strikes, batters faced, strike rate, and batters faced per inning
  - pitcher K/BB/hits rates per inning and per batter faced
  - opponent rolling pitcher-stat environment from historical logs
  - opponent rolling batting form: strikeout rate, walk rate, hit rate, runs, and plate appearances
  - opponent confirmed-lineup batter prior K rate, BB rate, hit rate, and PA volume
  - expected innings, pitches, and batters faced from opportunity models when targets are available
  - Statcast pitch-quality aggregates: velocity, CSW rate, swinging-strike rate, zone rate, and pitch mix
  - pitcher handedness
  - venue ID, temperature, wind speed, and home plate umpire ID when available
  - true Baseball Savant park factors for runs, hits, walks, strikeouts, HR, 1B, 2B, and 3B
  - historical venue and umpire rolling tendencies from prior games
  - opponent confirmed-lineup handedness counts when available
  - home/away flag
  - days rest

## Outputs

- Trained models: `data/processed/models/`
- Backtest predictions: `data/processed/backtest_predictions.csv`
- Daily projections: `data/exports/daily_pitcher_props_YYYY-MM-DD.csv`

Daily picks include:

```text
projection, fair_over_odds, fair_under_odds, edge_pct, ev, kelly_fraction
```

When odds are missing, projection and fair odds are still exported, while EV and Kelly fields are left blank.

## TODOs

- Add weather features: temperature, wind speed/direction, humidity.
- Convert raw weather, umpire, and venue fields into richer historical factors.
- Add umpire strike-zone and walk/strikeout tendencies.
- Add projected lineups before confirmed lineups post.
- Add park-factor handedness splits and roof/condition-specific park factors.
- Add historical odds movement and close-line value tracking.
- Replace baseline residual assumptions with calibrated distribution models.
- Daily fair odds use backtest calibration with conservative residuals and probability shrinkage.
- Add sportsbook-specific limits and bankroll state.
- Add model registry and experiment tracking.
- Add richer Statcast ingestion once source columns are confirmed.
