# MLB Pitcher Strikeout Props Model

Ensemble model (60% Poisson GLM + 40% XGBoost Poisson) that projects MLB pitcher strikeout totals, computes edge against market lines, and flags +EV bets daily.

**2026 live results:** 335 settled bets · 179W/156L · +9.0% ROI  
**2025 backtest:** 770 bets · 374W/396L · +2.4% ROI

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Msuresh32/pitcherKModel.git
cd pitcherKModel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the project root:

```
ODDS_API_KEY=your_key_here
```

Get a free key at [the-odds-api.com](https://the-odds-api.com). The free tier gives ~500 requests/month which covers daily use.

### 3. Fetch historical data (one-time setup)

```bash
# Pitcher game logs 2022-2025
python scripts/fetch_mlb_data.py logs --start 2022-01-01 --end 2025-12-31

# Opponent batting logs
python scripts/fetch_mlb_data.py extras --start 2022-01-01 --end 2025-12-31

# Batter-level logs (for lineup matchup features)
python scripts/fetch_mlb_data.py batters --start 2022-01-01 --end 2025-12-31 --max-workers 4

# Statcast pitch quality
python scripts/fetch_statcast.py --start 2022-01-01 --end 2025-12-31

# Park factors
python scripts/fetch_park_factors.py --start-year 2021 --end-year 2025 --rolling 3
```

### 4. Train the model

```bash
python scripts/train.py
```

Trains on 2022-2024 data. Takes ~5-10 minutes. Saves models to `data/processed/models/`.

### 5. Run today's slate

```bash
# Fetch today's probable pitchers
python scripts/fetch_probables_daily.py

# Fetch current odds
python scripts/fetch_odds_daily.py

# Generate projections + flag bets with edge >= 7%
python scripts/project_daily.py

# Build dashboard
python generate_dashboard.py
# Open dashboard.html in your browser
```

---

## Daily Automation (Windows Task Scheduler)

The script `run_daily_pipeline.ps1` runs the full pipeline automatically each morning:
1. Resolves yesterday's picks (fetches game logs, records W/L)
2. Fetches today's probable pitchers
3. Fetches today's odds
4. Runs projections and saves the picks log
5. Regenerates the dashboard

To schedule it at 11:03 AM daily:

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
             -Argument "-NonInteractive -WindowStyle Hidden -File `"$PWD\run_daily_pipeline.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At "11:03AM"
Register-ScheduledTask -TaskName "MLBPitcherPipeline" -Action $action -Trigger $trigger
```

Set the `PYTHON_PATH` environment variable if Python is not on your system PATH:

```powershell
$env:PYTHON_PATH = "C:\Users\YourName\anaconda3\python.exe"
```

---

## How It Works

### Model

- **Ensemble:** 60% Poisson GLM + 40% XGBoost Poisson
- **Features:** pitcher rolling stats (3/5/10/20 games), opponent batting environment, Statcast pitch quality (velocity, CSW, SwStr%), park factors, umpire tendencies, handedness, rest days
- **Probability:** Poisson CDF on the projected lambda with 50% shrinkage toward 50% to avoid overconfidence
- **Edge filter:** 7%+ EV, direction-agreement required (model must agree with bet direction)

### Edge Formula

```
EV = P(hit) × (decimal_odds − 1) − P(miss)
edge_pct = EV × 100
```

Break-even at +110 = 47.6%. The model flags a bet only when its shrunk probability exceeds the bookmaker's implied probability by 7%+.

### Key Files

| File | Purpose |
|---|---|
| `scripts/project_daily.py` | Daily projection + picks log |
| `scripts/resolve_picks.py` | Resolve prior day W/L from game logs |
| `scripts/fetch_probables_daily.py` | Fetch today's probable starters |
| `scripts/fetch_odds_daily.py` | Fetch today's prop odds |
| `generate_dashboard.py` | Build self-contained HTML dashboard |
| `data/exports/picks_log.csv` | Canonical record of every flagged bet |
| `data/exports/2026_backtest_extended.csv` | 2026 live results |
| `data/exports/2025_backtest.csv` | 2025 backtest results |
| `config/config.yaml` | All model/betting parameters |

---

## Project Structure

```
src/
  data/          # MLB Stats API + Statcast data fetchers
  features/      # Feature engineering
  models/        # Poisson GLM + XGBoost ensemble, calibration
  odds/          # The Odds API client, EV/Kelly math
  export/        # CSV + Excel exporters
scripts/         # CLI entry points
config/          # YAML config
data/
  raw/           # Source CSVs (gitignored — fetch with scripts above)
  processed/     # Trained models + calibration (gitignored)
  exports/       # Picks log + backtest results (committed)
  odds/          # Daily odds snapshots (gitignored)
```

---

## Results

| Season | Bets | W | L | ROI |
|--------|------|---|---|-----|
| 2025 (backtest) | 770 | 374 | 396 | +2.4% |
| 2026 (live, through Jun 14) | 335 | 179 | 156 | +9.0% |

Edge buckets (2026): 20%+ edge → +33% ROI · 7-12% edge → +10% ROI
