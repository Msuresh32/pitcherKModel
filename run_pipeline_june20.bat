@echo off
cd /d "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
set PYTHON="C:\Users\Mani Suresh\anaconda3\python.exe"
set GIT="C:\Users\Mani Suresh\AppData\Local\GitHubDesktop\app-3.1.3\resources\app\git\cmd\git.exe"
set LOG=logs\pipeline_june20.txt
mkdir logs 2>nul
echo Starting pipeline at %date% %time% > %LOG%

echo ========================================
echo  Step 1: Fetch MLB game logs for June 20
echo ========================================
echo [Step 1] Fetching MLB pitcher game logs for 2026-06-20... >> %LOG%
%PYTHON% -u scripts/fetch_mlb_data.py logs --start 2026-06-20 --end 2026-06-20 >> %LOG% 2>&1
echo [Step 1] exit code: %errorlevel% >> %LOG%

echo ========================================
echo  Step 2: Fetch historical closing odds June 17-20
echo ========================================
echo [Step 2] Fetching closing odds 2026-06-17 to 2026-06-20... >> %LOG%
%PYTHON% -u scripts/fetch_historical_odds.py --start 2026-06-17 --end 2026-06-20 --output data/odds/june_2026_odds.csv --resume >> %LOG% 2>&1
echo [Step 2] exit code: %errorlevel% >> %LOG%

echo ========================================
echo  Step 3: Resolve June 20 picks
echo ========================================
echo [Step 3] Resolving picks for 2026-06-20... >> %LOG%
%PYTHON% -u scripts/resolve_picks.py --date 2026-06-20 >> %LOG% 2>&1
echo [Step 3] exit code: %errorlevel% >> %LOG%

echo ========================================
echo  Step 4: Backtest (blend 0.7/0.3, min-edge 12, edge-shrink 0.7)
echo ========================================
echo [Step 4] Running backtest... >> %LOG%
%PYTHON% -u scripts/backtest.py --blend 0.7,0.3 --min-edge 12 --edge-shrink 0.7 --closing-odds data/odds/june_2026_odds.csv >> %LOG% 2>&1
echo [Step 4] exit code: %errorlevel% >> %LOG%

echo ========================================
echo  Step 5: Live CLV analysis
echo ========================================
echo [Step 5] Running CLV analysis... >> %LOG%
%PYTHON% -u scripts/compute_live_clv.py >> %LOG% 2>&1
echo [Step 5] exit code: %errorlevel% >> %LOG%

echo ========================================
echo  Step 6: Rebuild dashboard
echo ========================================
echo [Step 6] Rebuilding dashboard... >> %LOG%
%PYTHON% -u generate_dashboard.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    %PYTHON% -u scripts/dashboard.py >> %LOG% 2>&1
)
echo [Step 6] exit code: %errorlevel% >> %LOG%

echo ========================================
echo  Step 7: Git commit and push
echo ========================================
echo [Step 7] Committing and pushing... >> %LOG%
if exist .git\index.lock del .git\index.lock
if exist .git\HEAD.lock del .git\HEAD.lock
%GIT% add -A >> %LOG% 2>&1
%GIT% commit -m "Resolve June 20 picks and update backtest" >> %LOG% 2>&1
powershell -ExecutionPolicy Bypass -File push.ps1 >> %LOG% 2>&1
echo [Step 7] exit code: %errorlevel% >> %LOG%

echo. >> %LOG%
echo Pipeline complete at %date% %time% >> %LOG%
echo.
echo DONE - check logs\pipeline_june20.txt for full output
pause
