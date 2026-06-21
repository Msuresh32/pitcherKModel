@echo off
cd /d "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
echo Starting pipeline at %date% %time% > logs\pipeline_june21.txt

echo ========================================
echo  Step 1: Running backtest through June 21
echo ========================================
echo [Step 1] backtest starting... >> logs\pipeline_june21.txt
python -u scripts/backtest.py --blend 0.7,0.3 --min-edge 12 --edge-shrink 0.7 --closing-odds data/odds/june_2026_odds.csv >> logs\pipeline_june21.txt 2>&1
echo [Step 1] backtest exit code: %errorlevel% >> logs\pipeline_june21.txt

echo ========================================
echo  Step 2: CLV computation
echo ========================================
echo [Step 2] CLV starting... >> logs\pipeline_june21.txt
python -u scripts/compute_live_clv.py >> logs\pipeline_june21.txt 2>&1
echo [Step 2] CLV exit code: %errorlevel% >> logs\pipeline_june21.txt

echo ========================================
echo  Step 3: Rebuilding dashboard
echo ========================================
echo [Step 3] dashboard starting... >> logs\pipeline_june21.txt
python -u generate_dashboard.py >> logs\pipeline_june21.txt 2>&1
if %errorlevel% neq 0 (
    python -u scripts/dashboard.py >> logs\pipeline_june21.txt 2>&1
)
echo [Step 3] dashboard exit code: %errorlevel% >> logs\pipeline_june21.txt

echo ========================================
echo  Step 4: Git commit and push
echo ========================================
echo [Step 4] git commit/push starting... >> logs\pipeline_june21.txt
git add -A >> logs\pipeline_june21.txt 2>&1
git commit -m "Update backtest and CLV through June 21" >> logs\pipeline_june21.txt 2>&1
powershell -ExecutionPolicy Bypass -File push.ps1 >> logs\pipeline_june21.txt 2>&1
echo [Step 4] push exit code: %errorlevel% >> logs\pipeline_june21.txt

echo Pipeline complete at %date% %time% >> logs\pipeline_june21.txt
echo DONE - see logs\pipeline_june21.txt for full output
pause
