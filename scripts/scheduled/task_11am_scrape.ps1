# 11am daily — final pre-game odds scrape + rebuild Excel dashboard
$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\task_11am_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

"[$(Get-Date)] 11am scrape task started — $Today" | Tee-Object -FilePath $LogFile

# 1. Final odds snapshot before games start (closing lines)
& $Python scripts\fetch_odds_daily.py --date $Today --snapshot closing 2>&1 | Tee-Object -Append -FilePath $LogFile

# 2. Re-run projections with updated closing odds
& $Python scripts\project_daily.py --config config\config.yaml --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

# 3. Build clean daily picks Excel
& $Python scripts\_build_daily_excel.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

# 4. Rebuild picks.html dashboard with latest picks
& $Python generate_dashboard.py 2>&1 | Tee-Object -Append -FilePath $LogFile

"[$(Get-Date)] Done." | Tee-Object -Append -FilePath $LogFile
