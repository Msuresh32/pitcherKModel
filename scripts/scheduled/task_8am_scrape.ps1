# 8am daily — morning odds scrape + run projections (main pick-time)
$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\task_8am_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

"[$(Get-Date)] 8am scrape task started — $Today" | Tee-Object -FilePath $LogFile

# 1. Refresh probables in case any late changes from 11pm run
& $Python scripts\fetch_probables_daily.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

# 2. Morning odds (pick-time lines)
& $Python scripts\fetch_odds_daily.py --date $Today --snapshot morning 2>&1 | Tee-Object -Append -FilePath $LogFile

# 3. Run projections with morning odds
& $Python scripts\project_daily.py --config config\config.yaml --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

# 4. Build clean daily picks Excel (morning version)
& $Python scripts\_build_daily_excel.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

"[$(Get-Date)] Done." | Tee-Object -Append -FilePath $LogFile
