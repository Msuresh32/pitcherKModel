# 11pm daily - pull initial lines + probables for tomorrow
$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Tomorrow    = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\task_11pm_$((Get-Date).ToString('yyyy-MM-dd')).log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

"[$(Get-Date)] 11pm lines task started - fetching for $Tomorrow" | Tee-Object -FilePath $LogFile

# 1. Probable pitchers for tomorrow
& $Python scripts\fetch_probables_daily.py --date $Tomorrow 2>&1 | Tee-Object -Append -FilePath $LogFile

# 2. Initial odds read (early lines - labelled morning since its tomorrows first look)
& $Python scripts\fetch_odds_daily.py --date $Tomorrow --snapshot morning 2>&1 | Tee-Object -Append -FilePath $LogFile

# 3. Early projections so you have a first read overnight
& $Python scripts\project_daily.py --config config\config.yaml --date $Tomorrow 2>&1 | Tee-Object -Append -FilePath $LogFile

"[$(Get-Date)] Done." | Tee-Object -Append -FilePath $LogFile
