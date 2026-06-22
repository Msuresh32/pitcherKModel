$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Git         = "C:\Users\Mani Suresh\AppData\Local\GitHubDesktop\app-3.1.3\resources\app\git\cmd\git.exe"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$Yesterday   = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\daily_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

function Log($msg) { "[$((Get-Date).ToString('HH:mm:ss'))] $msg" | Tee-Object -Append -FilePath $LogFile }
function Ntfy($title, $body, $priority="default", $tags="baseball") {
    try { Invoke-RestMethod -Method Post -Uri "https://ntfy.sh/pitcher-model-mani" -Body $body -Headers @{"Title"=$title;"Priority"=$priority;"Tags"=$tags} | Out-Null }
    catch { Log "ntfy error: $_" }
}

Log "7am daily pipeline started"

& $Python scripts\resolve_picks.py --date $Yesterday 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_mlb_data.py extras --start $Yesterday --end $Yesterday 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_mlb_data.py batters --start $Yesterday --end $Yesterday --max-workers 4 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_statcast.py --start $Yesterday --end $Yesterday 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_probables_daily.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_pregame_lineups.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_odds_daily.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\project_daily.py --date $Today --config config/config_poisson.yaml --label poisson 2>&1 | Tee-Object -Append -FilePath $LogFile

Log "Running backtest (Poisson model, min-edge 20)..."
& $Python scripts\backtest.py --config config/config_poisson.yaml --min-edge 20 --closing-odds data/odds/closing_odds_master.csv 2>&1 | Tee-Object -Append -FilePath $LogFile

Log "Running live CLV analysis..."
& $Python scripts\compute_live_clv.py 2>&1 | Tee-Object -Append -FilePath $LogFile

& $Python generate_dashboard.py 2>&1 | Tee-Object -Append -FilePath $LogFile

& $Git -C $ProjectRoot add -A 2>&1 | Out-Null
& $Git -C $ProjectRoot commit -m "Daily pipeline $Today" 2>&1 | Out-Null
& $Git -C $ProjectRoot push origin main 2>&1 | Out-Null

Ntfy "Pipeline Done - $Today" "7am pipeline complete. Lineups + odds fetched. Check dashboard for picks." "default" "baseball,sun_with_face"
Log "Done."
