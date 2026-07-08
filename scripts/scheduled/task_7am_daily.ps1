$Python      = "py"
$PyArgs      = "-3.14"
$ProjectRoot = "C:\Users\shado\pitcherKModel"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$Yesterday   = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\daily_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

function Log($msg) { "[$((Get-Date).ToString('HH:mm:ss'))] $msg" | Tee-Object -Append -FilePath $LogFile }
function Ntfy($title, $body, $priority="default", $tags="baseball") {
    try { Invoke-RestMethod -Method Post -Uri "https://ntfy.sh/pitcher-model-mani" -Body $body `
        -Headers @{"Title"=$title;"Priority"=$priority;"Tags"=$tags} | Out-Null }
    catch { Log "ntfy error: $_" }
}

Log "=== 7am pipeline started — $Today ==="

# ── 1. Settle yesterday's bets ────────────────────────────────────────────────
Log "Settling results for $Yesterday..."
& $Python $PyArgs scripts\update_results.py --date $Yesterday 2>&1 | Tee-Object -Append -FilePath $LogFile

# ── 2. Fetch yesterday's game data ────────────────────────────────────────────
Log "Fetching game data for $Yesterday..."
& $Python $PyArgs scripts\fetch_mlb_data.py extras  --start $Yesterday --end $Yesterday 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python $PyArgs scripts\fetch_mlb_data.py batters --start $Yesterday --end $Yesterday --max-workers 4 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python $PyArgs scripts\fetch_statcast.py --start $Yesterday --end $Yesterday 2>&1 | Tee-Object -Append -FilePath $LogFile

# ── 3. Fetch today's pregame data ─────────────────────────────────────────────
Log "Fetching probables, lineups, odds for $Today..."
& $Python $PyArgs scripts\fetch_probables_daily.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python $PyArgs scripts\fetch_pregame_lineups.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python $PyArgs scripts\fetch_odds_daily.py      --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

# ── 4. Run model ──────────────────────────────────────────────────────────────
Log "Running model for $Today..."
& $Python $PyArgs scripts\project_daily.py --date $Today --config config\config_v4_production.yaml 2>&1 | Tee-Object -Append -FilePath $LogFile

# ── 5. Generate V4 candidates + log ──────────────────────────────────────────
Log "Generating picks..."
& $Python $PyArgs scripts\daily_runner.py --date $Today --config config\config_v4_production.yaml --skip-model 2>&1 | Tee-Object -Append -FilePath $LogFile

# ── 6. Notify with today's picks ─────────────────────────────────────────────
$SummaryFile = "$ProjectRoot\reports\daily\${Today}_summary.md"
if (Test-Path $SummaryFile) {
    $Body = (Get-Content $SummaryFile -Raw).Substring(0, [Math]::Min(3000, (Get-Content $SummaryFile -Raw).Length))
} else {
    $Body = "Pipeline complete. Check reports\daily\${Today}_summary.md"
}
Ntfy "Picks Ready — $Today" $Body "high" "baseball,dart"

Log "=== Done. ==="
