$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Git         = "C:\Users\Mani Suresh\AppData\Local\GitHubDesktop\app-3.1.3\resources\app\git\cmd\git.exe"
$Tomorrow    = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\nightly_$((Get-Date).ToString('yyyy-MM-dd')).log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

function Log($msg) { "[$((Get-Date).ToString('HH:mm:ss'))] $msg" | Tee-Object -Append -FilePath $LogFile }
function Ntfy($title, $body, $priority="default", $tags="baseball") {
    try {
        Invoke-RestMethod -Method Post -Uri "https://ntfy.sh/pitcher-model-mani" -Body $body `
            -Headers @{"Title"=$title;"Priority"=$priority;"Tags"=$tags} | Out-Null
    } catch { Log "ntfy error: $_" }
}

Log "Nightly pipeline started for $Tomorrow"

# 1. Fetch tomorrow's probable pitchers
Log "Fetching probable pitchers..."
& $Python scripts\fetch_probables_daily.py --date $Tomorrow 2>&1 | Tee-Object -Append -FilePath $LogFile

# 2. Fetch early odds for tomorrow (saved as nightly snapshot — 7am run saves the morning/open snapshot)
Log "Fetching early odds..."
& $Python scripts\fetch_odds_daily.py --date $Tomorrow --snapshot nightly 2>&1 | Tee-Object -Append -FilePath $LogFile

# 3. Run projections
Log "Running projections..."
& $Python scripts\project_daily.py --date $Tomorrow 2>&1 | Tee-Object -Append -FilePath $LogFile

# 4. Regenerate dashboard
Log "Regenerating dashboard..."
& $Python generate_dashboard.py 2>&1 | Tee-Object -Append -FilePath $LogFile

# 5. Read picks and build ntfy message
$CsvPath = "$ProjectRoot\data\exports\daily_pitcher_props_$Tomorrow.csv"
if (Test-Path $CsvPath) {
    $picks = Import-Csv $CsvPath | Where-Object {
        $_.market -eq "strikeouts" -and [double]($_.edge_pct) -ge 15
    } | Sort-Object { [double]$_.edge_pct } -Descending |
      Group-Object pitcher_name | ForEach-Object { $_.Group[0] }

    if ($picks) {
        $lines = $picks | ForEach-Object {
            $side = $_.best_side.ToUpper()
            $line = $_.line
            $odds = $_.recommended_odds
            $edge = [math]::Round([double]$_.edge_pct, 1)
            $gap  = [math]::Round([double]$_.gap, 2)
            $gap_str = if ($gap -ge 0) { "+$gap" } else { "$gap" }
            "$($_.pitcher_name) $side $line $odds | $edge% | $gap_str"
        }
        $body = "EARLY PICKS ($Tomorrow) - no lineups yet:`n" + ($lines -join "`n") + "`nLineups post ~9am ET"
        $title = "Early Picks $Tomorrow - $($picks.Count) total"
        Ntfy $title $body "default" "baseball,calendar"
        Log "Sent ntfy: $($picks.Count) picks"
    } else {
        Ntfy "No Early Picks $Tomorrow" "No 15%+ edge picks yet - check after lineups post" "low" "baseball"
        Log "No picks at edge>=15"
    }
} else {
    Log "CSV not found: $CsvPath"
    Ntfy "Nightly Pipeline Error" "CSV not found for $Tomorrow - check logs" "high" "warning"
}

# 6. Git push
Log "Pushing to GitHub..."
& $Git -C $ProjectRoot add -A 2>&1 | Out-Null
& $Git -C $ProjectRoot commit -m "Nightly pipeline $Tomorrow" 2>&1 | Out-Null
& $Git -C $ProjectRoot push origin main 2>&1 | Out-Null

Log "Done."
