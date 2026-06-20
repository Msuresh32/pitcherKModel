$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Git         = "C:\Users\Mani Suresh\AppData\Local\GitHubDesktop\app-3.1.3\resources\app\git\cmd\git.exe"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\rerun_530pm_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

function Log($msg) { "[$((Get-Date).ToString('HH:mm:ss'))] $msg" | Tee-Object -Append -FilePath $LogFile }
function Ntfy($title, $body, $priority="urgent", $tags="baseball") {
    try { Invoke-RestMethod -Method Post -Uri "https://ntfy.sh/pitcher-model-mani" -Body $body -Headers @{"Title"=$title;"Priority"=$priority;"Tags"=$tags} | Out-Null }
    catch { Log "ntfy error: $_" }
}

Log "5:30pm final rerun started"
& $Python scripts\fetch_pregame_lineups.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\fetch_odds_daily.py --date $Today --snapshot closing 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python scripts\project_daily.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
& $Python generate_dashboard.py 2>&1 | Tee-Object -Append -FilePath $LogFile

$CsvPath    = "$ProjectRoot\data\exports\daily_pitcher_props_$Today.csv"
$LineupPath = "$ProjectRoot\data\raw\today_lineups.csv"

if (Test-Path $CsvPath) {
    $picks = Import-Csv $CsvPath | Where-Object {
        $_.market -eq "strikeouts" -and [double]($_.edge_pct) -ge 15
    } | Sort-Object { [double]$_.edge_pct } -Descending |
      Group-Object pitcher_name | ForEach-Object { $_.Group[0] }

    $lineups = if (Test-Path $LineupPath) { Import-Csv $LineupPath | Where-Object { $_.game_date -like "$Today*" } } else { @() }

    if ($picks) {
        $lines = $picks | ForEach-Object {
            $opp = $_.opponent
            $confirmed = ($lineups | Where-Object { $_.team -eq $opp }).Count
            $side = $_.best_side.ToUpper()
            $edge = [math]::Round([double]$_.edge_pct, 1)
            $gap  = [math]::Round([double]$_.gap, 2)
            $gap_str = if ($gap -ge 0) { "+$gap" } else { "$gap" }
            "$($_.pitcher_name) $side $($_.line) $($_.recommended_odds) | $edge% | $gap_str | OPP: $confirmed/9"
        }
        $body = "FINAL PICKS ($Today):`n" + ($lines -join "`n") + "`nLast rerun before evening games."
        Ntfy "FINAL Picks $Today - $($picks.Count) total" $body "urgent" "baseball,bell"
        Log "Sent $($picks.Count) final picks"
    } else {
        Ntfy "FINAL Rerun - No Picks" "No 15%+ edge picks for $Today" "default" "baseball"
        Log "No picks at edge>=15"
    }
}

& $Git -C $ProjectRoot add -A 2>&1 | Out-Null
& $Git -C $ProjectRoot commit -m "5:30pm final rerun $Today" 2>&1 | Out-Null
& $Git -C $ProjectRoot push origin main 2>&1 | Out-Null
Log "Done."
