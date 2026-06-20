$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\lineups_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

function Log($msg) { "[$((Get-Date).ToString('HH:mm:ss'))] $msg" | Tee-Object -Append -FilePath $LogFile }
function Ntfy($title, $body, $priority="default", $tags="baseball") {
    try { Invoke-RestMethod -Method Post -Uri "https://ntfy.sh/pitcher-model-mani" -Body $body -Headers @{"Title"=$title;"Priority"=$priority;"Tags"=$tags} | Out-Null }
    catch { Log "ntfy error: $_" }
}

Log "9am lineup check"
& $Python scripts\fetch_pregame_lineups.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile

$ProbPath   = "$ProjectRoot\data\raw\probable_pitchers.csv"
$LineupPath = "$ProjectRoot\data\raw\today_lineups.csv"

if ((Test-Path $ProbPath) -and (Test-Path $LineupPath)) {
    $prob    = Import-Csv $ProbPath | Where-Object { $_.game_date -like "$Today*" }
    $lineups = Import-Csv $LineupPath | Where-Object { $_.game_date -like "$Today*" }

    $lines = $prob | ForEach-Object {
        $opp = $_.opponent
        $confirmed = ($lineups | Where-Object { $_.team -eq $opp }).Count
        $status = if ($confirmed -ge 7) { "LOCKED $confirmed/9" } else { "pending $confirmed/9" }
        "$($_.pitcher_name) vs $opp`: $status"
    }

    $body = "TODAY lineup status:`n" + ($lines -join "`n") + "`nFull rerun at 11:30am ET"
    Ntfy "MLB Lineups 9am" $body "default" "baseball"
    Log "Sent lineup status"
} else {
    Ntfy "MLB Lineups 9am" "No lineups posted yet - checking again at 11:30am ET" "low" "baseball"
    Log "No lineup data found"
}
