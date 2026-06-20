$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$Today       = (Get-Date).ToString("yyyy-MM-dd")
$LogFile     = "$ProjectRoot\logs\pregame_odds_$Today.log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

function Log($msg) { "[$((Get-Date).ToString('HH:mm:ss'))] $msg" | Tee-Object -Append -FilePath $LogFile }

$hour = (Get-Date).Hour
if ($hour -lt 11 -or $hour -ge 22) {
    Log "Outside 11am-10pm window, skipping."
    exit 0
}

Log "Running pregame odds capture..."
& $Python scripts\fetch_pregame_odds.py --date $Today 2>&1 | Tee-Object -Append -FilePath $LogFile
Log "Done."
