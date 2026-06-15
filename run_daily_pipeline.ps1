# Set PYTHON_PATH to your Python executable if it is not on your system PATH.
# Examples:
#   $python = "C:\Users\YourName\anaconda3\python.exe"
#   $python = "C:\Python311\python.exe"
$python    = if ($env:PYTHON_PATH) { $env:PYTHON_PATH } else { "python" }
$dir       = $PSScriptRoot
$today     = Get-Date -Format "yyyy-MM-dd"
$yesterday = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
$log       = "$dir\logs\pipeline_$today.log"

New-Item -ItemType Directory -Force -Path "$dir\logs" | Out-Null

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $log -Append
}

Set-Location $dir
Log "=== Daily pipeline starting for $today ==="

# 1. Resolve yesterday's picks (fetch game logs + update picks_log + rebuild backtest)
Log "Resolving picks for $yesterday..."
& $python "$dir\scripts\resolve_picks.py" --date $yesterday 2>&1 | ForEach-Object { Log $_ }

# 2. Fetch today's probable pitchers
Log "Fetching probable pitchers for $today..."
& $python "$dir\scripts\fetch_probables_daily.py" --date $today 2>&1 | ForEach-Object { Log $_ }

# 3. Fetch today's odds
Log "Fetching odds for $today..."
& $python "$dir\scripts\fetch_odds_daily.py" --date $today 2>&1 | ForEach-Object { Log $_ }

# 4. Run projections (saves picks + updates picks_log)
Log "Running projections for $today..."
& $python "$dir\scripts\project_daily.py" --date $today 2>&1 | ForEach-Object { Log $_ }

# 5. Regenerate dashboard
Log "Regenerating dashboard..."
& $python "$dir\generate_dashboard.py" 2>&1 | ForEach-Object { Log $_ }

Log "=== Pipeline complete ==="
