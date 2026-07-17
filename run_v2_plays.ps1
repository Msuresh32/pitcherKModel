# V2 daily plays pipeline - refresh data, fetch morning board, score, size stakes.
# Scheduled for ~8:45 AM daily (before games; T-12-style morning board).
# Output: reports\daily\v2_plays_<date>.csv + v2_plays_latest.html + full board CSV.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$py = if ($env:PYTHON_PATH) { $env:PYTHON_PATH } else { "py" }
$today = Get-Date -Format "yyyy-MM-dd"
$yday  = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
$log   = "logs\v2_plays_$($today).log"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

function Step($name, $cmd) {
    "=== $name - $(Get-Date -Format HH:mm:ss) ===" | Tee-Object -FilePath $log -Append
    & $py -3.14 $cmd.Split(" ") 2>&1 | Tee-Object -FilePath $log -Append | Select-Object -Last 3
}

# 1. settle yesterday: game logs + team batting/context
Step "pitcher logs ($yday)"  "scripts/fetch_mlb_data.py logs --start $yday --end $yday"
Step "extras ($yday)"        "scripts/fetch_mlb_data.py extras --start $yday --end $yday"
Step "statcast ($yday)"      "scripts/fetch_statcast.py --start $yday --end $yday"

# 2. today's slate
Step "probables ($today)"    "scripts/fetch_probables_daily.py --date $today"
Step "odds Ks ($today)"      "scripts/fetch_odds_daily.py --date $today --snapshot morning"
Step "odds hits ($today)"    "scripts/fetch_odds_daily.py --date $today --snapshot morning --markets pitcher_hits_allowed --output data/odds/pitcher_props_hits.csv"
Step "novig quotes ($today)" "research/v2/a9_novig_quotes.py"

# 3. score + size
"=== score ($today) - $(Get-Date -Format HH:mm:ss) ===" | Tee-Object -FilePath $log -Append
$env:SCORE_DATE = $today
$env:BANKROLL = if ($env:V2_BANKROLL) { $env:V2_BANKROLL } else { "10000" }
& $py -3.14 research/v2/92_today.py 2>&1 | Tee-Object -FilePath $log -Append

# 4. qualitative context + rebuild artifact HTML + push to GitHub (feeds the
#    cloud routine that writes per-play analyses and republishes the artifact)
"=== qual context + artifact build + push - $(Get-Date -Format HH:mm:ss) ===" | Tee-Object -FilePath $log -Append
& $py -3.14 research/v2/a7_qual_brief.py --all-props 2>&1 | Tee-Object -FilePath $log -Append | Select-Object -Last 1
& $py -3.14 research/v2/a8_hits_qual.py 2>&1 | Tee-Object -FilePath $log -Append | Select-Object -Last 1
& $py -3.14 research/v2/a6_build_artifact.py 2>&1 | Tee-Object -FilePath $log -Append
git add reports/daily reports/artifact data/daily/v2_predictions_log.csv 2>&1 | Out-Null
git commit -m "daily plays $today [auto]" 2>&1 | Tee-Object -FilePath $log -Append | Select-Object -Last 1
git push origin main 2>&1 | Tee-Object -FilePath $log -Append | Select-Object -Last 1

"done - $(Get-Date -Format HH:mm:ss)" | Tee-Object -FilePath $log -Append
