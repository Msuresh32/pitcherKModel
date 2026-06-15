param(
    [string]$Config = "config\config.yaml",
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Fetching probable pitchers for $Date"
& $Python scripts\fetch_mlb_data.py probables --config $Config --date $Date

Write-Host "Generating projections for $Date"
& $Python scripts\project_daily.py --config $Config --date $Date

Write-Host "Daily automation complete."
