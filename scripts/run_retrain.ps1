param(
    [string]$Config = "config\config.yaml",
    [string]$Start = "2022-01-01",
    [string]$End = (Get-Date -Format "yyyy-MM-dd"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Fetching MLB pitcher game logs from $Start to $End"
& $Python scripts\fetch_mlb_data.py logs --config $Config --start $Start --end $End

Write-Host "Fetching MLB team batting and game context logs from $Start to $End"
& $Python scripts\fetch_mlb_data.py extras --config $Config --start $Start --end $End

Write-Host "Fetching MLB batter game logs from $Start to $End"
& $Python scripts\fetch_mlb_data.py batters --config $Config --start $Start --end $End

Write-Host "Training models"
& $Python scripts\train.py --config $Config

Write-Host "Running backtest"
& $Python scripts\backtest.py --config $Config

Write-Host "Retrain automation complete."
