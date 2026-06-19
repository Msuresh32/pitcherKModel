# 2am daily — rebuild picks.html dashboard with any new results from yesterday
$Python      = "C:\Users\Mani Suresh\anaconda3\python.exe"
$ProjectRoot = "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
$LogFile     = "$ProjectRoot\logs\task_2am_$(Get-Date -Format 'yyyy-MM-dd').log"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force "$ProjectRoot\logs" | Out-Null

"[$(Get-Date)] 2am dashboard task started" | Tee-Object -FilePath $LogFile

& $Python generate_dashboard.py 2>&1 | Tee-Object -Append -FilePath $LogFile

"[$(Get-Date)] Done." | Tee-Object -Append -FilePath $LogFile
