@echo off
cd /d "C:\Users\Mani Suresh\Downloads\Pitcher-Model"
set GIT="C:\Users\Mani Suresh\AppData\Local\GitHubDesktop\app-3.1.3\resources\app\git\cmd\git.exe"
if exist .git\index.lock del .git\index.lock
if exist .git\HEAD.lock del .git\HEAD.lock
if exist .git\objects\maintenance.lock del .git\objects\maintenance.lock
%GIT% add -A
%GIT% commit -m "Update backtest and CLV through June 21"
powershell -ExecutionPolicy Bypass -File push.ps1
pause
