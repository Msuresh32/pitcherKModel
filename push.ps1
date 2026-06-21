# Push commit to GitHub using GitHub Desktop's git
$gitExe = "C:\Users\Mani Suresh\AppData\Local\GitHubDesktop\app-3.1.3\resources\app\git\cmd\git.exe"
if (Test-Path $gitExe) {
    & $gitExe push origin main
} else {
    git push origin main
}
