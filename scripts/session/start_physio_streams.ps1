# Start EDA and HR LSL streamers on the EEG laptop (inside Faraday cage).
# All three files (this script, stream_shimmer_eda.py, stream_polar_hr.py)
# must be in the same folder.

$ScriptDir = $PSScriptRoot
$EdaScript = Join-Path $ScriptDir "stream_shimmer_eda.py"
$HrScript  = Join-Path $ScriptDir "stream_polar_hr.py"

$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Host "ERROR: python not found. Install Python or add it to PATH." -ForegroundColor Red
    Read-Host "Press ENTER to exit"
    exit 1
}

Write-Host ""
Write-Host "EEG LAPTOP -- PHYSIO STREAM LAUNCHER" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Shimmer EDA streamer in background..." -ForegroundColor White
$EdaJob = Start-Process $PythonExe -ArgumentList "`"$EdaScript`" --auto-port" -NoNewWindow -PassThru
Write-Host "EDA streamer PID: $($EdaJob.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "Starting Polar H10 HR streamer (foreground)..." -ForegroundColor White
Write-Host "Press Ctrl+C to stop both streamers." -ForegroundColor Gray
Write-Host ""

try {
    & $PythonExe $HrScript
} finally {
    Write-Host ""
    Write-Host "Stopping EDA streamer..." -ForegroundColor Yellow
    if (-not $EdaJob.HasExited) {
        Stop-Process -Id $EdaJob.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Streamers stopped." -ForegroundColor Yellow
}