<#
.SYNOPSIS
    Start EDA and HR LSL streamers on the EEG laptop (inside Faraday cage).

.DESCRIPTION
    Run this script on the EEG laptop BEFORE starting the session on the main PC.
    It launches two background streamers that publish LSL outlets visible to the
    main PC via the existing Ethernet connection:

      - ShimmerEDA  (type=EDA)  — Shimmer GSR3 via Bluetooth
      - PolarHR/RR/ECG          — Polar H10 via BLE

    Both streamers stay alive until you close this window (Ctrl+C).

.NOTES
    Prerequisites on the EEG laptop:
      - Shimmer GSR3 paired via Bluetooth (check COM port in Device Manager)
      - Polar H10 worn by participant (BLE adapter enabled)
      - Project venv active: .\.venv\Scripts\Activate.ps1
      - Dependencies installed: pylsl, pyshimmer, pyserial, bleak
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent -ErrorAction SilentlyContinue
if (-not $RepoRoot) { $RepoRoot = Split-Path $PSScriptRoot -Parent }

$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
} else {
    Write-Host "WARNING: venv not found at $VenvActivate — using system Python" -ForegroundColor Yellow
}

$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Host "ERROR: python not found. Activate venv or add Python to PATH." -ForegroundColor Red
    Read-Host "Press ENTER to exit"
    exit 1
}

$EdaScript = Join-Path $RepoRoot "scripts\session\stream_shimmer_eda.py"
$HrScript  = Join-Path $RepoRoot "scripts\session\stream_polar_hr.py"

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host "  EEG LAPTOP — PHYSIO STREAM LAUNCHER" -ForegroundColor Cyan
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Starting Shimmer EDA streamer in background..." -ForegroundColor White
$EdaJob = Start-Process $PythonExe -ArgumentList "`"$EdaScript`" --auto-port" -NoNewWindow -PassThru
Write-Host "  EDA streamer PID: $($EdaJob.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "  Starting Polar H10 HR streamer (foreground)..." -ForegroundColor White
Write-Host "  Press Ctrl+C to stop both streamers." -ForegroundColor DarkGray
Write-Host ""

try {
    & $PythonExe $HrScript
} finally {
    Write-Host ""
    Write-Host "  Stopping EDA streamer..." -ForegroundColor Yellow
    if (-not $EdaJob.HasExited) {
        Stop-Process -Id $EdaJob.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  Streamers stopped." -ForegroundColor Yellow
}
