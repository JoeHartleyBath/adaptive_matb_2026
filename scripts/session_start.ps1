<#
.SYNOPSIS
    Interactive launcher for a single participant study session.

.DESCRIPTION
    Called by session_start_STUDY.bat or session_start_PILOT.bat.
    Asks for a participant number, optionally a resume phase, then
    constructs and runs run_full_study_session.py with the correct flags.

    LabRecorder RCS and EDA auto-port are always enabled.
    Do not edit flags here without updating the cheat sheet.

.PARAMETER Pilot
    Switch. When present, formats the ID as PPILOT0N instead of P0NN.
    Passed automatically by session_start_PILOT.bat.

.PARAMETER StaircaseOnly
    Switch. Runs only Phases 1-2 (Practice + Staircase, no sensors needed).
    Passed automatically by session_start_STAIRCASE_STUDY.bat / _PILOT.bat.
#>

param(
    [switch]$Pilot,
    [switch]$StaircaseOnly,
    [switch]$Dry
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path $PSScriptRoot -Parent

# ---------------------------------------------------------------------------
# Venv guard
# ---------------------------------------------------------------------------

$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Host ""
    Write-Host "  ERROR: Virtual environment not found at:" -ForegroundColor Red
    Write-Host "    $VenvActivate" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Set up the project venv before running a session." -ForegroundColor Red
    Write-Host "  See docs/WORKFLOW.md for setup instructions." -ForegroundColor Red
    Write-Host ""
    Read-Host "  Press ENTER to close"
    exit 1
}

& $VenvActivate

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Cyan
if ($StaircaseOnly) {
    if ($Dry) {
        Write-Host "  ADAPTIVE MATB --- DRY RUN STAIRCASE LAUNCHER" -ForegroundColor Cyan
    } elseif ($Pilot) {
        Write-Host "  ADAPTIVE MATB --- PILOT STAIRCASE LAUNCHER" -ForegroundColor Cyan
    } else {
        Write-Host "  ADAPTIVE MATB --- STUDY STAIRCASE LAUNCHER" -ForegroundColor Cyan
    }
    Write-Host "  (Phases 1-2 only, no sensors required)" -ForegroundColor DarkCyan
} elseif ($Dry) {
    Write-Host "  ADAPTIVE MATB --- DRY RUN LAUNCHER" -ForegroundColor Cyan
    Write-Host "  (Slots PDRY01-PDRY10 -- not analysed)" -ForegroundColor DarkCyan
} else {
    if ($Pilot) {
        Write-Host "  ADAPTIVE MATB --- PILOT SESSION LAUNCHER" -ForegroundColor Cyan
    } else {
        Write-Host "  ADAPTIVE MATB --- STUDY SESSION LAUNCHER" -ForegroundColor Cyan
    }
}
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Participant ID
# ---------------------------------------------------------------------------

$ParticipantID = $null

while ($true) {
    if ($Dry) {
        $UserInput = Read-Host "  Enter DRY RUN slot number (1-10)"
    } elseif ($Pilot) {
        $UserInput = Read-Host "  Enter PILOT number (1-10)"
    } else {
        $UserInput = Read-Host "  Enter PARTICIPANT number (1-30)"
    }

    $UserInput = $UserInput.Trim()

    if (-not ($UserInput -match '^\d+$')) {
        Write-Host "  Please enter a number only (e.g. 1)." -ForegroundColor Yellow
        continue
    }

    $Num = [int]$UserInput

    if ($Dry) {
        if ($Num -lt 1 -or $Num -gt 10) {
            Write-Host "  Dry run slot must be between 1 and 10." -ForegroundColor Yellow
            continue
        }
        $ParticipantID = "PDRY{0:D2}" -f $Num
    } elseif ($Pilot) {
        if ($Num -lt 1 -or $Num -gt 10) {
            Write-Host "  Pilot number must be between 1 and 10." -ForegroundColor Yellow
            continue
        }
        $ParticipantID = "PPILOT{0:D2}" -f $Num
    } else {
        if ($Num -lt 1 -or $Num -gt 30) {
            Write-Host "  Participant number must be between 1 and 30." -ForegroundColor Yellow
            continue
        }
        $ParticipantID = "P{0:D3}" -f $Num
    }

    break
}

Write-Host "  Participant ID : $ParticipantID" -ForegroundColor White

# ---------------------------------------------------------------------------
# Resume phase (full session only) / or set end phase for staircase
# ---------------------------------------------------------------------------

$Phase = 1
$EndPhase = 8

if ($StaircaseOnly) {
    # No phase prompt: always run from the beginning through Phase 2
    $EndPhase = 2
    Write-Host ""
    Write-Host "  Mode: STAIRCASE ONLY (Practice + Staircase, no sensors)" -ForegroundColor Yellow
    Write-Host "  After this finishes, fit sensors and use the Full Session icon." -ForegroundColor DarkGray
} elseif ($Dry) {
    Write-Host ""
    Write-Host "  Skip to which phase?" -ForegroundColor Yellow
    Write-Host "    3  Rest baseline        2-min fixation cross (EEG normalisation)"
    Write-Host "    4  Generate scenarios   Build task files from staircase result  (~1 min)"
    Write-Host "    5  Calibration runs     2 x 9-min task blocks"
    Write-Host "    6  Model calibration    Fit participant workload model  (~2 min)"
    Write-Host "    7  Experimental         Adaptation + control conditions (2 x 8 min)"
    Write-Host ""
    Write-Host "  NOTE: Phase 3 is the normal start for a dry run." -ForegroundColor DarkGray
    Write-Host ""

    $PhaseInput = Read-Host "  Phase [3-7, press ENTER for 3]"
    $PhaseInput = $PhaseInput.Trim()

    if (-not [string]::IsNullOrWhiteSpace($PhaseInput)) {
        if ($PhaseInput -match '^\d+$') {
            $ParsedPhase = [int]$PhaseInput
            if ($ParsedPhase -ge 3 -and $ParsedPhase -le 7) {
                $Phase = $ParsedPhase
            } else {
                Write-Host "  '$PhaseInput' is outside the valid range. Defaulting to 3." -ForegroundColor Yellow
                $Phase = 3
            }
        } else {
            Write-Host "  Invalid input. Defaulting to phase 3." -ForegroundColor Yellow
            $Phase = 3
        }
    } else {
        $Phase = 3
    }

    Write-Host ""
    Write-Host "  Mode: DRY RUN -- starting at Phase $Phase" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "  Start from which phase?" -ForegroundColor Yellow
    Write-Host "    3  Rest baseline        2-min fixation cross (EEG normalisation)"
    Write-Host "    4  Generate scenarios   Build task files from staircase result  (~1 min)"
    Write-Host "    5  Calibration runs     2 x 9-min task blocks"
    Write-Host "    6  Model calibration    Fit participant workload model  (~2 min)"
    Write-Host "    7  Experimental         Adaptation + control conditions (2 x 8 min)"
    Write-Host ""
    Write-Host "  NOTE: Phase 3 is the normal start (after sensor setup + staircase)." -ForegroundColor DarkGray
    Write-Host "  NOTE: Post-session analysis always runs at the end." -ForegroundColor DarkGray
    Write-Host ""

    $PhaseInput = Read-Host "  Phase [3-7, press ENTER for 3]"
    $PhaseInput = $PhaseInput.Trim()

    if (-not [string]::IsNullOrWhiteSpace($PhaseInput)) {
        if ($PhaseInput -match '^\d+$') {
            $ParsedPhase = [int]$PhaseInput
            if ($ParsedPhase -ge 3 -and $ParsedPhase -le 7) {
                $Phase = $ParsedPhase
            } else {
                Write-Host "  '$PhaseInput' is outside the valid range. Defaulting to 3." -ForegroundColor Yellow
                $Phase = 3
            }
        } else {
            Write-Host "  Invalid input. Defaulting to phase 3." -ForegroundColor Yellow
            $Phase = 3
        }
    } else {
        $Phase = 3
    }
}

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

$PythonExe  = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $RepoRoot "src\run_full_study_session.py"

if ($StaircaseOnly) {
    # Phases 1-2 only: no LabRecorder, no EDA, explicit end-phase
    $ScriptArgs = @(
        $ScriptPath,
        "--participant", $ParticipantID,
        "--end-phase", "2"
    )
} elseif ($Dry) {
    # Dry run: LabRecorder RCS enabled, no EDA, baseline refresh skipped
    $ScriptArgs = @(
        $ScriptPath,
        "--participant", $ParticipantID,
        "--labrecorder-rcs",
        "--skip-baseline-refresh"
    )
    $ScriptArgs += "--start-phase"
    $ScriptArgs += "$Phase"
} else {
    $ScriptArgs = @(
        $ScriptPath,
        "--participant", $ParticipantID,
        "--labrecorder-rcs",
        "--eda-auto-port"
    )
    if ($Phase -gt 1) {
        $ScriptArgs += "--start-phase"
        $ScriptArgs += "$Phase"
    }
}

# ---------------------------------------------------------------------------
# Confirm and launch
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "  READY TO LAUNCH" -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "  Participant : $ParticipantID" -ForegroundColor Green
if ($StaircaseOnly) {
    Write-Host "  Mode        : STAIRCASE ONLY (Phases 1-2, no sensors)" -ForegroundColor Green
} elseif ($Dry) {
    Write-Host "  Mode        : DRY RUN (no EDA, baseline refresh skipped)" -ForegroundColor Green
    Write-Host "  Start phase : $Phase" -ForegroundColor Green
} else {
    Write-Host "  Start phase : $Phase" -ForegroundColor Green
}
Write-Host ""
Write-Host "  Full command:" -ForegroundColor Green
Write-Host "    $PythonExe $($ScriptArgs -join ' ')" -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""

Read-Host "  Press ENTER to start (Ctrl-C to abort)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

Set-Location $RepoRoot
& $PythonExe @ScriptArgs
$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -ne 0) {
    Write-Host "  SESSION ENDED WITH ERRORS (exit code $ExitCode)" -ForegroundColor Red
    Write-Host "  Review the output above, then check the cheat sheet troubleshooting section." -ForegroundColor Red
} else {
    Write-Host "  SESSION COMPLETE." -ForegroundColor Green
}

Write-Host ""
Read-Host "  Press ENTER to close this window"
