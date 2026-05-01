<#
.SYNOPSIS
    Creates named desktop shortcuts for the MATB session launchers.

.DESCRIPTION
    Run this once after cloning the repo (or after moving the repo folder).
    Creates four .lnk files on the desktop:

        "Adaptive MATB - Study Staircase"   (Phase 1-2, no sensors)
        "Adaptive MATB - Study Session"     (Phase 3-8, all sensors)
        "Adaptive MATB - Pilot Staircase"   (Phase 1-2, no sensors)
        "Adaptive MATB - Pilot Session"     (Phase 3-8, all sensors)

    Both shortcuts point to the corresponding .bat launcher in scripts\ and
    use distinct Windows system icons so they are easy to tell apart at a
    glance.

    Re-run this script any time you move the repository folder.

.NOTES
    Icons are pulled from Windows built-in imageres.dll (ships with every
    Windows 10/11 installation — no external files needed).

    To use a custom icon instead:
        1. Place a .ico file anywhere in the repo (e.g. docs\assets\matb.ico)
        2. Change $StudyIcon / $PilotIcon below to the full path of that file
        3. Set the index to 0
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path $PSScriptRoot -Parent
$Shell      = New-Object -ComObject WScript.Shell
$Desktop    = $Shell.SpecialFolders("Desktop")

# ---------------------------------------------------------------------------
# Icons
#
# Source: shell32.dll — ships with every Windows 10/11 install.
# Well-known stable indices:
#
#   Index 131 : blue person silhouette       -> STUDY (participant)
#   Index 175 : coloured science/flask tiles  -> PILOT (test run)
#
# TO CHANGE AN ICON — easiest way:
#   1. After running this script, right-click the shortcut on the desktop.
#   2. Properties -> Shortcut tab -> Change Icon button.
#   3. Browse to any .dll (e.g. imageres.dll, shell32.dll) and pick visually.
#   4. Then update $StudyIconIndex / $PilotIconIndex here and re-run so it
#      survives the next time setup_desktop_shortcuts.ps1 is run.
#
# To list all available icons in a DLL (PowerShell one-liner):
#   & "$env:SystemRoot\System32\shell32.dll"   # paste in Run (Win+R)
# ---------------------------------------------------------------------------

$IconDll               = "$env:SystemRoot\System32\shell32.dll"
$StudyIconIndex        = 131   # blue person silhouette
$PilotIconIndex        = 175   # coloured flask / tiles (visually distinct)
$StaircaseIconIndex    = 24    # clock / timer (visually distinct from full-session icons)
$DryIconIndex          = 134   # wrench / tools (visually distinct — dry run / testing)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

function New-SessionShortcut {
    param(
        [string]$Name,
        [string]$BatFile,
        [string]$Description,
        [string]$IconFile,
        [int]   $IconIndex
    )

    $LinkPath = Join-Path $Desktop "$Name.lnk"
    $Shortcut = $Shell.CreateShortcut($LinkPath)

    $Shortcut.TargetPath       = $BatFile
    $Shortcut.WorkingDirectory = $RepoRoot
    $Shortcut.Description      = $Description
    $Shortcut.IconLocation     = "$IconFile,$IconIndex"
    $Shortcut.WindowStyle      = 1   # normal window

    $Shortcut.Save()

    Write-Host "  Created: $LinkPath" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Create shortcuts
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  Creating desktop shortcuts..." -ForegroundColor Cyan
Write-Host ""

New-SessionShortcut `
    -Name        "Adaptive MATB - Study Staircase" `
    -BatFile     (Join-Path $RepoRoot "scripts\session_start_STAIRCASE_STUDY.bat") `
    -Description "Run Practice + Staircase only for a study participant (no sensors needed)" `
    -IconFile    $IconDll `
    -IconIndex   $StaircaseIconIndex

New-SessionShortcut `
    -Name        "Adaptive MATB - Study Session" `
    -BatFile     (Join-Path $RepoRoot "scripts\session_start_STUDY.bat") `
    -Description "Launch a main-study session (P001-P030)" `
    -IconFile    $IconDll `
    -IconIndex   $StudyIconIndex

New-SessionShortcut `
    -Name        "Adaptive MATB - Pilot Staircase" `
    -BatFile     (Join-Path $RepoRoot "scripts\session_start_STAIRCASE_PILOT.bat") `
    -Description "Run Practice + Staircase only for a pilot participant (no sensors needed)" `
    -IconFile    $IconDll `
    -IconIndex   $StaircaseIconIndex

New-SessionShortcut `
    -Name        "Adaptive MATB - Pilot Session" `
    -BatFile     (Join-Path $RepoRoot "scripts\session_start_PILOT.bat") `
    -Description "Launch a pilot session (PPILOT01-PPILOT10)" `
    -IconFile    $IconDll `
    -IconIndex   $PilotIconIndex

New-SessionShortcut `
    -Name        "Adaptive MATB - Dry Run Staircase" `
    -BatFile     (Join-Path $RepoRoot "scripts\session_start_STAIRCASE_DRY.bat") `
    -Description "Run Practice + Staircase only for a dry-run participant (no sensors needed)" `
    -IconFile    $IconDll `
    -IconIndex   $StaircaseIconIndex

New-SessionShortcut `
    -Name        "Adaptive MATB - Dry Run Session" `
    -BatFile     (Join-Path $RepoRoot "scripts\session_start_DRY.bat") `
    -Description "Launch a dry-run session (PDRY01-PDRY05)" `
    -IconFile    $IconDll `
    -IconIndex   $DryIconIndex

Write-Host ""
Write-Host "  Done. Six shortcuts are now on the desktop." -ForegroundColor Green
Write-Host "  Staircase icons = run first (no sensors).  Session icons = run second (sensors fitted)." -ForegroundColor DarkGray
Write-Host "  If you move the repository folder, re-run this script." -ForegroundColor DarkGray
Write-Host ""
Read-Host "  Press ENTER to close"
