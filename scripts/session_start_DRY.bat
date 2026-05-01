@echo off
rem ---------------------------------------------------------------
rem  ADAPTIVE MATB -- DRY RUN SESSION LAUNCHER
rem
rem  Double-click this file (or a desktop shortcut to it) to start
rem  a session for a dry-run participant (PDRY01-PDRY05).
rem
rem  Use dry-run slots for lab colleagues / friends before ethics
rem  approval to verify hardware, software, and protocol flow.
rem  Data from these runs is never analysed.
rem
rem  To create a desktop shortcut:
rem    Right-click this file -> Send to -> Desktop (create shortcut)
rem ---------------------------------------------------------------
cd /d "%~dp0.."
powershell.exe -ExecutionPolicy Bypass -File "scripts\session_start.ps1" -Dry
