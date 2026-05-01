@echo off
rem ---------------------------------------------------------------
rem  ADAPTIVE MATB -- DRY RUN STAIRCASE LAUNCHER
rem
rem  Use this icon FIRST (before sensor setup).
rem  Runs Practice + Staircase (Phases 1-2) with no sensors needed.
rem
rem  After this finishes:
rem    1. Fit EEG cap and HR sensor.
rem    2. Double-click the DRY RUN FULL SESSION icon to continue.
rem ---------------------------------------------------------------
cd /d "%~dp0.."
powershell.exe -ExecutionPolicy Bypass -File "scripts\session_start.ps1" -StaircaseOnly -Dry
