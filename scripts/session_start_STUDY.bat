@echo off
rem ---------------------------------------------------------------
rem  ADAPTIVE MATB -- STUDY SESSION LAUNCHER
rem
rem  Double-click this file (or a desktop shortcut to it) to start
rem  a session for a main-study participant (P001-P030).
rem
rem  To create a desktop shortcut:
rem    Right-click this file -> Send to -> Desktop (create shortcut)
rem ---------------------------------------------------------------
cd /d "%~dp0.."
powershell.exe -ExecutionPolicy Bypass -File "scripts\session_start.ps1"
