@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-controller.ps1" %*
pause
