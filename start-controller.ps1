$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
$env:USE_TF = '0'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Write-Host 'Open http://127.0.0.1:8765 in your browser. Ctrl+C stops the controller.'
& '.\.venv\Scripts\python.exe' -u '.\controller.py' @args
