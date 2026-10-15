$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv }
    elseif (Get-Command uv -ErrorAction SilentlyContinue) { uv venv --python 3.12 .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
$archivePath = Join-Path $PSScriptRoot 'tools\scrcpy-win64-v5.0.1.zip'
$scrcpyPath = Join-Path $PSScriptRoot 'tools\scrcpy-win64-v5.0.1\scrcpy.exe'
if (-not (Test-Path -LiteralPath $scrcpyPath)) {
    New-Item -ItemType Directory -Force -Path '.\tools' | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/Genymobile/scrcpy/releases/download/v5.0.1/scrcpy-win64-v5.0.1.zip' -OutFile $archivePath
    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    if ($archiveHash -ine 'b12a2c4ee8be317422451fc7dcf8ee20a71b5ea7ef9ad73ddd825a25316227a5') { throw 'scrcpy archive hash mismatch.' }
    Expand-Archive -LiteralPath $archivePath -DestinationPath '.\tools' -Force
}
New-Item -ItemType Directory -Force -Path '.\logs' | Out-Null
if (-not (Test-Path -LiteralPath '.\config.json')) {
    Copy-Item -LiteralPath '.\config.example.json' -Destination '.\config.json'
}
& '.\tools\scrcpy-win64-v5.0.1\adb.exe' devices -l
Write-Host 'Setup complete. Run start-controller.cmd, then open http://127.0.0.1:8765.'
