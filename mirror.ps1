$ErrorActionPreference = 'Stop'
$scrcpyExe = Join-Path $PSScriptRoot 'tools\scrcpy-win64-v5.0.1\scrcpy.exe'
& $scrcpyExe --max-size=1600 --max-fps=60 --video-bit-rate=8M --no-audio --window-title='Mobile Legends - scrcpy' @args
