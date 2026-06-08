# gcc-release (run) then clang-debug (clangd PCM / compile_commands).
$ErrorActionPreference = 'Stop'

Set-Location (Split-Path -Parent $PSScriptRoot)

cmake --build --preset gcc-release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $PSScriptRoot 'Build-ClangDebug.ps1')
exit $LASTEXITCODE