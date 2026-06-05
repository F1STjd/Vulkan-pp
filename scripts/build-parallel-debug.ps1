# gcc-debug (run) then clang-debug (clangd PCM / compile_commands).
# Presets are built sequentially: on Windows, clangd often memory-maps
# build/clang-debug/*.pcm and a concurrent rebuild fails to overwrite them.
$ErrorActionPreference = 'Stop'

Set-Location (Split-Path -Parent $PSScriptRoot)

cmake --build --preset gcc-debug
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build --preset clang-debug
exit $LASTEXITCODE
