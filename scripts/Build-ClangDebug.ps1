# Build the clang-debug preset for clangd / compile_commands.json.
# On Windows, clangd memory-maps build/clang-debug/*.pcm; Cursor auto-restarts
# clangd, so a plain rebuild often fails with "user-mapped section open".
# Keep clangd stopped until the build finishes.
$ErrorActionPreference = 'Stop'

function Stop-ClangdProcesses {
    Get-Process -Name clangd -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

function Start-ClangdWatchdog {
    $script = @'
while ($true) {
    Get-Process -Name clangd -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 250
}
'@
    return Start-Job -ScriptBlock ([scriptblock]::Create($script))
}

Stop-ClangdProcesses
Start-Sleep -Milliseconds 300

$watchdog = Start-ClangdWatchdog
try {
    cmake --build --preset clang-debug
    $exitCode = $LASTEXITCODE
} finally {
    Stop-Job $watchdog -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $watchdog -Force -ErrorAction SilentlyContinue | Out-Null
}

if ($exitCode -ne 0) {
    Write-Host @"

clang-debug build failed. If the error mentions a locked '.pcm' file, restart
the clangd language server (Ctrl+Shift+P -> "clangd: Restart language server")
and run this script again.

"@ -ForegroundColor Yellow
}

exit $exitCode
