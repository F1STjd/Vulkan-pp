# Flips the `Suppress: "*"` line in .clangd between commented (squiggles ON)
# and uncommented (squiggles OFF). clangd hot-reloads .clangd on save.
$ErrorActionPreference = 'Stop'
$path = Join-Path $PSScriptRoot '..\.clangd'
$text = Get-Content -Raw -Path $path

if ($text -match '(?m)^\s*#\s*Suppress:\s*"\*"') {
    $text = [regex]::Replace($text, '(?m)^(\s*)#\s*(Suppress:\s*"\*")', '$1$2')
    $state = 'OFF (all squiggles hidden)'
}
elseif ($text -match '(?m)^\s*Suppress:\s*"\*"') {
    $text = [regex]::Replace($text, '(?m)^(\s*)(Suppress:\s*"\*")', '$1# $2')
    $state = 'ON (squiggles shown)'
}
else {
    Write-Host 'Could not find a `Suppress: "*"` line in .clangd; nothing changed.'
    exit 1
}

[System.IO.File]::WriteAllText((Resolve-Path $path), $text)
Write-Host "Diagnostics are now $state"
