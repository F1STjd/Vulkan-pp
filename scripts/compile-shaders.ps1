# Compile shaders/shader.slang -> shaders/slang.spv
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ShaderDir = Join-Path $ProjectRoot 'shaders'
$ShaderSource = Join-Path $ShaderDir 'shader.slang'
$ShaderOutput = Join-Path $ShaderDir 'slang.spv'

function Find-Slangc {
    if ($env:SLANGC) {
        return $env:SLANGC
    }

    $candidates = @(
        "$env:VULKAN_SDK\bin\slangc.exe",
        'D:\tools\slang\bin\slangc.exe',
        "$env:USERPROFILE\scoop\apps\slang\current\bin\slangc.exe",
        "$env:USERPROFILE\scoop\shims\slangc.exe"
    )

    foreach ($path in $candidates) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return $path
        }
    }

    $command = Get-Command slangc -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw @"
slangc not found. Install with: scoop install slang
Or set SLANGC to the full path of slangc.exe
"@
}

$slangc = Find-Slangc

& $slangc $ShaderSource `
    -target spirv `
    -profile spirv_1_4 `
    -emit-spirv-directly `
    -fvk-use-entrypoint-name `
    -entry vertex_main `
    -entry fragment_main `
    -o $ShaderOutput

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Wrote $ShaderOutput"
