# Fast iteration build: optimized gcc-release with basic warnings only
# (-Wall -Wextra -Wpedantic). Single compiler, no clang-debug rebuild, so it is
# the quickest way to compile an optimized binary while developing.
#
# For the strict, "feature finished" build use scripts/build-parallel-release.ps1.
$ErrorActionPreference = 'Stop'

Set-Location (Split-Path -Parent $PSScriptRoot)

# Configure once; cheap no-op on later runs (the build dir stays warm/incremental).
if (-not (Test-Path 'build/gcc-release-fast/CMakeCache.txt')) {
    cmake --preset gcc-release-fast
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

cmake --build --preset gcc-release-fast
exit $LASTEXITCODE
