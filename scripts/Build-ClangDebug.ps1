# Build the clang-debug preset for clangd / compile_commands.json.
$ErrorActionPreference = 'Stop'

cmake --build --preset clang-debug
exit $LASTEXITCODE
