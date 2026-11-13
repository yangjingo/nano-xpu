# Test-only HTTP server: synthetic output, no Qwen weights are loaded.
param(
    [string]$Python = 'python',
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$previousPythonPath = $env:PYTHONPATH
Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath '.cpu-deps') {
        $env:PYTHONPATH = (Join-Path $repoRoot '.cpu-deps') + [IO.Path]::PathSeparator + $env:PYTHONPATH
    }
    & $Python -m tests.cpu_server serve . --port $Port --served-model-name qwen3-0.6b
    if ($LASTEXITCODE -ne 0) { throw "CPU test server exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
