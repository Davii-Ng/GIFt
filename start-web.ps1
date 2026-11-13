[CmdletBinding()]
param(
    [switch]$IndexData,
    [string]$PythonPath,
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$giflyPython = $PythonPath
if (-not $giflyPython) {
    $giflyPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $giflyPython)) {
        $giflyPython = Join-Path $PSScriptRoot 'web\.venv\Scripts\python.exe'
    }
    if (-not (Test-Path -LiteralPath $giflyPython)) {
        $giflyPython = Join-Path $PSScriptRoot '..\..\GIF Search\.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $giflyPython)) {
    throw 'Python environment missing. Follow README.md to install the project first.'
}
$giflyPython = (Resolve-Path -LiteralPath $giflyPython).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    # Database updates are explicit so an import failure does not prevent serving.
    if ($IndexData) {
        & $giflyPython -X utf8 -m web.ingest
        if ($LASTEXITCODE -ne 0) { throw 'GIF indexing failed. See the error above.' }
    }
    Write-Host "gifly is starting at http://127.0.0.1:$Port. Press Ctrl+C to stop."
    & $giflyPython -X utf8 -m uvicorn api:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'GIF server stopped with an error. See the error above.' }
} finally {
    Pop-Location
}
