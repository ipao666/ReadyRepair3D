param([switch]$Full)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:R3DGUARD_HOME = $Root
$env:PYTHONPATH = "$Root;$Root\src"
Push-Location $Root
try {
    if ($Full) { python -m pytest tests -q --import-mode=importlib }
    else { python (Join-Path $Root "../tools/check_cpu.py") }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python (Join-Path $Root "ops/build_sha256s.py") --root $Root --verify
    exit $LASTEXITCODE
} finally { Pop-Location }
