$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$Root\src;$Root"
Push-Location $Root
try {
    python -m pytest tests -q --import-mode=importlib
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python ops\build_sha256s.py --root $Root --verify
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
