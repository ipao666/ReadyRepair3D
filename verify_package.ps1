# Compatibility entrypoint. Audit current committed evidence, not the old migration bundle.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $Root "tools/verify_portfolio.py")
exit $LASTEXITCODE
