$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Join-Path $Root "SHA256SUMS_PACKAGE.txt"

if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Missing SHA256SUMS_PACKAGE.txt"
}

$Failures = @()
foreach ($Line in Get-Content -LiteralPath $Manifest -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($Line)) { continue }
    $Parts = $Line -split "  ", 2
    if ($Parts.Count -ne 2) {
        $Failures += "Malformed manifest row: $Line"
        continue
    }
    $Expected = $Parts[0].Trim().ToUpperInvariant()
    $Relative = $Parts[1].Trim().Replace("/", [IO.Path]::DirectorySeparatorChar)
    $Path = Join-Path $Root $Relative
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $Failures += "Missing: $Relative"
        continue
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($Actual -ne $Expected) {
        $Failures += "Hash mismatch: $Relative"
    }
}

if ($Failures.Count -gt 0) {
    $Failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output "PACKAGE_VERIFY_OK"
