#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher that builds and evaluates learned-backend promotion evidence.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_promotion_evidence.ps1 --soak-artifact artifacts\...\learned_shadow_soak.json

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_promotion_evidence.ps1 --soak-artifact artifacts\...\learned_shadow_soak.json --ablation-verdict artifacts\...\verdict_p1.json

.NOTES
    All arguments are forwarded to scripts/build_learned_promotion_evidence.py.
    The generated evidence artifact is then evaluated with
    scripts/evaluate_learned_backend_promotion.py.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "artifacts/learned_backend_promotion" }
$EvidencePath = Join-Path $OutputDir "promotion_evidence.json"
$BuildArgs = New-Object System.Collections.Generic.List[string]

for ($Index = 0; $Index -lt $args.Count; $Index++) {
    $Item = [string]$args[$Index]
    if ($Item -eq "--output") {
        if ($Index + 1 -ge $args.Count) {
            throw "error: --output requires a path"
        }
        $Index++
        $EvidencePath = [string]$args[$Index]
    } elseif ($Item -eq "-h" -or $Item -eq "--help") {
        & $PythonBin scripts/build_learned_promotion_evidence.py --help
        exit $LASTEXITCODE
    } else {
        $BuildArgs.Add($Item)
    }
}

$ReportPath = Join-Path (Split-Path $EvidencePath -Parent) "promotion_report.json"

& $PythonBin scripts/build_learned_promotion_evidence.py @BuildArgs --output $EvidencePath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonBin scripts/evaluate_learned_backend_promotion.py --artifact $EvidencePath --output $ReportPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[learned-promotion] evidence: $EvidencePath"
Write-Host "[learned-promotion] report:   $ReportPath"
