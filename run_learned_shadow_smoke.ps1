#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for the learned-shadow P0 wiring smoke.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_shadow_smoke.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_shadow_smoke.ps1 --turns 8 --output-dir artifacts/learned_shadow_evidence_smoke

.NOTES
    All arguments are forwarded to scripts/run_learned_shadow_evidence_smoke.py.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin scripts/run_learned_shadow_evidence_smoke.py @args
exit $LASTEXITCODE
