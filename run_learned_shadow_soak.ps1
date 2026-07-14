#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for one continuous learned-shadow soak artifact.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_shadow_soak.ps1 --turns 500 --substrate-mode hf --substrate-device cuda

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_shadow_soak.ps1 --turns 50 --substrate-mode synthetic

.NOTES
    All arguments are forwarded to scripts/run_learned_shadow_soak.py.
    Use run_learned_active_evidence.ps1 for resume markers.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin -u scripts/run_learned_shadow_soak.py @args
exit $LASTEXITCODE
