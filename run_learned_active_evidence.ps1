#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for the resumable learned-backend ACTIVE evidence pipeline.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_active_evidence.ps1 --resume --substrate-mode hf --substrate-device cuda

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_active_evidence.ps1 --dry-run --skip-ablation

.NOTES
    All arguments are forwarded to scripts/run_learned_active_evidence.py.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin scripts/run_learned_active_evidence.py @args
exit $LASTEXITCODE
