#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for the capacity->gain ladder.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_capacity_ladder.ps1 --n-z 16,64,256 --turns 500

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_learned_capacity_ladder.ps1 --execute --n-z 16 --turns 50

.NOTES
    All arguments are forwarded to scripts/run_capacity_ladder.py.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin scripts/run_capacity_ladder.py @args
exit $LASTEXITCODE
