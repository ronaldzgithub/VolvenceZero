#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for the affordance score learner Stage 0 probe (G3).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_affordance_learner_probe.ps1

.NOTES
    Drives the real lifeform path (registry -> module -> invoker ->
    outcome listener) with deterministic in-process tools so the SHADOW
    affordance score learner accumulates settles. Machinery evidence only;
    promotion still gates on >=50 real-usage settles.
    All arguments are forwarded to scripts/probe_affordance_score_learner.py.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin -u scripts/probe_affordance_score_learner.py @args
exit $LASTEXITCODE
