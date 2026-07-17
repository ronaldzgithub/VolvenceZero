#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for the cross-session learned-state continuity lane.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_longitudinal_continuity.ps1

.NOTES
    Runs the longitudinal owner-hydration suites against real Brain
    constructors with per-user scoped persistence:
      - tests/longitudinal/test_cross_session_owner_hydration.py
      - tests/longitudinal/test_cross_session_learned_state_continuity.py
        (20 sessions, social/regime/PE-heads/dual-track-gate/credit-heads
         accumulation + cross-user isolation)
    Extra pytest arguments are forwarded (e.g. -k, -x).
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin -m pytest `
  tests/longitudinal/test_cross_session_owner_hydration.py `
  tests/longitudinal/test_cross_session_learned_state_continuity.py `
  -q @args
exit $LASTEXITCODE
