#requires -Version 5.1
<#
.SYNOPSIS
    Windows PowerShell launcher for the investor side-by-side demo
    (three arms race on the same task chains; only the carried context differs).

.EXAMPLE
    # Free offline rehearsal (no API key needed; verifies plumbing only,
    # its result shape is injected and is NOT evidence):
    powershell -ExecutionPolicy Bypass -File .\run_investor_demo.ps1 --hand scripted --chains 2 --episodes 6

.EXAMPLE
    # The real demo (needs DASHSCOPE_API_KEY; ~15 min at 3 chains):
    powershell -ExecutionPolicy Bypass -File .\run_investor_demo.ps1 --hand api --chains 3 --episodes 10

.EXAMPLE
    # Resume an interrupted long run without re-spending finished units:
    powershell -ExecutionPolicy Bypass -File .\run_investor_demo.ps1 --hand api --chains 8 --episodes 10 --resume

.NOTES
    All arguments are forwarded to scripts/run_investor_side_by_side_demo.py.
    Results land in %TEMP%\volvence-side-by-side-demo by default (report.json +
    live-demo-curve.png); override with --out, which must stay OUTSIDE this
    repository. The packages are put on PYTHONPATH directly, so no install
    step is required.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

# The live board prints Chinese; zh-CN consoles default to GBK and would garble
# the UTF-8 the script emits.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# No installed environment is required: expose every wheel's src directly.
$SrcDirs = Get-ChildItem (Join-Path $RepoRoot "packages") -Directory |
    ForEach-Object { Join-Path $_.FullName "src" } |
    Where-Object { Test-Path $_ }
$env:PYTHONPATH = ($SrcDirs -join ";")

if (($args -contains "--hand") -and ($args[($args.IndexOf("--hand") + 1)] -eq "api") -and -not $env:DASHSCOPE_API_KEY) {
    Write-Error "DASHSCOPE_API_KEY is not set. Set it first:  `$env:DASHSCOPE_API_KEY = `"sk-...`""
}

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonBin -X utf8 scripts/run_investor_side_by_side_demo.py @args
exit $LASTEXITCODE
