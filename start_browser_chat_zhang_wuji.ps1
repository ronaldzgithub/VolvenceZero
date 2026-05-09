#requires -Version 5.1
<#
.SYNOPSIS
    Real Qwen substrate + 张无忌 vertical (with the lived LifeformTemplate
    from the demo arc replay) — Windows / PowerShell port of
    start_browser_chat_zhang_wuji.sh.

.DESCRIPTION
    Pipeline:

      1. If $ZHANG_WUJI_TEMPLATE_PATH does not exist, run
         examples/train_zhang_wuji_template.py first. Set
         ZHANG_WUJI_RETRAIN=1 to retrain even when a template exists,
         or ZHANG_WUJI_SKIP_TEMPLATE=1 to fall back to the base
         profile.
      2. Export VERTICAL=zhang_wuji and ZHANG_WUJI_TEMPLATE_PATH so
         the service-side router builds a 张无忌 Lifeform via
         give_birth() under alpha mode (per-user filesystem-scoped
         memory + saved drives).
      3. Defer to start_browser_chat_qwen.ps1 which actually loads
         the HF Qwen runtime and starts the aiohttp service. All
         Qwen-related env vars (MODEL_ID, DEVICE, PORT, ALPHA_MODE,
         …) honored by that script are honored here too.

.NOTES
    Useful env vars on top of those documented by
    start_browser_chat_qwen.ps1:

      VERTICAL=zhang_wuji                       # default for this wrapper
      ZHANG_WUJI_TEMPLATE_PATH=<repo>\artifacts\lifeform-templates\zhang-wuji-demo.json
      ZHANG_WUJI_RETRAIN=0                      # 1 = retrain even if template exists
      ZHANG_WUJI_SKIP_TEMPLATE=0                # 1 = base profile, no give_birth
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
if (-not $RootDir) {
    $RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

function Set-DefaultEnv {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [AllowEmptyString()] [string] $Value
    )
    $current = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrEmpty($current)) {
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

Set-DefaultEnv 'VERTICAL'                 'zhang_wuji'
Set-DefaultEnv 'ZHANG_WUJI_TEMPLATE_PATH' (Join-Path $RootDir 'artifacts\lifeform-templates\zhang-wuji-demo.json')

if ($env:ZHANG_WUJI_SKIP_TEMPLATE -eq '1') {
    Write-Host "[start-browser-chat-zhang-wuji] ZHANG_WUJI_SKIP_TEMPLATE=1 - using base profile (no give_birth)."
    Remove-Item Env:\ZHANG_WUJI_TEMPLATE_PATH -ErrorAction SilentlyContinue
} else {
    $templatePath = $env:ZHANG_WUJI_TEMPLATE_PATH
    $needTrain = ($env:ZHANG_WUJI_RETRAIN -eq '1') -or (-not (Test-Path $templatePath))
    if ($needTrain) {
        if ($env:ZHANG_WUJI_RETRAIN -eq '1') {
            Write-Host "[start-browser-chat-zhang-wuji] ZHANG_WUJI_RETRAIN=1 - retraining template."
        } else {
            Write-Host "[start-browser-chat-zhang-wuji] template $templatePath not found - running trainer."
        }

        $packageSrcs = Get-ChildItem -Path (Join-Path $RootDir 'packages') -Directory -ErrorAction Stop |
            ForEach-Object { Join-Path $_.FullName 'src' } |
            Where-Object { Test-Path $_ }
        $packagePaths = ($packageSrcs -join ';')
        $oldPyPath = $env:PYTHONPATH
        if ($oldPyPath) {
            $env:PYTHONPATH = "$packagePaths;$oldPyPath"
        } else {
            $env:PYTHONPATH = $packagePaths
        }

        $trainArgs = @(
            (Join-Path $RootDir 'examples\train_zhang_wuji_template.py'),
            '--output', $templatePath
        )
        if ($env:ZHANG_WUJI_RETRAIN -eq '1') {
            $trainArgs += '--force'
        }
        & $PythonBin $trainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "trainer failed with exit code $LASTEXITCODE"
        }
        $env:PYTHONPATH = $oldPyPath
    } else {
        Write-Host "[start-browser-chat-zhang-wuji] template found at $templatePath; skipping trainer."
        Write-Host "[start-browser-chat-zhang-wuji] (set ZHANG_WUJI_RETRAIN=1 to retrain.)"
    }
}

$templateMsg = if ($env:ZHANG_WUJI_TEMPLATE_PATH) { $env:ZHANG_WUJI_TEMPLATE_PATH } else { '<none>' }
Write-Host "[start-browser-chat-zhang-wuji] vertical=$($env:VERTICAL) template=$templateMsg"

& (Join-Path $RootDir 'start_browser_chat_qwen.ps1')
exit $LASTEXITCODE
