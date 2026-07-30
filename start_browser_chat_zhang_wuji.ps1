#requires -Version 5.1
<#
.SYNOPSIS
    Start browser chat with the baked 张无忌 character vertical on Qwen 1.5B.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
if (-not $RootDir) {
    $RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

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

Set-DefaultEnv 'VERTICAL' 'zhang_wuji'
Set-DefaultEnv 'MODEL_ID' 'Qwen/Qwen2.5-1.5B-Instruct'
Set-DefaultEnv 'ALPHA_MODE' '0'
Set-DefaultEnv 'TEMPLATES_ROOT_DIR' (Join-Path $RootDir 'artifacts\lifeform-templates')
Set-DefaultEnv 'ZHANG_WUJI_TEMPLATE_PATH' (
    Join-Path $RootDir 'artifacts\lifeform-templates\zhang_wuji\zhang-wuji-live-through.json'
)

if (-not (Test-Path $env:ZHANG_WUJI_TEMPLATE_PATH)) {
    Write-Error @"
Cannot find the baked 张无忌 template:
  ZHANG_WUJI_TEMPLATE_PATH=$($env:ZHANG_WUJI_TEMPLATE_PATH)

Rebuild it with:
  .\.venv\Scripts\python.exe examples\bake_zhang_wuji_live_through.py --save-template

Or point ZHANG_WUJI_TEMPLATE_PATH at another saved LifeformTemplate JSON.
"@
    exit 1
}

& (Join-Path $RootDir 'start_browser_chat_qwen.ps1')
exit $LASTEXITCODE
