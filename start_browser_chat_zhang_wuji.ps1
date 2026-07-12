#requires -Version 5.1
<#
.SYNOPSIS
    Start browser chat with the 张无忌 character vertical.
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
Set-DefaultEnv 'TEMPLATES_ROOT_DIR' (Join-Path $RootDir 'artifacts\lifeform-templates')
Set-DefaultEnv 'ZHANG_WUJI_TEMPLATE_PATH' (
    Join-Path $RootDir 'artifacts\lifeform-templates\zhang_wuji\zhang-wuji-live-through.json'
)

& (Join-Path $RootDir 'start_browser_chat_qwen.ps1')
exit $LASTEXITCODE
