#requires -Version 5.1
<#
.SYNOPSIS
    Stop the browser chat service started by start_browser_chat_qwen.ps1.

.DESCRIPTION
    Windows / PowerShell port of stop_browser_chat_qwen.sh. Finds processes
    listening on PORT (default 8765) and stops them gracefully, then forces
    any listener still bound after one second.

.EXAMPLE
    .\stop_browser_chat_qwen.ps1

.EXAMPLE
    $env:PORT = '8766'
    .\stop_browser_chat_qwen.ps1
#>

$ErrorActionPreference = 'Stop'

if (-not $env:PORT) {
    $env:PORT = '8765'
}
$portNum = [int]$env:PORT

function Get-ListenerPids {
    param([int]$LocalPort)

    $pids = @()
    try {
        $pids = @(Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique)
    } catch {
        if ($_.FullyQualifiedErrorId -notlike 'CmdletizationQuery_NotFound*') {
            throw
        }
    }
    return $pids
}

$listenerPids = Get-ListenerPids -LocalPort $portNum

if ($listenerPids.Count -eq 0) {
    Write-Host "No browser chat service is listening on port ${portNum}."
    exit 0
}

Write-Host "Stopping browser chat service on port ${portNum}: $($listenerPids -join ', ')"
foreach ($procId in $listenerPids) {
    Stop-Process -Id $procId -ErrorAction Stop
}

Start-Sleep -Seconds 1

$remainingPids = Get-ListenerPids -LocalPort $portNum
if ($remainingPids.Count -gt 0) {
    Write-Host "Service is still listening on port ${portNum}; forcing stop: $($remainingPids -join ', ')"
    foreach ($procId in $remainingPids) {
        Stop-Process -Id $procId -Force -ErrorAction Stop
    }
}

Write-Host "Browser chat service stopped."
