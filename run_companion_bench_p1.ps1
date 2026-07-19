#requires -Version 5.1
<#
.SYNOPSIS
    Windows 一键启动 CompanionBench P1 same-substrate 五轨 directional 跑分。

.DESCRIPTION
    仓库根目录薄封装，委托给 SSOT 编排器：
    scripts/companion_bench/run_p1_windows.ps1

    前置条件（首次使用前）：
      1. .local/llm.env  — OPENROUTER_API_KEY 与 ABLATION_* judge/user-sim 配置
      2. 本地 HF 缓存    — 默认 Qwen/Qwen2.5-1.5B-Instruct（可用 $env:VZ_SUBSTRATE_MODEL_ID 覆盖）
      3. companion bootstrap — packages/lifeform-domain-emogpt/.../bootstraps/*.snap|*.bs
      4. NVIDIA GPU + CUDA

    产物目录：artifacts/companion-ablation/<run-id>/
    完成后查看：verdict_p1.json 与各轨 scores/*/summary.json

.PARAMETER ArtifactDir
    指定产物目录。省略时：新跑用 UTC 时间戳目录；-Resume 时自动选最近未完成的 run。

.PARAMETER DryRun
    只打印将执行的 score/verdict 命令，不启动 GPU 服务或付费 API。

.PARAMETER Resume
    从已有产物目录续跑（跳过已完成的 arc）。

.PARAMETER KeepServices
    跑分结束后保留 8000/8001/8500/8600 上的服务进程。

.PARAMETER FullMode
    关闭默认 SafeMode，使用完整 P1 资源配置运行。

.PARAMETER Stop
    停止指定（或最近）run 的 serve.pids 进程后退出，不启动新跑分。

.EXAMPLE
    .\run_companion_bench_p1.ps1
    全新 P1 跑分（五轨 × 30 公开场景，预计数小时）。

.EXAMPLE
    .\run_companion_bench_p1.ps1 -Resume
    续跑最近一次未完成的产物目录。

.EXAMPLE
    .\run_companion_bench_p1.ps1 -DryRun
    预检编排命令，不产生 GPU/API 费用。

.EXAMPLE
    .\run_companion_bench_p1.ps1 -Stop
    停止最近一次 run 残留的五轨服务。
#>

[CmdletBinding()]
param(
    [string]$ArtifactDir = "",
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$KeepServices,
    [switch]$FullMode,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

if ($ArtifactDir -like "--*") {
    switch ($ArtifactDir.ToLowerInvariant()) {
        "--resume" {
            $Resume = $true
            $ArtifactDir = ""
        }
        "--dry-run" {
            $DryRun = $true
            $ArtifactDir = ""
        }
        "--dryrun" {
            $DryRun = $true
            $ArtifactDir = ""
        }
        "--keep-services" {
            $KeepServices = $true
            $ArtifactDir = ""
        }
        "--full-mode" {
            $FullMode = $true
            $ArtifactDir = ""
        }
        "--fullmode" {
            $FullMode = $true
            $ArtifactDir = ""
        }
        "--stop" {
            $Stop = $true
            $ArtifactDir = ""
        }
        default {
            throw "unknown option '$ArtifactDir'. In PowerShell use -Resume/-Stop/-DryRun/-FullMode, or one of the supported --resume/--stop/--dry-run/--full-mode aliases."
        }
    }
}

$Runner = Join-Path $RepoRoot "scripts/companion_bench/run_p1_windows.ps1"
if (-not (Test-Path $Runner)) {
    throw "missing orchestrator: $Runner"
}

function Get-LatestAblationRunDir {
    $base = Join-Path $RepoRoot "artifacts/companion-ablation"
    if (-not (Test-Path $base)) {
        return $null
    }
    Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

function Stop-AblationRun {
    param([string]$RunDir)
    $pidFile = Join-Path $RunDir "serve.pids"
    $watchdogPidFile = Join-Path $RunDir "watchdog.pid"
    if (-not (Test-Path $pidFile)) {
        Write-Host "[p1] no serve.pids under $RunDir"
    } else {
        Get-Content $pidFile | Where-Object { $_ } | ForEach-Object {
            Stop-Process -Id ([int]$_) -Force -ErrorAction SilentlyContinue
        }
        Write-Host "[p1] stopped services from $pidFile"
    }
    if (Test-Path $watchdogPidFile) {
        Get-Content $watchdogPidFile | Where-Object { $_ } | ForEach-Object {
            Stop-Process -Id ([int]$_) -Force -ErrorAction SilentlyContinue
        }
        Write-Host "[p1] stopped watchdog from $watchdogPidFile"
    }
}

if ($Stop) {
    if (-not $ArtifactDir) {
        $latest = Get-LatestAblationRunDir
        if (-not $latest) {
            throw "no artifacts/companion-ablation run found to stop"
        }
        $ArtifactDir = $latest
    } elseif (-not [System.IO.Path]::IsPathRooted($ArtifactDir)) {
        $ArtifactDir = Join-Path $RepoRoot $ArtifactDir
    }
    Stop-AblationRun -RunDir $ArtifactDir
    exit 0
}

if ($Resume -and -not $ArtifactDir) {
    $latest = Get-LatestAblationRunDir
    if ($latest) {
        $ArtifactDir = $latest
        Write-Host "[p1] -Resume without -ArtifactDir; using latest run: $ArtifactDir"
    }
}

$invokeArgs = @{}
if ($ArtifactDir) {
    if (-not [System.IO.Path]::IsPathRooted($ArtifactDir)) {
        $ArtifactDir = Join-Path $RepoRoot $ArtifactDir
    }
    $invokeArgs["ArtifactDir"] = $ArtifactDir
}
if ($DryRun) { $invokeArgs["DryRun"] = $true }
if ($Resume) { $invokeArgs["Resume"] = $true }
if ($KeepServices) { $invokeArgs["KeepServices"] = $true }
if ($FullMode) { $invokeArgs["FullMode"] = $true }

Write-Host "[p1] repo=$RepoRoot"
Write-Host "[p1] orchestrator=$Runner"
if (-not $FullMode) {
    Write-Host "[p1] SafeMode enabled by default; pass -FullMode for unrestricted resource settings"
}
if ($ArtifactDir) {
    Write-Host "[p1] artifact-dir=$ArtifactDir"
}

& $Runner @invokeArgs
exit $LASTEXITCODE
