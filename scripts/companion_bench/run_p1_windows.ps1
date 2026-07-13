param(
    [string]$ArtifactDir = "",
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$KeepServices
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $RepoRoot

$EnvFile = Join-Path $RepoRoot ".local/llm.env"
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $name, $value = $trimmed.Split("=", 2)
        if (-not [Environment]::GetEnvironmentVariable($name.Trim(), "Process")) {
            [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), "Process")
        }
    }
}

if (-not $env:VZ_SUBSTRATE_MODEL_ID) {
    throw "set VZ_SUBSTRATE_MODEL_ID to the cached Qwen model id"
}
if (-not $ArtifactDir) {
    $dateTag = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $ArtifactDir = "artifacts/companion-ablation/$dateTag"
}
$env:ARTIFACT_DIR = $ArtifactDir

$OpenRouterBase = if ($env:OPENROUTER_BASE_URL) {
    $env:OPENROUTER_BASE_URL
} else {
    "https://openrouter.ai/api/v1"
}
$UserSimModel = if ($env:ABLATION_USER_SIM_MODEL) {
    $env:ABLATION_USER_SIM_MODEL
} else {
    "openai/gpt-5-mini"
}
$PerturnModel = if ($env:ABLATION_PERTURN_MODEL) {
    $env:ABLATION_PERTURN_MODEL
} else {
    $UserSimModel
}
$ArcModel = if ($env:ABLATION_ARC_MODEL) {
    $env:ABLATION_ARC_MODEL
} else {
    "anthropic/claude-3.7-sonnet"
}

$RunnerArgs = @(
    "scripts/companion_bench/run_same_substrate_ablation.py",
    "--phase", "p1",
    "--output-dir", $ArtifactDir,
    "--user-sim-base-url", $OpenRouterBase,
    "--user-sim-model", $UserSimModel,
    "--user-sim-key-env", "OPENROUTER_API_KEY",
    "--perturn-base-url", $OpenRouterBase,
    "--perturn-model", $PerturnModel,
    "--perturn-key-env", "OPENROUTER_API_KEY",
    "--arc-base-url", $OpenRouterBase,
    "--arc-model", $ArcModel,
    "--arc-key-env", "OPENROUTER_API_KEY"
)
if ($Resume) {
    $RunnerArgs += "--resume"
}
if ($DryRun) {
    $RunnerArgs += "--dry-run"
    Write-Host "[p1] dry-run only; no service, GPU, or paid API call will start"
    python @RunnerArgs
    exit $LASTEXITCODE
}

$PreflightArgs = @(
    "scripts/companion_bench/preflight_llm.py",
    "--model-id", $env:VZ_SUBSTRATE_MODEL_ID,
    "--artifact-dir", $ArtifactDir
)
if ($env:VZ_SUBSTRATE_WEIGHTS_PATH) {
    $PreflightArgs += @("--weights-path", $env:VZ_SUBSTRATE_WEIGHTS_PATH)
}

$PidFile = Join-Path $ArtifactDir "serve.pids"
try {
    python @PreflightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "P1 preflight failed"
    }

    & "$PSScriptRoot/serve_same_substrate_ablation.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "P1 service startup failed"
    }

    python @RunnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "P1 scoring or verdict failed"
    }
} finally {
    if (-not $KeepServices -and (Test-Path $PidFile)) {
        Get-Content $PidFile | Where-Object { $_ } | ForEach-Object {
            Stop-Process -Id ([int]$_) -Force -ErrorAction SilentlyContinue
        }
        Write-Host "[p1] stopped services from $PidFile"
    }
}
