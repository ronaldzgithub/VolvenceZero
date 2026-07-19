param(
    [string]$ArtifactDir = "",
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$KeepServices,
    [switch]$FullMode
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $RepoRoot

$PackageSrcs = Get-ChildItem -Path (Join-Path $RepoRoot "packages") -Directory -ErrorAction Stop |
    ForEach-Object { Join-Path $_.FullName "src" } |
    Where-Object { Test-Path $_ }
$PackagePaths = ($PackageSrcs -join ";")
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$PackagePaths;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $PackagePaths
}

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

function Set-P1CrossFamilyExtractorEnv {
    $openRouterBase = if ($env:OPENROUTER_BASE_URL) {
        $env:OPENROUTER_BASE_URL
    } else {
        "https://openrouter.ai/api/v1"
    }
    if (-not $env:REFH_EXTRACTOR_MODEL) {
        $refhModel = if ($env:ABLATION_REFH_EXTRACTOR_MODEL) {
            $env:ABLATION_REFH_EXTRACTOR_MODEL
        } elseif ($env:ABLATION_PERTURN_MODEL) {
            $env:ABLATION_PERTURN_MODEL
        } else {
            $null
        }
        if ($refhModel) {
            $env:REFH_EXTRACTOR_MODEL = $refhModel
            $env:REFH_EXTRACTOR_BASE_URL = $openRouterBase
            $env:REFH_EXTRACTOR_KEY_ENV = "OPENROUTER_API_KEY"
        }
    }
    if (-not $env:CAMEL_COMPACTION_MODEL) {
        $camelModel = if ($env:ABLATION_CAMEL_COMPACTION_MODEL) {
            $env:ABLATION_CAMEL_COMPACTION_MODEL
        } elseif ($env:ABLATION_REFH_EXTRACTOR_MODEL) {
            $env:ABLATION_REFH_EXTRACTOR_MODEL
        } elseif ($env:ABLATION_PERTURN_MODEL) {
            $env:ABLATION_PERTURN_MODEL
        } else {
            $null
        }
        if ($camelModel) {
            $env:CAMEL_COMPACTION_MODEL = $camelModel
            $env:CAMEL_COMPACTION_BASE_URL = $openRouterBase
            $env:CAMEL_COMPACTION_KEY_ENV = "OPENROUTER_API_KEY"
        }
    }
}

Set-P1CrossFamilyExtractorEnv

function Set-EnvDefault {
    param(
        [string]$Name,
        [string]$Value
    )
    if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Clear-SafeModeDefault {
    param(
        [string]$Name,
        [string]$Value
    )
    if ([Environment]::GetEnvironmentVariable($Name, "Process") -eq $Value) {
        [Environment]::SetEnvironmentVariable($Name, $null, "Process")
    }
}

$SafeMode = -not $FullMode
if ($SafeMode) {
    Set-EnvDefault "REFH_EMBEDDER" "hashing"
    Set-EnvDefault "OMP_NUM_THREADS" "2"
    Set-EnvDefault "MKL_NUM_THREADS" "2"
    Set-EnvDefault "NUMEXPR_NUM_THREADS" "2"
    Set-EnvDefault "TOKENIZERS_PARALLELISM" "false"
    Set-EnvDefault "VZ_P1_PROCESS_PRIORITY" "BelowNormal"
    Set-EnvDefault "VZ_P1_WATCHDOG" "1"
    Set-EnvDefault "VZ_P1_WATCHDOG_INTERVAL_SECONDS" "15"
    Set-EnvDefault "VZ_P1_MIN_AVAILABLE_MEMORY_GB" "4"
    Set-EnvDefault "VZ_P1_MAX_GPU_MEMORY_USED_PCT" "94"
    Set-EnvDefault "VZ_P1_WATCHDOG_SUSTAINED_BREACHES" "3"
    Write-Host "[p1] SafeMode: REFH_EMBEDDER=$($env:REFH_EMBEDDER), priority=$($env:VZ_P1_PROCESS_PRIORITY), watchdog=$($env:VZ_P1_WATCHDOG)"
} else {
    Clear-SafeModeDefault "REFH_EMBEDDER" "hashing"
    Clear-SafeModeDefault "OMP_NUM_THREADS" "2"
    Clear-SafeModeDefault "MKL_NUM_THREADS" "2"
    Clear-SafeModeDefault "NUMEXPR_NUM_THREADS" "2"
    Clear-SafeModeDefault "TOKENIZERS_PARALLELISM" "false"
    Clear-SafeModeDefault "VZ_P1_PROCESS_PRIORITY" "BelowNormal"
    Clear-SafeModeDefault "VZ_P1_WATCHDOG" "1"
    Clear-SafeModeDefault "VZ_P1_WATCHDOG_INTERVAL_SECONDS" "15"
    Clear-SafeModeDefault "VZ_P1_MIN_AVAILABLE_MEMORY_GB" "4"
    Clear-SafeModeDefault "VZ_P1_MAX_GPU_MEMORY_USED_PCT" "94"
    Clear-SafeModeDefault "VZ_P1_WATCHDOG_SUSTAINED_BREACHES" "3"
    $env:VZ_P1_FULL_MODE = "1"
    Write-Host "[p1] FullMode: default SafeMode resource limits disabled"
}

if (-not $env:VZ_SUBSTRATE_MODEL_ID) {
    $env:VZ_SUBSTRATE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    Write-Host "[p1] VZ_SUBSTRATE_MODEL_ID not set; defaulting to $($env:VZ_SUBSTRATE_MODEL_ID)"
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
$WatchdogPidFile = Join-Path $ArtifactDir "watchdog.pid"
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
    if (-not $KeepServices -and (Test-Path $WatchdogPidFile)) {
        Get-Content $WatchdogPidFile | Where-Object { $_ } | ForEach-Object {
            Stop-Process -Id ([int]$_) -Force -ErrorAction SilentlyContinue
        }
        Write-Host "[p1] stopped watchdog from $WatchdogPidFile"
    }
}
