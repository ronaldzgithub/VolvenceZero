# Same-substrate Companion Bench ablation — Windows launcher (mirrors .sh).
# Requires: VZ_SUBSTRATE_MODEL_ID, LIFEFORM_LOCAL_API_KEY, VZ_TORCH_BACKENDS=active (optional).

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

if (-not $env:VZ_SUBSTRATE_MODEL_ID) {
    $env:VZ_SUBSTRATE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
}
if (-not $env:LIFEFORM_LOCAL_API_KEY) {
    $env:LIFEFORM_LOCAL_API_KEY = "local-ablation-key"
}

$Device = if ($env:VZ_SUBSTRATE_DEVICE) { $env:VZ_SUBSTRATE_DEVICE } else { "cuda" }
$DateTag = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
if (-not $env:ARTIFACT_DIR) {
    $env:ARTIFACT_DIR = "artifacts/companion-ablation/$DateTag"
}
$ArtifactDir = $env:ARTIFACT_DIR
$LogDir = Join-Path $ArtifactDir "serve-logs"
$PidFile = Join-Path $ArtifactDir "serve.pids"
$CamelBackend = if ($env:CAMEL_BACKEND) { $env:CAMEL_BACKEND } else { "camel" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Content -Path $PidFile -Value ""

$RawUpstream = "http://127.0.0.1:8000/v1?mode=raw"

$PreflightArgs = @(
    "scripts/companion_bench/preflight_llm.py",
    "--offline",
    "--model-id", $env:VZ_SUBSTRATE_MODEL_ID,
    "--artifact-dir", $ArtifactDir
)
if ($env:VZ_SUBSTRATE_WEIGHTS_PATH) {
    $PreflightArgs += @("--weights-path", $env:VZ_SUBSTRATE_WEIGHTS_PATH)
}
python @PreflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "[serve] local P1 preflight failed"
}

Write-Host "[serve] substrate=$($env:VZ_SUBSTRATE_MODEL_ID) device=$Device"
Write-Host "[serve] artifacts -> $ArtifactDir"
Write-Host "[serve] VZ_TORCH_BACKENDS=$($env:VZ_TORCH_BACKENDS)"

function Start-AblationService {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Host "[serve] starting ${Name}: $($Command -join ' ')"
    $proc = Start-Process -FilePath $Command[0] -ArgumentList $Command[1..($Command.Length - 1)] `
        -RedirectStandardOutput (Join-Path $LogDir "$Name.log") `
        -RedirectStandardError (Join-Path $LogDir "$Name.err.log") `
        -PassThru -WindowStyle Hidden
    Add-Content -Path $PidFile -Value $proc.Id
}

function Stop-AblationServices {
    if (Test-Path $PidFile) {
        Get-Content $PidFile | Where-Object { $_ } | ForEach-Object {
            Stop-Process -Id ([int]$_) -Force -ErrorAction SilentlyContinue
        }
    }
}

function Wait-AblationEndpoint {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSeconds = 900
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                Write-Host "[serve] healthy ${Name}: $Url"
                return
            }
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    throw "[serve] timed out waiting for ${Name}: $Url"
}

try {
Start-AblationService "lifeform-companion" @(
    "lifeform-serve",
    "--vertical", "companion",
    "--port", "8000",
    "--substrate-mode", "hf-shared",
    "--substrate-model-id", $env:VZ_SUBSTRATE_MODEL_ID,
    "--substrate-device", $Device,
    "--enable-openai-compat"
)

Start-AblationService "lifeform-companion-cold" @(
    "lifeform-serve",
    "--vertical", "companion-cold",
    "--port", "8001",
    "--substrate-mode", "hf-shared",
    "--substrate-model-id", $env:VZ_SUBSTRATE_MODEL_ID,
    "--substrate-device", $Device,
    "--enable-openai-compat"
)

# ref-harness embed component now uses a real semantic embedder (bge-m3) by
# default. The model loads from the HF cache; if it needs downloading, the
# serve process honours HF_ENDPOINT + *_PROXY from the environment.
$RefhEmbedder = if ($env:REFH_EMBEDDER) { $env:REFH_EMBEDDER } else { "bge-m3" }
$RefhArgs = @(
    "companion-ref-harness", "serve",
    "--port", "8500",
    "--upstream-base-url", $RawUpstream,
    "--upstream-model", "lifeform-raw",
    "--upstream-key-env", "LIFEFORM_LOCAL_API_KEY",
    "--components", "summary,embed,user_model,episodic",
    "--embedder", $RefhEmbedder,
    "--store-mode", "sqlite",
    "--store-path", (Join-Path $ArtifactDir "ref-harness.sqlite3")
)
if ($env:REFH_EXTRACTOR_MODEL) {
    $RefhArgs += @(
        "--summary-extractor-base-url", $env:REFH_EXTRACTOR_BASE_URL,
        "--summary-extractor-model", $env:REFH_EXTRACTOR_MODEL,
        "--summary-extractor-key-env", $env:REFH_EXTRACTOR_KEY_ENV,
        "--summary-extractor-family", $(if ($env:REFH_EXTRACTOR_FAMILY) { $env:REFH_EXTRACTOR_FAMILY } else { "openai-compat" })
    )
} else {
    throw "[serve] ref-harness memory components require a cross-family extractor: set REFH_EXTRACTOR_MODEL / REFH_EXTRACTOR_BASE_URL / REFH_EXTRACTOR_KEY_ENV (a NON-Qwen family)."
}
Start-AblationService "ref-harness" $RefhArgs

# camel memory compaction now requires a cross-family extractor (fail-loud).
$CamelArgs = @(
    "companion-camel-baseline", "serve",
    "--port", "8600",
    "--backend", $CamelBackend,
    "--upstream-base-url", $RawUpstream,
    "--upstream-model", "lifeform-raw",
    "--upstream-key-env", "LIFEFORM_LOCAL_API_KEY",
    "--store-mode", "sqlite",
    "--store-path", (Join-Path $ArtifactDir "camel-baseline.sqlite3")
)
if ($CamelBackend -eq "camel") {
    if (-not $env:CAMEL_COMPACTION_MODEL) {
        throw "[serve] camel backend memory compaction requires a cross-family extractor: set CAMEL_COMPACTION_MODEL / CAMEL_COMPACTION_BASE_URL / CAMEL_COMPACTION_KEY_ENV (a NON-Qwen family)."
    }
    $CamelArgs += @(
        "--compaction-base-url", $env:CAMEL_COMPACTION_BASE_URL,
        "--compaction-model", $env:CAMEL_COMPACTION_MODEL,
        "--compaction-key-env", $env:CAMEL_COMPACTION_KEY_ENV
    )
}
Start-AblationService "camel-baseline" $CamelArgs

Wait-AblationEndpoint "lifeform-companion" "http://127.0.0.1:8000/v1/health"
Wait-AblationEndpoint "lifeform-companion-cold" "http://127.0.0.1:8001/v1/health"
Wait-AblationEndpoint "ref-harness" "http://127.0.0.1:8500/healthz"
Wait-AblationEndpoint "camel-baseline" "http://127.0.0.1:8600/healthz"

python scripts/companion_bench/assert_same_substrate.py `
  --require-weights-sha256 `
  --fingerprint-file "raw=$(Join-Path $ArtifactDir 'raw/substrate_fingerprint.json')" `
  --fingerprint-file "ref-harness=$(Join-Path $ArtifactDir 'ref-harness/substrate_fingerprint.json')" `
  --fingerprint-file "camel=$(Join-Path $ArtifactDir 'camel/substrate_fingerprint.json')" `
  --fingerprint-file "volvence-cold=$(Join-Path $ArtifactDir 'volvence-cold/substrate_fingerprint.json')" `
  --fingerprint-file "volvence=$(Join-Path $ArtifactDir 'volvence/substrate_fingerprint.json')"
if ($LASTEXITCODE -ne 0) {
    throw "[serve] same-substrate fingerprint gate failed"
}

Write-Host "[serve] all endpoints launched. PIDs in $PidFile"
Write-Host "[serve] stop with: Get-Content $PidFile | ForEach-Object { Stop-Process -Id `$_ -Force -ErrorAction SilentlyContinue }"
} catch {
    Write-Error $_
    foreach ($log in Get-ChildItem $LogDir -Filter "*.err.log" -ErrorAction SilentlyContinue) {
        Write-Host "[serve] stderr: $($log.FullName)"
        Get-Content $log.FullName -Tail 40
    }
    Stop-AblationServices
    throw
}
