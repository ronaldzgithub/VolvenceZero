# Same-substrate Companion Bench ablation — Windows launcher (mirrors .sh).
# Topology: one lifeform-serve --ablation-bundle process on :8000 owns the
# frozen HF runtime; ref-harness (:8500) and camel (:8600) consume :8000 raw.
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
Start-AblationService "lifeform-ablation-bundle" @(
    "lifeform-serve",
    "--ablation-bundle",
    "--port", "8000",
    "--substrate-mode", "hf-shared",
    "--substrate-model-id", $env:VZ_SUBSTRATE_MODEL_ID,
    "--substrate-device", $Device,
    # Preflight already verified + fingerprinted the local weight cache.
    # Serving must not depend on HF Hub reachability (a proxy flake here
    # previously triggered a silent builtin-fallback substrate).
    "--substrate-local-files-only",
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

# GAP-11: independent registry claim-2 arms. :8501 memory-only (summary +
# user_model + episodic, NO retrieval) and :8502 rag (embed retrieval ONLY).
$MemoryOnlyArgs = @(
    "companion-ref-harness", "serve",
    "--port", "8501",
    "--upstream-base-url", $RawUpstream,
    "--upstream-model", "lifeform-raw",
    "--upstream-key-env", "LIFEFORM_LOCAL_API_KEY",
    "--components", "summary,user_model,episodic",
    "--store-mode", "sqlite",
    "--store-path", (Join-Path $ArtifactDir "memory-only.sqlite3"),
    "--summary-extractor-base-url", $env:REFH_EXTRACTOR_BASE_URL,
    "--summary-extractor-model", $env:REFH_EXTRACTOR_MODEL,
    "--summary-extractor-key-env", $env:REFH_EXTRACTOR_KEY_ENV,
    "--summary-extractor-family", $(if ($env:REFH_EXTRACTOR_FAMILY) { $env:REFH_EXTRACTOR_FAMILY } else { "openai-compat" })
)
Start-AblationService "memory-only" $MemoryOnlyArgs

$RagArgs = @(
    "companion-ref-harness", "serve",
    "--port", "8502",
    "--upstream-base-url", $RawUpstream,
    "--upstream-model", "lifeform-raw",
    "--upstream-key-env", "LIFEFORM_LOCAL_API_KEY",
    "--components", "embed",
    "--embedder", $RefhEmbedder,
    "--store-mode", "sqlite",
    "--store-path", (Join-Path $ArtifactDir "rag.sqlite3")
)
Start-AblationService "rag" $RagArgs

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

Wait-AblationEndpoint "lifeform-ablation-bundle" "http://127.0.0.1:8000/v1/health"
Wait-AblationEndpoint "ref-harness" "http://127.0.0.1:8500/healthz"
Wait-AblationEndpoint "memory-only" "http://127.0.0.1:8501/healthz"
Wait-AblationEndpoint "rag" "http://127.0.0.1:8502/healthz"
Wait-AblationEndpoint "camel-baseline" "http://127.0.0.1:8600/healthz"

$Pids = @(Get-Content $PidFile | Where-Object { $_ })
$Topology = [ordered]@{
    schema_version = "companion-ablation-serving-topology.v1"
    serving_topology = "single-lifeform-ablation-bundle"
    lifeform_owner_pid = if ($Pids.Count -ge 1) { [int]$Pids[0] } else { $null }
    process_count = $Pids.Count
    ports = @(8000, 8500, 8501, 8502, 8600)
    ablation_verticals = @(
        "companion",
        "companion-cold",
        "companion-pe-drive-off",
        "companion-eta-off",
        "companion-active-learning-off",
        "companion-lora-adapter"
    )
}
# Write BOM-less UTF-8: PS 5.1's `Set-Content -Encoding utf8` emits a BOM,
# which strict json.loads(encoding="utf-8") consumers reject.
$TopologyJson = $Topology | ConvertTo-Json -Depth 4
$TopologyPath = Join-Path (Resolve-Path $ArtifactDir) "serve_topology.json"
[System.IO.File]::WriteAllText($TopologyPath, $TopologyJson, [System.Text.UTF8Encoding]::new($false))

python scripts/companion_bench/assert_same_substrate.py `
  --require-weights-sha256 `
  --fingerprint-file "raw=$(Join-Path $ArtifactDir 'raw/substrate_fingerprint.json')" `
  --fingerprint-file "ref-harness=$(Join-Path $ArtifactDir 'ref-harness/substrate_fingerprint.json')" `
  --fingerprint-file "memory-only=$(Join-Path $ArtifactDir 'memory-only/substrate_fingerprint.json')" `
  --fingerprint-file "rag=$(Join-Path $ArtifactDir 'rag/substrate_fingerprint.json')" `
  --fingerprint-file "camel=$(Join-Path $ArtifactDir 'camel/substrate_fingerprint.json')" `
  --fingerprint-file "volvence-cold=$(Join-Path $ArtifactDir 'volvence-cold/substrate_fingerprint.json')" `
  --fingerprint-file "volvence=$(Join-Path $ArtifactDir 'volvence/substrate_fingerprint.json')" `
  --fingerprint-file "pe-off=$(Join-Path $ArtifactDir 'pe-off/substrate_fingerprint.json')" `
  --fingerprint-file "eta-off=$(Join-Path $ArtifactDir 'eta-off/substrate_fingerprint.json')" `
  --fingerprint-file "active-learning-off=$(Join-Path $ArtifactDir 'active-learning-off/substrate_fingerprint.json')" `
  --fingerprint-file "lora-adapter=$(Join-Path $ArtifactDir 'lora-adapter/substrate_fingerprint.json')"
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
