# Same-substrate Companion Bench ablation — Windows launcher (mirrors .sh).
# Requires: VZ_SUBSTRATE_MODEL_ID, LIFEFORM_LOCAL_API_KEY, VZ_TORCH_BACKENDS=active (optional).

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $RepoRoot

if (-not $env:VZ_SUBSTRATE_MODEL_ID) {
    throw "set VZ_SUBSTRATE_MODEL_ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)"
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

$RefhArgs = @(
    "companion-ref-harness", "serve",
    "--port", "8500",
    "--upstream-base-url", $RawUpstream,
    "--upstream-model", "lifeform-raw",
    "--upstream-key-env", "LIFEFORM_LOCAL_API_KEY",
    "--components", "summary,embed,user_model,episodic",
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
    Write-Warning "[serve] no REFH_EXTRACTOR_MODEL -> ref-harness may use same-family upstream"
}
Start-AblationService "ref-harness" $RefhArgs

Start-AblationService "camel-baseline" @(
    "companion-camel-baseline", "serve",
    "--port", "8600",
    "--backend", $CamelBackend,
    "--upstream-base-url", $RawUpstream,
    "--upstream-model", "lifeform-raw",
    "--upstream-key-env", "LIFEFORM_LOCAL_API_KEY",
    "--store-mode", "sqlite",
    "--store-path", (Join-Path $ArtifactDir "camel-baseline.sqlite3")
)

foreach ($track in @("raw", "ref-harness", "camel", "volvence-cold", "volvence")) {
    $dir = Join-Path $ArtifactDir $track
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $fpPath = Join-Path $dir "substrate_fingerprint.json"
    $json = @"
{
  "track": "$track",
  "substrate_model_id": "$($env:VZ_SUBSTRATE_MODEL_ID)",
  "served_at": "$DateTag"
}
"@
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($fpPath, $json, $utf8NoBom)
}

Write-Host "[serve] waiting 90s for model load + endpoints..."
Start-Sleep -Seconds 90

python scripts/companion_bench/assert_same_substrate.py `
  --fingerprint-file "raw=$(Join-Path $ArtifactDir 'raw/substrate_fingerprint.json')" `
  --fingerprint-file "ref-harness=$(Join-Path $ArtifactDir 'ref-harness/substrate_fingerprint.json')" `
  --fingerprint-file "camel=$(Join-Path $ArtifactDir 'camel/substrate_fingerprint.json')" `
  --fingerprint-file "volvence-cold=$(Join-Path $ArtifactDir 'volvence-cold/substrate_fingerprint.json')" `
  --fingerprint-file "volvence=$(Join-Path $ArtifactDir 'volvence/substrate_fingerprint.json')"

Write-Host "[serve] all endpoints launched. PIDs in $PidFile"
Write-Host "[serve] stop with: Get-Content $PidFile | ForEach-Object { Stop-Process -Id `$_ -Force -ErrorAction SilentlyContinue }"
