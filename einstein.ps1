#requires -Version 5.1
<#
.SYNOPSIS
    Windows-native end-to-end Einstein figure-vertical demo driver
    (collect corpus -> bake curated bundle -> bake persona LoRA ->
    run 4-gate verification harness).

.DESCRIPTION
    PowerShell port of the 3-stage Unix pipeline:

      bash scripts/figure_collect_einstein.sh           (collect + curated bundle)
      bash scripts/figure_bake_einstein_persona_lora.sh (PEFT persona LoRA)
      bash scripts/figure_verify_einstein_persona.sh    (4-gate verdict)

    The Unix scripts assume Git Bash / WSL. This file produces the
    same artifacts (data\figure_corpus, data\figure_bundles,
    data\figure_audit, artifacts\figure_verify\<run_id>\) using pure
    PowerShell + ``python -m ...`` invocations -- no bash dependency.

    Two demo modes:

      Smoke (default, fully offline, ~5 minutes on CPU)
        $env:DEMO_MODE = 'smoke'    # or just omit; smoke is default
        .\einstein.ps1
        Defaults to ``tiny-gpt2`` backend + synthetic Einstein corpus.
        Proves the L1/L2/L3/L4 chain wires end-to-end. NO L2 forward
        delta is visible (synthetic LoRA delta is zeroed by Qwen
        LayerNorm; see debt #40). Good as a CI smoke / wiring check.

      Real (opt-in, requires Qwen-1.5B + GPU, ~30-45 minutes)
        $env:DEMO_MODE = 'real'
        $env:PEFT_DEVICE = 'cuda'      # or 'cuda:0', 'cuda:1', etc.
        .\einstein.ps1
        Uses ``Qwen/Qwen2.5-1.5B-Instruct`` + PEFT q/k/v/o LoRA on the
        real Wave K Einstein curated corpus (444 chunks). This is the
        path that produces the hand-on-table demo where ``einstein-raw``
        / ``einstein-bundle`` / ``einstein-full`` show distinct
        behaviour in the chat UI vertical dropdown.

    Phase skipping (re-run a single phase without redoing the earlier
    ones):

      .\einstein.ps1 -SkipCollect                       # bake + verify only
      .\einstein.ps1 -SkipCollect -SkipBake             # verify only
      .\einstein.ps1 -SkipVerify                        # collect + bake only

.EXAMPLE
    # 5-minute CPU smoke (verifies the chain end-to-end without a GPU).
    .\einstein.ps1

.EXAMPLE
    # Full real run (needs GPU + ~20 GB free disk for Qwen-1.5B weights).
    $env:DEMO_MODE = 'real'
    $env:PEFT_DEVICE = 'cuda'
    .\einstein.ps1

.EXAMPLE
    # Re-run only the 4-gate verification on the latest baked bundle.
    .\einstein.ps1 -SkipCollect -SkipBake

.NOTES
    Env vars (all optional; defaults shown):
      DEMO_MODE              smoke (default) | real
      RUN_ID                 einstein-2026Q2

      # Phase 1 (collect / re-clean / curated bundle)
      SEEDS_FILE             packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.jsonl
      METADATA_FILE          packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.curated_metadata.jsonl
      PROVENANCE_FILE        packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.verification_provenance.jsonl
      CORPUS_ROOT            data\figure_corpus
      BUNDLE_ROOT            data\figure_bundles
      AUDIT_ROOT             data\figure_audit
      MAX_PAGES              30
      RATE_RPS               0.5
      BURST                  5
      REQUIRE_VERIFY         1   (curator metadata gates bake-bundle on verifier PASS)

      # Phase 2 (persona LoRA bake)
      QWEN_MODEL_ID          smoke: sshleifer/tiny-gpt2
                             real:  Qwen/Qwen2.5-1.5B-Instruct
      PEFT_TARGET_MODULES    smoke: c_attn
                             real:  q_proj,k_proj,v_proj,o_proj
      PEFT_RANK              8
      PEFT_MAX_STEPS         smoke: 50
                             real:  200
      PEFT_DEVICE            smoke: cpu
                             real:  cuda  (override to 'cpu' to force CPU)
      BUNDLE_ID              autoresolve to newest manifest under BUNDLE_ROOT\einstein\

      # Phase 3 (4-gate verification harness)
      RUNTIME_BACKEND        smoke: synthetic
                             real:  transformers
      VERIFY_OUT             artifacts\figure_verify\<RUN_ID>-<timestamp>
      MAX_IN_CORPUS_QUESTIONS 20

    Outputs:
      data\figure_corpus\crawl\<RUN_ID>\results.jsonl     (crawl ledger)
      data\figure_corpus\cleaned\<sha>\v1\text.txt        (cleaned bodies)
      data\figure_corpus\verification\<sha>\checks.jsonl  (7-axis verifier)
      data\figure_bundles\einstein\<bundle_id>\manifest.json
      data\figure_audit\einstein\<bundle_id>.audit.jsonl  (gate decisions)
      artifacts\figure_verify\<run>\verdict.json          (4-gate pass/fail)
      artifacts\figure_verify\<run>\transcript.md         (raw / bundle / bundle_lora per question)
#>

[CmdletBinding()]
param(
    [switch] $SkipCollect,
    [switch] $SkipBake,
    [switch] $SkipVerify
)

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
if (-not $RootDir) {
    $RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $RootDir

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

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory)] [string] $Label,
        [Parameter(Mandatory)] [string[]] $Args,
        [int[]] $AllowedExitCodes = @(0)
    )
    Write-Host ""
    Write-Host "[$Label] $PythonBin $($Args -join ' ')" -ForegroundColor Cyan
    & $PythonBin @Args
    $code = $LASTEXITCODE
    if ($AllowedExitCodes -notcontains $code) {
        throw "[$Label] python exited with code $code (allowed: $($AllowedExitCodes -join ','))"
    }
    if ($code -ne 0) {
        Write-Host "[$Label] python exited with code $code (allowed)" -ForegroundColor Yellow
    }
    return $code
}

# ---------------------------------------------------------------------------
# Mode + defaults
# ---------------------------------------------------------------------------

Set-DefaultEnv 'DEMO_MODE' 'smoke'
$Mode = $env:DEMO_MODE.ToLowerInvariant()
if ($Mode -notin @('smoke', 'real')) {
    throw "DEMO_MODE must be 'smoke' or 'real', got '$Mode'."
}

Set-DefaultEnv 'RUN_ID' 'einstein-2026Q2'
Set-DefaultEnv 'SEEDS_FILE'      (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.jsonl')
Set-DefaultEnv 'METADATA_FILE'   (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.curated_metadata.jsonl')
Set-DefaultEnv 'PROVENANCE_FILE' (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.verification_provenance.jsonl')
Set-DefaultEnv 'CORPUS_ROOT'     (Join-Path $RootDir 'data\figure_corpus')
Set-DefaultEnv 'BUNDLE_ROOT'     (Join-Path $RootDir 'data\figure_bundles')
Set-DefaultEnv 'AUDIT_ROOT'      (Join-Path $RootDir 'data\figure_audit')
Set-DefaultEnv 'MAX_PAGES'       '30'
Set-DefaultEnv 'RATE_RPS'        '0.5'
Set-DefaultEnv 'BURST'           '5'
Set-DefaultEnv 'REQUIRE_VERIFY'  '1'

# Phase 2/3 defaults depend on mode
if ($Mode -eq 'real') {
    Set-DefaultEnv 'QWEN_MODEL_ID'       'Qwen/Qwen2.5-1.5B-Instruct'
    Set-DefaultEnv 'PEFT_TARGET_MODULES' 'q_proj,k_proj,v_proj,o_proj'
    Set-DefaultEnv 'PEFT_MAX_STEPS'      '200'
    Set-DefaultEnv 'PEFT_DEVICE'         'cuda'
    Set-DefaultEnv 'RUNTIME_BACKEND'     'transformers'
} else {
    Set-DefaultEnv 'QWEN_MODEL_ID'       'sshleifer/tiny-gpt2'
    Set-DefaultEnv 'PEFT_TARGET_MODULES' 'c_attn'
    Set-DefaultEnv 'PEFT_MAX_STEPS'      '50'
    Set-DefaultEnv 'PEFT_DEVICE'         'cpu'
    Set-DefaultEnv 'RUNTIME_BACKEND'     'synthetic'
}
Set-DefaultEnv 'PEFT_RANK' '8'
Set-DefaultEnv 'MAX_IN_CORPUS_QUESTIONS' '20'

$Timestamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
Set-DefaultEnv 'VERIFY_OUT' (Join-Path $RootDir "artifacts\figure_verify\$($env:RUN_ID)-$Timestamp")

# Make sure output dirs exist
New-Item -ItemType Directory -Path $env:CORPUS_ROOT -Force | Out-Null
New-Item -ItemType Directory -Path $env:BUNDLE_ROOT -Force | Out-Null
New-Item -ItemType Directory -Path $env:AUDIT_ROOT  -Force | Out-Null
New-Item -ItemType Directory -Path $env:VERIFY_OUT  -Force | Out-Null

# Build PYTHONPATH from packages\*\src -- Windows uses ';' as the separator.
$packageSrcs = Get-ChildItem -Path (Join-Path $RootDir 'packages') -Directory -ErrorAction Stop |
    ForEach-Object { Join-Path $_.FullName 'src' } |
    Where-Object { Test-Path $_ }
$packagePaths = ($packageSrcs -join ';')
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$packagePaths;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $packagePaths
}

Write-Host "================================================================"
Write-Host " Einstein figure-vertical pipeline (Windows-native)"
Write-Host " mode          = $Mode"
Write-Host " run_id        = $($env:RUN_ID)"
Write-Host " corpus_root   = $($env:CORPUS_ROOT)"
Write-Host " bundle_root   = $($env:BUNDLE_ROOT)"
Write-Host " audit_root    = $($env:AUDIT_ROOT)"
Write-Host " verify_out    = $($env:VERIFY_OUT)"
Write-Host " qwen_model_id = $($env:QWEN_MODEL_ID)"
Write-Host " peft_device   = $($env:PEFT_DEVICE)"
Write-Host " runtime       = $($env:RUNTIME_BACKEND)"
Write-Host " skip          = collect=$($SkipCollect.IsPresent) bake=$($SkipBake.IsPresent) verify=$($SkipVerify.IsPresent)"
Write-Host "================================================================"

# ---------------------------------------------------------------------------
# Phase 1 -- collect corpus + curated bundle
# Mirrors scripts/figure_collect_einstein.sh
# ---------------------------------------------------------------------------

if (-not $SkipCollect) {
    if (-not (Test-Path $env:SEEDS_FILE)) {
        throw "SEEDS_FILE not found: $($env:SEEDS_FILE)"
    }

    $CrawlCli = Join-Path $RootDir 'packages\lifeform-domain-figure\scripts\figure_crawl.py'
    $CleanCli = Join-Path $RootDir 'packages\lifeform-domain-figure\scripts\figure_clean.py'
    $VerifyCli = Join-Path $RootDir 'packages\lifeform-domain-figure\scripts\figure_verify.py'

    Invoke-PythonStep 'phase1.1/5 enqueue-batch' @(
        $CrawlCli, 'enqueue-batch',
        '--root', $env:CORPUS_ROOT,
        '--run-id', $env:RUN_ID,
        '--requests-file', $env:SEEDS_FILE
    )

    Invoke-PythonStep 'phase1.2/5 crawl run' @(
        $CrawlCli, 'run',
        '--root', $env:CORPUS_ROOT,
        '--run-id', $env:RUN_ID,
        '--cleaning-root', $env:CORPUS_ROOT,
        '--rate-rps', $env:RATE_RPS,
        '--burst', $env:BURST,
        '--max-pages', $env:MAX_PAGES
    )

    Invoke-PythonStep 'phase1.3/5 re-clean-all' @(
        $CleanCli, 're-clean-all',
        '--root', $env:CORPUS_ROOT
    )

    if ((Test-Path $env:PROVENANCE_FILE)) {
        Invoke-PythonStep 'phase1.4/5 verify run-batch' @(
            $VerifyCli, 'run-batch',
            '--root', $env:CORPUS_ROOT,
            '--provenance-file', $env:PROVENANCE_FILE,
            '--metadata-mode', 'offline'
        )
    } else {
        Write-Host "[phase1.4/5] skipped (PROVENANCE_FILE not found: $($env:PROVENANCE_FILE))" -ForegroundColor Yellow
    }

    if ((Test-Path $env:METADATA_FILE)) {
        $bakeArgs = @(
            '-m', 'lifeform_domain_figure.cli',
            '--bundle-root', $env:BUNDLE_ROOT,
            '--audit-root', $env:AUDIT_ROOT,
            'bake-bundle',
            '--figure', 'einstein',
            '--corpus-mode', 'curated',
            '--cleaning-root', $env:CORPUS_ROOT,
            '--curated-metadata-file', $env:METADATA_FILE
        )
        if ($env:REQUIRE_VERIFY -eq '1') {
            $bakeArgs += @('--verification-root', $env:CORPUS_ROOT, '--require-verification-pass')
        }
        Invoke-PythonStep 'phase1.5/5 bake-bundle (curated)' $bakeArgs
    } else {
        Write-Host "[phase1.5/5] skipped (METADATA_FILE not found: $($env:METADATA_FILE))" -ForegroundColor Yellow
    }
} else {
    Write-Host "[phase 1] SKIPPED" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# BUNDLE_ID resolution (auto-pick newest manifest under BUNDLE_ROOT\einstein\)
# Mirrors the inline Python in figure_bake_einstein_persona_lora.sh
# ---------------------------------------------------------------------------

function Resolve-BundleId {
    $einsteinRoot = Join-Path $env:BUNDLE_ROOT 'einstein'
    if (-not (Test-Path $einsteinRoot)) {
        throw "No bundle root at $einsteinRoot. Run phase 1 (bake-bundle) first."
    }
    $manifests = @()
    foreach ($d in (Get-ChildItem -Path $einsteinRoot -Directory -ErrorAction SilentlyContinue)) {
        $mf = Join-Path $d.FullName 'manifest.json'
        if (Test-Path $mf) {
            try {
                $payload = Get-Content -LiteralPath $mf -Raw | ConvertFrom-Json
                $createdAt = if ($payload.PSObject.Properties.Match('created_at_iso').Count -and $payload.created_at_iso) {
                    [string]$payload.created_at_iso
                } else { '' }
                $bundleId = if ($payload.PSObject.Properties.Match('bundle_id').Count -and $payload.bundle_id) {
                    [string]$payload.bundle_id
                } else { $d.Name }
                $manifests += [pscustomobject]@{
                    CreatedAt = $createdAt
                    BundleId  = $bundleId
                }
            } catch {
                Write-Host "[resolve-bundle-id] skipping unreadable manifest: $mf" -ForegroundColor Yellow
            }
        }
    }
    if ($manifests.Count -eq 0) {
        throw "No manifests found under $einsteinRoot. Did phase 1 succeed?"
    }
    $latest = $manifests | Sort-Object CreatedAt -Descending | Select-Object -First 1
    return $latest.BundleId
}

# ---------------------------------------------------------------------------
# Phase 2 -- persona LoRA bake (PEFT, OFFLINE-gated)
# Mirrors scripts/figure_bake_einstein_persona_lora.sh
# ---------------------------------------------------------------------------

if (-not $SkipBake) {
    if (-not (Test-Path $env:METADATA_FILE)) {
        throw "Phase 2 requires METADATA_FILE: $($env:METADATA_FILE)"
    }
    if ([string]::IsNullOrEmpty($env:BUNDLE_ID)) {
        $env:BUNDLE_ID = Resolve-BundleId
    }
    Set-DefaultEnv 'ROLLBACK_EVIDENCE' "prev_persona_lora=absent;base=$($env:BUNDLE_ID)"

    Write-Host ""
    Write-Host "[phase2] bake-lora bundle=$($env:BUNDLE_ID) backend=peft model=$($env:QWEN_MODEL_ID)" -ForegroundColor Cyan

    Invoke-PythonStep 'phase2/3 bake-lora (curated)' @(
        '-m', 'lifeform_domain_figure.cli',
        '--bundle-root', $env:BUNDLE_ROOT,
        '--audit-root', $env:AUDIT_ROOT,
        'bake-lora',
        '--figure', 'einstein',
        '--bundle', $env:BUNDLE_ID,
        '--corpus-mode', 'curated',
        '--cleaning-root', $env:CORPUS_ROOT,
        '--curated-metadata-file', $env:METADATA_FILE,
        '--backend', 'peft',
        '--rank', $env:PEFT_RANK,
        '--peft-model-id', $env:QWEN_MODEL_ID,
        '--peft-target-modules', $env:PEFT_TARGET_MODULES,
        '--peft-max-steps', $env:PEFT_MAX_STEPS,
        '--peft-device', $env:PEFT_DEVICE,
        '--evaluation-snapshot', 'default-clean',
        '--rollback-evidence', $env:ROLLBACK_EVIDENCE
    )
} else {
    Write-Host "[phase 2] SKIPPED" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Phase 3 -- 4-gate persona verification harness
# Mirrors scripts/figure_verify_einstein_persona.sh
# ---------------------------------------------------------------------------

if (-not $SkipVerify) {
    if ([string]::IsNullOrEmpty($env:BUNDLE_ID)) {
        $env:BUNDLE_ID = Resolve-BundleId
    }

    Write-Host ""
    Write-Host "[phase3] verification bundle=$($env:BUNDLE_ID) runtime=$($env:RUNTIME_BACKEND)" -ForegroundColor Cyan

    # Exit code semantics mirror the bash harness:
    #   0 = all 4 gates PASS
    #   2 = harness completed; one or more gates FAIL (verdict.json still written)
    #   3 = setup error (treated as a real failure)
    # Anything else = real crash. Treat 0 and 2 as legitimate so the
    # script keeps going and prints the verdict on smoke runs (where
    # synthetic LoRA forces bundle ≡ bundle_lora and the LoRA gate
    # FAILS by design — see debt #40).
    $verifyCode = Invoke-PythonStep 'phase3 persona-verification' @(
        '-m', 'lifeform_domain_figure.verification.persona.cli',
        '--bundle-id', $env:BUNDLE_ID,
        '--figure', 'einstein',
        '--bundle-root', $env:BUNDLE_ROOT,
        '--output-dir', $env:VERIFY_OUT,
        '--runtime', $env:RUNTIME_BACKEND,
        '--qwen-model-id', $env:QWEN_MODEL_ID,
        '--max-in-corpus-questions', $env:MAX_IN_CORPUS_QUESTIONS,
        '--conditions', 'raw,bundle,bundle_lora',
        '--questions-cache', (Join-Path $env:VERIFY_OUT 'questions.jsonl')
    ) -AllowedExitCodes @(0, 2)

    $verdictFile = Join-Path $env:VERIFY_OUT 'verdict.json'
    if (Test-Path $verdictFile) {
        $verdictColor = if ($verifyCode -eq 0) { 'Green' } else { 'Yellow' }
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor $verdictColor
        Write-Host " verdict.json:" -ForegroundColor $verdictColor
        Get-Content -LiteralPath $verdictFile -Raw
        Write-Host ""
        Write-Host " transcript: $(Join-Path $env:VERIFY_OUT 'transcript.md')" -ForegroundColor $verdictColor
        if ($verifyCode -eq 2) {
            Write-Host " note: one or more gates FAILED (exit 2). On a smoke run with" -ForegroundColor Yellow
            Write-Host "       runtime=synthetic this is expected (synthetic LoRA delta is" -ForegroundColor Yellow
            Write-Host "       zeroed by LayerNorm; bundle ≡ bundle_lora; see debt #40)." -ForegroundColor Yellow
        }
        Write-Host "================================================================" -ForegroundColor $verdictColor
    }
} else {
    Write-Host "[phase 3] SKIPPED" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "pipeline complete" -ForegroundColor Green
