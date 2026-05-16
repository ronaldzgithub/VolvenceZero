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

    Mode + Device + RequireVerify + RunId are first-class CLI
    parameters. Less-used overrides (file paths, model id, PEFT
    target modules, etc.) still fall back to the documented env vars
    listed in .NOTES for power users.

.PARAMETER Mode
    'smoke' (default; fully offline, ~5 minutes on CPU) -- proves the
    L1/L2/L3/L4 chain wires end-to-end with tiny-gpt2 + synthetic
    runtime. No L2 forward delta is visible (synthetic LoRA delta is
    zeroed by LayerNorm; see debt #40).

    'real' (opt-in, ~30-45 minutes, GPU/MPS strongly recommended) --
    Qwen/Qwen2.5-1.5B-Instruct + PEFT q/k/v/o LoRA on the Wave K
    Einstein curated corpus. Produces the hand-on-table demo where
    einstein-raw / einstein-bundle / einstein-full show distinct
    behaviour.

.PARAMETER Device
    Torch device for the PEFT persona bake AND the Phase 3 transformers
    runtime. 'cpu' / 'cuda' / 'mps' / 'auto' (default 'auto').

    'auto' resolves per-mode: smoke -> cpu; real -> first available of
    cuda / mps / cpu (probed once via python -c "import torch; ...").
    'mps' (Apple Silicon Metal) is honoured but only usable when this
    script is run via PowerShell Core on macOS.

.PARAMETER RequireVerify
    Whether bake-bundle gates on verifier PASS. '0' / '1' / 'auto'
    (default 'auto').

    'auto' resolves per-mode: smoke -> 0 (offline V1 metadata stubs
    guarantee NEEDS_REVIEW on 4 of 7 axes; gating would always BLOCK;
    smoke is a wiring check, not a curatorial verification);
    real -> 1 (curator metadata gates bake-bundle on verifier PASS;
    run with --metadata-mode=live + reviewed overrides to satisfy it).

.PARAMETER RunId
    Crawl + verification run identifier (default 'einstein-2026Q2').
    Affects data\figure_corpus\crawl\<RUN_ID>\ and
    artifacts\figure_verify\<RUN_ID>-<timestamp>\.

.PARAMETER SkipCollect
    Skip Phase 1 (corpus collect + curated bundle bake).

.PARAMETER SkipBake
    Skip Phase 2 (persona LoRA bake).

.PARAMETER SkipVerify
    Skip Phase 3 (4-gate persona verification).

.EXAMPLE
    # 5-minute CPU smoke (default; verifies the chain end-to-end
    # without a GPU).
    .\einstein.ps1

.EXAMPLE
    # Full real run, auto-detect the best torch device (cuda -> mps -> cpu).
    .\einstein.ps1 -Mode real

.EXAMPLE
    # Real run, pinned to CPU.
    .\einstein.ps1 -Mode real -Device cpu

.EXAMPLE
    # Real run on Apple Silicon (Metal backend).
    .\einstein.ps1 -Mode real -Device mps

.EXAMPLE
    # Real run but skip the curator verification gate (e.g. you have
    # no live-metadata override staged yet and just want a bundle).
    .\einstein.ps1 -Mode real -RequireVerify 0

.EXAMPLE
    # Re-run only the 4-gate verification on the latest baked bundle.
    .\einstein.ps1 -SkipCollect -SkipBake

.NOTES
    Less-used env-var overrides (consulted only when the matching CLI
    flag was NOT supplied). All optional; defaults shown.

      # Phase 1 (collect / re-clean / curated bundle)
      SEEDS_FILE             packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.jsonl
      METADATA_FILE          packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.curated_metadata.jsonl
      PROVENANCE_FILE        packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.verification_provenance.jsonl
      FIGURE_CONTEXT_FILE    packages\lifeform-domain-figure\data\seeds\einstein-figure-context.json
                             (per-figure constants for the 4 metadata-driven verifiers;
                             ignored unless --metadata-mode=live and curator overrides land)
      CORPUS_ROOT            data\figure_corpus
      BUNDLE_ROOT            data\figure_bundles
      AUDIT_ROOT             data\figure_audit
      MAX_PAGES              30
      RATE_RPS               0.5
      BURST                  5

      # Phase 2 (persona LoRA bake)
      QWEN_MODEL_ID          smoke: sshleifer/tiny-gpt2
                             real:  Qwen/Qwen2.5-1.5B-Instruct
      PEFT_TARGET_MODULES    smoke: c_attn
                             real:  q_proj,k_proj,v_proj,o_proj
      PEFT_RANK              8
      PEFT_MAX_STEPS         smoke: 50
                             real:  200
      BUNDLE_ID              autoresolve to newest manifest under BUNDLE_ROOT\einstein\

      # Phase 3 (4-gate verification harness)
      RUNTIME_BACKEND        smoke: synthetic
                             real:  transformers
      VERIFY_OUT             artifacts\figure_verify\<RUN_ID>-<timestamp>
      MAX_IN_CORPUS_QUESTIONS 20

      # Misc
      PYTHON                 python executable to use (default 'python')

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
    [Parameter(Position = 0)]
    [ValidateSet('smoke', 'real')]
    [string] $Mode = 'smoke',

    [Parameter(Position = 1)]
    [ValidateSet('cpu', 'cuda', 'mps', 'auto')]
    [string] $Device = 'auto',

    [ValidateSet('0', '1', 'auto')]
    [string] $RequireVerify = 'auto',

    [string] $RunId = 'einstein-2026Q2',

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

# Resolve a config value for *less-used* overrides: prefer a user-
# supplied env var, otherwise use the script default. **Read-only on
# the environment** -- never writes back. This is the power-user
# escape hatch for path-like / advanced knobs; everyday flags
# (Mode / Device / RequireVerify / RunId) live on the param block
# above and don't go through here.
#
# An earlier version of this script used a Set-DefaultEnv helper that
# stamped each derived default into the current PowerShell session's
# $env: on first run, so the next run (in the same shell) saw the
# previous run's mode-dependent defaults as "user-supplied" and
# refused to update them when the mode flipped. Never write back.
function Get-DefaultValue {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [AllowEmptyString()] [string] $Default
    )
    $current = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrEmpty($current)) {
        return $Default
    }
    return $current
}

# Probe torch for the best available accelerator. Returns one of
# 'cuda' / 'mps' / 'cpu'. Falls back to 'cpu' silently when torch is
# unimportable -- smoke mode never reaches torch so a fresh env is OK.
function Resolve-AutoDevice {
    $probeScript = @'
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
'@
    try {
        $detected = & $PythonBin -c $probeScript 2>$null
    } catch {
        return 'cpu'
    }
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($detected)) {
        return 'cpu'
    }
    $token = ($detected | Select-Object -Last 1).ToString().Trim()
    if ($token -notin @('cpu', 'cuda', 'mps')) {
        return 'cpu'
    }
    return $token
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory)] [string] $Label,
        [Parameter(Mandatory)] [string[]] $Args,
        [int[]] $AllowedExitCodes = @(0),
        # When set, capture stdout and also tee it to the console so the
        # caller can parse the JSON tail (e.g. bake-lora's bundle_id).
        # The captured value is returned as a string; the exit code is
        # exposed via the script-scoped $script:LastPythonExitCode.
        [switch] $CaptureStdout
    )
    Write-Host ""
    Write-Host "[$Label] $PythonBin $($Args -join ' ')" -ForegroundColor Cyan
    if ($CaptureStdout) {
        # Tee-Object -Variable writes to the variable AND forwards
        # down the pipeline. Inside a function the unconsumed pipeline
        # tail joins the function's *return value*, mixing every
        # stdout line with the joined string we explicitly return.
        # Pipe to Out-Host to render lines on the console *and* drain
        # them out of the function's output stream. We deliberately do
        # NOT use 2>&1 -- PowerShell wraps native-command stderr in
        # RemoteException records and `$ErrorActionPreference = 'Stop'`
        # would terminate the script on any HF/PEFT warning. Stderr
        # streams naturally to the console without merging.
        & $PythonBin @Args | Tee-Object -Variable teedOutput | Out-Host
        $code = $LASTEXITCODE
        $script:LastPythonExitCode = $code
        $captured = ($teedOutput | ForEach-Object { $_.ToString() }) -join "`n"
    } else {
        & $PythonBin @Args
        $code = $LASTEXITCODE
        $script:LastPythonExitCode = $code
        $captured = $null
    }
    if ($AllowedExitCodes -notcontains $code) {
        throw "[$Label] python exited with code $code (allowed: $($AllowedExitCodes -join ','))"
    }
    if ($code -ne 0) {
        Write-Host "[$Label] python exited with code $code (allowed)" -ForegroundColor Yellow
    }
    if ($CaptureStdout) {
        return $captured
    }
    return $code
}

# Extract bundle_id from a Python CLI JSON tail (bake-bundle / bake-lora /
# similar emit a single JSON object as stdout). We scan from the last '}'
# backwards to the matching '{' to tolerate warning lines interleaved
# above the JSON object (HF tokenizer warnings, PEFT UserWarnings, etc.).
function Get-BundleIdFromStdout {
    param(
        [Parameter(Mandatory)] [string] $Stdout
    )
    $closeIdx = $Stdout.LastIndexOf('}')
    if ($closeIdx -lt 0) {
        throw "Get-BundleIdFromStdout: no '}' in captured stdout"
    }
    # Walk back to the matching '{' by bracket counting.
    $depth = 0
    $openIdx = -1
    for ($i = $closeIdx; $i -ge 0; $i--) {
        $ch = $Stdout[$i]
        if ($ch -eq '}') { $depth++ }
        elseif ($ch -eq '{') {
            $depth--
            if ($depth -eq 0) { $openIdx = $i; break }
        }
    }
    if ($openIdx -lt 0) {
        throw "Get-BundleIdFromStdout: no balanced '{...}' found in captured stdout"
    }
    $jsonBlob = $Stdout.Substring($openIdx, $closeIdx - $openIdx + 1)
    try {
        $payload = $jsonBlob | ConvertFrom-Json
    } catch {
        throw "Get-BundleIdFromStdout: failed to parse JSON tail: $_"
    }
    if (-not $payload.PSObject.Properties.Match('bundle_id').Count) {
        throw "Get-BundleIdFromStdout: JSON payload missing 'bundle_id' key"
    }
    return [string]$payload.bundle_id
}

# ---------------------------------------------------------------------------
# Mode + defaults
# ---------------------------------------------------------------------------
# - Mode / RunId / SkipCollect / SkipBake / SkipVerify come straight
#   from the param block.
# - Device + RequireVerify resolve their 'auto' sentinel here per mode.
# - All other defaults are script-local (never written to $env:) so
#   re-invoking the script in the same shell never picks up a previous
#   run's derived state. Only PYTHONPATH is exported (python needs it).

# Phase 2/3 + verification-gate defaults depend on mode. Power users
# can still override individual axes via env var (consulted only when
# the env var is non-empty -- see Get-DefaultValue).
if ($Mode -eq 'real') {
    $QwenModelId       = Get-DefaultValue 'QWEN_MODEL_ID'       'Qwen/Qwen2.5-1.5B-Instruct'
    $PeftTargetModules = Get-DefaultValue 'PEFT_TARGET_MODULES' 'q_proj,k_proj,v_proj,o_proj'
    $PeftMaxSteps      = Get-DefaultValue 'PEFT_MAX_STEPS'      '200'
    $RuntimeBackend    = Get-DefaultValue 'RUNTIME_BACKEND'     'transformers'
    $defaultRequireVerify = '1'
    # Rationale: real mode opts into the curator verification gate by
    # default. The caller is expected to have arranged
    # --metadata-mode=live + human review overrides for the 4
    # metadata-driven axes (debt #26 closure path); without that,
    # expect BLOCK. Pass -RequireVerify 0 to opt out for now.
} else {
    $QwenModelId       = Get-DefaultValue 'QWEN_MODEL_ID'       'sshleifer/tiny-gpt2'
    $PeftTargetModules = Get-DefaultValue 'PEFT_TARGET_MODULES' 'c_attn'
    $PeftMaxSteps      = Get-DefaultValue 'PEFT_MAX_STEPS'      '50'
    $RuntimeBackend    = Get-DefaultValue 'RUNTIME_BACKEND'     'synthetic'
    $defaultRequireVerify = '0'
    # Rationale: smoke mode is offline-only and a wiring check; the V1
    # metadata stubs deliberately land NEEDS_REVIEW on 4 of 7 axes
    # (fenced by test_run_batch_metadata_axes_default_to_needs_review_offline),
    # so gating bake-bundle would be a guaranteed BLOCK with no path
    # forward. Leave gating off so smoke can complete Phase 2/3.
}

# Resolve -Device: 'auto' -> cpu in smoke, probe in real.
if ($Device -eq 'auto') {
    if ($Mode -eq 'real') {
        $PeftDevice = Resolve-AutoDevice
        Write-Host "[device-detect] -Device auto -> $PeftDevice (probed via torch)" -ForegroundColor DarkGray
    } else {
        $PeftDevice = 'cpu'
    }
} else {
    $PeftDevice = $Device
}

# Resolve -RequireVerify: 'auto' -> mode-derived default (0 smoke, 1 real).
if ($RequireVerify -eq 'auto') {
    $RequireVerify = $defaultRequireVerify
}

# Less-used path / numeric overrides keep their env-var fallback for
# power users. These are NOT mode-dependent and NOT writable back to
# env, so they don't suffer the previous-run pollution problem.
$SeedsFile         = Get-DefaultValue 'SEEDS_FILE'         (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.jsonl')
$MetadataFile      = Get-DefaultValue 'METADATA_FILE'      (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.curated_metadata.jsonl')
$ProvenanceFile    = Get-DefaultValue 'PROVENANCE_FILE'    (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-2026Q2.verification_provenance.jsonl')
$FigureContextFile = Get-DefaultValue 'FIGURE_CONTEXT_FILE' (Join-Path $RootDir 'packages\lifeform-domain-figure\data\seeds\einstein-figure-context.json')
$CorpusRoot        = Get-DefaultValue 'CORPUS_ROOT'        (Join-Path $RootDir 'data\figure_corpus')
$BundleRoot        = Get-DefaultValue 'BUNDLE_ROOT'        (Join-Path $RootDir 'data\figure_bundles')
$AuditRoot         = Get-DefaultValue 'AUDIT_ROOT'         (Join-Path $RootDir 'data\figure_audit')
$MaxPages          = Get-DefaultValue 'MAX_PAGES'          '30'
$RateRps           = Get-DefaultValue 'RATE_RPS'           '0.5'
$Burst             = Get-DefaultValue 'BURST'              '5'

$PeftRank             = Get-DefaultValue 'PEFT_RANK'             '8'
$MaxInCorpusQuestions = Get-DefaultValue 'MAX_IN_CORPUS_QUESTIONS' '20'

$Timestamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
$VerifyOut = Get-DefaultValue 'VERIFY_OUT' (Join-Path $RootDir "artifacts\figure_verify\$RunId-$Timestamp")

# BUNDLE_ID / ROLLBACK_EVIDENCE start as user overrides if supplied,
# otherwise stay empty and get resolved later (post bake-bundle).
$BundleId         = Get-DefaultValue 'BUNDLE_ID'         ''
$RollbackEvidence = Get-DefaultValue 'ROLLBACK_EVIDENCE' ''

# Make sure output dirs exist
New-Item -ItemType Directory -Path $CorpusRoot -Force | Out-Null
New-Item -ItemType Directory -Path $BundleRoot -Force | Out-Null
New-Item -ItemType Directory -Path $AuditRoot  -Force | Out-Null
New-Item -ItemType Directory -Path $VerifyOut  -Force | Out-Null

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
Write-Host " run_id        = $RunId"
Write-Host " corpus_root   = $CorpusRoot"
Write-Host " bundle_root   = $BundleRoot"
Write-Host " audit_root    = $AuditRoot"
Write-Host " verify_out    = $VerifyOut"
Write-Host " qwen_model_id = $QwenModelId"
Write-Host " peft_device   = $PeftDevice"
Write-Host " runtime       = $RuntimeBackend"
Write-Host " require_verify= $RequireVerify  (1=bake-bundle gates on verifier PASS)"
Write-Host " skip          = collect=$($SkipCollect.IsPresent) bake=$($SkipBake.IsPresent) verify=$($SkipVerify.IsPresent)"
Write-Host "================================================================"

# ---------------------------------------------------------------------------
# Phase 1 -- collect corpus + curated bundle
# Mirrors scripts/figure_collect_einstein.sh
# ---------------------------------------------------------------------------

if (-not $SkipCollect) {
    if (-not (Test-Path $SeedsFile)) {
        throw "SEEDS_FILE not found: $SeedsFile"
    }

    $CrawlCli = Join-Path $RootDir 'packages\lifeform-domain-figure\scripts\figure_crawl.py'
    $CleanCli = Join-Path $RootDir 'packages\lifeform-domain-figure\scripts\figure_clean.py'
    $VerifyCli = Join-Path $RootDir 'packages\lifeform-domain-figure\scripts\figure_verify.py'

    Invoke-PythonStep 'phase1.1/5 enqueue-batch' @(
        $CrawlCli, 'enqueue-batch',
        '--root', $CorpusRoot,
        '--run-id', $RunId,
        '--requests-file', $SeedsFile
    )

    Invoke-PythonStep 'phase1.2/5 crawl run' @(
        $CrawlCli, 'run',
        '--root', $CorpusRoot,
        '--run-id', $RunId,
        '--cleaning-root', $CorpusRoot,
        '--rate-rps', $RateRps,
        '--burst', $Burst,
        '--max-pages', $MaxPages
    )

    Invoke-PythonStep 'phase1.3/5 re-clean-all' @(
        $CleanCli, 're-clean-all',
        '--root', $CorpusRoot
    )

    if ((Test-Path $ProvenanceFile)) {
        $verifyArgs = @(
            $VerifyCli, 'run-batch',
            '--root', $CorpusRoot,
            '--provenance-file', $ProvenanceFile,
            '--metadata-mode', 'offline'
        )
        if ((-not [string]::IsNullOrEmpty($FigureContextFile)) -and (Test-Path $FigureContextFile)) {
            $verifyArgs += @('--figure-context-file', $FigureContextFile)
        }
        Invoke-PythonStep 'phase1.4/5 verify run-batch' $verifyArgs
    } else {
        Write-Host "[phase1.4/5] skipped (PROVENANCE_FILE not found: $ProvenanceFile)" -ForegroundColor Yellow
    }

    if ((Test-Path $MetadataFile)) {
        $bakeArgs = @(
            '-m', 'lifeform_domain_figure.cli',
            '--bundle-root', $BundleRoot,
            '--audit-root', $AuditRoot,
            'bake-bundle',
            '--figure', 'einstein',
            '--corpus-mode', 'curated',
            '--cleaning-root', $CorpusRoot,
            '--curated-metadata-file', $MetadataFile
        )
        if ($RequireVerify -eq '1') {
            $bakeArgs += @('--verification-root', $CorpusRoot, '--require-verification-pass')
        }
        # Exit code 2 = OFFLINE gate BLOCKed the bundle build. This is a
        # legitimate signal -- the audit row was written, the block_reason
        # is in stdout above. Not a crash. We allow it here and stop the
        # pipeline cleanly with actionable advice below (phases 2/3 cannot
        # proceed without a bundle so there is nothing else to do).
        Invoke-PythonStep 'phase1.5/5 bake-bundle (curated)' $bakeArgs -AllowedExitCodes @(0, 2)
        if ($script:LastPythonExitCode -eq 2) {
            Write-Host ""
            Write-Host "=================================================================" -ForegroundColor Yellow
            Write-Host " phase 1.5: OFFLINE gate BLOCKED bundle compilation (exit 2)." -ForegroundColor Yellow
            Write-Host "" -ForegroundColor Yellow
            Write-Host " The audit row is at" -ForegroundColor Yellow
            Write-Host "   $AuditRoot\<timestamp>_BAKE_BUNDLE_einstein_*.json" -ForegroundColor Yellow
            Write-Host " and carries the verifier failures that triggered the BLOCK." -ForegroundColor Yellow
            Write-Host "" -ForegroundColor Yellow
            Write-Host " Most common cause: --metadata-mode=offline ships V1 stubs that" -ForegroundColor Yellow
            Write-Host " mark 4 of 7 verifier axes as NEEDS_REVIEW; --require-verification-pass" -ForegroundColor Yellow
            Write-Host " then always BLOCKs unless live metadata + reviewed overrides are" -ForegroundColor Yellow
            Write-Host " staged. This is by design (R10 OFFLINE gate)." -ForegroundColor Yellow
            Write-Host "" -ForegroundColor Yellow
            Write-Host " Phases 2 and 3 cannot run without a bundle. Pick one of:" -ForegroundColor Yellow
            if ($RequireVerify -eq '1') {
                Write-Host "   .\einstein.ps1 -Mode $Mode -RequireVerify 0" -ForegroundColor Cyan
                Write-Host "       Skip the curator verification gate; bundle is baked" -ForegroundColor Gray
                Write-Host "       with verifier NEEDS_REVIEW notes attached. L3 evidence" -ForegroundColor Gray
                Write-Host "       + L4 refusal demo signal is still live." -ForegroundColor Gray
                Write-Host "" -ForegroundColor Yellow
            }
            Write-Host "   .\einstein.ps1" -ForegroundColor Cyan
            Write-Host "       Smoke mode (default); RequireVerify=0 auto, tiny-gpt2," -ForegroundColor Gray
            Write-Host "       fully offline, ~30s on CPU. Wiring check, not curation." -ForegroundColor Gray
            Write-Host "=================================================================" -ForegroundColor Yellow
            exit 2
        }
    } else {
        Write-Host "[phase1.5/5] skipped (METADATA_FILE not found: $MetadataFile)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[phase 1] SKIPPED" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# BUNDLE_ID resolution (auto-pick newest manifest under BUNDLE_ROOT\einstein\)
# Mirrors the inline Python in figure_bake_einstein_persona_lora.sh
# ---------------------------------------------------------------------------

function Resolve-BundleId {
    param(
        [Parameter(Mandatory)] [string] $BundleRoot
    )
    $einsteinRoot = Join-Path $BundleRoot 'einstein'
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
    if (-not (Test-Path $MetadataFile)) {
        throw "Phase 2 requires METADATA_FILE: $MetadataFile"
    }
    # The "source" bundle for bake-lora is the curated bundle from
    # Phase 1 (or the latest curated bundle on disk if Phase 1 was
    # skipped). Capture it under a distinct name -- the post-bake
    # bundle id (with `.lora` attached) replaces $BundleId below so
    # Phase 3 verifies the LoRA-bearing bundle, not the base.
    if ([string]::IsNullOrEmpty($BundleId)) {
        $BundleId = Resolve-BundleId -BundleRoot $BundleRoot
    }
    $SourceBundleId = $BundleId
    if ([string]::IsNullOrEmpty($RollbackEvidence)) {
        $RollbackEvidence = "prev_persona_lora=absent;base=$SourceBundleId"
    }

    Write-Host ""
    Write-Host "[phase2] bake-lora bundle=$SourceBundleId backend=peft model=$QwenModelId" -ForegroundColor Cyan

    # Exit code 2 = OFFLINE gate BLOCKed the LoRA artifact (e.g. the
    # downstream validation_delta / capacity_cost / rollback_evidence
    # contract failed). Same semantic as phase 1.5 -- legitimate gate
    # decision, not a script crash. We allow it here and report the
    # block cleanly so Phase 3 can be re-run against the latest
    # already-existing LoRA-bearing bundle (or none, in which case
    # Phase 3 uses the source bundle and BUNDLE_LORA falls through
    # to BUNDLE behaviour -- which is still useful for L4/L3 demo).
    $bakeLoraStdout = Invoke-PythonStep 'phase2/3 bake-lora (curated)' @(
        '-m', 'lifeform_domain_figure.cli',
        '--bundle-root', $BundleRoot,
        '--audit-root', $AuditRoot,
        'bake-lora',
        '--figure', 'einstein',
        '--bundle', $SourceBundleId,
        '--corpus-mode', 'curated',
        '--cleaning-root', $CorpusRoot,
        '--curated-metadata-file', $MetadataFile,
        '--backend', 'peft',
        '--rank', $PeftRank,
        '--peft-model-id', $QwenModelId,
        '--peft-target-modules', $PeftTargetModules,
        '--peft-max-steps', $PeftMaxSteps,
        '--peft-device', $PeftDevice,
        '--evaluation-snapshot', 'default-clean',
        '--rollback-evidence', $RollbackEvidence
    ) -CaptureStdout -AllowedExitCodes @(0, 2)

    if ($script:LastPythonExitCode -eq 2) {
        Write-Host ""
        Write-Host "=================================================================" -ForegroundColor Yellow
        Write-Host " phase 2: OFFLINE gate BLOCKED persona LoRA artifact (exit 2)." -ForegroundColor Yellow
        Write-Host " The audit row records the block_reason; the source bundle" -ForegroundColor Yellow
        Write-Host " ($SourceBundleId) is unchanged." -ForegroundColor Yellow
        Write-Host "" -ForegroundColor Yellow
        Write-Host " Phase 3 cannot verify a non-existent LoRA-bearing bundle." -ForegroundColor Yellow
        Write-Host " Re-run with -SkipBake to verify the base bundle's L4 refusal" -ForegroundColor Yellow
        Write-Host " + L3 evidence signal against raw (BUNDLE_LORA condition will" -ForegroundColor Yellow
        Write-Host " fall through to BUNDLE since no LoRA is registered):" -ForegroundColor Yellow
        Write-Host "" -ForegroundColor Yellow
        Write-Host "   .\einstein.ps1 -Mode $Mode -SkipCollect -SkipBake" -ForegroundColor Cyan
        Write-Host "=================================================================" -ForegroundColor Yellow
        exit 2
    }

    # bake-lora emits a new bundle with the `.lora` artifact attached;
    # the source bundle is left untouched. Phase 3 (persona verify)
    # needs the *post-bake* bundle so its in-process
    # `_ensure_pool_has_bundle_lora` helper can auto-register the
    # LoRA from `bundle.lora`. Otherwise BUNDLE_LORA silently falls
    # through to BUNDLE behaviour (see persona/cli.py:220-261), which
    # would mask the LoRA gate's real signal.
    $PostBakeBundleId = Get-BundleIdFromStdout -Stdout $bakeLoraStdout
    if ($PostBakeBundleId -ne $SourceBundleId) {
        Write-Host "[phase2] post-bake bundle_id=$PostBakeBundleId (was $SourceBundleId); Phase 3 will verify the LoRA-bearing bundle" -ForegroundColor Cyan
    }
    $BundleId = $PostBakeBundleId
} else {
    Write-Host "[phase 2] SKIPPED" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Phase 3 -- 4-gate persona verification harness
# Mirrors scripts/figure_verify_einstein_persona.sh
# ---------------------------------------------------------------------------

if (-not $SkipVerify) {
    if ([string]::IsNullOrEmpty($BundleId)) {
        $BundleId = Resolve-BundleId -BundleRoot $BundleRoot
    }

    Write-Host ""
    Write-Host "[phase3] verification bundle=$BundleId runtime=$RuntimeBackend" -ForegroundColor Cyan

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
        '--bundle-id', $BundleId,
        '--figure', 'einstein',
        '--bundle-root', $BundleRoot,
        '--output-dir', $VerifyOut,
        '--runtime', $RuntimeBackend,
        '--qwen-model-id', $QwenModelId,
        '--max-in-corpus-questions', $MaxInCorpusQuestions,
        '--conditions', 'raw,bundle,bundle_lora',
        '--questions-cache', (Join-Path $VerifyOut 'questions.jsonl')
    ) -AllowedExitCodes @(0, 2)

    $verdictFile = Join-Path $VerifyOut 'verdict.json'
    if (Test-Path $verdictFile) {
        $verdictColor = if ($verifyCode -eq 0) { 'Green' } else { 'Yellow' }
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor $verdictColor
        Write-Host " verdict.json:" -ForegroundColor $verdictColor
        Get-Content -LiteralPath $verdictFile -Raw
        Write-Host ""
        Write-Host " transcript: $(Join-Path $VerifyOut 'transcript.md')" -ForegroundColor $verdictColor
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
