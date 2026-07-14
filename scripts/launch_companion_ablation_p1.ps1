# Run 2 launcher - same-substrate Companion Bench ablation, phase P1 (debt #87).
#
# One-command usage from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_companion_ablation_p1.ps1
#
# Optional:
#   -ArtifactDir artifacts/companion-ablation/<tag>   reuse/resume a specific run dir
#   -Resume                                           first attempt already passes --resume
#   -MaxAttempts 3                                    auto-resume retries after judge/arc
#                                                     timeouts (default 3 total attempts)
#   -DryRun                                           print commands only, no GPU/paid calls
#
# What it does (delegates orchestration to run_p1_windows.ps1, the SSOT):
#   1. refuses to start if an ablation stack is already listening on
#      8000/8001/8500/8600 (an in-flight run would be corrupted),
#   2. runs preflight -> serve 5 tracks -> score -> verdict, teeing output to a
#      timestamped transcript in the artifact dir,
#   3. on failure (the 2026-07-13 run died on arc-judge TimeoutError) retries
#      with --resume so finished per-track summaries are not re-paid,
#   4. prints the verdict_p1.json state + per-claim status at the end.
#
# Requirements: .local/llm.env with OPENROUTER_API_KEY + ABLATION_*_MODEL set
# (already present on this machine), CUDA GPU, HF cache with the Qwen substrate.

param(
    [string]$ArtifactDir = "",
    [switch]$Resume,
    [switch]$DryRun,
    [int]$MaxAttempts = 3
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

# --- Guard: never start on top of a live ablation stack -----------------
$AblationPorts = @(8000, 8001, 8500, 8600)
$Busy = @()
foreach ($Port in $AblationPorts) {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($conn) { $Busy += "$Port(pid=$(($conn | Select-Object -First 1).OwningProcess))" }
}
if ($Busy.Count -gt 0 -and -not $DryRun) {
    Write-Host "[launch] ablation ports already in use: $($Busy -join ', ')" -ForegroundColor Yellow
    Write-Host "[launch] an ablation stack (possibly an in-flight run) is up. Either:"
    Write-Host "  a) let the in-flight run finish, or"
    Write-Host "  b) stop it first:  Get-Content <artifact-dir>\serve.pids | ForEach-Object { Stop-Process -Id `$_ -Force }"
    Write-Host "     (plus kill any run_same_substrate_ablation.py / score_reference_systems.py python processes)"
    throw "[launch] refusing to start: ports busy"
}

if (-not $ArtifactDir) {
    $dateTag = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $ArtifactDir = "artifacts/companion-ablation/$dateTag"
}
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null
$TranscriptPath = Join-Path $ArtifactDir ("launch_p1_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Write-Host "============================================================"
Write-Host " same-substrate companion ablation - P1"
Write-Host " artifact_dir = $ArtifactDir"
Write-Host " transcript   = $TranscriptPath"
Write-Host " max_attempts = $MaxAttempts"
Write-Host "============================================================"

Start-Transcript -Path $TranscriptPath -Append | Out-Null
$FinalRc = 1
try {
    if ($DryRun) {
        & "$PSScriptRoot/companion_bench/run_p1_windows.ps1" -ArtifactDir $ArtifactDir -DryRun
        $FinalRc = $LASTEXITCODE
    } else {
        for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
            $UseResume = $Resume -or ($Attempt -gt 1)
            Write-Host ""
            Write-Host "[launch] attempt $Attempt/$MaxAttempts (resume=$UseResume)..."
            $Failed = $false
            try {
                if ($UseResume) {
                    & "$PSScriptRoot/companion_bench/run_p1_windows.ps1" -ArtifactDir $ArtifactDir -Resume
                } else {
                    & "$PSScriptRoot/companion_bench/run_p1_windows.ps1" -ArtifactDir $ArtifactDir
                }
            } catch {
                $Failed = $true
                Write-Host "[launch] attempt $Attempt failed: $_" -ForegroundColor Yellow
            }
            if (-not $Failed) {
                $FinalRc = 0
                break
            }
            if ($Attempt -lt $MaxAttempts) {
                Write-Host "[launch] retrying with --resume in 30s (completed track summaries are reused, not re-paid)..."
                Start-Sleep -Seconds 30
            }
        }
    }
} finally {
    Stop-Transcript | Out-Null
}

if ($DryRun) { exit $FinalRc }
if ($FinalRc -ne 0) {
    Write-Host "[launch] P1 did not complete after $MaxAttempts attempts. See $TranscriptPath" -ForegroundColor Red
    Write-Host "[launch] partial per-track results are kept in $ArtifactDir\scores; rerun this script with:"
    Write-Host "         -ArtifactDir `"$ArtifactDir`" -Resume"
    exit $FinalRc
}

# --- Verdict summary -----------------------------------------------------
$VerdictPath = Join-Path $ArtifactDir "verdict_p1.json"
if (-not (Test-Path $VerdictPath)) {
    Write-Host "[launch] WARNING: run finished but $VerdictPath is missing" -ForegroundColor Yellow
    exit 1
}
$Verdict = Get-Content $VerdictPath -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "============================================================"
Write-Host " VERDICT (P1, single seed => stability not claimable)"
Write-Host " state = $($Verdict.state)"
Write-Host "------------------------------------------------------------"
foreach ($Claim in $Verdict.claims) {
    Write-Host " $($Claim.claim_id): $($Claim.status)"
    Write-Host "   $($Claim.detail)"
}
Write-Host "------------------------------------------------------------"
Write-Host " tracks (final_mean):"
$Verdict.tracks.PSObject.Properties | ForEach-Object {
    Write-Host ("   {0,-16} {1}" -f $_.Name, $_.Value)
}
if ($Verdict.recommendations) {
    Write-Host " recommendations:"
    foreach ($Rec in $Verdict.recommendations) { Write-Host "   - $Rec" }
}
Write-Host "============================================================"
Write-Host "[launch] verdict    : $VerdictPath"
Write-Host "[launch] transcript : $TranscriptPath"
exit 0
