# Run 1 launcher - 500-turn learned-shadow soak (synthetic lane).
# Windows mirror of scripts/run_learned_shadow_soak.sh.
#
# One-command usage from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_learned_shadow_soak.ps1
#
# Optional:
#   -Turns 500                                  turn count (default 500)
#   -OutputDir artifacts/learned_shadow_soak    artifact directory
#
# What it does:
#   * runs python -u scripts/run_learned_shadow_soak.py, teeing all output to a
#     timestamped log inside the output directory,
#   * prints the learned_active_gate verdict summary from the artifact.
#
# Notes (same as the .sh launcher):
#   * SYNTHETIC lane: verdicts are expected BLOCKED on the real-trace gates
#     (real_trace_turns=0 by definition). Directional evidence only; ACTIVE
#     promotion needs the real-trace lane.
#   * CPU-bound. The artifact records latency_slo_ok=false if the mean turn
#     time exceeds 5s - the run still completes, just plan for a long wall
#     clock (500 turns x mean-turn-seconds).

param(
    [int]$Turns = 500,
    [string]$OutputDir = "artifacts/learned_shadow_soak"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$LogPath = Join-Path $OutputDir ("soak_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Write-Host "============================================================"
Write-Host " learned-shadow soak (synthetic lane)"
Write-Host " turns      = $Turns"
Write-Host " output_dir = $OutputDir"
Write-Host " log        = $LogPath"
Write-Host "============================================================"

# cmd /c merges stderr into stdout as plain text, so Tee-Object never converts
# stderr lines into ErrorRecords (which would abort under ErrorActionPreference=Stop).
$SoakCmd = "python -u scripts/run_learned_shadow_soak.py --turns $Turns --output-dir `"$OutputDir`" 2>&1"
cmd /c $SoakCmd | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) {
    throw "[soak] run failed (exit $LASTEXITCODE); see $LogPath"
}

$ArtifactPath = Join-Path $OutputDir "learned_shadow_soak.json"
$Payload = Get-Content $ArtifactPath -Raw | ConvertFrom-Json
$Gate = $Payload.learned_active_gate

Write-Host ""
Write-Host "[soak] learned_active_gate verdict summary:"
Write-Host "  note: $($Gate.note)"
Write-Host "  latency_slo_ok: $($Gate.latency_slo_ok)"
foreach ($Verdict in $Gate.verdicts) {
    $Missing = if ($Verdict.missing_gates) { $Verdict.missing_gates -join ", " } else { "-" }
    Write-Host "  $($Verdict.component): eligible=$($Verdict.eligible) missing=[$Missing]"
}
Write-Host ""
Write-Host "[soak] artifact : $ArtifactPath"
Write-Host "[soak] manifest : $(Join-Path $OutputDir 'learned_shadow_soak_manifest.json')"
Write-Host "[soak] log      : $LogPath"
