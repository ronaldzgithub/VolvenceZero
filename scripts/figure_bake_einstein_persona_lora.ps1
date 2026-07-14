# Wave N — bake a real PEFT persona LoRA on the Wave K curated Einstein bundle.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $RepoRoot

$BundleRoot = if ($env:BUNDLE_ROOT) { $env:BUNDLE_ROOT } else { "data/figure_bundles" }
$AuditRoot = if ($env:AUDIT_ROOT) { $env:AUDIT_ROOT } else { "data/figure_audit" }
$CleaningRoot = if ($env:CLEANING_ROOT) { $env:CLEANING_ROOT } else { "data/figure_corpus" }
$MetadataFile = if ($env:METADATA_FILE) {
    $env:METADATA_FILE
} else {
    "packages/lifeform-domain-figure/data/seeds/einstein-2026Q2.curated_metadata.jsonl"
}
$QwenModelId = if ($env:QWEN_MODEL_ID) { $env:QWEN_MODEL_ID } else { "Qwen/Qwen2.5-1.5B-Instruct" }
$PeftTargetModules = if ($env:PEFT_TARGET_MODULES) { $env:PEFT_TARGET_MODULES } else { "q_proj,k_proj,v_proj,o_proj" }
$PeftRank = if ($env:PEFT_RANK) { $env:PEFT_RANK } else { "8" }
$PeftMaxSteps = if ($env:PEFT_MAX_STEPS) { $env:PEFT_MAX_STEPS } else { "200" }
$PeftDevice = if ($env:PEFT_DEVICE) { $env:PEFT_DEVICE } else { "cuda" }

if (-not (Test-Path $MetadataFile)) {
    throw "curator metadata file not found: $MetadataFile"
}
if (-not (Test-Path (Join-Path $CleaningRoot "raw"))) {
    throw "L1 cleaning store missing: $(Join-Path $CleaningRoot 'raw')"
}

if ($env:BUNDLE_ID) {
    $BundleId = $env:BUNDLE_ID
} else {
    $BundleId = & C:\ProgramData\miniconda3\python.exe -c @"
import json
from pathlib import Path
root = Path(r"$BundleRoot") / "einstein"
if not root.exists():
    raise SystemExit(f"ERROR: no bundle root at {root}")
rows = []
for path in sorted(root.iterdir()):
    manifest = path / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows.append((payload.get("created_at_iso", ""), payload.get("bundle_id", path.name)))
if not rows:
    raise SystemExit(f"ERROR: no manifests found under {root}")
rows.sort(reverse=True)
print(rows[0][1])
"@
}

$RollbackEvidence = if ($env:ROLLBACK_EVIDENCE) {
    $env:ROLLBACK_EVIDENCE
} else {
    "prev_persona_lora=absent;base=$BundleId"
}

Write-Host "============================================================"
Write-Host " Wave N persona LoRA bake (curated mode)"
Write-Host " bundle_id      = $BundleId"
Write-Host " qwen_model_id  = $QwenModelId"
Write-Host " target_modules = $PeftTargetModules"
Write-Host " rank           = $PeftRank"
Write-Host " max_steps      = $PeftMaxSteps"
Write-Host " device         = $PeftDevice"
Write-Host "============================================================"

New-Item -ItemType Directory -Force -Path $BundleRoot, $AuditRoot | Out-Null

C:\ProgramData\miniconda3\python.exe -m lifeform_domain_figure.cli `
    --bundle-root $BundleRoot `
    --audit-root $AuditRoot `
    bake-lora `
    --figure einstein `
    --bundle $BundleId `
    --corpus-mode curated `
    --cleaning-root $CleaningRoot `
    --curated-metadata-file $MetadataFile `
    --backend peft `
    --rank $PeftRank `
    --peft-model-id $QwenModelId `
    --peft-target-modules $PeftTargetModules `
    --peft-max-steps $PeftMaxSteps `
    --peft-device $PeftDevice `
    --evaluation-snapshot default-clean `
    --rollback-evidence $RollbackEvidence

