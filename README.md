# VolvenceZero

## Learned Backend ACTIVE Evidence

The root launchers below are thin shell wrappers around the Python evidence
tools under `scripts/`. They do not flip runtime defaults; they only run and
assemble evidence for SHADOW -> ACTIVE promotion.

### One-Command Resume Runner

Use this for the full resumable evidence pipeline:

```bash
bash run_learned_active_evidence.sh --resume --substrate-mode hf --substrate-device mps
```

On Windows / CUDA hosts, use:

```bash
bash run_learned_active_evidence.sh --resume --substrate-mode hf --substrate-device cuda
```

If you are running from Windows PowerShell without Git Bash, use the `.ps1`
launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_learned_active_evidence.ps1 --resume --substrate-mode hf --substrate-device cuda
```

The runner records per-stage markers under `artifacts/learned_active_evidence/`
and skips completed stages on the next `--resume`.

### Individual Launchers

```bash
bash run_learned_shadow_smoke.sh
bash run_learned_shadow_soak.sh --turns 500 --substrate-mode hf --substrate-device mps
bash run_learned_capacity_ladder.sh --n-z 16,64,256 --turns 500
bash run_learned_promotion_evidence.sh --soak-artifact artifacts/.../learned_shadow_soak.json --ablation-verdict artifacts/.../verdict_p1.json
bash run_affordance_learner_probe.sh
bash run_longitudinal_continuity.sh
```

PowerShell equivalents:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_learned_shadow_smoke.ps1
powershell -ExecutionPolicy Bypass -File .\run_learned_shadow_soak.ps1 --turns 500 --substrate-mode hf --substrate-device cuda
powershell -ExecutionPolicy Bypass -File .\run_learned_capacity_ladder.ps1 --n-z 16,64,256 --turns 500
powershell -ExecutionPolicy Bypass -File .\run_learned_promotion_evidence.ps1 --soak-artifact artifacts\...\learned_shadow_soak.json --ablation-verdict artifacts\...\verdict_p1.json
powershell -ExecutionPolicy Bypass -File .\run_affordance_learner_probe.ps1
powershell -ExecutionPolicy Bypass -File .\run_longitudinal_continuity.ps1
```

### Evidence Boundary

`run_learned_active_evidence.sh` and `run_learned_active_evidence.ps1` invoke
the same Python orchestrator. ACTIVE promotion still requires real evidence: a
continuous `hf` substrate soak, enough real trace turns, capacity/gain evidence,
component ablation verdicts, CMS anti-forgetting A/B, and a passing promotion
report. Chunked/platform soak artifacts are stability evidence only and are not
treated as promotion evidence.
