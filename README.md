# VolvenceZero

## State-KV Carrier Identification

The State-KV runner tests whether two users can receive distinguishable
responses through model-layer personal state while the pure arms send
byte-identical prompts and no conversation history:

```bash
# Zero-cost wiring smoke.
python scripts/run_state_kv_identification.py --lane smoke

# Real frozen Qwen2.5-0.5B; local weights only by default.
python scripts/run_state_kv_identification.py \
  --lane p1 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu

# Bake and explicitly test the reversible contrastive projector artifact.
python scripts/bake_state_kv_projector.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu
python scripts/run_state_kv_identification.py \
  --lane p1 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --projector-artifact \
  artifacts/state_kv/projectors/qwen2.5-0.5b-contrastive.json
```

The P1 lane reuses one frozen Transformers runtime across all four arms,
forces deterministic decoding, and gives both personas the same response
assembly so the sampling-layer carrier is closed. It writes the verdict,
full response transcript, and content-addressed substrate fingerprint under
`artifacts/state_kv/p1/`. Without a cross-family blind judge the identification
and causality claims remain `insufficient_data`; a real residual hook alone is
not promoted into a retained result. The learned-projector lane writes to
`artifacts/state_kv/p1-learned/`; omitting `--projector-artifact` is its rollback.
On the recorded Qwen 0.5B matched run, both fixed and contrastive projectors
failed output divergence on probes p0/p2, so the artifact remains evidence-only.

The P3 lane adds a fifth arm, `state-kv-arm-g-prefix-pure`, which carries the
same readout as a bounded per-layer key/value prefix instead of a single-layer
residual, and makes it the candidate arm. Train the generator with
`scripts/train_state_kv_prefix.py` (teacher is the text arm; the base model
stays frozen and only 123k generator parameters move), then pass
`--lane p3 --prefix-kv-artifact ...`; omitting the artifact is its rollback.
Running the same artifact on CPU and MPS is what separates carrier effects from
numerical noise: the residual arm's single divergence disappears when the
device changes, while the prefix arm diverges on probes p0/p2 under both. That
is a bandwidth result, not an identification one — the wrong-user negative
control sits at chance (0.508), probe p1 is still byte-identical across users,
so claim 2 fails and the blind judge stays unwired. Evidence is under
`artifacts/state_kv/p3/` and `artifacts/state_kv/p3-mps/`.

## Semantic Grounding Evidence

One command runs the two experiments of
`docs/specs/semantic-grounding-evidence.md` — the latent–semantic
grounding readout (D1 discrimination / D2 lead / D3 transfer with
shuffled controls) and the LLM-proposal dependency ablation (matched
on/off arms). These feed `claim_latent_abstraction_semantically_grounded`
and `claim_semantic_tracking_not_llm_dependent` in
`docs/specs/evidence_program.md`.

```bash
# Anytime (CI-safe, ~1 min): harness unit tests + synthetic smoke lane.
bash run_semantic_grounding_evidence.sh

# Milestone evidence run (real Qwen substrate; the citable one):
bash run_semantic_grounding_evidence.sh --lane hf --substrate-device mps
bash run_semantic_grounding_evidence.sh --lane hf --substrate-device cuda

# First hf run on a machine without cached weights:
bash run_semantic_grounding_evidence.sh --lane hf --substrate-allow-download

# Everything (unit + smoke + hf):
bash run_semantic_grounding_evidence.sh --lane all
```

Each run writes a fresh timestamped directory under
`artifacts/semantic_grounding_evidence/` containing per-stage logs, the
two report artifacts with sha256 manifests, and a `summary.json` with
per-stage status and extracted verdicts. Exit code is non-zero on any
stage failure.

Evidence tiers (enforced in the artifacts, not just by convention):

- `unit` — pytest acceptance for both harnesses. Validates the harness,
  produces no evidence.
- `smoke` — synthetic substrate. Wiring + differential-design check
  only; reports are stamped `evidence_tier: synthetic-smoke` and the
  grounding verdict is expected to be `insufficient-coverage`. Never
  citable for the claims.
- `hf` — shared real substrate for both ablation arms (identical
  residual path; only the proposal channel differs) plus a real-trace
  grounding capture. This is the lane that produces claim evidence. A
  grounding `fail` here is a kill signal for the "emergent abstraction
  is grounded" claim and must be reported as-is; an ablation
  `llm-proposal-dependent` verdict downgrades the external claim to
  "LLM-assisted typed semantic tracking".

If the hf grounding report says `insufficient-coverage`, raise
`--hf-turns-per-case` (coverage gate: >= 50 closed segments). The
channel-level runtime switch used by the off arm is also available for
manual A/B on any vertical: `VZ_SEMANTIC_PROPOSAL_CHANNEL=noop`.

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

### CompanionBench P1 on Windows

`run_learned_active_evidence.ps1` expects the CompanionBench P1 readiness
manifest when it reaches the same-substrate ablation stage. Generate or resume
that P1 run from PowerShell with:

```powershell
.\run_companion_bench_p1.ps1
```

The Windows P1 launcher defaults to SafeMode to keep RDP and the desktop
responsive on single-GPU machines. SafeMode uses the lightweight hashing
retrieval embedder, limits common math-library thread pools, starts service
processes at `BelowNormal` priority, and launches a watchdog for the current
run's `serve.pids`.

The watchdog writes logs under the run's `serve-logs/watchdog.log`. If available
RAM stays below `4GB` or GPU memory usage stays at or above `94%` for three
consecutive checks, it stops the P1 services for that run rather than letting
the machine become unreachable.

Use full-resource mode only when the host can tolerate it:

```powershell
.\run_companion_bench_p1.ps1 -FullMode
```

To stop a stuck or leftover P1 run, including its watchdog:

```powershell
.\run_companion_bench_p1.ps1 -Stop
```

### Evidence Boundary

`run_learned_active_evidence.sh` and `run_learned_active_evidence.ps1` invoke
the same Python orchestrator. ACTIVE promotion still requires real evidence: a
continuous `hf` substrate soak, enough real trace turns, capacity/gain evidence,
component ablation verdicts, CMS anti-forgetting A/B, and a passing promotion
report. Chunked/platform soak artifacts are stability evidence only and are not
treated as promotion evidence.
