# Learned backend SHADOW evidence — macOS / MPS

> Run completed: 2026-07-15 (UTC+8)  
> Evidence status: **partial run; ACTIVE promotion blocked**  
> Source commit: `6c36c5951c3f6330e4ee0aa3626955bf36f3616a` (`main`, working tree was dirty)

## Verdict

This run is useful positive SHADOW evidence, but it does **not** support promoting any learned backend to `ACTIVE`.

The strongest result is that the four torch candidates ran for 500 turns on the real Hugging Face substrate (`Qwen/Qwen2.5-1.5B-Instruct`, MPS), preserved forward parity, met latency and safety checks, passed rollback checks, and showed real parameter updates without writing SHADOW candidates back into owner state.

Promotion remains blocked for three substantive reasons:

1. The artifact contains 499 real-trace transitions, below the `>=500` gate, despite `turn_count=500`.
2. `validation_delta=0.0138`, below the required `0.02`.
3. The same-substrate PE-off and ETA-off component controls were not run, so both control-direction gates are false.

The orchestration stopped after the capacity manifest was generated. There is no same-substrate ablation verdict, promotion evidence bundle, promotion report, or final run summary.

## Completed stages

- SHADOW smoke: completed.
- Platform chunked soak: 25/25 chunks succeeded; 500 turns and 475 real-trace transitions. This is stability evidence only, not promotion evidence.
- Continuous real-substrate soak: completed; 500 turns and 499 real-trace transitions.
- Capacity ladder: manifest generated for 27 arms, but the arms were not executed.
- Same-substrate ablation: not completed.
- Promotion evidence build/evaluation: not completed.

## Real-soak configuration and throughput

- Substrate: `Qwen/Qwen2.5-1.5B-Instruct`
- Device: Apple MPS
- Temporal latent dimension: 16
- SHADOW candidates: temporal runtime, temporal SSL, internal RL, CMS torch
- Runtime: 2,426.57 seconds (40.44 minutes)
- Mean turn latency: 4.8531 seconds
- Latency SLO: 5.0 seconds; passed narrowly

## Positive evidence

### Forward parity and safety

- Temporal runtime WORLD/SELF parity passed at numerical-error scale (`~1e-16` maximum absolute differences).
- Temporal SSL post-training forward parity passed (`~1.2e-10`, tolerance `1e-7`).
- CMS torch parity passed (`5.55e-17`, tolerance `1e-9`).
- Runtime and SSL switch decisions and action-family selections matched their comparison paths.
- All four candidates passed strict ETA, rollback, latency, and safety gates.
- SHADOW candidates did not write back into owner state.

### Learned behavior

- Temporal SSL changed 3,085 parameters over 11 trained steps.
- Internal RL changed 49 parameters over 13 transitions.
- CMS torch changed all 1,024 tracked parameters; old-knowledge retention was `1.0`.
- Internal RL no-optimize control behaved correctly:
  - full return improved by `+1.5503`;
  - no-optimize control improvement was `0.0`.
- Strict ETA gate passed:
  - switch sparsity increased from `0.3808` at alpha `0.0` to `0.5335` at alpha `1.0`;
  - held-out reuse remained `1.0`.

### Predictive heads

The predictive heads improved steadily over the run:

- WORLD learned MAE: `0.0981` at turn 25 → `0.0337` at turn 500.
- WORLD baseline MAE at turn 500: `0.0411`; final learned improvement: `+0.0073`.
- SELF learned MAE: `0.0256` at turn 25 → `0.0251` at turn 500.
- SELF baseline MAE at turn 500: `0.0389`; final learned improvement: `+0.0138`.
- Six checkpoint round trips were verified.
- The report-only CP-11 kill criterion did not trigger.

## Cautions

- The active gate uses `validation_delta=0.0138`; this is positive but only 69% of the required `0.02`.
- Internal RL's approximate KL is high (`32.18`). The current gate did not reject it, but this deserves investigation before ACTIVE rollout.
- CMS new-knowledge absorption is low (`0.0093`) even though retention is perfect; this may indicate an overly conservative update.
- Social cognition was not exercised: all ToM record counts, common-ground atom counts, and pending settlement counts are zero. This run provides no evidence for social-learning behavior.
- Seven of nine semantic-owner forecast errors are exactly zero across 499 updates. That can mean stable targets, but it can also mean the synthetic trace lacks excitation; these numbers should not be treated as broad forecast validation.
- The run artifact records a dirty working tree, so exact reproducibility requires the uncommitted state or a clean rerun.
- The chunked lane and continuous lane are not additive for the `>=500 real_trace_turns` gate.

## Per-component promotion result

- `temporal_runtime_backend`: blocked by real traces, validation delta, PE-off control, and ETA-off control.
- `temporal_ssl_backend`: additionally blocked until temporal runtime is ACTIVE first.
- `internal_rl_backend`: additionally blocked until temporal runtime and SSL are ACTIVE first.
- `cms_torch_backend`: blocked by real traces, validation delta, PE-off control, and ETA-off control.

## Recommended next run

1. Fix or explicitly account for the 500-turn / 499-real-transition boundary, then rerun enough continuous turns to exceed the gate.
2. Run the same-substrate PE-off and ETA-off ablation arms.
3. Execute, rather than only generate, the capacity ladder with matched seeds and substrate.
4. Build and evaluate the promotion evidence only after those artifacts exist.
5. Keep all four backends in `SHADOW`; do not flip defaults from this result.

## Source artifacts

- `artifacts/learned_active_evidence/real_soak/learned_shadow_soak.json`  
  SHA-256: `f7d4fc107535f10627f195e3b74ff22d61268b46748daa1756382b2de918e625`
- `artifacts/learned_active_evidence/platform_chunked_soak/learned_shadow_soak_chunked.json`  
  SHA-256: `a4b299586d96ce7ae3d3523e5ff8920cf093474ad2aabd568d69e387d9abe3bf`
- `artifacts/learned_active_evidence/capacity_ladder/capacity_ladder_manifest.json`  
  SHA-256: `3f46d511191115773d00eaf153338e1ae8a4860a7e7d55a2a1e8a825ecfe9060`
