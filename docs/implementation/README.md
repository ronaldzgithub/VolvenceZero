# Implementation Documents

> Status: current index
> Last updated: 2026-04-26
> Scope: implementation notes that still reflect the current runtime, validation surfaces, and evidence plans

## Source of Truth Order

Use this order when judging the current system state:

1. Current code in `volvence_zero/`
2. `docs/DATA_CONTRACT.md`
3. `docs/SYSTEM_DESIGN.md`
4. `docs/specs/00_INDEX.md` and the target `docs/specs/*.md`
5. Current implementation notes in this directory

Implementation notes are not the architectural source of truth. They explain concrete runtime lanes, validation protocols, and evidence programs after the current contracts have already been established.

## Current Runtime And Validation Notes

- `05_real_llm_substrate_runtime.md` - local frozen LLM substrate runtime modes, strict-local semantics, rare-heavy substrate adapter path.
- `06_real_substrate_validation_protocol.md` - validation protocol for local frozen substrate runs.
- `07_real_substrate_calibration_report.md` - hook-layer calibration report for local substrate candidates.
- `08_three_path_benchmark_report.md` - benchmark comparison across builtin fallback, distilgpt2, and Qwen 0.5B paths.
- `09_prediction_error_first_cognitive_loop.md` - implemented PE-first runtime loop and downstream consumers.
- `10_pe_eta_dialogue_benchmark_harness.md` - dialogue proof harness for PE, ETA, multi-timescale, and slow-loop evidence.
- `11_eta_internal_rl_strong_proof_harness.md` - repo-native ETA internal-RL proof harness and matched controls.
- `12_eta_paper_grade_uplift_plan.md` - current staged path from engineering proof to paper-grade evidence discipline.
- `13_emogpt_prd_alignment_upgrade.md` - 12-gap upgrade plan that maps EmoGPT v4.0 product PRD requirements onto VolvenceZero's NL+ETA owner contracts; identifies what to borrow, what to refuse, and the phased rollout that preserves SSOT.

## Current Paper-Grade Uplift Stages

The active uplift stage documents are:

- `uplift/U05_sparse_reward_environment.md`
- `uplift/U06_batch_ssl_emergent_abstractions.md`
- `uplift/U07_hard_causal_takeover.md`
- `uplift/U08_real_residual_internal_rl.md`
- `uplift/U09_nl_slow_loop_support.md`
- `uplift/U10_paper_suite_closure.md`

These are the only remaining stage plans under `docs/implementation/uplift/`. Earlier P00-P09 package plans, U01-U04 uplift plans, and design-v2 gap documents were removed because their status snapshots had been superseded by the current code, data contract, system design, and specs.

## Removed Historical Plans

The old implementation docs that claimed items such as `prediction_error` missing, CMS being only `dim=3`, or reflection/temporal paths not being wired no longer reflected the current runtime. They were deleted rather than kept as stale guidance.

If historical context is needed, use git history. Do not use removed implementation plans as design evidence.

## Claim Boundary

The current system has meaningful NL/ETA mechanisms and proof surfaces, but paper-grade claims require the explicit evidence gates described in:

- `docs/specs/evidence_program.md`
- `docs/implementation/12_eta_paper_grade_uplift_plan.md`
- `docs/implementation/uplift/U10_paper_suite_closure.md`

In particular, synthetic or trace harness success must not be treated as real open-weight residual-control evidence unless the relevant open-weight claim gates retain.
