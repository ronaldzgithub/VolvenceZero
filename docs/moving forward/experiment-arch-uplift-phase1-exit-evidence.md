# Arch Uplift Plan — Phase 1 Exit Evidence (2026-05-13)

> Companion brief to [`experiment-arch-uplift.md`](./experiment-arch-uplift.md) §10 退出条件.
>
> Records the evidence collected after executing T1-T14 of the arch-uplift
> plan. Phase 1 covers **声明侧 + 骨架 + 接口扩展 + contract test**
> deliverables; the heavier follow-up packets (backbone.py refactoring,
> real ablation aggregation, OA-4 audit-agent tool loop, B2 阶段 2 abstract
> promotion) remain in their respective business packets.

## 1. Plan completion summary

| Todo | Scope | Status | Contract test |
|---|---|---|---|
| T1 | docs/specs/profile-registry.md (A1+A3 spec) | DONE | — |
| T2 | docs/specs/evaluation-cascade.md (A2 spec) | DONE | — |
| T3 | docs/specs/audit-owner.md (A5 spec) | DONE | — |
| T4 | A4 DATA_CONTRACT §6 sync + contract test | DONE | `test_data_contract_wiring_sync.py` — 6 PASS |
| T5 | A1 profile_registry.py + 12 profile registration | DONE | `test_profile_registry_sync.py` — 17 PASS |
| T6 | A3 CapabilityWiring + RuntimeModule + FinalRolloutConfig | DONE | `test_capability_wiring.py` — 13 PASS |
| T7 | A2 step 1 cheap_layer.py 收编 facade + EvaluationSnapshot field-identical guard | DONE | `test_evaluation_cascade.py::test_evaluation_snapshot_*` etc. |
| T8 | A2 step 2 mid_layer.py 骨架 + MidLayerSnapshot schema | DONE | `test_evaluation_cascade.py::test_mid_layer_*` |
| T9 | A2 step 3 expensive_layer + cross_generation_aggregator 骨架 + ModificationGateEvidence schema | DONE | `test_evaluation_cascade.py::test_expensive_*` / `::test_modification_gate_evidence_*` |
| T10 | B3 BenchmarkMetricDescriptor + RuntimeModule.declare_benchmark_metrics | DONE | `test_benchmark_metric_descriptor.py` — 6 PASS |
| T11 | A5 audit/ owner package + evaluate_gate_reasons 接口扩展 | DONE | `test_audit_owner.py` — 16 PASS; `test_credit_gate.py` 22/22 PASS (byte-equivalent) |
| T12 | B4 scripts/run_shadow_evidence_template.py | DONE | `test_shadow_evidence_harness.py` — 5 PASS |
| T13 | B2 阶段 1 substrate feature hook completeness (informational) | DONE | `test_substrate_feature_hook_completeness.py` — 3 PASS + 1 SKIP |
| T14 | B1 multi_party_scenarios.py + 3 fixture builders | DONE | `test_multi_party_scenarios.py` — 10 PASS |

Total new contract tests added by Phase 1: **96 PASS + 1 SKIP** (B2 阶段 1
informational SKIP is the expected behaviour until 阶段 2 promotion).

## 2. 退出条件逐项核对

Reference: [`experiment-arch-uplift.md`](./experiment-arch-uplift.md) §10.

### 条件 1: A1-A5 全部 ACTIVE

**Phase 1 status: PARTIAL — schema + skeleton layer ACTIVE; substantive
implementation deferred per Phase 1 scope.**

| Sub-piece | Phase 1 status |
|---|---|
| A1 ProfileRegistry singleton populated + validate() passes | DONE |
| A1 build_standard_dialogue_runner registry-first dispatch | DEFERRED (spec §迁移协议 阶段 2 / part of T16 follow-up) |
| A3 CapabilityWiring + RuntimeModule.capability_wiring() default behaviour preserved | DONE |
| A3 FinalRolloutConfig.capability_wirings nested map | DONE |
| A2 cheap_layer.py facade + EvaluationSnapshot field-identical contract test | DONE |
| A2 mid_layer / expensive_layer / aggregator skeletons (DISABLED default) | DONE |
| A2 backbone.py compute_* migration to cheap_layer.py | DEFERRED (size 2K+ LOC; tracked as T7 阶段 2 follow-up) |
| A4 DATA_CONTRACT §6 SSOT sync + contract test in CI | DONE |
| A5 evaluate_gate_reasons signature extension (audit_snapshot, audit_required) | DONE |
| A5 AuditModule skeleton + DATA_CONTRACT §6 audit slot registration | DONE |
| A5 N8 audit-agent tool loop | DEFERRED to OA-4 business packet (out of A5 scope) |

Phase 1 ACTIVE surface = schema + interface + skeleton. Business packets
(OA-4, COG-1, COG-3) consume the surface in follow-up rounds.

### 条件 2: B2/B3/B4 至少完成

**Phase 1 status: DONE.**

| Sub-piece | Status |
|---|---|
| B2 阶段 1 substrate hook completeness report (informational) | DONE |
| B2 阶段 2 abstract method promotion | DEFERRED (triggered by COG-3 packet) |
| B3 RuntimeModule.declare_benchmark_metrics + BenchmarkMetricDescriptor | DONE |
| B3 benchmark metric_means schema-driven extraction | DEFERRED (benchmark-side union refactor; tracked as T10 阶段 2 follow-up) |
| B4 scripts/run_shadow_evidence_template.py | DONE |

### 条件 3: contract test 在 CI 持续 PASS ≥1 release cycle

**Phase 1 status: BASELINE established.**

96 new contract tests added by Phase 1 all PASS. The release-cycle stability
requirement is a calendar-based exit criterion that must be observed after
this phase merges; it cannot be settled inside the current PR window.

### 条件 4: 现有 11 profile + 4 memprobe + ETA strong-proof 全 PASS

**Phase 1 status: NO REGRESSION OBSERVED for unit-test layer; full
benchmark suite not run here.**

- `tests/test_credit_gate.py` (22 tests, exercises `evaluate_gate_reasons`
  legacy path) — 22/22 PASS after A5 signature extension. This is the
  strongest local evidence that A5 default `audit_snapshot=None,
  audit_required=False` keeps the 4 existing callers byte-equivalent.
- A4 `test_data_contract_wiring_sync.py` validates §6.X social cognition
  slot states against `FinalRolloutConfig` defaults — PASS, confirming
  the 7 ToM / social slot states are now consistent.
- Full `tests/contracts/` suite: 2104 PASS, 1 SKIP, 3 FAILED. The 3
  failures are **pre-existing**:
  - `test_kernel_only_imports_declared_tier[vz-cognition-store.py]`
  - `test_kernel_only_imports_declared_tier[vz-runtime-owner_hydration_store.py]`
  - `test_verification_module_boundaries.py::test_no_verification_module_imports_kernel`

  Verified pre-existing by running the same suite on the stashed HEAD
  (no arch-uplift changes): 3 failed, 1281 passed. The arch-uplift plan
  did NOT introduce these failures and explicitly does not block on them.
- `scripts/run_dialogue_paper_suite.sh` / `run_eta_paper_suite.sh` /
  `tests/longitudinal/test_vz_memprobe_*.py` are heavy benchmark runs
  (each ≥10 minutes); they should be executed as part of the next
  release-gate run, not inside this Phase 1 unit-test pass.

### 条件 5: 3 份新 sub-spec 评审通过

**Phase 1 status: drafted, awaiting review.**

- [`docs/specs/profile-registry.md`](../specs/profile-registry.md) (A1 + A3, 348 lines)
- [`docs/specs/evaluation-cascade.md`](../specs/evaluation-cascade.md) (A2 + B3, 290 lines)
- [`docs/specs/audit-owner.md`](../specs/audit-owner.md) (A5, 260 lines)

All three reference each other and the relevant existing specs
([`contract-runtime.md`](../specs/contract-runtime.md), [`evaluation.md`](../specs/evaluation.md),
[`credit-and-self-modification.md`](../specs/credit-and-self-modification.md),
[`DATA_CONTRACT.md`](../DATA_CONTRACT.md)). Spec evaluation status remains
'draft' pending downstream review.

## 3. Files touched / created (Phase 1)

**New files (15):**

```
docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md  ← this file
docs/specs/profile-registry.md
docs/specs/evaluation-cascade.md
docs/specs/audit-owner.md
packages/vz-runtime/src/volvence_zero/agent/profile_registry.py
packages/vz-runtime/src/volvence_zero/agent/dialogue/multi_party_scenarios.py
packages/vz-cognition/src/volvence_zero/audit/__init__.py
packages/vz-cognition/src/volvence_zero/audit/module.py
packages/vz-cognition/src/volvence_zero/audit/types.py
packages/vz-cognition/src/volvence_zero/evaluation/cheap_layer.py
packages/vz-cognition/src/volvence_zero/evaluation/mid_layer.py
packages/vz-cognition/src/volvence_zero/evaluation/expensive_layer.py
packages/vz-cognition/src/volvence_zero/evaluation/cross_generation_aggregator.py
scripts/run_shadow_evidence_template.py
tests/contracts/test_data_contract_wiring_sync.py
tests/contracts/test_profile_registry_sync.py
tests/contracts/test_capability_wiring.py
tests/contracts/test_evaluation_cascade.py
tests/contracts/test_benchmark_metric_descriptor.py
tests/contracts/test_audit_owner.py
tests/contracts/test_shadow_evidence_harness.py
tests/contracts/test_substrate_feature_hook_completeness.py
tests/contracts/test_multi_party_scenarios.py
```

**Modified files (3):**

```
docs/DATA_CONTRACT.md            ← §6 audit slot row + §6.X wiring states
packages/vz-contracts/src/volvence_zero/runtime/kernel.py  ← BenchmarkMetricDescriptor, CapabilityWiring, RuntimeModule.capabilities + capability_overrides + declare_benchmark_metrics
packages/vz-runtime/src/volvence_zero/integration/final_wiring.py  ← capability_wirings nested map + audit field
packages/vz-cognition/src/volvence_zero/credit/gate.py  ← evaluate_gate_reasons signature extension + _evaluation_and_structural_gate_reasons helper extraction
```

## 4. Phase 1 follow-up tracking

The following items are explicitly out of Phase 1 scope and deferred to
business packets / Phase 2 of arch-uplift:

- **A1 阶段 2** (`build_standard_dialogue_runner` registry-first dispatch) — pending paper-suite-small SHADOW evidence comparison
- **A2 step 1 阶段 2** (move backbone.py compute_* helpers into cheap_layer.py) — large refactor; separate PR
- **A2 mid_layer / expensive_layer actual aggregation logic** — pending COG-1 + DM-7 packet
- **A5 audit-agent tool loop / N8 8-attack acceptance** — OA-4 business packet
- **B2 阶段 2** (`feature_surface` / `residual_activations` as abstract methods) — triggered by COG-3 packet
- **B3 阶段 2** (benchmark `metric_means` actually consumes `declare_benchmark_metrics` union) — benchmark-side refactor; separate PR
- **T16 legacy cleanup** — after ≥1 release cycle of SHADOW evidence stability

## 5. T16 legacy cleanup status (Phase 1 close-out)

T16 in the original plan calls for removing the 11 `if profile_label == "X"`
branches in `_legacy.py`, the legacy `EvaluationModule` path, and the
pre-A5 `evaluate_gate_reasons` signature. **None of this work is
appropriate in Phase 1**, because Phase 1 did not actually replace any
of those legacy structures — it only added schema / skeleton / contract
test layers alongside them:

- `build_standard_dialogue_runner` still uses the 11 if-elif branches as
  the正式 upstream. The new `ProfileRegistry` is registered + validated
  but **not** wired into dispatch. Removing the if-elif branches now
  would orphan every dialogue benchmark caller. Cleanup is gated on the
  阶段 2 registry-first dispatch landing (deferred follow-up).
- `EvaluationModule` is still the sole `evaluation` slot publisher; the
  new `EvaluationCheapLayer` facade is a marker, not a replacement. The
  4 cascade-tier slots (`evaluation_mid` / `evaluation_expensive` /
  `evaluation_cross_generation`) are all `WiringLevel.DISABLED` by
  default. No legacy compute path is yet retired.
- The pre-A5 `evaluate_gate_reasons` signature is preserved by the new
  keyword-only defaults (`audit_snapshot=None, audit_required=False`);
  the 22 existing `test_credit_gate.py` cases pass byte-equivalent. The
  helper extraction (`_evaluation_and_structural_gate_reasons`) is a
  private refactor, not a removable legacy surface.

**T16 status**: COMPLETED-NO-OP. Cleanup deferred to Phase 2 with the
following triggers documented in §4 above:

- A1 阶段 2 (registry-first dispatch) → triggers removal of 11 if-elif branches
- A2 backbone migration to cheap_layer.py → triggers removal of the
  shim wrapper layer
- OA-4 audit-required promotion → triggers removal of pre-A5 default
  arguments path

The exit evidence above shows Phase 1 introduced **zero new legacy
surface** to clean up, so no T16 work is required in this packet.

## 6. Exit decision

Phase 1 establishes the **schema + interface + skeleton + contract test
substrate** for the full 9-item arch-uplift program. All deferred work
is documented above with clear triggers / owners. The Phase 1 surface is
non-regressive (96 new tests PASS, existing 22-test credit_gate suite
passes, 3 pre-existing failures verified unrelated) and ready to support
the business packets that build on it (COG-1 / COG-2 / COG-3 / OA-4).
