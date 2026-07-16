# CMS ATLAS + Titans Update Rule Uplift

> Status: ACTIVE by default after SHADOW validation passed on 2026-05-06.
> Owner: `vz-memory.CMSMemoryCore` (memory owner internals only).
> Tier: 2 (substrate-internal learnable layer; base LLM untouched).

## 1. Why this spec exists

The current `CMSMemoryCore` already realizes the NL-essence parts of Continuum Memory:

- 3 frequency bands (online-fast / session-medium / background-slow), each a 2-layer residual MLP with momentum SGD, anti-forgetting backflow, cadence gating;
- a small `LearnedUpdateRule` that emits `LearnedUpdateDecision` per band update and learns from `improvement / stability` (Hope-style self-modifying scaffolding);
- nested variant that meta-learns the initial state for the next-faster band (NL Appendix A.5).

What it does **not** do, and what this uplift adds:

1. **ATLAS Omega rule (past-aware joint optimization)**: each test-time write currently only fits the *current* target. ATLAS argues this is the dominant failure mode of fixed-size memories on long context. Fix: every band write fits a small replay window (recent K observations) jointly, weighted exponentially.
2. **Titans surprise gating (PE-driven write gate)**: the `LearnedUpdateRule` decision features today only see current/target/delta/pending/source norms and hyperparameters. They do **not** see prediction error. So the write gate fires regardless of whether the user actually surprised the system. Fix: feed PE magnitudes (task / relationship / regime / action) into the decision features so the rule can learn surprise-driven writes.

These two changes together correspond to the parts of Titans (`2501.00663`) and ATLAS (`2505.23735`) that are *applicable to a frozen-substrate companion system*. The rest of those papers (deeper memory, replacing attention with recurrent memory) is Tier 3 and explicitly not in scope.

## 2. Owner boundary

All changes are inside `vz-memory`. No new owner-to-owner direct calls. Specifically:

- `CMSMemoryCore` (subordinate of `MemoryStore`) gains optional `prediction_error` parameters on `observe_substrate / observe_encoder_feedback / observe_fast_memory_signal` and an internal replay window. **No** new public API surface for non-memory owners.
- `MemoryStore` already reads `PredictionErrorSnapshot` via `apply_prediction_error_signal`. We extend `MemoryStore.observe_substrate` to accept the same snapshot and forward to CMS. The flow through `MemoryModule.process` continues to read `prediction_error` from `upstream` snapshots; no new owner→owner wires.

R8 invariants enforced:

- The new readouts about ATLAS replay / Titans gating are **published by CMS** in `CMSState` / `CMSCheckpointState`. Consumers do not reconstruct them.
- PE remains owned by `prediction` owner; CMS only consumes its snapshot.

## 3. Mechanism 1 — ATLAS Omega rule

### 3.1 Math

Let band `b` have current MLP state `x_b` and incoming target signal `t_t` at runtime step `t`. Define a replay buffer

```
R_b = [(t_{t-K+1}, w_{t-K+1}), ..., (t_t, w_t)]
```

with exponential weights `w_{t-i} = gamma^i / Z`, `gamma in (0,1)`, `Z = sum(gamma^i)`.

The replay update minimizes

```
L_b = sum_i w_i * || y(x_b) - t_i ||^2
```

with one weighted SGD step. K=1 collapses to the existing single-target update.

### 3.2 Defaults

- `online-fast`: K = 8, gamma = 0.6
- `session-medium`: K = 4, gamma = 0.7 (cadence-gated as today)
- `background-slow`: K = 2, gamma = 0.8 (cadence-gated as today)

Hard cap K = 32 to bound compute. Effective K is `min(configured_K, len(buffer))`.

### 3.3 Implementation surface

- `CMSBandMLP.update_with_replay(targets, weights, lr_scale, momentum_gate)` is added. K=1 must be **bit-equal** to the current `update(target=...)` path. Both methods share the same gradient kernel.
- `CMSMemoryCore` keeps three deque-backed buffers (`_online_replay`, `_session_replay`, `_background_replay`). Each `observe_*` call pushes the resolved target into the corresponding buffer before the band update.
- The legacy `update(target=...)` is preserved unchanged so the SHADOW-vs-ACTIVE comparison can switch on a flag without touching old code paths.

## 4. Mechanism 2 — Titans surprise gating

### 4.1 New decision features

`_decision_features` is extended from 12 dims to 16 dims by appending:

- `f12 = clamp(|task_error|)`
- `f13 = clamp(|relationship_error|)`
- `f14 = clamp(|regime_error|)`
- `f15 = clamp(|action_error|)`

When `prediction_error is None` (legacy path / bootstrap turn), the four new dims are zero, which keeps the rule numerically equivalent to the pre-uplift behavior **after** old checkpoints are restored as described in Section 5.

### 4.2 Implementation surface

- `LearnedUpdateRule.feature_dim` is bumped from configurable-12-or-d_in to a fixed `_BASE_FEATURE_DIM (12) + _PE_FEATURE_DIM (4) = 16`.
- `_decision_features` always returns a 16-tuple.
- `observe_substrate / observe_encoder_feedback / observe_fast_memory_signal` get an optional `prediction_error: PredictionErrorSnapshot | None = None` keyword. `_decision_features` is updated to read PE through a helper `_pe_features(prediction_error)`.

## 5. Backward compatibility

### 5.1 Old `LearnedUpdateRuleState` checkpoints

`LearnedUpdateRuleState` gains an optional `feature_version: int = 1` field (default value covers all checkpoints written before this uplift). New rule writes `feature_version = 2`.

`LearnedUpdateRule.restore_state` semantics:

| state.feature_version | state.feature_dim | restore behavior |
|---|---|---|
| 1 (legacy) | 12 | pad input_projection columns 12..15 with **zeros** (PE columns start neutral) |
| 1 (legacy) | 16 (legacy `max(12,d_in)` artifact) | **zero out** columns 12..15 (their old values came from modulo extension and are not aligned with the PE feature semantics) |
| 1 (legacy) | other | use modulo padding (existing behavior) for non-PE columns; columns 12..15 zero |
| 2 | 16 | direct copy |

This guarantees: **on a legacy checkpoint, calling `decide(features=12-tuple-or-16-with-PE-zero)` reproduces the pre-uplift decision exactly** (because PE columns are zero weight × zero feature = 0).

### 5.2 Old `CMSCheckpointState`

No structural changes. New `CMSCheckpointState` fields are additive frozen with default values. Old checkpoint deserializes by accepting defaults for the new fields.

### 5.3 Behavior parity contract

With `cms_replay_window_size = 1` and `cms_pe_features_enabled = False`, **the CMS must behave identically to the pre-uplift CMS**, modulo equivalent representation of the new state fields. This is asserted by unit test `test_cms_uplift_disabled_path_is_bit_equal_to_legacy`.

## 6. Public snapshot additions (frozen, additive only)

Added to `CMSBandState`:

- `replay_window_size: int = 0` — actual K for this band's last update (0 if disabled).
- `pe_feature_summary: tuple[float, ...] = ()` — last PE features fed into the rule for this band (4-tuple or empty).

Added to `CMSState`:

- `atlas_replay_active: bool = False`
- `titans_pe_gate_active: bool = False`
- `replay_window_sizes: tuple[tuple[str, int], ...] = ()` — (band_id, K) pairs.

Added to `CMSCheckpointState`:

- `atlas_replay_active: bool = False`
- `titans_pe_gate_active: bool = False`
- `replay_window_sizes: tuple[tuple[str, int], ...] = ()`

These let downstream evaluation (paper-suite SHADOW evidence builder) tell whether the uplift was active for any given turn.

## 7. SHADOW / ACTIVE protocol

### 7.1 Switch points

Two new construction-time flags on `CMSMemoryCore`:

- `pe_features_enabled: bool = False`
- `replay_window_sizes: Mapping[str, int] | None = None` — when `None`, all bands use K=1.

`build_default_memory_store(...)` exposes:

- `cms_pe_features_enabled: bool = True`
- `cms_replay_window_size: int | None = 8` — applies `{online: K, session: max(2, K//2), background: max(2, K//4)}`.

The default factory is ACTIVE. Rollback is explicit and local:

```python
build_default_memory_store(
    cms_pe_features_enabled=False,
    cms_replay_window_size=None,
)
```

Direct `CMSMemoryCore(...)` construction still defaults to the canonical off path so unit tests and low-level A/B setups can instantiate either side without going through the runtime factory.

### 7.2 SHADOW vs ACTIVE comparison

The dialogue paper-suite gains a new profile label `atlas-titans-cms-uplift`. It runs the same scenario as the canonical `pe-eta` profile but with both uplift flags on. Acceptance is **evidence-only** initially:

- the uplift profile must not regress NL essence default gates (`pe-first / multi-timescale-default / judge-gated-evolution / cross-session-growth`) below the canonical baseline minus 5%;
- the uplift profile must not regress ETA strong-proof `statistical-batch-evidence` below the canonical baseline minus 5%;
- the uplift profile may report higher `slow-shapes-fast` and `rare-heavy-net-benefit`; this is reported but does not count as retain-class success without separate review.

The SHADOW evidence held across the local validation ladder documented in `cms-atlas-titans-uplift-shadow-evidence-2026-05-06.md`; the default factory wiring is now ACTIVE.

### 7.3 Rollback

Pass `cms_pe_features_enabled=False, cms_replay_window_size=None` to `build_default_memory_store(...)` or construct `CMSMemoryCore(...)` directly with the default off flags. CMS reverts to legacy behavior. Old/new checkpoints remain interoperable per Section 5.

### 7.4 Anti-forgetting report-only readout + evidence (#89, 2026-07-04)

The learned CMS path (PE-gate + ATLAS replay) is CPU-ACTIVE by default via the factory; what was missing was *evidence*. Added (report-only, no effect on learning, does not enter any acceptance gate):

- **`CMSState.new_knowledge_absorption` / `old_knowledge_retention`** (both in `[0, 1]`): computed by `CMSMemoryCore` from the actual per-turn band drift — `new_knowledge_absorption` = normalized L2 movement of the online-fast band toward the new signal this turn; `old_knowledge_retention` = `1 - normalized drift of the background-slow band` (the slow substrate should barely move; ATLAS replay + anti-forgetting backflow preserve it).
- Mirrored into `MemoryStore.snapshot().lifecycle_metrics` as `cms_old_knowledge_retention` / `cms_new_knowledge_absorption`, plus observability flags `cms_pe_gate_active` / `cms_atlas_replay_active`.
- **A/B matched-control evidence** (`tests/test_cms_anti_forgetting_evidence.py`): the same interleaved trace (establish an old signal, then hammer a distinct new signal) through the uplift store vs an explicitly rolled-back store; the uplift path's background-slow drift is `<=` the legacy path's (deterministic, so a stable directional claim — uplift does not forget worse than legacy, and the two paths measurably differ). Report-only; this is unit-level evidence, NOT the >=500-turn scale evidence.

**torch band ACTIVE promotion (Stage 1, follow-up):** flipping `cms_torch_backend` DISABLED→SHADOW→ACTIVE (autograd write-back into W1/W2) and the >=500-turn real-substrate anti-forgetting gain curves + torch rollback drill are GPU-gated and tracked as known-debt #89 Stage 1 (same cadence as #88 / #6 / #7).

### 7.5 Stage 1 code-side closure (M2, 2026-07-16)

The code half of Stage 1 is now complete; only the >=500-turn real-trace evidence run remains GPU-gated:

- **SHADOW update-outcome dual-run**: each SHADOW band update now settles a pure-vs-torch comparison — both candidates start from the same pre-update weights and chase the same replay-averaged target; their one-step landing MSE (`update_outcome_pure_mse` / `update_outcome_torch_mse`) is published in `latest_cms_backend_evidence` and aggregated by the promotion tracker. Forward parity (`cms_band_shadow_dual_run`) results are aggregated in lockstep.
- **`CMSMemoryCore.cms_backend_promotion_readout()`** (report-only, frozen dataclass): exit conditions = `settled_comparisons >= 50` + forward-parity pass rate `>= 0.99` + torch update-outcome not worse than pure; kill condition = torch mean MSE worse by `>= 0.05` (recommend staying on / rolling back to the pure baseline). `promotable=True` means the CODE gate passes; the ACTIVE flip stays gated on the real-trace run. The readout ships inside the learned-shadow evidence artifact (`collect_learned_shadow_evidence` → `cms.promotion_readout`).
- **ACTIVE rollback drill (R15)**: every ACTIVE torch write-back records the band's full pre-update `export_params` tuple; `rollback_last_torch_writeback(band_id)` restores it exactly (single-shot, fails loudly without a recorded write-back).
- **Anti-forgetting window hooks**: absorption/retention proxies now also aggregate over a bounded 64-observation window (`absorption_window_mean` / `retention_window_mean` in the promotion readout), the code-side metric surface for the gain-curve evidence run.
- Tests: `tests/test_m2_cms_torch_closure.py` (SHADOW settlement, promotion gate dimensions, DISABLED no-op, rollback drill, window aggregates).

## 8. Acceptance ladder (must pass in order)

1. Unit: `test_cms_uplift_disabled_path_is_bit_equal_to_legacy` — disabled flags reproduce pre-uplift band MLP / decisions for a deterministic seed.
2. Unit: `test_band_mlp_update_with_replay_k1_equals_update` — `update_with_replay([t], [1.0], lr, m) == update(t, lr, m)` numerically (exact).
3. Unit: `test_learned_update_rule_legacy_state_zero_pads_pe_columns` — restoring a `feature_version=1` state into the new rule yields zero PE-column weights.
4. Contract: `test_cms_state_uplift_fields_default_for_canonical` — old consumer code does not break on new frozen fields.
5. Contract: `test_memory_store_observe_substrate_pe_optional` — PE absence path equals pre-uplift call signature behavior.
6. SHADOW smoke: synthetic 50-turn replay with mid-session relationship rupture; verify `online-fast` `update_gate` rises when `relationship_error > 0.4`, falls when `relationship_error < 0.05`, while canonical baseline does not show such modulation.
7. dialogue paper-suite-small: `atlas-titans-cms-uplift` does not regress NL essence default gates by more than 5%.
8. ETA strong-proof: `statistical-batch-evidence` does not regress by more than 5%.

Steps 1–8 passed on 2026-05-06; the uplift is default-ACTIVE via `build_default_memory_store(...)`.

## 9. Out of scope

- Base LLM updates (Tier 3).
- `vz-substrate` adapter / hook layer changes.
- A-Mem / HippoRAG-style external symbolic memory layers (rejected; see assessment §7).
- Replacing `_semantic_embedding` (separate Tier 0 plan).
- Curiosity-Critic critic head (separate Tier 1 plan).
- COCOA in `vz-cognition/credit` (separate Tier 0/1 plan).

## 10. References

- `research/papers/titans-learning-to-memorize-at-test-time-2501.00663.pdf`
- `research/papers/atlas-optimally-memorize-context-test-time-2505.23735.pdf`
- `research/papers/nested-learning-illusion-of-deep-architectures-2512.24695.pdf` (Hope, Continuum Memory System)
- `docs/specs/continuum-memory.md` (existing CMS spec)
- `docs/next_gen_emogpt.md` (R1 / R2 / R5 / R-PE)
- `research/papers/core-author-paper-assessment-2026-05.md` §7–§8 (companionship priority + Tier ladder)
