# P0 ecology mechanism audit — BLOCK

- run-id: `20260726T172736Z-seed0-e11c5959`
- schema: `digital-ant-ecology-mechanism-audit.v4`
- description: BLOCK: action_chain_final_sensitivity, action_chain_sign_consistency, action_head_update_applied, action_chain_no_rollback, backend_lane_coverage, backend_parity, frozen_evaluation

## Provenance (plan 05 §2.1)

| field | value |
|---|---|
| git SHA | `29c2d9d83a7cd875a30e0721dfe4786590bc3fe5` |
| git branch | `fix/ant-ecology-design-conformance` |
| working tree dirty | True |
| externally retainable | False |
| config digest | `e11c59598cdd2aeb5c0c142f580cbcc8146881b381b3c87124b96f465d7568bf` |
| model fingerprint | `be4f03b2cb7df14cd619bb511caae1cc439bf393d739865ce425449076857618` |
| requested device | `cpu` |
| effective backend | `pure:cpu` |
| python | 3.13.5 |
| platform | macOS-26.4.1-arm64-arm-64bit-Mach-O |
| training seeds | [0, 10000, 20000] |
| layout seeds | [43, 101, 307] |
| dependency versions | vz-contracts=0.1.0, vz-substrate=0.1.0, vz-runtime=0.1.0, numpy=2.5.1, torch=2.13.0 |

## Gates

| gate | verdict | observed |
|---|---|---|
| `action_chain_input_reachability` | PASS | passing_bodies=4/4 failures=() |
| `action_chain_final_sensitivity` | BLOCK | passing_bodies=0/4 failures=('body:0:food:turn-delta=6.52096121e-18', 'body:0:heat:turn-delta=3.32543291e-18', 'body:1:food:turn-delta=1.30173003e-17', 'body:1:heat:turn-delta=0', 'body:2:food:turn-delta=6.51931826e-18', 'body:2:heat:turn-delta=0', 'body:3:food:turn-delta=6.4995529e-18', 'body:3:heat:turn-delta=6.64893401e-18') |
| `action_chain_sign_consistency` | BLOCK | ((0, 'food', (0, 0, 0), (0, 0, 0)), (0, 'heat', (0, 0, 0), (0, 0, 0)), (1, 'food', (0, 0, 0), (0, 0, 0)), (1, 'heat', (0, 0, 0), (0, 0, 0)), (2, 'food', (0, 0, 0), (0, 0, 0)), (2, 'heat', (0, 0, 0), (0, 0, 0)), (3, 'food', (0, 0, 0), (0, 0, 0)), (3, 'heat', (0, 0, 0), (0, 0, 0))) |
| `action_chain_lateral_bias` | PASS | (('food', 8.101873381607797e-19, 2.4394012860814756e-18, False), ('heat', -4.1543763755887423e-19, -4.1543763755887423e-19, False)) |
| `action_head_update_applied` | BLOCK | ((0, 0, True, False), (1, 0, True, False), (2, 0, True, False), (3, 0, True, False)) |
| `action_chain_no_rollback` | BLOCK | {'rollback_episodes': (), 'action_probe_guard_enabled': False, 'gated_checkpoint_is_post_training': False} |
| `backend_lane_coverage` | BLOCK | (('pure', True, (), ''), ('runtime', True, ('temporal_runtime_backend',), ''), ('torch', False, ('temporal_ssl_backend', 'internal_rl_backend'), 'temporal_ssl_backend is active but the SSL trainer reported trained_steps=0 (MetacontrollerSSLTrainer.optimize early-returns on a trace shorter than two steps), so the torch SSL path never ran')) |
| `backend_parity` | BLOCK | (('pure', True, '', 0.0, 0.0, 0.0, 0.0, True), ('runtime', True, '', 0.01576995587103082, 3.4995945086252626e-17, 1.447156749989631e-08, 0.0, True), ('torch', True, '', 0.016333596319971666, 2.2062082594841544e-17, 0.0, 0.0, True)) |
| `no_optimize_policy_stable` | PASS | True |
| `temporal_positive_control` | PASS | {'switch_ticks': (19,), 'localizations': (('food_approach', 5, 14), ('pickup_carrying', 10, 9), ('home_approach_delivery', 14, 5), ('safe_to_harmful', 19, 0), ('harmful_to_cooling', 23, 4))} |
| `temporal_negative_control` | PASS | switch_rate=0.0000 switch_ticks=() |
| `temporal_segment_closure` | PASS | {'close_reasons': (('beta-switch', 1), ('environment-milestone', 3)), 'timeout_ratio': 0.0} |
| `segment_credit_parity` | PASS | {'sense': 4.5918824298496475e-12, 'turn': 4.203786852179319e-17, 'step': 2.3475665855698935e-13, 'lineage': (), 'first_misaligned_tick': -1} |
| `frozen_evaluation` | BLOCK | (('butter_only', 307, ('credit', 'dual-track-gate', 'joint-loop/memory', 'prediction', 'reflection', 'regime'), 'learned owner changed under learning_enabled=False: owner=joint-loop/memory field=learning_owner_fingerprints[joint-loop/memory] tick=0 body=0 before=4d4319d9b4cbff97 after=02a394dcebd8886c', 1.0, 1.0, 0), ('heat_forced_escape', 101, ('credit', 'dual-track-gate', 'joint-loop/memory', 'p … |

## Breakpoints

- `action_chain_final_sensitivity`
- `action_chain_sign_consistency`
- `action_head_update_applied`
- `action_chain_no_rollback`
- `backend_lane_coverage`
- `backend_parity`
- `frozen_evaluation`

## Gate thresholds

- `action_chain_input_reachability`: shared-initial food/heat/obstacle/home paired swaps reach code on the required body ratio; turn is NOT gated here because a cold exclusive-steering head is zero by design
- `action_chain_final_sensitivity`: final trained checkpoint: food/heat turn_delta >= 0.0001 and >= 0.25 of the peak acquired sensitivity, on the required body ratio
- `action_chain_sign_consistency`: 3 repeated probes at the final checkpoint agree on a non-zero left and right turn sign
- `action_chain_lateral_bias`: colony-mean side-independent turn does not dominate the colony-mean left/right contrast
- `action_head_update_applied`: learned bodies published a finite, non-NaN action-head residual, a positive update step and a changed policy fingerprint on >= 4 bodies
- `action_chain_no_rollback`: the gated checkpoint is the TRAINED one. The guard-disabled precondition is really enforced by a raise in _train_audit_arm (the curriculum only rolls back when both the flag and a baseline are supplied, and the audit supplies neither), so the first two clauses are a belt-and-braces publication of that fact and cannot fail on their own. The clause that CAN fail is the third: the final learned checkpoint's policy fingerprint must differ from the shared-initial one on every body, which is exactly the state a silent rollback -- or an optimizer that never moved -- would produce
- `backend_lane_coverage`: every declared backend of every lane is shown to have EXECUTED by its own owner's evidence readout (applied wiring on the live policy, SSL trained steps / torch backend, Internal-RL optimization report and write-back). Numeric difference from the reference is deliberately NOT admissible evidence of coverage
- `backend_parity`: pure/runtime/torch each ran on its declared wiring and all agree on final code, turn, step, action-head residual and the owner-published action distribution within 0.001, on the same abstract action. EXACT agreement passes: agreement is the ideal outcome, and whether a lane ran at all is the separate backend_lane_coverage gate
- `no_optimize_policy_stable`: no-optimize policy fingerprints equal shared initial
- `temporal_positive_control`: the scripted transition trace produces at least one real beta switch and at least one switch within 4 ticks of a declared state change
- `temporal_negative_control`: steady-state input switch rate <= 0.2
- `temporal_segment_closure`: at least one beta-switch closure, at least one milestone/terminal closure, and a bounded-horizon closure ratio strictly below 1
- `segment_credit_parity`: segment-credit on/off share sense, pre-credit action and rollout lineage within 1e-06
- `frozen_evaluation`: every gated learning owner is fingerprint-identical on every tick under learning_enabled=False; replay settlement and lineage >= 0.99 with no drops

## First failing learned episode (plan 05:130 bisect trigger)

- learned:butter:near:episode:0

## Declared diagnostic-only surfaces

- `action_chain_snapshots[learned:per-episode]` — gated=False: The intermediate learned snapshots are EVALUATED and recorded (each carries its own passed/failures), but no gate reads them: gating a mid-training episode would block on a checkpoint that later training is allowed to improve. plan 05:130's 'first failing episode -> bisect replay' branch is triggered by report.first_failing_learned_episode instead, which names the first one that failed.
- `action_chain_snapshots[no_optimize:per-episode]` — gated=False: The no-optimize arm's action sensitivity is a control, not a requirement: what the arm must prove is that its policy fingerprint never moved, and that is the no_optimize_policy_stable gate.
- `lateral_bias / sign_consistency probe seeds` — gated=True: Read by action_chain_lateral_bias and action_chain_sign_consistency.

## Declared gaps against the plan

### research/ant/05_ecology_p0_p1_p2_plan.md:121

- requirement: 训练后的 turn sensitivity 不得低于 shared-initial 的 25%，除非绝对值仍高于预先声明的任务有效阈值。
- status: DEVIATION: the retention floor is taken against the PEAK ACQUIRED sensitivity of the arm, not against shared-initial, and the plan's OR-form escape clause is not implemented -- both the absolute turn threshold AND the retention floor must hold. shared-initial is unusable as a baseline because a cold exclusive-steering head is exactly zero by design, so a floor derived from it evaluates to 0 and the gate becomes vacuous (_retention_floor raises rather than deriving one). The implemented rule is strictly stricter than the plan's, but the plan sentence above must be rewritten before P0 can be signed off; that edit belongs to the documentation package, not here.
- owner: research/ant/05_ecology_p0_p1_p2_plan.md
- currently failing a gate: False

### research/ant/05_ecology_p0_p1_p2_plan.md:149

- requirement: closure cause：world switch、self switch、milestone、terminal 或 max-step timeout。
- status: GAP: the runtime-replay owner publishes a single 'beta-switch' close reason for both tracks, so a world-track closure and a self-track closure are indistinguishable in close_reason_counts. The per-tick log does carry both tracks' beta separately, so the split is recoverable by hand but is not gated. Closing it means changing the close-reason vocabulary in the vz-temporal runtime-replay owner, which this package does not own.
- owner: vz-temporal runtime-replay segment owner
- currently failing a gate: False

### research/ant/05_ecology_p0_p1_p2_plan.md:150

- requirement: SSL 前后 switch 参数和 histogram。
- status: PARTIAL: switch parameters before/after each trace are published (EcologySwitchParameterSnapshot) and this audit derives its own ten-bin beta histogram per track from its per-tick log. The owner-published SwitchGateStats histogram is NOT available: the ant session's residual trace is shorter than two steps every turn, so MetacontrollerSSLTrainer.optimize early-returns and the SSL trainer never trains (trained_steps=0). That same fact is what fails the temporal_ssl_backend coverage lane.
- owner: vz-temporal MetacontrollerSSLTrainer / the ant trace length
- currently failing a gate: True

## Test command

```text
cd <repo> && export PYTHONPATH="$(ls -d packages/*/src | paste -sd: -)" && .venv/bin/python -m pytest packages/vz-embodiment-ant/tests/test_ecology_mechanism_audit.py packages/vz-embodiment-ant/tests/test_ecology_action_chain_audit.py packages/vz-embodiment-ant/tests/test_ecology_temporal_switch_audit.py packages/vz-embodiment-ant/tests/test_ecology_frozen_evaluation.py -q --no-header -p no:cacheprovider
```
