# P0 ecology mechanism audit — BLOCK

- run-id: `20260729T164048Z-seed0-e7ba7360`
- schema: `digital-ant-ecology-mechanism-audit.v4`
- description: BLOCK: action_chain_final_sensitivity, action_chain_sign_consistency, action_head_update_applied, action_chain_no_rollback, backend_lane_coverage, backend_parity, frozen_evaluation

## Provenance (plan 05 §2.1)

| field | value |
|---|---|
| git SHA | `82a8654dd3b6eb9ea9740a97e404aadb89b0878c` |
| git branch | `main` |
| working tree dirty | True |
| externally retainable | False |
| config digest | `e7ba73605f4d6a173105925253a0e3ebde52a3e5123b879ce07da0dbd00e856f` |
| model fingerprint | `7bc19fcf090d371b87774e69d3cacef4b75531a708ba596749fc10b069745e32` |
| requested device | `cpu` |
| effective backend | `pure:cpu` |
| python | 3.13.5 |
| platform | macOS-26.4.1-arm64-arm-64bit-Mach-O |
| training seeds | [1000003, 1000104, 1000205, 1010003, 1010104, 1010205, 1020003, 1020104, 1020205] |
| layout seeds | [43, 101, 307] |
| dependency versions | vz-contracts=0.1.0, vz-substrate=0.1.0, vz-runtime=0.1.0, numpy=2.3.3, torch=2.10.0 |

## Gates

| gate | verdict | observed |
|---|---|---|
| `action_chain_input_reachability` | PASS | passing_bodies=4/4 failures=() |
| `action_chain_final_sensitivity` | BLOCK | passing_bodies=2/4 failures=('body:0:food:turn-delta=2.06015707e-16', 'body:0:heat:turn-delta=1.96076236e-16', 'body:1:food:turn-delta=5.1462082e-17', 'body:1:heat:turn-delta=1.27866393e-16') |
| `action_chain_sign_consistency` | BLOCK | ((0, 'food', (0, 0, 0), (0, 0, 0)), (0, 'heat', (0, 0, 0), (0, 0, 0)), (1, 'food', (0, 0, 0), (0, 0, 0)), (1, 'heat', (0, 0, 0), (0, 0, 0)), (2, 'food', (0, 1, 1), (0, 1, 1)), (2, 'heat', (-1, 1, 1), (1, 1, 1)), (3, 'food', (-1, -1, -1), (1, 1, 1)), (3, 'heat', (-1, -1, -1), (1, 0, 0))) |
| `action_chain_lateral_bias` | PASS | (('food', 4.353351618704286e-06, -3.5968098427282504e-05, False), ('heat', 3.1873991581157236e-06, -9.101778588916977e-05, False)) |
| `action_head_update_applied` | BLOCK | ((0, 0, True, False), (1, 0, True, False), (2, 3, True, True), (3, 3, True, True)) |
| `action_chain_no_rollback` | BLOCK | {'rollback_episodes': (), 'action_probe_guard_enabled': False, 'gated_checkpoint_is_post_training': False} |
| `backend_lane_coverage` | BLOCK | (('pure', True, (), ''), ('runtime', True, ('temporal_runtime_backend',), ''), ('torch', False, ('temporal_ssl_backend', 'internal_rl_backend'), 'temporal_ssl_backend is active but the SSL trainer reported trained_steps=0 (MetacontrollerSSLTrainer.optimize early-returns on a trace shorter than two steps), so the torch SSL path never ran')) |
| `backend_parity` | BLOCK | (('pure', True, '', 0.0, 0.0, 0.0, 0.0, True), ('runtime', True, '', 0.011702860269352966, 7.812419486802346e-05, 0.0007253651260896166, 0.0, True), ('torch', True, '', 0.04006268787854428, 1.187261761261346e-05, 0.005847481591605919, 0.0, True)) |
| `no_optimize_policy_stable` | PASS | True |
| `temporal_positive_control` | PASS | {'switch_ticks': (0, 9, 10, 18, 19), 'localizations': (('food_approach', 5, 4), ('pickup_carrying', 10, 0), ('home_approach_delivery', 14, 4), ('safe_to_harmful', 19, 0), ('harmful_to_cooling', 23, 4))} |
| `temporal_negative_control` | PASS | switch_rate=0.0417 switch_ticks=(0,) |
| `temporal_segment_closure` | PASS | {'close_reasons': (('beta-switch', 1), ('environment-milestone', 3)), 'timeout_ratio': 0.0} |
| `segment_credit_parity` | PASS | {'sense': 0.0, 'turn': 0.0, 'step': 0.0, 'lineage': (), 'first_misaligned_tick': -1} |
| `frozen_evaluation` | BLOCK | (('butter_only', 307, ('credit', 'dual-track-gate', 'joint-loop/memory', 'prediction', 'reflection', 'regime'), 'learned owner changed under learning_enabled=False: owner=joint-loop/memory field=learning_owner_fingerprints[joint-loop/memory] tick=0 body=0 before=995fca26595136c7 after=2d264893ca3665f1', 1.0, 1.0, 0), ('heat_forced_escape', 101, ('credit', 'dual-track-gate', 'joint-loop/memory', 'p … |

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
