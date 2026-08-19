# Relationship Lab P1 development calibration

- artifact_id: `2b32f86d5277b938dc3fb2a662400c6faa9d7aa7d06daf59e6ff54d852c730ce`
- machinery_ready: **false**
- gate1_passed: **false**

## Checks

- `appendable_cross_process_recovery` — **pass**: Fresh-process recovery reproduced every scoped MemoryStore and companion-ref-harness record digest.
- `user_swap_scope_isolation` — **pass**: Mirrored current bytes stay fixed while user-scoped public histories remain distinct and isolated.
- `token_scaling` — **pass**: Full history grows with ordinary turns while ref-harness RAG and MemoryStore typed state remain bounded.
- `console_correction_and_delete` — **pass**: Owner-side correction and deletion survive restart without mutating the mirrored user's scope.
- `same_substrate_and_valid_outputs` — **fail**: P1 lineage diverged from Gate 0 or an arm emitted invalid output.
- `structured_state_user_swap_effect` — **pass**: The stateless arm cannot distinguish byte-identical users while the scoped structured state changes the selected action.
- `steelman_qualification` — **fail**: A steelman is too weak or the development set is saturated.

## Claim boundary

P1 is development evidence for append/recover/control and baseline qualification only. The preregistration and secret heldout remain unfrozen; no four-capability or product claim is allowed.
