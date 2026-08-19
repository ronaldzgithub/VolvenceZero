# Relationship Lab P1 development calibration

- artifact_id: `aedc381c26789a62fd22eeaade8d3eac070d58ea2602f3b79ebf387bd0cf9643`
- machinery_ready: **false**
- gate1_passed: **false**

## Checks

- `appendable_cross_process_recovery` — **pass**: Fresh-process recovery reproduced every scoped MemoryStore and companion-ref-harness record digest.
- `user_swap_scope_isolation` — **fail**: Mirrored context or user-scope isolation is broken.
- `token_scaling` — **pass**: Full history grows with ordinary turns while ref-harness RAG and MemoryStore typed state remain bounded.
- `console_correction_and_delete` — **pass**: Owner-side correction and deletion survive restart without mutating the mirrored user's scope.
- `same_substrate_and_valid_outputs` — **pass**: All arms share the frozen Gate 0 substrate and every decision is valid structured output.
- `structured_state_user_swap_effect` — **pass**: The stateless arm cannot distinguish byte-identical users while the scoped structured state changes the selected action.
- `steelman_qualification` — **fail**: A steelman is too weak or the development set is saturated.

## Claim boundary

P1 is development evidence for append/recover/control and baseline qualification only. The preregistration and secret heldout remain unfrozen; no four-capability or product claim is allowed.
