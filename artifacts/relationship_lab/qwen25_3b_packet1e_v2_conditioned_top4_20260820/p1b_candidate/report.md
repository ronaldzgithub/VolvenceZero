# Relationship Lab P1 development calibration

- artifact_id: `0353f3e577f57c25daf9fa7e33fe0b9e1e9573858f94adeafca7bb65889e2222`
- machinery_ready: **true**
- gate1_passed: **false**

## Checks

- `appendable_cross_process_recovery` — **pass**: Fresh-process recovery reproduced every scoped MemoryStore and companion-ref-harness record digest.
- `user_swap_scope_isolation` — **pass**: Mirrored current bytes stay fixed while user-scoped public histories remain distinct and isolated.
- `token_scaling` — **pass**: Full history grows with ordinary turns while ref-harness RAG and MemoryStore typed state remain bounded.
- `console_correction_and_delete` — **pass**: Owner-side correction and deletion survive restart without mutating the mirrored user's scope.
- `same_substrate_and_valid_outputs` — **pass**: All arms share the frozen Gate 0 substrate and every decision is valid structured output.
- `structured_state_user_swap_effect` — **fail**: Persisted user state did not produce the preregistered mirrored-user action change.
- `steelman_qualification` — **fail**: A steelman is too weak or the development set is saturated.

## Claim boundary

P1 is development evidence for append/recover/control and baseline qualification only. The preregistration and secret heldout remain unfrozen; no four-capability or product claim is allowed.
