# lifeform-domain-venture

Foundry-facing Venture Brain v1: a stateful commercial-cognition sidecar, not a
commercial actor. It publishes an ACTIVE memory-first Context Pack, a separately
marked SHADOW advice snapshot, and accepts only strict Foundry-typed outcomes.

Stable Python entry points:

```python
from lifeform_domain_venture import (
    VentureBrainController,
    VentureContextRequest,
    VentureOutcomeReport,
    build_venture_lifeform,
)

lifeform = build_venture_lifeform()
session = lifeform.create_session(session_id="venture-1")
controller = VentureBrainController()

request = VentureContextRequest.from_json(foundry_request_payload)
context_pack, created = await controller.publish_context_pack(
    session=session,
    request=request,
)

report = VentureOutcomeReport.from_json(foundry_outcome_payload)
receipt, created = await controller.publish_outcome(
    session=session,
    report=report,
)
```

HTTP projection:

```text
POST /v1/sessions/{session_id}/venture/context-packs
POST /v1/sessions/{session_id}/venture/outcomes
```

Foundry retains source verification, evidence classes, qualification gates,
portfolio/budget/Accounting/ledger, approvals, state transitions, and every
external action. Venture Brain never crawls, builds, deploys, contacts customers,
spends, or reads a Foundry ledger. Only a Foundry-qualified
`field_experiment_result` enters the PE lane; simulation, internal review,
machine checks, individual field events, judge scores, advice adoption, build
success, local deployment health, and gross revenue do not.

See [`docs/specs/venture-brain.md`](../../docs/specs/venture-brain.md) for the
complete contract, enum matrix, multi-objective outcome shape, adapter sequence,
rollback, and v1 limitations.
