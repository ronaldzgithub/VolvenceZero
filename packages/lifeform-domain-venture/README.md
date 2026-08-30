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

Foundry can pin the complete producer boundary without contacting the service:

```text
venture-foundry-contract show
venture-foundry-contract validate pinned-contract.json
```

The packaged `venture-foundry-public-contract.v1` manifest freezes all seven
decision points, OFF/SHADOW exposure ownership, permanent SHADOW Advice,
the unified `/brain/*` service entry, bounded content-policy lineage, typed
delayed outcomes, user/portfolio/venture isolation, OFF fallback, and
candidate-only production activation. Its content SHA-256 is
`a1767651e680ce4e350cfcaa320c664c94ffdb38fadca3b5029461f2713f69e5`.
The CLI is read-only and never contacts Foundry or writes its ledger.

HTTP projection:

```text
POST /v1/sessions/{session_id}/brain/context-packs
POST /v1/sessions/{session_id}/brain/outcomes
```

`/venture/{context-packs|outcomes}` remains a non-authoritative compatibility
alias for legacy clients.

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
