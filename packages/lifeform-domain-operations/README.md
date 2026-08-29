# lifeform-domain-operations

AutoCompany-facing Operations Brain v1/v2 is a stateful operational-cognition sidecar,
not a second COO or execution authority. It publishes an ACTIVE memory-first Context
Pack, a non-empty bounded policy surface that is SHADOW by default, and accepts only
strict AutoCompany-typed work-order outcomes. v2 ranks typed candidates and learns
intervention timing only from exact Prediction Error credit lineage. Every policy
decision is bound to an owner-issued source prediction; an outcome earns policy
credit only when AutoCompany explicitly records that the policy action was applied.
Rejecting or overriding an ACTIVE suggestion therefore cannot be misattributed to
the policy.

Stable Python entry points:

```python
from lifeform_domain_operations import (
    OperationsBrainController,
    OperationsContextRequest,
    OperationsOutcomeReport,
    build_operations_lifeform,
)

lifeform = build_operations_lifeform()
session = lifeform.create_session(session_id="operations-1")
controller = OperationsBrainController()

request = OperationsContextRequest.from_json(autocompany_request_payload)
context_pack, created = await controller.publish_context_pack(
    session=session,
    request=request,
)

report = OperationsOutcomeReport.from_json(autocompany_outcome_payload)
receipt, created = await controller.publish_outcome(
    session=session,
    report=report,
)
```

Staging ACTIVE requires both the promoted checkpoint and its exact
`OperationsPolicyActivationReceipt`; setting ACTIVE without a receipt fails at
controller construction. The checked evidence bundle is generated with:

```text
python scripts/run_operations_policy_benchmark.py \
  --output-dir artifacts/operations_brain/<new-run-id>
```

The current gate is based on deterministic simulation and is scoped only to
`autocompany_staging`; it does not change the production SHADOW default.

HTTP projection:

```text
POST /v1/sessions/{session_id}/operations/context-packs
POST /v1/sessions/{session_id}/operations/outcomes
```

AutoCompany retains OKR, budget, approval, work-order SSOT, dispatch, governance
ledger, state transitions, and every external action. Operations Brain never
dispatches a division, mutates an AutoCompany ledger, spends, deploys, or promotes
its own advice. Only an AutoCompany-qualified `field_operation_result` with matching
work-order evidence enters the PE lane; simulation, internal review, machine checks,
individual progress events, judge scores, and advice adoption do not.

See [`docs/specs/operations-brain.md`](../../docs/specs/operations-brain.md) for the
complete contract, enum matrix, multi-objective outcome shape, adapter sequence,
PE/credit learning path, promotion/rollback, and evidence limitations.
