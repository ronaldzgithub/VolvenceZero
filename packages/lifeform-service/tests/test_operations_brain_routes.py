from __future__ import annotations

from lifeform_domain_operations import build_operations_lifeform
from lifeform_service.app import create_app
from lifeform_service.verticals import VerticalSpec, _try_operations


_WORK_ORDER_REF = "autocompany://work-orders/work-order-1"


def _evidence(
    *,
    ref_id: str,
    role: str,
    observed_at_ms: int,
    digest_char: str,
    locator: str | None = None,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": "field",
        "role": role,
        "locator": locator or f"autocompany://evidence/{ref_id}",
        "content_sha256": digest_char * 64,
        "observed_at_ms": observed_at_ms,
    }


def _context_payload(
    *,
    request_id: str = "request-1",
    decision_id: str = "decision-1",
    cycle_id: str = "cycle-1",
) -> dict[str, object]:
    return {
        "schema_version": "operations-context-request.v1",
        "request_id": request_id,
        "company_id": "company-1",
        "cycle_id": cycle_id,
        "workstream_id": "reliability",
        "decision_id": decision_id,
        "decision_point": "work_prioritization",
        "division_ids": ["division-engineering", "division-support"],
        "action_catalog_ids": ["catalog:repair-service", "catalog:pause-rollout"],
        "confirmed_facts": [
            {
                "fact_id": "fact-1",
                "kind": "division_health",
                "division_id": "division-engineering",
                "statement": "The service error budget is nearly exhausted.",
                "evidence_ref_ids": ["operating-signal-1"],
                "as_of_ms": 100,
            }
        ],
        "constraints": [
            {
                "constraint_id": "constraint-1",
                "kind": "budget",
                "division_id": "",
                "description": "Use only the approved bounded operating budget.",
                "hard": True,
            }
        ],
        "operating_window": {
            "currency": "USD",
            "maximum_external_cost_minor": 10_000,
            "maximum_human_minutes": 240,
            "starts_at_ms": 100,
            "ends_at_ms": 10_000,
            "maximum_work_orders": 2,
        },
        "uncertainties": [
            {
                "uncertainty_id": "uncertainty-1",
                "statement": "The primary failure mechanism is not yet isolated.",
                "probability_lower": 0.3,
                "probability_upper": 0.7,
                "evidence_ref_ids": ["operating-signal-1"],
            }
        ],
        "evidence_refs": [
            _evidence(
                ref_id="operating-signal-1",
                role="operating_signal",
                observed_at_ms=100,
                digest_char="a",
            )
        ],
    }


def _field_outcome(*, context_pack_id: str, outcome_id: str = "outcome-1") -> dict[str, object]:
    costs = {
        "model_minor": 100,
        "data_minor": 100,
        "human_minor": 200,
        "infrastructure_minor": 300,
        "vendor_minor": 0,
        "incident_response_minor": 0,
        "other_minor": 0,
    }
    return {
        "schema_version": "operations-outcome-report.v1",
        "outcome_id": outcome_id,
        "context_pack_id": context_pack_id,
        "decision_id": "decision-1",
        "work_order_ref": _WORK_ORDER_REF,
        "decision": "accept",
        "outcome_kind": "field_operation_result",
        "evidence_class": "field",
        "verdict": "favorable",
        "summary": "AutoCompany supplied a favorable multi-objective field verdict.",
        "detail": "The bounded repair reduced errors without a new incident.",
        "observed_at_ms": 200,
        "evidence_refs": [
            _evidence(
                ref_id="field-result-1",
                role="field_observation",
                observed_at_ms=200,
                digest_char="b",
            ),
            _evidence(
                ref_id="field-work-order-1",
                role="work_order",
                observed_at_ms=200,
                digest_char="c",
                locator=_WORK_ORDER_REF,
            ),
            _evidence(
                ref_id="field-objective-1",
                role="objective_progress",
                observed_at_ms=200,
                digest_char="d",
            ),
            _evidence(
                ref_id="field-cost-1",
                role="cost",
                observed_at_ms=200,
                digest_char="e",
            ),
            _evidence(
                ref_id="field-human-1",
                role="human_load",
                observed_at_ms=200,
                digest_char="f",
            ),
        ],
        "execution_outcome": {
            "objective_result": "advanced",
            "metrics": [
                {
                    "metric_id": "error_rate",
                    "unit": "ratio",
                    "baseline_value": 0.08,
                    "observed_value": 0.03,
                    "evidence_ref_ids": ["field-result-1"],
                }
            ],
            "currency": "USD",
            "realized_costs": costs,
            "elapsed_ms": 86_400_000,
            "blocker_duration_ms": 3_600_000,
            "rework_count": 1,
            "incident_count": 0,
            "human_minutes": 90,
            "risk_level": "low",
            "reversibility": "reversible",
        },
    }


async def test_http_context_outcome_idempotency_and_next_turn_pe_lineage(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    client = await aiohttp_client(create_app(vertical=vertical))
    created_session = await client.post("/v1/sessions", json={"session_id": "operations-http"})
    assert created_session.status == 201

    first = await client.post(
        "/v1/sessions/operations-http/operations/context-packs",
        json=_context_payload(),
    )
    assert first.status == 201
    first_body = await first.json()
    assert first_body["schema_version"] == "operations-context-pack.v1"
    assert first_body["wiring_level"] == "active"
    assert first_body["advice"]["wiring_level"] == "shadow"
    assert first_body["advice"]["applied"] is False

    replay = await client.post(
        "/v1/sessions/operations-http/operations/context-packs",
        json=_context_payload(),
    )
    assert replay.status == 200
    assert (await replay.json())["context_pack_id"] == first_body["context_pack_id"]

    outcome_payload = _field_outcome(context_pack_id=first_body["context_pack_id"])
    outcome = await client.post(
        "/v1/sessions/operations-http/operations/outcomes",
        json=outcome_payload,
    )
    assert outcome.status == 201
    receipt = await outcome.json()
    assert receipt["schema_version"] == "operations-outcome-receipt.v1"
    assert receipt["learning_route"] == "field_operation_pe_memory_and_execution_result"
    assert receipt["work_order_ref"] == _WORK_ORDER_REF
    assert receipt["environment_outcome_id"]
    assert receipt["source_advice_applied"] is False

    outcome_replay = await client.post(
        "/v1/sessions/operations-http/operations/outcomes",
        json=outcome_payload,
    )
    assert outcome_replay.status == 200
    assert (await outcome_replay.json())["receipt_id"] == receipt["receipt_id"]

    second = await client.post(
        "/v1/sessions/operations-http/operations/context-packs",
        json=_context_payload(request_id="request-2", decision_id="decision-2", cycle_id="cycle-2"),
    )
    assert second.status == 201
    second_body = await second.json()
    assert receipt["environment_outcome_id"] in second_body["settled_outcome_ids"]
    assert "field-result-1" in second_body["settled_evidence_ref_ids"]
    assert receipt["memory_entry_id"] in second_body["source_entry_ids"]


async def test_http_contract_and_vertical_guards_fail_closed(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    client = await aiohttp_client(create_app(vertical=vertical))
    await client.post("/v1/sessions", json={"session_id": "operations-errors"})

    invalid = await client.post(
        "/v1/sessions/operations-errors/operations/context-packs",
        json={**_context_payload(), "guessed_evidence_class": "field"},
    )
    assert invalid.status == 400
    assert (await invalid.json())["error"] == "invalid_operations_context_request"

    malformed = await client.post(
        "/v1/sessions/operations-errors/operations/context-packs",
        data="{",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status == 400
    assert (await malformed.json())["error"] == "invalid_json_body"

    non_object = await client.post(
        "/v1/sessions/operations-errors/operations/outcomes",
        json=[],
    )
    assert non_object.status == 400
    assert (await non_object.json())["error"] == "invalid_json_body"

    unknown_pack = await client.post(
        "/v1/sessions/operations-errors/operations/outcomes",
        json=_field_outcome(context_pack_id="operations-context-pack:" + "c" * 64),
    )
    assert unknown_pack.status == 409
    assert (await unknown_pack.json())["error"] == "operations_context_lineage_error"

    other = VerticalSpec(
        name="other",
        factory=lambda runtime: build_operations_lifeform(substrate_runtime=runtime),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )
    other_client = await aiohttp_client(create_app(vertical=other))
    await other_client.post("/v1/sessions", json={"session_id": "other-session"})
    wrong_vertical = await other_client.post(
        "/v1/sessions/other-session/operations/context-packs",
        json=_context_payload(),
    )
    assert wrong_vertical.status == 409
    assert (await wrong_vertical.json())["error"] == "operations_vertical_required"
