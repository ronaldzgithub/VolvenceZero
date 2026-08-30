from __future__ import annotations

from lifeform_domain_venture import build_venture_lifeform
from lifeform_service.alpha import AlphaServiceConfig
from lifeform_service.app import create_app
from lifeform_service.verticals import VerticalSpec, _try_venture


def _evidence(
    *,
    ref_id: str,
    role: str,
    observed_at_ms: int,
    digest_char: str,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": "field",
        "role": role,
        "locator": f"foundry://evidence/{ref_id}",
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
        "schema_version": "venture-context-request.v1",
        "request_id": request_id,
        "portfolio_id": "portfolio-1",
        "cycle_id": cycle_id,
        "venture_id": "venture-1",
        "decision_id": decision_id,
        "decision_point": "experiment_planning",
        "confirmed_facts": [
            {
                "fact_id": "fact-1",
                "kind": "demand_signal",
                "statement": "Three customers requested the workflow.",
                "evidence_ref_ids": ["demand-1"],
                "as_of_ms": 100,
            }
        ],
        "constraints": [
            {
                "constraint_id": "constraint-1",
                "kind": "budget",
                "description": "Use only the approved reversible test budget.",
                "hard": True,
            }
        ],
        "resource_window": {
            "currency": "USD",
            "maximum_total_cost_minor": 10_000,
            "starts_at_ms": 100,
            "ends_at_ms": 10_000,
            "maximum_experiments": 2,
        },
        "uncertainties": [
            {
                "uncertainty_id": "uncertainty-1",
                "statement": "Paid conversion remains unknown.",
                "probability_lower": 0.1,
                "probability_upper": 0.5,
                "evidence_ref_ids": ["demand-1"],
            }
        ],
        "evidence_refs": [
            _evidence(
                ref_id="demand-1",
                role="demand_signal",
                observed_at_ms=100,
                digest_char="a",
            )
        ],
    }


def _field_outcome(*, context_pack_id: str, outcome_id: str = "outcome-1") -> dict[str, object]:
    costs = {
        "acquisition_minor": 100,
        "model_minor": 200,
        "data_minor": 100,
        "human_review_minor": 200,
        "delivery_minor": 300,
        "support_minor": 100,
        "risk_reserve_minor": 500,
    }
    return {
        "schema_version": "venture-outcome-report.v1",
        "outcome_id": outcome_id,
        "context_pack_id": context_pack_id,
        "decision_id": "decision-1",
        "decision": "continue",
        "outcome_kind": "field_experiment_result",
        "evidence_class": "field",
        "verdict": "favorable",
        "summary": "Foundry supplied a favorable multi-objective field verdict.",
        "detail": "A customer paid and completed the bounded field workflow.",
        "observed_at_ms": 200,
        "evidence_refs": [
            _evidence(
                ref_id="field-result-1",
                role="field_observation",
                observed_at_ms=200,
                digest_char="b",
            ),
            _evidence(
                ref_id="field-customer-1",
                role="customer_outcome",
                observed_at_ms=200,
                digest_char="c",
            ),
            _evidence(
                ref_id="field-payment-1",
                role="payment",
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
                ref_id="field-refund-1",
                role="refund",
                observed_at_ms=200,
                digest_char="f",
            ),
        ],
        "commercial_outcome": {
            "customer_result": "positive",
            "currency": "USD",
            "realized_revenue_minor": 5_000,
            "realized_costs": costs,
            "refund_minor": 100,
            "realized_net_value_minor": 3_400,
            "elapsed_ms": 86_400_000,
            "risk_level": "low",
            "reversibility": "reversible",
        },
    }


async def test_http_context_outcome_idempotency_and_next_turn_pe_lineage(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_venture()
    assert vertical is not None
    client = await aiohttp_client(create_app(vertical=vertical))
    created_session = await client.post("/v1/sessions", json={"session_id": "venture-http"})
    assert created_session.status == 201

    first = await client.post(
        "/v1/sessions/venture-http/venture/context-packs",
        json=_context_payload(),
    )
    assert first.status == 201
    first_body = await first.json()
    assert first_body["schema_version"] == "venture-context-pack.v1"
    assert first_body["wiring_level"] == "active"
    assert first_body["advice"]["wiring_level"] == "shadow"
    assert first_body["advice"]["applied"] is False

    replay = await client.post(
        "/v1/sessions/venture-http/venture/context-packs",
        json=_context_payload(),
    )
    assert replay.status == 200
    assert (await replay.json())["context_pack_id"] == first_body["context_pack_id"]

    outcome_payload = _field_outcome(context_pack_id=first_body["context_pack_id"])
    outcome = await client.post(
        "/v1/sessions/venture-http/venture/outcomes",
        json=outcome_payload,
    )
    assert outcome.status == 201
    receipt = await outcome.json()
    assert receipt["schema_version"] == "venture-outcome-receipt.v1"
    assert receipt["learning_route"] == "field_pe_memory_and_execution_result"
    assert receipt["environment_outcome_id"]
    assert receipt["source_advice_applied"] is False

    outcome_replay = await client.post(
        "/v1/sessions/venture-http/venture/outcomes",
        json=outcome_payload,
    )
    assert outcome_replay.status == 200
    assert (await outcome_replay.json())["receipt_id"] == receipt["receipt_id"]

    second = await client.post(
        "/v1/sessions/venture-http/venture/context-packs",
        json=_context_payload(request_id="request-2", decision_id="decision-2", cycle_id="cycle-2"),
    )
    assert second.status == 201
    second_body = await second.json()
    assert receipt["environment_outcome_id"] in second_body["settled_outcome_ids"]
    assert "field-result-1" in second_body["settled_evidence_ref_ids"]
    assert receipt["memory_entry_id"] in second_body["source_entry_ids"]


async def test_http_contract_unknown_fields_and_conflicts_fail_closed(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_venture()
    assert vertical is not None
    client = await aiohttp_client(create_app(vertical=vertical))
    await client.post("/v1/sessions", json={"session_id": "venture-errors"})

    invalid = await client.post(
        "/v1/sessions/venture-errors/venture/context-packs",
        json={**_context_payload(), "guessed_evidence_class": "field"},
    )
    assert invalid.status == 400
    assert (await invalid.json())["error"] == "invalid_venture_context_request"

    first = await client.post(
        "/v1/sessions/venture-errors/venture/context-packs",
        json=_context_payload(),
    )
    assert first.status == 201
    first_body = await first.json()
    conflict = await client.post(
        "/v1/sessions/venture-errors/venture/context-packs",
        json={**_context_payload(), "cycle_id": "different-cycle"},
    )
    assert conflict.status == 409
    assert (await conflict.json())["error"] == "venture_idempotency_conflict"

    unknown_pack = await client.post(
        "/v1/sessions/venture-errors/venture/outcomes",
        json=_field_outcome(context_pack_id="venture-context-pack:" + "c" * 64),
    )
    assert unknown_pack.status == 409
    assert (await unknown_pack.json())["error"] == "venture_context_lineage_error"

    invalid_pair = _field_outcome(context_pack_id=first_body["context_pack_id"])
    invalid_pair["evidence_class"] = "machine_check"
    invalid_pair["outcome_kind"] = "field_experiment_result"
    invalid_pair["evidence_refs"] = [
        {
            **invalid_pair["evidence_refs"][0],  # type: ignore[index]
            "evidence_class": "machine_check",
            "role": "machine_audit",
        }
    ]
    invalid_outcome = await client.post(
        "/v1/sessions/venture-errors/venture/outcomes",
        json=invalid_pair,
    )
    assert invalid_outcome.status == 400
    assert (await invalid_outcome.json())["error"] == "invalid_venture_outcome"


async def test_http_rejects_non_venture_and_historical_sessions(
    aiohttp_client,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    other = VerticalSpec(
        name="other",
        factory=lambda runtime: build_venture_lifeform(substrate_runtime=runtime),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )
    other_client = await aiohttp_client(create_app(vertical=other))
    await other_client.post("/v1/sessions", json={"session_id": "other-session"})
    wrong_vertical = await other_client.post(
        "/v1/sessions/other-session/venture/context-packs",
        json=_context_payload(),
    )
    assert wrong_vertical.status == 409
    assert (await wrong_vertical.json())["error"] == "venture_vertical_required"

    vertical = _try_venture()
    assert vertical is not None
    app = create_app(
        vertical=vertical,
        alpha_config=AlphaServiceConfig(
            enabled=True,
            memory_scope_root_dir=str(tmp_path),
            alpha_users=frozenset({"foundry-1"}),
        ),
    )
    client = await aiohttp_client(app)
    await client.post(
        "/v1/sessions",
        json={"session_id": "venture-live", "user_id": "foundry-1"},
    )
    manager = app["session_manager"]
    nodes = await manager.list_time_nodes(session_id="venture-live", limit=1)
    assert len(nodes) == 1
    await manager.fork_session(
        source_session_id="venture-live",
        fork_session_id="venture-historical",
        time_node_id=nodes[0].time_node_id,
        scope_key=nodes[0].scope_key,
        mode="historical_readonly",
        user_id="foundry-1",
    )
    historical = await client.post(
        "/v1/sessions/venture-historical/venture/context-packs",
        json=_context_payload(request_id="historical-request"),
        headers={"X-Alpha-User": "foundry-1"},
    )
    assert historical.status == 409
    assert (await historical.json())["error"] == "historical_session_readonly"


async def test_http_alpha_session_rejects_missing_or_cross_user_scope(
    aiohttp_client,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_venture()
    assert vertical is not None
    app = create_app(
        vertical=vertical,
        alpha_config=AlphaServiceConfig(
            enabled=True,
            memory_scope_root_dir=str(tmp_path),
            alpha_users=frozenset({"foundry-owner", "foundry-other"}),
        ),
    )
    client = await aiohttp_client(app)
    created = await client.post(
        "/v1/sessions",
        json={"session_id": "venture-owner", "user_id": "foundry-owner"},
    )
    assert created.status == 201

    missing = await client.post(
        "/v1/sessions/venture-owner/venture/context-packs",
        json=_context_payload(request_id="missing-user"),
    )
    assert missing.status == 400
    assert (await missing.json())["error"] == "missing_venture_user_scope"

    other = await client.post(
        "/v1/sessions/venture-owner/venture/context-packs",
        json=_context_payload(request_id="cross-user"),
        headers={"X-Alpha-User": "foundry-other"},
    )
    assert other.status == 403
    assert (await other.json())["error"] == "venture_user_scope_forbidden"

    owner = await client.post(
        "/v1/sessions/venture-owner/venture/context-packs",
        json=_context_payload(request_id="owner-user"),
        headers={"X-Alpha-User": "foundry-owner"},
    )
    assert owner.status == 201
