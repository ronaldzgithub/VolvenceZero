from __future__ import annotations

from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY, InstanceManager
from lifeform_service.app import create_app as create_lifeform_app
from lifeform_service.verticals import _try_operations

from dlaas_platform_api.app import attach_dlaas_routes


_WORK_ORDER_REF = "autocompany://work-orders/work-order-1"


def _evidence(
    *,
    ref_id: str,
    role: str,
    digest_char: str,
    locator: str | None = None,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "evidence_class": "field",
        "role": role,
        "locator": locator or f"autocompany://evidence/{ref_id}",
        "content_sha256": digest_char * 64,
        "observed_at_ms": 100,
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
        "division_ids": ["division-engineering"],
        "action_catalog_ids": ["catalog:repair-service"],
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
        "uncertainties": [],
        "evidence_refs": [
            _evidence(
                ref_id="operating-signal-1",
                role="operating_signal",
                digest_char="a",
            )
        ],
    }


def _outcome_payload(
    *,
    context_pack_id: str,
    outcome_id: str = "outcome-1",
) -> dict[str, object]:
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
        "summary": "The bounded work order completed.",
        "detail": "AutoCompany supplied the typed field aggregate.",
        "observed_at_ms": 200,
        "evidence_refs": [
            _evidence(
                ref_id="work-order-result-1",
                role="work_order",
                digest_char="b",
                locator=_WORK_ORDER_REF,
            ),
            _evidence(
                ref_id="objective-result-1",
                role="objective_progress",
                digest_char="c",
            ),
        ],
        "execution_outcome": {
            "objective_result": "advanced",
            "metrics": [],
            "currency": "USD",
            "realized_costs": {
                "model_minor": 0,
                "data_minor": 0,
                "human_minor": 0,
                "infrastructure_minor": 0,
                "vendor_minor": 0,
                "incident_response_minor": 0,
                "other_minor": 0,
            },
            "elapsed_ms": 1,
            "blocker_duration_ms": 0,
            "rework_count": 0,
            "incident_count": 0,
            "human_minutes": 0,
            "risk_level": "unassessed",
            "reversibility": "reversible",
        },
    }


async def test_dlaas_explicit_session_context_outcome_and_settlement(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    app = attach_dlaas_routes(create_lifeform_app(vertical=vertical))
    client = await aiohttp_client(app)

    created = await client.post(
        "/dlaas/v1/instances/ai-operations/sessions",
        json={"session_id": "session-1", "end_user_ref": "autocompany-1"},
    )
    assert created.status == 201
    assert (await created.json())["vertical"] == "operations"
    replay = await client.post(
        "/dlaas/v1/instances/ai-operations/sessions",
        json={"session_id": "session-1", "end_user_ref": "autocompany-1"},
    )
    assert replay.status == 200

    context = await client.post(
        "/dlaas/v1/instances/ai-operations/sessions/session-1/operations/context-packs",
        json=_context_payload(),
    )
    assert context.status == 201
    context_body = await context.json()
    assert context_body["session_id"] == "session-1"
    assert context_body["session_lineage_id"].startswith("operations-live-session:")

    outcome = await client.post(
        "/dlaas/v1/instances/ai-operations/sessions/session-1/operations/outcomes",
        json=_outcome_payload(context_pack_id=context_body["context_pack_id"]),
    )
    assert outcome.status == 201
    receipt = await outcome.json()
    assert receipt["work_order_ref"] == _WORK_ORDER_REF
    assert receipt["session_lineage_id"] == context_body["session_lineage_id"]
    assert receipt["learning_route"] == "field_operation_pe_memory_and_execution_result"

    settled = await client.post(
        "/dlaas/v1/instances/ai-operations/sessions/session-1/operations/context-packs",
        json=_context_payload(request_id="request-2", decision_id="decision-2", cycle_id="cycle-2"),
    )
    assert settled.status == 201
    assert receipt["environment_outcome_id"] in (await settled.json())["settled_outcome_ids"]


async def test_dlaas_operations_requires_explicit_session(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    client = await aiohttp_client(
        attach_dlaas_routes(create_lifeform_app(vertical=vertical))
    )
    missing = await client.post(
        "/dlaas/v1/instances/ai-operations/sessions/missing/operations/context-packs",
        json=_context_payload(),
    )
    assert missing.status == 404
    assert (await missing.json())["error"] == "session_not_found"


async def test_same_external_session_id_is_isolated_across_ai_ids(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    manager = InstanceManager(
        vertical_resolver=lambda name: vertical if name == "operations" else None,
        attach_default_mcp_bundle=False,
    )
    await manager.acquire(ai_id="ai-one", runtime_template_id="operations")
    await manager.acquire(ai_id="ai-two", runtime_template_id="operations")
    app = create_lifeform_app(vertical=vertical)
    app[INSTANCE_MANAGER_APP_KEY] = manager
    attach_dlaas_routes(app)
    client = await aiohttp_client(app)

    for ai_id in ("ai-one", "ai-two"):
        response = await client.post(
            f"/dlaas/v1/instances/{ai_id}/sessions",
            json={"session_id": "shared-session", "end_user_ref": ai_id},
        )
        assert response.status == 201

    one = await client.post(
        "/dlaas/v1/instances/ai-one/sessions/shared-session/operations/context-packs",
        json=_context_payload(),
    )
    two = await client.post(
        "/dlaas/v1/instances/ai-two/sessions/shared-session/operations/context-packs",
        json=_context_payload(),
    )
    assert one.status == two.status == 201
    one_body = await one.json()
    two_body = await two.json()
    assert one_body["session_lineage_id"] != two_body["session_lineage_id"]
    assert one_body["context_pack_id"] != two_body["context_pack_id"]

    crossed = await client.post(
        "/dlaas/v1/instances/ai-two/sessions/shared-session/operations/outcomes",
        json=_outcome_payload(context_pack_id=one_body["context_pack_id"]),
    )
    assert crossed.status == 409
    assert (await crossed.json())["error"] == "operations_context_lineage_error"


class _RemoteOperationsLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def forward_interaction(self, *, ai_id, envelope):
        raise AssertionError((ai_id, envelope))

    async def forward_session_create(self, *, ai_id, payload):
        self.calls.append((ai_id, "session"))
        return 201, {
            "status": "created",
            "ai_id": ai_id,
            "session_id": payload["session_id"],
            "created": True,
        }

    async def forward_operations_request(
        self,
        *,
        ai_id,
        session_id,
        operation,
        payload,
    ):
        del payload
        self.calls.append((ai_id, operation))
        return 201, {
            "schema_version": "operations-context-pack.v1",
            "session_id": session_id,
            "routing": "remote",
        }


class _LegacyRemoteLauncher:
    async def forward_interaction(self, *, ai_id, envelope):
        raise AssertionError((ai_id, envelope))


async def test_dlaas_routes_use_multi_pod_forwarding_when_available(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    launcher = _RemoteOperationsLauncher()
    app = create_lifeform_app(vertical=vertical)
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    attach_dlaas_routes(app)
    client = await aiohttp_client(app)

    session = await client.post(
        "/dlaas/v1/instances/ai-remote/sessions",
        json={"session_id": "session-remote"},
    )
    assert session.status == 201
    context = await client.post(
        "/dlaas/v1/instances/ai-remote/sessions/session-remote/operations/context-packs",
        json=_context_payload(),
    )
    assert context.status == 201
    assert (await context.json())["routing"] == "remote"
    assert launcher.calls == [
        ("ai-remote", "session"),
        ("ai-remote", "context-packs"),
    ]


async def test_legacy_multi_pod_launcher_fails_closed_without_operations_capability(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_operations()
    assert vertical is not None
    app = create_lifeform_app(vertical=vertical)
    app[INSTANCE_MANAGER_APP_KEY] = _LegacyRemoteLauncher()
    attach_dlaas_routes(app)
    client = await aiohttp_client(app)

    session = await client.post(
        "/dlaas/v1/instances/ai-legacy/sessions",
        json={"session_id": "session-legacy"},
    )
    assert session.status == 501
    assert (await session.json())["error"] == "pod_session_forwarding_unavailable"

    context = await client.post(
        "/dlaas/v1/instances/ai-legacy/sessions/session-legacy/operations/context-packs",
        json=_context_payload(),
    )
    assert context.status == 501
    assert (await context.json())["error"] == "pod_operations_forwarding_unavailable"
