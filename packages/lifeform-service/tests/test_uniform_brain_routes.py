from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from lifeform_service.app import create_app
from lifeform_service.verticals import _try_coding, _try_operations, _try_venture


def _coding_payload() -> dict[str, object]:
    return {
        "request_id": "coding-uniform-request",
        "project_id": "project-1",
        "repository_id": "repo-1",
        "task_id": "task-1",
        "task_kind": "bugfix",
        "task_summary": "Verify the shared vertical Brain route",
        "repository_revision": "abc123",
        "target_paths": ["src/state.py"],
    }


def _venture_payload() -> dict[str, object]:
    return {
        "schema_version": "venture-context-request.v1",
        "request_id": "venture-uniform-request",
        "portfolio_id": "portfolio-1",
        "cycle_id": "cycle-1",
        "venture_id": "venture-1",
        "decision_id": "decision-1",
        "decision_point": "experiment_planning",
        "confirmed_facts": [
            {
                "fact_id": "fact-1",
                "kind": "demand_signal",
                "statement": "The owner published a typed demand fact.",
                "evidence_ref_ids": ["demand-1"],
                "as_of_ms": 100,
            }
        ],
        "constraints": [
            {
                "constraint_id": "constraint-1",
                "kind": "budget",
                "description": "Use the approved reversible test budget.",
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
        "uncertainties": [],
        "evidence_refs": [
            {
                "ref_id": "demand-1",
                "evidence_class": "field",
                "role": "demand_signal",
                "locator": "foundry://evidence/demand-1",
                "content_sha256": "a" * 64,
                "observed_at_ms": 100,
            }
        ],
    }


def _operations_payload() -> dict[str, object]:
    return {
        "schema_version": "operations-context-request.v1",
        "request_id": "operations-uniform-request",
        "company_id": "company-1",
        "cycle_id": "cycle-1",
        "workstream_id": "reliability",
        "decision_id": "decision-1",
        "decision_point": "work_prioritization",
        "division_ids": ["division-engineering"],
        "action_catalog_ids": ["catalog:repair-service"],
        "confirmed_facts": [
            {
                "fact_id": "fact-1",
                "kind": "division_health",
                "division_id": "division-engineering",
                "statement": "The owner published a typed health fact.",
                "evidence_ref_ids": ["operating-signal-1"],
                "as_of_ms": 100,
            }
        ],
        "constraints": [
            {
                "constraint_id": "constraint-1",
                "kind": "budget",
                "division_id": "",
                "description": "Use the approved operating budget.",
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
            {
                "ref_id": "operating-signal-1",
                "evidence_class": "field",
                "role": "operating_signal",
                "locator": "autocompany://evidence/operating-signal-1",
                "content_sha256": "a" * 64,
                "observed_at_ms": 100,
            }
        ],
    }


async def _exercise_uniform_context_route(
    *,
    vertical,
    session_id: str,
    payload: dict[str, object],
    expected_brain: str,
    expected_schema: str,
) -> None:
    client = TestClient(TestServer(create_app(vertical=vertical)))
    await client.start_server()
    try:
        created = await client.post(
            "/v1/sessions",
            json={"session_id": session_id},
        )
        assert created.status == 201
        response = await client.post(
            f"/v1/sessions/{session_id}/brain/context-packs",
            json=payload,
        )
        assert response.status == 201
        assert response.headers["X-Volvence-Brain"] == expected_brain
        assert (await response.json())["schema_version"] == expected_schema
    finally:
        await client.close()


async def test_all_product_brains_share_one_session_api(monkeypatch) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    coding = _try_coding()
    venture = _try_venture()
    operations = _try_operations()
    assert coding is not None
    assert venture is not None
    assert operations is not None

    await _exercise_uniform_context_route(
        vertical=coding,
        session_id="coding-uniform",
        payload=_coding_payload(),
        expected_brain="coding",
        expected_schema="coding-context-pack.v1",
    )
    await _exercise_uniform_context_route(
        vertical=venture,
        session_id="venture-uniform",
        payload=_venture_payload(),
        expected_brain="venture",
        expected_schema="venture-context-pack.v1",
    )
    await _exercise_uniform_context_route(
        vertical=operations,
        session_id="operations-uniform",
        payload=_operations_payload(),
        expected_brain="operations",
        expected_schema="operations-context-pack.v1",
    )

    client = TestClient(TestServer(create_app(vertical=coding)))
    await client.start_server()
    try:
        discovery = await client.get("/v1/brains")
        assert discovery.status == 200
        manifests = {
            item["name"]: item
            for item in (await discovery.json())["brains"]
        }
        names = set(manifests)
        assert names == {"coding", "venture", "operations"}
        assert all(
            item["capabilities"]["shared_lifeform_kernel"]
            for item in manifests.values()
        )
        assert manifests["coding"]["capabilities"]["steerable"]["status"] == "shadow"
        assert manifests["venture"]["capabilities"]["shared_bounded_policy"] is False
        assert manifests["operations"]["capabilities"]["shared_bounded_policy"] is True
        assert (
            manifests["operations"]["capabilities"]["maximum_advice_scope"]
            == "staging_active"
        )
    finally:
        await client.close()


async def test_uniform_outcome_route_preserves_domain_contract(monkeypatch) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    coding = _try_coding()
    assert coding is not None
    client = TestClient(TestServer(create_app(vertical=coding)))
    await client.start_server()
    try:
        await client.post(
            "/v1/sessions",
            json={"session_id": "coding-uniform-outcome"},
        )
        context = await client.post(
            "/v1/sessions/coding-uniform-outcome/brain/context-packs",
            json=_coding_payload(),
        )
        context_body = await context.json()
        outcome_payload = {
            "outcome_id": "coding-uniform-outcome-1",
            "context_pack_id": context_body["context_pack_id"],
            "kind": "task_regressed",
            "source": "ci",
            "summary": "The deterministic check found a regression.",
            "detail": "The owner supplied the typed CI result.",
            "observed_at_ms": 1_000,
            "evidence_ref": "ci:run-uniform-1",
            "changed_paths": ["src/state.py"],
        }
        outcome = await client.post(
            "/v1/sessions/coding-uniform-outcome/brain/outcomes",
            json=outcome_payload,
        )
        assert outcome.status == 201
        assert outcome.headers["X-Volvence-Brain"] == "coding"
        assert (await outcome.json())["schema_version"] == "coding-outcome-receipt.v1"

        replay = await client.post(
            "/v1/sessions/coding-uniform-outcome/brain/outcomes",
            json=outcome_payload,
        )
        assert replay.status == 200
    finally:
        await client.close()
