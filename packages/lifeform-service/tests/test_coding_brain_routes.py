from __future__ import annotations

from lifeform_domain_coding import build_coding_lifeform
from lifeform_service.alpha import AlphaServiceConfig
from lifeform_service.app import create_app
from lifeform_service.verticals import VerticalSpec, _try_coding


def _context_payload(
    *,
    request_id: str = "request-1",
    task_id: str = "task-1",
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "project_id": "project-1",
        "repository_id": "repo-1",
        "task_id": task_id,
        "task_kind": "bugfix",
        "task_summary": "Fix state restoration after checkpoint failure",
        "repository_revision": "abc123",
        "target_paths": ["src/state.py", "tests/test_state.py"],
    }


async def test_http_context_outcome_and_next_turn_pe_closure(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_coding()
    assert vertical is not None
    client = await aiohttp_client(create_app(vertical=vertical))
    created_session = await client.post(
        "/v1/sessions",
        json={"session_id": "coding-http"},
    )
    assert created_session.status == 201

    first = await client.post(
        "/v1/sessions/coding-http/coding/context-packs",
        json=_context_payload(),
    )
    assert first.status == 201
    first_body = await first.json()
    assert first_body["wiring_level"] == "active"
    assert first_body["advice"]["wiring_level"] == "shadow"
    assert first_body["advice"]["applied"] is False

    replayed = await client.post(
        "/v1/sessions/coding-http/coding/context-packs",
        json=_context_payload(),
    )
    assert replayed.status == 200
    assert (await replayed.json())["context_pack_id"] == first_body["context_pack_id"]

    outcome_payload = {
        "outcome_id": "outcome-1",
        "context_pack_id": first_body["context_pack_id"],
        "kind": "task_regressed",
        "source": "ci",
        "summary": "Checkpoint regression",
        "detail": "test_restore expected the previously committed state",
        "observed_at_ms": 1_000,
        "evidence_ref": "ci:run-42",
        "changed_paths": ["src/state.py"],
    }
    outcome = await client.post(
        "/v1/sessions/coding-http/coding/outcomes",
        json=outcome_payload,
    )
    assert outcome.status == 201
    receipt = await outcome.json()
    assert receipt["learning_route"] == "dialogue_external_outcome"
    assert receipt["external_outcome_evidence_id"]

    outcome_replay = await client.post(
        "/v1/sessions/coding-http/coding/outcomes",
        json=outcome_payload,
    )
    assert outcome_replay.status == 200
    assert (await outcome_replay.json())["receipt_id"] == receipt["receipt_id"]

    second = await client.post(
        "/v1/sessions/coding-http/coding/context-packs",
        json=_context_payload(request_id="request-2", task_id="task-2"),
    )
    assert second.status == 201
    second_body = await second.json()
    assert receipt["external_outcome_evidence_id"] in second_body[
        "settled_outcome_evidence_refs"
    ]
    assert receipt["memory_entry_id"] in second_body["source_entry_ids"]
    assert "previously committed state" in second_body["rendered_context"]


async def test_http_contract_conflicts_and_typed_source_fail_closed(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_coding()
    assert vertical is not None
    client = await aiohttp_client(create_app(vertical=vertical))
    await client.post("/v1/sessions", json={"session_id": "coding-errors"})
    first = await client.post(
        "/v1/sessions/coding-errors/coding/context-packs",
        json=_context_payload(),
    )
    context_pack_id = (await first.json())["context_pack_id"]

    conflict = await client.post(
        "/v1/sessions/coding-errors/coding/context-packs",
        json={**_context_payload(), "task_summary": "Different immutable payload"},
    )
    assert conflict.status == 409
    assert (await conflict.json())["error"] == "coding_idempotency_conflict"

    invalid_pair = await client.post(
        "/v1/sessions/coding-errors/coding/outcomes",
        json={
            "outcome_id": "invalid-pair",
            "context_pack_id": context_pack_id,
            "kind": "review_approved",
            "source": "ci",
            "summary": "Invalid source",
            "detail": "A review verdict cannot originate from CI",
            "observed_at_ms": 2_000,
            "evidence_ref": "ci:run-1",
        },
    )
    assert invalid_pair.status == 400
    assert (await invalid_pair.json())["error"] == "invalid_coding_outcome"

    unknown_pack = await client.post(
        "/v1/sessions/coding-errors/coding/outcomes",
        json={
            "outcome_id": "unknown-pack",
            "context_pack_id": "coding-context-pack:" + "a" * 64,
            "kind": "merged",
            "source": "vcs",
            "summary": "Merged",
            "detail": "Merged to main",
            "observed_at_ms": 2_100,
            "evidence_ref": "git:abc123",
        },
    )
    assert unknown_pack.status == 409
    assert (await unknown_pack.json())["error"] == "coding_context_lineage_error"


async def test_http_rejects_non_coding_vertical(
    aiohttp_client,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    other = VerticalSpec(
        name="other",
        factory=lambda runtime: build_coding_lifeform(substrate_runtime=runtime),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )
    client = await aiohttp_client(create_app(vertical=other))
    await client.post("/v1/sessions", json={"session_id": "other-session"})
    response = await client.post(
        "/v1/sessions/other-session/coding/context-packs",
        json=_context_payload(),
    )
    assert response.status == 409
    assert (await response.json())["error"] == "coding_vertical_required"


async def test_alpha_coding_factory_restores_memory_across_http_sessions(
    aiohttp_client,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_coding()
    assert vertical is not None
    assert vertical.alpha_factory is not None
    app = create_app(
        vertical=vertical,
        alpha_config=AlphaServiceConfig(
            enabled=True,
            memory_scope_root_dir=str(tmp_path),
            alpha_users=frozenset({"coder-1"}),
        ),
    )
    client = await aiohttp_client(app)
    create_first = await client.post(
        "/v1/sessions",
        json={"session_id": "alpha-a", "user_id": "coder-1"},
    )
    assert create_first.status == 201
    first_context = await client.post(
        "/v1/sessions/alpha-a/coding/context-packs",
        json=_context_payload(),
    )
    first_body = await first_context.json()
    outcome = await client.post(
        "/v1/sessions/alpha-a/coding/outcomes",
        json={
            "outcome_id": "persistent-http-outcome",
            "context_pack_id": first_body["context_pack_id"],
            "kind": "task_verified",
            "source": "build_gate",
            "summary": "Restart verified",
            "detail": "Build gate restored the committed checkpoint",
            "observed_at_ms": 3_000,
            "evidence_ref": "build:run-9",
        },
    )
    receipt = await outcome.json()
    assert receipt["memory_persisted"] is True
    closed = await client.delete("/v1/sessions/alpha-a")
    assert closed.status == 200

    create_second = await client.post(
        "/v1/sessions",
        json={"session_id": "alpha-b", "user_id": "coder-1"},
    )
    assert create_second.status == 201
    second_context = await client.post(
        "/v1/sessions/alpha-b/coding/context-packs",
        json=_context_payload(request_id="request-b", task_id="task-b"),
    )
    second_body = await second_context.json()
    assert receipt["memory_entry_id"] in second_body["source_entry_ids"]
    assert "restored the committed checkpoint" in second_body["rendered_context"]
