"""DLaaS API v1 contract tests.

These tests cover the new public API layer without touching kernel
internals:

* `/dlaas/v1/...` aliases preserve the existing interaction dispatch.
* adoption config is frozen into the contract response.
* launcher-owned lifecycle status / sleep / wake works.
* environment / feedback aliases adapt to `InteractionEnvelope`.
* protocol/training intake routes expose SHADOW audit surfaces.
* OpenAI-compatible `/v1/chat/completions` can route by
  `metadata["dlaas.ai_id"]`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import attach_dlaas_routes, build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_api_v1"


@pytest.fixture
async def slice1_client(aiohttp_client):
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=8, idle_eviction_seconds=None)
    attach_dlaas_routes(app, default_ai_id="ai_v1_slice1")
    return await aiohttp_client(app)


async def _build_fullstack_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    return build_dlaas_app(
        db_path=str(tmp_path / "api_v1.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=8,
        idle_eviction_seconds=None,
    )


@pytest.fixture
async def fullstack_client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(await _build_fullstack_app(tmp_path))


async def _adopt_companion(client) -> tuple[dict, dict]:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "API v1 Tenant",
            "contact_email": "ops@api-v1.example",
            "business_type": "education",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    await client.post(
        "/dlaas/shells",
        headers=headers,
        json={
            "shell_id": "web_v1",
            "shell_kind": "deployment",
            "shell_type": "web_chat",
            "display_name": "Web v1",
            "embodiment": {"expression": ["text_streaming"]},
            "channel": {"type": "web"},
        },
    )
    resp = await client.post(
        "/dlaas/assets",
        headers=headers,
        json={
            "asset_type": "persona_kit",
            "title": "Persona",
            "uri": "test:persona.md",
            "mime_type": "text/markdown",
            "language": "zh-CN",
        },
    )
    assert resp.status == 200, await resp.text()
    asset_id = (await resp.json())["asset_id"]
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Companion API v1",
            "domain": "education",
            "description": "Companion template",
            "runtime_template_id": "companion",
            "base_persona": {"language": "zh-CN"},
            "persona_spec": {"display_name": "小鹿"},
            "seed_config": {"domain_seed": "api_v1"},
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]
    resp = await client.post(
        f"/dlaas/templates/{template_id}/assets",
        headers=headers,
        json={"asset_id": asset_id, "role": "persona_source"},
    )
    assert resp.status == 200, await resp.text()
    resp = await client.post(
        f"/dlaas/templates/{template_id}/activate",
        headers=headers,
        json={"seed_config_override": {}},
    )
    assert resp.status == 200, await resp.text()
    resp = await client.patch(
        f"/dlaas/templates/{template_id}",
        headers=headers,
        json={"status": "published", "version_note": "api v1"},
    )
    assert resp.status == 200, await resp.text()
    resp = await client.post(
        "/dlaas/v1/adoptions",
        headers=headers,
        json={
            "tenant_id": tenant["tenant_id"],
            "template_id": template_id,
            "shell_id": "web_v1",
            "owner_user_id": "operator_v1",
            "adoption_config": {
                "vertical": {
                    "vertical_id": "companion",
                    "runtime_template_id": "companion",
                    "profile_id": "default",
                },
                "substrate": {"substrate_profile_id": "synthetic-shared"},
                "protocols": {
                    "autoload": ["growth_advisor:cheng-laoshi"],
                    "library_ids": ["customer:no-hard-sell-v1"],
                },
                "training": {"allow_adapter_training": False},
            },
        },
    )
    assert resp.status == 200, await resp.text()
    return headers, await resp.json()


async def test_v1_interaction_alias_dispatches(slice1_client):
    resp = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/interactions",
        json={
            "contract_id": "ctr_v1",
            "session_id": "sess_v1",
            "end_user_ref": "user_v1",
            "interaction_type": "chat",
            "human_brief": "你好",
        },
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["status"] == "ok"
    assert body["ai_id"] == "ai_v1_slice1"


async def test_feedback_and_environment_aliases(slice1_client):
    feedback = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/feedback",
        json={
            "contract_id": "ctr_v1",
            "session_id": "sess_feedback",
            "end_user_ref": "user_v1",
            "feedback": {
                "valence": "felt_heard",
                "target_response_id": "resp_1",
                "intensity": 0.9,
                "evidence": "clicked helpful",
            },
        },
    )
    assert feedback.status == 200, await feedback.text()
    assert (await feedback.json())["feedback"]["outcome_kind"] == "felt_heard"

    outcome = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/environment/outcomes",
        json={
            "contract_id": "ctr_v1",
            "session_id": "sess_feedback",
            "end_user_ref": "user_v1",
            "event_id": "tool_evt_1",
            "tool_name": "wechat.send_message",
            "action_id": "act_1",
            "status": "ok",
            "summary": "delivered",
            "detail": "message delivered",
        },
    )
    assert outcome.status == 200, await outcome.text()
    body = await outcome.json()
    assert body["observation_type"] == "tool_result"


async def test_asset_intake_storage_only_stores_without_kernel_turn(slice1_client):
    resp = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/assets/intake",
        json={
            "contract_id": "ctr_asset",
            "session_id": "sess_asset_store",
            "end_user_ref": "operator",
            "intake_intent": "storage_only",
            "media_kind": "image",
            "title": "whiteboard photo",
            "mime_type": "image/png",
            "source_ref": "upload://whiteboard.png",
            "content_base64": "aW1hZ2UtYnl0ZXM=",
        },
    )

    assert resp.status == 201, await resp.text()
    body = await resp.json()
    assert body["asset"]["resolved_intent"] == "storage_only"
    assert body["decision"]["intent"] == "storage_only"
    asset_id = body["asset"]["asset_id"]

    fetched = await slice1_client.get(
        f"/dlaas/v1/instances/ai_v1_slice1/assets/intake/{asset_id}"
    )
    assert fetched.status == 200, await fetched.text()
    assert (await fetched.json())["asset"]["asset_id"] == asset_id


async def test_asset_intake_simple_ingest_text(slice1_client):
    resp = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/assets/intake",
        json={
            "contract_id": "ctr_asset",
            "session_id": "sess_asset_ingest",
            "end_user_ref": "operator",
            "intake_intent": "simple_ingest",
            "media_kind": "text",
            "title": "short note",
            "text": "This is a reviewed knowledge note for ingestion.",
        },
    )

    assert resp.status == 201, await resp.text()
    body = await resp.json()
    assert body["asset"]["resolved_intent"] == "simple_ingest"
    assert body["ingestion_report"]["processed_chunks"] >= 1
    assert body["ingestion_report"]["all_succeeded"] is True


async def test_debug_registration_ingest_and_analysis(slice1_client):
    app_resp = await slice1_client.post(
        "/dlaas/v1/debug/apps",
        json={
            "app_id": "repair30",
            "display_name": "repair30",
            "allowed_ai_ids": ["ai_v1_slice1"],
            "allowed_event_types": ["chat.completed"],
        },
    )
    assert app_resp.status == 201, await app_resp.text()

    schema_resp = await slice1_client.post(
        "/dlaas/v1/debug/apps/repair30/schemas",
        json={
            "schema_version": "repair30.debug.v1",
            "event_types": ["chat.completed"],
            "fields": [
                {
                    "name": "status",
                    "type": "number",
                    "meaning": "HTTP status returned by DLaaS.",
                    "owner": "dlaas",
                    "privacy_level": "internal",
                    "required": True,
                },
                {
                    "name": "ok",
                    "type": "boolean",
                    "meaning": "Whether the upstream call succeeded.",
                    "owner": "dlaas",
                    "privacy_level": "internal",
                    "required": True,
                },
            ],
        },
    )
    assert schema_resp.status == 201, await schema_resp.text()

    bad_event = await slice1_client.post(
        "/dlaas/v1/debug/events",
        json={
            "app_id": "repair30",
            "schema_version": "repair30.debug.v1",
            "ai_id": "ai_v1_slice1",
            "session_id": "sess_debug",
            "event_type": "chat.completed",
            "stage": "dlaas.chat",
            "fields": {"status": 200, "ok": True, "extra": "blocked"},
        },
    )
    assert bad_event.status == 400, await bad_event.text()

    event_resp = await slice1_client.post(
        "/dlaas/v1/debug/events",
        json={
            "app_id": "repair30",
            "schema_version": "repair30.debug.v1",
            "ai_id": "ai_v1_slice1",
            "session_id": "sess_debug",
            "event_type": "chat.completed",
            "stage": "dlaas.chat",
            "fields": {"status": 200, "ok": True},
        },
    )
    assert event_resp.status == 201, await event_resp.text()
    event = await event_resp.json()
    assert event["debug_event_id"].startswith("debug_evt_")

    listed = await slice1_client.get(
        "/dlaas/v1/debug/events?app_id=repair30&session_id=sess_debug"
    )
    assert listed.status == 200, await listed.text()
    assert (await listed.json())["events"][0]["debug_event_id"] == event["debug_event_id"]

    analysis_resp = await slice1_client.post(
        "/dlaas/v1/debug/analysis",
        json={
            "prompt": "Summarize debug evidence for this repair30 session.",
            "selectors": {
                "app_id": "repair30",
                "ai_id": "ai_v1_slice1",
                "session_id": "sess_debug",
            },
            "include_readouts": False,
            "include_explain": False,
            "include_audit": True,
        },
    )
    assert analysis_resp.status == 201, await analysis_resp.text()
    analysis = await analysis_resp.json()
    assert analysis["artifact_id"].startswith("debug_artifact_")
    assert analysis["evidence"]["debug_event_count"] == 1
    assert analysis["recommendations"]


async def test_asset_intake_deep_read_creates_training_job(slice1_client):
    resp = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/assets/intake",
        json={
            "contract_id": "ctr_asset",
            "session_id": "sess_asset_deep",
            "end_user_ref": "operator",
            "intake_intent": "deep_read",
            "media_kind": "markdown",
            "title": "book chapter",
            "text": "# Chapter\n\nLong material to read deeply later.",
        },
    )

    assert resp.status == 202, await resp.text()
    body = await resp.json()
    assert body["asset"]["resolved_intent"] == "deep_read"
    assert body["training_job"]["job_type"] == "corpus_ingestion"


async def test_adoption_config_and_lifecycle(fullstack_client):
    _, adopted = await _adopt_companion(fullstack_client)
    ai_id = adopted["ai_id"]
    assert adopted["instance_endpoint"].startswith("/dlaas/instances/")
    assert adopted["v1_instance_endpoint"].startswith("/dlaas/v1/instances/")
    assert adopted["adoption_config_version"] == 1
    assert adopted["resolved"]["vertical"] == "companion"
    assert adopted["resolved"]["protocol_ids"] == [
        "growth_advisor:cheng-laoshi",
        "customer:no-hard-sell-v1",
    ]

    status = await fullstack_client.get(f"/dlaas/v1/instances/{ai_id}/status")
    assert status.status == 200, await status.text()
    assert (await status.json())["lifecycle_state"] == "awake"

    sleep = await fullstack_client.post(
        f"/dlaas/v1/instances/{ai_id}/sleep",
        json={"reason": "test_sleep"},
    )
    assert sleep.status == 200, await sleep.text()
    assert (await sleep.json())["lifecycle_state"] == "asleep"

    wake = await fullstack_client.post(
        f"/dlaas/v1/instances/{ai_id}/wake",
        json={"reason": "test_wake"},
    )
    assert wake.status == 200, await wake.text()
    assert (await wake.json())["lifecycle_state"] == "awake"


async def test_protocol_and_training_intake_shadow_routes(slice1_client):
    created = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/protocols/submissions",
        json={
            "contract_id": "ctr_v1",
            "source_type": "json_payload",
            "submitted_by": "operator",
            "candidate_protocol_id": "customer:test-protocol",
        },
    )
    assert created.status == 201, await created.text()
    submission_id = (await created.json())["submission_id"]
    approved = await slice1_client.post(
        f"/dlaas/v1/instances/ai_v1_slice1/protocols/submissions/{submission_id}/approve"
    )
    assert approved.status == 200, await approved.text()
    library = await slice1_client.get(
        "/dlaas/v1/instances/ai_v1_slice1/protocols/library"
    )
    assert library.status == 200
    assert (await library.json())["count"] == 1
    loaded = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/protocols/library/customer:test-protocol/load"
    )
    assert loaded.status == 200, await loaded.text()
    assert (await loaded.json())["loaded"] is True

    job = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/training/jobs",
        json={
            "contract_id": "ctr_v1",
            "job_type": "adapter_candidate",
            "created_by": "operator",
        },
    )
    assert job.status == 201, await job.text()
    job_id = (await job.json())["job_id"]
    promote = await slice1_client.post(
        f"/dlaas/v1/instances/ai_v1_slice1/training/jobs/{job_id}/promote"
    )
    assert promote.status == 409
    assert (await promote.json())["error"] == "promotion_gate_required"


async def test_readouts_explain_and_admin_snapshots(fullstack_client):
    headers, adopted = await _adopt_companion(fullstack_client)
    ai_id = adopted["ai_id"]
    resp = await fullstack_client.post(
        f"/dlaas/v1/instances/{ai_id}/interactions",
        json={
            "contract_id": adopted["contract_id"],
            "session_id": "sess_observe",
            "end_user_ref": "user_obs",
            "interaction_type": "chat",
            "human_brief": "你好，帮我整理一下今天的状态。",
        },
    )
    assert resp.status == 200, await resp.text()

    readouts = await fullstack_client.get(
        f"/dlaas/v1/instances/{ai_id}/readouts?session_id=sess_observe"
    )
    assert readouts.status == 200, await readouts.text()
    readout_body = await readouts.json()
    assert readout_body["view"] == "summary"
    assert "cognition" in readout_body
    assert "protocol" in readout_body
    assert "safety" in readout_body

    explain = await fullstack_client.get(
        f"/dlaas/v1/instances/{ai_id}/explain?session_id=sess_observe"
    )
    assert explain.status == 200, await explain.text()
    chain = (await explain.json())["chain"]
    assert [step["step"] for step in chain] == [
        "input_event",
        "regime",
        "protocol",
        "boundary",
        "strategy",
        "knowledge",
        "response",
        "prediction_error",
    ]

    denied = await fullstack_client.get(
        f"/dlaas/v1/admin/instances/{ai_id}/snapshots?session_id=sess_observe"
    )
    assert denied.status == 403

    raw = await fullstack_client.get(
        f"/dlaas/v1/admin/instances/{ai_id}/snapshots"
        "?session_id=sess_observe&slot=response_assembly",
        headers={"X-Control-Plane-Secret": CONTROL_PLANE_SECRET},
    )
    assert raw.status == 200, await raw.text()
    raw_body = await raw.json()
    assert "response_assembly" in raw_body["active"]
    assert raw_body["active"]["response_assembly"]["slot_name"] == "response_assembly"


async def test_safety_protocol_alias_requires_boundary_and_loads(slice1_client):
    invalid = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/safety/protocols",
        json={
            "contract_id": "ctr_v1",
            "source_type": "json_payload",
            "candidate_protocol_id": "customer:no-boundary",
        },
    )
    assert invalid.status == 400
    assert (await invalid.json())["error"] == "missing_boundary_contracts"

    created = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/safety/protocols",
        json={
            "contract_id": "ctr_v1",
            "source_type": "json_payload",
            "candidate_protocol_id": "customer:safety-boundary",
            "protocol": {
                "boundaries": [
                    {
                        "boundary_id": "bd:safety:no-overclaim",
                        "description": "No overclaim",
                        "trigger_reasons": ["overclaim risk"],
                    }
                ]
            },
        },
    )
    assert created.status == 201, await created.text()
    submission_id = (await created.json())["submission_id"]
    approved = await slice1_client.post(
        f"/dlaas/v1/instances/ai_v1_slice1/safety/protocols/{submission_id}/approve"
    )
    assert approved.status == 200, await approved.text()
    loaded = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/safety/protocols/customer:safety-boundary/load"
    )
    assert loaded.status == 200, await loaded.text()
    assert (await loaded.json())["loaded"] is True


async def test_catalog_and_blueprint_adoption(fullstack_client):
    blueprints = await fullstack_client.get("/dlaas/v1/catalog/blueprints")
    assert blueprints.status == 200, await blueprints.text()
    listed = await blueprints.json()
    ids = {b["blueprint_id"] for b in listed["blueprints"]}
    assert "companion/default/dev-v1" in ids

    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await fullstack_client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Blueprint Tenant",
            "contact_email": "ops@blueprint.example",
            "business_type": "education",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    await fullstack_client.post(
        "/dlaas/shells",
        headers=headers,
        json={
            "shell_id": "blueprint_web",
            "shell_kind": "deployment",
            "shell_type": "web_chat",
            "display_name": "Blueprint Web",
            "embodiment": {"expression": ["text_streaming"]},
            "channel": {"type": "web"},
        },
    )
    asset = await fullstack_client.post(
        "/dlaas/assets",
        headers=headers,
        json={
            "asset_type": "persona_kit",
            "title": "Persona",
            "uri": "test:blueprint.md",
            "mime_type": "text/markdown",
            "language": "zh-CN",
        },
    )
    assert asset.status == 200, await asset.text()
    asset_id = (await asset.json())["asset_id"]
    template = await fullstack_client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Blueprint Template",
            "domain": "education",
            "description": "Template",
            "runtime_template_id": "companion",
            "base_persona": {"language": "zh-CN"},
            "persona_spec": {"display_name": "小鹿"},
            "seed_config": {"domain_seed": "blueprint"},
        },
    )
    assert template.status == 200, await template.text()
    template_id = (await template.json())["template_id"]
    await fullstack_client.post(
        f"/dlaas/templates/{template_id}/assets",
        headers=headers,
        json={"asset_id": asset_id, "role": "persona_source"},
    )
    activate = await fullstack_client.post(
        f"/dlaas/templates/{template_id}/activate",
        headers=headers,
        json={"seed_config_override": {}},
    )
    assert activate.status == 200, await activate.text()
    publish = await fullstack_client.patch(
        f"/dlaas/templates/{template_id}",
        headers=headers,
        json={"status": "published", "version_note": "blueprint"},
    )
    assert publish.status == 200, await publish.text()
    adopted = await fullstack_client.post(
        "/dlaas/v1/adoptions",
        headers=headers,
        json={
            "tenant_id": tenant["tenant_id"],
            "template_id": template_id,
            "shell_id": "blueprint_web",
            "blueprint_id": "companion/default/dev-v1",
        },
    )
    assert adopted.status == 200, await adopted.text()
    payload = await adopted.json()
    assert payload["resolved"]["vertical"] == "companion"


async def test_governance_data_audit_usage_and_consent(slice1_client):
    export = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/data/export",
        json={
            "contract_id": "ctr_gov",
            "end_user_ref": "user_gov",
        },
    )
    assert export.status == 201, await export.text()
    export_body = await export.json()
    assert export_body["status"] == "completed"
    assert export_body["artifact_ref"].startswith("artifact://data-export/")

    delete = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/data/delete",
        json={
            "contract_id": "ctr_gov",
            "end_user_ref": "user_gov",
            "scopes": ["platform"],
        },
    )
    assert delete.status == 202, await delete.text()
    delete_body = await delete.json()
    status = await slice1_client.get(
        "/dlaas/v1/instances/ai_v1_slice1/data/deletion-status/"
        + delete_body["job_id"]
    )
    assert status.status == 200, await status.text()
    assert (await status.json())["deleted_scopes"] == ["platform"]

    consent = await slice1_client.post(
        "/dlaas/v1/consents",
        json={
            "ai_id": "ai_v1_slice1",
            "end_user_ref": "user_gov",
            "consent_type": "training_use",
            "granted": True,
            "evidence_ref": "click:consent",
        },
    )
    assert consent.status == 201, await consent.text()
    listed = await slice1_client.get(
        "/dlaas/v1/consents/user_gov?ai_id=ai_v1_slice1"
    )
    assert listed.status == 200
    assert (await listed.json())["consents"][0]["consent_type"] == "training_use"

    policies = await slice1_client.get("/dlaas/v1/policies")
    assert policies.status == 200
    assert {p["policy_id"] for p in (await policies.json())["policies"]} >= {
        "default-data-governance",
        "default-training-consent",
    }

    audit = await slice1_client.get("/dlaas/v1/audit/events?ai_id=ai_v1_slice1")
    assert audit.status == 200
    event_types = {e["event_type"] for e in (await audit.json())["events"]}
    assert "data_export_requested" in event_types
    assert "data_delete_requested" in event_types
    assert "consent_recorded" in event_types

    usage = await slice1_client.get("/dlaas/v1/usage?ai_id=ai_v1_slice1")
    assert usage.status == 200
    usage_metrics = {r["metric"] for r in (await usage.json())["usage"]}
    assert {"data_export", "data_delete"} <= usage_metrics
    quota = await slice1_client.get("/dlaas/v1/quota?tenant_id=tenant_test")
    assert quota.status == 200
    assert (await quota.json())["tenant_id"] == "tenant_test"
    billing = await slice1_client.get("/dlaas/v1/billing/events")
    assert billing.status == 200
    assert len((await billing.json())["billing_events"]) >= 2


async def test_artifact_eval_and_webhook_surfaces(slice1_client):
    job = await slice1_client.post(
        "/dlaas/v1/instances/ai_v1_slice1/training/jobs",
        json={
            "contract_id": "ctr_artifact",
            "job_type": "adapter_candidate",
            "source_ref": "train://candidate",
        },
    )
    assert job.status == 201, await job.text()
    job_id = (await job.json())["job_id"]
    artifact_id = f"artifact:{job_id}"
    artifacts = await slice1_client.get("/dlaas/v1/artifacts")
    assert artifacts.status == 200
    assert artifact_id in {a["artifact_id"] for a in (await artifacts.json())["artifacts"]}

    blocked = await slice1_client.post(f"/dlaas/v1/artifacts/{artifact_id}/promote")
    assert blocked.status == 409
    assert (await blocked.json())["error"] == "promotion_gate_required"
    promoted = await slice1_client.post(
        f"/dlaas/v1/artifacts/{artifact_id}/promote",
        json={"gate_evidence": {"offline_gate": "pass"}},
    )
    assert promoted.status == 200, await promoted.text()
    assert (await promoted.json())["promotion_decision"] == "allow"

    eval_run = await slice1_client.post(
        "/dlaas/v1/eval/runs",
        json={
            "gate_id": "boundary-baseline",
            "ai_id": "ai_v1_slice1",
            "contract_id": "ctr_eval",
            "score": 0.9,
        },
    )
    assert eval_run.status == 201, await eval_run.text()
    run_id = (await eval_run.json())["run_id"]
    approved = await slice1_client.post(f"/dlaas/v1/eval/runs/{run_id}/approve")
    assert approved.status == 200, await approved.text()
    assert (await approved.json())["status"] == "approved"

    webhook = await slice1_client.post(
        "/dlaas/v1/webhooks",
        json={
            "target_url": "https://example.invalid/hook",
            "event_types": ["eval_run_approved"],
        },
    )
    assert webhook.status == 201, await webhook.text()
    assert (await webhook.json())["enabled"] is True
    stream = await slice1_client.get("/dlaas/v1/events/stream")
    assert stream.status == 200
    stream_types = {e["event_type"] for e in (await stream.json())["events"]}
    assert "eval_run_approved" in stream_types
    assert "webhook_created" in stream_types


async def test_governance_records_persist_across_fullstack_restart(
    aiohttp_client, tmp_path: Path
):
    from lifeform_service.verticals import discover_verticals

    db_path = tmp_path / "governance.sqlite"
    spec = discover_verticals()["companion"]

    app1 = build_dlaas_app(
        db_path=str(db_path),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=8,
        idle_eviction_seconds=None,
    )
    c1 = await aiohttp_client(app1)
    export = await c1.post(
        "/dlaas/v1/instances/ai_persist/data/export",
        json={"contract_id": "ctr_persist", "end_user_ref": "user_persist"},
    )
    assert export.status == 201, await export.text()
    consent = await c1.post(
        "/dlaas/v1/consents",
        json={
            "ai_id": "ai_persist",
            "end_user_ref": "user_persist",
            "consent_type": "training_use",
            "granted": True,
        },
    )
    assert consent.status == 201, await consent.text()
    job = await c1.post(
        "/dlaas/v1/instances/ai_persist/training/jobs",
        json={
            "contract_id": "ctr_persist",
            "job_type": "adapter_candidate",
            "source_ref": "train://persist",
        },
    )
    assert job.status == 201, await job.text()
    job_id = (await job.json())["job_id"]
    artifact_id = f"artifact:{job_id}"

    app2 = build_dlaas_app(
        db_path=str(db_path),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=8,
        idle_eviction_seconds=None,
    )
    c2 = await aiohttp_client(app2)
    audit = await c2.get("/dlaas/v1/audit/events?ai_id=ai_persist")
    assert audit.status == 200, await audit.text()
    event_types = {e["event_type"] for e in (await audit.json())["events"]}
    assert "data_export_requested" in event_types
    assert "consent_recorded" in event_types
    assert "training_job_created" in event_types

    persisted_consent = await c2.get(
        "/dlaas/v1/consents/user_persist?ai_id=ai_persist"
    )
    assert persisted_consent.status == 200, await persisted_consent.text()
    assert (await persisted_consent.json())["consents"][0]["consent_type"] == "training_use"

    artifacts = await c2.get("/dlaas/v1/artifacts")
    assert artifacts.status == 200, await artifacts.text()
    assert artifact_id in {a["artifact_id"] for a in (await artifacts.json())["artifacts"]}

    usage = await c2.get("/dlaas/v1/usage?ai_id=ai_persist")
    assert usage.status == 200, await usage.text()
    assert {r["metric"] for r in (await usage.json())["usage"]} >= {
        "data_export",
        "training_job",
    }


async def test_openai_compat_routes_by_dlaas_ai_id(aiohttp_client, tmp_path: Path):
    from lifeform_openai_compat import add_openai_routes

    app = await _build_fullstack_app(tmp_path)
    add_openai_routes(app)
    client = await aiohttp_client(app)
    _, adopted = await _adopt_companion(client)
    ai_id = adopted["ai_id"]
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "companion",
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {
                "session_id": "openai-dlaas-session",
                "dlaas.ai_id": ai_id,
            },
        },
    )
    assert resp.status == 200, await resp.text()
    assert resp.headers["x-lifeform-mode"] == "lifeform"
    body = await resp.json()
    assert body["id"] == "openai-dlaas-session"
