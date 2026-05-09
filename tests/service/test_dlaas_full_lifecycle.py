"""Slice 7.3 — full DLaaS lifecycle e2e.

This is the demo-target test: one client walks the full happy path
from tenant bootstrap to live runtime + ops + eval gate, exercising
every Slice (1 → 6) end-to-end:

1. Slice 3.1 — register tenant via control-plane secret
2. Slice 3.2 — declare deployment shell + create asset
3. Slice 3.3 — create template (runtime_template_id = ``companion``)
4. Slice 3.4 — activate (drives kernel ingestion + drain) + readiness
5. Slice 3.3 — publish via PATCH (status: published)
6. Slice 6.2 — exam questions + run, finalised with operator AI
   responses (Slice 6 LLM judge plug-in is fail-closed by default)
7. Slice 6.3 — license/evaluate + signoff
8. Slice 3.5 — adopt → ai_id + register focus_persons
9. Slice 4.2 — identity_links (canonical_end_user_ref ↔ channel_ref)
10. Slice 1 / 2 — interactions: chat, feedback, observe, command
11. Slice 5.1 — pause via admin → next chat returns operator_takeover
12. Slice 5.1 — operator-message + resume → chat resumes
13. Slice 5.2 — handoff_tickets CRUD + human_reply

The kernel runs the synthetic ``companion`` vertical without a real
substrate runtime, so every ``run_turn`` is sub-second. The intent
is to lock down the wiring contract; latency / throughput is out of
scope here (Slice 7.4 covers smoke perf).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_lifecycle"
SERVICE_SECRET = "svc_secret_lifecycle"


@pytest.fixture
async def lifecycle_client(aiohttp_client, tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = build_dlaas_app(
        db_path=str(tmp_path / "lifecycle.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        service_secret=SERVICE_SECRET,
        vertical=spec,
        max_sessions=8,
        idle_eviction_seconds=None,
    )
    return await aiohttp_client(app)


async def test_full_lifecycle_register_to_handoff(lifecycle_client):
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}

    # ----- 1. Tenant -----
    resp = await lifecycle_client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Lifecycle Tenant",
            "contact_email": "ops@lifecycle.example",
            "business_type": "education",
            "billing_plan": "pay_as_you_go",
            "quota": {"max_instances": 5},
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    tenant_id = tenant["tenant_id"]
    headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }

    # ----- 2. Shell + Asset -----
    resp = await lifecycle_client.post(
        "/dlaas/shells",
        headers=headers,
        json={
            "shell_id": "lifecycle_web",
            "shell_kind": "deployment",
            "shell_type": "web_chat",
            "display_name": "Lifecycle Web",
            "embodiment": {
                "perception": ["page_context", "user_profile"],
                "expression": ["text_streaming"],
                "action": ["open_url"],
                "constraints": {"max_response_chars": 2000},
            },
            "channel": {"type": "web", "region": "cn"},
            "scene_meta": {"product": "tutoring"},
        },
    )
    assert resp.status == 200
    shell_payload = await resp.json()
    assert shell_payload["capabilities_accepted"] == [
        "action",
        "expression",
        "perception",
    ]

    resp = await lifecycle_client.post(
        "/dlaas/assets",
        headers=headers,
        json={
            "asset_type": "persona_kit",
            "title": "Lifecycle Persona",
            "uri": "test:persona/lifecycle.md",
            "mime_type": "text/markdown",
            "language": "zh-CN",
        },
    )
    assert resp.status == 200
    asset_id = (await resp.json())["asset_id"]

    # ----- 3. Template -----
    resp = await lifecycle_client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Lifecycle Template",
            "domain": "education",
            "description": "Patient math tutor for primary-school students.",
            "runtime_template_id": "companion",
            "base_persona": {"language": "zh-CN", "audience": "primary"},
            "persona_spec": {
                "display_name": "小鹿数学老师",
                "role_archetype": "growth_advisor",
                "speaking_style": "warm, concise, encouraging",
                "value_boundaries": [
                    "never shame the learner",
                    "ask before escalating to parents",
                ],
                "background_story": "A long-term learning companion.",
            },
            "seed_config": {
                "domain_seed": "primary_math_tutor",
                "soul_template_kind": "education_growth_advisor",
            },
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    resp = await lifecycle_client.post(
        f"/dlaas/templates/{template_id}/assets",
        headers=headers,
        json={
            "asset_id": asset_id,
            "role": "persona_source",
            "link_meta": {"importance": "high"},
        },
    )
    assert resp.status == 200, await resp.text()

    # ----- 4. Activate + Readiness -----
    resp = await lifecycle_client.post(
        f"/dlaas/templates/{template_id}/activate",
        headers=headers,
        json={"seed_config_override": {}},
    )
    assert resp.status == 200, await resp.text()
    activation = await resp.json()
    assert activation["activation_status"] == "activated"
    assert activation["activation_result"]["ingestion_report"]["processed_chunks"] >= 1

    resp = await lifecycle_client.get(
        f"/dlaas/templates/{template_id}/readiness", headers=headers
    )
    assert resp.status == 200
    readiness = await resp.json()
    assert readiness["ready"] is True
    assert readiness["activation_status"] == "activated"
    assert readiness["has_runtime_template_id"] is True

    # ----- 5. Publish via PATCH -----
    resp = await lifecycle_client.patch(
        f"/dlaas/templates/{template_id}",
        headers=headers,
        json={
            "status": "published",
            "version_note": "Initial public launch candidate",
        },
    )
    assert resp.status == 200, await resp.text()
    template_after_publish = await resp.json()
    assert template_after_publish["status"] == "published"

    # ----- 6. Exam questions + run -----
    resp = await lifecycle_client.post(
        "/dlaas/exam_questions",
        headers=headers,
        json={
            "template_id": template_id,
            "scenario_tag": "fraction_help",
            "user_prompt": "我又错了，是不是很笨？",
            "context": {"student_age": 10},
            "rubric": [
                {
                    "criterion": "empathy_first",
                    "description": "先承认情绪",
                    "max_score": 10,
                    "weight": 0.5,
                },
                {
                    "criterion": "actionable_next_step",
                    "description": "给一个小步骤",
                    "max_score": 10,
                    "weight": 0.5,
                },
            ],
            "reference_answer": "你不是笨...",
            "tags": ["math", "emotion"],
            "difficulty": "medium",
        },
    )
    assert resp.status == 200
    question_id = (await resp.json())["question_id"]

    resp = await lifecycle_client.post(
        "/dlaas/exam_runs",
        headers=headers,
        json={
            "template_id": template_id,
            "template_version": template_after_publish["current_version"],
            "run_type": "launch_gate",
            "question_ids": [question_id],
            "pass_threshold": 0.4,
        },
    )
    assert resp.status == 200
    run_id = (await resp.json())["run_id"]

    resp = await lifecycle_client.post(
        f"/dlaas/exam_runs/{run_id}/complete",
        headers=headers,
        json={
            "ai_responses": {
                question_id: "你不是笨，我们先一起圈出题目里的关键词。"
            },
            "operator_id": "op_007",
            "operator_name": "Launch Reviewer",
        },
    )
    assert resp.status == 200
    completed_run = await resp.json()
    assert completed_run["status"] == "completed"
    assert completed_run["passed"] is True
    assert completed_run["aggregate_score"] >= 0.4

    # ----- 7. License -----
    resp = await lifecycle_client.post(
        f"/dlaas/templates/{template_id}/license/evaluate"
        f"?template_version={template_after_publish['current_version']}",
        headers=headers,
        json={},
    )
    assert resp.status == 200
    license_spec = await resp.json()
    assert license_spec["granted"] is True

    # ----- 8. Adopt -----
    resp = await lifecycle_client.post(
        "/dlaas/adopt",
        headers=headers,
        json={
            "tenant_id": tenant_id,
            "template_id": template_id,
            "template_version": template_after_publish["current_version"],
            "shell_id": "lifecycle_web",
            "owner_user_id": "owner_001",
            "engine_tools": {
                "web_browse": False,
                "web_search": True,
                "data_query": {
                    "enabled": True,
                    "allowed_sources": ["acme_lms"],
                },
                "content_analysis": True,
            },
            "service_contract": {
                "awake_strategy": "on_demand",
                "sla": "standard",
            },
            "focus_persons": [
                {
                    "person_id": "student_10001",
                    "name": "小明",
                    "role": "student",
                    "relationship_to_owner": "learner",
                    "age": 10,
                    "initial_profile": {
                        "grade": "四年级",
                        "weaknesses": ["应用题读题不仔细"],
                    },
                }
            ],
        },
    )
    assert resp.status == 200, await resp.text()
    adopt_payload = await resp.json()
    ai_id = adopt_payload["ai_id"]
    contract_id = adopt_payload["contract_id"]
    assert ai_id.startswith("ai_")
    assert contract_id.startswith("ctr_")
    assert adopt_payload["instance_endpoint"].endswith(
        f"/dlaas/instances/{ai_id}/interactions"
    )
    persons = adopt_payload["persons_registered"]
    assert any(
        p["person_id"] == "student_10001" and p["card_created"]
        for p in persons
    )
    tool_policy = adopt_payload["tool_policy_snapshot"]
    assert "web_search" in tool_policy["enabled_capabilities"]
    assert "data_query" in tool_policy["enabled_capabilities"]
    assert "content_analysis" in tool_policy["enabled_capabilities"]
    assert "web_browse" not in tool_policy["enabled_capabilities"]

    # ----- 9. Identity links -----
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/identity_links/batch",
        headers=headers,
        json={
            "links": [
                {
                    "canonical_end_user_ref": "student_10001",
                    "channel_type": "web",
                    "channel_ref": "web_user_123",
                    "link_meta": {},
                },
                {
                    "canonical_end_user_ref": "student_10001",
                    "channel_type": "wechat",
                    "channel_ref": "wx_openid_abc",
                    "link_meta": {"verified": True},
                },
            ]
        },
    )
    assert resp.status == 200
    link_body = await resp.json()
    assert len(link_body["links"]) == 2

    resp = await lifecycle_client.get(
        f"/dlaas/instances/{ai_id}/identity_links", headers=headers
    )
    assert resp.status == 200
    listed = await resp.json()
    assert len(listed["links"]) == 2

    # ----- 10. Live interactions -----
    chat_body = {
        "contract_id": contract_id,
        "session_id": "sess_lifecycle_001",
        "end_user_ref": "student_10001",
        "interaction_type": "chat",
        "human_brief": "我今天分数应用题又错了，怎么提高？",
        "structured_context": {"target_person_ids": ["student_10001"]},
        "lang": "cn",
    }
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/interactions",
        headers=headers,
        json=chat_body,
    )
    assert resp.status == 200, await resp.text()
    chat_response = await resp.json()
    assert chat_response["status"] == "ok"
    assert chat_response["ai_id"] == ai_id
    assert chat_response["output_acts"][0]["act_type"] == "text"

    # Feedback turn confirms the dispatch table reaches the kernel.
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/interactions",
        headers=headers,
        json={
            "contract_id": contract_id,
            "session_id": "sess_lifecycle_001",
            "end_user_ref": "teacher_003",
            "interaction_type": "feedback",
            "feedback": {
                "valence": "correct",
                "target_response_id": chat_response["response_id"],
                "intensity": 0.9,
                "scope": "response",
                "evidence": "学生听懂了。",
            },
            "lang": "cn",
        },
    )
    assert resp.status == 200, await resp.text()
    feedback_payload = await resp.json()
    assert feedback_payload["feedback"]["outcome_kind"] == "helped"

    # Observe homework_result
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/interactions",
        headers=headers,
        json={
            "contract_id": contract_id,
            "session_id": "sess_lifecycle_001",
            "end_user_ref": "student_10001",
            "interaction_type": "observe",
            "human_brief": "学生完成 10 道分数应用题。",
            "structured_context": {
                "observation_type": "homework_result",
                "event_id": "obs_001",
                "task_id": "math_hw_001",
                "status": "submitted",
                "summary": "fraction word problems",
                "detail": "10/15 correct",
                "target_person_ids": ["student_10001"],
            },
            "lang": "cn",
        },
    )
    assert resp.status == 200, await resp.text()
    observe_payload = await resp.json()
    assert observe_payload["observation_type"] == "homework_result"
    assert len(observe_payload["event_ids"]) >= 1

    # ----- 11. Pause + operator-message + 12. Resume -----
    resp = await lifecycle_client.post(
        "/dlaas/admin/ops/conversations/sess_lifecycle_001/pause",
        headers=cp_headers,
        json={
            "ai_id": ai_id,
            "operator_id": "op_007",
            "note": "Manual takeover",
        },
    )
    assert resp.status == 200, await resp.text()
    assert (await resp.json())["paused"] is True

    # While paused: chat returns operator_takeover placeholder, kernel untouched.
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/interactions",
        headers=headers,
        json={
            "contract_id": contract_id,
            "session_id": "sess_lifecycle_001",
            "end_user_ref": "student_10001",
            "interaction_type": "chat",
            "human_brief": "继续辅导我",
            "lang": "cn",
        },
    )
    assert resp.status == 200, await resp.text()
    paused_response = await resp.json()
    assert paused_response["status"] == "operator_takeover"
    assert paused_response["operator_takeover"] is True
    assert paused_response["output_acts"][0]["act_type"] == "system"

    resp = await lifecycle_client.post(
        "/dlaas/admin/ops/conversations/sess_lifecycle_001/operator-message",
        headers=cp_headers,
        json={
            "ai_id": ai_id,
            "operator_id": "op_007",
            "text": "我们一起圈出题目里的单位。",
            "inject_into_runtime": False,
        },
    )
    assert resp.status == 200

    resp = await lifecycle_client.post(
        "/dlaas/admin/ops/conversations/sess_lifecycle_001/resume",
        headers=cp_headers,
        json={
            "ai_id": ai_id,
            "operator_id": "op_007",
            "note": "Resolved",
        },
    )
    assert resp.status == 200
    assert (await resp.json())["paused"] is False

    # Chat resumes after resume
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/interactions",
        headers=headers,
        json={
            "contract_id": contract_id,
            "session_id": "sess_lifecycle_001",
            "end_user_ref": "student_10001",
            "interaction_type": "chat",
            "human_brief": "我们继续",
            "lang": "cn",
        },
    )
    assert resp.status == 200
    resumed = await resp.json()
    assert resumed["status"] == "ok"
    assert resumed["output_acts"][0]["act_type"] == "text"

    # ----- 13. Handoff ticket -----
    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/handoff_tickets",
        headers=headers,
        json={
            "contract_id": contract_id,
            "end_user_ref": "student_10001",
            "session_id": "sess_lifecycle_001",
            "trigger_reason": "low_confidence",
            "trigger_details": {"topic": "fractions"},
            "confidence_aggregate": 0.42,
            "recent_response_ids": [chat_response["response_id"]],
        },
    )
    assert resp.status == 200, await resp.text()
    ticket = await resp.json()
    ticket_id = ticket["ticket_id"]
    assert ticket["status"] == "open"

    resp = await lifecycle_client.get(
        f"/dlaas/instances/{ai_id}/handoff_queue?status=open",
        headers=headers,
    )
    assert resp.status == 200
    queue = await resp.json()
    assert any(t["ticket_id"] == ticket_id for t in queue["tickets"])

    resp = await lifecycle_client.post(
        f"/dlaas/instances/{ai_id}/handoff/{ticket_id}/human_reply",
        headers=headers,
        json={
            "operator_id": "op_007",
            "human_reply": "我来帮你，先只看第一句。",
            "resolution_notes": "Operator took over.",
        },
    )
    assert resp.status == 200
    resolved = await resp.json()
    assert resolved["status"] == "resolved"
    assert resolved["operator_id"] == "op_007"


async def test_adopt_rejects_unpublished_template(lifecycle_client):
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await lifecycle_client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Adopt Guard",
            "contact_email": "ops@adopt.example",
        },
    )
    tenant = await resp.json()
    headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    resp = await lifecycle_client.post(
        "/dlaas/shells",
        headers=headers,
        json={"shell_id": "adopt_web", "shell_kind": "deployment"},
    )
    assert resp.status == 200
    resp = await lifecycle_client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Draft Only",
            "domain": "education",
            "runtime_template_id": "companion",
        },
    )
    template_id = (await resp.json())["template_id"]
    # Skip activation. Adoption must reject because status != published.
    resp = await lifecycle_client.post(
        "/dlaas/adopt",
        headers=headers,
        json={
            "template_id": template_id,
            "shell_id": "adopt_web",
        },
    )
    assert resp.status == 409, await resp.text()
    payload = await resp.json()
    assert payload["error"] == "template_not_published"


async def test_adopt_rejects_studio_shell(lifecycle_client):
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await lifecycle_client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Studio Guard",
            "contact_email": "ops@studio.example",
        },
    )
    tenant = await resp.json()
    headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    resp = await lifecycle_client.post(
        "/dlaas/shells",
        headers=headers,
        json={"shell_id": "studio_only", "shell_kind": "studio"},
    )
    assert resp.status == 200
    resp = await lifecycle_client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Studio Adopt Test",
            "domain": "education",
            "runtime_template_id": "companion",
        },
    )
    template_id = (await resp.json())["template_id"]
    resp = await lifecycle_client.post(
        f"/dlaas/templates/{template_id}/activate",
        headers=headers,
        json={},
    )
    assert resp.status == 200
    resp = await lifecycle_client.patch(
        f"/dlaas/templates/{template_id}",
        headers=headers,
        json={"status": "published"},
    )
    assert resp.status == 200
    # Even with everything else right, studio shells cannot host a runtime
    # contract.
    resp = await lifecycle_client.post(
        "/dlaas/adopt",
        headers=headers,
        json={"template_id": template_id, "shell_id": "studio_only"},
    )
    assert resp.status == 409, await resp.text()
    payload = await resp.json()
    assert payload["error"] == "shell_not_deployable"
