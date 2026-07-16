"""D1 (#14) route tests: /audience/analyze real-analysis wiring.

Covers the three route modes:

* analyzer configured + readable assets → corpus is resolved from the
  asset store, the LLM analyzer (fake transport, no network) fills
  the empty profile slots, caller-supplied fields stay authoritative,
  and ``evidence_stats`` names the analyzer + corpus footprint;
* analyzer configured + unreadable asset scheme → typed 422
  ``asset_corpus_unreadable`` (never a hollow profile);
* no analyzer (LLM env unset) → honest SHADOW passthrough with
  ``evidence_stats.analyzer = "none"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app
from dlaas_platform_eval import EVAL_BUNDLE_APP_KEY
from dlaas_platform_eval.audience import LLMAudienceAnalyzer
from dlaas_platform_eval.llm_grader import EvalLLMConfig


CONTROL_PLANE_SECRET = "cp_secret_audience"

_PROFILE_JSON = {
    "common_questions": ["What's a healthy bedtime?"],
    "communication_style": "warm-pragmatic",
    "emotion_triggers": ["guilt"],
    "decision_patterns": ["seek confirmation"],
    "evidence_notes": "From uploaded chat log.",
}


def _fake_transport(config, *, system_prompt: str, user_prompt: str) -> str:
    assert "<<<CORPUS>>>" in user_prompt
    return json.dumps(_PROFILE_JSON)


def _fake_analyzer() -> LLMAudienceAnalyzer:
    return LLMAudienceAnalyzer(
        EvalLLMConfig(
            base_url="https://llm.example/v1",
            api_key="k",
            model="audience-model",
        ),
        transport=_fake_transport,
    )


@pytest.fixture
async def client(aiohttp_client, tmp_path: Path, monkeypatch):
    # The app must not pick up a real LLM env from the host.
    for name in (
        "EVAL_LLM_BASE_URL",
        "EVAL_LLM_API_KEY",
        "EVAL_LLM_MODEL",
        "PROTOCOL_LLM_BASE_URL",
        "PROTOCOL_LLM_API_KEY",
        "PROTOCOL_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = build_dlaas_app(
        db_path=str(tmp_path / "audience.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )
    return await aiohttp_client(app)


async def _setup_tenant_template(client, *, asset_uri: str = "") -> tuple[dict, str, str]:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Audience Tenant",
            "contact_email": "ops@audience.example",
            "business_type": "education",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    resp = await client.post(
        "/dlaas/assets",
        headers=headers,
        json={
            "asset_type": "chat_log",
            "title": "Parent chat log",
            "uri": asset_uri,
            "mime_type": "text/plain",
            "source_meta": (
                {}
                if asset_uri
                else {
                    "inline_text": (
                        "Parent: my kid refuses dinner, am I failing?\n"
                        "Parent: what bedtime is normal for a 6 year old?"
                    )
                }
            ),
        },
    )
    assert resp.status == 200, await resp.text()
    asset_id = (await resp.json())["asset_id"]
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Growth advisor",
            "domain": "education",
            "description": "advisor",
            "runtime_template_id": "companion",
            "base_persona": {},
            "persona_spec": {"display_name": "Advisor"},
            "seed_config": {},
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]
    return (headers, template_id, asset_id)


async def test_audience_analyze_runs_real_analysis(client) -> None:
    headers, template_id, asset_id = await _setup_tenant_template(client)
    client.server.app[EVAL_BUNDLE_APP_KEY].audience_analyzer = _fake_analyzer()

    resp = await client.post(
        f"/dlaas/templates/{template_id}/audience/analyze",
        headers=headers,
        json={
            "cohort_name": "anxious-parents",
            "asset_ids": [asset_id],
            # Caller-supplied field must survive the analyzer.
            "communication_style": "caller-explicit",
        },
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["cohort_name"] == "anxious-parents"
    assert body["common_questions"] == ["What's a healthy bedtime?"]
    assert body["communication_style"] == "caller-explicit"
    assert body["emotion_triggers"] == ["guilt"]
    stats = body["evidence_stats"]
    assert stats["analyzer"] == "llm:audience-model"
    assert stats["asset_count"] == 1
    assert stats["chunk_count"] == 1
    assert stats["evidence_notes"] == "From uploaded chat log."


async def test_audience_analyze_unreadable_asset_is_typed_422(client) -> None:
    headers, template_id, asset_id = await _setup_tenant_template(
        client, asset_uri="s3://bucket/log.txt"
    )
    client.server.app[EVAL_BUNDLE_APP_KEY].audience_analyzer = _fake_analyzer()

    resp = await client.post(
        f"/dlaas/templates/{template_id}/audience/analyze",
        headers=headers,
        json={"cohort_name": "x", "asset_ids": [asset_id]},
    )
    assert resp.status == 422
    body = await resp.json()
    assert body["error"] == "asset_corpus_unreadable"
    assert "s3" in body["detail"]


async def test_audience_analyze_without_analyzer_is_honest_passthrough(
    client,
) -> None:
    headers, template_id, asset_id = await _setup_tenant_template(client)
    assert client.server.app[EVAL_BUNDLE_APP_KEY].audience_analyzer is None

    resp = await client.post(
        f"/dlaas/templates/{template_id}/audience/analyze",
        headers=headers,
        json={
            "cohort_name": "form-only",
            "asset_ids": [asset_id],
            "communication_style": "declared-by-caller",
        },
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["communication_style"] == "declared-by-caller"
    assert body["common_questions"] == []
    assert body["evidence_stats"]["analyzer"] == "none"
    assert body["evidence_stats"]["asset_count"] == 1


async def test_audience_analyze_unknown_asset_is_404(client) -> None:
    headers, template_id, _asset_id = await _setup_tenant_template(client)
    client.server.app[EVAL_BUNDLE_APP_KEY].audience_analyzer = _fake_analyzer()

    resp = await client.post(
        f"/dlaas/templates/{template_id}/audience/analyze",
        headers=headers,
        json={"cohort_name": "x", "asset_ids": ["ast_missing"]},
    )
    assert resp.status == 404
    assert (await resp.json())["error"] == "asset_not_found"
