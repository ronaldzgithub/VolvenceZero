"""Contract test for U6 — `event: evidence` SSE frame propagates pointers.

The family-memorial product depends on every chat turn surfacing the
structured ``EvidencePointer`` list (not just the count summarized in
``x-lifeform-rationale-tags``) so apps/family-memorial's
``CitationCard`` can render clickable citations back to the original
corpus. Before U6, the GroundedDecoder's pointer list was thrown
away at the tag-string level.

This test exercises the OpenAI-compat streaming path end-to-end and
asserts (P7.8 audit closure):

1. **Bundle bound** — a ``stream=true`` chat completion with the
   einstein-bundle vertical MUST emit an ``event: evidence`` SSE
   frame; the audit insisted on a deterministic in-corpus query so
   the test does not silently pass on "no pointers generated".
2. **Schema invariant** — every pointer carries ``raw_locator`` /
   ``chunk_id`` / ``source_envelope_id`` plus the structured fields
   (locator_kind / document_id / …).
3. **No bundle bound** — a separate dlaas app built against the
   ``einstein-raw`` (no-bundle) vertical does NOT emit a spurious
   evidence frame. This used to be a pytest.skip; the audit forced
   a real test.
4. **Bridge-level invariants** — ``LifeformCompletionResult`` and
   ``AgentResponse`` expose ``evidence_pointers`` with sensible
   defaults so legacy callers never crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_u6_evidence_frame"


def _split_sse_frames(body: str) -> list[str]:
    """Split a UTF-8 SSE body into frames. Accepts both ``\\n\\n``
    (real spec) and ``\\r\\n\\r\\n`` (some HTTP libraries normalise
    here). Empty trailing frames are dropped.
    """

    raw = body.replace("\r\n", "\n")
    return [f for f in raw.split("\n\n") if f.strip()]


def _find_evidence_frame(body: str) -> str | None:
    for frame in _split_sse_frames(body):
        if frame.startswith("event: evidence"):
            return frame
    return None


async def _build_app_with_vertical(tmp_path: Path, vertical_id: str):
    from lifeform_openai_compat import add_openai_routes  # noqa: PLC0415
    from lifeform_service.verticals import discover_verticals  # noqa: PLC0415

    spec = discover_verticals()[vertical_id]
    app = build_dlaas_app(
        db_path=str(tmp_path / f"u6_{vertical_id}.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )
    add_openai_routes(app)
    return app


@pytest.fixture
async def bundle_client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(
        await _build_app_with_vertical(tmp_path, "einstein-bundle")
    )


@pytest.fixture
async def raw_client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(
        await _build_app_with_vertical(tmp_path / "raw", "einstein-raw")
    )


async def _wake(client, ai_id: str, template_id_for_wake: str | None = None) -> None:
    payload = {
        "runtime_template_id": "einstein-bundle",
        "reason": "u6-contract-test",
    }
    if template_id_for_wake is not None:
        payload["template_id"] = template_id_for_wake
    resp = await client.post(
        f"/dlaas/v1/instances/{ai_id}/wake", json=payload
    )
    assert resp.status == 200, await resp.text()


async def _stream_chat(client, ai_id: str, user_msg: str) -> tuple[int, str]:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-einstein",
            "stream": True,
            "messages": [{"role": "user", "content": user_msg}],
            "metadata": {"dlaas.ai_id": ai_id, "session_id": f"u6_{ai_id}"},
        },
    )
    body = await resp.text()
    return resp.status, body


# ---------------------------------------------------------------------------
# Happy path: bundle bound -> evidence frame MUST appear
# ---------------------------------------------------------------------------


async def test_bundle_query_emits_evidence_frame_with_pointers(
    bundle_client,
) -> None:
    """einstein-bundle has the synthetic einstein corpus loaded by
    default. We ask a *deterministically in-corpus* question
    (general relativity), and the GroundedDecoder must return at
    least one pointer; the evidence frame must appear.

    The audit explicitly rejected an "if frame present" branch — a
    bundle-bound query that yields zero pointers is itself a
    contract regression (L3 silently degraded), so we assert
    strictly here.
    """

    ai_id = "memorial_u6_bundle"
    await _wake(bundle_client, ai_id)
    status, body = await _stream_chat(
        bundle_client,
        ai_id,
        "Briefly describe Einstein's general theory of relativity.",
    )
    assert status == 200
    assert "data: [DONE]" in body, "stream must end with [DONE]"

    evidence_frame = _find_evidence_frame(body)
    assert evidence_frame is not None, (
        "bundle-bound query MUST emit an event: evidence frame; got body: "
        f"{body[:1500]}"
    )

    data_line = next(
        (line for line in evidence_frame.splitlines() if line.startswith("data: ")),
        None,
    )
    assert data_line is not None
    payload = json.loads(data_line[len("data: ") :])
    assert payload.get("schema_version") == 1
    pointers = payload.get("pointers")
    assert isinstance(pointers, list)
    assert len(pointers) >= 1, "bundle-bound query must produce >= 1 pointer"
    for pointer in pointers:
        assert "raw_locator" in pointer
        assert "chunk_id" in pointer
        assert "source_envelope_id" in pointer
        for k in (
            "locator_kind",
            "document_id",
            "paragraph_index",
            "offset_start",
            "offset_end",
            "language",
            "page",
        ):
            assert k in pointer, f"pointer missing structured field {k!r}"


# ---------------------------------------------------------------------------
# Negative path: no bundle -> NO evidence frame
# ---------------------------------------------------------------------------


async def test_no_bundle_omits_evidence_frame(raw_client) -> None:
    """einstein-raw deliberately does not attach a figure bundle.

    Streaming a chat completion against this vertical must NOT emit
    a spurious ``event: evidence`` frame — the legacy OpenAI 4-frame
    sequence (role / content / final / DONE) is preserved verbatim.
    """

    ai_id = "memorial_u6_raw"
    # Wake on the raw vertical: no template_id, no figure binding.
    resp = await raw_client.post(
        f"/dlaas/v1/instances/{ai_id}/wake",
        json={"runtime_template_id": "einstein-raw", "reason": "u6-raw"},
    )
    assert resp.status == 200, await resp.text()

    status, body = await _stream_chat(
        raw_client, ai_id, "Tell me something brief."
    )
    assert status == 200
    assert "data: [DONE]" in body, "stream must end with [DONE]"
    assert _find_evidence_frame(body) is None, (
        "raw vertical (no figure bundle) must NOT emit event: evidence; got body: "
        f"{body[:1500]}"
    )


# ---------------------------------------------------------------------------
# Bridge-level invariants (cheap, no aiohttp fixture)
# ---------------------------------------------------------------------------


def test_lifeform_completion_result_carries_evidence_pointers() -> None:
    from lifeform_openai_compat.session_bridge import LifeformCompletionResult

    result = LifeformCompletionResult(
        response=None,  # type: ignore[arg-type]
        resolution=None,  # type: ignore[arg-type]
        active_regime=None,
        active_abstract_action=None,
        pe_magnitude=0.0,
        rationale_tags=(),
    )
    assert result.evidence_pointers == ()

    result_with = LifeformCompletionResult(
        response=None,  # type: ignore[arg-type]
        resolution=None,  # type: ignore[arg-type]
        active_regime=None,
        active_abstract_action=None,
        pe_magnitude=0.0,
        rationale_tags=("l3_grounded_verify=passed:1;evidence:2",),
        evidence_pointers=(
            {
                "raw_locator": "letter:einstein->besso@1916-05-03",
                "chunk_id": "ck_abc",
                "source_envelope_id": "env_letters_001",
                "locator_kind": "letter",
                "document_id": "letter_001",
                "paragraph_index": 2,
                "offset_start": 12,
                "offset_end": 80,
                "language": "de",
                "sender_id": "einstein",
                "recipient_id": "besso",
                "date_iso": "1916-05-03",
                "venue_id": "",
                "volume": "",
                "page": -1,
                "rendered": "letter[einstein->besso@1916-05-03] para=2 off=12-80 lang=de | env_letters_001",
            },
        ),
    )
    assert len(result_with.evidence_pointers) == 1
    assert result_with.evidence_pointers[0]["chunk_id"] == "ck_abc"


def test_agent_response_carries_evidence_pointers_default_empty() -> None:
    from volvence_zero.agent.response import AgentResponse

    resp = AgentResponse(
        text="hello",
        regime_id="growth",
        abstract_action="anchor",
        rationale="test",
        rationale_tags=("regime=growth",),
    )
    assert resp.evidence_pointers == ()

    resp_with = AgentResponse(
        text="hello",
        regime_id="growth",
        abstract_action="anchor",
        rationale="test",
        rationale_tags=("regime=growth", "l3_grounded_verify=passed:1;evidence:1"),
        evidence_pointers=(
            {"raw_locator": "loc1", "chunk_id": "c1", "source_envelope_id": "e1"},
        ),
    )
    assert len(resp_with.evidence_pointers) == 1
