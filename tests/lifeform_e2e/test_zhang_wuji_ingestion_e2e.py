"""Wave C7 (Tier 2 evidence) — multi-chunk ingestion + R6 slow loop.

Drives a multi-chunk synthetic excerpt through the canonical
``LifeformSession.run_turn(..., trigger_kind=INGESTION)`` path via
``IngestionPipeline``, ends the scene to fire the session-post slow
loop, and verifies that **at least one observability surface in
vz-memory advanced**.

The disjunction-style assertion is intentional: we don't pin which
specific memory observable moves (attribute_summary length / CMS
band cadence counter / lifecycle metric / pending promotions are
all plausible response surfaces and the choice depends on the
session-post slow loop's internal scheduling), only that the
ingestion path produced ANY measurable owner-side response. A
future memory-owner refactor that changes which surface advances
should still pass; one that breaks the slow loop entirely should
fail.

The test runs deterministically without an LLM (synthetic substrate
default) and finishes in ~10 seconds.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lifeform_domain_character import (
    build_character_ingestion_envelope,
    build_zhang_wuji_lifeform,
    build_zhang_wuji_profile,
    zhang_wuji_long_arc_excerpt,
)
from lifeform_ingestion import IngestionComplianceProfile, IngestionPipeline


def _memory_observables(snapshot: Any) -> dict[str, float]:
    """Extract a small bundle of numeric memory observables.

    These are the surfaces a session-post slow loop or per-turn
    ingestion can plausibly move. Returning a dict makes diff
    output legible when the disjunction assertion fails.
    """
    out: dict[str, float] = {
        "entries_count": float(len(getattr(snapshot, "entries", ()) or ())),
        "attribute_summary_count": float(
            len(getattr(snapshot, "attribute_summary", ()) or ())
        ),
        "pending_promotions": float(getattr(snapshot, "pending_promotions", 0) or 0),
        "retrieved_entries_count": float(
            len(getattr(snapshot, "retrieved_entries", ()) or ())
        ),
    }
    lifecycle = getattr(snapshot, "lifecycle_metrics", None) or {}
    if isinstance(lifecycle, dict):
        for key in (
            "learned_recall_confidence",
            "core_guided_recall_evidence",
            "nested_context_reset_count",
            "slow_to_fast_init_benefit",
        ):
            value = lifecycle.get(key)
            if isinstance(value, (int, float)):
                out[f"lifecycle::{key}"] = float(value)
    cms_state = getattr(snapshot, "cms_state", None)
    if cms_state is not None:
        for band_name in ("online_fast", "session_medium", "background_slow"):
            band = getattr(cms_state, band_name, None)
            if band is None:
                continue
            obs = getattr(band, "observations_since_update", None)
            if isinstance(obs, (int, float)):
                out[f"cms::{band_name}::observations"] = float(obs)
    return out


def test_zhang_wuji_excerpt_drives_multi_chunk_ingestion() -> None:
    """Smoke: the canonical text produces at least 8 chunks at the
    test-time chunk size, and every chunk processes without raising.

    8 chunks is the floor that exercises both the ``per-paragraph``
    and the ``hard-cut`` paths inside ``chunk_plain_text``; it is
    also enough to span more than one R6 "session-post slow loop
    consolidation" boundary, which the next test asserts.
    """
    profile = build_zhang_wuji_profile()
    envelope = build_character_ingestion_envelope(
        profile,
        zhang_wuji_long_arc_excerpt(),
        uploader="test-suite-c7",
        upload_ts_ms=1_700_000_000_000,
        max_chunk_chars=512,
    )
    assert len(envelope.chunks) >= 8, (
        f"expected >= 8 chunks at max_chunk_chars=512, got "
        f"{len(envelope.chunks)}"
    )
    assert envelope.partial_failures == ()
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED


def test_ingestion_pipeline_drains_envelope_through_canonical_turns() -> None:
    """The ``IngestionPipeline`` processes every chunk via
    ``run_turn(trigger_kind=INGESTION)`` without any chunk being
    skipped, and ends the scene afterwards so the session-post slow
    loop has a chance to fire.
    """
    bundle = build_zhang_wuji_lifeform()
    session = bundle.lifeform.create_session(
        session_id="zhang-wuji-tier2-pipeline"
    )
    profile = bundle.profile
    envelope = build_character_ingestion_envelope(
        profile,
        zhang_wuji_long_arc_excerpt(),
        uploader="test-suite-c7",
        upload_ts_ms=1_700_000_000_000,
        max_chunk_chars=512,
    )

    async def _go() -> Any:
        pipeline = IngestionPipeline()
        return await pipeline.process_envelope(
            envelope,
            session=session,
            end_scene_after=True,
            scene_end_reason="ingestion-end",
            scene_end_drains_slow_loop=True,
        )

    report = asyncio.run(_go())
    # Every chunk processed; none skipped.
    assert report.processed_chunks == len(envelope.chunks), (
        f"expected all {len(envelope.chunks)} chunks processed, got "
        f"{report.processed_chunks}; skipped={report.skipped_chunks}"
    )
    assert report.skipped_chunks == 0
    assert report.ended_scene is True
    assert report.all_succeeded


def test_ingestion_advances_at_least_one_memory_observable() -> None:
    """**Tier 2 evidence**: ingestion of the multi-chunk excerpt
    moves at least one memory observability surface.

    Disjunction assertion: any of
    - memory.entries count
    - memory.attribute_summary length
    - memory.pending_promotions
    - any lifecycle_metric advancing (recall_confidence /
      slow_to_fast_init_benefit / nested_context_reset_count /
      core_guided_recall_evidence)
    - any CMS band's observations_since_update advancing
    must move strictly upward between baseline and post-ingestion.

    We pick this disjunction shape because the specific surface
    that advances depends on the slow loop's internal scheduling
    (which is owner-internal and may evolve). Rejecting the
    disjunction means R6 / session-post slow loop / memory-side
    ingestion handling all failed silently — which is the
    regression we want to catch.
    """
    bundle = build_zhang_wuji_lifeform()
    session = bundle.lifeform.create_session(
        session_id="zhang-wuji-tier2-observable"
    )

    async def _go():
        # Baseline: drive one trivial user turn so a valid memory
        # snapshot exists. Without this the kernel has not produced
        # any snapshot yet and our pre/post comparison is undefined.
        baseline_result = await session.run_turn("准备开始读一段。")
        baseline_memory = baseline_result.active_snapshots["memory"].value

        envelope = build_character_ingestion_envelope(
            bundle.profile,
            zhang_wuji_long_arc_excerpt(),
            uploader="test-suite-c7",
            upload_ts_ms=1_700_000_000_000,
            max_chunk_chars=512,
        )
        pipeline = IngestionPipeline()
        report = await pipeline.process_envelope(
            envelope,
            session=session,
            end_scene_after=True,
            scene_end_reason="ingestion-end",
            scene_end_drains_slow_loop=True,
        )

        # The session is now closed (end_scene fired). Do one more
        # turn to observe the post-ingestion memory snapshot. This
        # turn is a USER_INPUT turn (not ingestion) so it reads the
        # consolidated state.
        post_result = await session.run_turn("好了，回到正常对话。")
        post_memory = post_result.active_snapshots["memory"].value

        return baseline_memory, post_memory, report

    baseline_memory, post_memory, report = asyncio.run(_go())

    pre = _memory_observables(baseline_memory)
    post = _memory_observables(post_memory)
    advanced: list[str] = [
        key
        for key in pre
        if key in post and post[key] > pre[key]
    ]
    assert advanced, (
        f"Tier 2 ingestion produced no measurable memory advance. "
        f"baseline={pre!r}\npost={post!r}\n"
        f"chunks_processed={report.processed_chunks} "
        f"ended_scene={report.ended_scene}"
    )


def test_ingestion_envelope_is_book_kind_with_provenance_attribution() -> None:
    """The envelope must carry source_kind=BOOK and a stable provenance
    URI keyed to the profile, so reflection writeback / audit can
    later attribute records back to the ingestion event.
    """
    profile = build_zhang_wuji_profile()
    envelope = build_character_ingestion_envelope(
        profile,
        zhang_wuji_long_arc_excerpt(),
        uploader="test-suite-c7",
        upload_ts_ms=1_700_000_000_000,
    )
    assert envelope.envelope_id == "character-ingestion:zhang-wuji"
    assert envelope.provenance.uploader == "test-suite-c7"
    assert envelope.provenance.source_uri == profile.source_uri
    # Integrity hash is the SHA256 of the source text — non-empty.
    assert envelope.provenance.integrity_hash
    assert len(envelope.provenance.integrity_hash) == 64
