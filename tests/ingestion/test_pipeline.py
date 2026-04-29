"""IngestionPipeline unit tests.

Uses a fake session stub to validate the pipeline's contract:

* Only ``session.run_turn`` + ``session.end_scene`` are called \u2014
  no owner-store access.
* ``trigger_kind=INGESTION`` is passed to run_turn for FORCED
  compliance (and ``USER_INPUT`` for CONSULTATIVE).
* Parse-error chunks are skipped and surfaced in the report.
* Per-chunk kernel exceptions do not poison subsequent chunks;
  they appear in the report as skipped with the exception class.
* ``end_scene_after`` controls whether the scene closes after the
  envelope is drained.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lifeform_core import TurnTriggerKind

from lifeform_ingestion import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionPipeline,
    IngestionProvenance,
    IngestionSourceKind,
)


class _FakeSession:
    """Minimal ``LifeformSession`` protocol impl for pipeline tests."""

    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self._raise_on = raise_on or set()
        self.run_turn_calls: list[tuple[str, TurnTriggerKind]] = []
        self.end_scene_calls: list[tuple[str, bool]] = []

    async def run_turn(
        self,
        user_input: str,
        *,
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
    ):
        self.run_turn_calls.append((user_input, trigger_kind))
        if user_input in self._raise_on:
            raise RuntimeError(f"simulated kernel failure on {user_input!r}")
        return SimpleNamespace(
            response=SimpleNamespace(
                text=f"ack:{user_input[:32]}",
                regime_id=None,
                abstract_action=None,
                rationale="",
            ),
            active_regime=None,
            active_abstract_action=None,
            active_snapshots={},
        )

    async def end_scene(
        self, *, reason: str = "", drain_slow_loop: bool = True,
    ):
        self.end_scene_calls.append((reason, drain_slow_loop))
        # Mimic a closed scene object so the report sees ended_scene=True.
        return SimpleNamespace(scene_id="scene-fake", closed_at_tick=1)


def _provenance() -> IngestionProvenance:
    return IngestionProvenance(
        uploader="pipeline-test",
        upload_ts_ms=1000,
        source_uri="test:pipeline",
        integrity_hash="",
    )


def _envelope(
    *,
    chunks: tuple[IngestionChunk, ...],
    compliance: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    partial_failures: tuple[str, ...] = (),
) -> IngestionEnvelope:
    return IngestionEnvelope(
        envelope_id="env-pipeline-test",
        source_kind=IngestionSourceKind.CORPUS,
        chunks=chunks,
        provenance=_provenance(),
        compliance_profile=compliance,
        partial_failures=partial_failures,
    )


def _good(chunk_id: str, text: str) -> IngestionChunk:
    return IngestionChunk(
        chunk_id=chunk_id, text=text, locator=f"inline:{chunk_id}", confidence=1.0
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_pipeline_runs_one_turn_per_successful_chunk() -> None:
    session = _FakeSession()
    pipeline = IngestionPipeline()
    envelope = _envelope(
        chunks=(
            _good("c-1", "Chapter 1: introduction."),
            _good("c-2", "Chapter 2: framework."),
            _good("c-3", "Chapter 3: conclusion."),
        ),
    )
    report = await pipeline.process_envelope(envelope, session=session)
    assert report.total_chunks == 3
    assert report.processed_chunks == 3
    assert report.skipped_chunks == 0
    assert report.all_succeeded is True
    assert len(session.run_turn_calls) == 3
    # Every call must have trigger_kind=INGESTION for FORCED compliance.
    for _, kind in session.run_turn_calls:
        assert kind is TurnTriggerKind.INGESTION
    # end_scene runs after drain.
    assert session.end_scene_calls == [("ingestion-end", True)]
    assert report.ended_scene is True


async def test_pipeline_passes_user_input_for_consultative_profile() -> None:
    session = _FakeSession()
    pipeline = IngestionPipeline()
    envelope = _envelope(
        chunks=(_good("c-1", "Look at this!"),),
        compliance=IngestionComplianceProfile.CONSULTATIVE,
    )
    await pipeline.process_envelope(envelope, session=session)
    _, kind = session.run_turn_calls[0]
    assert kind is TurnTriggerKind.USER_INPUT


async def test_pipeline_skips_parse_error_chunks() -> None:
    session = _FakeSession()
    pipeline = IngestionPipeline()
    bad = IngestionChunk(
        chunk_id="c-bad",
        text="",
        locator="page=3",
        parse_error="PDF extraction failed",
    )
    envelope = _envelope(
        chunks=(_good("c-1", "ok"), bad, _good("c-3", "more ok")),
        partial_failures=("c-bad",),
    )
    report = await pipeline.process_envelope(envelope, session=session)
    assert report.total_chunks == 3
    assert report.processed_chunks == 2
    assert report.skipped_chunks == 1
    # run_turn was called twice (skipping the bad chunk).
    assert len(session.run_turn_calls) == 2
    # The skipped record is still in the report with an explicit reason.
    skipped = next(r for r in report.turns if r.chunk_id == "c-bad")
    assert skipped.skipped_reason == "parse_error"
    assert skipped.turn_succeeded is False
    # Ordering preserved: the skip record sits between the two good ones.
    assert [r.chunk_id for r in report.turns] == ["c-1", "c-bad", "c-3"]


async def test_pipeline_isolates_per_chunk_kernel_exceptions() -> None:
    session = _FakeSession(raise_on={"bad-one"})
    pipeline = IngestionPipeline()
    envelope = _envelope(
        chunks=(
            _good("c-1", "ok-one"),
            _good("c-2", "bad-one"),
            _good("c-3", "ok-two"),
        ),
    )
    report = await pipeline.process_envelope(envelope, session=session)
    assert report.total_chunks == 3
    assert report.processed_chunks == 2
    assert report.skipped_chunks == 1
    # The failing chunk got skipped with its exception class name.
    skipped = next(r for r in report.turns if r.chunk_id == "c-2")
    assert skipped.skipped_reason == "RuntimeError"
    # The good chunks still flowed through.
    ok_ids = {r.chunk_id for r in report.turns if r.turn_succeeded}
    assert ok_ids == {"c-1", "c-3"}


async def test_pipeline_end_scene_after_false_skips_scene_close() -> None:
    session = _FakeSession()
    pipeline = IngestionPipeline()
    envelope = _envelope(chunks=(_good("c-1", "hi"),))
    report = await pipeline.process_envelope(
        envelope,
        session=session,
        end_scene_after=False,
    )
    assert session.end_scene_calls == []
    assert report.ended_scene is False


async def test_pipeline_does_not_reach_for_owner_stores_on_fake_session() -> None:
    """Sanity: a session lacking any store attribute must still drive
    the pipeline. Verifies the pipeline uses only the public protocol.
    """

    class _BareMinimumSession:
        async def run_turn(self, user_input, *, trigger_kind=TurnTriggerKind.USER_INPUT):
            return SimpleNamespace(response=SimpleNamespace(text="ok"))

        async def end_scene(self, *, reason="", drain_slow_loop=True):
            return None

    pipeline = IngestionPipeline()
    envelope = _envelope(chunks=(_good("c-1", "hi"),))
    report = await pipeline.process_envelope(
        envelope,
        session=_BareMinimumSession(),  # type: ignore[arg-type]
    )
    assert report.processed_chunks == 1
    # end_scene returned None -> ended_scene stays False in the report.
    assert report.ended_scene is False
