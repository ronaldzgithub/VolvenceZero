"""IngestionPipeline \u2014 drives an IngestionEnvelope through a LifeformSession.

Contract:

* Every successful chunk is sent to the session as a standalone turn
  via ``session.run_turn(chunk.text, trigger_kind=INGESTION)``. The
  kernel sees a vanilla apprenticeship turn with no ingestion-specific
  handling \u2014 the whole point of Gap 3 is that we do not carve out a
  parallel learning path, we just surface external content through
  the same canonical pipeline.
* Chunks with a non-empty ``parse_error`` are SKIPPED at the
  pipeline layer and recorded in the report's ``skipped_chunks``.
  The envelope already carries the authoritative failure list in
  ``partial_failures`` (enforced by envelope construction), but the
  report re-surfaces skips in order so an operator watching a
  long-running ingestion sees progress.
* Optionally closes the scene at the end so the kernel's R6
  session-post slow loop fires and consolidates the ingested
  material.
* Never reaches for any kernel owner store \u2014 the pipeline only
  calls session public methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from lifeform_core import TurnTriggerKind

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
)


_LOG = logging.getLogger("lifeform_ingestion.pipeline")


class _SessionLike(Protocol):
    """Structural protocol for the session the pipeline drives.

    We intentionally use a Protocol rather than importing
    ``lifeform_core.LifeformSession`` at type-check time so test
    harnesses can substitute a fake session without monkey-patching.
    The only methods the pipeline ever calls are listed here.
    """

    async def run_turn(
        self,
        user_input: str,
        *,
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
    ) -> Any: ...

    async def end_scene(
        self,
        *,
        reason: str = ...,
        drain_slow_loop: bool = ...,
    ) -> Any: ...


@dataclass(frozen=True)
class IngestionTurnRecord:
    """What the pipeline did with one chunk.

    Immutable audit entry. Either the chunk produced a turn (and
    ``turn_succeeded=True`` with the chunk's response text echoed for
    correlation) or it was skipped (``skipped_reason`` non-empty with
    either ``"parse_error"`` or a per-chunk kernel exception class
    name).
    """

    chunk_id: str
    locator: str
    turn_succeeded: bool
    response_text_snippet: str = ""
    skipped_reason: str = ""


@dataclass(frozen=True)
class IngestionReport:
    """Aggregate outcome of one ``process_envelope`` call.

    Consumers treat this as a frozen audit record: it lists every
    chunk in order, the aggregate counts, and the scene close
    outcome. Keeping it shaped this way means a service endpoint
    that ingested a TeachingCase can return it verbatim to the
    operator for "what happened?".
    """

    envelope_id: str
    total_chunks: int
    processed_chunks: int
    skipped_chunks: int
    ended_scene: bool
    turns: tuple[IngestionTurnRecord, ...]

    @property
    def all_succeeded(self) -> bool:
        return self.skipped_chunks == 0 and self.processed_chunks == self.total_chunks


class IngestionPipeline:
    """Drives an ``IngestionEnvelope`` through a ``LifeformSession``.

    Stateless: construct once per host process and call
    ``process_envelope`` for each envelope. The pipeline does not
    retain any state between envelopes \u2014 audit trail lives in the
    envelope / returned ``IngestionReport`` / kernel owner snapshots
    only.
    """

    async def process_envelope(
        self,
        env: IngestionEnvelope,
        *,
        session: _SessionLike,
        end_scene_after: bool = True,
        scene_end_reason: str = "ingestion-end",
        scene_end_drains_slow_loop: bool = True,
    ) -> IngestionReport:
        """Feed every successful chunk through ``session.run_turn``.

        Chunks with ``parse_error`` are recorded as skipped and never
        sent to the kernel. After all chunks complete, optionally
        calls ``session.end_scene`` so the kernel's session-post
        slow loop fires on the ingested material.

        Per-chunk kernel exceptions are caught and recorded as a
        skipped chunk with ``skipped_reason`` set to the exception
        class name; the envelope's remaining chunks still run.
        A per-chunk failure does NOT poison subsequent chunks; the
        only way to stop the pipeline mid-envelope is the caller's
        cancellation of the awaited task.
        """
        # Determine the trigger kind from the envelope's compliance
        # profile. FORCED -> INGESTION (vitals apprentice override
        # active); CONSULTATIVE -> USER_INPUT (normal turn).
        trigger_kind = (
            TurnTriggerKind.INGESTION
            if env.compliance_profile is IngestionComplianceProfile.FORCED
            else TurnTriggerKind.USER_INPUT
        )

        records: list[IngestionTurnRecord] = []
        processed = 0
        skipped = 0

        for chunk in env.chunks:
            if chunk.has_parse_error:
                skipped += 1
                records.append(
                    IngestionTurnRecord(
                        chunk_id=chunk.chunk_id,
                        locator=chunk.locator,
                        turn_succeeded=False,
                        skipped_reason="parse_error",
                    )
                )
                continue
            try:
                result = await session.run_turn(
                    chunk.text, trigger_kind=trigger_kind
                )
            except Exception as exc:  # noqa: BLE001 \u2014 ingestion isolation boundary
                _LOG.exception(
                    "IngestionPipeline: kernel raised on chunk %s of envelope %s",
                    chunk.chunk_id,
                    env.envelope_id,
                )
                skipped += 1
                records.append(
                    IngestionTurnRecord(
                        chunk_id=chunk.chunk_id,
                        locator=chunk.locator,
                        turn_succeeded=False,
                        skipped_reason=type(exc).__name__,
                    )
                )
                continue
            processed += 1
            response_text = _extract_response_snippet(result)
            records.append(
                IngestionTurnRecord(
                    chunk_id=chunk.chunk_id,
                    locator=chunk.locator,
                    turn_succeeded=True,
                    response_text_snippet=response_text,
                )
            )

        ended_scene = False
        if end_scene_after:
            closed = await session.end_scene(
                reason=scene_end_reason,
                drain_slow_loop=scene_end_drains_slow_loop,
            )
            ended_scene = closed is not None

        return IngestionReport(
            envelope_id=env.envelope_id,
            total_chunks=env.total_chunks,
            processed_chunks=processed,
            skipped_chunks=skipped,
            ended_scene=ended_scene,
            turns=tuple(records),
        )


def _extract_response_snippet(run_turn_result: Any, *, max_chars: int = 160) -> str:
    """Pull a short response snippet from the kernel's AgentTurnResult.

    Kept defensive / tolerant because the pipeline needs to work with
    real kernel results AND test stubs. If the shape is unexpected,
    returns an empty string rather than raising \u2014 the snippet is a
    convenience audit field, not a contract surface.
    """
    response = getattr(run_turn_result, "response", None)
    if response is None:
        return ""
    text = getattr(response, "text", "")
    if not isinstance(text, str):
        return ""
    return text[:max_chars]


__all__ = [
    "IngestionPipeline",
    "IngestionReport",
    "IngestionTurnRecord",
]
