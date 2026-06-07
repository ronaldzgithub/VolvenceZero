"""Autonomous cultivation engine.

The engine drives the low-interaction self-study loop. Each *cycle* is:

    research(topic) -> ingest(corpus) -> study(apprentice turn)
                    -> (periodically) reflect (R6 slow-loop drain)

and after every cycle the engine re-reads the school-coherence readout
from the accumulated ``active_regime`` sequence. The engine holds no
cognitive state and writes none: it only sequences the canonical kernel
intake operations exposed by a :class:`CultivationSink` and aggregates
the readout. Convergence onto a single school is a *readout* decision
(coherence over threshold after a minimum number of cycles), never a
gradient the engine pushes back into the kernel.

This is intentionally NOT a token-space policy loop (R4): there is no
prompt that tells the kernel which school to pick. The expert forms its
school from the lived experience of researching and reconciling the
material; the engine just keeps the experience flowing and watches the
regime concentration settle.
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_cultivation.coherence import (
    assess_coherence,
    assess_protocol_coherence,
)
from lifeform_cultivation.curriculum import (
    CultivationCurriculum,
    CultivationDirection,
    CultivationSeed,
    build_charter_text,
    build_study_brief,
)
from lifeform_cultivation.sink import CultivationSink


@dataclass(frozen=True)
class CycleEvent:
    cycle_index: int
    topic: str
    docs_researched: int
    protocols_uptaken: tuple[str, ...]
    active_regime: str
    reflected: bool

    def to_json(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "topic": self.topic,
            "docs_researched": self.docs_researched,
            "protocols_uptaken": list(self.protocols_uptaken),
            "active_regime": self.active_regime,
            "reflected": self.reflected,
        }


@dataclass(frozen=True)
class CultivationProgress:
    cycles_completed: int
    coherence_score: float
    coherence_detail: dict
    readout_kind: str  # "protocol" | "regime_fallback"
    regime_history: tuple[str, ...]
    uptaken_protocols: tuple[str, ...]
    events: tuple[CycleEvent, ...]
    converged: bool

    def to_json(self) -> dict[str, object]:
        return {
            "cycles_completed": self.cycles_completed,
            "coherence_score": self.coherence_score,
            "coherence_detail": dict(self.coherence_detail),
            "readout_kind": self.readout_kind,
            "regime_history": list(self.regime_history),
            "uptaken_protocols": list(self.uptaken_protocols),
            "events": [e.to_json() for e in self.events],
            "converged": self.converged,
        }


class CultivationEngine:
    def __init__(
        self,
        *,
        curriculum: CultivationCurriculum,
        domain: str,
    ) -> None:
        self._curriculum = curriculum
        self._domain = domain

    async def seed(self, sink: CultivationSink, seed: CultivationSeed) -> int:
        """Ingest the seed charter as the cultivation's first experience.

        Returns the number of chunks ingested. The charter is plain
        ingestion material (not a prompt rule): it gives the kernel the
        rough identity + convergence intent to grow from.
        """

        charter = build_charter_text(seed)
        chunks = await sink.ingest(
            corpus_text=charter, source_uri="cultivation:charter"
        )
        await sink.reflect(reason="cultivation:charter-seeded")
        return chunks

    async def run_cycles(
        self,
        sink: CultivationSink,
        *,
        start_cycle: int,
        count: int,
        prior_regimes: tuple[str, ...] = (),
    ) -> CultivationProgress:
        """Run ``count`` autonomous study cycles starting at ``start_cycle``.

        ``prior_regimes`` carries the regime history observed in earlier
        ticks so coherence accumulates across the whole cultivation, not
        just this tick. Returns the updated progress (the caller
        persists it on the cultivation record).
        """

        regimes: list[str] = list(prior_regimes)
        uptaken: list[str] = []
        events: list[CycleEvent] = []
        reflect_every = max(1, self._curriculum.reflect_every)

        for i in range(max(0, count)):
            cycle_index = start_cycle + i
            topic = self._curriculum.topic_for_cycle(cycle_index)
            query = self._curriculum.research_query(cycle_index)

            docs = await sink.research(
                query, max_results=self._curriculum.docs_per_topic
            )
            cycle_protocols: list[str] = []
            for doc in docs:
                corpus_text = f"{doc.title}\n来源: {doc.url}\n\n{doc.text}"
                # Re-home: researched theory -> BehaviorProtocol (competes
                # in the active mixture on PE utility), NOT raw corpus.
                protocol_id = await sink.uptake_protocol(
                    corpus_text=corpus_text, source_label=doc.url
                )
                if protocol_id:
                    cycle_protocols.append(protocol_id)
            uptaken.extend(cycle_protocols)

            brief = build_study_brief(topic=topic, domain=self._domain)
            turn = await sink.study(brief)
            if turn.active_regime:
                regimes.append(turn.active_regime)

            reflected = (cycle_index + 1) % reflect_every == 0
            if reflected:
                await sink.reflect(reason=f"cultivation:cycle:{cycle_index}")

            events.append(
                CycleEvent(
                    cycle_index=cycle_index,
                    topic=topic,
                    docs_researched=len(docs),
                    protocols_uptaken=tuple(cycle_protocols),
                    active_regime=turn.active_regime,
                    reflected=reflected,
                )
            )

        cycles_completed = start_cycle + max(0, count)

        # Primary readout: protocol active-mixture convergence (the school
        # lives in the mixture, not the regime label). Fall back to regime
        # concentration only when the protocol runtime is not publishing
        # the active_mixture slot in this environment.
        mixture = sink.read_active_mixture()
        protocol_coherence = assess_protocol_coherence(mixture)
        if protocol_coherence.total_protocols > 0:
            score = protocol_coherence.score
            detail = protocol_coherence.to_json()
            readout_kind = "protocol"
            has_school = protocol_coherence.distinct_schools >= 1
        else:
            regime_coherence = assess_coherence(regimes)
            score = regime_coherence.score
            detail = regime_coherence.to_json()
            readout_kind = "regime_fallback"
            has_school = regime_coherence.total_observations > 0

        converged = (
            has_school
            and cycles_completed >= self._curriculum.min_cycles_for_convergence
            and score >= self._curriculum.coherence_threshold
        )
        return CultivationProgress(
            cycles_completed=cycles_completed,
            coherence_score=score,
            coherence_detail=detail,
            readout_kind=readout_kind,
            regime_history=tuple(regimes),
            uptaken_protocols=tuple(uptaken),
            events=tuple(events),
            converged=converged,
        )


# ---------------------------------------------------------------------------
# Multi-direction orchestration
# ---------------------------------------------------------------------------
#
# A single seed can fan out into several DIRECTIONS, each growing its own
# internally-consistent school. Each direction runs the *same* engine
# loop against its *own* sink (its own kernel session / ai_id), so the
# schools never cross-contaminate — there is no shared mutable cognitive
# state between tracks; the only thing they share is the seed's Identity
# Core anchor (loaded into each track independently by the caller).
#
# This orchestrator is pure: the caller supplies a bound sink per track
# (the platform layer binds a real per-ai_id session), so it stays inside
# the wheel boundary (no session / ai_id ownership here) and is testable
# with fake sinks.


@dataclass(frozen=True)
class TrackRun:
    """One direction + its bound (per-track) sink and resume state."""

    direction: CultivationDirection
    sink: CultivationSink
    start_cycle: int = 0
    prior_regimes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackProgress:
    track_id: str
    display_name: str
    progress: CultivationProgress

    def to_json(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "display_name": self.display_name,
            "progress": self.progress.to_json(),
        }


@dataclass(frozen=True)
class MultiTrackProgress:
    """Aggregate readout across every direction's track."""

    tracks: tuple[TrackProgress, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "tracks": [t.to_json() for t in self.tracks],
            "converged_track_ids": list(self.converged_track_ids),
            "all_converged": self.all_converged,
        }

    @property
    def converged_track_ids(self) -> tuple[str, ...]:
        return tuple(t.track_id for t in self.tracks if t.progress.converged)

    @property
    def all_converged(self) -> bool:
        return bool(self.tracks) and all(
            t.progress.converged for t in self.tracks
        )


async def run_direction_tracks(
    runs: tuple[TrackRun, ...],
    *,
    domain: str,
    count: int,
    base_curriculum: CultivationCurriculum | None = None,
) -> MultiTrackProgress:
    """Run ``count`` study cycles for each direction, independently.

    Each :class:`TrackRun` carries its own sink (bound by the caller to a
    distinct kernel session), so the per-track active-mixture readout and
    exam evidence are isolated. Returns a :class:`MultiTrackProgress`
    aggregating each track's :class:`CultivationProgress`.
    """

    results: list[TrackProgress] = []
    for run in runs:
        curriculum = run.direction.to_curriculum(base=base_curriculum)
        engine = CultivationEngine(curriculum=curriculum, domain=domain)
        progress = await engine.run_cycles(
            run.sink,
            start_cycle=run.start_cycle,
            count=count,
            prior_regimes=run.prior_regimes,
        )
        results.append(
            TrackProgress(
                track_id=run.direction.track_id,
                display_name=run.direction.display_name,
                progress=progress,
            )
        )
    return MultiTrackProgress(tracks=tuple(results))


__all__ = [
    "CultivationEngine",
    "CultivationProgress",
    "CycleEvent",
    "MultiTrackProgress",
    "TrackProgress",
    "TrackRun",
    "run_direction_tracks",
]
