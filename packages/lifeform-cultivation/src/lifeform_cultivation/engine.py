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

from lifeform_cultivation.coherence import CoherenceAssessment, assess_coherence
from lifeform_cultivation.curriculum import (
    CultivationCurriculum,
    CultivationSeed,
    build_charter_text,
    build_study_brief,
)
from lifeform_cultivation.sink import CultivationSink


@dataclass(frozen=True)
class CycleEvent:
    cycle_index: int
    topic: str
    docs_ingested: int
    chunks_ingested: int
    active_regime: str
    reflected: bool

    def to_json(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "topic": self.topic,
            "docs_ingested": self.docs_ingested,
            "chunks_ingested": self.chunks_ingested,
            "active_regime": self.active_regime,
            "reflected": self.reflected,
        }


@dataclass(frozen=True)
class CultivationProgress:
    cycles_completed: int
    coherence: CoherenceAssessment
    regime_history: tuple[str, ...]
    events: tuple[CycleEvent, ...]
    converged: bool

    def to_json(self) -> dict[str, object]:
        return {
            "cycles_completed": self.cycles_completed,
            "coherence": self.coherence.to_json(),
            "regime_history": list(self.regime_history),
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
        events: list[CycleEvent] = []
        reflect_every = max(1, self._curriculum.reflect_every)

        for i in range(max(0, count)):
            cycle_index = start_cycle + i
            topic = self._curriculum.topic_for_cycle(cycle_index)
            query = self._curriculum.research_query(cycle_index)

            docs = await sink.research(
                query, max_results=self._curriculum.docs_per_topic
            )
            chunks_ingested = 0
            for doc in docs:
                corpus_text = f"{doc.title}\n来源: {doc.url}\n\n{doc.text}"
                chunks_ingested += await sink.ingest(
                    corpus_text=corpus_text, source_uri=doc.url
                )

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
                    docs_ingested=len(docs),
                    chunks_ingested=chunks_ingested,
                    active_regime=turn.active_regime,
                    reflected=reflected,
                )
            )

        cycles_completed = start_cycle + max(0, count)
        coherence = assess_coherence(regimes)
        converged = (
            cycles_completed >= self._curriculum.min_cycles_for_convergence
            and coherence.score >= self._curriculum.coherence_threshold
        )
        return CultivationProgress(
            cycles_completed=cycles_completed,
            coherence=coherence,
            regime_history=tuple(regimes),
            events=tuple(events),
            converged=converged,
        )


__all__ = [
    "CultivationEngine",
    "CultivationProgress",
    "CycleEvent",
]
