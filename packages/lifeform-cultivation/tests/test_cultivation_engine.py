"""Unit tests for the autonomous cultivation engine + readouts.

Uses a fake sink so the loop is exercised without a live kernel. The
fake returns a controllable regime per study turn so we can assert the
school-coherence readout and convergence gate behave as designed.
"""

from __future__ import annotations

import asyncio

from lifeform_cultivation import (
    CultivationCurriculum,
    CultivationEngine,
    CultivationSeed,
    ResearchDoc,
    StudyTurn,
    assess_coherence,
    build_charter_text,
    build_study_brief,
)


class FakeSink:
    """Records calls and returns scripted regimes per study turn."""

    def __init__(self, regimes: list[str], *, docs_per_research: int = 2) -> None:
        self._regimes = regimes
        self._docs_per_research = docs_per_research
        self.research_queries: list[str] = []
        self.ingested: list[str] = []
        self.study_briefs: list[str] = []
        self.reflections: list[str] = []
        self._turn = 0

    async def research(self, query: str, *, max_results: int):
        self.research_queries.append(query)
        return tuple(
            ResearchDoc(title=f"doc{i}", url=f"https://x/{i}", text="theory body")
            for i in range(min(self._docs_per_research, max_results))
        )

    async def ingest(self, *, corpus_text: str, source_uri: str) -> int:
        self.ingested.append(source_uri)
        return 1

    async def study(self, brief: str) -> StudyTurn:
        self.study_briefs.append(brief)
        regime = self._regimes[self._turn % len(self._regimes)] if self._regimes else ""
        self._turn += 1
        return StudyTurn(text="ok", active_regime=regime, active_abstract_action="")

    async def reflect(self, *, reason: str) -> None:
        self.reflections.append(reason)


def _curriculum(**overrides) -> CultivationCurriculum:
    base = dict(
        topics=("依恋理论", "认知行为", "游戏治疗"),
        target_cycles=12,
        docs_per_topic=2,
        reflect_every=2,
        coherence_threshold=0.7,
        min_cycles_for_convergence=4,
    )
    base.update(overrides)
    return CultivationCurriculum(**base)


def test_coherence_concentration():
    a = assess_coherence(["psychodynamic"] * 8 + ["cbt"] * 2)
    assert a.dominant_regime == "psychodynamic"
    assert a.distinct_regimes == 2
    assert abs(a.score - 0.8) < 1e-9
    assert a.total_observations == 10


def test_coherence_empty_is_zero():
    a = assess_coherence([])
    assert a.score == 0.0
    assert a.dominant_regime == ""
    assert a.total_observations == 0


def test_engine_seed_ingests_charter():
    seed = CultivationSeed(
        display_name="儿童心理专家",
        domain="儿童心理",
        role_archetype="临床儿童心理学家",
        focus="儿童情绪与行为问题",
    )
    sink = FakeSink(regimes=[])
    engine = CultivationEngine(curriculum=_curriculum(), domain=seed.domain)
    chunks = asyncio.run(engine.seed(sink, seed))
    assert chunks == 1
    assert sink.ingested == ["cultivation:charter"]
    assert sink.reflections == ["cultivation:charter-seeded"]


def test_engine_converges_on_single_school():
    sink = FakeSink(regimes=["psychodynamic"])  # always one regime
    engine = CultivationEngine(curriculum=_curriculum(), domain="儿童心理")
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=6))
    assert progress.cycles_completed == 6
    assert progress.coherence.score == 1.0
    assert progress.converged is True
    # research + study happened each cycle
    assert len(sink.research_queries) == 6
    assert len(sink.study_briefs) == 6
    # reflect_every=2 -> reflections on cycles 1,3,5 (0-indexed +1)
    assert len(sink.reflections) == 3


def test_engine_not_converged_when_mixed():
    sink = FakeSink(regimes=["a", "b", "c"])  # evenly mixed -> low concentration
    engine = CultivationEngine(curriculum=_curriculum(), domain="儿童心理")
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=6))
    assert progress.converged is False
    assert progress.coherence.score < 0.7


def test_engine_min_cycles_gate():
    sink = FakeSink(regimes=["solo"])  # perfectly coherent
    engine = CultivationEngine(
        curriculum=_curriculum(min_cycles_for_convergence=10),
        domain="儿童心理",
    )
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=5))
    # coherence is 1.0 but fewer than min cycles -> not converged yet
    assert progress.coherence.score == 1.0
    assert progress.converged is False


def test_run_cycles_accumulates_prior_regimes():
    sink = FakeSink(regimes=["x"])
    engine = CultivationEngine(curriculum=_curriculum(), domain="d")
    progress = asyncio.run(
        engine.run_cycles(
            sink, start_cycle=4, count=2, prior_regimes=("x", "x", "x", "x")
        )
    )
    assert progress.cycles_completed == 6
    assert len(progress.regime_history) == 6


def test_charter_and_brief_builders_are_short_and_typed():
    seed = CultivationSeed(
        display_name="N",
        domain="儿童心理",
        role_archetype="临床心理学家",
        value_boundaries=("不做医疗诊断",),
    )
    charter = build_charter_text(seed)
    assert "养成宪章" in charter
    assert "不做医疗诊断" in charter
    brief = build_study_brief(topic="依恋理论", domain="儿童心理")
    assert "依恋理论" in brief
    assert "儿童心理" in brief


def test_curriculum_round_trip():
    c = _curriculum()
    again = CultivationCurriculum.from_json(c.to_json())
    assert again.topics == c.topics
    assert again.coherence_threshold == c.coherence_threshold


def test_seed_round_trip_and_validation():
    seed = CultivationSeed(
        display_name="N", domain="d", role_archetype="r", focus="f"
    )
    again = CultivationSeed.from_json(seed.to_json())
    assert again == seed
    try:
        CultivationSeed.from_json({"display_name": "", "domain": "", "role_archetype": ""})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty seed")
