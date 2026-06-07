"""Unit tests for the protocol-based cultivation engine + readouts.

Uses a fake sink that records protocol uptake and returns a scripted
``ActiveMixtureSnapshot`` so the school-coherence readout and the
convergence gate can be asserted without a live kernel or LLM.
"""

from __future__ import annotations

import asyncio

import pytest

from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
)

from lifeform_cultivation import (
    CultivationCurriculum,
    CultivationDirection,
    CultivationEngine,
    CultivationSeed,
    MultiTrackProgress,
    ResearchDoc,
    StudyTurn,
    TrackRun,
    assess_coherence,
    assess_protocol_coherence,
    build_charter_text,
    build_identity_core_protocol,
    build_study_brief,
    is_identity_core,
    parse_directions,
    run_direction_tracks,
)


def _mixture(entries: list[tuple[str, float]], *, boundaries: int = 0):
    return ActiveMixtureSnapshot(
        active_protocols=tuple(
            ActiveProtocolEntry(protocol_id=pid, activation_weight=w)
            for pid, w in entries
        ),
        boundary_union_ids=tuple(f"b{i}" for i in range(boundaries)),
    )


class FakeSink:
    """Records calls; returns scripted regimes + a fixed active_mixture."""

    def __init__(
        self,
        *,
        regimes: list[str] | None = None,
        mixture=None,
        protocol_id_prefix: str = "uptake:theory",
    ) -> None:
        self._regimes = regimes or []
        self._mixture = mixture
        self._prefix = protocol_id_prefix
        self.research_queries: list[str] = []
        self.uptaken: list[str] = []
        self.ingested: list[str] = []
        self.study_briefs: list[str] = []
        self.reflections: list[str] = []
        self._turn = 0
        self._uptake_n = 0

    async def research(self, query: str, *, max_results: int):
        self.research_queries.append(query)
        return (ResearchDoc(title="d", url=f"https://x/{len(self.research_queries)}", text="body"),)

    async def uptake_protocol(self, *, corpus_text: str, source_label: str):
        self._uptake_n += 1
        pid = f"{self._prefix}-{self._uptake_n}"
        self.uptaken.append(pid)
        return pid

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

    def read_active_mixture(self):
        return self._mixture


def _curriculum(**overrides) -> CultivationCurriculum:
    base = dict(
        topics=("依恋理论", "认知行为", "游戏治疗"),
        target_cycles=12,
        docs_per_topic=1,
        reflect_every=2,
        coherence_threshold=0.7,
        min_cycles_for_convergence=4,
    )
    base.update(overrides)
    return CultivationCurriculum(**base)


# --- Identity Core ---------------------------------------------------------


def test_identity_core_builder_is_active_anchor_with_boundaries():
    seed = CultivationSeed(
        display_name="儿童心理专家",
        domain="儿童心理",
        role_archetype="临床儿童心理学家",
        value_boundaries=("不做医疗诊断", "不替代危机干预"),
    )
    proto = build_identity_core_protocol(seed)
    assert is_identity_core(proto.protocol_id)
    assert proto.review_status.value == "active"
    assert len(proto.boundary_contracts) == 2
    assert proto.activation_conditions.minimum_weight_floor > 0.0
    assert len(proto.strategy_priors) == 1
    assert proto.success_signals and proto.failure_signals


# --- Protocol coherence readout -------------------------------------------


def test_protocol_coherence_dominant_school():
    mix = _mixture(
        [
            ("cultivation-identity:x", 0.25),  # anchor excluded
            ("uptake:psychodynamic", 0.6),
            ("uptake:cbt", 0.15),
        ]
    )
    a = assess_protocol_coherence(mix)
    assert a.dominant_protocol == "uptake:psychodynamic"
    assert a.identity_core_present is True
    assert a.distinct_schools == 2
    assert abs(a.score - (0.6 / 0.75)) < 1e-9


def test_protocol_coherence_none_is_zero():
    a = assess_protocol_coherence(None)
    assert a.score == 0.0
    assert a.total_protocols == 0


def test_protocol_coherence_only_identity_is_unconverged():
    mix = _mixture([("cultivation-identity:x", 0.3)])
    a = assess_protocol_coherence(mix)
    assert a.distinct_schools == 0
    assert a.score == 0.0
    assert a.identity_core_present is True


# --- Engine convergence on the protocol mixture ---------------------------


def test_engine_converges_when_one_school_dominates():
    mix = _mixture(
        [("cultivation-identity:x", 0.25), ("uptake:psychodynamic", 0.9), ("uptake:cbt", 0.1)]
    )
    sink = FakeSink(mixture=mix)
    engine = CultivationEngine(curriculum=_curriculum(), domain="儿童心理")
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=6))
    assert progress.cycles_completed == 6
    assert progress.readout_kind == "protocol"
    assert progress.converged is True
    assert len(sink.uptaken) == 6  # one researched protocol per cycle
    assert len(progress.uptaken_protocols) == 6


def test_engine_not_converged_when_schools_mixed():
    mix = _mixture(
        [("uptake:a", 0.34), ("uptake:b", 0.33), ("uptake:c", 0.33)]
    )
    sink = FakeSink(mixture=mix)
    engine = CultivationEngine(curriculum=_curriculum(), domain="儿童心理")
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=6))
    assert progress.readout_kind == "protocol"
    assert progress.converged is False
    assert progress.coherence_score < 0.7


def test_engine_falls_back_to_regime_when_no_mixture():
    sink = FakeSink(regimes=["solo"], mixture=None)
    engine = CultivationEngine(curriculum=_curriculum(), domain="儿童心理")
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=6))
    assert progress.readout_kind == "regime_fallback"
    assert progress.coherence_score == 1.0
    assert progress.converged is True


def test_engine_min_cycles_gate():
    mix = _mixture([("uptake:solo", 1.0)])
    sink = FakeSink(mixture=mix)
    engine = CultivationEngine(
        curriculum=_curriculum(min_cycles_for_convergence=10), domain="d"
    )
    progress = asyncio.run(engine.run_cycles(sink, start_cycle=0, count=5))
    assert progress.coherence_score == 1.0
    assert progress.converged is False


def test_engine_seed_ingests_charter():
    seed = CultivationSeed(
        display_name="儿童心理专家", domain="儿童心理", role_archetype="临床儿童心理学家"
    )
    sink = FakeSink(mixture=None)
    engine = CultivationEngine(curriculum=_curriculum(), domain=seed.domain)
    chunks = asyncio.run(engine.seed(sink, seed))
    assert chunks == 1
    assert sink.ingested == ["cultivation:charter"]
    assert sink.reflections == ["cultivation:charter-seeded"]


# --- Legacy regime readout retained ---------------------------------------


def test_regime_coherence_still_available():
    a = assess_coherence(["pd"] * 8 + ["cbt"] * 2)
    assert a.dominant_regime == "pd"
    assert abs(a.score - 0.8) < 1e-9


def test_charter_and_brief_builders():
    seed = CultivationSeed(
        display_name="N", domain="儿童心理", role_archetype="临床心理学家",
        value_boundaries=("不做医疗诊断",),
    )
    assert "养成宪章" in build_charter_text(seed)
    assert "依恋理论" in build_study_brief(topic="依恋理论", domain="儿童心理")


def test_curriculum_and_seed_round_trip():
    c = _curriculum()
    assert CultivationCurriculum.from_json(c.to_json()).topics == c.topics
    seed = CultivationSeed(display_name="N", domain="d", role_archetype="r")
    assert CultivationSeed.from_json(seed.to_json()) == seed


# --- Multi-direction cultivation ------------------------------------------


def test_direction_to_curriculum_inherits_base_cadence():
    base = _curriculum(docs_per_topic=5, reflect_every=3, target_cycles=30)
    direction = CultivationDirection(
        track_id="cbt",
        display_name="CBT",
        topics=("认知偏差", "行为"),
        coherence_threshold=0.8,
        min_cycles_for_convergence=6,
    )
    cur = direction.to_curriculum(base=base)
    assert cur.topics == ("认知偏差", "行为")
    assert cur.docs_per_topic == 5  # inherited from base
    assert cur.reflect_every == 3  # inherited from base
    assert cur.coherence_threshold == 0.8  # direction override
    assert cur.min_cycles_for_convergence == 6  # direction override


def test_parse_directions_round_trip_and_dedup():
    parsed = parse_directions(
        [
            {"track_id": "pd", "display_name": "PD", "topics": ["a"]},
            {"track_id": "cbt", "display_name": "CBT", "topics": ["b"]},
        ]
    )
    assert [d.track_id for d in parsed] == ["pd", "cbt"]
    assert CultivationDirection.from_json(parsed[0].to_json()) == parsed[0]


def test_parse_directions_empty_when_absent():
    assert parse_directions(None) == ()
    assert parse_directions({}) == ()


def test_parse_directions_rejects_duplicate_track_ids():
    with pytest.raises(ValueError):
        parse_directions(
            [
                {"track_id": "dup", "topics": ["a"]},
                {"track_id": "dup", "topics": ["b"]},
            ]
        )


def test_parse_directions_rejects_bad_track_id():
    with pytest.raises(ValueError):
        parse_directions([{"track_id": "Bad ID", "topics": ["a"]}])


def test_run_direction_tracks_isolates_each_school():
    converged_mix = _mixture(
        [("cultivation-identity:x", 0.25), ("uptake:psychodynamic", 0.9)]
    )
    mixed_mix = _mixture(
        [("uptake:a", 0.34), ("uptake:b", 0.33), ("uptake:c", 0.33)]
    )
    runs = (
        TrackRun(
            direction=CultivationDirection(
                track_id="pd",
                display_name="PD",
                topics=("依恋",),
                min_cycles_for_convergence=4,
            ),
            sink=FakeSink(mixture=converged_mix),
        ),
        TrackRun(
            direction=CultivationDirection(
                track_id="cbt",
                display_name="CBT",
                topics=("认知",),
                min_cycles_for_convergence=4,
            ),
            sink=FakeSink(mixture=mixed_mix),
        ),
    )
    result = asyncio.run(
        run_direction_tracks(runs, domain="儿童心理", count=6)
    )
    assert isinstance(result, MultiTrackProgress)
    assert len(result.tracks) == 2
    by_id = {t.track_id: t for t in result.tracks}
    assert by_id["pd"].progress.converged is True
    assert by_id["cbt"].progress.converged is False
    assert result.converged_track_ids == ("pd",)
    assert result.all_converged is False
