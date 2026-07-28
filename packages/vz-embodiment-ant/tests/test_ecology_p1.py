"""Small-budget P1 schema and fixed-gate smoke."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from volvence_ant.env.world_objects import BurningMatch, WoodStick
from volvence_ant.experiments.ecology_curriculum import (
    EcologyDataSplit,
    EcologyEvaluationScenario,
    EcologyStage,
    EcologyTrainingTier,
    _session_config,
    _world,
    ecology_schedule_milestone_shortfalls,
    ecology_training_min_stage_rounds,
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_ARM_NAMES,
    ECOLOGY_P1_FORMAL_LATENT_DIM,
    ECOLOGY_P1_FORMAL_MIN_ANTS,
    ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER,
    ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS,
    ECOLOGY_P1_GATE_NAMES,
    ECOLOGY_P1_REGIME_DETERMINISTIC,
    ECOLOGY_P1_SCHEMA_VERSION,
    ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX,
    EcologyP1Config,
    EcologyP1LayoutResult,
    EcologyP1ProgressPaused,
    _archive_state_fingerprints,
    _assert_repeat_reference_comparable,
    _curriculum_config,
    _direction_signature,
    _evaluation_specs,
    _fixed_schedule,
    _frozen_learned_fingerprints,
    _regime_row_from_dict,
    _repeat_run_same_direction_gate,
    _run_regime_layout,
    _scenario_stage,
    _temporal_non_timeout_closure_gate,
    _verify_p1_checkpoint_archives,
    ecology_p1_formal_budget_failures,
    load_p1_repeat_reference,
    run_ecology_p1,
    run_ecology_p1_diagnostics,
)
from volvence_ant.experiments.ecology_probe import (
    ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET,
    ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY,
    _max_consecutive_approach_steps,
    ecology_probe_world_config,
    run_ecology_checkpoint_post_pickup_uturn_probes,
)
from volvence_ant.runtime import KernelColonyRunner

_TINY = EcologyP1Config(
    n_ants=1,
    temporal_latent_dim=4,
    training_rounds=1,
    evaluation_rounds=3,
    layouts_per_tier=1,
    seed=7,
)
_FROZEN_BUDGET = EcologyP1Config(
    n_ants=ECOLOGY_P1_FORMAL_MIN_ANTS,
    temporal_latent_dim=ECOLOGY_P1_FORMAL_LATENT_DIM,
    training_rounds=ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS,
    evaluation_rounds=ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    layouts_per_tier=ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER,
    seed=0,
)


def _bootstrap_checkpoints(config: EcologyP1Config):
    """Untrained colony checkpoints; no rollout, no training."""

    curriculum = _curriculum_config(config)
    runner = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=config.seed,
            session_id="test:p1:bootstrap",
            optimize=True,
        ),
    )
    return runner.export_learning_checkpoints(
        checkpoint_prefix="test:p1:bootstrap",
        include_runtime_replay=False,
    )


def _layout(
    *,
    arm: str,
    capability: str,
    seed: int = 1,
    layout_success: bool = False,
    switch_count: int = 0,
    non_timeout_segment_closures: int = 0,
) -> EcologyP1LayoutResult:
    return EcologyP1LayoutResult(
        arm=arm,
        capability=capability,
        seed=seed,
        tier="far",
        successful_bodies=1 if layout_success else 0,
        required_bodies=1,
        layout_success=layout_success,
        harmful_tick_rate=0.0,
        escape_latencies=(),
        switch_count=switch_count,
        non_timeout_segment_closures=non_timeout_segment_closures,
        policy_fingerprint_stable=True,
        temporal_learning_fingerprint_stable=True,
        replay_settlement_coverage=1.0,
        replay_lineage_coverage=1.0,
        replay_drop_count=0,
    )


async def test_p1_bounded_work_budget_pauses_on_committed_episode(
    tmp_path,
) -> None:
    config = EcologyP1Config(
        n_ants=1,
        temporal_latent_dim=4,
        training_rounds=1,
        evaluation_rounds=1,
        layouts_per_tier=1,
        seed=17,
    )
    progress_dir = tmp_path / "bounded"

    with pytest.raises(EcologyP1ProgressPaused) as first:
        await run_ecology_p1(
            config,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    assert first.value.completed_work_items == 1
    state = json.loads(
        (progress_dir / "learned.json").read_text(encoding="utf-8")
    )
    assert state["completed_training_episodes"] == 1
    assert state["training_complete"] is False
    assert not (progress_dir / "no_optimize.json").exists()

    with pytest.raises(EcologyP1ProgressPaused):
        await run_ecology_p1(
            config,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    state = json.loads(
        (progress_dir / "learned.json").read_text(encoding="utf-8")
    )
    assert state["completed_training_episodes"] == 2


def test_p1_diagnostics_are_checkpoint_free_and_structured() -> None:
    report = run_ecology_p1_diagnostics(
        EcologyP1Config(
            n_ants=1,
            temporal_latent_dim=4,
            training_rounds=1,
            evaluation_rounds=3,
            layouts_per_tier=1,
            seed=7,
        )
    )

    # v3: ``heat_route_foraging`` now grades HEAT_ROUTE_AVOIDANCE (butter +
    # burning match, no wood stick) instead of a byte-identical copy of
    # COMPOSITE/FAR, so a v2 diagnostics report describes different layouts.
    # Pinned as a literal on purpose -- importing the constant would make this
    # assertion follow any future bump instead of failing on it.
    assert report.schema_version == "digital-ant-ecology-p1-diagnostics.v3"
    assert len(report.results) == 18
    assert len(report.oracle_success_by_capability) == 6


async def test_p1_uses_fixed_schedule_per_body_mastery(
    tmp_path,
) -> None:
    config = EcologyP1Config(
        n_ants=1,
        temporal_latent_dim=4,
        training_rounds=1,
        evaluation_rounds=3,
        layouts_per_tier=1,
        seed=7,
    )
    progress_dir = tmp_path / "progress"
    report = await run_ecology_p1(
        config,
        progress_dir=progress_dir,
    )

    assert report.schema_version == ECOLOGY_P1_SCHEMA_VERSION
    assert report.verdict in {"PASS", "BLOCK"}
    assert tuple(gate.name for gate in report.gates) == ECOLOGY_P1_GATE_NAMES
    assert {item.arm for item in report.layout_results} == set(
        ECOLOGY_P1_ARM_NAMES
    )
    assert len(report.layout_results) == len(ECOLOGY_P1_ARM_NAMES) * 6
    assert len(report.diagnostic_results) == 3 * 6
    assert {item.controller for item in report.diagnostic_results} == {
        "oracle_steering",
        "fixed_rule",
        "random",
    }
    assert all(item.required_bodies == 1 for item in report.layout_results)
    assert all(item.policy_fingerprint_stable for item in report.layout_results)

    # The four fields the report gained must actually be PASSED and
    # SERIALISED; before this they were computed and thrown away, and the
    # constructor call raised TypeError.
    assert {summary.source for summary in report.escape_latency_summaries} == {
        "learned",
        "random",
    }
    assert len(report.regime_diagnostic) == 2 * 6
    assert len(report.regime_gap_summary) == 6
    assert report.repeat_reference is None
    payload = report.to_dict()
    for field in (
        "escape_latency_summaries",
        "regime_diagnostic",
        "regime_gap_summary",
        "repeat_reference",
        "post_pickup_uturn_probes",
    ):
        assert field in payload, field
    assert len(report.post_pickup_uturn_probes) == config.n_ants
    assert all(
        len(probe.lanes) == 2
        for probe in report.post_pickup_uturn_probes
    )
    # allow_nan=False: an unserialisable sentinel must fail here, not in the
    # artifact writer after a formal run has already been spent.
    round_tripped = json.loads(
        json.dumps(payload, allow_nan=False, default=lambda item: item.value)
    )
    assert len(round_tripped["regime_diagnostic"]) == 2 * 6
    assert tuple(
        _regime_row_from_dict(row)
        for row in round_tripped["regime_diagnostic"]
    ) == report.regime_diagnostic
    # Memory drift under learning_enabled=False is published as evidence on
    # every regime row instead of aborting the diagnostic.
    assert all(
        row.memory_fingerprint_stable == (not row.drifted_memory_bodies)
        for row in report.regime_diagnostic
    )
    assert any(
        not row.memory_fingerprint_stable
        for row in report.regime_diagnostic
    )
    # The two gates that were declared but never built now carry real rows.
    gates = {gate.name: gate for gate in report.gates}
    assert gates["checkpoint_archive_roundtrip"].passed is True
    assert gates["repeat_run_same_direction"].passed is False
    assert (
        "no repeat reference report supplied"
        in gates["repeat_run_same_direction"].observed
    )
    # A tiny-budget run can never report PASS, whatever the capabilities do.
    assert gates["formal_configuration"].passed is False
    assert report.verdict == "BLOCK"
    assert report.schedule[:3] and all(
        item.tier.value == "near" for item in report.schedule[:3]
    )
    assert sum(item.forced_return for item in report.schedule) == (
        2 * config.layouts_per_tier
    )
    assert sum(
        item.forced_return and item.interleaved
        for item in report.schedule
    ) == config.layouts_per_tier
    assert sum(item.forced_approach for item in report.schedule) == 1
    assert all(
        item.stage.value == "butter" and item.tier.value == "near"
        for item in report.schedule
        if item.forced_approach
    )
    for arm in ECOLOGY_P1_ARM_NAMES:
        state = json.loads(
            (progress_dir / f"{arm}.json").read_text(
                encoding="utf-8"
            )
        )
        assert state["training_complete"] is True
        assert state["completed_training_episodes"] == (
            0 if arm == "cold" else len(report.schedule)
        )
    evaluation_state = json.loads(
        (progress_dir / "evaluations.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(evaluation_state["layout_results"]) == (
        len(ECOLOGY_P1_ARM_NAMES) * 6
    )

    # Rewind one arm to its penultimate immutable episode archive. Resume
    # must execute only the missing suffix and converge to the same report.
    learned_state_path = progress_dir / "learned.json"
    learned_state = json.loads(
        learned_state_path.read_text(encoding="utf-8")
    )
    penultimate = progress_dir / (
        f"learned.slot-{(len(report.schedule) - 1) % 2}.vzac"
    )
    learned_state.update(
        {
            "completed_training_episodes": len(report.schedule) - 1,
            "training_complete": False,
            "checkpoint_archive": penultimate.name,
            "checkpoint_sha256": hashlib.sha256(
                penultimate.read_bytes()
            ).hexdigest(),
        }
    )
    learned_state_path.write_text(
        json.dumps(
            learned_state,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    resumed = await run_ecology_p1(
        config,
        progress_dir=progress_dir,
    )
    assert resumed.to_dict() == report.to_dict()

    with pytest.raises(ValueError, match="progress mismatch"):
        await run_ecology_p1(
            EcologyP1Config(
                n_ants=1,
                temporal_latent_dim=4,
                training_rounds=1,
                evaluation_rounds=3,
                layouts_per_tier=1,
                seed=8,
            ),
            progress_dir=progress_dir,
        )


def test_fixed_schedule_interleaves_late_pickup_return_rehearsal() -> None:
    schedule = _fixed_schedule(_FROZEN_BUDGET)

    assert len(schedule) == 55
    assert tuple(
        item.episode_index
        for item in schedule
        if item.forced_return and item.interleaved
    ) == (38, 42, 46, 50, 54)
    assert all(
        item.stage is EcologyStage.BUTTER
        and item.tier is EcologyTrainingTier.NEAR
        for item in schedule
        if item.forced_return
    )
    assert sum(
        item.stage is EcologyStage.COMPOSITE
        and item.tier is EcologyTrainingTier.FAR
        and not item.interleaved
        for item in schedule
    ) == _FROZEN_BUDGET.layouts_per_tier


# ---------------------------------------------------------------------------
# formal_configuration: the budget predicate P2 must be able to re-derive
# ---------------------------------------------------------------------------


def test_formal_budget_fails_small_and_passes_frozen_budget() -> None:
    failures = ecology_p1_formal_budget_failures(_TINY)
    assert failures == (
        "n_ants=1<4",
        "temporal_latent_dim=4!=16",
        "layouts_per_tier=1<5",
        (
            "training_rounds=1<"
            f"{ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS}"
        ),
        "evaluation_rounds=3<120",
    )
    assert ecology_p1_formal_budget_failures(_FROZEN_BUDGET) == ()
    # Each threshold is enforced independently: one under-budget field is
    # enough to fail, so a run cannot trade held-out rounds for ants.
    for field, value in (
        ("n_ants", ECOLOGY_P1_FORMAL_MIN_ANTS - 1),
        ("layouts_per_tier", ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER - 1),
        ("training_rounds", ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS - 1),
        ("evaluation_rounds", ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS - 1),
        ("temporal_latent_dim", ECOLOGY_P1_FORMAL_LATENT_DIM + 1),
    ):
        under = replace(_FROZEN_BUDGET, **{field: value})
        assert len(ecology_p1_formal_budget_failures(under)) == 1, field


def test_formal_budget_is_a_predicate_over_a_written_config_mapping() -> None:
    """The hop the P2 loader needs (audit MAJOR 5).

    ``load_p1_prerequisite`` trusts the report's own
    ``formal_configuration.passed`` boolean and never re-derives the budget
    from the ``config`` block sitting in the same payload. This asserts the
    predicate is callable on exactly that mapping, so the P2 owner can close
    the hand-edited-report hole without duplicating a threshold.
    """

    frozen_payload = json.loads(json.dumps(asdict(_FROZEN_BUDGET)))
    assert ecology_p1_formal_budget_failures(frozen_payload) == ()

    # A hand-edited report: the gate row says passed, the config says smoke.
    forged = dict(frozen_payload)
    forged["n_ants"] = 1
    forged["layouts_per_tier"] = 1
    assert ecology_p1_formal_budget_failures(forged) == (
        "n_ants=1<4",
        "layouts_per_tier=1<5",
    )

    # A config that cannot be read is not a config whose budget is trusted.
    missing = {
        key: value
        for key, value in frozen_payload.items()
        if key != "evaluation_rounds"
    }
    with pytest.raises(ValueError, match="missing formal budget field"):
        ecology_p1_formal_budget_failures(missing)
    with pytest.raises(ValueError, match="must be an integer"):
        ecology_p1_formal_budget_failures(
            dict(frozen_payload, n_ants="4")
        )
    with pytest.raises(ValueError, match="must be an integer"):
        ecology_p1_formal_budget_failures(
            dict(frozen_payload, n_ants=True)
        )
    with pytest.raises(TypeError, match="config mapping"):
        ecology_p1_formal_budget_failures(["n_ants", 4])


# ---------------------------------------------------------------------------
# heat_route_foraging must grade a different scenario from composite
# ---------------------------------------------------------------------------


def test_heat_route_foraging_grades_a_different_scenario_than_composite() -> (
    None
):
    specs = {
        capability: (scenario, tier)
        for capability, scenario, tier in _evaluation_specs()
    }
    assert specs["heat_route_foraging"] == (
        EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE,
        EcologyTrainingTier.FAR,
    )
    assert specs["composite"] == (
        EcologyEvaluationScenario.COMPOSITE,
        EcologyTrainingTier.FAR,
    )
    assert len(specs) == len({spec for spec in specs.values()})

    # Not just a different enum: a different LAYOUT. Heat-route avoidance is
    # butter + burning match and NO wood stick; composite adds the stick.
    curriculum = _curriculum_config(_TINY)
    worlds = {}
    for capability in ("heat_route_foraging", "composite"):
        scenario, tier = specs[capability]
        worlds[capability] = _world(
            config=curriculum,
            stage=_scenario_stage(scenario),
            seed=4242,
            data_split=EcologyDataSplit.HELDOUT,
            tier=tier,
        ).world_objects()
    heat_route = worlds["heat_route_foraging"]
    composite = worlds["composite"]
    assert any(isinstance(item, BurningMatch) for item in heat_route)
    assert not any(isinstance(item, WoodStick) for item in heat_route)
    assert any(isinstance(item, BurningMatch) for item in composite)
    assert any(isinstance(item, WoodStick) for item in composite)


# ---------------------------------------------------------------------------
# temporal_non_timeout_closure: a per-layout ratio, in the right direction
# ---------------------------------------------------------------------------


def test_temporal_non_timeout_closure_is_a_per_layout_ratio() -> None:
    def rows(qualifying: int, total: int) -> tuple[EcologyP1LayoutResult, ...]:
        return tuple(
            _layout(
                arm="learned",
                capability="composite",
                seed=index,
                switch_count=1 if index < qualifying else 0,
                non_timeout_segment_closures=(
                    1 if index < qualifying else 0
                ),
            )
            for index in range(total)
        )

    assert ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX == pytest.approx(0.4)

    # 3/5 layouts qualify -> timeout-only rate 0.4, exactly at the ceiling.
    assert _temporal_non_timeout_closure_gate(rows(3, 5)).passed is True
    assert _temporal_non_timeout_closure_gate(rows(5, 5)).passed is True
    # 2/5 -> 0.6 timeout-only, above the ceiling.
    assert _temporal_non_timeout_closure_gate(rows(2, 5)).passed is False

    # THE conversion this replaces: one qualifying layout out of ten passed
    # the old aggregated existence test (sum(switch)>0 and sum(closures)>0).
    # It must now FAIL, otherwise the "ratio" is weaker than what it replaced.
    weak = rows(1, 10)
    assert sum(item.switch_count for item in weak) > 0
    assert sum(item.non_timeout_segment_closures for item in weak) > 0
    gate = _temporal_non_timeout_closure_gate(weak)
    assert gate.passed is False
    assert "timeout_only_layouts=9" in gate.observed

    # A layout needs BOTH facts itself; a switch here and a closure there is
    # exactly the aggregation defect.
    split = (
        _layout(
            arm="learned", capability="composite", seed=0, switch_count=5
        ),
        _layout(
            arm="learned",
            capability="composite",
            seed=1,
            non_timeout_segment_closures=5,
        ),
    )
    assert _temporal_non_timeout_closure_gate(split).passed is False

    # No learned held-out evidence at all is a failure, never a vacuous pass.
    assert _temporal_non_timeout_closure_gate(()).passed is False


# ---------------------------------------------------------------------------
# regime diagnostic: read-only against the frozen-LEARNED set; memory drift
# is published as evidence, not fatal
# ---------------------------------------------------------------------------


def test_frozen_learned_fingerprint_set_excludes_temporal_and_memory() -> None:
    checkpoints = _bootstrap_checkpoints(_TINY)
    drifted = tuple(
        replace(
            item,
            temporal_fingerprint="turn-local-mixture-moved",
            memory_fingerprint="memory-moved",
        )
        for item in checkpoints
    )
    # The established frozen-owner notion: policy + temporal-LEARNING only.
    # docs/specs/digital-ant-embodiment.md excludes the PE-driven turn-local
    # mixture from the temporal-learning fingerprint, so a moving
    # ``temporal_fingerprint`` is expected inference telemetry.
    assert _frozen_learned_fingerprints(
        drifted
    ) == _frozen_learned_fingerprints(checkpoints)
    # A real freeze violation still moves it.
    assert _frozen_learned_fingerprints(
        tuple(
            replace(item, policy_fingerprint="policy-moved")
            for item in checkpoints
        )
    ) != _frozen_learned_fingerprints(checkpoints)
    # The archive-identity set is deliberately the stricter full triple: an
    # export/restore roundtrip runs no rollout and must reproduce everything.
    assert _archive_state_fingerprints(
        drifted
    ) != _archive_state_fingerprints(checkpoints)


async def test_regime_diagnostic_publishes_memory_drift_instead_of_raising() -> (
    None
):
    checkpoints = _bootstrap_checkpoints(_TINY)
    capability, scenario, tier = _evaluation_specs()[0]
    row = await _run_regime_layout(
        config=_TINY,
        curriculum=_curriculum_config(_TINY),
        checkpoints=checkpoints,
        regime=ECOLOGY_P1_REGIME_DETERMINISTIC,
        capability=capability,
        scenario=scenario,
        tier=tier,
        seed=2_000_010,
    )

    assert row.regime == ECOLOGY_P1_REGIME_DETERMINISTIC
    assert row.capability == capability
    assert row.memory_fingerprint_stable == (not row.drifted_memory_bodies)
    # Measured today (1 ant, learning_enabled=False, optimize=False): the
    # memory fingerprint DOES move, driven by the credit / dual-track-gate /
    # joint-loop-memory / prediction / reflection / regime owners. The P0
    # ``frozen_evaluation`` gate owns that as a BLOCK. This diagnostic must
    # PUBLISH it and keep reporting -- if this assertion ever flips, the P0
    # defect was fixed and this evidence field should be re-examined, not
    # quietly dropped.
    assert row.drifted_memory_bodies == (0,)
    assert row.memory_fingerprint_stable is False

    # The row must survive the journal round trip byte-for-byte.
    assert _regime_row_from_dict(
        json.loads(json.dumps(asdict(row)))
    ) == row


# ---------------------------------------------------------------------------
# the two gates that were declared in ECOLOGY_P1_GATE_NAMES but never built
# ---------------------------------------------------------------------------


def test_checkpoint_archive_roundtrip_verifier_is_wired_and_real() -> None:
    assert "checkpoint_archive_roundtrip" in ECOLOGY_P1_GATE_NAMES
    passed, detail = _verify_p1_checkpoint_archives(
        config=_TINY,
        curriculum=_curriculum_config(_TINY),
        checkpoints=_bootstrap_checkpoints(_TINY),
    )
    assert passed is True, detail
    assert detail == "hydration verified and corrupt restore rolled back"


def _reference_payload(
    *,
    config: EcologyP1Config,
    seed: int,
    results: tuple[EcologyP1LayoutResult, ...],
    verdict: str = "BLOCK",
) -> dict[str, object]:
    return {
        "schema_version": ECOLOGY_P1_SCHEMA_VERSION,
        "config": dict(asdict(config), seed=seed),
        "gates": [
            {"name": name, "passed": False, "observed": "", "threshold": ""}
            for name in ECOLOGY_P1_GATE_NAMES
        ],
        "layout_results": [asdict(item) for item in results],
        "verdict": verdict,
    }


def _matrix(*, learned_wins: tuple[str, ...]) -> tuple[
    EcologyP1LayoutResult, ...
]:
    rows: list[EcologyP1LayoutResult] = []
    for capability, _, _ in _evaluation_specs():
        for arm in ("learned", "no_optimize", "cold"):
            rows.append(
                _layout(
                    arm=arm,
                    capability=capability,
                    layout_success=(
                        arm == "learned" and capability in learned_wins
                    ),
                )
            )
    return tuple(rows)


def test_repeat_gate_fails_without_a_reference_and_on_opposite_direction(
    tmp_path,
) -> None:
    results = _matrix(learned_wins=("butter_far", "composite"))

    # plan 4.7 is a conjunct: a single run is a NEGATIVE result here, never a
    # skipped or vacuously passing one.
    absent = _repeat_run_same_direction_gate(reference=None, results=results)
    assert absent.name == "repeat_run_same_direction"
    assert absent.passed is False
    assert "no repeat reference report supplied" in absent.observed

    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            _reference_payload(
                config=_TINY, seed=_TINY.seed + 1, results=results
            )
        ),
        encoding="utf-8",
    )
    reference = _assert_repeat_reference_comparable(
        config=_TINY, reference_path=reference_path, repo_root=None
    )
    assert reference.seed == _TINY.seed + 1
    assert reference.direction_signature == _direction_signature(results)
    assert _repeat_run_same_direction_gate(
        reference=reference, results=results
    ).passed is True

    # Same magnitude of success, opposite sign on one capability.
    flipped = _matrix(learned_wins=("butter_far",))
    opposite = _repeat_run_same_direction_gate(
        reference=reference, results=flipped
    )
    assert opposite.passed is False
    assert "'composite'" in opposite.observed

    # A repetition on the SAME training seed is not an independent repetition.
    same_seed = tmp_path / "same-seed.json"
    same_seed.write_text(
        json.dumps(
            _reference_payload(
                config=_TINY, seed=_TINY.seed, results=results
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="INDEPENDENT"):
        _assert_repeat_reference_comparable(
            config=_TINY, reference_path=same_seed, repo_root=None
        )

    # A reference produced under a different budget is not comparable.
    other_budget = tmp_path / "other-budget.json"
    other_budget.write_text(
        json.dumps(
            _reference_payload(
                config=replace(_TINY, evaluation_rounds=5),
                seed=_TINY.seed + 1,
                results=results,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different budget"):
        _assert_repeat_reference_comparable(
            config=_TINY, reference_path=other_budget, repo_root=None
        )

    # And a report from a retired schema is refused, never reinterpreted.
    stale = tmp_path / "stale.json"
    stale_payload = _reference_payload(
        config=_TINY, seed=_TINY.seed + 1, results=results
    )
    stale_payload["schema_version"] = "digital-ant-ecology-p1-development.v25"
    stale.write_text(json.dumps(stale_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_p1_repeat_reference(stale)


# ---------------------------------------------------------------------------
# the probe must measure the body the curriculum actually builds
# ---------------------------------------------------------------------------


def test_probe_world_matches_curriculum_sensor_geometry() -> None:
    curriculum_world = _world(
        config=_curriculum_config(_TINY),
        stage=EcologyStage.COMPOSITE,
        seed=11,
        data_split=EcologyDataSplit.HELDOUT,
        tier=EcologyTrainingTier.FAR,
    )
    probe_config = ecology_probe_world_config(seed=11)
    for field in (
        "antenna_offset_deg",
        "antenna_reach",
        "step_size",
        "nest_radius",
    ):
        assert getattr(probe_config, field) == getattr(
            curriculum_world.config, field
        ), field


def test_post_pickup_uturn_requires_sustained_distance_reduction() -> None:
    assert _max_consecutive_approach_steps((2.0, 1.9, 1.8, 1.7)) == 3
    assert _max_consecutive_approach_steps((2.0, 1.8, 1.9, 1.7)) == 1
    assert _max_consecutive_approach_steps((2.0, 2.0, 1.9, 1.8)) == 2


async def test_post_pickup_uturn_probe_is_real_balanced_and_frozen() -> None:
    probes = await run_ecology_checkpoint_post_pickup_uturn_probes(
        temporal_latent_dim=_TINY.temporal_latent_dim,
        seed=71,
        checkpoints=_bootstrap_checkpoints(_TINY),
    )

    assert len(probes) == _TINY.n_ants
    lanes = probes[0].lanes
    assert tuple(lane.side for lane in lanes) == ("left", "right")
    assert tuple(lane.heading_offset for lane in lanes) == pytest.approx(
        (
            ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET,
            -ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET,
        )
    )
    assert all(lane.picked_up for lane in lanes)
    assert all(lane.home_distances_after_pickup for lane in lanes)
    assert all(
        all(step >= 1 for step in lane.switch_steps_after_pickup)
        for lane in lanes
    )
    assert all(
        lane.post_pickup_switch_observed
        == (
            lane.first_post_pickup_switch_step is not None
            and lane.first_post_pickup_switch_step
            <= ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY
        )
        for lane in lanes
    )
    assert all(lane.policy_fingerprint_stable for lane in lanes)
    assert all(
        lane.temporal_learning_fingerprint_stable for lane in lanes
    )
    # A cold checkpoint must not receive credit merely for emitting an action
    # with the right sign on one tick.
    assert probes[0].passed is False


def test_milestone_budget_lever_is_bound_to_the_formal_budget_predicate() -> (
    None
):
    """The curriculum owner's THE-FAR-DECISION lever must not be hardcoded.

    ``EcologyCurriculumConfig.milestone_budget_enforced`` defaults to True and
    its owner documents False as the lever for "a plan section 7 small-budget
    diagnostic (never for a formal run)". P1 binds it to the same predicate
    the ``formal_configuration`` gate uses, so the milestone gate is fully
    enforced for every run that could produce a verdict, and only the tiny
    diagnostic tier -- which already fails ``formal_configuration`` -- turns
    it off.
    """

    curriculum = _curriculum_config(_FROZEN_BUDGET)
    assert curriculum.milestone_budget_enforced is True
    assert (
        ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS
        == ecology_training_min_stage_rounds()
    )
    assert curriculum.stage_rounds == ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS
    assert ecology_schedule_milestone_shortfalls(
        _fixed_schedule(_FROZEN_BUDGET),
        stage_rounds=curriculum.stage_rounds,
    ) == ()
    assert _curriculum_config(_TINY).milestone_budget_enforced is False
    # Any single under-budget field is enough to make the run a diagnostic.
    for field, value in (
        ("n_ants", ECOLOGY_P1_FORMAL_MIN_ANTS - 1),
        ("evaluation_rounds", ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS - 1),
    ):
        assert _curriculum_config(
            replace(_FROZEN_BUDGET, **{field: value})
        ).milestone_budget_enforced is False, field
