"""P2 confirmatory matrix: serial gate, matched levers, journal and verdict."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace

import pytest

from volvence_zero.runtime import WiringLevel

from volvence_ant.experiments.ecology_curriculum import (
    EcologyArmMetrics,
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyEvaluationScenario,
    EcologyStage,
    _ecology_outcome_score,
    _session_config,
    ecology_training_min_stage_rounds,
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_GATE_NAMES,
    ECOLOGY_P1_SCHEMA_VERSION,
    EcologyP1Config,
    _evaluation_specs,
    _fixed_schedule,
)
from volvence_ant.experiments import ecology_p2 as ecology_p2_module
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_ARM_LEVER_PARAMETERS,
    ECOLOGY_P2_ARM_NAMES,
    ECOLOGY_P2_ARM_SPEC_BY_NAME,
    ECOLOGY_P2_ARM_SPECS,
    ECOLOGY_P2_CORE_ARM_NAMES,
    ECOLOGY_P2_DIVERGENT_POLICY_ARM_NAMES,
    ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES,
    ECOLOGY_P2_GATE_NAMES,
    ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS,
    ECOLOGY_P2_PAIRED_COMPARISONS,
    ECOLOGY_P2_PRIMARY_ENDPOINTS,
    ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES,
    ECOLOGY_P2_SHARD_SCHEMA_VERSION,
    _ECOLOGY_P2_FROZEN_LOGIC,
    EcologyP2ArmLeverUnavailableError,
    EcologyP2Config,
    EcologyP2LayoutResult,
    EcologyP2Prerequisite,
    EcologyP2PrerequisiteError,
    EcologyP2ProbeSummary,
    EcologyP2ProgressPaused,
    EcologyP2ShardReport,
    EcologyP2SourceProvenance,
    _curriculum_config,
    _frozen_logic_source,
    _heldout_seed,
    _holm_adjusted,
    _hierarchical_paired_bootstrap,
    _scenario_stage,
    aggregate_ecology_p2_shards,
    heldout_layout_seeds,
    load_p1_prerequisite,
    load_shard_checkpoint_archives,
    outcome_score,
    p2_training_schedule,
    preregistration_digest,
    run_ecology_p2_preflight,
    run_ecology_p2_shard,
    shard_report_from_dict,
    unreachable_arm_levers,
)


_SOURCE = EcologyP2SourceProvenance(
    git_sha="0" * 40,
    git_branch="test",
    worktree_dirty=False,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_p1_report(
    path,
    *,
    verdict: str = "PASS",
    failed_gate: str | None = None,
    schema_version: str = ECOLOGY_P1_SCHEMA_VERSION,
    gate_names: tuple[str, ...] = ECOLOGY_P1_GATE_NAMES,
    training_seed: int = 17,
    config: EcologyP1Config | None = None,
    marker: str = "synthetic",
):
    """Synthetic P1 artifact.

    This exists only to exercise the P2 serial gate; it is never evidence that
    the real P1 matrix passed.

    The config block is a full ``EcologyP1Config`` because P2 re-derives the
    frozen P1 budget from it rather than trusting the report's own
    ``formal_configuration`` boolean.
    """

    resolved = replace(
        config or EcologyP1Config(), seed=training_seed
    )
    payload = {
        "schema_version": schema_version,
        "verdict": verdict,
        "config": asdict(resolved),
        "gates": [
            {
                "name": name,
                "passed": name != failed_gate,
                # ``marker`` distinguishes two runs of the SAME frozen
                # configuration -- their observed numbers differ, their
                # configuration identity does not.
                "observed": marker,
                "threshold": "synthetic",
            }
            for name in gate_names
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _small_config() -> EcologyP2Config:
    """The plan section 7 step 3 tier: one ant, one layout, fixed seed.

    ``training_rounds`` is the curriculum owner's PUBLISHED milestone floor,
    not a literal and not a smaller number with the owner's budget guard
    switched off: a schedule whose episodes cannot reach their own milestone
    is refused by ``_train_arm``, and P2 has no business declaring that guard
    away just to make an integration test cheap.
    """

    return EcologyP2Config(
        n_ants=1,
        temporal_latent_dim=4,
        training_rounds=ecology_training_min_stage_rounds(),
        validation_rounds=2,
        heldout_rounds=2,
        layouts_per_tier=1,
        training_seeds=(17,),
    )


def _layout_rows(
    *,
    config: EcologyP2Config,
    training_seed: int,
    arm: str,
    successful: bool,
    score: float,
    harmful_rate: float,
    composite_harmful_rate: float | None = None,
) -> tuple[EcologyP2LayoutResult, ...]:
    rows: list[EcologyP2LayoutResult] = []
    required = max(1, round(config.n_ants * config.body_success_ratio))
    for capability_index, (capability, _, tier) in enumerate(
        _evaluation_specs()
    ):
        for layout_index in range(config.layouts_per_tier):
            rate = (
                composite_harmful_rate
                if capability == "composite"
                and composite_harmful_rate is not None
                else harmful_rate
            )
            rows.append(
                EcologyP2LayoutResult(
                    training_seed=training_seed,
                    arm=arm,
                    capability=capability,
                    seed=_heldout_seed(
                        capability_index=capability_index,
                        layout_index=layout_index,
                    ),
                    tier=tier.value,
                    successful_bodies=config.n_ants if successful else 0,
                    required_bodies=required,
                    layout_success=successful,
                    pickups=4 if successful else 0,
                    deliveries=3 if successful else 0,
                    heat_escapes=2 if successful else 0,
                    harmful_heat_ticks=0,
                    total_ticks=config.heldout_rounds * config.n_ants,
                    harmful_tick_rate=rate,
                    # Small deterministic per-layout jitter so the paired
                    # bootstrap sees real within-seed spread instead of a
                    # degenerate constant.
                    outcome_score=score
                    + 0.01 * ((capability_index * 7 + layout_index) % 5),
                    escape_latencies=(3,) if successful else (),
                    switch_count=3 if arm == "learned" else 1,
                    non_timeout_segment_closures=2 if arm == "learned" else 1,
                    policy_fingerprint_stable=True,
                    temporal_learning_fingerprint_stable=True,
                    replay_settlement_coverage=1.0,
                    replay_lineage_coverage=1.0,
                    replay_drop_count=0,
                    first_pickup_tick=2 if successful else None,
                    mean_absolute_turn_delta=0.11,
                    applied_distance=12.5,
                    per_body_success=(successful,) * config.n_ants,
                )
            )
    return tuple(rows)


def _shard(
    *,
    config: EcologyP2Config,
    training_seed: int,
    arm: str,
    prerequisite: EcologyP2Prerequisite,
    successful: bool,
    score: float,
    harmful_rate: float = 0.0,
    composite_harmful_rate: float | None = None,
    preflight: bool = False,
    policy_digest: str | None = None,
    source: EcologyP2SourceProvenance = _SOURCE,
) -> EcologyP2ShardReport:
    spec = ECOLOGY_P2_ARM_SPEC_BY_NAME[arm]
    initial_digest = f"initial:{training_seed}" if spec.learning else ""
    if policy_digest is None:
        # Every arm whose construction persists no policy update -- cold
        # (trains=False), no_optimize and eta_off (optimize=False) -- must land
        # back on the shared initial digest; the ablations that DO optimize
        # must diverge from both the initial fork point and the learned arm.
        policy_digest = (
            initial_digest
            if arm in ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES
            else f"digest:{arm}:{training_seed}"
        )
    return EcologyP2ShardReport(
        schema_version=ECOLOGY_P2_SHARD_SCHEMA_VERSION,
        config=config,
        training_seed=training_seed,
        arm=arm,
        batch=spec.batch,
        arm_spec=spec,
        preregistration_digest=preregistration_digest(config),
        schedule_sha256="schedule",
        prerequisite=prerequisite,
        device=config.device,
        preflight=preflight,
        training_complete=True,
        completed_training_episodes=0,
        scheduled_training_episodes=0,
        policy_digest=policy_digest,
        initial_policy_digest=initial_digest,
        archive_roundtrip_ok=(
            True if spec.learning and spec.trains else None
        ),
        archive_corruption_rejected=(
            True if spec.learning and spec.trains else None
        ),
        archive_size_bytes=1024 if spec.learning else None,
        source_provenance=source,
        wall_clock_seconds=1.0,
        layout_results=_layout_rows(
            config=config,
            training_seed=training_seed,
            arm=arm,
            successful=successful,
            score=score,
            harmful_rate=harmful_rate,
            composite_harmful_rate=composite_harmful_rate,
        ),
        probe_summary=(
            EcologyP2ProbeSummary(
                action_chain_passed=True,
                action_chain_failures=(),
                home_probe_count=config.n_ants,
                home_aligned_bodies=config.n_ants,
                food_probe_count=config.n_ants,
                food_aligned_bodies=config.n_ants,
                required_aligned_bodies=max(
                    1, round(config.n_ants * config.body_success_ratio)
                ),
            )
            if arm == "learned"
            else None
        ),
        description="synthetic",
    )


def _passing_matrix(
    config: EcologyP2Config,
    prerequisites,
) -> tuple[EcologyP2ShardReport, ...]:
    """A complete matrix whose every gate passes.

    ``prerequisites`` is a per-seed mapping, exactly as a real matrix has: P1
    reports are per-seed artifacts, so binding each shard to the P1 run of its
    own training seed produces a DIFFERENT file digest per seed.
    """

    shards: list[EcologyP2ShardReport] = []
    for training_seed in config.training_seeds:
        for arm in ECOLOGY_P2_ARM_NAMES:
            learned = arm == "learned"
            shards.append(
                _shard(
                    config=config,
                    training_seed=training_seed,
                    arm=arm,
                    prerequisite=prerequisites[training_seed],
                    successful=learned,
                    score=9.0 if learned else 1.0,
                    harmful_rate=0.0 if learned else 0.02,
                    composite_harmful_rate=(
                        0.0 if learned else 0.01
                    ),
                )
            )
    return tuple(shards)


def _write_per_seed_prerequisites(
    directory,
    seeds,
    *,
    config: EcologyP1Config | None = None,
    marker: str = "synthetic",
) -> dict[int, EcologyP2Prerequisite]:
    return {
        seed: load_p1_prerequisite(
            _write_p1_report(
                directory / f"p1.seed{seed}.json",
                training_seed=seed,
                config=config,
                marker=marker,
            ),
            expected_training_seed=seed,
        )
        for seed in seeds
    }


@pytest.fixture()
def prerequisite(tmp_path) -> EcologyP2Prerequisite:
    """The single-seed prerequisite the small end-to-end shard tests use."""

    return load_p1_prerequisite(
        _write_p1_report(tmp_path / "p1.json", training_seed=17),
        expected_training_seed=17,
    )


@pytest.fixture()
def prerequisites(tmp_path) -> dict[int, EcologyP2Prerequisite]:
    """One P1 artifact per formal training seed, as a real matrix has."""

    return _write_per_seed_prerequisites(
        tmp_path, EcologyP2Config().training_seeds
    )


# ---------------------------------------------------------------------------
# Serial constraint: P2 cannot start before P1 is a complete PASS
# ---------------------------------------------------------------------------


def test_p1_prerequisite_accepts_only_a_complete_pass(tmp_path) -> None:
    accepted = load_p1_prerequisite(_write_p1_report(tmp_path / "pass.json"))
    assert accepted.verdict == "PASS"
    assert accepted.schema_version == ECOLOGY_P1_SCHEMA_VERSION
    assert len(accepted.report_sha256) == 64

    with pytest.raises(EcologyP2PrerequisiteError, match="not PASS"):
        load_p1_prerequisite(
            _write_p1_report(
                tmp_path / "block.json",
                verdict="BLOCK",
                failed_gate="butter_medium",
            )
        )
    # A report may not claim PASS while a frozen gate is failing.
    with pytest.raises(EcologyP2PrerequisiteError, match="not PASS"):
        load_p1_prerequisite(
            _write_p1_report(
                tmp_path / "contradictory.json",
                verdict="PASS",
                failed_gate="composite",
            )
        )
    with pytest.raises(EcologyP2PrerequisiteError, match="schema mismatch"):
        load_p1_prerequisite(
            _write_p1_report(
                tmp_path / "stale.json",
                schema_version="digital-ant-ecology-p1-development.v1",
            )
        )
    with pytest.raises(EcologyP2PrerequisiteError, match="gate set mismatch"):
        load_p1_prerequisite(
            _write_p1_report(
                tmp_path / "thin.json",
                gate_names=ECOLOGY_P1_GATE_NAMES[:-1],
            )
        )
    with pytest.raises(EcologyP2PrerequisiteError, match="not found"):
        load_p1_prerequisite(tmp_path / "missing.json")


async def test_p2_shard_refuses_to_spend_budget_without_a_p1_report(
    tmp_path,
) -> None:
    with pytest.raises(EcologyP2PrerequisiteError, match="frozen P1 report"):
        await run_ecology_p2_shard(
            _small_config(),
            training_seed=17,
            arm="random",
            source_provenance=_SOURCE,
            progress_dir=tmp_path / "progress",
        )


async def test_p2_shard_rejects_an_unregistered_training_seed(
    prerequisite,
) -> None:
    with pytest.raises(ValueError, match="not pre-registered"):
        await run_ecology_p2_shard(
            _small_config(),
            training_seed=999,
            arm="random",
            source_provenance=_SOURCE,
            prerequisite=prerequisite,
        )


async def test_p2_shard_refuses_an_unidentified_source_tree(
    prerequisite,
) -> None:
    """Plan section 5.4 cannot be checked from a tree with no commit."""

    with pytest.raises(ValueError, match="source git SHA"):
        await run_ecology_p2_shard(
            _small_config(),
            training_seed=17,
            arm="random",
            source_provenance=EcologyP2SourceProvenance(
                git_sha="", git_branch="", worktree_dirty=True
            ),
            prerequisite=prerequisite,
        )


def test_p1_prerequisite_is_bound_to_the_shard_training_seed(tmp_path) -> None:
    """One seed's P1 PASS must not unlock another seed's P2 budget."""

    report = _write_p1_report(tmp_path / "p1.json", training_seed=17)
    assert load_p1_prerequisite(report, expected_training_seed=17).training_seed == 17

    with pytest.raises(EcologyP2PrerequisiteError, match="different training seed"):
        load_p1_prerequisite(report, expected_training_seed=2)

    seedless = tmp_path / "seedless.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    del payload["config"]
    seedless.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EcologyP2PrerequisiteError, match="no config.seed"):
        load_p1_prerequisite(seedless)


def test_p1_budget_is_re_derived_instead_of_trusting_the_gate_boolean(
    tmp_path,
) -> None:
    """The P1 owner's predicate is re-run over the report's own config block.

    A P1 report whose ``formal_configuration`` gate says ``passed=True`` while
    its recorded configuration is a 1-ant smoke is self-contradictory, and the
    contradiction is only visible if P2 recomputes the budget rather than
    reading the boolean.
    """

    under_budget = _write_p1_report(
        tmp_path / "smoke.json",
        training_seed=0,
        config=EcologyP1Config(n_ants=1, temporal_latent_dim=4),
    )
    with pytest.raises(
        EcologyP2PrerequisiteError, match="below the frozen P1 budget"
    ):
        load_p1_prerequisite(under_budget, expected_training_seed=0)

    # A config block that cannot be re-derived at all is refused too: silently
    # skipping the check for a drifted schema is what the boolean already did.
    truncated = tmp_path / "truncated.json"
    payload = json.loads(
        _write_p1_report(tmp_path / "ok.json", training_seed=0).read_text(
            encoding="utf-8"
        )
    )
    del payload["config"]["layouts_per_tier"]
    truncated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EcologyP2PrerequisiteError, match="cannot be re-derived"):
        load_p1_prerequisite(truncated, expected_training_seed=0)


def test_p1_configuration_identity_ignores_the_seed_and_nothing_else(
    tmp_path,
) -> None:
    """The one field that MUST differ across a matrix is the one excluded."""

    first = load_p1_prerequisite(
        _write_p1_report(tmp_path / "a.json", training_seed=0),
        expected_training_seed=0,
    )
    other_seed = load_p1_prerequisite(
        _write_p1_report(tmp_path / "b.json", training_seed=1),
        expected_training_seed=1,
    )
    other_budget = load_p1_prerequisite(
        _write_p1_report(
            tmp_path / "c.json",
            training_seed=0,
            config=EcologyP1Config(layouts_per_tier=7),
        ),
        expected_training_seed=0,
    )

    assert first.report_sha256 != other_seed.report_sha256
    assert first.configuration_digest == other_seed.configuration_digest
    assert first.configuration_digest != other_budget.configuration_digest


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


def test_preregistration_digest_pins_config_arms_and_curriculum() -> None:
    config = EcologyP2Config()
    assert preregistration_digest(config) == preregistration_digest(
        EcologyP2Config()
    )
    for mutated in (
        replace(config, n_ants=16),
        replace(config, layouts_per_tier=6),
        replace(config, training_seeds=(0, 1, 2, 3)),
        replace(config, heldout_rounds=200),
        replace(config, device="cuda"),
    ):
        assert preregistration_digest(mutated) != preregistration_digest(
            config
        )


def test_frozen_logic_digest_covers_its_transitive_helpers() -> None:
    """plan section 5.4: a scoring change must invalidate an in-flight batch.

    Hashing only the ten named scoring/gate functions left a transitive hole:
    ``_run_baseline_layout`` decides WHICH WORLD every baseline arm is scored
    in by calling ``_scenario_stage``, and every lane trains under whatever
    ``_curriculum_config`` returns. Neither helper was hashed, so both could be
    rewritten under a byte-identical pre-registration digest and pre-change /
    post-change shards would merge silently.
    """

    source = _frozen_logic_source()
    for name in _ECOLOGY_P2_FROZEN_LOGIC:
        function = getattr(ecology_p2_module, name)
        assert inspect.getsource(function) in source, name
    # The two that were missing, named explicitly so a future narrowing of the
    # hashing unit fails here rather than in a formal batch.
    assert inspect.getsource(_scenario_stage) in source
    assert inspect.getsource(_curriculum_config) in source
    # ... and the helpers those two reach for in turn.
    for name in ("_heldout_seed", "heldout_layout_seeds", "p2_training_schedule"):
        assert inspect.getsource(getattr(ecology_p2_module, name)) in source


def test_frozen_thresholds_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="frozen at 0.6"):
        EcologyP2Config(layout_success_ratio=0.4)
    with pytest.raises(ValueError, match="frozen at 0.6"):
        EcologyP2Config(body_success_ratio=0.5)
    with pytest.raises(ValueError, match="frozen at 0.05"):
        EcologyP2Config(harmful_tick_rate_max=0.2)
    with pytest.raises(ValueError, match="unique"):
        EcologyP2Config(training_seeds=(0, 0, 1))
    with pytest.raises(ValueError, match="sorted"):
        EcologyP2Config(training_seeds=(2, 0, 1))


def test_p2_replays_the_p1_frozen_curriculum() -> None:
    config = EcologyP2Config()
    for training_seed in config.training_seeds:
        assert p2_training_schedule(
            config, training_seed=training_seed
        ) == _fixed_schedule(
            EcologyP1Config(
                n_ants=config.n_ants,
                temporal_latent_dim=config.temporal_latent_dim,
                training_rounds=config.training_rounds,
                evaluation_rounds=config.heldout_rounds,
                layouts_per_tier=config.layouts_per_tier,
                seed=training_seed,
            )
        )


def test_the_formal_p2_budget_can_pay_for_its_own_curriculum() -> None:
    """P2's frozen round floor must clear the curriculum owner's milestone floor.

    ``_curriculum_config`` forwards ``training_rounds`` as ``stage_rounds``, and
    ``_train_arm`` refuses a schedule whose episodes cannot reach their own
    milestone. If the owner's published floor ever rises above P2's frozen
    minimum, every confirmatory shard would die at runtime; that has to fail
    here, in a second, rather than mid-matrix.
    """

    assert (
        EcologyP2Config().training_rounds >= ecology_training_min_stage_rounds()
    )
    formal = _curriculum_config(EcologyP2Config(), training_seed=0)
    # P2 never declares the owner's budget guard away.
    assert formal.milestone_budget_enforced is True
    assert formal.stage_rounds == EcologyP2Config().training_rounds
    assert _small_config().training_rounds >= ecology_training_min_stage_rounds()
    assert (
        _curriculum_config(
            _small_config(), training_seed=17
        ).milestone_budget_enforced
        is True
    )


def test_p2_heldout_namespace_is_disjoint_from_p1() -> None:
    config = EcologyP2Config()
    p1_seeds = {
        0 + 2_000_003 + capability_index * 10_007 + index * 103
        for capability_index in range(len(_evaluation_specs()))
        for index in range(config.layouts_per_tier)
    }
    p2_seeds = set(
        _heldout_seed(capability_index=capability_index, layout_index=index)
        for capability_index in range(len(_evaluation_specs()))
        for index in range(config.layouts_per_tier)
    )
    assert not p1_seeds & p2_seeds


def test_outcome_score_matches_the_curriculum_owner() -> None:
    metrics = EcologyArmMetrics(
        arm="learned",
        data_split=EcologyDataSplit.HELDOUT,
        stage=EcologyStage.COMPOSITE,
        scenario=EcologyEvaluationScenario.COMPOSITE,
        seed=1,
        pickups=7,
        deliveries=3,
        obstacle_contacts=11,
        harmful_heat_ticks=5,
        heat_entries=2,
        heat_escapes=4,
        applied_distance=1.0,
        replay_captured=0,
        replay_settled=0,
        replay_lineage_matches=0,
        replay_pending_captures=0,
        replay_staged_rollouts=0,
        replay_drop_count=0,
        replay_settlement_coverage=1.0,
        replay_lineage_coverage=1.0,
        nonzero_ecology_payoffs=0,
        activated_sense_channels=(),
        first_pickup_tick=None,
        first_obstacle_contact_tick=None,
        first_heat_entry_tick=None,
        first_heat_escape_tick=None,
        minimum_food_distance=None,
        minimum_obstacle_distance=None,
        minimum_heat_distance=None,
        switch_count=0,
        mean_persistence_steps=0.0,
        closed_segment_count=0,
        longest_segment_length=0,
        mean_absolute_turn_delta=0.0,
        policy_fingerprint_stable=True,
        temporal_learning_fingerprint_stable=True,
    )
    assert outcome_score(
        pickups=metrics.pickups,
        deliveries=metrics.deliveries,
        heat_escapes=metrics.heat_escapes,
        harmful_heat_ticks=metrics.harmful_heat_ticks,
    ) == pytest.approx(_ecology_outcome_score(metrics))


# ---------------------------------------------------------------------------
# Matched-control levers (root cause: an ablation must actually be ablated)
# ---------------------------------------------------------------------------


def _config_for(**kwargs):
    return _session_config(
        config=EcologyCurriculumConfig(n_ants=1, temporal_latent_dim=4),
        seed=0,
        session_id="lever",
        optimize=True,
        **kwargs,
    )


def test_pe_lever_reaches_the_session_contract_without_taking_eta() -> None:
    learned = _config_for()
    assert learned.external_prediction_error_drive is True
    assert learned.joint_apply_writeback is True
    assert (
        learned.rollout_config.prediction_error_temporal_switch
        is WiringLevel.ACTIVE
    )

    pe_off = _config_for(prediction_error_enabled=False)
    assert pe_off.external_prediction_error_drive is False
    assert (
        pe_off.rollout_config.prediction_error_temporal_switch
        is WiringLevel.DISABLED
    )
    # PE-off must not silently take the reflection-consolidation lever with it.
    assert pe_off.joint_apply_writeback is True


def test_eta_off_is_the_frozen_learned_lite_construction() -> None:
    """spec section 7: ``eta_off`` = frozen learned-lite, ssl=rl=0.

    ``joint_apply_writeback=False`` is NOT that construction. In the kernel one
    ``owner_writeback_enabled`` boolean gates both the reflection -> temporal
    prior writeback and the ReflectionEngine/memory + regime consolidation, so
    flipping it removes a superset of the named mechanism -- while SSL and
    Internal-RL keep optimising the same world policy every cycle, so the arm
    does not even remove the mechanism it claims to.
    """

    eta_off = ECOLOGY_P2_ARM_SPEC_BY_NAME["eta_off"]
    assert eta_off.temporal_policy_kind == "learned_lite"
    assert eta_off.joint_ssl_interval == 0
    assert eta_off.joint_rl_interval == 0
    # The mislabelled lever is no longer what defines this arm.
    assert eta_off.temporal_writeback_enabled is True
    assert not any(
        spec.temporal_writeback_enabled is False
        for spec in ECOLOGY_P2_ARM_SPEC_BY_NAME.values()
    )


def test_an_arm_whose_lever_cannot_be_expressed_refuses_to_run() -> None:
    """A lever the session contract cannot express must block, not degrade.

    The ETA-off construction needs a ``JointLoopSchedule`` and a
    ``LearnedLiteTemporalPolicy``; both are vz-temporal internals this package
    may not import, and ``ecology_curriculum._session_config`` does not accept
    them. Until that hop lands, running the arm would label a *different*
    construction with the pre-registered arm name -- the exact
    "策略参数未按预期改变" kill condition. Refusing is the honest outcome.
    """

    unreachable = unreachable_arm_levers(
        ECOLOGY_P2_ARM_SPEC_BY_NAME["eta_off"]
    )
    assert unreachable == (
        "temporal_policy_kind",
        "joint_ssl_interval",
        "joint_rl_interval",
    )
    assert unreachable_arm_levers(ECOLOGY_P2_ARM_SPEC_BY_NAME["learned"]) == ()
    for name in ("no_optimize", "cold", "pe_off", "random", "fixed_rule"):
        assert unreachable_arm_levers(ECOLOGY_P2_ARM_SPEC_BY_NAME[name]) == ()


def test_unreachable_levers_are_read_off_the_exported_constant(
    monkeypatch,
) -> None:
    """``ECOLOGY_P2_ARM_LEVER_PARAMETERS`` is exported; it must also be USED.

    It was public API while the guard rebuilt the same three names inline, so
    a consumer reading the constant and the guard deciding the answer could
    drift apart silently.
    """

    assert ECOLOGY_P2_ARM_LEVER_PARAMETERS == (
        "temporal_policy_kind",
        "joint_ssl_interval",
        "joint_rl_interval",
    )
    spec = ECOLOGY_P2_ARM_SPEC_BY_NAME["eta_off"]

    monkeypatch.setattr(
        ecology_p2_module,
        "ECOLOGY_P2_ARM_LEVER_PARAMETERS",
        ("joint_rl_interval",),
    )
    assert unreachable_arm_levers(spec) == ("joint_rl_interval",)

    monkeypatch.setattr(
        ecology_p2_module, "ECOLOGY_P2_ARM_LEVER_PARAMETERS", ()
    )
    assert unreachable_arm_levers(spec) == ()


def test_frozen_and_divergent_policy_arms_partition_the_learning_arms() -> None:
    """The two policy-digest gates must not disagree about what an arm is.

    ``ablation_policy_divergence`` used to demand that every trained ablation
    arm move AWAY from the shared initial digest, and it included ``eta_off``.
    But ``eta_off`` is declared ``optimize=False`` with ssl=rl=0, and the
    sibling ``no_optimize_policy_stable`` gate asserts that ``optimize=False``
    means the digest EQUALS the initial -- so by the module's own semantics
    ``eta_off`` could only ever be classified ``never_trained``, and PASS was
    unreachable for a second reason.
    """

    learning = {spec.name for spec in ECOLOGY_P2_ARM_SPECS if spec.learning}
    frozen = set(ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES)
    divergent = set(ECOLOGY_P2_DIVERGENT_POLICY_ARM_NAMES)

    assert frozen & divergent == set()
    assert frozen | divergent | {"learned"} == learning
    # An arm belongs to the frozen set exactly when its construction persists
    # no policy update.
    for spec in ECOLOGY_P2_ARM_SPECS:
        if not spec.learning:
            continue
        persists_nothing = not spec.trains or not spec.optimize
        assert (spec.name in frozen) is persists_nothing, spec.name
    assert "eta_off" in frozen
    assert frozen == {"no_optimize", "cold", "eta_off"}
    assert divergent == {
        "pe_off",
        "dense_local_shaping_off",
        "segment_credit_off",
    }


async def test_eta_off_shard_blocks_instead_of_running_a_different_arm(
    prerequisite,
) -> None:
    with pytest.raises(
        EcologyP2ArmLeverUnavailableError, match="temporal_policy_kind"
    ):
        await run_ecology_p2_shard(
            _small_config(),
            training_seed=17,
            arm="eta_off",
            source_provenance=_SOURCE,
            prerequisite=prerequisite,
        )


def test_arm_specs_differ_from_learned_by_their_declared_levers() -> None:
    """Each ablation names one mechanism, in a pre-registered lever set."""

    learned = ECOLOGY_P2_ARM_SPEC_BY_NAME["learned"]
    levers = (
        "optimize",
        "local_valence_enabled",
        "segment_credit_enabled",
        "prediction_error_enabled",
        "temporal_writeback_enabled",
        "temporal_policy_kind",
        "joint_ssl_interval",
        "joint_rl_interval",
        "trains",
    )
    # Pre-registered: ETA-off is a construction, not a boolean flip, so it
    # legitimately moves three fields that together name one mechanism.
    expected = {
        "no_optimize": {"optimize"},
        "cold": {"trains"},
        "pe_off": {"prediction_error_enabled"},
        "eta_off": {
            "optimize",
            "temporal_policy_kind",
            "joint_ssl_interval",
            "joint_rl_interval",
        },
        "dense_local_shaping_off": {"local_valence_enabled"},
        "segment_credit_off": {"segment_credit_enabled"},
    }
    for name, declared in expected.items():
        spec = ECOLOGY_P2_ARM_SPEC_BY_NAME[name]
        differences = {
            lever
            for lever in levers
            if getattr(spec, lever) != getattr(learned, lever)
        }
        assert differences == declared, f"{name} changes {differences}"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_holm_adjustment_is_monotone_and_conservative() -> None:
    raw = (0.001, 0.02, 0.03, 0.5)
    adjusted = _holm_adjusted(raw)
    assert adjusted == tuple(sorted(adjusted))
    assert all(
        adjusted[index] >= raw[index] for index in range(len(raw))
    )
    assert adjusted[0] == pytest.approx(0.004)
    assert adjusted[-1] == pytest.approx(0.5)
    assert all(value <= 1.0 for value in _holm_adjusted((0.6, 0.7, 0.8, 0.9)))
    assert _holm_adjusted(()) == ()


def test_hierarchical_bootstrap_separates_seed_and_layout_levels() -> None:
    positive = ((1.0, 1.2, 0.8), (0.9, 1.1, 1.0), (1.3, 0.7, 1.0))
    mean, ci_low, ci_high, p_value = _hierarchical_paired_bootstrap(
        positive, seed=3, samples=2000
    )
    assert mean == pytest.approx(1.0, abs=0.05)
    assert ci_low > 0.0
    assert ci_high > ci_low
    assert p_value < 0.05

    null = ((0.5, -0.5, 0.1), (-0.3, 0.4, -0.2), (0.2, -0.1, 0.0))
    _, null_low, null_high, null_p = _hierarchical_paired_bootstrap(
        null, seed=3, samples=2000
    )
    assert null_low < 0.0 < null_high
    assert null_p > 0.05

    with pytest.raises(ValueError, match="at least one training seed"):
        _hierarchical_paired_bootstrap((), seed=1, samples=1000)
    with pytest.raises(ValueError, match="at least one paired layout"):
        _hierarchical_paired_bootstrap(((1.0,), ()), seed=1, samples=1000)


# ---------------------------------------------------------------------------
# Aggregation and promotion verdict
# ---------------------------------------------------------------------------


def test_full_matrix_reaches_pass_and_reports_every_endpoint(
    prerequisites,
) -> None:
    config = EcologyP2Config()
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisites),
        worktree_clean=True,
        config=config,
    )

    assert tuple(gate.name for gate in report.gates) == ECOLOGY_P2_GATE_NAMES
    assert report.verdict == "PASS", report.description
    assert report.diagnostic_breakpoints == ()
    assert tuple(item.name for item in report.primary_endpoints) == tuple(
        name for name, _ in ECOLOGY_P2_PRIMARY_ENDPOINTS
    )
    assert all(item.passed for item in report.primary_endpoints)
    assert {item.comparison for item in report.paired_effects} == {
        "learned_vs_no_optimize",
        "learned_vs_cold",
        "learned_vs_pe_off",
        "learned_vs_eta_off",
        # The random floor is a pre-registered comparison, not an ablation
        # substitute: the FORMAL stage may not be weaker than P1, which
        # already requires forced escape to clear the random floor.
        "learned_vs_random",
    }
    assert all(item.significant for item in report.paired_effects)
    assert all(item.complete for item in report.paired_effects)
    assert all(item.ci_low > 0.0 for item in report.paired_effects)
    assert report.source_git_sha == _SOURCE.git_sha
    assert tuple(
        item.name for item in report.secondary_endpoints
    ) == ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES

    # BLOCKER: a formal matrix has ONE P1 report PER TRAINING SEED, so their
    # file digests necessarily differ. Pinning "one identical P1 report" made
    # PASS structurally unreachable for any matrix with >=2 training seeds --
    # i.e. for every matrix the formal_configuration gate would accept.
    assert len(config.training_seeds) >= 3
    assert len(report.p1_prerequisites) == len(config.training_seeds)
    assert len({item.report_sha256 for item in report.p1_prerequisites}) == len(
        config.training_seeds
    )
    assert tuple(item.training_seed for item in report.p1_prerequisites) == (
        config.training_seeds
    )
    # What IS pinned across the matrix is the frozen P1 configuration.
    assert (
        len({item.configuration_digest for item in report.p1_prerequisites})
        == 1
    )
    assert all(item.configuration_digest for item in report.p1_prerequisites)
    # Plan section 2.1 provenance the promotion lane has to carry forward.
    assert report.heldout_layout_seeds == heldout_layout_seeds(config)
    assert len(report.heldout_layout_seeds) == len(_evaluation_specs()) * (
        config.layouts_per_tier
    )


def test_a_mixed_configuration_p1_matrix_cannot_pass(tmp_path) -> None:
    """Same-file is the wrong invariant; same-*configuration* is the right one.

    Seed 2's P1 run here is a different frozen configuration (a smaller
    colony). Every shard still carries a PASS P1 report of its own seed, and
    every file digest is distinct exactly as in the passing matrix -- so only a
    configuration-level identity can tell the two matrices apart.
    """

    config = EcologyP2Config()
    mixed = _write_per_seed_prerequisites(tmp_path, config.training_seeds[:-1])
    odd_directory = tmp_path / "odd"
    odd_directory.mkdir()
    mixed.update(
        _write_per_seed_prerequisites(
            odd_directory,
            config.training_seeds[-1:],
            config=EcologyP1Config(n_ants=8),
        )
    )
    assert len({item.report_sha256 for item in mixed.values()}) == 3
    assert len({item.configuration_digest for item in mixed.values()}) == 2

    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, mixed), worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "p1_prerequisite_pass" in report.diagnostic_breakpoints
    gate = next(
        item for item in report.gates if item.name == "p1_prerequisite_pass"
    )
    assert "p1_configuration_digests=2" in gate.observed


def test_a_shard_unlocked_by_another_seeds_p1_report_blocks(
    prerequisites,
) -> None:
    """Per-seed binding is re-checked at aggregation, not only at load."""

    config = EcologyP2Config()
    smuggled = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            # Seed 2's shards are unlocked by seed 0's P1 report.
            prerequisite=prerequisites[0 if training_seed == 2 else training_seed],
            successful=arm == "learned",
            score=9.0 if arm == "learned" else 1.0,
            harmful_rate=0.0 if arm == "learned" else 0.02,
            composite_harmful_rate=0.0 if arm == "learned" else 0.01,
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        smuggled, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    gate = next(
        item for item in report.gates if item.name == "p1_prerequisite_pass"
    )
    assert gate.passed is False
    assert "p1_seed=0" in gate.observed


def test_two_p1_reports_for_one_training_seed_block(prerequisites, tmp_path) -> None:
    """One seed, one P1 run: two reports for a seed is an unresolved lineage."""

    config = EcologyP2Config()
    duplicate_directory = tmp_path / "rerun"
    duplicate_directory.mkdir()
    duplicate = _write_per_seed_prerequisites(
        duplicate_directory, (1,), marker="second run"
    )[1]
    assert duplicate.report_sha256 != prerequisites[1].report_sha256
    assert duplicate.configuration_digest == (
        prerequisites[1].configuration_digest
    )

    shards = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=(
                duplicate
                if training_seed == 1 and arm == "learned"
                else prerequisites[training_seed]
            ),
            successful=arm == "learned",
            score=9.0 if arm == "learned" else 1.0,
            harmful_rate=0.0 if arm == "learned" else 0.02,
            composite_harmful_rate=0.0 if arm == "learned" else 0.01,
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        shards, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    gate = next(
        item for item in report.gates if item.name == "p1_prerequisite_pass"
    )
    assert gate.passed is False
    assert "seed1:2_reports" in gate.observed


def test_an_unidentified_p1_configuration_cannot_reach_pass(
    prerequisites,
) -> None:
    """A hand-built prerequisite with no configuration identity must block."""

    config = EcologyP2Config()
    blank = {
        seed: replace(item, configuration_digest="")
        for seed, item in prerequisites.items()
    }
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, blank), worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "p1_prerequisite_pass" in report.diagnostic_breakpoints


def test_a_missing_or_preflight_shard_blocks_the_matrix(prerequisites) -> None:
    config = EcologyP2Config()
    full = _passing_matrix(config, prerequisites)

    thinned = tuple(
        item
        for item in full
        if not (item.arm == "eta_off" and item.training_seed == 1)
    )
    # A missing cell used to abort the whole aggregation with a ValueError out
    # of ``_paired_differences``, leaving NOTHING on disk. Plan section 2.3
    # requires an artifact even on failure, so the matrix now produces a
    # complete diagnostic BLOCK that names the absent cells instead.
    report = aggregate_ecology_p2_shards(
        thinned, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "shard_completeness" in report.diagnostic_breakpoints
    assert "pe_eta_causal_degradation" in report.diagnostic_breakpoints
    assert tuple(gate.name for gate in report.gates) == ECOLOGY_P2_GATE_NAMES
    assert tuple(item.name for item in report.primary_endpoints) == tuple(
        name for name, _ in ECOLOGY_P2_PRIMARY_ENDPOINTS
    )
    completeness = next(
        gate for gate in report.gates if gate.name == "shard_completeness"
    )
    assert "eta_off@1" in completeness.observed
    eta_effect = next(
        item
        for item in report.paired_effects
        if item.comparison == "learned_vs_eta_off"
    )
    assert eta_effect.complete is False
    assert eta_effect.significant is False
    # Nothing was imputed in a favourable direction (plan section 5.6).
    assert eta_effect.mean_difference == 0.0
    assert eta_effect.ci_low == 0.0 and eta_effect.ci_high == 0.0
    assert len(eta_effect.missing_cells) == len(_evaluation_specs()) * (
        config.layouts_per_tier
    )
    assert all("eta_off@1" in cell for cell in eta_effect.missing_cells)
    # The comparisons that ARE complete keep their real statistics.
    cold_effect = next(
        item
        for item in report.paired_effects
        if item.comparison == "learned_vs_cold"
    )
    assert cold_effect.complete is True
    assert cold_effect.significant is True

    # A shard present but marked preflight must not be laundered into the
    # confirmatory statistics.
    tainted = tuple(
        replace(item, preflight=True) if item.arm == "learned" else item
        for item in full
    )
    report = aggregate_ecology_p2_shards(
        tainted, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "shard_completeness" in report.diagnostic_breakpoints


def test_an_unexecutable_arm_yields_a_complete_named_block_aggregate(
    prerequisites,
) -> None:
    """The whole confirmatory matrix must still emit a diagnostic artifact.

    ``eta_off`` cannot be executed as pre-registered today: its frozen
    construction needs ``temporal_policy_kind`` / ``joint_ssl_interval`` /
    ``joint_rl_interval`` on the curriculum session builder. Refusing to fake
    the arm is correct; producing no artifact at all is not (plan section 2.3).
    The aggregate must therefore say *which* arm is missing and *why*.
    """

    config = EcologyP2Config()
    without_eta = tuple(
        item
        for item in _passing_matrix(config, prerequisites)
        if item.arm != "eta_off"
    )
    report = aggregate_ecology_p2_shards(
        without_eta, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    completeness = next(
        gate for gate in report.gates if gate.name == "shard_completeness"
    )
    assert completeness.passed is False
    assert "unexecutable_arms=['eta_off']" in completeness.observed
    for training_seed in config.training_seeds:
        assert (
            f"eta_off@{training_seed}:unexecutable"
            f"(levers=temporal_policy_kind+joint_ssl_interval+joint_rl_interval)"
        ) in completeness.observed
    causal = next(
        gate
        for gate in report.gates
        if gate.name == "pe_eta_causal_degradation"
    )
    assert "NOT COMPUTED" in causal.observed
    # Everything else the matrix CAN answer is still answered, so the artifact
    # is a usable diagnosis rather than an abort.
    passed = {gate.name for gate in report.gates if gate.passed}
    assert {"butter_medium", "composite", "provenance_clean"} <= passed


def test_small_budget_evidence_cannot_claim_promotion(prerequisites) -> None:
    config = EcologyP2Config(
        n_ants=2,
        training_rounds=4,
        validation_rounds=4,
        heldout_rounds=4,
        layouts_per_tier=2,
        training_seeds=(0,),
    )
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisites),
        worktree_clean=True,
        config=config,
    )
    assert report.verdict == "BLOCK"
    assert "formal_configuration" in report.diagnostic_breakpoints


def test_dirty_worktree_and_blocked_p1_are_hard_failures(
    prerequisites,
) -> None:
    config = EcologyP2Config()
    dirty = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisites),
        worktree_clean=False,
        config=config,
    )
    assert dirty.verdict == "BLOCK"
    assert "provenance_clean" in dirty.diagnostic_breakpoints

    blocked = {
        seed: replace(item, verdict="BLOCK")
        for seed, item in prerequisites.items()
    }
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, blocked),
        worktree_clean=True,
        config=config,
    )
    assert report.verdict == "BLOCK"
    assert "p1_prerequisite_pass" in report.diagnostic_breakpoints


def test_learned_without_an_effect_blocks_the_causal_endpoints(
    prerequisites,
) -> None:
    config = EcologyP2Config()
    # Every arm scores identically: capability gates still pass but no paired
    # effect exists, so the learning and PE/ETA endpoints must block.
    flat = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisites[training_seed],
            successful=True,
            score=5.0,
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        flat, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "learned_paired_effect" in report.diagnostic_breakpoints
    assert "pe_eta_causal_degradation" in report.diagnostic_breakpoints
    failed = {
        item.name: item.failed_gates for item in report.primary_endpoints
    }
    assert failed["learned_paired_effect"] == ("learned_paired_effect",)
    assert failed["butter_distance_transfer"] == ()


def test_random_arm_enters_a_gate_instead_of_burning_free_budget(
    prerequisites,
) -> None:
    """P1 already gates the random floor; P2 may not be weaker."""

    assert ("learned", "random") in ECOLOGY_P2_PAIRED_COMPARISONS
    assert "above_random_floor" in ECOLOGY_P2_GATE_NAMES

    config = EcologyP2Config()
    # Learned clears every capability gate but scores no better than random.
    shards = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisites[training_seed],
            successful=arm == "learned",
            score=9.0 if arm in {"learned", "random"} else 1.0,
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        shards, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "above_random_floor" in report.diagnostic_breakpoints


def test_an_inert_ablation_lever_blocks_the_matrix(prerequisites) -> None:
    """spec section 6: "策略参数未按预期改变" must be caught.

    Each shard's policy digest used to be consumed only for resume comparison,
    never across arms -- so a ``no_optimize`` arm that silently kept persisting
    updates, or an ablation whose lever did nothing, passed.
    """

    config = EcologyP2Config()

    def _matrix(**overrides: str):
        return tuple(
            _shard(
                config=config,
                training_seed=training_seed,
                arm=arm,
                prerequisite=prerequisites[training_seed],
                successful=arm == "learned",
                score=9.0 if arm == "learned" else 1.0,
                harmful_rate=0.0 if arm == "learned" else 0.02,
                composite_harmful_rate=0.0 if arm == "learned" else 0.01,
                policy_digest=(
                    overrides[arm].format(training_seed=training_seed)
                    if arm in overrides
                    else None
                ),
            )
            for training_seed in config.training_seeds
            for arm in ECOLOGY_P2_ARM_NAMES
        )

    # Training left the learned policy exactly where it forked from.
    frozen_learned = aggregate_ecology_p2_shards(
        _matrix(learned="initial:{training_seed}"),
        worktree_clean=True,
        config=config,
    )
    assert "policy_changed" in frozen_learned.diagnostic_breakpoints

    # no-optimize kept persisting its updates after all.
    leaking = aggregate_ecology_p2_shards(
        _matrix(no_optimize="leaked:{training_seed}"),
        worktree_clean=True,
        config=config,
    )
    assert "no_optimize_policy_stable" in leaking.diagnostic_breakpoints

    # The PE-off lever did nothing: identical parameters to learned.
    inert = aggregate_ecology_p2_shards(
        _matrix(pe_off="digest:learned:{training_seed}"),
        worktree_clean=True,
        config=config,
    )
    assert "ablation_policy_divergence" in inert.diagnostic_breakpoints

    # An ablation arm that never trained at all.
    untrained = aggregate_ecology_p2_shards(
        _matrix(segment_credit_off="initial:{training_seed}"),
        worktree_clean=True,
        config=config,
    )
    assert "ablation_policy_divergence" in untrained.diagnostic_breakpoints

    # eta_off is graded by the FROZEN-policy gate, not the divergence gate:
    # its construction is optimize=False with ssl=rl=0, so a drifted digest
    # means the ablation did not actually freeze the policy.
    eta_drifted = aggregate_ecology_p2_shards(
        _matrix(eta_off="drifted:{training_seed}"),
        worktree_clean=True,
        config=config,
    )
    assert "no_optimize_policy_stable" in eta_drifted.diagnostic_breakpoints
    assert (
        "ablation_policy_divergence"
        not in eta_drifted.diagnostic_breakpoints
    )
    stable = next(
        item
        for item in eta_drifted.gates
        if item.name == "no_optimize_policy_stable"
    )
    assert "eta_off@0" in stable.observed

    # cold never trains, so it must land on the initial digest too.
    cold_drifted = aggregate_ecology_p2_shards(
        _matrix(cold="drifted:{training_seed}"),
        worktree_clean=True,
        config=config,
    )
    assert "no_optimize_policy_stable" in cold_drifted.diagnostic_breakpoints


def test_the_frozen_eta_off_construction_reaches_pass(prerequisites) -> None:
    """The coherent classification must be REACHABLE, not merely consistent.

    ``_shard`` gives every frozen-policy arm the shared initial digest, which
    is what the real runner produces for ``optimize=False`` / ``trains=False``
    arms. Under the previous rule that exact matrix was classified
    ``eta_off:never_trained`` and could never pass.
    """

    config = EcologyP2Config()
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisites),
        worktree_clean=True,
        config=config,
    )
    eta_shard = next(
        item
        for item in _passing_matrix(config, prerequisites)
        if item.arm == "eta_off" and item.training_seed == 0
    )
    assert eta_shard.policy_digest == eta_shard.initial_policy_digest
    assert report.verdict == "PASS", report.description


def test_learned_may_not_be_less_safe_than_fixed_rule(prerequisites) -> None:
    """plan section 5.7: not weaker than FixedRule's safety threshold.

    ``max(fixed_rule, cap)`` let learned be strictly less safe than the
    hand-written FSM and still pass; the floor is the ``min``.
    """

    config = EcologyP2Config()
    shards = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisites[training_seed],
            successful=arm == "learned",
            score=9.0 if arm == "learned" else 1.0,
            # Learned burns more harmful ticks than FixedRule but stays under
            # the absolute 0.05 cap.
            harmful_rate=(
                0.04
                if arm == "learned"
                else (0.001 if arm == "fixed_rule" else 0.02)
            ),
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        shards, worktree_clean=True, config=config
    )
    assert "fixed_rule_safety_floor" in report.diagnostic_breakpoints


def test_learned_must_show_its_own_advantage_over_fixed_rule(
    prerequisites,
) -> None:
    """plan section 5.7's second FixedRule clause had no gate at all."""

    config = EcologyP2Config()
    shards = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisites[training_seed],
            successful=arm == "learned",
            score=9.0 if arm in {"learned", "fixed_rule"} else 1.0,
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        shards, worktree_clean=True, config=config
    )
    assert "fixed_rule_learning_advantage" in report.diagnostic_breakpoints


def test_mixed_commits_and_dirty_shards_block_the_batch(prerequisites) -> None:
    """plan section 5.4, enforced by the aggregator rather than a driver."""

    config = EcologyP2Config()
    other = EcologyP2SourceProvenance(
        git_sha="1" * 40, git_branch="test", worktree_dirty=False
    )
    mixed = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisites[training_seed],
            successful=arm == "learned",
            score=9.0 if arm == "learned" else 1.0,
            source=other if arm == "cold" else _SOURCE,
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    report = aggregate_ecology_p2_shards(
        mixed, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "provenance_clean" in report.diagnostic_breakpoints
    assert report.source_git_sha == ""

    dirty = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisites[training_seed],
            successful=arm == "learned",
            score=9.0 if arm == "learned" else 1.0,
            source=EcologyP2SourceProvenance(
                git_sha=_SOURCE.git_sha,
                git_branch="test",
                worktree_dirty=arm == "e2e_rl",
            ),
        )
        for training_seed in config.training_seeds
        for arm in ECOLOGY_P2_ARM_NAMES
    )
    assert "provenance_clean" in aggregate_ecology_p2_shards(
        dirty, worktree_clean=True, config=config
    ).diagnostic_breakpoints


def test_corrupted_archive_rollback_is_a_promotion_gate(prerequisites) -> None:
    config = EcologyP2Config()
    shards = tuple(
        replace(item, archive_corruption_rejected=False)
        if item.arm == "learned"
        else item
        for item in _passing_matrix(config, prerequisites)
    )
    report = aggregate_ecology_p2_shards(
        shards, worktree_clean=True, config=config
    )
    assert report.verdict == "BLOCK"
    assert "archive_corruption_rollback" in report.diagnostic_breakpoints


def test_preregistration_digest_binds_the_frozen_score_weights() -> None:
    """plan section 5.4 covers the estimand, not only the declared config."""

    assert ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS == (
        ("pickups", 0.5),
        ("deliveries", 1.0),
        ("heat_escapes", 0.25),
        ("harmful_heat_ticks", -0.02),
    )
    config = EcologyP2Config()
    baseline = preregistration_digest(config)
    import volvence_ant.experiments.ecology_p2 as module

    original = module.ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS
    try:
        module.ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS = (
            ("pickups", 0.5),
            ("deliveries", 2.0),
            ("heat_escapes", 0.25),
            ("harmful_heat_ticks", -0.02),
        )
        assert preregistration_digest(config) != baseline
    finally:
        module.ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS = original
    assert preregistration_digest(config) == baseline


def test_secondary_endpoints_are_reported_or_declared_absent(
    prerequisites,
) -> None:
    """plan section 5.5 names six; none may simply go missing."""

    assert ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES == (
        "path_efficiency",
        "first_pickup_tick",
        "escape_latency",
        "per_ant_variance",
        "action_smoothness",
        "action_probe_sensitivity",
    )
    config = EcologyP2Config()
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisites),
        worktree_clean=True,
        config=config,
    )
    by_name = {item.name: item for item in report.secondary_endpoints}
    assert by_name["path_efficiency"].collected is False
    assert "NOT COLLECTED" in by_name["path_efficiency"].note
    for name in (
        "first_pickup_tick",
        "escape_latency",
        "per_ant_variance",
        "action_smoothness",
        "action_probe_sensitivity",
    ):
        assert by_name[name].collected is True, name
    # A diagnostic can never rescue a primary: no secondary name is a gate.
    assert not set(ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES) & set(
        ECOLOGY_P2_GATE_NAMES
    )


def test_shard_reports_survive_the_disk_round_trip(prerequisites) -> None:
    config = EcologyP2Config()
    original = _passing_matrix(config, prerequisites)[0]
    restored = shard_report_from_dict(
        json.loads(json.dumps(original.to_dict()))
    )
    assert restored == original

    drifted = json.loads(json.dumps(original.to_dict()))
    drifted["arm_spec"]["prediction_error_enabled"] = False
    with pytest.raises(ValueError, match="drifted from the pre-registered"):
        shard_report_from_dict(drifted)

    stale = json.loads(json.dumps(original.to_dict()))
    stale["schema_version"] = "digital-ant-ecology-p2-shard.v1"
    with pytest.raises(ValueError, match="shard schema mismatch"):
        shard_report_from_dict(stale)


# ---------------------------------------------------------------------------
# End-to-end shard execution (small budget)
# ---------------------------------------------------------------------------


async def test_baseline_shard_runs_and_journals_every_heldout_layout(
    tmp_path, prerequisite
) -> None:
    config = _small_config()
    progress_dir = tmp_path / "progress"
    report = await run_ecology_p2_shard(
        config,
        training_seed=17,
        arm="random",
        source_provenance=_SOURCE,
        prerequisite=prerequisite,
        progress_dir=progress_dir,
    )

    assert report.schema_version == ECOLOGY_P2_SHARD_SCHEMA_VERSION
    assert report.training_complete is True
    assert report.preflight is False
    assert len(report.layout_results) == len(_evaluation_specs())
    assert {item.capability for item in report.layout_results} == {
        capability for capability, _, _ in _evaluation_specs()
    }
    assert report.probe_summary is None

    state = json.loads(
        (progress_dir / "seed17" / "random" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["preregistration_digest"] == preregistration_digest(config)
    assert state["p1_report_sha256"] == prerequisite.report_sha256
    assert len(state["layout_results"]) == len(_evaluation_specs())

    # A resumed shard reuses journalled rows and converges to the same report.
    resumed = await run_ecology_p2_shard(
        config,
        training_seed=17,
        arm="random",
        source_provenance=_SOURCE,
        prerequisite=prerequisite,
        progress_dir=progress_dir,
    )
    assert resumed.layout_results == report.layout_results

    # A different pre-registration must never merge into an existing journal.
    with pytest.raises(ValueError, match="progress mismatch"):
        await run_ecology_p2_shard(
            replace(config, heldout_rounds=3),
            training_seed=17,
            arm="random",
            source_provenance=_SOURCE,
            prerequisite=prerequisite,
            progress_dir=progress_dir,
        )


async def test_learning_shard_trains_journals_and_pauses_on_budget(
    tmp_path, prerequisite
) -> None:
    config = _small_config()
    progress_dir = tmp_path / "progress"

    with pytest.raises(EcologyP2ProgressPaused) as paused:
        await run_ecology_p2_shard(
            config,
            training_seed=17,
            arm="learned",
            source_provenance=_SOURCE,
            prerequisite=prerequisite,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    assert paused.value.completed_work_items == 1
    state_path = progress_dir / "seed17" / "learned" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed_training_episodes"] == 1
    assert state["training_complete"] is False
    assert state["checkpoint_sha256"]

    with pytest.raises(EcologyP2ProgressPaused):
        await run_ecology_p2_shard(
            config,
            training_seed=17,
            arm="learned",
            source_provenance=_SOURCE,
            prerequisite=prerequisite,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed_training_episodes"] == 2

    # An interrupted shard must never supply a promotion checkpoint.
    with pytest.raises(ValueError, match="not training-complete"):
        load_shard_checkpoint_archives(
            progress_dir=progress_dir,
            config=config,
            training_seed=17,
            arm="learned",
            prerequisite=prerequisite,
        )
    # Neither may a shard that was never journalled at all.
    with pytest.raises(ValueError, match="no P2 shard journal"):
        load_shard_checkpoint_archives(
            progress_dir=progress_dir,
            config=config,
            training_seed=17,
            arm="pe_off",
            prerequisite=prerequisite,
        )


async def test_preflight_blocks_with_an_artifact_when_an_arm_cannot_run(
    prerequisite,
) -> None:
    """P2-A must refuse the rehearsal AND leave a report behind.

    The default preflight set is the P2-B core matrix, which contains
    ``eta_off``. Raising spent no budget but produced no artifact either, so
    the driver exited with nothing on disk -- plan section 2.3 requires every
    stage to leave a report, and a blocked one has to say what is missing.
    """

    assert "eta_off" in ECOLOGY_P2_CORE_ARM_NAMES
    report = await run_ecology_p2_preflight(
        _small_config(),
        source_provenance=_SOURCE,
        training_seed=17,
        prerequisite=prerequisite,
    )
    assert report.passed is False
    assert report.arms == ECOLOGY_P2_CORE_ARM_NAMES
    assert report.breakpoints == (
        "unexecutable_arms=['eta_off(temporal_policy_kind+"
        "joint_ssl_interval+joint_rl_interval)']",
    )
    # Zero budget was spent: no arm ran, so no wall clock and no archive.
    assert report.shard_wall_clock_seconds == ()
    assert report.shard_archive_size_bytes == ()
    assert report.determinism_repeat_matches is False
    # The exact missing hop, named in the artifact an operator reads.
    assert "_session_config" in report.description
    assert "temporal_policy" in report.description
    assert "joint_schedule" in report.description


async def test_preflight_rehearses_executable_arms_and_checks_determinism(
    tmp_path, prerequisite
) -> None:
    """P2-A on arms that CAN run: timing, artifact size and a replay probe."""

    config = _small_config()
    report = await run_ecology_p2_preflight(
        config,
        source_provenance=_SOURCE,
        training_seed=17,
        prerequisite=prerequisite,
        progress_dir=tmp_path / "progress",
        arms=("fixed_rule", "random"),
    )
    assert report.training_seed == 17
    assert report.arms == ("fixed_rule", "random")
    assert report.preregistration_digest == preregistration_digest(config)
    assert report.determinism_repeat_matches is True, report.determinism_detail
    assert {arm for arm, _ in report.shard_wall_clock_seconds} == {
        "fixed_rule",
        "random",
    }
    # A rehearsal below the frozen formal budget is still a rehearsal, but it
    # may never report ``passed``.
    assert report.passed is False
    assert any(
        item.startswith("below_formal_budget") for item in report.breakpoints
    )

    with pytest.raises(ValueError, match="unknown P2 preflight arms"):
        await run_ecology_p2_preflight(
            config,
            source_provenance=_SOURCE,
            training_seed=17,
            prerequisite=prerequisite,
            arms=("not_an_arm",),
        )


async def test_preflight_refuses_to_start_without_a_p1_report() -> None:
    with pytest.raises(EcologyP2PrerequisiteError, match="frozen P1 report"):
        await run_ecology_p2_preflight(
            _small_config(),
            source_provenance=_SOURCE,
            training_seed=17,
        )


async def test_cold_shard_trains_nothing_but_still_evaluates(
    tmp_path, prerequisite
) -> None:
    config = _small_config()
    report = await run_ecology_p2_shard(
        config,
        training_seed=17,
        arm="cold",
        source_provenance=_SOURCE,
        prerequisite=prerequisite,
        progress_dir=tmp_path / "progress",
    )
    assert report.scheduled_training_episodes == 0
    assert report.completed_training_episodes == 0
    assert report.training_complete is True
    assert report.archive_roundtrip_ok is None
    assert len(report.layout_results) == len(_evaluation_specs())
    assert all(
        item.policy_fingerprint_stable for item in report.layout_results
    )
