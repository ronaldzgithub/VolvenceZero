"""P2 confirmatory matrix: serial gate, matched levers, journal and verdict."""

from __future__ import annotations

import json
from dataclasses import replace

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
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_GATE_NAMES,
    ECOLOGY_P1_SCHEMA_VERSION,
    EcologyP1Config,
    _evaluation_specs,
    _fixed_schedule,
)
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_ARM_NAMES,
    ECOLOGY_P2_ARM_SPEC_BY_NAME,
    ECOLOGY_P2_GATE_NAMES,
    ECOLOGY_P2_PRIMARY_ENDPOINTS,
    ECOLOGY_P2_SHARD_SCHEMA_VERSION,
    EcologyP2Config,
    EcologyP2LayoutResult,
    EcologyP2Prerequisite,
    EcologyP2PrerequisiteError,
    EcologyP2ProbeSummary,
    EcologyP2ProgressPaused,
    EcologyP2ShardReport,
    _heldout_seed,
    _holm_adjusted,
    _hierarchical_paired_bootstrap,
    aggregate_ecology_p2_shards,
    load_p1_prerequisite,
    outcome_score,
    p2_training_schedule,
    preregistration_digest,
    run_ecology_p2_shard,
    shard_report_from_dict,
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
):
    """Synthetic P1 artifact.

    This exists only to exercise the P2 serial gate; it is never evidence that
    the real P1 matrix passed.
    """

    payload = {
        "schema_version": schema_version,
        "verdict": verdict,
        "gates": [
            {
                "name": name,
                "passed": name != failed_gate,
                "observed": "synthetic",
                "threshold": "synthetic",
            }
            for name in gate_names
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _small_config() -> EcologyP2Config:
    return EcologyP2Config(
        n_ants=1,
        temporal_latent_dim=4,
        training_rounds=2,
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
) -> EcologyP2ShardReport:
    spec = ECOLOGY_P2_ARM_SPEC_BY_NAME[arm]
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
        policy_digest=f"digest:{arm}:{training_seed}",
        archive_roundtrip_ok=(
            True if spec.learning and spec.trains else None
        ),
        archive_size_bytes=1024 if spec.learning else None,
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
    prerequisite: EcologyP2Prerequisite,
) -> tuple[EcologyP2ShardReport, ...]:
    shards: list[EcologyP2ShardReport] = []
    for training_seed in config.training_seeds:
        for arm in ECOLOGY_P2_ARM_NAMES:
            learned = arm == "learned"
            shards.append(
                _shard(
                    config=config,
                    training_seed=training_seed,
                    arm=arm,
                    prerequisite=prerequisite,
                    successful=learned,
                    score=9.0 if learned else 1.0,
                    harmful_rate=0.0 if learned else 0.02,
                    composite_harmful_rate=(
                        0.0 if learned else 0.01
                    ),
                )
            )
    return tuple(shards)


@pytest.fixture()
def prerequisite(tmp_path) -> EcologyP2Prerequisite:
    return load_p1_prerequisite(_write_p1_report(tmp_path / "p1.json"))


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
            prerequisite=prerequisite,
        )


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


def test_pe_and_eta_levers_reach_the_session_contract() -> None:
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
    # PE-off must not silently take the ETA lever with it.
    assert pe_off.joint_apply_writeback is True

    eta_off = _config_for(temporal_writeback_enabled=False)
    assert eta_off.joint_apply_writeback is False
    assert eta_off.external_prediction_error_drive is True
    assert (
        eta_off.rollout_config.prediction_error_temporal_switch
        is WiringLevel.ACTIVE
    )


def test_arm_specs_differ_by_exactly_one_lever_from_learned() -> None:
    learned = ECOLOGY_P2_ARM_SPEC_BY_NAME["learned"]
    levers = (
        "optimize",
        "local_valence_enabled",
        "segment_credit_enabled",
        "prediction_error_enabled",
        "temporal_writeback_enabled",
        "trains",
    )
    for name in (
        "no_optimize",
        "cold",
        "pe_off",
        "eta_off",
        "dense_local_shaping_off",
        "segment_credit_off",
    ):
        spec = ECOLOGY_P2_ARM_SPEC_BY_NAME[name]
        differences = tuple(
            lever
            for lever in levers
            if getattr(spec, lever) != getattr(learned, lever)
        )
        assert differences != (), f"{name} is identical to learned"
        assert len(differences) == 1, f"{name} changes {differences}"


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
    prerequisite,
) -> None:
    config = EcologyP2Config()
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisite),
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
    }
    assert all(item.significant for item in report.paired_effects)
    assert all(item.ci_low > 0.0 for item in report.paired_effects)


def test_a_missing_or_preflight_shard_blocks_the_matrix(prerequisite) -> None:
    config = EcologyP2Config()
    full = _passing_matrix(config, prerequisite)

    thinned = tuple(
        item
        for item in full
        if not (item.arm == "eta_off" and item.training_seed == 1)
    )
    with pytest.raises(ValueError, match="missing a matched layout"):
        aggregate_ecology_p2_shards(
            thinned, worktree_clean=True, config=config
        )

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


def test_small_budget_evidence_cannot_claim_promotion(prerequisite) -> None:
    config = EcologyP2Config(
        n_ants=2,
        training_rounds=4,
        validation_rounds=4,
        heldout_rounds=4,
        layouts_per_tier=2,
        training_seeds=(0,),
    )
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisite),
        worktree_clean=True,
        config=config,
    )
    assert report.verdict == "BLOCK"
    assert "formal_configuration" in report.diagnostic_breakpoints


def test_dirty_worktree_and_blocked_p1_are_hard_failures(
    prerequisite,
) -> None:
    config = EcologyP2Config()
    dirty = aggregate_ecology_p2_shards(
        _passing_matrix(config, prerequisite),
        worktree_clean=False,
        config=config,
    )
    assert dirty.verdict == "BLOCK"
    assert "provenance_clean" in dirty.diagnostic_breakpoints

    blocked = replace(prerequisite, verdict="BLOCK")
    report = aggregate_ecology_p2_shards(
        _passing_matrix(config, blocked),
        worktree_clean=True,
        config=config,
    )
    assert report.verdict == "BLOCK"
    assert "p1_prerequisite_pass" in report.diagnostic_breakpoints


def test_learned_without_an_effect_blocks_the_causal_endpoints(
    prerequisite,
) -> None:
    config = EcologyP2Config()
    # Every arm scores identically: capability gates still pass but no paired
    # effect exists, so the learning and PE/ETA endpoints must block.
    flat = tuple(
        _shard(
            config=config,
            training_seed=training_seed,
            arm=arm,
            prerequisite=prerequisite,
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


def test_shard_reports_survive_the_disk_round_trip(prerequisite) -> None:
    config = EcologyP2Config()
    original = _passing_matrix(config, prerequisite)[0]
    restored = shard_report_from_dict(
        json.loads(json.dumps(original.to_dict()))
    )
    assert restored == original

    drifted = json.loads(json.dumps(original.to_dict()))
    drifted["arm_spec"]["prediction_error_enabled"] = False
    with pytest.raises(ValueError, match="drifted from the pre-registered"):
        shard_report_from_dict(drifted)


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
            prerequisite=prerequisite,
            progress_dir=progress_dir,
            max_new_work_items=1,
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed_training_episodes"] == 2


async def test_cold_shard_trains_nothing_but_still_evaluates(
    tmp_path, prerequisite
) -> None:
    config = _small_config()
    report = await run_ecology_p2_shard(
        config,
        training_seed=17,
        arm="cold",
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
