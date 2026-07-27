"""Small-budget smoke plus curriculum-semantics contracts.

The geometry and mastery assertions here are the frozen record of the P1
repair: near layouts must not be solvable by standing still or spinning, and
mastery must be a per-layout / per-body success rate rather than a
stage-cumulative event count.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from volvence_ant.env.ant_world import AntWorldConfig
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.world_objects import BurningMatch, ButterSource
from volvence_ant.evidence.ecology_checkpoint import (
    AntArtifactIntegrityError,
    _validated_report_verdict,
)
from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_CURRICULUM_SCHEMA_VERSION,
    ECOLOGY_HEAT_FREE_DELIVERY_CLEARANCE,
    ECOLOGY_HEAT_FREE_PICKUP_LOBE,
    ECOLOGY_MEASURED_POLICY_STEP,
    ECOLOGY_MIN_DELIVERY_LEG,
    ECOLOGY_NEST_RADIUS,
    ECOLOGY_PLANT_MAX_TURN_RATE,
    ECOLOGY_PLANT_STEP_CEILING,
    ECOLOGY_PLANT_STEP_FLOOR,
    ECOLOGY_REQUIRED_GATE_NAMES,
    ECOLOGY_TIER_GEOMETRY,
    ECOLOGY_UNDIRECTED_ORBIT_REACH,
    EcologyBodyEpisodeLineage,
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyLayoutCapability,
    EcologyMilestoneBudgetError,
    EcologyStage,
    EcologyTierGeometry,
    EcologyTrainingEpisodePlan,
    EcologyTrainingEpisodeReport,
    EcologyTrainingTier,
    _ECOLOGY_MATCH_HARM_RADIUS,
    _scene_objects,
    _stage_mastery,
    _synchronize_curriculum_navigators,
    _train_arm,
    _world,
    _session_config,
    ecology_heat_layout_clearances,
    ecology_schedule_milestone_shortfalls,
    ecology_tier_round_budget,
    ecology_training_min_stage_rounds,
    train_and_evaluate_ecology_checkpoint,
)
from volvence_ant.runtime import KernelColonyRunner


def _config(**overrides: object) -> EcologyCurriculumConfig:
    base: dict[str, object] = {
        "n_ants": 2,
        "temporal_latent_dim": 4,
        "stage_rounds": 1,
        "stage_episodes": 1,
        "mastery_min_episodes": 1,
        "validation_rounds": 1,
        "validation_seeds": (13,),
        "heldout_rounds": 1,
        "heldout_seeds": (19,),
        "seed": 2,
        # plan section 7 step 3 small-budget tier: a 1-round episode cannot
        # reach any tier's milestone, so the schedule refusal has to be
        # declared off explicitly. Formal lanes leave it on.
        "milestone_budget_enforced": False,
    }
    base.update(overrides)
    return EcologyCurriculumConfig(**base)  # type: ignore[arg-type]


def _lineage(
    *,
    body_id: int,
    picked_up: bool,
    delivered: bool,
    heat_escapes: int = 0,
    harmful_heat_ticks: int = 0,
    total_ticks: int = 24,
) -> EcologyBodyEpisodeLineage:
    return EcologyBodyEpisodeLineage(
        body_id=body_id,
        episode_id=f"test:{body_id}",
        layout_seed=7,
        stage=EcologyStage.BUTTER,
        tier=EcologyTrainingTier.MEDIUM,
        encountered_food=picked_up,
        encountered_heat=harmful_heat_ticks > 0,
        picked_up=picked_up,
        delivered=delivered,
        pickup_tick=0 if picked_up else None,
        delivery_tick=1 if delivered else None,
        harmful_heat_ticks=harmful_heat_ticks,
        heat_entries=heat_escapes,
        heat_escapes=heat_escapes,
        escape_latencies=(3,) * heat_escapes,
        applied_distance=1.0,
        switch_count=0,
        non_timeout_segment_closures=0,
        timed_out=not delivered,
        total_ticks=total_ticks,
    )


def _episode(
    *,
    stage: EcologyStage,
    tier: EcologyTrainingTier,
    seed: int,
    lineage: tuple[EcologyBodyEpisodeLineage, ...],
    forced_escape: bool = False,
    interleaved: bool = False,
    milestone_samplable: bool = True,
) -> EcologyTrainingEpisodeReport:
    plan = EcologyTrainingEpisodePlan(
        stage=stage,
        tier=tier,
        seed=seed,
        episode_index=0,
        interleaved=interleaved,
        forced_escape=forced_escape,
    )
    return EcologyTrainingEpisodeReport(
        arm="learned",
        plan=plan,
        pickups=sum(item.picked_up for item in lineage),
        deliveries=sum(item.delivered for item in lineage),
        obstacle_contacts=0,
        heat_entries=sum(item.heat_entries for item in lineage),
        heat_escapes=sum(item.heat_escapes for item in lineage),
        nonzero_ecology_payoffs=0,
        activated_sense_channels=(),
        minimum_food_distance=None,
        minimum_obstacle_distance=None,
        minimum_heat_distance=None,
        switch_count=0,
        mean_persistence_steps=0.0,
        closed_segment_count=0,
        longest_segment_length=0,
        policy_fingerprints_before=(),
        policy_fingerprints_after=(),
        memory_entries_evicted=0,
        rounds=(
            ecology_tier_round_budget(tier)
            if milestone_samplable
            else ecology_tier_round_budget(tier) - 1
        ),
        milestone_round_budget=ecology_tier_round_budget(tier),
        milestone_samplable=milestone_samplable,
        body_lineage=lineage,
    )


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "retired",
    [
        # v9's forced-return pressure rewarded a straight non-delivery path.
        "digital-ant-ecology-curriculum.v9",
        # v8's milestone budgets and mastery eligibility are not v9's.
        "digital-ant-ecology-curriculum.v8",
        "digital-ant-ecology-curriculum.v7",
    ],
)
def test_curriculum_schema_bump_rejects_earlier_reports(
    retired: str,
) -> None:
    """v10 semantics must never be read out of a v9 (or older) journal."""

    assert ECOLOGY_CURRICULUM_SCHEMA_VERSION == (
        "digital-ant-ecology-curriculum.v10"
    )
    legacy = {
        "schema_version": retired,
        "verdict": "PASS",
        "gates": [
            {"name": name, "passed": True}
            for name in ECOLOGY_REQUIRED_GATE_NAMES
        ],
    }

    with pytest.raises(AntArtifactIntegrityError, match="unexpected ecology"):
        _validated_report_verdict(legacy)


def test_required_gates_publish_layout_mastery_and_budget() -> None:
    """The retired event-count gate name must not ship under v9."""

    assert "training_event_coverage" not in ECOLOGY_REQUIRED_GATE_NAMES
    assert ECOLOGY_REQUIRED_GATE_NAMES[:2] == (
        "training_layout_mastery",
        "training_tier_milestone_samplable",
    )


def test_plan_mastery_thresholds_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="frozen at 0.6"):
        _config(mastery_layout_success_ratio=0.4)
    with pytest.raises(ValueError, match="frozen at 0.6"):
        _config(mastery_body_success_ratio=0.5)
    with pytest.raises(ValueError, match="frozen at 0.05"):
        _config(mastery_harmful_tick_rate_max=0.2)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_every_tier_pickup_disc_clears_the_delivery_disc_and_orbit() -> None:
    """Both degeneracies that made the near tier a free pickup are closed."""

    for tier, geometry in ECOLOGY_TIER_GEOMETRY.items():
        # A body that never leaves the nest cannot reach the pickup disc, and
        # the pickup point is never already inside the delivery disc.
        assert geometry.min_delivery_leg >= ECOLOGY_MIN_DELIVERY_LEG, tier
        # Undirected maximum-curvature spinning cannot sweep the pickup disc.
        assert geometry.min_outbound_leg > ECOLOGY_UNDIRECTED_ORBIT_REACH, tier


def test_frozen_tier_geometry_cannot_be_edited_after_import() -> None:
    """``_validate_tier_geometry`` runs once, so the table must be read-only.

    A plain dict would let any importer install a tier the validator never
    saw, and every budget, gate and layout downstream would then describe a
    world nothing checked.
    """

    forged = EcologyTierGeometry(
        pickup_radius=1.1,
        distance_min=0.9,
        distance_max=1.0,
    )
    with pytest.raises(TypeError):
        ECOLOGY_TIER_GEOMETRY[  # type: ignore[index]
            EcologyTrainingTier.NEAR
        ] = forged
    with pytest.raises(AttributeError):
        ECOLOGY_TIER_GEOMETRY.update(  # type: ignore[attr-defined]
            {EcologyTrainingTier.NEAR: forged}
        )
    assert ECOLOGY_TIER_GEOMETRY[EcologyTrainingTier.NEAR] != forged


def test_heat_layouts_admit_a_zero_harmful_tick_route() -> None:
    """plan 4.4's 5% harmful bound must grade the policy, not the layout.

    The plane minus a closed disc is path-connected, so a harm-free
    pickup->delivery route exists on every layout whose delivery point is
    outside the harm disc and whose pickup disc keeps a lobe outside it.  Both
    clearances are checked here on the layouts the generator actually emits,
    against the analytic worst case the import-time validator uses.
    """

    config = _config(n_ants=1)
    for tier, geometry in ECOLOGY_TIER_GEOMETRY.items():
        worst_delivery, worst_lobe = ecology_heat_layout_clearances(
            tier,
            food_distance=geometry.distance_min,
        )
        assert worst_delivery >= ECOLOGY_HEAT_FREE_DELIVERY_CLEARANCE, tier
        assert worst_lobe >= ECOLOGY_HEAT_FREE_PICKUP_LOBE, tier
        for stage in (EcologyStage.BURNING_MATCH, EcologyStage.COMPOSITE):
            for seed in range(8):
                world = _world(
                    config=config,
                    stage=stage,
                    seed=seed,
                    data_split=EcologyDataSplit.TRAIN,
                    tier=tier,
                )
                match = next(
                    item
                    for item in world.world_objects()
                    if isinstance(item, BurningMatch)
                )
                butter = next(
                    item
                    for item in world.world_objects()
                    if isinstance(item, ButterSource)
                )
                # A body can spawn, and deliver, without burning.
                assert not world.heat_harmful(*world.nest)
                delivery = math.hypot(
                    match.x - world.nest[0],
                    match.y - world.nest[1],
                ) - match.harm_radius
                assert delivery >= worst_delivery - 1e-9
                # ...and a pickup point exists outside the harm disc.
                lobe = (
                    math.hypot(match.x - butter.x, match.y - butter.y)
                    + butter.radius
                    - match.harm_radius
                )
                assert lobe >= worst_lobe - 1e-9
                assert match.harm_radius == pytest.approx(
                    _ECOLOGY_MATCH_HARM_RADIUS
                )


def test_layouts_are_not_solvable_without_directed_motion() -> None:
    """Measured on the real world contacts, not on the geometry alone.

    Before the repair a frozen body delivered on 34% of near layouts, the
    measured-speed max-curvature orbit picked up on 98.5% of them, and the
    full-speed orbit still reached 32/96 near and 1/96 medium layouts after
    the first geometry pass.
    """

    config = _config(n_ants=4)
    # "frozen" never moves. The orbit policies hold maximum curvature at the
    # plant's slowest, measured and fastest speeds; the fastest traces the
    # WIDEST max-curvature circle and is the strongest of the three.
    policies = {
        "frozen": (0.0, 0.0),
        "orbit_floor": (math.radians(45.0), ECOLOGY_PLANT_STEP_FLOOR),
        "orbit_measured": (math.radians(45.0), ECOLOGY_MEASURED_POLICY_STEP),
        "orbit_ceiling": (
            math.radians(45.0),
            ECOLOGY_PLANT_STEP_CEILING,
        ),
    }
    outcomes = dict.fromkeys(policies, 0)
    layouts = 48
    for tier in EcologyTrainingTier:
        for name, (turn, step) in policies.items():
            for seed in range(layouts):
                world = _world(
                    config=config,
                    stage=EcologyStage.BUTTER,
                    seed=seed,
                    data_split=EcologyDataSplit.TRAIN,
                    tier=tier,
                )
                for _ in range(24):
                    for body_id in range(config.n_ants):
                        world.act(
                            turn_command=turn,
                            step_command=step,
                            body_id=body_id,
                        )
                outcomes[name] += int(world.food_pickups > 0)

    assert outcomes == dict.fromkeys(policies, 0)


def test_near_layouts_stay_reachable_for_a_straight_running_body() -> None:
    """Stage-0 bootstrap still works: near must not become unsamplable."""

    geometry = ECOLOGY_TIER_GEOMETRY[EcologyTrainingTier.NEAR]
    solved = 0
    layouts = 32
    for seed in range(layouts):
        objects = _scene_objects(
            stage=EcologyStage.BUTTER,
            seed=seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        )
        butter = next(
            item for item in objects if isinstance(item, ButterSource)
        )
        world = ColonyWorld(
            config=AntWorldConfig(
                seed=seed,
                antenna_offset_deg=45.0,
                antenna_reach=0.9,
                nest_radius=ECOLOGY_NEST_RADIUS,
            ),
            world_objects=objects,
            n_bodies=1,
        )
        # Straight out at the plant's SLOWEST step, then straight back: the
        # cheapest directed policy the graded controller can express.
        outbound = math.atan2(butter.y, butter.x)
        world.set_body_pose(x=0.0, y=0.0, heading=outbound)
        for _ in range(
            ecology_tier_round_budget(EcologyTrainingTier.NEAR)
        ):
            body = world.body()
            heading = (
                outbound
                if not body.carrying_food
                else math.atan2(-body.y, -body.x)
            )
            world.set_body_pose(x=body.x, y=body.y, heading=heading)
            world.act(
                turn_command=0.0,
                step_command=ECOLOGY_PLANT_STEP_FLOOR,
                body_id=0,
            )
        solved += int(world.food_delivered > 0)

    # The published near budget must cover EVERY near layout, not most.
    assert solved == layouts
    assert geometry.worst_case_round_trip <= (
        ecology_tier_round_budget(EcologyTrainingTier.NEAR)
        * ECOLOGY_PLANT_STEP_FLOOR
    )


def test_forced_approach_no_longer_leaves_the_delivery_leg_free() -> None:
    """A forced-approach pickup always leaves a real carry home.

    ``forced_approach`` only ever fixed the OUTBOUND leg: with the retired
    geometry the pickup point was still inside the delivery disc, so delivery
    followed for free.  Drive each body straight at the butter and check the
    pose at the tick the pickup actually lands.
    """

    config = _config(n_ants=4)
    checked = 0
    for seed in range(8):
        world = _world(
            config=config,
            stage=EcologyStage.BUTTER,
            seed=seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
            forced_approach=True,
        )
        butter = next(
            item
            for item in world.world_objects()
            if isinstance(item, ButterSource)
        )
        for body_id in range(config.n_ants):
            for _ in range(32):
                body = world.body(body_id)
                if body.carrying_food:
                    break
                world.set_body_pose(
                    body_id=body_id,
                    x=body.x,
                    y=body.y,
                    heading=math.atan2(
                        butter.y - body.y,
                        butter.x - body.x,
                    ),
                )
                world.act(
                    turn_command=0.0,
                    step_command=ECOLOGY_PLANT_STEP_FLOOR,
                    body_id=body_id,
                )
            carrying = world.body(body_id)
            assert carrying.carrying_food
            home_distance = math.hypot(carrying.x, carrying.y)
            assert home_distance > ECOLOGY_NEST_RADIUS
            checked += 1

    assert checked == 8 * config.n_ants


def test_tier_round_budget_exposes_the_far_training_shortfall() -> None:
    """No tier can sample its own milestone inside a 24-round episode.

    The retired derivation counted straight legs only and concluded that near
    (12) and medium (16) fitted a 24-round episode while far (32) did not, so
    the defect was published as far-specific.  Against the frozen plant that
    was never true: bodies spawn at the nest with a uniform random heading and
    the plant cannot stop to re-aim, so every leg also pays a heading
    correction and all three tiers overrun 24 rounds.
    """

    budgets = {
        tier: ecology_tier_round_budget(tier)
        for tier in EcologyTrainingTier
    }
    assert budgets[EcologyTrainingTier.NEAR] < budgets[
        EcologyTrainingTier.MEDIUM
    ] < budgets[EcologyTrainingTier.FAR]
    # The retired P1 budget was 24; the curriculum owner now publishes the
    # sufficient floor and the formal P1 default consumes it.
    assert min(budgets.values()) > 24
    assert ecology_training_min_stage_rounds() == max(budgets.values())
    far = ECOLOGY_TIER_GEOMETRY[EcologyTrainingTier.FAR]
    # Even at the audit-measured policy speed the worst far layout does not
    # fit; this is a statement about the world, not about a threshold.
    assert far.worst_case_round_trip > 24 * ECOLOGY_MEASURED_POLICY_STEP


def test_round_budget_charges_the_heading_correction_every_body_pays() -> None:
    """The published budget must be sufficient, not just the straight legs.

    A body that runs the BEST trajectory the graded plant can express -- turn
    at ``max_turn_rate`` toward the target, hold the floor speed the gate
    names -- is simulated here against the real ``AntWorld``.  The worst spawn
    heading is a half turn away from the food, which is exactly the case the
    straight-leg-only budget ignored.
    """

    config = _config(n_ants=1)
    for tier in EcologyTrainingTier:
        geometry = ECOLOGY_TIER_GEOMETRY[tier]
        straight_legs = math.ceil(
            geometry.max_outbound_leg / ECOLOGY_PLANT_STEP_FLOOR
        ) + math.ceil(
            (
                geometry.max_outbound_leg
                + ECOLOGY_PLANT_STEP_FLOOR
                - ECOLOGY_NEST_RADIUS
            )
            / ECOLOGY_PLANT_STEP_FLOOR
        )
        budget = ecology_tier_round_budget(tier)
        # The correction is charged, not assumed away.
        assert budget > straight_legs

        worst = 0
        for seed in range(6):
            world = _world(
                config=config,
                stage=EcologyStage.BUTTER,
                seed=seed,
                data_split=EcologyDataSplit.TRAIN,
                tier=tier,
            )
            butter = next(
                item
                for item in world.world_objects()
                if isinstance(item, ButterSource)
            )
            body = world.body(0)
            # Worst case: spawn pointing exactly away from the food.
            world.set_body_pose(
                body_id=0,
                x=body.x,
                y=body.y,
                heading=math.atan2(butter.y, butter.x) + math.pi,
            )
            # The budget is derived from this constant, so it must be the
            # plant's own turn rate and not a restated literal.
            assert world.config.max_turn_rate == pytest.approx(
                ECOLOGY_PLANT_MAX_TURN_RATE
            )
            rounds = 0
            while rounds < 4 * budget and world.food_delivered == 0:
                body = world.body(0)
                target = (
                    world.nest
                    if body.carrying_food
                    else (butter.x, butter.y)
                )
                desired = math.atan2(
                    target[1] - body.y,
                    target[0] - body.x,
                )
                relative = (
                    desired - body.heading + math.pi
                ) % (2.0 * math.pi) - math.pi
                world.act(
                    turn_command=max(
                        -world.config.max_turn_rate,
                        min(world.config.max_turn_rate, relative),
                    ),
                    # The graded plant cannot stop; the gate names the floor.
                    step_command=ECOLOGY_PLANT_STEP_FLOOR,
                    body_id=0,
                )
                rounds += 1
            assert world.food_delivered > 0
            worst = max(worst, rounds)
        # Sufficiency: the published budget covers the best expressible
        # policy on every layout, which the straight-leg budget did not.
        assert worst <= budget
        assert worst > straight_legs


def test_train_arm_refuses_a_schedule_it_cannot_sample() -> None:
    """The far decision is enforced where every lane passes, not in a gate.

    ``training_tier_milestone_samplable`` lives in ``_build_gates``, which the
    fixed-schedule P1/P2 lanes never call, so a schedule refusal has to sit on
    the shared training entry point.
    """

    schedule = tuple(
        EcologyTrainingEpisodePlan(
            stage=EcologyStage.BUTTER,
            tier=tier,
            seed=11 + index,
            episode_index=index,
            interleaved=False,
            forced_escape=False,
        )
        for index, tier in enumerate(EcologyTrainingTier)
    )
    shortfalls = ecology_schedule_milestone_shortfalls(
        schedule,
        stage_rounds=24,
    )
    assert {row[1] for row in shortfalls} == {
        tier.value for tier in EcologyTrainingTier
    }
    assert all(row[3] == 24 < row[4] for row in shortfalls)
    assert not ecology_schedule_milestone_shortfalls(
        schedule,
        stage_rounds=ecology_training_min_stage_rounds(),
    )

    # ``initial=()`` is deliberately invalid: the refusal must fire BEFORE
    # ``_train_arm`` touches a checkpoint or opens a world, so a build that
    # dropped it does not merely mis-report -- it walks on into the colony
    # runner with an empty checkpoint tuple.
    with pytest.raises(EcologyMilestoneBudgetError, match="far"):
        asyncio.run(
            _train_arm(
                config=_config(
                    stage_rounds=24,
                    milestone_budget_enforced=True,
                ),
                initial=(),
                arm="learned",
                optimize=True,
                local_valence_enabled=True,
                segment_credit_enabled=True,
                schedule=schedule,
            )
        )

    # The generated schedule is built one episode at a time, so it needs the
    # same refusal or the curriculum's own lane would be the way around it.
    with pytest.raises(EcologyMilestoneBudgetError, match="near"):
        asyncio.run(
            _train_arm(
                config=_config(
                    stage_rounds=24,
                    milestone_budget_enforced=True,
                ),
                initial=(),
                arm="learned",
                optimize=True,
                local_valence_enabled=True,
                segment_credit_enabled=True,
            )
        )


def test_layout_that_could_not_sample_its_milestone_is_not_eligible() -> None:
    """A too-short episode cannot certify a capability either way."""

    config = _config(n_ants=4, stage_episodes=1, mastery_min_episodes=1)
    lineage = tuple(
        _lineage(body_id=body_id, picked_up=True, delivered=True)
        for body_id in range(4)
    )
    samplable = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=(
            _episode(
                stage=EcologyStage.BUTTER,
                tier=EcologyTrainingTier.MEDIUM,
                seed=41,
                lineage=lineage,
            ),
        ),
    )
    assert samplable.reached
    assert samplable.layout_results[0].mastery_eligible

    starved = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=(
            _episode(
                stage=EcologyStage.BUTTER,
                tier=EcologyTrainingTier.MEDIUM,
                seed=41,
                lineage=lineage,
                milestone_samplable=False,
            ),
        ),
    )
    # Same four perfect bodies, but the episode was one round too short.
    assert not starved.reached
    assert not starved.layout_results[0].mastery_eligible
    assert "cannot certify" in starved.layout_results[0].eligibility_reason


# ---------------------------------------------------------------------------
# mastery semantics
# ---------------------------------------------------------------------------


def test_stage_mastery_rejects_aggregate_event_counts() -> None:
    """Many events, no body closing the loop -> no mastery."""

    config = _config(n_ants=4, stage_episodes=3, mastery_min_episodes=3)
    # Six pickups and three deliveries across the stage, but pickup and
    # delivery never happen in the SAME body: the retired v7 rule
    # (pickups>=2 and deliveries>=1) would have declared mastery.
    reports = tuple(
        _episode(
            stage=EcologyStage.BUTTER,
            tier=EcologyTrainingTier.MEDIUM,
            seed=100 + index,
            lineage=(
                _lineage(body_id=0, picked_up=True, delivered=False),
                _lineage(body_id=1, picked_up=True, delivered=False),
                _lineage(body_id=2, picked_up=False, delivered=True),
                _lineage(body_id=3, picked_up=False, delivered=False),
            ),
        )
        for index in range(3)
    )
    mastery = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=reports,
    )

    assert mastery.mastery_layouts == 3
    assert mastery.successful_layouts == 0
    assert not mastery.reached
    assert all(
        item.successful_bodies == 0 for item in mastery.layout_results
    )


def test_stage_mastery_needs_a_body_success_rate_per_layout() -> None:
    config = _config(n_ants=4, stage_episodes=3, mastery_min_episodes=3)
    full = tuple(
        _lineage(body_id=index, picked_up=True, delivered=True)
        for index in range(3)
    ) + (_lineage(body_id=3, picked_up=False, delivered=False),)
    thin = (
        _lineage(body_id=0, picked_up=True, delivered=True),
        _lineage(body_id=1, picked_up=True, delivered=True),
        _lineage(body_id=2, picked_up=False, delivered=False),
        _lineage(body_id=3, picked_up=False, delivered=False),
    )
    reached = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=tuple(
            _episode(
                stage=EcologyStage.BUTTER,
                tier=EcologyTrainingTier.MEDIUM,
                seed=200 + index,
                lineage=full,
            )
            for index in range(3)
        ),
    )
    # 2/4 bodies is below the frozen 60% body threshold (needs 3 of 4).
    missed = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=tuple(
            _episode(
                stage=EcologyStage.BUTTER,
                tier=EcologyTrainingTier.MEDIUM,
                seed=300 + index,
                lineage=thin,
            )
            for index in range(3)
        ),
    )

    assert reached.required_bodies == 3
    assert reached.successful_layouts == 3
    assert reached.reached
    assert missed.successful_layouts == 0
    assert not missed.reached


def test_near_foraging_layouts_are_bootstrap_only() -> None:
    """plan 4.3 Stage 0: near results never make a capability gate pass."""

    config = _config(n_ants=4, stage_episodes=3, mastery_min_episodes=1)
    perfect = tuple(
        _lineage(body_id=index, picked_up=True, delivered=True)
        for index in range(4)
    )
    mastery = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=tuple(
            _episode(
                stage=EcologyStage.BUTTER,
                tier=EcologyTrainingTier.NEAR,
                seed=400 + index,
                lineage=perfect,
            )
            for index in range(3)
        ),
    )

    assert mastery.bootstrap_layouts == 3
    assert mastery.mastery_layouts == 0
    assert mastery.successful_layouts == 0
    assert not mastery.reached
    assert all(
        not item.mastery_eligible
        and "Stage-0 bootstrap" in item.eligibility_reason
        for item in mastery.layout_results
    )


def test_near_forced_escape_layouts_still_certify_escape() -> None:
    """Only FORAGING is bootstrap-limited at the near tier."""

    config = _config(n_ants=4, stage_episodes=1, mastery_min_episodes=1)
    mastery = _stage_mastery(
        config=config,
        stage=EcologyStage.BURNING_MATCH,
        reports=(
            _episode(
                stage=EcologyStage.BURNING_MATCH,
                tier=EcologyTrainingTier.NEAR,
                seed=500,
                forced_escape=True,
                lineage=tuple(
                    _lineage(
                        body_id=index,
                        picked_up=False,
                        delivered=False,
                        heat_escapes=1,
                        harmful_heat_ticks=4,
                    )
                    for index in range(4)
                ),
            ),
        ),
    )

    assert mastery.layout_results[0].capability is (
        EcologyLayoutCapability.FORCED_ESCAPE
    )
    assert mastery.mastery_layouts == 1
    assert mastery.reached


def test_escape_success_cannot_substitute_for_foraging_success() -> None:
    """A mixed stage must clear the rate on each capability separately."""

    config = _config(n_ants=4, stage_episodes=4, mastery_min_episodes=2)
    escaped = tuple(
        _lineage(
            body_id=index,
            picked_up=False,
            delivered=False,
            heat_escapes=1,
        )
        for index in range(4)
    )
    failed = tuple(
        _lineage(body_id=index, picked_up=True, delivered=False)
        for index in range(4)
    )
    mastery = _stage_mastery(
        config=config,
        stage=EcologyStage.BURNING_MATCH,
        # Three escape layouts pass, the single foraging layout fails.
        reports=tuple(
            _episode(
                stage=EcologyStage.BURNING_MATCH,
                tier=EcologyTrainingTier.NEAR,
                seed=800 + index,
                forced_escape=True,
                lineage=escaped,
            )
            for index in range(3)
        )
        + (
            _episode(
                stage=EcologyStage.BURNING_MATCH,
                tier=EcologyTrainingTier.FAR,
                seed=810,
                lineage=failed,
            ),
        ),
    )
    by_capability = {
        item.capability: item for item in mastery.capability_results
    }

    # Aggregated over the stage this is 3/4 successful layouts -- above the
    # frozen 60% -- yet the foraging capability was never demonstrated.
    assert mastery.successful_layouts == 3
    assert mastery.mastery_layouts == 4
    assert by_capability[EcologyLayoutCapability.FORCED_ESCAPE].reached
    assert not by_capability[EcologyLayoutCapability.FORAGING].reached
    assert not mastery.reached


def test_heat_foraging_layout_fails_on_harmful_exposure() -> None:
    """plan 4.4: route foraging needs the harmful tick rate held down too."""

    config = _config(n_ants=4, stage_episodes=1, mastery_min_episodes=1)
    lineage = tuple(
        _lineage(
            body_id=index,
            picked_up=True,
            delivered=True,
            harmful_heat_ticks=6,
            total_ticks=24,
        )
        for index in range(4)
    )
    mastery = _stage_mastery(
        config=config,
        stage=EcologyStage.COMPOSITE,
        reports=(
            _episode(
                stage=EcologyStage.COMPOSITE,
                tier=EcologyTrainingTier.FAR,
                seed=600,
                lineage=lineage,
            ),
        ),
    )
    result = mastery.layout_results[0]

    assert result.successful_bodies == 4
    assert result.harmful_tick_rate == pytest.approx(0.25)
    assert not result.safety_respected
    assert not result.layout_success
    assert not mastery.reached


def test_interleaved_replays_do_not_certify_a_stage() -> None:
    config = _config(n_ants=4, stage_episodes=1, mastery_min_episodes=1)
    perfect = tuple(
        _lineage(body_id=index, picked_up=True, delivered=True)
        for index in range(4)
    )
    mastery = _stage_mastery(
        config=config,
        stage=EcologyStage.BUTTER,
        reports=(
            _episode(
                stage=EcologyStage.BUTTER,
                tier=EcologyTrainingTier.MEDIUM,
                seed=700,
                lineage=perfect,
                interleaved=True,
            ),
        ),
    )

    assert mastery.mastery_layouts == 0
    assert not mastery.reached


# ---------------------------------------------------------------------------
# forced start conditions
# ---------------------------------------------------------------------------


def test_forced_return_curriculum_balances_state_without_action_labels() -> None:
    config = _config()
    world = _world(
        config=config,
        stage=EcologyStage.BUTTER,
        seed=10,
        data_split=EcologyDataSplit.TRAIN,
        tier=EcologyTrainingTier.NEAR,
        forced_return=True,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=10,
            session_id="forced-return-test",
            optimize=False,
        ),
    )

    _synchronize_curriculum_navigators(runner)

    assert all(world.body(body_id).carrying_food for body_id in range(2))
    assert all(
        session.navigator.state.home_distance > world.config.nest_radius
        for session in runner.sessions
    )
    home_sides = tuple(
        session.navigator.egocentric_home()[1]
        for session in runner.sessions
    )
    assert home_sides[0] * home_sides[1] < 0.0
    assert all(abs(value) == pytest.approx(1.0) for value in home_sides)

    # A zero-turn policy must not harvest dense home-progress while still
    # missing the delivery disc. Tangent starts make the first straight step
    # increase home distance for both left/right-balanced bodies.
    before = tuple(
        session.navigator.state.home_distance for session in runner.sessions
    )
    for body_id in range(2):
        world.act(
            turn_command=0.0,
            step_command=world.config.step_size * 0.5,
            body_id=body_id,
        )
    after = tuple(
        math.hypot(
            world.body(body_id).x - world.nest[0],
            world.body(body_id).y - world.nest[1],
        )
        for body_id in range(2)
    )
    assert all(
        current > initial
        for initial, current in zip(before, after, strict=True)
    )


def test_forced_approach_curriculum_demands_steering_without_action_labels() -> None:
    config = _config()
    world = _world(
        config=config,
        stage=EcologyStage.BUTTER,
        seed=10,
        data_split=EcologyDataSplit.TRAIN,
        tier=EcologyTrainingTier.NEAR,
        forced_approach=True,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=10,
            session_id="forced-approach-test",
            optimize=False,
        ),
    )

    _synchronize_curriculum_navigators(runner)

    butter = next(
        item
        for item in world.world_objects()
        if isinstance(item, ButterSource)
    )
    relative_bearings = []
    for body_id in range(2):
        body = world.body(body_id)
        # State only: nothing to carry, no free pickup at spawn.
        assert not body.carrying_food
        distance = math.hypot(body.x - butter.x, body.y - butter.y)
        assert butter.radius * 1.45 <= distance <= butter.radius * 2.9
        observation = world.observe(body_id)
        assert not observation.at_food
        # The scent gradient must be sensable from the spawn pose.
        assert observation.food_left + observation.food_right > 0.05
        # A straight path can never enter the pickup disc: the butter is
        # either behind the body or its closest approach stays outside the
        # disc, so only an active turn toward the gradient reaches reward.
        to_food = math.atan2(butter.y - body.y, butter.x - body.x)
        forward_projection = distance * math.cos(to_food - body.heading)
        if forward_projection > 0.0:
            closest_approach = distance * abs(
                math.sin(to_food - body.heading)
            )
            assert closest_approach > butter.radius
        # Path integration agrees with the true forced pose.
        assert (
            abs(
                runner.sessions[body_id].navigator.state.home_distance
                - observation.eval_home_distance
            )
            < 1e-6
        )
        relative_bearings.append(
            (to_food - body.heading + math.pi) % (2.0 * math.pi) - math.pi
        )
    # The required correction turn is side-balanced across bodies, so no
    # single turning direction can solve the block by accident.
    assert relative_bearings[0] * relative_bearings[1] < 0.0


# ---------------------------------------------------------------------------
# end-to-end smoke
# ---------------------------------------------------------------------------


async def test_ecology_curriculum_exports_checkpoint_and_honest_gates() -> None:
    candidate = await train_and_evaluate_ecology_checkpoint(
        EcologyCurriculumConfig(
            n_ants=1,
            temporal_latent_dim=4,
            stage_rounds=1,
            stage_episodes=1,
            mastery_min_episodes=1,
            validation_rounds=1,
            validation_seeds=(13,),
            heldout_rounds=3,
            heldout_seeds=(19,),
            seed=3,
            # plan section 7 step 3 small-budget smoke: the refusal is
            # declared off so the gate itself can be observed reporting the
            # shortfall instead of the run aborting.
            milestone_budget_enforced=False,
        )
    )

    assert len(candidate.checkpoints) == 1
    assert len(candidate.checkpoint_archives) == 1
    assert candidate.report.verdict in {"PASS", "BLOCK"}
    assert candidate.report.gates
    assert tuple(gate.name for gate in candidate.report.gates) == ECOLOGY_REQUIRED_GATE_NAMES
    assert len(candidate.report.learned_metrics) == 5
    assert {
        item.scenario.value
        for item in candidate.report.learned_metrics
    } == {
        "butter_only",
        "butter_with_neutral_stick",
        "heat_route_avoidance",
        "heat_forced_escape",
        "composite",
    }
    assert candidate.report.training_schedule
    assert candidate.report.learned_training
    # The ablation arm is published under its P1 name.
    assert candidate.report.dense_local_shaping_off_training
    assert all(
        item.arm == "dense_local_shaping_off"
        for item in candidate.report.dense_local_shaping_off_metrics
    )
    assert all(
        len(item.body_lineage) == 1
        and item.body_lineage[0].body_id == 0
        and item.body_lineage[0].episode_id
        and item.body_lineage[0].layout_seed == item.plan.seed
        for item in candidate.report.learned_training
    )
    # A 1-round episode cannot reach any milestone; the report must say so
    # rather than silently scoring the episode as a capability sample.
    assert all(
        not item.milestone_samplable
        and item.milestone_round_budget > item.rounds == 1
        for item in candidate.report.learned_training
    )
    samplability = next(
        gate
        for gate in candidate.report.gates
        if gate.name == "training_tier_milestone_samplable"
    )
    assert not samplability.passed
    assert candidate.report.action_probes
    assert all(
        item.policy_fingerprint_stable
        and item.temporal_learning_fingerprint_stable
        for item in candidate.report.learned_metrics
    )
    assert {
        gate.name for gate in candidate.report.gates
    } >= {
        "burning_match_route_avoidance",
        "burning_match_forced_escape",
        "checkpoint_archive_roundtrip",
    }
    assert candidate.report.learned_metrics[0].replay_captured >= 0
    assert all(
        item.replay_pending_captures >= 0 and item.replay_settlement_coverage == 1.0
        for item in candidate.report.learned_metrics
    )
    if candidate.report.verdict == "BLOCK":
        assert candidate.report.diagnostic_breakpoints
