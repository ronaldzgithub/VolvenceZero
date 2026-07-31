"""Realtime app contracts and deterministic runner tests."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import json

import pytest

from volvence_zero.agent import decode_agent_learning_archive

from volvence_ant.app import (
    AntAppManager,
    AppArm,
    AppCommand,
    AppCommandKind,
    AppDisturbance,
    AppDisturbanceKind,
    AppExperimentConfig,
    AppMode,
    AppObjective,
    AppRunState,
    WorldObjectKind,
)
from volvence_ant.env import AntWorld, ButterSource
from volvence_ant.evidence.ecology_checkpoint import (
    EcologyP2PromotionEvidence,
    LoadedEcologyCheckpoint,
)
from volvence_ant.experiments.ecology_curriculum import (
    EcologyCurriculumConfig,
)
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_GATE_NAMES,
    ECOLOGY_P2_SCHEMA_VERSION,
    EcologyP2Config,
    heldout_layout_seeds,
)
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
)


def _p2_promotion(
    *, verdict: str = "PASS", n_ants: int = 1
) -> EcologyP2PromotionEvidence:
    """The confirmatory evidence a loadable checkpoint always carries.

    ``load_promoted_ecology_checkpoint`` refuses a bundle without it, so an
    in-process fixture that omits it is not a shape the admission point should
    accept either.  Synthetic: it exercises the admission contract and is never
    evidence that the real confirmatory matrix passed.
    """

    failed = () if verdict == "PASS" else (ECOLOGY_P2_GATE_NAMES[0],)
    return EcologyP2PromotionEvidence(
        schema_version=ECOLOGY_P2_SCHEMA_VERSION,
        preregistration_digest="d" * 64,
        verdict=verdict,
        gate_names=ECOLOGY_P2_GATE_NAMES,
        failed_gates=failed,
        training_seeds=(0, 1, 2),
        arms=("learned",),
        n_ants=n_ants,
        temporal_latent_dim=4,
        device="cpu",
        heldout_layout_seeds=heldout_layout_seeds(EcologyP2Config()),
        source_git_sha="a" * 40,
        report_path="synthetic-p2.json",
        report_sha256="b" * 64,
    )


def test_app_contracts_are_frozen_and_reject_motor_commands() -> None:
    config = AppExperimentConfig(autostart=False)
    with pytest.raises(FrozenInstanceError):
        config.seed = 3  # type: ignore[misc]

    assert tuple(AppCommand.__dataclass_fields__) == (
        "command_id",
        "kind",
        "value",
    )
    with pytest.raises(ValueError, match="does not accept"):
        AppCommand(
            command_id="pause-with-value",
            kind=AppCommandKind.PAUSE,
            value=1.0,
        )


def test_solo_contract_rejects_multiple_bodies() -> None:
    with pytest.raises(ValueError, match="solo mode"):
        AppExperimentConfig(mode=AppMode.SOLO, n_ants=2)


async def _wait_for_tick(run, tick: int) -> None:
    async with asyncio.timeout(3.0):
        while run.world.tick < tick:
            await asyncio.sleep(0.01)


async def test_disturbance_is_applied_before_one_authoritative_step() -> None:
    manager = AntAppManager()
    run = await manager.create_run(
        AppExperimentConfig(
            arm=AppArm.FIXED_RULE,
            autostart=False,
            tick_interval_ms=0,
            max_ticks=4,
        ),
        run_id="single-step",
    )
    await run.queue_disturbance(
        AppDisturbance(
            event_id="move-food",
            kind=AppDisturbanceKind.RELOCATE_FOOD,
            x=-4.0,
            y=4.0,
        )
    )
    await run.apply_command(AppCommand(command_id="step-1", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 1)

    assert run.state is AppRunState.PAUSED
    assert tuple((source.x, source.y) for source in run.world.food_sources()) == ((-4.0, 4.0),)
    replay = run.replay_payload()
    assert len(replay["frames"]) == 1
    audit = replay["audit_events"]
    applied = [event for event in audit if event["kind"] == "disturbance" and event["payload"]["status"] == "applied"]
    assert len(applied) == 1
    assert applied[0]["payload"]["applied_tick"] == 0
    await manager.close()


async def test_colony_frame_uses_public_world_and_pheromone_snapshots() -> None:
    manager = AntAppManager()
    run = await manager.create_run(
        AppExperimentConfig(
            mode=AppMode.COLONY,
            arm=AppArm.FIXED_RULE,
            n_ants=3,
            autostart=False,
            tick_interval_ms=0,
            max_ticks=2,
        ),
        run_id="colony",
    )
    await run.apply_command(AppCommand(command_id="step-colony", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 1)
    frame_events = [event for event in run.events_after(0) if event.kind.value == "frame"]
    assert len(frame_events) == 1
    frame = json.loads(frame_events[0].payload_json)
    assert len(frame["ants"]) == 3
    assert frame["tick"] == 1
    assert frame["mode"] == "colony"
    assert frame["trail"] and frame["trail"][0]
    await manager.close()


async def test_future_disturbance_waits_for_requested_tick_boundary() -> None:
    manager = AntAppManager()
    run = await manager.create_run(
        AppExperimentConfig(
            arm=AppArm.FIXED_RULE,
            autostart=False,
            tick_interval_ms=0,
            max_ticks=5,
        ),
        run_id="scheduled-disturbance",
    )
    original_food = tuple((source.x, source.y) for source in run.world.food_sources())
    await run.queue_disturbance(
        AppDisturbance(
            event_id="future-food",
            kind=AppDisturbanceKind.RELOCATE_FOOD,
            requested_tick=2,
            x=-6.0,
            y=-2.0,
        )
    )
    for target_tick in (1, 2):
        await run.apply_command(
            AppCommand(
                command_id=f"step-{target_tick}",
                kind=AppCommandKind.STEP,
            )
        )
        await _wait_for_tick(run, target_tick)
        assert tuple((source.x, source.y) for source in run.world.food_sources()) == original_food
    await run.apply_command(AppCommand(command_id="step-3", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 3)
    assert tuple((source.x, source.y) for source in run.world.food_sources()) == ((-6.0, -2.0),)
    await manager.close()


async def test_world_object_disturbances_require_ecology_objective() -> None:
    manager = AntAppManager()
    run = await manager.create_run(
        AppExperimentConfig(
            arm=AppArm.FIXED_RULE,
            objective=AppObjective.FORAGING,
            autostart=False,
        ),
        run_id="non-ecology-objects",
    )

    with pytest.raises(ValueError, match="requires objective=ecology"):
        await run.queue_disturbance(
            AppDisturbance(
                event_id="invalid-stick",
                kind=AppDisturbanceKind.UPSERT_WORLD_OBJECT,
                object_id="stick-1",
                object_kind=WorldObjectKind.WOOD_STICK,
                start_x=0.0,
                start_y=0.0,
                end_x=1.0,
                end_y=1.0,
            )
        )

    assert not run.world.world_object_snapshots()
    assert run.status().pending_disturbances == 0
    await manager.close()


async def test_ecology_object_upsert_move_and_remove_are_boundary_applied() -> None:
    manager = AntAppManager()
    run = await manager.create_run(
        AppExperimentConfig(
            arm=AppArm.FIXED_RULE,
            objective=AppObjective.ECOLOGY,
            autostart=False,
            tick_interval_ms=0,
            max_ticks=5,
        ),
        run_id="ecology-objects",
    )
    await run.queue_disturbance(
        AppDisturbance(
            event_id="place-stick",
            kind=AppDisturbanceKind.UPSERT_WORLD_OBJECT,
            object_id="stick-1",
            object_kind=WorldObjectKind.WOOD_STICK,
            start_x=1.0,
            start_y=-1.0,
            end_x=1.0,
            end_y=1.0,
        )
    )
    await run.apply_command(AppCommand(command_id="place-step", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 1)
    stick = next(
        item
        for item in run.world.world_object_snapshots()
        if item.object_id == "stick-1"
    )
    assert stick.center == (1.0, 0.0)

    await run.queue_disturbance(
        AppDisturbance(
            event_id="move-stick",
            kind=AppDisturbanceKind.MOVE_WORLD_OBJECT,
            object_id="stick-1",
            delta_x=2.0,
            delta_y=0.0,
        )
    )
    await run.apply_command(AppCommand(command_id="move-step", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 2)
    moved_stick = next(
        item
        for item in run.world.world_object_snapshots()
        if item.object_id == "stick-1"
    )
    assert moved_stick.center[0] == 3.0

    await run.queue_disturbance(
        AppDisturbance(
            event_id="remove-stick",
            kind=AppDisturbanceKind.REMOVE_WORLD_OBJECT,
            object_id="stick-1",
        )
    )
    await run.apply_command(AppCommand(command_id="remove-step", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 3)
    assert all(
        item.object_id != "stick-1"
        for item in run.world.world_object_snapshots()
    )
    await manager.close()


async def test_live_frame_backpressure_preserves_audit_events() -> None:
    manager = AntAppManager()
    run = await manager.create_run(
        AppExperimentConfig(
            arm=AppArm.FIXED_RULE,
            autostart=False,
            tick_interval_ms=0,
            max_ticks=None,
        ),
        run_id="backpressure",
    )
    await run.apply_command(AppCommand(command_id="resume", kind=AppCommandKind.RESUME))
    await _wait_for_tick(run, 70)
    await run.apply_command(AppCommand(command_id="pause", kind=AppCommandKind.PAUSE))
    status = run.status()
    assert status.frames_retained == 64
    assert status.frames_dropped >= 6
    assert any(event.kind.value == "status" for event in run.events_after(0))
    assert len(run.replay_payload()["frames"]) >= 70
    await manager.close()


async def test_learned_ecology_drag_drop_three_objects_keeps_kernel_arm() -> None:
    """Runtime drag/drop of butter, stick, and match must stay on learned kernel."""

    bootstrap = AntSession(
        AntWorld(world_objects=(ButterSource(object_id="butter", x=2.0, y=0.0),)),
        config=AntSessionConfig(
            temporal_latent_dim=4,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    checkpoint_archive = bootstrap.export_learning_checkpoint_archive(
        checkpoint_id="ecology-drag-demo"
    )
    checkpoint_info = decode_agent_learning_archive(checkpoint_archive).info
    loaded = LoadedEcologyCheckpoint(
        checkpoint_archives=(checkpoint_archive,),
        fingerprint=checkpoint_info.state_fingerprint,
        verdict="PASS",
        config=EcologyCurriculumConfig(
            n_ants=1,
            temporal_latent_dim=4,
            stage_rounds=1,
            stage_episodes=1,
            mastery_min_episodes=1,
            heldout_rounds=1,
            heldout_seeds=(1,),
        ),
        report_path="local-drag-demo",
        p2_promotion=_p2_promotion(),
    )
    manager = AntAppManager(ecology_checkpoint=loaded)
    run = await manager.create_run(
        AppExperimentConfig(
            objective=AppObjective.ECOLOGY,
            arm=AppArm.LEARNED,
            temporal_latent_dim=4,
            autostart=False,
            tick_interval_ms=0,
            max_ticks=6,
        ),
        run_id="drag-three-objects",
    )
    assert run.config.arm is AppArm.LEARNED

    await run.queue_disturbance(
        AppDisturbance(
            event_id="place-butter",
            kind=AppDisturbanceKind.UPSERT_WORLD_OBJECT,
            object_id="butter-demo",
            object_kind=WorldObjectKind.BUTTER,
            x=3.5,
            y=-1.2,
        )
    )
    await run.queue_disturbance(
        AppDisturbance(
            event_id="place-stick",
            kind=AppDisturbanceKind.UPSERT_WORLD_OBJECT,
            object_id="stick-demo",
            object_kind=WorldObjectKind.WOOD_STICK,
            start_x=1.2,
            start_y=-1.5,
            end_x=1.2,
            end_y=1.5,
        )
    )
    await run.queue_disturbance(
        AppDisturbance(
            event_id="place-match",
            kind=AppDisturbanceKind.UPSERT_WORLD_OBJECT,
            object_id="match-demo",
            object_kind=WorldObjectKind.BURNING_MATCH,
            x=-2.4,
            y=1.8,
        )
    )
    await run.apply_command(
        AppCommand(command_id="place-step", kind=AppCommandKind.STEP)
    )
    await _wait_for_tick(run, 1)

    object_ids = {
        item.object_id for item in run.world.world_object_snapshots()
    }
    assert {"butter-demo", "stick-demo", "match-demo"} <= object_ids

    await run.queue_disturbance(
        AppDisturbance(
            event_id="nudge-butter",
            kind=AppDisturbanceKind.MOVE_WORLD_OBJECT,
            object_id="butter-demo",
            delta_x=-1.0,
            delta_y=0.5,
        )
    )
    await run.apply_command(
        AppCommand(command_id="nudge-step", kind=AppCommandKind.STEP)
    )
    await _wait_for_tick(run, 2)

    frame_event = next(
        event for event in run.events_after(0) if event.kind.value == "frame"
    )
    frame = json.loads(frame_event.payload_json)
    assert frame["evidence"]["checkpoint_loaded"] is True
    assert frame["evidence"]["checkpoint_verdict"] == "PASS"
    assert frame["evidence"]["checkpoint_fingerprint"] == (
        checkpoint_info.state_fingerprint
    )
    assert run.config.arm is AppArm.LEARNED
    assert run.config.arm is not AppArm.FIXED_RULE
    await manager.close()


async def test_promoted_ecology_checkpoint_is_restored_and_projected() -> None:
    bootstrap = AntSession(
        AntWorld(world_objects=(ButterSource(object_id="butter", x=2.0, y=0.0),)),
        config=AntSessionConfig(
            temporal_latent_dim=4,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    checkpoint_archive = bootstrap.export_learning_checkpoint_archive(
        checkpoint_id="ecology-promoted"
    )
    checkpoint_info = decode_agent_learning_archive(checkpoint_archive).info
    curriculum_config = EcologyCurriculumConfig(
        n_ants=1,
        temporal_latent_dim=4,
        stage_rounds=1,
        stage_episodes=1,
        mastery_min_episodes=1,
        heldout_rounds=1,
        heldout_seeds=(1,),
    )
    loaded = LoadedEcologyCheckpoint(
        checkpoint_archives=(checkpoint_archive,),
        fingerprint=checkpoint_info.state_fingerprint,
        verdict="PASS",
        config=curriculum_config,
        report_path="local-test",
        p2_promotion=_p2_promotion(),
    )
    manager = AntAppManager(ecology_checkpoint=loaded)
    run = await manager.create_run(
        AppExperimentConfig(
            objective=AppObjective.ECOLOGY,
            temporal_latent_dim=4,
            autostart=False,
            max_ticks=2,
        ),
        run_id="loaded-ecology",
    )
    await run.apply_command(AppCommand(command_id="loaded-step", kind=AppCommandKind.STEP))
    await _wait_for_tick(run, 1)
    frame_event = next(event for event in run.events_after(0) if event.kind.value == "frame")
    frame = json.loads(frame_event.payload_json)
    assert frame["evidence"]["checkpoint_loaded"] is True
    assert frame["evidence"]["checkpoint_verdict"] == "PASS"
    assert (
        frame["evidence"]["checkpoint_fingerprint"]
        == checkpoint_info.state_fingerprint
    )
    assert "runtime_replay_pending_captures" in frame["evidence"]
    await manager.close()


async def test_a_checkpoint_without_p2_evidence_is_refused_at_admission() -> None:
    """The defence-in-depth check may not be optional on its own evidence.

    ``p2_promotion`` is optional on ``LoadedEcologyCheckpoint`` only so a
    fixture can be built in-process.  Skipping the verdict check when it is
    absent made the whole defence conditional on the one field an in-process
    caller can simply omit -- and this test file used to construct exactly that
    shape twice and be admitted.
    """

    bootstrap = AntSession(
        AntWorld(world_objects=(ButterSource(object_id="butter", x=2.0, y=0.0),)),
        config=AntSessionConfig(
            temporal_latent_dim=4,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    archive = bootstrap.export_learning_checkpoint_archive(
        checkpoint_id="ecology-unconfirmed"
    )
    info = decode_agent_learning_archive(archive).info
    curriculum_config = EcologyCurriculumConfig(
        n_ants=1,
        temporal_latent_dim=4,
        stage_rounds=1,
        stage_episodes=1,
        mastery_min_episodes=1,
        heldout_rounds=1,
        heldout_seeds=(1,),
    )

    run_config = AppExperimentConfig(
        objective=AppObjective.ECOLOGY,
        arm=AppArm.LEARNED,
        temporal_latent_dim=4,
        autostart=False,
        max_ticks=1,
    )

    unconfirmed = AntAppManager(
        ecology_checkpoint=LoadedEcologyCheckpoint(
            checkpoint_archives=(archive,),
            fingerprint=info.state_fingerprint,
            verdict="PASS",
            config=curriculum_config,
            report_path="no-p2-evidence",
        )
    )
    with pytest.raises(ValueError, match="carries no P2 confirmatory evidence"):
        await unconfirmed.create_run(run_config, run_id="no-p2")
    await unconfirmed.close()

    blocked = AntAppManager(
        ecology_checkpoint=LoadedEcologyCheckpoint(
            checkpoint_archives=(archive,),
            fingerprint=info.state_fingerprint,
            verdict="PASS",
            config=curriculum_config,
            report_path="blocked-p2",
            p2_promotion=_p2_promotion(verdict="BLOCK"),
        )
    )
    with pytest.raises(ValueError, match="P2 confirmatory verdict is BLOCK"):
        await blocked.create_run(run_config, run_id="blocked-p2")
    await blocked.close()
