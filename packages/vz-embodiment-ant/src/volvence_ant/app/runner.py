"""Deterministic realtime orchestration for the digital-ant app."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import math
from time import perf_counter
from typing import Iterable
from uuid import uuid4

from volvence_ant.app.contracts import (
    APP_SCHEMA_VERSION,
    AppAntFrame,
    AppArm,
    AppCommand,
    AppCommandKind,
    AppDisturbance,
    AppDisturbanceKind,
    AppDisturbanceRecord,
    AppEventKind,
    AppEvidenceProjection,
    AppExperimentConfig,
    AppFrame,
    AppMode,
    AppObjective,
    AppRunState,
    AppRunStatus,
    AppStreamEvent,
)
from volvence_ant.controllers.fixed_rule_ant import (
    FixedRuleAnt,
    FixedRuleConfig,
    FixedRuleStep,
)
from volvence_ant.env.ant_world import (
    AntWorld,
    AntWorldConfig,
    FoodSource,
    MotorDistortionProfile,
)
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.world_objects import (
    BurningMatch,
    ButterSource,
    WoodStick,
    WorldObject,
    WorldObjectKind,
)
from volvence_ant.evidence.ecology_checkpoint import (
    LoadedEcologyCheckpoint,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
    AntStepRecord,
    KernelColonyRunner,
)


_LIVE_FRAME_CAPACITY = 64


class AntAppRun:
    """One authoritative simulation run and its immutable event stream."""

    def __init__(
        self,
        *,
        run_id: str,
        config: AppExperimentConfig,
        formal_verdict: str = "BLOCK",
        formal_verdict_reason: str = "formal evidence has not passed",
        ecology_checkpoint: LoadedEcologyCheckpoint | None = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        if formal_verdict not in {"PASS", "BLOCK"}:
            raise ValueError("formal_verdict must be PASS or BLOCK")
        self._formal_verdict = formal_verdict
        self._formal_verdict_reason = formal_verdict_reason
        self._ecology_checkpoint = ecology_checkpoint
        self._checkpoint_loaded = False
        self._checkpoint_fingerprint = ""
        self._checkpoint_verdict = "UNAVAILABLE"
        self._state = AppRunState.RUNNING if config.autostart else AppRunState.PAUSED
        self._tick_interval_ms = config.tick_interval_ms
        self._sequence = 0
        self._step_budget = 0
        self._last_error = ""
        self._pending_disturbances: list[AppDisturbance] = []
        self._live_frame_events: deque[AppStreamEvent] = deque(maxlen=_LIVE_FRAME_CAPACITY)
        self._audit_events: list[AppStreamEvent] = []
        self._replay_frames: list[AppFrame] = []
        self._frames_dropped = 0
        self._condition = asyncio.Condition()
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

        world_config = AntWorldConfig(
            seed=config.seed,
            motor_distortions=(
                MotorDistortionProfile(
                    turn_gain=config.motor_turn_gain,
                    turn_bias=config.motor_turn_bias,
                    switch_tick=config.motor_switch_tick,
                    switched_turn_gain=config.motor_switched_turn_gain,
                    switched_turn_bias=config.motor_switched_turn_bias,
                ),
            ),
        )
        ecology_objects: tuple[WorldObject, ...] = ()
        food: tuple[FoodSource, ...] = ()
        if config.objective is AppObjective.ECOLOGY:
            ecology_objects = (
                ButterSource(
                    object_id="butter-initial",
                    x=config.food_x,
                    y=config.food_y,
                    strength=1.6 if config.mode is AppMode.COLONY else 1.0,
                    decay=5.0,
                    radius=1.6,
                ),
            )
        else:
            food = (
                FoodSource(
                    x=config.food_x,
                    y=config.food_y,
                    strength=1.6 if config.mode is AppMode.COLONY else 1.0,
                    decay=5.0,
                    radius=1.6,
                ),
            )
        if config.mode is AppMode.COLONY:
            self.world: AntWorld = ColonyWorld(
                config=world_config,
                food_sources=food,
                world_objects=ecology_objects,
                n_bodies=config.n_ants,
            )
        else:
            self.world = AntWorld(
                config=world_config,
                food_sources=food,
                world_objects=ecology_objects,
            )

        self._initial_headings = tuple(self.world.body(body_id).heading for body_id in range(self.world.n_bodies))
        self._kernel_single: AntSession | None = None
        self._kernel_colony: KernelColonyRunner | None = None
        self._fixed_ants: tuple[FixedRuleAnt, ...] = ()
        self._configure_controller()
        self._restore_ecology_checkpoint()

    def _configure_controller(self) -> None:
        if self.config.arm is AppArm.FIXED_RULE:
            self._fixed_ants = tuple(
                FixedRuleAnt(
                    self.world,
                    config=FixedRuleConfig(seed=self.config.seed + body_id),
                    body_id=body_id,
                )
                for body_id in range(self.world.n_bodies)
            )
            return

        if self.config.objective is AppObjective.HEADING_STABILITY:
            objective = AntObjectiveKind.HEADING_STABILITY
        elif self.config.objective is AppObjective.ECOLOGY:
            objective = AntObjectiveKind.ECOLOGY
        else:
            objective = AntObjectiveKind.FORAGING
        session_config = AntSessionConfig(
            temporal_latent_dim=self.config.temporal_latent_dim,
            session_id=f"digital-ant-app:{self.run_id}",
            seed=self.config.seed,
            rollout_config=ant_runtime_replay_rollout_config(
                enable_sparse_exploration=(self.config.objective in {AppObjective.FORAGING, AppObjective.ECOLOGY})
            ),
            joint_apply_writeback=True,
            joint_apply_policy_optimization=(self.config.arm is AppArm.LEARNED),
            objective=objective,
            sense_schema=(AntSenseSchema.ECOLOGY_V2 if objective is AntObjectiveKind.ECOLOGY else AntSenseSchema.V1),
        )
        if self.config.mode is AppMode.COLONY:
            self._kernel_colony = KernelColonyRunner(self.world, base_config=session_config)
        else:
            self._kernel_single = AntSession(self.world, config=session_config)

    def _restore_ecology_checkpoint(self) -> None:
        checkpoint = self._ecology_checkpoint
        if (
            checkpoint is None
            or self.config.objective is not AppObjective.ECOLOGY
            or self.config.arm is not AppArm.LEARNED
        ):
            return
        if self.config.n_ants != checkpoint.config.n_ants:
            raise ValueError(
                "ecology checkpoint ant count mismatch: "
                f"run={self.config.n_ants}, checkpoint={checkpoint.config.n_ants}"
            )
        if self.config.temporal_latent_dim != checkpoint.config.temporal_latent_dim:
            raise ValueError(
                "ecology checkpoint latent dim mismatch: "
                f"run={self.config.temporal_latent_dim}, "
                f"checkpoint={checkpoint.config.temporal_latent_dim}"
            )
        if self._kernel_colony is not None:
            self._kernel_colony.restore_learning_checkpoint_archives(
                checkpoint.checkpoint_archives
            )
        elif self._kernel_single is not None:
            if len(checkpoint.checkpoint_archives) != 1:
                raise ValueError("solo ecology run requires a one-ant checkpoint")
            self._kernel_single.restore_learning_checkpoint_archive(
                checkpoint.checkpoint_archives[0]
            )
        else:
            raise RuntimeError("learned ecology checkpoint requires a kernel controller")
        self._checkpoint_loaded = True
        self._checkpoint_fingerprint = checkpoint.fingerprint
        self._checkpoint_verdict = checkpoint.verdict

    @property
    def state(self) -> AppRunState:
        return self._state

    @property
    def terminal(self) -> bool:
        return self._state in {
            AppRunState.COMPLETED,
            AppRunState.FAILED,
            AppRunState.STOPPED,
        }

    @property
    def latest_sequence(self) -> int:
        return self._sequence

    def status(self) -> AppRunStatus:
        return AppRunStatus(
            schema_version=APP_SCHEMA_VERSION,
            run_id=self.run_id,
            state=self._state,
            tick=self.world.tick,
            sequence=self._sequence,
            mode=self.config.mode,
            arm=self.config.arm,
            objective=self.config.objective,
            seed=self.config.seed,
            n_ants=self.world.n_bodies,
            tick_interval_ms=self._tick_interval_ms,
            pending_disturbances=len(self._pending_disturbances),
            frames_retained=len(self._live_frame_events),
            frames_dropped=self._frames_dropped,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError(f"run {self.run_id} already started")
        await self._publish_status()
        self._task = asyncio.create_task(self._run_loop(), name=f"digital-ant-app:{self.run_id}")
        self._wake.set()

    async def close(self) -> None:
        if not self.terminal:
            self._state = AppRunState.STOPPED
            await self._publish_status()
        self._wake.set()
        if self._task is not None:
            await self._task

    async def apply_command(self, command: AppCommand) -> AppRunStatus:
        async with self._lock:
            if self.terminal:
                raise RuntimeError(f"run {self.run_id} is terminal ({self._state.value})")
            if command.kind is AppCommandKind.PAUSE:
                self._state = AppRunState.PAUSED
            elif command.kind is AppCommandKind.RESUME:
                self._state = AppRunState.RUNNING
            elif command.kind is AppCommandKind.STEP:
                self._state = AppRunState.PAUSED
                self._step_budget += 1
            elif command.kind is AppCommandKind.SET_SPEED:
                if command.value is None:
                    raise ValueError("set_speed value missing after validation")
                self._tick_interval_ms = int(round(command.value))
            elif command.kind is AppCommandKind.STOP:
                self._state = AppRunState.STOPPED
            else:
                raise ValueError(f"unsupported command {command.kind.value}")
        await self._publish_status()
        self._wake.set()
        return self.status()

    async def queue_disturbance(self, disturbance: AppDisturbance) -> AppDisturbanceRecord:
        async with self._lock:
            if self.terminal:
                raise RuntimeError(f"run {self.run_id} is terminal ({self._state.value})")
            self._pending_disturbances.append(disturbance)
        record = AppDisturbanceRecord(
            disturbance=disturbance,
            status="queued",
            applied_tick=None,
            detail="queued for a deterministic tick/round boundary",
        )
        await self._publish_audit(AppEventKind.DISTURBANCE, record)
        self._wake.set()
        return record

    async def _run_loop(self) -> None:
        try:
            while not self.terminal:
                should_step = False
                async with self._lock:
                    if self._state is AppRunState.RUNNING:
                        should_step = True
                    elif self._state is AppRunState.PAUSED and self._step_budget > 0:
                        self._step_budget -= 1
                        should_step = True
                if not should_step:
                    self._wake.clear()
                    await self._wake.wait()
                    continue

                await self._apply_due_disturbances()
                await self._advance()
                if self.config.max_ticks is not None and self.world.tick >= self.config.max_ticks:
                    self._state = AppRunState.COMPLETED
                    await self._publish_status()
                    break
                if self._state is AppRunState.RUNNING:
                    await asyncio.sleep(self._tick_interval_ms / 1000.0)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._state = AppRunState.FAILED
            await self._publish_audit(
                AppEventKind.ERROR,
                {"run_id": self.run_id, "error": self._last_error},
            )
            await self._publish_status()
            raise

    async def _apply_due_disturbances(self) -> None:
        async with self._lock:
            current_tick = self.world.tick
            due = [
                event
                for event in self._pending_disturbances
                if event.requested_tick is None or event.requested_tick <= current_tick
            ]
            self._pending_disturbances = [event for event in self._pending_disturbances if event not in due]
        for event in due:
            try:
                detail = self._apply_disturbance(event)
                record = AppDisturbanceRecord(
                    disturbance=event,
                    status="applied",
                    applied_tick=current_tick,
                    detail=detail,
                )
            except (IndexError, KeyError, ValueError) as exc:
                record = AppDisturbanceRecord(
                    disturbance=event,
                    status="rejected",
                    applied_tick=current_tick,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            await self._publish_audit(AppEventKind.DISTURBANCE, record)

    def _apply_disturbance(self, event: AppDisturbance) -> str:
        if event.kind is AppDisturbanceKind.RELOCATE_FOOD:
            if event.x is None or event.y is None:
                raise ValueError("relocate_food values missing after validation")
            self.world.move_food(
                index=event.food_index,
                x=event.x,
                y=event.y,
            )
            return f"food[{event.food_index}] moved to ({event.x}, {event.y})"
        if event.kind is AppDisturbanceKind.TRIGGER_ALARM:
            if event.magnitude is None:
                raise ValueError("alarm magnitude missing after validation")
            self.world.trigger_alarm(body_id=event.body_id, magnitude=event.magnitude)
            return f"alarm magnitude {event.magnitude} applied"
        if event.kind is AppDisturbanceKind.MOTOR_DISTORTION:
            if event.turn_gain is None or event.turn_bias is None:
                raise ValueError("motor values missing after validation")
            self.world.set_motor_distortion(
                MotorDistortionProfile(
                    turn_gain=event.turn_gain,
                    turn_bias=event.turn_bias,
                ),
                body_id=event.body_id,
            )
            return "hidden actuator transfer replaced at boundary"
        if event.kind is AppDisturbanceKind.UPSERT_WORLD_OBJECT:
            world_object = self._world_object_from_disturbance(event)
            self.world.upsert_world_object(world_object)
            return f"{world_object.object_id} ({event.object_kind.value}) upserted at boundary"
        if event.kind is AppDisturbanceKind.MOVE_WORLD_OBJECT:
            if event.object_id is None or event.delta_x is None or event.delta_y is None:
                raise ValueError("move_world_object values missing after validation")
            self.world.move_world_object(
                event.object_id,
                delta_x=event.delta_x,
                delta_y=event.delta_y,
            )
            return f"{event.object_id} translated at boundary"
        if event.kind is AppDisturbanceKind.REMOVE_WORLD_OBJECT:
            if event.object_id is None:
                raise ValueError("remove_world_object id missing after validation")
            self.world.remove_world_object(event.object_id)
            return f"{event.object_id} removed at boundary"
        raise ValueError(f"unsupported disturbance {event.kind.value}")

    @staticmethod
    def _world_object_from_disturbance(
        event: AppDisturbance,
    ) -> WorldObject:
        if event.object_id is None or event.object_kind is None:
            raise ValueError("world object identity missing after validation")
        if event.object_kind is WorldObjectKind.BUTTER:
            if event.x is None or event.y is None:
                raise ValueError("butter position missing after validation")
            return ButterSource(
                object_id=event.object_id,
                x=event.x,
                y=event.y,
                strength=event.strength if event.strength is not None else 1.6,
                decay=event.decay if event.decay is not None else 4.0,
                radius=event.radius if event.radius is not None else 1.2,
                remaining=(event.remaining if event.remaining is not None else float("inf")),
            )
        if event.object_kind is WorldObjectKind.WOOD_STICK:
            coordinates = (
                event.start_x,
                event.start_y,
                event.end_x,
                event.end_y,
            )
            if any(value is None for value in coordinates):
                raise ValueError("wood stick geometry missing after validation")
            start_x, start_y, end_x, end_y = (float(value) for value in coordinates if value is not None)
            return WoodStick(
                object_id=event.object_id,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                radius=event.radius if event.radius is not None else 0.22,
            )
        if event.object_kind is WorldObjectKind.BURNING_MATCH:
            if event.x is None or event.y is None:
                raise ValueError("burning match position missing after validation")
            return BurningMatch(
                object_id=event.object_id,
                x=event.x,
                y=event.y,
                angle=event.angle if event.angle is not None else 0.0,
                length=event.length if event.length is not None else 1.8,
                heat_strength=(event.strength if event.strength is not None else 1.0),
                heat_decay=event.decay if event.decay is not None else 1.8,
                harm_threshold=(event.harm_threshold if event.harm_threshold is not None else 0.55),
            )
        raise ValueError(f"unsupported world object kind {event.object_kind.value}")

    async def _advance(self) -> None:
        started_at = perf_counter()
        if self._kernel_single is not None:
            record = await self._kernel_single.step()
            ants = (self._kernel_ant_frame(0, record),)
            evidence = self._evidence_from_records((record,))
        elif self._kernel_colony is not None:
            round_record = await self._kernel_colony.step_round()
            ants = tuple(
                self._kernel_ant_frame(body_id, record) for body_id, record in enumerate(round_record.ant_steps)
            )
            evidence = self._evidence_from_records(round_record.ant_steps)
        else:
            fixed_records = tuple(ant.step() for ant in self._fixed_ants)
            ants = tuple(self._fixed_ant_frame(body_id, record) for body_id, record in enumerate(fixed_records))
            evidence = AppEvidenceProjection(
                verdict="BLOCK",
                verdict_reason="fixed-rule is a baseline, not learned evidence",
            )
        await self._publish_frame(
            ants=ants,
            evidence=evidence,
            tick_latency_ms=(perf_counter() - started_at) * 1000.0,
        )

    def _kernel_ant_frame(self, body_id: int, record: AntStepRecord) -> AppAntFrame:
        return AppAntFrame(
            body_id=body_id,
            x=record.x,
            y=record.y,
            heading=record.true_heading,
            target_heading=self._initial_headings[body_id],
            carrying_food=record.carrying_food,
            action=record.abstract_action,
            turn_command=record.command.turn_command,
            applied_turn=record.applied_turn,
            step_command=record.command.step_command,
            code=record.code,
            switch_gate=record.switch_gate,
            pe_magnitude=record.pe_magnitude,
            cumulative_credit=record.cumulative_credit,
            heading_stability_error=record.heading_stability_error,
            motor_execution_error=record.motor_execution_error,
            heat_center=record.heat_center,
            heat_harmful=record.heat_harmful,
        )

    def _fixed_ant_frame(self, body_id: int, record: FixedRuleStep) -> AppAntFrame:
        body = self.world.body(body_id)
        transition = self.world.last_transition(body_id)
        target = self._initial_headings[body_id]
        heading_error = abs((body.heading - target + math.pi) % (2.0 * math.pi) - math.pi) / math.pi
        return AppAntFrame(
            body_id=body_id,
            x=body.x,
            y=body.y,
            heading=body.heading,
            target_heading=target,
            carrying_food=body.carrying_food,
            action=record.mode,
            turn_command=record.turn_command,
            applied_turn=transition.applied_turn,
            step_command=record.step_command,
            heading_stability_error=heading_error,
            motor_execution_error=abs(transition.applied_turn - transition.commanded_turn),
            heat_center=self.world.observe(body_id).heat_center,
            heat_harmful=self.world.observe(body_id).heat_harmful,
        )

    def _evidence_from_records(
        self,
        records: Iterable[AntStepRecord],
    ) -> AppEvidenceProjection:
        materialized = tuple(records)
        if not materialized:
            return AppEvidenceProjection()
        latest = materialized[-1]
        return AppEvidenceProjection(
            backend_wiring=latest.backend_wiring,
            runtime_replay_captured=sum(record.runtime_replay_captured for record in materialized),
            runtime_replay_settled=sum(record.runtime_replay_settled for record in materialized),
            runtime_replay_transitions=sum(record.runtime_replay_transitions for record in materialized),
            runtime_replay_lineage_matches=sum(record.runtime_replay_lineage_matches for record in materialized),
            runtime_replay_drop_reasons=tuple(
                reason for record in materialized for reason in record.runtime_replay_drop_reasons
            ),
            runtime_replay_pending_captures=sum(record.runtime_replay_pending_captures for record in materialized),
            runtime_replay_staged_rollouts=sum(record.runtime_replay_staged_rollouts for record in materialized),
            verdict=self._formal_verdict,
            verdict_reason=self._formal_verdict_reason,
            checkpoint_loaded=self._checkpoint_loaded,
            checkpoint_fingerprint=self._checkpoint_fingerprint,
            checkpoint_verdict=self._checkpoint_verdict,
        )

    async def _publish_frame(
        self,
        *,
        ants: tuple[AppAntFrame, ...],
        evidence: AppEvidenceProjection,
        tick_latency_ms: float,
    ) -> None:
        sequence = self._next_sequence()
        trail: tuple[tuple[float, ...], ...] = ()
        if isinstance(self.world, ColonyWorld):
            trail = tuple(tuple(float(value) for value in row) for row in self.world.pheromone.trail)
        frame = AppFrame(
            schema_version=APP_SCHEMA_VERSION,
            run_id=self.run_id,
            sequence=sequence,
            tick=self.world.tick,
            tick_latency_ms=tick_latency_ms,
            mode=self.config.mode,
            arm=self.config.arm,
            objective=self.config.objective,
            nest=self.world.nest,
            food=tuple((source.x, source.y) for source in self.world.food_sources() if source.remaining > 0.0),
            delivered=self.world.food_delivered,
            pickups=self.world.food_pickups,
            ants=ants,
            trail=trail,
            evidence=evidence,
            objects=self.world.world_object_snapshots(),
        )
        self._replay_frames.append(frame)
        event = self._stream_event(sequence, AppEventKind.FRAME, frame)
        if len(self._live_frame_events) == self._live_frame_events.maxlen:
            self._frames_dropped += 1
        self._live_frame_events.append(event)
        await self._notify()

    async def _publish_status(self) -> None:
        sequence = self._next_sequence()
        self._audit_events.append(self._stream_event(sequence, AppEventKind.STATUS, self.status()))
        await self._notify()

    async def _publish_audit(self, kind: AppEventKind, payload: object) -> None:
        sequence = self._next_sequence()
        if isinstance(payload, dict):
            payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            event = AppStreamEvent(sequence=sequence, kind=kind, payload_json=payload_json)
        else:
            event = self._stream_event(sequence, kind, payload)
        self._audit_events.append(event)
        await self._notify()

    @staticmethod
    def _stream_event(sequence: int, kind: AppEventKind, payload: object) -> AppStreamEvent:
        from dataclasses import asdict

        return AppStreamEvent(
            sequence=sequence,
            kind=kind,
            payload_json=json.dumps(asdict(payload), separators=(",", ":"), sort_keys=True),
        )

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def events_after(self, sequence: int) -> tuple[AppStreamEvent, ...]:
        events = [event for event in self._audit_events if event.sequence > sequence]
        events.extend(event for event in self._live_frame_events if event.sequence > sequence)
        return tuple(sorted(events, key=lambda event: event.sequence))

    async def wait_for_events(self, sequence: int, *, timeout: float = 15.0) -> tuple[AppStreamEvent, ...]:
        available = self.events_after(sequence)
        if available:
            return available
        async with self._condition:
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
            except TimeoutError:
                return ()
        return self.events_after(sequence)

    def replay_payload(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            "schema_version": APP_SCHEMA_VERSION,
            "run_id": self.run_id,
            "config": asdict(self.config),
            "status": asdict(self.status()),
            "frames": tuple(asdict(frame) for frame in self._replay_frames),
            "audit_events": tuple(
                {
                    "sequence": event.sequence,
                    "kind": event.kind.value,
                    "payload": json.loads(event.payload_json),
                }
                for event in self._audit_events
            ),
        }


class AntAppManager:
    """Owns app-run lifecycles; each run owns exactly one world."""

    def __init__(
        self,
        *,
        formal_verdict: str = "BLOCK",
        formal_verdict_reason: str = "formal evidence has not passed",
        ecology_checkpoint: LoadedEcologyCheckpoint | None = None,
    ) -> None:
        self._runs: dict[str, AntAppRun] = {}
        self._formal_verdict = formal_verdict
        self._formal_verdict_reason = formal_verdict_reason
        self._ecology_checkpoint = ecology_checkpoint

    @classmethod
    def from_evidence_artifact(
        cls,
        path: str,
        *,
        ecology_checkpoint: LoadedEcologyCheckpoint | None = None,
    ) -> "AntAppManager":
        from pathlib import Path

        artifact_path = Path(path)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("formal evidence artifact must be a JSON object")
        raw_verdict = payload.get("overall_verdict", payload.get("verdict"))
        if raw_verdict is None and "eligible" in payload:
            raw_verdict = "PASS" if payload["eligible"] is True else "BLOCK"
        verdict = str(raw_verdict or "BLOCK").upper()
        if verdict not in {"PASS", "BLOCK"}:
            verdict = "BLOCK"
        return cls(
            formal_verdict=verdict,
            formal_verdict_reason=(f"read-only formal verdict from {artifact_path.name}"),
            ecology_checkpoint=ecology_checkpoint,
        )

    async def create_run(self, config: AppExperimentConfig, *, run_id: str | None = None) -> AntAppRun:
        resolved_id = run_id or uuid4().hex
        if resolved_id in self._runs:
            raise ValueError(f"run_id already exists: {resolved_id}")
        run = AntAppRun(
            run_id=resolved_id,
            config=config,
            formal_verdict=self._formal_verdict,
            formal_verdict_reason=self._formal_verdict_reason,
            ecology_checkpoint=self._ecology_checkpoint,
        )
        self._runs[resolved_id] = run
        await run.start()
        return run

    def get_run(self, run_id: str) -> AntAppRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown digital-ant run: {run_id}") from exc

    async def close(self) -> None:
        for run in tuple(self._runs.values()):
            await run.close()


# Public plan terminology; AntAppRun remains a compatibility spelling for the
# concrete per-run runner.
AntAppRunner = AntAppRun


__all__ = ["AntAppManager", "AntAppRun", "AntAppRunner"]
