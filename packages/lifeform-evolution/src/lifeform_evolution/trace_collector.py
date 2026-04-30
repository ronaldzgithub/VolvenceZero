"""SSL training-data collector — the seed for "往下生长" (downward growth).

Drives a ``Lifeform`` through one or more scripted scenarios and captures
the kernel state at every turn into a structured trace suitable for
offline ``vz-temporal`` SSL training (R3, R4, R13 — SSL phase).

What we capture per turn:

* substrate residual feature surface (when published)
* temporal_abstraction snapshot — z_t, beta_t, switch markers
* regime_id, expression_intent
* prediction_error magnitude + 4-dim breakdown
* dual_track pressures, evaluation scores
* memory lifecycle counters
* lifeform-side scene_id and tick_index

What we do **not** do:

* train a model here (that is the SSL trainer's job in vz-temporal)
* mutate kernel owners in any way (this collector is read-only)
* leak any user-identifying data — the trace stores text inputs the
  scenarios were already given, plus typed kernel state

The on-disk format is line-delimited JSON (``.ndjson``). One line per turn.
Schema is intentionally flat so downstream training scripts do not need to
import any vz-* type to consume traces.

Usage::

    from lifeform_evolution import (
        TraceCollector,
        all_built_in_scenarios,
    )
    collector = TraceCollector(output_path="data/traces.ndjson")
    for scenario in all_built_in_scenarios():
        collector.collect_scenario(scenario)
    collector.close()
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from lifeform_core import Lifeform, LifeformConfig
from lifeform_domain_emogpt import build_companion_package
from lifeform_evolution.benchmark import ScriptedScenario


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceTurnRecord:
    """One JSON-serialisable training row.

    Every field is a primitive or a tuple of primitives so the row survives
    ``json.dumps`` without custom encoders.
    """

    schema_version: str
    scenario_id: str
    session_id: str
    turn_index: int
    scene_id: str
    tick_index: int
    user_input: str
    response_text: str
    active_regime: str | None
    active_abstract_action: str | None
    expression_intent: str | None
    pe_magnitude: float
    pe_task: float
    pe_relationship: float
    pe_regime: float
    pe_action: float
    dual_track_world_pressure: float
    dual_track_self_pressure: float
    eval_alert_count: int
    memory_entry_count: int
    open_loop_count: int
    commitment_count: int
    has_substrate_residuals: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        # The dataclass-as-dict route would be cleaner but ``dict`` field
        # would be deepcopied. Instead we emit a flat dict explicitly.
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "scenario_id": self.scenario_id,
                "session_id": self.session_id,
                "turn_index": self.turn_index,
                "scene_id": self.scene_id,
                "tick_index": self.tick_index,
                "user_input": self.user_input,
                "response_text": self.response_text,
                "active_regime": self.active_regime,
                "active_abstract_action": self.active_abstract_action,
                "expression_intent": self.expression_intent,
                "pe_magnitude": self.pe_magnitude,
                "pe_task": self.pe_task,
                "pe_relationship": self.pe_relationship,
                "pe_regime": self.pe_regime,
                "pe_action": self.pe_action,
                "dual_track_world_pressure": self.dual_track_world_pressure,
                "dual_track_self_pressure": self.dual_track_self_pressure,
                "eval_alert_count": self.eval_alert_count,
                "memory_entry_count": self.memory_entry_count,
                "open_loop_count": self.open_loop_count,
                "commitment_count": self.commitment_count,
                "has_substrate_residuals": self.has_substrate_residuals,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class TraceScenarioReport:
    scenario_id: str
    record_count: int
    distinct_regimes: tuple[str, ...]
    distinct_intents: tuple[str, ...]
    pe_max: float
    pe_mean: float


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


_SCHEMA_VERSION = "trace.v1"


class TraceCollector:
    """Run scenarios and emit one training row per turn.

    The collector owns its own ``Lifeform`` so traces produced under the
    same collector instance share substrate / domain pack — that keeps the
    trace file internally self-consistent for downstream SSL trainers.

    Lifeforms are NOT shared with running production sessions. Each
    ``collect_scenario`` call creates a fresh ``LifeformSession``.
    """

    def __init__(
        self,
        *,
        output_path: str | pathlib.Path | None = None,
        config: LifeformConfig | None = None,
        temporal_bootstrap: object | None = None,
        substrate_runtime: object | None = None,
    ) -> None:
        """Args:
            output_path: optional ``.ndjson`` file to stream rows to.
            config: optional ``LifeformConfig``; defaults force rare-heavy off
                so traces are deterministic.
            temporal_bootstrap: a ``MetacontrollerParameterSnapshot`` to seed
                the underlying lifeform with a previously-trained policy.
                When supplied, each new session starts from this trained
                policy, so traces collected reflect that policy's behaviour
                rather than a fresh one. This is the seam that lets the
                multi-round learning loop have policy and traces co-evolve.
            substrate_runtime: optional pre-built ``OpenWeightResidualRuntime``
                (synthetic OR real HF transformer). When supplied the
                underlying lifeform is built with ``substrate_mode="injected"``
                so traces collected reflect THAT substrate's residual
                activations / semantic pulls rather than a freshly-built
                synthetic default. This is the seam slice 2a phase 1.5
                uses to recalibrate the companion regime classifier on
                real-LLM signals (Gap 9 follow-up). When ``None`` (default)
                the lifeform builds its own synthetic runtime, matching
                pre-phase-1.5 behaviour.
        """
        base_config = config or LifeformConfig()
        # Force rare-heavy off for deterministic traces — the SSL pipeline
        # doesn't want online artifact import noise mixed into the dataset.
        from dataclasses import replace as _replace
        base_config = _replace(
            base_config,
            brain_config=_replace(base_config.brain_config, rare_heavy_enabled=False),
        )
        # When an injected substrate is supplied, force ``substrate_mode``
        # to ``injected`` so the brain consumes the supplied runtime
        # instead of building a fresh one per session. Matches the
        # discipline in ``build_companion_lifeform``.
        if substrate_runtime is not None:
            base_config = _replace(
                base_config,
                brain_config=_replace(
                    base_config.brain_config, substrate_mode="injected"
                ),
            )
        base_config = base_config.with_domain_experience((build_companion_package(),))

        self._lifeform = Lifeform(
            base_config,
            temporal_bootstrap=temporal_bootstrap,
            substrate_runtime=substrate_runtime,
        )
        self._records: list[TraceTurnRecord] = []
        self._reports: list[TraceScenarioReport] = []
        self._output_path = pathlib.Path(output_path) if output_path is not None else None
        self._sink = None
        if self._output_path is not None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._sink = self._output_path.open("w", encoding="utf-8")

    @property
    def output_path(self) -> pathlib.Path | None:
        return self._output_path

    @property
    def records(self) -> tuple[TraceTurnRecord, ...]:
        return tuple(self._records)

    @property
    def reports(self) -> tuple[TraceScenarioReport, ...]:
        return tuple(self._reports)

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    # ------------------------------------------------------------------
    # Sync convenience
    # ------------------------------------------------------------------

    def collect_scenario(self, scenario: ScriptedScenario) -> TraceScenarioReport:
        return asyncio.run(self.collect_scenario_async(scenario))

    def collect_scenarios(self, scenarios: tuple[ScriptedScenario, ...]) -> tuple[TraceScenarioReport, ...]:
        return tuple(self.collect_scenario(scenario) for scenario in scenarios)

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def collect_scenario_async(self, scenario: ScriptedScenario) -> TraceScenarioReport:
        session = self._lifeform.create_session(
            session_id=f"trace::{scenario.scenario_id}",
        )
        scenario_records: list[TraceTurnRecord] = []
        for index, turn in enumerate(scenario.turns, start=1):
            await session.advance_tick(1, reason="trace-step")
            result = await session.run_turn(turn.user_input)
            record = self._build_record(
                scenario_id=scenario.scenario_id,
                session_id=session.session_id,
                turn_index=index,
                tick_index=session.tick_engine.tick_index,
                scene_id=(session.open_scene.scene_id if session.open_scene else "scene-?"),
                user_input=turn.user_input,
                result=result,
            )
            self._records.append(record)
            scenario_records.append(record)
            if self._sink is not None:
                self._sink.write(record.to_json_line() + "\n")

        await session.end_scene(reason="trace-end", drain_slow_loop=True)

        report = self._summarise(scenario.scenario_id, scenario_records)
        self._reports.append(report)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(
        scenario_id: str, records: list[TraceTurnRecord]
    ) -> TraceScenarioReport:
        regimes = sorted({r.active_regime for r in records if r.active_regime})
        intents = sorted({r.expression_intent for r in records if r.expression_intent})
        pe_values = [r.pe_magnitude for r in records]
        return TraceScenarioReport(
            scenario_id=scenario_id,
            record_count=len(records),
            distinct_regimes=tuple(regimes),
            distinct_intents=tuple(intents),
            pe_max=max(pe_values) if pe_values else 0.0,
            pe_mean=(sum(pe_values) / len(pe_values)) if pe_values else 0.0,
        )

    @staticmethod
    def _build_record(
        *,
        scenario_id: str,
        session_id: str,
        turn_index: int,
        tick_index: int,
        scene_id: str,
        user_input: str,
        result: Any,
    ) -> TraceTurnRecord:
        snapshots = result.active_snapshots

        pe_snap = snapshots.get("prediction_error")
        pe_magnitude = 0.0
        pe_task = pe_relationship = pe_regime = pe_action = 0.0
        if pe_snap is not None:
            error = getattr(pe_snap.value, "error", None)
            if error is not None:
                pe_magnitude = float(getattr(error, "magnitude", 0.0))
                pe_task = float(getattr(error, "task", 0.0))
                pe_relationship = float(getattr(error, "relationship", 0.0))
                pe_regime = float(getattr(error, "regime", 0.0))
                pe_action = float(getattr(error, "action", 0.0))

        dual_snap = snapshots.get("dual_track")
        world_pressure = self_pressure = 0.0
        if dual_snap is not None:
            world_pressure = float(getattr(dual_snap.value, "world_pressure", 0.0))
            self_pressure = float(getattr(dual_snap.value, "self_pressure", 0.0))

        eval_snap = snapshots.get("evaluation")
        alert_count = 0
        if eval_snap is not None:
            alerts = getattr(eval_snap.value, "alerts", ()) or ()
            alert_count = len(alerts)

        memory_snap = snapshots.get("memory")
        memory_entry_count = 0
        if memory_snap is not None:
            entries = getattr(memory_snap.value, "entries", ()) or ()
            memory_entry_count = len(entries)

        open_loop_snap = snapshots.get("open_loop")
        open_loop_count = 0
        if open_loop_snap is not None:
            open_loop_count = len(getattr(open_loop_snap.value, "unresolved_loops", ()) or ())

        commitment_snap = snapshots.get("commitment")
        commitment_count = 0
        if commitment_snap is not None:
            commitment_count = len(getattr(commitment_snap.value, "active_commitments", ()) or ())

        substrate_snap = snapshots.get("substrate")
        has_substrate_residuals = False
        if substrate_snap is not None:
            has_substrate_residuals = bool(
                getattr(substrate_snap.value, "residual_activations", None)
            )

        assembly_snap = snapshots.get("response_assembly")
        expression_intent: str | None = None
        if assembly_snap is not None:
            expression_intent = getattr(assembly_snap.value, "expression_intent", None)

        return TraceTurnRecord(
            schema_version=_SCHEMA_VERSION,
            scenario_id=scenario_id,
            session_id=session_id,
            turn_index=turn_index,
            scene_id=scene_id,
            tick_index=tick_index,
            user_input=user_input,
            response_text=result.response.text,
            active_regime=result.active_regime,
            active_abstract_action=result.active_abstract_action,
            expression_intent=expression_intent,
            pe_magnitude=pe_magnitude,
            pe_task=pe_task,
            pe_relationship=pe_relationship,
            pe_regime=pe_regime,
            pe_action=pe_action,
            dual_track_world_pressure=world_pressure,
            dual_track_self_pressure=self_pressure,
            eval_alert_count=alert_count,
            memory_entry_count=memory_entry_count,
            open_loop_count=open_loop_count,
            commitment_count=commitment_count,
            has_substrate_residuals=has_substrate_residuals,
        )
