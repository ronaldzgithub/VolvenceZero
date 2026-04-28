"""Lifeform / LifeformSession facade.

Composes:

* a Volvence Zero ``Brain`` for the cognitive kernel
* a ``TickEngine`` for the lifeform's own metabolic clock
* a ``SceneManager`` for scene lifecycle (drives kernel ``begin_new_context``)
* a ``FollowupManager`` for advisory follow-up scheduling

The Lifeform layer's invariants:

1. **One Brain in, one organism out.** Lifeform owns Brain, never the other
   way around (R8 — single ownership).
2. **Lifeform never auto-emits user turns.** Tick events can update internal
   state and surface follow-ups, but cannot fabricate ``run_turn`` calls.
   That would make the lifeform a second owner of conversation initiation.
3. **Scene closure → kernel boundary.** When a scene closes the lifeform
   calls ``runner.begin_new_context(reason='scene-end')`` so the kernel's
   session-post slow loop fires (R6).
4. **No prompt rendering here.** Prompt assembly is the job of
   ``lifeform-expression`` and is injected via ``response_synthesizer``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lifeform_core.followup_manager import FollowupManager
from lifeform_core.scene_manager import SceneManager
from lifeform_core.tick_engine import TickEngine, TickEngineConfig
from lifeform_core.types import FollowupItem, Scene, TickEvent, TurnSummary

from volvence_zero.agent.response import ResponseSynthesizer
from volvence_zero.application.domain_experience import DomainExperiencePackage
from volvence_zero.brain import Brain, BrainConfig, BrainSession
from volvence_zero.semantic_state import (
    ExternalSemanticEventBatch,
    SemanticProposalRuntime,
)
from volvence_zero.regime import RegimeBootstrap
from volvence_zero.substrate import OpenWeightResidualRuntime, SubstrateAdapter
from volvence_zero.temporal import MetacontrollerParameterSnapshot


@dataclass(frozen=True)
class LifeformConfig:
    """Configuration for the lifeform.

    Mirrors the most useful ``BrainConfig`` knobs but layered for the lifeform
    surface — products use ``LifeformConfig`` not ``BrainConfig`` so the API
    stays product-shaped.
    """

    brain_config: BrainConfig = field(default_factory=BrainConfig)
    tick: TickEngineConfig = field(default_factory=TickEngineConfig)
    idle_close_after_system_ticks: int | None = 60
    followup_default_due_delay_ticks: int = 90
    followup_max_pending: int = 32

    def with_domain_experience(
        self,
        packages: tuple[DomainExperiencePackage, ...],
    ) -> "LifeformConfig":
        from dataclasses import replace as _replace
        return _replace(
            self,
            brain_config=_replace(
                self.brain_config,
                domain_experience_packages=self.brain_config.domain_experience_packages + packages,
            ),
        )


class Lifeform:
    """Stable product-facing factory for lifeform sessions.

    Construct once per product process; reuse for many sessions. A pre-trained
    metacontroller can be injected via ``temporal_bootstrap`` so newly-created
    sessions start from learned \u03b2_t / z_t structure rather than a fresh
    random policy. This is the closure of the SSL feedback loop:
    ``lifeform-trace`` \u2192 ``lifeform-ssl`` \u2192 trained snapshot \u2192 inject here \u2192
    ``lifeform-bench`` shows behaviour difference.
    """

    def __init__(
        self,
        config: LifeformConfig | None = None,
        *,
        substrate_runtime: OpenWeightResidualRuntime | None = None,
        substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
        response_synthesizer: ResponseSynthesizer | None = None,
        semantic_proposal_runtime: SemanticProposalRuntime | None = None,
        temporal_bootstrap: MetacontrollerParameterSnapshot | None = None,
        regime_bootstrap: RegimeBootstrap | None = None,
    ) -> None:
        self._config = config or LifeformConfig()
        self._brain = Brain(
            self._config.brain_config,
            substrate_runtime=substrate_runtime,
            substrate_adapter_factory=substrate_adapter_factory,
            response_synthesizer=response_synthesizer,
            semantic_proposal_runtime=semantic_proposal_runtime,
            temporal_bootstrap=temporal_bootstrap,
            regime_bootstrap=regime_bootstrap,
        )
        self._init_kwargs = {
            "substrate_runtime": substrate_runtime,
            "substrate_adapter_factory": substrate_adapter_factory,
            "response_synthesizer": response_synthesizer,
            "semantic_proposal_runtime": semantic_proposal_runtime,
            "temporal_bootstrap": temporal_bootstrap,
            "regime_bootstrap": regime_bootstrap,
        }

    @property
    def config(self) -> LifeformConfig:
        return self._config

    @property
    def brain(self) -> Brain:
        return self._brain

    @property
    def temporal_bootstrap(self) -> MetacontrollerParameterSnapshot | None:
        return self._brain.temporal_bootstrap

    @property
    def regime_bootstrap(self) -> RegimeBootstrap | None:
        return self._brain.regime_bootstrap

    def with_domain_experience(
        self,
        packages: tuple[DomainExperiencePackage, ...],
    ) -> "Lifeform":
        return Lifeform(
            self._config.with_domain_experience(packages),
            **self._init_kwargs,
        )

    def with_temporal_bootstrap(
        self,
        snapshot: MetacontrollerParameterSnapshot | None,
    ) -> "Lifeform":
        """Return a clone of this lifeform with the given trained metacontroller."""
        new_kwargs = dict(self._init_kwargs)
        new_kwargs["temporal_bootstrap"] = snapshot
        return Lifeform(self._config, **new_kwargs)

    def with_regime_bootstrap(
        self,
        bootstrap: RegimeBootstrap | None,
    ) -> "Lifeform":
        """Return a clone of this lifeform with calibrated regime weights."""
        new_kwargs = dict(self._init_kwargs)
        new_kwargs["regime_bootstrap"] = bootstrap
        return Lifeform(self._config, **new_kwargs)

    def create_session(self, *, session_id: str = "lifeform-session") -> "LifeformSession":
        brain_session = self._brain.create_session(session_id=session_id)
        return LifeformSession(
            brain_session=brain_session,
            tick=TickEngine(self._config.tick),
            scene=SceneManager(idle_close_after_system_ticks=self._config.idle_close_after_system_ticks),
            followups=FollowupManager(
                default_due_delay_ticks=self._config.followup_default_due_delay_ticks,
                max_pending=self._config.followup_max_pending,
            ),
        )


class LifeformSession:
    """A live lifeform session.

    Wraps ``BrainSession`` and adds tick / scene / followup coordination.
    """

    def __init__(
        self,
        *,
        brain_session: BrainSession,
        tick: TickEngine,
        scene: SceneManager,
        followups: FollowupManager,
    ) -> None:
        self._brain_session = brain_session
        self._tick = tick
        self._scene = scene
        self._followups = followups
        self._turn_summaries: list[TurnSummary] = []
        # Re-publish snapshot fields after every turn so consumers can see
        # the cross-cutting lifeform state without poking at internals.
        self._latest_active_snapshots: dict[str, Any] = {}
        self._latest_response_text: str = ""

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._brain_session.session_id

    @property
    def brain_session(self) -> BrainSession:
        return self._brain_session

    @property
    def tick_engine(self) -> TickEngine:
        return self._tick

    @property
    def scene_manager(self) -> SceneManager:
        return self._scene

    @property
    def followup_manager(self) -> FollowupManager:
        return self._followups

    @property
    def open_scene(self) -> Scene | None:
        return self._scene.open_scene

    @property
    def closed_scenes(self) -> tuple[Scene, ...]:
        return self._scene.closed_scenes

    @property
    def turn_summaries(self) -> tuple[TurnSummary, ...]:
        return tuple(self._turn_summaries)

    @property
    def latest_active_snapshots(self) -> dict[str, Any]:
        return dict(self._latest_active_snapshots)

    @property
    def latest_response_text(self) -> str:
        return self._latest_response_text

    def due_followups(self) -> tuple[FollowupItem, ...]:
        return self._followups.due_now(current_tick=self._tick.tick_index)

    def all_pending_followups(self) -> tuple[FollowupItem, ...]:
        return self._followups.pending

    # ------------------------------------------------------------------
    # External event ingestion (delegated to BrainSession)
    # ------------------------------------------------------------------

    def submit_semantic_events(self, events: ExternalSemanticEventBatch) -> tuple[str, ...]:
        return self._brain_session.submit_semantic_events(events)

    def submit_tool_result(self, **kwargs: Any) -> tuple[str, ...]:
        return self._brain_session.submit_tool_result(**kwargs)

    def submit_profile_event(self, **kwargs: Any) -> tuple[str, ...]:
        return self._brain_session.submit_profile_event(**kwargs)

    def submit_task_event(self, **kwargs: Any) -> tuple[str, ...]:
        return self._brain_session.submit_task_event(**kwargs)

    def submit_reviewed_knowledge_event(self, **kwargs: Any) -> tuple[str, ...]:
        return self._brain_session.submit_reviewed_knowledge_event(**kwargs)

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    async def run_turn(self, user_input: str) -> Any:
        """Run one turn through the kernel.

        Returns the kernel's ``AgentTurnResult`` unchanged so callers can
        inspect every snapshot. Side effects:

        * Open a new scene if none exists.
        * Increment the open scene's turn counter.
        * Record a compact ``TurnSummary``.
        * Pull ``open_loop`` / ``commitment`` snapshots and feed the
          ``FollowupManager``.
        """
        # Open scene if needed; this is the only place a scene auto-opens.
        if self._scene.open_scene is None:
            self._scene.open_scene_now(current_tick=self._tick.tick_index)

        result = await self._brain_session.run_turn_async(user_input)

        scene = self._scene.record_turn(current_tick=self._tick.tick_index)
        self._latest_active_snapshots = dict(result.active_snapshots)
        self._latest_response_text = result.response.text

        open_loops_snapshot = result.active_snapshots.get("open_loop")
        commitment_snapshot = result.active_snapshots.get("commitment")

        unresolved_loops: tuple[Any, ...] = ()
        if open_loops_snapshot is not None:
            unresolved_loops = tuple(getattr(open_loops_snapshot.value, "unresolved_loops", ()) or ())
            if unresolved_loops:
                self._followups.ingest_open_loops(
                    unresolved_loops=unresolved_loops,
                    current_tick=self._tick.tick_index,
                )

        if commitment_snapshot is not None:
            at_risk = tuple(getattr(commitment_snapshot.value, "at_risk_commitment_refs", ()) or ())
            if at_risk:
                self._followups.ingest_at_risk_commitments(
                    at_risk_refs=at_risk,
                    current_tick=self._tick.tick_index,
                )

        pe_snapshot = result.active_snapshots.get("prediction_error")
        pe_magnitude = 0.0
        if pe_snapshot is not None:
            error = getattr(pe_snapshot.value, "error", None)
            if error is not None:
                pe_magnitude = float(getattr(error, "magnitude", 0.0))

        self._turn_summaries.append(
            TurnSummary(
                turn_index=len(self._turn_summaries) + 1,
                scene_id=scene.scene_id,
                user_input=user_input,
                response_text=result.response.text,
                active_regime=result.active_regime,
                active_abstract_action=result.active_abstract_action,
                open_loop_count=len(unresolved_loops),
                commitment_count=(
                    len(getattr(commitment_snapshot.value, "active_commitments", ()) or ())
                    if commitment_snapshot is not None
                    else 0
                ),
                pe_magnitude=pe_magnitude,
                elapsed_at_tick=self._tick.tick_index,
            )
        )
        return result

    async def end_scene(
        self,
        *,
        reason: str = "scene-end",
        drain_slow_loop: bool = True,
    ) -> Scene | None:
        """Close the open scene, fire the kernel boundary, optionally drain.

        Calling this when no scene is open is a no-op (returns ``None``).
        """
        scene = self._scene.open_scene
        if scene is None:
            return None

        # Capture open-loop / commitment refs at scene close for the record.
        open_loop_keys = self._extract_open_loop_keys()
        commitment_keys = self._extract_commitment_keys()

        closed = self._scene.close_open_scene(
            current_tick=self._tick.tick_index,
            open_loops=open_loop_keys,
            commitments=commitment_keys,
        )

        # Schedule a scene-end followup if there are unresolved open loops.
        self._followups.ingest_scene_close(
            scene_id=scene.scene_id,
            open_loops=open_loop_keys,
            current_tick=self._tick.tick_index,
        )

        # Hit the kernel boundary so session-post slow loop is enqueued.
        runner = self._brain_session.runner
        runner.begin_new_context(reason=reason)
        if drain_slow_loop:
            await runner.drain_session_post_slow_loop()
        return closed

    # ------------------------------------------------------------------
    # Tick advancement
    # ------------------------------------------------------------------

    async def advance_tick(self, system_ticks: int = 1, *, reason: str = "") -> tuple[TickEvent, ...]:
        """Advance the metabolic clock.

        After tick advancement the SceneManager is consulted for idle-close
        eligibility; if eligible AND there is an open scene, the scene is
        closed automatically (which fires ``end_scene``).
        """
        events = await self._tick.advance(system_ticks, reason=reason)
        if any(self._scene.on_tick(ev) for ev in events):
            await self.end_scene(reason="idle-timeout", drain_slow_loop=False)
        return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_open_loop_keys(self) -> tuple[str, ...]:
        snap = self._latest_active_snapshots.get("open_loop")
        if snap is None:
            return ()
        loops = getattr(snap.value, "unresolved_loops", ()) or ()
        out: list[str] = []
        for entry in loops:
            for attr in ("loop_id", "id", "key", "ref"):
                value = getattr(entry, attr, None)
                if value:
                    out.append(str(value))
                    break
            else:
                if isinstance(entry, str):
                    out.append(entry)
        return tuple(out)

    def _extract_commitment_keys(self) -> tuple[str, ...]:
        snap = self._latest_active_snapshots.get("commitment")
        if snap is None:
            return ()
        active = getattr(snap.value, "active_commitments", ()) or ()
        out: list[str] = []
        for entry in active:
            for attr in ("commitment_ref", "id", "ref"):
                value = getattr(entry, attr, None)
                if value:
                    out.append(str(value))
                    break
            else:
                if isinstance(entry, str):
                    out.append(entry)
        return tuple(out)
