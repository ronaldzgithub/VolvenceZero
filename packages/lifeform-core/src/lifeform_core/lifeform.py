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
5. **Thinking loop wiring is optional + protocol-only.** ``LifeformSession``
   never imports ``lifeform-thinking`` directly. It accepts an adapter
   object (conforming to ``ThinkingAdapterProtocol``) and calls three
   narrow methods at turn / scene lifecycle points. Keeps
   ``lifeform-core`` independent of the thinking wheel.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lifeform_core.followup_manager import FollowupManager
from lifeform_core.scene_manager import SceneManager
from lifeform_core.tick_engine import TickEngine, TickEngineConfig
from lifeform_core.types import (
    FollowupItem,
    Scene,
    TickEvent,
    TurnSummary,
    TurnTriggerKind,
    is_apprenticeship_trigger,
)
from lifeform_core.vitals import VitalsBootstrap, VitalsModule, VitalsSnapshot

from volvence_zero.agent.response import (
    ResponseSynthesizer,
)
from volvence_zero.application.domain_experience import DomainExperiencePackage
from volvence_zero.brain import Brain, BrainConfig, BrainSession
from volvence_zero.semantic_state import (
    ExternalSemanticEventBatch,
    SemanticProposalRuntime,
)
from volvence_zero.regime import RegimeBootstrap
from volvence_zero.substrate import OpenWeightResidualRuntime, SubstrateAdapter
from volvence_zero.temporal import MetacontrollerParameterSnapshot


@runtime_checkable
class ThinkingAdapterProtocol(Protocol):
    """Narrow structural protocol for a thinking-loop adapter.

    ``LifeformSession`` calls exactly these three methods on the adapter
    at well-defined turn / scene lifecycle points. The session NEVER
    pokes at scheduler internals directly \u2014 that would couple
    ``lifeform-core`` to ``lifeform-thinking``'s concrete types and
    violate the wheel-boundary rule from SPLIT.md.

    Concrete implementation lives in ``lifeform-thinking.adapter``.
    Tests can supply a fake adapter implementing only these three
    methods. Missing-method errors raise at call time (duck-typed), not
    at adapter construction.
    """

    async def on_turn_begin(
        self, *, snapshots: Mapping[str, Any], turn_index: int
    ) -> None: ...

    async def on_turn_end(
        self, *, snapshots: Mapping[str, Any], turn_index: int
    ) -> None: ...

    async def drain(self) -> Any: ...


ThinkingAdapterFactory = Callable[[], Any]
"""Factory that builds a per-session thinking adapter.

Returning ``None`` is legal (= "don't wire thinking for this session").
The return type is ``Any`` \u2014 not ``ThinkingAdapterProtocol`` \u2014 because
Python's ``runtime_checkable`` ``Protocol`` does not statically enforce
method signatures; adapters duck-type at call time anyway.
"""


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
    vitals_bootstrap: VitalsBootstrap | None = None

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

    def with_vitals(self, bootstrap: VitalsBootstrap | None) -> "LifeformConfig":
        from dataclasses import replace as _replace
        return _replace(self, vitals_bootstrap=bootstrap)


class _LateBoundSessionHolder:
    """Mutable ref to a ``LifeformSession`` for late-binding closures.

    Why this exists: the synthesizer's ``interlocutor_state_provider``
    must be wired BEFORE the brain session is created (the brain
    session captures the synthesizer at construction time), but the
    callable it wraps must read from a ``LifeformSession`` that
    doesn't exist until AFTER the brain session is built. The
    closure captures THIS holder by identity, the factory back-fills
    ``self.session`` after the session is constructed, and the
    closure reads through to the live attribute.

    Single-attribute holder rather than a full delayed-init wrapper
    so it is impossible to mistake for a real session and accidentally
    invoke ``LifeformSession`` methods on it.
    """

    __slots__ = ("session",)

    def __init__(self) -> None:
        self.session: "LifeformSession | None" = None


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
        thinking_adapter_factory: ThinkingAdapterFactory | None = None,
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
        self._thinking_adapter_factory = thinking_adapter_factory
        self._init_kwargs = {
            "substrate_runtime": substrate_runtime,
            "substrate_adapter_factory": substrate_adapter_factory,
            "response_synthesizer": response_synthesizer,
            "semantic_proposal_runtime": semantic_proposal_runtime,
            "temporal_bootstrap": temporal_bootstrap,
            "regime_bootstrap": regime_bootstrap,
            "thinking_adapter_factory": thinking_adapter_factory,
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

    def with_thinking_adapter_factory(
        self,
        factory: ThinkingAdapterFactory | None,
    ) -> "Lifeform":
        """Return a clone of this lifeform with the given thinking-adapter factory.

        ``factory`` is called once per ``create_session`` to build a
        per-session adapter. Pass ``None`` to disable mid-frequency
        thinking for sessions created from the returned lifeform.

        The typical invocation (Gap 4 slice 2c default) is::

            from lifeform_thinking import build_default_thinking_adapter

            lf = base_lf.with_thinking_adapter_factory(
                build_default_thinking_adapter
            )

        ``build_default_thinking_adapter`` has all-defaulted keyword
        arguments, so it works directly as the factory.
        """
        new_kwargs = dict(self._init_kwargs)
        new_kwargs["thinking_adapter_factory"] = factory
        return Lifeform(self._config, **new_kwargs)

    def create_session(self, *, session_id: str = "lifeform-session") -> "LifeformSession":
        vitals = (
            VitalsModule(self._config.vitals_bootstrap)
            if self._config.vitals_bootstrap is not None
            else None
        )

        # Late-bound holder for ``LifeformSession.interlocutor_state``.
        # The synthesizer's ``interlocutor_state_provider`` closure
        # is captured BEFORE the session is constructed (we need the
        # synthesizer to wire the brain session); the closure reads
        # through this holder, which the factory back-fills after
        # the session is built. This is the canonical late-binding
        # pattern \u2014 not silent mutation: the holder's identity is
        # captured by the closure, only the ``session`` attribute
        # is set later.
        session_holder = _LateBoundSessionHolder()

        def _interlocutor_provider() -> Any:
            sess = session_holder.session
            return sess.interlocutor_state if sess is not None else None

        # Per-session synthesizer when vitals are wired AND/OR the
        # brain-level synthesizer can be cloned with a vitals
        # / interlocutor provider. We deliberately NEVER mutate the
        # Brain's own synthesizer \u2014 we only construct session-local
        # clones whose closures capture THIS session's state. That
        # preserves single-ownership for the brain default while
        # still letting drives + 12-axis readouts reach the planner.
        session_synthesizer = self._maybe_clone_synthesizer_for_session(
            vitals=vitals,
            interlocutor_provider=_interlocutor_provider,
        )
        brain_session = self._brain.create_session(
            session_id=session_id,
            response_synthesizer=session_synthesizer,
        )
        thinking_adapter: Any = None
        if self._thinking_adapter_factory is not None:
            thinking_adapter = self._thinking_adapter_factory()
        session = LifeformSession(
            brain_session=brain_session,
            tick=TickEngine(self._config.tick),
            scene=SceneManager(idle_close_after_system_ticks=self._config.idle_close_after_system_ticks),
            followups=FollowupManager(
                default_due_delay_ticks=self._config.followup_default_due_delay_ticks,
                max_pending=self._config.followup_max_pending,
            ),
            vitals=vitals,
            thinking_adapter=thinking_adapter,
        )
        session_holder.session = session
        return session

    def _maybe_clone_synthesizer_for_session(
        self,
        *,
        vitals: VitalsModule | None,
        interlocutor_provider: Callable[[], Any],
    ) -> ResponseSynthesizer | None:
        """Return a per-session synthesizer clone bound to this session's state.

        Two providers are wired in one place because both are
        session-scoped and both target ``GroundedResponseSynthesizer``:

        * ``vitals.current_snapshot`` \u2014 always available when
          ``vitals`` is non-None;
        * ``interlocutor_provider`` \u2014 a late-bound closure that
          reads ``LifeformSession.interlocutor_state`` after the
          session is constructed.

        Returns ``None`` (i.e. fall back to the Brain's default) when:

        * the brain-level synthesizer is not a ``GroundedResponseSynthesizer``
          (the only synthesizer shape that knows how to consume these
          providers). Plain ``ResponseSynthesizer`` and custom subclasses
          are passed through unchanged \u2014 not silently downgraded \u2014 so
          callers see the synthesizer they constructed;
        * neither vitals nor interlocutor wiring is available (in
          practice the interlocutor provider is always available
          since it just reads the readout, but we keep the guard
          symmetric for future shape changes).
        """
        synth = self._init_kwargs.get("response_synthesizer")
        if synth is None:
            return None
        # Lazy import to keep ``lifeform-core`` from depending on
        # ``lifeform-expression`` at module import time. The dependency
        # is already implicit (the user supplied a Grounded synthesizer
        # constructed from lifeform-expression), so the import only
        # actually runs when the user opted into that synthesizer.
        try:
            from lifeform_expression.response_synthesizer import (
                GroundedResponseSynthesizer,
            )
        except ImportError:
            return None
        if not isinstance(synth, GroundedResponseSynthesizer):
            return None
        cloned = synth
        if vitals is not None:
            cloned = cloned.with_vitals_provider(vitals.current_snapshot)
        cloned = cloned.with_interlocutor_provider(interlocutor_provider)
        return cloned


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
        vitals: VitalsModule | None = None,
        thinking_adapter: Any = None,
    ) -> None:
        self._brain_session = brain_session
        self._tick = tick
        self._scene = scene
        self._followups = followups
        self._vitals = vitals
        self._turn_summaries: list[TurnSummary] = []
        # Re-publish snapshot fields after every turn so consumers can see
        # the cross-cutting lifeform state without poking at internals.
        self._latest_active_snapshots: dict[str, Any] = {}
        self._latest_response_text: str = ""
        # Gap 4 slice 2a observability: last scene-end case-memory
        # reconcile result, if any. Typed ``Any`` here to avoid a hard
        # kernel import at module load time \u2014 the value is a kernel
        # ``ProvisionalReconcileResult``. None means no scene has
        # closed yet.
        self._latest_case_reconcile: Any = None
        # Gap 4 slice 2c: optional thinking adapter. Duck-typed (Any)
        # so lifeform-core does not import lifeform-thinking. The
        # adapter (when present) is called at three well-defined
        # lifecycle points; see ``_invoke_thinking_*`` helpers below.
        self._thinking_adapter: Any = thinking_adapter

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
    def vitals_module(self) -> VitalsModule | None:
        return self._vitals

    @property
    def vitals_snapshot(self) -> VitalsSnapshot | None:
        return self._vitals.current_snapshot() if self._vitals is not None else None

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

    @property
    def latest_case_memory_reconcile(self) -> Any:
        """Last scene-end case-memory provisional reconcile result.

        Returns a kernel-side ``ProvisionalReconcileResult`` with
        ``promoted`` / ``retired`` / ``expired`` case_id tuples plus a
        per-decision audit trail, or ``None`` when no scene has closed
        yet in this session. Provides observability into the Gap 4
        case-memory lifecycle without poking owner internals.
        """
        return self._latest_case_reconcile

    @property
    def thinking_adapter(self) -> Any:
        """Return the attached thinking adapter (or None).

        Exposed for tests / observability; product code should read
        ``thinking_adapter_snapshot`` instead of poking at the
        adapter directly. Returning ``Any`` here avoids a hard
        import of the ``lifeform-thinking`` wheel.
        """
        return self._thinking_adapter

    @property
    def thinking_adapter_snapshot(self) -> Any:
        """Return the adapter's ``snapshot()`` or None when no adapter.

        Type is ``Any`` (not a concrete dataclass) to keep
        ``lifeform-core`` wheel-independent of ``lifeform-thinking``.
        Consumers that want the typed view cast it themselves.
        """
        if self._thinking_adapter is None:
            return None
        snapshot_fn = getattr(self._thinking_adapter, "snapshot", None)
        if snapshot_fn is None:
            return None
        return snapshot_fn()

    @property
    def interlocutor_state(self) -> Any:
        """Gap 9 slice 1: 12-axis readout of the current interlocutor.

        Derived at read time from the latest kernel snapshots
        (``regime`` / ``dual_track`` / ``evaluation`` /
        ``prediction_error`` / ``memory`` / ``commitment``) via
        the pure-function readout in ``volvence_zero.interlocutor``.
        Returns a default ``InterlocutorState`` (neutral mid-point
        + low ``readout_confidence``) when no turn has run yet.

        The readout is **not** a kernel owner; it's a continuous
        view computed at consumer time, same pattern as
        ``latest_thinking_artifacts_by_consumer``. Downstream
        product code that wants to style its response against
        the interlocutor's resistance / openness / rapport should
        read this property + gate on ``readout_confidence``.
        """
        # Local import to keep the module load graph thin; the
        # readout is pure + stateless, so per-call import is cheap
        # after the first.
        from volvence_zero.interlocutor import (
            build_interlocutor_readout_context_from_snapshots,
            readout_interlocutor_state,
        )

        snaps = self._latest_active_snapshots
        context = build_interlocutor_readout_context_from_snapshots(
            regime_snapshot=snaps.get("regime"),
            dual_track_snapshot=snaps.get("dual_track"),
            evaluation_snapshot=snaps.get("evaluation"),
            prediction_error_snapshot=snaps.get("prediction_error"),
            memory_snapshot=snaps.get("memory"),
            commitment_snapshot=snaps.get("commitment"),
        )
        return readout_interlocutor_state(context)

    @property
    def latest_thinking_artifacts_by_consumer(self) -> Mapping[str, Any]:
        """Return the latest appliable mid-reflection artifacts.

        Keyed by consumer owner name (``world_temporal`` /
        ``self_temporal`` by default). Empty dict when no adapter
        is wired OR no artifacts have completed yet.

        Downstream consumers that want to ACT on an artifact should
        still check ``artifact.is_appliable()`` before reading
        ``payload`` \u2014 the adapter filters non-appliable ones, but
        defense in depth is cheap.
        """
        if self._thinking_adapter is None:
            return {}
        getter = getattr(
            self._thinking_adapter, "latest_artifacts_by_consumer", None
        )
        if getter is None:
            return {}
        return dict(getter)

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

    async def run_turn(
        self,
        user_input: str,
        *,
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
    ) -> Any:
        """Run one turn through the kernel.

        Returns the kernel's ``AgentTurnResult`` unchanged so callers can
        inspect every snapshot. Side effects:

        * Open a new scene if none exists.
        * Increment the open scene's turn counter.
        * Record a compact ``TurnSummary`` (tagged with ``trigger_kind``).
        * Pull ``open_loop`` / ``commitment`` snapshots and feed the
          ``FollowupManager``.

        ``trigger_kind`` (Gap 2) is a pure observability label EXCEPT
        for two values \u2014 ``APPRENTICE`` / ``INGESTION`` \u2014 which
        activate the vitals apprentice override for the duration of
        this turn only. See ``is_apprenticeship_trigger``. The
        override is restored in a ``finally`` block so even exceptions
        from the kernel cannot leak the flag into subsequent turns.
        """
        # Open scene if needed; this is the only place a scene auto-opens.
        if self._scene.open_scene is None:
            self._scene.open_scene_now(current_tick=self._tick.tick_index)

        # Gap 4 slice 2c: collect any thinking artifacts submitted
        # at the end of the previous turn. This runs BEFORE the
        # kernel executes this turn so the fingerprint guard sees
        # the previous-turn snapshots (which are what tasks were
        # submitted against). After this turn runs, any still-pending
        # tasks will mismatch on the next collect and go STALE.
        await self._invoke_thinking_on_turn_begin(
            turn_index=len(self._turn_summaries),
        )

        apprentice_turn = is_apprenticeship_trigger(trigger_kind)
        vitals_override_was_active = (
            self._vitals.apprentice_override_active
            if self._vitals is not None
            else False
        )
        if apprentice_turn and self._vitals is not None:
            self._vitals.set_apprentice_override(True)
        try:
            result = await self._brain_session.run_turn_async(user_input)
        finally:
            # Leak-free invariant: restore the prior override state
            # regardless of whether the kernel raised. Nested calls
            # (tests, scripted scenarios) see the same pre-turn state
            # they had.
            if apprentice_turn and self._vitals is not None:
                self._vitals.set_apprentice_override(vitals_override_was_active)

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
            commitment_value = commitment_snapshot.value
            # Owner-side ``at_risk_commitments`` (status=blocked records)
            # are still surfaced via the classic ingest path so product
            # layers that consumed the old shape keep working.
            at_risk_records = tuple(
                getattr(commitment_value, "at_risk_commitments", ()) or ()
            )
            if at_risk_records:
                self._followups.ingest_at_risk_commitments(
                    at_risk_refs=at_risk_records,
                    current_tick=self._tick.tick_index,
                )
            # Gap 7: lifecycle-aware ingestion. We hand the entire
            # ``lifecycle_entries`` tuple to the follow-up manager and
            # let it apply the ``followup_policy`` per record \u2014
            # GENTLE_CHECKIN vs DEFER_ONLY \u2014 rather than fabricating a
            # classifier here. This keeps the lifeform layer out of the
            # business of interpreting commitment state.
            lifecycle_entries = tuple(
                getattr(commitment_value, "lifecycle_entries", ()) or ()
            )
            if lifecycle_entries:
                self._followups.ingest_commitment_lifecycle(
                    lifecycle_entries=lifecycle_entries,
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
                trigger_kind=trigger_kind,
            )
        )

        # Vitals: this turn recharges drives based on the active regime.
        # ``user_input_present`` applies the per-turn baseline recharge
        # \u2014 we set it to True for USER_INPUT / FOLLOWUP_DUE /
        # INTERNAL_DRIVE turns (a real exchange happened) and False
        # for APPRENTICE / INGESTION turns (operator-supplied content,
        # not a user engagement signal). Regime-keyed bonuses still
        # apply in both cases, matching the bootstrap semantics.
        if self._vitals is not None:
            self._vitals.on_turn(
                regime=result.active_regime,
                user_input_present=not apprentice_turn,
            )

        # Gap 4 slice 2c: submit mid-reflection tasks with the
        # snapshots this kernel turn just produced. The adapter
        # closes over its own scheduler; this call returns as soon
        # as the tasks are enqueued (workers run concurrently).
        await self._invoke_thinking_on_turn_end(
            snapshots=result.active_snapshots,
            turn_index=len(self._turn_summaries),
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
        # Gap 4 slice 2a: after drain (so any provisional cases the
        # slow loop just wrote are part of the decision set) sweep
        # case_memory with the lifeform's current tick. Idle-timeout
        # scenes ``drain_slow_loop=False`` would starve the reconcile,
        # so we still invoke it \u2014 it is cheap, pure per-record work
        # gated on ``lifecycle != VALIDATED``.
        self._latest_case_reconcile = self._brain_session.reconcile_case_memory_provisional(
            now_tick=self._tick.tick_index,
        )
        # Gap 4 slice 2c: drain the thinking scheduler so no worker
        # outlives the scene. Terminal artifacts are preserved for
        # post-mortem observability (``thinking_adapter_snapshot``).
        await self._invoke_thinking_drain()
        return closed

    # ------------------------------------------------------------------
    # Tick advancement
    # ------------------------------------------------------------------

    async def advance_tick(self, system_ticks: int = 1, *, reason: str = "") -> tuple[TickEvent, ...]:
        """Advance the metabolic clock.

        After tick advancement:

        1. The ``VitalsModule`` (if any) decays drive levels on every
           ``SYSTEM`` tick. When the resulting slow-scale PE crosses the
           configured threshold AND we are outside the cooldown, a
           proactive ``FollowupItem`` is surfaced via ``FollowupManager``
           \u2014 the lifeform layer's "I am alive between turns" signal.
        2. The ``SceneManager`` is consulted for idle-close eligibility;
           if eligible AND there is an open scene, the scene is closed
           automatically (which fires ``end_scene``).
        """
        events = await self._tick.advance(system_ticks, reason=reason)

        if self._vitals is not None:
            for ev in events:
                self._vitals.on_tick(ev)
            if self._vitals.consider_proactive_followup(
                current_tick=self._tick.tick_index
            ):
                snap = self._vitals.current_snapshot()
                self._followups.ingest_proactive_drive_pressure(
                    total_pe=snap.total_pe,
                    out_of_band_drive_names=tuple(
                        d.name for d in snap.drive_levels if d.out_of_band
                    ),
                    current_tick=self._tick.tick_index,
                    priority=self._vitals.bootstrap.proactive_followup_priority,
                )

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

    # ------------------------------------------------------------------
    # Gap 4 slice 2c: thinking-loop invocation helpers
    # ------------------------------------------------------------------
    #
    # The three helpers below are the ONLY places in ``lifeform-core``
    # that call into the thinking adapter. They are deliberately:
    #
    # * **Duck-typed.** No ``isinstance`` check against a concrete
    #   ``ThinkingAdapter`` class \u2014 ``lifeform-core`` must not import
    #   ``lifeform-thinking``. The adapter conforms to
    #   ``ThinkingAdapterProtocol`` (defined above) but at call time
    #   we just look up the method and call it.
    # * **Non-fatal.** An adapter raising inside any of these hooks
    #   would leak a failure mode into the normal turn path; we
    #   catch and log so a buggy adapter cannot break the turn.
    #   A contract test enforces that the SUPPLIED adapter
    #   (lifeform-thinking's default one) never actually raises;
    #   this try/except is defence in depth.

    async def _invoke_thinking_on_turn_begin(self, *, turn_index: int) -> None:
        if self._thinking_adapter is None:
            return
        hook: Callable[..., Awaitable[None]] | None = getattr(
            self._thinking_adapter, "on_turn_begin", None
        )
        if hook is None:
            return
        try:
            await hook(
                snapshots=self._latest_active_snapshots,
                turn_index=turn_index,
            )
        except Exception:  # noqa: BLE001 - adapter isolation boundary
            import logging as _logging

            _logging.getLogger("lifeform_core.lifeform").exception(
                "thinking_adapter.on_turn_begin raised; continuing turn"
            )

    async def _invoke_thinking_on_turn_end(
        self,
        *,
        snapshots: Mapping[str, Any],
        turn_index: int,
    ) -> None:
        if self._thinking_adapter is None:
            return
        hook: Callable[..., Awaitable[None]] | None = getattr(
            self._thinking_adapter, "on_turn_end", None
        )
        if hook is None:
            return
        try:
            await hook(snapshots=snapshots, turn_index=turn_index)
        except Exception:  # noqa: BLE001 - adapter isolation boundary
            import logging as _logging

            _logging.getLogger("lifeform_core.lifeform").exception(
                "thinking_adapter.on_turn_end raised; continuing turn"
            )

    async def _invoke_thinking_drain(self) -> None:
        if self._thinking_adapter is None:
            return
        hook: Callable[..., Awaitable[Any]] | None = getattr(
            self._thinking_adapter, "drain", None
        )
        if hook is None:
            return
        try:
            await hook()
        except Exception:  # noqa: BLE001 - adapter isolation boundary
            import logging as _logging

            _logging.getLogger("lifeform_core.lifeform").exception(
                "thinking_adapter.drain raised; continuing scene close"
            )
