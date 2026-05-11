"""``ProtocolRegistryModule`` — application-tier owner of the ``active_mixture`` slot.

Reads upstream ``dual_track`` and ``regime`` snapshots (packet
1.3a; packet 1.0 had no upstream); publishes a frozen
:class:`ActiveMixtureSnapshot` each turn at ``WiringLevel.SHADOW``.

Lifecycle: external callers (FixtureUptake adapters in
``lifeform-domain-*``; future DocumentUptake in
``lifeform-protocol-runtime``) call ``load_protocol(protocol)``
between turns to register protocols. The module reads the registry
inside ``process()`` and constructs the snapshot via
``compute_active_mixture``.

Why SHADOW in packet 1.0:

* SHADOW lets ``propagate`` validate the snapshot shape and surface
  any contract violations early without affecting baseline
  behaviour. ``shadow_snapshots`` capture the published value for
  dual-run diffing (see ``vz-runtime`` propagate flow).

Compile path (packet 1.2 / 1.3b / 1.4a / 1.4b):

* When constructed with an injected
  ``ApplicationRareHeavyState`` (kernel ``vz-application`` owner),
  ``load_protocol`` additionally compiles:
    - ``BehaviorProtocol.boundary_contracts`` →
      ``BoundaryPriorHint`` (packet 1.2)
    - ``BehaviorProtocol.strategy_priors`` → ``PlaybookRule``
      (packet 1.3b)
  …and upserts both kinds into the rare-heavy state.
* When additionally constructed with an injected
  ``ApplicationDomainKnowledgeStore``, ``load_protocol`` compiles:
    - ``BehaviorProtocol.knowledge_seeds`` →
      ``DomainKnowledgeRecord`` (packet 1.4a)
* When additionally constructed with an injected
  ``ApplicationCaseMemoryStore``, ``load_protocol`` compiles:
    - ``BehaviorProtocol.signature_cases`` →
      ``CaseMemoryRecord`` (packet 1.4b)
* The application owners (``boundary_policy`` /
  ``strategy_playbook`` / ``domain_knowledge`` /
  ``case_memory``) remain the canonical writers of their own
  state; ProtocolRuntime only feeds them.
* Without an injected state, ``load_protocol`` is registry-only
  (snapshot-shape verification path). Tests use this to exercise
  the shape contract without setting up a full application
  state.

Unload path (deferred):

* ``unload_protocol`` raises ``NotImplementedError`` when a
  rare-heavy state was injected and the protocol had any
  non-empty compile output (boundary or strategy).
  ``ApplicationRareHeavyState`` has no per-key remove API today;
  clean unload requires either extending the state with
  ``remove_*_by_id_prefix`` APIs or rebuilding from a checkpoint
  without the protocol's entries. Both are out of packet 1.2 /
  1.3b scope and tracked for packet 1.6+. Unload of an unused
  (no rare-heavy state injected) registry entry is allowed.

See ``docs/specs/protocol-runtime.md`` for the full design.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    ProtocolRevisionProposal,
    ReviewStatus,
)
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.runtime.kernel import ContractViolationError

from volvence_zero.protocol_runtime.activation import (
    compute_active_mixture,
    is_fallback_mode,
)
from volvence_zero.protocol_runtime.compiler import (
    compile_protocol_to_application_artifacts,
)
from volvence_zero.protocol_runtime.registry import ProtocolRegistry


# Packet 1.5b: per-protocol rolling PE EMA learning rate. The owner
# updates ``pe_utility[i] ← (1-η) · pe_utility[i] + η · attributed_reward``
# each turn for every protocol that was active in the previous turn.
# η = 0.25 is a moderate setting: ~4 turns to converge to a step
# response, conservative enough to not jitter. Hard-coded for
# packet 1.5b; learned online in packet 1.5c (the metacontroller
# learns α/β plus a per-protocol effective learning rate).
_PE_HISTORY_LEARNING_RATE: float = 0.25

# Bound the EMA to ``[-1, 1]`` mirroring ``signed_reward`` range.
# Prevents pathological accumulation if the same protocol keeps
# being attributed (small-eligibility regimes). Acts as a soft
# saturation; real saturation comes from the EMA decay.
_PE_UTILITY_CLAMP: float = 1.0


# Packet 1.5c-iii: α / β online learning rate. The owner adjusts
# the activation-formula coefficients each PE turn using a
# REINFORCE-style proxy gradient:
#
#     α_grad = signed_reward × range(context_match across last actives)
#     β_grad = signed_reward × range(pe_utility across last actives)
#     α ← clamp(α + η_meta · α_grad, [α_min, α_max])
#     β ← clamp(β + η_meta · β_grad, [α_min, α_max])
#
# Intuition: when one signal differentiated protocols sharply (large
# range) and the outcome was positive (signed_reward > 0), the
# decision was good and the differentiating signal was useful →
# raise its coefficient. Negative reward inverts the sign.
# Single-protocol mixtures have range = 0, so cheng_laoshi-shape
# fixtures never trigger learning (cheng_laoshi byte-equivalence
# preserved across this packet — pinned by tests).
#
# η_meta = 0.05 chosen ~5x slower than the pe_utility EMA's η = 0.25:
# α / β are global hyperparameters that change the shape of every
# decision, while pe_utility is per-protocol and self-localising.
# Slow meta updates avoid jitter under noisy PE.
_ALPHA_BETA_LEARNING_RATE: float = 0.05

# Hard clamps on α / β to prevent collapse (toward 0, in which case
# the corresponding signal is silently muted forever) or runaway
# (which would saturate softmax to argmax). 0.1 is a soft floor
# that still lets a signal contribute meaningfully; 5.0 is plenty
# of dynamic range for differentiated mixtures.
_ALPHA_BETA_MIN: float = 0.1
_ALPHA_BETA_MAX: float = 5.0


class FallbackActivationActiveError(ContractViolationError):
    """Raised when ``ProtocolRegistryModule`` is constructed at ACTIVE
    while the activation controller is still in fallback mode.

    Packet 1.0 ships placeholder activation logic (equal-weight,
    hard-coded ``identity_gate=1.0``, lexicographic incompatibility
    arbitration). Promoting the slot to ACTIVE without real
    machinery would let downstream consumers act on outputs that are
    not a learned posterior. The SHADOW → ACTIVE checklist in
    ``docs/specs/protocol-runtime.md`` lists the required upgrades.

    Future packets (1.3 / 1.5) that land the real machinery flip
    ``activation._ACTIVATION_CONTROLLER_FALLBACK_MODE`` to False in
    the same landing PR; this error is then unreachable.
    """


class ProtocolRegistryModule(RuntimeModule[ActiveMixtureSnapshot]):
    """SHADOW owner of the ``active_mixture`` slot (packet 1.0)."""

    slot_name: ClassVar[str] = "active_mixture"
    owner: ClassVar[str] = "ProtocolRegistryModule"
    value_type: ClassVar[type[Any]] = ActiveMixtureSnapshot
    # Identity gate (packet 1.3a/1.3'): ``dual_track`` (R7 Self
    # track + IdentitySeed traits) + ``regime`` (R14 RegimeIdentity).
    # context_match (packet 1.5a + 1.5a'): ``interlocutor_state``
    # (zone activity) + ``rupture_state`` (rupture_kind /
    # USER_DROPOUT_OBSERVED) + ``boundary_policy`` (decision
    # triggered) + ``regime`` (REGIME_TRANSITION_RECENT, shared
    # with identity_gate) + ``retrieval_policy``
    # (RETRIEVAL_HITS_PRESENT).
    # PE utility (packet 1.5b): ``prediction_error`` (signed_reward
    # attributed to last-turn active mixture, EMA-accumulated
    # internally). All upstream readers are SHADOW-tolerant —
    # missing snapshots yield "no signal fires / no PE update"
    # rather than fail-loud. As of packet 1.5a' the activation
    # controller is no longer in fallback mode, so ACTIVE wiring
    # is legal (gated only by the runtime's normal owner-graph
    # validation).
    dependencies: ClassVar[tuple[str, ...]] = (
        "dual_track",
        "regime",
        "interlocutor_state",
        "rupture_state",
        "boundary_policy",
        "prediction_error",
        "retrieval_policy",
    )
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        registry: ProtocolRegistry | None = None,
        application_rare_heavy_state: ApplicationRareHeavyState | None = None,
        domain_knowledge_store: ApplicationDomainKnowledgeStore | None = None,
        case_memory_store: ApplicationCaseMemoryStore | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        # Packet 1.0.1 ACTIVE-gate guard: refuse to construct at
        # ACTIVE while the activation controller is in fallback. The
        # default wiring level is SHADOW; reaching this check means
        # someone explicitly passed ``wiring_level=ACTIVE`` (or a
        # config flipped ``FinalRolloutConfig.protocol_runtime`` to
        # ACTIVE) without first landing the upgrades enumerated in
        # the spec's SHADOW → ACTIVE checklist.
        if (
            self.wiring_level is WiringLevel.ACTIVE
            and is_fallback_mode()
        ):
            raise FallbackActivationActiveError(
                "ProtocolRegistryModule cannot be wired ACTIVE while "
                "the activation controller is still in packet-1.0 "
                "fallback mode. See docs/specs/protocol-runtime.md "
                "§SHADOW → ACTIVE 升级 checklist for the required "
                "upgrades (real identity_gate, learned α/β, typed "
                "context_match signals, PE-driven arbitration, plus "
                "at least one matched-control consumer test)."
            )
        self._registry = registry if registry is not None else ProtocolRegistry()
        self._application_rare_heavy_state = application_rare_heavy_state
        self._domain_knowledge_store = domain_knowledge_store
        self._case_memory_store = case_memory_store
        # Track which protocols had artifacts applied to *any* of the
        # injected stores (rare-heavy / domain-knowledge / case-memory).
        # Used by unload_protocol to fail-loud when artifacts can't be
        # cleanly removed (no per-key remove API on the stores today).
        self._applied_to_application_state: set[str] = set()

        # Packet 1.5b: per-protocol rolling pe_utility (EMA of
        # signed_reward attributed to last-turn weights). Lives
        # inside the registry owner because protocol load/unload
        # is the natural lifecycle anchor (unload → drop entry).
        # An R8-clean split into a dedicated ``ProtocolPerformanceModule``
        # owner is tracked for packet 1.5c-ii. Until then keeping
        # both responsibilities co-located avoids an extra slot
        # without weakening any actual contract.
        self._pe_utility: dict[str, float] = {}
        # Last published mixture's per-protocol weights, used to
        # attribute the current turn's signed_reward back to the
        # protocols that were active when the action was taken (PE
        # at turn t reflects outcome of turn t-1's action).
        self._last_active_weights: dict[str, float] = {}
        # Last seen prediction_error turn_index, so duplicate
        # snapshots from the same turn (e.g. retries / replay)
        # don't double-count attribution.
        self._last_pe_turn_index: int | None = None

        # Packet 1.5c-iii: α / β online learning state.
        # Initialised to 1.0 each — this matches packet 1.5b's
        # hardcoded values, so cold-start behaviour is identical.
        # Updated by ``_update_alpha_beta`` after every (non-bootstrap,
        # non-duplicate) PE turn that has cached signal data from
        # the previous turn.
        self._alpha: float = 1.0
        self._beta: float = 1.0
        # Snapshots of last turn's per-protocol context_match and
        # pe_utility values, used to compute ``range(signal)`` for
        # the α / β gradient. context_scores capture the bare
        # context_match per eligible protocol (not α·context_match).
        self._last_context_scores: dict[str, float] = {}
        # pe_utility snapshot at LAST turn's mixture-computation time
        # (before this turn's pe_utility EMA update). Needed because
        # ``self._pe_utility`` evolves on every turn; if we read it
        # AFTER the EMA update we'd be using a different slice than
        # what last turn's softmax actually saw.
        self._last_pe_utilities: dict[str, float] = {}

    @property
    def registry(self) -> ProtocolRegistry:
        """Public handle for adapters (``FixtureUptake`` etc.) to load protocols.

        Adapters call ``module.registry.load(protocol)`` between
        turns. Reads (``loaded()`` / ``get(...)``) are safe
        concurrently with ``process()``.
        """

        return self._registry

    def load_protocol(self, protocol: BehaviorProtocol) -> None:
        """Register a protocol; if a rare-heavy state was injected,
        compile artifacts and apply to the application owners.

        Order matters: registry first, application state second. If
        any application state apply step fails (e.g. application
        validator rejects an empty trigger list), the protocol is
        rolled back out of the registry so the system stays
        consistent.

        Packet 1.2 applied boundary artifacts only; packet 1.3b
        adds strategy artifacts (``PlaybookRule``). Both are upserted
        atomically — the rollback covers both. If a future packet
        adds case / knowledge artifacts, follow the same pattern.
        """

        self._registry.load(protocol)
        # Short-circuit: no application stores injected → registry-only.
        if (
            self._application_rare_heavy_state is None
            and self._domain_knowledge_store is None
            and self._case_memory_store is None
        ):
            return
        try:
            artifacts = compile_protocol_to_application_artifacts(protocol)
            applied_anything = False
            if (
                self._application_rare_heavy_state is not None
                and artifacts.boundary_prior_hints
            ):
                self._application_rare_heavy_state.upsert_boundary_prior_hints(
                    artifacts.boundary_prior_hints
                )
                applied_anything = True
            if (
                self._application_rare_heavy_state is not None
                and artifacts.playbook_rules
            ):
                self._application_rare_heavy_state.upsert_distilled_playbook_rules(
                    artifacts.playbook_rules
                )
                applied_anything = True
            if (
                self._domain_knowledge_store is not None
                and artifacts.domain_knowledge_records
            ):
                self._domain_knowledge_store.upsert_records(
                    artifacts.domain_knowledge_records
                )
                applied_anything = True
            if (
                self._case_memory_store is not None
                and artifacts.case_memory_records
            ):
                self._case_memory_store.upsert_records(
                    artifacts.case_memory_records
                )
                applied_anything = True
            if applied_anything:
                self._applied_to_application_state.add(protocol.protocol_id)
        except Exception:
            # Roll the registry back so a failed apply doesn't leave
            # a half-loaded protocol behind. Note: any partial application
            # state writes that succeeded BEFORE the failure remain in
            # state; full transactional rollback would require checkpoint
            # snapshotting (deferred to packet 1.6+ alongside unload).
            self._registry.unload(protocol.protocol_id)
            raise

    def load_protocol_candidate(
        self,
        candidate: BehaviorProtocolCandidate,
        *,
        force: bool = False,
    ) -> None:
        """Packet 2.4: load a DocumentUptake-derived candidate.

        Differs from :meth:`load_protocol` in that the candidate
        carries a ``requires_review`` flag plus provenance audit.
        Routing rules:

        * ``candidate.requires_review=True`` (default for any
          LLM-extracted candidate): reject unless the inner
          protocol's ``review_status`` is ``SHADOW`` or ``ACTIVE``
          (i.e. an external reviewer has already stamped it via
          ``approve_candidate``). DRAFT candidates are blocked
          unless ``force=True``.
        * ``candidate.requires_review=False`` (trusted upstream
          like API_INJECTION from a reviewed source): accepted
          directly.
        * ``force=True`` is an explicit override for emergency
          paths (test fixtures, recovery). The override is logged
          in the audit trail of any caller that uses it; the
          schema does NOT trust ``force`` silently.

        On accept, delegates to :meth:`load_protocol` so the
        compile path runs uniformly for fixture / candidate
        loads.
        """

        if (
            candidate.requires_review
            and not force
            and candidate.protocol.review_status is ReviewStatus.DRAFT
        ):
            raise PermissionError(
                f"candidate {candidate.protocol.protocol_id!r} requires "
                f"review; got review_status=DRAFT. Run review.approve_candidate "
                f"first or pass force=True if you have an explicit override."
            )
        self.load_protocol(candidate.protocol)

    def unload_protocol(self, protocol_id: str) -> bool:
        """Convenience for adapters: equivalent to ``self.registry.unload(...)``.

        Packet 1.2 deferral: when a protocol has artifacts already
        applied to the application state, unload raises
        ``NotImplementedError``. ``ApplicationRareHeavyState`` has
        no per-key remove API today; clean unload requires either
        extending the state with
        ``remove_boundary_prior_hints_by_id_prefix`` or rebuilding
        from a checkpoint without the protocol's entries. This
        path will land alongside protocol revision (packet 1.6+)
        and is intentionally fail-loud here so callers know the
        operation is unsupported.
        """

        if protocol_id in self._applied_to_application_state:
            raise NotImplementedError(
                f"unload_protocol({protocol_id!r}) cannot remove "
                "already-applied application artifacts (boundary, "
                "playbook, domain-knowledge, or case-memory). "
                "ApplicationRareHeavyState / ApplicationDomainKnowledgeStore / "
                "ApplicationCaseMemoryStore lack per-key remove APIs. "
                "Wait for packet 1.6+ or rebuild from a checkpoint "
                "without this protocol's entries. See "
                "docs/specs/protocol-runtime.md §packet 1.2 / 1.3b / 1.4a / "
                "1.4b unload deferral."
            )
        return self._registry.unload(protocol_id)

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ActiveMixtureSnapshot]:
        # Packet 1.5b/1.5c-iii: PE attribution + rolling EMA update
        # for pe_utility AND α / β happens BEFORE computing this
        # turn's mixture, so the current turn sees the freshly
        # updated values. The order is important: α/β learning
        # uses LAST turn's context_match / pe_utility snapshots,
        # so it must run before this turn's mixture overwrites
        # the caches.
        self._update_pe_history(upstream)
        self._update_alpha_beta(upstream)

        loaded = self._registry.loaded()
        # Capture this turn's per-eligible-protocol context_match
        # scores via the audit out-param so we can use them in
        # NEXT turn's α/β learning (range(context_match) needs
        # them snapshot-aligned with the mixture they produced).
        audit_context: dict[str, float] = {}
        snapshot_value = compute_active_mixture(
            loaded_protocols=loaded,
            upstream=upstream,
            pe_utility_by_id=self._pe_utility,
            alpha=self._alpha,
            beta=self._beta,
            audit_context_scores=audit_context,
        )

        # Cache this turn's published weights AND signal scores for
        # the next turn's PE attribution + α/β gradient. The
        # pe_utility snapshot mirrors the dict the softmax just
        # consumed (not the freshly-updated post-attribution one).
        self._last_active_weights = {
            entry.protocol_id: entry.activation_weight
            for entry in snapshot_value.active_protocols
        }
        self._last_context_scores = audit_context
        self._last_pe_utilities = {
            protocol_id: self._pe_utility.get(protocol_id, 0.0)
            for protocol_id in audit_context
        }
        return self.publish(snapshot_value)

    def _update_pe_history(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        """Attribute current turn's PE to last turn's active mixture.

        Updates ``self._pe_utility`` in place. SHADOW-tolerant:
        when ``prediction_error`` is missing from upstream (e.g.
        partial-graph SHADOW dual-run, isolated tests calling the
        owner directly without a full pipeline), the method
        returns immediately without raising. When
        ``_last_active_weights`` is empty (first turn / no
        protocols active last turn), no attribution happens — the
        EMA stays at its current values.

        Per-protocol update rule:

            Δ_i = signed_reward × last_weight_i
            ema_i ← (1-η) · ema_i + η · Δ_i

        η = ``_PE_HISTORY_LEARNING_RATE`` (0.25). Result clamped
        to ``[-_PE_UTILITY_CLAMP, +_PE_UTILITY_CLAMP]``.
        """

        pe_snapshot = upstream.get("prediction_error")
        if pe_snapshot is None:
            return
        pe_value = pe_snapshot.value
        if not isinstance(pe_value, PredictionErrorSnapshot):
            return
        # Bootstrap turns produce a placeholder PE that the system
        # (per ``PredictionErrorSnapshot.bootstrap=True`` path)
        # treats as not-yet-actionable. Skip attribution to avoid
        # baking placeholder noise into the EMA.
        if pe_value.bootstrap:
            return
        # De-dup by turn_index so retries / replay don't
        # double-attribute the same outcome.
        if (
            self._last_pe_turn_index is not None
            and pe_value.turn_index <= self._last_pe_turn_index
        ):
            return
        self._last_pe_turn_index = pe_value.turn_index

        if not self._last_active_weights:
            return

        signed_reward = float(pe_value.error.signed_reward)
        eta = _PE_HISTORY_LEARNING_RATE
        # Decay every protocol's EMA toward 0 each PE turn (so
        # protocols that haven't been active gradually forget),
        # then add the per-protocol contribution for those that
        # were active. Equivalent to ``ema ← (1-η)·ema + η·Δ``
        # where Δ = signed_reward·weight (and Δ=0 for inactive).
        for protocol_id in list(self._pe_utility.keys()):
            self._pe_utility[protocol_id] = (
                (1.0 - eta) * self._pe_utility[protocol_id]
            )
        for protocol_id, weight in self._last_active_weights.items():
            attributed = signed_reward * weight
            current = self._pe_utility.get(protocol_id, 0.0)
            updated = current + eta * attributed
            if updated > _PE_UTILITY_CLAMP:
                updated = _PE_UTILITY_CLAMP
            elif updated < -_PE_UTILITY_CLAMP:
                updated = -_PE_UTILITY_CLAMP
            self._pe_utility[protocol_id] = updated

    def apply_revision(
        self,
        proposal: ProtocolRevisionProposal,
        *,
        revised_by: str = "ProtocolReflectionEngine",
        revised_at_tick: int = 0,
    ) -> BehaviorProtocol:
        """Packet 3.3: apply an approved revision proposal.

        Delegates to :meth:`ProtocolRegistry.apply_revision` for
        the in-memory mutation, then re-runs the compile path
        against any injected application state stores so
        downstream owners see the new content.

        The recompile is best-effort: if any of the upserts
        fail, the registry mutation is *not* rolled back (the
        revision is committed). Reviewers / orchestrators are
        expected to handle partial failures by issuing another
        revision (or invoking :meth:`ProtocolRegistry.checkout_revision`
        once that lands).
        """

        revised = self._registry.apply_revision(
            proposal,
            revised_by=revised_by,
            revised_at_tick=revised_at_tick,
        )
        # Re-run compile so application owners see the new content.
        if (
            self._application_rare_heavy_state is None
            and self._domain_knowledge_store is None
            and self._case_memory_store is None
        ):
            return revised

        artifacts = compile_protocol_to_application_artifacts(revised)
        if (
            self._application_rare_heavy_state is not None
            and artifacts.boundary_prior_hints
        ):
            self._application_rare_heavy_state.upsert_boundary_prior_hints(
                artifacts.boundary_prior_hints
            )
        if (
            self._application_rare_heavy_state is not None
            and artifacts.playbook_rules
        ):
            self._application_rare_heavy_state.upsert_distilled_playbook_rules(
                artifacts.playbook_rules
            )
        if (
            self._domain_knowledge_store is not None
            and artifacts.domain_knowledge_records
        ):
            self._domain_knowledge_store.upsert_records(
                artifacts.domain_knowledge_records
            )
        if (
            self._case_memory_store is not None
            and artifacts.case_memory_records
        ):
            self._case_memory_store.upsert_records(
                artifacts.case_memory_records
            )
        return revised

    @property
    def pe_utility(self) -> Mapping[str, float]:
        """Read-only view of current per-protocol pe_utility EMA.

        Exposed for tests and audit. Mutation must go through
        ``_update_pe_history``.
        """

        return dict(self._pe_utility)

    @property
    def alpha(self) -> float:
        """Current α (context_match coefficient). Updated each PE turn."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Current β (pe_utility coefficient). Updated each PE turn."""
        return self._beta

    def _update_alpha_beta(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        """REINFORCE-style proxy gradient update for α / β.

        Runs after ``_update_pe_history`` and uses LAST turn's
        cached signal snapshots (``_last_context_scores`` /
        ``_last_pe_utilities``) plus the current turn's
        ``signed_reward``. Skips when:

        * No PE this turn (SHADOW-tolerant; same condition as
          ``_update_pe_history``).
        * Bootstrap PE (placeholder, not actionable).
        * Duplicate ``turn_index`` (replay / retry).
        * Fewer than 2 cached protocols (no differential signal
          to credit; cheng_laoshi-shape singleton mixtures).

        Update rule:

            α_grad = signed_reward × (max cm − min cm)
            β_grad = signed_reward × (max pe_utility − min pe_utility)
            α ← clamp(α + η_meta · α_grad, [α_min, α_max])
            β ← clamp(β + η_meta · β_grad, [α_min, α_max])

        Note: ``_update_pe_history`` already ran and may have
        updated ``self._last_pe_turn_index``. We re-validate the
        same conditions here independently — duplicate / bootstrap
        gating is cheap, and the explicit re-check makes the
        function safe to call standalone (e.g. in tests).
        """

        pe_snapshot = upstream.get("prediction_error")
        if pe_snapshot is None:
            return
        pe_value = pe_snapshot.value
        if not isinstance(pe_value, PredictionErrorSnapshot):
            return
        if pe_value.bootstrap:
            return
        # ``_update_pe_history`` advances ``_last_pe_turn_index`` on
        # successful attribution. Re-check is therefore inverted:
        # if last_pe_turn_index reflects a turn STRICTLY GREATER
        # than ours, we're a duplicate / out-of-order PE → skip.
        # When it equals ours, the pe_utility update DID run so we
        # should run too. (Owner calls _update_alpha_beta after
        # _update_pe_history, so the typical path is "_last_pe_turn_index
        # equals current pe.turn_index", meaning attribution just
        # happened on this turn.)
        if (
            self._last_pe_turn_index is not None
            and pe_value.turn_index < self._last_pe_turn_index
        ):
            return

        if len(self._last_context_scores) < 2:
            # Singleton or empty mixture last turn → no differential
            # signal to credit. (cheng_laoshi default-shape lands here.)
            return

        signed_reward = float(pe_value.error.signed_reward)
        cm_values = list(self._last_context_scores.values())
        pe_values = list(self._last_pe_utilities.values())
        cm_range = max(cm_values) - min(cm_values)
        pe_range = max(pe_values) - min(pe_values)

        if cm_range == 0.0 and pe_range == 0.0:
            # Last mixture had multiple protocols but zero signal
            # differentiation (e.g. all signals 0, all pe history 0).
            # No basis to credit either coefficient; skip to avoid
            # accumulating numeric noise.
            return

        eta_meta = _ALPHA_BETA_LEARNING_RATE
        alpha_grad = signed_reward * cm_range
        beta_grad = signed_reward * pe_range

        new_alpha = self._alpha + eta_meta * alpha_grad
        new_beta = self._beta + eta_meta * beta_grad
        if new_alpha < _ALPHA_BETA_MIN:
            new_alpha = _ALPHA_BETA_MIN
        elif new_alpha > _ALPHA_BETA_MAX:
            new_alpha = _ALPHA_BETA_MAX
        if new_beta < _ALPHA_BETA_MIN:
            new_beta = _ALPHA_BETA_MIN
        elif new_beta > _ALPHA_BETA_MAX:
            new_beta = _ALPHA_BETA_MAX

        self._alpha = new_alpha
        self._beta = new_beta


__all__ = ["FallbackActivationActiveError", "ProtocolRegistryModule"]
