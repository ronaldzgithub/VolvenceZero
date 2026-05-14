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

from dataclasses import dataclass
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
    ProtocolPhaseSnapshot,
    ProtocolRevisionProposal,
    ReviewStatus,
    StrategyWeightEntry,
)
from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
)
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.runtime.kernel import ContractViolationError

from volvence_zero.protocol_runtime.activation import (
    compute_active_mixture,
    is_fallback_mode,
)
from volvence_zero.protocol_runtime.compiler import (
    _protocol_rule_id,
    compile_protocol_to_application_artifacts,
    merge_protocol_chain,
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


# ---------------------------------------------------------------------------
# Per-strategy weight learning (``protocol-online-learning-active`` packet,
# sub-packet B1)
# ---------------------------------------------------------------------------
#
# Each ``StrategyPrior`` of every loaded protocol carries
# ``initial_weight``, ``pe_decay_rate``, ``pe_reinforce_rate`` and
# ``minimum_weight_floor``. Pre-this-packet those fields were
# explicitly dropped at compile time (see compiler.py docstring).
# This packet keeps the per-rule weights as runtime state on the
# owner so the prior's PE-rate fields actually drive in-session
# adaptation.
#
# Update rule (multiplicative, per active turn with non-bootstrap PE):
#
#   if signed_reward >= 0:
#       w' = w * (1 + reinforce_rate * |signed_reward|)
#   else:
#       w' = w * (1 - decay_rate * |signed_reward|)
#   w' = max(minimum_weight_floor, w')
#
# Attribution: only protocols with ``_last_active_weights[pid] > 0``
# get their per-rule weights updated this turn (PE outcome of last
# turn's action is attributed to the protocols that were active
# then). The factor is scaled by ``last_active_weight`` so a protocol
# that barely contributed only weakly steers its own rules.
#
# The reinforce/decay rates are bounded to [0, 1) at the contract
# layer; the multiplicative factor is therefore always in (0, 2)
# for any single-turn |signed_reward| <= 1.0. For larger PE
# magnitudes the factor still stays bounded by clamping to
# ``_STRATEGY_WEIGHT_MAX`` from above.
_STRATEGY_WEIGHT_MAX: float = 1_000.0


# ---------------------------------------------------------------------------
# Cross-session persistence (``protocol-online-learning-active`` packet,
# sub-packet C; schema-evolution guidance closed out by
# ``protocol-online-learning-followups`` packet, sub-packet F3)
# ---------------------------------------------------------------------------
#
# Owner identifier used both as the persistence key prefix in
# ``OwnerHydrationStore`` and as the contract sentinel inside
# ``hydrate_from_persistence`` (mismatched name → fail-loud
# ``HydrationOwnerMismatchError``). Stable across builds.
#
# ``schema_version`` is OWNER-internal. Current payload (v1) carries:
#
#   * ``alpha`` (float)            — global softmax-α coefficient
#   * ``beta`` (float)             — global softmax-β coefficient
#   * ``pe_utility`` (dict[str,
#         float])                  — per-protocol PE EMA
#   * ``strategy_weights`` (dict[
#         str, dict[str, float]])  — per-(protocol, raw rule_id)
#                                    learnt weight (B1)
#   * ``last_strategy_reward``
#         (dict[str, dict[str,
#         float]])                 — per-(protocol, raw rule_id)
#                                    last signed_reward mirror (B1)
#   * ``last_pe_turn_index``
#         (int | None)             — replay-dedup cursor
#
# Defensive filtering: the export path intersects
# ``strategy_weights`` / ``last_strategy_reward`` with the currently-
# loaded protocols' ``strategy_priors`` rule_ids (sub-packet F2).
# Stale rule_ids cannot leak into v1 payload.
#
# ---------------------------------------------------------------------------
# Schema-evolution policy (sub-packet F3)
# ---------------------------------------------------------------------------
#
# Wider repo convention (mirrored by ``SemanticStateStore`` /
# ``VitalsModule`` / ``FollowupManager``): each hydratable owner
# carries a single integer ``_*_SCHEMA_VERSION`` constant; the
# ``hydrate_from_persistence`` path requires byte-equality and
# raises ``HydrationVersionMismatchError`` otherwise. There is NO
# central migration registry, NO chained version-step framework.
# Mismatch is fail-loud by design — it forces operators to roll
# the build forward together with the persistence directory.
#
# Bumping this owner to a future ``vN+1`` is a 5-step procedure;
# perform ALL of them in the same change set:
#
# 1. Bump ``_PROTOCOL_REGISTRY_SCHEMA_VERSION`` to ``N + 1``.
# 2. Update ``export_persistence_snapshot`` to write the new
#    payload shape (add fields, change types, split entries).
#    Keep the new shape JSON-serialisable (the
#    ``OwnerHydrationStore`` backend is text-based).
# 3. In ``hydrate_from_persistence``, **before** the
#    ``snapshot.schema_version != _PROTOCOL_REGISTRY_SCHEMA_VERSION``
#    fail-loud check, branch on ``snapshot.schema_version == N``
#    and migrate the payload in-process::
#
#        if snapshot.schema_version == N:
#            payload = _migrate_vN_to_vNplus1(dict(snapshot.payload))
#            snapshot = OwnerPersistenceSnapshot(
#                owner_name=snapshot.owner_name,
#                schema_version=N + 1,
#                payload=payload,
#                description=snapshot.description,
#            )
#
#    The migrate helper is a pure function on dicts; defensively
#    copy the input. Older snapshots (``schema_version < N``)
#    that need multi-step upgrades chain through the same branch
#    series (``vN-1 → vN → vN+1``); two-or-more-version gaps that
#    cross a deliberate-break boundary should still fail-loud.
# 4. Add a contract test under ``tests/contracts/`` that constructs
#    a hand-built ``OwnerPersistenceSnapshot`` at the OLD
#    schema_version, calls ``hydrate_from_persistence``, then
#    asserts the resulting owner state is semantically equivalent
#    to what an in-place ``vN+1`` export would have produced.
#    Mirrors the round-trip + fail-loud tests in
#    ``tests/contracts/test_protocol_registry_module_hydration.py``.
# 5. Update the field list in this docstring AND the matching
#    ``docs/specs/owner-hydration.md`` table so future readers can
#    see the payload at a glance without grepping.
#
# Cross-process compatibility warning: in deployments where one
# process writes the snapshot and another reads it, run a
# co-deployment of build versions during the transition. The
# hydration path is intentionally not lenient — silently accepting
# an unknown schema_version would leak partially-correct state
# into the new owner and corrupt α / β learning trajectories.
_PROTOCOL_REGISTRY_OWNER_NAME: str = "protocol_registry"
_PROTOCOL_REGISTRY_SCHEMA_VERSION: int = 1


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
        "protocol_phase",
        # Packet 7.0: COMMITMENT_FULFILLED / COMMITMENT_BROKEN
        # detectors read the typed commitment snapshot.
        "commitment",
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

        # ``protocol-online-learning-active`` packet (sub-packet B1):
        # per-rule weight table, learning the per-strategy weight
        # encoded in ``StrategyPrior.initial_weight`` /
        # ``pe_decay_rate`` / ``pe_reinforce_rate`` /
        # ``minimum_weight_floor``. Outer key is ``protocol_id``,
        # inner key is ``rule_id``; value is the current weight.
        # Initialised on ``load_protocol`` from each prior's
        # ``initial_weight``. Updated on every PE turn by
        # ``_update_strategy_weights``: positive signed_reward ⇒
        # multiplicative reinforcement scaled by the prior's
        # ``pe_reinforce_rate``; negative signed_reward ⇒
        # multiplicative decay scaled by ``pe_decay_rate``; result
        # is clamped to ``minimum_weight_floor`` from below.
        # Published verbatim on
        # ``ActiveMixtureSnapshot.strategy_weights`` so downstream
        # ``StrategyPlaybookModule`` / planner can read the
        # PE-tuned weights when they want to. SHADOW for now: no
        # consumer reads the field yet (B2 follow-up packet).
        self._strategy_weights: dict[str, dict[str, float]] = {}
        # ``last_signed_reward`` per (protocol_id, rule_id) for
        # observability — surfaced verbatim on the snapshot entry
        # so external monitors can debug the learning step
        # without poking at the module's private state.
        self._last_strategy_reward: dict[str, dict[str, float]] = {}

    @property
    def registry(self) -> ProtocolRegistry:
        """Public handle for adapters (``FixtureUptake`` etc.) to load protocols.

        Adapters call ``module.registry.load(protocol)`` between
        turns. Reads (``loaded()`` / ``get(...)``) are safe
        concurrently with ``process()``.
        """

        return self._registry

    def load_protocol(
        self,
        protocol: BehaviorProtocol,
        *,
        load_context: "LoadContext | None" = None,
    ) -> None:
        """Register a protocol; if a rare-heavy state was injected,
        compile artifacts and apply to the application owners.

        Packet 6.6: ``load_context`` may carry a reviewer-supplied
        review level. If supplied, the engine asserts the level
        is sufficient for the protocol's content (compares against
        ``required_review_level`` derived from the protocol's
        boundary / identity content). Missing context defaults to
        the legacy "trust caller" path (used by FIXTURE loads and
        tests).

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

        if load_context is not None:
            required = _required_review_level_for_protocol(protocol)
            if not _review_level_sufficient(
                load_context.reviewer_level, required
            ):
                raise PermissionError(
                    f"protocol {protocol.protocol_id!r} requires "
                    f"reviewer_level={required.value}; got "
                    f"{load_context.reviewer_level.value} (reviewer_id="
                    f"{load_context.reviewer_id!r})"
                )
        self._registry.load(protocol)
        # ``protocol-online-learning-active`` packet (sub-packet B1):
        # seed per-rule weights from each StrategyPrior's
        # ``initial_weight``. Idempotent across re-loads of the same
        # protocol id (existing weights for known rule ids are
        # preserved so PE-driven learning state survives a hot-reload
        # of the same protocol; new rule ids get their initial_weight;
        # rule ids that disappeared get pruned). The
        # ``last_signed_reward`` mirror table is updated in lockstep.
        self._reseed_strategy_weights_for_protocol(protocol)
        # Short-circuit: no application stores injected → registry-only.
        if (
            self._application_rare_heavy_state is None
            and self._domain_knowledge_store is None
            and self._case_memory_store is None
        ):
            return
        try:
            # Packet 6.5: if this protocol declares a parent_protocol_id,
            # merge the chain into a flat protocol before compiling so
            # parent boundary / strategy / knowledge / case content is
            # also applied to application owners.
            compile_target = protocol
            if protocol.parent_protocol_id is not None:
                compile_target = merge_protocol_chain(
                    protocol, lookup=self._registry.get_optional
                )
            artifacts = compile_protocol_to_application_artifacts(compile_target)
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
        """Packet 6.9: full unload — drops registry entry and
        removes any protocol-prefixed entries from the injected
        application stores.

        Returns True if the protocol was loaded and removed, False
        if it wasn't loaded.

        Cleanup uses the lineage prefix shape established by the
        compile path:

        * Boundary prior hints: ``protocol:{protocol_id}:boundary:{...}``
        * Playbook rules: ``protocol:{protocol_id}:playbook:{...}``
        * Domain knowledge records: ``protocol:{protocol_id}:knowledge:{...}``
        * Case memory records: ``protocol:{protocol_id}:case:{...}``

        Other vertical-driven entries (no ``protocol:`` prefix) are
        untouched.
        """

        if protocol_id in self._applied_to_application_state:
            self._unload_protocol_artifacts(protocol_id)
            self._applied_to_application_state.discard(protocol_id)
        # ``protocol-online-learning-active`` packet (sub-packet B1):
        # drop per-rule weight + reward state for the unloaded
        # protocol so the snapshot doesn't carry stale entries.
        # Pre-emptive: also clear pe_utility / last_active_weights
        # entries so the next α/β / EMA pass cannot attribute
        # signal to a no-longer-loaded protocol.
        self._strategy_weights.pop(protocol_id, None)
        self._last_strategy_reward.pop(protocol_id, None)
        self._pe_utility.pop(protocol_id, None)
        self._last_active_weights.pop(protocol_id, None)
        return self._registry.unload(protocol_id)

    def _unload_protocol_artifacts(self, protocol_id: str) -> None:
        """Drop application-store entries with the given protocol prefix."""

        prefix = f"protocol:{protocol_id}:"
        if self._application_rare_heavy_state is not None:
            self._application_rare_heavy_state.remove_boundary_prior_hints_by_id_prefix(
                f"{prefix}boundary:"
            )
            self._application_rare_heavy_state.remove_distilled_playbook_rules_by_id_prefix(
                f"{prefix}playbook:"
            )
        if self._domain_knowledge_store is not None:
            self._domain_knowledge_store.remove_records_by_id_prefix(
                f"{prefix}knowledge:"
            )
        if self._case_memory_store is not None:
            self._case_memory_store.remove_records_by_id_prefix(
                f"{prefix}case:"
            )

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
        # ``protocol-online-learning-active`` packet (sub-packet B1):
        # per-rule weight learning runs on the same PE attribution
        # window as ``_update_pe_history`` (uses last turn's
        # ``_last_active_weights`` to credit the protocols that
        # were active when the action was taken). Must run BEFORE
        # ``_last_active_weights`` is overwritten with THIS turn's
        # mixture below.
        self._update_strategy_weights(upstream)

        loaded = self._registry.loaded()
        # Capture this turn's per-eligible-protocol context_match
        # scores via the audit out-param so we can use them in
        # NEXT turn's α/β learning (range(context_match) needs
        # them snapshot-aligned with the mixture they produced).
        audit_context: dict[str, float] = {}
        # Packet 5.0: read protocol_phase snapshot if present so
        # ActiveProtocolEntry.current_phase_id reflects the
        # PE-driven phase pointer maintained by ProtocolPhaseModule.
        # SHADOW-tolerant: missing snapshot → fall back to first
        # declared phase per protocol (cheng_laoshi default shape).
        phase_by_id = _read_phase_by_id(upstream)
        snapshot_value = compute_active_mixture(
            loaded_protocols=loaded,
            upstream=upstream,
            pe_utility_by_id=self._pe_utility,
            alpha=self._alpha,
            beta=self._beta,
            audit_context_scores=audit_context,
            phase_by_id=phase_by_id,
        )
        # ``protocol-online-learning-active`` packet (sub-packet B1):
        # attach the per-rule weight table to the snapshot. The
        # mixture computation in ``activation.py`` is owner-agnostic
        # and only knows about per-protocol weights, so we splice
        # the per-rule table in here using ``dataclasses.replace``
        # to keep the snapshot immutable.
        from dataclasses import replace as _replace
        snapshot_value = _replace(
            snapshot_value,
            strategy_weights=self._build_strategy_weight_entries(),
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

    # ------------------------------------------------------------------
    # Per-strategy weight learning (``protocol-online-learning-active``
    # packet, sub-packet B1)
    # ------------------------------------------------------------------

    def _reseed_strategy_weights_for_protocol(
        self, protocol: BehaviorProtocol
    ) -> None:
        """(Re)initialise per-rule weights for a freshly loaded protocol.

        Idempotent: weights for known rule ids are preserved (so a
        hot-reload of the same protocol id does not reset learning
        state); new rule ids get their ``initial_weight``; rule ids
        that disappeared (revision dropped them) get pruned.
        """

        rule_ids_in_protocol = {
            prior.rule_id for prior in protocol.strategy_priors
        }
        existing = self._strategy_weights.get(protocol.protocol_id, {})
        new_weights: dict[str, float] = {}
        new_rewards: dict[str, float] = {}
        existing_rewards = self._last_strategy_reward.get(
            protocol.protocol_id, {}
        )
        for prior in protocol.strategy_priors:
            if prior.rule_id in existing:
                # Preserve learned weight across re-loads (clamped at
                # the floor though, in case the prior's floor was
                # raised between revisions).
                preserved = max(
                    existing[prior.rule_id], prior.minimum_weight_floor
                )
                new_weights[prior.rule_id] = preserved
            else:
                new_weights[prior.rule_id] = max(
                    prior.initial_weight, prior.minimum_weight_floor
                )
            new_rewards[prior.rule_id] = existing_rewards.get(
                prior.rule_id, 0.0
            )
        # Prune rule ids that no longer exist in the protocol.
        del rule_ids_in_protocol  # used implicitly via `prior.rule_id`
        self._strategy_weights[protocol.protocol_id] = new_weights
        self._last_strategy_reward[protocol.protocol_id] = new_rewards

    def _update_strategy_weights(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> None:
        """Multiplicatively reinforce / decay per-rule weights from PE.

        Same attribution window as ``_update_pe_history``: PE at
        turn t reflects last turn's action; only protocols that
        had non-zero ``_last_active_weights[pid]`` participate.
        SHADOW-tolerant on missing PE / bootstrap PE / duplicate
        turn_index (returns silently — same skip rules as the
        sibling EMA path).
        """

        pe_snapshot = upstream.get("prediction_error")
        if pe_snapshot is None:
            return
        pe_value = pe_snapshot.value
        if not isinstance(pe_value, PredictionErrorSnapshot):
            return
        if pe_value.bootstrap:
            return
        # NOTE: we share the de-dup state with ``_update_pe_history``
        # which runs immediately before us. ``_update_pe_history``
        # has already advanced ``_last_pe_turn_index`` for this
        # turn_index, so we cannot use the same ``<=`` check here.
        # Instead we read ``_last_active_weights``: it is empty on
        # cold start, and any duplicate replay would have skipped
        # the EMA path anyway (so on duplicate the weights here
        # already reflect this turn's update). Net effect: replay
        # is a no-op here too.
        if not self._last_active_weights:
            return

        signed_reward = float(pe_value.error.signed_reward)
        magnitude = abs(signed_reward)
        if magnitude == 0.0:
            return

        for protocol in self._registry.loaded():
            pid = protocol.protocol_id
            last_weight = self._last_active_weights.get(pid, 0.0)
            if last_weight <= 0.0:
                continue
            rule_table = self._strategy_weights.get(pid)
            reward_table = self._last_strategy_reward.setdefault(pid, {})
            if rule_table is None:
                continue
            for prior in protocol.strategy_priors:
                if prior.rule_id not in rule_table:
                    # Defensive: a freshly added rule we haven't seeded
                    # yet (e.g. mid-session apply_revision before the
                    # caller re-ran load_protocol). Seed it lazily
                    # rather than dropping the update.
                    rule_table[prior.rule_id] = max(
                        prior.initial_weight, prior.minimum_weight_floor
                    )
                current = rule_table[prior.rule_id]
                # Scale the multiplicative factor by the protocol's
                # last-turn weight so a barely-active protocol only
                # weakly steers its own rules. ``magnitude`` is in
                # [0, +inf); typical PE magnitudes from the kernel
                # PE module sit in [0, ~2].
                attributed = magnitude * last_weight
                if signed_reward >= 0.0:
                    factor = 1.0 + prior.pe_reinforce_rate * attributed
                else:
                    factor = 1.0 - prior.pe_decay_rate * attributed
                # Negative factor would flip the sign of the weight;
                # that's pathological. Clamp factor at 0 so an
                # extreme decay step takes the weight to the floor
                # rather than below it.
                if factor < 0.0:
                    factor = 0.0
                updated = current * factor
                if updated < prior.minimum_weight_floor:
                    updated = prior.minimum_weight_floor
                if updated > _STRATEGY_WEIGHT_MAX:
                    updated = _STRATEGY_WEIGHT_MAX
                rule_table[prior.rule_id] = updated
                reward_table[prior.rule_id] = signed_reward

    def _filter_to_loaded_priors(
        self,
        per_protocol: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Return a copy of ``per_protocol`` keeping only (pid, rule_id)
        pairs that still appear on a currently-loaded protocol's
        ``strategy_priors``.

        Used by both :meth:`export_persistence_snapshot` and the
        snapshot-build path so internal stale entries (which
        ``apply_revision`` + ``_reseed_strategy_weights_for_protocol``
        should already have GC'd) cannot leak out to either
        persistence or downstream consumers. Pure / read-only on the
        owner state; the caller decides whether to assign the
        result back over the internal dicts.
        """

        loaded_priors_by_pid: dict[str, set[str]] = {
            p.protocol_id: {prior.rule_id for prior in p.strategy_priors}
            for p in self._registry.loaded_all()
        }
        filtered: dict[str, dict[str, float]] = {}
        for pid, rule_table in per_protocol.items():
            valid_rule_ids = loaded_priors_by_pid.get(pid, set())
            if not valid_rule_ids:
                continue
            kept = {
                rid: value
                for rid, value in rule_table.items()
                if rid in valid_rule_ids
            }
            if kept:
                filtered[pid] = kept
        return filtered

    def _build_strategy_weight_entries(
        self,
    ) -> tuple[StrategyWeightEntry, ...]:
        """Project the current ``_strategy_weights`` table onto the
        snapshot's read-only entry shape.

        Sorted by ``(protocol_id, rule_id)`` so the snapshot is
        deterministic across runs (matches the ordering
        convention used elsewhere in this module).

        ``compiled_rule_id`` (follow-up packet ``protocol-online-learning-followups``,
        sub-packet F1) is filled via :func:`_protocol_rule_id`, the
        same helper used by ``compile_protocol_to_application_artifacts``
        so the value is byte-identical with the corresponding
        ``PlaybookRule.rule_id`` written into the application stores.
        Downstream consumers (``StrategyPlaybookModule``) join on
        this field — no reverse-parsing of the namespace prefix.

        Defensive filtering (sub-packet F2): only entries whose
        ``rule_id`` still appears on a currently-loaded protocol's
        ``strategy_priors`` are emitted. This keeps stale rule_ids
        out of the published snapshot even if some path forgot to
        call ``_reseed_strategy_weights_for_protocol`` after a
        revision dropped a strategy. ``apply_revision`` and
        ``load_protocol`` both already call the reseed helper; the
        filter is belt-and-braces.
        """

        loaded_priors_by_pid: dict[str, set[str]] = {
            p.protocol_id: {prior.rule_id for prior in p.strategy_priors}
            for p in self._registry.loaded_all()
        }
        entries: list[StrategyWeightEntry] = []
        for pid in sorted(self._strategy_weights):
            valid_rule_ids = loaded_priors_by_pid.get(pid, set())
            rule_table = self._strategy_weights[pid]
            reward_table = self._last_strategy_reward.get(pid, {})
            for rid in sorted(rule_table):
                if rid not in valid_rule_ids:
                    continue
                entries.append(
                    StrategyWeightEntry(
                        protocol_id=pid,
                        rule_id=rid,
                        weight=rule_table[rid],
                        last_signed_reward=reward_table.get(rid, 0.0),
                        compiled_rule_id=_protocol_rule_id(pid, rid),
                    )
                )
        return tuple(entries)

    # ------------------------------------------------------------------
    # Cross-session persistence (``protocol-online-learning-active``
    # packet, sub-packet C)
    # ------------------------------------------------------------------
    #
    # Implements the duck-typed ``HydratableOwnerProtocol`` so the
    # owner participates in ``OwnerHydrationStore``. The persisted
    # state covers everything that drives in-session adaptation:
    #
    # * ``_alpha`` / ``_beta`` — global softmax learning coefficients
    # * ``_pe_utility`` — per-protocol EMA of attributed PE
    # * ``_strategy_weights`` — per-rule weights (B1)
    # * ``_last_strategy_reward`` — per-rule last signed_reward mirror
    # * ``_last_pe_turn_index`` — replay-dedup cursor
    #
    # The protocol REGISTRY itself is NOT persisted here. Loaded
    # protocols come from ``LifeformConfig.seed_protocols`` (which
    # the SessionManager re-supplies from
    # ``ProtocolUptakeService.loaded_approved_snapshot()`` on each
    # session creation), so reconstructing the registry is the
    # caller's job. Persisting the registry too would create a
    # second source of truth for "which protocols are loaded",
    # violating R8.

    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot:
        """Dump per-session learning state to a versioned, JSON-friendly
        ``OwnerPersistenceSnapshot``.

        Read-only on the owner state. Round-trip stable: calling
        ``export → hydrate → export`` on a fresh module yields the
        same payload.
        """

        return OwnerPersistenceSnapshot(
            owner_name=_PROTOCOL_REGISTRY_OWNER_NAME,
            schema_version=_PROTOCOL_REGISTRY_SCHEMA_VERSION,
            payload={
                "alpha": float(self._alpha),
                "beta": float(self._beta),
                "pe_utility": dict(self._pe_utility),
                # Sub-packet F2 defensive filtering: only persist
                # rule_ids that still appear on a currently-loaded
                # protocol's strategy_priors. Mirrors the same
                # intersection ``_build_strategy_weight_entries``
                # applies to the published snapshot, and the same
                # intersection ``hydrate_from_persistence`` applies
                # on read. End result: stale rule_ids that survived
                # an apply_revision (or were left over from earlier
                # builds) cannot leak into the cross-session
                # persistence payload.
                "strategy_weights": self._filter_to_loaded_priors(
                    self._strategy_weights
                ),
                "last_strategy_reward": self._filter_to_loaded_priors(
                    self._last_strategy_reward
                ),
                "last_pe_turn_index": self._last_pe_turn_index,
            },
            description=(
                "ProtocolRegistryModule online learning state "
                "(alpha/beta + pe_utility + strategy_weights)."
            ),
        )

    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None:
        """Restore per-session learning state from a previously-exported
        snapshot.

        Idempotent (applying the same snapshot twice yields the
        same module state). Fail-loud on owner mismatch / unknown
        schema version / missing required keys, per the
        :class:`HydratableOwnerProtocol` contract.

        Strategy-weight payloads are intersected with the currently-
        loaded protocols: a rule id that exists in the snapshot
        but not in the current ``BehaviorProtocol.strategy_priors``
        is dropped (the protocol may have been revised between
        sessions); a rule id present in the protocol but not in
        the snapshot is left at its just-seeded ``initial_weight``.
        This keeps hydration safe across protocol-revision
        boundaries without silently re-introducing weights for
        rules that no longer exist.
        """

        if snapshot.owner_name != _PROTOCOL_REGISTRY_OWNER_NAME:
            raise HydrationOwnerMismatchError(
                f"ProtocolRegistryModule.hydrate_from_persistence: "
                f"owner_name mismatch — expected "
                f"{_PROTOCOL_REGISTRY_OWNER_NAME!r}, got "
                f"{snapshot.owner_name!r}"
            )
        if snapshot.schema_version != _PROTOCOL_REGISTRY_SCHEMA_VERSION:
            raise HydrationVersionMismatchError(
                f"ProtocolRegistryModule.hydrate_from_persistence: "
                f"unknown schema_version={snapshot.schema_version!r}; "
                f"this build only knows version "
                f"{_PROTOCOL_REGISTRY_SCHEMA_VERSION}."
            )
        payload = snapshot.payload
        try:
            alpha = float(payload["alpha"])
            beta = float(payload["beta"])
            pe_utility_blob = payload["pe_utility"]
            strategy_weights_blob = payload["strategy_weights"]
            last_strategy_reward_blob = payload["last_strategy_reward"]
            last_pe_turn_index = payload["last_pe_turn_index"]
        except KeyError as exc:
            raise HydrationPayloadInvalidError(
                f"ProtocolRegistryModule.hydrate_from_persistence: "
                f"missing required key {exc.args[0]!r} in payload"
            ) from exc

        # Clamp α / β to the allowed band on rehydrate so a
        # malformed snapshot can't poison the owner. Same band
        # ``_update_alpha_beta`` enforces.
        if alpha < _ALPHA_BETA_MIN:
            alpha = _ALPHA_BETA_MIN
        elif alpha > _ALPHA_BETA_MAX:
            alpha = _ALPHA_BETA_MAX
        if beta < _ALPHA_BETA_MIN:
            beta = _ALPHA_BETA_MIN
        elif beta > _ALPHA_BETA_MAX:
            beta = _ALPHA_BETA_MAX
        self._alpha = alpha
        self._beta = beta

        # PE utility: keep entries for currently-loaded protocols
        # only, and clamp to the EMA band.
        loaded_ids = {p.protocol_id for p in self._registry.loaded_all()}
        pe_utility: dict[str, float] = {}
        for pid, value in pe_utility_blob.items():
            if pid not in loaded_ids:
                continue
            v = float(value)
            if v > _PE_UTILITY_CLAMP:
                v = _PE_UTILITY_CLAMP
            elif v < -_PE_UTILITY_CLAMP:
                v = -_PE_UTILITY_CLAMP
            pe_utility[pid] = v
        self._pe_utility = pe_utility

        # Strategy weights: for each loaded protocol, intersect the
        # snapshot's rule ids with the protocol's current rule ids;
        # honour each prior's minimum_weight_floor.
        new_weights: dict[str, dict[str, float]] = {}
        new_rewards: dict[str, dict[str, float]] = {}
        for protocol in self._registry.loaded_all():
            pid = protocol.protocol_id
            persisted_rules = strategy_weights_blob.get(pid, {})
            persisted_rewards = last_strategy_reward_blob.get(pid, {})
            rule_table: dict[str, float] = {}
            reward_table: dict[str, float] = {}
            for prior in protocol.strategy_priors:
                if prior.rule_id in persisted_rules:
                    w = float(persisted_rules[prior.rule_id])
                    if w < prior.minimum_weight_floor:
                        w = prior.minimum_weight_floor
                    if w > _STRATEGY_WEIGHT_MAX:
                        w = _STRATEGY_WEIGHT_MAX
                    rule_table[prior.rule_id] = w
                else:
                    # Rule wasn't in the snapshot: leave the
                    # currently-seeded initial value (set by
                    # _reseed_strategy_weights_for_protocol on the
                    # earlier load_protocol).
                    existing = self._strategy_weights.get(pid, {}).get(
                        prior.rule_id,
                        max(prior.initial_weight, prior.minimum_weight_floor),
                    )
                    rule_table[prior.rule_id] = existing
                reward_table[prior.rule_id] = float(
                    persisted_rewards.get(prior.rule_id, 0.0)
                )
            new_weights[pid] = rule_table
            new_rewards[pid] = reward_table
        self._strategy_weights = new_weights
        self._last_strategy_reward = new_rewards

        self._last_pe_turn_index = (
            int(last_pe_turn_index)
            if last_pe_turn_index is not None
            else None
        )

    def checkout_revision(
        self,
        protocol_id: str,
        revision_id: str | None = None,
    ) -> BehaviorProtocol:
        """Packet 6.3: roll a protocol back to an earlier revision.

        Delegates to :meth:`ProtocolRegistry.checkout_revision` for
        the in-memory state restore; then re-runs the compile path
        against any injected application stores so the application
        owners see the restored content (overwrite semantics —
        same-id entries replace; entries that no longer exist in
        the rolled-back state may persist as orphans until packet
        6.9 adds full unload).
        """

        restored = self._registry.checkout_revision(
            protocol_id, revision_id
        )
        if (
            self._application_rare_heavy_state is None
            and self._domain_knowledge_store is None
            and self._case_memory_store is None
        ):
            return restored

        artifacts = compile_protocol_to_application_artifacts(restored)
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
        return restored

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
        # ``protocol-online-learning-followups`` packet, sub-packet F2:
        # re-sync per-rule strategy weight state with the revised
        # protocol's strategy_priors. ``_reseed_strategy_weights_for_protocol``
        # preserves learnt weights for unchanged rule_ids, seeds new
        # rule_ids at their initial_weight, and prunes weights for
        # rule_ids that the revision removed. Mirrors what
        # ``load_protocol`` already does for fresh loads. Called
        # unconditionally (independent of injected stores) because
        # the weight table is a per-owner internal state, not an
        # application-store side effect.
        self._reseed_strategy_weights_for_protocol(revised)
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


@dataclass(frozen=True)
class LoadContext:
    """Packet 6.6: reviewer authority context for protocol load.

    Supplied by the caller when loading non-trivial protocols
    (e.g. PDF-extracted candidates, API-injected protocols).
    The engine compares ``reviewer_level`` against the protocol's
    derived ``required_review_level`` and raises ``PermissionError``
    if insufficient.
    """

    reviewer_id: str
    reviewer_level: "ReviewLevel"
    note: str = ""


def _required_review_level_for_protocol(
    protocol: BehaviorProtocol,
) -> "ReviewLevel":
    """Mirror of ``review.required_review_level`` for a fully-built
    protocol (post-approval). Used by ``LoadContext`` enforcement.
    """

    from volvence_zero.behavior_protocol import (
        BoundarySeverity,
        ReviewLevel,
    )

    if any(
        b.severity is BoundarySeverity.HARD_BLOCK
        or b.severity is BoundarySeverity.ESCALATE_HUMAN
        for b in protocol.boundary_contracts
    ):
        return ReviewLevel.L4
    if (
        protocol.identity_assertion.requires_self_traits
        or protocol.identity_assertion.forbidden_self_traits
    ):
        return ReviewLevel.L3
    if any(
        b.severity is BoundarySeverity.SOFT_REMIND
        for b in protocol.boundary_contracts
    ):
        return ReviewLevel.L2
    return ReviewLevel.L1


def _review_level_sufficient(
    actual: "ReviewLevel", required: "ReviewLevel"
) -> bool:
    """Compare two ReviewLevel enums.

    ``L4`` ≥ ``L3`` ≥ ``L2`` ≥ ``L1`` (numerically reversed so
    ``L4`` is the highest authority).
    """

    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    return order[actual.name] >= order[required.name]


def _read_phase_by_id(
    upstream: Mapping[str, Snapshot[Any]],
) -> dict[str, str] | None:
    """Read ``protocol_phase`` upstream into a flat dict.

    Returns ``None`` when the slot is absent (packet 5.0 SHADOW
    pre-rollout / tests that bypass the runtime). Returns an
    empty dict when the slot is present but has no entries
    (packet 5.0 first turn before any protocol is tracked).
    """

    snapshot = upstream.get("protocol_phase")
    if snapshot is None:
        return None
    if not isinstance(snapshot.value, ProtocolPhaseSnapshot):
        return None
    return {
        protocol_id: phase_id
        for protocol_id, phase_id in snapshot.value.phase_by_protocol_id
    }


__all__ = [
    "FallbackActivationActiveError",
    "LoadContext",
    "ProtocolRegistryModule",
]
