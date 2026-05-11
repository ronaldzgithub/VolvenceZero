"""In-memory store for loaded BehaviorProtocols (packet 1.0).

Per-module store backing ``ProtocolRegistryModule``. Does NOT
publish snapshots itself; the module reads ``loaded()`` each turn and
hands the tuple to ``compute_active_mixture``.

Lifecycle ops are synchronous mutations on the registry (``load`` /
``unload`` / ``mark_status``). Adapters in ``lifeform-domain-*``
wheels (and future ``lifeform-protocol-runtime``) call ``load(...)``
to register protocols; nothing else mutates the registry.

Packet 1.0 keeps this minimal:

* No revision_log mutation API (``ProtocolRevision`` records are
  declared in vz-contracts but no PE-driven writeback exists yet).
* No persistence; in-memory only — module rebuild = empty registry.
* No cross-session sharing; one ``ProtocolRegistryModule`` instance
  per lifeform.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import replace as _replace
from threading import RLock

from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    ProtocolRevision,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewStatus,
    StrategyPrior,
    StrategyPriorRevision,
)


class ProtocolRegistry:
    """Mutable, threadsafe in-memory registry of loaded BehaviorProtocols.

    Holds the canonical protocol identity → ``BehaviorProtocol``
    mapping. Snapshot publication and activation weighting are NOT
    its concern; ``ProtocolRegistryModule.process`` reads
    ``loaded()`` and constructs the snapshot via
    ``compute_active_mixture``.

    Threadsafety: an ``RLock`` guards the dict so that adapter calls
    (``load_protocol`` from session-side code) cannot race with the
    kernel ``process()`` call. Reads return tuples (immutable
    snapshots of the dict at read-time) so consumers don't see
    in-flight mutations.
    """

    # Packet 6.7: hard upper bound on protocols counted as "active"
    # (review_status in {SHADOW, ACTIVE}). DRAFT and RETIRED do not
    # count. Loading an additional protocol when the count is already
    # at the cap raises ProtocolLimitExceededError. This protects
    # softmax-mixed activation from getting smeared across too many
    # protocols (Open Q8 resolve).
    ACTIVE_PROTOCOL_HARD_CAP: int = 8

    def __init__(self) -> None:
        self._lock = RLock()
        self._loaded: dict[str, BehaviorProtocol] = {}
        # Packet 6.3: per-protocol revision snapshots for content
        # rollback. Each entry is a list of (revision_id, post-apply-state)
        # tuples in chronological order. The first entry is
        # ("initial:<protocol_id>", original loaded state). Later
        # revisions append. ``checkout_revision`` truncates back to
        # any prior point.
        self._revision_snapshots: dict[
            str, list[tuple[str, BehaviorProtocol]]
        ] = {}

    def load(self, protocol: BehaviorProtocol) -> None:
        """Register a protocol. Replaces an existing entry of the same id.

        Replacement is idempotent for FixtureUptake (same protocol
        loaded twice = no-op semantically). Future packets will
        gate replacement behind ``ModificationGate`` + version
        bookkeeping; packet 1.0 trusts the caller.
        """

        if not isinstance(protocol, BehaviorProtocol):
            raise TypeError(
                f"ProtocolRegistry.load expects BehaviorProtocol, got "
                f"{type(protocol).__name__}"
            )
        with self._lock:
            # Packet 6.7: enforce active-protocol cap (counts SHADOW
            # + ACTIVE; DRAFT and RETIRED don't count). Re-loading
            # an already-loaded protocol id is allowed (overwrite).
            if protocol.protocol_id not in self._loaded:
                active_count = sum(
                    1
                    for p in self._loaded.values()
                    if p.review_status in {ReviewStatus.SHADOW, ReviewStatus.ACTIVE}
                )
                this_active = protocol.review_status in {
                    ReviewStatus.SHADOW,
                    ReviewStatus.ACTIVE,
                }
                if (
                    this_active
                    and active_count >= self.ACTIVE_PROTOCOL_HARD_CAP
                ):
                    raise ProtocolLimitExceededError(
                        f"cannot load active protocol "
                        f"{protocol.protocol_id!r}: "
                        f"already at hard cap of "
                        f"{self.ACTIVE_PROTOCOL_HARD_CAP} active protocols. "
                        "Retire / unload one first."
                    )
            self._loaded[protocol.protocol_id] = protocol
            # Packet 6.3: stamp initial snapshot for rollback baseline.
            initial_id = f"initial:{protocol.protocol_id}"
            self._revision_snapshots[protocol.protocol_id] = [
                (initial_id, protocol)
            ]

    def unload(self, protocol_id: str) -> bool:
        """Remove a protocol; return True if it was present."""

        with self._lock:
            return self._loaded.pop(protocol_id, None) is not None

    def mark_status(self, protocol_id: str, status: ReviewStatus) -> None:
        """Replace a loaded protocol with one carrying a new review status.

        ``BehaviorProtocol`` is frozen; transitioning lifecycle state
        means swapping the dataclass instance. Packet 1.0 callers are
        expected to use this directly; later packets will route through
        ``ModificationGate``.
        """

        with self._lock:
            existing = self._loaded.get(protocol_id)
            if existing is None:
                raise KeyError(
                    f"ProtocolRegistry: no loaded protocol with id "
                    f"{protocol_id!r}"
                )
            self._loaded[protocol_id] = _replace(
                existing, review_status=status
            )

    def get(self, protocol_id: str) -> BehaviorProtocol | None:
        with self._lock:
            return self._loaded.get(protocol_id)

    def get_optional(self, protocol_id: str) -> BehaviorProtocol | None:
        """Alias for :meth:`get` (explicit None-tolerant signature).

        Used by ``merge_protocol_chain`` to look up parents without
        forcing the caller to handle KeyError vs None.
        """
        with self._lock:
            return self._loaded.get(protocol_id)

    def apply_revision(
        self,
        proposal: ProtocolRevisionProposal,
        *,
        revised_by: str = "ProtocolReflectionEngine",
        revised_at_tick: int = 0,
    ) -> BehaviorProtocol:
        """Packet 3.3: apply a reviewed revision proposal.

        Mutation rules (per change_kind):

        * ``WEIGHT_DECAY`` on ``STRATEGY_PRIOR``: multiply each
          ``initial_weight`` by ``proposed_payload['weight_multiplier']``
          (defaults to 0.5). When ``target_entry_id`` matches a
          specific ``rule_id``, only that strategy decays; when
          it matches the ``protocol_id`` (the "protocol-granular"
          path used by :func:`propose_strategy_decay`), all
          strategies decay.
        * ``DEACTIVATE`` on ``STRATEGY_PRIOR``: set
          ``initial_weight=0`` for the matched strategies.
        * ``ARCHIVE`` on ``KNOWLEDGE_SEED`` / ``SIGNATURE_CASE``:
          remove the targeted entry from the protocol's
          collection.
        * Any other combination raises NotImplementedError —
          keeps the surface narrow until a specific rule needs
          it (failing loud beats silent no-op).

        Side effects:

        * Appends a :class:`ProtocolRevision` entry to
          ``revision_log``.
        * Replaces the registry slot for the protocol with the
          new (frozen) ``BehaviorProtocol`` instance.

        Returns the new protocol instance. Caller is expected to
        re-run the compile path against application owners
        (handled at the owner level for clean injection).
        """

        with self._lock:
            existing = self._loaded.get(proposal.target_protocol_id)
            if existing is None:
                raise KeyError(
                    f"ProtocolRegistry.apply_revision: no loaded "
                    f"protocol with id {proposal.target_protocol_id!r}"
                )

            mutated = _apply_change(existing, proposal)

            revision = ProtocolRevision(
                revision_id=(
                    f"revision:{proposal.proposal_id}:"
                    f"{_now_iso()}"
                ),
                revised_at_tick=revised_at_tick,
                revised_by=revised_by,
                description=(
                    f"{proposal.change_kind.value} on "
                    f"{proposal.target_field.value} entry "
                    f"{proposal.target_entry_id!r}: "
                    f"{proposal.evidence.summary}"
                ),
                affected_field=proposal.target_field.value,
            )
            mutated = _replace(
                mutated,
                revision_log=mutated.revision_log + (revision,),
            )
            self._loaded[proposal.target_protocol_id] = mutated
            # Packet 6.3: stamp post-apply snapshot for rollback.
            history = self._revision_snapshots.setdefault(
                proposal.target_protocol_id,
                [(f"initial:{proposal.target_protocol_id}", existing)],
            )
            history.append((revision.revision_id, mutated))
            return mutated

    def checkout_revision(
        self,
        protocol_id: str,
        revision_id: str | None = None,
    ) -> BehaviorProtocol:
        """R15 rollback: restore a protocol from its revision history.

        ``revision_id=None`` rolls back to the **initial** loaded
        state (before any revision was applied); a specific
        ``revision_id`` restores the state that existed *immediately
        after* that revision was applied.

        Packet 6.3 ships full content rollback by replaying from
        per-revision content snapshots stored at apply time
        (see ``_revision_snapshots``). After rollback the registry
        entry is replaced and the revision_log is truncated to
        match the restored state. The caller is expected to
        re-run the compile path against application owners to
        propagate the restored content; ``ProtocolRegistryModule.checkout_revision``
        does this automatically.
        """

        with self._lock:
            existing = self._loaded.get(protocol_id)
            if existing is None:
                raise KeyError(
                    f"ProtocolRegistry.checkout_revision: no loaded "
                    f"protocol with id {protocol_id!r}"
                )
            history = self._revision_snapshots.get(protocol_id) or []
            if not history:
                raise ValueError(
                    f"protocol {protocol_id!r} has no revision snapshots"
                )

            if revision_id is None:
                # Restore initial loaded state (first entry).
                target_idx = 0
            else:
                target_idx = next(
                    (
                        i for i, (rid, _) in enumerate(history)
                        if rid == revision_id
                    ),
                    None,
                )
                if target_idx is None:
                    raise KeyError(
                        f"protocol {protocol_id!r} has no revision "
                        f"with id {revision_id!r}"
                    )

            target_id, target_state = history[target_idx]
            # Replace the live protocol with the target state.
            self._loaded[protocol_id] = target_state
            # Truncate history past the target so further rollbacks
            # don't see undone revisions.
            self._revision_snapshots[protocol_id] = history[: target_idx + 1]
            return target_state

    def loaded(self) -> tuple[BehaviorProtocol, ...]:
        """Return active (non-RETIRED) protocols as an immutable tuple.

        Ordering is by ``protocol_id`` so snapshots are
        deterministic across runs (important for stable
        ``revision_fingerprint``).

        Packet 6.4: protocols with ``review_status == RETIRED`` are
        filtered out — they no longer participate in the active
        mixture nor the compile path. Use :meth:`loaded_all` to
        get the full set including RETIRED (audit / rollback paths).
        """

        with self._lock:
            return tuple(
                self._loaded[pid] for pid in sorted(self._loaded)
                if self._loaded[pid].review_status is not ReviewStatus.RETIRED
            )

    def loaded_all(self) -> tuple[BehaviorProtocol, ...]:
        """Return ALL loaded protocols including RETIRED ones.

        Used by audit / R15 rollback / queue introspection.
        """

        with self._lock:
            return tuple(
                self._loaded[pid] for pid in sorted(self._loaded)
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._loaded)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class ProtocolLimitExceededError(RuntimeError):
    """Raised when loading a protocol would exceed
    :attr:`ProtocolRegistry.ACTIVE_PROTOCOL_HARD_CAP`."""


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _build_strategy_from_payload(
    payload: dict, target_entry_id: str
) -> StrategyPrior:
    """Construct a ``StrategyPrior`` from a proposal payload (packet 5.2).

    Required keys: ``problem_pattern``, ``recommended_ordering``,
    ``recommended_pacing``. Optional keys mirror StrategyPrior
    defaults. ``target_entry_id`` is used as the new rule_id when
    the payload doesn't supply one — keeps the proposal
    self-identifying.
    """

    rule_id = (payload.get("rule_id") or target_entry_id).strip()
    if not rule_id:
        raise ValueError(
            "ADD_STRATEGY payload must yield a non-empty rule_id "
            "(via 'rule_id' key or target_entry_id)"
        )
    problem_pattern = (payload.get("problem_pattern") or "").strip()
    if not problem_pattern:
        raise ValueError(
            f"ADD_STRATEGY payload for rule_id={rule_id!r} must have "
            "'problem_pattern'"
        )
    ordering = tuple(
        s for s in (payload.get("recommended_ordering") or ())
        if isinstance(s, str) and s.strip()
    )
    if not ordering:
        raise ValueError(
            f"ADD_STRATEGY payload for rule_id={rule_id!r} must have "
            "non-empty 'recommended_ordering'"
        )
    pacing = (payload.get("recommended_pacing") or "moderate").strip() or "moderate"

    return StrategyPrior(
        rule_id=rule_id,
        problem_pattern=problem_pattern,
        recommended_ordering=ordering,
        recommended_pacing=pacing,
        avoid_patterns=tuple(
            p for p in (payload.get("avoid_patterns") or ())
            if isinstance(p, str)
        ),
        applicability_phase=tuple(
            p for p in (payload.get("applicability_phase") or ())
            if isinstance(p, str)
        ),
        recommended_regime=payload.get("recommended_regime"),
        knowledge_weight_hint=float(
            payload.get("knowledge_weight_hint", 0.45)
        ),
        experience_weight_hint=float(
            payload.get("experience_weight_hint", 0.65)
        ),
        initial_weight=float(payload.get("initial_weight", 1.0)),
        confidence=float(payload.get("confidence", 0.7)),
        description=(payload.get("description") or "").strip(),
    )


def _apply_change(
    protocol: BehaviorProtocol,
    proposal: ProtocolRevisionProposal,
) -> BehaviorProtocol:
    """Pure mutation: produce a new BehaviorProtocol per the change kind."""

    field = proposal.target_field
    kind = proposal.change_kind
    target_entry_id = proposal.target_entry_id

    if field is ProtocolRevisionTargetField.STRATEGY_PRIOR:
        if kind is ProtocolRevisionChangeKind.ADD_STRATEGY:
            payload = proposal.proposed_payload or {}
            new_prior = _build_strategy_from_payload(payload, target_entry_id)
            # Skip if same rule_id already exists (idempotent).
            if any(s.rule_id == new_prior.rule_id for s in protocol.strategy_priors):
                return protocol
            return _replace(
                protocol,
                strategy_priors=protocol.strategy_priors + (new_prior,),
            )
        if kind in (
            ProtocolRevisionChangeKind.WEIGHT_DECAY,
            ProtocolRevisionChangeKind.DEACTIVATE,
            ProtocolRevisionChangeKind.WEIGHT_REINFORCE,
        ):
            payload = proposal.proposed_payload or {}
            if kind is ProtocolRevisionChangeKind.DEACTIVATE:
                multiplier = 0.0
            elif kind is ProtocolRevisionChangeKind.WEIGHT_REINFORCE:
                multiplier = float(payload.get("weight_multiplier", 1.5))
            else:
                multiplier = float(payload.get("weight_multiplier", 0.5))
            new_strategies = []
            for strategy in protocol.strategy_priors:
                if (
                    target_entry_id == protocol.protocol_id
                    or target_entry_id == strategy.rule_id
                ):
                    new_weight = strategy.initial_weight * multiplier
                    # Clamp to schema-valid range [0, 1].
                    if new_weight > 1.0:
                        new_weight = 1.0
                    elif new_weight < 0.0:
                        new_weight = 0.0
                    history_entry = StrategyPriorRevision(
                        revision_id=f"{proposal.proposal_id}:{strategy.rule_id}",
                        revised_at_tick=0,
                        delta=new_weight - strategy.initial_weight,
                        reason=(
                            f"applied {kind.value} via reflection "
                            f"proposal {proposal.proposal_id}"
                        ),
                    )
                    new_strategies.append(
                        _replace(
                            strategy,
                            initial_weight=new_weight,
                            revision_history=strategy.revision_history
                            + (history_entry,),
                        )
                    )
                else:
                    new_strategies.append(strategy)
            return _replace(
                protocol, strategy_priors=tuple(new_strategies)
            )

    if field is ProtocolRevisionTargetField.KNOWLEDGE_SEED:
        if kind is ProtocolRevisionChangeKind.ARCHIVE:
            kept = tuple(
                seed
                for seed in protocol.knowledge_seeds
                if seed.seed_id != target_entry_id
            )
            return _replace(protocol, knowledge_seeds=kept)

    if field is ProtocolRevisionTargetField.SIGNATURE_CASE:
        if kind is ProtocolRevisionChangeKind.ARCHIVE:
            kept = tuple(
                case
                for case in protocol.signature_cases
                if case.case_id != target_entry_id
            )
            return _replace(protocol, signature_cases=kept)

    if field is ProtocolRevisionTargetField.BOUNDARY_CONTRACT:
        if kind is ProtocolRevisionChangeKind.BOUNDARY_REFINEMENT:
            return _apply_boundary_refinement(
                protocol, target_entry_id, proposal.proposed_payload or {}
            )

    if field is ProtocolRevisionTargetField.IDENTITY_ASSERTION:
        if kind is ProtocolRevisionChangeKind.IDENTITY_CLARIFICATION:
            return _apply_identity_clarification(
                protocol, proposal.proposed_payload or {}
            )

    # PROTOCOL_RETIREMENT is field-agnostic — it changes review_status.
    if kind is ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT:
        from volvence_zero.behavior_protocol import ReviewStatus
        return _replace(protocol, review_status=ReviewStatus.RETIRED)

    raise NotImplementedError(
        f"apply_revision: change_kind={kind.value!r} on "
        f"target_field={field.value!r} is not yet supported"
    )


def _apply_boundary_refinement(
    protocol: BehaviorProtocol,
    target_entry_id: str,
    payload: dict,
) -> BehaviorProtocol:
    """Modify one BoundaryContract in protocol.boundary_contracts.

    Payload keys (all optional; only supplied fields are updated):
    ``trigger_reasons`` / ``blocked_topics`` / ``required_disclaimers``
    / ``description`` / ``severity`` / ``refer_out_required``.
    """

    new_boundaries = []
    found = False
    for b in protocol.boundary_contracts:
        if b.boundary_id != target_entry_id:
            new_boundaries.append(b)
            continue
        found = True
        kwargs: dict = {}
        for field_name in (
            "trigger_reasons",
            "blocked_topics",
            "required_disclaimers",
            "description",
        ):
            if field_name in payload:
                value = payload[field_name]
                if field_name in ("trigger_reasons", "blocked_topics", "required_disclaimers"):
                    kwargs[field_name] = tuple(
                        v for v in value if isinstance(v, str)
                    )
                else:
                    kwargs[field_name] = str(value)
        if "refer_out_required" in payload:
            kwargs["refer_out_required"] = bool(payload["refer_out_required"])
        if "severity" in payload:
            from volvence_zero.behavior_protocol import BoundarySeverity
            sev = payload["severity"]
            if isinstance(sev, BoundarySeverity):
                kwargs["severity"] = sev
            elif isinstance(sev, str):
                # accept string form
                for member in BoundarySeverity:
                    if member.value == sev or member.name == sev:
                        kwargs["severity"] = member
                        break
        if not kwargs:
            new_boundaries.append(b)
            continue
        new_boundaries.append(_replace(b, **kwargs))
    if not found:
        return protocol
    return _replace(protocol, boundary_contracts=tuple(new_boundaries))


def _apply_identity_clarification(
    protocol: BehaviorProtocol,
    payload: dict,
) -> BehaviorProtocol:
    """Modify identity_assertion fields.

    Payload keys: ``requires_self_traits`` / ``forbidden_self_traits``
    / ``required_regime_compatibility`` (all optional, tuple-of-str).
    """

    from volvence_zero.behavior_protocol import IdentityAssertion

    current = protocol.identity_assertion
    kwargs: dict = {}
    for field_name in (
        "requires_self_traits",
        "forbidden_self_traits",
        "required_regime_compatibility",
    ):
        if field_name in payload:
            value = payload[field_name]
            kwargs[field_name] = tuple(
                v for v in (value or ()) if isinstance(v, str)
            )
    if not kwargs:
        return protocol
    new_identity: IdentityAssertion = _replace(current, **kwargs)
    return _replace(protocol, identity_assertion=new_identity)


__all__ = ["ProtocolLimitExceededError", "ProtocolRegistry"]
