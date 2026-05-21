"""Lossless serialization for :class:`BehaviorProtocol`.

This module is the **service-level persistence contract** for the
Behavior Protocol Runtime. It provides a round-trip pair:

* :func:`protocol_to_payload` — ``BehaviorProtocol -> dict`` suitable
  for ``json.dumps``.
* :func:`protocol_from_payload` — ``dict -> BehaviorProtocol``
  reconstructing the exact same protocol (modulo whitespace
  normalisation done by dataclass ``__post_init__`` validators).

Why this lives next to the existing API-injection adapter rather
than in ``vz-contracts``:

* ``vz-contracts`` is pure schema (data only). Adding JSON I/O
  there would expand its surface beyond its charter.
* The existing :func:`inject_protocol_from_payload` already lives
  in this wheel and is the dict-shaped consumer-facing builder.
  However, that builder is intentionally **lossy** — it forces
  ``source_kind=API_INJECTION``, drops ``revision_log``, ignores
  ``signature_cases`` / ``knowledge_seeds``, etc. — because its
  purpose is to accept external orchestrator payloads. Our
  persistence path wants the opposite: round-trip exactness.

Schema versioning
-----------------

Payloads carry ``"schema_version": "1.0"`` at the top so future
schema changes can migrate at read-time. Reading a payload whose
``schema_version`` is missing or differs from the current value
raises :class:`ProtocolPayloadSchemaError` rather than silently
producing a stale-shaped protocol.

R8 / SSOT
---------

The serialiser is a **pure function** of the protocol; it does
not consult registry state and does not read or write files.
Anything stateful (which directory the JSON lands in, whether it
is currently loaded into a registry) lives upstream in
:class:`lifeform_service.protocol_persistence.ProtocolPersistenceStore`.
"""

from __future__ import annotations

from typing import Any

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    ContextMatchSignal,
    DriveExpectation,
    FailureSignal,
    IdentityAssertion,
    KnowledgeSeed,
    ProgressionSignal,
    ProtocolRevision,
    ProtocolSourceKind,
    ReviewLevel,
    ReviewStatus,
    SignatureCase,
    StrategyPrior,
    StrategyPriorRevision,
    SuccessSignal,
    TemporalArc,
    TemporalPhase,
)


SCHEMA_VERSION = "1.0"


class ProtocolPayloadSchemaError(ValueError):
    """Raised when a payload's schema_version doesn't match SCHEMA_VERSION."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def protocol_to_payload(protocol: BehaviorProtocol) -> dict[str, Any]:
    """Convert a :class:`BehaviorProtocol` into a JSON-safe ``dict``.

    The returned dict is suitable for ``json.dumps`` with no custom
    encoder. All enums are serialised via ``.value`` (their string
    form), tuples become lists, and ``None`` is preserved for
    optional fields.

    Round-trip: ``protocol_from_payload(protocol_to_payload(p)) == p``
    for any well-formed protocol.
    """

    if not isinstance(protocol, BehaviorProtocol):
        raise TypeError(
            f"protocol_to_payload expects BehaviorProtocol, got "
            f"{type(protocol).__name__}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "version": protocol.version,
        "advisor_name": protocol.advisor_name,
        "description": protocol.description,
        "source_kind": protocol.source_kind.value,
        "source_locator": protocol.source_locator,
        "identity_assertion": _identity_to_payload(protocol.identity_assertion),
        "boundary_contracts": [
            _boundary_to_payload(b) for b in protocol.boundary_contracts
        ],
        "activation_conditions": _activation_to_payload(
            protocol.activation_conditions
        ),
        "strategy_priors": [
            _strategy_to_payload(s) for s in protocol.strategy_priors
        ],
        "temporal_arc": _temporal_arc_to_payload(protocol.temporal_arc),
        "success_signals": [
            _success_signal_to_payload(s) for s in protocol.success_signals
        ],
        "failure_signals": [
            _failure_signal_to_payload(s) for s in protocol.failure_signals
        ],
        "knowledge_seeds": [
            _knowledge_seed_to_payload(k) for k in protocol.knowledge_seeds
        ],
        "signature_cases": [
            _signature_case_to_payload(c) for c in protocol.signature_cases
        ],
        "parent_protocol_id": protocol.parent_protocol_id,
        "review_status": protocol.review_status.value,
        "revision_log": [
            _revision_to_payload(r) for r in protocol.revision_log
        ],
        "legacy_fixture": protocol.legacy_fixture,
    }


def protocol_from_payload(payload: dict[str, Any]) -> BehaviorProtocol:
    """Reconstruct a :class:`BehaviorProtocol` from a payload dict.

    Inverse of :func:`protocol_to_payload`. Raises
    :class:`ProtocolPayloadSchemaError` when the payload's
    ``schema_version`` is missing or different from
    :data:`SCHEMA_VERSION` — fail loudly rather than produce a
    silently-degraded protocol.

    Dataclass ``__post_init__`` validators still fire as normal so a
    payload with out-of-range floats / duplicate ids / missing
    required signals will raise ``ValueError`` from the dataclass
    layer (not this function).
    """

    if not isinstance(payload, dict):
        raise TypeError(
            f"protocol_from_payload expects dict, got {type(payload).__name__}"
        )
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ProtocolPayloadSchemaError(
            f"protocol_from_payload: payload schema_version="
            f"{schema_version!r} does not match expected {SCHEMA_VERSION!r}; "
            "this persisted protocol was written by a different version "
            "and needs a migration step."
        )
    return BehaviorProtocol(
        protocol_id=str(payload["protocol_id"]),
        version=str(payload["version"]),
        advisor_name=str(payload["advisor_name"]),
        description=str(payload["description"]),
        source_kind=ProtocolSourceKind(payload["source_kind"]),
        source_locator=str(payload["source_locator"]),
        identity_assertion=_identity_from_payload(payload["identity_assertion"]),
        boundary_contracts=tuple(
            _boundary_from_payload(b) for b in payload["boundary_contracts"]
        ),
        activation_conditions=_activation_from_payload(
            payload["activation_conditions"]
        ),
        strategy_priors=tuple(
            _strategy_from_payload(s) for s in payload["strategy_priors"]
        ),
        temporal_arc=_temporal_arc_from_payload(payload["temporal_arc"]),
        success_signals=tuple(
            _success_signal_from_payload(s) for s in payload["success_signals"]
        ),
        failure_signals=tuple(
            _failure_signal_from_payload(s) for s in payload["failure_signals"]
        ),
        knowledge_seeds=tuple(
            _knowledge_seed_from_payload(k)
            for k in payload.get("knowledge_seeds", ())
        ),
        signature_cases=tuple(
            _signature_case_from_payload(c)
            for c in payload.get("signature_cases", ())
        ),
        parent_protocol_id=payload.get("parent_protocol_id"),
        review_status=ReviewStatus(payload.get("review_status", "draft")),
        revision_log=tuple(
            _revision_from_payload(r) for r in payload.get("revision_log", ())
        ),
        legacy_fixture=bool(payload.get("legacy_fixture", False)),
    )


# ---------------------------------------------------------------------------
# Nested dataclass helpers (one to/from pair per type)
# ---------------------------------------------------------------------------


def _identity_to_payload(ia: IdentityAssertion) -> dict[str, Any]:
    return {
        "requires_self_traits": list(ia.requires_self_traits),
        "forbidden_self_traits": list(ia.forbidden_self_traits),
        "required_regime_compatibility": list(
            ia.required_regime_compatibility
        ),
    }


def _identity_from_payload(raw: dict[str, Any]) -> IdentityAssertion:
    return IdentityAssertion(
        requires_self_traits=tuple(raw.get("requires_self_traits", ())),
        forbidden_self_traits=tuple(raw.get("forbidden_self_traits", ())),
        required_regime_compatibility=tuple(
            raw.get("required_regime_compatibility", ())
        ),
    )


def _boundary_to_payload(b: BoundaryContract) -> dict[str, Any]:
    return {
        "boundary_id": b.boundary_id,
        "description": b.description,
        "trigger_reasons": list(b.trigger_reasons),
        "blocked_topics": list(b.blocked_topics),
        "required_disclaimers": list(b.required_disclaimers),
        "refer_out_required": b.refer_out_required,
        "regime_id": b.regime_id,
        "answer_depth_limit_hint": b.answer_depth_limit_hint,
        "clarification_required": b.clarification_required,
        "severity": b.severity.value,
        "review_level": b.review_level.value,
        "confidence": b.confidence,
    }


def _boundary_from_payload(raw: dict[str, Any]) -> BoundaryContract:
    return BoundaryContract(
        boundary_id=str(raw["boundary_id"]),
        description=str(raw["description"]),
        trigger_reasons=tuple(raw["trigger_reasons"]),
        blocked_topics=tuple(raw.get("blocked_topics", ())),
        required_disclaimers=tuple(raw.get("required_disclaimers", ())),
        refer_out_required=bool(raw.get("refer_out_required", False)),
        regime_id=raw.get("regime_id"),
        answer_depth_limit_hint=str(raw.get("answer_depth_limit_hint", "")),
        clarification_required=bool(raw.get("clarification_required", False)),
        severity=BoundarySeverity(raw.get("severity", "hard_block")),
        review_level=ReviewLevel(raw.get("review_level", "l3")),
        confidence=float(raw.get("confidence", 0.9)),
    )


def _strategy_revision_to_payload(r: StrategyPriorRevision) -> dict[str, Any]:
    return {
        "revision_id": r.revision_id,
        "revised_at_tick": r.revised_at_tick,
        "delta": r.delta,
        "reason": r.reason,
    }


def _strategy_revision_from_payload(
    raw: dict[str, Any],
) -> StrategyPriorRevision:
    return StrategyPriorRevision(
        revision_id=str(raw["revision_id"]),
        revised_at_tick=int(raw["revised_at_tick"]),
        delta=float(raw["delta"]),
        reason=str(raw.get("reason", "")),
    )


def _strategy_to_payload(s: StrategyPrior) -> dict[str, Any]:
    return {
        "rule_id": s.rule_id,
        "problem_pattern": s.problem_pattern,
        "recommended_ordering": list(s.recommended_ordering),
        "recommended_pacing": s.recommended_pacing,
        "avoid_patterns": list(s.avoid_patterns),
        "applicability_phase": list(s.applicability_phase),
        "recommended_regime": s.recommended_regime,
        "knowledge_weight_hint": s.knowledge_weight_hint,
        "experience_weight_hint": s.experience_weight_hint,
        "initial_weight": s.initial_weight,
        "pe_decay_rate": s.pe_decay_rate,
        "pe_reinforce_rate": s.pe_reinforce_rate,
        "minimum_weight_floor": s.minimum_weight_floor,
        "revision_history": [
            _strategy_revision_to_payload(r) for r in s.revision_history
        ],
        "confidence": s.confidence,
        "description": s.description,
    }


def _strategy_from_payload(raw: dict[str, Any]) -> StrategyPrior:
    return StrategyPrior(
        rule_id=str(raw["rule_id"]),
        problem_pattern=str(raw["problem_pattern"]),
        recommended_ordering=tuple(raw["recommended_ordering"]),
        recommended_pacing=str(raw["recommended_pacing"]),
        avoid_patterns=tuple(raw.get("avoid_patterns", ())),
        applicability_phase=tuple(raw.get("applicability_phase", ())),
        recommended_regime=raw.get("recommended_regime"),
        knowledge_weight_hint=float(raw.get("knowledge_weight_hint", 0.45)),
        experience_weight_hint=float(raw.get("experience_weight_hint", 0.65)),
        initial_weight=float(raw.get("initial_weight", 1.0)),
        pe_decay_rate=float(raw.get("pe_decay_rate", 0.0)),
        pe_reinforce_rate=float(raw.get("pe_reinforce_rate", 0.0)),
        minimum_weight_floor=float(raw.get("minimum_weight_floor", 0.0)),
        revision_history=tuple(
            _strategy_revision_from_payload(r)
            for r in raw.get("revision_history", ())
        ),
        confidence=float(raw.get("confidence", 0.85)),
        description=str(raw.get("description", "")),
    )


def _progression_signal_to_payload(p: ProgressionSignal) -> dict[str, Any]:
    return {
        "signal_id": p.signal_id,
        "measurable_via": p.measurable_via.value,
        "threshold": p.threshold,
        "description": p.description,
    }


def _progression_signal_from_payload(raw: dict[str, Any]) -> ProgressionSignal:
    return ProgressionSignal(
        signal_id=str(raw["signal_id"]),
        measurable_via=BehaviorProtocolSignalSource(raw["measurable_via"]),
        threshold=float(raw.get("threshold", 0.0)),
        description=str(raw.get("description", "")),
    )


def _drive_expectation_to_payload(d: DriveExpectation) -> dict[str, Any]:
    return {
        "drive_name": d.drive_name,
        "expected_band": list(d.expected_band),
    }


def _drive_expectation_from_payload(raw: dict[str, Any]) -> DriveExpectation:
    band = raw["expected_band"]
    return DriveExpectation(
        drive_name=str(raw["drive_name"]),
        expected_band=(float(band[0]), float(band[1])),
    )


def _temporal_phase_to_payload(tp: TemporalPhase) -> dict[str, Any]:
    return {
        "phase_id": tp.phase_id,
        "description": tp.description,
        "entry_conditions": [
            _progression_signal_to_payload(p) for p in tp.entry_conditions
        ],
        "exit_conditions": [
            _progression_signal_to_payload(p) for p in tp.exit_conditions
        ],
        "expected_drives_state": [
            _drive_expectation_to_payload(d) for d in tp.expected_drives_state
        ],
    }


def _temporal_phase_from_payload(raw: dict[str, Any]) -> TemporalPhase:
    return TemporalPhase(
        phase_id=str(raw["phase_id"]),
        description=str(raw.get("description", "")),
        entry_conditions=tuple(
            _progression_signal_from_payload(p)
            for p in raw.get("entry_conditions", ())
        ),
        exit_conditions=tuple(
            _progression_signal_from_payload(p)
            for p in raw.get("exit_conditions", ())
        ),
        expected_drives_state=tuple(
            _drive_expectation_from_payload(d)
            for d in raw.get("expected_drives_state", ())
        ),
    )


def _temporal_arc_to_payload(arc: TemporalArc) -> dict[str, Any]:
    return {
        "phases": [_temporal_phase_to_payload(p) for p in arc.phases],
        "progression_signals": [
            _progression_signal_to_payload(s)
            for s in arc.progression_signals
        ],
    }


def _temporal_arc_from_payload(raw: dict[str, Any]) -> TemporalArc:
    return TemporalArc(
        phases=tuple(
            _temporal_phase_from_payload(p) for p in raw.get("phases", ())
        ),
        progression_signals=tuple(
            _progression_signal_from_payload(s)
            for s in raw.get("progression_signals", ())
        ),
    )


def _context_match_to_payload(c: ContextMatchSignal) -> dict[str, Any]:
    return {
        "signal_id": c.signal_id,
        "measurable_via": c.measurable_via.value,
        "weight": c.weight,
        "description": c.description,
    }


def _context_match_from_payload(raw: dict[str, Any]) -> ContextMatchSignal:
    return ContextMatchSignal(
        signal_id=str(raw["signal_id"]),
        measurable_via=BehaviorProtocolSignalSource(raw["measurable_via"]),
        weight=float(raw.get("weight", 1.0)),
        description=str(raw.get("description", "")),
    )


def _activation_to_payload(a: ActivationConditions) -> dict[str, Any]:
    return {
        "context_match_signals": [
            _context_match_to_payload(c) for c in a.context_match_signals
        ],
        "co_activation_compatible": list(a.co_activation_compatible),
        "co_activation_incompatible": list(a.co_activation_incompatible),
        "minimum_weight_floor": a.minimum_weight_floor,
    }


def _activation_from_payload(raw: dict[str, Any]) -> ActivationConditions:
    return ActivationConditions(
        context_match_signals=tuple(
            _context_match_from_payload(c)
            for c in raw.get("context_match_signals", ())
        ),
        co_activation_compatible=tuple(
            raw.get("co_activation_compatible", ())
        ),
        co_activation_incompatible=tuple(
            raw.get("co_activation_incompatible", ())
        ),
        minimum_weight_floor=float(raw.get("minimum_weight_floor", 0.0)),
    )


def _success_signal_to_payload(s: SuccessSignal) -> dict[str, Any]:
    return {
        "signal_id": s.signal_id,
        "description": s.description,
        "measurable_via": s.measurable_via.value,
        "expected_value_range": list(s.expected_value_range),
        "weight_in_pe": s.weight_in_pe,
    }


def _success_signal_from_payload(raw: dict[str, Any]) -> SuccessSignal:
    band = raw.get("expected_value_range", (0.0, 1.0))
    return SuccessSignal(
        signal_id=str(raw["signal_id"]),
        description=str(raw.get("description", "")),
        measurable_via=BehaviorProtocolSignalSource(raw["measurable_via"]),
        expected_value_range=(float(band[0]), float(band[1])),
        weight_in_pe=float(raw.get("weight_in_pe", 1.0)),
    )


def _failure_signal_to_payload(f: FailureSignal) -> dict[str, Any]:
    return {
        "signal_id": f.signal_id,
        "description": f.description,
        "measurable_via": f.measurable_via.value,
        "threshold": f.threshold,
        "weight_in_pe": f.weight_in_pe,
    }


def _failure_signal_from_payload(raw: dict[str, Any]) -> FailureSignal:
    return FailureSignal(
        signal_id=str(raw["signal_id"]),
        description=str(raw.get("description", "")),
        measurable_via=BehaviorProtocolSignalSource(raw["measurable_via"]),
        threshold=float(raw.get("threshold", 0.0)),
        weight_in_pe=float(raw.get("weight_in_pe", 1.0)),
    )


def _knowledge_seed_to_payload(k: KnowledgeSeed) -> dict[str, Any]:
    return {
        "seed_id": k.seed_id,
        "domain": k.domain,
        "title": k.title,
        "summary": k.summary,
        "snippet": k.snippet,
        "evidence_locator": k.evidence_locator,
        "confidence": k.confidence,
        "evidence_strength": k.evidence_strength,
        "topic_tags": list(k.topic_tags),
        "source_type": k.source_type,
        "freshness_label": k.freshness_label,
        "jurisdiction_tags": list(k.jurisdiction_tags),
        "conflict_markers": list(k.conflict_markers),
    }


def _knowledge_seed_from_payload(raw: dict[str, Any]) -> KnowledgeSeed:
    return KnowledgeSeed(
        seed_id=str(raw["seed_id"]),
        domain=str(raw["domain"]),
        title=str(raw["title"]),
        summary=str(raw["summary"]),
        snippet=str(raw["snippet"]),
        evidence_locator=str(raw["evidence_locator"]),
        confidence=float(raw["confidence"]),
        evidence_strength=str(raw.get("evidence_strength", "medium")),
        topic_tags=tuple(raw.get("topic_tags", ())),
        source_type=str(raw.get("source_type", "internal-guide")),
        freshness_label=str(raw.get("freshness_label", "reviewed")),
        jurisdiction_tags=tuple(raw.get("jurisdiction_tags", ())),
        conflict_markers=tuple(raw.get("conflict_markers", ())),
    )


def _signature_case_to_payload(c: SignatureCase) -> dict[str, Any]:
    return {
        "case_id": c.case_id,
        "domain": c.domain,
        "problem_pattern": c.problem_pattern,
        "user_state_pattern": c.user_state_pattern,
        "risk_markers": list(c.risk_markers),
        "track_tags": list(c.track_tags),
        "regime_tags": list(c.regime_tags),
        "intervention_ordering": list(c.intervention_ordering),
        "outcome_label": c.outcome_label,
        "confidence": c.confidence,
        "description": c.description,
        "relevance_score": c.relevance_score,
        "escalation_observed": c.escalation_observed,
        "repair_observed": c.repair_observed,
        "delayed_signal_count": c.delayed_signal_count,
        "reconstruction_source": c.reconstruction_source,
    }


def _signature_case_from_payload(raw: dict[str, Any]) -> SignatureCase:
    return SignatureCase(
        case_id=str(raw["case_id"]),
        domain=str(raw["domain"]),
        problem_pattern=str(raw["problem_pattern"]),
        user_state_pattern=str(raw["user_state_pattern"]),
        risk_markers=tuple(raw.get("risk_markers", ())),
        track_tags=tuple(raw.get("track_tags", ())),
        regime_tags=tuple(raw.get("regime_tags", ())),
        intervention_ordering=tuple(raw["intervention_ordering"]),
        outcome_label=str(raw["outcome_label"]),
        confidence=float(raw["confidence"]),
        description=str(raw["description"]),
        relevance_score=float(raw.get("relevance_score", 0.75)),
        escalation_observed=bool(raw.get("escalation_observed", False)),
        repair_observed=bool(raw.get("repair_observed", False)),
        delayed_signal_count=int(raw.get("delayed_signal_count", 0)),
        reconstruction_source=str(
            raw.get("reconstruction_source", "behavior-protocol")
        ),
    )


def _revision_to_payload(r: ProtocolRevision) -> dict[str, Any]:
    return {
        "revision_id": r.revision_id,
        "revised_at_tick": r.revised_at_tick,
        "revised_by": r.revised_by,
        "description": r.description,
        "affected_field": r.affected_field,
    }


def _revision_from_payload(raw: dict[str, Any]) -> ProtocolRevision:
    return ProtocolRevision(
        revision_id=str(raw["revision_id"]),
        revised_at_tick=int(raw["revised_at_tick"]),
        revised_by=str(raw["revised_by"]),
        description=str(raw["description"]),
        affected_field=str(raw["affected_field"]),
    )


__all__ = [
    "SCHEMA_VERSION",
    "ProtocolPayloadSchemaError",
    "protocol_from_payload",
    "protocol_to_payload",
]
