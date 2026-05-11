"""Packet 7.3: API injection adapter — payload dict → candidate.

The payload is a structured dict (already-parsed JSON) describing
the protocol. Required fields: ``protocol_id`` / ``advisor_name``
/ ``description``. Optional: ``boundaries`` / ``strategies`` /
``identity`` / ``activation_conditions`` / ``temporal_arc`` /
``success_signals`` / ``failure_signals``.

The adapter performs minimal schema validation and field
mapping; it does NOT run the LLM (no extraction). For protocols
that need LLM enrichment, callers should use DocumentUptake
or TaskDescriptionUptake instead.
"""

from __future__ import annotations

import datetime as _dt

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    ContextMatchSignal,
    FailureSignal,
    IdentityAssertion,
    ProtocolProvenance,
    ProtocolSourceKind,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)


def inject_protocol_from_payload(
    payload: dict,
    *,
    request_id: str,
    extractor_id: str = "lifeform-protocol-runtime/api-injection",
) -> BehaviorProtocolCandidate:
    """Convert a JSON-shaped payload into a candidate protocol.

    Args:
        payload: dict-shaped protocol spec.
        request_id: opaque API request identifier (carried as
            provenance.source_locator).
        extractor_id: provenance attribution string.

    Returns:
        A ``BehaviorProtocolCandidate`` with
        ``provenance.source_kind == API_INJECTION``,
        ``requires_review=True`` (default), and
        ``review_status=DRAFT`` so the standard ModificationGate
        review path applies.

    Raises:
        ValueError: payload missing required fields or has
        invalid schema.
    """

    protocol_id = (payload.get("protocol_id") or "").strip()
    advisor_name = (payload.get("advisor_name") or "").strip()
    description = (payload.get("description") or "").strip()
    if not protocol_id or not advisor_name or not description:
        raise ValueError(
            "inject_protocol_from_payload: payload must include "
            "non-empty 'protocol_id', 'advisor_name', and 'description'"
        )

    identity = _build_identity(payload.get("identity") or {})
    boundaries = _build_boundaries(payload.get("boundaries") or [])
    strategies = _build_strategies(payload.get("strategies") or [])
    activation_conditions = _build_activation_conditions(
        payload.get("activation_conditions") or {}
    )
    success_signals = _build_success_signals(payload.get("success_signals") or [])
    failure_signals = _build_failure_signals(payload.get("failure_signals") or [])
    # Schema invariant: BehaviorProtocol requires non-empty
    # success_signals + failure_signals (legacy_fixture opt-out is
    # for fixture protocols only). API-injected protocols that
    # don't supply signals get a default pair derived from the
    # most generic detector.
    if not success_signals:
        success_signals = (
            SuccessSignal(
                signal_id="api:default:engagement",
                description="default success signal: any interlocutor zone fired",
                measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            ),
        )
    if not failure_signals:
        failure_signals = (
            FailureSignal(
                signal_id="api:default:rupture",
                description="default failure signal: any rupture fired",
                measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            ),
        )

    protocol = BehaviorProtocol(
        protocol_id=protocol_id,
        version=str(payload.get("version", "0.1.0")),
        advisor_name=advisor_name,
        description=description,
        source_kind=ProtocolSourceKind.API_INJECTION,
        source_locator=f"api-injection://{request_id}",
        identity_assertion=identity,
        boundary_contracts=boundaries,
        activation_conditions=activation_conditions,
        strategy_priors=strategies,
        temporal_arc=TemporalArc(),
        success_signals=success_signals,
        failure_signals=failure_signals,
        review_status=ReviewStatus.DRAFT,
        legacy_fixture=False,
    )
    provenance = ProtocolProvenance(
        source_kind=ProtocolSourceKind.API_INJECTION,
        source_locator=f"api-injection://{request_id}",
        extracted_at_iso=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        extractor_id=extractor_id,
        confidence=float(payload.get("confidence", 0.85)),
    )
    return BehaviorProtocolCandidate(
        protocol=protocol,
        provenance=provenance,
        requires_review=True,
    )


def _build_identity(raw: dict) -> IdentityAssertion:
    return IdentityAssertion(
        requires_self_traits=tuple(
            t for t in (raw.get("requires_self_traits") or ())
            if isinstance(t, str)
        ),
        forbidden_self_traits=tuple(
            t for t in (raw.get("forbidden_self_traits") or ())
            if isinstance(t, str)
        ),
        required_regime_compatibility=tuple(
            r for r in (raw.get("required_regime_compatibility") or ())
            if isinstance(r, str)
        ),
    )


def _resolve_severity(raw_severity) -> BoundarySeverity:
    if isinstance(raw_severity, BoundarySeverity):
        return raw_severity
    if isinstance(raw_severity, str):
        for member in BoundarySeverity:
            if member.value == raw_severity or member.name == raw_severity:
                return member
    return BoundarySeverity.SOFT_REMIND


def _build_boundaries(raw_list) -> tuple[BoundaryContract, ...]:
    out = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        boundary_id = (item.get("boundary_id") or "").strip()
        if not boundary_id:
            continue
        out.append(
            BoundaryContract(
                boundary_id=boundary_id,
                description=str(item.get("description", "")).strip()
                or "API-injected boundary",
                severity=_resolve_severity(item.get("severity")),
                trigger_reasons=tuple(
                    s for s in (item.get("trigger_reasons") or ())
                    if isinstance(s, str)
                ) or ("api-injection trigger",),
                blocked_topics=tuple(
                    s for s in (item.get("blocked_topics") or ())
                    if isinstance(s, str)
                ),
                required_disclaimers=tuple(
                    s for s in (item.get("required_disclaimers") or ())
                    if isinstance(s, str)
                ),
                refer_out_required=bool(item.get("refer_out_required", False)),
            )
        )
    return tuple(out)


def _build_strategies(raw_list) -> tuple[StrategyPrior, ...]:
    out = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        rule_id = (item.get("rule_id") or "").strip()
        problem = (item.get("problem_pattern") or "").strip()
        ordering = tuple(
            s for s in (item.get("recommended_ordering") or ())
            if isinstance(s, str)
        )
        if not rule_id or not problem or not ordering:
            continue
        out.append(
            StrategyPrior(
                rule_id=rule_id,
                problem_pattern=problem,
                recommended_ordering=ordering,
                recommended_pacing=str(
                    item.get("recommended_pacing", "moderate")
                ),
                avoid_patterns=tuple(
                    s for s in (item.get("avoid_patterns") or ())
                    if isinstance(s, str)
                ),
                applicability_phase=tuple(
                    s for s in (item.get("applicability_phase") or ())
                    if isinstance(s, str)
                ),
                recommended_regime=item.get("recommended_regime"),
                initial_weight=float(item.get("initial_weight", 1.0)),
                confidence=float(item.get("confidence", 0.7)),
                description=str(item.get("description", "")),
            )
        )
    return tuple(out)


def _build_activation_conditions(raw: dict) -> ActivationConditions:
    raw_signals = raw.get("context_match_signals") or []
    signals: list[ContextMatchSignal] = []
    for item in raw_signals:
        if not isinstance(item, dict):
            continue
        try:
            measurable_via = BehaviorProtocolSignalSource(
                item.get("measurable_via")
            )
        except ValueError:
            continue
        signals.append(
            ContextMatchSignal(
                signal_id=str(item.get("signal_id", measurable_via.value)),
                measurable_via=measurable_via,
                weight=float(item.get("weight", 1.0)),
            )
        )
    return ActivationConditions(
        context_match_signals=tuple(signals),
        co_activation_compatible=tuple(
            s for s in (raw.get("co_activation_compatible") or ())
            if isinstance(s, str)
        ),
        co_activation_incompatible=tuple(
            s for s in (raw.get("co_activation_incompatible") or ())
            if isinstance(s, str)
        ),
        minimum_weight_floor=float(raw.get("minimum_weight_floor", 0.0)),
    )


def _build_success_signals(raw_list) -> tuple[SuccessSignal, ...]:
    out = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            source = BehaviorProtocolSignalSource(item.get("measurable_via"))
        except ValueError:
            continue
        out.append(
            SuccessSignal(
                signal_id=str(item.get("signal_id", source.value)),
                description=str(item.get("description", "")),
                measurable_via=source,
            )
        )
    return tuple(out)


def _build_failure_signals(raw_list) -> tuple[FailureSignal, ...]:
    out = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            source = BehaviorProtocolSignalSource(item.get("measurable_via"))
        except ValueError:
            continue
        out.append(
            FailureSignal(
                signal_id=str(item.get("signal_id", source.value)),
                description=str(item.get("description", "")),
                measurable_via=source,
            )
        )
    return tuple(out)


__all__ = ["inject_protocol_from_payload"]
