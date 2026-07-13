"""CP-04 (intent-alignment W2.E): boundary_consent gate on affordances.

The AffordanceModule reads the published ``boundary_consent`` snapshot.
When the consent owner has decided ``external_action_blocked=True``, every
external-kind candidate (TOOL / SHELL) is published as typed-blocked with a
``consent_blocked:external_action`` reason; internal kinds (ACTION / ORGAN)
stay unblocked. The module consumes the owner's decision as-is - it never
re-derives consent from records.
"""

from __future__ import annotations

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceModule,
    AffordanceRegistry,
    AffordanceSafety,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.semantic_state import BoundaryConsentSnapshot

_HINT = (
    "Use this consent-gate probe when validating that boundary_consent "
    "denials surface as typed-blocked affordance candidates."
)


def _descriptor(name: str, kind: AffordanceKind) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=kind,
        version="1.0.0",
        display_name=name.replace("_", " ").title(),
        description="Consent-gate probe descriptor.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Do not use outside this test.",
        parameters_schema={"type": "object", "properties": {}},
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.INSTANT),
        safety_model=AffordanceSafety(),
    )


def _registry() -> AffordanceRegistry:
    registry = AffordanceRegistry()
    registry.register(_descriptor("external_tool_probe", AffordanceKind.TOOL))
    registry.register(_descriptor("shell_capability_probe", AffordanceKind.SHELL))
    registry.register(_descriptor("internal_action_probe", AffordanceKind.ACTION))
    return registry


def _consent_snapshot(*, external_action_blocked: bool) -> Snapshot[BoundaryConsentSnapshot]:
    value = BoundaryConsentSnapshot(
        granted_consents=(),
        missing_consents=(),
        denied_boundaries=(),
        memory_consent="unknown",
        external_action_consent="denied" if external_action_blocked else "granted",
        compliance_score=1.0,
        control_signal=0.0,
        description="consent-gate probe snapshot",
        external_action_blocked=external_action_blocked,
    )
    return Snapshot(
        slot_name="boundary_consent",
        owner="BoundaryConsentModule",
        version=1,
        timestamp_ms=0,
        value=value,
    )


def _candidates_by_name(snapshot_value) -> dict[str, object]:
    return {
        candidate.descriptor_name: candidate
        for candidate in snapshot_value.candidates_for_turn
    }


def test_module_declares_boundary_consent_dependency() -> None:
    assert "boundary_consent" in AffordanceModule.dependencies


async def test_consent_denial_blocks_external_kinds_only() -> None:
    module = AffordanceModule(registry=_registry())
    result = await module.process(
        {"boundary_consent": _consent_snapshot(external_action_blocked=True)}
    )
    by_name = _candidates_by_name(result.value)

    tool = by_name["external_tool_probe"]
    shell = by_name["shell_capability_probe"]
    action = by_name["internal_action_probe"]

    assert tool.is_blocked
    assert tool.blocked_reason.startswith("consent_blocked:external_action")
    assert "kind=tool" in tool.blocked_reason
    assert shell.is_blocked
    assert "kind=shell" in shell.blocked_reason
    assert not action.is_blocked

    # Blocked candidates never become the selected offer.
    assert result.value.selected is None or result.value.selected.descriptor_name == (
        "internal_action_probe"
    )
    assert "consent_external_blocked=True" in result.value.description


async def test_consent_granted_keeps_external_kinds_unblocked() -> None:
    module = AffordanceModule(registry=_registry())
    result = await module.process(
        {"boundary_consent": _consent_snapshot(external_action_blocked=False)}
    )
    by_name = _candidates_by_name(result.value)
    assert not by_name["external_tool_probe"].is_blocked
    assert not by_name["shell_capability_probe"].is_blocked
    assert "consent_external_blocked=False" in result.value.description


async def test_missing_consent_snapshot_is_a_no_op() -> None:
    module = AffordanceModule(registry=_registry())
    result = await module.process({})
    by_name = _candidates_by_name(result.value)
    assert not any(candidate.is_blocked for candidate in by_name.values())
    assert "consent_external_blocked=None" in result.value.description
