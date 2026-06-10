from __future__ import annotations

from lifeform_core import LifeformSession
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    FailureSignal,
    IdentityAssertion,
    ProtocolSourceKind,
    ReviewLevel,
    ReviewStatus,
    SuccessSignal,
    TemporalArc,
)


class _FakeProtocolRegistryModule:
    def __init__(self) -> None:
        self.loaded: list[tuple[BehaviorProtocol, object]] = []

    def load_protocol(self, protocol: BehaviorProtocol, *, load_context=None) -> None:
        self.loaded.append((protocol, load_context))


class _FakeRunner:
    def __init__(self, registry_module: _FakeProtocolRegistryModule) -> None:
        self._protocol_registry_module = registry_module


class _FakeBrainSession:
    session_id = "sess-test"

    def __init__(self, registry_module: _FakeProtocolRegistryModule) -> None:
        self.runner = _FakeRunner(registry_module)


def _protocol() -> BehaviorProtocol:
    return BehaviorProtocol(
        protocol_id="mentor:test-protocol",
        version="0.1.0",
        advisor_name="Mentor",
        description="Test mentor protocol.",
        source_kind=ProtocolSourceKind.TASK_DESCRIPTION,
        source_locator="mentor://test",
        identity_assertion=IdentityAssertion(
            requires_self_traits=(),
            forbidden_self_traits=(),
            required_regime_compatibility=(),
        ),
        boundary_contracts=(),
        activation_conditions=ActivationConditions(),
        strategy_priors=(),
        temporal_arc=TemporalArc(),
        success_signals=(
            SuccessSignal(
                signal_id="success",
                description="success",
                measurable_via=BehaviorProtocolSignalSource.RETRIEVAL_HITS_PRESENT,
            ),
        ),
        failure_signals=(
            FailureSignal(
                signal_id="failure",
                description="failure",
                measurable_via=BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED,
            ),
        ),
        review_status=ReviewStatus.ACTIVE,
    )


def test_apply_mentor_intake_loads_only_session_registry():
    registry_module = _FakeProtocolRegistryModule()
    session = LifeformSession(
        brain_session=_FakeBrainSession(registry_module),
        tick=object(),
        scene=object(),
        followups=object(),
    )
    uptake_service = ProtocolUptakeService(config=ProtocolUptakeConfig())

    result = session.apply_mentor_intake(
        _protocol(),
        reviewer_id="mentor:alice",
        reviewer_level=ReviewLevel.L4,
        note="test",
    )

    assert result["applies_to_current_session"] is True
    assert len(registry_module.loaded) == 1
    loaded_protocol, load_context = registry_module.loaded[0]
    assert loaded_protocol.protocol_id == "mentor:test-protocol"
    assert load_context.reviewer_id == "mentor:alice"
    assert uptake_service.loaded_approved_snapshot() == ()
