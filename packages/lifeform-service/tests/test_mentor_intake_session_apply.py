from __future__ import annotations

from lifeform_core import LifeformSession
from lifeform_protocol_runtime import build_protocol_revision_proposal
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
    SignatureCase,
    SuccessSignal,
    TemporalArc,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)


class _FakeProtocolRegistryModule:
    def __init__(self) -> None:
        self.loaded: list[tuple[BehaviorProtocol, object]] = []
        self.revisions: list[object] = []

    def load_protocol(self, protocol: BehaviorProtocol, *, load_context=None) -> None:
        self.loaded.append((protocol, load_context))

    def apply_revision(self, proposal, **kwargs) -> None:
        self.revisions.append(proposal)


class _FakeRunner:
    def __init__(self, registry_module: _FakeProtocolRegistryModule) -> None:
        self._protocol_registry_module = registry_module


class _FakeBrainSession:
    session_id = "sess-test"

    def __init__(self, registry_module: _FakeProtocolRegistryModule) -> None:
        self.runner = _FakeRunner(registry_module)
        self.knowledge_events: list[dict] = []
        self.dialogue_outcomes: list[dict] = []

    def submit_reviewed_knowledge_event(self, **kwargs) -> tuple[str, ...]:
        self.knowledge_events.append(kwargs)
        return ("evt-1",)

    def submit_dialogue_outcome(self, **kwargs):
        self.dialogue_outcomes.append(kwargs)
        return kwargs


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


def _session() -> tuple[LifeformSession, _FakeProtocolRegistryModule, _FakeBrainSession]:
    registry_module = _FakeProtocolRegistryModule()
    brain = _FakeBrainSession(registry_module)
    session = LifeformSession(
        brain_session=brain,
        tick=object(),
        scene=object(),
        followups=object(),
    )
    return session, registry_module, brain


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


def test_apply_mentor_knowledge_routes_to_domain_knowledge():
    session, _registry, brain = _session()

    result = session.apply_mentor_knowledge(
        knowledge_id="k1",
        summary="Fragile children need emotional safety first.",
        detail="Co-regulate before reasoning.",
        confidence=0.8,
        reviewer_id="mentor:alice",
        source_label="mentor-intake",
        relevance_hint="when distressed",
    )

    assert result["routed_owner"] == "domain_knowledge"
    assert result["applies_to_current_session"] is True
    assert len(brain.knowledge_events) == 1
    event = brain.knowledge_events[0]
    assert event["knowledge_id"] == "k1"
    assert event["event_id"] == "mentor:k1"


def test_apply_mentor_case_seeds_case_memory_via_carrier_protocol():
    session, registry, _brain = _session()
    case = SignatureCase(
        case_id="case-1",
        domain="child-emotional-support",
        problem_pattern="shuts down when corrected",
        user_state_pattern="high emotional weight",
        risk_markers=(),
        track_tags=(),
        regime_tags=(),
        intervention_ordering=("emotional support", "then logic"),
        outcome_label="repaired",
        confidence=0.7,
        description="repair-first worked example",
    )

    result = session.apply_mentor_case(
        case,
        protocol_id="mentor:case-carrier",
        advisor_name="Mentor",
        reviewer_id="mentor:alice",
    )

    assert result["routed_owner"] == "case_memory"
    assert result["case_id"] == "case-1"
    assert len(registry.loaded) == 1
    carrier, _ctx = registry.loaded[0]
    assert carrier.signature_cases == (case,)
    assert carrier.legacy_fixture is True


def test_record_mentor_experience_records_human_review_outcome():
    session, _registry, brain = _session()

    result = session.record_mentor_experience(
        outcome_kind=DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        reviewer_id="mentor:alice",
        description="led with logic, child disengaged",
    )

    # background-slow: recorded but not applied to the next turn.
    assert result["applies_to_current_session"] is False
    assert result["routed_owner"] == "experience_consolidation"
    assert len(brain.dialogue_outcomes) == 1
    outcome = brain.dialogue_outcomes[0]
    assert outcome["kind"] is DialogueExternalOutcomeKind.OVER_DIRECTIVE
    assert outcome["source"] is DialogueExternalOutcomeEvidenceSource.HUMAN_REVIEW


def test_apply_mentor_protocol_revision_auto_approved_applies():
    session, registry, _brain = _session()
    proposal = build_protocol_revision_proposal(
        proposal_id="rev-1",
        target_protocol_id="mentor:test-protocol",
        target_field="strategy_prior",
        target_entry_id="rule-1",
        change_kind="weight_decay",
        summary="over-firing strategy",
        required_review_level=ReviewLevel.L2,
    )

    result = session.apply_mentor_protocol_revision(
        proposal, reviewer_id="mentor:alice"
    )

    assert result["gate_outcome"] == "auto_approved"
    assert result["applies_to_current_session"] is True
    assert registry.revisions == [proposal]


def test_apply_mentor_protocol_revision_l4_queues_not_applied():
    session, registry, _brain = _session()
    proposal = build_protocol_revision_proposal(
        proposal_id="rev-2",
        target_protocol_id="mentor:test-protocol",
        target_field="strategy_prior",
        target_entry_id="rule-1",
        change_kind="deactivate",
        summary="risky deactivate",
        required_review_level=ReviewLevel.L4,
    )

    result = session.apply_mentor_protocol_revision(
        proposal, reviewer_id="mentor:alice"
    )

    assert result["gate_outcome"] == "queued_for_human"
    assert result["applies_to_current_session"] is False
    assert registry.revisions == []
