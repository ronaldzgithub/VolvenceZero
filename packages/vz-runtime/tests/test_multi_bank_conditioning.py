"""Multi-bank conditioning delivery and lineage (State KV P4-b).

Pins the consumer-side semantics of the second bank: the Relationship
readout is delivered through exactly one text or versioned residual
carrier, the prompt state block and audit ref combine
per-bank deliveries in bank order, the action lineage records both banks
sorted with the versioned ``static-all.v1`` routing policy, and every
gate (SHADOW default, residual mode, revocation, cold start) keeps the
single-bank behaviour unchanged.

These tests intentionally avoid a full ``AgentSessionRunner``: the
delivery helpers and the lineage builder are the units this packet
changed, and the existing arm-E attribution end-to-end test remains the
integration regression for the personal-only path.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from companion_standard.semantic_state import SemanticRecord
import pytest

from volvence_zero.agent.conditioning_lineage import (
    bank_fingerprint,
    build_conditioning_lineage,
)
from volvence_zero.temporal.conditioning_router import (
    TOPK_SEMANTIC_ROUTER_VERSION,
    select_conditioning_banks,
)
from volvence_zero.agent.session_observation import (
    _STATIC_ROUTER_VERSION,
    _merge_text_delivery,
    _relationship_conditioning_delivery_from_config,
    _relationship_conditioning_text_delivery,
)
from volvence_zero.conditioning_bank_adapters import (
    bank_readout_to_bank,
    personal_conditioning_to_bank,
)
from volvence_zero.conditioning_bank_contracts import (
    CONDITIONING_BANK_READOUT_SCHEMA_VERSION,
    ConditioningBankReadout,
    ConditioningBankType,
    ConditioningRevocationState,
    ConditioningScope,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.relationship_conditioning import (
    RELATIONSHIP_CONDITIONING_COMPILER_VERSION,
    RELATIONSHIP_CONDITIONING_READOUT_LABELS,
    RELATIONSHIP_CONDITIONING_SLOT,
    RelationshipConditioningModule,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    RelationshipStateSnapshot,
)


def _scope() -> ConditioningScope:
    return ConditioningScope(
        tenant_scope="runner-local",
        user_scope="user-1",
        session_scope="session-1",
    )


def _relationship_readout(
    *,
    confidence: float = 0.6,
    fingerprint: str = "rel-fp-000000000001",
) -> ConditioningBankReadout:
    return ConditioningBankReadout(
        schema_version=CONDITIONING_BANK_READOUT_SCHEMA_VERSION,
        bank_type=ConditioningBankType.RELATIONSHIP,
        readout=tuple(
            0.5 for _ in RELATIONSHIP_CONDITIONING_READOUT_LABELS
        ),
        readout_labels=RELATIONSHIP_CONDITIONING_READOUT_LABELS,
        source_versions=(
            ("relationship_state", 3),
            ("boundary_consent", 2),
        ),
        source_fingerprint=fingerprint,
        confidence=confidence,
        provenance="owner:RelationshipConditioningModule/test",
        is_cold_start=False,
        description="relationship bank test readout",
        rendered_statement=(
            "Current dyad relationship estimate (typed readout only, "
            f"confidence {confidence:.2f}): steady."
        ),
    )


def _personal_snapshot() -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(
            0.4 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
        ),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(
            ("user_model", 1),
            ("relationship_state", 1),
            ("goal_value", 1),
            ("boundary_consent", 1),
        ),
        source_fingerprint="personal-fp-0001",
        confidence=0.7,
        is_cold_start=False,
        description="personal test snapshot",
        rendered_statement="Current relational state estimate: steady.",
    )


def _trajectory_owner_readout(
    relationship: RelationshipStateSnapshot,
) -> ConditioningBankReadout:
    record = SemanticRecord(
        record_id="relationship-evidence",
        summary="typed relationship evidence",
        detail="typed relationship evidence detail",
        confidence=0.8,
        status="active",
        source_turn=1,
        evidence="reviewed:test",
    )
    boundary = BoundaryConsentSnapshot(
        granted_consents=(record,),
        missing_consents=(),
        denied_boundaries=(),
        memory_consent="granted",
        external_action_consent="granted",
        compliance_score=0.8,
        control_signal=0.1,
        description="typed boundary test snapshot",
        consent_clarity=0.75,
    )
    upstream = {
        "relationship_state": Snapshot(
            slot_name="relationship_state",
            owner="RelationshipStateModule",
            version=3,
            timestamp_ms=0,
            value=relationship,
        ),
        "boundary_consent": Snapshot(
            slot_name="boundary_consent",
            owner="BoundaryConsentModule",
            version=2,
            timestamp_ms=0,
            value=boundary,
        ),
    }
    return asyncio.run(RelationshipConditioningModule().process(upstream)).value


def test_relationship_compiler_publishes_versioned_trajectory_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = SemanticRecord(
        record_id="rapport",
        summary="typed rapport",
        detail="typed rapport detail",
        confidence=0.8,
        status="active",
        source_turn=1,
        evidence="reviewed:test",
    )
    base = RelationshipStateSnapshot(
        trust_level=0.6,
        continuity_level=0.5,
        repair_pressure=0.2,
        rapport_signals=(record,),
        relational_tensions=(),
        control_signal=0.1,
        description="typed relationship test snapshot",
        emotional_load=0.4,
        stabilization_need=0.3,
        recent_repair_count=3,
        unresolved_tension_count=1,
        attunement_trend=0.8,
        trust_recovery_signal=0.7,
        relationship_continuity_score=0.75,
        cumulative_trust_level=0.55,
        relationship_age_turns=18,
    )
    recovering = _trajectory_owner_readout(base)
    newly_stable = _trajectory_owner_readout(
        replace(
            base,
            recent_repair_count=0,
            attunement_trend=0.35,
            relationship_continuity_score=0.45,
            relationship_age_turns=2,
        )
    )
    recovering_values = dict(
        zip(recovering.readout_labels, recovering.readout, strict=True)
    )
    newly_stable_values = dict(
        zip(newly_stable.readout_labels, newly_stable.readout, strict=True)
    )

    assert recovering.readout_labels == RELATIONSHIP_CONDITIONING_READOUT_LABELS
    assert RELATIONSHIP_CONDITIONING_COMPILER_VERSION in recovering.provenance
    assert recovering_values["rel_repair_progress"] == pytest.approx(0.75)
    assert newly_stable_values["rel_repair_progress"] == pytest.approx(0.0)
    assert recovering_values["rel_relationship_depth"] == pytest.approx(0.9)
    assert newly_stable_values["rel_relationship_depth"] == pytest.approx(0.1)
    assert recovering.source_fingerprint != newly_stable.source_fingerprint
    assert recovering.rendered_statement != newly_stable.rendered_statement
    monkeypatch.setattr(
        "volvence_zero.relationship_conditioning."
        "RELATIONSHIP_CONDITIONING_COMPILER_VERSION",
        "relationship-conditioning.test-version",
    )
    recompiled = _trajectory_owner_readout(base)
    assert recompiled.source_fingerprint != recovering.source_fingerprint


# ---------------------------------------------------------------------------
# Relationship text delivery gates
# ---------------------------------------------------------------------------


def test_relationship_delivers_only_in_text_mode() -> None:
    readout = _relationship_readout()

    statement, ref = _relationship_conditioning_text_delivery(
        relationship_readout=readout,
        personal_conditioning_mode="text",
        revocation_state=ConditioningRevocationState.ACTIVE,
    )
    assert statement == readout.rendered_statement
    assert ref == (
        f"{readout.schema_version}:{readout.confidence:.2f}:"
        f"{readout.source_fingerprint[:12]}"
    )

    # This helper is the legacy text-only surface. Latent delivery is owned by
    # the explicit carrier selector below.
    for mode in ("residual", "prefix_kv"):
        assert _relationship_conditioning_text_delivery(
            relationship_readout=readout,
            personal_conditioning_mode=mode,
            revocation_state=ConditioningRevocationState.ACTIVE,
        ) == ("", "")


def test_relationship_residual_delivery_builds_versioned_carrier() -> None:
    from volvence_zero.substrate import (
        RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION,
    )

    readout = _relationship_readout()
    bank = bank_readout_to_bank(
        readout=readout,
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
    )
    carrier, statement, statement_ref = (
        _relationship_conditioning_delivery_from_config(
            relationship_readout=readout,
            relationship_bank=bank,
            relationship_conditioning_mode="residual",
            revocation_state=ConditioningRevocationState.ACTIVE,
        )
    )
    assert carrier is not None
    assert carrier.bank is bank
    assert carrier.projector_version == RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION
    assert statement == ""
    assert statement_ref == ""

    learned_version = "relationship-contrastive-residual.v1:artifact-123"
    learned_carrier, _, _ = _relationship_conditioning_delivery_from_config(
        relationship_readout=readout,
        relationship_bank=bank,
        relationship_conditioning_mode="residual",
        revocation_state=ConditioningRevocationState.ACTIVE,
        projector_version=learned_version,
    )
    assert learned_carrier is not None
    assert learned_carrier.projector_version == learned_version


def test_relationship_residual_delivery_is_revocation_safe() -> None:
    readout = _relationship_readout()
    bank = bank_readout_to_bank(
        readout=readout,
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
    )
    assert _relationship_conditioning_delivery_from_config(
        relationship_readout=readout,
        relationship_bank=bank,
        relationship_conditioning_mode="residual",
        revocation_state=ConditioningRevocationState.REVOKED,
    ) == (None, "", "")


def test_relationship_delivery_gates_on_absence_and_revocation() -> None:
    assert _relationship_conditioning_text_delivery(
        relationship_readout=None,
        personal_conditioning_mode="text",
        revocation_state=ConditioningRevocationState.ACTIVE,
    ) == ("", "")

    assert _relationship_conditioning_text_delivery(
        relationship_readout=_relationship_readout(),
        personal_conditioning_mode="text",
        revocation_state=ConditioningRevocationState.REVOKED,
    ) == ("", "")


def test_relationship_delivery_requires_rendered_statement() -> None:
    from dataclasses import replace

    broken = replace(_relationship_readout(), rendered_statement="")
    with pytest.raises(ValueError, match="rendered_statement"):
        _relationship_conditioning_text_delivery(
            relationship_readout=broken,
            personal_conditioning_mode="text",
            revocation_state=ConditioningRevocationState.ACTIVE,
        )


# ---------------------------------------------------------------------------
# Prompt block / audit ref combination
# ---------------------------------------------------------------------------


def test_merge_appends_second_bank_as_own_paragraph() -> None:
    statement, ref = _merge_text_delivery(
        statement="Personal statement.",
        statement_ref="personal-ref",
        extra_statement="Relationship statement.",
        extra_statement_ref="relationship-ref",
    )
    assert statement == "Personal statement.\n\nRelationship statement."
    assert ref == "personal-ref;relationship-ref"


def test_merge_with_cold_personal_delivers_relationship_alone() -> None:
    statement, ref = _merge_text_delivery(
        statement="",
        statement_ref="",
        extra_statement="Relationship statement.",
        extra_statement_ref="relationship-ref",
    )
    assert statement == "Relationship statement."
    assert ref == "relationship-ref"


def test_merge_without_second_bank_is_identity() -> None:
    statement, ref = _merge_text_delivery(
        statement="Personal statement.",
        statement_ref="personal-ref",
        extra_statement="",
        extra_statement_ref="",
    )
    assert statement == "Personal statement."
    assert ref == "personal-ref"


# ---------------------------------------------------------------------------
# Multi-bank lineage
# ---------------------------------------------------------------------------


def test_lineage_records_both_banks_sorted_with_versioned_router() -> None:
    personal_bank = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
    )
    relationship_bank = bank_readout_to_bank(
        readout=_relationship_readout(),
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
    )

    # Deliberately passed in reverse order: the lineage must sort by bank
    # type so identical bank sets always produce identical rows.
    lineage = build_conditioning_lineage(
        session_scope="session-1",
        banks=(relationship_bank, personal_bank),
        router_version=_STATIC_ROUTER_VERSION,
    )

    assert lineage is not None
    assert lineage.selected_bank_set == ("personal", "relationship")
    assert lineage.bank_fingerprints == (
        ("personal", bank_fingerprint(personal_bank)),
        ("relationship", bank_fingerprint(relationship_bank)),
    )
    assert lineage.router_version == "static-all.v1"


def test_lineage_single_personal_bank_stays_single() -> None:
    personal_bank = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
    )
    lineage = build_conditioning_lineage(
        session_scope="session-1",
        banks=(personal_bank,),
        router_version=_STATIC_ROUTER_VERSION,
    )
    assert lineage is not None
    assert lineage.selected_bank_set == ("personal",)
    assert lineage.router_version == "static-all.v1"


def test_revoked_relationship_bank_never_reaches_lineage() -> None:
    revoked_bank = bank_readout_to_bank(
        readout=_relationship_readout(),
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
        revocation_state=ConditioningRevocationState.REVOKED,
    )
    assert revoked_bank.is_injectable is False
    lineage = build_conditioning_lineage(
        session_scope="session-1",
        banks=(revoked_bank,),
        router_version=_STATIC_ROUTER_VERSION,
    )
    assert lineage is None


def test_zero_freshness_bank_is_expired_and_never_reaches_lineage() -> None:
    expired_bank = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
        freshness=0.0,
    )

    assert expired_bank.is_injectable is False
    assert build_conditioning_lineage(
        session_scope="session-1",
        banks=(expired_bank,),
        router_version=_STATIC_ROUTER_VERSION,
    ) is None


# ---------------------------------------------------------------------------
# P4-c Top-K semantic routing
# ---------------------------------------------------------------------------


def test_topk_router_scores_all_injectable_banks_and_selects_highest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    personal_bank = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
        freshness=0.5,
    )
    relationship_bank = bank_readout_to_bank(
        readout=_relationship_readout(confidence=0.6),
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
        freshness=0.8,
    )
    relevance_by_statement = {
        personal_bank.rendered_statement: 0.9,
        relationship_bank.rendered_statement: 0.7,
    }
    monkeypatch.setattr(
        "volvence_zero.temporal.conditioning_router.semantic_topic_similarity",
        lambda _query, statement: relevance_by_statement[statement],
    )

    decision = select_conditioning_banks(
        user_input="How should we repair this?",
        banks=(relationship_bank, personal_bank),
        k=1,
    )

    assert decision.router_version == TOPK_SEMANTIC_ROUTER_VERSION
    assert decision.selected_bank_set == ("relationship",)
    assert decision.scores == (
        ("personal", pytest.approx(0.9 * 0.7 * 0.5)),
        ("relationship", pytest.approx(0.7 * 0.6 * 0.8)),
    )


def test_topk_router_breaks_score_ties_by_bank_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    personal_bank = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
    )
    relationship_bank = bank_readout_to_bank(
        readout=_relationship_readout(confidence=0.7),
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
    )
    monkeypatch.setattr(
        "volvence_zero.temporal.conditioning_router.semantic_topic_similarity",
        lambda _query, _statement: 1.0,
    )

    decision = select_conditioning_banks(
        user_input="same score",
        banks=(relationship_bank, personal_bank),
        k=1,
    )

    assert decision.selected_bank_set == ("personal",)


def test_topk_router_applies_injectable_hard_gate_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_personal = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
    )
    revoked_relationship = bank_readout_to_bank(
        readout=_relationship_readout(),
        slot_name=RELATIONSHIP_CONDITIONING_SLOT,
        scope=_scope(),
        revocation_state=ConditioningRevocationState.REVOKED,
    )
    scored_statements: list[str] = []

    def _score(_query: str, statement: str) -> float:
        scored_statements.append(statement)
        return 1.0

    monkeypatch.setattr(
        "volvence_zero.temporal.conditioning_router.semantic_topic_similarity",
        _score,
    )
    decision = select_conditioning_banks(
        user_input="route",
        banks=(revoked_relationship, live_personal),
        k=2,
    )

    assert decision.selected_bank_set == ("personal",)
    assert decision.scores == (("personal", pytest.approx(0.7)),)
    assert scored_statements == [live_personal.rendered_statement]


def test_topk_router_fails_loudly_for_broken_injectable_publisher() -> None:
    from dataclasses import replace

    broken = replace(
        personal_conditioning_to_bank(
            snapshot=_personal_snapshot(),
            scope=_scope(),
        ),
        rendered_statement="",
    )

    with pytest.raises(ValueError, match="rendered_statement"):
        select_conditioning_banks(
            user_input="route",
            banks=(broken,),
            k=1,
        )


def test_router_config_defaults_to_shadow_with_bounded_top_k() -> None:
    from volvence_zero.integration.final_wiring import FinalRolloutConfig
    from volvence_zero.runtime import WiringLevel

    config = FinalRolloutConfig()
    assert config.conditioning_router is WiringLevel.SHADOW
    assert config.conditioning_router_top_k >= 1
    with pytest.raises(ValueError, match="conditioning_router_top_k"):
        FinalRolloutConfig(conditioning_router_top_k=0)


def test_lineage_separates_active_and_shadow_router_audit() -> None:
    personal_bank = personal_conditioning_to_bank(
        snapshot=_personal_snapshot(),
        scope=_scope(),
    )
    scores = (("personal", 0.42),)

    active = build_conditioning_lineage(
        session_scope="session-1",
        banks=(personal_bank,),
        router_version=TOPK_SEMANTIC_ROUTER_VERSION,
        router_scores=scores,
    )
    shadow = build_conditioning_lineage(
        session_scope="session-1",
        banks=(personal_bank,),
        router_version=_STATIC_ROUTER_VERSION,
        shadow_router_version=TOPK_SEMANTIC_ROUTER_VERSION,
        shadow_router_scores=scores,
    )

    assert active is not None
    assert active.router_version == TOPK_SEMANTIC_ROUTER_VERSION
    assert active.router_scores == scores
    assert active.shadow_router_version == ""
    assert shadow is not None
    assert shadow.router_version == _STATIC_ROUTER_VERSION
    assert shadow.router_scores == ()
    assert shadow.shadow_router_version == TOPK_SEMANTIC_ROUTER_VERSION
    assert shadow.shadow_router_scores == scores


@pytest.mark.parametrize(
    (
        "profile_label",
        "personal_level",
        "personal_mode",
        "relationship_level",
        "relationship_mode",
    ),
    (
        ("state-kv-bank-none", "shadow", "text", "disabled", "text"),
        ("state-kv-bank-personal-only", "active", "text", "disabled", "text"),
        (
            "state-kv-bank-relationship-only",
            "shadow",
            "text",
            "active",
            "text",
        ),
        (
            "state-kv-bank-relationship-latent-pure",
            "shadow",
            "residual",
            "active",
            "residual",
        ),
        ("state-kv-bank-dual", "active", "text", "active", "text"),
    ),
)
def test_bank_gain_profiles_hold_carrier_and_dynamic_residual_fixed(
    profile_label: str,
    personal_level: str,
    personal_mode: str,
    relationship_level: str,
    relationship_mode: str,
) -> None:
    from volvence_zero.agent.dialogue import (
        DEFAULT_DIALOGUE_PROOF_CASES,
        build_standard_dialogue_runner,
    )

    runner = build_standard_dialogue_runner(
        profile_label=profile_label,
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
    )
    config = runner._config
    assert config.personal_conditioning.value == personal_level
    assert config.personal_conditioning_mode == personal_mode
    assert config.relationship_conditioning.value == relationship_level
    assert config.relationship_conditioning_mode == relationship_mode
    assert config.generation_dynamic_residual.value == "disabled"
    assert config.conditioning_router.value == "shadow"
    assert config.conditioning_router_top_k == 1
    assert config.prompt_state_delivery == (
        "suppressed"
        if profile_label == "state-kv-bank-relationship-latent-pure"
        else "text"
    )


@pytest.mark.asyncio
async def test_active_router_prunes_delivery_and_records_selected_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volvence_zero.agent.dialogue import (
        DEFAULT_DIALOGUE_PROOF_CASES,
        build_standard_dialogue_runner,
    )

    monkeypatch.setattr(
        "volvence_zero.temporal.conditioning_router.semantic_topic_similarity",
        lambda _query, statement: (
            1.0
            if "dyad relationship estimate" in statement.lower()
            else 0.01
        ),
    )
    runner = build_standard_dialogue_runner(
        profile_label="state-kv-bank-dual-router-active",
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
    )
    await runner.run_turn(
        "I felt dismissed in our last exchange and trust is still fragile."
    )
    await runner.run_turn(
        "Please acknowledge the rupture before suggesting a plan."
    )
    result = await runner.run_turn(
        "How should we repair the agreement while keeping the boundary?"
    )

    lineage = result.dialogue_trace.conditioning_lineage
    assert lineage is not None
    assert lineage.router_version == TOPK_SEMANTIC_ROUTER_VERSION
    assert lineage.selected_bank_set == ("relationship",)
    assert lineage.shadow_router_version == ""
    assert dict(lineage.router_scores).keys() == {
        "personal",
        "relationship",
    }
