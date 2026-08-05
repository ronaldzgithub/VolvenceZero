from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from volvence_zero.agent.steering_human_anchor import (
    STEERING_HUMAN_ANCHOR_CONSENT_SCOPE,
    SteeringAnchorArm,
    SteeringAnchorCaptureUnit,
    SteeringAnchorConsentAttestation,
    SteeringAnchorInternalKeyEntry,
    SteeringAnchorPacketBundle,
    SteeringAnchorPreference,
    SteeringAnchorPrivacyAttestation,
    SteeringAnchorRaterAttestation,
    SteeringAnchorRating,
    SteeringAnchorTurn,
    SteeringAnchorUnratableReason,
    SteeringAnchorWithdrawal,
    analyze_steering_human_anchor_pilot,
    apply_steering_anchor_withdrawals,
    build_steering_human_anchor_pilot_packet,
)
from volvence_zero.steering_contracts import SteeringTerminalPredictionError


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _settlement(
    *,
    index: int,
    decision_id: str,
    improvement: float,
) -> SteeringTerminalPredictionError:
    return SteeringTerminalPredictionError(
        episode_id=f"episode-{index:02d}",
        decision_ids=(decision_id,),
        action_batch_id=f"action-{index:02d}",
        noop_batch_id=f"noop-{index:02d}",
        sample_ids=(f"sample-{index:02d}",),
        prediction_head_fingerprint="1" * 64,
        target_lineage_fingerprint="2" * 64,
        target_model_id="tiny-steering-model",
        target_model_weights_sha256="3" * 64,
        action_mean_squared_error=0.1 if improvement > 0 else 0.4,
        noop_mean_squared_error=0.4 if improvement > 0 else 0.1,
        relative_mse_improvement=improvement,
        action_mean_cosine_similarity=0.8 if improvement > 0 else 0.2,
        noop_mean_cosine_similarity=0.2 if improvement > 0 else 0.8,
        cosine_error_improvement=0.6 if improvement > 0 else -0.6,
        terminal=True,
        description="Matched terminal PE for C2 test.",
    )


def _materials() -> tuple[
    tuple[SteeringAnchorCaptureUnit, ...],
    tuple[SteeringAnchorInternalKeyEntry, ...],
]:
    captures = []
    keys = []
    for index in range(48):
        unit_id = _digest(f"unit-{index:02d}")
        decision_id = f"gate-decision-{index:02d}"
        improvement = 0.5 if index % 2 == 0 else -0.5
        steered_arm = SteeringAnchorArm.A if index % 3 else SteeringAnchorArm.B
        capture = SteeringAnchorCaptureUnit(
            unit_id=unit_id,
            episode_ref_sha256=_digest(f"episode-{index:02d}"),
            decision_ref_sha256=_digest(decision_id),
            capture_source_sha256=_digest(f"source-{index:02d}"),
            captured_at_unix_ms=1_800_000_000_000,
            context_turns=(
                SteeringAnchorTurn(
                    speaker="user",
                    text=f"Deidentified context {index:02d}.",
                ),
            ),
            arm_a_response=f"Deidentified response A {index:02d}.",
            arm_b_response=f"Deidentified response B {index:02d}.",
            consent=SteeringAnchorConsentAttestation(
                consent_record_sha256=_digest(f"consent-{index:02d}"),
                consent_document_sha256=_digest("consent-document-v1"),
                subject_ref_sha256=_digest(f"subject-{index:02d}"),
                scope=STEERING_HUMAN_ANCHOR_CONSENT_SCOPE,
                active_at_capture=True,
                learning_use_authorized=False,
                withdrawal_channel_sha256=_digest("withdrawal-channel"),
                retention_deadline_unix_ms=1_900_000_000_000,
            ),
            privacy=SteeringAnchorPrivacyAttestation(
                review_artifact_sha256=_digest(f"privacy-{index:02d}"),
                reviewer_role="independent-privacy-reviewer",
                human_review_completed=True,
                direct_identifiers_removed=True,
                quasi_identifiers_generalized=True,
                third_party_content_cleared=True,
                raw_source_excluded=True,
            ),
        )
        settlement = _settlement(
            index=index,
            decision_id=decision_id,
            improvement=improvement,
        )
        captures.append(capture)
        keys.append(
            SteeringAnchorInternalKeyEntry.from_terminal_prediction_error(
                unit_id=unit_id,
                decision_id=decision_id,
                steered_arm=steered_arm,
                policy_version=1,
                settlement=settlement,
            )
        )
    return tuple(captures), tuple(keys)


def _raters() -> tuple[SteeringAnchorRaterAttestation, ...]:
    return tuple(
        SteeringAnchorRaterAttestation(
            rater_id=rater_id,
            eligibility_review_sha256=_digest(f"eligibility-{rater_id}"),
            human_expert_attested=True,
            domain_expertise_attested=True,
            independent_from_policy_training_attested=True,
        )
        for rater_id in ("expert-a", "expert-b")
    )


def _rating(
    *,
    unit_id: str,
    key: SteeringAnchorInternalKeyEntry,
    rater: SteeringAnchorRaterAttestation,
    align_with_pe: bool,
) -> SteeringAnchorRating:
    pe_positive = key.relative_mse_improvement > 0
    prefer_steered = pe_positive if align_with_pe else not pe_positive
    noop_arm = (
        SteeringAnchorArm.B
        if key.steered_arm is SteeringAnchorArm.A
        else SteeringAnchorArm.A
    )
    preferred = key.steered_arm if prefer_steered else noop_arm
    steer_relationship = 5 if prefer_steered else 2
    noop_relationship = 2 if prefer_steered else 5
    scores = {
        "a_relationship": (
            steer_relationship
            if key.steered_arm is SteeringAnchorArm.A
            else noop_relationship
        ),
        "b_relationship": (
            steer_relationship
            if key.steered_arm is SteeringAnchorArm.B
            else noop_relationship
        ),
    }
    return SteeringAnchorRating(
        unit_id=unit_id,
        rater_id=rater.rater_id,
        eligibility_review_sha256=rater.eligibility_review_sha256,
        preferred_arm=SteeringAnchorPreference(preferred.value),
        arm_a_relationship_support=scores["a_relationship"],
        arm_b_relationship_support=scores["b_relationship"],
        arm_a_boundary_respect=4,
        arm_b_boundary_respect=4,
        arm_a_task_preservation=4,
        arm_b_task_preservation=4,
        unratable_reason=SteeringAnchorUnratableReason.NONE,
    )


def _ratings(
    *,
    keys: tuple[SteeringAnchorInternalKeyEntry, ...],
    align_with_pe: bool,
) -> tuple[SteeringAnchorRating, ...]:
    raters = _raters()
    return tuple(
        _rating(
            unit_id=key.unit_id,
            key=key,
            rater=rater,
            align_with_pe=align_with_pe,
        )
        for key in keys
        for rater in raters
    )


def test_c2_packet_is_blinded_hashed_and_validation_only() -> None:
    captures, keys = _materials()
    bundle = build_steering_human_anchor_pilot_packet(
        captures=captures,
        internal_keys=keys,
        created_at_unix_ms=1_810_000_000_000,
    )
    assert bundle.public_packet["unit_count"] == 48
    assert len(bundle.rating_template) == 96
    assert bundle.public_packet["learning_use_authorized"] is False
    assert bundle.manifest["production_promotion_authorized"] is False
    packet_text = str(bundle.public_packet)
    assert "relative_mse_improvement" not in packet_text
    assert "steered_arm" not in packet_text
    assert "consent_record_sha256" not in packet_text


def test_c2_agreement_and_directional_alignment_pass_independently() -> None:
    captures, keys = _materials()
    bundle = build_steering_human_anchor_pilot_packet(
        captures=captures,
        internal_keys=keys,
        created_at_unix_ms=1_810_000_000_000,
    )
    report = analyze_steering_human_anchor_pilot(
        bundle=bundle,
        raters=_raters(),
        ratings=_ratings(keys=keys, align_with_pe=True),
    )
    assert report.exact_agreement == 1.0
    assert report.cohen_kappa == 1.0
    assert report.agreement_gate_passed is True
    assert report.resolved_direction_count == 48
    assert report.directional_alignment_rate == 1.0
    assert report.c1_alignment_review_required is False
    assert report.expansion_admissible is True
    assert report.learning_use_authorized is False


def test_c2_disagreement_with_pe_triggers_review_but_never_learning() -> None:
    captures, keys = _materials()
    bundle = build_steering_human_anchor_pilot_packet(
        captures=captures,
        internal_keys=keys,
        created_at_unix_ms=1_810_000_000_000,
    )
    report = analyze_steering_human_anchor_pilot(
        bundle=bundle,
        raters=_raters(),
        ratings=_ratings(keys=keys, align_with_pe=False),
    )
    assert report.agreement_gate_passed is True
    assert report.directional_alignment_rate == 0.0
    assert report.c1_alignment_review_required is True
    assert report.learning_use_authorized is False
    assert report.production_promotion_authorized is False


def test_c2_withdrawal_removes_content_and_requires_packet_rebuild() -> None:
    captures, keys = _materials()
    withdrawal = SteeringAnchorWithdrawal(
        consent_record_sha256=captures[0].consent.consent_record_sha256,
        requested_at_unix_ms=1_811_000_000_000,
        completed_at_unix_ms=1_811_000_001_000,
        raw_source_deleted=True,
        capture_removed=True,
        ratings_removed=True,
        reidentification_mapping_deleted=True,
    )
    retained, tombstones = apply_steering_anchor_withdrawals(
        captures=captures,
        withdrawals=(withdrawal,),
    )
    assert len(retained) == 47
    assert tombstones[0]["content_retained"] is False
    with pytest.raises(ValueError, match="exactly 48"):
        build_steering_human_anchor_pilot_packet(
            captures=retained,
            internal_keys=keys[1:],
            created_at_unix_ms=1_812_000_000_000,
        )


def test_c2_consent_cannot_authorize_learning_use() -> None:
    captures, _ = _materials()
    consent = captures[0].consent
    with pytest.raises(ValueError, match="must not authorize learning"):
        replace(consent, learning_use_authorized=True)


def test_c2_packet_tamper_fails_before_analysis() -> None:
    captures, keys = _materials()
    bundle = build_steering_human_anchor_pilot_packet(
        captures=captures,
        internal_keys=keys,
        created_at_unix_ms=1_810_000_000_000,
    )
    tampered_packet = dict(bundle.public_packet)
    tampered_packet["unit_count"] = 47
    tampered = SteeringAnchorPacketBundle(
        public_packet=tampered_packet,
        internal_key=bundle.internal_key,
        rating_template=bundle.rating_template,
        manifest=bundle.manifest,
    )
    with pytest.raises(ValueError, match="hash drift"):
        analyze_steering_human_anchor_pilot(
            bundle=tampered,
            raters=_raters(),
            ratings=_ratings(keys=keys, align_with_pe=True),
        )
