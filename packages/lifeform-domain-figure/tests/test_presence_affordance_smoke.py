"""Smoke + contract tests for the talking_head_video.v1 SHELL descriptor."""

from __future__ import annotations

from lifeform_affordance import AffordanceKind

from lifeform_domain_figure import (
    CONSENT_LIKENESS_RENDER,
    CONSENT_VOICE_CLONE,
    PRESENCE_AFFORDANCE_DESCRIPTORS,
    TALKING_HEAD_VIDEO_DESCRIPTOR,
)


def test_talking_head_descriptor_is_shell_kind() -> None:
    assert TALKING_HEAD_VIDEO_DESCRIPTOR.kind == AffordanceKind.SHELL


def test_talking_head_descriptor_requires_both_consents() -> None:
    grants = TALKING_HEAD_VIDEO_DESCRIPTOR.safety_model.requires_consent_grant
    assert CONSENT_LIKENESS_RENDER in grants
    assert CONSENT_VOICE_CLONE in grants


def test_talking_head_descriptor_is_irreversible() -> None:
    # A frame that left the lab to a third-party SFU cannot be unsent;
    # the descriptor encodes this so the modification gate forces the
    # audit_required path.
    assert TALKING_HEAD_VIDEO_DESCRIPTOR.safety_model.irreversible is True
    assert TALKING_HEAD_VIDEO_DESCRIPTOR.safety_model.audit_required is True


def test_talking_head_descriptor_blocks_minor_and_crisis_regimes() -> None:
    blocked = TALKING_HEAD_VIDEO_DESCRIPTOR.safety_model.blocked_in_regimes
    assert "minor_user" in blocked
    assert "crisis" in blocked


def test_talking_head_descriptor_no_per_utterance_user_confirmation() -> None:
    # Confirmation happens at session-open time on the rendering plane;
    # per-utterance prompts would make the conversation unusable.
    assert (
        TALKING_HEAD_VIDEO_DESCRIPTOR.safety_model.requires_user_confirmation
        is False
    )


def test_presence_affordance_descriptors_export() -> None:
    assert TALKING_HEAD_VIDEO_DESCRIPTOR in PRESENCE_AFFORDANCE_DESCRIPTORS


def test_talking_head_descriptor_parameter_schema_required_fields() -> None:
    schema = TALKING_HEAD_VIDEO_DESCRIPTOR.parameters_schema
    required = set(schema.get("required", ()))
    assert {"persona_id", "session_token", "scope_key"} <= required


def test_talking_head_descriptor_when_to_use_long_enough() -> None:
    # AffordanceDescriptor.__post_init__ enforces ≥ 50 chars but we
    # additionally want to lock that the actual text mentions the
    # likeness consent gate, since a short string would also pass the
    # length check while losing the operational warning.
    assert "consent" in TALKING_HEAD_VIDEO_DESCRIPTOR.when_to_use
    assert "regime" in TALKING_HEAD_VIDEO_DESCRIPTOR.when_to_use
