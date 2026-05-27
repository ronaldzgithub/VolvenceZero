"""ExperienceLoop platform-contract schemas."""

from __future__ import annotations

import pytest

from dlaas_platform_contracts import (
    CAPABILITY_EXPERIENCE_BINDING,
    ExperienceBriefSpec,
    ExperienceReceiptSpec,
    ExperienceReflectionSpec,
    find_binding_for_brief_kind,
)


def test_experience_brief_accepts_canonical_payload():
    spec = ExperienceBriefSpec.from_json(
        {
            "schema_version": "experience.brief.v1",
            "domain": "marketing",
            "goal": "launch a landing page",
            "kpi": "qualified_signup",
            "channels": ["twitter_organic"],
            "autonomy_level": "draft_only",
        }
    )
    assert spec.domain == "marketing"
    assert spec.to_json()["schema_version"] == "experience.brief.v1"


def test_experience_brief_rejects_invalid_autonomy():
    with pytest.raises(ValueError, match="autonomy_level"):
        ExperienceBriefSpec.from_json(
            {
                "schema_version": "experience.brief.v1",
                "domain": "marketing",
                "goal": "launch",
                "kpi": "signup",
                "autonomy_level": "self_modify_model",
            }
        )


def test_experience_receipt_requires_experience_id():
    with pytest.raises(ValueError, match="experience_id"):
        ExperienceReceiptSpec.from_json(
            {
                "schema_version": "experience.receipt.v1",
                "domain": "reader_dialogue",
                "event_kind": "feedback_observed",
                "summary": "helped",
            }
        )


def test_experience_receipt_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        ExperienceReceiptSpec.from_json(
            {
                "schema_version": "experience.receipt.v1",
                "domain": "reader_dialogue",
                "experience_id": "resp_1",
                "event_kind": "feedback_observed",
                "summary": "helped",
                "confidence": 2.0,
            }
        )


def test_experience_reflection_requires_title():
    with pytest.raises(ValueError, match="title"):
        ExperienceReflectionSpec.from_json(
            {
                "schema_version": "experience.reflection.v1",
                "domain": "marketing",
                "experience_id": "campaign_1",
                "summary": "learned something",
            }
        )


def test_builtin_binding_lookup_round_trips():
    binding = find_binding_for_brief_kind("capability.brief")
    assert binding == CAPABILITY_EXPERIENCE_BINDING
