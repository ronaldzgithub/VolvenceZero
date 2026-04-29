"""Contract tests for ``AffordanceDescriptor`` invariants (Gap 1).

Authoring discipline enforced at construction time:

* ``when_to_use`` / ``when_not_to_use`` must each be \u2265 50 chars
  (MIN_SELECTION_HINT_CHARS). This is the single biggest knob for
  LLM selection quality; the rule exists because shorter hints
  consistently degrade selection accuracy.
* ``parameters_schema`` must be a Mapping with a ``type`` key
  (JSON-Schema-shaped).
* ``AffordanceKind`` / ``AffordanceLatencyClass`` /
  ``AffordanceMonetaryClass`` enums are finite and exhaustively
  named so schema drift is caught early.
* ``AffordanceSafety.requires_consent_grant`` entries must be
  non-empty, unique strings.
* ``affordance_tags`` must be unique.
"""

from __future__ import annotations

import pytest

from volvence_zero.affordance import (
    MIN_SELECTION_HINT_CHARS,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


def test_affordance_kind_values_are_exhaustive() -> None:
    assert set(AffordanceKind) == {
        AffordanceKind.TOOL,
        AffordanceKind.ACTION,
        AffordanceKind.ORGAN,
        AffordanceKind.SHELL,
    }


def test_affordance_latency_class_values_are_exhaustive() -> None:
    assert set(AffordanceLatencyClass) == {
        AffordanceLatencyClass.INSTANT,
        AffordanceLatencyClass.FAST,
        AffordanceLatencyClass.SLOW,
        AffordanceLatencyClass.VERY_SLOW,
    }


def test_affordance_monetary_class_values_are_exhaustive() -> None:
    assert set(AffordanceMonetaryClass) == {
        AffordanceMonetaryClass.FREE,
        AffordanceMonetaryClass.LOW,
        AffordanceMonetaryClass.MEDIUM,
        AffordanceMonetaryClass.HIGH,
    }


def test_min_selection_hint_chars_is_fifty() -> None:
    """Locked to 50. Changing this value requires re-validating every
    registered descriptor and this test reminds the author.
    """
    assert MIN_SELECTION_HINT_CHARS == 50


# ---------------------------------------------------------------------------
# Cost invariants
# ---------------------------------------------------------------------------


def test_cost_rejects_nonpositive_rate_limit() -> None:
    with pytest.raises(ValueError, match="rate_limit_per_minute"):
        AffordanceCost(
            latency_class=AffordanceLatencyClass.FAST, rate_limit_per_minute=0
        )
    with pytest.raises(ValueError, match="rate_limit_per_minute"):
        AffordanceCost(
            latency_class=AffordanceLatencyClass.FAST, rate_limit_per_minute=-1
        )


def test_cost_accepts_none_rate_limit() -> None:
    cost = AffordanceCost(latency_class=AffordanceLatencyClass.FAST)
    assert cost.rate_limit_per_minute is None
    assert cost.monetary_class is AffordanceMonetaryClass.FREE


# ---------------------------------------------------------------------------
# Safety invariants
# ---------------------------------------------------------------------------


def test_safety_rejects_empty_consent_grant_name() -> None:
    with pytest.raises(ValueError, match="requires_consent_grant"):
        AffordanceSafety(requires_consent_grant=("   ",))


def test_safety_rejects_duplicate_consent_grant_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        AffordanceSafety(requires_consent_grant=("tool_use", "tool_use"))


def test_safety_rejects_empty_blocked_regime() -> None:
    with pytest.raises(ValueError, match="blocked_in_regimes"):
        AffordanceSafety(blocked_in_regimes=("",))


def test_safety_defaults_are_permissive() -> None:
    safety = AffordanceSafety()
    assert safety.requires_user_confirmation is False
    assert safety.irreversible is False
    assert safety.requires_consent_grant == ()
    assert safety.blocked_in_regimes == ()
    assert safety.audit_required is False


# ---------------------------------------------------------------------------
# Descriptor invariants
# ---------------------------------------------------------------------------


def _good_hint() -> str:
    # Exactly 50+ chars; tests vary length explicitly where needed.
    return (
        "This is a deliberately long enough hint so the "
        "AffordanceDescriptor post_init check will accept it."
    )


def _good_descriptor(**overrides: object) -> AffordanceDescriptor:
    defaults: dict[str, object] = dict(
        name="read_file",
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Read file",
        description="Read a UTF-8 text file from the workspace.",
        when_to_use=_good_hint(),
        when_not_to_use=_good_hint() + " (negative).",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        safety_model=AffordanceSafety(),
    )
    defaults.update(overrides)
    return AffordanceDescriptor(**defaults)


def test_descriptor_accepts_well_formed_construction() -> None:
    d = _good_descriptor()
    assert d.name == "read_file"
    assert d.kind is AffordanceKind.TOOL


def test_descriptor_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _good_descriptor(name="")


def test_descriptor_rejects_empty_version() -> None:
    with pytest.raises(ValueError, match="version"):
        _good_descriptor(version="")


def test_descriptor_rejects_empty_display_name() -> None:
    with pytest.raises(ValueError, match="display_name"):
        _good_descriptor(display_name="  ")


def test_descriptor_rejects_short_when_to_use() -> None:
    # Just under the threshold.
    short_hint = "a" * (MIN_SELECTION_HINT_CHARS - 1)
    with pytest.raises(ValueError, match="when_to_use"):
        _good_descriptor(when_to_use=short_hint)


def test_descriptor_rejects_short_when_not_to_use() -> None:
    short_hint = "b" * (MIN_SELECTION_HINT_CHARS - 1)
    with pytest.raises(ValueError, match="when_not_to_use"):
        _good_descriptor(when_not_to_use=short_hint)


def test_descriptor_accepts_exactly_threshold_hints() -> None:
    exact_hint = "x" * MIN_SELECTION_HINT_CHARS
    d = _good_descriptor(when_to_use=exact_hint, when_not_to_use=exact_hint + "_neg")
    assert len(d.when_to_use) == MIN_SELECTION_HINT_CHARS


def test_descriptor_rejects_parameters_schema_without_type() -> None:
    with pytest.raises(ValueError, match="parameters_schema"):
        _good_descriptor(parameters_schema={"properties": {"x": {"type": "string"}}})


def test_descriptor_rejects_non_mapping_parameters_schema() -> None:
    with pytest.raises(TypeError, match="parameters_schema"):
        _good_descriptor(parameters_schema="not a dict")  # type: ignore[arg-type]


def test_descriptor_rejects_non_mapping_output_schema() -> None:
    with pytest.raises(TypeError, match="output_schema"):
        _good_descriptor(output_schema=["list", "instead", "of", "dict"])  # type: ignore[arg-type]


def test_descriptor_rejects_duplicate_affordance_tags() -> None:
    with pytest.raises(ValueError, match="affordance_tags"):
        _good_descriptor(affordance_tags=("read", "code", "read"))
