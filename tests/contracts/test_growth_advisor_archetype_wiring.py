"""Contract test: D1 archetype classifier wiring (debt #66).

Validates the wiring contract between
:class:`LLMArchetypeClassifier` and
:func:`build_growth_advisor_lifeform`:

1. ``build_growth_advisor_lifeform(archetype_classifier=...)``
   accepts the new optional kwarg and stores it on the bundle.
2. ``GrowthAdvisorLifeformBundle.maybe_classify_archetype`` is a
   no-op when no classifier is wired (back-compat with v0.1).
3. ``maybe_classify_archetype`` invokes the classifier exactly
   every ``call_every_n_turns`` turns (default 3); skipped turns
   do NOT call the LLM provider.
4. After enough turns, ``classifier.state_for(end_user_id)``
   returns a frozen ``ArchetypeStateSnapshot`` (R8 SSOT — owner
   is the classifier, not the bundle).
5. Fake provider returning a malformed JSON raises with context
   (no-swallow-errors invariant).

Refs:

* docs/known-debts.md #66
* docs/specs/growth-advisor-archetype-detection.md
* packages/lifeform-domain-growth-advisor/.../archetype_classifier.py
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Mapping

import pytest

from lifeform_domain_growth_advisor.archetype_classifier import (
    ARCHETYPE_IDS,
    ArchetypeClassification,
    ArchetypeStateSnapshot,
    LLMArchetypeClassifier,
    LLMArchetypeClassifierConfig,
)
from lifeform_domain_growth_advisor.lifeform_builder import (
    build_growth_advisor_lifeform,
)


# ---------------------------------------------------------------------------
# Fake LLM provider — deterministic JSON; counts calls so we can assert
# the every-N-turns cadence.
# ---------------------------------------------------------------------------


class _CountingFakeProvider:
    """Returns a fixed valid classification JSON; counts invocations."""

    def __init__(self, primary: str = "anxious") -> None:
        self.calls = 0
        self._primary = primary

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps(
            {
                "primary": self._primary,
                "probabilities": {
                    "anxious": 0.20,
                    "comparing": 0.20,
                    "standard_seeking": 0.20,
                    "venting": 0.20,
                    "product_seeking": 0.20,
                },
                "confidence": 0.40,
                "rationale": f"fake (call #{self.calls})",
            }
        )


def _classifier(*, every_n: int = 3, min_turns: int = 2) -> LLMArchetypeClassifier:
    fake = _CountingFakeProvider()
    config = LLMArchetypeClassifierConfig(
        call_every_n_turns=every_n,
        min_turns_to_classify=min_turns,
    )
    classifier = LLMArchetypeClassifier(llm_provider=fake, config=config)
    classifier._fake_provider = fake  # surface for assertion
    return classifier


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_build_growth_advisor_lifeform_accepts_archetype_classifier() -> None:
    classifier = _classifier()
    bundle = build_growth_advisor_lifeform(archetype_classifier=classifier)
    assert bundle.archetype_classifier is classifier


def test_bundle_maybe_classify_no_op_without_classifier() -> None:
    """v0.1 back-compat: bundles built without a classifier do nothing."""
    bundle = build_growth_advisor_lifeform()
    snapshot = bundle.maybe_classify_archetype(
        end_user_id="alice",
        recent_user_turns=("hi", "i need help with my child's height"),
    )
    assert snapshot is None


def test_bundle_invokes_classifier_only_every_n_turns() -> None:
    """Cadence contract: every N=3 turns, classifier fires once."""
    classifier = _classifier(every_n=3, min_turns=1)
    bundle = build_growth_advisor_lifeform(archetype_classifier=classifier)
    fake = classifier._fake_provider
    for i in range(1, 8):  # turns 1..7 (counts 1, 2, 3, 4, 5, 6, 7)
        bundle.maybe_classify_archetype(
            end_user_id="alice",
            recent_user_turns=tuple(f"turn-{j}" for j in range(1, i + 1)),
        )
    # turns 3 and 6 are multiples of 3 → classifier called twice.
    assert fake.calls == 2


def test_classifier_owns_state_snapshot_after_classification() -> None:
    """After enough turns, ``state_for`` returns a frozen typed snapshot."""
    classifier = _classifier(every_n=2, min_turns=1)
    bundle = build_growth_advisor_lifeform(archetype_classifier=classifier)
    bundle.maybe_classify_archetype(
        end_user_id="alice",
        recent_user_turns=("turn-1",),
    )
    bundle.maybe_classify_archetype(
        end_user_id="alice",
        recent_user_turns=("turn-1", "turn-2"),
    )
    snapshot = classifier.state_for("alice")
    assert snapshot is not None
    assert isinstance(snapshot, ArchetypeStateSnapshot)
    # Frozen snapshot — assignment must raise.
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.stability = 0.5  # type: ignore[misc]
    assert snapshot.current.primary in ARCHETYPE_IDS
    assert 0.0 <= snapshot.stability <= 1.0


def test_per_user_turn_counters_isolated() -> None:
    """One end_user crossing the N boundary doesn't trigger classify for others."""
    classifier = _classifier(every_n=3, min_turns=1)
    bundle = build_growth_advisor_lifeform(archetype_classifier=classifier)
    fake = classifier._fake_provider
    # Drive alice 3 turns (hits boundary at turn 3 → 1 call)
    for i in range(1, 4):
        bundle.maybe_classify_archetype(
            end_user_id="alice",
            recent_user_turns=tuple(f"a-{j}" for j in range(1, i + 1)),
        )
    # Bob has only 2 turns so no call for bob.
    for i in range(1, 3):
        bundle.maybe_classify_archetype(
            end_user_id="bob",
            recent_user_turns=tuple(f"b-{j}" for j in range(1, i + 1)),
        )
    assert fake.calls == 1
    assert classifier.state_for("alice") is not None
    assert classifier.state_for("bob") is None


def test_malformed_provider_output_raises_with_context() -> None:
    """no-swallow-errors: bad JSON propagates ValueError with context."""

    def bad_provider(prompt: str) -> str:
        del prompt
        return "not-json-at-all"

    classifier = LLMArchetypeClassifier(
        llm_provider=bad_provider,
        config=LLMArchetypeClassifierConfig(min_turns_to_classify=1),
    )
    with pytest.raises(ValueError, match="failed to parse"):
        classifier.classify(
            recent_user_turns=("turn-1",),
            end_user_id="alice",
        )


# ---------------------------------------------------------------------------
# R8: classifier is the SSOT — downstream reads via state_for, never
# reconstructs from raw turns.
# ---------------------------------------------------------------------------


def test_bundle_does_not_expose_archetype_state_directly() -> None:
    """R8 boundary: bundle has no parallel ``archetype_state`` field.

    Catches the regression of "downstream consumer started caching the
    state on the bundle itself rather than reading from the classifier."
    """

    bundle = build_growth_advisor_lifeform()
    fields = {f.name for f in dataclasses.fields(bundle)}
    forbidden = {"archetype_state", "archetype_state_snapshot"}
    overlap = fields & forbidden
    assert not overlap, (
        f"GrowthAdvisorLifeformBundle must not own archetype_state directly; "
        f"R8 SSOT is the classifier. Forbidden fields present: {sorted(overlap)}"
    )


def test_archetype_classification_protocol_returns_typed_payload() -> None:
    """Smoke: directly invoking classifier.classify(...) returns typed dataclass."""
    classifier = _classifier(every_n=1, min_turns=1)
    result = classifier.classify(
        recent_user_turns=("worried about my child's growth",),
        end_user_id="alice",
    )
    assert isinstance(result, ArchetypeClassification)
    assert result.primary in ARCHETYPE_IDS
    assert isinstance(result.probabilities, Mapping)
    assert all(k in result.probabilities for k in ARCHETYPE_IDS)
