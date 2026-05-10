"""Contract tests for the rupture_state owner and dialogue_external_outcome slot.

These tests enforce the v0 invariants from
``docs/specs/rupture-and-repair.md``:

* ``RuptureStateSnapshot`` is frozen and rejects out-of-bound fields.
* Every ``RuptureKind`` value has at least one typed non-PE
  ``RuptureEvidenceSource`` that can emit it (enum completeness).
* ``internal_suspected_only`` is only allowed when the sole source is
  ``INTERNAL_PE``; a kind can be resolved only when a non-PE source
  contributes.
* The closed ``ExternalOutcomeKind -> RuptureKind`` mapping is 1:1 for
  every rupture-producing external kind, and ``HELPED`` / ``FELT_HEARD``
  / ``DECISION_CLEARER`` deliberately produce no rupture evidence.
* The ``dialogue_external_outcome`` slot is the single legal channel;
  ``DialogueExternalOutcomeEvidence`` rejects out-of-range confidence.
* The module defaults: ``rupture_state`` at SHADOW,
  ``dialogue_external_outcome`` at ACTIVE.
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.rupture_state import (
    EXTERNAL_OUTCOME_TO_RUPTURE_KIND,
    RUPTURE_KIND_LABEL,
    RUPTURE_KIND_SEVERITY,
    RuptureContributingSignal,
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateModule,
    RuptureStateSnapshot,
    rupture_kind_label,
)


# External outcomes that never produce a turn-level rupture.
#
# Three groups live here:
#
# 1. Positive in-turn signals (``HELPED`` / ``FELT_HEARD`` /
#    ``DECISION_CLEARER``) — by definition not rupture-producing.
# 2. W3-A positive LTV funnel events (``LEAD_QUALIFIED`` /
#    ``RECOMMENDATION_MADE`` / ``PURCHASE_CONFIRMED`` / ``REPURCHASE``)
#    — confirmed business wins; PE / regime treat them as positive.
# 3. ``CHURNED`` — negative but long-horizon. The user has already
#    disengaged, so there is no current turn to anchor a rupture to;
#    the damage is captured by W3-A's negative PE bias + low regime
#    score, and a typed "churn rupture" RuptureKind is intentionally
#    deferred (see the comment in ``EXTERNAL_OUTCOME_TO_RUPTURE_KIND``).
#
# The set name is historical ("positive") but the meaning is
# precisely "non-rupture-producing"; the test below relies on it.
_POSITIVE_EXTERNAL_KINDS: frozenset[DialogueExternalOutcomeKind] = frozenset(
    {
        DialogueExternalOutcomeKind.HELPED,
        DialogueExternalOutcomeKind.FELT_HEARD,
        DialogueExternalOutcomeKind.DECISION_CLEARER,
        DialogueExternalOutcomeKind.LEAD_QUALIFIED,
        DialogueExternalOutcomeKind.RECOMMENDATION_MADE,
        DialogueExternalOutcomeKind.PURCHASE_CONFIRMED,
        DialogueExternalOutcomeKind.REPURCHASE,
        DialogueExternalOutcomeKind.CHURNED,
    }
)


def test_rupture_kind_enum_is_closed() -> None:
    assert set(RuptureKind) == {
        RuptureKind.MISREAD,
        RuptureKind.OVER_DIRECTIVE,
        RuptureKind.PUSHED_TOO_FAST,
        RuptureKind.COLD,
        RuptureKind.UNSAFE,
        RuptureKind.ABANDONED,
    }


def test_rupture_evidence_source_enum_is_closed() -> None:
    assert set(RuptureEvidenceSource) == {
        RuptureEvidenceSource.INTERNAL_PE,
        RuptureEvidenceSource.BEHAVIORAL_TRACE,
        RuptureEvidenceSource.SELF_CHECK_ASSEMBLY,
        RuptureEvidenceSource.EXTERNAL_USER,
        RuptureEvidenceSource.ENVIRONMENT,
        RuptureEvidenceSource.LLM_PROPOSAL,
    }


def test_dialogue_external_outcome_kind_enum_is_closed() -> None:
    """Lock the ``DialogueExternalOutcomeKind`` vocabulary.

    Adding a new value here is a contract change — it MUST come with
    explicit downstream mappings in PE bias, regime score, structural
    projection, and rupture mapping (or a documented opt-out). The
    W3-A LTV block was added to support conversion-funnel evidence
    (CRM / payments) flowing through ``submit_dialogue_outcome`` via
    typed ``FeedbackValence`` — see
    ``packages/dlaas-platform-contracts/src/dlaas_platform_contracts/dispatch_vocab.py``.
    """
    assert set(DialogueExternalOutcomeKind) == {
        DialogueExternalOutcomeKind.HELPED,
        DialogueExternalOutcomeKind.FELT_HEARD,
        DialogueExternalOutcomeKind.MISSED,
        DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        DialogueExternalOutcomeKind.DECISION_CLEARER,
        DialogueExternalOutcomeKind.COME_BACK,
        DialogueExternalOutcomeKind.UNSAFE,
        DialogueExternalOutcomeKind.ABANDONED,
        # W3-A LTV / conversion-funnel block.
        DialogueExternalOutcomeKind.LEAD_QUALIFIED,
        DialogueExternalOutcomeKind.RECOMMENDATION_MADE,
        DialogueExternalOutcomeKind.PURCHASE_CONFIRMED,
        DialogueExternalOutcomeKind.REPURCHASE,
        DialogueExternalOutcomeKind.CHURNED,
    }


def test_every_rupture_kind_has_a_producing_typed_source() -> None:
    """Enum completeness: adding a new RuptureKind requires a typed source.

    Each rupture kind must be reachable via either (a) the closed 1:1
    mapping from an external outcome kind, or (b) an explicit
    compositional rule handled inside the owner (currently only COLD).
    """

    producible_by_mapping = set(EXTERNAL_OUTCOME_TO_RUPTURE_KIND.values())
    # COLD is the only kind intentionally absent from the 1:1 table; it
    # is produced by the owner's compositional rule (MISREAD external +
    # elevated behavioral repair pressure). This test documents that
    # exception so a new kind cannot sneak in without an equivalent
    # typed-source commitment.
    compositional_kinds = {RuptureKind.COLD}
    covered = producible_by_mapping | compositional_kinds
    assert covered == set(RuptureKind), (
        "Every RuptureKind must be reachable via a typed external outcome or a "
        "documented compositional rule. Missing coverage: "
        f"{set(RuptureKind) - covered}"
    )


def test_positive_external_kinds_produce_no_rupture() -> None:
    for kind in _POSITIVE_EXTERNAL_KINDS:
        assert kind not in EXTERNAL_OUTCOME_TO_RUPTURE_KIND, (
            f"{kind.value} is a positive outcome and must not map to any RuptureKind."
        )


def test_external_to_rupture_mapping_is_one_to_one_on_rupture_producing_kinds() -> None:
    rupture_producing = {
        kind
        for kind in DialogueExternalOutcomeKind
        if kind not in _POSITIVE_EXTERNAL_KINDS
    }
    # Every rupture-producing external kind has exactly one mapping.
    for kind in rupture_producing:
        assert kind in EXTERNAL_OUTCOME_TO_RUPTURE_KIND, (
            f"Rupture-producing external kind {kind.value} has no mapping."
        )
    # The mapping is injective (no two external kinds map to the same
    # rupture kind) among rupture-producing externals.
    mapped_values = [
        EXTERNAL_OUTCOME_TO_RUPTURE_KIND[kind] for kind in rupture_producing
    ]
    assert len(set(mapped_values)) == len(mapped_values), (
        "External-to-rupture mapping must be 1:1 (no two external kinds "
        f"map to the same RuptureKind); saw {mapped_values}"
    )


def test_rupture_kind_severity_covers_all_kinds_uniquely() -> None:
    assert set(RUPTURE_KIND_SEVERITY.keys()) == set(RuptureKind)
    severities = list(RUPTURE_KIND_SEVERITY.values())
    assert len(set(severities)) == len(severities), (
        "RUPTURE_KIND_SEVERITY must assign a unique severity to each kind."
    )


def test_rupture_kind_label_covers_every_kind() -> None:
    """W3 SSOT: every RuptureKind has a canonical human-readable label
    in ``RUPTURE_KIND_LABEL``. Adding a new kind without a label
    fails this test before the synthesizer silently falls back to the
    enum value.
    """

    assert set(RUPTURE_KIND_LABEL.keys()) == set(RuptureKind), (
        "RUPTURE_KIND_LABEL must cover every RuptureKind. "
        f"Missing: {set(RuptureKind) - set(RUPTURE_KIND_LABEL.keys())}; "
        f"unexpected: {set(RUPTURE_KIND_LABEL.keys()) - set(RuptureKind)}"
    )
    for kind in RuptureKind:
        assert rupture_kind_label(kind), f"empty label for {kind!r}"
    assert rupture_kind_label(None) == ""


def test_rupture_state_snapshot_publishes_kind_label() -> None:
    """The snapshot publishes ``kind_label`` derived from the
    canonical label map; consumers do not maintain a parallel dict.
    """

    pe_signal = RuptureContributingSignal(
        source=RuptureEvidenceSource.INTERNAL_PE,
        signal_strength=0.4,
        confidence=0.5,
        kind_hint=None,
        detail="pe",
    )
    external_signal = RuptureContributingSignal(
        source=RuptureEvidenceSource.EXTERNAL_USER,
        signal_strength=0.9,
        confidence=0.95,
        kind_hint=RuptureKind.OVER_DIRECTIVE,
        detail="user signal",
    )
    snapshot = RuptureStateSnapshot(
        rupture_signal_strength=0.9,
        rupture_kind=RuptureKind.OVER_DIRECTIVE,
        confidence=0.7,
        internal_suspected_only=False,
        evidence_sources=(
            RuptureEvidenceSource.INTERNAL_PE,
            RuptureEvidenceSource.EXTERNAL_USER,
        ),
        contributing_signals=(pe_signal, external_signal),
        description="rupture",
    )
    assert snapshot.kind_label == "over-directive"
    # And the bootstrap (no rupture) snapshot has empty label.
    bootstrap = RuptureStateSnapshot(
        rupture_signal_strength=0.0,
        rupture_kind=None,
        confidence=0.0,
        internal_suspected_only=False,
        evidence_sources=(),
        contributing_signals=(),
        description="no rupture",
    )
    assert bootstrap.kind_label == ""


def test_rupture_state_snapshot_overwrites_caller_supplied_label() -> None:
    """A caller cannot drift the label by hand; ``__post_init__`` always
    rewrites it to the canonical value.
    """

    pe_signal = RuptureContributingSignal(
        source=RuptureEvidenceSource.INTERNAL_PE,
        signal_strength=0.4,
        confidence=0.5,
        kind_hint=None,
        detail="pe",
    )
    external_signal = RuptureContributingSignal(
        source=RuptureEvidenceSource.EXTERNAL_USER,
        signal_strength=0.9,
        confidence=0.95,
        kind_hint=RuptureKind.MISREAD,
        detail="user signal",
    )
    snapshot = RuptureStateSnapshot(
        rupture_signal_strength=0.9,
        rupture_kind=RuptureKind.MISREAD,
        confidence=0.7,
        internal_suspected_only=False,
        evidence_sources=(
            RuptureEvidenceSource.INTERNAL_PE,
            RuptureEvidenceSource.EXTERNAL_USER,
        ),
        contributing_signals=(pe_signal, external_signal),
        description="rupture",
        kind_label="WRONG LABEL",
    )
    assert snapshot.kind_label == "a misread"


def test_rupture_state_snapshot_is_frozen() -> None:
    snapshot = RuptureStateSnapshot(
        rupture_signal_strength=0.0,
        rupture_kind=None,
        confidence=0.0,
        internal_suspected_only=False,
        evidence_sources=(),
        contributing_signals=(),
        description="no rupture",
    )
    assert dataclasses.is_dataclass(snapshot)
    assert snapshot.__dataclass_params__.frozen


def test_rupture_state_snapshot_rejects_kind_without_non_pe_source() -> None:
    pe_only_signal = RuptureContributingSignal(
        source=RuptureEvidenceSource.INTERNAL_PE,
        signal_strength=0.7,
        confidence=0.4,
        kind_hint=None,
        detail="pe spike",
    )
    with pytest.raises(ValueError, match="rupture_kind can only be resolved"):
        RuptureStateSnapshot(
            rupture_signal_strength=0.7,
            rupture_kind=RuptureKind.MISREAD,
            confidence=0.4,
            internal_suspected_only=True,
            evidence_sources=(RuptureEvidenceSource.INTERNAL_PE,),
            contributing_signals=(pe_only_signal,),
            description="invalid",
        )


def test_rupture_state_snapshot_requires_internal_only_when_only_pe_fires() -> None:
    pe_only_signal = RuptureContributingSignal(
        source=RuptureEvidenceSource.INTERNAL_PE,
        signal_strength=0.7,
        confidence=0.4,
        kind_hint=None,
        detail="pe spike",
    )
    with pytest.raises(ValueError, match="internal_suspected_only"):
        RuptureStateSnapshot(
            rupture_signal_strength=0.7,
            rupture_kind=None,
            confidence=0.4,
            internal_suspected_only=False,
            evidence_sources=(RuptureEvidenceSource.INTERNAL_PE,),
            contributing_signals=(pe_only_signal,),
            description="invalid",
        )


def test_rupture_state_snapshot_rejects_out_of_range_fields() -> None:
    with pytest.raises(ValueError, match="rupture_signal_strength"):
        RuptureStateSnapshot(
            rupture_signal_strength=1.5,
            rupture_kind=None,
            confidence=0.0,
            internal_suspected_only=False,
            evidence_sources=(),
            contributing_signals=(),
            description="",
        )
    with pytest.raises(ValueError, match="confidence"):
        RuptureStateSnapshot(
            rupture_signal_strength=0.0,
            rupture_kind=None,
            confidence=-0.1,
            internal_suspected_only=False,
            evidence_sources=(),
            contributing_signals=(),
            description="",
        )


def test_dialogue_external_outcome_evidence_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        DialogueExternalOutcomeEvidence(
            evidence_id="ev-1",
            turn_index=1,
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=1.5,
            evidence_ref="user:explicit:missed",
        )


def test_dialogue_external_outcome_snapshot_rejects_future_evidence() -> None:
    ev = DialogueExternalOutcomeEvidence(
        evidence_id="ev-1",
        turn_index=5,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        evidence_ref="user:explicit:missed",
    )
    with pytest.raises(ValueError, match="later turn"):
        DialogueExternalOutcomeSnapshot(
            turn_index=1,
            entries=(ev,),
            description="evidence from future turn not allowed",
        )


def test_dialogue_external_outcome_module_rejects_llm_by_default() -> None:
    module = DialogueExternalOutcomeModule()
    assert module.allow_llm_outcome_proposals is False
    assert module.default_wiring_level is WiringLevel.ACTIVE
    ev = DialogueExternalOutcomeEvidence(
        evidence_id="ev-llm",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
        confidence=0.9,
        evidence_ref="llm:proposal:test",
    )
    module.set_turn_index(1)
    with pytest.raises(ValueError, match="allow_llm_outcome_proposals"):
        module.append_evidence(ev)


def test_dialogue_external_outcome_module_accepts_llm_when_enabled() -> None:
    module = DialogueExternalOutcomeModule(allow_llm_outcome_proposals=True)
    ev = DialogueExternalOutcomeEvidence(
        evidence_id="ev-llm-2",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
        confidence=0.3,
        evidence_ref="llm:proposal:test",
    )
    module.set_turn_index(1)
    module.append_evidence(ev)
    assert module.pending_entry_count() == 1


def test_rupture_state_module_defaults_to_shadow() -> None:
    module = RuptureStateModule()
    assert module.slot_name == "rupture_state"
    assert module.owner == "RuptureStateModule"
    assert module.value_type is RuptureStateSnapshot
    assert module.default_wiring_level is WiringLevel.SHADOW
    assert module.dependencies == (
        "prediction_error",
        "relationship_state",
        "response_assembly",
        "dialogue_external_outcome",
    )
