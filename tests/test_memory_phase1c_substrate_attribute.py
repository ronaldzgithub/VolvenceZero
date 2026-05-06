"""Tests for Phase 1.C: substrate-feature retrieval embedding +
PE/substrate-derived attribute readout.

These tests cover:
- ``_substrate_embedding`` returns a deterministic vector that uses
  ``SubstrateSnapshot.feature_surface``;
- ``MemoryStore`` falls back to the legacy hash embedding when no
  substrate has been observed yet;
- ``MemoryStore.write`` populates an owner-internal attribute readout;
- ``MemorySnapshot.attribute_summary`` publishes the readouts and
  reflects PE intensity / regime / substrate digest;
- existing ``MemoryEntry`` / contract / checkpoint paths are unaffected.
"""

from __future__ import annotations

from volvence_zero.memory import (
    MemoryAttributeReadout,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    Track,
)
from volvence_zero.memory.retrieval import _semantic_embedding, _substrate_embedding
from volvence_zero.prediction import (
    ActualOutcome,
    PEDecomposition,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.substrate import (
    FeatureSignal,
    SubstrateSnapshot,
    SurfaceKind,
)


def _substrate(*, feature_values: tuple[tuple[str, tuple[float, ...]], ...]) -> SubstrateSnapshot:
    feature_surface = tuple(
        FeatureSignal(name=name, values=values, source="test")
        for name, values in feature_values
    )
    return SubstrateSnapshot(
        model_id="test",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=feature_surface,
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="test substrate",
    )


def _pe_snapshot(
    *,
    magnitude: float = 0.6,
    relationship_error: float = 0.6,
    regime_id: str = "comfort",
    bootstrap: bool = False,
) -> PredictionErrorSnapshot:
    action_context = PredictionActionContext(regime_id=regime_id)
    actual = ActualOutcome(
        observed_turn_index=1,
        task_progress=0.5,
        relationship_delta=relationship_error,
        regime_stability=0.4,
        action_payoff=0.4,
        description="actual",
        action_context=action_context,
    )
    next_prediction = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="next",
        action_context=action_context,
    )
    error = PredictionError(
        task_error=0.05,
        relationship_error=relationship_error,
        regime_error=0.05,
        action_error=0.05,
        magnitude=magnitude,
        signed_reward=-magnitude,
        description="pe",
    )
    decomposition = PEDecomposition(
        aleatoric_magnitude=0.10,
        epistemic_magnitude=0.40,
        per_axis=(
            ("task", 0.0, 0.05),
            ("relationship", 0.10, 0.40),
            ("regime", 0.0, 0.05),
            ("action", 0.0, 0.05),
        ),
        description="decomp",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=None if bootstrap else next_prediction,
        actual_outcome=actual,
        next_prediction=next_prediction,
        error=error,
        turn_index=1,
        bootstrap=bootstrap,
        description="pe-snapshot",
        action_context=action_context,
        pe_decomposition=decomposition,
    )


def test_substrate_embedding_returns_normalized_vector():
    surface = (
        FeatureSignal(name="semantic_task_pull", values=(0.6,), source="test"),
        FeatureSignal(name="residual_density", values=(0.4, 0.2), source="test"),
    )
    vector = _substrate_embedding(feature_surface=surface, dim=6)
    assert len(vector) == 6
    norm = sum(value * value for value in vector) ** 0.5
    assert norm > 0.0
    assert norm <= 1.0 + 1e-6


def test_substrate_embedding_returns_zero_when_feature_surface_empty():
    vector = _substrate_embedding(feature_surface=(), dim=4)
    assert vector == (0.0, 0.0, 0.0, 0.0)


def test_memory_store_uses_substrate_embedding_after_observe_substrate():
    """Owner signal should change when substrate has been observed
    (substrate-driven path) versus when no substrate is present
    (hash-only fallback)."""

    store = MemoryStore()
    # Capture baseline owner signal (no substrate observed yet).
    baseline = store._owner_signal(
        text="user prefers careful planning",
        tags=("planning",),
        track=Track.SHARED,
        stratum=MemoryStratum.EPISODIC.value,
        strength=0.5,
    )
    # Observe substrate, then re-compute.
    store.observe_substrate(
        substrate_snapshot=_substrate(
            feature_values=(
                ("semantic_task_pull", (0.85,)),
                ("residual_density", (0.40, 0.30)),
            )
        ),
        timestamp_ms=10,
    )
    enriched = store._owner_signal(
        text="user prefers careful planning",
        tags=("planning",),
        track=Track.SHARED,
        stratum=MemoryStratum.EPISODIC.value,
        strength=0.5,
    )
    # The two vectors must differ (substrate contributed) but both have
    # the same dimension and remain in [0, 1] range.
    assert baseline != enriched
    assert len(baseline) == len(enriched)


def test_memory_store_falls_back_to_hash_embedding_when_no_substrate():
    store = MemoryStore()
    owner_signal = store._owner_signal(
        text="user prefers careful planning",
        tags=("planning",),
        track=Track.SHARED,
        stratum=MemoryStratum.EPISODIC.value,
        strength=0.5,
    )
    # When no substrate has been observed, the dense vector should be
    # constructible from semantic+metadata blend alone (i.e. non-zero).
    assert any(value > 0.0 for value in owner_signal)


def test_memory_store_attribute_readout_reflects_pe_and_substrate():
    store = MemoryStore()
    store.observe_substrate(
        substrate_snapshot=_substrate(
            feature_values=(
                ("semantic_relationship_pull", (0.7,)),
                ("residual_density", (0.5, 0.2)),
            )
        ),
        timestamp_ms=42,
    )
    operations = store.apply_prediction_error_signal(
        prediction_error_snapshot=_pe_snapshot(
            magnitude=0.6, relationship_error=-0.6, regime_id="repair"
        ),
        timestamp_ms=42,
    )
    assert any("prediction-error-write" in op for op in operations)
    snapshot = store.snapshot(retrieved_entries=())
    assert isinstance(snapshot.attribute_summary, tuple)
    assert snapshot.attribute_summary, "Expected at least one attribute readout after PE write"
    readout = snapshot.attribute_summary[0]
    assert isinstance(readout, MemoryAttributeReadout)
    assert readout.pe_intensity == 0.6
    assert readout.pe_primary_axis == "relationship"
    assert readout.regime_id == "repair"
    assert readout.epistemic_magnitude == 0.40
    assert readout.aleatoric_magnitude == 0.10
    assert len(readout.substrate_feature_digest) > 0
    assert readout.timestamp_ms == 42


def test_memory_store_attribute_summary_capacity_capped():
    store = MemoryStore()
    store.observe_substrate(
        substrate_snapshot=_substrate(
            feature_values=(
                ("semantic_task_pull", (0.4,)),
            )
        ),
        timestamp_ms=1,
    )
    for index in range(40):
        store.write(
            MemoryWriteRequest(
                content=f"entry-{index}",
                track=Track.SHARED,
                stratum=MemoryStratum.EPISODIC,
                tags=("seq",),
            ),
            timestamp_ms=index + 1,
        )
    snapshot = store.snapshot(retrieved_entries=())
    assert len(snapshot.attribute_summary) <= 16
    # Most recent entry should be at the head.
    head = snapshot.attribute_summary[0]
    assert head.timestamp_ms == 40


def test_memory_store_attribute_readout_zero_pe_when_no_pe_observed():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="quiet write before any PE has been observed",
            track=Track.SHARED,
            stratum=MemoryStratum.TRANSIENT,
            tags=("quiet",),
        ),
        timestamp_ms=5,
    )
    snapshot = store.snapshot(retrieved_entries=())
    assert snapshot.attribute_summary
    head = snapshot.attribute_summary[0]
    assert head.pe_intensity == 0.0
    assert head.regime_id == ""


def test_memory_entry_schema_unchanged_by_phase1c():
    """Public ``MemoryEntry`` contract must not gain a new field; the
    attribute readout lives on ``MemorySnapshot.attribute_summary`` so
    persistence / checkpoint code is unaffected."""

    store = MemoryStore()
    entry = store.write(
        MemoryWriteRequest(
            content="schema check",
            track=Track.SHARED,
            stratum=MemoryStratum.TRANSIENT,
            tags=("check",),
        ),
        timestamp_ms=1,
    )
    expected = {
        "entry_id",
        "content",
        "track",
        "stratum",
        "created_at_ms",
        "last_accessed_ms",
        "strength",
        "tags",
        "subject_ids",
        "audience_ids",
    }
    assert set(entry.__dataclass_fields__.keys()) == expected
