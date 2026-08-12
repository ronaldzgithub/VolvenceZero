from __future__ import annotations

from dataclasses import replace
import math

import pytest

from volvence_zero.substrate import (
    SUBSTRATE_FORWARD_CENTERED_READOUT_KIND,
    SUBSTRATE_FORWARD_READOUT_KIND,
    SubstrateFingerprint,
    SubstrateForwardRepresentationPublisher,
    SubstrateReadoutReferenceStatistics,
    SyntheticOpenWeightResidualRuntime,
    fit_forward_readout_reference_statistics,
    layer_normalized_readout_vector,
    publish_runtime_capture_representation,
)


def _fingerprint(model_id: str = "synthetic-target") -> SubstrateFingerprint:
    return SubstrateFingerprint(
        model_id=model_id,
        version="fixture-v1",
        weights_sha256="1" * 64,
    )


def test_publisher_emits_deterministic_frozen_lineage_without_raw_text() -> None:
    publisher = SubstrateForwardRepresentationPublisher(
        SyntheticOpenWeightResidualRuntime(model_id="synthetic-target"),
        model_fingerprint=_fingerprint(),
    )
    sources = (
        ("sample-a", "I need help planning tomorrow."),
        ("sample-b", "I feel calmer than yesterday."),
    )
    first = publisher.publish(sources)
    second = publisher.publish(sources)

    assert first == second
    assert first.lineage.model_fingerprint == _fingerprint()
    assert first.lineage.representation_dim == sum(
        first.lineage.activation_widths
    )
    assert first.lineage.layer_indices == tuple(sorted(first.lineage.layer_indices))
    assert all(
        math.isclose(
            math.sqrt(sum(value * value for value in row.values)),
            1.0,
            rel_tol=1e-6,
        )
        for row in first.representations
    )
    assert all(source_text not in repr(first) for _, source_text in sources)


def test_publisher_rejects_unfrozen_or_mismatched_runtime() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    runtime.is_frozen = False
    with pytest.raises(ValueError, match="must be frozen"):
        SubstrateForwardRepresentationPublisher(
            runtime,
            model_fingerprint=_fingerprint(),
        )

    with pytest.raises(ValueError, match="runtime/model fingerprint mismatch"):
        SubstrateForwardRepresentationPublisher(
            SyntheticOpenWeightResidualRuntime(model_id="other-model"),
            model_fingerprint=_fingerprint(),
        )


def test_publisher_rejects_residual_geometry_drift() -> None:
    class _DriftingRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-target")
            self.calls = 0

        def capture(self, *, source_text: str):
            capture = super().capture(source_text=source_text)
            self.calls += 1
            if self.calls == 1:
                return capture
            first, *rest = capture.residual_activations
            return replace(
                capture,
                residual_activations=(
                    replace(first, activation=first.activation + (0.25,)),
                    *rest,
                ),
            )

    publisher = SubstrateForwardRepresentationPublisher(
        _DriftingRuntime(),
        model_fingerprint=_fingerprint(),
    )
    with pytest.raises(ValueError, match="geometry drifted"):
        publisher.publish((('a', 'first text'), ('b', 'second text')))


def test_publisher_rejects_conditioned_target_capture() -> None:
    class _ConditionedRuntime(SyntheticOpenWeightResidualRuntime):
        def capture(self, *, source_text: str):
            return replace(
                super().capture(source_text=source_text),
                personal_conditioning_applied=True,
            )

    publisher = SubstrateForwardRepresentationPublisher(
        _ConditionedRuntime(model_id="synthetic-target"),
        model_fingerprint=_fingerprint(),
    )
    with pytest.raises(ValueError, match="must be unconditioned"):
        publisher.publish((("a", "future utterance"),))


def test_runtime_context_publisher_accepts_conditioned_capture_without_raw_text() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    capture = replace(
        runtime.capture(source_text="private runtime prompt"),
        personal_conditioning_applied=True,
    )
    snapshot = publish_runtime_capture_representation(
        sample_id="runtime-context",
        source_sha256="2" * 64,
        capture=capture,
        model_fingerprint=_fingerprint(),
        runtime_origin="hf-local",
    )

    assert len(snapshot.representations) == 1
    assert snapshot.lineage.runtime_origin == "hf-local"
    assert "private runtime prompt" not in repr(snapshot)
    assert math.sqrt(
        sum(value * value for value in snapshot.representations[0].values)
    ) == pytest.approx(1.0)


def test_publisher_requires_full_weights_fingerprint() -> None:
    with pytest.raises(ValueError, match="full model weights SHA-256"):
        SubstrateForwardRepresentationPublisher(
            SyntheticOpenWeightResidualRuntime(model_id="synthetic-target"),
            model_fingerprint=SubstrateFingerprint(
                model_id="synthetic-target",
                version="fixture-v1",
                weights_sha256="legacy",
            ),
        )


def _capture_geometry_and_vectors(
    runtime: SyntheticOpenWeightResidualRuntime,
    texts: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[float, ...], ...]]:
    layer_indices: tuple[int, ...] = ()
    activation_widths: tuple[int, ...] = ()
    vectors: list[tuple[float, ...]] = []
    for text in texts:
        capture = runtime.capture(source_text=text)
        activations = tuple(
            sorted(capture.residual_activations, key=lambda row: row.layer_index)
        )
        layer_indices = tuple(row.layer_index for row in activations)
        activation_widths = tuple(len(row.activation) for row in activations)
        vectors.append(
            tuple(value for row in activations for value in row.activation)
        )
    return layer_indices, activation_widths, tuple(vectors)


def _fitted_statistics(
    runtime: SyntheticOpenWeightResidualRuntime,
    *,
    principal_component_count: int = 0,
) -> SubstrateReadoutReferenceStatistics:
    layer_indices, activation_widths, vectors = _capture_geometry_and_vectors(
        runtime,
        (
            "reference corpus utterance one",
            "reference corpus utterance two",
            "reference corpus utterance three",
        ),
    )
    return fit_forward_readout_reference_statistics(
        corpus_id="synthetic-train-split-fixture",
        layer_indices=layer_indices,
        activation_widths=activation_widths,
        vectors=vectors,
        principal_component_count=principal_component_count,
    )


def test_layer_normalization_is_scale_invariant_per_block() -> None:
    raw = (3.0, 4.0, 0.5, 1.2, -0.9)
    widths = (2, 3)
    normalized = layer_normalized_readout_vector(raw, activation_widths=widths)
    scaled = layer_normalized_readout_vector(
        tuple(value * 7.5 for value in raw), activation_widths=widths
    )
    assert normalized == pytest.approx(scaled)
    assert math.sqrt(sum(v * v for v in normalized[:2])) == pytest.approx(1.0)
    assert math.sqrt(sum(v * v for v in normalized[2:])) == pytest.approx(1.0)


def test_centered_publisher_emits_v2_lineage_bound_to_statistics() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    statistics = _fitted_statistics(runtime)
    centered_publisher = SubstrateForwardRepresentationPublisher(
        runtime,
        model_fingerprint=_fingerprint(),
        reference_statistics=statistics,
    )
    plain_publisher = SubstrateForwardRepresentationPublisher(
        runtime,
        model_fingerprint=_fingerprint(),
    )
    sources = (
        ("sample-a", "I need help planning tomorrow."),
        ("sample-b", "I feel calmer than yesterday."),
    )
    centered = centered_publisher.publish(sources)
    again = centered_publisher.publish(sources)
    plain = plain_publisher.publish(sources)

    assert centered == again
    assert centered.lineage.readout_kind == SUBSTRATE_FORWARD_CENTERED_READOUT_KIND
    assert centered.lineage.reference_corpus_id == "synthetic-train-split-fixture"
    assert (
        centered.lineage.reference_statistics_sha256
        == statistics.statistics_sha256
    )
    assert plain.lineage.readout_kind == SUBSTRATE_FORWARD_READOUT_KIND
    assert plain.lineage.reference_corpus_id is None
    assert centered.lineage.snapshot_fingerprint != plain.lineage.snapshot_fingerprint
    for row in centered.representations:
        assert math.sqrt(sum(v * v for v in row.values)) == pytest.approx(1.0)
    assert all(
        centered_row.values != plain_row.values
        for centered_row, plain_row in zip(
            centered.representations, plain.representations, strict=True
        )
    )


def test_centered_publisher_rejects_statistics_geometry_mismatch() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    statistics = _fitted_statistics(runtime)
    mismatched = fit_forward_readout_reference_statistics(
        corpus_id=statistics.corpus_id,
        layer_indices=tuple(index + 1 for index in statistics.layer_indices),
        activation_widths=statistics.activation_widths,
        vectors=tuple(
            (statistics.mean, tuple(value + 0.1 for value in statistics.mean))
        ),
    )
    publisher = SubstrateForwardRepresentationPublisher(
        runtime,
        model_fingerprint=_fingerprint(),
        reference_statistics=mismatched,
    )
    with pytest.raises(ValueError, match="geometry mismatch"):
        publisher.publish((("sample-a", "any utterance"),))


def test_centering_math_matches_hand_computation() -> None:
    vectors = (
        (1.0, 0.0, 0.0, 2.0),
        (0.0, 1.0, 2.0, 0.0),
    )
    statistics = fit_forward_readout_reference_statistics(
        corpus_id="hand-check",
        layer_indices=(0, 1),
        activation_widths=(2, 2),
        vectors=vectors,
    )
    # Layer-normalized inputs are ((1,0),(0,1)) and ((0,1),(1,0)); the mean
    # is (0.5, 0.5, 0.5, 0.5).
    assert statistics.mean == pytest.approx((0.5, 0.5, 0.5, 0.5))
    centered = statistics.apply((1.0, 0.0, 0.0, 1.0))
    expected_direction = (0.5, -0.5, -0.5, 0.5)
    norm = math.sqrt(sum(v * v for v in expected_direction))
    assert centered == pytest.approx(
        tuple(value / norm for value in expected_direction)
    )
    with pytest.raises(ValueError, match="zero norm"):
        statistics.apply(statistics.mean)


def test_statistics_reject_tampering_and_bad_components() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    statistics = _fitted_statistics(runtime)
    with pytest.raises(ValueError, match="statistics_sha256 mismatch"):
        replace(
            statistics,
            mean=tuple(value + 1e-3 for value in statistics.mean),
        )
    with pytest.raises(ValueError, match="unit norm"):
        replace(
            statistics,
            principal_components=(
                tuple(2.0 if index == 0 else 0.0 for index in range(len(statistics.mean))),
            ),
        )


def test_statistics_payload_round_trip() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    statistics = _fitted_statistics(runtime, principal_component_count=1)
    assert len(statistics.principal_components) == 1
    restored = SubstrateReadoutReferenceStatistics.from_payload(
        statistics.to_payload()
    )
    assert restored == statistics


def test_fit_rejects_evaluation_scale_misuse_signatures() -> None:
    with pytest.raises(ValueError, match="at least two vectors"):
        fit_forward_readout_reference_statistics(
            corpus_id="too-small",
            layer_indices=(0,),
            activation_widths=(2,),
            vectors=((1.0, 0.0),),
        )
    with pytest.raises(ValueError, match="non-negative"):
        fit_forward_readout_reference_statistics(
            corpus_id="bad-count",
            layer_indices=(0,),
            activation_widths=(2,),
            vectors=((1.0, 0.0), (0.0, 1.0)),
            principal_component_count=-1,
        )


def test_runtime_context_publisher_supports_centered_readout() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-target")
    statistics = _fitted_statistics(runtime)
    capture = runtime.capture(source_text="private runtime prompt")
    snapshot = publish_runtime_capture_representation(
        sample_id="runtime-context",
        source_sha256="2" * 64,
        capture=capture,
        model_fingerprint=_fingerprint(),
        runtime_origin="hf-local",
        reference_statistics=statistics,
    )
    assert snapshot.lineage.readout_kind == SUBSTRATE_FORWARD_CENTERED_READOUT_KIND
    assert (
        snapshot.lineage.reference_statistics_sha256
        == statistics.statistics_sha256
    )
    assert math.sqrt(
        sum(value * value for value in snapshot.representations[0].values)
    ) == pytest.approx(1.0)
