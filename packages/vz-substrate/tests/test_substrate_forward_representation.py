from __future__ import annotations

from dataclasses import replace
import math

import pytest

from volvence_zero.substrate import (
    SubstrateFingerprint,
    SubstrateForwardRepresentationPublisher,
    SyntheticOpenWeightResidualRuntime,
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
