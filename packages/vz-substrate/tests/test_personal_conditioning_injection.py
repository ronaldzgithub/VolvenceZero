"""Personal conditioning injection contract tests (State KV P0-a).

Covers the three defect fixes of the personal-conditioning defect
packet:

1. the ABC default ``generate()`` fails loudly instead of silently
   dropping a ``personal_conditioning`` contract input;
2. the synthetic runtime takes conditioning trace-only and reports
   ``personal_conditioning_applied=False`` so no consumer can mistake
   the trace for a real injection;
3. the projection math extracted from the transformers runtime
   (``build_personal_conditioning_basis`` / ``..._delta``) honours the
   cold-start / zero-confidence gates, scales linearly with
   confidence, and respects the hard 0.12 scale cap — without needing
   a loaded model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.substrate.residual_backend import (
    PERSONAL_CONDITIONING_SCALE_CAP,
    build_personal_conditioning_basis,
    build_personal_conditioning_delta,
    clamp_personal_conditioning_scale,
)
from volvence_zero.substrate.residual_interfaces import (
    OpenWeightResidualRuntime,
)
from volvence_zero.substrate.residual_synthetic import (
    SyntheticOpenWeightResidualRuntime,
)


def _conditioning(
    *,
    confidence: float = 0.8,
    cold_start: bool = False,
    coordinate_value: float = 0.5,
) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(
            (0.0 if cold_start else coordinate_value)
            for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
        ),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 1),),
        source_fingerprint="injection-contract-test",
        confidence=0.0 if cold_start else confidence,
        is_cold_start=cold_start,
        description="test conditioning",
    )


class _MinimalRuntime(OpenWeightResidualRuntime):
    """Smallest concrete runtime keeping the ABC default ``generate``."""

    def __init__(self) -> None:
        self.model_id = "minimal-abc-runtime"
        self.is_frozen = True

    def capture(self, *, source_text: str):
        raise NotImplementedError("capture is out of scope for this test")

    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot,
        applied_control,
        track_scale=(1.0, 1.0, 1.0),
    ):
        raise NotImplementedError("apply_control is out of scope for this test")


# ---------------------------------------------------------------------------
# 1. ABC default generate(): fail loud on conditioning, no regression without
# ---------------------------------------------------------------------------


def test_abc_default_generate_fails_loud_on_personal_conditioning() -> None:
    runtime = _MinimalRuntime()
    with pytest.raises(NotImplementedError, match="residual hooks"):
        runtime.generate(prompt="hello", personal_conditioning=_conditioning())


def test_abc_default_generate_without_conditioning_keeps_placeholder() -> None:
    runtime = _MinimalRuntime()
    result = runtime.generate(prompt="hello")
    assert "generation not supported" in result.text
    assert result.token_count == 0
    assert result.personal_conditioning_applied is False


# ---------------------------------------------------------------------------
# 2. Synthetic runtime: trace-only intake, never claims injection
# ---------------------------------------------------------------------------


def test_synthetic_runtime_records_conditioning_trace_only() -> None:
    runtime = SyntheticOpenWeightResidualRuntime()
    conditioning = _conditioning()

    result = runtime.generate(prompt="hello", personal_conditioning=conditioning)

    assert result.personal_conditioning_applied is False
    assert runtime.personal_conditioning_trace == [conditioning]
    assert "trace-only" in result.description
    assert "not injected" in result.description


def test_synthetic_runtime_without_conditioning_leaves_trace_empty() -> None:
    runtime = SyntheticOpenWeightResidualRuntime()

    result = runtime.generate(prompt="hello")

    assert result.personal_conditioning_applied is False
    assert runtime.personal_conditioning_trace == []
    assert "trace-only" not in result.description


# ---------------------------------------------------------------------------
# 3. Projection math (pure functions, no model required)
# ---------------------------------------------------------------------------


def _torch():
    return pytest.importorskip("torch")


def _basis(torch_module, *, hidden_size: int = 32):
    return build_personal_conditioning_basis(
        torch_module=torch_module,
        hidden_size=hidden_size,
        vector_dim=len(PERSONAL_CONDITIONING_VECTOR_LABELS),
    )


def test_projection_basis_shape_and_row_normalisation() -> None:
    torch = _torch()
    basis = _basis(torch, hidden_size=32)
    assert basis.shape == (len(PERSONAL_CONDITIONING_VECTOR_LABELS), 32)
    row_norms = basis.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)


def test_projection_delta_gates_absent_cold_start_and_zero_confidence() -> None:
    torch = _torch()
    basis = _basis(torch)
    for conditioning in (
        None,
        _conditioning(cold_start=True),
        _conditioning(confidence=0.0),
    ):
        delta = build_personal_conditioning_delta(
            torch_module=torch,
            conditioning=conditioning,
            basis=basis,
            scale=0.08,
        )
        assert delta is None


def test_projection_delta_norm_scales_linearly_with_confidence() -> None:
    torch = _torch()
    basis = _basis(torch)
    deltas = {
        confidence: build_personal_conditioning_delta(
            torch_module=torch,
            conditioning=_conditioning(confidence=confidence),
            basis=basis,
            scale=0.08,
        )
        for confidence in (0.4, 0.8)
    }
    low_norm = float(deltas[0.4].norm())
    high_norm = float(deltas[0.8].norm())
    assert low_norm > 0.0
    assert high_norm == pytest.approx(2.0 * low_norm, rel=1e-5)


def test_projection_delta_rejects_wrong_vector_width() -> None:
    # The snapshot contract already validates width at construction, so a
    # drifted-schema value can only reach the substrate as a stale/foreign
    # object. A duck-typed stub simulates exactly that drift.
    torch = _torch()
    basis = _basis(torch)
    drifted = SimpleNamespace(
        is_cold_start=False,
        confidence=0.8,
        state_vector=(0.5,) * 8,
    )
    with pytest.raises(ValueError, match="projection contract"):
        build_personal_conditioning_delta(
            torch_module=torch,
            conditioning=drifted,
            basis=basis,
            scale=0.08,
        )


def test_scale_clamp_enforces_hard_cap() -> None:
    assert PERSONAL_CONDITIONING_SCALE_CAP == 0.12
    assert clamp_personal_conditioning_scale(0.5) == 0.12
    assert clamp_personal_conditioning_scale(-1.0) == 0.0
    assert clamp_personal_conditioning_scale(0.08) == 0.08
