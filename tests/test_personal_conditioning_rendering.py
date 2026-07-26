"""Pure-function tests for the personal-conditioning statement renderer.

State KV P0-b: the renderer must consume only the typed readout (labelled
coordinates + confidence), bucket values deterministically, and stay empty
for cold-start / zero-confidence readouts.
"""

from __future__ import annotations

import pytest

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)
from volvence_zero.personal_conditioning_rendering import (
    render_personal_conditioning_statement,
)


def _vector(fill: float = 0.5) -> tuple[float, ...]:
    return tuple(fill for _ in PERSONAL_CONDITIONING_VECTOR_LABELS)


def _render(
    *,
    state_vector: tuple[float, ...] | None = None,
    confidence: float = 0.6,
    is_cold_start: bool = False,
) -> str:
    return render_personal_conditioning_statement(
        state_vector=state_vector if state_vector is not None else _vector(),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        confidence=confidence,
        is_cold_start=is_cold_start,
    )


def test_cold_start_renders_empty_string() -> None:
    assert _render(state_vector=_vector(0.0), confidence=0.0, is_cold_start=True) == ""


def test_zero_confidence_renders_empty_string() -> None:
    assert _render(confidence=0.0) == ""


def test_rendering_is_deterministic_for_same_readout() -> None:
    assert _render() == _render()


def test_every_coordinate_appears_with_bucket_and_value() -> None:
    values = tuple((index + 1) / 20.0 for index in range(16))
    statement = _render(state_vector=values)

    # One line per display group plus the confidence header.
    assert statement.splitlines()[0].startswith("Current relational state estimate")
    assert "confidence 0.60" in statement
    for value in values:
        assert f"({value:.2f})" in statement


def test_bucket_boundaries() -> None:
    low = _render(state_vector=_vector(0.33))
    moderate = _render(state_vector=_vector(0.34))
    high = _render(state_vector=_vector(0.67))

    assert "moderate" not in low and "low" in low
    assert "low" not in moderate and "moderate" in moderate
    assert "high" in high and "moderate" not in high


def test_wrong_label_contract_fails_loud() -> None:
    with pytest.raises(ValueError, match="frozen"):
        render_personal_conditioning_statement(
            state_vector=(0.5,),
            vector_labels=("something_else",),
            confidence=0.5,
            is_cold_start=False,
        )


def test_mismatched_vector_width_fails_loud() -> None:
    with pytest.raises(ValueError, match="one coordinate per label"):
        render_personal_conditioning_statement(
            state_vector=_vector()[:-1],
            vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            confidence=0.5,
            is_cold_start=False,
        )


@pytest.mark.parametrize(
    ("state_vector", "confidence"),
    (
        (_vector(-0.01), 0.5),
        (_vector(1.01), 0.5),
        (_vector(), -0.01),
        (_vector(), 1.01),
    ),
)
def test_out_of_bounds_readout_fails_loud(
    state_vector: tuple[float, ...],
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        render_personal_conditioning_statement(
            state_vector=state_vector,
            vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            confidence=confidence,
            is_cold_start=False,
        )


def test_inconsistent_cold_start_fails_loud() -> None:
    with pytest.raises(ValueError, match="all-zero vector"):
        render_personal_conditioning_statement(
            state_vector=_vector(0.1),
            vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            confidence=0.0,
            is_cold_start=True,
        )
