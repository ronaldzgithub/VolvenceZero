"""Contract tests for learned control-basis extraction and installation."""

from __future__ import annotations

import math

import pytest

from volvence_zero.substrate import (
    build_builtin_transformers_runtime,
    control_basis_fingerprint,
    fit_transition_control_basis,
)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _demo_deltas(*, width: int = 12, count: int = 9) -> list[list[float]]:
    deltas = []
    for sample in range(count):
        row = [
            1.0 if index == 0 else 0.0
            for index in range(width)
        ]
        # Structured variation on two secondary axes plus a small tail.
        row[1] += 0.4 * math.sin(sample * 1.7)
        row[2] += 0.3 * math.cos(sample * 0.9)
        row[3 + sample % 3] += 0.05
        deltas.append(row)
    return deltas


def test_fit_transition_control_basis_is_orthonormal_and_deterministic() -> None:
    deltas = _demo_deltas()

    basis = fit_transition_control_basis(deltas, basis_rank=3)

    assert len(basis) == 3
    for row in basis:
        assert math.isclose(math.sqrt(_dot(row, row)), 1.0, abs_tol=1e-9)
    assert abs(_dot(basis[0], basis[1])) < 1e-9
    assert abs(_dot(basis[0], basis[2])) < 1e-9
    assert abs(_dot(basis[1], basis[2])) < 1e-9

    # Row 0 is the normalized mean transition direction.
    width = len(deltas[0])
    mean_delta = tuple(
        sum(row[index] for row in deltas) / len(deltas)
        for index in range(width)
    )
    mean_norm = math.sqrt(_dot(mean_delta, mean_delta))
    normalized_mean = tuple(value / mean_norm for value in mean_delta)
    assert math.isclose(_dot(basis[0], normalized_mean), 1.0, abs_tol=1e-9)

    refit = fit_transition_control_basis(deltas, basis_rank=3)
    assert refit == basis
    assert control_basis_fingerprint(refit) == control_basis_fingerprint(basis)


def test_fit_transition_control_basis_rejects_degenerate_corpora() -> None:
    with pytest.raises(ValueError, match="at least 4 transition deltas"):
        fit_transition_control_basis(
            [[1.0, 0.0], [1.0, 0.0]],
            basis_rank=3,
        )
    with pytest.raises(ValueError, match="share one width"):
        fit_transition_control_basis(
            [[1.0, 0.0, 0.0], [1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            basis_rank=2,
        )
    with pytest.raises(ValueError, match="non-finite"):
        fit_transition_control_basis(
            [
                [1.0, 0.0, 0.0],
                [float("nan"), 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [1.0, 0.0, 0.1],
            ],
            basis_rank=2,
        )
    # Zero-mean corpus cannot define the forward direction.
    with pytest.raises(ValueError, match="degenerate mean-transition"):
        fit_transition_control_basis(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            basis_rank=2,
        )


def test_install_control_basis_supports_arbitrary_rank_and_layer_gains() -> None:
    runtime = build_builtin_transformers_runtime(activation_width=48)
    assert runtime.control_basis_provenance == "fixed-sinusoid-v1"
    hidden_size = runtime._hidden_size
    deltas = [
        [
            (1.0 if index == 0 else 0.0)
            + 0.2 * math.sin((sample + 1) * (index + 1) * 0.37)
            for index in range(hidden_size)
        ]
        for sample in range(8)
    ]
    basis = fit_transition_control_basis(deltas, basis_rank=3)

    runtime.install_control_basis(
        basis=basis,
        provenance="train-transition-pca-v1:demo",
        layer_indices=(runtime.hook_layer_indices[0],),
        layer_gains=(0.5,),
    )

    assert runtime.control_basis_provenance == "train-transition-pca-v1:demo"
    score = runtime.score_continuation(
        source_text="alpha bravo charlie",
        continuation_text="delta",
        applied_control=(1.0, 0.0, 0.0),
    )
    assert math.isfinite(score.mean_negative_log_likelihood)

    full_basis = tuple(
        tuple(
            math.sin((row_index + 1) * (column_index + 1) * 0.17)
            + math.cos((row_index + 2) * (column_index + 1) * 0.11)
            for column_index in range(hidden_size)
        )
        for row_index in range(16)
    )
    runtime.install_control_basis(
        basis=full_basis,
        provenance="full-code-sinusoid-v1:demo",
    )
    full_score = runtime.score_continuation(
        source_text="alpha bravo charlie",
        continuation_text="delta",
        applied_control=tuple(0.05 for _ in range(16)),
    )
    assert math.isfinite(full_score.mean_negative_log_likelihood)

    with pytest.raises(ValueError, match="hidden size"):
        runtime.install_control_basis(
            basis=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            provenance="bad-width",
        )
    with pytest.raises(ValueError, match="not hooked"):
        runtime.install_control_basis(
            basis=basis,
            provenance="bad-layer",
            layer_indices=(max(runtime.hook_layer_indices) + 1,),
        )
    with pytest.raises(ValueError, match="align"):
        runtime.install_control_basis(
            basis=basis,
            provenance="bad-gains",
            layer_indices=(runtime.hook_layer_indices[0],),
            layer_gains=(1.0, 0.5),
        )
    with pytest.raises(ValueError, match="non-empty provenance"):
        runtime.install_control_basis(basis=basis, provenance="  ")
