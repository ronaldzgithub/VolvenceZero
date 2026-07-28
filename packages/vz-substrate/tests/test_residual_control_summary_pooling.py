from __future__ import annotations

import pytest
import torch

from volvence_zero.substrate import TransformersOpenWeightResidualRuntime
from volvence_zero.substrate import build_builtin_transformers_runtime


def _runtime() -> TransformersOpenWeightResidualRuntime:
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime._torch = torch
    return runtime


def test_control_summary_uses_latest_token_like_public_capture() -> None:
    hidden = torch.tensor(
        (
            ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
            ((2.0, 4.0), (6.0, 8.0), (10.0, 12.0)),
        )
    )

    pooled = _runtime()._latest_token_control_activation(hidden)

    assert pooled.tolist() == [7.5, 9.0]
    assert pooled.tolist() != hidden.mean(dim=1)[0].tolist()


def test_control_summary_rejects_empty_or_non_sequence_hidden_state() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError, match="nonempty sequence"):
        runtime._latest_token_control_activation(torch.empty((1, 0, 4)))

    with pytest.raises(ValueError, match="shape"):
        runtime._latest_token_control_activation(torch.empty((1, 4)))


def test_continuation_score_uses_real_controlled_forward() -> None:
    runtime = build_builtin_transformers_runtime(
        model_id="continuation-score-test",
    )

    zero = runtime.score_continuation(
        source_text="repair inspect",
        continuation_text="continue",
        applied_control=(0.0, 0.0, 0.0),
        track_scale=(0.7, 0.7, 0.7),
    )
    controlled = runtime.score_continuation(
        source_text="repair inspect",
        continuation_text="continue",
        applied_control=(0.2, -0.1, 0.3),
        track_scale=(0.7, 0.7, 0.7),
    )

    assert zero.token_count == controlled.token_count == 1
    assert zero.mean_negative_log_likelihood > 0.0
    assert 0.0 < zero.geometric_mean_probability < 1.0
    assert (
        controlled.mean_negative_log_likelihood
        != zero.mean_negative_log_likelihood
    )
    assert controlled.applied_control == (0.2, -0.1, 0.3)


def test_continuation_score_rejects_empty_target() -> None:
    runtime = build_builtin_transformers_runtime(
        model_id="continuation-score-empty-target",
    )

    with pytest.raises(ValueError, match="continuation_text"):
        runtime.score_continuation(
            source_text="repair",
            continuation_text=" ",
            applied_control=(0.0, 0.0, 0.0),
        )
