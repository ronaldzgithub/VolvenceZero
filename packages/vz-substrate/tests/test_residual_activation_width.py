from __future__ import annotations

import pytest

from volvence_zero.substrate import build_builtin_transformers_runtime


def _activation_widths(runtime) -> set[int]:
    capture = runtime.capture(source_text="alpha beta gamma")
    return {
        len(activation.activation)
        for step in capture.residual_sequence
        for activation in step.residual_activations
    }


def test_transformers_capture_activation_width_is_configurable() -> None:
    compressed = build_builtin_transformers_runtime(
        model_id="activation-width-16",
        activation_width=16,
    )
    exact = build_builtin_transformers_runtime(
        model_id="activation-width-exact",
        activation_width=64,
    )

    assert _activation_widths(compressed) == {16}
    assert _activation_widths(exact) == {48}


def test_transformers_capture_activation_width_fails_loudly() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_builtin_transformers_runtime(
            model_id="activation-width-invalid",
            activation_width=0,
        )
