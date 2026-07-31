from __future__ import annotations

from volvence_zero.temporal import MetacontrollerParameterStore


def test_prediction_error_temporal_weights_reach_ndim_code() -> None:
    store = MetacontrollerParameterStore(n_z=4)
    code = (0.2, 0.4, 0.6, 0.8)

    assert (
        store.runtime_prediction_error_modulated_code(
            code,
            strength=0.0,
        )
        is code
    )

    store.fit_temporal_from_signals(
        residual_strength=0.7,
        memory_strength=0.2,
        reflection_strength=0.1,
    )
    modulated = store.runtime_prediction_error_modulated_code(
        code,
        strength=1.0,
    )

    assert modulated != code
    assert all(0.0 <= value <= 1.0 for value in modulated)
