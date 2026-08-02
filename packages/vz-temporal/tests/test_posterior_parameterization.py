"""The smooth posterior parameterization removes clamp saturation.

The 2026-08-02 Gate-1 sweep showed a noisy, non-monotonic rate axis. The root
cause was the legacy posterior surface ``sigma = clamp(|W h|, 0.05, 0.95)``:
``abs`` is non-differentiable at 0 and the clamp zeroes the gradient at its
boundaries, so the KL/rate could not respond smoothly to ``alpha``.

These tests pin the two properties the repair depends on:

- legacy keeps its historical bounded surface (byte-compatible default), and
- smooth (``sigma = softplus(W h) + floor``) is strictly positive, escapes the
  legacy [0.05, 0.95] band, and keeps a non-vanishing gradient into
  ``posterior_std_proj`` even where the legacy path would saturate.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from volvence_zero.temporal.interface import (  # noqa: E402
    MetacontrollerParameterStore,
)
from volvence_zero.temporal.metacontroller_components import (  # noqa: E402
    POSTERIOR_STD_SMOOTH_FLOOR,
    _softplus_scalar,
)
from volvence_zero.temporal.torch_store_ssl import (  # noqa: E402
    _TorchNdimMetacontroller,
)


def _module(*, scale_std_proj: float = 1.0) -> _TorchNdimMetacontroller:
    store = MetacontrollerParameterStore(n_z=16, initialization_seed=7)
    encoder = store.ndim_encoder_parameters
    assert encoder is not None
    if scale_std_proj != 1.0:
        # Amplify the std projection so softplus can exceed the legacy 0.95
        # clamp, giving a crisp discriminator between the two surfaces.
        encoder = dataclasses.replace(
            encoder,
            posterior_std_proj=tuple(
                tuple(value * scale_std_proj for value in row)
                for row in encoder.posterior_std_proj
            ),
        )
    return _TorchNdimMetacontroller(
        n_z=16,
        encoder=encoder,
        switch=store.ndim_switch_parameters,
        decoder=store.ndim_decoder_parameters,
    )


def _step_inputs(module: _TorchNdimMetacontroller) -> list[tuple[float, ...]]:
    generator = torch.Generator().manual_seed(20260802)
    return [
        tuple(
            float(v)
            for v in torch.randn(
                module.n_input, generator=generator, dtype=torch.float64
            )
        )
        for _ in range(5)
    ]


def test_softplus_scalar_matches_reference() -> None:
    import math

    for value in (-30.0, -1.0, 0.0, 0.5, 25.0, 40.0):
        assert _softplus_scalar(value) == pytest.approx(
            math.log1p(math.exp(value)) if abs(value) < 20 else max(value, 0.0)
            if value > 0
            else math.exp(value),
            rel=1e-9,
            abs=1e-9,
        )


def test_legacy_keeps_the_bounded_posterior_surface() -> None:
    module = _module(scale_std_proj=8.0)
    inputs = _step_inputs(module)
    out = module.rollout(
        inputs,
        switch_threshold=0.55,
        generator=torch.Generator().manual_seed(1),
        posterior_parameterization="legacy",
    )
    for std in out["stds"]:
        assert torch.all(std >= 0.05 - 1e-9)
        assert torch.all(std <= 0.95 + 1e-9)
    for mean in out["means"]:
        assert torch.all(mean >= 0.0 - 1e-9)
        assert torch.all(mean <= 1.0 + 1e-9)


def test_smooth_posterior_escapes_the_legacy_band() -> None:
    module = _module(scale_std_proj=8.0)
    inputs = _step_inputs(module)
    out = module.rollout(
        inputs,
        switch_threshold=0.55,
        generator=torch.Generator().manual_seed(1),
        posterior_parameterization="smooth",
    )
    stds = torch.stack(out["stds"])
    # softplus is strictly positive and, with an amplified projection, exceeds
    # the legacy 0.95 ceiling that the clamp could never cross.
    assert torch.all(stds > 0.0)
    assert torch.max(stds) > 0.95
    assert torch.all(stds >= POSTERIOR_STD_SMOOTH_FLOOR - 1e-9)


def test_smooth_posterior_keeps_a_live_gradient_without_saturation() -> None:
    inputs = _step_inputs(_module(scale_std_proj=8.0))

    def run(parameterization: str):
        module = _module(scale_std_proj=8.0)
        out = module.rollout(
            inputs,
            switch_threshold=0.55,
            generator=torch.Generator().manual_seed(1),
            posterior_parameterization=parameterization,
        )
        stds = torch.stack(out["stds"])
        objective = stds.sum()
        objective.backward()
        grad = module.posterior_std_proj.grad
        assert grad is not None
        assert torch.all(torch.isfinite(grad))
        return stds, float(grad.norm())

    legacy_stds, legacy_norm = run("legacy")
    smooth_stds, smooth_norm = run("smooth")

    # Legacy: the amplified projection pushes many std entries onto the clamp
    # boundary, where their local derivative is zero (the dead-gradient region
    # that made the rate axis noisy).
    legacy_saturated = int(
        torch.sum(torch.abs(legacy_stds - 0.95) < 1e-9)
        + torch.sum(torch.abs(legacy_stds - 0.05) < 1e-9)
    )
    assert legacy_saturated > 0

    # Smooth: no entry sits on those boundaries and the gradient into the std
    # projection is alive.
    smooth_saturated = int(
        torch.sum(torch.abs(smooth_stds - 0.95) < 1e-9)
        + torch.sum(torch.abs(smooth_stds - 0.05) < 1e-9)
    )
    assert smooth_saturated == 0
    assert smooth_norm > 1e-6
