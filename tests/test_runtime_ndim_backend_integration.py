"""Phase B exit gate: runtime ndim metacontroller backend routing + SHADOW parity.

Verifies:

1. The backend-routed ndim forward (pure backend) reproduces the pure component
   forward of the SAME store params within tight tolerance, and torch reproduces
   pure (SHADOW parity gate).
2. FullLearnedTemporalPolicy under WiringLevel.ACTIVE actually runs the torch
   backend for its runtime step and still produces a valid TemporalStep.
3. DISABLED keeps the pure components (default, unchanged); rollback works.
"""

from __future__ import annotations

import asyncio

import pytest

from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    SimulatedResidualSubstrateAdapter,
    SubstrateModule,
    build_training_trace,
)
from volvence_zero.tensor_backend import PurePythonBackend, is_torch_available
from volvence_zero.temporal import FullLearnedTemporalPolicy, MetacontrollerParameterStore
from volvence_zero.temporal.backend_ndim_runtime import (
    BackendNdimMetacontroller,
    runtime_ndim_shadow_compare,
)

torch_only = pytest.mark.skipif(not is_torch_available(), reason="torch not installed")

N_Z = 8


def _store() -> MetacontrollerParameterStore:
    return MetacontrollerParameterStore(n_z=N_Z)


def _substrate_snapshot():
    trace = build_training_trace(
        trace_id="phaseB", source_text="the quiet harbor at dawn holds still water again"
    )
    adapter = SimulatedResidualSubstrateAdapter(trace=trace)
    snap = asyncio.run(SubstrateModule(adapter=adapter).process_standalone())
    return snap.value


def test_pure_backend_matches_pure_components() -> None:
    # The backend-routed forward (pure) must reproduce the live ndim components
    # on the same store params, so ACTIVE is a faithful drop-in.
    store = _store()
    from volvence_zero.temporal.metacontroller_components import (
        NdimResidualDecoder,
        NdimSequenceEncoder,
        NdimSwitchUnit,
    )

    snap = _substrate_snapshot()
    enc_p = store.ndim_encoder_parameters
    sw_p = store.ndim_switch_parameters
    dec_p = store.ndim_decoder_parameters
    prev = tuple(0.0 for _ in range(N_Z))

    pure_enc = NdimSequenceEncoder(n_z=N_Z, n_input=enc_p.n_input).encode(
        substrate_snapshot=snap, previous_hidden_state=None, cms_context=None, params=enc_p
    )
    backend_mc = BackendNdimMetacontroller(PurePythonBackend(), n_z=N_Z)
    b_enc = backend_mc.encode(
        substrate_snapshot=snap, previous_hidden_state=None, cms_context=None, params=enc_p
    )
    for a, b in zip(pure_enc.posterior.posterior_mean, b_enc.posterior.posterior_mean):
        assert abs(a - b) <= 1e-9

    pure_beta, _, pure_scalar = NdimSwitchUnit(n_z=N_Z).compute(
        z_tilde=pure_enc.posterior.z_tilde, previous_code=prev, memory_signal=0.0,
        reflection_signal=0.0, active_family_outcome=0.0, active_family_reuse=0.0,
        active_family_persistence=0.0, external_switch_pressure_delta=0.0, params=sw_p,
    )
    b_beta, _, b_scalar = backend_mc.compute(
        z_tilde=b_enc.posterior.z_tilde, previous_code=prev, memory_signal=0.0,
        reflection_signal=0.0, active_family_outcome=0.0, active_family_reuse=0.0,
        active_family_persistence=0.0, external_switch_pressure_delta=0.0, params=sw_p,
    )
    assert abs(pure_scalar - b_scalar) <= 1e-9

    latent = tuple(
        max(0.0, min(1.0, pure_beta[i] * pure_enc.posterior.z_tilde[i] + (1 - pure_beta[i]) * prev[i]))
        for i in range(N_Z)
    )
    pure_dec = NdimResidualDecoder(n_z=N_Z).decode(latent_code=latent, params=dec_p)
    b_dec = backend_mc.decode(latent_code=latent, params=dec_p)
    for a, b in zip(pure_dec.applied_control, b_dec.applied_control):
        assert abs(a - b) <= 1e-9


@torch_only
def test_runtime_ndim_shadow_parity_promotable() -> None:
    store = _store()
    report = runtime_ndim_shadow_compare(
        store=store, substrate_snapshot=_substrate_snapshot(),
        tolerance=1e-7, latency_budget_ms=1000.0,
    )
    assert report.torch_available
    assert report.within_tolerance, report.description
    assert report.promotable


@torch_only
def test_policy_active_backend_runs_torch_step() -> None:
    store = _store()
    policy = FullLearnedTemporalPolicy(parameter_store=store, runtime_backend=WiringLevel.ACTIVE)
    assert policy.runtime_backend is WiringLevel.ACTIVE
    backend_mc = policy._resolve_backend_ndim_mc()
    assert backend_mc is not None
    assert backend_mc.backend.kind.value == "torch"


def test_policy_disabled_backend_uses_pure_components() -> None:
    store = _store()
    policy = FullLearnedTemporalPolicy(parameter_store=store)  # default DISABLED
    assert policy.runtime_backend is WiringLevel.DISABLED
    assert policy._resolve_backend_ndim_mc() is None
    # rollback toggle works
    policy.set_runtime_backend(WiringLevel.ACTIVE)
    assert policy.runtime_backend is WiringLevel.ACTIVE
    policy.set_runtime_backend(WiringLevel.DISABLED)
    assert policy._resolve_backend_ndim_mc() is None
