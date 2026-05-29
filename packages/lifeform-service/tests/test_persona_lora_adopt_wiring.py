"""Regression tests for the DLaaS adopt -> persona LoRA wiring.

The adopt path (:func:`register_bundle_persona_lora`) used to drop the
artifact's ``peft_checkpoint_dir``, so an adopted figure with a real
baked PEFT checkpoint silently fell back to the projected-summary
forward hook (the LayerNorm-eaten path). These tests pin:

* the checkpoint dir is forwarded into the pool record, and
* it is resolved against the configurable checkpoint root so a service
  CWD that differs from the bake CWD still locates the checkpoint, and
* once registered, :meth:`PersonaLoRAPool.activate` routes to the real
  :meth:`activate_peft_adapter` path rather than ``activate_lora``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from lifeform_service import register_bundle_persona_lora
from volvence_zero.substrate import (
    PersonaLoRAPool,
    SubstrateDeltaAdapterLayer,
)


@dataclass
class _FakeArtifact:
    figure_id: str
    backend_id: str
    training_plan_hash: str
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...]
    parameter_count: int
    description: str
    peft_checkpoint_dir: str


@dataclass
class _FakeBundle:
    bundle_id: str
    figure_id: str
    lora: _FakeArtifact | None


def _adapter_layers() -> tuple[SubstrateDeltaAdapterLayer, ...]:
    return (
        SubstrateDeltaAdapterLayer(
            layer_index=0,
            delta_vector=(0.01, -0.02, 0.03),
            mean_abs_delta=0.02,
            description="test-delta",
        ),
    )


def _bundle_with_checkpoint(checkpoint_dir: str) -> _FakeBundle:
    return _FakeBundle(
        bundle_id="figure-bundle:einstein:abc123",
        figure_id="einstein",
        lora=_FakeArtifact(
            figure_id="einstein",
            backend_id="peft-v1",
            training_plan_hash="plan-hash",
            adapter_layers=_adapter_layers(),
            parameter_count=128,
            description="einstein persona lora",
            peft_checkpoint_dir=checkpoint_dir,
        ),
    )


def test_register_forwards_peft_checkpoint_dir(tmp_path) -> None:
    checkpoint = tmp_path / "peft-checkpoints" / "einstein" / "abc123"
    checkpoint.mkdir(parents=True)
    pool = PersonaLoRAPool()

    record_id = register_bundle_persona_lora(
        _bundle_with_checkpoint(str(checkpoint)), pool=pool
    )

    assert record_id is not None
    record = pool.lookup("einstein")
    assert record.peft_checkpoint_dir == str(checkpoint.resolve())


def test_register_resolves_against_configured_root(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "served-root"
    served_checkpoint = root / "einstein" / "abc123"
    served_checkpoint.mkdir(parents=True)
    monkeypatch.setenv("VZ_PEFT_CHECKPOINT_ROOT", str(root))
    pool = PersonaLoRAPool()

    # The bundle records a path baked under a *different* root; the
    # resolver re-roots it under the configured root via the stable
    # ``peft-checkpoints/<figure>/<hash>`` tail.
    baked_elsewhere = (
        "/some/other/machine/.local/peft-checkpoints/einstein/abc123"
    )
    register_bundle_persona_lora(
        _bundle_with_checkpoint(baked_elsewhere), pool=pool
    )

    record = pool.lookup("einstein")
    assert record.peft_checkpoint_dir == str(served_checkpoint.resolve())


def test_register_returns_none_without_lora() -> None:
    pool = PersonaLoRAPool()
    bundle = _FakeBundle(
        bundle_id="figure-bundle:einstein:nolora",
        figure_id="einstein",
        lora=None,
    )
    assert register_bundle_persona_lora(bundle, pool=pool) is None
    assert not pool.has("einstein")


class _FakeRuntime:
    """Records which activation path the pool chose."""

    def __init__(self) -> None:
        self.peft_calls: list[str] = []
        self.hook_calls: list[object] = []

    @contextlib.contextmanager
    def activate_peft_adapter(self, checkpoint_dir):
        self.peft_calls.append(str(checkpoint_dir))
        yield

    @contextlib.contextmanager
    def activate_lora(self, layers):
        self.hook_calls.append(layers)
        yield


def test_adopted_figure_activates_peft_path_not_hook(tmp_path) -> None:
    checkpoint = tmp_path / "peft-checkpoints" / "einstein" / "abc123"
    checkpoint.mkdir(parents=True)
    pool = PersonaLoRAPool()
    register_bundle_persona_lora(
        _bundle_with_checkpoint(str(checkpoint)), pool=pool
    )

    runtime = _FakeRuntime()
    with pool.activate("einstein", runtime=runtime):
        pass

    assert runtime.peft_calls == [str(checkpoint.resolve())]
    assert runtime.hook_calls == []


def test_register_skips_when_adapter_policy_forbids(tmp_path) -> None:
    checkpoint = tmp_path / "peft-checkpoints" / "einstein" / "abc123"
    checkpoint.mkdir(parents=True)
    pool = PersonaLoRAPool()
    record_id = register_bundle_persona_lora(
        _bundle_with_checkpoint(str(checkpoint)),
        pool=pool,
        persona_lora_enabled=False,
    )
    assert record_id is None
    assert not pool.has("einstein")


def test_scoped_pools_isolate_same_figure_id(tmp_path) -> None:
    # Two tenants adopt DIFFERENT bundles for the same figure_id; in the
    # global default pool this is last-register-wins, but per-ai_id
    # scoped pools must keep distinct records.
    ck_a = tmp_path / "peft-checkpoints" / "einstein" / "tenantA"
    ck_b = tmp_path / "peft-checkpoints" / "einstein" / "tenantB"
    ck_a.mkdir(parents=True)
    ck_b.mkdir(parents=True)

    pool_a = PersonaLoRAPool()
    pool_b = PersonaLoRAPool()

    bundle_a = _bundle_with_checkpoint(str(ck_a))
    bundle_a.bundle_id = "figure-bundle:einstein:tenantA"
    bundle_b = _bundle_with_checkpoint(str(ck_b))
    bundle_b.bundle_id = "figure-bundle:einstein:tenantB"

    register_bundle_persona_lora(bundle_a, pool=pool_a)
    register_bundle_persona_lora(bundle_b, pool=pool_b)

    assert pool_a.lookup("einstein").peft_checkpoint_dir == str(ck_a.resolve())
    assert pool_b.lookup("einstein").peft_checkpoint_dir == str(ck_b.resolve())
