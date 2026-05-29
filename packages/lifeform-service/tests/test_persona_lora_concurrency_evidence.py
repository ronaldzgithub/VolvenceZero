"""Concurrency evidence for per-tenant persona-LoRA isolation (debt #61/#45).

Drives N concurrent "turns", each a different tenant with its own
SessionManager-scoped :class:`PersonaLoRAPool` registering a DISTINCT
checkpoint for the SAME ``figure_id``. Asserts every turn activates its
own checkpoint with no cross-tenant leakage even when the activation
contexts interleave on the event loop.

This is the load-bearing correctness evidence for the concurrent
multi-LoRA path: the process-wide default pool is last-register-wins, but
the scoped pools (one per ai_id) keep tenants isolated.
"""

from __future__ import annotations

import asyncio
import contextlib

from lifeform_expression.llm_synthesizer import _maybe_activate_persona_lora
from volvence_zero.substrate import (
    PersonaLoRAPool,
    SubstrateDeltaAdapterLayer,
)

_FIGURE = "einstein"
_TENANTS = 12


class _Bundle:
    def __init__(self, figure_id: str) -> None:
        self.figure_id = figure_id


class _RecordingRuntime:
    """LoRA-aware fake that records which checkpoint it activated."""

    def __init__(self) -> None:
        self.activated: list[str] = []

    @contextlib.contextmanager
    def activate_peft_adapter(self, checkpoint_dir):
        self.activated.append(str(checkpoint_dir))
        yield

    @contextlib.contextmanager
    def activate_lora(self, layers):  # pragma: no cover - not used here
        yield


def _scoped_pool_for(tenant: int) -> PersonaLoRAPool:
    pool = PersonaLoRAPool()
    pool.register(
        figure_id=_FIGURE,
        source_bundle_id=f"figure-bundle:{_FIGURE}:tenant{tenant}",
        backend_id="peft-v1",
        training_plan_hash=f"plan-{tenant}",
        adapter_layers=(
            SubstrateDeltaAdapterLayer(
                layer_index=0,
                delta_vector=(0.01 * tenant, -0.01 * tenant),
                mean_abs_delta=0.01 * tenant + 0.001,
                description=f"tenant-{tenant}",
            ),
        ),
        parameter_count=16,
        peft_checkpoint_dir=f"/ck/tenant{tenant}/{_FIGURE}",
    )
    return pool


async def test_no_cross_tenant_lora_leakage_under_concurrency() -> None:
    bundle = _Bundle(_FIGURE)

    async def turn(tenant: int) -> tuple[int, list[str]]:
        pool = _scoped_pool_for(tenant)
        runtime = _RecordingRuntime()
        with _maybe_activate_persona_lora(
            bundle=bundle, runtime=runtime, enabled=True, pool=pool
        ):
            # Interleave with other tenants while "inside" activation.
            await asyncio.sleep(0)
        return tenant, runtime.activated

    results = await asyncio.gather(*(turn(t) for t in range(_TENANTS)))

    for tenant, activated in results:
        assert activated == [f"/ck/tenant{tenant}/{_FIGURE}"], (
            f"tenant {tenant} activated {activated!r}; cross-tenant leakage"
        )


async def test_disabled_policy_blocks_all_activation_under_concurrency() -> None:
    bundle = _Bundle(_FIGURE)

    async def turn(tenant: int) -> list[str]:
        pool = _scoped_pool_for(tenant)
        runtime = _RecordingRuntime()
        with _maybe_activate_persona_lora(
            bundle=bundle, runtime=runtime, enabled=False, pool=pool
        ):
            await asyncio.sleep(0)
        return runtime.activated

    results = await asyncio.gather(*(turn(t) for t in range(_TENANTS)))
    assert all(activated == [] for activated in results)
