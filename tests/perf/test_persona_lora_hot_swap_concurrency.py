"""F-A subtask 3: ``PersonaLoRAPool`` hot-swap concurrency.

Verifies that N=10 concurrent sessions, each activating a distinct
figure LoRA on the same shared ``TransformersOpenWeightResidualRuntime``
(Wave D), get correct per-session logits with no race condition and
that the frozen base ``state_dict_hash`` is byte-identical before /
during / after activation (R2 守门).

Debts: known-debts.md #45 (perf 床) + #20 closure / #61 (LoRA 并发)
Packets:
* docs/moving forward/cross-cutting-foundation-packet.md §2.1
* docs/moving forward/figure-evidence-packet.md §2.4 (#61)
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.perf, pytest.mark.hf]


N_CONCURRENT_FIGURES: int = 10


def test_persona_lora_concurrent_activation_isolation(
    asyncio_harness,  # noqa: ANN001
    gpu_mem_tracker,  # noqa: ANN001
) -> None:
    """SHADOW scaffold."""

    pytest.skip(
        f"SHADOW scaffold: PersonaLoRAPool concurrent activation × "
        f"{N_CONCURRENT_FIGURES} figures test lands once F-A subtask 3 "
        "and figure-evidence #61 ship. Target: per-session logits "
        "deterministic; frozen base state_dict_hash unchanged "
        "throughout. See cross-cutting-foundation-packet §2.1 + "
        "figure-evidence-packet §2.4."
    )


def test_persona_lora_swap_overhead_under_target(
    asyncio_harness,  # noqa: ANN001
) -> None:
    """SHADOW scaffold: per-turn LoRA swap overhead vs no-swap baseline."""

    pytest.skip(
        "SHADOW scaffold: per-turn LoRA swap overhead measurement "
        "lands with F-A subtask 3."
    )
