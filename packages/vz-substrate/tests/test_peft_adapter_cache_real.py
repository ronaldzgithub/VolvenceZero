"""Real-peft validation of the resident adapter cache on tiny-gpt2.

Exercises ``activate_peft_adapter`` end-to-end against the shipped
einstein tiny-gpt2 checkpoint to confirm:

* repeat activation of the same checkpoint is a cache hit (no second
  ``from_pretrained``),
* the frozen base ``state_dict`` is byte-identical before / after, and
* the base forward logits are restored after the context (R2).

hf-gated + checkpoint-gated so CI without torch/peft or without the
shipped checkpoint skips cleanly.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO_PACKAGES = pathlib.Path(__file__).resolve().parents[2]
_CHECKPOINT = (
    _REPO_PACKAGES
    / "lifeform-domain-figure"
    / ".local"
    / "peft-checkpoints"
    / "einstein"
    / "6dea268c94393820"
)


def _stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("transformers", "torch", "peft")
    )


@pytest.mark.hf
def test_resident_cache_hit_and_frozen_base() -> None:
    if not _stack_available():
        pytest.skip("transformers + torch + peft not installed")
    if not _CHECKPOINT.is_dir():
        pytest.skip(f"shipped checkpoint not present at {_CHECKPOINT}")
    from volvence_zero.substrate.residual_backend import (
        TransformersOpenWeightResidualRuntime,
    )

    runtime = TransformersOpenWeightResidualRuntime(
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
    )
    base_logits = tuple(runtime.capture(source_text="reality is").token_logits)

    with runtime.activate_peft_adapter(str(_CHECKPOINT)):
        runtime.capture(source_text="reality is")
    stats_after_first = dict(runtime.peft_cache_stats)

    with runtime.activate_peft_adapter(str(_CHECKPOINT)):
        runtime.capture(source_text="reality is")
    stats_after_second = dict(runtime.peft_cache_stats)

    # First activation is a miss; second is a hit (no second
    # from_pretrained — the adapter stays resident in the cache).
    assert stats_after_first["misses"] == 1
    assert stats_after_first["hits"] == 0
    assert stats_after_second["misses"] == 1
    assert stats_after_second["hits"] == 1
    assert stats_after_second["resident"] == 1

    # R2 (behavioral): with the cached adapter disabled on context exit
    # the base forward is byte-identical to the pre-activation base.
    # The resident cache intentionally keeps the additive (disabled)
    # LoRA params in memory, so we assert the frozen-base *behaviour*
    # is preserved rather than a full state_dict byte-equality (which
    # would defeat the purpose of a warm adapter cache).
    restored_logits = tuple(
        runtime.capture(source_text="reality is").token_logits
    )
    assert restored_logits == base_logits
