"""Offline forward hooks for distilling artifacts on an admitted adapter.

The runtime and both offline distillers must observe the same residual delta.
This module is substrate-owned and deliberately contains no training policy;
it only installs a validated ``SubstrateRareHeavyCheckpoint`` on a frozen
Transformers model and returns removable hook handles.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from volvence_zero.substrate.residual_contracts import (
    SubstrateRareHeavyCheckpoint,
)


def resolve_transformer_blocks(model: Any) -> tuple[Any, ...]:
    for path in (
        ("transformer", "h"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
        ("backbone", "layers"),
        ("layers",),
    ):
        current = model
        for segment in path:
            try:
                current = getattr(current, segment)
            except AttributeError:
                break
        else:
            try:
                blocks = tuple(current)
            except TypeError:
                continue
            if blocks:
                return blocks
    raise ValueError(
        f"could not resolve transformer blocks for {type(model).__name__}."
    )


def resolve_hidden_size(model: Any) -> int:
    for field in ("hidden_size", "n_embd", "d_model"):
        value = getattr(model.config, field, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        f"could not resolve hidden size for {type(model).__name__}."
    )


def install_rare_heavy_checkpoint_hooks(
    *,
    model: Any,
    checkpoint: SubstrateRareHeavyCheckpoint,
    expected_model_id: str,
) -> tuple[Any, ...]:
    """Install constant residual deltas and return their hook handles."""

    if checkpoint.model_id != expected_model_id:
        raise ValueError(
            "rare-heavy checkpoint model_id does not match offline model: "
            f"checkpoint={checkpoint.model_id!r}, expected={expected_model_id!r}."
        )
    if not checkpoint.adapter_layers:
        raise ValueError(
            "offline adapter activation requires a non-empty rare-heavy payload."
        )
    blocks = resolve_transformer_blocks(model)
    hidden_size = resolve_hidden_size(model)
    indices = tuple(layer.layer_index for layer in checkpoint.adapter_layers)
    if len(indices) != len(set(indices)):
        raise ValueError("rare-heavy checkpoint layer indices must be unique.")
    unavailable = sorted(index for index in indices if index not in range(len(blocks)))
    if unavailable:
        raise ValueError(
            f"rare-heavy checkpoint targets unavailable layers {unavailable}."
        )
    scale = float(checkpoint.adapter_scale)
    if scale <= 0.0:
        raise ValueError("rare-heavy checkpoint adapter_scale must be positive.")

    torch = __import__("torch")

    def make_hook(delta: Sequence[float]):
        if len(delta) != hidden_size:
            raise ValueError(
                "rare-heavy checkpoint delta width does not match model hidden "
                f"size: got={len(delta)}, expected={hidden_size}."
            )
        vector = torch.tensor(tuple(delta), dtype=torch.float32)

        def hook(module: Any, args: tuple[Any, ...], output: Any):
            del module
            del args
            hidden = output[0] if isinstance(output, tuple) else output
            if not hasattr(hidden, "shape") or hidden.shape[-1] != hidden_size:
                raise ValueError(
                    "transformer block output is incompatible with the admitted "
                    "rare-heavy delta."
                )
            adjusted = hidden + vector.to(
                device=hidden.device,
                dtype=hidden.dtype,
            ).view(1, 1, -1) * scale
            if isinstance(output, tuple):
                return (adjusted, *output[1:])
            return adjusted

        return hook

    return tuple(
        blocks[layer.layer_index].register_forward_hook(
            make_hook(layer.delta_vector)
        )
        for layer in checkpoint.adapter_layers
    )


def remove_forward_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


__all__ = [
    "install_rare_heavy_checkpoint_hooks",
    "remove_forward_hooks",
    "resolve_hidden_size",
    "resolve_transformer_blocks",
]
