#!/usr/bin/env python3
"""Train a target-model residual adapter from Zhang Wuji live-through traces.

The Qwen base is frozen. Only one bounded residual vector per selected target
layer is optimized against the reviewed action/outcome completions. The
resulting artifact is target-model-specific and is separate from Prefix/KV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.substrate import (  # noqa: E402
    CHARACTER_RESIDUAL_ADAPTER_MODE,
    CHARACTER_RESIDUAL_DELTA_CAP,
    CharacterResidualAdapterPackage,
    SubstrateDeltaAdapterLayer,
)


CHARACTER_ID = "zhang-wuji"
CHARACTER_NAME = "张无忌"
SOURCE_LIVE_THROUGH_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TARGET_SNAPSHOT = (
    REPO_ROOT
    / ".local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "artifacts/lifeform-templates/zhang_wuji/zhang-wuji-live-through.json"
)
DEFAULT_PROOF = (
    REPO_ROOT
    / "artifacts/character-live-through/zhang_wuji.ch-11.bake-proof.json"
)
DEFAULT_LEDGER = (
    REPO_ROOT
    / "artifacts/character-live-through/zhang_wuji.reviewed_ledger.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/character-packages/zhang_wuji/"
    "zhang-wuji-qwen2.5-1.5b.character-residual.json"
)


@dataclass(frozen=True)
class TrainingRow:
    prompt: str
    target: str
    scene_id: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=TARGET_MODEL_ID)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_TARGET_SNAPSHOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--layers", default="13,14,15")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-sequence-tokens", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--delta-cap", type=float, default=CHARACTER_RESIDUAL_DELTA_CAP)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path, *, max_cases: int) -> tuple[TrainingRow, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("chapters"), list):
        raise ValueError("reviewed ledger must contain a chapters list")
    rows: list[TrainingRow] = []
    for chapter in raw["chapters"]:
        if not isinstance(chapter, dict):
            raise ValueError("reviewed ledger chapter must be an object")
        for scene in chapter.get("scenes", []):
            if not isinstance(scene, dict):
                raise ValueError("reviewed ledger scene must be an object")
            setting = str(scene.get("setting") or "").strip()
            decision = str(scene.get("decision_point") or "").strip()
            action = str(scene.get("canonical_action") or "").strip()
            outcome = str(scene.get("canonical_outcome") or "").strip()
            if not setting or not decision or not action:
                continue
            rows.append(
                TrainingRow(
                    prompt=(
                        f"场景：{setting}\n"
                        f"抉择：{decision}\n"
                        "请以张无忌的第一人称说明现在的行动和直接理由。"
                    ),
                    target=f"{action}。{outcome}" if outcome else action,
                    scene_id=str(scene.get("scene_id") or "unknown"),
                )
            )
            if len(rows) >= max_cases:
                return tuple(rows)
    if not rows:
        raise ValueError("reviewed ledger has no usable decision scenes")
    return tuple(rows)


def _resolve_device(torch: Any, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_layers(raw: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("--layers must contain non-negative layer indices")
    return layers


def _chat_prompt(tokenizer: Any, row: TrainingRow) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "你是张无忌。你亲历光明顶之后的江湖局势，"
                    "止杀、守义、体恤无辜为先，但仁恕不等于纵容。"
                ),
            },
            {"role": "user", "content": row.prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_training_rows(
    *,
    tokenizer: Any,
    rows: tuple[TrainingRow, ...],
    torch: Any,
    device: Any,
    max_sequence_tokens: int,
    max_target_tokens: int,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for row in rows:
        prompt_ids = tokenizer(
            _chat_prompt(tokenizer, row),
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence_tokens,
        )["input_ids"]
        target_ids = tokenizer(
            row.target,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][..., :max_target_tokens]
        full_ids = torch.cat((prompt_ids, target_ids), dim=-1)
        if full_ids.shape[1] < 2:
            continue
        if full_ids.shape[1] > max_sequence_tokens:
            full_ids = full_ids[:, -max_sequence_tokens:]
            prompt_length = max(0, max_sequence_tokens - int(target_ids.shape[1]))
        else:
            prompt_length = int(prompt_ids.shape[1])
        labels = full_ids.clone()
        labels[:, :prompt_length] = -100
        if bool((labels != -100).any().item()) is False:
            continue
        result.append(
            {
                "input_ids": full_ids.to(device),
                "attention_mask": torch.ones_like(full_ids, device=device),
                "labels": labels.to(device),
                "scene_id": row.scene_id,
            }
        )
    if not result:
        raise ValueError("no training rows survived tokenization")
    return tuple(result)


def _make_hooks(*, model: Any, blocks: tuple[Any, ...], deltas: dict[int, Any]):
    hooks = []
    for layer_index, delta in deltas.items():
        def hook(module, args, output, delta=delta):
            del module, args
            hidden = output[0] if isinstance(output, tuple) else output
            adjusted = hidden + delta.to(dtype=hidden.dtype).view(1, 1, -1)
            if isinstance(output, tuple):
                return (adjusted, *output[1:])
            return adjusted

        hooks.append(blocks[layer_index].register_forward_hook(hook))
    return hooks


def _mean_loss(model: Any, rows: tuple[dict[str, Any], ...]) -> float:
    torch = __import__("torch")
    losses = []
    with torch.no_grad():
        for row in rows:
            outputs = model(
                input_ids=row["input_ids"],
                attention_mask=row["attention_mask"],
                labels=row["labels"],
            )
            losses.append(float(outputs.loss.detach().cpu().item()))
    return sum(losses) / len(losses)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.max_cases <= 0:
        raise ValueError("epochs and max-cases must be positive")
    if args.delta_cap <= 0.0 or args.delta_cap > CHARACTER_RESIDUAL_DELTA_CAP:
        raise ValueError(
            f"delta-cap must be in (0, {CHARACTER_RESIDUAL_DELTA_CAP}]"
        )
    for required in (args.model_source, args.template, args.proof, args.ledger):
        if not required.exists():
            raise FileNotFoundError(required)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(torch, args.device)
    layers = _parse_layers(args.layers)
    rows = _load_rows(args.ledger, max_cases=args.max_cases)
    tokenizer = AutoTokenizer.from_pretrained(args.model_source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_source,
        local_files_only=True,
        torch_dtype=torch.float32 if device.type == "cpu" else None,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = model.config
    hidden_size = int(config.hidden_size)
    blocks = tuple(model.model.layers)
    if any(layer >= len(blocks) for layer in layers):
        raise ValueError(
            f"requested layers {layers} exceed model block count {len(blocks)}"
        )
    training_rows = _build_training_rows(
        tokenizer=tokenizer,
        rows=rows,
        torch=torch,
        device=device,
        max_sequence_tokens=args.max_sequence_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    parameters = {
        layer: torch.nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32, device=device)
        )
        for layer in layers
    }
    hooks = _make_hooks(model=model, blocks=blocks, deltas=parameters)
    try:
        initial_loss = _mean_loss(model, training_rows)
        optimizer = torch.optim.AdamW(
            tuple(parameters.values()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for _epoch in range(args.epochs):
            for row in training_rows:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=row["input_ids"],
                    attention_mask=row["attention_mask"],
                    labels=row["labels"],
                )
                outputs.loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for parameter in parameters.values():
                        parameter.clamp_(-args.delta_cap, args.delta_cap)
        final_loss = _mean_loss(model, training_rows)
    finally:
        for hook in hooks:
            hook.remove()

    adapter_layers = tuple(
        SubstrateDeltaAdapterLayer(
            layer_index=layer,
            delta_vector=tuple(
                max(-args.delta_cap, min(args.delta_cap, float(value)))
                for value in parameters[layer].detach().cpu().tolist()
            ),
            mean_abs_delta=float(parameters[layer].detach().abs().mean().cpu().item()),
            description=(
                f"Target-model teacher-forced Zhang Wuji residual delta; "
                f"layer={layer} hidden_size={hidden_size}."
            ),
        )
        for layer in layers
    )
    template_raw = json.loads(args.template.read_text(encoding="utf-8"))
    manifest = template_raw.get("manifest")
    if not isinstance(manifest, dict) or not str(manifest.get("integrity_hash", "")).strip():
        raise ValueError("template manifest must carry an integrity_hash")
    package = CharacterResidualAdapterPackage.create(
        character_id=CHARACTER_ID,
        character_name=CHARACTER_NAME,
        model_id=args.model_id,
        source_live_through_model_id=SOURCE_LIVE_THROUGH_MODEL_ID,
        source_template_id=str(manifest.get("template_id", "zhang-wuji-live-through")),
        source_template_integrity_hash=str(manifest["integrity_hash"]),
        source_live_through_proof=(
            f"{args.proof.relative_to(REPO_ROOT)};sha256={_sha256(args.proof)}"
        ),
        hidden_size=hidden_size,
        adapter_layers=adapter_layers,
        training_mode=CHARACTER_RESIDUAL_ADAPTER_MODE,
        training_loss=final_loss,
        sample_count=len(training_rows),
        description=(
            f"Frozen {args.model_id} residual adapter trained from "
            f"{len(training_rows)} reviewed Zhang Wuji live-through scenes; "
            f"initial_loss={initial_loss:.6f} final_loss={final_loss:.6f}."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(package.to_json() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "package_id": package.package_id,
                "model_id": package.model_id,
                "layers": package.layer_indices,
                "hidden_size": package.hidden_size,
                "sample_count": package.sample_count,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
