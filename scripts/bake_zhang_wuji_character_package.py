#!/usr/bin/env python3
"""Bake a Zhang Wuji model-side Prefix/KV package for Qwen 1.5B.

The reviewed chapter ledger is the source of training cases. The Qwen model is
frozen; only one static per-layer K/V prefix is optimized against the reviewed
canonical actions. The resulting package contains no dialogue memory and is
loaded independently from the LifeformTemplate.

Usage:
    python scripts/bake_zhang_wuji_character_package.py --device mps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.substrate import (  # noqa: E402
    CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
    CharacterPrefixKVPackage,
    build_teacher_distilled_prefix_artifact,
)


CHARACTER_ID = "zhang-wuji"
CHARACTER_NAME = "张无忌"
SOURCE_LIVE_THROUGH_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CHARACTER_VECTOR_LABELS = ("zhang_wuji_live_through_identity",)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=REPO_ROOT
        / ".local/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=REPO_ROOT
        / "artifacts/lifeform-templates/zhang_wuji/zhang-wuji-live-through.json",
    )
    parser.add_argument(
        "--proof",
        type=Path,
        default=REPO_ROOT
        / "artifacts/character-live-through/zhang_wuji.ch-11.bake-proof.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT
        / "artifacts/character-live-through/zhang_wuji.reviewed_ledger.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/character-packages/zhang_wuji/"
        "zhang-wuji-qwen2.5-1.5b.character-prefix.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-target-tokens", type=int, default=96)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--norm-cap", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases(path: Path, *, max_cases: int) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("chapters"), list):
        raise ValueError("reviewed ledger must contain a chapters list")
    cases: list[dict[str, str]] = []
    for chapter in raw["chapters"]:
        if not isinstance(chapter, dict):
            raise ValueError("reviewed ledger chapter must be an object")
        for scene in chapter.get("scenes", []):
            if not isinstance(scene, dict):
                raise ValueError("reviewed ledger scene must be an object")
            decision = str(scene.get("decision_point") or "").strip()
            action = str(scene.get("canonical_action") or "").strip()
            outcome = str(scene.get("canonical_outcome") or "").strip()
            setting = str(scene.get("setting") or "").strip()
            if not decision or not action:
                continue
            cases.append(
                {
                    "prompt": f"场景：{setting}\n抉择：{decision}",
                    "target": f"{action}。{outcome}" if outcome else action,
                    "scene_id": str(scene.get("scene_id") or "unknown"),
                }
            )
            if len(cases) >= max_cases:
                return cases
    if not cases:
        raise ValueError("reviewed ledger has no trainable decision scenes")
    return cases


def _device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _messages(tokenizer: Any, prompt: str) -> Any:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "你是张无忌。你的判断以止杀、守义、体恤无辜为先，"
                    "但不把仁恕当成纵容；回答眼前的抉择，简明说明取舍。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _reference_norms(torch: Any, model: Any, prompts: list[Any], layers: int):
    key_totals = [0.0] * layers
    value_totals = [0.0] * layers
    for input_ids in prompts:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        cache = outputs.past_key_values
        for index in range(layers):
            layer = cache.layers[index]
            key_totals[index] += float(layer.keys.to(torch.float32).norm(dim=-1).mean())
            value_totals[index] += float(layer.values.to(torch.float32).norm(dim=-1).mean())
    count = float(len(prompts))
    return (
        [max(total / count, 1e-6) for total in key_totals],
        [max(total / count, 1e-6) for total in value_totals],
    )


def _prefix_pairs(
    torch: Any,
    raw_key: Any,
    raw_value: Any,
    key_caps: Any,
    value_caps: Any,
    *,
    dtype: Any,
):
    def capped(raw: Any, caps: Any) -> Any:
        norms = raw.norm(dim=-1, keepdim=True)
        limits = caps.reshape(-1, 1, 1, 1)
        return raw * torch.clamp(limits / norms.clamp_min(1e-8), max=1.0)

    keys = capped(raw_key, key_caps).permute(0, 2, 1, 3).unsqueeze(1).to(dtype)
    values = capped(raw_value, value_caps).permute(0, 2, 1, 3).unsqueeze(1).to(dtype)
    return [(keys[index], values[index]) for index in range(raw_key.shape[0])]


def _loss_for_case(
    torch: Any,
    transformers: Any,
    model: Any,
    tokenizer: Any,
    case: dict[str, str],
    prefix_pairs: Any,
    device: Any,
    max_target_tokens: int,
):
    prompt_ids = tokenizer(
        _messages(tokenizer, case["prompt"]), return_tensors="pt"
    )["input_ids"].to(device)
    target_ids = tokenizer(
        case["target"], add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)[..., :max_target_tokens]
    if int(target_ids.shape[-1]) == 0:
        raise ValueError(f"scene {case['scene_id']} produced an empty target")
    full_ids = torch.cat((prompt_ids, target_ids), dim=-1)
    slots = int(prefix_pairs[0][0].shape[-2])
    attention_mask = torch.ones(
        (1, slots + int(full_ids.shape[-1])), dtype=torch.long, device=device
    )
    position_ids = torch.arange(int(full_ids.shape[-1]), device=device).unsqueeze(0)
    cache = transformers.DynamicCache(ddp_cache_data=prefix_pairs)
    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    prompt_length = int(prompt_ids.shape[-1])
    logits = outputs.logits[:, prompt_length - 1 : -1, :].to(torch.float32)
    labels = full_ids[:, prompt_length:]
    if logits.shape[1] != labels.shape[1]:
        raise RuntimeError("prefix teacher-forcing logits and labels are misaligned")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.max_cases <= 0 or args.slots <= 0:
        raise ValueError("epochs, max-cases, and slots must be positive")
    if not 0.0 < args.norm_cap <= 0.5:
        raise ValueError("norm-cap must be in (0, 0.5]")
    for required in (args.template, args.proof, args.ledger, args.model_source):
        if not required.exists():
            raise FileNotFoundError(required)

    template = json.loads(args.template.read_text(encoding="utf-8"))
    manifest = template.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("template is missing manifest")
    if manifest.get("character_id") != CHARACTER_ID:
        raise ValueError("template character_id is not zhang-wuji")
    template_integrity = str(manifest.get("integrity_hash") or "")
    if not template_integrity:
        raise ValueError("template manifest has no integrity_hash")
    proof_digest = _sha256(args.proof)
    source_fingerprint = hashlib.sha256(
        (args.template.read_bytes() + args.proof.read_bytes() + args.ledger.read_bytes())
    ).hexdigest()
    cases = _load_cases(args.ledger, max_cases=args.max_cases)

    import torch
    import transformers

    torch.manual_seed(args.seed)
    device = _device(torch, args.device)
    dtype = torch.float16 if str(device).startswith(("mps", "cuda")) else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model_source), local_files_only=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.model_source), local_files_only=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = model.config
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size // config.num_attention_heads)
    prompts = [
        tokenizer(_messages(tokenizer, case["prompt"]), return_tensors="pt")["input_ids"].to(device)
        for case in cases[: min(4, len(cases))]
    ]
    reference_keys, reference_values = _reference_norms(
        torch, model, prompts, layers
    )
    width = args.slots * kv_heads * head_dim
    raw_keys = (
        torch.randn(
            (layers, args.slots, kv_heads, head_dim), device=device, dtype=torch.float32
        )
        * 0.01
    ).requires_grad_(True)
    raw_values = (
        torch.randn(
            (layers, args.slots, kv_heads, head_dim), device=device, dtype=torch.float32
        )
        * 0.01
    ).requires_grad_(True)
    key_caps = torch.tensor(reference_keys, device=device) * args.norm_cap
    value_caps = torch.tensor(reference_values, device=device) * args.norm_cap
    optimizer = torch.optim.AdamW((raw_keys, raw_values), lr=args.learning_rate)

    for epoch in range(args.epochs):
        total = 0.0
        for case in cases:
            optimizer.zero_grad(set_to_none=True)
            pairs = _prefix_pairs(
                torch, raw_keys, raw_values, key_caps, value_caps, dtype=dtype
            )
            loss = _loss_for_case(
                torch,
                transformers,
                model,
                tokenizer,
                case,
                pairs,
                device,
                args.max_target_tokens,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_((raw_keys, raw_values), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        print(
            f"[train] epoch={epoch + 1}/{args.epochs} "
            f"mean_loss={total / len(cases):.4f} device={device}",
            flush=True,
        )

    with torch.no_grad():
        trained_keys = _prefix_pairs(
            torch, raw_keys, raw_values, key_caps, value_caps, dtype=dtype
        )
        key_bias = [pair[0].squeeze(0).permute(1, 0, 2).reshape(-1).cpu().tolist() for pair in trained_keys]
        value_bias = [pair[1].squeeze(0).permute(1, 0, 2).reshape(-1).cpu().tolist() for pair in trained_keys]
    artifact = build_teacher_distilled_prefix_artifact(
        model_id=args.model_id,
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        num_slots=args.slots,
        bottleneck_rank=1,
        encoder_rows=((0.0,),),
        encoder_bias=(0.0,),
        key_projection=tuple(tuple((0.0,) for _ in range(width)) for _ in range(layers)),
        key_bias=tuple(tuple(float(value) for value in block) for block in key_bias),
        value_projection=tuple(tuple((0.0,) for _ in range(width)) for _ in range(layers)),
        value_bias=tuple(tuple(float(value) for value in block) for block in value_bias),
        reference_key_norms=reference_keys,
        reference_value_norms=reference_values,
        norm_cap=args.norm_cap,
        source_fingerprint=source_fingerprint,
        sample_count=len(cases) * args.epochs,
        training_mode=CHARACTER_TEACHER_FORCED_PREFIX_TRAINING_MODE,
        vector_labels=CHARACTER_VECTOR_LABELS,
        description=(
            "Static Zhang Wuji character Prefix/KV distilled from reviewed "
            "chapter-live-through canonical decisions on frozen Qwen 1.5B; "
            f"cases={len(cases)} epochs={args.epochs} slots={args.slots}."
        ),
    )
    package = CharacterPrefixKVPackage.create(
        character_id=CHARACTER_ID,
        character_name=CHARACTER_NAME,
        model_id=args.model_id,
        source_live_through_model_id=SOURCE_LIVE_THROUGH_MODEL_ID,
        source_template_id=str(manifest["template_id"]),
        source_template_integrity_hash=template_integrity,
        source_live_through_proof=f"{args.proof}:{proof_digest}",
        state_vector=(0.0,),
        prefix_artifact=artifact,
        description=(
            "Zhang Wuji model-side character package: reviewed 0.5B "
            "live-through provenance is retained, while the Prefix/KV carrier "
            "is baked against the frozen Qwen 1.5B substrate."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(package.to_json() + "\n", encoding="utf-8")
    print(
        f"[package] wrote {args.output} package_id={package.package_id} "
        f"artifact_id={artifact.artifact_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
