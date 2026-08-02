#!/usr/bin/env python3
"""Train a dedicated 14-dimensional Relationship Prefix-KV artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in [
    Path(__file__).resolve().parent,
    *sorted((REPO_ROOT / "packages").glob("*/src")),
]:
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_bank_gain_gate import (  # noqa: E402
    GAIN_PROBES,
    IRRELEVANT_PROBES,
)
from run_state_kv_identification import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _fingerprint_weights,
    _resolve_local_weights,
)
from train_state_kv_prefix import (  # noqa: E402
    DEFAULT_ROUTE_TEMPERATURE,
    PrefixGenerator,
    _encode,
    _measure_reference_norms,
    _student_logprobs,
    _student_route_loss,
    _target_tokens_from_text,
)
from volvence_zero.relationship_conditioning import (  # noqa: E402
    RELATIONSHIP_CONDITIONING_COMPILER_VERSION,
    RELATIONSHIP_CONDITIONING_READOUT_LABELS,
)
from volvence_zero.substrate import (  # noqa: E402
    MAX_PREFIX_NORM_CAP,
    PrefixKVArtifact,
    bind_relationship_prefix_artifact,
    build_teacher_distilled_prefix_artifact,
)
from volvence_zero.substrate.prefix_kv_artifact import (  # noqa: E402
    STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE,
)

DEFAULT_MATERIAL = (
    REPO_ROOT
    / "packages"
    / "vz-substrate"
    / "src"
    / "volvence_zero"
    / "substrate"
    / "prompts"
    / "relationship_prefix_kv_training.json"
)
DEFAULT_WARM_START = (
    REPO_ROOT
    / "artifacts"
    / "state_kv"
    / "projectors"
    / "qwen2.5-0.5b-state-strategy-routed-prefix.json"
)
SYSTEM_PROMPT = (
    "You are a careful assistant. Respond to the user directly and keep the "
    "next step bounded."
)


def _load_material(path: Path) -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[str, ...],
    str,
]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = tuple(str(value) for value in raw["readout_labels"])
    if labels != RELATIONSHIP_CONDITIONING_READOUT_LABELS:
        raise ValueError(
            "Relationship Prefix-KV material labels must match the owner "
            "readout exactly."
        )
    owner_version = str(raw["owner_schema_version"])
    if owner_version != RELATIONSHIP_CONDITIONING_COMPILER_VERSION:
        raise ValueError(
            "Relationship Prefix-KV material owner_schema_version is stale: "
            f"{owner_version!r} != "
            f"{RELATIONSHIP_CONDITIONING_COMPILER_VERSION!r}."
        )
    repair = tuple(float(value) for value in raw["endpoints"]["repair"])
    steady = tuple(float(value) for value in raw["endpoints"]["steady"])
    probes = tuple(str(value).strip() for value in raw["training_probes"])
    if len(repair) != len(labels) or len(steady) != len(labels):
        raise ValueError("Relationship Prefix-KV endpoints have wrong width.")
    if not probes or not all(probes):
        raise ValueError("Relationship Prefix-KV probes must be non-empty.")
    evaluation = {text for _, text in (*GAIN_PROBES, *IRRELEVANT_PROBES)}
    overlap = set(probes) & evaluation
    if overlap:
        raise ValueError(
            "Relationship Prefix-KV training probes overlap pilot probes: "
            f"{sorted(overlap)}"
        )
    return labels, repair, steady, probes, owner_version


def _sample_states(
    *,
    repair: tuple[float, ...],
    steady: tuple[float, ...],
    count: int,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    if count < 2:
        raise ValueError("Relationship Prefix-KV needs at least two states.")
    rng = random.Random(seed)
    center = tuple(
        (repair_value + steady_value) / 2.0
        for repair_value, steady_value in zip(repair, steady, strict=True)
    )
    primary = tuple(
        (steady_value - repair_value) / 2.0
        for repair_value, steady_value in zip(repair, steady, strict=True)
    )
    primary_norm_sq = sum(value * value for value in primary)
    if primary_norm_sq <= 0.0:
        raise ValueError("Relationship endpoints must define a non-zero axis")
    # A deterministic second direction in the owner's 14-D readout space,
    # Gram-Schmidt orthogonalized against repair<->steady. This adds off-axis
    # states without inventing a second semantic owner or touching held-out
    # endpoints.
    seed_axis = tuple(
        (
            0.0
            if center_value in (0.0, 1.0) and primary_value == 0.0
            else math.sin(index + 1.0) + 0.5 * math.cos(2.0 * index + 1.0)
        )
        for index, (center_value, primary_value) in enumerate(
            zip(center, primary, strict=True)
        )
    )
    projection = sum(
        left * right for left, right in zip(seed_axis, primary, strict=True)
    ) / primary_norm_sq
    orthogonal = tuple(
        seed_value - projection * primary_value
        for seed_value, primary_value in zip(seed_axis, primary, strict=True)
    )
    orthogonal_norm = math.sqrt(sum(value * value for value in orthogonal))
    if orthogonal_norm <= 0.0:
        raise ValueError("Relationship second state axis collapsed")
    unit_orthogonal = tuple(value / orthogonal_norm for value in orthogonal)
    max_orthogonal_scale = min(
        min(center_value, 1.0 - center_value) / abs(axis_value)
        for center_value, axis_value in zip(
            center, unit_orthogonal, strict=True
        )
        if abs(axis_value) > 1e-12
    )
    orthogonal_scale = min(
        0.35 * math.sqrt(primary_norm_sq),
        0.8 * max_orthogonal_scale,
    )
    secondary = tuple(
        value * orthogonal_scale for value in unit_orthogonal
    )
    states = []
    for index in range(count):
        # Stratified two-dimensional interior samples. Neither axis reaches
        # its endpoint, so repair/steady (u=+/-1,v=0) remain held out.
        u = -0.8 + 1.6 * ((index + 0.5) / count)
        v = rng.uniform(-0.8, 0.8)
        state = tuple(
            center_value + u * primary_value + v * secondary_value
            for center_value, primary_value, secondary_value in zip(
                center, primary, secondary, strict=True
            )
        )
        if any(not 0.0 <= value <= 1.0 for value in state):
            raise RuntimeError("Relationship 2-D sampler escaped owner bounds")
        states.append(state)
    return tuple(states)


def _wilson_lower(*, correct: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def _coordinate(state: tuple[float, ...], label: str) -> float:
    return state[RELATIONSHIP_CONDITIONING_READOUT_LABELS.index(label)]


def _strategy_target(*, state: tuple[float, ...], probe: str) -> str:
    repair = _coordinate(state, "rel_repair_pressure")
    load = _coordinate(state, "rel_emotional_load")
    stabilize = _coordinate(state, "rel_stabilization_need")
    tension = _coordinate(state, "rel_tension_load")
    trust = _coordinate(state, "rel_trust")
    consent = _coordinate(state, "rel_consent_clarity")
    score = (repair + load + stabilize + tension - trust - consent) / 4.0
    if score >= 0.02:
        return (
            "Pause before pushing forward. Acknowledge the strain, protect "
            "trust and consent, and stabilize the relationship first. For "
            f"'{probe}', choose one reversible step and check permission."
        )
    if score <= -0.08:
        return (
            "The relationship is stable enough to proceed within the agreed "
            f"scope. For '{probe}', state the goal clearly, choose one "
            "concrete step, and keep consent visible."
        )
    return (
        "Move carefully while checking the relationship. Confirm consent and "
        f"trust, then take one bounded step for '{probe}'."
    )


def _messages(probe: str) -> tuple[tuple[str, str], ...]:
    return (("system", SYSTEM_PROMPT), ("user", probe))


def _copy_warm_start(*, torch, generator, artifact: PrefixKVArtifact) -> None:
    expected = (
        generator.num_layers,
        generator.num_kv_heads,
        generator.head_dim,
        generator.num_slots,
        generator.rank,
    )
    declared = (
        artifact.num_layers,
        artifact.num_kv_heads,
        artifact.head_dim,
        artifact.num_slots,
        artifact.bottleneck_rank,
    )
    if declared != expected:
        raise ValueError(
            "warm-start Prefix-KV geometry does not match Relationship "
            f"trainer: {declared!r} != {expected!r}."
        )
    with torch.no_grad():
        for target, source in (
            (generator.key_projection, artifact.key_projection),
            (generator.key_bias, artifact.key_bias),
            (generator.value_projection, artifact.value_projection),
            (generator.value_bias, artifact.value_bias),
        ):
            target.copy_(
                torch.tensor(source, dtype=torch.float32, device=target.device)
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--material", default=str(DEFAULT_MATERIAL))
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--norm-cap", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument(
        "--route-temperature", type=float, default=DEFAULT_ROUTE_TEMPERATURE
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--warm-start", default=str(DEFAULT_WARM_START))
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "artifacts"
            / "state_kv"
            / "projectors"
            / "qwen2.5-0.5b-relationship-prefix-v2.json"
        ),
    )
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 < args.norm_cap <= min(MAX_PREFIX_NORM_CAP, 0.12):
        parser.error("--norm-cap must be in (0, 0.12]")
    if args.epochs <= 0 or args.states < 2:
        parser.error("--epochs must be positive and --states >= 2")
    if args.margin <= 0.0 or args.margin_weight < 0.0:
        parser.error("--margin must be positive and --margin-weight non-negative")
    if args.states * 4 != 128 or args.epochs != 3 or args.max_new_tokens != 48:
        parser.error(
            "Relationship v2 preregistration requires 128 samples, 3 epochs, "
            "and a 48-token target"
        )

    material_path = Path(args.material).expanduser().resolve()
    labels, repair, steady, probes, owner_version = _load_material(
        material_path
    )
    states = _sample_states(
        repair=repair,
        steady=steady,
        count=args.states,
        seed=args.seed,
    )

    import torch
    import transformers

    weights_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    weights = _fingerprint_weights(
        model_id=args.model_id, weights_root=weights_root
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(weights_root), local_files_only=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(weights_root),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    device = torch.device(args.device)
    model.to(device)
    config = model.config
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(
        getattr(config, "num_key_value_heads", config.num_attention_heads)
    )
    head_dim = int(
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
    )
    dtype = next(model.parameters()).dtype
    eos_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.add(token_id)

    samples = []
    prompts = []
    for state_index, state in enumerate(states):
        for probe in probes:
            student_ids = _encode(
                tokenizer,
                torch,
                messages=_messages(probe),
                device=device,
            )
            target = _target_tokens_from_text(
                torch=torch,
                tokenizer=tokenizer,
                text=_strategy_target(state=state, probe=probe),
                max_new_tokens=args.max_new_tokens,
                eos_ids=eos_ids,
                device=device,
            )
            if target is None:
                raise RuntimeError("Relationship strategy target was empty.")
            continuation, top_values, top_indices = target
            samples.append(
                {
                    "state": state,
                    "state_index": state_index,
                    "student_ids": student_ids,
                    "continuation": continuation,
                    "top_values": top_values,
                    "top_indices": top_indices,
                }
            )
            prompts.append(student_ids)

    key_norms, value_norms = _measure_reference_norms(
        torch=torch,
        model=model,
        prompts=prompts[: min(8, len(prompts))],
        num_layers=num_layers,
    )
    generator = PrefixGenerator(
        torch=torch,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_slots=args.slots,
        rank=args.rank,
        reference_key_norms=key_norms,
        reference_value_norms=value_norms,
        norm_cap=args.norm_cap,
        device=device,
        seed=args.seed,
        coordinate_count=len(labels),
    )
    warm_start_path = Path(args.warm_start).expanduser().resolve()
    warm_start = PrefixKVArtifact.from_json(
        warm_start_path.read_text(encoding="utf-8")
    )
    _copy_warm_start(
        torch=torch,
        generator=generator,
        artifact=warm_start,
    )
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=args.learning_rate
    )
    order = list(range(len(samples)))
    rng = random.Random(args.seed)
    started = time.time()

    def state_tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    for epoch in range(args.epochs):
        rng.shuffle(order)
        distil_total = 0.0
        margin_total = 0.0
        route_total = 0.0
        for step, index in enumerate(order):
            sample = samples[index]
            state = state_tensor(sample["state"])
            logprobs = _student_logprobs(
                torch=torch,
                transformers=transformers,
                model=model,
                generator=generator,
                state=state,
                student_ids=sample["student_ids"],
                continuation=sample["continuation"],
                dtype=dtype,
            )
            selected = logprobs.gather(-1, sample["top_indices"])
            distil = -(sample["top_values"] * selected).sum(dim=-1).mean()
            margin = torch.zeros((), device=logprobs.device)
            partners = [
                other
                for other in order
                if samples[other]["state_index"] != sample["state_index"]
                and not torch.equal(
                    samples[other]["continuation"], sample["continuation"]
                )
            ]
            if partners and args.margin_weight > 0.0:
                partner = samples[rng.choice(partners)]
                other_logprobs = _student_logprobs(
                    torch=torch,
                    transformers=transformers,
                    model=model,
                    generator=generator,
                    state=state,
                    student_ids=sample["student_ids"],
                    continuation=partner["continuation"],
                    dtype=dtype,
                )
                own = logprobs.gather(
                    -1, sample["continuation"].unsqueeze(-1)
                ).mean()
                foreign = other_logprobs.gather(
                    -1, partner["continuation"].unsqueeze(-1)
                ).mean()
                margin = torch.clamp(
                    args.margin - (own - foreign), min=0.0
                )
            route = _student_route_loss(
                torch=torch,
                transformers=transformers,
                model=model,
                generator=generator,
                state=state,
                student_ids=sample["student_ids"],
                dtype=dtype,
                temperature=args.route_temperature,
            )
            loss = (
                distil
                + args.margin_weight * margin
                + args.route_weight * route
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            distil_total += float(distil.detach())
            margin_total += float(margin.detach())
            route_total += float(route.detach())
            if (step + 1) % 8 == 0:
                print(
                    f"epoch {epoch + 1} step {step + 1}/{len(order)} "
                    f"distil={distil_total / (step + 1):.4f} "
                    f"margin={margin_total / (step + 1):.4f} "
                    f"route={route_total / (step + 1):.4f} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    correct = 0
    checked = 0
    with torch.no_grad():
        for index, sample in enumerate(samples):
            candidates = [
                candidate
                for candidate in samples[index + 1 :] + samples[:index]
                if candidate["state_index"] != sample["state_index"]
                and not torch.equal(
                    candidate["continuation"], sample["continuation"]
                )
            ]
            if not candidates:
                continue
            partner = candidates[0]
            own = float(
                _student_logprobs(
                    torch=torch,
                    transformers=transformers,
                    model=model,
                    generator=generator,
                    state=state_tensor(sample["state"]),
                    student_ids=sample["student_ids"],
                    continuation=sample["continuation"],
                    dtype=dtype,
                )
                .gather(-1, sample["continuation"].unsqueeze(-1))
                .mean()
            )
            foreign = float(
                _student_logprobs(
                    torch=torch,
                    transformers=transformers,
                    model=model,
                    generator=generator,
                    state=state_tensor(partner["state"]),
                    student_ids=sample["student_ids"],
                    continuation=sample["continuation"],
                    dtype=dtype,
                )
                .gather(-1, sample["continuation"].unsqueeze(-1))
                .mean()
            )
            checked += 1
            correct += int(own > foreign)
    wrong_user_accuracy = correct / checked if checked else 0.0
    wrong_user_ci_lower = _wilson_lower(correct=correct, total=checked)
    print(
        f"wrong-user control: {correct}/{checked} "
        f"accuracy={wrong_user_accuracy:.3f} ci_lower={wrong_user_ci_lower:.3f}"
    )

    def as_list(tensor):
        return tensor.detach().to("cpu", torch.float32).tolist()

    material_bytes = material_path.read_bytes()
    source_fingerprint = hashlib.sha256(
        (
            str(weights["weights_sha256"])
            + ":"
            + hashlib.sha256(material_bytes).hexdigest()
            + ":"
            + warm_start.artifact_id
        ).encode("utf-8")
    ).hexdigest()
    nested = build_teacher_distilled_prefix_artifact(
        model_id=args.model_id,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_slots=args.slots,
        bottleneck_rank=args.rank,
        encoder_rows=as_list(generator.encoder),
        encoder_bias=as_list(generator.encoder_bias),
        key_projection=as_list(generator.key_projection),
        key_bias=as_list(generator.key_bias),
        value_projection=as_list(generator.value_projection),
        value_bias=as_list(generator.value_bias),
        reference_key_norms=key_norms,
        reference_value_norms=value_norms,
        norm_cap=args.norm_cap,
        source_fingerprint=source_fingerprint,
        sample_count=len(samples),
        training_mode=STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE,
        vector_labels=labels,
        description=(
            "Relationship State-to-KV generator trained on owner-derived "
            "interior states with routed attention."
        ),
    )
    artifact = bind_relationship_prefix_artifact(
        prefix_artifact=nested,
        owner_schema_version=owner_version,
        readout_labels=labels,
        description=(
            "Dedicated Relationship Prefix-KV artifact; evaluation owner "
            "endpoints and pilot probes held out."
        ),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.to_json() + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "relationship-prefix-kv-bake.v1",
        "artifact_id": artifact.artifact_id,
        "carrier_version": artifact.carrier_version,
        "nested_prefix_artifact_id": nested.artifact_id,
        "model_id": args.model_id,
        "weights_sha256": weights["weights_sha256"],
        "material_sha256": hashlib.sha256(material_bytes).hexdigest(),
        "warm_start_artifact_id": warm_start.artifact_id,
        "state_count": len(states),
        "probe_count": len(probes),
        "sample_count": len(samples),
        "epochs": args.epochs,
        "num_slots": args.slots,
        "bottleneck_rank": args.rank,
        "norm_cap": args.norm_cap,
        "route_weight": args.route_weight,
        "route_temperature": args.route_temperature,
        "margin": args.margin,
        "margin_weight": args.margin_weight,
        "wrong_user_control_accuracy": round(wrong_user_accuracy, 6),
        "wrong_user_control_ci_lower": round(wrong_user_ci_lower, 6),
        "wrong_user_control_samples": checked,
        "wrong_user_control_passed": wrong_user_ci_lower > 0.5,
        "state_geometry": "two-orthogonal-axes-interior-v1",
        "evaluation_endpoints_held_out": True,
        "evaluation_probes_held_out": True,
        "base_model_mutated": False,
        "trainable_parameter_count": sum(
            math.prod(tuple(parameter.shape))
            for parameter in generator.parameters()
        ),
    }
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"artifact_id={artifact.artifact_id}")
    print(f"carrier_version={artifact.carrier_version}")
    print(f"written: {output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
