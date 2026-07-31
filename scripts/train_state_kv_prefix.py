#!/usr/bin/env python3
"""Distil a State-KV prefix generator from the frozen substrate's own text arm.

The teacher is arm B-prime: the same frozen Qwen, reading the owner-rendered
state statement in its system prompt. The student is arm G: the same frozen
Qwen, reading a prompt with no state sections at all, conditioned only by a
bounded per-layer key/value prefix generated from the 16-dimensional readout.
Only the generator trains; the base model is frozen and never touched.

Pure distillation alone measured out as a degenerate multi-layer residual
bias: attention stayed nearly constant while values changed with state. The
default objective therefore adds a routed-attention term that pushes the final
prompt query's prefix-slot attention toward a deterministic state-conditioned
distribution. That makes this trainer a mechanism fix for P4 Gate A, not a new
semantic owner and not a claim of beating prompt engineering.

Three properties keep the resulting artifact from being self-confirming:

* **The evaluation probes are held out.** Training uses a disjoint sentence
  pool; the three identification probes are never seen.
* **The evaluation personas are outside the training envelope.** States are
  sampled inside ``|u| <= 0.8`` along the persona axis, so the two evaluation
  personas (``u = +-1``) are extrapolation, not memorised points.
* **The wrong-user direction is trained against, then measured.** The
  counterfactual margin term is the negative control, and the final report
  states how often the student prefers its own teacher's continuation.
* **The key route is trained directly.** A fixed numeric state->slot target is
  matched against real attention weights, so the generator cannot satisfy the
  default loss purely by changing values under near-constant attention.

Usage:
    python scripts/train_state_kv_prefix.py --device mps
"""

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
# Also importable as a module (contract tests load it by path), so the sibling
# runner has to be resolvable without relying on the script's own cwd.
for _src in [
    Path(__file__).resolve().parent,
    *sorted((REPO_ROOT / "packages").glob("*/src")),
]:
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_identification import (  # noqa: E402
    DEFAULT_MODEL_ID,
    P2_PROBE_SENTENCES,
    PERSONAS,
    PROBE_SENTENCES,
    _assembly,
    _base_context,
    _conditioning,
    _fingerprint_weights,
    _resolve_local_weights,
)
from volvence_zero.agent.prompts import (  # noqa: E402
    build_chat_messages,
    build_system_prompt,
)
from volvence_zero.personal_conditioning_contracts import (  # noqa: E402
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)
from volvence_zero.state_kv_identification import (  # noqa: E402
    PREFIX_ARM_LABEL,
    ProbeCase,
    arm_from_profile,
    context_for_arm,
)
from volvence_zero.substrate.prefix_kv_artifact import (  # noqa: E402
    MAX_PREFIX_NORM_CAP,
    ROUTED_TEACHER_DISTILLED_PREFIX_TRAINING_MODE,
    STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE,
    build_teacher_distilled_prefix_artifact,
)
from volvence_zero.substrate import (  # noqa: E402
    install_rare_heavy_checkpoint_hooks,
    rare_heavy_checkpoint_from_json,
    remove_forward_hooks,
)

TEACHER_ARM_LABEL = "state-kv-arm-bprime"

# Training probes. Disjoint from PROBE_SENTENCES by construction; the assertion
# below is what keeps that true when either pool is edited.
TRAIN_PROBE_SENTENCES: tuple[str, ...] = (
    "今天特别累",
    "我不知道接下来该做什么",
    "刚刚和家里人吵了一架",
    "这个方案你怎么看",
    "我想再确认一次",
    "明天要做个决定",
    "最近总是提不起劲",
    "帮我理一下思路",
)

# States are sampled inside this fraction of the persona axis so the two
# evaluation personas sit outside the training envelope.
TRAIN_AXIS_LIMIT = 0.8

DEFAULT_ROUTE_WEIGHT = 1.0
DEFAULT_ROUTE_TEMPERATURE = 0.18
TARGET_SOURCE_TEACHER = "teacher"
TARGET_SOURCE_STATE_STRATEGY = "state-strategy"
TARGET_SOURCES = (TARGET_SOURCE_STATE_STRATEGY, TARGET_SOURCE_TEACHER)


def _coordinate(state: tuple[float, ...], label: str) -> float:
    return state[PERSONAL_CONDITIONING_VECTOR_LABELS.index(label)]


def _state_strategy_target(*, state: tuple[float, ...], probe: str) -> str:
    """Audit target that makes the typed readout legible in the reply.

    This is training material only. Runtime still receives no text-state
    section on arm G; the prefix must carry enough of this strategy for the
    frozen substrate to express it from identical prompt bytes.
    """

    overwhelm = _coordinate(state, "user_overwhelm")
    control = _coordinate(state, "user_control")
    trust = _coordinate(state, "relationship_trust")
    repair = _coordinate(state, "relationship_repair_need")
    emotional_load = _coordinate(state, "relationship_emotional_load")
    readiness = _coordinate(state, "goal_decision_readiness")
    reversibility = _coordinate(state, "goal_reversibility_need")
    autonomy_risk = _coordinate(state, "boundary_autonomy_risk")
    caution = (
        overwhelm
        + repair
        + emotional_load
        + reversibility
        + autonomy_risk
        - control
        - trust
        - readiness
    ) / 5.0
    if caution >= 0.18:
        return (
            "我会先放慢，承认你现在压力很高、控制感偏低，也需要修复和退路；"
            f"关于「{probe}」，我们先稳住情绪，只拆一个很小的下一步。"
        )
    if caution <= -0.18:
        return (
            "我会更直接地给结构，因为你现在稳定、信任和决策准备度都较高；"
            f"关于「{probe}」，我们可以列标准、定下一步，并推进验证。"
        )
    return (
        "我会先确认你的状态再推进：既给一点结构，也保留退路；"
        f"关于「{probe}」，我们先把选择、风险和下一步放在桌面上。"
    )


def _target_tokens_from_text(
    *,
    torch,
    tokenizer,
    text: str,
    max_new_tokens: int,
    eos_ids: set[int],
    device,
):
    token_ids = [
        token
        for token in tokenizer(text, add_special_tokens=False)["input_ids"]
        if token not in eos_ids
    ][:max_new_tokens]
    if not token_ids:
        return None
    continuation = torch.tensor(
        token_ids, dtype=torch.long, device=device
    )
    top_indices = continuation.unsqueeze(-1)
    top_values = torch.ones(
        (continuation.numel(), 1), dtype=torch.float32, device=device
    )
    return continuation, top_values, top_indices


def _assert_probe_holdout() -> None:
    evaluation = {
        sentence
        for _, sentence in (*PROBE_SENTENCES, *P2_PROBE_SENTENCES)
    }
    overlap = evaluation & set(TRAIN_PROBE_SENTENCES)
    if overlap:
        raise ValueError(
            "training probes overlap the identification probes "
            f"({sorted(overlap)}); the resulting verdict would be scored on "
            "material the generator was fit to."
        )


def _persona_axes() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Centre plus two coherent directions derived from the eval personas.

    Sampling each coordinate independently would describe internally
    contradictory people (high trust and high repair-need at once), and the
    teacher's rendered statement would be incoherent text. Deriving the axes
    from the two hand-checked personas keeps every sampled state a plausible
    person while still covering the space between and around them.
    """

    vector_a = PERSONAS[0][1]
    vector_b = PERSONAS[1][1]
    centre = tuple((a + b) / 2.0 for a, b in zip(vector_a, vector_b, strict=True))
    primary = tuple((a - b) / 2.0 for a, b in zip(vector_a, vector_b, strict=True))
    # Second axis: flip the boundary block against the rest, so boundary risk
    # can vary independently of relationship distress.
    boundary_start = next(
        index
        for index, label in enumerate(PERSONAL_CONDITIONING_VECTOR_LABELS)
        if label.startswith("boundary_")
    )
    flipped = tuple(
        value * (-1.0 if index >= boundary_start else 1.0)
        for index, value in enumerate(primary)
    )
    # Orthogonalise against the primary axis. Without this the second factor
    # leaks into the first, and a sampled state could project past |u| = 1 --
    # i.e. land on or beyond an evaluation persona while the sampler still
    # reported drawing u inside the training envelope.
    primary_energy = sum(value * value for value in primary)
    overlap = (
        sum(f * p for f, p in zip(flipped, primary, strict=True))
        / primary_energy
        if primary_energy > 1e-12
        else 0.0
    )
    secondary = tuple(
        f - overlap * p for f, p in zip(flipped, primary, strict=True)
    )
    return centre, primary, secondary


def _sample_states(*, count: int, seed: int) -> tuple[tuple[float, ...], ...]:
    centre, primary, secondary = _persona_axes()
    rng = random.Random(seed)
    states: list[tuple[float, ...]] = []
    for _ in range(count):
        u = rng.uniform(-TRAIN_AXIS_LIMIT, TRAIN_AXIS_LIMIT)
        v = rng.uniform(-TRAIN_AXIS_LIMIT, TRAIN_AXIS_LIMIT) * 0.5
        state = tuple(
            min(
                0.98,
                max(
                    0.02,
                    c + u * p + v * s + rng.gauss(0.0, 0.02),
                ),
            )
            for c, p, s in zip(centre, primary, secondary, strict=True)
        )
        states.append(state)
    return tuple(states)


def _chat_messages_for(*, arm_label: str, state, probe: str, user_id: str):
    """Build one arm's actual chat messages for one (state, probe) pair.

    Goes through the same ``context_for_arm`` / ``build_system_prompt`` path the
    identification runner uses. A trainer that assembled its own prompts would
    fit the generator to a prompt distribution the evidence run never sends.
    """

    case = ProbeCase(
        user_id=user_id,
        probe_id="train",
        user_input=probe,
        conditioning=_conditioning(user_id=user_id, state_vector=state),
        assembly=_assembly(residue="", ordering_driver="playbook-only"),
    )
    context = context_for_arm(
        arm=arm_from_profile(arm_label),
        case=case,
        base_context=_base_context(),
    )
    system_prompt = build_system_prompt(
        assembly=case.assembly, context=context
    )
    messages = build_chat_messages(assembly=case.assembly, context=context)
    if not messages:
        raise ValueError(f"arm {arm_label!r} produced no chat messages")
    return system_prompt, messages


class PrefixGenerator:
    """Trainable mirror of :class:`PrefixKVArtifact`'s inference math.

    Kept as an explicit parameter list rather than an ``nn.Module`` tree so the
    export path is a direct read of the same tensors the artifact stores; a
    mismatch between trained and exported math would be invisible otherwise.
    """

    def __init__(
        self,
        *,
        torch,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_slots: int,
        rank: int,
        reference_key_norms,
        reference_value_norms,
        norm_cap: float,
        device,
        seed: int,
        coordinate_count: int = len(PERSONAL_CONDITIONING_VECTOR_LABELS),
    ) -> None:
        self._torch = torch
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_slots = num_slots
        self.rank = rank
        self.norm_cap = norm_cap
        self.width = num_slots * num_kv_heads * head_dim
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def parameter(*shape, scale: float):
            data = (
                torch.randn(*shape, generator=generator, dtype=torch.float32)
                * scale
            )
            return data.to(device).requires_grad_(True)

        def zeros(*shape):
            return torch.zeros(
                *shape, dtype=torch.float32, device=device
            ).requires_grad_(True)

        if coordinate_count <= 0:
            raise ValueError("coordinate_count must be positive")
        coordinates = coordinate_count
        self.encoder = parameter(rank, coordinates, scale=0.5)
        self.encoder_bias = zeros(rank)
        self.key_projection = parameter(num_layers, self.width, rank, scale=0.05)
        self.key_bias = zeros(num_layers, self.width)
        self.value_projection = parameter(
            num_layers, self.width, rank, scale=0.05
        )
        self.value_bias = zeros(num_layers, self.width)
        self.key_caps = (
            torch.tensor(
                reference_key_norms, dtype=torch.float32, device=device
            )
            * norm_cap
        )
        self.value_caps = (
            torch.tensor(
                reference_value_norms, dtype=torch.float32, device=device
            )
            * norm_cap
        )

    def parameters(self) -> list:
        return [
            self.encoder,
            self.encoder_bias,
            self.key_projection,
            self.key_bias,
            self.value_projection,
            self.value_bias,
        ]

    def _cap(self, tensor, caps):
        torch = self._torch
        norms = tensor.norm(dim=-1, keepdim=True)
        limits = caps.reshape(-1, 1, 1, 1)
        return tensor * torch.clamp(limits / norms.clamp_min(1e-8), max=1.0)

    def build(self, state, *, dtype):
        """Return per-layer ``(key, value)`` with gradients attached."""

        torch = self._torch
        hidden = torch.tanh(self.encoder @ state + self.encoder_bias)
        shape = (self.num_layers, self.num_slots, self.num_kv_heads, self.head_dim)
        keys = (self.key_projection @ hidden + self.key_bias).reshape(shape)
        values = (self.value_projection @ hidden + self.value_bias).reshape(shape)
        keys = self._cap(keys, self.key_caps).permute(0, 2, 1, 3).unsqueeze(1)
        values = self._cap(values, self.value_caps).permute(0, 2, 1, 3).unsqueeze(1)
        return [
            (keys[index].to(dtype), values[index].to(dtype))
            for index in range(self.num_layers)
        ]


def _encode(tokenizer, torch, *, messages, device):
    payload = [{"role": role, "content": content} for role, content in messages]
    text = tokenizer.apply_chat_template(
        payload, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, return_tensors="pt")
    return encoded["input_ids"].to(device)


def _measure_reference_norms(*, torch, model, prompts, num_layers):
    """Measure per-layer key/value norms on real prompts.

    The cap is meaningful only against measured norms: on Qwen2.5-0.5B the mean
    key norm runs from 259.7 at layer 0 to ~14 in the middle layers, so one
    global bound would be simultaneously far too loose and far too tight.
    """

    key_totals = [0.0] * num_layers
    value_totals = [0.0] * num_layers
    for input_ids in prompts:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        cache = outputs.past_key_values
        for index in range(num_layers):
            layer = cache.layers[index]
            key_totals[index] += float(
                layer.keys.to(torch.float32).norm(dim=-1).mean()
            )
            value_totals[index] += float(
                layer.values.to(torch.float32).norm(dim=-1).mean()
            )
    count = float(len(prompts))
    return (
        [total / count for total in key_totals],
        [total / count for total in value_totals],
    )


def _route_anchors(*, torch, slots: int, coordinates: int, device):
    """Deterministic slot anchors for the State-KV routing objective.

    The anchors are not trainable semantic labels. They are a fixed measurement
    basis that gives each state a distinct target distribution over prefix
    slots, forcing the key side of the prefix to participate instead of
    letting the generator satisfy distillation through values alone.
    """

    rows = []
    scale = 1.0 / math.sqrt(float(coordinates))
    for slot in range(slots):
        row = []
        for index in range(coordinates):
            angle = (slot + 1) * (index + 1)
            row.append(
                scale
                * (
                    math.sin(angle * 1.61803398875)
                    + math.cos(angle * 0.75487766625)
                )
            )
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _slot_route_target(*, torch, state, slots: int, temperature: float):
    """State-conditioned target distribution over prefix slots."""

    if slots <= 1:
        return torch.ones(slots, dtype=torch.float32, device=state.device)
    if temperature <= 0.0:
        raise ValueError("route temperature must be positive.")
    anchors = _route_anchors(
        torch=torch,
        slots=slots,
        coordinates=int(state.shape[0]),
        device=state.device,
    )
    centred = state.to(torch.float32) * 2.0 - 1.0
    logits = anchors @ centred
    return torch.softmax(logits / float(temperature), dim=0)


def _slot_attention_route_loss_from_attentions(
    *,
    torch,
    attentions,
    slots: int,
    target,
):
    """Cross-entropy from final-query slot attention to a state route target."""

    losses = []
    for layer_attention in attentions:
        head_view = layer_attention[0, :, -1, :slots].to(torch.float32)
        slot_view = head_view.mean(dim=0)
        slot_dist = slot_view / slot_view.sum().clamp_min(1e-8)
        losses.append(-(target * torch.log(slot_dist.clamp_min(1e-8))).sum())
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=target.device)
    return torch.stack(losses).mean()


def _teacher_targets(
    *,
    torch,
    model,
    tokenizer,
    teacher_ids,
    max_new_tokens: int,
    top_k: int,
    eos_ids: set[int],
):
    """Greedy continuation plus its top-k next-token distribution."""

    with torch.no_grad():
        generated = model.generate(
            input_ids=teacher_ids,
            attention_mask=torch.ones_like(teacher_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=sorted(eos_ids),
        )
        continuation = generated[0, teacher_ids.shape[-1]:]
        continuation = torch.tensor(
            [token for token in continuation.tolist() if token not in eos_ids],
            dtype=teacher_ids.dtype,
            device=teacher_ids.device,
        )
        if continuation.numel() == 0:
            return None
        full = torch.cat([teacher_ids, continuation.unsqueeze(0)], dim=-1)
        logits = model(input_ids=full, use_cache=False).logits[0]
        start = teacher_ids.shape[-1] - 1
        window = logits[start : start + continuation.numel()].to(torch.float32)
        probabilities = torch.softmax(window, dim=-1)
        values, indices = probabilities.topk(top_k, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True)
    return continuation, values.detach(), indices.detach()


def _student_logprobs(
    *,
    torch,
    transformers,
    model,
    generator,
    state,
    student_ids,
    continuation,
    dtype,
):
    """Log-probabilities the prefixed student assigns to a continuation."""

    prefix_pairs = generator.build(state, dtype=dtype)
    slots = prefix_pairs[0][0].shape[-2]
    cache = transformers.DynamicCache(ddp_cache_data=prefix_pairs)
    full = torch.cat([student_ids, continuation.unsqueeze(0)], dim=-1)
    total = full.shape[-1]
    attention_mask = torch.ones(
        (1, slots + total), dtype=torch.long, device=full.device
    )
    position_ids = torch.arange(total, device=full.device).unsqueeze(0)
    logits = model(
        input_ids=full,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    ).logits[0]
    start = student_ids.shape[-1] - 1
    window = logits[start : start + continuation.numel()].to(torch.float32)
    return torch.log_softmax(window, dim=-1)


def _student_route_loss(
    *,
    torch,
    transformers,
    model,
    generator,
    state,
    student_ids,
    dtype,
    temperature: float,
):
    """Train prefix keys so slot attention is a function of state.

    The pure distillation loss can be solved as an almost-constant attention
    weight over state-dependent values, which P4 diagnosed as a multi-layer
    residual bias. This loss looks only at final-prompt-position attention over
    the prefix slots and pushes that slot distribution toward a deterministic
    state route.
    """

    prefix_pairs = generator.build(state, dtype=dtype)
    slots = prefix_pairs[0][0].shape[-2]
    cache = transformers.DynamicCache(ddp_cache_data=prefix_pairs)
    attention_mask = torch.ones(
        (1, slots + student_ids.shape[-1]),
        dtype=torch.long,
        device=student_ids.device,
    )
    position_ids = torch.arange(
        student_ids.shape[-1], device=student_ids.device
    ).unsqueeze(0)
    outputs = model(
        input_ids=student_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        output_attentions=True,
    )
    if not outputs.attentions:
        raise RuntimeError(
            "routed State-KV training requires output_attentions=True to "
            "return per-layer attention weights; use an attention backend "
            "that exposes them instead of silently dropping the route loss."
        )
    target = _slot_route_target(
        torch=torch,
        state=state,
        slots=slots,
        temperature=temperature,
    )
    return _slot_attention_route_loss_from_attentions(
        torch=torch,
        attentions=outputs.attentions,
        slots=slots,
        target=target,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--norm-cap", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=DEFAULT_ROUTE_WEIGHT)
    parser.add_argument(
        "--route-temperature", type=float, default=DEFAULT_ROUTE_TEMPERATURE
    )
    parser.add_argument(
        "--target-source",
        choices=TARGET_SOURCES,
        default=TARGET_SOURCE_STATE_STRATEGY,
        help=(
            "training continuation source: state-strategy uses an "
            "owner-readout-derived audit target; teacher preserves the older "
            "B-prime text-arm distillation path"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "artifacts"
            / "state_kv"
            / "projectors"
            / "qwen2.5-0.5b-prefix.json"
        ),
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--common-adapter-checkpoint",
        type=Path,
        help=(
            "standalone rare-heavy checkpoint activated before all teacher "
            "and student forwards"
        ),
    )
    parser.add_argument(
        "--common-adapter-version",
        default="",
        help="common adapter version bound into State-KV provenance",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.norm_cap <= MAX_PREFIX_NORM_CAP:
        parser.error(
            f"--norm-cap must be in (0, {MAX_PREFIX_NORM_CAP}]"
        )
    if args.route_weight < 0.0:
        parser.error("--route-weight must be non-negative")
    if args.route_temperature <= 0.0:
        parser.error("--route-temperature must be positive")
    if bool(args.common_adapter_checkpoint) != bool(
        args.common_adapter_version.strip()
    ):
        parser.error(
            "--common-adapter-checkpoint and --common-adapter-version must be "
            "provided together"
        )
    _assert_probe_holdout()

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
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(weights_root))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(weights_root), dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    model.requires_grad_(False)
    device = torch.device(args.device)
    model.to(device)
    adapter_hooks = ()
    checkpoint_sha256 = ""
    checkpoint_fingerprint = ""
    if args.common_adapter_checkpoint is not None:
        checkpoint_path = args.common_adapter_checkpoint.expanduser().resolve()
        checkpoint_payload = checkpoint_path.read_text(encoding="utf-8")
        checkpoint = rare_heavy_checkpoint_from_json(checkpoint_payload)
        checkpoint_sha256 = hashlib.sha256(
            checkpoint_payload.encode("utf-8")
        ).hexdigest()
        checkpoint_fingerprint = checkpoint.compatibility_fingerprint
        adapter_hooks = install_rare_heavy_checkpoint_hooks(
            model=model,
            checkpoint=checkpoint,
            expected_model_id=args.model_id,
        )

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

    states = _sample_states(count=args.states, seed=args.seed)
    print(
        f"material: {len(states)} states x {len(TRAIN_PROBE_SENTENCES)} probes "
        f"(evaluation probes held out, |u| <= {TRAIN_AXIS_LIMIT})"
    )

    # ---- teacher pass -----------------------------------------------------
    samples: list[dict[str, object]] = []
    student_prompts: list[object] = []
    started = time.time()
    for state_index, state in enumerate(states):
        user_id = f"train-{state_index:03d}"
        for probe in TRAIN_PROBE_SENTENCES:
            _, teacher_messages = _chat_messages_for(
                arm_label=TEACHER_ARM_LABEL,
                state=state,
                probe=probe,
                user_id=user_id,
            )
            _, student_messages = _chat_messages_for(
                arm_label=PREFIX_ARM_LABEL,
                state=state,
                probe=probe,
                user_id=user_id,
            )
            teacher_ids = _encode(
                tokenizer, torch, messages=teacher_messages, device=device
            )
            student_ids = _encode(
                tokenizer, torch, messages=student_messages, device=device
            )
            if args.target_source == TARGET_SOURCE_STATE_STRATEGY:
                target = _target_tokens_from_text(
                    torch=torch,
                    tokenizer=tokenizer,
                    text=_state_strategy_target(state=state, probe=probe),
                    max_new_tokens=args.max_new_tokens,
                    eos_ids=eos_ids,
                    device=device,
                )
            else:
                target = _teacher_targets(
                    torch=torch,
                    model=model,
                    tokenizer=tokenizer,
                    teacher_ids=teacher_ids,
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.top_k,
                    eos_ids=eos_ids,
                )
            if target is None:
                # An empty teacher continuation carries no behaviour to
                # distil; counting it as a sample would dilute the loss with
                # zero-signal rows.
                continue
            continuation, top_values, top_indices = target
            samples.append(
                {
                    "state": state,
                    "state_index": state_index,
                    "probe": probe,
                    "student_ids": student_ids,
                    "continuation": continuation,
                    "top_values": top_values,
                    "top_indices": top_indices,
                }
            )
            student_prompts.append(student_ids)
        print(
            f"  teacher {state_index + 1}/{len(states)} "
            f"({len(samples)} samples, {time.time() - started:.0f}s)"
        )
    if not samples:
        raise RuntimeError("teacher produced no usable continuations")

    key_norms, value_norms = _measure_reference_norms(
        torch=torch,
        model=model,
        prompts=student_prompts[: min(8, len(student_prompts))],
        num_layers=num_layers,
    )
    print(
        "reference key norms: "
        f"layer0={key_norms[0]:.1f} mean={sum(key_norms) / num_layers:.1f}"
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
    )
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate)
    order = list(range(len(samples)))
    rng = random.Random(args.seed)

    def state_tensor(values) -> object:
        return torch.tensor(
            list(values), dtype=torch.float32, device=device
        )

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

            # Counterfactual margin: the same prefix must prefer its own
            # teacher's continuation over another state's. Without this the
            # generator can satisfy the distillation loss by learning one
            # state-independent prefix, which would pass claim 2 only by
            # accident and fail every negative control.
            margin = torch.zeros((), device=logprobs.device)
            partners = [
                other
                for other in order
                if samples[other]["state_index"] != sample["state_index"]
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
                own = (
                    logprobs.gather(
                        -1, sample["continuation"].unsqueeze(-1)
                    ).mean()
                )
                foreign = (
                    other_logprobs.gather(
                        -1, partner["continuation"].unsqueeze(-1)
                    ).mean()
                )
                margin = torch.clamp(args.margin - (own - foreign), min=0.0)

            route = torch.zeros((), device=logprobs.device)
            if args.route_weight > 0.0:
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
            if (step + 1) % 20 == 0:
                print(
                    f"  epoch {epoch + 1} step {step + 1}/{len(order)} "
                    f"distil={distil_total / (step + 1):.4f} "
                    f"margin={margin_total / (step + 1):.4f} "
                    f"route={route_total / (step + 1):.4f} "
                    f"({time.time() - started:.0f}s)"
                )
        print(
            f"epoch {epoch + 1}/{args.epochs} distil="
            f"{distil_total / len(order):.4f} "
            f"margin={margin_total / len(order):.4f} "
            f"route={route_total / len(order):.4f}"
        )

    # ---- wrong-user negative control -------------------------------------
    correct = 0
    checked = 0
    with torch.no_grad():
        for index, sample in enumerate(samples):
            partner = samples[(index + len(samples) // 2) % len(samples)]
            if partner["state_index"] == sample["state_index"]:
                continue
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
    print(
        f"wrong-user negative control: {correct}/{checked} "
        f"({wrong_user_accuracy:.3f}) prefer their own state's prefix"
    )

    # ---- export -----------------------------------------------------------
    def as_list(tensor) -> list:
        return tensor.detach().to("cpu", torch.float32).tolist()

    material = json.dumps(
        {
            "probes": list(TRAIN_PROBE_SENTENCES),
            "states": [list(state) for state in states],
            "axis_limit": TRAIN_AXIS_LIMIT,
            "route_weight": args.route_weight,
            "route_temperature": args.route_temperature,
            "target_source": args.target_source,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    source_fingerprint = hashlib.sha256(
        (
            str(weights["weights_sha256"])
            + ":"
            + hashlib.sha256(material).hexdigest()
            + ":"
            + args.common_adapter_version.strip()
            + ":"
            + checkpoint_sha256
        ).encode("utf-8")
    ).hexdigest()

    artifact = build_teacher_distilled_prefix_artifact(
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
        training_mode=(
            STATE_STRATEGY_ROUTED_PREFIX_TRAINING_MODE
            if args.target_source == TARGET_SOURCE_STATE_STRATEGY
            else ROUTED_TEACHER_DISTILLED_PREFIX_TRAINING_MODE
        ),
    )
    remove_forward_hooks(adapter_hooks)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.to_json() + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "state-kv-prefix-bake.v1",
        "artifact_id": artifact.artifact_id,
        "training_mode": artifact.training_mode,
        "model_id": args.model_id,
        "weights_sha256": weights["weights_sha256"],
        "common_adapter_version": args.common_adapter_version.strip() or None,
        "rare_heavy_checkpoint_sha256": checkpoint_sha256 or None,
        "rare_heavy_compatibility_fingerprint": (
            checkpoint_fingerprint or None
        ),
        "training_order": (
            "base+rare-heavy->state-kv"
            if checkpoint_sha256
            else "base-only-legacy"
        ),
        "material_sha256": hashlib.sha256(material).hexdigest(),
        "teacher_arm": TEACHER_ARM_LABEL,
        "student_arm": PREFIX_ARM_LABEL,
        "target_source": args.target_source,
        "state_count": len(states),
        "train_probe_count": len(TRAIN_PROBE_SENTENCES),
        "sample_count": len(samples),
        "epochs": args.epochs,
        "num_slots": args.slots,
        "bottleneck_rank": args.rank,
        "norm_cap": args.norm_cap,
        "route_weight": args.route_weight,
        "route_temperature": args.route_temperature,
        "evaluation_probes_held_out": True,
        "training_axis_limit": TRAIN_AXIS_LIMIT,
        "wrong_user_control_accuracy": round(wrong_user_accuracy, 4),
        "wrong_user_control_samples": checked,
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
    print(f"written: {output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
