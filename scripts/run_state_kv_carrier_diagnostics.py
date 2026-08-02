#!/usr/bin/env python3
"""Run the two mechanism gates on the State-KV prefix carrier.

Gate A asks whether attention reads the state slots. Gate B asks whether the
readout is linearly recoverable from the *prompt* tokens after prefill. Both
are cheaper than the identification lane and, unlike it, they separate the two
diagnoses that a chance-level wrong-user control leaves entangled: nothing is
read, versus something is read but carries no state.

The prefix content comes from the same ``PrefixKVGenerator`` the runtime
injects, so these measurements describe the arm that actually runs. The forward
pass is loaded separately with eager attention because attention weights are
not observable under the fused kernels the generation path uses; the generation
path itself is untouched.

Usage:
    python scripts/run_state_kv_carrier_diagnostics.py \\
        --device mps \\
        --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-prefix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in [
    Path(__file__).resolve().parent,
    *sorted((REPO_ROOT / "packages").glob("*/src")),
]:
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_identification import (  # noqa: E402
    DEFAULT_MODEL_ID,
    PERSONAS,
    PROBE_SENTENCES,
    _assembly,
    _base_context,
    _fingerprint_weights,
    _resolve_local_weights,
)
from train_state_kv_prefix import (  # noqa: E402
    TRAIN_PROBE_SENTENCES,
    _chat_messages_for,
    _sample_states,
)
from train_relationship_prefix_kv import (  # noqa: E402
    DEFAULT_MATERIAL as RELATIONSHIP_MATERIAL,
    SYSTEM_PROMPT as RELATIONSHIP_SYSTEM_PROMPT,
    _load_material as _load_relationship_material,
    _sample_states as _sample_relationship_states,
)
from volvence_zero.state_kv_carrier_diagnostics import (  # noqa: E402
    build_carrier_diagnostics_verdict,
    evaluate_slot_attention_read,
    evaluate_state_linearly_readable,
)
from volvence_zero.state_kv_identification import (  # noqa: E402
    PREFIX_ARM_LABEL,
)
from volvence_zero.substrate.prefix_kv_artifact import (  # noqa: E402
    PrefixKVArtifact,
    load_prefix_generator,
)
from volvence_zero.substrate import RelationshipPrefixKVArtifact  # noqa: E402
from volvence_zero.substrate.prefix_kv_diagnostics import (  # noqa: E402
    capture_prefix_diagnostics,
    fit_ridge_probe,
    profile_spread,
    select_ridge_alpha,
)

# Sentences used for the diagnostics. Kept disjoint from the identification
# probes so a passing gate cannot be an artefact of the evaluation material,
# and small because every sentence multiplies the forward-pass count.
DIAGNOSTIC_SENTENCES: tuple[str, ...] = TRAIN_PROBE_SENTENCES[:4]


def _encode(
    tokenizer,
    torch,
    *,
    state,
    sentence,
    user_id,
    device,
    state_domain="personal",
):
    if state_domain == "relationship":
        payload = [
            {"role": "system", "content": RELATIONSHIP_SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ]
        text = tokenizer.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    _, messages = _chat_messages_for(
        arm_label=PREFIX_ARM_LABEL,
        state=state,
        probe=sentence,
        user_id=user_id,
    )
    payload = [{"role": role, "content": content} for role, content in messages]
    text = tokenizer.apply_chat_template(
        payload, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")["input_ids"].to(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix-kv-artifact", required=True)
    parser.add_argument(
        "--state-domain",
        choices=("personal", "relationship"),
        default="personal",
    )
    parser.add_argument("--train-states", type=int, default=96)
    parser.add_argument("--eval-states", type=int, default=32)
    parser.add_argument("--random-draws", type=int, default=4)
    parser.add_argument("--shuffle-draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "artifacts"
            / "state_kv"
            / "p4-diagnostics"
            / "verdict_carrier_diagnostics.json"
        ),
    )
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)

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
    artifact_payload = Path(args.prefix_kv_artifact).expanduser().read_text(
        encoding="utf-8"
    )
    if args.state_domain == "relationship":
        relationship_artifact = RelationshipPrefixKVArtifact.from_json(
            artifact_payload
        )
        artifact = relationship_artifact.prefix_artifact
        bound_artifact_id = relationship_artifact.artifact_id
    else:
        artifact = PrefixKVArtifact.from_json(artifact_payload)
        bound_artifact_id = artifact.artifact_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(weights_root))
    # Eager attention: the fused kernels the generation path uses do not
    # materialise the weight matrix this diagnostic reads.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(weights_root), dtype=torch.float32, attn_implementation="eager"
    ).eval()
    device = torch.device(args.device)
    model.to(device)
    config = model.config
    generator = load_prefix_generator(
        torch_module=torch,
        artifact=artifact,
        expected_model_id=args.model_id,
        expected_num_layers=int(config.num_hidden_layers),
        expected_num_kv_heads=int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)
        ),
        expected_head_dim=int(
            getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
        ),
        device=device,
        dtype=torch.float32,
    )

    def cache_factory(pairs=None):
        if pairs is None:
            return transformers.DynamicCache()
        return transformers.DynamicCache(ddp_cache_data=list(pairs))

    if args.state_domain == "relationship":
        _, repair, steady, _, _ = _load_relationship_material(
            RELATIONSHIP_MATERIAL
        )
        train_states = _sample_relationship_states(
            repair=repair,
            steady=steady,
            count=args.train_states,
            seed=args.seed,
        )
        eval_states = list(
            _sample_relationship_states(
                repair=repair,
                steady=steady,
                count=args.eval_states,
                seed=args.seed + 1,
            )
        )
        eval_states.extend((repair, steady))
    else:
        train_states = _sample_states(count=args.train_states, seed=args.seed)
        eval_states = list(
            _sample_states(count=args.eval_states, seed=args.seed + 1)
        )
        # The two hand-checked identification personas are extrapolation for
        # the generator (trained inside |u| <= 0.8).
        eval_states.extend(vector for _, vector, _, _ in PERSONAS)
    print(
        f"states: {len(train_states)} train / {len(eval_states)} held-out; "
        f"sentences: {len(DIAGNOSTIC_SENTENCES)}"
    )

    layers = int(config.num_hidden_layers)
    zero_pairs = [
        (
            torch.zeros(1, generator.artifact.num_kv_heads,
                        artifact.num_slots, artifact.head_dim, device=device),
            torch.zeros(1, generator.artifact.num_kv_heads,
                        artifact.num_slots, artifact.head_dim, device=device),
        )
        for _ in range(layers)
    ]

    def random_pairs(seed: int):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pairs = []
        for index in range(layers):
            shape = (1, artifact.num_kv_heads, artifact.num_slots,
                     artifact.head_dim)
            key = torch.randn(*shape, generator=gen)
            value = torch.randn(*shape, generator=gen)
            key = key / key.norm(dim=-1, keepdim=True) * (
                artifact.reference_key_norms[index] * artifact.norm_cap
            )
            value = value / value.norm(dim=-1, keepdim=True) * (
                artifact.reference_value_norms[index] * artifact.norm_cap
            )
            pairs.append((key.to(device), value.to(device)))
        return pairs

    # ---- capture -----------------------------------------------------------
    learned: dict[tuple[int, int], object] = {}
    hidden_rows: list[tuple[list[float], tuple[float, ...]]] = []
    all_states = list(train_states) + eval_states
    for state_index, state in enumerate(all_states):
        user_id = f"diag-{state_index:04d}"
        for sentence_index, sentence in enumerate(DIAGNOSTIC_SENTENCES):
            input_ids = _encode(
                tokenizer, torch, state=state, sentence=sentence,
                user_id=user_id, device=device,
                state_domain=args.state_domain,
            )
            profile = capture_prefix_diagnostics(
                torch_module=torch,
                model=model,
                input_ids=input_ids,
                prefix_pairs=generator.build(state),
                cache_factory=cache_factory,
            )
            learned[(state_index, sentence_index)] = profile
            hidden_rows.append((list(profile.final_hidden), tuple(state)))
        if (state_index + 1) % 20 == 0:
            print(f"  captured {state_index + 1}/{len(all_states)} states")

    # Controls: zero content, and random content at the same per-layer norms.
    reference_ids = _encode(
        tokenizer, torch, state=all_states[0],
        sentence=DIAGNOSTIC_SENTENCES[0], user_id="diag-control", device=device,
        state_domain=args.state_domain,
    )
    zero_profile = capture_prefix_diagnostics(
        torch_module=torch, model=model, input_ids=reference_ids,
        prefix_pairs=zero_pairs, cache_factory=cache_factory,
        capture_hidden=False,
    )
    random_profiles = [
        capture_prefix_diagnostics(
            torch_module=torch, model=model, input_ids=reference_ids,
            prefix_pairs=random_pairs(args.seed + 100 + draw),
            cache_factory=cache_factory, capture_hidden=False,
        )
        for draw in range(args.random_draws)
    ]
    random_nonuniformity = [
        sum(p.slot_nonuniformity[layer] for p in random_profiles)
        / len(random_profiles)
        for layer in range(layers)
    ]
    random_mass = [
        sum(p.slot_mass[layer] for p in random_profiles) / len(random_profiles)
        for layer in range(layers)
    ]
    learned_nonuniformity = [
        sum(learned[(s, t)].slot_nonuniformity[layer]
            for s in range(len(all_states))
            for t in range(len(DIAGNOSTIC_SENTENCES)))
        / (len(all_states) * len(DIAGNOSTIC_SENTENCES))
        for layer in range(layers)
    ]
    learned_mass = [
        sum(learned[(s, t)].slot_mass[layer]
            for s in range(len(all_states))
            for t in range(len(DIAGNOSTIC_SENTENCES)))
        / (len(all_states) * len(DIAGNOSTIC_SENTENCES))
        for layer in range(layers)
    ]

    # A3: does the slot-attention profile move with who, or with what was said?
    state_spread = sum(
        profile_spread(
            [learned[(s, t)].slot_mass for s in range(len(all_states))]
        )
        for t in range(len(DIAGNOSTIC_SENTENCES))
    ) / len(DIAGNOSTIC_SENTENCES)
    sentence_spread = sum(
        profile_spread(
            [learned[(s, t)].slot_mass
             for t in range(len(DIAGNOSTIC_SENTENCES))]
        )
        for s in range(len(all_states))
    ) / len(all_states)

    attention_claim = evaluate_slot_attention_read(
        learned_nonuniformity=learned_nonuniformity,
        control_nonuniformity=random_nonuniformity,
        state_spread=state_spread,
        sentence_spread=sentence_spread,
    )
    print(f"gate A: {attention_claim.state.value} — {attention_claim.detail}")

    # ---- gate B ------------------------------------------------------------
    train_count = len(train_states) * len(DIAGNOSTIC_SENTENCES)
    targets = torch.tensor(
        [list(state) for _, state in hidden_rows],
        dtype=torch.float32, device=device,
    )
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    # Several draws, because the null this control estimates is the maximum of
    # a finite-sample R^2 over layers, which is positively biased.
    shuffles = [
        torch.randperm(train_count, generator=rng).to(device)
        for _ in range(args.shuffle_draws)
    ]

    # The no-prefix control: with the prompt bytes identical across states, the
    # hidden state is a function of the sentence alone. Verified, not assumed.
    control_hidden: list[list[float]] = []
    control_identical = True
    per_sentence: dict[int, tuple[float, ...]] = {}
    for sentence_index, sentence in enumerate(DIAGNOSTIC_SENTENCES):
        seen = None
        for state_index in (0, 1, len(all_states) - 1):
            ids = _encode(
                tokenizer, torch, state=all_states[state_index],
                sentence=sentence, user_id=f"diag-{state_index:04d}",
                device=device,
                state_domain=args.state_domain,
            )
            profile = capture_prefix_diagnostics(
                torch_module=torch, model=model, input_ids=ids,
                prefix_pairs=None, cache_factory=cache_factory,
            )
            if seen is None:
                seen = profile.final_hidden
            elif profile.final_hidden != seen:
                control_identical = False
        per_sentence[sentence_index] = seen
    for _ in range(len(all_states)):
        for sentence_index in range(len(DIAGNOSTIC_SENTENCES)):
            control_hidden.append(list(per_sentence[sentence_index]))

    held_out: dict[int, float] = {}
    shuffled: dict[int, float] = {}
    control: dict[int, float] = {}
    selected_alpha: dict[int, float] = {}
    # Fold assignment is by state, not by row: the same state under a
    # different probe sentence is not an independent validation point.
    groups = [
        index // len(DIAGNOSTIC_SENTENCES) for index in range(train_count)
    ]
    for layer in range(layers):
        features = torch.tensor(
            [row[layer] for row, _ in hidden_rows],
            dtype=torch.float32, device=device,
        )
        alpha = select_ridge_alpha(
            torch_module=torch,
            features=features[:train_count],
            targets=targets[:train_count],
            groups=groups,
        )
        selected_alpha[layer] = alpha
        held_out[layer] = fit_ridge_probe(
            torch_module=torch,
            train_features=features[:train_count],
            train_targets=targets[:train_count],
            eval_features=features[train_count:],
            eval_targets=targets[train_count:],
            layer_index=layer,
            alpha=alpha,
        ).mean_r2
        shuffled[layer] = max(
            fit_ridge_probe(
                torch_module=torch,
                train_features=features[:train_count],
                train_targets=targets[:train_count][draw],
                eval_features=features[train_count:],
                eval_targets=targets[train_count:],
                layer_index=layer,
                alpha=alpha,
            ).mean_r2
            for draw in shuffles
        )
        control_features = torch.tensor(
            [row[layer] for row in control_hidden],
            dtype=torch.float32, device=device,
        )
        control[layer] = fit_ridge_probe(
            torch_module=torch,
            train_features=control_features[:train_count],
            train_targets=targets[:train_count],
            eval_features=control_features[train_count:],
            eval_targets=targets[train_count:],
            layer_index=layer,
            alpha=alpha,
        ).mean_r2

    readable_claim = evaluate_state_linearly_readable(
        held_out_r2=held_out,
        shuffled_r2=shuffled,
        control_r2=control,
        control_hidden_identical=control_identical,
    )
    print(f"gate B: {readable_claim.state.value} — {readable_claim.detail}")

    verdict = build_carrier_diagnostics_verdict(
        substrate_fingerprint=(
            f"{args.model_id}@{str(weights['weights_sha256'])[:16]}"
        ),
        prefix_artifact_id=bound_artifact_id,
        attention_claim=attention_claim,
        readable_claim=readable_claim,
        slot_mass_report={
            "learned": learned_mass,
            "zero": list(zero_profile.slot_mass),
            "random": random_mass,
            "uniform_expectation": [
                artifact.num_slots / (artifact.num_slots + int(reference_ids.shape[-1]))
            ],
        },
        nonuniformity_report={
            "learned": learned_nonuniformity,
            "zero": list(zero_profile.slot_nonuniformity),
            "random": random_nonuniformity,
        },
        probe_report={
            "held_out": held_out,
            "shuffled_label_control": shuffled,
            "no_prefix_control": control,
            "selected_ridge_alpha": selected_alpha,
        },
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")
    fingerprint = output.with_name("substrate_fingerprint.json")
    fingerprint.write_text(
        json.dumps(
            {
                **weights,
                "prefix_artifact_id": bound_artifact_id,
                "state_domain": args.state_domain,
                "device_request": args.device,
                "attn_implementation": "eager",
                "diagnostic_sentences": list(DIAGNOSTIC_SENTENCES),
                "identification_probe_sentences": [s for _, s in PROBE_SENTENCES],
                "assembly": _assembly(
                    residue="", ordering_driver="playbook-only"
                ).description,
                "base_context_regime": _base_context().regime_id,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"carrier_is_live = {verdict.carrier_is_live}")
    print(f"verdict: {output}")
    print(f"substrate fingerprint: {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
