"""Linear probe: does the step-0 encoder input still carry the plan identity?

Under observation protocol v2 the ordered route plan is stated exactly once, in
the step-0 observation text. The metacontroller never sees that text directly;
it sees a 16-dim folded summary of the frozen substrate's residual capture
(``_step_input_vectors``, mirroring ``_summarize_substrate_ndim``). If that
summary destroys the plan identity, no rate objective -- gated or not -- can
make the latent code carry the active subgoal across boundaries, and
``boundary_f1 = 0`` is a foregone conclusion regardless of the rate economics.

This diagnostic captures the *same* traces the Gate-1 sweep trains on (same
frozen corpus seed, same v2 protocol, same real substrate) and fits a
multinomial linear probe from each route's step-0 input vector to:

- ``target_1``: the first planned subgoal (near-trivially present if anything
  survives, because step 0 is where the expert immediately heads there), and
- ``target_2``: the second planned subgoal, the piece of plan identity the
  latent must transport across the first boundary.

A shuffled-label null run calibrates what "chance" looks like for this probe
capacity and sample size. Reading:

- target_2 well above the null  -> the input surface is sufficient; a
  switching failure is then attributable to the objective/optimization.
- target_2 at the null          -> the activation summary is the bottleneck;
  widen the input surface before spending authoritative compute.

Not authoritative evidence; a capacity diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    _build_eta_open_weight_runtime,
    _validate_eta_open_weight_runtime,
    generate_eta_proof_corpus,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V2,
    _rate_distortion_observation_texts,
)
from volvence_zero.temporal.metacontroller_components import (
    _fold_residual_to_ndim,
)


def _fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    steps: int = 800,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> torch.nn.Linear:
    torch.manual_seed(seed)
    probe = torch.nn.Linear(features.shape[1], class_count, dtype=torch.float64)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=0.05, weight_decay=weight_decay
    )
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(features), labels)
        loss.backward()
        optimizer.step()
    return probe


def _cross_validated_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    folds: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    fold_accuracies: list[float] = []
    for fold in range(folds):
        test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask[permutation[fold::folds]] = True
        probe = _fit_linear_probe(
            features[~test_mask],
            labels[~test_mask],
            class_count=class_count,
            seed=seed + fold,
        )
        with torch.no_grad():
            predictions = probe(features[test_mask]).argmax(dim=1)
        fold_accuracies.append(
            float((predictions == labels[test_mask]).double().mean())
        )
    return {
        "mean_test_accuracy": statistics.fmean(fold_accuracies),
        "std_test_accuracy": (
            statistics.pstdev(fold_accuracies)
            if len(fold_accuracies) > 1
            else 0.0
        ),
        "fold_accuracies": fold_accuracies,
    }


def _capture_last_raw(capture) -> tuple[float, ...]:
    raw: list[float] = []
    for act in capture.residual_activations:
        raw.extend(act.activation)
    return tuple(raw)


def _capture_mean_raw(capture, *, tail: int | None = None) -> tuple[float, ...]:
    steps = capture.residual_sequence[-tail:] if tail else capture.residual_sequence
    per_layer_sums: dict[int, list[float]] = {}
    for sequence_step in steps:
        for act in sequence_step.residual_activations:
            bucket = per_layer_sums.setdefault(
                act.layer_index, [0.0] * len(act.activation)
            )
            for index, value in enumerate(act.activation):
                bucket[index] += value
    raw: list[float] = []
    for layer_index in sorted(per_layer_sums):
        raw.extend(value / len(steps) for value in per_layer_sums[layer_index])
    return tuple(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Linear probe for plan identity in the step-0 encoder input "
            "(frozen Gate-1 corpus, v2 protocol, real substrate)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eta_step0_plan_probe_20260802"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--n-inputs",
        type=int,
        nargs="+",
        default=(16,),
        help=(
            "Controller input widths to evaluate. Folding is applied "
            "post-hoc per width from one shared capture, so several widths "
            "cost one sweep of model forwards."
        ),
    )
    parser.add_argument(
        "--activation-width",
        type=int,
        default=8,
        help=(
            "Per-layer capture width (chunk-mean pooling of the 896-dim "
            "hidden state). The Gate-1 default is 8; wider widths test "
            "whether the pooling itself is what destroys plan identity."
        ),
    )
    parser.add_argument(
        "--probe-route-count",
        type=int,
        default=None,
        help=(
            "Override the train-route count for probe power. Default keeps "
            "the frozen Gate-1 corpus (64 train + 24 heldout routes)."
        ),
    )
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    # Identical corpus family to the frozen Gate-1 protocol (route count may
    # be raised for probe power; the graph/renderer are the same).
    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=8,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=args.probe_route_count or 64,
        heldout_route_count=24,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )
    config = ETAOpenWeightRuntimeConfig(
        model_id=args.model_id,
        device=args.device,
        model_dtype="float32",
        activation_width=args.activation_width,
    )
    runtime = _build_eta_open_weight_runtime(config)
    _validate_eta_open_weight_runtime(runtime=runtime, config=config)

    cases = tuple(corpus.train_cases) + tuple(corpus.heldout_cases)

    # Two captures per route (step-0 observation only), three candidate
    # encoder inputs:
    # - last-token: what the Gate-1 pipeline feeds the encoder today
    #   (``capture.residual_activations`` is the final token position);
    # - token-mean: true mean over every prompt token position, computed from
    #   ``residual_sequence`` (the pipeline currently strips this); and
    # - plan-at-end last-token: a counterfactual v2 rendering with the route
    #   plan restated at the END of the step-0 text, testing whether causal
    #   recency (not capture pooling) is what hides the plan from the final
    #   token's hidden state.
    last_raws: list[tuple[float, ...]] = []
    mean_raws: list[tuple[float, ...]] = []
    plan_end_raws: list[tuple[float, ...]] = []
    plan_end_tail_raws: list[tuple[float, ...]] = []
    plan_alone_last_raws: list[tuple[float, ...]] = []
    plan_alone_mean_raws: list[tuple[float, ...]] = []
    label_target_1: list[str] = []
    label_target_2: list[str] = []
    label_plan_length: list[str] = []
    label_last_transition: list[str] = []
    for case in cases:
        texts, _targets = _rate_distortion_observation_texts(
            case,
            environment=corpus.environment,
            protocol_version=OBSERVATION_PROTOCOL_V2,
        )
        step0_text = texts[0]
        capture = runtime.capture(source_text=step0_text)
        last_raws.append(_capture_last_raw(capture))
        mean_raws.append(_capture_mean_raw(capture))

        plan_prefix = f"Route plan: {case.source_text}. "
        if not step0_text.startswith(plan_prefix):
            raise RuntimeError(
                f"v2 step-0 text for {case.case_id!r} does not start with "
                "the expected route-plan prefix; the counterfactual "
                "rendering below would drift from the real protocol."
            )
        plan_end_text = (
            f"{step0_text[len(plan_prefix):]} Route plan: {case.source_text}."
        )
        plan_end_capture = runtime.capture(source_text=plan_end_text)
        plan_end_raws.append(_capture_last_raw(plan_end_capture))
        plan_end_tail_raws.append(_capture_mean_raw(plan_end_capture, tail=8))

        plan_alone_capture = runtime.capture(
            source_text=f"Route plan: {case.source_text}."
        )
        plan_alone_last_raws.append(_capture_last_raw(plan_alone_capture))
        plan_alone_mean_raws.append(_capture_mean_raw(plan_alone_capture))

        # Readability control: the final transition name is spelled by the
        # very last content tokens of the step-0 prompt; if even this is not
        # linearly readable from the last-token capture, no prompt content is.
        transitions_marker = "Available transitions: "
        marker_index = step0_text.rindex(transitions_marker)
        transition_list = step0_text[
            marker_index + len(transitions_marker) :
        ].rstrip(".")
        label_last_transition.append(
            transition_list.split(", ")[-1].strip()
        )

        # route_signature is the full waypoint path including corridor hops;
        # the plan stated in the step-0 text is the ordered OBJECTIVE
        # sequence. Label with objectives only.
        objectives = tuple(
            waypoint
            for waypoint in case.route_signature
            if corpus.environment.location(waypoint).is_objective
        )
        if len(objectives) < 2:
            raise RuntimeError(
                f"route {case.case_id!r} has fewer than two planned "
                "objectives; probe labels would be undefined."
            )
        label_target_1.append(objectives[0])
        label_target_2.append(objectives[1])
        # Positive control: plan length (objective count) is trivially
        # encoded by prompt length; if the probe cannot recover even this,
        # the probe method itself (not the representation) is broken.
        label_plan_length.append(str(len(objectives)))

    results: dict[str, object] = {}
    for pooling, raws in (
        ("last-token", last_raws),
        ("token-mean", mean_raws),
        ("plan-at-end-last-token", plan_end_raws),
        ("plan-at-end-last8-mean", plan_end_tail_raws),
        ("plan-alone-last-token", plan_alone_last_raws),
        ("plan-alone-mean", plan_alone_mean_raws),
        ("plan-alone-last-token", plan_alone_last_raws),
        ("plan-alone-mean", plan_alone_mean_raws),
    ):
        for n_input in args.n_inputs:
            features = torch.tensor(
                [_fold_residual_to_ndim(raw, n_input) for raw in raws],
                dtype=torch.float64,
            )
            features = (features - features.mean(dim=0)) / (
                features.std(dim=0) + 1e-9
            )
            width_results: dict[str, object] = {}
            for name, raw_labels in (
                ("target_1", label_target_1),
                ("target_2", label_target_2),
                ("plan_length_control", label_plan_length),
                ("last_transition_control", label_last_transition),
                ("last_transition_control", label_last_transition),
            ):
                vocabulary = sorted(set(raw_labels))
                labels = torch.tensor(
                    [vocabulary.index(value) for value in raw_labels]
                )
                counts = torch.bincount(labels, minlength=len(vocabulary))
                majority_share = float(counts.max()) / len(raw_labels)
                real = _cross_validated_accuracy(
                    features,
                    labels,
                    class_count=len(vocabulary),
                    folds=args.folds,
                )
                shuffle_generator = torch.Generator().manual_seed(1234)
                null = _cross_validated_accuracy(
                    features,
                    labels[
                        torch.randperm(
                            len(raw_labels), generator=shuffle_generator
                        )
                    ],
                    class_count=len(vocabulary),
                    folds=args.folds,
                )
                width_results[name] = {
                    "class_count": len(vocabulary),
                    "majority_class_share": majority_share,
                    "probe": real,
                    "shuffled_label_null": null,
                }
                print(
                    f"{pooling} n_input={n_input} {name}: "
                    f"classes={len(vocabulary)} "
                    f"majority={majority_share:.3f} "
                    f"probe={real['mean_test_accuracy']:.3f}"
                    f"+/-{real['std_test_accuracy']:.3f} "
                    f"null={null['mean_test_accuracy']:.3f}"
                    f"+/-{null['std_test_accuracy']:.3f}"
                )
            results[f"{pooling}:n_input_{n_input}"] = width_results

    payload = {
        "schema_version": "eta-step0-plan-identity-probe.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Capacity diagnostic for the step-0 encoder input under protocol "
            "v2 on the frozen Gate-1 corpus. Not authoritative evidence."
        ),
        "model_id": runtime.model_id,
        "runtime_origin": runtime.runtime_origin,
        "fallback_active": runtime.fallback_active,
        "device": args.device,
        "observation_protocol": OBSERVATION_PROTOCOL_V2,
        "n_inputs": list(args.n_inputs),
        "activation_width": args.activation_width,
        "corpus_seed": args.corpus_seed,
        "route_count": len(cases),
        "folds": args.folds,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "step0_plan_probe.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
