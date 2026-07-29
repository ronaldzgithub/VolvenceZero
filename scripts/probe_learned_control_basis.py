"""GO/NO-GO probe: does a learned control basis create transferable action value?

Development diagnostic for the Gate 2 learned-control-basis convergence
packet. The v33 falsification showed the sinusoid control basis has real
per-prefix causal power (realized-continuation NLL effect ranges >= 0.02)
but zero transfer across contexts (cross-prefix Spearman ~0.14, global
action-main train->validation R^2 ~ -0.01). Hypothesis: steering along
directions the frozen model's own hidden state traverses when a route
advances (train-transition PCA) yields candidate rankings that transfer.

Protocol (frozen-split hygiene):
- Basis is fit ONLY from train-split route captures.
- Transfer is evaluated on train-subset (reference) plus eval + heldout
  routes (development splits). The frozen validation split is never
  touched by this probe; it is reserved for the official evidence run.
- Both bases (fixed sinusoid baseline, learned) are scored on identical
  prefix/candidate grids in one process for a paired comparison.

Usage:
    python scripts/probe_learned_control_basis.py --out /tmp/learned-basis-probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    _build_case_snapshot_bundle,
    _build_eta_open_weight_runtime,
    _continuation_counterfactual_candidates,
    _continuation_nlls,
    eta_gate2_expected_value_cases,
)
from volvence_zero.substrate import (
    TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE,
    control_basis_fingerprint,
    fit_transition_control_basis,
)


def _snapshot_state_vector(snapshot) -> tuple[float, ...]:
    """Mean over hook layers of the last-token hidden activation."""

    last_step = snapshot.residual_sequence[-1]
    activations = last_step.residual_activations
    if not activations:
        raise ValueError("snapshot carries no residual activations")
    width = len(activations[0].activation)
    return tuple(
        sum(layer.activation[index] for layer in activations) / len(activations)
        for index in range(width)
    )


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        tie_end = position
        while (
            tie_end + 1 < len(order)
            and values[order[tie_end + 1]] == values[order[position]]
        ):
            tie_end += 1
        mean_rank = (position + tie_end) / 2.0
        for tied_position in range(position, tie_end + 1):
            ranks[order[tied_position]] = mean_rank
        position = tie_end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("spearman requires two equal-length vectors, n >= 3")
    ranks_left = _ranks(left)
    ranks_right = _ranks(right)
    mean_left = sum(ranks_left) / len(ranks_left)
    mean_right = sum(ranks_right) / len(ranks_right)
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(ranks_left, ranks_right, strict=True)
    )
    var_left = sum((a - mean_left) ** 2 for a in ranks_left)
    var_right = sum((b - mean_right) ** 2 for b in ranks_right)
    if var_left <= 0.0 or var_right <= 0.0:
        return 0.0
    return numerator / math.sqrt(var_left * var_right)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score_basis(
    *,
    runtime,
    contexts,
    candidates,
) -> dict:
    """Score all candidate controls on every prefix context for one basis."""

    score_cache: dict = {}
    rows = []
    for context in contexts:
        zero_nll = _continuation_nlls(
            runtime=runtime,
            source_text=context["prefix"],
            continuation_texts=(context["segment"],),
            applied_control=(0.0, 0.0, 0.0),
            score_cache=score_cache,
        )[0]
        improvements = []
        for candidate in candidates:
            if candidate == (0.0, 0.0, 0.0):
                improvements.append(0.0)
                continue
            candidate_nll = _continuation_nlls(
                runtime=runtime,
                source_text=context["prefix"],
                continuation_texts=(context["segment"],),
                applied_control=candidate,
                score_cache=score_cache,
            )[0]
            improvements.append(zero_nll - candidate_nll)
        row = {
            "case_id": context["case_id"],
            "split": context["split"],
            "index": context["index"],
            "zero_nll": zero_nll,
            "improvements": improvements,
        }
        rows.append(row)
        print("ROW " + json.dumps(row), flush=True)
    return _summarize_rows(rows=rows, candidates=candidates)


def _summarize_rows(*, rows, candidates) -> dict:
    nonzero_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate != (0.0, 0.0, 0.0)
    ]
    effect_ranges = [
        max(row["improvements"]) - min(row["improvements"]) for row in rows
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] != "train"]

    def mean_credit(subset):
        return [
            _mean([row["improvements"][index] for row in subset])
            for index in nonzero_indices
        ]

    train_mean = mean_credit(train_rows)
    dev_mean = mean_credit(dev_rows)
    pairwise = []
    for i in range(len(dev_rows)):
        for j in range(i + 1, len(dev_rows)):
            pairwise.append(
                _spearman(
                    [dev_rows[i]["improvements"][k] for k in nonzero_indices],
                    [dev_rows[j]["improvements"][k] for k in nonzero_indices],
                )
            )
    within_case_adjacent = []
    by_case: dict = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_rows in by_case.values():
        case_rows.sort(key=lambda row: row["index"])
        for left, right in zip(case_rows, case_rows[1:], strict=False):
            within_case_adjacent.append(
                _spearman(
                    [left["improvements"][k] for k in nonzero_indices],
                    [right["improvements"][k] for k in nonzero_indices],
                )
            )
    sorted_ranges = sorted(effect_ranges)
    return {
        "rows": rows,
        "context_count": len(rows),
        "effect_range_min": min(effect_ranges),
        "effect_range_median": sorted_ranges[len(sorted_ranges) // 2],
        "effect_range_max": max(effect_ranges),
        "train_dev_mean_credit_spearman": _spearman(train_mean, dev_mean),
        "train_mean_credit": train_mean,
        "dev_mean_credit": dev_mean,
        "dev_mean_credit_positive_rate": (
            sum(1 for value in dev_mean if value > 0.0) / len(dev_mean)
        ),
        "dev_pairwise_spearman_mean": _mean(pairwise),
        "within_case_adjacent_spearman_mean": _mean(within_case_adjacent),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/learned-basis-probe.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-eval-cases", type=int, default=3)
    args = parser.parse_args()

    config = ETAOpenWeightRuntimeConfig(
        device=args.device,
        activation_width=896,
        max_prefix_steps=8,
    )
    runtime = _build_eta_open_weight_runtime(config)
    if runtime.fallback_active:
        raise RuntimeError("probe requires the real Qwen backend, got fallback")

    corpus = eta_gate2_expected_value_cases()
    train_cases = [case for case in corpus if case.split == "train"]
    dev_cases = [case for case in corpus if case.split in ("eval", "heldout")]
    validation_case_ids = [
        case.case_id for case in corpus if case.split == "validation"
    ]
    print(
        f"corpus: train={len(train_cases)} dev={len(dev_cases)} "
        f"validation(untouched)={len(validation_case_ids)}",
        flush=True,
    )

    started = time.time()
    transition_deltas: list[tuple[float, ...]] = []
    snapshot_bundles: dict = {}
    for case in train_cases + dev_cases:
        snapshots, prefixes = _build_case_snapshot_bundle(
            case,
            open_weight_runtime=runtime,
            open_weight_config=config,
        )
        snapshot_bundles[case.case_id] = (snapshots, prefixes)
        if case.split == "train":
            states = [_snapshot_state_vector(snapshot) for snapshot in snapshots]
            for left, right in zip(states, states[1:], strict=False):
                transition_deltas.append(
                    tuple(b - a for a, b in zip(left, right, strict=True))
                )
        print(
            f"captured case={case.case_id} split={case.split} "
            f"prefixes={len(prefixes)} elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    basis = fit_transition_control_basis(transition_deltas, basis_rank=3)
    fingerprint = control_basis_fingerprint(basis)
    print(
        f"fitted basis from {len(transition_deltas)} train transition deltas, "
        f"fingerprint={fingerprint[:16]}",
        flush=True,
    )

    eval_cases = train_cases[: args.train_eval_cases] + dev_cases
    contexts = []
    for case in eval_cases:
        _snapshots, prefixes = snapshot_bundles[case.case_id]
        for index in range(len(prefixes) - 1):
            segment = prefixes[index + 1][len(prefixes[index]):].strip()
            if not segment:
                raise ValueError(
                    f"non-growing prefix at case={case.case_id} index={index}"
                )
            contexts.append(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "index": index,
                    "prefix": prefixes[index],
                    "segment": segment,
                }
            )
    candidates = _continuation_counterfactual_candidates()
    print(
        f"scoring {len(contexts)} contexts x {len(candidates)} candidates "
        f"x 2 bases",
        flush=True,
    )

    print("=== pass 1: fixed sinusoid baseline ===", flush=True)
    sinusoid_summary = _score_basis(
        runtime=runtime,
        contexts=contexts,
        candidates=candidates,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in sinusoid_summary.items()
                if key not in ("rows", "train_mean_credit", "dev_mean_credit")
            },
            indent=2,
        ),
        flush=True,
    )

    print("=== pass 2: learned train-transition PCA basis ===", flush=True)
    runtime.install_control_basis(
        basis=basis,
        provenance=(
            f"{TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE}:{fingerprint[:16]}"
        ),
    )
    learned_summary = _score_basis(
        runtime=runtime,
        contexts=contexts,
        candidates=candidates,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in learned_summary.items()
                if key not in ("rows", "train_mean_credit", "dev_mean_credit")
            },
            indent=2,
        ),
        flush=True,
    )

    report = {
        "probe": "learned-control-basis-transfer",
        "runtime_model_id": runtime.model_id,
        "device": args.device,
        "basis_fingerprint": fingerprint,
        "transition_delta_count": len(transition_deltas),
        "context_count": len(contexts),
        "candidate_count": len(candidates),
        "validation_untouched": validation_case_ids,
        "sinusoid": sinusoid_summary,
        "learned": learned_summary,
        "elapsed_seconds": time.time() - started,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.out} elapsed={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
