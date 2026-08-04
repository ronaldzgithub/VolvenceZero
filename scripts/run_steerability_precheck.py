"""Read-only steerability pre-check (Braun et al. 2505.22637 indicators).

This diagnostic does NOT train, write back, or touch production wiring.  It
reuses the exact S1/S2 forward-capture path (merged Stage-2 v2 base, layer 20,
896-wide, L2-normalized latest-token residual, staged-plan v4 cumulative
prefixes) to answer one question left open by the S2 causal-steering FAIL:

    Is the active-subgoal a coherent, separable linear direction in the frozen
    residual stream (Braun's precondition for steerability), and how much does
    the S1 probe axis diverge from the causally-grounded difference-of-means
    axis (research/steering-2026-08 sec 3.2)?

For each subgoal class k it computes, on the heldout prefixes S2 was scored on:
  * axis_dom_k  = normalize(mean(rep | subgoal=k) - mean(rep | subgoal!=k))
  * axis_probe_k = S1 v2 artifact.axis_for(k)  (class-vs-rest ridge contrast)
  * cosine(axis_dom_k, axis_probe_k)
  * directional agreement: mean over positive samples of
        cos(rep - mean_neg, axis)   (Braun Fig 2/6; steerable ~0.48, not ~0.19)
  * discriminability index d' along each axis (Braun Fig 3)
  * reversal fraction: positives that project below the class midpoint

It is intentionally read-only: no admission gate, no artifact promotion.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from companion_test_plan_common import (
    exclusive_mps_lock,
    require_mps,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V4,
    eta_stage2_probe_rows,
)
from volvence_zero.substrate import (
    SubstrateFingerprint,
    TransformersOpenWeightResidualRuntime,
)
from volvence_zero.substrate.forward_representation import (
    SubstrateForwardRepresentationPublisher,
)
from volvence_zero.substrate.frozen_residual_readout import (
    FrozenResidualReadoutArtifact,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "steerability-precheck-readonly-mps.v1"

# Braun (2505.22637) reference bands for the 36 assistant-behaviour datasets.
BRAUN_STEERABLE_MEAN_COSINE = 0.48
BRAUN_UNSTEERABLE_MEAN_COSINE = 0.19


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: tuple[float, ...]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: tuple[float, ...]) -> tuple[float, ...]:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(x / n for x in a)


def _mean_vec(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    count = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            acc[i] += value
    return tuple(value / count for value in acc)


def _discriminability(pos_proj: list[float], neg_proj: list[float]) -> float:
    mu_p = sum(pos_proj) / len(pos_proj)
    mu_n = sum(neg_proj) / len(neg_proj)
    var_p = sum((x - mu_p) ** 2 for x in pos_proj) / len(pos_proj)
    var_n = sum((x - mu_n) ** 2 for x in neg_proj) / len(neg_proj)
    denom = math.sqrt(0.5 * (var_p + var_n))
    if denom == 0.0:
        return float("inf")
    return abs(mu_p - mu_n) / denom


def _axis_report(
    *,
    axis: tuple[float, ...],
    pos: list[tuple[float, ...]],
    neg: list[tuple[float, ...]],
    mean_neg: tuple[float, ...],
) -> dict[str, float]:
    # Directional agreement: cosine of each positive sample's offset from the
    # negative mean with the axis (Braun Fig 2/6 analogue).
    cosines: list[float] = []
    for vec in pos:
        offset = tuple(v - m for v, m in zip(vec, mean_neg, strict=True))
        offset_norm = _norm(offset)
        if offset_norm == 0.0:
            continue
        cosines.append(_dot(offset, axis) / offset_norm)
    mean_cosine = sum(cosines) / len(cosines) if cosines else 0.0
    pos_proj = [_dot(vec, axis) for vec in pos]
    neg_proj = [_dot(vec, axis) for vec in neg]
    midpoint = 0.5 * (
        sum(pos_proj) / len(pos_proj) + sum(neg_proj) / len(neg_proj)
    )
    reversal = sum(1 for p in pos_proj if p < midpoint) / len(pos_proj)
    return {
        "mean_directional_cosine": mean_cosine,
        "discriminability_dprime": _discriminability(pos_proj, neg_proj),
        "reversal_fraction": reversal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument(
        "--model-source", default="artifacts/eta_stage2_merged_v2_20260803"
    )
    parser.add_argument("--model-version", default="eta-stage2-merged-v2")
    parser.add_argument(
        "--s1-artifact",
        default="artifacts/eta_s1_residual_readout_v2_20260804/readout_artifact.json",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--min-class-support", type=int, default=8)
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.steerability-precheck-mps.lock"),
    )
    parser.add_argument(
        "--output",
        default="research/steering-2026-08/steerability-precheck-result.json",
    )
    args = parser.parse_args()

    model_root = (_REPO_ROOT / args.model_source).resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"model source not found: {model_root}")
    artifact_path = (_REPO_ROOT / args.s1_artifact).resolve()
    artifact = FrozenResidualReadoutArtifact.from_json(
        artifact_path.read_text(encoding="utf-8")
    )

    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )
    heldout_rows, class_ids = eta_stage2_probe_rows(
        corpus.heldout_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    if tuple(class_ids) != tuple(artifact.class_ids):
        raise RuntimeError(
            "class vocabulary differs between corpus and S1 artifact"
        )

    fingerprint = SubstrateFingerprint(
        model_id=args.model_id,
        version=args.model_version,
        weights_sha256=artifact.model_fingerprint.weights_sha256,
    )

    started = time.perf_counter()
    with ExitStack() as stack:
        if args.device.startswith("mps"):
            stack.enter_context(
                exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID)
            )
            require_mps()
        runtime = TransformersOpenWeightResidualRuntime(
            model_id=args.model_id,
            pretrained_source=str(model_root),
            device=args.device,
            max_length=args.max_length,
            fail_on_truncation=True,
            layer_indices=(20,),
            activation_width=896,
            local_files_only=True,
            runtime_origin="hf-local",
            allow_live_substrate_mutation=False,
            allow_offline_substrate_training=False,
            model_dtype="float32",
        )
        publisher = SubstrateForwardRepresentationPublisher(
            runtime, model_fingerprint=fingerprint
        )
        sample_sources = tuple(
            (
                f"{row.split}:{row.case_id}:step-{row.step_index}",
                row.observation_text,
            )
            for row in heldout_rows
        )

        def progress(sample_id: str, completed: int, total: int) -> None:
            if completed == total or completed % 25 == 0:
                print(
                    f"capture {completed}/{total} sample={sample_id}",
                    flush=True,
                )

        snapshot = publisher.publish(sample_sources, progress=progress)
    elapsed = time.perf_counter() - started

    reps = [tuple(row.values) for row in snapshot.representations]
    labels = [class_ids[row.subgoal_label] for row in heldout_rows]
    if len(reps) != len(labels):
        raise RuntimeError("representation/label count mismatch")

    by_class: dict[str, list[tuple[float, ...]]] = {cid: [] for cid in class_ids}
    for vec, label in zip(reps, labels, strict=True):
        by_class[label].append(vec)

    per_class: list[dict[str, object]] = []
    dom_cosines: list[float] = []
    probe_cosines: list[float] = []
    dom_dprimes: list[float] = []
    probe_dprimes: list[float] = []
    dom_probe_alignment: list[float] = []
    for cid in class_ids:
        pos = by_class[cid]
        neg = [vec for vec, label in zip(reps, labels, strict=True) if label != cid]
        if len(pos) < args.min_class_support or not neg:
            per_class.append(
                {
                    "class_id": cid,
                    "support": len(pos),
                    "skipped": True,
                    "reason": "insufficient-support",
                }
            )
            continue
        mean_pos = _mean_vec(pos)
        mean_neg = _mean_vec(neg)
        axis_dom = _normalize(
            tuple(p - n for p, n in zip(mean_pos, mean_neg, strict=True))
        )
        axis_probe = artifact.axis_for(cid)
        alignment = _dot(axis_dom, axis_probe)
        dom = _axis_report(axis=axis_dom, pos=pos, neg=neg, mean_neg=mean_neg)
        probe = _axis_report(
            axis=axis_probe, pos=pos, neg=neg, mean_neg=mean_neg
        )
        dom_cosines.append(dom["mean_directional_cosine"])
        probe_cosines.append(probe["mean_directional_cosine"])
        dom_dprimes.append(dom["discriminability_dprime"])
        probe_dprimes.append(probe["discriminability_dprime"])
        dom_probe_alignment.append(alignment)
        per_class.append(
            {
                "class_id": cid,
                "support": len(pos),
                "skipped": False,
                "dom_probe_axis_cosine": alignment,
                "diff_of_means_axis": dom,
                "s1_probe_axis": probe,
            }
        )

    def _avg(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    aggregate = {
        "evaluated_classes": len(dom_cosines),
        "mean_dom_directional_cosine": _avg(dom_cosines),
        "mean_probe_directional_cosine": _avg(probe_cosines),
        "mean_dom_dprime": _avg(dom_dprimes),
        "mean_probe_dprime": _avg(probe_dprimes),
        "mean_dom_probe_axis_cosine": _avg(dom_probe_alignment),
        "braun_steerable_reference_cosine": BRAUN_STEERABLE_MEAN_COSINE,
        "braun_unsteerable_reference_cosine": BRAUN_UNSTEERABLE_MEAN_COSINE,
    }

    result = {
        "schema_version": "steerability-precheck.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "claim_scope": "steerability-precheck-no-admission-no-writeback",
        "model_id": args.model_id,
        "model_source": args.model_source,
        "model_weights_sha256": artifact.model_fingerprint.weights_sha256,
        "s1_artifact_id": artifact.artifact_id,
        "observation_protocol": OBSERVATION_PROTOCOL_V4,
        "layer_index": 20,
        "representation_dim": artifact.representation_dim,
        "heldout_prefix_count": len(reps),
        "heldout_route_count": corpus.heldout_route_count,
        "class_ids": list(class_ids),
        "wall_seconds": elapsed,
        "aggregate": aggregate,
        "per_class": per_class,
    }

    output_path = (_REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== steerability pre-check (read-only) ===", flush=True)
    print(
        f"heldout prefixes={len(reps)} classes_evaluated={aggregate['evaluated_classes']}",
        flush=True,
    )
    print(
        f"diff-of-means: mean directional cosine="
        f"{aggregate['mean_dom_directional_cosine']:.3f} "
        f"d'={aggregate['mean_dom_dprime']:.3f}",
        flush=True,
    )
    print(
        f"s1 probe axis: mean directional cosine="
        f"{aggregate['mean_probe_directional_cosine']:.3f} "
        f"d'={aggregate['mean_probe_dprime']:.3f}",
        flush=True,
    )
    print(
        f"mean cosine(dom, probe axis)="
        f"{aggregate['mean_dom_probe_axis_cosine']:.3f}",
        flush=True,
    )
    print(
        f"Braun reference: steerable~{BRAUN_STEERABLE_MEAN_COSINE} "
        f"unsteerable~{BRAUN_UNSTEERABLE_MEAN_COSINE}",
        flush=True,
    )
    print(f"written {output_path}", flush=True)


if __name__ == "__main__":
    main()
