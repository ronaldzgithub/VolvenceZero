# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""G2 evaluation harness.

Produces one JSON report with the encoder and every baseline scored by
the same metric functions on the same val split:

* structured prediction — phase accuracy + macro-F1, per-target MAE
  (trust / continuity / repair), expected calibration error (ECE);
* embedding quality — trajectory-level retrieval: for each val
  trajectory, does the nearest neighbour (cosine, leave-one-out within
  val) share its scenario family? Compared against the standard's
  deterministic stub embedding, which is the "no semantics" floor.

Metric functions are pure Python over ``StatePrediction`` tuples, so
baseline columns never need torch and the encoder column cannot drift to
a different scoring path.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

from companion_standard import stub_semantic_embedding

from companion_encoder.baselines import MajorityBaseline, StatePrediction
from companion_encoder.dataset import (
    PHASE_VOCAB,
    REGRESSION_TARGETS,
    AnchorExample,
    DatasetSplits,
)
from companion_encoder.serialization import render_full

# ---------------------------------------------------------------- metrics


def phase_metrics(
    predictions: tuple[StatePrediction, ...], examples: tuple[AnchorExample, ...]
) -> dict:
    if len(predictions) != len(examples):
        raise ValueError(
            f"prediction/example count mismatch: {len(predictions)} != {len(examples)}"
        )
    correct = sum(
        1 for p, e in zip(predictions, examples, strict=True) if p.phase is e.phase
    )
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    for prediction, example in zip(predictions, examples, strict=True):
        if prediction.phase is example.phase:
            per_class[example.phase.value]["tp"] += 1
        else:
            per_class[prediction.phase.value]["fp"] += 1
            per_class[example.phase.value]["fn"] += 1
    f1_scores = []
    for phase in PHASE_VOCAB:
        counts = per_class[phase.value]
        support = counts["tp"] + counts["fn"]
        if support == 0:
            continue  # macro-F1 over classes present in the val split
        precision_denominator = counts["tp"] + counts["fp"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / support
        f1_scores.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return {
        "accuracy": correct / len(examples),
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "invalid_rate": sum(1 for p in predictions if not p.valid) / len(predictions),
    }


def regression_metrics(
    predictions: tuple[StatePrediction, ...], examples: tuple[AnchorExample, ...]
) -> dict:
    maes = {}
    for target_position, target_name in enumerate(REGRESSION_TARGETS):
        maes[f"{target_name}_mae"] = sum(
            abs(p.regression_values[target_position] - e.regression_targets[target_position])
            for p, e in zip(predictions, examples, strict=True)
        ) / len(examples)
    maes["mean_mae"] = sum(maes.values()) / len(REGRESSION_TARGETS)
    return maes


def expected_calibration_error(
    predictions: tuple[StatePrediction, ...],
    examples: tuple[AnchorExample, ...],
    *,
    bin_count: int = 10,
) -> float:
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(bin_count)]
    for prediction, example in zip(predictions, examples, strict=True):
        bin_index = min(int(prediction.confidence * bin_count), bin_count - 1)
        bins[bin_index].append(
            (prediction.confidence, prediction.phase is example.phase)
        )
    total = len(examples)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        mean_accuracy = sum(1 for _, hit in bucket if hit) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_confidence - mean_accuracy)
    return ece


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieval_family_top1(
    embeddings: tuple[tuple[float, ...], ...], families: tuple[str, ...]
) -> float:
    """Leave-one-out nearest-neighbour family agreement."""
    if len(embeddings) != len(families):
        raise ValueError("embedding/family count mismatch")
    if len(embeddings) < 2:
        raise ValueError("retrieval needs at least 2 trajectories")
    hits = 0
    for query_index, query in enumerate(embeddings):
        best_index = max(
            (index for index in range(len(embeddings)) if index != query_index),
            key=lambda index: _cosine(query, embeddings[index]),
        )
        if families[best_index] == families[query_index]:
            hits += 1
    return hits / len(embeddings)


# ------------------------------------------------------- encoder predictor


def predict_with_encoder(
    checkpoint_path: pathlib.Path | str,
    examples: tuple[AnchorExample, ...],
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[StatePrediction, ...]:
    import torch

    from companion_encoder.model import load_checkpoint

    model = load_checkpoint(checkpoint_path, device=device)
    predictions: list[StatePrediction] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            outputs = model([example.text for example in batch])
            probabilities = torch.softmax(outputs["phase_logits"], dim=-1)
            confidences, phase_indices = probabilities.max(dim=-1)
            regressions = outputs["regressions"]
            for row in range(len(batch)):
                predictions.append(
                    StatePrediction(
                        phase=PHASE_VOCAB[int(phase_indices[row])],
                        trust_level=float(regressions[row][0]),
                        continuity_level=float(regressions[row][1]),
                        repair_pressure=float(regressions[row][2]),
                        confidence=float(confidences[row]),
                    )
                )
    return tuple(predictions)


def embed_trajectories_with_encoder(
    checkpoint_path: pathlib.Path | str,
    texts: tuple[str, ...],
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[tuple[float, ...], ...]:
    import torch

    from companion_encoder.model import load_checkpoint

    model = load_checkpoint(checkpoint_path, device=device)
    embeddings: list[tuple[float, ...]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            for row in model(batch)["embedding"]:
                embeddings.append(tuple(float(value) for value in row))
    return tuple(embeddings)


# ----------------------------------------------------------------- report


def score_predictions(
    predictions: tuple[StatePrediction, ...], examples: tuple[AnchorExample, ...]
) -> dict:
    return {
        **phase_metrics(predictions, examples),
        **regression_metrics(predictions, examples),
        "ece": expected_calibration_error(predictions, examples),
    }


def build_g2_report(
    *,
    splits: DatasetSplits,
    checkpoint_path: pathlib.Path | str,
    zero_shot_predictions: tuple[StatePrediction, ...] | None = None,
    device: str = "cpu",
) -> dict:
    """Score encoder + baselines on the val split; return the report dict."""
    val_examples = splits.val
    majority = MajorityBaseline.fit(splits.train)

    columns: dict[str, dict] = {
        "encoder": score_predictions(
            predict_with_encoder(checkpoint_path, val_examples, device=device),
            val_examples,
        ),
        "majority_baseline": score_predictions(
            majority.predict(val_examples), val_examples
        ),
    }
    if zero_shot_predictions is not None:
        columns["llm_zero_shot_baseline"] = score_predictions(
            zero_shot_predictions, val_examples
        )

    val_texts = tuple(render_full(t) for t in splits.val_trajectories)
    val_families = tuple(t.family for t in splits.val_trajectories)
    retrieval = {
        "encoder": retrieval_family_top1(
            embed_trajectories_with_encoder(checkpoint_path, val_texts, device=device),
            val_families,
        ),
        "stub_embedding_baseline": retrieval_family_top1(
            tuple(stub_semantic_embedding(text, dim=64) for text in val_texts),
            val_families,
        ),
    }

    return {
        "val_example_count": len(val_examples),
        "val_trajectory_count": len(splits.val_trajectories),
        "structured_prediction": columns,
        "retrieval_family_top1": retrieval,
        "g2_note": (
            "G2 passes only if 'encoder' significantly beats "
            "'majority_baseline' AND 'llm_zero_shot_baseline' on phase "
            "macro_f1 and mean_mae; see the public RFC release gates."
        ),
    }


def write_report(report: dict, path: pathlib.Path | str) -> pathlib.Path:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


__all__ = [
    "build_g2_report",
    "embed_trajectories_with_encoder",
    "expected_calibration_error",
    "phase_metrics",
    "predict_with_encoder",
    "regression_metrics",
    "retrieval_family_top1",
    "score_predictions",
    "write_report",
]
