#!/usr/bin/env python3
"""Evaluate one structured residual reader on frozen cross-view development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

TASK_ROOT = Path(__file__).resolve().parents[2]
READER_SCHEMA_PATH = TASK_ROOT / "assets" / "harness" / "reader.schema.json"
CORPUS_PATH = (
    TASK_ROOT / "assets" / "dataset_metadata" / "development_corpus.json"
)

EVALUATOR_VERSION = "volvence-readout-cross-view-causal-validity.v1"
READER_VERSION = "volvence-readout-reader.v1"
LABEL_TO_INT = {"agency_displacement": 0, "belonging_erasure": 1}
ANCHOR_VIEW = "original_en"
MODE_GROUP_KEY = {
    "preliminary": "preliminary_group_ids",
    "complete": "complete_group_ids",
}
MODE_VIEW_KEY = {
    "preliminary": "preliminary_views",
    "complete": "complete_views",
}


class EvaluationError(RuntimeError):
    """Raised when candidate, corpus, or local model integrity is invalid."""


@dataclass(frozen=True, slots=True)
class Sample:
    group_id: str
    split: str
    label: str
    view: str
    text: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.group_id, self.view)

    @property
    def target(self) -> int:
        return LABEL_TO_INT[self.label]


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    device: Any
    torch: Any
    snapshot_path: Path
    target_token_ids: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReaderFit:
    direction: np.ndarray
    intercept: float
    calibrator: LogisticRegression
    training_scores: np.ndarray


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"missing {context}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {context} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{context} must contain a JSON object: {path}")
    return payload


def _load_reader(variant_dir: Path) -> dict[str, Any]:
    if variant_dir.is_symlink() or not variant_dir.is_dir():
        raise EvaluationError(f"variant path must be a real directory: {variant_dir}")
    children = sorted(variant_dir.iterdir(), key=lambda item: item.name)
    if any(child.is_symlink() for child in children):
        raise EvaluationError("candidate directory may not contain symlinks")
    unexpected = [child.name for child in children if child.name != "reader.json"]
    if unexpected:
        raise EvaluationError(
            f"candidate directory may contain only reader.json; found {unexpected}"
        )
    reader = _load_json(variant_dir / "reader.json", context="candidate reader")
    schema = _load_json(READER_SCHEMA_PATH, context="reader schema")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(reader),
        key=lambda item: list(item.path),
    )
    if errors:
        details = "; ".join(error.message for error in errors[:5])
        raise EvaluationError(f"candidate reader violates reader.schema.json: {details}")
    return reader


def _load_corpus() -> tuple[dict[str, Any], tuple[Sample, ...]]:
    corpus = _load_json(CORPUS_PATH, context="development corpus")
    if corpus.get("schema_version") != "readout-cross-view-development-corpus.v1":
        raise EvaluationError("unsupported development corpus schema_version")
    labels = [item.get("id") for item in corpus.get("labels", [])]
    if labels != list(LABEL_TO_INT):
        raise EvaluationError("development corpus labels or ordering changed")
    views = corpus.get("views")
    if not isinstance(views, list) or views[0] != ANCHOR_VIEW or len(views) < 4:
        raise EvaluationError("development corpus must freeze an anchor plus cross views")
    samples: list[Sample] = []
    group_ids: set[str] = set()
    for group in corpus.get("groups", []):
        if not isinstance(group, dict):
            raise EvaluationError("development corpus group must be an object")
        group_id = str(group.get("group_id", ""))
        split = str(group.get("split", ""))
        label = str(group.get("label", ""))
        group_views = group.get("views")
        if not group_id or group_id in group_ids:
            raise EvaluationError(f"duplicate or empty development group_id: {group_id!r}")
        if split not in {"train", "evaluation"} or label not in LABEL_TO_INT:
            raise EvaluationError(f"invalid split or label in group {group_id}")
        if not isinstance(group_views, dict) or list(group_views) != views:
            raise EvaluationError(f"group {group_id} does not provide every frozen view")
        group_ids.add(group_id)
        for view, text in group_views.items():
            if not isinstance(text, str) or not text.strip():
                raise EvaluationError(f"group {group_id} view {view} has no text")
            lowered = text.casefold()
            if "agency_displacement" in lowered or "belonging_erasure" in lowered:
                raise EvaluationError(f"group {group_id} leaks a protocol label")
            samples.append(
                Sample(
                    group_id=group_id,
                    split=split,
                    label=label,
                    view=str(view),
                    text=text.strip(),
                )
            )
    protocol = corpus["protocol"]
    declared_complete = set(protocol["complete_group_ids"])
    observed_evaluation = {
        sample.group_id for sample in samples if sample.split == "evaluation"
    }
    if declared_complete != observed_evaluation:
        raise EvaluationError("complete group ids do not equal the evaluation split")
    train_counts = {
        label: len(
            {
                sample.group_id
                for sample in samples
                if sample.split == "train" and sample.label == label
            }
        )
        for label in LABEL_TO_INT
    }
    eval_counts = {
        label: len(
            {
                sample.group_id
                for sample in samples
                if sample.split == "evaluation" and sample.label == label
            }
        )
        for label in LABEL_TO_INT
    }
    if len(set(train_counts.values())) != 1 or len(set(eval_counts.values())) != 1:
        raise EvaluationError("development corpus splits must remain label-balanced")
    return corpus, tuple(samples)


def _prompt(sample: Sample, corpus: dict[str, Any]) -> str:
    definitions = {item["id"]: item["definition"] for item in corpus["labels"]}
    return (
        "Classify the relationship condition in the scenario. Use only one letter.\n"
        f"A: {definitions['agency_displacement']}.\n"
        f"B: {definitions['belonging_erasure']}.\n"
        f"Scenario: {sample.text}\n"
        "Answer:"
    )


def _selected_samples(
    *, corpus: dict[str, Any], samples: tuple[Sample, ...], mode: str
) -> tuple[tuple[Sample, ...], tuple[Sample, ...], tuple[Sample, ...]]:
    protocol = corpus["protocol"]
    train = tuple(sample for sample in samples if sample.split == "train")
    selected_groups = set(protocol[MODE_GROUP_KEY[mode]])
    selected_views = set(protocol[MODE_VIEW_KEY[mode]])
    evaluation = tuple(
        sample
        for sample in samples
        if sample.group_id in selected_groups and sample.view in selected_views
    )
    capture = tuple((*train, *evaluation))
    if not evaluation or {sample.target for sample in evaluation} != {0, 1}:
        raise EvaluationError(f"{mode} evaluation slice is not label-balanced")
    return train, evaluation, capture


def _resolve_model_snapshot(corpus: dict[str, Any]) -> Path:
    override = os.environ.get("PRAXIST_READOUT_MODEL_ROOT", "").strip()
    if override:
        snapshot = Path(override).expanduser().resolve(strict=True)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise EvaluationError("huggingface_hub is required to resolve the frozen model") from exc
        model_spec = corpus["model"]
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=model_spec["model_id"],
                    revision=model_spec["revision"],
                    local_files_only=True,
                )
            ).resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise EvaluationError(
                "the exact frozen Qwen snapshot is not available in the local cache"
            ) from exc
    if not snapshot.is_dir():
        raise EvaluationError(f"frozen model snapshot is not a directory: {snapshot}")
    for filename, expected in corpus["model"]["files"].items():
        path = snapshot / filename
        if not path.is_file():
            raise EvaluationError(f"frozen model file is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise EvaluationError(
                f"frozen model file digest changed for {filename}: {actual}"
            )
    return snapshot


def _load_model_bundle(corpus: dict[str, Any]) -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise EvaluationError("torch and transformers are required by this evaluator") from exc
    snapshot = _resolve_model_snapshot(corpus)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot load frozen model snapshot {snapshot}: {exc}") from exc
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    if int(model.config.num_hidden_layers) != int(corpus["model"]["num_hidden_layers"]):
        raise EvaluationError("frozen model layer count changed")
    if int(model.config.hidden_size) != int(corpus["model"]["hidden_size"]):
        raise EvaluationError("frozen model hidden size changed")
    target_ids: dict[str, int] = {}
    for label, token_text in corpus["protocol"]["target_token_text"].items():
        encoded = tokenizer.encode(token_text, add_special_tokens=False)
        expected = int(corpus["protocol"]["target_token_ids"][label])
        if encoded != [expected]:
            raise EvaluationError(
                f"target token {token_text!r} no longer has exact id {expected}: {encoded}"
            )
        target_ids[label] = expected
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch=torch,
        snapshot_path=snapshot,
        target_token_ids=target_ids,
    )


def _pool_hidden(hidden: Any, attention_mask: Any, pooling: str, torch: Any) -> Any:
    if pooling == "last_token":
        return hidden[:, -1, :]
    mask = attention_mask.to(hidden.dtype)
    if pooling == "mean_all":
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
    if pooling == "mean_last_4":
        rows = []
        for row_index in range(hidden.shape[0]):
            valid = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
            rows.append(hidden[row_index, valid[-4:], :].mean(dim=0))
        return torch.stack(rows, dim=0)
    raise EvaluationError(f"unsupported pooling: {pooling}")


def _capture(
    *,
    bundle: ModelBundle,
    corpus: dict[str, Any],
    reader: dict[str, Any],
    samples: tuple[Sample, ...],
    batch_size: int = 8,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    vectors: dict[tuple[str, str], np.ndarray] = {}
    margins: dict[tuple[str, str], float] = {}
    layer_index = int(reader["layer"]) + 1
    torch = bundle.torch
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        prompts = [_prompt(sample, corpus) for sample in batch]
        encoded = bundle.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
        )
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = bundle.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or layer_index >= len(hidden_states):
            raise EvaluationError(f"model did not expose hidden state {layer_index}")
        pooled = _pool_hidden(
            hidden_states[layer_index],
            encoded["attention_mask"],
            str(reader["pooling"]),
            torch,
        )
        logits = output.logits[:, -1, :]
        for index, sample in enumerate(batch):
            vectors[sample.key] = pooled[index].detach().float().cpu().numpy()
            a_id = bundle.target_token_ids["agency_displacement"]
            b_id = bundle.target_token_ids["belonging_erasure"]
            margins[sample.key] = float((logits[index, b_id] - logits[index, a_id]).item())
    return vectors, margins


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise EvaluationError("reader direction has zero or non-finite norm")
    return np.asarray(vector, dtype=np.float64) / norm


def _view_direction(
    samples: tuple[Sample, ...],
    vectors: dict[tuple[str, str], np.ndarray],
    view: str,
) -> np.ndarray:
    rows = [sample for sample in samples if sample.view == view]
    by_target = {
        target: np.stack([vectors[sample.key] for sample in rows if sample.target == target])
        for target in (0, 1)
    }
    if any(values.size == 0 for values in by_target.values()):
        raise EvaluationError(f"training view {view} lacks one label")
    return by_target[1].mean(axis=0) - by_target[0].mean(axis=0)


def _training_rows(
    *,
    train: tuple[Sample, ...],
    aggregation: str,
) -> tuple[Sample, ...]:
    if aggregation == "original_only":
        return tuple(sample for sample in train if sample.view == ANCHOR_VIEW)
    return train


def _fit_reader(
    *,
    bundle: ModelBundle,
    corpus: dict[str, Any],
    reader: dict[str, Any],
    train: tuple[Sample, ...],
    vectors: dict[tuple[str, str], np.ndarray],
) -> ReaderFit:
    family = str(reader["reader_family"])
    aggregation = str(reader["view_aggregation"])
    rows = _training_rows(train=train, aggregation=aggregation)
    matrix = np.stack([vectors[sample.key] for sample in rows]).astype(np.float64)
    targets = np.asarray([sample.target for sample in rows], dtype=np.int64)
    regularization = float(reader["regularization"])
    seed = int(reader["seed"])
    if family == "linear_probe":
        classifier = LogisticRegression(
            C=regularization,
            solver="liblinear",
            random_state=seed,
            max_iter=1000,
        )
        classifier.fit(matrix, targets)
        direction = np.asarray(classifier.coef_[0], dtype=np.float64)
        intercept = float(classifier.intercept_[0])
    elif family in {"diff_of_means", "rsa_aligned_centroid"}:
        if family == "diff_of_means" and aggregation == "original_only":
            direction = _view_direction(train, vectors, ANCHOR_VIEW)
        else:
            original = _unit(_view_direction(train, vectors, ANCHOR_VIEW))
            directions = []
            weights = []
            for view in corpus["views"]:
                candidate = _unit(_view_direction(train, vectors, str(view)))
                if float(np.dot(candidate, original)) < 0.0:
                    candidate = -candidate
                coherence = max(0.0, float(np.dot(candidate, original)))
                directions.append(candidate)
                if aggregation == "coherence_weighted" or family == "rsa_aligned_centroid":
                    weights.append(max(coherence, 1e-6))
                else:
                    weights.append(1.0)
            direction = np.average(np.stack(directions), axis=0, weights=weights)
        direction = _unit(direction)
        anchor_rows = tuple(sample for sample in train if sample.view == ANCHOR_VIEW)
        anchor_scores = np.asarray(
            [float(np.dot(vectors[sample.key], direction)) for sample in anchor_rows]
        )
        anchor_targets = np.asarray([sample.target for sample in anchor_rows])
        midpoint = 0.5 * (
            anchor_scores[anchor_targets == 0].mean()
            + anchor_scores[anchor_targets == 1].mean()
        )
        intercept = -float(midpoint)
    elif family == "j_lens_like":
        weight = bundle.model.get_output_embeddings().weight.detach().float().cpu().numpy()
        a_id = bundle.target_token_ids["agency_displacement"]
        b_id = bundle.target_token_ids["belonging_erasure"]
        direction = _unit(weight[b_id] - weight[a_id])
        anchor_rows = tuple(sample for sample in train if sample.view == ANCHOR_VIEW)
        anchor_scores = np.asarray(
            [float(np.dot(vectors[sample.key], direction)) for sample in anchor_rows]
        )
        anchor_targets = np.asarray([sample.target for sample in anchor_rows])
        midpoint = 0.5 * (
            anchor_scores[anchor_targets == 0].mean()
            + anchor_scores[anchor_targets == 1].mean()
        )
        intercept = -float(midpoint)
    elif family == "random_control":
        generator = np.random.default_rng(seed)
        direction = _unit(generator.standard_normal(matrix.shape[1]))
        observed = _view_direction(train, vectors, ANCHOR_VIEW)
        if float(np.dot(direction, observed)) < 0.0:
            direction = -direction
        anchor_rows = tuple(sample for sample in train if sample.view == ANCHOR_VIEW)
        anchor_scores = np.asarray(
            [float(np.dot(vectors[sample.key], direction)) for sample in anchor_rows]
        )
        anchor_targets = np.asarray([sample.target for sample in anchor_rows])
        midpoint = 0.5 * (
            anchor_scores[anchor_targets == 0].mean()
            + anchor_scores[anchor_targets == 1].mean()
        )
        intercept = -float(midpoint)
    else:  # pragma: no cover - JSON schema owns the family set.
        raise EvaluationError(f"unsupported reader family: {family}")
    direction = _unit(direction)
    training_scores = matrix @ direction + intercept
    calibrator = LogisticRegression(
        C=regularization,
        solver="liblinear",
        random_state=seed,
        max_iter=1000,
    )
    calibrator.fit(training_scores.reshape(-1, 1), targets)
    return ReaderFit(
        direction=direction,
        intercept=intercept,
        calibrator=calibrator,
        training_scores=training_scores,
    )


def _probabilities(fit: ReaderFit, scores: np.ndarray) -> np.ndarray:
    return fit.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def _balanced_accuracy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return float(balanced_accuracy_score(targets, predictions))


def _cohen_d(scores: np.ndarray, targets: np.ndarray) -> float:
    negative = scores[targets == 0]
    positive = scores[targets == 1]
    if len(negative) < 2 or len(positive) < 2:
        return 0.0
    denominator = len(negative) + len(positive) - 2
    pooled_variance = (
        (len(negative) - 1) * negative.var(ddof=1)
        + (len(positive) - 1) * positive.var(ddof=1)
    ) / denominator
    if pooled_variance <= 1e-18:
        return 0.0
    return float((positive.mean() - negative.mean()) / math.sqrt(pooled_variance))


def _direction_coherence(
    *,
    train: tuple[Sample, ...],
    vectors: dict[tuple[str, str], np.ndarray],
    views: list[str],
    direction: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    values = {
        str(view): float(np.dot(_unit(_view_direction(train, vectors, str(view))), direction))
        for view in views
    }
    return float(np.mean(list(values.values()))), min(values.values()), values


def _identity_retrieval(
    *,
    evaluation: tuple[Sample, ...],
    vectors: dict[tuple[str, str], np.ndarray],
    train: tuple[Sample, ...],
) -> float:
    anchor_rows = [sample for sample in evaluation if sample.view == ANCHOR_VIEW]
    queries = [sample for sample in evaluation if sample.view != ANCHOR_VIEW]
    if not anchor_rows or not queries:
        return 0.0
    center = np.stack([vectors[sample.key] for sample in train]).mean(axis=0)
    anchors = np.stack([_unit(vectors[sample.key] - center) for sample in anchor_rows])
    correct = 0
    for sample in queries:
        query = _unit(vectors[sample.key] - center)
        nearest = int(np.argmax(anchors @ query))
        correct += anchor_rows[nearest].group_id == sample.group_id
    return correct / len(queries)


def _patched_margin_effects(
    *,
    bundle: ModelBundle,
    corpus: dict[str, Any],
    reader: dict[str, Any],
    evaluation: tuple[Sample, ...],
    baseline_margins: dict[tuple[str, str], float],
    direction: np.ndarray,
    activation_scale: float,
    batch_size: int = 8,
) -> np.ndarray:
    model = bundle.model
    torch = bundle.torch
    layer = model.model.layers[int(reader["layer"])]
    direction_tensor = torch.tensor(
        direction,
        dtype=next(model.parameters()).dtype,
        device=bundle.device,
    )
    effects: list[float] = []
    for offset in range(0, len(evaluation), batch_size):
        batch = evaluation[offset : offset + batch_size]
        encoded = bundle.tokenizer(
            [_prompt(sample, corpus) for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
        )
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
        signs = torch.tensor(
            [1.0 if sample.target == 1 else -1.0 for sample in batch],
            dtype=direction_tensor.dtype,
            device=bundle.device,
        )
        delta = (
            signs[:, None]
            * float(reader["dose"])
            * activation_scale
            * direction_tensor[None, :]
        )

        def patch_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            patch_delta: Any = delta,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            patched[:, -1, :] = patched[:, -1, :] + patch_delta
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handle = layer.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                output = model(**encoded, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        logits = output.logits[:, -1, :]
        a_id = bundle.target_token_ids["agency_displacement"]
        b_id = bundle.target_token_ids["belonging_erasure"]
        for index, sample in enumerate(batch):
            patched_b_minus_a = float((logits[index, b_id] - logits[index, a_id]).item())
            baseline_b_minus_a = baseline_margins[sample.key]
            target_sign = 1.0 if sample.target == 1 else -1.0
            effects.append(target_sign * (patched_b_minus_a - baseline_b_minus_a))
    return np.asarray(effects, dtype=np.float64)


def _metric_margins(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, float]:
    return {
        "same_view_balanced_accuracy": (
            metrics["same_view_balanced_accuracy"]
            - thresholds["same_view_balanced_accuracy"]
        )
        / 0.25,
        "cross_view_balanced_accuracy": (
            metrics["cross_view_balanced_accuracy"]
            - thresholds["cross_view_balanced_accuracy"]
        )
        / 0.35,
        "worst_view_balanced_accuracy": (
            metrics["worst_view_balanced_accuracy"]
            - thresholds["worst_view_balanced_accuracy"]
        )
        / 0.5,
        "brier_score": (
            thresholds["brier_score_max"] - metrics["brier_score"]
        )
        / thresholds["brier_score_max"],
        "heldout_cohen_d": (
            metrics["heldout_cohen_d"] - thresholds["heldout_cohen_d"]
        )
        / thresholds["heldout_cohen_d"],
        "cross_view_identity_retrieval": (
            metrics["cross_view_identity_retrieval"]
            - thresholds["cross_view_identity_retrieval"]
        )
        / thresholds["cross_view_identity_retrieval"],
        "mean_direction_coherence": (
            metrics["mean_direction_coherence"]
            - thresholds["mean_direction_coherence"]
        )
        / 0.25,
        "causal_target_margin_effect": (
            metrics["causal_target_margin_effect"]
            - thresholds["causal_target_margin_effect"]
        )
        / thresholds["causal_target_margin_effect"],
        "random_control_separation": (
            metrics["random_control_separation"]
            - thresholds["random_control_separation"]
        )
        / thresholds["random_control_separation"],
    }


def _measure(
    *,
    bundle: ModelBundle,
    corpus: dict[str, Any],
    reader: dict[str, Any],
    train: tuple[Sample, ...],
    evaluation: tuple[Sample, ...],
    vectors: dict[tuple[str, str], np.ndarray],
    baseline_margins: dict[tuple[str, str], float],
    elapsed_before_patch: float,
) -> dict[str, Any]:
    fit = _fit_reader(
        bundle=bundle,
        corpus=corpus,
        reader=reader,
        train=train,
        vectors=vectors,
    )
    rows = list(evaluation)
    matrix = np.stack([vectors[sample.key] for sample in rows]).astype(np.float64)
    targets = np.asarray([sample.target for sample in rows], dtype=np.int64)
    scores = matrix @ fit.direction + fit.intercept
    probabilities = _probabilities(fit, scores)
    same_indices = np.asarray([sample.view == ANCHOR_VIEW for sample in rows])
    cross_indices = ~same_indices
    per_view = {}
    for view in corpus["views"]:
        indices = np.asarray([sample.view == view for sample in rows])
        if indices.any():
            per_view[str(view)] = _balanced_accuracy(
                targets[indices], probabilities[indices]
            )
    cross_view_values = [
        value for view, value in per_view.items() if view != ANCHOR_VIEW
    ]
    mean_coherence, worst_coherence, coherence_by_view = _direction_coherence(
        train=train,
        vectors=vectors,
        views=[str(view) for view in corpus["views"]],
        direction=fit.direction,
    )
    train_anchor = [sample for sample in train if sample.view == ANCHOR_VIEW]
    train_anchor_matrix = np.stack([vectors[sample.key] for sample in train_anchor])
    residual_norm = float(np.median(np.linalg.norm(train_anchor_matrix, axis=1)))
    activation_scale = residual_norm / math.sqrt(train_anchor_matrix.shape[1])
    candidate_effects = _patched_margin_effects(
        bundle=bundle,
        corpus=corpus,
        reader=reader,
        evaluation=evaluation,
        baseline_margins=baseline_margins,
        direction=fit.direction,
        activation_scale=activation_scale,
    )
    generator = np.random.default_rng(int(reader["seed"]) + 104729)
    random_direction = _unit(generator.standard_normal(fit.direction.shape[0]))
    random_effects = _patched_margin_effects(
        bundle=bundle,
        corpus=corpus,
        reader=reader,
        evaluation=evaluation,
        baseline_margins=baseline_margins,
        direction=random_direction,
        activation_scale=activation_scale,
    )
    metrics: dict[str, Any] = {
        "same_view_balanced_accuracy": _balanced_accuracy(
            targets[same_indices], probabilities[same_indices]
        ),
        "cross_view_balanced_accuracy": _balanced_accuracy(
            targets[cross_indices], probabilities[cross_indices]
        ),
        "worst_view_balanced_accuracy": min(cross_view_values),
        "per_view_balanced_accuracy": per_view,
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "heldout_cohen_d": _cohen_d(scores[same_indices], targets[same_indices]),
        "cross_view_identity_retrieval": _identity_retrieval(
            evaluation=evaluation,
            vectors=vectors,
            train=train,
        ),
        "mean_direction_coherence": mean_coherence,
        "worst_direction_coherence": worst_coherence,
        "direction_coherence_by_view": coherence_by_view,
        "causal_target_margin_effect": float(candidate_effects.mean()),
        "causal_target_margin_effect_std": float(candidate_effects.std()),
        "causal_reverse_effect_rate": float(np.mean(candidate_effects <= 0.0)),
        "random_control_margin_effect": float(random_effects.mean()),
        "random_control_separation": float(
            candidate_effects.mean() - random_effects.mean()
        ),
        "activation_scale": activation_scale,
        "intervention_delta_norm": float(reader["dose"]) * activation_scale,
        "median_residual_norm": residual_norm,
        "intervention_to_residual_norm_ratio": (
            float(reader["dose"]) * activation_scale / residual_norm
        ),
        "fit_examples": len(_training_rows(
            train=train,
            aggregation=str(reader["view_aggregation"]),
        )),
        "evaluation_examples": len(evaluation),
        "protocol_integrity_passed": True,
        "protocol_integrity_failed": False,
        "suspect_protocol": False,
        "suspect_leakage": False,
        "late_after_generation_boundary": False,
        "evaluator_wall_seconds_before_patch": elapsed_before_patch,
    }
    margins = _metric_margins(metrics, corpus["thresholds"])
    metrics["qualification_gate_margins"] = margins
    metrics["qualification_margin"] = min(margins.values())
    metrics["qualification_passed"] = all(value >= 0.0 for value in margins.values())
    same_instrument_valid = (
        metrics["same_view_balanced_accuracy"] >= 0.65
        and metrics["heldout_cohen_d"] >= 0.5
        and metrics["causal_target_margin_effect"]
        > metrics["random_control_margin_effect"] + 0.005
        and reader["reader_family"] != "random_control"
    )
    cross_valid = (
        metrics["cross_view_balanced_accuracy"]
        >= corpus["thresholds"]["cross_view_balanced_accuracy"]
        and metrics["worst_view_balanced_accuracy"]
        >= corpus["thresholds"]["worst_view_balanced_accuracy"]
    )
    if metrics["qualification_passed"] and reader["reader_family"] != "random_control":
        exit_classification = "PASS"
    elif same_instrument_valid and not cross_valid:
        exit_classification = "DOMAIN_LOCAL"
    else:
        exit_classification = "INSTRUMENT_INVALID"
    metrics["exit_classification"] = exit_classification
    metrics["domain_local"] = exit_classification == "DOMAIN_LOCAL"
    metrics["instrument_invalid"] = exit_classification == "INSTRUMENT_INVALID"
    metrics["random_control"] = reader["reader_family"] == "random_control"
    return metrics


def _producer_identity() -> dict[str, Any]:
    result: dict[str, Any] = {}
    generation = os.environ.get("PRAXIST_GENERATION_ID", "").strip()
    peer = os.environ.get("PRAXIST_PEER_ID", "").strip()
    if generation:
        result["generation_id"] = int(generation)
    if peer:
        result["peer_id"] = peer
    return result


def _design_dimensions(reader: dict[str, Any]) -> dict[str, str]:
    return {
        "mechanism_family": str(reader["reader_family"]),
        "intervention_surface": f"residual_layer_{reader['layer']}",
        "intent": "falsify" if reader["reader_family"] == "random_control" else "explore",
        "semantic_family": f"{reader['pooling']}_{reader['view_aggregation']}",
        "parent_lineage": "frozen_qwen25_05b_baseline",
        "novelty_axis": f"dose_{reader['dose']}",
    }


def _failure_modes(metrics: dict[str, Any]) -> list[str]:
    return [
        name
        for name, value in metrics["qualification_gate_margins"].items()
        if value < 0.0
    ]


def _build_summary(
    *,
    mode: str,
    reader: dict[str, Any],
    corpus: dict[str, Any],
    metrics: dict[str, Any],
    corpus_sha256: str,
    elapsed: float,
    replication_of_effective_config_sha256: str = "",
) -> dict[str, Any]:
    complete = mode == "complete"
    reference_units = len(corpus["protocol"]["complete_group_ids"])
    actual_units = len(corpus["protocol"][MODE_GROUP_KEY[mode]])
    effort_ratio = actual_units / reference_units
    metrics = {**metrics, "evaluator_wall_seconds": elapsed}
    metrics["scored_complete"] = complete
    metrics["is_smoke_eval"] = not complete
    metrics["partial"] = not complete
    metrics["scout_only"] = not complete
    metrics["validation_only"] = not complete
    metrics["validation_only_result"] = not complete
    metrics["promotion_eligible"] = (
        complete
        and metrics["exit_classification"] == "PASS"
        and reader["reader_family"] != "random_control"
    )
    effective_config = {
        "evaluator_version": EVALUATOR_VERSION,
        "mode": mode,
        "reader": reader,
        "corpus_sha256": corpus_sha256,
        "model": corpus["model"],
        "selected_group_ids": corpus["protocol"][MODE_GROUP_KEY[mode]],
        "selected_views": corpus["protocol"][MODE_VIEW_KEY[mode]],
        "reference_units": reference_units,
    }
    effective_digest = _digest(effective_config)
    if replication_of_effective_config_sha256:
        replication_status = (
            "matched"
            if replication_of_effective_config_sha256 == effective_digest
            else "mismatched"
        )
    else:
        replication_status = "not_requested"
    if reader["reader_family"] == "random_control":
        source_lane = "negative_control"
    elif complete:
        source_lane = "performance"
    else:
        source_lane = "task_candidate"
    failures = _failure_modes(metrics)
    if complete and metrics["promotion_eligible"]:
        valence = "positive"
    elif complete and metrics["exit_classification"] == "DOMAIN_LOCAL":
        valence = "mixed"
    elif complete:
        valence = "negative"
    else:
        valence = "neutral"
    producer = _producer_identity()
    return {
        "schema_version": 1,
        "protocol": EVALUATOR_VERSION,
        "protocol_version": 1,
        "variant_id": reader["variant_id"],
        "variant_name": reader["variant_id"],
        **producer,
        "score": metrics["qualification_margin"],
        "metrics": metrics,
        "frontier_lane": source_lane,
        "promotion_lane": source_lane,
        "evidence_stage": mode,
        "eval_stage": mode,
        "tier": mode,
        "result_status": "scored_complete" if complete else "preliminary",
        "scored_complete": complete,
        "complete_eval": complete,
        "promotion_eligible": metrics["promotion_eligible"],
        "parent_authorized": complete and reader["reader_family"] != "random_control",
        "close_eligible": complete,
        "effort_ratio": effort_ratio,
        "coverage_ratio": effort_ratio,
        "actual_evaluation_units": actual_units,
        "reference_evaluation_units": reference_units,
        "evaluation_units_completed": actual_units,
        "evaluation_units_required": reference_units,
        "effective_config": effective_config,
        "effective_config_complete": True,
        "effective_config_digest": effective_digest,
        "effective_config_schema": "readout-cross-view-effective-config.v1",
        "replication_of_effective_config_sha256": (
            replication_of_effective_config_sha256
        ),
        "replication_effective_config_status": replication_status,
        "design_dimensions": _design_dimensions(reader),
        "changed_modules": ["reader.json"],
        "method_class": "bounded_residual_reader",
        "protocol_integrity": {
            "passed": True,
            "frozen_assets_attested": True,
            "outcome_dependent_selection": False,
            "candidate_contains_executable_code": False,
            "development_only": True,
            "formal_validation_performed": False,
        },
        "dataset": {
            "visibility": corpus["visibility"],
            "corpus_sha256": corpus_sha256,
            "selected_group_ids": corpus["protocol"][MODE_GROUP_KEY[mode]],
            "selected_views": corpus["protocol"][MODE_VIEW_KEY[mode]],
            "formal_validation_overlap": False,
        },
        "extra": {
            "frontier_lane": source_lane,
            "promotion_lane": source_lane,
            "evidence_stage": mode,
            "protocol_name": EVALUATOR_VERSION,
            "effort_ratio": effort_ratio,
            "coverage_ratio": effort_ratio,
            "completed_required_eval_units": actual_units,
            "complete_protocol_evaluation_units": reference_units,
            "is_negative": valence == "negative",
            "evidence_valence": valence,
            "failure_mode": ",".join(failures),
            "disconfirming_claim_ids": ["claim:readout-cross-view-causal-validity"]
            if failures
            else [],
            "next_step_intent": (
                "loop_external_formal_validation"
                if metrics["promotion_eligible"]
                else "repair_or_falsify_instrument"
            ),
            "development_evaluation_is_not_learning_source": True,
            "formal_validation_performed": False,
            "production_promotion_authorized": False,
            "design_dimensions": _design_dimensions(reader),
            "effective_config_digest": effective_digest,
            **producer,
        },
    }


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "evaluation_summary.json"
    temporary = output_dir / ".evaluation_summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def evaluate(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    variant_dir = args.variant_dir.expanduser().resolve(strict=True)
    reader = _load_reader(variant_dir)
    corpus, samples = _load_corpus()
    train, evaluation, capture = _selected_samples(
        corpus=corpus,
        samples=samples,
        mode=args.mode,
    )
    bundle = _load_model_bundle(corpus)
    vectors, baseline_margins = _capture(
        bundle=bundle,
        corpus=corpus,
        reader=reader,
        samples=capture,
    )
    before_patch = time.perf_counter() - started
    metrics = _measure(
        bundle=bundle,
        corpus=corpus,
        reader=reader,
        train=train,
        evaluation=evaluation,
        vectors=vectors,
        baseline_margins=baseline_margins,
        elapsed_before_patch=before_patch,
    )
    elapsed = time.perf_counter() - started
    summary = _build_summary(
        mode=args.mode,
        reader=reader,
        corpus=corpus,
        metrics=metrics,
        corpus_sha256=_sha256_file(CORPUS_PATH),
        elapsed=elapsed,
        replication_of_effective_config_sha256=(
            args.replication_of_effective_config_sha256
        ),
    )
    return _write_summary(args.output_dir.expanduser().resolve(), summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a structured residual reader on frozen public-development "
            "cross-view and causal-patch checks."
        )
    )
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_GROUP_KEY), default="preliminary")
    parser.add_argument(
        "--replication-of-effective-config-sha256",
        default="",
        help="Optional exact effective-config digest for a replication claim.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        summary_path = evaluate(args)
    except (EvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
