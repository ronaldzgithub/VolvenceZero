"""Stage-2 step 3: linear classification probe + Gate 2 (base vs pretrained).

Gate 2 asks whether the LLM residual stream can carry the domain's behavioural
hierarchy at all. It decodes the *active subgoal* (next objective on the route)
from the final-prompt-position hidden state with a linear classifier, on
held-out routes, and compares the continued-pretrained base against the
original Qwen (a causal contrast). Three conditions must all hold:

1. pretrained held-out accuracy >= 2x uniform chance;
2. accuracy rises with observation prefix (late-step >= early-step);
3. pretrained accuracy > original-base accuracy.

Failing means "the residual cannot host the behaviour hierarchy": the plan
kills the LLM-transfer route here and no Stage-3 rerun is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from volvence_zero.agent.eta_proof_benchmark import (
    eta_route_probe_rows,
    generate_eta_proof_corpus,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V3,
    OBSERVATION_PROTOCOL_V4,
    eta_stage2_probe_rows,
)
from volvence_zero.substrate import (
    capture_prefix_diagnostics,
    fit_linear_classification_probe,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Legacy rows: single-step observations whose plan carrier is the corpus
# hash-fingerprint (2026-08-03 instrument audit: heldout information ceiling
# ~0.18 against the 0.25 gate bar, so the 2x-chance condition was unpassable
# by construction). Kept for byte-identical reproduction of sealed runs.
PROBE_PROTOCOL_LEGACY = "legacy-single-step.v1"
_PROBE_PROTOCOLS = (
    PROBE_PROTOCOL_LEGACY,
    OBSERVATION_PROTOCOL_V3,
    OBSERVATION_PROTOCOL_V4,
)

# Final-layer readout was the v1 preregistration; small-model final blocks are
# output-head aligned (the v1 run's base arm peaked mid-stack at layer 10),
# so v2 offers a train-split-selected fixed layer: the layer is chosen on a
# deterministic case-id split of the *train* rows only, never on heldout.
LAYER_SELECTION_FINAL = "final"
LAYER_SELECTION_TRAIN_SPLIT = "train-split"
_LAYER_SELECTIONS = (LAYER_SELECTION_FINAL, LAYER_SELECTION_TRAIN_SPLIT)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True, cwd=_REPO_ROOT
    )
    return result.stdout.strip()


def _cache_factory(pairs=None):
    if pairs is None:
        return transformers.DynamicCache()
    return transformers.DynamicCache(ddp_cache_data=list(pairs))


def _capture_layer_features(*, model, tokenizer, device, texts, max_length):
    """Return per-layer stacked final-position hidden features for the texts."""

    per_layer_rows: list[list[list[float]]] = []
    for text in texts:
        encoded = tokenizer(
            text, return_tensors="pt", truncation=False
        )["input_ids"]
        if encoded.shape[1] > max_length:
            raise ValueError(
                f"probe text has {encoded.shape[1]} tokens > max_length "
                f"{max_length}; truncating would cut the current step off the "
                "final position. Raise --max-length."
            )
        input_ids = encoded.to(device)
        profile = capture_prefix_diagnostics(
            torch_module=torch,
            model=model,
            input_ids=input_ids,
            prefix_pairs=None,
            cache_factory=_cache_factory,
            capture_hidden=True,
        )
        if not per_layer_rows:
            per_layer_rows = [[] for _ in profile.final_hidden]
        for layer_index, hidden in enumerate(profile.final_hidden):
            per_layer_rows[layer_index].append(list(hidden))
    return [torch.tensor(rows, dtype=torch.float32) for rows in per_layer_rows]


def _train_split_masks(train_rows):
    """Deterministic case-id split of the train rows into fit/val (~80/20).

    Grouping by case id keeps every view of one route on one side, mirroring
    the train/eval hygiene of the outer split. Never touches heldout rows.
    """

    fit_mask: list[bool] = []
    for row in train_rows:
        digest = hashlib.sha256(row.case_id.encode("utf-8")).digest()
        fit_mask.append(digest[0] % 5 != 0)
    if all(fit_mask) or not any(fit_mask):
        raise RuntimeError(
            "train-split layer selection produced an empty fit or val side."
        )
    return fit_mask


def _probe_arm(
    *,
    model_source,
    device,
    train_rows,
    eval_rows,
    class_count,
    layer_selection,
    max_length,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    train_texts = [row.observation_text for row in train_rows]
    eval_texts = [row.observation_text for row in eval_rows]
    train_labels = torch.tensor(
        [row.subgoal_label for row in train_rows], dtype=torch.long
    )
    eval_labels = torch.tensor(
        [row.subgoal_label for row in eval_rows], dtype=torch.long
    )
    train_feats = _capture_layer_features(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        texts=train_texts,
        max_length=max_length,
    )
    eval_feats = _capture_layer_features(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        texts=eval_texts,
        max_length=max_length,
    )
    del model

    per_layer = []
    for layer_index, (xtr, xev) in enumerate(zip(train_feats, eval_feats, strict=True)):
        fit = fit_linear_classification_probe(
            torch_module=torch,
            train_features=xtr.to(torch_device),
            train_labels=train_labels.to(torch_device),
            eval_features=xev.to(torch_device),
            eval_labels=eval_labels.to(torch_device),
            layer_index=layer_index,
            class_count=class_count,
            alpha=1.0,
        )
        per_layer.append(fit)

    if not per_layer:
        raise RuntimeError("probe captured no transformer layers.")
    # Layer selection is fixed before held-out scoring. Choosing the best
    # held-out layer would turn the Gate-2 readout into a validation-set
    # search and inflate the causal contrast.
    layer_validation_accuracies: list[float] | None = None
    if layer_selection == LAYER_SELECTION_FINAL:
        # v1 preregistration: the final transformer block for both arms.
        selected = per_layer[-1]
    else:
        # v2: pick the layer on a deterministic fit/val split of the *train*
        # rows only; heldout rows never inform the choice.
        fit_mask = _train_split_masks(train_rows)
        fit_idx = torch.tensor(
            [i for i, keep in enumerate(fit_mask) if keep], dtype=torch.long
        )
        val_idx = torch.tensor(
            [i for i, keep in enumerate(fit_mask) if not keep],
            dtype=torch.long,
        )
        layer_validation_accuracies = []
        for xtr in train_feats:
            xtr_device = xtr.to(torch_device)
            predictions = _predict_labels(
                xtr_device[fit_idx],
                train_labels.to(torch_device)[fit_idx],
                xtr_device[val_idx],
                class_count,
            )
            val_labels = train_labels.to(torch_device)[val_idx]
            accuracy = float(
                (predictions == val_labels).to(torch.float32).mean()
            )
            layer_validation_accuracies.append(round(accuracy, 6))
        best_layer = max(
            range(len(layer_validation_accuracies)),
            key=lambda index: (layer_validation_accuracies[index], -index),
        )
        selected = per_layer[best_layer]

    # Prefix monotonicity on the fixed selected layer: early vs late bucket.
    step_indices = sorted({row.step_index for row in eval_rows})
    median_step = step_indices[len(step_indices) // 2] if step_indices else 0
    selected_layer = selected.layer_index
    xev_selected = eval_feats[selected_layer].to(torch_device)
    predictions = _predict_labels(
        train_feats[selected_layer].to(torch_device),
        train_labels.to(torch_device),
        xev_selected,
        class_count,
    )
    early_correct = early_total = late_correct = late_total = 0
    for idx, row in enumerate(eval_rows):
        hit = int(predictions[idx]) == row.subgoal_label
        if row.step_index < median_step:
            early_total += 1
            early_correct += int(hit)
        else:
            late_total += 1
            late_correct += int(hit)
    early_acc = early_correct / early_total if early_total else 0.0
    late_acc = late_correct / late_total if late_total else 0.0
    return {
        "model_source": str(model_source),
        "layer_selection": layer_selection,
        "selected_layer": selected_layer,
        "selected_accuracy": round(selected.accuracy, 6),
        "chance_accuracy": round(selected.chance_accuracy, 6),
        "majority_accuracy": round(selected.majority_accuracy, 6),
        "support": selected.support,
        "early_prefix_accuracy": round(early_acc, 6),
        "late_prefix_accuracy": round(late_acc, 6),
        "layer_validation_accuracies": layer_validation_accuracies,
        "per_layer": [fit.as_json_dict() for fit in per_layer],
    }


def _predict_labels(train_x, train_y, eval_x, class_count):
    one_hot = torch.zeros(
        (int(train_y.shape[0]), class_count),
        dtype=train_x.dtype,
        device=train_x.device,
    )
    one_hot.scatter_(1, train_y.unsqueeze(1), 1.0)
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    centred = (train_x - x_mean) / x_scale
    width = centred.shape[1]
    y_mean = one_hot.mean(dim=0, keepdim=True)
    gram = centred.T @ centred + 1.0 * torch.eye(
        width, dtype=centred.dtype, device=centred.device
    )
    weights = torch.linalg.solve(gram, centred.T @ (one_hot - y_mean))
    scores = ((eval_x - x_mean) / x_scale) @ weights + y_mean
    return scores.argmax(dim=1)


def assess_gate2(*, base_arm, pretrained_arm) -> dict:
    """Pure Gate-2 decision from the two arms' probe readouts."""

    acc = pretrained_arm["selected_accuracy"]
    chance = pretrained_arm["chance_accuracy"]
    cond_2x = acc >= 2.0 * chance
    cond_prefix = (
        pretrained_arm["late_prefix_accuracy"]
        >= pretrained_arm["early_prefix_accuracy"]
    )
    cond_causal = acc > base_arm["selected_accuracy"]
    passed = cond_2x and cond_prefix and cond_causal
    return {
        "gate_id": "gate-2-residual-carries-subgoal",
        "passed": passed,
        "condition_2x_chance": cond_2x,
        "condition_rises_with_prefix": cond_prefix,
        "condition_beats_base": cond_causal,
        "pretrained_accuracy": acc,
        "base_accuracy": base_arm["selected_accuracy"],
        "chance_accuracy": chance,
        "verdict": (
            "gate-2-pass-proceed-to-stage-3"
            if passed
            else "gate-2-fail-kill-llm-transfer"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-2 linear classification probe + Gate 2."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-source", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument(
        "--pretrained-model-source",
        required=True,
        help="Path to the Stage-2 merged continued-pretrained model.",
    )
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--train-routes", type=int, default=120)
    parser.add_argument("--heldout-routes", type=int, default=60)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--probe-protocol",
        choices=_PROBE_PROTOCOLS,
        default=PROBE_PROTOCOL_LEGACY,
        help=(
            "Probe row construction. Legacy rows are single-step observations "
            "with the unreadable fingerprint plan carrier; observation-"
            "protocol rows are cumulative trajectory prefixes rendered by the "
            "rate-distortion protocol (v4 = staged revelation), under which "
            "the label is a deterministic function of the visible text."
        ),
    )
    parser.add_argument(
        "--layer-selection",
        choices=_LAYER_SELECTIONS,
        default=LAYER_SELECTION_FINAL,
        help=(
            "Readout layer rule: 'final' (v1 preregistration) or "
            "'train-split' (fixed layer chosen on a deterministic fit/val "
            "split of the train rows only)."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help=(
            "Fail-loud token budget per probe text; texts longer than this "
            "raise instead of being truncated."
        ),
    )
    args = parser.parse_args()

    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=args.objective_count,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
    )
    if args.probe_protocol == PROBE_PROTOCOL_LEGACY:
        train_rows, vocab = eta_route_probe_rows(
            corpus.environment, corpus.train_cases
        )
        eval_rows, vocab_eval = eta_route_probe_rows(
            corpus.environment, corpus.heldout_cases
        )
    else:
        train_rows, vocab = eta_stage2_probe_rows(
            corpus.train_cases,
            environment=corpus.environment,
            protocol_version=args.probe_protocol,
        )
        eval_rows, vocab_eval = eta_stage2_probe_rows(
            corpus.heldout_cases,
            environment=corpus.environment,
            protocol_version=args.probe_protocol,
        )
    if vocab != vocab_eval:
        raise RuntimeError("probe label vocabulary differs between splits.")
    class_count = len(vocab)

    started = time.perf_counter()
    base_arm = _probe_arm(
        model_source=args.base_model_source,
        device=args.device,
        train_rows=train_rows,
        eval_rows=eval_rows,
        class_count=class_count,
        layer_selection=args.layer_selection,
        max_length=args.max_length,
    )
    pretrained_arm = _probe_arm(
        model_source=args.pretrained_model_source,
        device=args.device,
        train_rows=train_rows,
        eval_rows=eval_rows,
        class_count=class_count,
        layer_selection=args.layer_selection,
        max_length=args.max_length,
    )
    elapsed = time.perf_counter() - started
    gate = assess_gate2(base_arm=base_arm, pretrained_arm=pretrained_arm)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "eta-stage2-probe.v2",
        "experiment_id": "eta-stage2-linear-classification-probe",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "probe_protocol": args.probe_protocol,
        "layer_selection": args.layer_selection,
        "max_length": args.max_length,
        "corpus_seed": args.corpus_seed,
        "objective_count": args.objective_count,
        "class_count": class_count,
        "subgoal_vocabulary": list(vocab),
        "train_probe_rows": len(train_rows),
        "eval_probe_rows": len(eval_rows),
        "base_arm": base_arm,
        "pretrained_arm": pretrained_arm,
        "gate2": gate,
        "device": args.device,
        "elapsed_seconds": round(elapsed, 1),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
    }
    (output_dir / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate2": gate, "output_dir": str(output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
