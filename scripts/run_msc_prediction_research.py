#!/usr/bin/env python3
"""Run the real-human MSC N+1 capacity/four-arm research lane.

This runner never vendors corpus text and never routes mismatch around the PE
owner.  It intentionally labels its bounded-state ``volvence`` arm as a
prototype: only an external collector using the complete runtime may attest
``volvence_full_stack=True`` for formal R4 adjudication.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

from companion_bench.msc_corpus import MSCDyad, load_msc_split
from companion_bench.prediction_research import (
    PREDICTION_ARMS,
    CapacityObservation,
    MSCDialogueTurn,
    MSCNextTurnExample,
    PredictionObservation,
    adjudicate_capacity_ladder,
    adjudicate_prediction_experiment,
    build_msc_next_turn_examples,
    examples_fingerprint,
    render_long_context,
    render_stateless_context,
    render_summary_retrieval_context,
)
from volvence_zero.prediction import ForwardRepresentationBatch, PredictionErrorModule


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[arg-type]
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n",
        encoding="utf-8",
    )


def _stable_subset(dyads: tuple[MSCDyad, ...], limit: int | None) -> tuple[MSCDyad, ...]:
    if limit is None:
        return dyads
    if limit < 1:
        raise ValueError("dyad limits must be positive")
    return tuple(
        sorted(
            dyads,
            key=lambda row: hashlib.sha256(row.dyad_id.encode("utf-8")).hexdigest(),
        )[:limit]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenSentenceEncoder:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_seq_length: int,
        batch_size: int,
    ) -> None:
        if max_seq_length < 8 or batch_size < 1:
            raise ValueError("encoder max_seq_length/batch_size are invalid")
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "MSC prediction research requires sentence-transformers and "
                "huggingface-hub"
            ) from exc
        snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
        self._model = SentenceTransformer(
            str(snapshot), local_files_only=True, device=device
        )
        transformer = self._model[0]
        declared_limit = int(self._model.max_seq_length)
        position_limit = int(transformer.auto_model.config.max_position_embeddings)
        self.max_seq_length = min(max_seq_length, declared_limit, position_limit)
        self._model.max_seq_length = self.max_seq_length
        self._tokenizer = self._model.tokenizer
        self._tokenizer.truncation_side = "left"
        self.batch_size = batch_size
        self.model_id = model_id
        self.device = device
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        digest = hashlib.sha256()
        digest.update(model_id.encode("utf-8"))
        digest.update(str(self.max_seq_length).encode("ascii"))
        digest.update(b"recency-left-truncation")
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            relative = path.relative_to(snapshot).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(_sha256_file(path).encode("ascii"))
        self.fingerprint = digest.hexdigest()

    def encode(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, ...], ...], float]:
        if not texts:
            return (), 0.0
        started = time.perf_counter()
        values = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            tuple(tuple(float(value) for value in row) for row in values),
            elapsed_ms / len(texts),
        )

    def token_cost(self, text: str) -> tuple[int, int]:
        token_count = len(
            self._tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        )
        return (
            max(1, min(token_count, self.max_seq_length)),
            max(0, token_count - self.max_seq_length),
        )


def _unit(values: Iterable[float]) -> tuple[float, ...]:
    row = tuple(values)
    norm = math.sqrt(sum(value * value for value in row))
    if norm <= 1e-12:
        return tuple(0.0 for _ in row)
    return tuple(value / norm for value in row)


def _mean_vectors(
    vectors: tuple[tuple[float, ...], ...], *, dimension: int
) -> tuple[float, ...]:
    if not vectors:
        return tuple(0.0 for _ in range(dimension))
    return tuple(
        sum(row[index] for row in vectors) / len(vectors)
        for index in range(dimension)
    )


def _ema_vectors(
    vectors: tuple[tuple[float, ...], ...], *, dimension: int, decay: float = 0.85
) -> tuple[float, ...]:
    state = tuple(0.0 for _ in range(dimension))
    for vector in vectors:
        state = tuple(
            decay * previous + (1.0 - decay) * current
            for previous, current in zip(state, vector, strict=True)
        )
    return state


def _all_atomic_texts(
    examples_by_split: dict[str, tuple[MSCNextTurnExample, ...]]
) -> tuple[str, ...]:
    texts: dict[str, None] = {}
    for examples in examples_by_split.values():
        for example in examples:
            texts[example.target_text] = None
            for persona in example.personas:
                texts[persona] = None
            for turn in example.history:
                texts[turn.text] = None
    return tuple(texts)


def _embedding_map(
    encoder: FrozenSentenceEncoder,
    texts: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], float]:
    rows, per_item_ms = encoder.encode(texts)
    return dict(zip(texts, rows, strict=True)), per_item_ms


def _retrieved_turns(
    example: MSCNextTurnExample,
    *,
    embeddings: dict[str, tuple[float, ...]],
    count: int,
) -> tuple[MSCDialogueTurn, ...]:
    query = embeddings[example.latest_text]
    candidates = tuple(example.history[:-1])
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: sum(
            left * right
            for left, right in zip(
                query, embeddings[item[1].text], strict=True
            )
        ),
        reverse=True,
    )[:count]
    return tuple(candidates[index] for index, _ in sorted(ranked))


def _volvence_state(
    example: MSCNextTurnExample,
    *,
    embeddings: dict[str, tuple[float, ...]],
    dimension: int,
) -> tuple[float, ...]:
    target_history = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
    )
    previous_sessions = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
        and turn.session_index < example.session_index
    )
    current_session = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
        and turn.session_index == example.session_index
    )
    personas = tuple(embeddings[text] for text in example.personas)
    latest = embeddings[example.latest_text]
    ema = _ema_vectors(target_history, dimension=dimension)
    persona = _mean_vectors(personas, dimension=dimension)
    consolidated = _mean_vectors(previous_sessions, dimension=dimension)
    within_session = _mean_vectors(current_session, dimension=dimension)
    return _unit(
        0.45 * latest[index]
        + 0.25 * ema[index]
        + 0.15 * persona[index]
        + 0.10 * consolidated[index]
        + 0.05 * within_session[index]
        for index in range(dimension)
    )


def _prepare_arm_vectors(
    *,
    arm: str,
    examples: tuple[MSCNextTurnExample, ...],
    encoder: FrozenSentenceEncoder,
    atomic_embeddings: dict[str, tuple[float, ...]],
    atomic_latency_ms: float,
    retrieval_count: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    if arm == "volvence":
        started = time.perf_counter()
        vectors = tuple(
            _volvence_state(
                example,
                embeddings=atomic_embeddings,
                dimension=encoder.embedding_dim,
            )
            for example in examples
        )
        state_ms = (time.perf_counter() - started) * 1000.0 / max(len(examples), 1)
        costs = tuple(encoder.token_cost(example.latest_text) for example in examples)
        return (
            vectors,
            tuple(cost[0] for cost in costs),
            tuple(cost[1] for cost in costs),
            tuple(atomic_latency_ms + state_ms for _ in examples),
        )
    if arm == "stateless":
        texts = tuple(render_stateless_context(example) for example in examples)
    elif arm == "long_context":
        texts = tuple(render_long_context(example) for example in examples)
    elif arm == "summary_retrieval":
        texts = tuple(
            render_summary_retrieval_context(
                example,
                retrieved_turns=_retrieved_turns(
                    example,
                    embeddings=atomic_embeddings,
                    count=retrieval_count,
                ),
            )
            for example in examples
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    vectors, per_item_ms = encoder.encode(texts)
    costs = tuple(encoder.token_cost(text) for text in texts)
    return (
        vectors,
        tuple(cost[0] for cost in costs),
        tuple(cost[1] for cost in costs),
        tuple(per_item_ms for _ in examples),
    )


def _batches(indices: tuple[int, ...], batch_size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _make_batch(
    *,
    batch_id: str,
    examples: tuple[MSCNextTurnExample, ...],
    context_vectors: tuple[tuple[float, ...], ...],
    target_vectors: tuple[tuple[float, ...], ...],
    persistence_vectors: tuple[tuple[float, ...], ...],
    indices: tuple[int, ...],
) -> ForwardRepresentationBatch:
    return ForwardRepresentationBatch(
        batch_id=batch_id,
        sample_ids=tuple(examples[index].sample_id for index in indices),
        context_representations=tuple(context_vectors[index] for index in indices),
        target_representations=tuple(target_vectors[index] for index in indices),
        persistence_representations=tuple(
            persistence_vectors[index] for index in indices
        ),
        history_turns=tuple(examples[index].history_turns for index in indices),
    )


def _train_head(
    *,
    train_examples: tuple[MSCNextTurnExample, ...],
    train_contexts: tuple[tuple[float, ...], ...],
    train_targets: tuple[tuple[float, ...], ...],
    train_persistence: tuple[tuple[float, ...], ...],
    n_z: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> PredictionErrorModule:
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=len(train_contexts[0]),
        target_dim=len(train_targets[0]),
        n_z=n_z,
        seed=seed,
        learning_rate=learning_rate,
        device=device,
    )
    rng = random.Random(seed)
    for epoch in range(epochs):
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)
        for batch_index, batch_indices in enumerate(
            _batches(tuple(indices), batch_size)
        ):
            module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"train-e{epoch}-b{batch_index}",
                    examples=train_examples,
                    context_vectors=train_contexts,
                    target_vectors=train_targets,
                    persistence_vectors=train_persistence,
                    indices=batch_indices,
                ),
                update=True,
            )
    return module


def _evaluate_head(
    *,
    module: PredictionErrorModule,
    examples: tuple[MSCNextTurnExample, ...],
    context_vectors: tuple[tuple[float, ...], ...],
    target_vectors: tuple[tuple[float, ...], ...],
    persistence_vectors: tuple[tuple[float, ...], ...],
    batch_size: int,
) -> tuple[tuple[Any, ...], tuple[float, ...], str]:
    settlements = []
    latencies = []
    fingerprint = ""
    indices = tuple(range(len(examples)))
    for batch_index, batch_indices in enumerate(_batches(indices, batch_size)):
        snapshot = module.process_forward_representation_batch(
            _make_batch(
                batch_id=f"eval-b{batch_index}",
                examples=examples,
                context_vectors=context_vectors,
                target_vectors=target_vectors,
                persistence_vectors=persistence_vectors,
                indices=batch_indices,
            ),
            update=False,
        )
        settlements.extend(snapshot.settlements)
        latencies.extend(
            snapshot.elapsed_ms / snapshot.sample_count
            for _ in snapshot.settlements
        )
        fingerprint = snapshot.parameter_fingerprint
    return tuple(settlements), tuple(latencies), fingerprint


def _targets(
    examples: tuple[MSCNextTurnExample, ...],
    embeddings: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(embeddings[example.target_text] for example in examples)


def _persistence(
    examples: tuple[MSCNextTurnExample, ...],
    embeddings: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(embeddings[example.latest_text] for example in examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accept-noncommercial-license", action="store_true", required=True
    )
    parser.add_argument(
        "--encoder", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--retrieval-count", type=int, default=4)
    parser.add_argument("--train-dyads", type=int, default=24)
    parser.add_argument("--validation-dyads", type=int, default=12)
    parser.add_argument("--heldout-dyads", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_noncommercial_license:
        raise SystemExit("MSC license acceptance flag is required")
    if args.epochs < 1 or args.retrieval_count < 1:
        raise ValueError("epochs/retrieval-count must be positive")
    seeds = tuple(args.seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("research run requires at least three unique non-negative seeds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    split_payload: dict[str, tuple[MSCDyad, ...]] = {}
    audits = {}
    limits = {
        "train": args.train_dyads,
        "validation": args.validation_dyads,
        "heldout": args.heldout_dyads,
    }
    for split in ("train", "validation", "heldout"):
        dyads, audit = load_msc_split(args.msc_root, split=split, strict=True)
        split_payload[split] = _stable_subset(dyads, limits[split])
        audits[split] = audit
    examples_by_split = {
        split: build_msc_next_turn_examples(dyads)
        for split, dyads in split_payload.items()
    }

    encoder = FrozenSentenceEncoder(
        model_id=args.encoder,
        device=args.device,
        max_seq_length=args.max_seq_length,
        batch_size=args.encoder_batch_size,
    )
    atomic_embeddings, atomic_latency_ms = _embedding_map(
        encoder, _all_atomic_texts(examples_by_split)
    )
    targets_by_split = {
        split: _targets(examples, atomic_embeddings)
        for split, examples in examples_by_split.items()
    }
    persistence_by_split = {
        split: _persistence(examples, atomic_embeddings)
        for split, examples in examples_by_split.items()
    }

    prepared: dict[
        tuple[str, str],
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[float, ...],
        ],
    ] = {}
    for arm in PREDICTION_ARMS:
        for split, examples in examples_by_split.items():
            prepared[(arm, split)] = _prepare_arm_vectors(
                arm=arm,
                examples=examples,
                encoder=encoder,
                atomic_embeddings=atomic_embeddings,
                atomic_latency_ms=atomic_latency_ms,
                retrieval_count=args.retrieval_count,
            )

    capacity_rows = []
    for n_z in (3, 16, 64, 256):
        for seed in seeds:
            module = _train_head(
                train_examples=examples_by_split["train"],
                train_contexts=prepared[("volvence", "train")][0],
                train_targets=targets_by_split["train"],
                train_persistence=persistence_by_split["train"],
                n_z=n_z,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.head_batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            )
            settlements, _, _ = _evaluate_head(
                module=module,
                examples=examples_by_split["validation"],
                context_vectors=prepared[("volvence", "validation")][0],
                target_vectors=targets_by_split["validation"],
                persistence_vectors=persistence_by_split["validation"],
                batch_size=args.head_batch_size,
            )
            capacity_rows.append(
                CapacityObservation(
                    n_z=n_z,
                    seed=seed,
                    split="validation",
                    mean_cosine_similarity=sum(
                        row.cosine_similarity for row in settlements
                    )
                    / len(settlements),
                    mean_squared_error=sum(row.mean_squared_error for row in settlements)
                    / len(settlements),
                )
            )
    capacity_verdict = adjudicate_capacity_ladder(
        tuple(capacity_rows),
        complete_train=(len(split_payload["train"]) == audits["train"].conversation_count),
        complete_validation=(
            len(split_payload["validation"])
            == audits["validation"].conversation_count
        ),
    )

    observations = []
    head_fingerprints = {}
    chosen_n_z = capacity_verdict.best_n_z
    for arm in PREDICTION_ARMS:
        for seed in seeds:
            module = _train_head(
                train_examples=examples_by_split["train"],
                train_contexts=prepared[(arm, "train")][0],
                train_targets=targets_by_split["train"],
                train_persistence=persistence_by_split["train"],
                n_z=chosen_n_z,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.head_batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            )
            settlements, head_latency, fingerprint = _evaluate_head(
                module=module,
                examples=examples_by_split["heldout"],
                context_vectors=prepared[(arm, "heldout")][0],
                target_vectors=targets_by_split["heldout"],
                persistence_vectors=persistence_by_split["heldout"],
                batch_size=args.head_batch_size,
            )
            head_fingerprints[f"{arm}:seed{seed}"] = fingerprint
            tokens = prepared[(arm, "heldout")][1]
            truncated = prepared[(arm, "heldout")][2]
            context_latency = prepared[(arm, "heldout")][3]
            for index, (example, settlement) in enumerate(
                zip(examples_by_split["heldout"], settlements, strict=True)
            ):
                observations.append(
                    PredictionObservation(
                        arm=arm,
                        seed=seed,
                        sample_id=example.sample_id,
                        dyad_id=example.dyad_id,
                        session_index=example.session_index,
                        history_turns=example.history_turns,
                        cosine_similarity=settlement.cosine_similarity,
                        mean_squared_error=settlement.mean_squared_error,
                        persistence_cosine_similarity=(
                            settlement.persistence_cosine_similarity
                        ),
                        persistence_mean_squared_error=(
                            settlement.persistence_mean_squared_error
                        ),
                        context_token_count=tokens[index],
                        context_truncated_tokens=truncated[index],
                        latency_ms=context_latency[index] + head_latency[index],
                    )
                )
    prediction_verdict = adjudicate_prediction_experiment(
        tuple(observations),
        heldout_sorted_id_sha256=audits["heldout"].sorted_id_sha256,
        encoder_fingerprint=encoder.fingerprint,
        volvence_full_stack=False,
    )

    manifest = {
        "schema_version": "msc-n-plus-one-research.v1",
        "license_policy": "noncommercial-research-only",
        "corpus": {
            split: {
                "official_audit": asdict(audits[split]),
                "selected_dyads": len(split_payload[split]),
                "prediction_examples": len(examples_by_split[split]),
                "example_fingerprint": examples_fingerprint(examples_by_split[split]),
            }
            for split in ("train", "validation", "heldout")
        },
        "encoder": {
            "model_id": encoder.model_id,
            "fingerprint": encoder.fingerprint,
            "embedding_dim": encoder.embedding_dim,
            "max_seq_length": encoder.max_seq_length,
            "device": encoder.device,
        },
        "training": {
            "seeds": seeds,
            "capacity_n_z": (3, 16, 64, 256),
            "selected_n_z": chosen_n_z,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "head_batch_size": args.head_batch_size,
            "head_fingerprints": head_fingerprints,
        },
        "arms": {
            "volvence": (
                "PE-owned bounded recurrent relationship-state prototype; "
                "NOT a full runtime-stack attestation"
            ),
            "stateless": "persona plus latest partner message",
            "long_context": (
                "complete rendered relationship history, recency-truncated by "
                "the frozen encoder at the audited max sequence length"
            ),
            "summary_retrieval": (
                "persona summary plus frozen-encoder top-k extractive memories"
            ),
        },
        "claim_boundary": (
            "This runner produces pilot mechanism evidence only because the "
            "volvence arm bypasses propagate and is not the complete runtime stack."
        ),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "capacity_ladder.json", capacity_verdict)
    _write_json(output / "prediction_verdict.json", prediction_verdict)
    with (output / "prediction_observations.jsonl").open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(asdict(observation), sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "\n".join(
            (
                "# MSC N+1 prediction research report",
                "",
                f"- Evidence level: `{prediction_verdict.evidence_level}`",
                f"- Thesis exit: `{prediction_verdict.thesis_exit}`",
                f"- Capacity exit: `{capacity_verdict.eta_claim_exit}`",
                f"- Selected n_z: `{chosen_n_z}`",
                (
                    "- Longest-session cosine advantage vs long context: "
                    f"`{prediction_verdict.longest_quality_advantage:.6f}`"
                ),
                f"- Token ratio: `{prediction_verdict.longest_token_ratio:.6f}`",
                f"- Latency ratio: `{prediction_verdict.longest_latency_ratio:.6f}`",
                "",
                "This is pilot-only: the bounded-state arm is not the complete ",
                "Volvence runtime stack, and partial corpus runs cannot select thesis v3.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(output)
    print(prediction_verdict.description)
    print(capacity_verdict.description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
