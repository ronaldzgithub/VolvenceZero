"""Frozen-substrate N+1 measurement for seven-day companion evidence.

The target is the unconditioned residual representation of the next user
utterance published by ``vz-substrate``.  The context is a canonical product
transcript prefix, so it may differ by arm without reading owner metrics or
feeding evaluation back into the runtime.  A PE-owned bounded head is trained
on target turns from days 1--5 and frozen before days 6--7 are evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
from typing import Mapping, Sequence

from volvence_zero.prediction import ForwardRepresentationBatch, PredictionErrorModule
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SubstrateForwardRepresentationPublisher,
    SubstrateForwardRepresentationSnapshot,
    build_transformers_runtime_with_fallback,
)


SEVEN_DAY_N_PLUS_ONE_SCHEMA_VERSION = "seven-day-substrate-n-plus-one.v1"
SEVEN_DAY_N_PLUS_ONE_CONTRACT_SCHEMA_VERSION = (
    "seven-day-substrate-n-plus-one-contract.v1"
)
SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER = "vz-substrate"
SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER = "vz-cognition.prediction_error"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _vector_sha256(values: Sequence[float]) -> str:
    return hashlib.sha256(
        struct.pack(f"!{len(values)}d", *values)
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def build_seven_day_n_plus_one_contract(
    *,
    sut_model: Mapping[str, object],
    execution_device: str,
) -> dict[str, object]:
    """Freeze the measurement configuration against the formal SUT model."""

    model_id = sut_model.get("model_id")
    weights_sha256 = sut_model.get("weights_sha256")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("N+1 target model_id must be non-empty")
    if not _is_sha256(weights_sha256):
        raise ValueError("N+1 target weights_sha256 must be SHA-256")
    if sut_model.get("frozen") is not True or sut_model.get("local_files_only") is not True:
        raise ValueError("N+1 target must use frozen local-only SUT weights")
    if execution_device not in {"mps", "cuda", "cuda:0"}:
        raise ValueError("N+1 formal measurement requires mps or cuda")
    return {
        "schema_version": SEVEN_DAY_N_PLUS_ONE_CONTRACT_SCHEMA_VERSION,
        "target_owner": SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER,
        "predictor_owner": SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER,
        "target_model_role": "formal_models.sut",
        "target_model": {
            "model_id": model_id,
            "weights_sha256": weights_sha256,
            "frozen": True,
            "local_files_only": True,
        },
        "target_capture": {
            "personal_conditioning_applied": False,
            "layer_selection": "middle",
            "activation_width": 896,
            "readout_kind": "latest-token-selected-layer-residual-l2.v1",
        },
        "context": {
            "source": "canonical user/assistant product transcript prefix",
            "owner_readouts_included": False,
            "evaluation_values_included": False,
        },
        "predictor": {
            "kind": "Linear-Tanh-Linear",
            "n_z": 16,
            "seed": 7301,
            "epochs": 12,
            "batch_size": 8,
            "learning_rate": 0.003,
            "device": execution_device,
            "gradient_clip_norm": 1.0,
        },
        "split": {
            "training_target_days": [1, 2, 3, 4, 5],
            "heldout_target_days": [6, 7],
            "head_frozen_during_heldout": True,
            "training_example_count": 24,
            "heldout_example_count": 10,
        },
        "primary_readout": "heldout_mean_cosine_similarity",
        "error_readouts": [
            "heldout_mean_cosine_error",
            "heldout_mean_squared_error",
        ],
        "persistence_baseline_required": True,
        "target_arm_independent": True,
        "evaluation_writeback_allowed": False,
    }


@dataclass(frozen=True)
class SevenDayNPlusOneReadout:
    heldout_mean_cosine_similarity: float
    heldout_mean_cosine_error: float
    heldout_mean_squared_error: float
    heldout_persistence_mean_cosine_similarity: float
    heldout_persistence_mean_squared_error: float
    target_sequence_sha256: str
    target_snapshot_fingerprint: str
    head_parameter_fingerprint: str


def _turn_rows(
    run: Mapping[str, object],
) -> tuple[tuple[int, int, str, str], ...]:
    raw_days = run.get("days")
    if not isinstance(raw_days, (list, tuple)) or len(raw_days) != 7:
        raise ValueError("N+1 run must contain seven days")
    rows: list[tuple[int, int, str, str]] = []
    for expected_day, raw_day in enumerate(raw_days, start=1):
        day = _mapping(raw_day, field=f"N+1 day-{expected_day}")
        if day.get("day_index") != expected_day:
            raise ValueError("N+1 run day order drift")
        raw_turns = day.get("turns")
        if not isinstance(raw_turns, (list, tuple)) or len(raw_turns) != 5:
            raise ValueError("N+1 run day must contain five turns")
        for expected_exchange, raw_turn in enumerate(raw_turns, start=1):
            turn = _mapping(
                raw_turn,
                field=f"N+1 day-{expected_day}.turn-{expected_exchange}",
            )
            if turn.get("exchange_index") != expected_exchange:
                raise ValueError("N+1 run exchange order drift")
            user_text = turn.get("user_text")
            assistant_text = turn.get("assistant_text")
            if not isinstance(user_text, str) or not user_text.strip():
                raise ValueError("N+1 user_text must be non-empty")
            if not isinstance(assistant_text, str) or not assistant_text.strip():
                raise ValueError("N+1 assistant_text must be non-empty")
            rows.append(
                (expected_day, expected_exchange, user_text, assistant_text)
            )
    return tuple(rows)


def _context_source(
    rows: Sequence[tuple[int, int, str, str]], *, end_index: int
) -> str:
    payload = {
        "schema_version": "seven-day-product-transcript-prefix.v1",
        "turns": [
            {
                "day_index": day,
                "exchange_index": exchange,
                "user": user,
                "assistant": assistant,
            }
            for day, exchange, user, assistant in rows[: end_index + 1]
        ],
    }
    return _canonical_bytes(payload).decode("utf-8")


def _batch(
    *,
    batch_id: str,
    indices: Sequence[int],
    contexts: Sequence[tuple[float, ...]],
    targets: Sequence[tuple[float, ...]],
    persistence: Sequence[tuple[float, ...]],
    target_snapshot: SubstrateForwardRepresentationSnapshot,
) -> ForwardRepresentationBatch:
    return ForwardRepresentationBatch(
        batch_id=batch_id,
        sample_ids=tuple(f"prediction-target-turn-{index + 2}" for index in indices),
        context_representations=tuple(contexts[index] for index in indices),
        target_representations=tuple(targets[index] for index in indices),
        persistence_representations=tuple(persistence[index] for index in indices),
        history_turns=tuple(index + 1 for index in indices),
        target_lineage=target_snapshot.lineage,
    )


class SevenDayNPlusOneCompiler:
    """Compile a complete N+1 artifact from one frozen product-path run."""

    def __init__(
        self,
        *,
        publisher: SubstrateForwardRepresentationPublisher,
        contract: Mapping[str, object],
    ) -> None:
        self._publisher = publisher
        self._contract = dict(contract)
        self._target_cache: dict[str, SubstrateForwardRepresentationSnapshot] = {}
        # Validate exact public shape up front.
        target_model = _mapping(self._contract.get("target_model"), field="target_model")
        expected = build_seven_day_n_plus_one_contract(
            sut_model={
                **target_model,
                "frozen": True,
                "local_files_only": True,
            },
            execution_device=str(
                _mapping(self._contract.get("predictor"), field="predictor").get(
                    "device"
                )
            ),
        )
        if self._contract != expected:
            raise ValueError("seven-day N+1 measurement contract drift")

    def compile(self, run: Mapping[str, object]) -> dict[str, object]:
        rows = _turn_rows(run)
        case_id = f"{run.get('scenario_id')}:seed-{run.get('paraphrase_seed')}"
        target_sources = tuple(
            (f"{case_id}:user-turn-{index + 1}", row[2])
            for index, row in enumerate(rows)
        )
        target_source_key = _sha256(target_sources)
        target_snapshot = self._target_cache.get(target_source_key)
        if target_snapshot is None:
            target_snapshot = self._publisher.publish(target_sources)
            self._target_cache[target_source_key] = target_snapshot
        context_sources = tuple(
            (
                f"{case_id}:{run.get('arm_label')}:context-turn-{index + 1}",
                _context_source(rows, end_index=index),
            )
            for index in range(34)
        )
        context_snapshot = self._publisher.publish(context_sources)
        if (
            context_snapshot.lineage.model_fingerprint
            != target_snapshot.lineage.model_fingerprint
            or context_snapshot.lineage.layer_indices
            != target_snapshot.lineage.layer_indices
            or context_snapshot.lineage.activation_widths
            != target_snapshot.lineage.activation_widths
        ):
            raise ValueError("N+1 context/target substrate geometry drift")

        user_vectors = tuple(row.values for row in target_snapshot.representations)
        contexts = tuple(row.values for row in context_snapshot.representations)
        targets = user_vectors[1:]
        persistence = user_vectors[:-1]
        predictor = _mapping(self._contract["predictor"], field="predictor")
        module = PredictionErrorModule()
        device = str(predictor["device"])
        module.configure_forward_representation_head(
            input_dim=len(contexts[0]),
            target_dim=len(targets[0]),
            n_z=int(predictor["n_z"]),
            seed=int(predictor["seed"]),
            learning_rate=float(predictor["learning_rate"]),
            device="cuda" if device == "cuda:0" else device,
        )
        training_indices = tuple(range(24))
        rng = random.Random(int(predictor["seed"]))
        batch_size = int(predictor["batch_size"])
        for epoch in range(int(predictor["epochs"])):
            shuffled = list(training_indices)
            rng.shuffle(shuffled)
            for batch_index, start in enumerate(range(0, len(shuffled), batch_size)):
                indices = tuple(shuffled[start : start + batch_size])
                module.process_forward_representation_batch(
                    _batch(
                        batch_id=f"{case_id}:train-e{epoch}-b{batch_index}",
                        indices=indices,
                        contexts=contexts,
                        targets=targets,
                        persistence=persistence,
                        target_snapshot=target_snapshot,
                    ),
                    update=True,
                )
        heldout_indices = tuple(range(24, 34))
        heldout = module.process_forward_representation_batch(
            _batch(
                batch_id=f"{case_id}:heldout-days-6-7",
                indices=heldout_indices,
                contexts=contexts,
                targets=targets,
                persistence=persistence,
                target_snapshot=target_snapshot,
            ),
            update=False,
        )
        target_rows = [
            {
                "sample_id": representation.sample_id,
                "day_index": rows[index][0],
                "exchange_index": rows[index][1],
                "source_sha256": representation.source_sha256,
                "values_sha256": representation.values_sha256,
            }
            for index, representation in enumerate(target_snapshot.representations)
        ]
        context_rows = [
            {
                "sample_id": representation.sample_id,
                "history_turns": index + 1,
                "source_sha256": representation.source_sha256,
                "values_sha256": representation.values_sha256,
            }
            for index, representation in enumerate(context_snapshot.representations)
        ]
        observations = []
        for index, settlement in zip(
            heldout_indices, heldout.settlements, strict=True
        ):
            target_day, target_exchange, _user, _assistant = rows[index + 1]
            observations.append(
                {
                    "sample_id": settlement.sample_id,
                    "target_day_index": target_day,
                    "target_exchange_index": target_exchange,
                    "history_turns": settlement.history_turns,
                    "predicted_representation": list(
                        settlement.predicted_representation
                    ),
                    "actual_representation": list(
                        settlement.actual_representation
                    ),
                    "persistence_representation": list(persistence[index]),
                    "mean_squared_error": settlement.mean_squared_error,
                    "cosine_similarity": settlement.cosine_similarity,
                    "cosine_error": 1.0 - settlement.cosine_similarity,
                    "persistence_mean_squared_error": (
                        settlement.persistence_mean_squared_error
                    ),
                    "persistence_cosine_similarity": (
                        settlement.persistence_cosine_similarity
                    ),
                    "target_values_sha256": target_rows[index + 1][
                        "values_sha256"
                    ],
                }
            )
        payload = {
            "schema_version": SEVEN_DAY_N_PLUS_ONE_SCHEMA_VERSION,
            "contract_sha256": _sha256(self._contract),
            "target_owner": SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER,
            "predictor_owner": SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER,
            "target_arm_independent": True,
            "target_personal_conditioning_applied": False,
            "evaluation_writeback_allowed": False,
            "target_lineage": asdict(target_snapshot.lineage),
            "context_lineage": asdict(context_snapshot.lineage),
            "target_rows": target_rows,
            "context_rows": context_rows,
            "target_sequence_sha256": _sha256(target_rows),
            "context_sequence_sha256": _sha256(context_rows),
            "training_example_count": len(training_indices),
            "heldout_example_count": len(heldout_indices),
            "head_parameter_fingerprint": heldout.parameter_fingerprint,
            "heldout_mean_cosine_similarity": heldout.mean_cosine_similarity,
            "heldout_mean_cosine_error": 1.0 - heldout.mean_cosine_similarity,
            "heldout_mean_squared_error": heldout.mean_squared_error,
            "heldout_persistence_mean_cosine_similarity": (
                heldout.persistence_mean_cosine_similarity
            ),
            "heldout_persistence_mean_squared_error": (
                heldout.persistence_mean_squared_error
            ),
            "observations": observations,
        }
        validate_seven_day_n_plus_one_evidence(
            run={**run, "n_plus_one_representation_evidence": payload},
            contract=self._contract,
        )
        return payload


def build_seven_day_n_plus_one_compiler(
    *,
    model_source: str | Path,
    contract: Mapping[str, object],
) -> SevenDayNPlusOneCompiler:
    target_model = _mapping(contract.get("target_model"), field="target_model")
    capture = _mapping(contract.get("target_capture"), field="target_capture")
    predictor = _mapping(contract.get("predictor"), field="predictor")
    model_id = str(target_model.get("model_id"))
    model_fingerprint = SubstrateFingerprint(
        model_id=model_id,
        version=Path(model_source).name,
        weights_sha256=str(target_model.get("weights_sha256")),
    )
    device = str(predictor.get("device"))
    runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        model_source=str(model_source),
        device=device,
        layer_indices=None,
        hook_layer_selection=str(capture.get("layer_selection")),
        activation_width=int(capture.get("activation_width", 0)),
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
    )
    return SevenDayNPlusOneCompiler(
        publisher=SubstrateForwardRepresentationPublisher(
            runtime,
            model_fingerprint=model_fingerprint,
        ),
        contract=contract,
    )


def _vector(value: object, *, field: str, dimension: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValueError(f"{field} dimension mismatch")
    result = tuple(_finite(item, field=field) for item in value)
    return result


def _mse(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def validate_seven_day_n_plus_one_evidence(
    *,
    run: Mapping[str, object],
    contract: Mapping[str, object],
) -> SevenDayNPlusOneReadout:
    """Validate the stored vectors, lineage, split, and aggregate metrics."""

    evidence = _mapping(
        run.get("n_plus_one_representation_evidence"),
        field="n_plus_one_representation_evidence",
    )
    if evidence.get("schema_version") != SEVEN_DAY_N_PLUS_ONE_SCHEMA_VERSION:
        raise ValueError("seven-day N+1 evidence schema drift")
    fixed = {
        "contract_sha256": _sha256(contract),
        "target_owner": SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER,
        "predictor_owner": SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER,
        "target_arm_independent": True,
        "target_personal_conditioning_applied": False,
        "evaluation_writeback_allowed": False,
    }
    for field, expected in fixed.items():
        if evidence.get(field) != expected:
            raise ValueError(f"seven-day N+1 {field} drift")
    rows = _turn_rows(run)
    target_rows = evidence.get("target_rows")
    context_rows = evidence.get("context_rows")
    if not isinstance(target_rows, list) or len(target_rows) != 35:
        raise ValueError("seven-day N+1 target rows must cover 35 turns")
    if not isinstance(context_rows, list) or len(context_rows) != 34:
        raise ValueError("seven-day N+1 context rows must cover 34 prefixes")
    for index, raw in enumerate(target_rows):
        row = _mapping(raw, field=f"target_rows[{index}]")
        if (
            row.get("day_index") != rows[index][0]
            or row.get("exchange_index") != rows[index][1]
            or row.get("source_sha256")
            != hashlib.sha256(rows[index][2].encode("utf-8")).hexdigest()
            or not _is_sha256(row.get("values_sha256"))
        ):
            raise ValueError("seven-day N+1 target row lineage drift")
    for index, raw in enumerate(context_rows):
        row = _mapping(raw, field=f"context_rows[{index}]")
        expected_source = _context_source(rows, end_index=index)
        if (
            row.get("history_turns") != index + 1
            or row.get("source_sha256")
            != hashlib.sha256(expected_source.encode("utf-8")).hexdigest()
            or not _is_sha256(row.get("values_sha256"))
        ):
            raise ValueError("seven-day N+1 context row lineage drift")
    if evidence.get("target_sequence_sha256") != _sha256(target_rows):
        raise ValueError("seven-day N+1 target sequence SHA drift")
    if evidence.get("context_sequence_sha256") != _sha256(context_rows):
        raise ValueError("seven-day N+1 context sequence SHA drift")

    target_lineage = _mapping(evidence.get("target_lineage"), field="target_lineage")
    context_lineage = _mapping(evidence.get("context_lineage"), field="context_lineage")
    target_model = _mapping(contract.get("target_model"), field="target_model")
    model_fingerprint = _mapping(
        target_lineage.get("model_fingerprint"), field="target model_fingerprint"
    )
    if (
        model_fingerprint.get("model_id") != target_model.get("model_id")
        or model_fingerprint.get("weights_sha256")
        != target_model.get("weights_sha256")
        or context_lineage.get("model_fingerprint")
        != target_lineage.get("model_fingerprint")
        or context_lineage.get("layer_indices")
        != target_lineage.get("layer_indices")
        or context_lineage.get("activation_widths")
        != target_lineage.get("activation_widths")
    ):
        raise ValueError("seven-day N+1 substrate lineage drift")
    dimension = target_lineage.get("representation_dim")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("seven-day N+1 representation dimension is invalid")
    if evidence.get("training_example_count") != 24:
        raise ValueError("seven-day N+1 training split drift")
    if evidence.get("heldout_example_count") != 10:
        raise ValueError("seven-day N+1 heldout split drift")
    if not _is_sha256(evidence.get("head_parameter_fingerprint")):
        raise ValueError("seven-day N+1 head fingerprint is invalid")
    observations = evidence.get("observations")
    if not isinstance(observations, list) or len(observations) != 10:
        raise ValueError("seven-day N+1 observations must cover days 6-7")
    cosine_values = []
    mse_values = []
    persistence_cosine_values = []
    persistence_mse_values = []
    for offset, raw in enumerate(observations):
        observation = _mapping(raw, field=f"observations[{offset}]")
        source_index = offset + 25
        expected_day, expected_exchange, _user, _assistant = rows[source_index]
        if (
            observation.get("target_day_index") != expected_day
            or observation.get("target_exchange_index") != expected_exchange
            or observation.get("history_turns") != source_index
            or expected_day not in {6, 7}
        ):
            raise ValueError("seven-day N+1 heldout coordinate drift")
        predicted = _vector(
            observation.get("predicted_representation"),
            field="predicted_representation",
            dimension=dimension,
        )
        actual = _vector(
            observation.get("actual_representation"),
            field="actual_representation",
            dimension=dimension,
        )
        persistence = _vector(
            observation.get("persistence_representation"),
            field="persistence_representation",
            dimension=dimension,
        )
        if (
            observation.get("target_values_sha256")
            != target_rows[source_index].get("values_sha256")
            or _vector_sha256(actual)
            != observation.get("target_values_sha256")
        ):
            raise ValueError("seven-day N+1 heldout target vector drift")
        cosine = _cosine(predicted, actual)
        mse = _mse(predicted, actual)
        persistence_cosine = _cosine(persistence, actual)
        persistence_mse = _mse(persistence, actual)
        checks = {
            "cosine_similarity": cosine,
            "cosine_error": 1.0 - cosine,
            "mean_squared_error": mse,
            "persistence_cosine_similarity": persistence_cosine,
            "persistence_mean_squared_error": persistence_mse,
        }
        for field, expected in checks.items():
            if not math.isclose(
                _finite(observation.get(field), field=field),
                expected,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(f"seven-day N+1 observation {field} drift")
        cosine_values.append(cosine)
        mse_values.append(mse)
        persistence_cosine_values.append(persistence_cosine)
        persistence_mse_values.append(persistence_mse)
    aggregates = {
        "heldout_mean_cosine_similarity": statistics.fmean(cosine_values),
        "heldout_mean_cosine_error": 1.0 - statistics.fmean(cosine_values),
        "heldout_mean_squared_error": statistics.fmean(mse_values),
        "heldout_persistence_mean_cosine_similarity": statistics.fmean(
            persistence_cosine_values
        ),
        "heldout_persistence_mean_squared_error": statistics.fmean(
            persistence_mse_values
        ),
    }
    for field, expected in aggregates.items():
        if not math.isclose(
            _finite(evidence.get(field), field=field),
            expected,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(f"seven-day N+1 aggregate {field} drift")
    target_snapshot_fingerprint = target_lineage.get("snapshot_fingerprint")
    if not _is_sha256(target_snapshot_fingerprint):
        raise ValueError("seven-day N+1 target snapshot fingerprint is invalid")
    return SevenDayNPlusOneReadout(
        heldout_mean_cosine_similarity=aggregates[
            "heldout_mean_cosine_similarity"
        ],
        heldout_mean_cosine_error=aggregates["heldout_mean_cosine_error"],
        heldout_mean_squared_error=aggregates["heldout_mean_squared_error"],
        heldout_persistence_mean_cosine_similarity=aggregates[
            "heldout_persistence_mean_cosine_similarity"
        ],
        heldout_persistence_mean_squared_error=aggregates[
            "heldout_persistence_mean_squared_error"
        ],
        target_sequence_sha256=str(evidence["target_sequence_sha256"]),
        target_snapshot_fingerprint=str(target_snapshot_fingerprint),
        head_parameter_fingerprint=str(evidence["head_parameter_fingerprint"]),
    )


__all__ = (
    "SEVEN_DAY_N_PLUS_ONE_CONTRACT_SCHEMA_VERSION",
    "SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER",
    "SEVEN_DAY_N_PLUS_ONE_SCHEMA_VERSION",
    "SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER",
    "SevenDayNPlusOneCompiler",
    "SevenDayNPlusOneReadout",
    "build_seven_day_n_plus_one_compiler",
    "build_seven_day_n_plus_one_contract",
    "validate_seven_day_n_plus_one_evidence",
)
