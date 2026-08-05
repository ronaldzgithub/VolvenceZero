# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Preregistered MSC N+1 prediction design and evidence adjudication.

The module is deliberately model-agnostic.  It freezes sample construction,
paired metrics, cost accounting, and fail-closed evidence levels.  A runner may
use any frozen encoder, but the actual predictor/mismatch must stay with the PE
owner.  Synthetic and partial runs are labelled pilots and cannot emit a thesis
verdict.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import random
import struct
from typing import Iterable, Mapping

from companion_bench.msc_corpus import MSCDyad


PREDICTION_ARMS = (
    "volvence",
    "stateless",
    "long_context",
    "summary_retrieval",
)
MSC_HELDOUT_ID_SHA256 = (
    "58a61e1b08a9d0ae384b413a677e161d2e809cacc1bd81ba79beb557588e5777"
)


@dataclass(frozen=True)
class MSCDialogueTurn:
    session_index: int
    speaker: str
    text: str
    utterance_index: int
    preceding_empty_utterance_count: int = 0


@dataclass(frozen=True)
class MSCNextTurnExample:
    sample_id: str
    dyad_id: str
    split: str
    target_speaker: str
    session_index: int
    history_turns: int
    personas: tuple[str, ...]
    history: tuple[MSCDialogueTurn, ...]
    target_text: str

    @property
    def latest_text(self) -> str:
        for turn in reversed(self.history):
            if turn.speaker != self.target_speaker:
                return turn.text
        raise ValueError("MSC prediction example has no observed partner turn")


@dataclass(frozen=True)
class PredictionObservation:
    arm: str
    seed: int
    sample_id: str
    dyad_id: str
    session_index: int
    history_turns: int
    cosine_similarity: float
    mean_squared_error: float
    persistence_cosine_similarity: float
    persistence_mean_squared_error: float
    context_token_count: int
    context_truncated_tokens: int
    latency_ms: float
    prediction_zero_norm: bool = False

    def __post_init__(self) -> None:
        if self.arm not in PREDICTION_ARMS:
            raise ValueError(f"unknown prediction arm {self.arm!r}")
        if self.seed < 0:
            raise ValueError("prediction observation seed must be non-negative")
        if not self.sample_id or not self.dyad_id:
            raise ValueError("prediction observation lineage ids must be non-empty")
        if self.session_index < 1 or self.history_turns < 1:
            raise ValueError("prediction observation history indices must be positive")
        numeric = (
            self.cosine_similarity,
            self.mean_squared_error,
            self.persistence_cosine_similarity,
            self.persistence_mean_squared_error,
            self.latency_ms,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("prediction observation metrics must be finite")
        if self.context_token_count < 1 or self.context_truncated_tokens < 0:
            raise ValueError("prediction observation token counts are invalid")
        if self.latency_ms < 0.0:
            raise ValueError("prediction observation latency must be non-negative")
        if not isinstance(self.prediction_zero_norm, bool):
            raise ValueError("prediction_zero_norm must be boolean")


@dataclass(frozen=True)
class PredictionThresholds:
    quality_min_cosine_advantage: float = 0.02
    quality_min_advantage_slope: float = 0.0
    scaling_min_cosine_equivalence: float = -0.01
    scaling_max_token_ratio: float = 0.10
    scaling_max_latency_ratio: float = 0.50
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260801
    formal_heldout_dyads: int = 501
    formal_min_seeds: int = 3
    formal_heldout_id_sha256: str = MSC_HELDOUT_ID_SHA256
    formal_longest_session: int = 5


@dataclass(frozen=True)
class SameSubstrateContextAttestation:
    """R3 proof that context and target use one frozen residual surface."""

    context_model_id: str
    target_model_id: str
    context_weights_sha256: str
    target_weights_sha256: str
    context_readout_kind: str
    target_readout_kind: str
    context_layer_indices: tuple[int, ...]
    target_layer_indices: tuple[int, ...]
    context_activation_widths: tuple[int, ...]
    target_activation_widths: tuple[int, ...]
    context_limit: int
    maximum_observed_tokens: int
    truncated_token_count: int
    schema_version: str = "msc-same-substrate-context-attestation.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "msc-same-substrate-context-attestation.v1":
            raise ValueError("same-substrate context attestation schema is unsupported")
        if not self.context_model_id.strip() or not self.target_model_id.strip():
            raise ValueError("same-substrate model ids must be non-empty")
        for field_name, value in (
            ("context_weights_sha256", self.context_weights_sha256),
            ("target_weights_sha256", self.target_weights_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256")
        if not self.context_readout_kind or not self.target_readout_kind:
            raise ValueError("same-substrate readout kinds must be non-empty")
        if not self.context_layer_indices or not self.target_layer_indices:
            raise ValueError("same-substrate layer indices must be non-empty")
        if (
            len(self.context_layer_indices)
            != len(self.context_activation_widths)
            or len(self.target_layer_indices)
            != len(self.target_activation_widths)
        ):
            raise ValueError("same-substrate residual geometry is malformed")
        if any(
            value < 1
            for value in (
                *self.context_activation_widths,
                *self.target_activation_widths,
                self.context_limit,
                self.maximum_observed_tokens,
            )
        ):
            raise ValueError("same-substrate dimensions/token limits must be positive")
        if self.truncated_token_count < 0:
            raise ValueError("same-substrate truncated token count cannot be negative")

    @property
    def passed(self) -> bool:
        return (
            self.context_model_id == self.target_model_id
            and self.context_weights_sha256 == self.target_weights_sha256
            and self.context_readout_kind == self.target_readout_kind
            and self.context_layer_indices == self.target_layer_indices
            and self.context_activation_widths == self.target_activation_widths
            and self.maximum_observed_tokens <= self.context_limit
            and self.truncated_token_count == 0
        )


@dataclass(frozen=True)
class MSCFullRuntimeContext:
    """One R4 context emitted after a complete service/runtime turn."""

    sample_id: str
    active_speaker_id: str
    values: tuple[float, ...]
    values_sha256: str
    source_sha256: str
    model_id: str
    model_version: str
    weights_sha256: str
    runtime_origin: str
    readout_kind: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    temporal_n_z: int
    input_token_count: int
    output_token_count: int
    total_token_count: int
    generation_latency_ms: float
    latency_ms: float
    propagate_event_count: int
    runtime_slot_surface_sha256: str
    schema_version: str = "msc-full-runtime-context.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "msc-full-runtime-context.v1":
            raise ValueError("MSC full-runtime context schema is unsupported")
        if not self.sample_id.strip():
            raise ValueError("MSC full-runtime context sample_id must be non-empty")
        if self.active_speaker_id not in {"speaker_1", "speaker_2"}:
            raise ValueError("MSC full-runtime context active speaker is invalid")
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise ValueError("MSC full-runtime context values must be finite")
        norm = math.sqrt(sum(value * value for value in self.values))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("MSC full-runtime context values must be L2-normalized")
        expected_values_sha = hashlib.sha256(
            struct.pack(f"!{len(self.values)}d", *self.values)
        ).hexdigest()
        if self.values_sha256 != expected_values_sha:
            raise ValueError("MSC full-runtime context values SHA drift")
        for field_name, value in (
            ("source_sha256", self.source_sha256),
            ("weights_sha256", self.weights_sha256),
            ("runtime_slot_surface_sha256", self.runtime_slot_surface_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"MSC full-runtime {field_name} must be SHA-256")
        if (
            not self.model_id.strip()
            or not self.model_version.strip()
            or not self.runtime_origin.strip()
        ):
            raise ValueError("MSC full-runtime model identity must be non-empty")
        if not self.readout_kind.strip() or not self.layer_indices:
            raise ValueError("MSC full-runtime residual readout is incomplete")
        if (
            len(self.layer_indices) != len(self.activation_widths)
            or sum(self.activation_widths) != len(self.values)
        ):
            raise ValueError("MSC full-runtime residual geometry is inconsistent")
        if self.temporal_n_z not in {3, 16, 64, 256}:
            raise ValueError("MSC full-runtime temporal_n_z is not preregistered")
        if (
            self.input_token_count < 1
            or self.output_token_count < 0
            or self.total_token_count
            != self.input_token_count + self.output_token_count
        ):
            raise ValueError("MSC full-runtime token accounting is inconsistent")
        if (
            not math.isfinite(self.generation_latency_ms)
            or self.generation_latency_ms < 0.0
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < self.generation_latency_ms
        ):
            raise ValueError("MSC full-runtime latency must be finite/non-negative")
        if self.propagate_event_count < 1:
            raise ValueError("MSC full-runtime propagate evidence is missing")


def parse_msc_full_runtime_context(
    payload: Mapping[str, object], *, sample_id: str
) -> MSCFullRuntimeContext:
    """Validate the service DTO without importing any internal wheel."""

    if payload.get("schema_version") != "msc-full-runtime-context.v1":
        raise ValueError("MSC full-runtime payload schema is unsupported")
    if payload.get("volvence_full_stack") is not True:
        raise ValueError("MSC full-runtime payload lacks full-stack attestation")
    if payload.get("acceptance_passed") is not True:
        raise ValueError("MSC full-runtime payload failed runtime acceptance")
    if payload.get("substrate_fallback_active") is not False:
        raise ValueError("MSC full-runtime payload used substrate fallback")
    if payload.get("raw_text_retained") is not False:
        raise ValueError("MSC full-runtime payload retained raw text")
    if payload.get("evaluation_writeback_allowed") is not False:
        raise ValueError("MSC full-runtime payload allows evaluation writeback")
    context = payload.get("context_representation")
    lineage = payload.get("context_lineage")
    if not isinstance(context, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError("MSC full-runtime context/lineage must be objects")
    model = lineage.get("model_fingerprint")
    if not isinstance(model, Mapping):
        raise ValueError("MSC full-runtime model fingerprint must be an object")
    values_raw = context.get("values")
    layers_raw = lineage.get("layer_indices")
    widths_raw = lineage.get("activation_widths")
    if not isinstance(values_raw, (list, tuple)) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values_raw
    ):
        raise ValueError("MSC full-runtime context values must be numeric")
    if not isinstance(layers_raw, (list, tuple)) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in layers_raw
    ):
        raise ValueError("MSC full-runtime layer indices must be integers")
    if not isinstance(widths_raw, (list, tuple)) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in widths_raw
    ):
        raise ValueError("MSC full-runtime activation widths must be integers")
    try:
        return MSCFullRuntimeContext(
            sample_id=sample_id,
            active_speaker_id=str(payload["active_speaker_id"]),
            values=tuple(float(value) for value in values_raw),
            values_sha256=str(context["values_sha256"]),
            source_sha256=str(context["source_sha256"]),
            model_id=str(model["model_id"]),
            model_version=str(model["version"]),
            weights_sha256=str(model["weights_sha256"]),
            runtime_origin=str(lineage["runtime_origin"]),
            readout_kind=str(lineage["readout_kind"]),
            layer_indices=tuple(layers_raw),
            activation_widths=tuple(widths_raw),
            temporal_n_z=int(payload["temporal_n_z"]),
            input_token_count=int(payload["input_token_count"]),
            output_token_count=int(payload["output_token_count"]),
            total_token_count=int(payload["total_token_count"]),
            generation_latency_ms=float(payload["generation_latency_ms"]),
            latency_ms=float(payload["end_to_end_latency_ms"]),
            propagate_event_count=int(payload["propagate_event_count"]),
            runtime_slot_surface_sha256=str(
                payload["runtime_slot_surface_sha256"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MSC full-runtime payload fields are invalid") from exc


@dataclass(frozen=True)
class SessionPredictionCurve:
    arm: str
    session_index: int
    observation_count: int
    dyad_count: int
    mean_cosine_similarity: float
    mean_squared_error: float
    mean_context_tokens: float
    mean_truncated_tokens: float
    mean_latency_ms: float
    zero_norm_prediction_count: int


@dataclass(frozen=True)
class PairedAdvantage:
    session_index: int
    pair_count: int
    dyad_count: int
    mean_cosine_advantage: float
    confidence_interval_95: tuple[float, float]


@dataclass(frozen=True)
class PredictionExperimentVerdict:
    evidence_level: str
    quality_condition_met: bool
    scaling_condition_met: bool
    thesis_exit: str
    longest_session: int
    longest_quality_advantage: float
    longest_quality_confidence_interval_95: tuple[float, float]
    advantage_slope: float
    longest_token_ratio: float
    longest_latency_ratio: float
    zero_norm_prediction_count: int
    formal_requirements: tuple[tuple[str, bool], ...]
    curves: tuple[SessionPredictionCurve, ...]
    paired_advantages: tuple[PairedAdvantage, ...]
    description: str


@dataclass(frozen=True)
class CapacityObservation:
    forward_head_n_z: int
    seed: int
    split: str
    mean_cosine_similarity: float
    mean_squared_error: float
    zero_norm_prediction_count: int = 0

    def __post_init__(self) -> None:
        if self.forward_head_n_z not in {3, 16, 64, 256}:
            raise ValueError(
                "capacity observation forward_head_n_z must be one of 3/16/64/256"
            )
        if self.seed < 0 or self.split not in {"validation", "heldout"}:
            raise ValueError("capacity observation seed/split is invalid")
        if not math.isfinite(self.mean_cosine_similarity) or not math.isfinite(
            self.mean_squared_error
        ):
            raise ValueError("capacity metrics must be finite")
        if (
            isinstance(self.zero_norm_prediction_count, bool)
            or self.zero_norm_prediction_count < 0
        ):
            raise ValueError("capacity zero-norm count must be non-negative")


@dataclass(frozen=True)
class CapacityLadderVerdict:
    evidence_level: str
    best_forward_head_n_z: int
    chosen_forward_head_n_z: int
    best_mean_cosine: float
    gain_over_forward_head_n_z_3: float
    forward_head_capacity_is_flat: bool
    zero_norm_prediction_count: int
    forward_head_claim_exit: str
    observations: tuple[CapacityObservation, ...]
    description: str


@dataclass(frozen=True)
class TemporalCapacityObservation:
    """One R5 validation result with PE-head capacity held fixed."""

    temporal_n_z: int
    forward_head_n_z: int
    seed: int
    split: str
    mean_cosine_similarity: float
    mean_squared_error: float
    zero_norm_prediction_count: int = 0

    def __post_init__(self) -> None:
        if self.temporal_n_z not in {3, 16, 64, 256}:
            raise ValueError("temporal capacity n_z must be one of 3/16/64/256")
        if self.forward_head_n_z != 3:
            raise ValueError("R5 temporal capacity must hold forward_head_n_z at 3")
        if self.seed < 0 or self.split != "validation":
            raise ValueError("temporal capacity seed/split is invalid")
        if not math.isfinite(self.mean_cosine_similarity) or not math.isfinite(
            self.mean_squared_error
        ):
            raise ValueError("temporal capacity metrics must be finite")
        if (
            isinstance(self.zero_norm_prediction_count, bool)
            or self.zero_norm_prediction_count < 0
        ):
            raise ValueError("temporal capacity zero-norm count must be non-negative")


@dataclass(frozen=True)
class TemporalCapacityLadderVerdict:
    evidence_level: str
    capacity_integrity_passed: bool
    best_temporal_n_z: int
    chosen_temporal_n_z: int
    fixed_forward_head_n_z: int
    best_mean_cosine: float
    gain_over_temporal_n_z_3: float
    temporal_capacity_is_flat: bool
    zero_norm_prediction_count: int
    temporal_capacity_claim_exit: str
    observations: tuple[TemporalCapacityObservation, ...]
    description: str


def _flatten_dyad(dyad: MSCDyad) -> tuple[MSCDialogueTurn, ...]:
    return tuple(
        MSCDialogueTurn(
            session_index=session.session_index,
            speaker=utterance.speaker,
            text=utterance.text,
            utterance_index=utterance.utterance_index,
            preceding_empty_utterance_count=(
                utterance.preceding_empty_utterance_count
            ),
        )
        for session in dyad.sessions
        for utterance in session.utterances
    )


def build_msc_next_turn_examples(
    dyads: tuple[MSCDyad, ...],
    *,
    target_speaker: str = "speaker_1",
) -> tuple[MSCNextTurnExample, ...]:
    """Freeze human N+1 targets without manufacturing labels.

    The first MSC role is treated consistently as the predicted person.  Each
    sample ends immediately before one of that role's observed utterances; all
    prior sessions remain in ``history``.
    """

    if target_speaker not in {"speaker_1", "speaker_2"}:
        raise ValueError("target_speaker must be speaker_1 or speaker_2")
    examples: list[MSCNextTurnExample] = []
    seen_ids: set[str] = set()
    persona_position = 0 if target_speaker == "speaker_1" else 1
    for dyad in dyads:
        turns = _flatten_dyad(dyad)
        for target_position in range(1, len(turns)):
            target = turns[target_position]
            if target.speaker != target_speaker:
                continue
            sample_id = (
                f"{dyad.dyad_id}:s{target.session_index}:"
                f"u{target.utterance_index}:p{target_position}"
            )
            if sample_id in seen_ids:
                raise ValueError(f"duplicate MSC prediction sample id {sample_id!r}")
            seen_ids.add(sample_id)
            history = turns[:target_position]
            examples.append(
                MSCNextTurnExample(
                    sample_id=sample_id,
                    dyad_id=dyad.dyad_id,
                    split=dyad.split,
                    target_speaker=target_speaker,
                    session_index=target.session_index,
                    history_turns=len(history),
                    personas=dyad.initial_personas[persona_position],
                    history=history,
                    target_text=target.text,
                )
            )
    return tuple(examples)


def render_stateless_context(example: MSCNextTurnExample) -> str:
    persona = "\n".join(f"- {item}" for item in example.personas)
    return (
        f"Predicted-person persona:\n{persona}\n\n"
        f"Latest partner message:\n{example.latest_text}"
    )


def render_long_context(example: MSCNextTurnExample) -> str:
    persona = "\n".join(f"- {item}" for item in example.personas)
    lines = [f"Predicted-person persona:\n{persona}"]
    active_session = 0
    for turn in example.history:
        if turn.session_index != active_session:
            active_session = turn.session_index
            lines.append(f"\n[session {active_session}]")
        if turn.preceding_empty_utterance_count:
            lines.append(
                "[omitted empty turns: "
                f"{turn.preceding_empty_utterance_count}]"
            )
        lines.append(f"{turn.speaker}: {turn.text}")
    return "\n".join(lines)


def render_summary_retrieval_context(
    example: MSCNextTurnExample,
    *,
    retrieved_turns: tuple[MSCDialogueTurn, ...],
) -> str:
    persona = "\n".join(f"- {item}" for item in example.personas)
    memories = "\n".join(
        f"[session {turn.session_index}] {turn.speaker}: {turn.text}"
        for turn in retrieved_turns
    )
    return (
        f"Predicted-person persona summary:\n{persona}\n\n"
        f"Retrieved relationship memories:\n{memories or '[none]'}\n\n"
        f"Latest partner message:\n{example.latest_text}"
    )


def examples_fingerprint(examples: tuple[MSCNextTurnExample, ...]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.target_text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot compute a mean over no values")
    return sum(materialized) / len(materialized)


def _bootstrap_mean_interval(
    values: tuple[float, ...], *, resamples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    if resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    rng = random.Random(seed)
    estimates = sorted(
        _mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(resamples)
    )
    lower = estimates[int(0.025 * (resamples - 1))]
    upper = estimates[int(0.975 * (resamples - 1))]
    return (lower, upper)


def _slope(points: tuple[tuple[float, float], ...]) -> float:
    if len(points) < 2:
        return 0.0
    mean_x = _mean(point[0] for point in points)
    mean_y = _mean(point[1] for point in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 1e-12:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def _validate_matched_observations(
    observations: tuple[PredictionObservation, ...],
) -> None:
    if not observations:
        raise ValueError("prediction experiment requires observations")
    arms = {observation.arm for observation in observations}
    if arms != set(PREDICTION_ARMS):
        raise ValueError(
            "prediction experiment requires exactly all four arms; "
            f"got {sorted(arms)!r}"
        )
    keys_by_arm_seed: dict[tuple[str, int], set[str]] = defaultdict(set)
    for observation in observations:
        key = (observation.arm, observation.seed)
        if observation.sample_id in keys_by_arm_seed[key]:
            raise ValueError(
                f"duplicate prediction observation for {key!r}/{observation.sample_id}"
            )
        keys_by_arm_seed[key].add(observation.sample_id)
    reference = next(iter(keys_by_arm_seed.values()))
    for key, sample_ids in keys_by_arm_seed.items():
        if sample_ids != reference:
            raise ValueError(f"prediction arms/seeds are not sample-matched: {key!r}")


def adjudicate_prediction_experiment(
    observations: tuple[PredictionObservation, ...],
    *,
    heldout_sorted_id_sha256: str,
    encoder_fingerprint: str,
    volvence_full_stack: bool,
    same_substrate_context: bool = False,
    temporal_controller_capacity: bool = False,
    formal_preregistered: bool = False,
    thresholds: PredictionThresholds | None = None,
) -> PredictionExperimentVerdict:
    thresholds = thresholds or PredictionThresholds()
    _validate_matched_observations(observations)
    if not encoder_fingerprint.strip():
        raise ValueError("prediction experiment encoder_fingerprint is required")

    grouped: dict[tuple[str, int], list[PredictionObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.arm, observation.session_index)].append(observation)
    curves = tuple(
        SessionPredictionCurve(
            arm=arm,
            session_index=session,
            observation_count=len(rows),
            dyad_count=len({row.dyad_id for row in rows}),
            mean_cosine_similarity=_mean(row.cosine_similarity for row in rows),
            mean_squared_error=_mean(row.mean_squared_error for row in rows),
            mean_context_tokens=_mean(row.context_token_count for row in rows),
            mean_truncated_tokens=_mean(
                row.context_truncated_tokens for row in rows
            ),
            mean_latency_ms=_mean(row.latency_ms for row in rows),
            zero_norm_prediction_count=sum(
                row.prediction_zero_norm for row in rows
            ),
        )
        for (arm, session), rows in sorted(grouped.items())
    )

    by_key = {
        (row.arm, row.seed, row.sample_id): row for row in observations
    }
    sessions = sorted({row.session_index for row in observations})
    paired: list[PairedAdvantage] = []
    for session in sessions:
        session_rows = tuple(
            row
            for row in observations
            if row.arm == "volvence" and row.session_index == session
        )
        per_dyad: dict[str, list[float]] = defaultdict(list)
        for row in session_rows:
            control = by_key[("long_context", row.seed, row.sample_id)]
            per_dyad[row.dyad_id].append(
                row.cosine_similarity - control.cosine_similarity
            )
        dyad_deltas = tuple(
            _mean(values) for _, values in sorted(per_dyad.items())
        )
        paired.append(
            PairedAdvantage(
                session_index=session,
                pair_count=len(session_rows),
                dyad_count=len(dyad_deltas),
                mean_cosine_advantage=_mean(dyad_deltas),
                confidence_interval_95=_bootstrap_mean_interval(
                    dyad_deltas,
                    resamples=thresholds.bootstrap_resamples,
                    seed=thresholds.bootstrap_seed + session,
                ),
            )
        )

    longest = max(sessions)
    longest_advantage = next(
        item for item in paired if item.session_index == longest
    )
    advantage_slope = _slope(
        tuple(
            (float(item.session_index), item.mean_cosine_advantage)
            for item in paired
        )
    )
    curve_by_key = {(curve.arm, curve.session_index): curve for curve in curves}
    volv_longest = curve_by_key[("volvence", longest)]
    context_longest = curve_by_key[("long_context", longest)]
    token_ratio = volv_longest.mean_context_tokens / max(
        context_longest.mean_context_tokens, 1e-12
    )
    latency_ratio = volv_longest.mean_latency_ms / max(
        context_longest.mean_latency_ms, 1e-12
    )

    arm_set = {row.arm for row in observations}
    seeds = {row.seed for row in observations}
    heldout_dyads = {row.dyad_id for row in observations}
    observed_id_payload = "\n".join(sorted(heldout_dyads)) + "\n"
    observed_id_sha256 = hashlib.sha256(
        observed_id_payload.encode("utf-8")
    ).hexdigest()
    zero_norm_prediction_count = sum(
        row.prediction_zero_norm for row in observations
    )
    requirements = (
        (
            "official-heldout-hash",
            heldout_sorted_id_sha256 == thresholds.formal_heldout_id_sha256,
        ),
        (
            "observation-heldout-id-hash",
            observed_id_sha256 == thresholds.formal_heldout_id_sha256,
        ),
        ("all-four-arms", arm_set == set(PREDICTION_ARMS)),
        ("same-substrate-zero-truncation-context", same_substrate_context),
        ("volvence-full-stack", volvence_full_stack),
        ("temporal-controller-capacity-ladder", temporal_controller_capacity),
        ("formal-preregistration", formal_preregistered),
        ("minimum-three-seeds", len(seeds) >= thresholds.formal_min_seeds),
        (
            "complete-heldout-dyads",
            len(heldout_dyads) == thresholds.formal_heldout_dyads,
        ),
        (
            "longest-session-is-preregistered-session-five",
            longest == thresholds.formal_longest_session,
        ),
        ("zero-norm-prediction-count-zero", zero_norm_prediction_count == 0),
        ("frozen-encoder-fingerprint", bool(encoder_fingerprint.strip())),
    )
    formal = all(passed for _, passed in requirements)
    quality = (
        longest_advantage.mean_cosine_advantage
        >= thresholds.quality_min_cosine_advantage
        and longest_advantage.confidence_interval_95[0] > 0.0
        and advantage_slope > thresholds.quality_min_advantage_slope
    )
    scaling = (
        longest_advantage.mean_cosine_advantage
        >= thresholds.scaling_min_cosine_equivalence
        and token_ratio <= thresholds.scaling_max_token_ratio
        and latency_ratio <= thresholds.scaling_max_latency_ratio
    )
    evidence_level = "formal" if formal else "pilot"
    if zero_norm_prediction_count:
        thesis_exit = "ZERO_NORM_HEAD_COLLAPSE"
    elif not formal:
        thesis_exit = "INELIGIBLE_PILOT"
    elif quality:
        thesis_exit = "QUALITY_ADVANTAGE"
    elif scaling:
        thesis_exit = "SCALING_ADVANTAGE"
    else:
        thesis_exit = "REJECT_AND_SIMPLIFY"
    return PredictionExperimentVerdict(
        evidence_level=evidence_level,
        quality_condition_met=formal and quality,
        scaling_condition_met=formal and scaling,
        thesis_exit=thesis_exit,
        longest_session=longest,
        longest_quality_advantage=longest_advantage.mean_cosine_advantage,
        longest_quality_confidence_interval_95=(
            longest_advantage.confidence_interval_95
        ),
        advantage_slope=advantage_slope,
        longest_token_ratio=token_ratio,
        longest_latency_ratio=latency_ratio,
        zero_norm_prediction_count=zero_norm_prediction_count,
        formal_requirements=requirements,
        curves=curves,
        paired_advantages=tuple(paired),
        description=(
            f"MSC N+1 four-arm adjudication is {evidence_level}; "
            f"thesis_exit={thesis_exit}, longest-session cosine advantage="
            f"{longest_advantage.mean_cosine_advantage:.6f}, token ratio="
            f"{token_ratio:.4f}, latency ratio={latency_ratio:.4f}, "
            f"zero-norm predictions={zero_norm_prediction_count}."
        ),
    )


def adjudicate_capacity_ladder(
    observations: tuple[CapacityObservation, ...],
    *,
    complete_train: bool,
    complete_validation: bool,
    minimum_gain: float = 0.01,
) -> CapacityLadderVerdict:
    if not observations:
        raise ValueError("capacity ladder requires observations")
    expected = {3, 16, 64, 256}
    if {row.forward_head_n_z for row in observations} != expected:
        raise ValueError(
            "capacity ladder must contain forward_head_n_z 3/16/64/256"
        )
    validation = tuple(row for row in observations if row.split == "validation")
    keys = {(row.forward_head_n_z, row.seed) for row in validation}
    if len(keys) != len(validation):
        raise ValueError("capacity ladder contains duplicate n_z/seed rows")
    seeds_by_nz = {
        n_z: {
            row.seed for row in validation if row.forward_head_n_z == n_z
        }
        for n_z in expected
    }
    if len({frozenset(seeds) for seeds in seeds_by_nz.values()}) != 1:
        raise ValueError("capacity ladder seeds must be matched across n_z")
    means = {
        n_z: _mean(
            row.mean_cosine_similarity
            for row in validation
            if row.forward_head_n_z == n_z
        )
        for n_z in expected
    }
    best_n_z = max(sorted(means), key=means.__getitem__)
    gain = means[best_n_z] - means[3]
    formal = complete_train and complete_validation and len(seeds_by_nz[3]) >= 3
    flat = gain < minimum_gain
    zero_norm_prediction_count = sum(
        row.zero_norm_prediction_count for row in validation
    )
    chosen_n_z = 3 if flat or zero_norm_prediction_count else best_n_z
    if not formal:
        head_exit = "INELIGIBLE_PILOT"
    elif zero_norm_prediction_count:
        head_exit = "FAIL_ZERO_NORM_PREDICTIONS"
    elif flat:
        head_exit = "KEEP_MINIMAL_FORWARD_HEAD"
    else:
        head_exit = f"SELECT_FORWARD_HEAD_N_Z_{best_n_z}"
    return CapacityLadderVerdict(
        evidence_level="formal" if formal else "pilot",
        best_forward_head_n_z=best_n_z,
        chosen_forward_head_n_z=chosen_n_z,
        best_mean_cosine=means[best_n_z],
        gain_over_forward_head_n_z_3=gain,
        forward_head_capacity_is_flat=(
            formal and not zero_norm_prediction_count and flat
        ),
        zero_norm_prediction_count=zero_norm_prediction_count,
        forward_head_claim_exit=head_exit,
        observations=observations,
        description=(
            "Real-target forward-head capacity ladder is "
            f"{'formal' if formal else 'pilot'}; "
            f"best forward_head_n_z={best_n_z}, cosine gain over "
            f"forward_head_n_z=3 is {gain:.6f}, chosen={chosen_n_z}, "
            f"zero-norm predictions={zero_norm_prediction_count}, "
            f"exit={head_exit}. "
            "This does not test temporal-controller capacity."
        ),
    )


def adjudicate_temporal_capacity_ladder(
    observations: tuple[TemporalCapacityObservation, ...],
    *,
    complete_train: bool,
    complete_validation: bool,
    minimum_gain: float = 0.01,
) -> TemporalCapacityLadderVerdict:
    """Adjudicate R5 while holding the PE forward head fixed at n_z=3."""

    if not observations:
        raise ValueError("temporal capacity ladder requires observations")
    expected = {3, 16, 64, 256}
    if {row.temporal_n_z for row in observations} != expected:
        raise ValueError("temporal capacity ladder must contain n_z 3/16/64/256")
    if {row.forward_head_n_z for row in observations} != {3}:
        raise ValueError("temporal capacity ladder changed PE forward-head capacity")
    keys = {(row.temporal_n_z, row.seed) for row in observations}
    if len(keys) != len(observations):
        raise ValueError("temporal capacity ladder contains duplicate n_z/seed rows")
    seeds_by_nz = {
        n_z: {row.seed for row in observations if row.temporal_n_z == n_z}
        for n_z in expected
    }
    if len({frozenset(seeds) for seeds in seeds_by_nz.values()}) != 1:
        raise ValueError("temporal capacity ladder seeds must be matched across n_z")
    means = {
        n_z: _mean(
            row.mean_cosine_similarity
            for row in observations
            if row.temporal_n_z == n_z
        )
        for n_z in expected
    }
    best_n_z = max(sorted(means), key=means.__getitem__)
    gain = means[best_n_z] - means[3]
    flat = gain < minimum_gain
    chosen_n_z = 3 if flat else best_n_z
    zero_norm_prediction_count = sum(
        row.zero_norm_prediction_count for row in observations
    )
    complete = (
        complete_train
        and complete_validation
        and len(seeds_by_nz[3]) >= 3
    )
    integrity_passed = complete and zero_norm_prediction_count == 0
    if not complete:
        capacity_exit = "INELIGIBLE_PILOT"
    elif zero_norm_prediction_count:
        capacity_exit = "FAIL_ZERO_NORM_PREDICTIONS"
        chosen_n_z = 3
    elif flat:
        capacity_exit = "KEEP_MINIMAL_TEMPORAL_CONTROLLER"
    else:
        capacity_exit = f"SELECT_TEMPORAL_N_Z_{best_n_z}"
    return TemporalCapacityLadderVerdict(
        evidence_level="formal" if complete else "pilot",
        capacity_integrity_passed=integrity_passed,
        best_temporal_n_z=best_n_z,
        chosen_temporal_n_z=chosen_n_z,
        fixed_forward_head_n_z=3,
        best_mean_cosine=means[best_n_z],
        gain_over_temporal_n_z_3=gain,
        temporal_capacity_is_flat=integrity_passed and flat,
        zero_norm_prediction_count=zero_norm_prediction_count,
        temporal_capacity_claim_exit=capacity_exit,
        observations=observations,
        description=(
            "R5 temporal-controller capacity ladder is "
            f"{'formal' if complete else 'pilot'}; fixed forward_head_n_z=3, "
            f"best temporal_n_z={best_n_z}, chosen={chosen_n_z}, cosine gain "
            f"over temporal_n_z=3 is {gain:.6f}, zero-norm predictions="
            f"{zero_norm_prediction_count}, exit={capacity_exit}."
        ),
    )


__all__ = (
    "MSC_HELDOUT_ID_SHA256",
    "PREDICTION_ARMS",
    "CapacityLadderVerdict",
    "CapacityObservation",
    "MSCDialogueTurn",
    "MSCFullRuntimeContext",
    "MSCNextTurnExample",
    "PairedAdvantage",
    "PredictionExperimentVerdict",
    "PredictionObservation",
    "PredictionThresholds",
    "SameSubstrateContextAttestation",
    "SessionPredictionCurve",
    "TemporalCapacityLadderVerdict",
    "TemporalCapacityObservation",
    "adjudicate_capacity_ladder",
    "adjudicate_prediction_experiment",
    "adjudicate_temporal_capacity_ladder",
    "build_msc_next_turn_examples",
    "examples_fingerprint",
    "parse_msc_full_runtime_context",
    "render_long_context",
    "render_stateless_context",
    "render_summary_retrieval_context",
)
