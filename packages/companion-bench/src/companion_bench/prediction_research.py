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
from typing import Iterable

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
        return self.history[-1].text


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
    formal_requirements: tuple[tuple[str, bool], ...]
    curves: tuple[SessionPredictionCurve, ...]
    paired_advantages: tuple[PairedAdvantage, ...]
    description: str


@dataclass(frozen=True)
class CapacityObservation:
    n_z: int
    seed: int
    split: str
    mean_cosine_similarity: float
    mean_squared_error: float

    def __post_init__(self) -> None:
        if self.n_z not in {3, 16, 64, 256}:
            raise ValueError("capacity observation n_z must be one of 3/16/64/256")
        if self.seed < 0 or self.split not in {"validation", "heldout"}:
            raise ValueError("capacity observation seed/split is invalid")
        if not math.isfinite(self.mean_cosine_similarity) or not math.isfinite(
            self.mean_squared_error
        ):
            raise ValueError("capacity metrics must be finite")


@dataclass(frozen=True)
class CapacityLadderVerdict:
    evidence_level: str
    best_n_z: int
    best_mean_cosine: float
    gain_over_n_z_3: float
    capacity_is_flat: bool
    eta_claim_exit: str
    observations: tuple[CapacityObservation, ...]
    description: str


def _flatten_dyad(dyad: MSCDyad) -> tuple[MSCDialogueTurn, ...]:
    return tuple(
        MSCDialogueTurn(
            session_index=session.session_index,
            speaker=utterance.speaker,
            text=utterance.text,
            utterance_index=utterance.utterance_index,
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
        ("volvence-full-stack", volvence_full_stack),
        ("minimum-three-seeds", len(seeds) >= thresholds.formal_min_seeds),
        (
            "complete-heldout-dyads",
            len(heldout_dyads) == thresholds.formal_heldout_dyads,
        ),
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
    if not formal:
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
        formal_requirements=requirements,
        curves=curves,
        paired_advantages=tuple(paired),
        description=(
            f"MSC N+1 four-arm adjudication is {evidence_level}; "
            f"thesis_exit={thesis_exit}, longest-session cosine advantage="
            f"{longest_advantage.mean_cosine_advantage:.6f}, token ratio="
            f"{token_ratio:.4f}, latency ratio={latency_ratio:.4f}."
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
    if {row.n_z for row in observations} != expected:
        raise ValueError("capacity ladder must contain n_z 3/16/64/256")
    validation = tuple(row for row in observations if row.split == "validation")
    seeds_by_nz = {
        n_z: {row.seed for row in validation if row.n_z == n_z}
        for n_z in expected
    }
    if len({frozenset(seeds) for seeds in seeds_by_nz.values()}) != 1:
        raise ValueError("capacity ladder seeds must be matched across n_z")
    means = {
        n_z: _mean(
            row.mean_cosine_similarity for row in validation if row.n_z == n_z
        )
        for n_z in expected
    }
    best_n_z = max(sorted(means), key=means.__getitem__)
    gain = means[best_n_z] - means[3]
    formal = complete_train and complete_validation and len(seeds_by_nz[3]) >= 3
    flat = gain < minimum_gain
    if not formal:
        eta_exit = "INELIGIBLE_PILOT"
    elif flat:
        eta_exit = "KILL_ETA_CAPACITY_CLAIM"
    else:
        eta_exit = f"PROMOTE_N_Z_{best_n_z}"
    return CapacityLadderVerdict(
        evidence_level="formal" if formal else "pilot",
        best_n_z=best_n_z,
        best_mean_cosine=means[best_n_z],
        gain_over_n_z_3=gain,
        capacity_is_flat=formal and flat,
        eta_claim_exit=eta_exit,
        observations=observations,
        description=(
            f"Real-target capacity ladder is {'formal' if formal else 'pilot'}; "
            f"best n_z={best_n_z}, cosine gain over n_z=3 is {gain:.6f}, "
            f"exit={eta_exit}."
        ),
    )


__all__ = (
    "MSC_HELDOUT_ID_SHA256",
    "PREDICTION_ARMS",
    "CapacityLadderVerdict",
    "CapacityObservation",
    "MSCDialogueTurn",
    "MSCNextTurnExample",
    "PairedAdvantage",
    "PredictionExperimentVerdict",
    "PredictionObservation",
    "PredictionThresholds",
    "SessionPredictionCurve",
    "adjudicate_capacity_ladder",
    "adjudicate_prediction_experiment",
    "build_msc_next_turn_examples",
    "examples_fingerprint",
    "render_long_context",
    "render_stateless_context",
    "render_summary_retrieval_context",
)
