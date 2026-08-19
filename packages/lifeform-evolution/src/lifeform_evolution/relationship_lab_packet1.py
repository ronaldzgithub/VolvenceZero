"""Relationship Lab P1: strong baselines and Appendable evidence.

The four arms share one frozen substrate instance. They differ only in the
public context surface handed to the model. This module owns read-only
evaluation records and gate verdicts; it does not write PE, credit, semantic
owners, regime, or steering state.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol

from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    RelationshipDatasetSplit,
    RelationshipTransferDataset,
    canonical_json,
    load_relationship_transfer_dataset,
    sha256_json,
)
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_contexts import (
    RELATIONSHIP_P1_ARMS,
    PersistedRelationshipP1StateDigest,
    RelationshipP1Arm,
    RelationshipP1ConsoleControlEvidence,
    RelationshipP1ContextBundle,
    relationship_p1_structural_metrics,
)
from lifeform_evolution.relationship_lab_gate0 import FrozenBaselineAttestation


RELATIONSHIP_PACKET1_DECISION_SCHEMA_VERSION = "relationship-p1-decision.v1"
RELATIONSHIP_PACKET1_RUN_SCHEMA_VERSION = "relationship-p1-run.v1"
RELATIONSHIP_PACKET1_REPORT_SCHEMA_VERSION = "relationship-p1-report.v3"


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def relationship_p1_prompt_path(arm: RelationshipP1Arm) -> pathlib.Path:
    names = {
        RelationshipP1Arm.STATELESS: "relationship_lab_stateless_v1.txt",
        RelationshipP1Arm.PROMPT_STEELMAN: ("relationship_lab_full_history_steelman_v2.txt"),
        RelationshipP1Arm.RAG_STEELMAN: "relationship_lab_rag_steelman_v2.txt",
        RelationshipP1Arm.STRUCTURED_STATE: ("relationship_lab_structured_state_v2.txt"),
    }
    return _asset_dir() / "prompts" / names[arm]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ContextualRelationshipActionPolicy(Protocol):
    model_id: str
    weights_sha256: str
    prompt_sha256: str
    generation_config_sha256: str

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion: ...

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class RelationshipP1Decision:
    decision_id: str
    arm: RelationshipP1Arm
    scene_id: str
    mirror_pair_id: str
    split: RelationshipDatasetSplit
    seed: int
    current_input_sha256: str
    context_sha256: str
    arm_prompt_sha256: str
    raw_output: str
    chosen_action_id: RelationshipAction | None
    expected_action_id: RelationshipAction
    valid: bool
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    schema_version: str = RELATIONSHIP_PACKET1_DECISION_SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "arm": self.arm.value,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "split": self.split.value,
            "seed": self.seed,
            "current_input_sha256": self.current_input_sha256,
            "context_sha256": self.context_sha256,
            "arm_prompt_sha256": self.arm_prompt_sha256,
            "raw_output": self.raw_output,
            "chosen_action_id": (self.chosen_action_id.value if self.chosen_action_id is not None else None),
            "expected_action_id": self.expected_action_id.value,
            "valid": self.valid,
            "correct": self.correct,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class RelationshipP1ContextTokenCount:
    arm: RelationshipP1Arm
    scene_id: str
    background_depth: int
    context_tokens: int

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "scene_id": self.scene_id,
            "background_depth": self.background_depth,
            "context_tokens": self.context_tokens,
        }


@dataclass(frozen=True)
class RelationshipP1Run:
    dataset_fingerprint: str
    context_bundle_artifact_id: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    seed_schedule: tuple[int, ...]
    arm_prompt_hashes: tuple[tuple[str, str], ...]
    decisions: tuple[RelationshipP1Decision, ...]
    context_token_counts: tuple[RelationshipP1ContextTokenCount, ...]
    schema_version: str = RELATIONSHIP_PACKET1_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PACKET1_RUN_SCHEMA_VERSION:
            raise ValueError("P1 run schema_version mismatch")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(self.seed_schedule):
            raise ValueError("P1 seed_schedule must be non-empty and unique")
        decision_ids = tuple(item.decision_id for item in self.decisions)
        if not self.decisions or len(set(decision_ids)) != len(decision_ids):
            raise ValueError("P1 decision ids must be non-empty and unique")

    @property
    def seed_schedule_sha256(self) -> str:
        return sha256_json(self.seed_schedule)

    def decision_ledger_jsonl(self) -> str:
        return "".join(canonical_json(item.to_payload()) + "\n" for item in self.decisions)

    @property
    def decision_ledger_sha256(self) -> str:
        return hashlib.sha256(self.decision_ledger_jsonl().encode("utf-8")).hexdigest()

    def decisions_for_arm(self, arm: RelationshipP1Arm) -> tuple[RelationshipP1Decision, ...]:
        return tuple(item for item in self.decisions if item.arm is arm)

    def arm_metrics(self, arm: RelationshipP1Arm) -> dict[str, object]:
        rows = self.decisions_for_arm(arm)
        if not rows:
            raise ValueError(f"P1 run has no decisions for {arm.value}")
        valid = sum(int(item.valid) for item in rows)
        correct = sum(int(item.correct) for item in rows)
        grouped: dict[tuple[str, int], list[RelationshipP1Decision]] = {}
        for item in rows:
            grouped.setdefault((item.mirror_pair_id, item.seed), []).append(item)
        flip_groups = 0
        valid_groups = 0
        for group in grouped.values():
            if len(group) != 2:
                raise ValueError("P1 mirrored pair decision group must contain two rows")
            if all(item.chosen_action_id is not None for item in group):
                valid_groups += 1
                flip_groups += int(
                    {item.chosen_action_id for item in group}
                    == {
                        RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                        RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                    }
                )
        return {
            "decisions": len(rows),
            "valid_decisions": valid,
            "valid_rate": valid / len(rows),
            "correct_decisions": correct,
            "accuracy": correct / len(rows),
            "pair_groups": len(grouped),
            "valid_pair_groups": valid_groups,
            "pair_flip_rate": (flip_groups / valid_groups if valid_groups else 0.0),
        }

    def to_summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "context_bundle_artifact_id": self.context_bundle_artifact_id,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "seed_schedule": list(self.seed_schedule),
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "arm_prompt_hashes": dict(self.arm_prompt_hashes),
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "arm_metrics": {arm.value: self.arm_metrics(arm) for arm in RELATIONSHIP_P1_ARMS},
            "context_token_counts": [item.to_payload() for item in self.context_token_counts],
        }


def relationship_p1_completion_to_decision(
    *,
    completion: StatelessActionCompletion,
    arm: RelationshipP1Arm,
    scene_id: str,
    mirror_pair_id: str,
    split: RelationshipDatasetSplit,
    seed: int,
    current_input_sha256: str,
    context_sha256: str,
    arm_prompt_sha256: str,
    expected_action_id: RelationshipAction,
    model_id: str,
) -> RelationshipP1Decision:
    decision_id = sha256_json(
        {
            "model_id": model_id,
            "arm": arm.value,
            "scene_id": scene_id,
            "seed": seed,
            "current_input_sha256": current_input_sha256,
            "context_sha256": context_sha256,
            "arm_prompt_sha256": arm_prompt_sha256,
        }
    )
    valid = completion.chosen_action_id is not None
    return RelationshipP1Decision(
        decision_id=decision_id,
        arm=arm,
        scene_id=scene_id,
        mirror_pair_id=mirror_pair_id,
        split=split,
        seed=seed,
        current_input_sha256=current_input_sha256,
        context_sha256=context_sha256,
        arm_prompt_sha256=arm_prompt_sha256,
        raw_output=completion.raw_output,
        chosen_action_id=completion.chosen_action_id,
        expected_action_id=expected_action_id,
        valid=valid,
        correct=completion.chosen_action_id is expected_action_id,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )


def run_relationship_packet1_arms(
    policy: ContextualRelationshipActionPolicy,
    *,
    contexts: RelationshipP1ContextBundle,
    dataset: RelationshipTransferDataset | None = None,
    seed_schedule: tuple[int, ...] = (101,),
    decision_observer: Callable[[RelationshipP1Decision], None] | None = None,
) -> RelationshipP1Run:
    effective_dataset = dataset or load_relationship_transfer_dataset()
    if contexts.dataset_fingerprint != effective_dataset.dataset_fingerprint:
        raise ValueError("P1 context bundle dataset fingerprint mismatch")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("P1 seed_schedule must be non-empty and unique")
    prompts = {
        arm: relationship_p1_prompt_path(arm).read_text(encoding="utf-8").strip() for arm in RELATIONSHIP_P1_ARMS
    }
    prompt_hashes = {arm: _sha256_file(relationship_p1_prompt_path(arm)) for arm in RELATIONSHIP_P1_ARMS}
    if prompt_hashes[RelationshipP1Arm.STATELESS] != policy.prompt_sha256:
        raise ValueError("P1 stateless prompt does not match frozen Gate 0 policy")

    decisions: list[RelationshipP1Decision] = []
    allowed_splits = {
        RelationshipDatasetSplit.TRAIN,
        RelationshipDatasetSplit.VALIDATION,
    }
    for mirror_pair_id, members in effective_dataset.mirrored_pairs():
        split = members[0][1].split
        if split not in allowed_splits:
            continue
        current_input = members[0][0].current_input
        current_input_sha256 = hashlib.sha256(current_input.encode("utf-8")).hexdigest()
        for seed in seed_schedule:
            stateless_completion = policy.choose(
                current_input=current_input,
                seed=seed,
            )
            for observation, dynamic in members:
                stateless_context = contexts.context(
                    scene_id=observation.scene_id,
                    arm=RelationshipP1Arm.STATELESS,
                )
                decision = relationship_p1_completion_to_decision(
                    completion=stateless_completion,
                    arm=RelationshipP1Arm.STATELESS,
                    scene_id=observation.scene_id,
                    mirror_pair_id=mirror_pair_id,
                    split=split,
                    seed=seed,
                    current_input_sha256=current_input_sha256,
                    context_sha256=stateless_context.context_sha256,
                    arm_prompt_sha256=prompt_hashes[RelationshipP1Arm.STATELESS],
                    expected_action_id=dynamic.preferred_action,
                    model_id=policy.model_id,
                )
                decisions.append(decision)
                if decision_observer is not None:
                    decision_observer(decision)
            for arm in (
                RelationshipP1Arm.PROMPT_STEELMAN,
                RelationshipP1Arm.RAG_STEELMAN,
                RelationshipP1Arm.STRUCTURED_STATE,
            ):
                for observation, dynamic in members:
                    context = contexts.context(
                        scene_id=observation.scene_id,
                        arm=arm,
                    )
                    completion = policy.choose_from_messages(
                        messages=(
                            {"role": "system", "content": prompts[arm]},
                            {
                                "role": "user",
                                "content": context.render_user_message(observation.current_input),
                            },
                        ),
                        seed=seed,
                    )
                    decision = relationship_p1_completion_to_decision(
                        completion=completion,
                        arm=arm,
                        scene_id=observation.scene_id,
                        mirror_pair_id=mirror_pair_id,
                        split=split,
                        seed=seed,
                        current_input_sha256=current_input_sha256,
                        context_sha256=context.context_sha256,
                        arm_prompt_sha256=prompt_hashes[arm],
                        expected_action_id=dynamic.preferred_action,
                        model_id=policy.model_id,
                    )
                    decisions.append(decision)
                    if decision_observer is not None:
                        decision_observer(decision)

    token_counts = tuple(
        RelationshipP1ContextTokenCount(
            arm=context.arm,
            scene_id=context.scene_id,
            background_depth=context.background_depth,
            context_tokens=policy.count_tokens(context.context_text),
        )
        for context in contexts.contexts
    )
    return RelationshipP1Run(
        dataset_fingerprint=effective_dataset.dataset_fingerprint,
        context_bundle_artifact_id=contexts.artifact_id,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule=seed_schedule,
        arm_prompt_hashes=tuple(sorted((arm.value, digest) for arm, digest in prompt_hashes.items())),
        decisions=tuple(decisions),
        context_token_counts=token_counts,
    )


@dataclass(frozen=True)
class RelationshipP1RecoveryEvidence:
    expected_state_artifact_id: str
    recovered_state_artifact_id: str
    fresh_process: bool

    @property
    def passed(self) -> bool:
        return self.fresh_process and self.expected_state_artifact_id == self.recovered_state_artifact_id

    def to_payload(self) -> dict[str, object]:
        return {
            "expected_state_artifact_id": self.expected_state_artifact_id,
            "recovered_state_artifact_id": self.recovered_state_artifact_id,
            "fresh_process": self.fresh_process,
            "passed": self.passed,
        }


class P1CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class RelationshipP1GateConfig:
    minimum_decisions_per_arm: int = 8
    minimum_steelman_accuracy: float = 0.625
    maximum_steelman_accuracy: float = 0.875
    minimum_steelman_pair_flip_rate: float = 0.5
    minimum_structured_state_pair_flip_rate: float = 0.5
    maximum_rag_to_full_history_token_ratio: float = 0.55
    maximum_structured_to_full_history_token_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.minimum_decisions_per_arm < 2:
            raise ValueError("minimum_decisions_per_arm must be >= 2")
        if not (0.0 <= self.minimum_steelman_accuracy < self.maximum_steelman_accuracy <= 1.0):
            raise ValueError("P1 steelman accuracy band is invalid")
        if not 0.0 <= self.minimum_steelman_pair_flip_rate <= 1.0:
            raise ValueError("P1 pair flip threshold must be in [0, 1]")
        if not 0.0 <= self.minimum_structured_state_pair_flip_rate <= 1.0:
            raise ValueError("P1 structured-state pair flip threshold must be in [0, 1]")
        for value in (
            self.maximum_rag_to_full_history_token_ratio,
            self.maximum_structured_to_full_history_token_ratio,
        ):
            if not 0.0 < value < 1.0:
                raise ValueError("P1 token ratios must be in (0, 1)")

    def to_payload(self) -> dict[str, object]:
        return {
            "minimum_decisions_per_arm": self.minimum_decisions_per_arm,
            "minimum_steelman_accuracy": self.minimum_steelman_accuracy,
            "maximum_steelman_accuracy": self.maximum_steelman_accuracy,
            "minimum_steelman_pair_flip_rate": (self.minimum_steelman_pair_flip_rate),
            "minimum_structured_state_pair_flip_rate": (self.minimum_structured_state_pair_flip_rate),
            "maximum_rag_to_full_history_token_ratio": (self.maximum_rag_to_full_history_token_ratio),
            "maximum_structured_to_full_history_token_ratio": (self.maximum_structured_to_full_history_token_ratio),
        }


@dataclass(frozen=True)
class RelationshipP1Check:
    check_id: str
    status: P1CheckStatus
    summary: str
    metrics: tuple[tuple[str, object], ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "summary": self.summary,
            "metrics": dict(self.metrics),
        }


def _mean_context_tokens(
    run: RelationshipP1Run,
    *,
    arm: RelationshipP1Arm,
    depth: int,
) -> float:
    values = [
        item.context_tokens for item in run.context_token_counts if item.arm is arm and item.background_depth == depth
    ]
    if not values:
        raise ValueError(f"no context token counts for {arm.value} depth={depth}")
    return statistics.fmean(values)


def _scaling_check(
    *,
    run: RelationshipP1Run,
    contexts: RelationshipP1ContextBundle,
    config: RelationshipP1GateConfig,
) -> RelationshipP1Check:
    minimum_depth = contexts.background_depths[0]
    maximum_depth = contexts.background_depths[-1]
    full_min = _mean_context_tokens(
        run,
        arm=RelationshipP1Arm.PROMPT_STEELMAN,
        depth=minimum_depth,
    )
    full_max = _mean_context_tokens(
        run,
        arm=RelationshipP1Arm.PROMPT_STEELMAN,
        depth=maximum_depth,
    )
    rag_min = _mean_context_tokens(
        run,
        arm=RelationshipP1Arm.RAG_STEELMAN,
        depth=minimum_depth,
    )
    rag_max = _mean_context_tokens(
        run,
        arm=RelationshipP1Arm.RAG_STEELMAN,
        depth=maximum_depth,
    )
    structured_min = _mean_context_tokens(
        run,
        arm=RelationshipP1Arm.STRUCTURED_STATE,
        depth=minimum_depth,
    )
    structured_max = _mean_context_tokens(
        run,
        arm=RelationshipP1Arm.STRUCTURED_STATE,
        depth=maximum_depth,
    )
    rag_ratio = rag_max / full_max
    structured_ratio = structured_max / full_max
    passed = (
        full_max > full_min
        and (rag_max - rag_min) < (full_max - full_min)
        and structured_max == structured_min
        and rag_ratio <= config.maximum_rag_to_full_history_token_ratio
        and structured_ratio <= config.maximum_structured_to_full_history_token_ratio
    )
    return RelationshipP1Check(
        check_id="token_scaling",
        status=P1CheckStatus.PASS if passed else P1CheckStatus.FAIL,
        summary=(
            "Full history grows with ordinary turns while ref-harness RAG and MemoryStore typed state remain bounded."
            if passed
            else "P1 bounded contexts did not beat full-history token growth."
        ),
        metrics=(
            ("minimum_depth", minimum_depth),
            ("maximum_depth", maximum_depth),
            ("full_history_tokens_at_min", round(full_min, 3)),
            ("full_history_tokens_at_max", round(full_max, 3)),
            ("rag_tokens_at_min", round(rag_min, 3)),
            ("rag_tokens_at_max", round(rag_max, 3)),
            ("structured_tokens_at_min", round(structured_min, 3)),
            ("structured_tokens_at_max", round(structured_max, 3)),
            ("rag_to_full_history_ratio", round(rag_ratio, 6)),
            ("structured_to_full_history_ratio", round(structured_ratio, 6)),
        ),
    )


def _steelman_check(
    *,
    run: RelationshipP1Run,
    config: RelationshipP1GateConfig,
) -> RelationshipP1Check:
    prompt = run.arm_metrics(RelationshipP1Arm.PROMPT_STEELMAN)
    rag = run.arm_metrics(RelationshipP1Arm.RAG_STEELMAN)
    return _steelman_check_from_metrics(
        prompt=prompt,
        rag=rag,
        config=config,
    )


def _steelman_check_from_metrics(
    *,
    prompt: dict[str, object],
    rag: dict[str, object],
    config: RelationshipP1GateConfig,
) -> RelationshipP1Check:
    prompt_qualified = (
        config.minimum_steelman_accuracy <= float(prompt["accuracy"]) <= config.maximum_steelman_accuracy
        and float(prompt["pair_flip_rate"]) >= config.minimum_steelman_pair_flip_rate
    )
    rag_qualified = (
        config.minimum_steelman_accuracy <= float(rag["accuracy"]) <= config.maximum_steelman_accuracy
        and float(rag["pair_flip_rate"]) >= config.minimum_steelman_pair_flip_rate
    )
    passed = prompt_qualified and rag_qualified
    return RelationshipP1Check(
        check_id="steelman_qualification",
        status=P1CheckStatus.PASS if passed else P1CheckStatus.FAIL,
        summary=(
            "Full-history and ref-harness RAG steelmen both use personal "
            "history without saturating the development set."
            if passed
            else "A steelman is too weak or the development set is saturated."
        ),
        metrics=(
            ("prompt_accuracy", round(float(prompt["accuracy"]), 6)),
            ("prompt_pair_flip_rate", round(float(prompt["pair_flip_rate"]), 6)),
            ("prompt_qualified", prompt_qualified),
            ("rag_accuracy", round(float(rag["accuracy"]), 6)),
            ("rag_pair_flip_rate", round(float(rag["pair_flip_rate"]), 6)),
            ("rag_qualified", rag_qualified),
        ),
    )


def _structured_state_user_swap_check(
    *,
    run: RelationshipP1Run,
    config: RelationshipP1GateConfig,
) -> RelationshipP1Check:
    stateless = run.arm_metrics(RelationshipP1Arm.STATELESS)
    structured = run.arm_metrics(RelationshipP1Arm.STRUCTURED_STATE)
    return _structured_state_user_swap_check_from_metrics(
        stateless=stateless,
        structured=structured,
        config=config,
    )


def _structured_state_user_swap_check_from_metrics(
    *,
    stateless: dict[str, object],
    structured: dict[str, object],
    config: RelationshipP1GateConfig,
) -> RelationshipP1Check:
    stateless_pair_flip_rate = float(stateless["pair_flip_rate"])
    structured_pair_flip_rate = float(structured["pair_flip_rate"])
    passed = (
        stateless_pair_flip_rate == 0.0 and structured_pair_flip_rate >= config.minimum_structured_state_pair_flip_rate
    )
    return RelationshipP1Check(
        check_id="structured_state_user_swap_effect",
        status=P1CheckStatus.PASS if passed else P1CheckStatus.FAIL,
        summary=(
            "The stateless arm cannot distinguish byte-identical users while "
            "the scoped structured state changes the selected action."
            if passed
            else "Persisted user state did not produce the preregistered mirrored-user action change."
        ),
        metrics=(
            ("stateless_pair_flip_rate", stateless_pair_flip_rate),
            (
                "structured_state_accuracy",
                round(float(structured["accuracy"]), 6),
            ),
            (
                "structured_state_pair_flip_rate",
                round(structured_pair_flip_rate, 6),
            ),
            (
                "minimum_structured_state_pair_flip_rate",
                config.minimum_structured_state_pair_flip_rate,
            ),
        ),
    )


@dataclass(frozen=True)
class RelationshipP1Report:
    created_at_iso: str
    dataset_fingerprint: str
    context_bundle_artifact_id: str
    decision_ledger_sha256: str
    gate0_baseline_attestation_id: str
    config: RelationshipP1GateConfig
    checks: tuple[RelationshipP1Check, ...]
    arm_metrics: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    machinery_ready: bool
    gate1_passed: bool
    source_report_artifact_id: str | None = None
    schema_version: str = RELATIONSHIP_PACKET1_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PACKET1_REPORT_SCHEMA_VERSION:
            raise ValueError("P1 report schema_version mismatch")
        if self.source_report_artifact_id is not None and (
            len(self.source_report_artifact_id) != 64
            or any(char not in "0123456789abcdef" for char in self.source_report_artifact_id)
        ):
            raise ValueError("P1 source report artifact id must be a sha256 digest")
        check_ids = tuple(item.check_id for item in self.checks)
        if not check_ids or len(set(check_ids)) != len(check_ids):
            raise ValueError("P1 report check ids must be non-empty and unique")
        arm_ids = tuple(arm for arm, _ in self.arm_metrics)
        if set(arm_ids) != {arm.value for arm in RELATIONSHIP_P1_ARMS}:
            raise ValueError("P1 report must contain every frozen arm exactly once")
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("P1 report arm metrics must be unique")
        if self.gate1_passed and not self.machinery_ready:
            raise ValueError("P1 Gate 1 cannot pass when machinery is not ready")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "dataset_fingerprint": self.dataset_fingerprint,
            "context_bundle_artifact_id": self.context_bundle_artifact_id,
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "gate0_baseline_attestation_id": self.gate0_baseline_attestation_id,
            "source_report_artifact_id": self.source_report_artifact_id,
            "config": self.config.to_payload(),
            "checks": [item.to_payload() for item in self.checks],
            "arm_metrics": {arm: dict(metrics) for arm, metrics in self.arm_metrics},
            "verdicts": {
                "machinery_ready": self.machinery_ready,
                "gate1_passed": self.gate1_passed,
            },
            "claim_boundary": (
                "P1 establishes bounded append/recover/control machinery and "
                "qualifies development baselines only. It does not prove a "
                "Readable ToM abstraction, PE-driven learning, steering, or "
                "formal hidden-test superiority."
            ),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def assess_relationship_packet1(
    *,
    run: RelationshipP1Run,
    contexts: RelationshipP1ContextBundle,
    recovery: RelationshipP1RecoveryEvidence,
    console: RelationshipP1ConsoleControlEvidence,
    gate0_baseline: FrozenBaselineAttestation,
    config: RelationshipP1GateConfig | None = None,
    dataset: RelationshipTransferDataset | None = None,
    created_at_iso: str | None = None,
) -> RelationshipP1Report:
    effective_config = config or RelationshipP1GateConfig()
    effective_dataset = dataset or load_relationship_transfer_dataset()
    structural = relationship_p1_structural_metrics(
        bundle=contexts,
        dataset=effective_dataset,
    )
    appendable_passed = recovery.passed and contexts.persisted_state.artifact_id == recovery.expected_state_artifact_id
    appendable_check = RelationshipP1Check(
        check_id="appendable_cross_process_recovery",
        status=P1CheckStatus.PASS if appendable_passed else P1CheckStatus.FAIL,
        summary=(
            "Fresh-process recovery reproduced every scoped MemoryStore and companion-ref-harness record digest."
            if appendable_passed
            else "Fresh-process persistence digest did not reproduce."
        ),
        metrics=tuple(sorted(recovery.to_payload().items())),
    )
    structural_check = RelationshipP1Check(
        check_id="user_swap_scope_isolation",
        status=(P1CheckStatus.PASS if bool(structural["passed"]) else P1CheckStatus.FAIL),
        summary=(
            "Mirrored current bytes stay fixed while user-scoped public histories remain distinct and isolated."
            if bool(structural["passed"])
            else "Mirrored context or user-scope isolation is broken."
        ),
        metrics=tuple(sorted(structural.items())),
    )
    console_check = RelationshipP1Check(
        check_id="console_correction_and_delete",
        status=P1CheckStatus.PASS if console.passed else P1CheckStatus.FAIL,
        summary=(
            "Owner-side correction and deletion survive restart without mutating the mirrored user's scope."
            if console.passed
            else "Correction/delete persistence or sibling isolation failed."
        ),
        metrics=tuple(sorted(console.to_payload().items())),
    )
    lineage_matches = (
        run.dataset_fingerprint == gate0_baseline.dataset_fingerprint
        and run.model_id == gate0_baseline.model_id
        and run.weights_sha256 == gate0_baseline.weights_sha256
        and run.generation_config_sha256 == gate0_baseline.generation_config_sha256
        and dict(run.arm_prompt_hashes)[RelationshipP1Arm.STATELESS.value] == gate0_baseline.prompt_sha256
    )
    per_arm = {arm: run.arm_metrics(arm) for arm in RELATIONSHIP_P1_ARMS}
    all_valid = all(
        metrics["valid_decisions"] == metrics["decisions"]
        and int(metrics["decisions"]) >= effective_config.minimum_decisions_per_arm
        for metrics in per_arm.values()
    )
    lineage_check = RelationshipP1Check(
        check_id="same_substrate_and_valid_outputs",
        status=(P1CheckStatus.PASS if lineage_matches and all_valid else P1CheckStatus.FAIL),
        summary=(
            "All arms share the frozen Gate 0 substrate and every decision is valid structured output."
            if lineage_matches and all_valid
            else "P1 lineage diverged from Gate 0 or an arm emitted invalid output."
        ),
        metrics=(
            ("lineage_matches_gate0", lineage_matches),
            ("all_outputs_valid", all_valid),
            (
                "minimum_decisions_observed",
                min(int(item["decisions"]) for item in per_arm.values()),
            ),
        ),
    )
    scaling_check = _scaling_check(
        run=run,
        contexts=contexts,
        config=effective_config,
    )
    structured_state_effect_check = _structured_state_user_swap_check(
        run=run,
        config=effective_config,
    )
    steelman_check = _steelman_check(run=run, config=effective_config)
    checks = (
        appendable_check,
        structural_check,
        scaling_check,
        console_check,
        lineage_check,
        structured_state_effect_check,
        steelman_check,
    )
    machinery_ready = all(item.status is P1CheckStatus.PASS for item in checks[:5])
    gate1_passed = machinery_ready and all(item.status is P1CheckStatus.PASS for item in checks[5:])
    arm_metrics = tuple(
        (
            arm.value,
            tuple(sorted(metrics.items())),
        )
        for arm, metrics in per_arm.items()
    )
    return RelationshipP1Report(
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=run.dataset_fingerprint,
        context_bundle_artifact_id=contexts.artifact_id,
        decision_ledger_sha256=run.decision_ledger_sha256,
        gate0_baseline_attestation_id=gate0_baseline.artifact_id,
        config=effective_config,
        checks=checks,
        arm_metrics=arm_metrics,
        machinery_ready=machinery_ready,
        gate1_passed=gate1_passed,
    )


def reassess_relationship_packet1_report_v1(
    *,
    source_report_path: pathlib.Path,
    minimum_structured_state_pair_flip_rate: float = 0.5,
) -> RelationshipP1Report:
    """Add the behavioral user-swap gate without rerunning frozen decisions."""

    raw = json.loads(pathlib.Path(source_report_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P1 source report must be a JSON object")
    expected_keys = {
        "artifact_id",
        "schema_version",
        "created_at_iso",
        "dataset_fingerprint",
        "context_bundle_artifact_id",
        "decision_ledger_sha256",
        "gate0_baseline_attestation_id",
        "config",
        "checks",
        "arm_metrics",
        "verdicts",
        "claim_boundary",
    }
    if set(raw) != expected_keys:
        raise ValueError("P1 v1 source report keys do not match the frozen schema")
    if raw["schema_version"] != "relationship-p1-report.v1":
        raise ValueError("P1 reassessment accepts report.v1 only")
    source_artifact_id = raw["artifact_id"]
    if not isinstance(source_artifact_id, str):
        raise ValueError("P1 source artifact_id must be a string")
    canonical_source = dict(raw)
    del canonical_source["artifact_id"]
    if sha256_json(canonical_source) != source_artifact_id:
        raise ValueError("P1 source report artifact_id mismatch")

    raw_config = raw["config"]
    if not isinstance(raw_config, dict) or set(raw_config) != {
        "minimum_decisions_per_arm",
        "minimum_steelman_accuracy",
        "maximum_steelman_accuracy",
        "minimum_steelman_pair_flip_rate",
        "maximum_rag_to_full_history_token_ratio",
        "maximum_structured_to_full_history_token_ratio",
    }:
        raise ValueError("P1 v1 source config is invalid")
    config = RelationshipP1GateConfig(
        minimum_decisions_per_arm=int(raw_config["minimum_decisions_per_arm"]),
        minimum_steelman_accuracy=float(raw_config["minimum_steelman_accuracy"]),
        maximum_steelman_accuracy=float(raw_config["maximum_steelman_accuracy"]),
        minimum_steelman_pair_flip_rate=float(raw_config["minimum_steelman_pair_flip_rate"]),
        minimum_structured_state_pair_flip_rate=(minimum_structured_state_pair_flip_rate),
        maximum_rag_to_full_history_token_ratio=float(raw_config["maximum_rag_to_full_history_token_ratio"]),
        maximum_structured_to_full_history_token_ratio=float(
            raw_config["maximum_structured_to_full_history_token_ratio"]
        ),
    )
    raw_arm_metrics = raw["arm_metrics"]
    if not isinstance(raw_arm_metrics, dict) or set(raw_arm_metrics) != {arm.value for arm in RELATIONSHIP_P1_ARMS}:
        raise ValueError("P1 v1 arm metrics are invalid")
    for arm, metrics in raw_arm_metrics.items():
        if not isinstance(arm, str) or not isinstance(metrics, dict):
            raise ValueError("P1 v1 arm metric row is invalid")

    raw_checks = raw["checks"]
    if not isinstance(raw_checks, list):
        raise ValueError("P1 v1 checks must be a list")
    expected_technical_ids = (
        "appendable_cross_process_recovery",
        "user_swap_scope_isolation",
        "token_scaling",
        "console_correction_and_delete",
        "same_substrate_and_valid_outputs",
    )
    if tuple(item.get("check_id") for item in raw_checks) != (
        *expected_technical_ids,
        "steelman_qualification",
    ):
        raise ValueError("P1 v1 check sequence is invalid")
    technical_checks: list[RelationshipP1Check] = []
    for item in raw_checks[:5]:
        if not isinstance(item, dict) or set(item) != {
            "check_id",
            "status",
            "summary",
            "metrics",
        }:
            raise ValueError("P1 v1 technical check is invalid")
        metrics = item["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("P1 v1 check metrics must be an object")
        technical_checks.append(
            RelationshipP1Check(
                check_id=str(item["check_id"]),
                status=P1CheckStatus(str(item["status"])),
                summary=str(item["summary"]),
                metrics=tuple(sorted(metrics.items())),
            )
        )
    structured_state_effect_check = _structured_state_user_swap_check_from_metrics(
        stateless=raw_arm_metrics[RelationshipP1Arm.STATELESS.value],
        structured=raw_arm_metrics[RelationshipP1Arm.STRUCTURED_STATE.value],
        config=config,
    )
    steelman_check = _steelman_check_from_metrics(
        prompt=raw_arm_metrics[RelationshipP1Arm.PROMPT_STEELMAN.value],
        rag=raw_arm_metrics[RelationshipP1Arm.RAG_STEELMAN.value],
        config=config,
    )
    checks = (
        *technical_checks,
        structured_state_effect_check,
        steelman_check,
    )
    machinery_ready = all(item.status is P1CheckStatus.PASS for item in technical_checks)
    gate1_passed = machinery_ready and all(item.status is P1CheckStatus.PASS for item in checks[5:])
    return RelationshipP1Report(
        created_at_iso=str(raw["created_at_iso"]),
        dataset_fingerprint=str(raw["dataset_fingerprint"]),
        context_bundle_artifact_id=str(raw["context_bundle_artifact_id"]),
        decision_ledger_sha256=str(raw["decision_ledger_sha256"]),
        gate0_baseline_attestation_id=str(raw["gate0_baseline_attestation_id"]),
        config=config,
        checks=checks,
        arm_metrics=tuple((arm, tuple(sorted(metrics.items()))) for arm, metrics in sorted(raw_arm_metrics.items())),
        machinery_ready=machinery_ready,
        gate1_passed=gate1_passed,
        source_report_artifact_id=source_artifact_id,
    )


def write_relationship_packet1_artifacts(
    *,
    run: RelationshipP1Run,
    report: RelationshipP1Report,
    recovery: RelationshipP1RecoveryEvidence,
    console: RelationshipP1ConsoleControlEvidence,
    persisted_state: PersistedRelationshipP1StateDigest,
    contexts: RelationshipP1ContextBundle,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, ...]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = (
        target / "decisions.jsonl",
        target / "run.json",
        target / "recovery.json",
        target / "console_control.json",
        target / "persisted_state.json",
        target / "contexts.json",
        target / "report.json",
        target / "report.md",
    )
    existing = tuple(path for path in paths if path.exists())
    if existing:
        raise FileExistsError(f"P1 output files already exist: {existing}")
    paths[0].write_text(run.decision_ledger_jsonl(), encoding="utf-8")
    if _sha256_file(paths[0]) != run.decision_ledger_sha256:
        raise RuntimeError("written P1 decision ledger hash mismatch")
    paths[1].write_text(
        json.dumps(
            run.to_summary_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths[2].write_text(
        json.dumps(
            recovery.to_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths[3].write_text(
        json.dumps(
            console.to_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state_payload = persisted_state.to_payload()
    state_payload["artifact_id"] = persisted_state.artifact_id
    paths[4].write_text(
        json.dumps(state_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    context_payload = contexts.to_summary_payload()
    context_payload["artifact_id"] = contexts.artifact_id
    paths[5].write_text(
        json.dumps(
            context_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths[6].write_text(report.to_json(), encoding="utf-8")
    lines = [
        "# Relationship Lab P1 development calibration",
        "",
        f"- artifact_id: `{report.artifact_id}`",
        f"- machinery_ready: **{str(report.machinery_ready).lower()}**",
        f"- gate1_passed: **{str(report.gate1_passed).lower()}**",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{item.check_id}` — **{item.status.value}**: {item.summary}" for item in report.checks)
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "P1 is development evidence for append/recover/control and baseline "
            "qualification only. The preregistration and secret heldout remain "
            "unfrozen; no four-capability or product claim is allowed.",
            "",
        ]
    )
    paths[7].write_text("\n".join(lines), encoding="utf-8")
    return paths


__all__ = [
    "ContextualRelationshipActionPolicy",
    "P1CheckStatus",
    "RELATIONSHIP_PACKET1_DECISION_SCHEMA_VERSION",
    "RELATIONSHIP_PACKET1_REPORT_SCHEMA_VERSION",
    "RELATIONSHIP_PACKET1_RUN_SCHEMA_VERSION",
    "RelationshipP1Check",
    "RelationshipP1ContextTokenCount",
    "RelationshipP1Decision",
    "RelationshipP1GateConfig",
    "RelationshipP1RecoveryEvidence",
    "RelationshipP1Report",
    "RelationshipP1Run",
    "assess_relationship_packet1",
    "reassess_relationship_packet1_report_v1",
    "relationship_p1_prompt_path",
    "relationship_p1_completion_to_decision",
    "run_relationship_packet1_arms",
    "write_relationship_packet1_artifacts",
]
