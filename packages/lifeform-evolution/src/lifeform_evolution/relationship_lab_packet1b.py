"""Relationship Lab P1b: schema-bound evidence-readout steelmen.

P1b keeps every P1 owner and context surface frozen.  A contextual arm asks
the same frozen substrate to publish two typed evidence scores, then compiles
those scores into the closed action protocol without reading text, ids, or
evaluation truth.  Expected actions are attached only after the readout has
been published to the observer.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

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
    RelationshipP1Arm,
    RelationshipP1ContextBundle,
    relationship_p1_evaluated_context_surface_sha256,
)
from lifeform_evolution.relationship_lab_packet1 import (
    ContextualRelationshipActionPolicy,
    RelationshipP1ContextTokenCount,
    RelationshipP1Decision,
    RelationshipP1Report,
    RelationshipP1Run,
    relationship_p1_completion_to_decision,
    relationship_p1_prompt_path,
)


RELATIONSHIP_P1B_READOUT_SCHEMA_VERSION = "relationship-p1b-readout.v2"
RELATIONSHIP_P1B_RUN_SCHEMA_VERSION = "relationship-p1b-run.v2"
RELATIONSHIP_P1B_REPORT_SCHEMA_VERSION = "relationship-p1b-report.v4"
RELATIONSHIP_P1B_COMPILER_VERSION = "relationship-evidence-argmax.v1"
RELATIONSHIP_P1B_RAG_TOP_K = 2
_SCORE_FIELDS = (
    "stay_present_without_probe_score",
    "respect_space_with_return_option_score",
)
_ALLOWED_SCORES = frozenset((-1, 0, 1))
_HEX_DIGITS = frozenset("0123456789abcdef")
_REQUEST_CONTEXT_MARKER = "{{PUBLIC_HISTORY_EVIDENCE}}"
_REQUEST_CURRENT_INPUT_MARKER = "{{CURRENT_USER_MESSAGE}}"
_P1B_REPORT_ARMS = (
    RelationshipP1Arm.PROMPT_STEELMAN.value,
    RelationshipP1Arm.RAG_STEELMAN.value,
    RelationshipP1Arm.STRUCTURED_STATE.value,
)
_P1B_REPORT_METRIC_FIELDS = frozenset(
    {
        "accuracy",
        "completion_tokens_total",
        "correct_decisions",
        "decisions",
        "pair_flip_rate",
        "pair_groups",
        "prompt_tokens_total",
        "readouts",
        "valid_decisions",
        "valid_pair_groups",
        "valid_rate",
        "valid_readouts",
    }
)


class RelationshipP1bReadoutProfile(str, Enum):
    V1_ACTION_TALLY = "v1_action_tally"
    V2_CONDITION_AWARE = "v2_condition_aware"


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def relationship_p1b_readout_prompt_path(
    profile: RelationshipP1bReadoutProfile = RelationshipP1bReadoutProfile.V1_ACTION_TALLY,
) -> pathlib.Path:
    if profile is RelationshipP1bReadoutProfile.V1_ACTION_TALLY:
        filename = "relationship_lab_evidence_readout_v3.txt"
    elif profile is RelationshipP1bReadoutProfile.V2_CONDITION_AWARE:
        filename = "relationship_lab_conditioned_evidence_readout_v1.txt"
    else:
        raise ValueError("unsupported P1b readout profile")
    return _asset_dir() / "prompts" / filename


def relationship_p1b_readout_schema_path() -> pathlib.Path:
    return _asset_dir() / "schemas" / "relationship_evidence_readout.schema.json"


def relationship_p1b_readout_request_template_path(
    profile: RelationshipP1bReadoutProfile = RelationshipP1bReadoutProfile.V1_ACTION_TALLY,
) -> pathlib.Path:
    if profile is RelationshipP1bReadoutProfile.V1_ACTION_TALLY:
        filename = "relationship_lab_evidence_readout_request_v1.txt"
    elif profile is RelationshipP1bReadoutProfile.V2_CONDITION_AWARE:
        filename = "relationship_lab_conditioned_evidence_readout_request_v1.txt"
    else:
        raise ValueError("unsupported P1b readout profile")
    return _asset_dir() / "prompts" / filename


def render_relationship_p1b_readout_request(
    *,
    context_text: str,
    current_input: str,
    profile: RelationshipP1bReadoutProfile = RelationshipP1bReadoutProfile.V1_ACTION_TALLY,
) -> str:
    if not context_text.strip() or not current_input.strip():
        raise ValueError("P1b readout request requires context and current input")
    template = relationship_p1b_readout_request_template_path(profile).read_text(
        encoding="utf-8"
    )
    if template.count(_REQUEST_CONTEXT_MARKER) != 1 or template.count(_REQUEST_CURRENT_INPUT_MARKER) != 1:
        raise ValueError("P1b request template markers must each occur exactly once")
    return (
        template.replace(_REQUEST_CONTEXT_MARKER, context_text)
        .replace(_REQUEST_CURRENT_INPUT_MARKER, current_input)
        .strip()
    )


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def _require_iso_timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


def parse_relationship_evidence_scores(
    raw_output: str,
) -> tuple[int | None, int | None]:
    """Strictly parse the frozen two-score readout schema."""

    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None, None
    if not isinstance(payload, dict) or set(payload) != set(_SCORE_FIELDS):
        return None, None
    values = tuple(payload[field] for field in _SCORE_FIELDS)
    if any(type(value) is not int or value not in _ALLOWED_SCORES for value in values):
        return None, None
    return values[0], values[1]


def compile_relationship_evidence_scores(
    *,
    stay_score: int,
    space_score: int,
) -> RelationshipAction:
    """Project typed scores to the action enum without consuming semantics."""

    if type(stay_score) is not int or stay_score not in _ALLOWED_SCORES:
        raise ValueError("stay_score must be one of -1, 0, 1")
    if type(space_score) is not int or space_score not in _ALLOWED_SCORES:
        raise ValueError("space_score must be one of -1, 0, 1")
    if stay_score > space_score:
        return RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    if space_score > stay_score:
        return RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
    return RelationshipAction.NEUTRAL_NOOP


@dataclass(frozen=True)
class RelationshipEvidenceReadout:
    arm: RelationshipP1Arm
    scene_id: str
    seed: int
    current_input_sha256: str
    context_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    prompt_sha256: str
    request_template_sha256: str
    schema_sha256: str
    raw_output: str
    stay_score: int | None
    space_score: int | None
    prompt_tokens: int
    completion_tokens: int
    schema_version: str = RELATIONSHIP_P1B_READOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1B_READOUT_SCHEMA_VERSION:
            raise ValueError("P1b readout schema_version mismatch")
        if self.arm is RelationshipP1Arm.STATELESS:
            raise ValueError("stateless arm cannot publish a P1b evidence readout")
        if not self.scene_id.strip() or not self.model_id.strip() or self.seed < 0:
            raise ValueError("P1b readout identity is invalid")
        for field_name, value in (
            ("current_input_sha256", self.current_input_sha256),
            ("context_sha256", self.context_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("prompt_sha256", self.prompt_sha256),
            ("request_template_sha256", self.request_template_sha256),
            ("schema_sha256", self.schema_sha256),
        ):
            _require_sha256(value, field_name)
        if (self.stay_score is None) is not (self.space_score is None):
            raise ValueError("P1b readout scores must both be present or absent")
        if self.stay_score is not None:
            compile_relationship_evidence_scores(
                stay_score=self.stay_score,
                space_score=self.space_score,
            )
        for field_name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def valid(self) -> bool:
        return self.stay_score is not None and self.space_score is not None

    @property
    def compiled_action(self) -> RelationshipAction | None:
        if self.stay_score is None or self.space_score is None:
            return None
        return compile_relationship_evidence_scores(
            stay_score=self.stay_score,
            space_score=self.space_score,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arm": self.arm.value,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "current_input_sha256": self.current_input_sha256,
            "context_sha256": self.context_sha256,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "prompt_sha256": self.prompt_sha256,
            "request_template_sha256": self.request_template_sha256,
            "schema_sha256": self.schema_sha256,
            "raw_output": self.raw_output,
            "stay_score": self.stay_score,
            "space_score": self.space_score,
            "valid": self.valid,
            "compiled_action_id": (self.compiled_action.value if self.compiled_action is not None else None),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class RelationshipP1bRun:
    action_run: RelationshipP1Run
    readout_prompt_sha256: str
    readout_request_template_sha256: str
    readout_schema_sha256: str
    compiler_version: str
    readouts: tuple[RelationshipEvidenceReadout, ...]
    schema_version: str = RELATIONSHIP_P1B_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1B_RUN_SCHEMA_VERSION:
            raise ValueError("P1b run schema_version mismatch")
        _require_sha256(self.readout_prompt_sha256, "readout_prompt_sha256")
        _require_sha256(
            self.readout_request_template_sha256,
            "readout_request_template_sha256",
        )
        _require_sha256(self.readout_schema_sha256, "readout_schema_sha256")
        if self.compiler_version != RELATIONSHIP_P1B_COMPILER_VERSION:
            raise ValueError("P1b compiler version mismatch")
        readout_keys = tuple((item.arm, item.scene_id, item.seed) for item in self.readouts)
        if not readout_keys or len(set(readout_keys)) != len(readout_keys):
            raise ValueError("P1b readout keys must be non-empty and unique")
        contextual_decisions = {
            (item.arm, item.scene_id, item.seed): item
            for item in self.action_run.decisions
            if item.arm is not RelationshipP1Arm.STATELESS
        }
        if set(readout_keys) != set(contextual_decisions):
            raise ValueError("P1b readouts must cover every contextual decision")
        pipeline_sha256 = sha256_json(
            {
                "prompt_sha256": self.readout_prompt_sha256,
                "request_template_sha256": self.readout_request_template_sha256,
                "schema_sha256": self.readout_schema_sha256,
                "compiler_version": self.compiler_version,
            }
        )
        arm_prompt_hashes = dict(self.action_run.arm_prompt_hashes)
        for arm in (
            RelationshipP1Arm.PROMPT_STEELMAN,
            RelationshipP1Arm.RAG_STEELMAN,
            RelationshipP1Arm.STRUCTURED_STATE,
        ):
            if arm_prompt_hashes.get(arm.value) != pipeline_sha256:
                raise ValueError("P1b action run does not bind the readout pipeline")
        for readout in self.readouts:
            if (
                readout.prompt_sha256 != self.readout_prompt_sha256
                or readout.request_template_sha256 != self.readout_request_template_sha256
                or readout.schema_sha256 != self.readout_schema_sha256
                or readout.model_id != self.action_run.model_id
                or readout.weights_sha256 != self.action_run.weights_sha256
                or readout.generation_config_sha256 != self.action_run.generation_config_sha256
            ):
                raise ValueError("P1b readout lineage diverges from its run")
            decision = contextual_decisions[(readout.arm, readout.scene_id, readout.seed)]
            if (
                decision.current_input_sha256 != readout.current_input_sha256
                or decision.context_sha256 != readout.context_sha256
                or decision.chosen_action_id is not readout.compiled_action
            ):
                raise ValueError("P1b readout-to-decision projection mismatch")

    def readout_ledger_jsonl(self) -> str:
        return "".join(
            canonical_json({**item.to_payload(), "artifact_id": item.artifact_id}) + "\n" for item in self.readouts
        )

    @property
    def readout_ledger_sha256(self) -> str:
        return hashlib.sha256(self.readout_ledger_jsonl().encode("utf-8")).hexdigest()

    def readout_metrics(self, arm: RelationshipP1Arm) -> dict[str, object]:
        rows = tuple(item for item in self.readouts if item.arm is arm)
        if not rows:
            raise ValueError(f"P1b has no readouts for {arm.value}")
        valid = sum(int(item.valid) for item in rows)
        return {
            "readouts": len(rows),
            "valid_readouts": valid,
            "valid_rate": valid / len(rows),
            "prompt_tokens_total": sum(item.prompt_tokens for item in rows),
            "completion_tokens_total": sum(item.completion_tokens for item in rows),
        }

    def to_summary_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_run": self.action_run.to_summary_payload(),
            "readout_prompt_sha256": self.readout_prompt_sha256,
            "readout_request_template_sha256": (self.readout_request_template_sha256),
            "readout_schema_sha256": self.readout_schema_sha256,
            "compiler_version": self.compiler_version,
            "readout_ledger_sha256": self.readout_ledger_sha256,
            "readout_metrics": {
                arm.value: self.readout_metrics(arm)
                for arm in RELATIONSHIP_P1_ARMS
                if arm is not RelationshipP1Arm.STATELESS
            },
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_summary_payload())


def _readout_to_action_completion(
    readout: RelationshipEvidenceReadout,
) -> StatelessActionCompletion:
    action = readout.compiled_action
    if action is None:
        return StatelessActionCompletion(
            raw_output=readout.raw_output,
            chosen_action_id=None,
            prompt_tokens=readout.prompt_tokens,
            completion_tokens=readout.completion_tokens,
        )
    return StatelessActionCompletion(
        raw_output=canonical_json({"action_id": action.value}),
        chosen_action_id=action,
        prompt_tokens=readout.prompt_tokens,
        completion_tokens=readout.completion_tokens,
    )


def run_relationship_packet1b_arms(
    policy: ContextualRelationshipActionPolicy,
    *,
    contexts: RelationshipP1ContextBundle,
    dataset: RelationshipTransferDataset | None = None,
    seed_schedule: tuple[int, ...] = (101,),
    readout_profile: RelationshipP1bReadoutProfile = (
        RelationshipP1bReadoutProfile.V1_ACTION_TALLY
    ),
    readout_observer: Callable[[RelationshipEvidenceReadout], None] | None = None,
    decision_observer: Callable[[RelationshipP1Decision], None] | None = None,
) -> RelationshipP1bRun:
    effective_dataset = dataset or load_relationship_transfer_dataset()
    if contexts.dataset_fingerprint != effective_dataset.dataset_fingerprint:
        raise ValueError("P1b context bundle dataset fingerprint mismatch")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("P1b seed_schedule must be non-empty and unique")
    if not isinstance(readout_profile, RelationshipP1bReadoutProfile):
        raise ValueError("P1b readout_profile must be typed")

    prompt_path = relationship_p1b_readout_prompt_path(readout_profile)
    request_template_path = relationship_p1b_readout_request_template_path(
        readout_profile
    )
    schema_path = relationship_p1b_readout_schema_path()
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt_sha256 = _sha256_file(prompt_path)
    request_template_sha256 = _sha256_file(request_template_path)
    schema_sha256 = _sha256_file(schema_path)
    pipeline_sha256 = sha256_json(
        {
            "prompt_sha256": prompt_sha256,
            "request_template_sha256": request_template_sha256,
            "schema_sha256": schema_sha256,
            "compiler_version": RELATIONSHIP_P1B_COMPILER_VERSION,
        }
    )
    stateless_prompt_sha256 = _sha256_file(relationship_p1_prompt_path(RelationshipP1Arm.STATELESS))
    if stateless_prompt_sha256 != policy.prompt_sha256:
        raise ValueError("P1b stateless prompt does not match Gate 0 policy")

    decisions: list[RelationshipP1Decision] = []
    readouts: list[RelationshipEvidenceReadout] = []
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
                    arm_prompt_sha256=stateless_prompt_sha256,
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
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": render_relationship_p1b_readout_request(
                                    context_text=context.context_text,
                                    current_input=observation.current_input,
                                    profile=readout_profile,
                                ),
                            },
                        ),
                        seed=seed,
                    )
                    stay_score, space_score = parse_relationship_evidence_scores(completion.raw_output)
                    readout = RelationshipEvidenceReadout(
                        arm=arm,
                        scene_id=observation.scene_id,
                        seed=seed,
                        current_input_sha256=current_input_sha256,
                        context_sha256=context.context_sha256,
                        model_id=policy.model_id,
                        weights_sha256=policy.weights_sha256,
                        generation_config_sha256=(policy.generation_config_sha256),
                        prompt_sha256=prompt_sha256,
                        request_template_sha256=request_template_sha256,
                        schema_sha256=schema_sha256,
                        raw_output=completion.raw_output,
                        stay_score=stay_score,
                        space_score=space_score,
                        prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens,
                    )
                    readouts.append(readout)
                    if readout_observer is not None:
                        readout_observer(readout)
                    decision = relationship_p1_completion_to_decision(
                        completion=_readout_to_action_completion(readout),
                        arm=arm,
                        scene_id=observation.scene_id,
                        mirror_pair_id=mirror_pair_id,
                        split=split,
                        seed=seed,
                        current_input_sha256=current_input_sha256,
                        context_sha256=context.context_sha256,
                        arm_prompt_sha256=pipeline_sha256,
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
    action_run = RelationshipP1Run(
        dataset_fingerprint=effective_dataset.dataset_fingerprint,
        context_bundle_artifact_id=contexts.artifact_id,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule=seed_schedule,
        arm_prompt_hashes=tuple(
            sorted(
                (
                    (RelationshipP1Arm.STATELESS.value, stateless_prompt_sha256),
                    (RelationshipP1Arm.PROMPT_STEELMAN.value, pipeline_sha256),
                    (RelationshipP1Arm.RAG_STEELMAN.value, pipeline_sha256),
                    (RelationshipP1Arm.STRUCTURED_STATE.value, pipeline_sha256),
                )
            )
        ),
        decisions=tuple(decisions),
        context_token_counts=token_counts,
    )
    return RelationshipP1bRun(
        action_run=action_run,
        readout_prompt_sha256=prompt_sha256,
        readout_request_template_sha256=request_template_sha256,
        readout_schema_sha256=schema_sha256,
        compiler_version=RELATIONSHIP_P1B_COMPILER_VERSION,
        readouts=tuple(readouts),
    )


class RelationshipP1bVerdict(str, Enum):
    QUALIFIED = "qualified"
    DATASET_SATURATED = "dataset_saturated"
    BASELINE_UNDERQUALIFIED = "baseline_underqualified"


@dataclass(frozen=True)
class RelationshipP1bReport:
    created_at_iso: str
    dataset_fingerprint: str
    context_bundle_artifact_id: str
    evaluated_context_surface_sha256: str
    background_templates_sha256: str
    rag_config_sha256: str
    seed_schedule_sha256: str
    p1_gate_config_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    gate0_baseline_attestation_id: str
    readout_prompt_sha256: str
    readout_request_template_sha256: str
    readout_schema_sha256: str
    compiler_version: str
    run_artifact_id: str
    p1_report_artifact_id: str
    readout_ledger_sha256: str
    verdict: RelationshipP1bVerdict
    p1_machinery_ready: bool
    all_readouts_valid: bool
    saturated_arms: tuple[str, ...]
    arm_metrics: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    schema_version: str = RELATIONSHIP_P1B_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1B_REPORT_SCHEMA_VERSION:
            raise ValueError("P1b report schema_version mismatch")
        _require_iso_timestamp(self.created_at_iso, "created_at_iso")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("P1b report model_id must be non-empty")
        if not isinstance(self.verdict, RelationshipP1bVerdict):
            raise ValueError("P1b report verdict must be typed")
        for field_name, value in (
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("context_bundle_artifact_id", self.context_bundle_artifact_id),
            (
                "evaluated_context_surface_sha256",
                self.evaluated_context_surface_sha256,
            ),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
            ("seed_schedule_sha256", self.seed_schedule_sha256),
            ("p1_gate_config_sha256", self.p1_gate_config_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            (
                "gate0_baseline_attestation_id",
                self.gate0_baseline_attestation_id,
            ),
            ("readout_prompt_sha256", self.readout_prompt_sha256),
            (
                "readout_request_template_sha256",
                self.readout_request_template_sha256,
            ),
            ("readout_schema_sha256", self.readout_schema_sha256),
            ("run_artifact_id", self.run_artifact_id),
            ("p1_report_artifact_id", self.p1_report_artifact_id),
            ("readout_ledger_sha256", self.readout_ledger_sha256),
        ):
            _require_sha256(value, field_name)
        if self.compiler_version != RELATIONSHIP_P1B_COMPILER_VERSION:
            raise ValueError("P1b report compiler version mismatch")
        if not isinstance(self.p1_machinery_ready, bool):
            raise ValueError("P1b p1_machinery_ready must be boolean")
        if not isinstance(self.all_readouts_valid, bool):
            raise ValueError("P1b all_readouts_valid must be boolean")
        if self.saturated_arms != tuple(sorted(set(self.saturated_arms))):
            raise ValueError("P1b saturated arms must be sorted and unique")
        if not set(self.saturated_arms).issubset(
            {
                RelationshipP1Arm.PROMPT_STEELMAN.value,
                RelationshipP1Arm.RAG_STEELMAN.value,
            }
        ):
            raise ValueError("P1b saturated arms must be steelman arms")
        if self.verdict is RelationshipP1bVerdict.DATASET_SATURATED and (
            not self.saturated_arms or not self.p1_machinery_ready or not self.all_readouts_valid
        ):
            raise ValueError("P1b saturation verdict requires ready machinery, valid readouts, and saturated arms")
        if self.verdict is RelationshipP1bVerdict.QUALIFIED and self.saturated_arms:
            raise ValueError("P1b qualified verdict cannot contain saturated arms")
        if self.verdict is RelationshipP1bVerdict.QUALIFIED and not self.all_readouts_valid:
            raise ValueError("P1b qualified verdict requires valid readouts")
        arm_ids = tuple(arm for arm, _metrics in self.arm_metrics)
        if arm_ids != _P1B_REPORT_ARMS:
            raise ValueError("P1b arm metrics must contain the frozen arms in order")
        for arm, metrics_items in self.arm_metrics:
            metrics = dict(metrics_items)
            if len(metrics) != len(metrics_items) or set(metrics) != _P1B_REPORT_METRIC_FIELDS:
                raise ValueError(f"P1b metrics for {arm} do not match the frozen schema")
            for field_name in (
                "completion_tokens_total",
                "correct_decisions",
                "decisions",
                "pair_groups",
                "prompt_tokens_total",
                "readouts",
                "valid_decisions",
                "valid_pair_groups",
                "valid_readouts",
            ):
                value = metrics[field_name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"P1b {arm}.{field_name} must be a non-negative integer")
            for field_name in ("accuracy", "pair_flip_rate", "valid_rate"):
                value = metrics[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"P1b {arm}.{field_name} must be in [0, 1]")
            if metrics["readouts"] != metrics["decisions"]:
                raise ValueError(f"P1b {arm} readout/decision counts diverge")
            if metrics["valid_readouts"] != metrics["valid_decisions"]:
                raise ValueError(f"P1b {arm} readout/decision validity diverges")
            decisions = metrics["decisions"]
            if decisions < 1:
                raise ValueError(f"P1b {arm} requires at least one decision")
            if not 0 <= metrics["correct_decisions"] <= metrics["valid_decisions"] <= decisions:
                raise ValueError(f"P1b {arm} decision counts are inconsistent")
            if not 0 <= metrics["valid_pair_groups"] <= metrics["pair_groups"]:
                raise ValueError(f"P1b {arm} mirrored-pair counts are inconsistent")
            if metrics["accuracy"] != metrics["correct_decisions"] / decisions:
                raise ValueError(f"P1b {arm} accuracy diverges from decision counts")
            if metrics["valid_rate"] != metrics["valid_decisions"] / decisions:
                raise ValueError(f"P1b {arm} valid_rate diverges from decision counts")
        metrics_all_valid = all(
            dict(metrics)["valid_readouts"] == dict(metrics)["readouts"] for _arm, metrics in self.arm_metrics
        )
        if self.all_readouts_valid != metrics_all_valid:
            raise ValueError("P1b all_readouts_valid diverges from arm metrics")
        if self.gate1_passed and not self.p1_machinery_ready:
            raise ValueError("P1b cannot qualify when P1 machinery is not ready")

    @property
    def gate1_passed(self) -> bool:
        return self.verdict is RelationshipP1bVerdict.QUALIFIED

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "dataset_fingerprint": self.dataset_fingerprint,
            "context_bundle_artifact_id": self.context_bundle_artifact_id,
            "evaluated_context_surface_sha256": (self.evaluated_context_surface_sha256),
            "background_templates_sha256": self.background_templates_sha256,
            "rag_config_sha256": self.rag_config_sha256,
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "p1_gate_config_sha256": self.p1_gate_config_sha256,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "gate0_baseline_attestation_id": (self.gate0_baseline_attestation_id),
            "readout_prompt_sha256": self.readout_prompt_sha256,
            "readout_request_template_sha256": (self.readout_request_template_sha256),
            "readout_schema_sha256": self.readout_schema_sha256,
            "compiler_version": self.compiler_version,
            "run_artifact_id": self.run_artifact_id,
            "p1_report_artifact_id": self.p1_report_artifact_id,
            "readout_ledger_sha256": self.readout_ledger_sha256,
            "verdict": self.verdict.value,
            "gate1_passed": self.gate1_passed,
            "p1_machinery_ready": self.p1_machinery_ready,
            "all_readouts_valid": self.all_readouts_valid,
            "saturated_arms": list(self.saturated_arms),
            "arm_metrics": {arm: dict(metrics) for arm, metrics in self.arm_metrics},
            "claim_boundary": (
                "P1b qualifies a frozen schema-bound baseline or detects "
                "development-set saturation. It does not prove Readable, "
                "Learnable, Steerable, formal held-out superiority, or product value."
            ),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1bReport":
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise ValueError("P1b report must be a JSON object")
        expected = {
            "all_readouts_valid",
            "arm_metrics",
            "artifact_id",
            "claim_boundary",
            "compiler_version",
            "context_bundle_artifact_id",
            "created_at_iso",
            "dataset_fingerprint",
            "evaluated_context_surface_sha256",
            "gate0_baseline_attestation_id",
            "gate1_passed",
            "generation_config_sha256",
            "model_id",
            "background_templates_sha256",
            "p1_gate_config_sha256",
            "p1_machinery_ready",
            "p1_report_artifact_id",
            "readout_ledger_sha256",
            "readout_prompt_sha256",
            "readout_request_template_sha256",
            "readout_schema_sha256",
            "rag_config_sha256",
            "run_artifact_id",
            "saturated_arms",
            "schema_version",
            "seed_schedule_sha256",
            "verdict",
            "weights_sha256",
        }
        if set(raw) != expected:
            raise ValueError("P1b report fields do not match schema v4")
        artifact_id = raw.pop("artifact_id")
        gate1_passed = raw.pop("gate1_passed")
        claim_boundary = raw.pop("claim_boundary")
        if not isinstance(gate1_passed, bool) or not isinstance(claim_boundary, str):
            raise ValueError("P1b report derived fields are invalid")
        metrics_raw = raw["arm_metrics"]
        if not isinstance(metrics_raw, dict):
            raise ValueError("P1b arm_metrics must be an object")
        arm_metrics: list[tuple[str, tuple[tuple[str, object], ...]]] = []
        for arm, metrics in metrics_raw.items():
            if not isinstance(arm, str) or not isinstance(metrics, dict):
                raise ValueError("P1b arm_metrics entries are invalid")
            arm_metrics.append((arm, tuple(sorted(metrics.items()))))
        saturated_raw = raw["saturated_arms"]
        if not isinstance(saturated_raw, list) or any(not isinstance(item, str) for item in saturated_raw):
            raise ValueError("P1b saturated_arms must be a string list")
        try:
            verdict = RelationshipP1bVerdict(raw["verdict"])
        except (TypeError, ValueError) as exc:
            raise ValueError("P1b verdict is invalid") from exc
        report = cls(
            schema_version=raw["schema_version"],
            created_at_iso=raw["created_at_iso"],
            dataset_fingerprint=raw["dataset_fingerprint"],
            context_bundle_artifact_id=raw["context_bundle_artifact_id"],
            evaluated_context_surface_sha256=raw["evaluated_context_surface_sha256"],
            background_templates_sha256=raw["background_templates_sha256"],
            rag_config_sha256=raw["rag_config_sha256"],
            seed_schedule_sha256=raw["seed_schedule_sha256"],
            p1_gate_config_sha256=raw["p1_gate_config_sha256"],
            model_id=raw["model_id"],
            weights_sha256=raw["weights_sha256"],
            generation_config_sha256=raw["generation_config_sha256"],
            gate0_baseline_attestation_id=(raw["gate0_baseline_attestation_id"]),
            readout_prompt_sha256=raw["readout_prompt_sha256"],
            readout_request_template_sha256=(raw["readout_request_template_sha256"]),
            readout_schema_sha256=raw["readout_schema_sha256"],
            compiler_version=raw["compiler_version"],
            run_artifact_id=raw["run_artifact_id"],
            p1_report_artifact_id=raw["p1_report_artifact_id"],
            readout_ledger_sha256=raw["readout_ledger_sha256"],
            verdict=verdict,
            p1_machinery_ready=raw["p1_machinery_ready"],
            all_readouts_valid=raw["all_readouts_valid"],
            saturated_arms=tuple(saturated_raw),
            arm_metrics=tuple(sorted(arm_metrics)),
        )
        _require_sha256(artifact_id, "artifact_id")
        if artifact_id != report.artifact_id or gate1_passed != report.gate1_passed:
            raise ValueError("P1b report derived values or artifact_id mismatch")
        return report


def assess_relationship_packet1b(
    *,
    run: RelationshipP1bRun,
    p1_report: RelationshipP1Report,
    contexts: RelationshipP1ContextBundle,
    dataset: RelationshipTransferDataset | None = None,
    created_at_iso: str | None = None,
) -> RelationshipP1bReport:
    effective_dataset = dataset or load_relationship_transfer_dataset()
    if effective_dataset.dataset_fingerprint != run.action_run.dataset_fingerprint:
        raise ValueError("P1b assessment dataset does not match its run")
    if run.action_run.decision_ledger_sha256 != p1_report.decision_ledger_sha256:
        raise ValueError("P1b action run does not match the P1 report")
    if run.action_run.dataset_fingerprint != p1_report.dataset_fingerprint:
        raise ValueError("P1b dataset does not match the P1 report")
    if (
        run.action_run.context_bundle_artifact_id != contexts.artifact_id
        or p1_report.context_bundle_artifact_id != contexts.artifact_id
    ):
        raise ValueError("P1b context bundle does not match its run and P1 report")
    all_readouts_valid = all(item.valid for item in run.readouts)
    p1_metrics = {arm: dict(metrics) for arm, metrics in p1_report.arm_metrics}
    saturated_arms = tuple(
        sorted(
            arm.value
            for arm in (
                RelationshipP1Arm.PROMPT_STEELMAN,
                RelationshipP1Arm.RAG_STEELMAN,
            )
            if float(p1_metrics[arm.value]["accuracy"]) > p1_report.config.maximum_steelman_accuracy
        )
    )
    if p1_report.machinery_ready and all_readouts_valid and saturated_arms:
        verdict = RelationshipP1bVerdict.DATASET_SATURATED
    elif all_readouts_valid and p1_report.gate1_passed:
        verdict = RelationshipP1bVerdict.QUALIFIED
    else:
        verdict = RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED
    arm_metrics = tuple(
        (
            arm.value,
            tuple(
                sorted(
                    {
                        **run.readout_metrics(arm),
                        **run.action_run.arm_metrics(arm),
                    }.items()
                )
            ),
        )
        for arm in (
            RelationshipP1Arm.PROMPT_STEELMAN,
            RelationshipP1Arm.RAG_STEELMAN,
            RelationshipP1Arm.STRUCTURED_STATE,
        )
    )
    return RelationshipP1bReport(
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
        dataset_fingerprint=run.action_run.dataset_fingerprint,
        context_bundle_artifact_id=run.action_run.context_bundle_artifact_id,
        evaluated_context_surface_sha256=(
            relationship_p1_evaluated_context_surface_sha256(
                bundle=contexts,
                dataset=effective_dataset,
            )
        ),
        background_templates_sha256=contexts.background_templates_sha256,
        rag_config_sha256=contexts.rag_config_sha256,
        seed_schedule_sha256=run.action_run.seed_schedule_sha256,
        p1_gate_config_sha256=sha256_json(p1_report.config.to_payload()),
        model_id=run.action_run.model_id,
        weights_sha256=run.action_run.weights_sha256,
        generation_config_sha256=run.action_run.generation_config_sha256,
        gate0_baseline_attestation_id=(p1_report.gate0_baseline_attestation_id),
        readout_prompt_sha256=run.readout_prompt_sha256,
        readout_request_template_sha256=(run.readout_request_template_sha256),
        readout_schema_sha256=run.readout_schema_sha256,
        compiler_version=run.compiler_version,
        run_artifact_id=run.artifact_id,
        p1_report_artifact_id=p1_report.artifact_id,
        readout_ledger_sha256=run.readout_ledger_sha256,
        verdict=verdict,
        p1_machinery_ready=p1_report.machinery_ready,
        all_readouts_valid=all_readouts_valid,
        saturated_arms=saturated_arms,
        arm_metrics=arm_metrics,
    )


def write_relationship_packet1b_artifacts(
    *,
    run: RelationshipP1bRun,
    report: RelationshipP1bReport,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, ...]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = (
        target / "readouts.jsonl",
        target / "packet1b_run.json",
        target / "packet1b_report.json",
        target / "packet1b_report.md",
    )
    existing = tuple(path for path in paths if path.exists())
    if existing:
        raise FileExistsError(f"P1b output files already exist: {existing}")
    paths[0].write_text(run.readout_ledger_jsonl(), encoding="utf-8")
    if _sha256_file(paths[0]) != run.readout_ledger_sha256:
        raise RuntimeError("written P1b readout ledger hash mismatch")
    run_payload = run.to_summary_payload()
    run_payload["artifact_id"] = run.artifact_id
    paths[1].write_text(
        json.dumps(run_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths[2].write_text(report.to_json(), encoding="utf-8")
    metrics_by_arm = {arm: dict(metrics) for arm, metrics in report.arm_metrics}
    lines = [
        "# Relationship Lab P1b development calibration",
        "",
        f"- artifact_id: `{report.artifact_id}`",
        f"- verdict: **{report.verdict.value}**",
        f"- gate1_passed: **{str(report.gate1_passed).lower()}**",
        f"- all_readouts_valid: **{str(report.all_readouts_valid).lower()}**",
        f"- saturated_arms: `{', '.join(report.saturated_arms) or 'none'}`",
        f"- readout_prompt_sha256: `{run.readout_prompt_sha256}`",
        (f"- readout_request_template_sha256: `{run.readout_request_template_sha256}`"),
        f"- readout_schema_sha256: `{run.readout_schema_sha256}`",
        f"- compiler_version: `{run.compiler_version}`",
        "",
        "| Arm | valid | accuracy | pair flip | prompt tokens |",
        "|---|---:|---:|---:|---:|",
        *(
            "| "
            f"{arm} | {int(metrics_by_arm[arm]['valid_readouts'])}/"
            f"{int(metrics_by_arm[arm]['readouts'])} | "
            f"{float(metrics_by_arm[arm]['accuracy']):.3f} | "
            f"{float(metrics_by_arm[arm]['pair_flip_rate']):.3f} | "
            f"{int(metrics_by_arm[arm]['prompt_tokens_total'])} |"
            for arm in (
                RelationshipP1Arm.PROMPT_STEELMAN.value,
                RelationshipP1Arm.RAG_STEELMAN.value,
                RelationshipP1Arm.STRUCTURED_STATE.value,
            )
        ),
        "",
        "P1b is development-only evidence. Formal preregistration and secret heldout remain closed.",
        "",
    ]
    paths[3].write_text("\n".join(lines), encoding="utf-8")
    return paths


def load_relationship_packet1b_report(
    path: pathlib.Path,
) -> RelationshipP1bReport:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return RelationshipP1bReport.from_json(file_path.read_text(encoding="utf-8"))


__all__ = [
    "RELATIONSHIP_P1B_COMPILER_VERSION",
    "RELATIONSHIP_P1B_RAG_TOP_K",
    "RELATIONSHIP_P1B_READOUT_SCHEMA_VERSION",
    "RELATIONSHIP_P1B_REPORT_SCHEMA_VERSION",
    "RELATIONSHIP_P1B_RUN_SCHEMA_VERSION",
    "RelationshipEvidenceReadout",
    "RelationshipP1bReport",
    "RelationshipP1bReadoutProfile",
    "RelationshipP1bRun",
    "RelationshipP1bVerdict",
    "assess_relationship_packet1b",
    "compile_relationship_evidence_scores",
    "load_relationship_packet1b_report",
    "parse_relationship_evidence_scores",
    "relationship_p1b_readout_prompt_path",
    "relationship_p1b_readout_request_template_path",
    "relationship_p1b_readout_schema_path",
    "render_relationship_p1b_readout_request",
    "run_relationship_packet1b_arms",
    "write_relationship_packet1b_artifacts",
]
