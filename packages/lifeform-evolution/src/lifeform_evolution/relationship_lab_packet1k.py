"""Relationship Lab P1k: evaluator-only oracle-concept disclosure ladder.

P1g/P1i/P1j established that the frozen ordinary-Qwen consumer does not reach
the mirrored-flip qualification bar, but they cannot say *where* the failure
lives: recognizing which abstract relational condition a probe instantiates,
inducing the individual condition-to-action policy from controlled outcome
evidence, or merely projecting an already-known rule onto the typed score
schema.

P1k answers that by running the same frozen substrate over the same already
burned ``relationship_transfer_v3`` consumer-training package three times,
disclosing one additional layer of sealed generator truth each time:

``oracle_condition_v1``   the two abstract conditions and which one the probe is
``oracle_binding_v1``     additionally which condition each history instantiates
``oracle_policy_v1``      additionally this user's condition-to-action policy

The most-disclosed rung runs first, because a failure there is a machinery or
substrate floor and makes the lower rungs uninterpretable.  All three rungs are
non-competitive by construction: they consume sealed truth that the qualified
arms may never see.  They exist to localize a failure, never to score the
system.  P1k therefore revises no consumer, no gate, no dataset, and writes
nothing into Volvence memory, PE, credit, reward, controller, or steering.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipAction,
    RelationshipDatasetSplit,
    RelationshipTransferDataset,
    canonical_json,
    sha256_json,
)
from lifeform_evolution.relationship_lab_contexts import (
    RelationshipP1Arm,
    RelationshipP1ContextBundle,
    relationship_p1_evaluated_context_surface_sha256,
)
from lifeform_evolution.relationship_lab_packet1 import (
    ContextualRelationshipActionPolicy,
)
from lifeform_evolution.relationship_lab_packet1b import (
    RELATIONSHIP_P1B_COMPILER_VERSION,
    compile_relationship_evidence_scores,
    parse_relationship_evidence_scores,
    relationship_p1b_readout_schema_path,
)
from lifeform_evolution.relationship_lab_packet1i import (
    RelationshipP1iFrozenConsumerProtocol,
)


RELATIONSHIP_P1K_PROTOCOL_SCHEMA_VERSION = "relationship-p1k-protocol.v1"
RELATIONSHIP_P1K_CHECKPOINT_SCHEMA_VERSION = "relationship-p1k-checkpoint.v1"
RELATIONSHIP_P1K_READOUT_SCHEMA_VERSION = "relationship-p1k-readout.v1"
RELATIONSHIP_P1K_DECISION_SCHEMA_VERSION = "relationship-p1k-decision.v1"
RELATIONSHIP_P1K_REPORT_SCHEMA_VERSION = "relationship-p1k-report.v1"
RELATIONSHIP_P1K_PREPARED_NEXT_ACTION = "execute_frozen_oracle_disclosure_ladder"

RELATIONSHIP_P1K_READOUT_PROMPT_ASSET = "relationship_lab_oracle_concept_readout_v1.txt"
RELATIONSHIP_P1K_REQUEST_TEMPLATE_ASSET = (
    "relationship_lab_oracle_concept_readout_request_v1.txt"
)

# The disclosure ladder is evaluated on the prompt-steelman public surface so
# that every rung differs from the P1i prompt arm by disclosure alone.
RELATIONSHIP_P1K_CONTEXT_ARM = RelationshipP1Arm.PROMPT_STEELMAN

RELATIONSHIP_P1K_RUNG_MINIMUM_ACCURACY = 0.75
RELATIONSHIP_P1K_RUNG_MINIMUM_PAIR_FLIP_RATE = 0.5
RELATIONSHIP_P1K_REQUIRED_VALID_RATE = 1.0

_HEX_DIGITS = frozenset("0123456789abcdef")
_REQUEST_CONTEXT_MARKER = "{{PUBLIC_HISTORY_EVIDENCE}}"
_REQUEST_DISCLOSURE_MARKER = "{{SEALED_CONCEPT_DISCLOSURE}}"
_REQUEST_CURRENT_INPUT_MARKER = "{{CURRENT_USER_MESSAGE}}"
_CONDITION_LABELS = ("甲", "乙")

_PROTOCOL_CLAIM_BOUNDARY = (
    "P1k freezes an evaluator-only oracle disclosure ladder over the already "
    "burned v3 consumer-training package before its first output. Every rung "
    "consumes sealed generator truth and is therefore non-competitive. It does "
    "not revise the P1i consumer, the P1h gate, or any dataset, does not open "
    "formal hidden test or P2, and does not write Volvence learning or control "
    "state."
)
_REPORT_CLAIM_BOUNDARY = (
    "P1k localizes where the frozen ordinary-Qwen consumer stops being able to "
    "act on the sealed relational concept. Oracle rungs consume truth that "
    "competitive arms may never see, so their accuracy is a diagnostic ceiling "
    "and never evidence of consumer qualification, Volvence advantage, "
    "Readable/Learnable/Steerable capability, or product value. The result "
    "cannot revise the consumer, the qualification gate, or the scenario data."
)


class RelationshipP1kOracleTier(str, Enum):
    """Increasing sealed disclosure over one fixed public evidence surface."""

    CONDITION_ONLY = "oracle_condition_v1"
    CONDITION_AND_BINDING = "oracle_binding_v1"
    CONDITION_AND_POLICY = "oracle_policy_v1"


# Ordered most-disclosed first: a failure at the ceiling invalidates the rest.
RELATIONSHIP_P1K_TIERS: tuple[RelationshipP1kOracleTier, ...] = (
    RelationshipP1kOracleTier.CONDITION_AND_POLICY,
    RelationshipP1kOracleTier.CONDITION_AND_BINDING,
    RelationshipP1kOracleTier.CONDITION_ONLY,
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text


def _require_object(
    value: object,
    expected_keys: set[str],
    *,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    if set(value) != expected_keys:
        raise ValueError(f"{field_name} keys do not match the frozen schema")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        temporary = pathlib.Path(handle.name)
    temporary.replace(target)


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def relationship_p1k_readout_prompt_path() -> pathlib.Path:
    return _asset_dir() / "prompts" / RELATIONSHIP_P1K_READOUT_PROMPT_ASSET


def relationship_p1k_request_template_path() -> pathlib.Path:
    return _asset_dir() / "prompts" / RELATIONSHIP_P1K_REQUEST_TEMPLATE_ASSET


def render_relationship_p1k_request(
    *,
    context_text: str,
    disclosure_text: str,
    current_input: str,
) -> str:
    """Render the frozen oracle request from public context plus disclosure."""

    if not context_text.strip() or not current_input.strip():
        raise ValueError("P1k request requires context and current input")
    if not disclosure_text.strip():
        raise ValueError("P1k request requires a sealed disclosure block")
    template = relationship_p1k_request_template_path().read_text(encoding="utf-8")
    for marker in (
        _REQUEST_CONTEXT_MARKER,
        _REQUEST_DISCLOSURE_MARKER,
        _REQUEST_CURRENT_INPUT_MARKER,
    ):
        if template.count(marker) != 1:
            raise ValueError("P1k request markers must each occur exactly once")
    return (
        template.replace(_REQUEST_CONTEXT_MARKER, context_text)
        .replace(_REQUEST_DISCLOSURE_MARKER, disclosure_text)
        .replace(_REQUEST_CURRENT_INPUT_MARKER, current_input)
        .strip()
    )


def _condition_labels(dataset: RelationshipTransferDataset) -> dict[str, str]:
    condition_ids = sorted(item.condition_id for item in dataset.abstract_conditions)
    if len(condition_ids) != len(_CONDITION_LABELS):
        raise ValueError("P1k disclosure requires exactly two abstract conditions")
    return dict(zip(condition_ids, _CONDITION_LABELS, strict=True))


def build_relationship_p1k_disclosure(
    *,
    dataset: RelationshipTransferDataset,
    scene_id: str,
    tier: RelationshipP1kOracleTier,
) -> str:
    """Build the evaluator-only sealed disclosure block for one scene and rung.

    Raw sealed identifiers are never emitted: conditions are relabelled by a
    dataset-wide canonical ordering so the model reads the concept rather than
    an English mnemonic, and the preferred action is only ever implied by the
    disclosed condition-to-action policy.
    """

    if not isinstance(tier, RelationshipP1kOracleTier):
        raise ValueError("P1k disclosure tier must be typed")
    dynamic = dataset.dynamic_for_scene(scene_id)
    if dynamic.probe_condition_id is None or dynamic.policy_id is None:
        raise ValueError("P1k disclosure requires a compositional sealed dynamic")
    labels = _condition_labels(dataset)
    summaries = {
        item.condition_id: item.hidden_summary for item in dataset.abstract_conditions
    }
    policies = {item.policy_id: item for item in dataset.policy_profiles}
    policy = policies[dynamic.policy_id]
    if policy.action_for(dynamic.probe_condition_id) is not dynamic.preferred_action:
        raise ValueError("P1k sealed policy disagrees with the preferred action")

    lines = [
        "[evaluator-only sealed disclosure]",
        "以下内容为真值，直接采用，不需要重新推断。",
        "",
        "抽象关系条件：",
    ]
    for condition_id in sorted(summaries):
        lines.append(f"- 条件{labels[condition_id]}：{summaries[condition_id]}")
    lines.append("")
    lines.append(f"当前消息所处的抽象关系条件：条件{labels[dynamic.probe_condition_id]}")

    if tier in (
        RelationshipP1kOracleTier.CONDITION_AND_BINDING,
        RelationshipP1kOracleTier.CONDITION_AND_POLICY,
    ):
        bindings = dict(dataset.history_condition_bindings)
        observation = _observation_for_scene(dataset, scene_id)
        lines.append("")
        lines.append("每条公开历史所处的抽象关系条件：")
        for history in observation.histories:
            condition_id = bindings.get(history.event_id)
            if condition_id is None:
                raise ValueError(
                    f"P1k disclosure is missing a condition binding for {history.event_id}"
                )
            lines.append(f"- {history.event_id}：条件{labels[condition_id]}")

    if tier is RelationshipP1kOracleTier.CONDITION_AND_POLICY:
        lines.append("")
        lines.append("该用户在每种抽象关系条件下需要的关系动作：")
        for condition_id, action in policy.condition_actions:
            lines.append(f"- 条件{labels[condition_id]}：{action.value}")

    return "\n".join(lines)


def _observation_for_scene(dataset: RelationshipTransferDataset, scene_id: str):
    for observation in dataset.observations:
        if observation.scene_id == scene_id:
            return observation
    raise KeyError(scene_id)


@dataclass(frozen=True)
class RelationshipP1kRecordKey:
    tier: RelationshipP1kOracleTier
    scene_id: str
    mirror_pair_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.tier not in RELATIONSHIP_P1K_TIERS:
            raise ValueError("P1k record tier is not part of the frozen ladder")
        _require_text(self.scene_id, "P1k record scene")
        _require_text(self.mirror_pair_id, "P1k record mirror pair")
        if self.seed < 0:
            raise ValueError("P1k record seed must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "tier": self.tier.value,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "seed": self.seed,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1kRecordKey":
        raw = _require_object(
            value,
            {"tier", "scene_id", "mirror_pair_id", "seed"},
            field_name="P1k record key",
        )
        return cls(
            tier=RelationshipP1kOracleTier(_require_text(raw["tier"], "P1k record tier")),
            scene_id=_require_text(raw["scene_id"], "P1k record scene"),
            mirror_pair_id=_require_text(raw["mirror_pair_id"], "P1k record mirror pair"),
            seed=_require_int(raw["seed"], "P1k record seed"),
        )


def relationship_p1k_record_plan(
    *,
    dataset: RelationshipTransferDataset,
    seed_schedule: tuple[int, ...],
) -> tuple[RelationshipP1kRecordKey, ...]:
    """Freeze the deterministic record order, most-disclosed rung first."""

    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("P1k seed schedule must be non-empty and unique")
    keys: list[RelationshipP1kRecordKey] = []
    for tier in RELATIONSHIP_P1K_TIERS:
        for mirror_pair_id, members in dataset.mirrored_pairs():
            for seed in seed_schedule:
                for observation, _dynamic in members:
                    keys.append(
                        RelationshipP1kRecordKey(
                            tier=tier,
                            scene_id=observation.scene_id,
                            mirror_pair_id=mirror_pair_id,
                            seed=seed,
                        )
                    )
    if not keys:
        raise ValueError("P1k record plan is empty")
    return tuple(keys)


@dataclass(frozen=True)
class RelationshipP1kReadout:
    tier: RelationshipP1kOracleTier
    scene_id: str
    seed: int
    current_input_sha256: str
    context_sha256: str
    disclosure_sha256: str
    model_input_sha256: str
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
    schema_version: str = RELATIONSHIP_P1K_READOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1K_READOUT_SCHEMA_VERSION:
            raise ValueError("P1k readout schema_version mismatch")
        if self.tier not in RELATIONSHIP_P1K_TIERS:
            raise ValueError("P1k readout tier is not part of the frozen ladder")
        _require_text(self.scene_id, "P1k readout scene")
        _require_text(self.model_id, "P1k readout model")
        if self.seed < 0:
            raise ValueError("P1k readout seed must be non-negative")
        for field_name, value in (
            ("current_input_sha256", self.current_input_sha256),
            ("context_sha256", self.context_sha256),
            ("disclosure_sha256", self.disclosure_sha256),
            ("model_input_sha256", self.model_input_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("prompt_sha256", self.prompt_sha256),
            ("request_template_sha256", self.request_template_sha256),
            ("schema_sha256", self.schema_sha256),
        ):
            _require_sha256(value, field_name)
        if (self.stay_score is None) is not (self.space_score is None):
            raise ValueError("P1k readout scores must both be present or absent")
        if self.stay_score is not None:
            compile_relationship_evidence_scores(
                stay_score=self.stay_score,
                space_score=self.space_score,
            )
        for field_name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
        action = self.compiled_action
        return {
            "schema_version": self.schema_version,
            "tier": self.tier.value,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "current_input_sha256": self.current_input_sha256,
            "context_sha256": self.context_sha256,
            "disclosure_sha256": self.disclosure_sha256,
            "model_input_sha256": self.model_input_sha256,
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
            "compiled_action_id": None if action is None else action.value,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1kReadout":
        raw = _require_object(
            value,
            {
                "schema_version",
                "tier",
                "scene_id",
                "seed",
                "current_input_sha256",
                "context_sha256",
                "disclosure_sha256",
                "model_input_sha256",
                "model_id",
                "weights_sha256",
                "generation_config_sha256",
                "prompt_sha256",
                "request_template_sha256",
                "schema_sha256",
                "raw_output",
                "stay_score",
                "space_score",
                "valid",
                "compiled_action_id",
                "prompt_tokens",
                "completion_tokens",
                "artifact_id",
            },
            field_name="P1k readout record",
        )
        artifact_id = _require_sha256(raw["artifact_id"], "P1k readout artifact id")
        scores: list[int | None] = []
        for field_name in ("stay_score", "space_score"):
            value_raw = raw[field_name]
            if value_raw is None:
                scores.append(None)
            else:
                scores.append(_require_int(value_raw, f"P1k readout {field_name}"))
        readout = cls(
            schema_version=_require_text(raw["schema_version"], "P1k readout schema"),
            tier=RelationshipP1kOracleTier(_require_text(raw["tier"], "P1k readout tier")),
            scene_id=_require_text(raw["scene_id"], "P1k readout scene"),
            seed=_require_int(raw["seed"], "P1k readout seed"),
            current_input_sha256=raw["current_input_sha256"],
            context_sha256=raw["context_sha256"],
            disclosure_sha256=raw["disclosure_sha256"],
            model_input_sha256=raw["model_input_sha256"],
            model_id=_require_text(raw["model_id"], "P1k readout model"),
            weights_sha256=raw["weights_sha256"],
            generation_config_sha256=raw["generation_config_sha256"],
            prompt_sha256=raw["prompt_sha256"],
            request_template_sha256=raw["request_template_sha256"],
            schema_sha256=raw["schema_sha256"],
            raw_output=raw["raw_output"] if isinstance(raw["raw_output"], str) else "",
            stay_score=scores[0],
            space_score=scores[1],
            prompt_tokens=_require_int(raw["prompt_tokens"], "P1k prompt tokens"),
            completion_tokens=_require_int(
                raw["completion_tokens"], "P1k completion tokens"
            ),
        )
        if readout.artifact_id != artifact_id:
            raise ValueError("P1k readout artifact id mismatch")
        if raw["valid"] != readout.valid:
            raise ValueError("P1k readout validity mismatch")
        return readout


@dataclass(frozen=True)
class RelationshipP1kDecision:
    tier: RelationshipP1kOracleTier
    scene_id: str
    mirror_pair_id: str
    split: RelationshipDatasetSplit
    seed: int
    readout_artifact_id: str
    chosen_action_id: RelationshipAction | None
    expected_action_id: RelationshipAction
    schema_version: str = RELATIONSHIP_P1K_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1K_DECISION_SCHEMA_VERSION:
            raise ValueError("P1k decision schema_version mismatch")
        if self.tier not in RELATIONSHIP_P1K_TIERS:
            raise ValueError("P1k decision tier is not part of the frozen ladder")
        _require_text(self.scene_id, "P1k decision scene")
        _require_text(self.mirror_pair_id, "P1k decision mirror pair")
        _require_sha256(self.readout_artifact_id, "P1k decision readout artifact id")
        if self.seed < 0:
            raise ValueError("P1k decision seed must be non-negative")
        if self.expected_action_id is RelationshipAction.NEUTRAL_NOOP:
            raise ValueError("P1k expected action cannot be the negative control")

    @property
    def valid(self) -> bool:
        return self.chosen_action_id is not None

    @property
    def correct(self) -> bool:
        return self.chosen_action_id is self.expected_action_id

    @property
    def decision_id(self) -> str:
        return sha256_json(
            {
                "tier": self.tier.value,
                "scene_id": self.scene_id,
                "seed": self.seed,
                "readout_artifact_id": self.readout_artifact_id,
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "tier": self.tier.value,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "split": self.split.value,
            "seed": self.seed,
            "readout_artifact_id": self.readout_artifact_id,
            "chosen_action_id": (
                None if self.chosen_action_id is None else self.chosen_action_id.value
            ),
            "expected_action_id": self.expected_action_id.value,
            "valid": self.valid,
            "correct": self.correct,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1kDecision":
        raw = _require_object(
            value,
            {
                "schema_version",
                "decision_id",
                "tier",
                "scene_id",
                "mirror_pair_id",
                "split",
                "seed",
                "readout_artifact_id",
                "chosen_action_id",
                "expected_action_id",
                "valid",
                "correct",
            },
            field_name="P1k decision record",
        )
        chosen_raw = raw["chosen_action_id"]
        decision = cls(
            schema_version=_require_text(raw["schema_version"], "P1k decision schema"),
            tier=RelationshipP1kOracleTier(
                _require_text(raw["tier"], "P1k decision tier")
            ),
            scene_id=_require_text(raw["scene_id"], "P1k decision scene"),
            mirror_pair_id=_require_text(
                raw["mirror_pair_id"], "P1k decision mirror pair"
            ),
            split=RelationshipDatasetSplit(
                _require_text(raw["split"], "P1k decision split")
            ),
            seed=_require_int(raw["seed"], "P1k decision seed"),
            readout_artifact_id=raw["readout_artifact_id"],
            chosen_action_id=(
                None
                if chosen_raw is None
                else RelationshipAction(_require_text(chosen_raw, "P1k chosen action"))
            ),
            expected_action_id=RelationshipAction(
                _require_text(raw["expected_action_id"], "P1k expected action")
            ),
        )
        if (
            raw["decision_id"] != decision.decision_id
            or raw["valid"] != decision.valid
            or raw["correct"] != decision.correct
        ):
            raise ValueError("P1k decision derived values mismatch")
        return decision


@dataclass(frozen=True)
class RelationshipP1kProtocol:
    frozen_at_iso: str
    consumer_protocol_id: str
    training_dataset_fingerprint: str
    training_package_name: str
    context_arm: str
    context_surface_sha256: str
    background_templates_sha256: str
    rag_config_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    readout_prompt_sha256: str
    request_template_sha256: str
    readout_schema_sha256: str
    compiler_version: str
    tiers: tuple[str, ...]
    seed_schedule: tuple[int, ...]
    record_plan_sha256: str
    planned_output_count: int
    observation_count: int
    rung_minimum_accuracy: float
    rung_minimum_pair_flip_rate: float
    required_valid_rate: float
    claim_boundary: str = _PROTOCOL_CLAIM_BOUNDARY
    next_action: str = RELATIONSHIP_P1K_PREPARED_NEXT_ACTION
    schema_version: str = RELATIONSHIP_P1K_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1K_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1k protocol schema_version mismatch")
        _require_timestamp(self.frozen_at_iso, "P1k protocol timestamp")
        for field_name, value in (
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
            ("context_surface_sha256", self.context_surface_sha256),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("readout_prompt_sha256", self.readout_prompt_sha256),
            ("request_template_sha256", self.request_template_sha256),
            ("readout_schema_sha256", self.readout_schema_sha256),
            ("record_plan_sha256", self.record_plan_sha256),
        ):
            _require_sha256(value, field_name)
        if self.training_package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1k must run on the burned v3 consumer-training package")
        if self.context_arm != RELATIONSHIP_P1K_CONTEXT_ARM.value:
            raise ValueError("P1k context arm is not frozen")
        if self.compiler_version != RELATIONSHIP_P1B_COMPILER_VERSION:
            raise ValueError("P1k compiler version mismatch")
        if self.tiers != tuple(tier.value for tier in RELATIONSHIP_P1K_TIERS):
            raise ValueError("P1k tiers are not the frozen ladder")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(
            self.seed_schedule
        ):
            raise ValueError("P1k seed schedule must be non-empty and unique")
        if self.observation_count <= 0 or self.planned_output_count != (
            self.observation_count * len(self.tiers) * len(self.seed_schedule)
        ):
            raise ValueError("P1k planned output count diverges from the ladder")
        if self.rung_minimum_accuracy != RELATIONSHIP_P1K_RUNG_MINIMUM_ACCURACY:
            raise ValueError("P1k rung accuracy threshold is not frozen")
        if (
            self.rung_minimum_pair_flip_rate
            != RELATIONSHIP_P1K_RUNG_MINIMUM_PAIR_FLIP_RATE
        ):
            raise ValueError("P1k rung pair-flip threshold is not frozen")
        if self.required_valid_rate != RELATIONSHIP_P1K_REQUIRED_VALID_RATE:
            raise ValueError("P1k valid-rate threshold is not frozen")
        if self.claim_boundary != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1k protocol claim boundary is not frozen")
        if self.next_action != RELATIONSHIP_P1K_PREPARED_NEXT_ACTION:
            raise ValueError("P1k protocol next action is not frozen")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source_lineage": {
                "consumer_protocol_id": self.consumer_protocol_id,
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
                "training_package_name": self.training_package_name,
            },
            "frozen_context": {
                "context_arm": self.context_arm,
                "context_surface_sha256": self.context_surface_sha256,
                "background_templates_sha256": self.background_templates_sha256,
                "rag_config_sha256": self.rag_config_sha256,
            },
            "runtime": {
                "model_id": self.model_id,
                "weights_sha256": self.weights_sha256,
                "generation_config_sha256": self.generation_config_sha256,
                "readout_prompt_sha256": self.readout_prompt_sha256,
                "request_template_sha256": self.request_template_sha256,
                "readout_schema_sha256": self.readout_schema_sha256,
                "compiler_version": self.compiler_version,
                "seed_schedule": list(self.seed_schedule),
            },
            "execution_plan": {
                "tiers": list(self.tiers),
                "record_plan_sha256": self.record_plan_sha256,
                "planned_output_count": self.planned_output_count,
                "observation_count": self.observation_count,
            },
            "diagnostic_gate": {
                "rung_minimum_accuracy": self.rung_minimum_accuracy,
                "rung_minimum_pair_flip_rate": self.rung_minimum_pair_flip_rate,
                "required_valid_rate": self.required_valid_rate,
            },
            "experiment_guards": {
                "competitive": False,
                "consumes_sealed_generator_truth": True,
                "diagnostic_feedback_to_consumer": False,
                "diagnostic_feedback_to_dataset": False,
                "diagnostic_feedback_to_qualification_gate": False,
                "evaluation_feedback_to_pe_credit_reward_or_steering": False,
                "formal_hidden_test_opened": False,
                "p2_enabled": False,
            },
            "claim_boundary": self.claim_boundary,
            "next_action": self.next_action,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self.to_payload())


def freeze_relationship_p1k_protocol(
    *,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    dataset: RelationshipTransferDataset,
    contexts: RelationshipP1ContextBundle,
    seed_schedule: tuple[int, ...],
    frozen_at_iso: str | None = None,
) -> RelationshipP1kProtocol:
    """Freeze the complete oracle ladder before its first model output."""

    if dataset.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
        raise ValueError("P1k diagnostic must consume the burned v3 package")
    if dataset.dataset_fingerprint != consumer.training_dataset_fingerprint:
        raise ValueError("P1k dataset diverges from the frozen consumer training split")
    if contexts.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("P1k context bundle dataset fingerprint mismatch")
    if contexts.background_templates_sha256 != consumer.background_templates_sha256:
        raise ValueError("P1k background templates diverge from frozen consumer")
    if contexts.rag_config_sha256 != consumer.rag_config_sha256:
        raise ValueError("P1k RAG config diverges from frozen consumer")
    if tuple(seed_schedule) != consumer.seed_schedule:
        raise ValueError("P1k seed schedule diverges from frozen consumer")
    record_plan = relationship_p1k_record_plan(
        dataset=dataset,
        seed_schedule=seed_schedule,
    )
    return RelationshipP1kProtocol(
        frozen_at_iso=frozen_at_iso or datetime.now(timezone.utc).isoformat(),
        consumer_protocol_id=consumer.protocol_id,
        training_dataset_fingerprint=dataset.dataset_fingerprint,
        training_package_name=dataset.package_name,
        context_arm=RELATIONSHIP_P1K_CONTEXT_ARM.value,
        context_surface_sha256=relationship_p1_evaluated_context_surface_sha256(
            bundle=contexts,
            dataset=dataset,
        ),
        background_templates_sha256=contexts.background_templates_sha256,
        rag_config_sha256=contexts.rag_config_sha256,
        model_id=consumer.model_id,
        weights_sha256=consumer.expected_weights_sha256,
        generation_config_sha256=consumer.expected_generation_config_sha256,
        readout_prompt_sha256=_sha256_file(relationship_p1k_readout_prompt_path()),
        request_template_sha256=_sha256_file(relationship_p1k_request_template_path()),
        readout_schema_sha256=_sha256_file(relationship_p1b_readout_schema_path()),
        compiler_version=RELATIONSHIP_P1B_COMPILER_VERSION,
        tiers=tuple(tier.value for tier in RELATIONSHIP_P1K_TIERS),
        seed_schedule=tuple(seed_schedule),
        record_plan_sha256=sha256_json(
            [key.to_payload() for key in record_plan]
        ),
        planned_output_count=len(record_plan),
        observation_count=len(dataset.observations),
        rung_minimum_accuracy=RELATIONSHIP_P1K_RUNG_MINIMUM_ACCURACY,
        rung_minimum_pair_flip_rate=RELATIONSHIP_P1K_RUNG_MINIMUM_PAIR_FLIP_RATE,
        required_valid_rate=RELATIONSHIP_P1K_REQUIRED_VALID_RATE,
    )


def write_relationship_p1k_protocol(
    protocol: RelationshipP1kProtocol,
    path: pathlib.Path,
) -> pathlib.Path:
    payload = protocol.to_payload()
    payload["protocol_id"] = protocol.protocol_id
    _atomic_write_text(
        pathlib.Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return pathlib.Path(path)


def load_relationship_p1k_protocol(path: pathlib.Path) -> RelationshipP1kProtocol:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P1k protocol must be a JSON object")
    protocol_id = _require_sha256(raw.pop("protocol_id", None), "P1k protocol id")
    lineage = raw["source_lineage"]
    context = raw["frozen_context"]
    runtime = raw["runtime"]
    plan = raw["execution_plan"]
    gate = raw["diagnostic_gate"]
    protocol = RelationshipP1kProtocol(
        frozen_at_iso=raw["frozen_at_iso"],
        consumer_protocol_id=lineage["consumer_protocol_id"],
        training_dataset_fingerprint=lineage["training_dataset_fingerprint"],
        training_package_name=lineage["training_package_name"],
        context_arm=context["context_arm"],
        context_surface_sha256=context["context_surface_sha256"],
        background_templates_sha256=context["background_templates_sha256"],
        rag_config_sha256=context["rag_config_sha256"],
        model_id=runtime["model_id"],
        weights_sha256=runtime["weights_sha256"],
        generation_config_sha256=runtime["generation_config_sha256"],
        readout_prompt_sha256=runtime["readout_prompt_sha256"],
        request_template_sha256=runtime["request_template_sha256"],
        readout_schema_sha256=runtime["readout_schema_sha256"],
        compiler_version=runtime["compiler_version"],
        tiers=tuple(plan["tiers"]),
        seed_schedule=tuple(runtime["seed_schedule"]),
        record_plan_sha256=plan["record_plan_sha256"],
        planned_output_count=_require_int(
            plan["planned_output_count"], "P1k planned outputs"
        ),
        observation_count=_require_int(
            plan["observation_count"], "P1k observation count"
        ),
        rung_minimum_accuracy=_require_number(
            gate["rung_minimum_accuracy"], "P1k rung accuracy"
        ),
        rung_minimum_pair_flip_rate=_require_number(
            gate["rung_minimum_pair_flip_rate"], "P1k rung pair flip"
        ),
        required_valid_rate=_require_number(
            gate["required_valid_rate"], "P1k valid rate"
        ),
        claim_boundary=raw["claim_boundary"],
        next_action=raw["next_action"],
        schema_version=raw["schema_version"],
    )
    if protocol.protocol_id != protocol_id:
        raise ValueError("P1k protocol id mismatch")
    return protocol


def validate_relationship_p1k_protocol_lineage(
    protocol: RelationshipP1kProtocol,
    *,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    dataset: RelationshipTransferDataset,
    contexts: RelationshipP1ContextBundle,
) -> None:
    """Reject a freeze whose public surface or substrate drifted."""

    expected = freeze_relationship_p1k_protocol(
        consumer=consumer,
        dataset=dataset,
        contexts=contexts,
        seed_schedule=protocol.seed_schedule,
        frozen_at_iso=protocol.frozen_at_iso,
    )
    if protocol != expected:
        raise ValueError("P1k protocol lineage mismatch")
    if protocol.consumer_protocol_id != consumer.protocol_id:
        raise ValueError("P1k protocol consumer id mismatch")
    if protocol.model_id != consumer.model_id:
        raise ValueError("P1k protocol substrate identity mismatch")
    if protocol.weights_sha256 != consumer.expected_weights_sha256:
        raise ValueError("P1k protocol weights mismatch")
    if protocol.generation_config_sha256 != consumer.expected_generation_config_sha256:
        raise ValueError("P1k protocol generation config mismatch")


@dataclass(frozen=True)
class RelationshipP1kCheckpoint:
    protocol_id: str
    consumer_protocol_id: str
    training_dataset_fingerprint: str
    context_surface_sha256: str
    planned_record_keys: tuple[RelationshipP1kRecordKey, ...]
    schema_version: str = RELATIONSHIP_P1K_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1K_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("P1k checkpoint schema_version mismatch")
        for field_name, value in (
            ("protocol_id", self.protocol_id),
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
            ("context_surface_sha256", self.context_surface_sha256),
        ):
            _require_sha256(value, field_name)
        if not self.planned_record_keys:
            raise ValueError("P1k checkpoint requires a record plan")
        identities = tuple(
            (item.tier, item.scene_id, item.seed) for item in self.planned_record_keys
        )
        if len(set(identities)) != len(identities):
            raise ValueError("P1k checkpoint record keys must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "consumer_protocol_id": self.consumer_protocol_id,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "context_surface_sha256": self.context_surface_sha256,
            "planned_record_keys": [
                item.to_payload() for item in self.planned_record_keys
            ],
        }


def build_relationship_p1k_checkpoint(
    *,
    protocol: RelationshipP1kProtocol,
    dataset: RelationshipTransferDataset,
) -> RelationshipP1kCheckpoint:
    record_plan = relationship_p1k_record_plan(
        dataset=dataset,
        seed_schedule=protocol.seed_schedule,
    )
    if sha256_json([key.to_payload() for key in record_plan]) != (
        protocol.record_plan_sha256
    ):
        raise ValueError("P1k record plan diverges from the frozen protocol")
    return RelationshipP1kCheckpoint(
        protocol_id=protocol.protocol_id,
        consumer_protocol_id=protocol.consumer_protocol_id,
        training_dataset_fingerprint=protocol.training_dataset_fingerprint,
        context_surface_sha256=protocol.context_surface_sha256,
        planned_record_keys=record_plan,
    )


def write_relationship_p1k_checkpoint(
    *,
    checkpoint: RelationshipP1kCheckpoint,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "checkpoint.json"
    _atomic_write_text(
        path,
        json.dumps(
            checkpoint.to_payload(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    )
    return path


def load_relationship_p1k_checkpoint(
    output_dir: pathlib.Path,
) -> RelationshipP1kCheckpoint:
    path = pathlib.Path(output_dir) / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _require_object(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "protocol_id",
            "consumer_protocol_id",
            "training_dataset_fingerprint",
            "context_surface_sha256",
            "planned_record_keys",
        },
        field_name="P1k checkpoint",
    )
    keys_raw = raw["planned_record_keys"]
    if not isinstance(keys_raw, list):
        raise ValueError("P1k checkpoint record keys must be a list")
    return RelationshipP1kCheckpoint(
        schema_version=_require_text(raw["schema_version"], "P1k checkpoint schema"),
        protocol_id=raw["protocol_id"],
        consumer_protocol_id=raw["consumer_protocol_id"],
        training_dataset_fingerprint=raw["training_dataset_fingerprint"],
        context_surface_sha256=raw["context_surface_sha256"],
        planned_record_keys=tuple(
            RelationshipP1kRecordKey.from_payload(item) for item in keys_raw
        ),
    )


@dataclass(frozen=True)
class RelationshipP1kProgress:
    checkpoint: RelationshipP1kCheckpoint
    readouts: tuple[RelationshipP1kReadout, ...]
    decisions: tuple[RelationshipP1kDecision, ...]

    def __post_init__(self) -> None:
        if len(self.decisions) > len(self.readouts) or (
            len(self.readouts) - len(self.decisions) > 1
        ):
            raise ValueError("P1k progress readout/decision counts are invalid")
        planned = tuple(
            (item.tier, item.scene_id, item.seed)
            for item in self.checkpoint.planned_record_keys
        )
        readout_keys = tuple(
            (item.tier, item.scene_id, item.seed) for item in self.readouts
        )
        decision_keys = tuple(
            (item.tier, item.scene_id, item.seed) for item in self.decisions
        )
        if readout_keys != planned[: len(readout_keys)]:
            raise ValueError("P1k readouts are not a contiguous planned prefix")
        if decision_keys != planned[: len(decision_keys)]:
            raise ValueError("P1k decisions are not a contiguous planned prefix")

    @property
    def is_complete(self) -> bool:
        expected = len(self.checkpoint.planned_record_keys)
        return len(self.readouts) == len(self.decisions) == expected


def _record_path(output_dir: pathlib.Path, index: int, kind: str) -> pathlib.Path:
    return pathlib.Path(output_dir) / "records" / f"{index:04d}.{kind}.json"


def persist_relationship_p1k_readout(
    *,
    checkpoint: RelationshipP1kCheckpoint,
    output_dir: pathlib.Path,
    index: int,
    readout: RelationshipP1kReadout,
) -> pathlib.Path:
    if index < 0 or index >= len(checkpoint.planned_record_keys):
        raise IndexError("P1k readout index is outside the record plan")
    key = checkpoint.planned_record_keys[index]
    if (readout.tier, readout.scene_id, readout.seed) != (
        key.tier,
        key.scene_id,
        key.seed,
    ):
        raise ValueError("P1k readout key diverges from record plan")
    path = _record_path(output_dir, index, "readout")
    if path.exists():
        raise FileExistsError(f"P1k readout already exists: {path}")
    payload = readout.to_payload()
    payload["artifact_id"] = readout.artifact_id
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path


def persist_relationship_p1k_decision(
    *,
    checkpoint: RelationshipP1kCheckpoint,
    output_dir: pathlib.Path,
    index: int,
    decision: RelationshipP1kDecision,
) -> pathlib.Path:
    readout_path = _record_path(output_dir, index, "readout")
    if not readout_path.is_file():
        raise FileNotFoundError("P1k decision cannot precede its durable readout")
    key = checkpoint.planned_record_keys[index]
    if (decision.tier, decision.scene_id, decision.seed) != (
        key.tier,
        key.scene_id,
        key.seed,
    ):
        raise ValueError("P1k decision key diverges from record plan")
    path = _record_path(output_dir, index, "decision")
    if path.exists():
        raise FileExistsError(f"P1k decision already exists: {path}")
    _atomic_write_text(
        path,
        json.dumps(decision.to_payload(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return path


def load_relationship_p1k_progress(
    output_dir: pathlib.Path,
) -> RelationshipP1kProgress:
    checkpoint = load_relationship_p1k_checkpoint(output_dir)
    readouts: list[RelationshipP1kReadout] = []
    decisions: list[RelationshipP1kDecision] = []
    missing_seen = False
    for index in range(len(checkpoint.planned_record_keys)):
        readout_path = _record_path(output_dir, index, "readout")
        decision_path = _record_path(output_dir, index, "decision")
        if readout_path.is_file():
            if missing_seen:
                raise ValueError("P1k readout files contain a non-contiguous gap")
            readouts.append(
                RelationshipP1kReadout.from_payload(
                    json.loads(readout_path.read_text(encoding="utf-8"))
                )
            )
        else:
            missing_seen = True
        if decision_path.is_file():
            if not readout_path.is_file() or len(decisions) != index:
                raise ValueError("P1k decision files contain a gap or orphan")
            decisions.append(
                RelationshipP1kDecision.from_payload(
                    json.loads(decision_path.read_text(encoding="utf-8"))
                )
            )
    return RelationshipP1kProgress(
        checkpoint=checkpoint,
        readouts=tuple(readouts),
        decisions=tuple(decisions),
    )


def validate_relationship_p1k_progress(
    progress: RelationshipP1kProgress,
    *,
    protocol: RelationshipP1kProtocol,
    dataset: RelationshipTransferDataset,
    contexts: RelationshipP1ContextBundle,
) -> None:
    """Reject any resumed prefix whose lineage diverges from the freeze."""

    checkpoint = progress.checkpoint
    if (
        checkpoint.protocol_id != protocol.protocol_id
        or checkpoint.consumer_protocol_id != protocol.consumer_protocol_id
        or checkpoint.training_dataset_fingerprint
        != protocol.training_dataset_fingerprint
        or checkpoint.context_surface_sha256 != protocol.context_surface_sha256
    ):
        raise ValueError("P1k checkpoint diverges from the frozen protocol")
    if dataset.dataset_fingerprint != protocol.training_dataset_fingerprint:
        raise ValueError("P1k dataset diverges from the frozen protocol")
    if contexts.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("P1k context bundle diverges from the dataset")
    for index, readout in enumerate(progress.readouts):
        key = checkpoint.planned_record_keys[index]
        observation = _observation_for_scene(dataset, key.scene_id)
        context = contexts.context(
            scene_id=key.scene_id,
            arm=RELATIONSHIP_P1K_CONTEXT_ARM,
        )
        disclosure = build_relationship_p1k_disclosure(
            dataset=dataset,
            scene_id=key.scene_id,
            tier=key.tier,
        )
        expected_input = render_relationship_p1k_request(
            context_text=context.context_text,
            disclosure_text=disclosure,
            current_input=observation.current_input,
        )
        if (
            readout.current_input_sha256
            != _sha256_text(observation.current_input)
            or readout.context_sha256 != context.context_sha256
            or readout.disclosure_sha256 != _sha256_text(disclosure)
            or readout.model_input_sha256 != _sha256_text(expected_input)
        ):
            raise ValueError("P1k resumed readout input lineage mismatch")
        if (
            readout.model_id != protocol.model_id
            or readout.weights_sha256 != protocol.weights_sha256
            or readout.generation_config_sha256 != protocol.generation_config_sha256
            or readout.prompt_sha256 != protocol.readout_prompt_sha256
            or readout.request_template_sha256 != protocol.request_template_sha256
            or readout.schema_sha256 != protocol.readout_schema_sha256
        ):
            raise ValueError("P1k resumed readout runtime lineage mismatch")
        if index < len(progress.decisions):
            decision = progress.decisions[index]
            expected = _expected_decision(
                readout=readout,
                key=key,
                dataset=dataset,
            )
            if decision != expected:
                raise ValueError("P1k resumed decision diverges from evaluator truth")


def _expected_decision(
    *,
    readout: RelationshipP1kReadout,
    key: RelationshipP1kRecordKey,
    dataset: RelationshipTransferDataset,
) -> RelationshipP1kDecision:
    dynamic = dataset.dynamic_for_scene(key.scene_id)
    return RelationshipP1kDecision(
        tier=key.tier,
        scene_id=key.scene_id,
        mirror_pair_id=key.mirror_pair_id,
        split=dynamic.split,
        seed=key.seed,
        readout_artifact_id=readout.artifact_id,
        chosen_action_id=readout.compiled_action,
        expected_action_id=dynamic.preferred_action,
    )


@dataclass(frozen=True)
class RelationshipP1kExecution:
    readouts: tuple[RelationshipP1kReadout, ...]
    decisions: tuple[RelationshipP1kDecision, ...]
    new_outputs: int


def execute_relationship_p1k_diagnostic(
    policy: ContextualRelationshipActionPolicy,
    *,
    protocol: RelationshipP1kProtocol,
    dataset: RelationshipTransferDataset,
    contexts: RelationshipP1ContextBundle,
    existing_progress: RelationshipP1kProgress,
    max_new_readouts: int | None = None,
    readout_observer: Callable[[int, RelationshipP1kReadout], None] | None = None,
    decision_observer: Callable[[int, RelationshipP1kDecision], None] | None = None,
) -> RelationshipP1kExecution:
    """Advance the frozen ladder, persisting each readout before its truth."""

    if max_new_readouts is not None and max_new_readouts < 0:
        raise ValueError("P1k max_new_readouts must be non-negative")
    validate_relationship_p1k_progress(
        existing_progress,
        protocol=protocol,
        dataset=dataset,
        contexts=contexts,
    )
    if (
        policy.model_id != protocol.model_id
        or policy.weights_sha256 != protocol.weights_sha256
        or policy.generation_config_sha256 != protocol.generation_config_sha256
    ):
        raise ValueError("P1k policy diverges from the frozen substrate")
    prompt_path = relationship_p1k_readout_prompt_path()
    if _sha256_file(prompt_path) != protocol.readout_prompt_sha256:
        raise ValueError("P1k readout prompt asset drifted")
    if _sha256_file(relationship_p1k_request_template_path()) != (
        protocol.request_template_sha256
    ):
        raise ValueError("P1k request template asset drifted")
    prompt = prompt_path.read_text(encoding="utf-8").strip()

    readouts = list(existing_progress.readouts)
    decisions = list(existing_progress.decisions)
    new_outputs = 0
    for index, key in enumerate(existing_progress.checkpoint.planned_record_keys):
        if index < len(readouts):
            readout = readouts[index]
        else:
            if max_new_readouts is not None and new_outputs >= max_new_readouts:
                break
            observation = _observation_for_scene(dataset, key.scene_id)
            context = contexts.context(
                scene_id=key.scene_id,
                arm=RELATIONSHIP_P1K_CONTEXT_ARM,
            )
            disclosure = build_relationship_p1k_disclosure(
                dataset=dataset,
                scene_id=key.scene_id,
                tier=key.tier,
            )
            request = render_relationship_p1k_request(
                context_text=context.context_text,
                disclosure_text=disclosure,
                current_input=observation.current_input,
            )
            completion = policy.choose_from_messages(
                messages=(
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request},
                ),
                seed=key.seed,
            )
            stay_score, space_score = parse_relationship_evidence_scores(
                completion.raw_output
            )
            readout = RelationshipP1kReadout(
                tier=key.tier,
                scene_id=key.scene_id,
                seed=key.seed,
                current_input_sha256=_sha256_text(observation.current_input),
                context_sha256=context.context_sha256,
                disclosure_sha256=_sha256_text(disclosure),
                model_input_sha256=_sha256_text(request),
                model_id=policy.model_id,
                weights_sha256=policy.weights_sha256,
                generation_config_sha256=policy.generation_config_sha256,
                prompt_sha256=protocol.readout_prompt_sha256,
                request_template_sha256=protocol.request_template_sha256,
                schema_sha256=protocol.readout_schema_sha256,
                raw_output=completion.raw_output,
                stay_score=stay_score,
                space_score=space_score,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
            readouts.append(readout)
            new_outputs += 1
            if readout_observer is not None:
                readout_observer(index, readout)
        if index < len(decisions):
            continue
        decision = _expected_decision(readout=readout, key=key, dataset=dataset)
        decisions.append(decision)
        if decision_observer is not None:
            decision_observer(index, decision)
    return RelationshipP1kExecution(
        readouts=tuple(readouts),
        decisions=tuple(decisions),
        new_outputs=new_outputs,
    )


@dataclass(frozen=True)
class RelationshipP1kTierMetric:
    tier: str
    decisions: int
    valid_decisions: int
    valid_rate: float
    correct_decisions: int
    accuracy: float
    pair_groups: int
    valid_pair_groups: int
    pair_flip_rate: float
    prompt_tokens_total: int
    completion_tokens_total: int

    def __post_init__(self) -> None:
        if self.tier not in {item.value for item in RELATIONSHIP_P1K_TIERS}:
            raise ValueError("P1k metric tier is not part of the frozen ladder")
        if self.decisions <= 0:
            raise ValueError("P1k metric requires decisions")
        for field_name in ("valid_rate", "accuracy", "pair_flip_rate"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"P1k metric {field_name} must be in [0, 1]")

    @property
    def functional(self) -> bool:
        return (
            self.accuracy >= RELATIONSHIP_P1K_RUNG_MINIMUM_ACCURACY
            and self.pair_flip_rate >= RELATIONSHIP_P1K_RUNG_MINIMUM_PAIR_FLIP_RATE
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "decisions": self.decisions,
            "valid_decisions": self.valid_decisions,
            "valid_rate": self.valid_rate,
            "correct_decisions": self.correct_decisions,
            "accuracy": self.accuracy,
            "pair_groups": self.pair_groups,
            "valid_pair_groups": self.valid_pair_groups,
            "pair_flip_rate": self.pair_flip_rate,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "functional": self.functional,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1kTierMetric":
        raw = _require_object(
            value,
            {
                "tier",
                "decisions",
                "valid_decisions",
                "valid_rate",
                "correct_decisions",
                "accuracy",
                "pair_groups",
                "valid_pair_groups",
                "pair_flip_rate",
                "prompt_tokens_total",
                "completion_tokens_total",
                "functional",
            },
            field_name="P1k tier metric",
        )
        metric = cls(
            tier=_require_text(raw["tier"], "P1k metric tier"),
            decisions=_require_int(raw["decisions"], "P1k metric decisions"),
            valid_decisions=_require_int(
                raw["valid_decisions"], "P1k metric valid decisions"
            ),
            valid_rate=_require_number(raw["valid_rate"], "P1k metric valid rate"),
            correct_decisions=_require_int(
                raw["correct_decisions"], "P1k metric correct decisions"
            ),
            accuracy=_require_number(raw["accuracy"], "P1k metric accuracy"),
            pair_groups=_require_int(raw["pair_groups"], "P1k metric pair groups"),
            valid_pair_groups=_require_int(
                raw["valid_pair_groups"], "P1k metric valid pair groups"
            ),
            pair_flip_rate=_require_number(
                raw["pair_flip_rate"], "P1k metric pair flip"
            ),
            prompt_tokens_total=_require_int(
                raw["prompt_tokens_total"], "P1k metric prompt tokens"
            ),
            completion_tokens_total=_require_int(
                raw["completion_tokens_total"], "P1k metric completion tokens"
            ),
        )
        if raw["functional"] != metric.functional:
            raise ValueError("P1k metric functional flag mismatch")
        return metric


def _tier_metric(
    *,
    tier: RelationshipP1kOracleTier,
    decisions: tuple[RelationshipP1kDecision, ...],
    readouts: tuple[RelationshipP1kReadout, ...],
) -> RelationshipP1kTierMetric:
    selected = tuple(item for item in decisions if item.tier is tier)
    if not selected:
        raise ValueError(f"P1k has no decisions for {tier.value}")
    by_key = {(item.tier, item.scene_id, item.seed): item for item in readouts}
    selected_readouts = tuple(
        by_key[(item.tier, item.scene_id, item.seed)] for item in selected
    )
    groups: dict[tuple[str, int], list[RelationshipP1kDecision]] = {}
    for item in selected:
        groups.setdefault((item.mirror_pair_id, item.seed), []).append(item)
    valid_groups = 0
    flip_groups = 0
    for group in groups.values():
        if len(group) != 2:
            raise ValueError("P1k mirrored metric group must contain two decisions")
        if all(item.chosen_action_id is not None for item in group):
            valid_groups += 1
            flip_groups += int(
                {item.chosen_action_id for item in group}
                == {
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                }
            )
    valid = sum(int(item.valid) for item in selected)
    correct = sum(int(item.correct) for item in selected)
    return RelationshipP1kTierMetric(
        tier=tier.value,
        decisions=len(selected),
        valid_decisions=valid,
        valid_rate=valid / len(selected),
        correct_decisions=correct,
        accuracy=correct / len(selected),
        pair_groups=len(groups),
        valid_pair_groups=valid_groups,
        pair_flip_rate=flip_groups / valid_groups if valid_groups else 0.0,
        prompt_tokens_total=sum(item.prompt_tokens for item in selected_readouts),
        completion_tokens_total=sum(
            item.completion_tokens for item in selected_readouts
        ),
    )


class RelationshipP1kVerdict(str, Enum):
    MACHINERY_REGRESSION = "oracle_machinery_regression"
    SUBSTRATE_APPLICATION_FLOOR = "substrate_cannot_apply_disclosed_policy"
    POLICY_INDUCTION_BOTTLENECK = "cannot_induce_policy_from_labelled_evidence"
    CONDITION_RECOGNITION_BOTTLENECK = "cannot_recognize_probe_condition"
    UNAIDED_INDUCTION_BOTTLENECK = "oracle_ladder_intact_gap_is_unaided_induction"


_NEXT_ACTIONS = {
    RelationshipP1kVerdict.MACHINERY_REGRESSION: (
        "stop_diagnostic_lane_preserve_failed_attempt"
    ),
    RelationshipP1kVerdict.SUBSTRATE_APPLICATION_FLOOR: (
        "stop_scenario_lane_change_substrate_or_readout_floor"
    ),
    RelationshipP1kVerdict.POLICY_INDUCTION_BOTTLENECK: (
        "scope_next_packet_to_policy_induction_owner"
    ),
    RelationshipP1kVerdict.CONDITION_RECOGNITION_BOTTLENECK: (
        "scope_next_packet_to_condition_recognition_owner"
    ),
    RelationshipP1kVerdict.UNAIDED_INDUCTION_BOTTLENECK: (
        "scope_next_packet_to_unaided_induction_owner"
    ),
}


def _verdict_from_metrics(
    metrics: tuple[RelationshipP1kTierMetric, ...],
) -> RelationshipP1kVerdict:
    by_tier = {item.tier: item for item in metrics}
    if tuple(by_tier) != tuple(tier.value for tier in RELATIONSHIP_P1K_TIERS):
        raise ValueError("P1k metrics are incomplete or unordered")
    if any(
        item.valid_rate != RELATIONSHIP_P1K_REQUIRED_VALID_RATE for item in metrics
    ):
        return RelationshipP1kVerdict.MACHINERY_REGRESSION
    if not by_tier[RelationshipP1kOracleTier.CONDITION_AND_POLICY.value].functional:
        return RelationshipP1kVerdict.SUBSTRATE_APPLICATION_FLOOR
    if not by_tier[RelationshipP1kOracleTier.CONDITION_AND_BINDING.value].functional:
        return RelationshipP1kVerdict.POLICY_INDUCTION_BOTTLENECK
    if not by_tier[RelationshipP1kOracleTier.CONDITION_ONLY.value].functional:
        return RelationshipP1kVerdict.CONDITION_RECOGNITION_BOTTLENECK
    return RelationshipP1kVerdict.UNAIDED_INDUCTION_BOTTLENECK


@dataclass(frozen=True)
class RelationshipP1kReport:
    created_at_iso: str
    protocol_id: str
    consumer_protocol_id: str
    training_dataset_fingerprint: str
    context_surface_sha256: str
    model_id: str
    weights_sha256: str
    readout_ledger_sha256: str
    decision_ledger_sha256: str
    tier_metrics: tuple[RelationshipP1kTierMetric, ...]
    output_count: int
    verdict: RelationshipP1kVerdict
    claim_boundary: str = _REPORT_CLAIM_BOUNDARY
    schema_version: str = RELATIONSHIP_P1K_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1K_REPORT_SCHEMA_VERSION:
            raise ValueError("P1k report schema_version mismatch")
        _require_timestamp(self.created_at_iso, "P1k report timestamp")
        for field_name, value in (
            ("protocol_id", self.protocol_id),
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
            ("context_surface_sha256", self.context_surface_sha256),
            ("weights_sha256", self.weights_sha256),
            ("readout_ledger_sha256", self.readout_ledger_sha256),
            ("decision_ledger_sha256", self.decision_ledger_sha256),
        ):
            _require_sha256(value, field_name)
        if tuple(item.tier for item in self.tier_metrics) != tuple(
            tier.value for tier in RELATIONSHIP_P1K_TIERS
        ):
            raise ValueError("P1k report tier metrics are incomplete or unordered")
        if self.output_count != sum(item.decisions for item in self.tier_metrics):
            raise ValueError("P1k report output count diverges from tier metrics")
        if self.verdict is not _verdict_from_metrics(self.tier_metrics):
            raise ValueError("P1k report verdict diverges from its own metrics")
        if self.claim_boundary != _REPORT_CLAIM_BOUNDARY:
            raise ValueError("P1k report claim boundary is not frozen")

    @property
    def next_action(self) -> str:
        return _NEXT_ACTIONS[self.verdict]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "protocol_id": self.protocol_id,
            "consumer_protocol_id": self.consumer_protocol_id,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "context_surface_sha256": self.context_surface_sha256,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "readout_ledger_sha256": self.readout_ledger_sha256,
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "tier_metrics": [item.to_payload() for item in self.tier_metrics],
            "output_count": self.output_count,
            "verdict": self.verdict.value,
            "next_action": self.next_action,
            "experiment_guards": {
                "competitive": False,
                "diagnostic_feedback_to_consumer": False,
                "diagnostic_feedback_to_dataset": False,
                "diagnostic_feedback_to_qualification_gate": False,
                "evaluation_feedback_to_pe_credit_reward_or_steering": False,
                "formal_hidden_test_opened": False,
                "p2_enabled": False,
            },
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def assess_relationship_p1k_diagnostic(
    *,
    protocol: RelationshipP1kProtocol,
    progress: RelationshipP1kProgress,
    created_at_iso: str | None = None,
) -> RelationshipP1kReport:
    if not progress.is_complete:
        raise ValueError("P1k assessment requires a complete ladder")
    metrics = tuple(
        _tier_metric(
            tier=tier,
            decisions=progress.decisions,
            readouts=progress.readouts,
        )
        for tier in RELATIONSHIP_P1K_TIERS
    )
    readout_ledger = "".join(
        canonical_json({**item.to_payload(), "artifact_id": item.artifact_id}) + "\n"
        for item in progress.readouts
    )
    decision_ledger = "".join(
        canonical_json(item.to_payload()) + "\n" for item in progress.decisions
    )
    return RelationshipP1kReport(
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
        protocol_id=protocol.protocol_id,
        consumer_protocol_id=protocol.consumer_protocol_id,
        training_dataset_fingerprint=protocol.training_dataset_fingerprint,
        context_surface_sha256=protocol.context_surface_sha256,
        model_id=protocol.model_id,
        weights_sha256=protocol.weights_sha256,
        readout_ledger_sha256=_sha256_text(readout_ledger),
        decision_ledger_sha256=_sha256_text(decision_ledger),
        tier_metrics=metrics,
        output_count=len(progress.decisions),
        verdict=_verdict_from_metrics(metrics),
    )


def write_relationship_p1k_report(
    *,
    report: RelationshipP1kReport,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    json_path = target / "packet1k_report.json"
    markdown_path = target / "packet1k_report.md"
    existing = tuple(path for path in (json_path, markdown_path) if path.exists())
    if existing:
        raise FileExistsError(f"P1k report already exists: {existing}")
    payload = report.to_payload()
    payload["artifact_id"] = report.artifact_id
    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# Relationship Lab P1k oracle disclosure ladder",
        "",
        f"- artifact_id: `{report.artifact_id}`",
        f"- protocol_id: `{report.protocol_id}`",
        f"- consumer_protocol_id: `{report.consumer_protocol_id}`",
        f"- verdict: **{report.verdict.value}**",
        f"- next_action: `{report.next_action}`",
        "",
        "| Tier | valid | accuracy | pair flip | functional |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {item.tier} | {item.valid_decisions}/{item.decisions} | "
            f"{item.accuracy:.3f} | {item.pair_flip_rate:.3f} | "
            f"{str(item.functional).lower()} |"
            for item in report.tier_metrics
        ),
        "",
        report.claim_boundary,
        "",
    ]
    _atomic_write_text(markdown_path, "\n".join(lines))
    return json_path, markdown_path


def load_relationship_p1k_report(path: pathlib.Path) -> RelationshipP1kReport:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P1k report must be a JSON object")
    artifact_id = _require_sha256(raw.pop("artifact_id", None), "P1k report artifact id")
    next_action = _require_text(raw.pop("next_action"), "P1k report next action")
    guards = _require_object(
        raw.pop("experiment_guards"),
        {
            "competitive",
            "diagnostic_feedback_to_consumer",
            "diagnostic_feedback_to_dataset",
            "diagnostic_feedback_to_qualification_gate",
            "evaluation_feedback_to_pe_credit_reward_or_steering",
            "formal_hidden_test_opened",
            "p2_enabled",
        },
        field_name="P1k report guards",
    )
    if any(_require_bool(value, f"P1k guard {name}") for name, value in guards.items()):
        raise ValueError("P1k report guards must all be false")
    metrics_raw = raw["tier_metrics"]
    if not isinstance(metrics_raw, list):
        raise ValueError("P1k report tier metrics must be a list")
    report = RelationshipP1kReport(
        schema_version=_require_text(raw["schema_version"], "P1k report schema"),
        created_at_iso=raw["created_at_iso"],
        protocol_id=raw["protocol_id"],
        consumer_protocol_id=raw["consumer_protocol_id"],
        training_dataset_fingerprint=raw["training_dataset_fingerprint"],
        context_surface_sha256=raw["context_surface_sha256"],
        model_id=_require_text(raw["model_id"], "P1k report model"),
        weights_sha256=raw["weights_sha256"],
        readout_ledger_sha256=raw["readout_ledger_sha256"],
        decision_ledger_sha256=raw["decision_ledger_sha256"],
        tier_metrics=tuple(
            RelationshipP1kTierMetric.from_payload(item) for item in metrics_raw
        ),
        output_count=_require_int(raw["output_count"], "P1k report output count"),
        verdict=RelationshipP1kVerdict(
            _require_text(raw["verdict"], "P1k report verdict")
        ),
        claim_boundary=raw["claim_boundary"],
    )
    if report.artifact_id != artifact_id or report.next_action != next_action:
        raise ValueError("P1k report derived values mismatch")
    return report


__all__ = [
    "RELATIONSHIP_P1K_CONTEXT_ARM",
    "RELATIONSHIP_P1K_PREPARED_NEXT_ACTION",
    "RELATIONSHIP_P1K_REQUIRED_VALID_RATE",
    "RELATIONSHIP_P1K_RUNG_MINIMUM_ACCURACY",
    "RELATIONSHIP_P1K_RUNG_MINIMUM_PAIR_FLIP_RATE",
    "RELATIONSHIP_P1K_TIERS",
    "RelationshipP1kCheckpoint",
    "RelationshipP1kDecision",
    "RelationshipP1kExecution",
    "RelationshipP1kOracleTier",
    "RelationshipP1kProgress",
    "RelationshipP1kProtocol",
    "RelationshipP1kReadout",
    "RelationshipP1kReport",
    "RelationshipP1kTierMetric",
    "RelationshipP1kVerdict",
    "assess_relationship_p1k_diagnostic",
    "build_relationship_p1k_checkpoint",
    "build_relationship_p1k_disclosure",
    "execute_relationship_p1k_diagnostic",
    "freeze_relationship_p1k_protocol",
    "load_relationship_p1k_checkpoint",
    "load_relationship_p1k_progress",
    "load_relationship_p1k_protocol",
    "load_relationship_p1k_report",
    "persist_relationship_p1k_decision",
    "persist_relationship_p1k_readout",
    "relationship_p1k_readout_prompt_path",
    "relationship_p1k_record_plan",
    "relationship_p1k_request_template_path",
    "render_relationship_p1k_request",
    "validate_relationship_p1k_progress",
    "validate_relationship_p1k_protocol_lineage",
    "write_relationship_p1k_checkpoint",
    "write_relationship_p1k_protocol",
    "write_relationship_p1k_report",
]
