"""First-attempt P1m development-instrument qualification.

This module freezes two independent consumers before their first answer:

* a strong full-history/RAG Qwen baseline scored by exact A/B next-token
  logits with mirrored-pair label rotation; and
* the owner-persisted structured-state path with a frozen named semantic
  condition reader.

Generator truth is attached only after each readout is durable.  Qualification
outputs never enter memory, PE, credit, reward, controller state, or steering.
Passing admits an instrument; it is not evidence for any four-able axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Callable

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME,
    P2DevelopmentEpisode,
    P2DevelopmentHistorySession,
    P2DevelopmentProbeSession,
    RelationshipAction,
    RelationshipObservation,
    RelationshipTransferDataset,
    sha256_json,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
)


RELATIONSHIP_P1M_QUALIFICATION_PLAN_SCHEMA_VERSION = (
    "relationship-p1m-qualification-plan.v1"
)
RELATIONSHIP_P1M_QUALIFICATION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-p1m-qualification-protocol.v1"
)
RELATIONSHIP_P1M_QWEN_READOUT_SCHEMA_VERSION = (
    "relationship-p1m-qwen-logit-readout.v1"
)
RELATIONSHIP_P1M_STRUCTURED_READOUT_SCHEMA_VERSION = (
    "relationship-p1m-structured-readout.v1"
)
RELATIONSHIP_P1M_QUALIFICATION_DECISION_SCHEMA_VERSION = (
    "relationship-p1m-qualification-decision.v1"
)
RELATIONSHIP_P1M_QUALIFICATION_REPORT_SCHEMA_VERSION = (
    "relationship-p1m-qualification-report.v1"
)
RELATIONSHIP_P1M_QUALIFICATION_NEXT_ACTION = (
    "execute_first_and_only_p1m_development_qualification"
)

RELATIONSHIP_P1M_FORCED_CHOICE_PROMPT_ASSET = (
    "relationship_p1m_forced_choice_v1.txt"
)
RELATIONSHIP_P1M_FORCED_CHOICE_REQUEST_ASSET = (
    "relationship_p1m_forced_choice_request_v1.txt"
)

RELATIONSHIP_P1M_REQUIRED_PAIRS = 24
RELATIONSHIP_P1M_REQUIRED_DECISIONS_PER_ARM = 48
RELATIONSHIP_P1M_ACCURACY_MINIMUM = 0.625
RELATIONSHIP_P1M_ACCURACY_MAXIMUM = 0.875
RELATIONSHIP_P1M_ACCURACY_WILSON_LOWER_MINIMUM = 0.50
RELATIONSHIP_P1M_PAIR_FLIP_WILSON_LOWER_EXCLUSIVE = 0.35
RELATIONSHIP_P1M_WILSON_ONE_SIDED_CONFIDENCE = 0.95
RELATIONSHIP_P1M_WILSON_Z = 1.6448536269514722
RELATIONSHIP_P1M_RAG_TOP_K = 4

_HEX_DIGITS = frozenset("0123456789abcdef")
_HISTORY_MARKER = "{{PUBLIC_HISTORY_EVIDENCE}}"
_CURRENT_MARKER = "{{CURRENT_USER_MESSAGE}}"
_CANDIDATE_A_MARKER = "{{CANDIDATE_A}}"
_CANDIDATE_B_MARKER = "{{CANDIDATE_B}}"
_NON_NOOP_ACTIONS = (
    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
)
_ACTION_SURFACES = {
    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: (
        "stay_present_without_probe（不追问、不催促，也不急着解决，"
        "只是明确表示仍会在这里陪着）"
    ),
    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: (
        "respect_space_with_return_option（停止施加回应压力，把节奏和决定权"
        "还给对方，并说明愿意时可以回来）"
    ),
}
_CONDITION_LABELS = {
    "condition_agency_displacement_p1m": "agency_displacement",
    "condition_belonging_erasure_p1m": "belonging_erasure",
}
_PROTOCOL_CLAIM_BOUNDARY = (
    "P1m freezes one first-attempt development qualification over the sealed "
    "24-pair generated package before any Qwen or structured-state answer. "
    "Prompt and RAG use exact A/B next-token logits with pairwise label "
    "rotation; structured-state uses owner persistence and a frozen named BGE "
    "reader. Evaluation cannot revise the dataset, consumers, thresholds, PE, "
    "credit, reward, controller, or steering."
)
_REPORT_CLAIM_BOUNDARY = (
    "This report only decides whether the generated P1m development instrument "
    "has an informative strong-baseline range and a functioning structured-state "
    "path. Passing is not formal held-out evidence, Volvence advantage, or proof "
    "of Appendable, Readable, Learnable, Steerable, production ACTIVE, safety, "
    "or product value. Failure permanently closes scenario versioning for this "
    "frozen P1m recipe."
)


class RelationshipP1mQualificationArm(str, Enum):
    PROMPT_STEELMAN = "prompt-steelman-forced-choice"
    RAG_STEELMAN = "rag-steelman-observational"
    STRUCTURED_STATE = "structured-state-named-reader"


class RelationshipP1mQualificationVerdict(str, Enum):
    QUALIFIED = "instrument_qualified_for_causal_four_able_experiments"
    MACHINERY_INVALID = "qualification_machinery_invalid"
    BASELINE_TOO_WEAK = "prompt_steelman_baseline_too_weak"
    BASELINE_SATURATED = "prompt_steelman_baseline_saturated"
    BASELINE_FLIP_INSUFFICIENT = "prompt_steelman_pair_flip_insufficient"
    STRUCTURED_FLIP_INSUFFICIENT = "structured_state_pair_flip_insufficient"


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "prompts"


def relationship_p1m_forced_choice_prompt_path() -> pathlib.Path:
    return _asset_dir() / RELATIONSHIP_P1M_FORCED_CHOICE_PROMPT_ASSET


def relationship_p1m_forced_choice_request_path() -> pathlib.Path:
    return _asset_dir() / RELATIONSHIP_P1M_FORCED_CHOICE_REQUEST_ASSET


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_snapshot_manifest_sha256(snapshot: pathlib.Path) -> str:
    """Hash one frozen model snapshot with platform-neutral relative paths."""

    root = pathlib.Path(snapshot)
    manifest = tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            _sha256_file(path),
        )
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    )
    if not manifest:
        raise FileNotFoundError(f"empty frozen snapshot: {root}")
    return sha256_json(manifest)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if len(text) != 64 or any(character not in _HEX_DIGITS for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{field_name} fields do not match; missing={missing}, extra={extra}"
        )
    return value


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


def _history_text(
    observation: RelationshipObservation,
    ordered_event_ids: tuple[str, ...],
) -> str:
    histories = {item.event_id: item for item in observation.histories}
    if set(ordered_event_ids) != set(histories) or len(ordered_event_ids) != 4:
        raise ValueError("P1m history order must cover exactly four public events")
    blocks: list[str] = []
    for index, event_id in enumerate(ordered_event_ids, start=1):
        event = histories[event_id]
        blocks.append(
            "\n".join(
                (
                    f"历史 {index}",
                    f"user_event: {event.user_utterance}",
                    f"assistant_action: {event.assistant_action.value}",
                    f"typed_external_outcome: {event.typed_outcome.value}",
                    f"user_reaction: {event.user_reaction}",
                )
            )
        )
    return "\n\n".join(blocks)


def render_relationship_p1m_forced_choice_request(
    *,
    observation: RelationshipObservation,
    ordered_event_ids: tuple[str, ...],
    candidate_a: RelationshipAction,
    candidate_b: RelationshipAction,
) -> str:
    if {candidate_a, candidate_b} != set(_NON_NOOP_ACTIONS):
        raise ValueError("P1m A/B mapping must cover the two non-noop actions")
    template = relationship_p1m_forced_choice_request_path().read_text(
        encoding="utf-8"
    )
    replacements = {
        _HISTORY_MARKER: _history_text(observation, ordered_event_ids),
        _CURRENT_MARKER: observation.current_input,
        _CANDIDATE_A_MARKER: _ACTION_SURFACES[candidate_a],
        _CANDIDATE_B_MARKER: _ACTION_SURFACES[candidate_b],
    }
    for marker in replacements:
        if template.count(marker) != 1:
            raise ValueError("P1m forced-choice template marker drift")
    rendered = template
    for marker, replacement in replacements.items():
        rendered = rendered.replace(marker, replacement)
    return rendered.strip()


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("P1m RAG embedding width mismatch")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("P1m RAG embedding norm must be positive")
    return math.fsum(
        left_item * right_item
        for left_item, right_item in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


@dataclass(frozen=True)
class RelationshipP1mQwenPlanRecord:
    record_index: int
    arm: RelationshipP1mQualificationArm
    scene_id: str
    mirror_pair_id: str
    candidate_a: RelationshipAction
    candidate_b: RelationshipAction
    ordered_history_event_ids: tuple[str, ...]
    request_text: str
    request_sha256: str
    model_input_sha256: str
    prompt_tokens: int

    def __post_init__(self) -> None:
        if self.record_index < 0:
            raise ValueError("P1m Qwen plan index must be non-negative")
        if self.arm not in {
            RelationshipP1mQualificationArm.PROMPT_STEELMAN,
            RelationshipP1mQualificationArm.RAG_STEELMAN,
        }:
            raise ValueError("P1m Qwen plan arm is unsupported")
        _require_text(self.scene_id, "P1m Qwen plan scene")
        _require_text(self.mirror_pair_id, "P1m Qwen plan pair")
        if {self.candidate_a, self.candidate_b} != set(_NON_NOOP_ACTIONS):
            raise ValueError("P1m Qwen plan candidate mapping drift")
        if len(self.ordered_history_event_ids) != 4 or len(
            set(self.ordered_history_event_ids)
        ) != 4:
            raise ValueError("P1m Qwen plan requires four unique history events")
        if _sha256_text(self.request_text) != self.request_sha256:
            raise ValueError("P1m Qwen plan request hash mismatch")
        _require_sha256(self.model_input_sha256, "P1m model input hash")
        if self.prompt_tokens <= 0:
            raise ValueError("P1m prompt token count must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "arm": self.arm.value,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "candidate_a": self.candidate_a.value,
            "candidate_b": self.candidate_b.value,
            "ordered_history_event_ids": list(self.ordered_history_event_ids),
            "request_text": self.request_text,
            "request_sha256": self.request_sha256,
            "model_input_sha256": self.model_input_sha256,
            "prompt_tokens": self.prompt_tokens,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1mQwenPlanRecord":
        raw = _require_exact_keys(
            value,
            {
                "record_index",
                "arm",
                "scene_id",
                "mirror_pair_id",
                "candidate_a",
                "candidate_b",
                "ordered_history_event_ids",
                "request_text",
                "request_sha256",
                "model_input_sha256",
                "prompt_tokens",
            },
            field_name="P1m Qwen plan record",
        )
        history_ids = raw["ordered_history_event_ids"]
        if not isinstance(history_ids, list):
            raise ValueError("P1m Qwen history ids must be an array")
        return cls(
            record_index=_require_int(raw["record_index"], "record index"),
            arm=RelationshipP1mQualificationArm(
                _require_text(raw["arm"], "P1m plan arm")
            ),
            scene_id=_require_text(raw["scene_id"], "P1m plan scene"),
            mirror_pair_id=_require_text(
                raw["mirror_pair_id"], "P1m plan pair"
            ),
            candidate_a=RelationshipAction(
                _require_text(raw["candidate_a"], "P1m candidate A")
            ),
            candidate_b=RelationshipAction(
                _require_text(raw["candidate_b"], "P1m candidate B")
            ),
            ordered_history_event_ids=tuple(
                _require_text(item, "P1m history event id") for item in history_ids
            ),
            request_text=_require_text(raw["request_text"], "P1m request"),
            request_sha256=_require_sha256(
                raw["request_sha256"], "P1m request hash"
            ),
            model_input_sha256=_require_sha256(
                raw["model_input_sha256"], "P1m model input hash"
            ),
            prompt_tokens=_require_int(raw["prompt_tokens"], "P1m prompt tokens"),
        )


@dataclass(frozen=True)
class RelationshipP1mStructuredPlanRecord:
    record_index: int
    scene_id: str
    mirror_pair_id: str
    public_episode_sha256: str

    def __post_init__(self) -> None:
        if self.record_index < 0:
            raise ValueError("P1m structured plan index must be non-negative")
        _require_text(self.scene_id, "P1m structured scene")
        _require_text(self.mirror_pair_id, "P1m structured pair")
        _require_sha256(self.public_episode_sha256, "P1m public episode hash")

    def to_payload(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "public_episode_sha256": self.public_episode_sha256,
        }

    @classmethod
    def from_payload(
        cls, value: object
    ) -> "RelationshipP1mStructuredPlanRecord":
        raw = _require_exact_keys(
            value,
            {
                "record_index",
                "scene_id",
                "mirror_pair_id",
                "public_episode_sha256",
            },
            field_name="P1m structured plan record",
        )
        return cls(
            record_index=_require_int(raw["record_index"], "record index"),
            scene_id=_require_text(raw["scene_id"], "structured scene"),
            mirror_pair_id=_require_text(raw["mirror_pair_id"], "structured pair"),
            public_episode_sha256=_require_sha256(
                raw["public_episode_sha256"], "public episode hash"
            ),
        )


@dataclass(frozen=True)
class RelationshipP1mQualificationPlan:
    dataset_fingerprint: str
    qwen_records: tuple[RelationshipP1mQwenPlanRecord, ...]
    structured_records: tuple[RelationshipP1mStructuredPlanRecord, ...]
    schema_version: str = RELATIONSHIP_P1M_QUALIFICATION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_QUALIFICATION_PLAN_SCHEMA_VERSION:
            raise ValueError("P1m qualification plan schema mismatch")
        _require_sha256(self.dataset_fingerprint, "P1m dataset fingerprint")
        if len(self.qwen_records) != 96:
            raise ValueError("P1m plan requires 96 Qwen records")
        if len(self.structured_records) != 48:
            raise ValueError("P1m plan requires 48 structured records")
        if tuple(item.record_index for item in self.qwen_records) != tuple(range(96)):
            raise ValueError("P1m Qwen plan indices are not contiguous")
        if tuple(item.record_index for item in self.structured_records) != tuple(
            range(48)
        ):
            raise ValueError("P1m structured plan indices are not contiguous")
        identities = tuple(
            (item.arm, item.scene_id, item.mirror_pair_id)
            for item in self.qwen_records
        )
        if len(set(identities)) != len(identities):
            raise ValueError("P1m Qwen plan identities are not unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "qwen_records": [item.to_payload() for item in self.qwen_records],
            "structured_records": [
                item.to_payload() for item in self.structured_records
            ],
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def write_relationship_p1m_qualification_plan(
    plan: RelationshipP1mQualificationPlan,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "qualification_plan.json"
    _atomic_write_text(
        path,
        json.dumps(
            {**plan.to_payload(), "artifact_id": plan.artifact_id},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


def load_relationship_p1m_qualification_plan(
    path: pathlib.Path,
) -> RelationshipP1mQualificationPlan:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "dataset_fingerprint",
            "qwen_records",
            "structured_records",
            "artifact_id",
        },
        field_name="P1m qualification plan",
    )
    qwen_raw = root["qwen_records"]
    structured_raw = root["structured_records"]
    if not isinstance(qwen_raw, list) or not isinstance(structured_raw, list):
        raise ValueError("P1m qualification plan records must be arrays")
    plan = RelationshipP1mQualificationPlan(
        dataset_fingerprint=_require_sha256(
            root["dataset_fingerprint"], "dataset fingerprint"
        ),
        qwen_records=tuple(
            RelationshipP1mQwenPlanRecord.from_payload(item) for item in qwen_raw
        ),
        structured_records=tuple(
            RelationshipP1mStructuredPlanRecord.from_payload(item)
            for item in structured_raw
        ),
        schema_version=_require_text(root["schema_version"], "plan schema"),
    )
    if plan.artifact_id != _require_sha256(root["artifact_id"], "plan id"):
        raise ValueError("P1m qualification plan artifact id mismatch")
    return plan


def build_relationship_p1m_reader_artifact(
    dataset: RelationshipTransferDataset,
    *,
    embedding_model_id: str,
    embedding_weights_sha256: str,
) -> RelationshipConditionReaderArtifact:
    conditions = {item.condition_id: item for item in dataset.abstract_conditions}
    if set(conditions) != set(_CONDITION_LABELS):
        raise ValueError("P1m reader requires the two frozen abstract conditions")
    return RelationshipConditionReaderArtifact(
        embedding_model_id=embedding_model_id,
        embedding_weights_sha256=embedding_weights_sha256,
        prototypes=tuple(
            RelationshipConditionPrototype(
                label=_CONDITION_LABELS[condition_id],
                summary=conditions[condition_id].hidden_summary,
            )
            for condition_id in sorted(conditions)
        ),
    )


def relationship_p1m_public_episode(
    observation: RelationshipObservation,
) -> P2DevelopmentEpisode:
    histories = tuple(
        P2DevelopmentHistorySession(
            episode_id=observation.scene_id,
            session_id=f"{observation.scene_id}:session:{index}",
            session_index=index,
            event_id=event.event_id,
            observation_summary=event.user_utterance,
            action_id=event.assistant_action.value,
            observed_outcome_id=event.typed_outcome.value,
            reaction_summary=event.user_reaction,
            observation_ref=f"public:{_sha256_text(event.user_utterance)}",
        )
        for index, event in enumerate(observation.histories)
    )
    return P2DevelopmentEpisode(
        episode_id=observation.scene_id,
        history_sessions=histories,
        probe_session=P2DevelopmentProbeSession(
            episode_id=observation.scene_id,
            session_id=f"{observation.scene_id}:session:4",
            session_index=4,
            decision_id=f"{observation.scene_id}:decision",
            current_observation=observation.current_input,
            observation_ref=f"public:{_sha256_text(observation.current_input)}",
            candidate_action_ids=tuple(item.value for item in RELATIONSHIP_ACTIONS),
            outcome_ids=tuple(item.value for item in RELATIONSHIP_OUTCOMES),
        ),
    )


def _episode_sha256(episode: P2DevelopmentEpisode) -> str:
    return sha256_json(episode.to_sut_sequence())


def build_relationship_p1m_qualification_plan(
    dataset: RelationshipTransferDataset,
    *,
    render_model_input: Callable[[str], tuple[str, int]],
    embed: Callable[[str], tuple[float, ...]],
) -> RelationshipP1mQualificationPlan:
    if dataset.package_name != RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME:
        raise ValueError("P1m qualification requires the generated P1m package")
    pairs = dataset.mirrored_pairs()
    if len(pairs) != RELATIONSHIP_P1M_REQUIRED_PAIRS:
        raise ValueError("P1m qualification requires 24 mirrored pairs")
    qwen_records: list[RelationshipP1mQwenPlanRecord] = []
    structured_records: list[RelationshipP1mStructuredPlanRecord] = []
    for pair_index, (mirror_pair_id, members) in enumerate(pairs):
        if len(members) != 2:
            raise ValueError("P1m mirrored pair must contain two scenes")
        candidate_a, candidate_b = (
            _NON_NOOP_ACTIONS
            if pair_index % 2 == 0
            else tuple(reversed(_NON_NOOP_ACTIONS))
        )
        for observation, _dynamic in sorted(
            members,
            key=lambda item: item[0].scene_id,
        ):
            natural_order = tuple(item.event_id for item in observation.histories)
            query_vector = embed(observation.current_input)
            scored = tuple(
                (
                    event.event_id,
                    _cosine(
                        query_vector,
                        embed(f"{event.user_utterance}\n{event.user_reaction}"),
                    ),
                )
                for event in observation.histories
            )
            rag_order = tuple(
                event_id
                for event_id, _score in sorted(
                    scored,
                    key=lambda item: (-item[1], item[0]),
                )[:RELATIONSHIP_P1M_RAG_TOP_K]
            )
            for arm, order in (
                (RelationshipP1mQualificationArm.PROMPT_STEELMAN, natural_order),
                (RelationshipP1mQualificationArm.RAG_STEELMAN, rag_order),
            ):
                request = render_relationship_p1m_forced_choice_request(
                    observation=observation,
                    ordered_event_ids=order,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                )
                model_input, prompt_tokens = render_model_input(request)
                qwen_records.append(
                    RelationshipP1mQwenPlanRecord(
                        record_index=len(qwen_records),
                        arm=arm,
                        scene_id=observation.scene_id,
                        mirror_pair_id=mirror_pair_id,
                        candidate_a=candidate_a,
                        candidate_b=candidate_b,
                        ordered_history_event_ids=order,
                        request_text=request,
                        request_sha256=_sha256_text(request),
                        model_input_sha256=_sha256_text(model_input),
                        prompt_tokens=prompt_tokens,
                    )
                )
            episode = relationship_p1m_public_episode(observation)
            structured_records.append(
                RelationshipP1mStructuredPlanRecord(
                    record_index=len(structured_records),
                    scene_id=observation.scene_id,
                    mirror_pair_id=mirror_pair_id,
                    public_episode_sha256=_episode_sha256(episode),
                )
            )
    return RelationshipP1mQualificationPlan(
        dataset_fingerprint=dataset.dataset_fingerprint,
        qwen_records=tuple(qwen_records),
        structured_records=tuple(structured_records),
    )


@dataclass(frozen=True)
class RelationshipP1mQualificationProtocol:
    frozen_at_iso: str
    source_p1k_report_artifact_id: str
    source_generation_attestation_id: str
    source_generation_protocol_id: str
    source_transport_id: str
    source_seed_inventory_sha256: str
    package_name: str
    dataset_fingerprint: str
    pair_count: int
    scene_count: int
    qwen_model_source: str
    qwen_model_revision: str
    qwen_model_id: str
    qwen_weights_sha256: str
    qwen_snapshot_sha256: str
    qwen_device: str
    qwen_torch_dtype: str
    prompt_sha256: str
    request_template_sha256: str
    token_a_id: int
    token_b_id: int
    scoring_method: str
    qwen_config_sha256: str
    bge_model_source: str
    bge_model_revision: str
    bge_weights_sha256: str
    reader_artifact: RelationshipConditionReaderArtifact
    rag_top_k: int
    plan_artifact_id: str
    planned_qwen_readouts: int
    planned_structured_readouts: int
    qualification_inputs_observed_before_freeze: int
    qwen_outputs_observed_before_freeze: int
    structured_outputs_observed_before_freeze: int
    first_qualification_attempt_only: bool
    evaluation_feedback_allowed: bool
    claim_boundary: str = _PROTOCOL_CLAIM_BOUNDARY
    next_action: str = RELATIONSHIP_P1M_QUALIFICATION_NEXT_ACTION
    schema_version: str = RELATIONSHIP_P1M_QUALIFICATION_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_QUALIFICATION_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1m qualification protocol schema mismatch")
        _require_timestamp(self.frozen_at_iso, "P1m qualification frozen_at")
        for field_name, value in (
            ("source_p1k_report_artifact_id", self.source_p1k_report_artifact_id),
            ("source_generation_attestation_id", self.source_generation_attestation_id),
            ("source_generation_protocol_id", self.source_generation_protocol_id),
            ("source_transport_id", self.source_transport_id),
            ("source_seed_inventory_sha256", self.source_seed_inventory_sha256),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("qwen_weights_sha256", self.qwen_weights_sha256),
            ("qwen_snapshot_sha256", self.qwen_snapshot_sha256),
            ("prompt_sha256", self.prompt_sha256),
            ("request_template_sha256", self.request_template_sha256),
            ("qwen_config_sha256", self.qwen_config_sha256),
            ("bge_weights_sha256", self.bge_weights_sha256),
            ("plan_artifact_id", self.plan_artifact_id),
        ):
            _require_sha256(value, field_name)
        if self.package_name != RELATIONSHIP_TRANSFER_P1M_V1_PACKAGE_NAME:
            raise ValueError("P1m qualification package mismatch")
        if self.pair_count != 24 or self.scene_count != 48:
            raise ValueError("P1m qualification package size mismatch")
        if self.token_a_id < 0 or self.token_b_id < 0 or self.token_a_id == self.token_b_id:
            raise ValueError("P1m A/B token ids are invalid")
        if self.scoring_method != "exact_first_assistant_token_logits_A_vs_B":
            raise ValueError("P1m Qwen scoring method drift")
        if self.rag_top_k != 4:
            raise ValueError("P1m RAG top-k drift")
        if self.reader_artifact.embedding_weights_sha256 != self.bge_weights_sha256:
            raise ValueError("P1m reader/BGE weights mismatch")
        if self.planned_qwen_readouts != 96 or self.planned_structured_readouts != 48:
            raise ValueError("P1m qualification output plan drift")
        if any(
            value != 0
            for value in (
                self.qualification_inputs_observed_before_freeze,
                self.qwen_outputs_observed_before_freeze,
                self.structured_outputs_observed_before_freeze,
            )
        ):
            raise ValueError("P1m qualification must freeze before any answer")
        if not self.first_qualification_attempt_only or self.evaluation_feedback_allowed:
            raise ValueError("P1m first-attempt/evaluation firewall drift")
        if self.claim_boundary != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1m qualification claim boundary drift")
        if self.next_action != RELATIONSHIP_P1M_QUALIFICATION_NEXT_ACTION:
            raise ValueError("P1m qualification next action drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source": {
                "p1k_report_artifact_id": self.source_p1k_report_artifact_id,
                "generation_attestation_id": self.source_generation_attestation_id,
                "generation_protocol_id": self.source_generation_protocol_id,
                "transport_id": self.source_transport_id,
                "surface_seed_inventory_sha256": self.source_seed_inventory_sha256,
            },
            "dataset": {
                "package_name": self.package_name,
                "dataset_fingerprint": self.dataset_fingerprint,
                "pair_count": self.pair_count,
                "scene_count": self.scene_count,
            },
            "qwen": {
                "model_source": self.qwen_model_source,
                "model_revision": self.qwen_model_revision,
                "model_id": self.qwen_model_id,
                "weights_sha256": self.qwen_weights_sha256,
                "snapshot_sha256": self.qwen_snapshot_sha256,
                "device": self.qwen_device,
                "torch_dtype": self.qwen_torch_dtype,
                "prompt_sha256": self.prompt_sha256,
                "request_template_sha256": self.request_template_sha256,
                "token_a_id": self.token_a_id,
                "token_b_id": self.token_b_id,
                "scoring_method": self.scoring_method,
                "config_sha256": self.qwen_config_sha256,
            },
            "structured_state": {
                "bge_model_source": self.bge_model_source,
                "bge_model_revision": self.bge_model_revision,
                "bge_weights_sha256": self.bge_weights_sha256,
                "reader_artifact": self.reader_artifact.to_payload(),
                "reader_artifact_id": self.reader_artifact.artifact_id,
                "rag_top_k": self.rag_top_k,
            },
            "plan": {
                "artifact_id": self.plan_artifact_id,
                "planned_qwen_readouts": self.planned_qwen_readouts,
                "planned_structured_readouts": self.planned_structured_readouts,
            },
            "qualification_gate": {
                "required_decisions_per_arm": RELATIONSHIP_P1M_REQUIRED_DECISIONS_PER_ARM,
                "accuracy_interval": [
                    RELATIONSHIP_P1M_ACCURACY_MINIMUM,
                    RELATIONSHIP_P1M_ACCURACY_MAXIMUM,
                ],
                "accuracy_wilson_lower_minimum": (
                    RELATIONSHIP_P1M_ACCURACY_WILSON_LOWER_MINIMUM
                ),
                "pair_flip_wilson_lower_exclusive": (
                    RELATIONSHIP_P1M_PAIR_FLIP_WILSON_LOWER_EXCLUSIVE
                ),
                "wilson_one_sided_confidence": (
                    RELATIONSHIP_P1M_WILSON_ONE_SIDED_CONFIDENCE
                ),
                "primary_arm": (
                    RelationshipP1mQualificationArm.PROMPT_STEELMAN.value
                ),
                "structured_pair_flip_required": True,
                "rag_role": "observational_at_four_histories",
            },
            "freeze_guards": {
                "qualification_inputs_observed_before_freeze": (
                    self.qualification_inputs_observed_before_freeze
                ),
                "qwen_outputs_observed_before_freeze": (
                    self.qwen_outputs_observed_before_freeze
                ),
                "structured_outputs_observed_before_freeze": (
                    self.structured_outputs_observed_before_freeze
                ),
                "first_qualification_attempt_only": self.first_qualification_attempt_only,
                "evaluation_feedback_allowed": self.evaluation_feedback_allowed,
            },
            "claim_boundary": self.claim_boundary,
            "next_action": self.next_action,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self.to_payload())


def _reader_artifact_from_payload(value: object) -> RelationshipConditionReaderArtifact:
    raw = _require_exact_keys(
        value,
        {
            "schema_version",
            "embedding_model_id",
            "embedding_weights_sha256",
            "semantic_similarity",
            "softmax_temperature",
            "prototypes",
        },
        field_name="P1m reader artifact",
    )
    prototypes_raw = raw["prototypes"]
    if not isinstance(prototypes_raw, list):
        raise ValueError("P1m reader prototypes must be an array")
    prototypes: list[RelationshipConditionPrototype] = []
    for value_raw in prototypes_raw:
        item = _require_exact_keys(
            value_raw,
            {"label", "summary", "summary_sha256"},
            field_name="P1m reader prototype",
        )
        prototype = RelationshipConditionPrototype(
            label=_require_text(item["label"], "prototype label"),
            summary=_require_text(item["summary"], "prototype summary"),
        )
        if prototype.summary_sha256 != _require_sha256(
            item["summary_sha256"], "prototype summary hash"
        ):
            raise ValueError("P1m reader prototype summary hash mismatch")
        prototypes.append(prototype)
    return RelationshipConditionReaderArtifact(
        embedding_model_id=_require_text(
            raw["embedding_model_id"], "reader embedding model"
        ),
        embedding_weights_sha256=_require_sha256(
            raw["embedding_weights_sha256"], "reader embedding weights"
        ),
        prototypes=tuple(prototypes),
        softmax_temperature=float(raw["softmax_temperature"]),
        semantic_similarity=_require_text(
            raw["semantic_similarity"], "reader similarity"
        ),
        schema_version=_require_text(raw["schema_version"], "reader schema"),
    )


def write_relationship_p1m_qualification_protocol(
    protocol: RelationshipP1mQualificationProtocol,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "qualification_protocol.json"
    _atomic_write_text(
        path,
        json.dumps(
            {**protocol.to_payload(), "protocol_id": protocol.protocol_id},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


def load_relationship_p1m_qualification_protocol(
    path: pathlib.Path,
) -> RelationshipP1mQualificationProtocol:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "frozen_at_iso",
            "source",
            "dataset",
            "qwen",
            "structured_state",
            "plan",
            "qualification_gate",
            "freeze_guards",
            "claim_boundary",
            "next_action",
            "protocol_id",
        },
        field_name="P1m qualification protocol",
    )
    source = root["source"]
    dataset = root["dataset"]
    qwen = root["qwen"]
    structured = root["structured_state"]
    plan = root["plan"]
    guards = root["freeze_guards"]
    if not all(isinstance(item, dict) for item in (source, dataset, qwen, structured, plan, guards)):
        raise ValueError("P1m qualification protocol sections must be objects")
    protocol = RelationshipP1mQualificationProtocol(
        frozen_at_iso=_require_timestamp(root["frozen_at_iso"], "frozen_at"),
        source_p1k_report_artifact_id=_require_sha256(source["p1k_report_artifact_id"], "p1k report"),
        source_generation_attestation_id=_require_sha256(source["generation_attestation_id"], "generation attestation"),
        source_generation_protocol_id=_require_sha256(source["generation_protocol_id"], "generation protocol"),
        source_transport_id=_require_sha256(source["transport_id"], "transport id"),
        source_seed_inventory_sha256=_require_sha256(source["surface_seed_inventory_sha256"], "seed inventory"),
        package_name=_require_text(dataset["package_name"], "package name"),
        dataset_fingerprint=_require_sha256(dataset["dataset_fingerprint"], "dataset fingerprint"),
        pair_count=_require_int(dataset["pair_count"], "pair count"),
        scene_count=_require_int(dataset["scene_count"], "scene count"),
        qwen_model_source=_require_text(qwen["model_source"], "Qwen model source"),
        qwen_model_revision=_require_text(qwen["model_revision"], "Qwen revision"),
        qwen_model_id=_require_text(qwen["model_id"], "Qwen model id"),
        qwen_weights_sha256=_require_sha256(qwen["weights_sha256"], "Qwen weights"),
        qwen_snapshot_sha256=_require_sha256(qwen["snapshot_sha256"], "Qwen snapshot"),
        qwen_device=_require_text(qwen["device"], "Qwen device"),
        qwen_torch_dtype=_require_text(qwen["torch_dtype"], "Qwen dtype"),
        prompt_sha256=_require_sha256(qwen["prompt_sha256"], "prompt hash"),
        request_template_sha256=_require_sha256(qwen["request_template_sha256"], "request template hash"),
        token_a_id=_require_int(qwen["token_a_id"], "token A"),
        token_b_id=_require_int(qwen["token_b_id"], "token B"),
        scoring_method=_require_text(qwen["scoring_method"], "scoring method"),
        qwen_config_sha256=_require_sha256(qwen["config_sha256"], "Qwen config"),
        bge_model_source=_require_text(structured["bge_model_source"], "BGE model source"),
        bge_model_revision=_require_text(structured["bge_model_revision"], "BGE revision"),
        bge_weights_sha256=_require_sha256(structured["bge_weights_sha256"], "BGE weights"),
        reader_artifact=_reader_artifact_from_payload(structured["reader_artifact"]),
        rag_top_k=_require_int(structured["rag_top_k"], "RAG top-k"),
        plan_artifact_id=_require_sha256(plan["artifact_id"], "plan id"),
        planned_qwen_readouts=_require_int(plan["planned_qwen_readouts"], "planned Qwen"),
        planned_structured_readouts=_require_int(plan["planned_structured_readouts"], "planned structured"),
        qualification_inputs_observed_before_freeze=_require_int(
            guards["qualification_inputs_observed_before_freeze"],
            "prior inputs",
        ),
        qwen_outputs_observed_before_freeze=_require_int(
            guards["qwen_outputs_observed_before_freeze"],
            "prior Qwen outputs",
        ),
        structured_outputs_observed_before_freeze=_require_int(
            guards["structured_outputs_observed_before_freeze"],
            "prior structured outputs",
        ),
        first_qualification_attempt_only=_require_bool(guards["first_qualification_attempt_only"], "first attempt"),
        evaluation_feedback_allowed=_require_bool(guards["evaluation_feedback_allowed"], "evaluation feedback"),
        claim_boundary=_require_text(root["claim_boundary"], "claim boundary"),
        next_action=_require_text(root["next_action"], "next action"),
        schema_version=_require_text(root["schema_version"], "schema version"),
    )
    if protocol.reader_artifact.artifact_id != _require_sha256(
        structured["reader_artifact_id"], "reader artifact id"
    ):
        raise ValueError("P1m reader artifact id mismatch")
    if protocol.protocol_id != _require_sha256(root["protocol_id"], "protocol id"):
        raise ValueError("P1m qualification protocol id mismatch")
    expected_gate = protocol.to_payload()["qualification_gate"]
    if root["qualification_gate"] != expected_gate:
        raise ValueError("P1m qualification gate drift")
    return protocol


@dataclass(frozen=True)
class RelationshipP1mQwenReadout:
    protocol_id: str
    record_index: int
    arm: RelationshipP1mQualificationArm
    scene_id: str
    model_input_sha256: str
    logit_a: float
    logit_b: float
    chosen_label: str | None
    chosen_action_id: RelationshipAction | None
    prompt_tokens: int
    schema_version: str = RELATIONSHIP_P1M_QWEN_READOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_QWEN_READOUT_SCHEMA_VERSION:
            raise ValueError("P1m Qwen readout schema mismatch")
        _require_sha256(self.protocol_id, "P1m readout protocol")
        _require_sha256(self.model_input_sha256, "P1m readout model input")
        if self.record_index < 0 or self.prompt_tokens <= 0:
            raise ValueError("P1m Qwen readout index/token count invalid")
        if self.arm is RelationshipP1mQualificationArm.STRUCTURED_STATE:
            raise ValueError("P1m Qwen readout cannot use structured arm")
        if not math.isfinite(self.logit_a) or not math.isfinite(self.logit_b):
            raise ValueError("P1m Qwen logits must be finite")
        expected_label = None if self.logit_a == self.logit_b else ("A" if self.logit_a > self.logit_b else "B")
        if self.chosen_label != expected_label:
            raise ValueError("P1m chosen label does not match logits")
        if (self.chosen_label is None) != (self.chosen_action_id is None):
            raise ValueError("P1m chosen label/action validity mismatch")

    @property
    def valid(self) -> bool:
        return self.chosen_action_id is not None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "record_index": self.record_index,
            "arm": self.arm.value,
            "scene_id": self.scene_id,
            "model_input_sha256": self.model_input_sha256,
            "logit_a": self.logit_a,
            "logit_b": self.logit_b,
            "chosen_label": self.chosen_label,
            "chosen_action_id": None if self.chosen_action_id is None else self.chosen_action_id.value,
            "prompt_tokens": self.prompt_tokens,
            "valid": self.valid,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1mQwenReadout":
        raw = _require_exact_keys(
            value,
            {
                "schema_version",
                "protocol_id",
                "record_index",
                "arm",
                "scene_id",
                "model_input_sha256",
                "logit_a",
                "logit_b",
                "chosen_label",
                "chosen_action_id",
                "prompt_tokens",
                "valid",
                "artifact_id",
            },
            field_name="P1m Qwen readout",
        )
        chosen_action_raw = raw["chosen_action_id"]
        readout = cls(
            protocol_id=_require_sha256(raw["protocol_id"], "readout protocol"),
            record_index=_require_int(raw["record_index"], "readout index"),
            arm=RelationshipP1mQualificationArm(
                _require_text(raw["arm"], "readout arm")
            ),
            scene_id=_require_text(raw["scene_id"], "readout scene"),
            model_input_sha256=_require_sha256(
                raw["model_input_sha256"], "model input hash"
            ),
            logit_a=float(raw["logit_a"]),
            logit_b=float(raw["logit_b"]),
            chosen_label=(
                None
                if raw["chosen_label"] is None
                else _require_text(raw["chosen_label"], "chosen label")
            ),
            chosen_action_id=(
                None
                if chosen_action_raw is None
                else RelationshipAction(
                    _require_text(chosen_action_raw, "chosen action")
                )
            ),
            prompt_tokens=_require_int(raw["prompt_tokens"], "prompt tokens"),
            schema_version=_require_text(raw["schema_version"], "readout schema"),
        )
        if readout.valid != _require_bool(raw["valid"], "readout valid"):
            raise ValueError("P1m Qwen readout validity drift")
        if readout.artifact_id != _require_sha256(
            raw["artifact_id"], "readout artifact id"
        ):
            raise ValueError("P1m Qwen readout artifact id mismatch")
        return readout


@dataclass(frozen=True)
class RelationshipP1mStructuredReadout:
    protocol_id: str
    record_index: int
    scene_id: str
    recommended_action_id: RelationshipAction
    forecast_id: str
    confidence: float
    condition_label: str
    condition_confidence: float
    condition_margin: float
    condition_candidate_scores: tuple[tuple[str, float], ...]
    reader_artifact_id: str
    source_observation_sha256: str
    persistence_payload_sha256: tuple[str, ...]
    persisted_record_count: int
    persisted_action_outcome_count: int
    raw_history_replayed_at_probe: bool
    schema_version: str = RELATIONSHIP_P1M_STRUCTURED_READOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_STRUCTURED_READOUT_SCHEMA_VERSION:
            raise ValueError("P1m structured readout schema mismatch")
        for field_name, value in (
            ("protocol_id", self.protocol_id),
            ("reader_artifact_id", self.reader_artifact_id),
            ("source_observation_sha256", self.source_observation_sha256),
            *(
                ("persistence_payload_sha256", item)
                for item in self.persistence_payload_sha256
            ),
        ):
            _require_sha256(value, field_name)
        if self.record_index < 0 or len(self.persistence_payload_sha256) != 4:
            raise ValueError("P1m structured readout persistence lineage invalid")
        if self.raw_history_replayed_at_probe:
            raise ValueError("P1m structured probe cannot replay raw history")
        if self.persisted_record_count != 4 or self.persisted_action_outcome_count != 4:
            raise ValueError("P1m structured owner did not restore four histories")
        for value in (self.confidence, self.condition_confidence, self.condition_margin):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("P1m structured confidence/margin invalid")

    @property
    def valid(self) -> bool:
        return self.recommended_action_id in _NON_NOOP_ACTIONS

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "record_index": self.record_index,
            "scene_id": self.scene_id,
            "recommended_action_id": self.recommended_action_id.value,
            "forecast_id": self.forecast_id,
            "confidence": self.confidence,
            "condition_readout": {
                "condition_label": self.condition_label,
                "confidence": self.condition_confidence,
                "normalized_margin": self.condition_margin,
                "candidate_scores": [
                    {"label": label, "score": score}
                    for label, score in self.condition_candidate_scores
                ],
                "reader_artifact_id": self.reader_artifact_id,
                "source_observation_sha256": self.source_observation_sha256,
            },
            "persistence_payload_sha256": list(self.persistence_payload_sha256),
            "persisted_record_count": self.persisted_record_count,
            "persisted_action_outcome_count": self.persisted_action_outcome_count,
            "raw_history_replayed_at_probe": self.raw_history_replayed_at_probe,
            "valid": self.valid,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1mStructuredReadout":
        raw = _require_exact_keys(
            value,
            {
                "schema_version",
                "protocol_id",
                "record_index",
                "scene_id",
                "recommended_action_id",
                "forecast_id",
                "confidence",
                "condition_readout",
                "persistence_payload_sha256",
                "persisted_record_count",
                "persisted_action_outcome_count",
                "raw_history_replayed_at_probe",
                "valid",
                "artifact_id",
            },
            field_name="P1m structured readout",
        )
        condition = _require_exact_keys(
            raw["condition_readout"],
            {
                "condition_label",
                "confidence",
                "normalized_margin",
                "candidate_scores",
                "reader_artifact_id",
                "source_observation_sha256",
            },
            field_name="P1m structured condition readout",
        )
        scores_raw = condition["candidate_scores"]
        persistence_raw = raw["persistence_payload_sha256"]
        if not isinstance(scores_raw, list) or not isinstance(persistence_raw, list):
            raise ValueError("P1m structured arrays have invalid shape")
        scores: list[tuple[str, float]] = []
        for value_raw in scores_raw:
            score = _require_exact_keys(
                value_raw,
                {"label", "score"},
                field_name="P1m structured candidate score",
            )
            scores.append(
                (
                    _require_text(score["label"], "condition score label"),
                    float(score["score"]),
                )
            )
        readout = cls(
            protocol_id=_require_sha256(raw["protocol_id"], "structured protocol"),
            record_index=_require_int(raw["record_index"], "structured index"),
            scene_id=_require_text(raw["scene_id"], "structured scene"),
            recommended_action_id=RelationshipAction(
                _require_text(raw["recommended_action_id"], "recommended action")
            ),
            forecast_id=_require_text(raw["forecast_id"], "forecast id"),
            confidence=float(raw["confidence"]),
            condition_label=_require_text(
                condition["condition_label"], "condition label"
            ),
            condition_confidence=float(condition["confidence"]),
            condition_margin=float(condition["normalized_margin"]),
            condition_candidate_scores=tuple(scores),
            reader_artifact_id=_require_sha256(
                condition["reader_artifact_id"], "reader artifact id"
            ),
            source_observation_sha256=_require_sha256(
                condition["source_observation_sha256"], "source observation"
            ),
            persistence_payload_sha256=tuple(
                _require_sha256(item, "persistence payload")
                for item in persistence_raw
            ),
            persisted_record_count=_require_int(
                raw["persisted_record_count"], "persisted records"
            ),
            persisted_action_outcome_count=_require_int(
                raw["persisted_action_outcome_count"], "persisted outcomes"
            ),
            raw_history_replayed_at_probe=_require_bool(
                raw["raw_history_replayed_at_probe"], "raw history replay"
            ),
            schema_version=_require_text(raw["schema_version"], "structured schema"),
        )
        if readout.valid != _require_bool(raw["valid"], "structured valid"):
            raise ValueError("P1m structured readout validity drift")
        if readout.artifact_id != _require_sha256(
            raw["artifact_id"], "structured artifact id"
        ):
            raise ValueError("P1m structured readout artifact id mismatch")
        return readout


@dataclass(frozen=True)
class RelationshipP1mQualificationDecision:
    protocol_id: str
    arm: RelationshipP1mQualificationArm
    record_index: int
    scene_id: str
    mirror_pair_id: str
    readout_artifact_id: str
    chosen_action_id: RelationshipAction | None
    expected_action_id: RelationshipAction
    schema_version: str = RELATIONSHIP_P1M_QUALIFICATION_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_QUALIFICATION_DECISION_SCHEMA_VERSION:
            raise ValueError("P1m qualification decision schema mismatch")
        _require_sha256(self.protocol_id, "P1m decision protocol")
        _require_sha256(self.readout_artifact_id, "P1m decision readout")
        if self.record_index < 0:
            raise ValueError("P1m decision index invalid")
        if self.expected_action_id not in _NON_NOOP_ACTIONS:
            raise ValueError("P1m expected action must be non-noop")

    @property
    def valid(self) -> bool:
        return self.chosen_action_id is not None

    @property
    def correct(self) -> bool:
        return self.chosen_action_id is self.expected_action_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "arm": self.arm.value,
            "record_index": self.record_index,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "readout_artifact_id": self.readout_artifact_id,
            "chosen_action_id": None if self.chosen_action_id is None else self.chosen_action_id.value,
            "expected_action_id": self.expected_action_id.value,
            "valid": self.valid,
            "correct": self.correct,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1mQualificationDecision":
        raw = _require_exact_keys(
            value,
            {
                "schema_version",
                "protocol_id",
                "arm",
                "record_index",
                "scene_id",
                "mirror_pair_id",
                "readout_artifact_id",
                "chosen_action_id",
                "expected_action_id",
                "valid",
                "correct",
                "artifact_id",
            },
            field_name="P1m qualification decision",
        )
        chosen_raw = raw["chosen_action_id"]
        decision = cls(
            protocol_id=_require_sha256(raw["protocol_id"], "decision protocol"),
            arm=RelationshipP1mQualificationArm(
                _require_text(raw["arm"], "decision arm")
            ),
            record_index=_require_int(raw["record_index"], "decision index"),
            scene_id=_require_text(raw["scene_id"], "decision scene"),
            mirror_pair_id=_require_text(
                raw["mirror_pair_id"], "decision pair"
            ),
            readout_artifact_id=_require_sha256(
                raw["readout_artifact_id"], "decision readout"
            ),
            chosen_action_id=(
                None
                if chosen_raw is None
                else RelationshipAction(_require_text(chosen_raw, "chosen action"))
            ),
            expected_action_id=RelationshipAction(
                _require_text(raw["expected_action_id"], "expected action")
            ),
            schema_version=_require_text(raw["schema_version"], "decision schema"),
        )
        if (
            decision.valid != _require_bool(raw["valid"], "decision valid")
            or decision.correct != _require_bool(raw["correct"], "decision correct")
        ):
            raise ValueError("P1m decision derived fields drift")
        if decision.artifact_id != _require_sha256(
            raw["artifact_id"], "decision artifact id"
        ):
            raise ValueError("P1m decision artifact id mismatch")
        return decision


def wilson_one_sided_lower(successes: int, trials: int) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson successes/trials are invalid")
    proportion = successes / trials
    z_squared = RELATIONSHIP_P1M_WILSON_Z**2
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    spread = RELATIONSHIP_P1M_WILSON_Z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials**2)
    )
    return (centre - spread) / denominator


@dataclass(frozen=True)
class RelationshipP1mArmMetrics:
    arm: RelationshipP1mQualificationArm
    decisions: int
    valid_decisions: int
    correct_decisions: int
    accuracy: float
    accuracy_wilson_lower: float
    mirrored_pairs: int
    pair_flips: int
    pair_flip_rate: float
    pair_flip_wilson_lower: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("decisions", self.decisions),
            ("valid_decisions", self.valid_decisions),
            ("correct_decisions", self.correct_decisions),
            ("mirrored_pairs", self.mirrored_pairs),
            ("pair_flips", self.pair_flips),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"P1m metric {field_name} must be non-negative")
        if self.decisions <= 0 or self.mirrored_pairs <= 0:
            raise ValueError("P1m metrics require decisions and mirrored pairs")
        if not 0 <= self.correct_decisions <= self.valid_decisions <= self.decisions:
            raise ValueError("P1m decision metric counts are inconsistent")
        if not 0 <= self.pair_flips <= self.mirrored_pairs:
            raise ValueError("P1m pair metric counts are inconsistent")
        expected_accuracy = self.correct_decisions / self.decisions
        expected_flip_rate = self.pair_flips / self.mirrored_pairs
        expected_accuracy_lower = wilson_one_sided_lower(
            self.correct_decisions,
            self.decisions,
        )
        expected_flip_lower = wilson_one_sided_lower(
            self.pair_flips,
            self.mirrored_pairs,
        )
        for field_name, observed, expected in (
            ("accuracy", self.accuracy, expected_accuracy),
            ("pair_flip_rate", self.pair_flip_rate, expected_flip_rate),
            (
                "accuracy_wilson_lower",
                self.accuracy_wilson_lower,
                expected_accuracy_lower,
            ),
            (
                "pair_flip_wilson_lower",
                self.pair_flip_wilson_lower,
                expected_flip_lower,
            ),
        ):
            if not math.isfinite(observed) or not math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"P1m metric {field_name} drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "decisions": self.decisions,
            "valid_decisions": self.valid_decisions,
            "correct_decisions": self.correct_decisions,
            "accuracy": self.accuracy,
            "accuracy_wilson_lower": self.accuracy_wilson_lower,
            "mirrored_pairs": self.mirrored_pairs,
            "pair_flips": self.pair_flips,
            "pair_flip_rate": self.pair_flip_rate,
            "pair_flip_wilson_lower": self.pair_flip_wilson_lower,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1mArmMetrics":
        raw = _require_exact_keys(
            value,
            {
                "arm",
                "decisions",
                "valid_decisions",
                "correct_decisions",
                "accuracy",
                "accuracy_wilson_lower",
                "mirrored_pairs",
                "pair_flips",
                "pair_flip_rate",
                "pair_flip_wilson_lower",
            },
            field_name="P1m qualification arm metrics",
        )
        return cls(
            arm=RelationshipP1mQualificationArm(
                _require_text(raw["arm"], "metric arm")
            ),
            decisions=_require_int(raw["decisions"], "metric decisions"),
            valid_decisions=_require_int(
                raw["valid_decisions"], "metric valid decisions"
            ),
            correct_decisions=_require_int(
                raw["correct_decisions"], "metric correct decisions"
            ),
            accuracy=_require_float(raw["accuracy"], "metric accuracy"),
            accuracy_wilson_lower=_require_float(
                raw["accuracy_wilson_lower"], "metric accuracy Wilson lower"
            ),
            mirrored_pairs=_require_int(
                raw["mirrored_pairs"], "metric mirrored pairs"
            ),
            pair_flips=_require_int(raw["pair_flips"], "metric pair flips"),
            pair_flip_rate=_require_float(
                raw["pair_flip_rate"], "metric pair flip rate"
            ),
            pair_flip_wilson_lower=_require_float(
                raw["pair_flip_wilson_lower"],
                "metric pair flip Wilson lower",
            ),
        )


def relationship_p1m_arm_metrics(
    decisions: tuple[RelationshipP1mQualificationDecision, ...],
    *,
    arm: RelationshipP1mQualificationArm,
) -> RelationshipP1mArmMetrics:
    selected = tuple(item for item in decisions if item.arm is arm)
    if len(selected) != RELATIONSHIP_P1M_REQUIRED_DECISIONS_PER_ARM:
        raise ValueError("P1m arm does not contain 48 decisions")
    valid = sum(int(item.valid) for item in selected)
    correct = sum(int(item.correct) for item in selected)
    by_pair: dict[str, list[RelationshipP1mQualificationDecision]] = {}
    for item in selected:
        by_pair.setdefault(item.mirror_pair_id, []).append(item)
    if len(by_pair) != 24 or any(len(items) != 2 for items in by_pair.values()):
        raise ValueError("P1m arm mirrored-pair shape drift")
    pair_flips = sum(
        int(
            items[0].chosen_action_id is not None
            and items[1].chosen_action_id is not None
            and items[0].chosen_action_id is not items[1].chosen_action_id
        )
        for items in by_pair.values()
    )
    return RelationshipP1mArmMetrics(
        arm=arm,
        decisions=len(selected),
        valid_decisions=valid,
        correct_decisions=correct,
        accuracy=correct / len(selected),
        accuracy_wilson_lower=wilson_one_sided_lower(correct, len(selected)),
        mirrored_pairs=len(by_pair),
        pair_flips=pair_flips,
        pair_flip_rate=pair_flips / len(by_pair),
        pair_flip_wilson_lower=wilson_one_sided_lower(
            pair_flips, len(by_pair)
        ),
    )


@dataclass(frozen=True)
class RelationshipP1mQualificationReport:
    created_at_iso: str
    protocol_id: str
    plan_artifact_id: str
    dataset_fingerprint: str
    qwen_readout_ledger_sha256: str
    structured_readout_ledger_sha256: str
    qwen_decision_ledger_sha256: str
    structured_decision_ledger_sha256: str
    arm_metrics: tuple[RelationshipP1mArmMetrics, ...]
    verdict: RelationshipP1mQualificationVerdict
    qualification_passed: bool
    scenario_versioning_closed: bool
    evaluation_feedback_to_system: bool
    claim_boundary: str = _REPORT_CLAIM_BOUNDARY
    schema_version: str = RELATIONSHIP_P1M_QUALIFICATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1M_QUALIFICATION_REPORT_SCHEMA_VERSION:
            raise ValueError("P1m qualification report schema mismatch")
        _require_timestamp(self.created_at_iso, "P1m report created_at")
        for value in (
            self.protocol_id,
            self.plan_artifact_id,
            self.dataset_fingerprint,
            self.qwen_readout_ledger_sha256,
            self.structured_readout_ledger_sha256,
            self.qwen_decision_ledger_sha256,
            self.structured_decision_ledger_sha256,
        ):
            _require_sha256(value, "P1m report lineage")
        expected_arms = tuple(RelationshipP1mQualificationArm)
        if tuple(item.arm for item in self.arm_metrics) != expected_arms:
            raise ValueError("P1m report arm metrics order drift")
        if self.qualification_passed != (
            self.verdict is RelationshipP1mQualificationVerdict.QUALIFIED
        ):
            raise ValueError("P1m report pass/verdict mismatch")
        if self.scenario_versioning_closed == self.qualification_passed:
            raise ValueError("P1m scenario closure must equal qualification failure")
        if self.evaluation_feedback_to_system:
            raise ValueError("P1m evaluation feedback firewall is open")
        if self.claim_boundary != _REPORT_CLAIM_BOUNDARY:
            raise ValueError("P1m report claim boundary drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "protocol_id": self.protocol_id,
            "plan_artifact_id": self.plan_artifact_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "ledger_sha256": {
                "qwen_readouts": self.qwen_readout_ledger_sha256,
                "structured_readouts": self.structured_readout_ledger_sha256,
                "qwen_decisions": self.qwen_decision_ledger_sha256,
                "structured_decisions": self.structured_decision_ledger_sha256,
            },
            "arm_metrics": [item.to_payload() for item in self.arm_metrics],
            "verdict": self.verdict.value,
            "qualification_passed": self.qualification_passed,
            "scenario_versioning_closed": self.scenario_versioning_closed,
            "evaluation_feedback_to_system": self.evaluation_feedback_to_system,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1mQualificationReport":
        raw = _require_exact_keys(
            value,
            {
                "schema_version",
                "created_at_iso",
                "protocol_id",
                "plan_artifact_id",
                "dataset_fingerprint",
                "ledger_sha256",
                "arm_metrics",
                "verdict",
                "qualification_passed",
                "scenario_versioning_closed",
                "evaluation_feedback_to_system",
                "claim_boundary",
                "artifact_id",
            },
            field_name="P1m qualification report",
        )
        ledger = _require_exact_keys(
            raw["ledger_sha256"],
            {
                "qwen_readouts",
                "structured_readouts",
                "qwen_decisions",
                "structured_decisions",
            },
            field_name="P1m qualification report ledgers",
        )
        metrics_raw = raw["arm_metrics"]
        if not isinstance(metrics_raw, list):
            raise ValueError("P1m qualification arm metrics must be an array")
        report = cls(
            created_at_iso=_require_timestamp(
                raw["created_at_iso"], "P1m report created_at"
            ),
            protocol_id=_require_sha256(raw["protocol_id"], "P1m protocol id"),
            plan_artifact_id=_require_sha256(
                raw["plan_artifact_id"], "P1m plan id"
            ),
            dataset_fingerprint=_require_sha256(
                raw["dataset_fingerprint"], "P1m dataset fingerprint"
            ),
            qwen_readout_ledger_sha256=_require_sha256(
                ledger["qwen_readouts"], "P1m Qwen readout ledger"
            ),
            structured_readout_ledger_sha256=_require_sha256(
                ledger["structured_readouts"], "P1m structured readout ledger"
            ),
            qwen_decision_ledger_sha256=_require_sha256(
                ledger["qwen_decisions"], "P1m Qwen decision ledger"
            ),
            structured_decision_ledger_sha256=_require_sha256(
                ledger["structured_decisions"], "P1m structured decision ledger"
            ),
            arm_metrics=tuple(
                RelationshipP1mArmMetrics.from_payload(item)
                for item in metrics_raw
            ),
            verdict=RelationshipP1mQualificationVerdict(
                _require_text(raw["verdict"], "P1m report verdict")
            ),
            qualification_passed=_require_bool(
                raw["qualification_passed"], "P1m report qualification pass"
            ),
            scenario_versioning_closed=_require_bool(
                raw["scenario_versioning_closed"],
                "P1m report scenario versioning closure",
            ),
            evaluation_feedback_to_system=_require_bool(
                raw["evaluation_feedback_to_system"],
                "P1m report evaluation feedback",
            ),
            claim_boundary=_require_text(
                raw["claim_boundary"], "P1m report claim boundary"
            ),
            schema_version=_require_text(
                raw["schema_version"], "P1m report schema"
            ),
        )
        if report.artifact_id != _require_sha256(
            raw["artifact_id"], "P1m report artifact id"
        ):
            raise ValueError("P1m qualification report artifact id mismatch")
        return report


def load_relationship_p1m_qualification_report(
    path: pathlib.Path,
) -> RelationshipP1mQualificationReport:
    return RelationshipP1mQualificationReport.from_payload(
        json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    )


def validate_relationship_p1m_qualification_report_files(
    report: RelationshipP1mQualificationReport,
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mQualificationProtocol,
    plan: RelationshipP1mQualificationPlan,
) -> None:
    if (
        report.protocol_id != protocol.protocol_id
        or report.plan_artifact_id != plan.artifact_id
        or report.dataset_fingerprint != plan.dataset_fingerprint
        or protocol.plan_artifact_id != plan.artifact_id
        or protocol.dataset_fingerprint != plan.dataset_fingerprint
    ):
        raise ValueError("P1m qualification report lineage drift")
    ledgers = (
        (
            "qwen_readouts.jsonl",
            report.qwen_readout_ledger_sha256,
            protocol.planned_qwen_readouts,
        ),
        (
            "qwen_decisions.jsonl",
            report.qwen_decision_ledger_sha256,
            protocol.planned_qwen_readouts,
        ),
        (
            "structured_readouts.jsonl",
            report.structured_readout_ledger_sha256,
            protocol.planned_structured_readouts,
        ),
        (
            "structured_decisions.jsonl",
            report.structured_decision_ledger_sha256,
            protocol.planned_structured_readouts,
        ),
    )
    for filename, expected_sha256, expected_lines in ledgers:
        path = pathlib.Path(output_dir) / filename
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"P1m qualification ledger drift: {filename}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != expected_lines or any(not line.strip() for line in lines):
            raise ValueError(f"P1m qualification ledger shape drift: {filename}")


def assess_relationship_p1m_qualification(
    *,
    protocol: RelationshipP1mQualificationProtocol,
    plan: RelationshipP1mQualificationPlan,
    decisions: tuple[RelationshipP1mQualificationDecision, ...],
    qwen_readout_ledger_sha256: str,
    structured_readout_ledger_sha256: str,
    qwen_decision_ledger_sha256: str,
    structured_decision_ledger_sha256: str,
    created_at_iso: str,
) -> RelationshipP1mQualificationReport:
    if protocol.plan_artifact_id != plan.artifact_id:
        raise ValueError("P1m report protocol/plan lineage mismatch")
    metrics = tuple(
        relationship_p1m_arm_metrics(decisions, arm=arm)
        for arm in RelationshipP1mQualificationArm
    )
    by_arm = {item.arm: item for item in metrics}
    prompt = by_arm[RelationshipP1mQualificationArm.PROMPT_STEELMAN]
    structured = by_arm[RelationshipP1mQualificationArm.STRUCTURED_STATE]
    if any(
        item.valid_decisions != RELATIONSHIP_P1M_REQUIRED_DECISIONS_PER_ARM
        for item in metrics
    ):
        verdict = RelationshipP1mQualificationVerdict.MACHINERY_INVALID
    elif (
        prompt.accuracy < RELATIONSHIP_P1M_ACCURACY_MINIMUM
        or prompt.accuracy_wilson_lower
        < RELATIONSHIP_P1M_ACCURACY_WILSON_LOWER_MINIMUM
    ):
        verdict = RelationshipP1mQualificationVerdict.BASELINE_TOO_WEAK
    elif prompt.accuracy > RELATIONSHIP_P1M_ACCURACY_MAXIMUM:
        verdict = RelationshipP1mQualificationVerdict.BASELINE_SATURATED
    elif (
        prompt.pair_flip_wilson_lower
        <= RELATIONSHIP_P1M_PAIR_FLIP_WILSON_LOWER_EXCLUSIVE
    ):
        verdict = RelationshipP1mQualificationVerdict.BASELINE_FLIP_INSUFFICIENT
    elif (
        structured.pair_flip_wilson_lower
        <= RELATIONSHIP_P1M_PAIR_FLIP_WILSON_LOWER_EXCLUSIVE
    ):
        verdict = RelationshipP1mQualificationVerdict.STRUCTURED_FLIP_INSUFFICIENT
    else:
        verdict = RelationshipP1mQualificationVerdict.QUALIFIED
    passed = verdict is RelationshipP1mQualificationVerdict.QUALIFIED
    return RelationshipP1mQualificationReport(
        created_at_iso=created_at_iso,
        protocol_id=protocol.protocol_id,
        plan_artifact_id=plan.artifact_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        qwen_readout_ledger_sha256=qwen_readout_ledger_sha256,
        structured_readout_ledger_sha256=structured_readout_ledger_sha256,
        qwen_decision_ledger_sha256=qwen_decision_ledger_sha256,
        structured_decision_ledger_sha256=structured_decision_ledger_sha256,
        arm_metrics=metrics,
        verdict=verdict,
        qualification_passed=passed,
        scenario_versioning_closed=not passed,
        evaluation_feedback_to_system=False,
    )


def write_relationship_p1m_qualification_report(
    report: RelationshipP1mQualificationReport,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "qualification_report.json"
    _atomic_write_text(
        path,
        json.dumps(
            {**report.to_payload(), "artifact_id": report.artifact_id},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


__all__ = [
    "RELATIONSHIP_P1M_QUALIFICATION_PLAN_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_QUALIFICATION_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1M_QUALIFICATION_REPORT_SCHEMA_VERSION",
    "RelationshipP1mArmMetrics",
    "RelationshipP1mQualificationArm",
    "RelationshipP1mQualificationDecision",
    "RelationshipP1mQualificationPlan",
    "RelationshipP1mQualificationProtocol",
    "RelationshipP1mQualificationReport",
    "RelationshipP1mQualificationVerdict",
    "RelationshipP1mQwenPlanRecord",
    "RelationshipP1mQwenReadout",
    "RelationshipP1mStructuredPlanRecord",
    "RelationshipP1mStructuredReadout",
    "assess_relationship_p1m_qualification",
    "build_relationship_p1m_qualification_plan",
    "build_relationship_p1m_reader_artifact",
    "load_relationship_p1m_qualification_plan",
    "load_relationship_p1m_qualification_protocol",
    "load_relationship_p1m_qualification_report",
    "relationship_p1m_arm_metrics",
    "relationship_p1m_forced_choice_prompt_path",
    "relationship_p1m_forced_choice_request_path",
    "relationship_p1m_public_episode",
    "render_relationship_p1m_forced_choice_request",
    "validate_relationship_p1m_qualification_report_files",
    "wilson_one_sided_lower",
    "write_relationship_p1m_qualification_plan",
    "write_relationship_p1m_qualification_protocol",
    "write_relationship_p1m_qualification_report",
]
