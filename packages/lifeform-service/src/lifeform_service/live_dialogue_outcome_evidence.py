"""Privacy-bounded artifact sink for explicit live dialogue outcomes.

The service does not infer failures from conversation text.  It only exports
the typed ``DialogueExternalOutcomeEvidence`` already accepted by the runtime
owner, plus a compact action-turn readout.  Raw user/model text, free-form
descriptions, evidence refs, and plaintext identity keys are deliberately not
part of this artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from volvence_zero.dialogue_trace import DialogueExternalOutcomeEvidence

if TYPE_CHECKING:
    from lifeform_core.types import TurnSummary


LIVE_DIALOGUE_OUTCOME_SCHEMA_VERSION = "lifeform-live-dialogue-outcome.v1"
LIVE_DIALOGUE_OUTCOME_PRIVACY_PROFILE = "typed-metadata-only.v1"


def _hash_text(value: str) -> str:
    if not value:
        raise ValueError("live dialogue outcome hash input must be non-empty")
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_timezone(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("recorded_at_iso must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("recorded_at_iso must include a timezone")


@dataclass(frozen=True)
class LiveDialogueActionContext:
    """Text-free public readout for the action turn being evaluated."""

    turn_index: int
    scene_id_sha256: str
    trigger_kind: str
    active_regime: str | None
    active_abstract_action: str | None
    prediction_error_magnitude: float
    open_loop_count: int
    commitment_count: int
    elapsed_at_tick: int

    @classmethod
    def from_turn_summary(
        cls,
        summary: "TurnSummary",
    ) -> "LiveDialogueActionContext":
        return cls(
            turn_index=summary.turn_index,
            scene_id_sha256=_hash_text(summary.scene_id),
            trigger_kind=summary.trigger_kind.value,
            active_regime=summary.active_regime,
            active_abstract_action=summary.active_abstract_action,
            prediction_error_magnitude=float(summary.pe_magnitude),
            open_loop_count=summary.open_loop_count,
            commitment_count=summary.commitment_count,
            elapsed_at_tick=summary.elapsed_at_tick,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "scene_id_sha256": self.scene_id_sha256,
            "trigger_kind": self.trigger_kind,
            "active_regime": self.active_regime,
            "active_abstract_action": self.active_abstract_action,
            "prediction_error_magnitude": self.prediction_error_magnitude,
            "open_loop_count": self.open_loop_count,
            "commitment_count": self.commitment_count,
            "elapsed_at_tick": self.elapsed_at_tick,
        }


@dataclass(frozen=True)
class LiveDialogueOutcomeArtifact:
    """Immutable, de-identified service evidence artifact."""

    artifact_id: str
    recorded_at_iso: str
    subject_scope_sha256: str
    session_scope_sha256: str
    source_evidence_sha256: str
    outcome_kind: str
    evidence_source: str
    confidence: float
    consuming_turn_index: int
    action_turn_index: int
    action_context: LiveDialogueActionContext | None
    service_version: str
    policy_version: str

    def __post_init__(self) -> None:
        _require_timezone(self.recorded_at_iso)
        if not self.artifact_id:
            raise ValueError("live dialogue outcome artifact_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("live dialogue outcome confidence must be in [0, 1]")
        if self.consuming_turn_index < 0:
            raise ValueError("consuming_turn_index must be non-negative")
        if self.action_turn_index < -1:
            raise ValueError("action_turn_index must be non-negative or -1")
        if (
            self.action_context is not None
            and self.action_context.turn_index != self.action_turn_index
        ):
            raise ValueError("action_context must describe action_turn_index")

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": LIVE_DIALOGUE_OUTCOME_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "recorded_at_iso": self.recorded_at_iso,
            "subject_scope_sha256": self.subject_scope_sha256,
            "session_scope_sha256": self.session_scope_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "outcome_kind": self.outcome_kind,
            "evidence_source": self.evidence_source,
            "confidence": self.confidence,
            "consuming_turn_index": self.consuming_turn_index,
            "action_turn_index": self.action_turn_index,
            "action_context": (
                self.action_context.to_json()
                if self.action_context is not None
                else None
            ),
            "service_version": self.service_version,
            "policy_version": self.policy_version,
            "privacy_profile": LIVE_DIALOGUE_OUTCOME_PRIVACY_PROFILE,
        }
        payload["content_sha256"] = sha256(_canonical_bytes(payload)).hexdigest()
        return payload


def build_live_dialogue_outcome_artifact(
    *,
    subject_scope: str,
    session_id: str,
    evidence: DialogueExternalOutcomeEvidence,
    turn_summaries: tuple["TurnSummary", ...],
    service_version: str,
    policy_version: str,
    recorded_at_iso: str | None = None,
) -> LiveDialogueOutcomeArtifact:
    """Project one owner-issued outcome into the privacy-bounded contract."""

    action_summary = next(
        (
            summary
            for summary in turn_summaries
            if summary.turn_index == evidence.action_turn_index
        ),
        None,
    )
    source_evidence_sha256 = _hash_text(evidence.evidence_id)
    return LiveDialogueOutcomeArtifact(
        artifact_id=f"live-dialogue-outcome:{source_evidence_sha256[:24]}",
        recorded_at_iso=(
            recorded_at_iso
            or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        ),
        subject_scope_sha256=_hash_text(subject_scope),
        session_scope_sha256=_hash_text(session_id),
        source_evidence_sha256=source_evidence_sha256,
        outcome_kind=evidence.kind.value,
        evidence_source=evidence.source.value,
        confidence=evidence.confidence,
        consuming_turn_index=evidence.turn_index,
        action_turn_index=evidence.action_turn_index,
        action_context=(
            LiveDialogueActionContext.from_turn_summary(action_summary)
            if action_summary is not None
            else None
        ),
        service_version=service_version,
        policy_version=policy_version,
    )


def _validate_existing_artifact(
    path: Path,
    *,
    expected: LiveDialogueOutcomeArtifact,
) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing live outcome artifact {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"existing live outcome artifact {path} must be a JSON object")
    expected_payload = expected.to_json()
    if set(raw) != set(expected_payload):
        raise ValueError(
            f"existing live outcome artifact {path} has unexpected schema fields"
        )
    stored_digest = raw.get("content_sha256")
    body = {key: value for key, value in raw.items() if key != "content_sha256"}
    computed_digest = sha256(_canonical_bytes(body)).hexdigest()
    if stored_digest != computed_digest:
        raise ValueError(f"existing live outcome artifact {path} failed content_sha256 validation")
    stable_keys = set(expected_payload) - {"recorded_at_iso", "content_sha256"}
    mismatches = sorted(
        key for key in stable_keys if raw.get(key) != expected_payload[key]
    )
    if mismatches:
        raise ValueError(
            f"existing live outcome artifact {path} conflicts on fields: {mismatches}"
        )


def write_live_dialogue_outcome_artifact(
    *,
    evidence_root: Path,
    artifact: LiveDialogueOutcomeArtifact,
) -> Path:
    """Create one content-verified artifact without overwriting prior evidence."""

    root = evidence_root.expanduser().resolve() / "live_dialogue_outcomes"
    shard = artifact.source_evidence_sha256[:2]
    target_dir = root / shard
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{artifact.source_evidence_sha256}.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                artifact.to_json(),
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError:
        _validate_existing_artifact(path, expected=artifact)
    return path


__all__ = (
    "LIVE_DIALOGUE_OUTCOME_PRIVACY_PROFILE",
    "LIVE_DIALOGUE_OUTCOME_SCHEMA_VERSION",
    "LiveDialogueActionContext",
    "LiveDialogueOutcomeArtifact",
    "build_live_dialogue_outcome_artifact",
    "write_live_dialogue_outcome_artifact",
)
