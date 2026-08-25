"""Invitation-only seven-day relationship assistant pilot evidence harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

_METRIC_NAMES = (
    "callback_hit_rate",
    "boundary_violation_rate",
    "wrong_user_attribution_rate",
    "open_loop_closure_rate",
    "user_correction_rate",
    "remembered_item_usefulness",
    "seven_day_trust_delta",
)


@dataclass(frozen=True)
class PilotTranscriptTurn:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("pilot transcript role must be user or assistant")
        if not self.text.strip():
            raise ValueError("pilot transcript text must be non-empty")


@dataclass(frozen=True)
class PilotDayEvidence:
    schema_version: str
    pilot_id: str
    participant_hash: str
    day_index: int
    captured_at_ms: int
    metrics: Mapping[str, float | None]
    metric_sample_sizes: Mapping[str, int]
    transcript_ref: str
    transcript_sha256: str
    l4_material_ready: bool


class RelationshipAssistantPilotHarness:
    """Writes deidentified daily metrics and transcript material for L4 review."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        pilot_id: str,
        invited_user_ids: frozenset[str],
    ) -> None:
        if not pilot_id.strip():
            raise ValueError("pilot_id must be non-empty")
        if not invited_user_ids:
            raise ValueError("pilot requires at least one invited user")
        self._root = Path(root_dir)
        self._pilot_id = pilot_id
        self._invited = invited_user_ids

    def capture_day(
        self,
        *,
        user_id: str,
        day_index: int,
        captured_at_ms: int,
        continuity_metrics: Mapping[str, Any],
        transcript: tuple[PilotTranscriptTurn, ...],
    ) -> PilotDayEvidence:
        if user_id not in self._invited:
            raise PermissionError("user is not invited to this pilot")
        if day_index < 1 or day_index > 7:
            raise ValueError("day_index must be in [1, 7]")
        if captured_at_ms < 0:
            raise ValueError("captured_at_ms must be non-negative")
        missing = tuple(name for name in _METRIC_NAMES if name not in continuity_metrics)
        if missing:
            raise ValueError(f"continuity metrics missing fields: {missing!r}")
        participant_hash = _participant_hash(self._pilot_id, user_id)
        participant_dir = self._root / self._pilot_id / participant_hash
        participant_dir.mkdir(parents=True, exist_ok=True)
        transcript_rows = [
            {
                "role": turn.role,
                "text": turn.text.replace(user_id, "[participant]"),
            }
            for turn in transcript
        ]
        transcript_payload = {
            "schema_version": "relationship-assistant-transcript.v1",
            "pilot_id": self._pilot_id,
            "participant_hash": participant_hash,
            "day_index": day_index,
            "turns": transcript_rows,
        }
        transcript_bytes = (
            json.dumps(
                transcript_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        transcript_path = participant_dir / f"day-{day_index}-transcript.json"
        transcript_path.write_bytes(transcript_bytes)
        sample_sizes = continuity_metrics.get("sample_sizes", {})
        if not isinstance(sample_sizes, dict):
            raise ValueError("continuity metric sample_sizes must be an object")
        evidence = PilotDayEvidence(
            schema_version="relationship-assistant-pilot-day.v1",
            pilot_id=self._pilot_id,
            participant_hash=participant_hash,
            day_index=day_index,
            captured_at_ms=captured_at_ms,
            metrics={name: continuity_metrics[name] for name in _METRIC_NAMES},
            metric_sample_sizes={
                str(name): int(value) for name, value in sample_sizes.items()
            },
            transcript_ref=str(transcript_path.relative_to(self._root)),
            transcript_sha256=sha256(transcript_bytes).hexdigest(),
            l4_material_ready=bool(transcript_rows),
        )
        evidence_path = participant_dir / f"day-{day_index}-metrics.json"
        temporary = evidence_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(evidence), ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(evidence_path)
        return evidence


def _participant_hash(pilot_id: str, user_id: str) -> str:
    return sha256(f"{pilot_id}\0{user_id}".encode("utf-8")).hexdigest()[:24]


__all__ = [
    "PilotDayEvidence",
    "PilotTranscriptTurn",
    "RelationshipAssistantPilotHarness",
]
