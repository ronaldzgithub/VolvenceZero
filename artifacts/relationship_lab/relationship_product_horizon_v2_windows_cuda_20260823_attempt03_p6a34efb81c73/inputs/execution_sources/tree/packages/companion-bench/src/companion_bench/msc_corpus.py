# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Verified adapter for the official Multi-Session Chat v0.1 release.

The raw corpus is intentionally not vendored.  This module reads an operator-
downloaded official archive extraction, verifies the frozen file and dyad-id
hashes, and publishes immutable conversation values for offline research.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.resources as resources
import json
from pathlib import Path
from typing import Any, Mapping


_MANIFEST_RESOURCE = "corpora/msc_v0_1_manifest.json"
_SPLITS = frozenset({"train", "validation", "heldout"})


@dataclass(frozen=True)
class MSCUtterance:
    speaker: str
    text: str
    utterance_index: int
    preceding_empty_utterance_count: int = 0


@dataclass(frozen=True)
class MSCSession:
    session_index: int
    utterances: tuple[MSCUtterance, ...]
    elapsed_value: int | None = None
    elapsed_unit: str = ""
    elapsed_description: str = ""


@dataclass(frozen=True)
class MSCDyad:
    dyad_id: str
    split: str
    sessions: tuple[MSCSession, ...]
    initial_personas: tuple[tuple[str, ...], tuple[str, ...]]

    @property
    def utterance_count(self) -> int:
        return sum(len(session.utterances) for session in self.sessions)


@dataclass(frozen=True)
class MSCSplitAudit:
    split: str
    path: str
    conversation_count: int
    file_sha256: str
    sorted_id_sha256: str
    minimum_session_count: int
    maximum_session_count: int
    dropped_empty_utterance_count: int
    verified: bool


def load_msc_manifest() -> dict[str, Any]:
    root = resources.files("companion_bench")
    payload = json.loads(root.joinpath(_MANIFEST_RESOURCE).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "msc-corpus-manifest.v1":
        raise ValueError(
            "unsupported MSC manifest schema_version "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_id_sha256(ids: tuple[str, ...]) -> str:
    payload = "\n".join(sorted(ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_dialogue_root(dataset_root: Path) -> Path:
    candidates = (
        dataset_root,
        dataset_root / "msc_dialogue",
        dataset_root / "msc" / "msc_dialogue",
    )
    for candidate in candidates:
        if candidate.name == "msc_dialogue" and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "MSC extraction must contain msc/msc_dialogue or msc_dialogue; "
        f"searched under {dataset_root}"
    )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MSC {field} must be a mapping; got {type(value).__name__}")
    return value


def _utterances(
    raw_dialog: object, *, session_index: int
) -> tuple[tuple[MSCUtterance, ...], int]:
    if not isinstance(raw_dialog, list) or not raw_dialog:
        raise ValueError(f"MSC session {session_index} dialog must be a non-empty list")
    utterances: list[MSCUtterance] = []
    dropped_empty = 0
    pending_empty = 0
    speaker_by_source_id: dict[str, str] = {}
    for index, raw_turn in enumerate(raw_dialog, start=1):
        turn = _require_mapping(raw_turn, field=f"session {session_index} turn {index}")
        text = turn.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"MSC session {session_index} turn {index} requires string text"
            )
        expected_speaker = "speaker_1" if index % 2 == 1 else "speaker_2"
        raw_speaker = turn.get("id")
        if raw_speaker is not None:
            if not isinstance(raw_speaker, str) or not raw_speaker.strip():
                raise ValueError(
                    f"MSC session {session_index} turn {index} has invalid speaker id"
                )
            if raw_speaker in {"Speaker 1", "Speaker 2"}:
                observed_speaker = (
                    "speaker_1" if raw_speaker == "Speaker 1" else "speaker_2"
                )
            else:
                observed_speaker = speaker_by_source_id.setdefault(
                    raw_speaker,
                    expected_speaker,
                )
                if len(speaker_by_source_id) > 2:
                    raise ValueError(
                        f"MSC session {session_index} has more than two speaker ids"
                    )
            if observed_speaker != expected_speaker:
                raise ValueError(
                    f"MSC session {session_index} turn {index} speaker position drift"
                )
        if not text.strip():
            dropped_empty += 1
            pending_empty += 1
            continue
        if (
            utterances
            and utterances[-1].speaker == expected_speaker
            and pending_empty == 0
        ):
            raise ValueError(
                f"MSC session {session_index} retained speakers do not alternate"
            )
        utterances.append(
            MSCUtterance(
                speaker=expected_speaker,
                text=text.strip(),
                utterance_index=index,
                preceding_empty_utterance_count=pending_empty,
            )
        )
        pending_empty = 0
    if not utterances:
        raise ValueError(
            f"MSC session {session_index} has no non-empty utterances after cleaning"
        )
    return tuple(utterances), dropped_empty


def _personas(raw: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("MSC init_personas must contain exactly two persona lists")
    result: list[tuple[str, ...]] = []
    for speaker_index, values in enumerate(raw, start=1):
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(
                f"MSC init_personas speaker {speaker_index} must be a string list"
            )
        result.append(tuple(value.strip() for value in values if value.strip()))
    return (result[0], result[1])


def _parse_dyad(
    raw: object, *, split: str, minimum_sessions: int
) -> tuple[MSCDyad, int]:
    row = _require_mapping(raw, field="row")
    metadata = _require_mapping(row.get("metadata"), field="metadata")
    dyad_id = metadata.get("initial_data_id")
    if not isinstance(dyad_id, str) or not dyad_id.strip():
        raise ValueError("MSC metadata.initial_data_id must be non-empty")
    previous = row.get("previous_dialogs")
    if not isinstance(previous, list):
        raise ValueError("MSC previous_dialogs must be a list")
    sessions: list[MSCSession] = []
    dropped_empty = 0
    for session_index, raw_session in enumerate(previous, start=1):
        session = _require_mapping(raw_session, field=f"previous session {session_index}")
        elapsed_value = session.get("time_num")
        if elapsed_value is not None and (
            isinstance(elapsed_value, bool) or not isinstance(elapsed_value, int)
        ):
            raise ValueError(
                f"MSC session {session_index} time_num must be integer or null"
            )
        elapsed_unit = session.get("time_unit", "")
        elapsed_description = session.get("time_back", "")
        if not isinstance(elapsed_unit, str) or not isinstance(elapsed_description, str):
            raise ValueError(f"MSC session {session_index} elapsed fields must be strings")
        utterances, session_dropped = _utterances(
            session.get("dialog"), session_index=session_index
        )
        dropped_empty += session_dropped
        sessions.append(
            MSCSession(
                session_index=session_index,
                utterances=utterances,
                elapsed_value=elapsed_value,
                elapsed_unit=elapsed_unit.strip(),
                elapsed_description=elapsed_description.strip(),
            )
        )
    current_index = len(sessions) + 1
    current_utterances, current_dropped = _utterances(
        row.get("dialog"), session_index=current_index
    )
    dropped_empty += current_dropped
    sessions.append(
        MSCSession(
            session_index=current_index,
            utterances=current_utterances,
        )
    )
    if len(sessions) < minimum_sessions:
        raise ValueError(
            f"MSC dyad {dyad_id!r} has {len(sessions)} sessions; "
            f"minimum is {minimum_sessions}"
        )
    return (
        MSCDyad(
            dyad_id=dyad_id.strip(),
            split=split,
            sessions=tuple(sessions),
            initial_personas=_personas(row.get("init_personas")),
        ),
        dropped_empty,
    )


def load_msc_split(
    dataset_root: Path | str,
    *,
    split: str,
    strict: bool = True,
    limit: int | None = None,
) -> tuple[tuple[MSCDyad, ...], MSCSplitAudit]:
    """Load one frozen split and return immutable dyads plus an audit record."""

    if split not in _SPLITS:
        raise ValueError(f"MSC split must be one of {sorted(_SPLITS)}; got {split!r}")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError(f"MSC limit must be a positive integer or None; got {limit!r}")
    manifest = load_msc_manifest()
    split_manifest = _require_mapping(manifest["splits"][split], field=f"manifest {split}")
    dialogue_root = _resolve_dialogue_root(Path(dataset_root))
    relative_path = Path(str(split_manifest["relative_path"]))
    if relative_path.parts[0] != "msc_dialogue":
        raise ValueError(f"MSC manifest path must start with msc_dialogue: {relative_path}")
    path = dialogue_root.joinpath(*relative_path.parts[1:])
    if not path.is_file():
        raise FileNotFoundError(f"MSC split file not found: {path}")
    file_sha256 = _sha256_path(path)
    expected_file_sha = str(split_manifest["file_sha256"])
    if strict and file_sha256 != expected_file_sha:
        raise ValueError(
            f"MSC {split} file SHA-256 mismatch: expected {expected_file_sha}, "
            f"got {file_sha256}"
        )

    rows: list[MSCDyad] = []
    dropped_empty_utterances = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MSC {split} line {line_number} is invalid JSON") from exc
            dyad, dropped = _parse_dyad(
                payload,
                split=split,
                minimum_sessions=int(manifest["minimum_sessions"]),
            )
            rows.append(dyad)
            dropped_empty_utterances += dropped
    ids = tuple(row.dyad_id for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError(f"MSC {split} contains duplicate initial_data_id values")
    id_sha256 = _sorted_id_sha256(ids)
    if strict:
        expected_count = int(split_manifest["conversation_count"])
        expected_id_sha = str(split_manifest["sorted_id_sha256"])
        if len(rows) != expected_count:
            raise ValueError(
                f"MSC {split} count mismatch: expected {expected_count}, got {len(rows)}"
            )
        if id_sha256 != expected_id_sha:
            raise ValueError(
                f"MSC {split} id SHA-256 mismatch: expected {expected_id_sha}, "
                f"got {id_sha256}"
            )
    session_counts = tuple(len(row.sessions) for row in rows)
    audit = MSCSplitAudit(
        split=split,
        path=str(path),
        conversation_count=len(rows),
        file_sha256=file_sha256,
        sorted_id_sha256=id_sha256,
        minimum_session_count=min(session_counts, default=0),
        maximum_session_count=max(session_counts, default=0),
        dropped_empty_utterance_count=dropped_empty_utterances,
        verified=(
            file_sha256 == expected_file_sha
            and len(rows) == int(split_manifest["conversation_count"])
            and id_sha256 == str(split_manifest["sorted_id_sha256"])
        ),
    )
    selected = rows if limit is None else rows[:limit]
    return tuple(selected), audit


__all__ = (
    "MSCDyad",
    "MSCSession",
    "MSCSplitAudit",
    "MSCUtterance",
    "load_msc_manifest",
    "load_msc_split",
)
