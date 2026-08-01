"""Crash-safe checkpoint journal for the MSC N+1 research runner.

The journal is execution control, not an evaluation owner.  Immutable unit
files contain only numeric representations, hashes, sample ids, and metrics;
raw MSC utterances are never persisted here.  ``run_state.json`` is mutable
and intentionally excluded from evidence hashes because it only records
which immutable units are available for resume.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping


MSC_PREDICTION_RUN_STATE_SCHEMA_VERSION = "msc-prediction-run-state.v1"
MSC_PREDICTION_JSON_CHECKPOINT_SCHEMA_VERSION = (
    "msc-prediction-json-checkpoint.v1"
)
MSC_PREDICTION_ARRAY_CHECKPOINT_SCHEMA_VERSION = (
    "msc-prediction-array-checkpoint.v1"
)


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


def _json_normalize(value: object) -> object:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"checkpoint relative path is unsafe: {value!r}")
    return Path(*pure.parts)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class CheckpointFile:
    unit: str
    relative_path: str
    kind: str
    sha256: str
    size_bytes: int


class PredictionRunCheckpointStore:
    """Own a resumable run journal bound to one exact configuration."""

    def __init__(
        self,
        *,
        output_dir: Path,
        configuration: Mapping[str, object],
        resume: bool,
    ) -> None:
        self.output_dir = output_dir.resolve()
        normalized_configuration = _json_normalize(dict(configuration))
        if not isinstance(normalized_configuration, dict):
            raise ValueError("prediction checkpoint configuration must be an object")
        self.configuration = normalized_configuration
        self.configuration_fingerprint = _sha256_bytes(
            _canonical_bytes(self.configuration)
        )
        self._state_path = self.output_dir / "run_state.json"
        if self.output_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"prediction output is immutable without --resume: {self.output_dir}"
                )
            if not self._state_path.is_file():
                raise FileNotFoundError(
                    "prediction resume requires run_state.json; legacy or partial "
                    f"output cannot be adopted: {self.output_dir}"
                )
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("prediction run_state root must be an object")
            self._state = state
            self._validate_state()
        else:
            if resume:
                raise FileNotFoundError(
                    f"prediction --resume output does not exist: {self.output_dir}"
                )
            self.output_dir.mkdir(parents=True)
            self._state: dict[str, object] = {
                "schema_version": MSC_PREDICTION_RUN_STATE_SCHEMA_VERSION,
                "configuration": self.configuration,
                "configuration_fingerprint": self.configuration_fingerprint,
                "status": "running",
                "last_completed_unit": None,
                "completed_units": {},
                "analysis_allowed": False,
                "formal_claim_allowed": False,
                "raw_corpus_text_retained": False,
            }
            self._write_state()

    def _validate_state(self) -> None:
        if (
            self._state.get("schema_version")
            != MSC_PREDICTION_RUN_STATE_SCHEMA_VERSION
        ):
            raise ValueError("prediction run_state schema_version mismatch")
        if self._state.get("configuration") != self.configuration:
            raise ValueError("prediction resume configuration drift")
        if (
            self._state.get("configuration_fingerprint")
            != self.configuration_fingerprint
        ):
            raise ValueError("prediction resume configuration fingerprint drift")
        units = self._state.get("completed_units")
        if not isinstance(units, dict):
            raise ValueError("prediction run_state completed_units is invalid")
        status = self._state.get("status")
        analysis_allowed = self._state.get("analysis_allowed")
        if status not in {"running", "complete"}:
            raise ValueError("prediction run_state status is invalid")
        if (
            not isinstance(analysis_allowed, bool)
            or analysis_allowed != (status == "complete")
        ):
            raise ValueError("prediction run_state analysis gate is invalid")
        if self._state.get("formal_claim_allowed") is not False:
            raise ValueError("prediction run_state cannot authorize a formal claim")
        if self._state.get("raw_corpus_text_retained") is not False:
            raise ValueError("prediction checkpoint cannot retain raw corpus text")
        registered_paths: set[str] = set()
        for unit, raw_entry in units.items():
            if not isinstance(unit, str) or not isinstance(raw_entry, dict):
                raise ValueError("prediction run_state unit entry is invalid")
            entry = self._parse_entry(unit, raw_entry)
            if entry.relative_path in registered_paths:
                raise ValueError(
                    "prediction checkpoint path is registered to multiple units: "
                    f"{entry.relative_path}"
                )
            registered_paths.add(entry.relative_path)
            path = self.output_dir / _safe_relative_path(entry.relative_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"registered prediction checkpoint is missing: {path}"
                )
            if _sha256_file(path) != entry.sha256:
                raise ValueError(
                    f"registered prediction checkpoint hash drift: {path}"
                )

    def _write_state(self) -> None:
        _atomic_write(self._state_path, _canonical_bytes(self._state))

    @staticmethod
    def _parse_entry(unit: str, value: Mapping[str, object]) -> CheckpointFile:
        relative_path = value.get("relative_path")
        kind = value.get("kind")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or kind not in {"json", "arrays"}
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 1
        ):
            raise ValueError(f"prediction checkpoint entry is invalid: {unit!r}")
        _safe_relative_path(relative_path)
        return CheckpointFile(
            unit=unit,
            relative_path=relative_path,
            kind=kind,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def _registered_entry(self, unit: str) -> CheckpointFile | None:
        units = self._state["completed_units"]
        if not isinstance(units, dict):
            raise RuntimeError("prediction checkpoint unit registry changed type")
        value = units.get(unit)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"prediction checkpoint entry is invalid: {unit!r}")
        return self._parse_entry(unit, value)

    def _register(self, *, unit: str, path: Path, kind: str) -> None:
        relative = path.relative_to(self.output_dir).as_posix()
        entry = {
            "relative_path": relative,
            "kind": kind,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        units = self._state["completed_units"]
        if not isinstance(units, dict):
            raise RuntimeError("prediction checkpoint unit registry changed type")
        existing = units.get(unit)
        if existing is not None and existing != entry:
            raise ValueError(f"prediction checkpoint unit registration drift: {unit}")
        units[unit] = entry
        self._state["last_completed_unit"] = unit
        self._write_state()

    def _resolve_unit_path(
        self,
        *,
        unit: str,
        relative_path: str,
        kind: str,
    ) -> tuple[Path, CheckpointFile | None]:
        if not unit.strip():
            raise ValueError("prediction checkpoint unit must be non-empty")
        path = self.output_dir / _safe_relative_path(relative_path)
        registered = self._registered_entry(unit)
        if registered is not None:
            if registered.relative_path != relative_path or registered.kind != kind:
                raise ValueError(f"prediction checkpoint unit path/kind drift: {unit}")
            if not path.is_file() or _sha256_file(path) != registered.sha256:
                raise ValueError(f"prediction checkpoint unit file drift: {unit}")
        return path, registered

    def load_json(
        self,
        *,
        unit: str,
        relative_path: str,
    ) -> object | None:
        path, registered = self._resolve_unit_path(
            unit=unit,
            relative_path=relative_path,
            kind="json",
        )
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"prediction JSON checkpoint is not an object: {path}")
        if (
            payload.get("schema_version")
            != MSC_PREDICTION_JSON_CHECKPOINT_SCHEMA_VERSION
            or payload.get("unit") != unit
            or payload.get("configuration_fingerprint")
            != self.configuration_fingerprint
            or "payload" not in payload
        ):
            raise ValueError(f"prediction JSON checkpoint metadata drift: {path}")
        if registered is None:
            self._register(unit=unit, path=path, kind="json")
        return payload["payload"]

    def save_json(
        self,
        *,
        unit: str,
        relative_path: str,
        payload: object,
    ) -> None:
        path, registered = self._resolve_unit_path(
            unit=unit,
            relative_path=relative_path,
            kind="json",
        )
        if registered is not None or path.exists():
            raise FileExistsError(f"prediction JSON checkpoint already exists: {path}")
        envelope = {
            "schema_version": MSC_PREDICTION_JSON_CHECKPOINT_SCHEMA_VERSION,
            "unit": unit,
            "configuration_fingerprint": self.configuration_fingerprint,
            "payload": payload,
        }
        _atomic_write(path, _canonical_bytes(envelope))
        self._register(unit=unit, path=path, kind="json")

    def load_arrays(
        self,
        *,
        unit: str,
        relative_path: str,
        expected_metadata: Mapping[str, object],
    ) -> dict[str, object] | None:
        path, registered = self._resolve_unit_path(
            unit=unit,
            relative_path=relative_path,
            kind="arrays",
        )
        if not path.exists():
            return None
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("prediction array checkpoints require numpy") from exc
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError(f"prediction array checkpoint lacks metadata: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"prediction array checkpoint metadata is not an object: {path}"
                )
            expected_envelope = {
                "schema_version": MSC_PREDICTION_ARRAY_CHECKPOINT_SCHEMA_VERSION,
                "unit": unit,
                "configuration_fingerprint": self.configuration_fingerprint,
                "payload": _json_normalize(dict(expected_metadata)),
            }
            if metadata != expected_envelope:
                raise ValueError(f"prediction array checkpoint metadata drift: {path}")
            arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name != "metadata_json"
            }
        if registered is None:
            self._register(unit=unit, path=path, kind="arrays")
        return arrays

    def save_arrays(
        self,
        *,
        unit: str,
        relative_path: str,
        metadata: Mapping[str, object],
        arrays: Mapping[str, object],
    ) -> None:
        path, registered = self._resolve_unit_path(
            unit=unit,
            relative_path=relative_path,
            kind="arrays",
        )
        if registered is not None or path.exists():
            raise FileExistsError(f"prediction array checkpoint already exists: {path}")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("prediction array checkpoints require numpy") from exc
        envelope = {
            "schema_version": MSC_PREDICTION_ARRAY_CHECKPOINT_SCHEMA_VERSION,
            "unit": unit,
            "configuration_fingerprint": self.configuration_fingerprint,
            "payload": _json_normalize(dict(metadata)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(
                    json.dumps(envelope, ensure_ascii=False, sort_keys=True)
                ),
                **dict(arrays),
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._register(unit=unit, path=path, kind="arrays")

    def immutable_file_manifest(self) -> dict[str, dict[str, object]]:
        units = self._state["completed_units"]
        if not isinstance(units, dict):
            raise RuntimeError("prediction checkpoint unit registry changed type")
        manifest: dict[str, dict[str, object]] = {}
        for unit in sorted(units):
            raw = units[unit]
            if not isinstance(raw, dict):
                raise ValueError(f"prediction checkpoint entry is invalid: {unit!r}")
            entry = self._parse_entry(unit, raw)
            manifest[entry.relative_path] = {
                "unit": entry.unit,
                "kind": entry.kind,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
        return manifest

    def mark_complete(self) -> None:
        self._state["status"] = "complete"
        self._state["analysis_allowed"] = True
        self._state["formal_claim_allowed"] = False
        self._write_state()


__all__ = [
    "CheckpointFile",
    "MSC_PREDICTION_ARRAY_CHECKPOINT_SCHEMA_VERSION",
    "MSC_PREDICTION_JSON_CHECKPOINT_SCHEMA_VERSION",
    "MSC_PREDICTION_RUN_STATE_SCHEMA_VERSION",
    "PredictionRunCheckpointStore",
]
