"""Content-addressed, resumable storage for generated corpus runs."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonical_json, trajectory_from_json
from .contracts import ExperienceTrajectory


@dataclass(frozen=True)
class CompletedRecord:
    trajectory_id: str
    trajectory_hash: str
    object_uri: str
    model_id: str | None
    prompt_hash: str | None
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class AppendOnlyJournal:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def completed(self) -> dict[str, CompletedRecord]:
        if not self._path.exists():
            return {}
        output: dict[str, CompletedRecord] = {}
        with self._path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid journal JSON at line {line_number}") from error
                if not isinstance(event, dict):
                    raise ValueError(f"journal line {line_number} must be an object")
                if event.get("event") != "completed":
                    continue
                record = _completed_from_event(event, line_number=line_number)
                previous = output.get(record.trajectory_id)
                if previous is not None and previous.trajectory_hash != record.trajectory_hash:
                    raise ValueError(f"journal contains conflicting hashes for {record.trajectory_id!r}")
                output[record.trajectory_id] = record
        return output

    def append_completed(self, record: CompletedRecord, *, timestamp: str) -> None:
        self._append(
            {
                "event": "completed",
                "timestamp": timestamp,
                "trajectory_id": record.trajectory_id,
                "trajectory_hash": record.trajectory_hash,
                "object_uri": record.object_uri,
                "model_id": record.model_id,
                "prompt_hash": record.prompt_hash,
                "cost_usd": record.cost_usd,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
            }
        )

    def append_quarantined(
        self,
        *,
        trajectory_id: str,
        error_type: str,
        message: str,
        timestamp: str,
    ) -> None:
        self._append(
            {
                "event": "quarantined",
                "timestamp": timestamp,
                "trajectory_id": trajectory_id,
                "error_type": error_type,
                "message": message,
            }
        )

    def _append(self, event: dict[str, object]) -> None:
        line = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            with self._path.open("a", encoding="utf-8", newline="\n") as sink:
                sink.write(line)
                sink.write("\n")
                sink.flush()


class ContentAddressedStore:
    def __init__(self, run_root: Path) -> None:
        self._run_root = run_root
        self._objects_root = run_root / "objects"
        self._snapshots_root = run_root / "snapshot_sidecars"
        self._lock = threading.Lock()

    def put_trajectory(self, trajectory: ExperienceTrajectory) -> CompletedRecord:
        payload = canonical_json(trajectory)
        trajectory_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        relative = Path("objects") / trajectory_hash[:2] / f"{trajectory_hash}.json.gz"
        path = self._run_root / relative
        with self._lock:
            if path.exists():
                existing = _read_deterministic_gzip(path).decode("utf-8").rstrip("\n")
                if existing != payload:
                    raise ValueError(f"content-address collision at {path.as_posix()}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_deterministic_gzip(path, f"{payload}\n".encode("utf-8"))
            if trajectory.snapshot_frames:
                self._put_snapshot_sidecar(trajectory, trajectory_hash)
        return CompletedRecord(
            trajectory_id=trajectory.trajectory_id,
            trajectory_hash=trajectory_hash,
            object_uri=relative.as_posix(),
            model_id=trajectory.provenance.model_id,
            prompt_hash=trajectory.provenance.prompt_hash,
            cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    def load(self, record: CompletedRecord) -> ExperienceTrajectory:
        path = self._run_root / Path(record.object_uri)
        if not path.is_file():
            raise FileNotFoundError(f"journal object is missing: {path}")
        payload = _read_deterministic_gzip(path).decode("utf-8").rstrip("\n")
        observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if observed != record.trajectory_hash:
            raise ValueError(f"journal object hash mismatch for {record.trajectory_id!r}")
        trajectory = trajectory_from_json(payload)
        if trajectory.trajectory_id != record.trajectory_id:
            raise ValueError("journal trajectory_id does not match stored object")
        return trajectory

    def materialize_master_shards(
        self,
        records: tuple[CompletedRecord, ...],
        *,
        shard_size: int,
    ) -> tuple[Path, ...]:
        master_root = self._run_root / "master"
        master_root.mkdir(parents=True, exist_ok=True)
        for stale in master_root.glob("shard-*.jsonl.gz"):
            stale.unlink()
        for stale in master_root.glob("shard-*.jsonl.gz.sha256"):
            stale.unlink()
        ordered = tuple(sorted(records, key=lambda item: item.trajectory_id))
        shard_paths: list[Path] = []
        for offset in range(0, len(ordered), shard_size):
            shard_index = offset // shard_size
            shard_records = ordered[offset : offset + shard_size]
            buffer = io.BytesIO()
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=buffer,
                mtime=0,
            ) as compressed:
                for record in shard_records:
                    object_path = self._run_root / Path(record.object_uri)
                    compressed.write(_read_deterministic_gzip(object_path))
            path = master_root / f"shard-{shard_index:05d}.jsonl.gz"
            path.write_bytes(buffer.getvalue())
            checksum = sha256_file(path)
            path.with_suffix(path.suffix + ".sha256").write_text(
                f"{checksum}  {path.name}\n",
                encoding="utf-8",
            )
            shard_paths.append(path)
        return tuple(shard_paths)

    def _put_snapshot_sidecar(
        self,
        trajectory: ExperienceTrajectory,
        trajectory_hash: str,
    ) -> Path:
        payload = canonical_json(
            {
                "schema_version": trajectory.schema_version,
                "trajectory_id": trajectory.trajectory_id,
                "trajectory_hash": trajectory_hash,
                "snapshot_frames": trajectory.snapshot_frames,
            }
        )
        sidecar_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path = self._snapshots_root / sidecar_hash[:2] / f"{sidecar_hash}.json.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_deterministic_gzip(path, f"{payload}\n".encode("utf-8"))
        return path


def write_run_config(path: Path, payload: dict[str, object]) -> str:
    encoded = canonical_json(payload)
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing != encoded:
            raise ValueError("run config differs from the existing resume checkpoint")
        return fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{encoded}\n", encoding="utf-8")
    return fingerprint


def write_quarantine_record(
    path: Path,
    *,
    trajectory_id: str,
    error_type: str,
    message: str,
    timestamp: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "trajectory_id": trajectory_id,
        "error_type": error_type,
        "message": message,
        "timestamp": timestamp,
        "training_use": "quarantined",
    }
    line = json.dumps(
        event,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as sink:
        sink.write(line)
        sink.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_deterministic_gzip(path: Path, payload: bytes) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(payload)


def _read_deterministic_gzip(path: Path) -> bytes:
    with gzip.open(path, "rb") as source:
        return source.read()


def _completed_from_event(
    event: dict[str, object],
    *,
    line_number: int,
) -> CompletedRecord:
    required = {
        "event",
        "timestamp",
        "trajectory_id",
        "trajectory_hash",
        "object_uri",
        "model_id",
        "prompt_hash",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
    }
    if set(event) != required:
        raise ValueError(f"journal completed event line {line_number} has invalid fields")
    trajectory_id = event["trajectory_id"]
    trajectory_hash = event["trajectory_hash"]
    object_uri = event["object_uri"]
    model_id = event["model_id"]
    prompt_hash = event["prompt_hash"]
    cost_usd = event["cost_usd"]
    prompt_tokens = event["prompt_tokens"]
    completion_tokens = event["completion_tokens"]
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"journal line {line_number} trajectory_id is invalid")
    if not isinstance(trajectory_hash, str) or len(trajectory_hash) != 64:
        raise ValueError(f"journal line {line_number} trajectory_hash is invalid")
    if not isinstance(object_uri, str) or not object_uri:
        raise ValueError(f"journal line {line_number} object_uri is invalid")
    if model_id is not None and not isinstance(model_id, str):
        raise ValueError(f"journal line {line_number} model_id is invalid")
    if prompt_hash is not None and not isinstance(prompt_hash, str):
        raise ValueError(f"journal line {line_number} prompt_hash is invalid")
    if type(cost_usd) not in {int, float} or float(cost_usd) < 0:
        raise ValueError(f"journal line {line_number} cost_usd is invalid")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError(f"journal line {line_number} prompt_tokens is invalid")
    if type(completion_tokens) is not int or completion_tokens < 0:
        raise ValueError(f"journal line {line_number} completion_tokens is invalid")
    return CompletedRecord(
        trajectory_id=trajectory_id,
        trajectory_hash=trajectory_hash,
        object_uri=object_uri,
        model_id=model_id,
        prompt_hash=prompt_hash,
        cost_usd=float(cost_usd),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


__all__ = [
    "AppendOnlyJournal",
    "CompletedRecord",
    "ContentAddressedStore",
    "sha256_file",
    "write_quarantine_record",
    "write_run_config",
]
