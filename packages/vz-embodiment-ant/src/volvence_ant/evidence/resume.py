"""Config-bound, atomically committed partial state for ant evidence runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from volvence_ant.evidence.provenance import atomic_write_json, stable_json_digest

ANT_PARTIAL_SCHEMA_VERSION = "digital-ant-partial.v1"


class AntResumeStateError(RuntimeError):
    """A partial run cannot be trusted for the requested configuration."""


def ant_stage_fingerprint(*, stage: str, config: Mapping[str, Any]) -> str:
    return stable_json_digest(
        {
            "schema_version": ANT_PARTIAL_SCHEMA_VERSION,
            "stage": stage,
            "config": dict(config),
        }
    )


class SeedPartialStore:
    """One atomic partial payload per complete seed; no owner state is stored."""

    def __init__(
        self,
        *,
        results_root: Path,
        stage: str,
        fingerprint: str,
        requested_seeds: Sequence[int],
    ) -> None:
        if not requested_seeds:
            raise ValueError("requested_seeds must be non-empty")
        self.stage = stage
        self.fingerprint = fingerprint
        self.requested_seeds = tuple(int(seed) for seed in requested_seeds)
        self.root = results_root / ".partials" / stage / fingerprint
        self._metadata_path = self.root / "run.json"
        self._seed_dir = self.root / "seeds"

    def initialize(self) -> None:
        expected = self._metadata()
        if self._metadata_path.is_file():
            actual = self._read_object(self._metadata_path)
            if actual != expected:
                raise AntResumeStateError(
                    f"partial metadata mismatch at {self._metadata_path}"
                )
        else:
            atomic_write_json(self._metadata_path, expected)

    def load(self) -> dict[int, Mapping[str, Any]]:
        self.initialize()
        completed: dict[int, Mapping[str, Any]] = {}
        if not self._seed_dir.is_dir():
            return completed
        for path in sorted(self._seed_dir.glob("seed-*.json")):
            payload = self._read_object(path)
            if payload.get("schema_version") != ANT_PARTIAL_SCHEMA_VERSION:
                raise AntResumeStateError(f"unsupported partial schema: {path}")
            if payload.get("stage") != self.stage:
                raise AntResumeStateError(f"partial stage mismatch: {path}")
            if payload.get("fingerprint") != self.fingerprint:
                raise AntResumeStateError(f"partial fingerprint mismatch: {path}")
            seed = payload.get("seed")
            if not isinstance(seed, int) or seed not in self.requested_seeds:
                raise AntResumeStateError(f"unexpected partial seed {seed!r}: {path}")
            if seed in completed:
                raise AntResumeStateError(f"duplicate partial seed {seed}: {path}")
            report = payload.get("report")
            if not isinstance(report, dict):
                raise AntResumeStateError(f"partial report must be an object: {path}")
            completed[seed] = report
        return completed

    def commit(self, *, seed: int, report: Mapping[str, Any]) -> Path:
        self.initialize()
        if seed not in self.requested_seeds:
            raise AntResumeStateError(f"seed {seed} was not requested")
        path = self._seed_dir / f"seed-{seed:06d}.json"
        payload = {
            "schema_version": ANT_PARTIAL_SCHEMA_VERSION,
            "stage": self.stage,
            "fingerprint": self.fingerprint,
            "seed": seed,
            "report": dict(report),
        }
        if path.is_file():
            if self._read_object(path) != payload:
                raise AntResumeStateError(f"conflicting partial seed {seed}: {path}")
            return path
        atomic_write_json(path, payload)
        return path

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ANT_PARTIAL_SCHEMA_VERSION,
            "stage": self.stage,
            "fingerprint": self.fingerprint,
            "requested_seeds": list(self.requested_seeds),
        }

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AntResumeStateError(f"invalid partial JSON: {path}") from exc
        if not isinstance(value, dict):
            raise AntResumeStateError(f"partial JSON must be an object: {path}")
        return value


__all__ = [
    "ANT_PARTIAL_SCHEMA_VERSION",
    "AntResumeStateError",
    "SeedPartialStore",
    "ant_stage_fingerprint",
]
