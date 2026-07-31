"""Forge paths, editable-surface policy and frozen validation configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class ForgeConfigError(ValueError):
    """Raised when Forge governance configuration violates its contract."""


@dataclass(frozen=True)
class ForgePaths:
    repo_root: Path
    forge_root: Path
    artifacts_root: Path
    transcripts_root: Path
    plans_root: Path
    editable_surface_path: Path
    ledger_path: Path

    @classmethod
    def discover(
        cls,
        repo_root: Path | None = None,
        transcripts_root: Path | None = None,
    ) -> ForgePaths:
        root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        transcript_env = os.environ.get("FORGE_TRANSCRIPTS_ROOT")
        default_transcripts = (
            Path.home()
            / ".cursor"
            / "projects"
            / "Users-mengfu-Documents-GitHub-volvence"
            / "agent-transcripts"
        )
        transcript_path = transcripts_root or (Path(transcript_env) if transcript_env else default_transcripts)
        forge_root = root / "forge"
        return cls(
            repo_root=root,
            forge_root=forge_root,
            artifacts_root=root / "artifacts",
            transcripts_root=transcript_path.expanduser().resolve(),
            plans_root=root / ".cursor" / "plans",
            editable_surface_path=forge_root / "editable_surface.yaml",
            ledger_path=forge_root / "ledger.jsonl",
        )


@dataclass(frozen=True)
class EditableSurfaceEntry:
    component: str
    glob: str
    semantic_description: str


@dataclass(frozen=True)
class ValidationPolicy:
    command_timeout_seconds: int
    static_commands: tuple[tuple[str, ...], ...]
    held_in_commands: tuple[tuple[str, ...], ...]
    held_out_commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ForgeConfig:
    paths: ForgePaths
    schema_version: str
    optimization_stage: str
    editable: tuple[EditableSurfaceEntry, ...]
    read_only: tuple[str, ...]
    minimum_surface_similarity: float
    cluster_similarity: float
    proposal_duplicate_similarity: float
    validation: ValidationPolicy

    @classmethod
    def load(cls, paths: ForgePaths) -> ForgeConfig:
        try:
            raw = yaml.safe_load(paths.editable_surface_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ForgeConfigError(f"Missing editable-surface policy: {paths.editable_surface_path}") from exc
        except yaml.YAMLError as exc:
            raise ForgeConfigError(f"Invalid YAML in {paths.editable_surface_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ForgeConfigError("editable_surface.yaml must contain a mapping")
        if raw.get("schema_version") != "forge-editable-surface.v1":
            raise ForgeConfigError("Unsupported editable-surface schema_version")

        editable_raw = _require_list(raw, "editable")
        editable_entries: list[EditableSurfaceEntry] = []
        for index, item in enumerate(editable_raw):
            if not isinstance(item, dict):
                raise ForgeConfigError(f"editable[{index}] must be a mapping")
            editable_entries.append(
                EditableSurfaceEntry(
                    component=_require_nonempty_string(item, "component", f"editable[{index}]"),
                    glob=_require_safe_glob(item, "glob", f"editable[{index}]"),
                    semantic_description=_require_nonempty_string(
                        item, "semantic_description", f"editable[{index}]"
                    ),
                )
            )

        read_only_raw = _require_list(raw, "read_only")
        read_only = tuple(
            _validate_glob(value, f"read_only[{index}]") for index, value in enumerate(read_only_raw)
        )
        mapping = _require_mapping(raw, "semantic_mapping")
        validation_raw = _require_mapping(raw, "validation")
        timeout = validation_raw.get("command_timeout_seconds")
        if not isinstance(timeout, int) or timeout <= 0:
            raise ForgeConfigError("validation.command_timeout_seconds must be a positive integer")
        validation = ValidationPolicy(
            command_timeout_seconds=timeout,
            static_commands=_parse_commands(validation_raw, "static"),
            held_in_commands=_parse_commands(validation_raw, "held_in"),
            held_out_commands=_parse_commands(validation_raw, "held_out"),
        )
        config = cls(
            paths=paths,
            schema_version="forge-editable-surface.v1",
            optimization_stage=_require_nonempty_string(raw, "optimization_stage", "root"),
            editable=tuple(editable_entries),
            read_only=read_only,
            minimum_surface_similarity=_bounded_float(mapping, "minimum_similarity"),
            cluster_similarity=_bounded_float(mapping, "cluster_similarity"),
            proposal_duplicate_similarity=_bounded_float(mapping, "proposal_duplicate_similarity"),
            validation=validation,
        )
        config._assert_policy_disjoint()
        return config

    def normalize_relative_path(self, value: str | Path) -> str:
        raw = value.as_posix() if isinstance(value, Path) else value.replace("\\", "/")
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ForgeConfigError(f"Unsafe repository-relative path: {value!r}")
        resolved = (self.paths.repo_root / Path(*path.parts)).resolve(strict=False)
        if not resolved.is_relative_to(self.paths.repo_root):
            raise ForgeConfigError(f"Path escapes repository root: {value!r}")
        return path.as_posix()

    def is_read_only(self, relative_path: str | Path) -> bool:
        normalized = self.normalize_relative_path(relative_path)
        path = PurePosixPath(normalized)
        return any(_glob_matches(path, pattern) for pattern in self.read_only)

    def editable_entry_for(self, relative_path: str | Path) -> EditableSurfaceEntry | None:
        normalized = self.normalize_relative_path(relative_path)
        if self.is_read_only(normalized):
            return None
        path = PurePosixPath(normalized)
        return next((entry for entry in self.editable if path.match(entry.glob)), None)

    def resolve_target(self, relative_path: str | Path, *, must_exist: bool) -> Path:
        normalized = self.normalize_relative_path(relative_path)
        target = self.paths.repo_root / normalized
        resolved = target.resolve(strict=must_exist)
        if not resolved.is_relative_to(self.paths.repo_root):
            raise ForgeConfigError(f"Resolved target escapes repository root: {relative_path!r}")
        return resolved

    def editable_assets(self) -> tuple[tuple[EditableSurfaceEntry, str, Path], ...]:
        assets: list[tuple[EditableSurfaceEntry, str, Path]] = []
        for entry in self.editable:
            for path in sorted(self.paths.repo_root.glob(entry.glob)):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.paths.repo_root).as_posix()
                if self.editable_entry_for(relative) is not None:
                    assets.append((entry, relative, path))
        return tuple(assets)

    def _assert_policy_disjoint(self) -> None:
        protected_governance = (
            "forge/src/volvence_forge/config.py",
            "forge/editable_surface.yaml",
            "forge/ledger.jsonl",
            "tests/contracts/test_forge_boundaries.py",
            "docs/specs/rsi-forge.md",
        )
        for path in protected_governance:
            if not self.is_read_only(path):
                raise ForgeConfigError(f"Governance path must remain read-only: {path}")
            if self.editable_entry_for(path) is not None:
                raise ForgeConfigError(f"Read-only governance path also matched editable surface: {path}")


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ForgeConfigError(f"{key} must be a mapping")
    return value


def _require_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ForgeConfigError(f"{key} must be a list")
    return value


def _require_nonempty_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ForgeConfigError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _validate_glob(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForgeConfigError(f"{context} must be a non-empty glob string")
    glob = value.strip().replace("\\", "/")
    path = PurePosixPath(glob)
    if path.is_absolute() or ".." in path.parts:
        raise ForgeConfigError(f"{context} must be repository-relative: {glob!r}")
    return glob


def _require_safe_glob(raw: dict[str, Any], key: str, context: str) -> str:
    return _validate_glob(raw.get(key), f"{context}.{key}")


def _bounded_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ForgeConfigError(f"semantic_mapping.{key} must be numeric")
    numeric = float(value)
    if not -1.0 <= numeric <= 1.0:
        raise ForgeConfigError(f"semantic_mapping.{key} must be in [-1, 1]")
    return numeric


def _parse_commands(raw: dict[str, Any], key: str) -> tuple[tuple[str, ...], ...]:
    commands = raw.get(key)
    if not isinstance(commands, list) or not commands:
        raise ForgeConfigError(f"validation.{key} must be a non-empty command list")
    parsed: list[tuple[str, ...]] = []
    for index, command in enumerate(commands):
        if not isinstance(command, list) or not command:
            raise ForgeConfigError(f"validation.{key}[{index}] must be a non-empty argv list")
        if not all(isinstance(part, str) and part for part in command):
            raise ForgeConfigError(f"validation.{key}[{index}] contains an invalid argv item")
        parsed.append(tuple(command))
    return tuple(parsed)


def _glob_matches(path: PurePosixPath, pattern: str) -> bool:
    """Match repository globs with explicit recursive ``/**`` semantics."""

    if pattern.endswith("/**"):
        prefix = PurePosixPath(pattern[:-3].rstrip("/"))
        return path == prefix or prefix in path.parents
    return path.match(pattern)
