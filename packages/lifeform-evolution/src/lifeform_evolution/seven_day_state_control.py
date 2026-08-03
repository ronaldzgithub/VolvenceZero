"""Filesystem state interventions for seven-day evidence arms.

The controller operates only under one explicitly supplied evidence root.  It
never deletes owner state: the just-finished scope directory is atomically
renamed into an immutable day archive, then the preregistered source snapshot
is copied into the active scope for the next service instance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

from volvence_zero.memory import scoped_memory_dir
from volvence_zero.seven_day_evidence_contract import (
    SEVEN_DAY_SHUFFLED_SOURCE_DAYS,
)


_MEASUREMENT_CHECKPOINT_NAME = "evaluation__relationship_continuity_v1.json"
_STATE_POLICY_BY_ARM = {
    "correct-user-state": "correct-user-state",
    "stateless": "stateless",
    "swapped-user-state": "swapped-user-state",
    "shuffled-history": "shuffled-history",
    "sleep-consolidation": "correct-user-state",
    "no-sleep": "correct-user-state",
}
_STATE_LOADING_POLICIES = frozenset(_STATE_POLICY_BY_ARM.values())


def _require_descendant(*, root: Path, path: Path, field: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved == resolved_root:
        raise ValueError(f"{field} may not equal the evidence root")
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} must be inside the evidence root") from exc
    return resolved


def _directory_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"state directory does not exist: {root}")
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != _MEASUREMENT_CHECKPOINT_NAME
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class StateInterventionEvidence:
    experiment_arm_label: str
    state_loading_policy: str
    after_day_index: int
    archived_state_ref: str
    archived_state_sha256: str
    measurement_checkpoint_sha256: str
    next_day_source_arm: str | None
    next_day_source_day_index: int | None
    next_day_loaded_state_sha256: str | None

    def __post_init__(self) -> None:
        expected = _STATE_POLICY_BY_ARM.get(self.experiment_arm_label)
        if expected is not None and self.state_loading_policy != expected:
            raise ValueError("state intervention arm/policy mismatch")
        if self.state_loading_policy not in _STATE_LOADING_POLICIES:
            raise ValueError("unsupported state loading policy")
        if self.after_day_index < 1 or self.after_day_index > 6:
            raise ValueError("state intervention day must be in [1, 6]")
        if len(self.archived_state_sha256) != 64:
            raise ValueError("archived state digest must be SHA-256")
        if len(self.measurement_checkpoint_sha256) != 64:
            raise ValueError("measurement checkpoint digest must be SHA-256")
        if self.state_loading_policy == "stateless":
            if any(
                value is not None
                for value in (
                    self.next_day_source_arm,
                    self.next_day_source_day_index,
                    self.next_day_loaded_state_sha256,
                )
            ):
                raise ValueError("stateless arm may not stage prior state")
        else:
            if self.next_day_source_arm is None:
                raise ValueError("stateful arm must identify source arm")
            if self.next_day_source_day_index is None:
                raise ValueError("stateful arm must identify source day")
            if self.next_day_source_day_index > self.after_day_index:
                raise ValueError("state intervention may not load future state")
            if (
                self.next_day_loaded_state_sha256 is None
                or len(self.next_day_loaded_state_sha256) != 64
            ):
                raise ValueError("loaded state digest must be SHA-256")


class SevenDayFilesystemStateController:
    """Archive and stage owner scopes for one formal run arm."""

    def __init__(
        self,
        *,
        evidence_root: str | Path,
        active_scope_root: str | Path,
        archive_root: str | Path,
        user_id: str,
        experiment_arm_label: str,
        state_loading_policy: str | None = None,
        correct_reference_archive_root: str | Path | None = None,
        donor_archive_root: str | Path | None = None,
    ) -> None:
        if not user_id.strip():
            raise ValueError("state controller user_id must be non-empty")
        policy = (
            state_loading_policy
            if state_loading_policy is not None
            else _STATE_POLICY_BY_ARM.get(experiment_arm_label)
        )
        if policy is None:
            raise ValueError("unsupported seven-day experiment arm")
        if policy not in _STATE_LOADING_POLICIES:
            raise ValueError("unsupported state loading policy")
        expected_policy = _STATE_POLICY_BY_ARM.get(experiment_arm_label)
        if expected_policy is not None and policy != expected_policy:
            raise ValueError("state intervention arm/policy mismatch")
        root = Path(evidence_root).resolve()
        self._evidence_root = root
        self._active_root = _require_descendant(
            root=root,
            path=Path(active_scope_root),
            field="active_scope_root",
        )
        self._archive_root = _require_descendant(
            root=root,
            path=Path(archive_root),
            field="archive_root",
        )
        self._reference_root = (
            _require_descendant(
                root=root,
                path=Path(correct_reference_archive_root),
                field="correct_reference_archive_root",
            )
            if correct_reference_archive_root is not None
            else None
        )
        self._donor_root = (
            _require_descendant(
                root=root,
                path=Path(donor_archive_root),
                field="donor_archive_root",
            )
            if donor_archive_root is not None
            else None
        )
        if policy == "shuffled-history" and self._reference_root is None:
            raise ValueError("shuffled-history requires correct reference archives")
        if policy == "swapped-user-state" and self._donor_root is None:
            raise ValueError("swapped-user-state requires donor archives")
        self._user_id = user_id
        self._experiment_arm_label = experiment_arm_label
        self._policy = policy

    @property
    def active_scope_dir(self) -> Path:
        return scoped_memory_dir(
            root_dir=self._active_root,
            user_id=self._user_id,
        )

    def prepare_initial_day(self) -> None:
        active = self.active_scope_dir
        if active.exists():
            raise FileExistsError(
                f"formal run active scope must start absent: {active}"
            )
        self._active_root.mkdir(parents=True, exist_ok=True)
        self._archive_root.mkdir(parents=True, exist_ok=True)

    def archive_and_stage_after_day(
        self, *, day_index: int
    ) -> StateInterventionEvidence:
        if day_index < 1 or day_index > 6:
            raise ValueError("day_index must be in [1, 6]")
        active = self.active_scope_dir
        if not active.is_dir():
            raise FileNotFoundError(
                "owner persistence scope was not created before restart"
            )
        archive = self._archive_root / f"day-{day_index}"
        if archive.exists():
            raise FileExistsError(f"state archive is immutable: {archive}")
        measurement_path = active / _MEASUREMENT_CHECKPOINT_NAME
        if not measurement_path.is_file():
            raise FileNotFoundError(
                "relationship continuity measurement checkpoint is missing"
            )
        measurement_bytes = measurement_path.read_bytes()
        measurement_sha = hashlib.sha256(measurement_bytes).hexdigest()
        archive.parent.mkdir(parents=True, exist_ok=True)
        active.rename(archive)
        archived_sha = _directory_sha256(archive)
        source_arm, source_day, source = self._source_for_next_day(
            after_day_index=day_index,
            just_archived=archive,
        )
        loaded_sha = None
        if source is not None:
            shutil.copytree(source, active)
            loaded_sha = _directory_sha256(active)
            if loaded_sha != _directory_sha256(source):
                raise RuntimeError("staged owner state digest drift")
        else:
            active.mkdir(parents=True)
        staged_measurement_path = active / _MEASUREMENT_CHECKPOINT_NAME
        staged_measurement_path.write_bytes(measurement_bytes)
        if hashlib.sha256(staged_measurement_path.read_bytes()).hexdigest() != (
            measurement_sha
        ):
            raise RuntimeError("measurement checkpoint changed during state staging")
        return StateInterventionEvidence(
            experiment_arm_label=self._experiment_arm_label,
            state_loading_policy=self._policy,
            after_day_index=day_index,
            archived_state_ref=str(archive.relative_to(self._evidence_root)),
            archived_state_sha256=archived_sha,
            measurement_checkpoint_sha256=measurement_sha,
            next_day_source_arm=source_arm,
            next_day_source_day_index=source_day,
            next_day_loaded_state_sha256=loaded_sha,
        )

    def _source_for_next_day(
        self,
        *,
        after_day_index: int,
        just_archived: Path,
    ) -> tuple[str | None, int | None, Path | None]:
        if self._policy == "stateless":
            return None, None, None
        if self._policy == "correct-user-state":
            return "correct-user-state", after_day_index, just_archived
        if self._policy == "swapped-user-state":
            assert self._donor_root is not None
            source = self._donor_root / f"day-{after_day_index}"
            if not source.is_dir():
                raise FileNotFoundError(f"donor state archive is missing: {source}")
            return "matched-donor-correct-user-state", after_day_index, source
        assert self._policy == "shuffled-history"
        assert self._reference_root is not None
        source_day = SEVEN_DAY_SHUFFLED_SOURCE_DAYS[after_day_index - 1]
        source = self._reference_root / f"day-{source_day}"
        if not source.is_dir():
            raise FileNotFoundError(
                f"correct reference state archive is missing: {source}"
            )
        return "same-user-correct-reference", source_day, source


__all__ = [
    "SEVEN_DAY_SHUFFLED_SOURCE_DAYS",
    "SevenDayFilesystemStateController",
    "StateInterventionEvidence",
]
