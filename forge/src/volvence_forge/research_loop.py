"""Bounded orchestration for demand-driven discovery and approved research runs."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ForgeConfig
from .foundation import ForgeError, sha256_bytes
from .research_control import (
    PraxistCommandRunner,
    ResearchControlStatus,
    list_research_inbox,
    reconcile_research_control,
    validate_research_request,
)
from .research_discovery import (
    DISCOVERY_LOOP_OWNER,
    ResearchDiscoveryBackend,
    discover_research_topics,
    submit_bound_topic_for_a0,
    validate_research_demand,
    validate_research_demand_binding,
)

LOOP_RESULT_VERSION = "forge-demand-research-loop-result.v1"

_DISCOVERY_ROOT = "research_discovery"
_RECONCILABLE_STATES = frozenset(
    {"APPROVED", "PREFLIGHT_RESOLVED", "STARTING", "RUNNING"}
)
_TERMINAL_STATES = frozenset(
    {"REJECTED", "SUPERSEDED", "BLOCKED", "RUN_COMPLETED", "RUN_FAILED"}
)


class ResearchLoopError(ForgeError):
    """Raised when one automatic loop pass cannot preserve exact boundaries."""


@dataclass(frozen=True, slots=True)
class ResearchLoopDiscovery:
    demand_id: str
    demand_path: Path
    run_id: str
    run_path: Path
    proposal_count: int
    reused: bool

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "demand_id": self.demand_id,
            "demand": str(self.demand_path),
            "run_id": self.run_id,
            "run": str(self.run_path),
            "proposal_count": self.proposal_count,
            "reused": self.reused,
        }


@dataclass(frozen=True, slots=True)
class ResearchLoopSubmission:
    binding_id: str
    binding_path: Path
    request_id: str
    request_path: Path
    reused: bool

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "binding": str(self.binding_path),
            "request_id": self.request_id,
            "request": str(self.request_path),
            "reused": self.reused,
            "state": "AWAITING_RESEARCH_APPROVAL",
        }


@dataclass(frozen=True, slots=True)
class ResearchLoopReconciliation:
    request_id: str
    request_path: Path
    state_before: str
    state_after: str
    run_id: str | None
    run_dir: str | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request": str(self.request_path),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
        }


@dataclass(frozen=True, slots=True)
class ResearchLoopResult:
    demand_root: Path
    backend: str
    model: str
    max_demands: int
    max_new_discoveries: int
    max_new_requests: int
    max_reconciles: int
    demand_count: int
    open_demand_count: int
    binding_count: int
    approved_binding_count: int
    awaiting_a0_count: int
    terminal_request_count: int
    discoveries: tuple[ResearchLoopDiscovery, ...]
    submissions: tuple[ResearchLoopSubmission, ...]
    reconciliations: tuple[ResearchLoopReconciliation, ...]

    @property
    def new_discovery_count(self) -> int:
        return sum(not item.reused for item in self.discoveries)

    @property
    def new_request_count(self) -> int:
        return sum(not item.reused for item in self.submissions)

    @property
    def blocked_count(self) -> int:
        return sum(
            item.state_after in {"BLOCKED", "RUN_FAILED"}
            for item in self.reconciliations
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": LOOP_RESULT_VERSION,
            "mode": "bounded_once",
            "demand_root": str(self.demand_root),
            "execution": {
                "backend": self.backend,
                "model": self.model,
                "limits": {
                    "max_demands": self.max_demands,
                    "max_new_discoveries": self.max_new_discoveries,
                    "max_new_requests": self.max_new_requests,
                    "max_reconciles": self.max_reconciles,
                },
            },
            "summary": {
                "demand_count": self.demand_count,
                "open_demand_count": self.open_demand_count,
                "binding_count": self.binding_count,
                "approved_binding_count": self.approved_binding_count,
                "new_discovery_count": self.new_discovery_count,
                "new_request_count": self.new_request_count,
                "awaiting_a0_count": self.awaiting_a0_count,
                "reconciled_count": len(self.reconciliations),
                "terminal_request_count": self.terminal_request_count,
                "blocked_count": self.blocked_count,
            },
            "discoveries": [item.to_jsonable() for item in self.discoveries],
            "submissions": [item.to_jsonable() for item in self.submissions],
            "reconciliations": [
                item.to_jsonable() for item in self.reconciliations
            ],
            "authority": {
                "human_topic_binding_required": True,
                "human_a0_required": True,
                "automatic_a0_authorized": False,
                "automatic_candidate_import_authorized": False,
                "production_promotion_authorized": False,
                "runtime_wiring_changed": False,
                "evaluation_is_learning_source": False,
            },
        }


def run_demand_research_loop_once(
    *,
    config: ForgeConfig,
    backend: ResearchDiscoveryBackend,
    demand_root: Path | None = None,
    max_demands: int = 128,
    max_new_discoveries: int = 1,
    max_new_requests: int = 8,
    max_reconciles: int = 8,
    runner: PraxistCommandRunner | None = None,
    allowed_demand_ids: frozenset[str] | None = None,
    blocked_demand_ids: frozenset[str] | None = None,
    preferred_demand_ids: tuple[str, ...] = (),
) -> ResearchLoopResult:
    """Run one serialized discovery, submission, and approved reconcile pass."""

    _validate_limit("max_demands", max_demands, minimum=1, maximum=1024)
    _validate_limit(
        "max_new_discoveries", max_new_discoveries, minimum=0, maximum=32
    )
    _validate_limit("max_new_requests", max_new_requests, minimum=0, maximum=128)
    _validate_limit("max_reconciles", max_reconciles, minimum=0, maximum=128)
    if backend.backend_name not in {"codex_sdk", "replay"}:
        raise ResearchLoopError(
            f"unsupported research discovery backend: {backend.backend_name!r}"
        )
    allowed = allowed_demand_ids or frozenset()
    blocked = blocked_demand_ids or frozenset()
    if allowed_demand_ids is not None and allowed & blocked:
        raise ResearchLoopError(
            "allowed and blocked Demand sets must not overlap"
        )
    if len(preferred_demand_ids) != len(set(preferred_demand_ids)):
        raise ResearchLoopError("preferred Demand identities must be unique")

    selected_root = _resolve_demand_root(config, demand_root)
    lock_path = (
        config.paths.artifacts_root / _DISCOVERY_ROOT / ".loop.lock"
    )
    with _exclusive_lock(lock_path):
        all_demand_records = _load_demands(
            config=config,
            demand_root=selected_root,
            max_demands=max_demands,
        )
        known_demand_ids = {
            str(demand["demand_id"]) for demand, _ in all_demand_records
        }
        constrained_ids = allowed | blocked | set(preferred_demand_ids)
        unknown = constrained_ids - known_demand_ids
        if unknown:
            raise ResearchLoopError(
                "Demand scheduling set contains identities outside the validated "
                f"Demand root: {sorted(unknown)}"
            )
        demand_records = tuple(
            (demand, path)
            for demand, path in all_demand_records
            if (
                (allowed_demand_ids is None or demand["demand_id"] in allowed)
                and demand["demand_id"] not in blocked
            )
        )
        selected_demand_ids = {
            str(demand["demand_id"]) for demand, _ in demand_records
        }
        preferred_not_selected = set(preferred_demand_ids) - selected_demand_ids
        if preferred_not_selected:
            raise ResearchLoopError(
                "preferred Demand set contains blocked or disallowed identities: "
                f"{sorted(preferred_not_selected)}"
            )
        preferred_rank = {
            demand_id: index for index, demand_id in enumerate(preferred_demand_ids)
        }
        demand_records = tuple(
            sorted(
                demand_records,
                key=lambda item: (
                    preferred_rank.get(
                        str(item[0]["demand_id"]), len(preferred_rank)
                    ),
                    str(item[1]),
                ),
            )
        )
        ignored_demand_refs = frozenset(
            (
                path.relative_to(config.paths.repo_root).as_posix(),
                sha256_bytes(path.read_bytes()),
            )
            for demand, path in all_demand_records
            if str(demand["demand_id"]) not in selected_demand_ids
        )
        binding_records = _load_bindings(config=config)
        binding_records = tuple(
            (binding, path)
            for binding, path in binding_records
            if binding["demand"]["artifact_id"] in selected_demand_ids
        )
        binding_records = tuple(
            sorted(
                binding_records,
                key=lambda item: (
                    preferred_rank.get(
                        str(item[0]["demand"]["artifact_id"]),
                        len(preferred_rank),
                    ),
                    str(item[1]),
                ),
            )
        )

        discoveries: list[ResearchLoopDiscovery] = []
        new_discoveries = 0
        for demand, path in demand_records:
            if demand["status"] != "OPEN":
                continue
            if new_discoveries >= max_new_discoveries:
                break
            result = discover_research_topics(
                config=config,
                demand_path=path,
                backend=backend,
            )
            discoveries.append(
                ResearchLoopDiscovery(
                    demand_id=str(demand["demand_id"]),
                    demand_path=path,
                    run_id=result.run_id,
                    run_path=result.run_path,
                    proposal_count=len(result.proposal_paths),
                    reused=result.reused,
                )
            )
            if not result.reused:
                new_discoveries += 1

        submissions: list[ResearchLoopSubmission] = []
        new_requests = 0
        for binding, path in binding_records:
            if binding["decision"] != "APPROVE":
                continue
            if new_requests >= max_new_requests:
                break
            submitted = submit_bound_topic_for_a0(
                config=config,
                binding_path=path,
            )
            submissions.append(
                ResearchLoopSubmission(
                    binding_id=str(binding["binding_id"]),
                    binding_path=path,
                    request_id=submitted.request_id,
                    request_path=submitted.request_path,
                    reused=submitted.reused,
                )
            )
            if not submitted.reused:
                new_requests += 1

        approved_lineages = _approved_binding_lineages(
            config=config,
            binding_records=binding_records,
        )
        owned_statuses = _discovery_owned_statuses(
            config=config,
            approved_lineages=approved_lineages,
            ignored_demand_refs=ignored_demand_refs,
        )
        awaiting_a0_count = sum(
            status.state == "AWAITING_RESEARCH_APPROVAL"
            for status in owned_statuses
        )
        terminal_request_count = sum(
            status.state in _TERMINAL_STATES for status in owned_statuses
        )
        reconciliations: list[ResearchLoopReconciliation] = []
        for status in owned_statuses:
            if len(reconciliations) >= max_reconciles:
                break
            if status.state not in _RECONCILABLE_STATES:
                continue
            reconciled = reconcile_research_control(
                config=config,
                request_path=status.request_path,
                runner=runner,
            )
            if len(reconciled) != 1:
                raise ResearchLoopError(
                    "targeted research reconcile returned an ambiguous status set"
                )
            current = reconciled[0]
            if current.request_id != status.request_id:
                raise ResearchLoopError(
                    "targeted research reconcile changed the exact Request identity"
                )
            reconciliations.append(
                ResearchLoopReconciliation(
                    request_id=current.request_id,
                    request_path=current.request_path,
                    state_before=status.state,
                    state_after=current.state,
                    run_id=current.run_id,
                    run_dir=current.run_dir,
                )
            )

        return ResearchLoopResult(
            demand_root=selected_root,
            backend=backend.backend_name,
            model=backend.model_name,
            max_demands=max_demands,
            max_new_discoveries=max_new_discoveries,
            max_new_requests=max_new_requests,
            max_reconciles=max_reconciles,
            demand_count=len(demand_records),
            open_demand_count=sum(
                demand["status"] == "OPEN" for demand, _ in demand_records
            ),
            binding_count=len(binding_records),
            approved_binding_count=sum(
                binding["decision"] == "APPROVE"
                for binding, _ in binding_records
            ),
            awaiting_a0_count=awaiting_a0_count,
            terminal_request_count=terminal_request_count,
            discoveries=tuple(discoveries),
            submissions=tuple(submissions),
            reconciliations=tuple(reconciliations),
        )


def _resolve_demand_root(
    config: ForgeConfig,
    demand_root: Path | None,
) -> Path:
    raw = demand_root or (config.paths.repo_root / "research" / "demands")
    candidate = raw if raw.is_absolute() else config.paths.repo_root / raw
    if candidate.is_symlink():
        raise ResearchLoopError(f"ResearchDemand root may not be a symlink: {candidate}")
    resolved = candidate.expanduser().resolve(strict=False)
    if not resolved.is_relative_to(config.paths.repo_root):
        raise ResearchLoopError("ResearchDemand root must remain inside the repository")
    if resolved.exists() and not resolved.is_dir():
        raise ResearchLoopError(f"ResearchDemand root must be a directory: {resolved}")
    return resolved


def _load_demands(
    *,
    config: ForgeConfig,
    demand_root: Path,
    max_demands: int,
) -> tuple[tuple[dict[str, Any], Path], ...]:
    if not demand_root.exists():
        return ()
    entries = tuple(sorted(demand_root.rglob("*")))
    symlinks = tuple(path for path in entries if path.is_symlink())
    if symlinks:
        raise ResearchLoopError(
            f"ResearchDemand root contains a symlink: {symlinks[0]}"
        )
    paths = tuple(
        path.resolve(strict=True)
        for path in entries
        if path.suffix == ".json" and path.is_file()
    )
    if len(paths) > max_demands:
        raise ResearchLoopError(
            f"ResearchDemand root contains {len(paths)} files above max_demands={max_demands}"
        )
    records: list[tuple[dict[str, Any], Path]] = []
    identities: set[str] = set()
    for path in paths:
        demand = validate_research_demand(config=config, demand_path=path)
        demand_id = str(demand["demand_id"])
        if demand_id in identities:
            raise ResearchLoopError(
                f"duplicate ResearchDemand identity in demand root: {demand_id}"
            )
        identities.add(demand_id)
        records.append((demand, path))
    return tuple(records)


def _load_bindings(
    *,
    config: ForgeConfig,
) -> tuple[tuple[dict[str, Any], Path], ...]:
    root = config.paths.artifacts_root / _DISCOVERY_ROOT
    if not root.exists():
        return ()
    paths = tuple(
        path
        for path in sorted(root.rglob("*.json"))
        if "bindings" in path.relative_to(root).parts
    )
    records: list[tuple[dict[str, Any], Path]] = []
    identities: set[str] = set()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ResearchLoopError(
                f"ResearchDemand Binding must be a regular non-symlink file: {path}"
            )
        binding = validate_research_demand_binding(
            config=config,
            binding_path=path,
        )
        binding_id = str(binding["binding_id"])
        if binding_id in identities:
            raise ResearchLoopError(
                f"duplicate ResearchDemand Binding identity: {binding_id}"
            )
        identities.add(binding_id)
        records.append((binding, path.resolve(strict=True)))
    return tuple(records)


def _discovery_owned_statuses(
    *,
    config: ForgeConfig,
    approved_lineages: dict[
        tuple[tuple[str, str], ...],
        tuple[str, str],
    ],
    ignored_demand_refs: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[ResearchControlStatus, ...]:
    statuses = []
    for status in list_research_inbox(config=config):
        request = validate_research_request(
            config=config,
            request_path=status.request_path,
            verify_bindings=False,
        )
        if request["schema_version"] != "forge-research-request.v1":
            continue
        trigger = request["trigger"]
        if (
            trigger["kind"] != "typed_signal"
            or trigger["submitted_by"] != DISCOVERY_LOOP_OWNER
        ):
            continue
        evidence_lineage = tuple(
            (str(ref["locator"]), str(ref["sha256"]))
            for ref in trigger["evidence"]
        )
        if any(ref in ignored_demand_refs for ref in evidence_lineage):
            continue
        expected = approved_lineages.get(evidence_lineage)
        if expected is None:
            raise ResearchLoopError(
                "discovery-owned Request does not preserve the exact approved "
                "Demand -> TopicProposal -> DemandBinding lineage"
            )
        expected_task_id, expected_rationale = expected
        if request["task_id"] != expected_task_id:
            raise ResearchLoopError(
                "discovery-owned Request task_id does not match its approved DemandBinding"
            )
        if trigger["rationale"] != expected_rationale:
            raise ResearchLoopError(
                "discovery-owned Request rationale does not match its approved DemandBinding"
            )
        statuses.append(status)
    return tuple(sorted(statuses, key=lambda item: str(item.request_path)))


def _approved_binding_lineages(
    *,
    config: ForgeConfig,
    binding_records: tuple[tuple[dict[str, Any], Path], ...],
) -> dict[tuple[tuple[str, str], ...], tuple[str, str]]:
    lineages: dict[tuple[tuple[str, str], ...], tuple[str, str]] = {}
    for binding, path in binding_records:
        if binding["decision"] != "APPROVE":
            continue
        lineage = (
            _stored_content_ref_key(binding["demand"]["artifact"]),
            _stored_content_ref_key(binding["proposal"]["artifact"]),
            _content_ref_key(config, path),
        )
        if lineage in lineages:
            raise ResearchLoopError(
                "multiple approved DemandBindings resolve to the same exact lineage"
            )
        rationale = (
            f"Approved DemandBinding {binding['binding_id']} mapped by "
            f"{binding['mapping']['mapping_id']}."
        )
        lineages[lineage] = (str(binding["mapping"]["task_id"]), rationale)
    return lineages


def _stored_content_ref_key(value: dict[str, Any]) -> tuple[str, str]:
    return str(value["locator"]), str(value["sha256"])


def _content_ref_key(config: ForgeConfig, path: Path) -> tuple[str, str]:
    try:
        resolved = path.resolve(strict=True)
        content = resolved.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ResearchLoopError(f"cannot read approved DemandBinding {path}: {exc}") from exc
    locator = (
        resolved.relative_to(config.paths.repo_root).as_posix()
        if resolved.is_relative_to(config.paths.repo_root)
        else str(resolved)
    )
    return locator, sha256_bytes(content)


def _validate_limit(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ResearchLoopError(
            f"{name} must be an integer from {minimum} to {maximum}"
        )


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Forge targets POSIX hosts.
        raise ResearchLoopError("research loop requires POSIX file locking") from exc
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ResearchLoopError(f"cannot open research loop lock {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
