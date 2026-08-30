"""Content-addressed portfolio scheduling for Forge research lifecycles.

The portfolio decides which exact Demand lineages are eligible for the existing
Research Loop.  It never owns Praxist processes, human gates, scientific
validation, ModificationGate decisions, or production wiring.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    SchemaStore,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
    utc_now,
)
from .research_control import (
    PraxistCommandRunner,
    ResearchControlStatus,
    list_research_inbox,
    validate_research_request,
)
from .research_discovery import (
    ResearchDiscoveryBackend,
    resolve_registered_task_binding,
    validate_research_demand,
)
from .research_loop import ResearchLoopResult, run_demand_research_loop_once

SCHEMA_NAME = "research_portfolio.schema.json"
PORTFOLIO_VERSION = "forge-research-portfolio.v1"
OUTCOME_VERSION = "forge-research-study-outcome.v1"

_TERMINAL_REQUEST_STATES = frozenset(
    {
        "REJECTED",
        "SUPERSEDED",
        "BLOCKED",
        "RUN_COMPLETED",
        "RUN_FAILED",
        "CANCELLED",
    }
)
_RUNNING_REQUEST_STATES = frozenset(
    {"PREFLIGHT_RESOLVED", "STARTING", "RUNNING", "STOPPING", "RESUMING"}
)
_LOOP_ELIGIBLE_STATES = frozenset(
    {
        "REGISTERED",
        "AWAITING_RESEARCH_APPROVAL",
        "APPROVED",
        "WAITING_FOR_CAPACITY",
        "PREFLIGHT_RESOLVED",
        "STARTING",
        "RUNNING",
        "STOPPING",
        "PAUSED",
        "RESUMING",
    }
)


class ResearchPortfolioError(ForgeError):
    """Raised when portfolio identity, ordering, or authority is unsafe."""


@dataclass(frozen=True, slots=True)
class ResearchPortfolioSealResult:
    portfolio_id: str
    portfolio_path: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class ResearchPortfolioStudyStatus:
    study_id: str
    state: str
    priority: int
    concurrency_lane: str
    dependency_states: tuple[tuple[str, str], ...]
    request_id: str | None
    request_path: Path | None
    run_id: str | None
    run_dir: str | None
    outcome_decision: str | None
    outcome_path: Path | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "state": self.state,
            "priority": self.priority,
            "concurrency_lane": self.concurrency_lane,
            "dependencies": [
                {"study_id": study_id, "state": state}
                for study_id, state in self.dependency_states
            ],
            "request_id": self.request_id,
            "request": str(self.request_path) if self.request_path else None,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "outcome_decision": self.outcome_decision,
            "outcome": str(self.outcome_path) if self.outcome_path else None,
        }


@dataclass(frozen=True, slots=True)
class ResearchPortfolioStatus:
    portfolio_id: str
    portfolio_path: Path
    studies: tuple[ResearchPortfolioStudyStatus, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": "forge-research-portfolio-status.v1",
            "portfolio_id": self.portfolio_id,
            "portfolio": str(self.portfolio_path),
            "studies": [study.to_jsonable() for study in self.studies],
            "summary": {
                "study_count": len(self.studies),
                "running_count": sum(
                    study.state in _RUNNING_REQUEST_STATES for study in self.studies
                ),
                "awaiting_a0_count": sum(
                    study.state == "AWAITING_RESEARCH_APPROVAL"
                    for study in self.studies
                ),
                "needs_task_design_count": sum(
                    study.state == "NEEDS_TASK_DESIGN" for study in self.studies
                ),
                "accepted_count": sum(
                    study.state == "COMPLETED_ACCEPTED" for study in self.studies
                ),
            },
            "authority": {
                "portfolio_scheduling_only": True,
                "automatic_human_gates_authorized": False,
                "production_promotion_authorized": False,
                "runtime_wiring_changed": False,
                "evaluation_is_learning_source": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ResearchStudyOutcomeResult:
    outcome_id: str
    outcome_path: Path
    decision: str


@dataclass(frozen=True, slots=True)
class ResearchPortfolioLoopResult:
    before: ResearchPortfolioStatus
    loop: ResearchLoopResult
    after: ResearchPortfolioStatus
    eligible_study_ids: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": "forge-research-portfolio-loop-result.v1",
            "mode": "bounded_once",
            "portfolio_id": self.after.portfolio_id,
            "portfolio": str(self.after.portfolio_path),
            "eligible_study_ids": list(self.eligible_study_ids),
            "loop": self.loop.to_jsonable(),
            "status": self.after.to_jsonable(),
            "authority": {
                "portfolio_scheduling_only": True,
                "automatic_human_gates_authorized": False,
                "automatic_candidate_import_authorized": False,
                "production_promotion_authorized": False,
                "runtime_wiring_changed": False,
                "evaluation_is_learning_source": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ResearchPortfolioMembership:
    portfolio_id: str
    portfolio_path: Path
    study_id: str
    demand_id: str


@dataclass(frozen=True, slots=True)
class ResearchManagedStudy:
    portfolio_id: str
    portfolio_path: Path
    study_id: str
    demand_id: str
    state: str

    def to_jsonable(self) -> dict[str, str]:
        return {
            "portfolio_id": self.portfolio_id,
            "portfolio": str(self.portfolio_path),
            "study_id": self.study_id,
            "demand_id": self.demand_id,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class ResearchManagedLoopResult:
    portfolio_root: Path
    portfolio_statuses: tuple[ResearchPortfolioStatus, ...]
    eligible_studies: tuple[ResearchManagedStudy, ...]
    blocked_studies: tuple[ResearchManagedStudy, ...]
    loop: ResearchLoopResult

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": "forge-research-managed-loop-result.v1",
            "mode": "bounded_once",
            "portfolio_root": str(self.portfolio_root),
            "portfolio_count": len(self.portfolio_statuses),
            "eligible_studies": [item.to_jsonable() for item in self.eligible_studies],
            "blocked_studies": [item.to_jsonable() for item in self.blocked_studies],
            "loop": self.loop.to_jsonable(),
            "portfolio_statuses": [
                status.to_jsonable() for status in self.portfolio_statuses
            ],
            "authority": {
                "portfolio_dependency_enforced": True,
                "automatic_human_gates_authorized": False,
                "automatic_candidate_import_authorized": False,
                "production_promotion_authorized": False,
                "runtime_wiring_changed": False,
                "evaluation_is_learning_source": False,
            },
        }


def validate_research_portfolio(
    *,
    config: ForgeConfig,
    portfolio_path: Path,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    """Validate one immutable portfolio, its Demand refs, DAG, and mappings."""

    portfolio, _ = _load_portfolio(config, portfolio_path)
    _validate_structure(portfolio)
    if verify_bindings:
        _verify_portfolio_bindings(config, portfolio)
    return portfolio


def find_research_portfolio_memberships(
    *,
    config: ForgeConfig,
    request_path: Path,
    portfolio_root: Path | None = None,
) -> tuple[ResearchPortfolioMembership, ...]:
    """Find exact Portfolio studies that own one ResearchRequest Demand lineage."""

    request = validate_research_request(config=config, request_path=request_path)
    if request["schema_version"] != "forge-research-request.v1":
        return ()
    records, _ = _load_registered_portfolios(
        config=config,
        portfolio_root=portfolio_root,
    )
    memberships: list[ResearchPortfolioMembership] = []
    for portfolio, path in records:
        for study in portfolio["studies"]:
            if not _request_has_demand(request, study["demand"]["artifact"]):
                continue
            if study["task_id"] is not None and request["task_id"] != study["task_id"]:
                raise ResearchPortfolioError(
                    f"Request task_id does not match registered Portfolio study "
                    f"{study['study_id']}"
                )
            memberships.append(
                ResearchPortfolioMembership(
                    portfolio_id=str(portfolio["portfolio_id"]),
                    portfolio_path=path,
                    study_id=str(study["study_id"]),
                    demand_id=str(study["demand"]["artifact_id"]),
                )
            )
    return tuple(
        sorted(
            memberships,
            key=lambda item: (item.portfolio_id, item.study_id),
        )
    )


def seal_research_portfolio(
    *,
    config: ForgeConfig,
    draft_path: Path,
    output_path: Path | None = None,
) -> ResearchPortfolioSealResult:
    """Seal one human-authored Portfolio draft into the repository registry."""

    resolved_draft = _resolve_regular_file(draft_path, "research portfolio draft")
    if not resolved_draft.is_relative_to(config.paths.repo_root):
        raise ResearchPortfolioError(
            "research portfolio draft must be stored inside the repository"
        )
    draft = read_json(resolved_draft)
    if draft.get("schema_version") != PORTFOLIO_VERSION:
        raise ResearchPortfolioError(
            f"research portfolio draft must declare {PORTFOLIO_VERSION}"
        )
    portfolio = dict(draft)
    supplied_identity = portfolio.pop("portfolio_id", None)
    identity = _artifact_id("research-portfolio", portfolio, "portfolio_id")
    if supplied_identity is not None and supplied_identity != identity:
        raise ResearchPortfolioError(
            "research portfolio draft portfolio_id does not match its canonical payload"
        )
    portfolio["portfolio_id"] = identity
    _validate_payload(config, portfolio, PORTFOLIO_VERSION)
    _validate_structure(portfolio)
    _verify_portfolio_bindings(config, portfolio)

    digest = identity.partition(":")[2]
    destination = output_path or (
        config.paths.repo_root / "research" / "portfolios" / f"{digest}.json"
    )
    target = (
        destination
        if destination.is_absolute()
        else config.paths.repo_root / destination
    )
    target = target.expanduser().resolve(strict=False)
    portfolio_root = (
        config.paths.repo_root / "research" / "portfolios"
    ).resolve(strict=False)
    if not target.is_relative_to(portfolio_root) or target.suffix != ".json":
        raise ResearchPortfolioError(
            "sealed research Portfolio must be a JSON file below research/portfolios"
        )
    if target.is_symlink():
        raise ResearchPortfolioError(
            f"research Portfolio output may not be a symlink: {target}"
        )
    if target.exists():
        existing = validate_research_portfolio(
            config=config,
            portfolio_path=target,
        )
        if _identity_body(existing, "portfolio_id") != _identity_body(
            portfolio, "portfolio_id"
        ):
            raise ResearchPortfolioError(
                f"refusing to overwrite a different research Portfolio: {target}"
            )
        return ResearchPortfolioSealResult(
            portfolio_id=str(existing["portfolio_id"]),
            portfolio_path=target,
            reused=True,
        )
    _write_create_only_json(target, portfolio)
    validate_research_portfolio(config=config, portfolio_path=target)
    return ResearchPortfolioSealResult(
        portfolio_id=identity,
        portfolio_path=target,
        reused=False,
    )


def build_research_execution_policy(
    *,
    config: ForgeConfig,
    portfolio_path: Path,
    study_id: str,
    request_path: Path,
) -> dict[str, Any]:
    """Bind one exact A0 approval to its portfolio concurrency budget."""

    portfolio, resolved = _load_portfolio(config, portfolio_path)
    _validate_structure(portfolio)
    _verify_portfolio_bindings(config, portfolio)
    matches = [
        study for study in portfolio["studies"] if study["study_id"] == study_id
    ]
    if len(matches) != 1:
        raise ResearchPortfolioError(
            f"portfolio has no unique study_id {study_id!r}"
        )
    study = matches[0]
    if study["readiness"] != "RUNNABLE_MAPPING":
        raise ResearchPortfolioError(
            f"portfolio study {study_id} is not runnable and cannot receive A0 capacity"
        )
    request = validate_research_request(config=config, request_path=request_path)
    if request["schema_version"] != "forge-research-request.v1":
        raise ResearchPortfolioError(
            "Volvence research portfolio capacity cannot bind an external-domain Request"
        )
    if not _request_has_demand(request, study["demand"]["artifact"]):
        raise ResearchPortfolioError(
            f"Request does not bind portfolio study {study_id}'s exact Demand"
        )
    if request["task_id"] != study["task_id"]:
        raise ResearchPortfolioError(
            f"Request task_id does not match portfolio study {study_id}"
        )

    status = inspect_research_portfolio(config=config, portfolio_path=resolved)
    by_id = {item.study_id: item for item in status.studies}
    unsatisfied = [
        dependency
        for dependency in study["depends_on"]
        if by_id[str(dependency)].state != "COMPLETED_ACCEPTED"
    ]
    if unsatisfied:
        raise ResearchPortfolioError(
            f"portfolio study {study_id} has unsatisfied dependencies: {unsatisfied}"
        )
    lane_limits = {
        str(lane["name"]): int(lane["max_active_runs"])
        for lane in portfolio["scheduling"]["lanes"]
    }
    lane = str(study["concurrency_lane"])
    policy = {
        "portfolio": {
            "artifact_id": portfolio["portfolio_id"],
            "artifact": _content_ref(config, resolved),
        },
        "study_id": study_id,
        "concurrency_lane": lane,
        "max_active_runs_global": int(
            portfolio["scheduling"]["max_active_runs_global"]
        ),
        "max_active_runs_lane": lane_limits[lane],
        "unknown_active_run_policy": "BLOCK",
        "resume_policy": "completed_generation",
    }
    validate_research_execution_policy(
        config=config,
        execution_policy=policy,
        request=request,
    )
    return policy


def validate_research_execution_policy(
    *,
    config: ForgeConfig,
    execution_policy: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate a frozen A0 concurrency policy against its exact Portfolio."""

    portfolio_ref = execution_policy["portfolio"]
    portfolio_path = _verify_content_ref(
        config,
        portfolio_ref["artifact"],
        context="A0 execution policy Portfolio",
    )
    portfolio = validate_research_portfolio(
        config=config,
        portfolio_path=portfolio_path,
    )
    if portfolio["portfolio_id"] != portfolio_ref["artifact_id"]:
        raise ResearchPortfolioError(
            "A0 execution policy Portfolio identity mismatch"
        )
    study_id = str(execution_policy["study_id"])
    matches = [
        study for study in portfolio["studies"] if study["study_id"] == study_id
    ]
    if len(matches) != 1:
        raise ResearchPortfolioError(
            f"A0 execution policy has no unique Portfolio study {study_id!r}"
        )
    study = matches[0]
    lane_limits = {
        str(lane["name"]): int(lane["max_active_runs"])
        for lane in portfolio["scheduling"]["lanes"]
    }
    expected = {
        "concurrency_lane": study["concurrency_lane"],
        "max_active_runs_global": portfolio["scheduling"][
            "max_active_runs_global"
        ],
        "max_active_runs_lane": lane_limits[str(study["concurrency_lane"])],
        "unknown_active_run_policy": "BLOCK",
        "resume_policy": "completed_generation",
    }
    for field, value in expected.items():
        if execution_policy[field] != value:
            raise ResearchPortfolioError(
                f"A0 execution policy {field} does not match its Portfolio"
            )
    if request["schema_version"] != "forge-research-request.v1":
        raise ResearchPortfolioError(
            "Portfolio execution policy cannot bind an external-domain Request"
        )
    if request["task_id"] != study["task_id"]:
        raise ResearchPortfolioError(
            "A0 execution policy Request task_id does not match its Portfolio study"
        )
    if not _request_has_demand(request, study["demand"]["artifact"]):
        raise ResearchPortfolioError(
            "A0 execution policy Request lacks its Portfolio study's exact Demand"
        )
    binding, _ = resolve_registered_task_binding(
        config=config,
        mapping_id=str(study["mapping_id"]),
        identity_key=f"portfolio-a0:{portfolio['portfolio_id']}:{study_id}",
    )
    expected_task_ref = _content_ref(config, binding.task_manifest)
    if request["bindings"]["research_task"] != expected_task_ref:
        raise ResearchPortfolioError(
            "A0 execution policy Request does not bind the registered ResearchTask"
        )
    project = request["bindings"]["task_project"]
    if project["root"] != str(binding.task_project):
        raise ResearchPortfolioError(
            "A0 execution policy Request does not bind the registered task project"
        )
    executable = request["bindings"]["praxist"]["executable"]
    if (
        executable["locator"] != str(binding.praxist_executable)
        or executable["sha256"]
        != sha256_bytes(binding.praxist_executable.read_bytes())
    ):
        raise ResearchPortfolioError(
            "A0 execution policy Request does not bind the registered Praxist executable"
        )
    profile = request["launch"]["profile"]
    expected_profile = {
        "config_file": (
            {
                "locator": str(binding.config_file),
                "sha256": sha256_bytes(binding.config_file.read_bytes()),
            }
            if binding.config_file is not None
            else None
        ),
        "agent_system": binding.agent_system,
        "runtime": binding.runtime,
        "codex_native": binding.codex_native,
        "model_provider": binding.model_provider,
        "model": binding.model,
        "strategy": binding.strategy,
        "cohort": binding.cohort,
        "generations": binding.generations,
        "startup_timeout_seconds": binding.startup_timeout_seconds,
    }
    if profile != expected_profile:
        raise ResearchPortfolioError(
            "A0 execution policy Request launch profile does not match its mapping"
        )
    if Path(request["launch"]["run_dir"]).parent != binding.run_dir.parent:
        raise ResearchPortfolioError(
            "A0 execution policy Request run_dir is outside the registered run root"
        )
    return portfolio


def inspect_research_portfolio(
    *,
    config: ForgeConfig,
    portfolio_path: Path,
) -> ResearchPortfolioStatus:
    """Project current immutable Demand/Request/outcome state without mutation."""

    portfolio, resolved_portfolio = _load_portfolio(config, portfolio_path)
    _validate_structure(portfolio)
    demands = _verify_portfolio_bindings(config, portfolio)
    request_records = _portfolio_request_records(config, portfolio)
    outcomes = _load_outcomes(
        config=config,
        portfolio=portfolio,
        portfolio_path=resolved_portfolio,
        requests=request_records,
    )

    study_states: dict[str, str] = {}
    statuses: list[ResearchPortfolioStudyStatus] = []
    for study in _topological_studies(portfolio):
        study_id = str(study["study_id"])
        dependency_states = tuple(
            (dependency, study_states[dependency])
            for dependency in study["depends_on"]
        )
        request_record = request_records.get(study_id)
        outcome_record = outcomes.get(study_id)
        state = _study_state(
            study=study,
            demand=demands[study_id],
            dependency_states=dependency_states,
            request_record=request_record,
            outcome_record=outcome_record,
        )
        study_states[study_id] = state
        status = request_record[0] if request_record is not None else None
        outcome = outcome_record[0] if outcome_record is not None else None
        statuses.append(
            ResearchPortfolioStudyStatus(
                study_id=study_id,
                state=state,
                priority=int(study["priority"]),
                concurrency_lane=str(study["concurrency_lane"]),
                dependency_states=dependency_states,
                request_id=status.request_id if status else None,
                request_path=status.request_path if status else None,
                run_id=status.run_id if status else None,
                run_dir=status.run_dir if status else None,
                outcome_decision=str(outcome["decision"]) if outcome else None,
                outcome_path=outcome_record[1] if outcome_record else None,
            )
        )
    return ResearchPortfolioStatus(
        portfolio_id=str(portfolio["portfolio_id"]),
        portfolio_path=resolved_portfolio,
        studies=tuple(statuses),
    )


def run_research_portfolio_once(
    *,
    config: ForgeConfig,
    portfolio_path: Path,
    backend: ResearchDiscoveryBackend,
    max_new_discoveries: int = 1,
    max_new_requests: int = 8,
    max_reconciles: int = 8,
    runner: PraxistCommandRunner | None = None,
) -> ResearchPortfolioLoopResult:
    """Delegate one eligible portfolio slice to the existing bounded loop."""

    portfolio, resolved = _load_portfolio(config, portfolio_path)
    _validate_structure(portfolio)
    _verify_portfolio_bindings(config, portfolio)
    before = inspect_research_portfolio(config=config, portfolio_path=resolved)
    eligible_studies = list(
        _eligible_portfolio_studies(portfolio=portfolio, status=before)
    )
    eligible_ids = tuple(str(study["study_id"]) for study in eligible_studies)
    allowed_demand_ids = frozenset(
        str(study["demand"]["artifact_id"]) for study in eligible_studies
    )
    loop = run_demand_research_loop_once(
        config=config,
        backend=backend,
        max_demands=1024,
        max_new_discoveries=max_new_discoveries,
        max_new_requests=max_new_requests,
        max_reconciles=max_reconciles,
        runner=runner,
        allowed_demand_ids=allowed_demand_ids,
    )
    after = inspect_research_portfolio(config=config, portfolio_path=resolved)
    return ResearchPortfolioLoopResult(
        before=before,
        loop=loop,
        after=after,
        eligible_study_ids=eligible_ids,
    )


def run_managed_research_loop_once(
    *,
    config: ForgeConfig,
    backend: ResearchDiscoveryBackend,
    portfolio_root: Path | None = None,
    max_demands: int = 128,
    max_new_discoveries: int = 1,
    max_new_requests: int = 8,
    max_reconciles: int = 8,
    runner: PraxistCommandRunner | None = None,
) -> ResearchManagedLoopResult:
    """Run one global bounded pass while enforcing every registered Portfolio DAG."""

    records, resolved_root = _load_registered_portfolios(
        config=config,
        portfolio_root=portfolio_root,
    )
    statuses: list[ResearchPortfolioStatus] = []
    eligible: list[ResearchManagedStudy] = []
    blocked: list[ResearchManagedStudy] = []
    claimed_demands: dict[str, tuple[str, str]] = {}

    for portfolio, path in records:
        status = inspect_research_portfolio(config=config, portfolio_path=path)
        statuses.append(status)
        by_status = {item.study_id: item for item in status.studies}
        eligible_ids = {
            str(item["study_id"])
            for item in _eligible_portfolio_studies(
                portfolio=portfolio,
                status=status,
            )
        }
        for study in _topological_studies(portfolio):
            study_id = str(study["study_id"])
            demand_id = str(study["demand"]["artifact_id"])
            prior = claimed_demands.get(demand_id)
            if prior is not None:
                raise ResearchPortfolioError(
                    "one exact Demand may have only one Portfolio scheduling owner: "
                    f"{demand_id} is claimed by {prior} and "
                    f"{(portfolio['portfolio_id'], study_id)}"
                )
            claimed_demands[demand_id] = (
                str(portfolio["portfolio_id"]),
                study_id,
            )
            item = ResearchManagedStudy(
                portfolio_id=str(portfolio["portfolio_id"]),
                portfolio_path=path,
                study_id=study_id,
                demand_id=demand_id,
                state=by_status[study_id].state,
            )
            if study_id in eligible_ids:
                eligible.append(item)
            else:
                blocked.append(item)

    preferred_demand_ids = tuple(item.demand_id for item in eligible)
    loop = run_demand_research_loop_once(
        config=config,
        backend=backend,
        max_demands=max_demands,
        max_new_discoveries=max_new_discoveries,
        max_new_requests=max_new_requests,
        max_reconciles=max_reconciles,
        runner=runner,
        blocked_demand_ids=frozenset(item.demand_id for item in blocked),
        preferred_demand_ids=preferred_demand_ids,
    )
    after = tuple(
        inspect_research_portfolio(config=config, portfolio_path=path)
        for _, path in records
    )
    return ResearchManagedLoopResult(
        portfolio_root=resolved_root,
        portfolio_statuses=after,
        eligible_studies=tuple(eligible),
        blocked_studies=tuple(blocked),
        loop=loop,
    )


def review_research_study_outcome(
    *,
    config: ForgeConfig,
    portfolio_path: Path,
    study_id: str,
    request_path: Path,
    evidence_paths: Sequence[Path],
    reviewed_by: str,
    reason: str,
    decision: str,
) -> ResearchStudyOutcomeResult:
    """Seal a named-human dependency decision for one completed exact Request."""

    normalized_decision = decision.upper()
    if normalized_decision not in {"PROCEED", "REVISE", "STOP"}:
        raise ResearchPortfolioError(
            "research study outcome decision must be PROCEED, REVISE, or STOP"
        )
    actor = reviewed_by.strip()
    rationale = reason.strip()
    if not actor or not rationale:
        raise ResearchPortfolioError(
            "research study outcome requires a named reviewer and non-empty reason"
        )
    if not evidence_paths:
        raise ResearchPortfolioError(
            "research study outcome requires at least one exact evidence artifact"
        )

    portfolio, resolved_portfolio = _load_portfolio(config, portfolio_path)
    _validate_structure(portfolio)
    _verify_portfolio_bindings(config, portfolio)
    studies = {str(item["study_id"]): item for item in portfolio["studies"]}
    if study_id not in studies:
        raise ResearchPortfolioError(f"unknown portfolio study_id: {study_id!r}")

    request_statuses = {
        status.request_path.resolve(strict=True): status
        for status in list_research_inbox(config=config)
    }
    resolved_request = _resolve_regular_file(request_path, "ResearchRequest")
    status = request_statuses.get(resolved_request)
    if status is None:
        raise ResearchPortfolioError("ResearchRequest is not registered in the Forge inbox")
    if status.state != "RUN_COMPLETED":
        raise ResearchPortfolioError(
            "study outcome may only be sealed for a RUN_COMPLETED Request"
        )
    request = validate_research_request(
        config=config,
        request_path=resolved_request,
    )
    demand_ref = studies[study_id]["demand"]["artifact"]
    if not _request_has_demand(request, demand_ref):
        raise ResearchPortfolioError(
            "ResearchRequest does not preserve the study's exact Demand lineage"
        )

    outcome: dict[str, Any] = {
        "schema_version": OUTCOME_VERSION,
        "portfolio": {
            "artifact_id": portfolio["portfolio_id"],
            "artifact": _content_ref(config, resolved_portfolio),
        },
        "study_id": study_id,
        "request": {
            "artifact_id": request["request_id"],
            "artifact": _content_ref(config, resolved_request),
        },
        "evidence": [_content_ref(config, path) for path in evidence_paths],
        "decision": normalized_decision,
        "review": {"reviewed_by": actor, "reason": rationale},
        "authority": {
            "dependency_scheduling_only": True,
            "research_start_authorized": False,
            "formal_validation_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    outcome["outcome_id"] = _artifact_id(
        "research-study-outcome", outcome, "outcome_id"
    )
    _validate_payload(config, outcome, OUTCOME_VERSION)
    destination_root = _outcome_root(config, portfolio, study_id)
    existing = sorted(destination_root.glob("*.json")) if destination_root.exists() else []
    if existing:
        if len(existing) != 1:
            raise ResearchPortfolioError(
                f"study has multiple outcome artifacts: {study_id}"
            )
        prior = _load_outcome(config, existing[0])
        if _identity_body(prior, "outcome_id") != _identity_body(outcome, "outcome_id"):
            raise ResearchPortfolioError(
                "study already has a different immutable outcome decision"
            )
        return ResearchStudyOutcomeResult(
            outcome_id=str(prior["outcome_id"]),
            outcome_path=existing[0],
            decision=str(prior["decision"]),
        )
    digest = str(outcome["outcome_id"]).partition(":")[2]
    destination = destination_root / f"{digest}.json"
    _write_create_only_json(destination, outcome)
    return ResearchStudyOutcomeResult(
        outcome_id=str(outcome["outcome_id"]),
        outcome_path=destination,
        decision=normalized_decision,
    )


def _eligible_portfolio_studies(
    *,
    portfolio: Mapping[str, Any],
    status: ResearchPortfolioStatus,
) -> tuple[dict[str, Any], ...]:
    by_status = {item.study_id: item for item in status.studies}
    eligible: list[dict[str, Any]] = []
    for study in _topological_studies(portfolio):
        study_id = str(study["study_id"])
        current = by_status[study_id]
        dependencies_ready = all(
            by_status[str(dependency)].state == "COMPLETED_ACCEPTED"
            for dependency in study["depends_on"]
        )
        continuing = current.state in _RUNNING_REQUEST_STATES
        if (
            study["readiness"] == "RUNNABLE_MAPPING"
            and current.state in _LOOP_ELIGIBLE_STATES
            and (dependencies_ready or continuing)
        ):
            eligible.append(dict(study))
    eligible.sort(
        key=lambda item: (int(item["priority"]), str(item["study_id"]))
    )
    return tuple(eligible)


def _load_registered_portfolios(
    *,
    config: ForgeConfig,
    portfolio_root: Path | None,
) -> tuple[tuple[tuple[dict[str, Any], Path], ...], Path]:
    raw_root = portfolio_root or (
        config.paths.repo_root / "research" / "portfolios"
    )
    candidate = raw_root if raw_root.is_absolute() else config.paths.repo_root / raw_root
    if candidate.is_symlink():
        raise ResearchPortfolioError(
            f"research Portfolio registry may not be a symlink: {candidate}"
        )
    resolved_root = candidate.expanduser().resolve(strict=False)
    if not resolved_root.is_relative_to(config.paths.repo_root):
        raise ResearchPortfolioError(
            "research Portfolio registry must remain inside the repository"
        )
    if not resolved_root.exists():
        return (), resolved_root
    if not resolved_root.is_dir():
        raise ResearchPortfolioError(
            f"research Portfolio registry must be a directory: {resolved_root}"
        )

    records: list[tuple[dict[str, Any], Path]] = []
    identities: set[str] = set()
    for path in sorted(resolved_root.glob("*.json")):
        if path.is_symlink():
            raise ResearchPortfolioError(
                f"registered research Portfolio may not be a symlink: {path}"
            )
        portfolio = validate_research_portfolio(
            config=config,
            portfolio_path=path,
        )
        identity = str(portfolio["portfolio_id"])
        if identity in identities:
            raise ResearchPortfolioError(
                f"duplicate registered research Portfolio identity: {identity}"
            )
        identities.add(identity)
        records.append((portfolio, path.resolve(strict=True)))
    records.sort(
        key=lambda item: (
            str(item[0]["created_at"]),
            str(item[0]["portfolio_id"]),
        )
    )
    return tuple(records), resolved_root


def _verify_portfolio_bindings(
    config: ForgeConfig,
    portfolio: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    demands: dict[str, dict[str, Any]] = {}
    mapping_ids: set[str] = set()
    task_ids: set[str] = set()
    for study in portfolio["studies"]:
        study_id = str(study["study_id"])
        demand_path = _verify_content_ref(
            config,
            study["demand"]["artifact"],
            context=f"portfolio study {study_id} Demand",
        )
        demand = validate_research_demand(config=config, demand_path=demand_path)
        if demand["demand_id"] != study["demand"]["artifact_id"]:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} Demand identity mismatch"
            )
        expected = {
            "claim_id": demand["claim_id"],
            "owner": demand["owner"],
            "capability_axes": demand["capability_axes"],
        }
        for field, value in expected.items():
            if study[field] != value:
                raise ResearchPortfolioError(
                    f"portfolio study {study_id} {field} does not match its Demand"
                )
        mapping_id = study["mapping_id"]
        task_id = study["task_id"]
        if study["readiness"] == "NEEDS_TASK_DESIGN":
            if mapping_id is not None or task_id is not None:
                raise ResearchPortfolioError(
                    f"NEEDS_TASK_DESIGN study {study_id} may not claim a runnable mapping"
                )
            if demand["routing"]["requested_mapping_id"] is not None:
                raise ResearchPortfolioError(
                    f"NEEDS_TASK_DESIGN study {study_id} Demand must not request a mapping"
                )
        else:
            if mapping_id is None or task_id is None:
                raise ResearchPortfolioError(
                    f"RUNNABLE_MAPPING study {study_id} requires mapping_id and task_id"
                )
            if demand["routing"]["requested_mapping_id"] != mapping_id:
                raise ResearchPortfolioError(
                    f"portfolio study {study_id} mapping does not match its Demand"
                )
            if str(mapping_id) in mapping_ids or str(task_id) in task_ids:
                raise ResearchPortfolioError(
                    "runnable portfolio mappings and task_ids must be unique"
                )
            mapping_ids.add(str(mapping_id))
            task_ids.add(str(task_id))
            binding, _ = resolve_registered_task_binding(
                config=config,
                mapping_id=str(mapping_id),
                identity_key=f"portfolio:{portfolio['portfolio_id']}:{study_id}",
            )
            if binding.task_id != task_id:
                raise ResearchPortfolioError(
                    f"portfolio study {study_id} task_id does not match its registry mapping"
                )
            if binding.owner != study["owner"]:
                raise ResearchPortfolioError(
                    f"portfolio study {study_id} owner does not match its ResearchTask"
                )
            if not set(study["capability_axes"]).issubset(binding.capability_axes):
                raise ResearchPortfolioError(
                    f"portfolio study {study_id} axes are not covered by its ResearchTask"
                )
        demands[study_id] = demand
    return demands


def _validate_structure(portfolio: Mapping[str, Any]) -> None:
    studies = {str(item["study_id"]): item for item in portfolio["studies"]}
    if len(studies) != len(portfolio["studies"]):
        raise ResearchPortfolioError("portfolio study_id values must be unique")
    lanes = {
        str(item["name"]): int(item["max_active_runs"])
        for item in portfolio["scheduling"]["lanes"]
    }
    if len(lanes) != len(portfolio["scheduling"]["lanes"]):
        raise ResearchPortfolioError("portfolio concurrency lane names must be unique")
    global_limit = int(portfolio["scheduling"]["max_active_runs_global"])
    if any(limit > global_limit for limit in lanes.values()):
        raise ResearchPortfolioError(
            "a concurrency lane may not exceed max_active_runs_global"
        )
    for study_id, study in studies.items():
        if study["concurrency_lane"] not in lanes:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} references an unknown concurrency lane"
            )
        if study_id in study["depends_on"]:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} may not depend on itself"
            )
        unknown = set(study["depends_on"]) - studies.keys()
        if unknown:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} has unknown dependencies: {sorted(unknown)}"
            )
    _topological_studies(portfolio)


def _topological_studies(portfolio: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    remaining = {
        str(item["study_id"]): dict(item) for item in portfolio["studies"]
    }
    emitted: list[dict[str, Any]] = []
    completed: set[str] = set()
    while remaining:
        ready = [
            study
            for study in remaining.values()
            if set(study["depends_on"]).issubset(completed)
        ]
        if not ready:
            raise ResearchPortfolioError("portfolio dependency graph contains a cycle")
        ready.sort(key=lambda item: (int(item["priority"]), str(item["study_id"])))
        for study in ready:
            study_id = str(study["study_id"])
            emitted.append(study)
            completed.add(study_id)
            del remaining[study_id]
    return tuple(emitted)


def _portfolio_request_records(
    config: ForgeConfig,
    portfolio: Mapping[str, Any],
) -> dict[str, tuple[ResearchControlStatus, dict[str, Any]]]:
    statuses = list_research_inbox(config=config)
    loaded = [
        (
            status,
            validate_research_request(
                config=config,
                request_path=status.request_path,
                verify_bindings=False,
            ),
        )
        for status in statuses
    ]
    records: dict[str, tuple[ResearchControlStatus, dict[str, Any]]] = {}
    for study in portfolio["studies"]:
        matches = [
            (status, request)
            for status, request in loaded
            if request["schema_version"] == "forge-research-request.v1"
            and _request_has_demand(request, study["demand"]["artifact"])
        ]
        nonterminal = [
            item for item in matches if item[0].state not in _TERMINAL_REQUEST_STATES
        ]
        if len(nonterminal) > 1:
            raise ResearchPortfolioError(
                f"portfolio study {study['study_id']} has multiple non-terminal Requests"
            )
        if nonterminal:
            selected = nonterminal[0]
        elif matches:
            selected = max(
                matches,
                key=lambda item: (
                    str(item[1]["created_at"]),
                    str(item[1]["request_id"]),
                ),
            )
        else:
            continue
        if study["task_id"] is not None and selected[1]["task_id"] != study["task_id"]:
            raise ResearchPortfolioError(
                f"portfolio study {study['study_id']} Request task_id mismatch"
            )
        records[str(study["study_id"])] = selected
    return records


def _request_has_demand(
    request: Mapping[str, Any],
    demand_ref: Mapping[str, Any],
) -> bool:
    expected = (str(demand_ref["locator"]), str(demand_ref["sha256"]))
    return any(
        (str(item["locator"]), str(item["sha256"])) == expected
        for item in request["trigger"]["evidence"]
    )


def _study_state(
    *,
    study: Mapping[str, Any],
    demand: Mapping[str, Any],
    dependency_states: tuple[tuple[str, str], ...],
    request_record: tuple[ResearchControlStatus, dict[str, Any]] | None,
    outcome_record: tuple[dict[str, Any], Path] | None,
) -> str:
    del demand
    if outcome_record is not None:
        decision = outcome_record[0]["decision"]
        return {
            "PROCEED": "COMPLETED_ACCEPTED",
            "REVISE": "REVISION_REQUIRED",
            "STOP": "STOPPED_BY_OUTCOME",
        }[str(decision)]
    if request_record is not None:
        return request_record[0].state
    if any(state != "COMPLETED_ACCEPTED" for _, state in dependency_states):
        return "WAITING_FOR_DEPENDENCIES"
    if study["readiness"] == "NEEDS_TASK_DESIGN":
        return "NEEDS_TASK_DESIGN"
    return "REGISTERED"


def _load_outcomes(
    *,
    config: ForgeConfig,
    portfolio: Mapping[str, Any],
    portfolio_path: Path,
    requests: Mapping[str, tuple[ResearchControlStatus, dict[str, Any]]],
) -> dict[str, tuple[dict[str, Any], Path]]:
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    portfolio_ref = _content_ref(config, portfolio_path)
    for study in portfolio["studies"]:
        study_id = str(study["study_id"])
        root = _outcome_root(config, portfolio, study_id)
        paths = sorted(root.glob("*.json")) if root.exists() else []
        if len(paths) > 1:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} has multiple outcome decisions"
            )
        if not paths:
            continue
        outcome = _load_outcome(config, paths[0])
        if outcome["portfolio"] != {
            "artifact_id": portfolio["portfolio_id"],
            "artifact": portfolio_ref,
        }:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} outcome portfolio binding mismatch"
            )
        if outcome["study_id"] != study_id:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} outcome study_id mismatch"
            )
        request_record = requests.get(study_id)
        if request_record is None:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} outcome has no registered Request"
            )
        status, request = request_record
        request_ref = _content_ref(config, status.request_path)
        if outcome["request"] != {
            "artifact_id": request["request_id"],
            "artifact": request_ref,
        }:
            raise ResearchPortfolioError(
                f"portfolio study {study_id} outcome Request binding mismatch"
            )
        for index, evidence in enumerate(outcome["evidence"]):
            _verify_content_ref(
                config,
                evidence,
                context=f"portfolio study {study_id} outcome evidence {index}",
            )
        records[study_id] = (outcome, paths[0])
    return records


def _load_portfolio(
    config: ForgeConfig,
    portfolio_path: Path,
) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_regular_file(portfolio_path, "research portfolio")
    if not resolved.is_relative_to(config.paths.repo_root):
        raise ResearchPortfolioError("research portfolio must be stored inside the repository")
    portfolio = read_json(resolved)
    _validate_payload(config, portfolio, PORTFOLIO_VERSION)
    expected_id = _artifact_id("research-portfolio", portfolio, "portfolio_id")
    if portfolio["portfolio_id"] != expected_id:
        raise ResearchPortfolioError(
            "research portfolio identity does not match its canonical payload"
        )
    return portfolio, resolved


def _load_outcome(config: ForgeConfig, path: Path) -> dict[str, Any]:
    resolved = _resolve_regular_file(path, "research study outcome")
    outcome = read_json(resolved)
    _validate_payload(config, outcome, OUTCOME_VERSION)
    expected = _artifact_id("research-study-outcome", outcome, "outcome_id")
    if outcome["outcome_id"] != expected:
        raise ResearchPortfolioError(
            f"research study outcome identity is invalid: {resolved}"
        )
    if resolved.name != str(outcome["outcome_id"]).partition(":")[2] + ".json":
        raise ResearchPortfolioError(
            f"research study outcome filename is not canonical: {resolved}"
        )
    return outcome


def _outcome_root(
    config: ForgeConfig,
    portfolio: Mapping[str, Any],
    study_id: str,
) -> Path:
    digest = str(portfolio["portfolio_id"]).partition(":")[2]
    return (
        config.paths.artifacts_root
        / "research_portfolio"
        / digest
        / study_id
        / "outcomes"
    )


def _validate_payload(
    config: ForgeConfig,
    payload: Mapping[str, Any],
    version: str,
) -> None:
    SchemaStore(config.paths.forge_root / "schemas").validate(dict(payload), SCHEMA_NAME)
    if payload.get("schema_version") != version:
        raise ResearchPortfolioError(
            f"expected schema_version {version!r}, got {payload.get('schema_version')!r}"
        )


def _artifact_id(prefix: str, payload: Mapping[str, Any], identity_field: str) -> str:
    return f"{prefix}:{sha256_text(canonical_json(_identity_body(payload, identity_field)))}"


def _identity_body(
    payload: Mapping[str, Any],
    identity_field: str,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {identity_field, "created_at"}
    }


def _content_ref(config: ForgeConfig, path: Path) -> dict[str, str]:
    resolved = _resolve_regular_file(path, "content reference")
    locator = (
        resolved.relative_to(config.paths.repo_root).as_posix()
        if resolved.is_relative_to(config.paths.repo_root)
        else str(resolved)
    )
    return {"locator": locator, "sha256": sha256_bytes(resolved.read_bytes())}


def _verify_content_ref(
    config: ForgeConfig,
    value: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    locator = str(value["locator"])
    raw = Path(locator)
    if raw.is_absolute():
        candidate = raw
    else:
        relative = PurePosixPath(locator)
        if not locator or ".." in relative.parts or "." in relative.parts:
            raise ResearchPortfolioError(f"unsafe {context} locator: {locator!r}")
        candidate = config.paths.repo_root / Path(*relative.parts)
    resolved = _resolve_regular_file(candidate, context)
    if sha256_bytes(resolved.read_bytes()) != value["sha256"]:
        raise ResearchPortfolioError(f"{context} SHA-256 drift: {resolved}")
    return resolved


def _resolve_regular_file(path: Path, context: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ResearchPortfolioError(f"{context} may not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchPortfolioError(f"missing {context}: {candidate}") from exc
    if not resolved.is_file():
        raise ResearchPortfolioError(f"{context} must be a regular file: {resolved}")
    return resolved


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ResearchPortfolioError(f"refusing to overwrite immutable artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
