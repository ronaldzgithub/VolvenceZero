"""Read-only aggregation of formal Forge and Praxist operator artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    ArtifactRef,
    AuthoritySnapshot,
    EvidenceSnapshot,
    HealthStatus,
    LifecycleSnapshot,
    LifecycleStage,
    NamedCount,
    PortalWarning,
    PraxistRunSnapshot,
    ResearchLabItem,
    ResearchLabSnapshot,
    ResearchLabSummary,
    SourceHealth,
    WarningSeverity,
)

StatusLoader = Callable[[], object]
Clock = Callable[[], datetime]
RevisionLoader = Callable[[Path], str]


class PraxistStatusError(RuntimeError):
    """Raised when a configured read-only Praxist status command fails."""


@dataclass(frozen=True, slots=True)
class _LoadedArtifact:
    payload: Mapping[str, Any]
    ref: ArtifactRef
    path: Path


@dataclass(frozen=True, slots=True)
class _ControlBundle:
    request: _LoadedArtifact | None
    approval: _LoadedArtifact | None
    event: _LoadedArtifact | None


@dataclass(frozen=True, slots=True)
class _PromotionBundle:
    candidate: _LoadedArtifact | None
    validation: _LoadedArtifact | None
    gate: _LoadedArtifact | None
    receipt: _LoadedArtifact | None


_EXPECTED_IDS = {
    "forge-research-task.v1": "task_id",
    "forge-research-opportunity.v1": "opportunity_id",
    "forge-research-opportunity-routing.v1": "routing_id",
    "forge-research-request.v1": "request_id",
    "forge-external-research-request.v1": "request_id",
    "forge-research-approval.v1": "approval_id",
    "forge-research-control-event.v1": "event_id",
    "forge-external-research-handoff.v1": "handoff_id",
    "forge-research-candidate.v1": "candidate_id",
    "forge-research-promotion-receipt.v1": "receipt_id",
}

_PROMOTION_SOURCES = (
    (
        "research_promotion",
        frozenset(
            {
                "forge-research-candidate.v1",
                "forge-research-validation.v1",
                "forge-research-gate.v1",
                "forge-research-promotion-receipt.v1",
            }
        ),
    ),
    ("research_validation", frozenset({"forge-research-validation.v1"})),
    ("research_gate", frozenset({"forge-research-gate.v1"})),
)
_CANONICAL_HANDOFF_NAME = "volvence_handoff.json"


def command_status_loader(executable: str | os.PathLike[str], *, timeout_seconds: float = 15.0) -> StatusLoader:
    """Build a fixed-argv status loader from an explicitly selected executable."""

    resolved = Path(executable).expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"Praxist executable is not an executable file: {resolved}")

    def load() -> object:
        completed = subprocess.run(
            [str(resolved), "status", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            raise PraxistStatusError(f"praxist status failed with exit {completed.returncode}: {detail}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PraxistStatusError(f"praxist status returned invalid JSON: {exc}") from exc

    return load


class ResearchLabCollector:
    """Publish one immutable cross-owner view without gaining mutation authority."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        *,
        status_loader: StatusLoader | None = None,
        clock: Clock | None = None,
        revision_loader: RevisionLoader | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.status_loader = status_loader
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.revision_loader = revision_loader or _git_revision

    def collect(self) -> ResearchLabSnapshot:
        warnings: list[PortalWarning] = []
        source_counts: dict[str, int] = {
            "tasks": 0,
            "opportunities": 0,
            "control": 0,
            "praxist": 0,
            "promotion": 0,
        }

        repo_revision = self._load_revision(warnings)
        tasks = self._load_tasks(warnings, source_counts)
        opportunities, routes = self._load_opportunities(warnings, source_counts)
        status_rows = self._load_status(warnings, source_counts)
        promotions = self._load_promotions(warnings, source_counts)

        items: list[ResearchLabItem] = []
        formal_task_ids: set[str] = set()
        for task in tasks:
            task_id = _require_str(task.payload, "task_id", task.path)
            formal_task_ids.add(task_id)
            item_warnings = [warning for warning in warnings if warning.task_id == task_id]
            inherited_warning_count = len(item_warnings)
            control = self._load_control(task_id, item_warnings, source_counts)
            route = routes.get(task_id)
            opportunity = self._opportunity_for_route(route, opportunities)
            item = self._build_item(
                task=task,
                opportunity=opportunity,
                route=route,
                control=control,
                status_rows=status_rows,
                promotion=promotions.get(task_id, _PromotionBundle(None, None, None, None)),
                warnings=item_warnings,
            )
            items.append(item)
            warnings.extend(item_warnings[inherited_warning_count:])

        control_root = self.repo_root / "artifacts" / "research_control"
        for task_root in sorted(control_root.glob("external_*")):
            task_id = task_root.name
            if task_id in formal_task_ids or not task_root.is_dir():
                continue
            item_warnings = [warning for warning in warnings if warning.task_id == task_id]
            inherited_warning_count = len(item_warnings)
            control = self._load_control(
                task_id,
                item_warnings,
                source_counts,
                request_versions=frozenset({"forge-external-research-request.v1"}),
            )
            if control.request is None:
                continue
            items.append(
                self._build_external_item(
                    control=control,
                    status_rows=status_rows,
                    warnings=item_warnings,
                )
            )
            warnings.extend(item_warnings[inherited_warning_count:])

        items.sort(key=lambda item: (item.lifecycle.stage.value, item.task_id))
        source_health = self._source_health(source_counts, warnings)
        summary = _summary(items)
        generated_at = _timestamp(self.clock())
        revision_payload = {
            "repo_revision": repo_revision,
            "source_health": [
                {"source": health.source, "status": health.status.value, "count": health.artifacts_seen}
                for health in source_health
            ],
            "items": [_item_revision_view(item) for item in items],
            "warnings": [
                {"code": warning.code, "source": warning.source, "task_id": warning.task_id} for warning in warnings
            ],
        }
        revision = hashlib.sha256(_canonical_json(revision_payload)).hexdigest()
        return ResearchLabSnapshot(
            schema_version="volvence-research-lab-snapshot.v1",
            generated_at=generated_at,
            revision=revision,
            repo_revision=repo_revision,
            summary=summary,
            source_health=tuple(source_health),
            items=tuple(items),
            warnings=tuple(warnings),
        )

    def _load_revision(self, warnings: list[PortalWarning]) -> str:
        try:
            return self.revision_loader(self.repo_root)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            warnings.append(
                PortalWarning(
                    code="REPOSITORY_REVISION_UNAVAILABLE",
                    message=str(exc),
                    source="repository",
                )
            )
            return "unknown"

    def _load_tasks(
        self,
        warnings: list[PortalWarning],
        counts: dict[str, int],
    ) -> list[_LoadedArtifact]:
        root = self.repo_root / "research" / "tasks"
        loaded: list[_LoadedArtifact] = []
        for path in sorted(root.glob("*/task.json")):
            artifact = self._load_artifact(path, "forge-research-task.v1", "task", warnings)
            if artifact is None:
                continue
            try:
                _require_str(artifact.payload, "task_id", path)
                _require_str(artifact.payload, "claim_id", path)
                _require_str(artifact.payload, "owner", path)
                _require_sequence(artifact.payload, "capability_axes", path)
                _require_mapping(artifact.payload, "release", path)
            except ValueError as exc:
                warnings.append(_malformed_warning(path, exc, source="tasks"))
                continue
            loaded.append(artifact)
        counts["tasks"] = len(loaded)
        if not root.is_dir():
            warnings.append(
                PortalWarning(
                    code="TASK_ROOT_MISSING",
                    message=f"formal Task root does not exist: {root}",
                    source="tasks",
                    severity=WarningSeverity.ERROR,
                )
            )
        return loaded

    def _load_opportunities(
        self,
        warnings: list[PortalWarning],
        counts: dict[str, int],
    ) -> tuple[dict[str, _LoadedArtifact], dict[str, _LoadedArtifact]]:
        root = self.repo_root / "artifacts" / "research_opportunities"
        opportunities: dict[str, _LoadedArtifact] = {}
        routes: dict[str, _LoadedArtifact] = {}
        for path in sorted(root.glob("*/*/opportunity.json")):
            artifact = self._load_artifact(
                path,
                "forge-research-opportunity.v1",
                "research opportunity",
                warnings,
            )
            if artifact is not None and artifact.ref.artifact_id is not None:
                opportunities[artifact.ref.artifact_id] = artifact
        for path in sorted(root.glob("*/*/routes/*.json")):
            artifact = self._load_artifact(
                path,
                "forge-research-opportunity-routing.v1",
                "research opportunity route",
                warnings,
            )
            if artifact is None:
                continue
            try:
                mapping = _require_mapping(artifact.payload, "mapping", path)
                task_id = _require_str(mapping, "task_id", path)
            except ValueError as exc:
                warnings.append(_malformed_warning(path, exc, source="opportunities"))
                continue
            existing = routes.get(task_id)
            if existing is None or _created_at(artifact.payload) > _created_at(existing.payload):
                routes[task_id] = artifact
        counts["opportunities"] = len(opportunities) + len(routes)
        return opportunities, routes

    def _load_control(
        self,
        task_id: str,
        warnings: list[PortalWarning],
        counts: dict[str, int],
        request_versions: frozenset[str] = frozenset({"forge-research-request.v1"}),
    ) -> _ControlBundle:
        root = self.repo_root / "artifacts" / "research_control" / task_id
        requests: list[_LoadedArtifact] = []
        for path in sorted(root.glob("*/request.json")):
            payload = self._load_json_object(path, "research request", warnings, task_id=task_id)
            if payload is None:
                continue
            version = payload.get("schema_version")
            if version not in request_versions:
                warnings.append(
                    PortalWarning(
                        code="ARTIFACT_SCHEMA_MISMATCH",
                        message=(
                            f"expected one of {sorted(request_versions)}, got {version!r}: "
                            f"{_portable(path, self.repo_root)}"
                        ),
                        source="control",
                        severity=WarningSeverity.ERROR,
                        task_id=task_id,
                    )
                )
                continue
            requests.append(
                _LoadedArtifact(
                    payload=payload,
                    ref=_artifact_ref(path, payload, "research request", self.repo_root),
                    path=path,
                )
            )
        counts["control"] += len(requests)
        if not requests:
            return _ControlBundle(None, None, None)

        request = max(requests, key=lambda artifact: _created_at(artifact.payload))
        request_id = request.ref.artifact_id
        approvals: list[_LoadedArtifact] = []
        for path in sorted(request.path.parent.glob("approvals/*.json")):
            approval = self._load_artifact(path, "forge-research-approval.v1", "research approval", warnings)
            if approval is None:
                continue
            counts["control"] += 1
            if approval.payload.get("request_id") != request_id:
                warnings.append(
                    PortalWarning(
                        code="APPROVAL_REQUEST_MISMATCH",
                        message=f"approval does not bind selected Request: {_portable(path, self.repo_root)}",
                        source="control",
                        severity=WarningSeverity.ERROR,
                        task_id=task_id,
                    )
                )
                continue
            if approval.payload.get("request_sha256") != request.ref.sha256:
                warnings.append(
                    PortalWarning(
                        code="APPROVAL_BINDING_MISMATCH",
                        message=f"approval Request SHA does not match current bytes: {_portable(path, self.repo_root)}",
                        source="control",
                        severity=WarningSeverity.ERROR,
                        task_id=task_id,
                    )
                )
                continue
            decision = approval.payload.get("decision")
            authority = approval.payload.get("authority")
            expected_authorized = decision == "APPROVE"
            if (
                decision not in {"APPROVE", "REJECT"}
                or not isinstance(authority, Mapping)
                or authority.get("research_start_authorized") is not expected_authorized
            ):
                warnings.append(
                    PortalWarning(
                        code="APPROVAL_AUTHORITY_MALFORMED",
                        message=f"approval decision and authority disagree: {_portable(path, self.repo_root)}",
                        source="control",
                        severity=WarningSeverity.ERROR,
                        task_id=task_id,
                    )
                )
                continue
            approvals.append(approval)
        approval = max(approvals, key=lambda artifact: _created_at(artifact.payload), default=None)

        events: list[_LoadedArtifact] = []
        for path in sorted(request.path.parent.glob("events/*.json")):
            event = self._load_artifact(path, "forge-research-control-event.v1", "research event", warnings)
            if event is not None and event.payload.get("request_id") == request_id:
                counts["control"] += 1
                events.append(event)
        event = max(events, key=lambda artifact: int(artifact.payload.get("sequence", -1)), default=None)
        return _ControlBundle(request, approval, event)

    def _load_status(
        self,
        warnings: list[PortalWarning],
        counts: dict[str, int],
    ) -> tuple[Mapping[str, Any], ...]:
        if self.status_loader is None:
            warnings.append(
                PortalWarning(
                    code="PRAXIST_STATUS_UNAVAILABLE",
                    message="no explicit Praxist status loader was configured",
                    source="praxist",
                )
            )
            return ()
        try:
            payload = self.status_loader()
        except (OSError, subprocess.SubprocessError, PraxistStatusError, ValueError) as exc:
            warnings.append(
                PortalWarning(
                    code="PRAXIST_STATUS_FAILED",
                    message=str(exc),
                    source="praxist",
                    severity=WarningSeverity.ERROR,
                )
            )
            return ()
        if not isinstance(payload, list):
            warnings.append(
                PortalWarning(
                    code="PRAXIST_STATUS_MALFORMED",
                    message="praxist status JSON must be an array",
                    source="praxist",
                    severity=WarningSeverity.ERROR,
                )
            )
            return ()
        rows: list[Mapping[str, Any]] = []
        for index, row in enumerate(payload):
            if not isinstance(row, Mapping) or not isinstance(row.get("run_id"), str):
                warnings.append(
                    PortalWarning(
                        code="PRAXIST_STATUS_ROW_MALFORMED",
                        message=f"praxist status row {index} lacks a string run_id",
                        source="praxist",
                        severity=WarningSeverity.ERROR,
                    )
                )
                continue
            rows.append(row)
        counts["praxist"] = len(rows)
        return tuple(rows)

    def _load_promotions(
        self,
        warnings: list[PortalWarning],
        counts: dict[str, int],
    ) -> dict[str, _PromotionBundle]:
        versions = {
            "forge-research-candidate.v1": "candidate",
            "forge-research-validation.v1": "validation",
            "forge-research-gate.v1": "gate",
            "forge-research-promotion-receipt.v1": "receipt",
        }
        by_task: dict[str, dict[str, list[_LoadedArtifact]]] = {}
        seen_paths: set[Path] = set()
        for directory, allowed_versions in _PROMOTION_SOURCES:
            root = self.repo_root / "artifacts" / directory
            for path in sorted(root.glob("**/*.json")):
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                raw = self._load_json_object(path, "research promotion", warnings)
                if raw is None:
                    continue
                version = raw.get("schema_version")
                if not isinstance(version, str) or version not in allowed_versions:
                    warnings.append(
                        PortalWarning(
                            code="UNSUPPORTED_PROMOTION_ARTIFACT",
                            message=f"unsupported schema_version in {_portable(path, self.repo_root)}",
                            source="promotion",
                        )
                    )
                    continue
                kind = versions[version]
                task_id = raw.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    warnings.append(
                        _malformed_warning(path, ValueError("task_id must be a string"), source="promotion")
                    )
                    continue
                artifact = _LoadedArtifact(
                    payload=raw,
                    ref=_artifact_ref(path, raw, kind, self.repo_root),
                    path=path,
                )
                by_task.setdefault(task_id, {}).setdefault(kind, []).append(artifact)
                counts["promotion"] += 1

        result: dict[str, _PromotionBundle] = {}
        for task_id, kinds in by_task.items():
            result[task_id] = _select_exact_promotion_bundle(
                task_id=task_id,
                candidates=kinds.get("candidate", []),
                validations=kinds.get("validation", []),
                gates=kinds.get("gate", []),
                receipts=kinds.get("receipt", []),
                warnings=warnings,
            )
        return result

    def _build_item(
        self,
        *,
        task: _LoadedArtifact,
        opportunity: _LoadedArtifact | None,
        route: _LoadedArtifact | None,
        control: _ControlBundle,
        status_rows: Sequence[Mapping[str, Any]],
        promotion: _PromotionBundle,
        warnings: list[PortalWarning],
    ) -> ResearchLabItem:
        task_id = _require_str(task.payload, "task_id", task.path)
        task_project = self.repo_root / "research" / "praxist_tasks" / task_id
        if control.request is not None:
            bindings = control.request.payload.get("bindings")
            if isinstance(bindings, Mapping):
                project = bindings.get("task_project")
                if isinstance(project, Mapping) and isinstance(project.get("root"), str):
                    task_project = Path(str(project["root"])).expanduser()

        matching_rows = [row for row in status_rows if _same_path(row.get("task_path"), task_project)]
        active_rows = [row for row in matching_rows if row.get("state") == "running"]
        run: PraxistRunSnapshot | None = None
        if len(active_rows) == 1:
            run = self._run_snapshot(active_rows[0], task_id, warnings)
        elif len(active_rows) > 1:
            warnings.append(
                PortalWarning(
                    code="DUPLICATE_ACTIVE_RUNS",
                    message=f"{len(active_rows)} active Praxist runs map to one Task",
                    source="praxist",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
        elif matching_rows:
            latest_row = max(
                matching_rows,
                key=lambda row: str(row.get("updated_at") or row.get("started_at") or row.get("run_id") or ""),
            )
            run = self._run_snapshot(latest_row, task_id, warnings)

        handoff = self._load_completed_handoff(run, task_id, warnings)

        authority = _authority(control, promotion, task.payload)
        lifecycle = _lifecycle(
            control=control,
            promotion=promotion,
            run=run,
            handoff=handoff,
            duplicate_active_runs=len(active_rows) > 1,
        )
        title = task_id.replace("_", " ").title()
        if opportunity is not None:
            source = opportunity.payload.get("source")
            if isinstance(source, Mapping):
                record = source.get("record")
                if isinstance(record, Mapping) and isinstance(record.get("title"), str):
                    title = str(record["title"])

        refs = [task.ref]
        for artifact in (
            opportunity,
            route,
            control.request,
            control.approval,
            control.event,
            promotion.candidate,
            promotion.validation,
            promotion.gate,
            promotion.receipt,
            handoff,
        ):
            if artifact is not None:
                refs.append(artifact.ref)

        release = _require_mapping(task.payload, "release", task.path)
        axes = tuple(str(value) for value in _require_sequence(task.payload, "capability_axes", task.path))
        updated_at = _latest_timestamp(
            run.updated_at if run else None,
            *(
                _created_at(artifact.payload)
                for artifact in (
                    control.event,
                    control.approval,
                    control.request,
                    promotion.receipt,
                    promotion.gate,
                    promotion.validation,
                    promotion.candidate,
                    handoff,
                )
                if artifact is not None
            ),
        )
        return ResearchLabItem(
            item_id=f"research-lab-item:{task_id}",
            task_id=task_id,
            research_mode="volvence_promotion",
            claim_id=_require_str(task.payload, "claim_id", task.path),
            title=title,
            objective=_require_str(task.payload, "objective", task.path),
            owner=_require_str(task.payload, "owner", task.path),
            capability_axes=axes,
            release_target=str(release.get("target", "unknown")),
            lifecycle=lifecycle,
            authority=authority,
            evidence=_evidence(control, promotion, run, handoff),
            bindings=tuple(refs),
            run=run,
            available_actions=_actions(lifecycle.stage, control, promotion, handoff),
            warnings=tuple(warnings),
            updated_at=updated_at,
        )

    def _build_external_item(
        self,
        *,
        control: _ControlBundle,
        status_rows: Sequence[Mapping[str, Any]],
        warnings: list[PortalWarning],
    ) -> ResearchLabItem:
        request = control.request
        if request is None:  # pragma: no cover - caller narrows this invariant.
            raise AssertionError("external item requires a ResearchRequest")
        payload = request.payload
        task_id = _require_str(payload, "task_id", request.path)
        external = _require_mapping(payload, "external_domain", request.path)
        bindings = _require_mapping(payload, "bindings", request.path)
        project = _require_mapping(bindings, "task_project", request.path)
        task_project = Path(_require_str(project, "root", request.path)).expanduser()

        matching_rows = [row for row in status_rows if _same_path(row.get("task_path"), task_project)]
        active_rows = [row for row in matching_rows if row.get("state") == "running"]
        run: PraxistRunSnapshot | None = None
        terminal_control = (
            control.event is not None
            and control.event.payload.get("state") in {"RUN_COMPLETED", "RUN_FAILED", "BLOCKED"}
        )
        if terminal_control and control.event is not None:
            run = _external_event_run(payload, control.event.payload)
            if active_rows:
                warnings.append(
                    PortalWarning(
                        code="PRAXIST_STATUS_AFTER_TERMINAL_EVENT",
                        message="live status conflicts with the exact terminal Research Control event",
                        source="praxist",
                        severity=WarningSeverity.ERROR,
                        task_id=task_id,
                    )
                )
        elif len(active_rows) == 1:
            run = self._run_snapshot(active_rows[0], task_id, warnings)
        elif len(active_rows) > 1:
            warnings.append(
                PortalWarning(
                    code="DUPLICATE_ACTIVE_RUNS",
                    message=f"{len(active_rows)} active Praxist runs map to one external Request",
                    source="praxist",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
        elif matching_rows:
            latest_row = max(
                matching_rows,
                key=lambda row: str(row.get("updated_at") or row.get("started_at") or row.get("run_id") or ""),
            )
            run = self._run_snapshot(latest_row, task_id, warnings)
        elif control.event is not None:
            run = _external_event_run(payload, control.event.payload)

        handoff = self._load_external_handoff(control, task_id, warnings)
        lifecycle = _external_lifecycle(
            control=control,
            run=run,
            handoff=handoff,
            duplicate_active_runs=not terminal_control and len(active_rows) > 1,
        )
        refs = [request.ref]
        for artifact in (control.approval, control.event, handoff):
            if artifact is not None:
                refs.append(artifact.ref)
        declared = (
            ("external descriptor", bindings.get("external_descriptor"), external.get("descriptor_id")),
            ("external intent", bindings.get("external_intent"), external.get("intent_id")),
            ("external budget", bindings.get("external_budget"), external.get("intent_id")),
        )
        for kind, raw_ref, artifact_id in declared:
            ref = self._external_declared_ref(
                kind=kind,
                raw_ref=raw_ref,
                artifact_id=artifact_id,
                task_id=task_id,
                warnings=warnings,
            )
            if ref is not None:
                refs.append(ref)
        raw_evidence = bindings.get("external_evidence")
        if isinstance(raw_evidence, list):
            for index, raw_ref in enumerate(raw_evidence):
                ref = self._external_declared_ref(
                    kind=f"external evidence {index}",
                    raw_ref=raw_ref,
                    artifact_id=None,
                    task_id=task_id,
                    warnings=warnings,
                )
                if ref is not None:
                    refs.append(ref)

        authority = AuthoritySnapshot(
            a0_research_start_authorized=_is_a0_approved(control),
            formal_validation_status="external_domain_owned",
            modification_gate_decision="not_applicable",
            authorized_wiring="not_applicable",
            runtime_wiring="not_applicable",
            target_adapter_apply_required=False,
            production_default_changed=False,
            evaluation_is_learning_source=False,
        )
        development = "simulation_pending"
        if run is not None and run.state == "running":
            development = "simulation_running"
        if run is not None and run.state == "completed":
            development = "simulation_completed"
        if handoff is not None:
            development = "simulation_handoff_sealed"
        evidence = EvidenceSnapshot(
            development=development,
            formal="external_domain_owned",
            shadow="not_applicable",
            canary="not_applicable",
        )
        updated_at = _latest_timestamp(
            run.updated_at if run else None,
            *(
                _created_at(artifact.payload)
                for artifact in (handoff, control.event, control.approval, request)
                if artifact is not None
            ),
        )
        external_task_id = str(external.get("task_id", task_id))
        return ResearchLabItem(
            item_id=f"research-lab-item:{task_id}",
            task_id=task_id,
            research_mode="external_simulation",
            claim_id=_require_str(payload, "claim_id", request.path),
            title=f"Foundry · {external_task_id}",
            objective=_require_str(payload, "objective", request.path),
            owner=_require_str(payload, "owner", request.path),
            capability_axes=(),
            release_target="foundry:simulation:proposal_only",
            lifecycle=lifecycle,
            authority=authority,
            evidence=evidence,
            bindings=tuple(refs),
            run=run,
            available_actions=_external_actions(lifecycle.stage, handoff),
            warnings=tuple(warnings),
            updated_at=updated_at,
        )

    def _load_external_handoff(
        self,
        control: _ControlBundle,
        task_id: str,
        warnings: list[PortalWarning],
    ) -> _LoadedArtifact | None:
        request = control.request
        if request is None:
            return None
        paths = sorted(request.path.parent.glob("handoffs/*.json"))
        if len(paths) > 1:
            warnings.append(
                PortalWarning(
                    code="EXTERNAL_HANDOFF_AMBIGUOUS",
                    message="external Request has more than one immutable handoff",
                    source="control",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        if not paths:
            return None
        handoff = self._load_artifact(
            paths[0],
            "forge-external-research-handoff.v1",
            "external handoff",
            warnings,
        )
        if handoff is None:
            return None
        request_binding = handoff.payload.get("request")
        result = handoff.payload.get("result")
        authority = handoff.payload.get("authority")
        if (
            not isinstance(request_binding, Mapping)
            or request_binding.get("request_id") != request.ref.artifact_id
            or request_binding.get("sha256") != request.ref.sha256
            or not isinstance(result, Mapping)
            or result.get("evidence_class") != "simulation"
            or result.get("adoption_mode") != "proposal_only"
            or not isinstance(authority, Mapping)
            or authority.get("volvence_promotion_eligible") is not False
            or authority.get("modification_gate_applicable") is not False
            or authority.get("runtime_wiring_applicable") is not False
        ):
            warnings.append(
                PortalWarning(
                    code="EXTERNAL_HANDOFF_BINDING_MISMATCH",
                    message="external handoff does not preserve its exact simulation-only Request boundary",
                    source="control",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        return handoff

    def _external_declared_ref(
        self,
        *,
        kind: str,
        raw_ref: object,
        artifact_id: object,
        task_id: str,
        warnings: list[PortalWarning],
    ) -> ArtifactRef | None:
        if not isinstance(raw_ref, Mapping):
            warnings.append(
                PortalWarning(
                    code="EXTERNAL_BINDING_MALFORMED",
                    message=f"{kind} binding is not an object",
                    source="control",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        locator = raw_ref.get("locator")
        declared_sha = raw_ref.get("sha256")
        if not isinstance(locator, str) or not isinstance(declared_sha, str):
            warnings.append(
                PortalWarning(
                    code="EXTERNAL_BINDING_MALFORMED",
                    message=f"{kind} binding lacks locator or SHA-256",
                    source="control",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        path = Path(locator).expanduser()
        path = path if path.is_absolute() else self.repo_root / path
        try:
            actual_sha = _sha256_file(path.resolve())
        except OSError as exc:
            warnings.append(
                PortalWarning(
                    code="EXTERNAL_BINDING_UNREADABLE",
                    message=f"cannot read {kind}: {exc}",
                    source="control",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
        else:
            if actual_sha != declared_sha:
                warnings.append(
                    PortalWarning(
                        code="EXTERNAL_BINDING_DRIFT",
                        message=f"{kind} bytes changed after Request submission",
                        source="control",
                        severity=WarningSeverity.ERROR,
                        task_id=task_id,
                    )
                )
        return ArtifactRef(
            kind=kind,
            locator=_portable(path, self.repo_root),
            sha256=declared_sha,
            artifact_id=str(artifact_id) if isinstance(artifact_id, str) else None,
        )

    def _run_snapshot(
        self,
        row: Mapping[str, Any],
        task_id: str,
        warnings: list[PortalWarning],
    ) -> PraxistRunSnapshot:
        run_dir_value = row.get("run_dir")
        run_dir = Path(str(run_dir_value)).expanduser() if isinstance(run_dir_value, str) else None
        runtime = None
        model_provider = row.get("model_provider_ref") if isinstance(row.get("model_provider_ref"), str) else None
        model = row.get("model") if isinstance(row.get("model"), str) else None
        if run_dir is not None:
            startup_path = run_dir / "startup_config.json"
            startup = self._load_json_object(startup_path, "Praxist startup config", warnings, task_id=task_id)
            if startup is not None:
                if startup.get("schema_version") != "praxist.startup.v1":
                    warnings.append(
                        PortalWarning(
                            code="PRAXIST_STARTUP_SCHEMA_MISMATCH",
                            message=f"unexpected startup schema in {startup_path}",
                            source="praxist",
                            task_id=task_id,
                        )
                    )
                canonical = startup.get("canonical_args")
                if isinstance(canonical, Mapping):
                    runtime = canonical.get("runtime") if isinstance(canonical.get("runtime"), str) else None
                    model_provider = (
                        canonical.get("model_provider")
                        if isinstance(canonical.get("model_provider"), str)
                        else model_provider
                    )
                    model = canonical.get("model") if isinstance(canonical.get("model"), str) else model

        peer_health_raw = row.get("peer_health_summary")
        peer_health: list[NamedCount] = []
        if isinstance(peer_health_raw, Mapping):
            for name in sorted(peer_health_raw):
                count = peer_health_raw[name]
                if isinstance(count, int) and not isinstance(count, bool):
                    peer_health.append(NamedCount(str(name), count))
        peers = row.get("peers")
        peers_total = len(peers) if isinstance(peers, list) else sum(value.count for value in peer_health)
        generation = row.get("generation")
        pid = row.get("pid")
        findings = row.get("findings_total", 0)
        return PraxistRunSnapshot(
            run_id=str(row["run_id"]),
            state=str(row.get("state", "unknown")),
            source=str(row.get("source", "unknown")),
            pid=(pid if row.get("state") == "running" and isinstance(pid, int) and not isinstance(pid, bool) else None),
            task_path=str(row.get("task_path", "")),
            run_dir=str(run_dir) if run_dir is not None else None,
            generation=generation if isinstance(generation, int) and not isinstance(generation, bool) else None,
            findings_total=findings if isinstance(findings, int) and not isinstance(findings, bool) else 0,
            peers_total=peers_total,
            peer_health=tuple(peer_health),
            runtime=runtime,
            model_provider=model_provider,
            model=model,
            started_at=row.get("started_at") if isinstance(row.get("started_at"), str) else None,
            updated_at=row.get("updated_at") if isinstance(row.get("updated_at"), str) else None,
        )

    def _load_completed_handoff(
        self,
        run: PraxistRunSnapshot | None,
        task_id: str,
        warnings: list[PortalWarning],
    ) -> _LoadedArtifact | None:
        if run is None or run.state != "completed" or run.run_dir is None:
            return None
        path = Path(run.run_dir) / _CANONICAL_HANDOFF_NAME
        if not path.exists():
            return None
        handoff = self._load_artifact(
            path,
            "forge-praxist-candidate-handoff.v1",
            "praxist handoff",
            warnings,
        )
        if handoff is None:
            return None
        if handoff.payload.get("task_id") != task_id or handoff.payload.get("run_id") != run.run_id:
            warnings.append(
                PortalWarning(
                    code="PRAXIST_HANDOFF_BINDING_MISMATCH",
                    message=f"canonical handoff does not bind task/run: {_portable(path, self.repo_root)}",
                    source="praxist",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        return handoff

    def _opportunity_for_route(
        self,
        route: _LoadedArtifact | None,
        opportunities: Mapping[str, _LoadedArtifact],
    ) -> _LoadedArtifact | None:
        if route is None:
            return None
        opportunity_id = route.payload.get("opportunity_id")
        return opportunities.get(opportunity_id) if isinstance(opportunity_id, str) else None

    def _load_artifact(
        self,
        path: Path,
        expected_version: str,
        kind: str,
        warnings: list[PortalWarning],
    ) -> _LoadedArtifact | None:
        payload = self._load_json_object(path, kind, warnings)
        if payload is None:
            return None
        if payload.get("schema_version") != expected_version:
            warnings.append(
                PortalWarning(
                    code="ARTIFACT_SCHEMA_MISMATCH",
                    message=(
                        f"expected {expected_version}, got {payload.get('schema_version')!r}: "
                        f"{_portable(path, self.repo_root)}"
                    ),
                    source=_source_for_version(expected_version),
                    severity=WarningSeverity.ERROR,
                )
            )
            return None
        return _LoadedArtifact(
            payload=payload,
            ref=_artifact_ref(path, payload, kind, self.repo_root),
            path=path,
        )

    def _load_json_object(
        self,
        path: Path,
        kind: str,
        warnings: list[PortalWarning],
        *,
        task_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            warnings.append(
                PortalWarning(
                    code="ARTIFACT_MISSING",
                    message=f"{kind} is missing: {_portable(path, self.repo_root)}",
                    source=_source_for_kind(kind),
                    task_id=task_id,
                )
            )
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(
                PortalWarning(
                    code="INVALID_JSON_ARTIFACT",
                    message=f"cannot read {kind} {_portable(path, self.repo_root)}: {exc}",
                    source=_source_for_kind(kind),
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        if not isinstance(value, Mapping):
            warnings.append(
                PortalWarning(
                    code="MALFORMED_ARTIFACT",
                    message=f"{kind} must contain a JSON object: {_portable(path, self.repo_root)}",
                    source=_source_for_kind(kind),
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
            return None
        return value

    @staticmethod
    def _source_health(
        counts: Mapping[str, int],
        warnings: Sequence[PortalWarning],
    ) -> list[SourceHealth]:
        result: list[SourceHealth] = []
        for source in ("tasks", "opportunities", "control", "praxist", "promotion"):
            related = [warning for warning in warnings if warning.source == source]
            errors = [warning for warning in related if warning.severity is WarningSeverity.ERROR]
            if source == "praxist" and any(
                warning.code in {"PRAXIST_STATUS_UNAVAILABLE", "PRAXIST_STATUS_FAILED"} for warning in related
            ):
                status = HealthStatus.UNAVAILABLE
            elif errors or related:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            detail = "healthy"
            if related:
                detail = f"{len(errors)} errors, {len(related) - len(errors)} warnings"
            elif counts[source] == 0:
                detail = "no artifacts yet"
            result.append(SourceHealth(source, status, counts[source], detail))
        return result


def _external_event_run(
    request: Mapping[str, Any],
    event: Mapping[str, Any],
) -> PraxistRunSnapshot | None:
    raw_run = event.get("run")
    if not isinstance(raw_run, Mapping):
        return None
    event_state = str(event.get("state", ""))
    state = raw_run.get("praxist_state")
    if not isinstance(state, str) or not state:
        state = {
            "RUNNING": "running",
            "RUN_COMPLETED": "completed",
            "RUN_FAILED": "failed",
            "STARTING": "starting",
        }.get(event_state, "unknown")
    profile = request.get("launch")
    profile = profile.get("profile") if isinstance(profile, Mapping) else None
    bindings = request.get("bindings")
    project = bindings.get("task_project") if isinstance(bindings, Mapping) else None
    task_path = project.get("root") if isinstance(project, Mapping) else ""
    pid = raw_run.get("pid")
    generation = raw_run.get("generation")
    findings = raw_run.get("findings_total")
    return PraxistRunSnapshot(
        run_id=str(raw_run.get("run_id", "")),
        state=state,
        source="research_control_event",
        pid=pid if state == "running" and isinstance(pid, int) and not isinstance(pid, bool) else None,
        task_path=str(task_path),
        run_dir=str(raw_run["run_dir"]) if isinstance(raw_run.get("run_dir"), str) else None,
        generation=(
            generation if isinstance(generation, int) and not isinstance(generation, bool) else None
        ),
        findings_total=(
            findings if isinstance(findings, int) and not isinstance(findings, bool) else 0
        ),
        peers_total=0,
        peer_health=(),
        runtime=(
            str(profile["runtime"])
            if isinstance(profile, Mapping) and isinstance(profile.get("runtime"), str)
            else None
        ),
        model_provider=(
            str(profile["model_provider"])
            if isinstance(profile, Mapping) and isinstance(profile.get("model_provider"), str)
            else None
        ),
        model=(
            str(profile["model"])
            if isinstance(profile, Mapping) and isinstance(profile.get("model"), str)
            else None
        ),
        started_at=None,
        updated_at=(
            str(raw_run["updated_at"])
            if isinstance(raw_run.get("updated_at"), str)
            else _created_at(event)
        ),
    )


def _external_lifecycle(
    *,
    control: _ControlBundle,
    run: PraxistRunSnapshot | None,
    handoff: _LoadedArtifact | None,
    duplicate_active_runs: bool,
) -> LifecycleSnapshot:
    if duplicate_active_runs:
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            "multiple active Praxist runs map to one external Request",
            run.updated_at if run else None,
        )
    if control.approval is not None and control.approval.payload.get("decision") == "REJECT":
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            "exact A0 review rejected this external ResearchRequest",
            _created_at(control.approval.payload),
        )
    if control.event is not None and control.event.payload.get("state") in {"BLOCKED", "RUN_FAILED"}:
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            f"Research Control state is {control.event.payload.get('state')}",
            _created_at(control.event.payload),
        )
    if handoff is not None:
        return LifecycleSnapshot(
            LifecycleStage.RESEARCH_COMPLETE,
            None,
            None,
            _created_at(handoff.payload),
        )
    if run is not None and run.state == "completed":
        return LifecycleSnapshot(
            LifecycleStage.RESEARCH_COMPLETE,
            None,
            "simulation result is complete and awaits immutable external handoff",
            run.updated_at,
        )
    if run is not None and run.state == "running":
        return LifecycleSnapshot(
            LifecycleStage.RESEARCH_RUNNING,
            LifecycleStage.RESEARCH_COMPLETE,
            None,
            run.started_at or run.updated_at,
        )
    if run is not None and run.state in {"failed", "stale", "stopped", "status_inconsistent", "unknown"}:
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            f"Praxist run is {run.state}; explicit lifecycle repair is required",
            run.updated_at,
        )
    if _is_a0_approved(control):
        blocker = None
        if control.event is not None and control.event.payload.get("state") == "WAITING_FOR_CAPACITY":
            blocker = "waiting for Praxist capacity"
        return LifecycleSnapshot(
            LifecycleStage.PREFLIGHT,
            LifecycleStage.RESEARCH_RUNNING,
            blocker,
            _created_at(control.event.payload if control.event else control.approval.payload),
        )
    return LifecycleSnapshot(
        LifecycleStage.AWAITING_A0,
        LifecycleStage.PREFLIGHT,
        "exact named-human research approval is required",
        _created_at(control.request.payload) if control.request is not None else None,
    )


def _external_actions(
    stage: LifecycleStage,
    handoff: _LoadedArtifact | None,
) -> tuple[str, ...]:
    if stage is LifecycleStage.AWAITING_A0:
        return ("review_a0",)
    if stage is LifecycleStage.PREFLIGHT:
        return ("reconcile",)
    if stage is LifecycleStage.RESEARCH_RUNNING:
        return ("reconcile", "view_run")
    if stage is LifecycleStage.RESEARCH_COMPLETE and handoff is None:
        return ("record_external_handoff",)
    if stage is LifecycleStage.BLOCKED:
        return ("inspect_blocker",)
    return ()


def _authority(
    control: _ControlBundle,
    promotion: _PromotionBundle,
    task: Mapping[str, Any],
) -> AuthoritySnapshot:
    validation_status = "not_performed"
    if promotion.validation is not None:
        validation_status = str(promotion.validation.payload.get("status", "unknown")).lower()
    gate_decision = "not_evaluated"
    if promotion.gate is not None:
        gate_decision = str(promotion.gate.payload.get("decision", "unknown")).lower()
    authorized_wiring = "disabled"
    target_apply_required = False
    if promotion.receipt is not None and promotion.receipt.payload.get("outcome") == "AUTHORIZED":
        transition = promotion.receipt.payload.get("transition")
        authority = promotion.receipt.payload.get("authority")
        if isinstance(transition, Mapping):
            authorized_wiring = str(transition.get("resulting_wiring", "disabled"))
        if isinstance(authority, Mapping):
            target_apply_required = authority.get("target_adapter_apply_required") is True
    release = task.get("release")
    initial_wiring = "disabled"
    if isinstance(release, Mapping) and isinstance(release.get("initial_wiring"), str):
        initial_wiring = str(release["initial_wiring"])
    return AuthoritySnapshot(
        a0_research_start_authorized=_is_a0_approved(control),
        formal_validation_status=validation_status,
        modification_gate_decision=gate_decision,
        authorized_wiring=authorized_wiring,
        runtime_wiring=initial_wiring,
        target_adapter_apply_required=target_apply_required,
        production_default_changed=False,
        evaluation_is_learning_source=False,
    )


def _lifecycle(
    *,
    control: _ControlBundle,
    promotion: _PromotionBundle,
    run: PraxistRunSnapshot | None,
    handoff: _LoadedArtifact | None,
    duplicate_active_runs: bool,
) -> LifecycleSnapshot:
    if duplicate_active_runs:
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            "multiple active Praxist runs map to one Task",
            run.updated_at if run else None,
        )
    if control.approval is not None and control.approval.payload.get("decision") == "REJECT":
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            "exact A0 review rejected this ResearchRequest",
            _created_at(control.approval.payload),
        )
    if run is not None and run.state == "running":
        return LifecycleSnapshot(
            LifecycleStage.RESEARCH_RUNNING,
            LifecycleStage.RESEARCH_COMPLETE,
            None,
            run.started_at,
        )
    if promotion.receipt is not None:
        payload = promotion.receipt.payload
        if payload.get("outcome") == "BLOCKED":
            if _has_fresh_authorization_evidence(promotion):
                transition = payload.get("transition")
                resulting = transition.get("resulting_wiring") if isinstance(transition, Mapping) else None
                retry_stage = LifecycleStage.AWAITING_A2 if resulting == "shadow" else LifecycleStage.AWAITING_A1
                return LifecycleSnapshot(retry_stage, None, None, _created_at(promotion.gate.payload))
            reasons = payload.get("blocking_reasons")
            reason = "; ".join(str(value) for value in reasons) if isinstance(reasons, list) else "promotion blocked"
            return LifecycleSnapshot(LifecycleStage.BLOCKED, None, reason, _created_at(payload))
        transition = payload.get("transition")
        action = payload.get("action")
        resulting = transition.get("resulting_wiring") if isinstance(transition, Mapping) else None
        if action == "rollback" and resulting == "disabled":
            if _has_fresh_authorization_evidence(promotion):
                return LifecycleSnapshot(
                    LifecycleStage.AWAITING_A1,
                    LifecycleStage.SHADOW,
                    None,
                    _created_at(promotion.gate.payload),
                )
            return LifecycleSnapshot(LifecycleStage.ROLLED_BACK, None, None, _created_at(payload))
        if resulting == "active":
            return LifecycleSnapshot(
                LifecycleStage.AWAITING_A2,
                LifecycleStage.ACTIVE,
                "target adapter apply receipt is not present",
                _created_at(payload),
            )
        if resulting == "shadow":
            if _has_fresh_authorization_evidence(promotion):
                return LifecycleSnapshot(
                    LifecycleStage.AWAITING_A2,
                    LifecycleStage.ACTIVE,
                    None,
                    _created_at(promotion.gate.payload),
                )
            return LifecycleSnapshot(
                LifecycleStage.AWAITING_A1,
                LifecycleStage.SHADOW,
                "target adapter apply receipt is not present",
                _created_at(payload),
            )
    if promotion.gate is not None and promotion.gate.payload.get("decision") == "ALLOW":
        return LifecycleSnapshot(
            LifecycleStage.AWAITING_A1,
            LifecycleStage.SHADOW,
            None,
            _created_at(promotion.gate.payload),
        )
    if promotion.validation is not None:
        status = promotion.validation.payload.get("status")
        if status == "BLOCK":
            return LifecycleSnapshot(
                LifecycleStage.BLOCKED,
                None,
                "loop-external formal validation blocked the candidate",
                _created_at(promotion.validation.payload),
            )
        return LifecycleSnapshot(
            LifecycleStage.FORMAL_VALIDATION,
            LifecycleStage.AWAITING_A1,
            "ModificationGate evidence is not present",
            _created_at(promotion.validation.payload),
        )
    if promotion.candidate is not None:
        return LifecycleSnapshot(
            LifecycleStage.CANDIDATE_RETAINED,
            LifecycleStage.FORMAL_VALIDATION,
            None,
            _created_at(promotion.candidate.payload),
        )
    if run is not None and run.state == "completed":
        return LifecycleSnapshot(
            LifecycleStage.RESEARCH_COMPLETE,
            LifecycleStage.CANDIDATE_RETAINED,
            None if handoff is not None else "committed Praxist handoff is not present",
            run.updated_at,
        )
    if run is not None and run.state in {
        "failed",
        "stale",
        "stopped",
        "status_inconsistent",
        "unknown",
    }:
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            f"Praxist run is {run.state}; explicit lifecycle repair is required",
            run.updated_at,
        )
    if _is_a0_approved(control):
        blocker = None
        if control.event is not None and control.event.payload.get("state") == "WAITING_FOR_CAPACITY":
            blocker = "waiting for Praxist capacity"
        return LifecycleSnapshot(
            LifecycleStage.PREFLIGHT,
            LifecycleStage.RESEARCH_RUNNING,
            blocker,
            _created_at(control.event.payload if control.event else control.approval.payload),
        )
    if control.request is not None:
        return LifecycleSnapshot(
            LifecycleStage.AWAITING_A0,
            LifecycleStage.PREFLIGHT,
            "exact named-human research approval is required",
            _created_at(control.request.payload),
        )
    return LifecycleSnapshot(
        LifecycleStage.NEEDS_TASK_DESIGN,
        LifecycleStage.AWAITING_A0,
        "no exact ResearchRequest is present",
        None,
    )


def _evidence(
    control: _ControlBundle,
    promotion: _PromotionBundle,
    run: PraxistRunSnapshot | None,
    handoff: _LoadedArtifact | None,
) -> EvidenceSnapshot:
    development = "running" if run is not None else "baseline_registered"
    if run is not None and run.state == "completed":
        development = "completed"
    if handoff is not None:
        development = "handoff_committed"
    if promotion.candidate is not None:
        development = "candidate_retained"
    formal = "not_performed"
    if promotion.validation is not None:
        formal = str(promotion.validation.payload.get("status", "unknown")).lower()
    shadow = "not_started"
    canary = "not_started"
    if promotion.receipt is not None and promotion.receipt.payload.get("outcome") == "AUTHORIZED":
        transition = promotion.receipt.payload.get("transition")
        resulting = transition.get("resulting_wiring") if isinstance(transition, Mapping) else None
        if resulting == "shadow":
            shadow = "authorized_not_applied"
        elif resulting == "active":
            canary = "authorized_not_applied"
    return EvidenceSnapshot(development, formal, shadow, canary)


def _actions(
    stage: LifecycleStage,
    control: _ControlBundle,
    promotion: _PromotionBundle,
    handoff: _LoadedArtifact | None,
) -> tuple[str, ...]:
    if stage is LifecycleStage.AWAITING_A0 and control.request is not None:
        return ("review_a0",)
    if stage is LifecycleStage.PREFLIGHT and _is_a0_approved(control):
        return ("reconcile",)
    if stage is LifecycleStage.RESEARCH_RUNNING:
        return ("view_run",)
    if stage is LifecycleStage.RESEARCH_COMPLETE:
        return ("import_candidate",) if handoff is not None else ("inspect_handoff",)
    if stage is LifecycleStage.CANDIDATE_RETAINED:
        return ("run_formal_validation",)
    if stage is LifecycleStage.FORMAL_VALIDATION:
        return ("view_formal_evidence",)
    if stage is LifecycleStage.AWAITING_A1:
        if _receipt_can_rollback(promotion.receipt):
            return ("rollback",)
        return ("authorize_shadow",)
    if stage is LifecycleStage.AWAITING_A2:
        return ("authorize_active",) if _can_authorize_active(promotion) else ("rollback",)
    if stage in {LifecycleStage.SHADOW, LifecycleStage.ACTIVE}:
        return ("rollback",)
    if stage is LifecycleStage.BLOCKED:
        return ("inspect_blocker",)
    return ()


def _has_fresh_authorization_evidence(promotion: _PromotionBundle) -> bool:
    if promotion.validation is None or promotion.gate is None or promotion.receipt is None:
        return False
    if promotion.validation.payload.get("status") != "PASS" or promotion.gate.payload.get("decision") != "ALLOW":
        return False
    bindings = promotion.receipt.payload.get("bindings")
    if not isinstance(bindings, Mapping):
        return False
    return (
        bindings.get("validation_sha256") != promotion.validation.ref.sha256
        and bindings.get("gate_sha256") != promotion.gate.ref.sha256
    )


def _receipt_can_rollback(receipt: _LoadedArtifact | None) -> bool:
    if receipt is None or receipt.payload.get("outcome") != "AUTHORIZED":
        return False
    transition = receipt.payload.get("transition")
    return isinstance(transition, Mapping) and transition.get("resulting_wiring") in {"shadow", "active"}


def _can_authorize_active(promotion: _PromotionBundle) -> bool:
    if not _has_fresh_authorization_evidence(promotion) or promotion.receipt is None:
        return False
    transition = promotion.receipt.payload.get("transition")
    return (
        promotion.receipt.payload.get("outcome") == "AUTHORIZED"
        and isinstance(transition, Mapping)
        and transition.get("resulting_wiring") == "shadow"
    )


def _summary(items: Sequence[ResearchLabItem]) -> ResearchLabSummary:
    stage_counts = tuple(
        NamedCount(stage.value, sum(1 for item in items if item.lifecycle.stage is stage)) for stage in LifecycleStage
    )
    return ResearchLabSummary(
        registered_tasks=len(items),
        stage_counts=stage_counts,
        active_runs=sum(1 for item in items if item.run is not None and item.run.state == "running"),
        blocked=sum(1 for item in items if item.lifecycle.stage is LifecycleStage.BLOCKED),
        awaiting_human=sum(
            1
            for item in items
            if item.lifecycle.stage
            in {LifecycleStage.AWAITING_A0, LifecycleStage.AWAITING_A1, LifecycleStage.AWAITING_A2}
        ),
        production_active=sum(1 for item in items if item.authority.runtime_wiring == "active"),
    )


def _is_a0_approved(control: _ControlBundle) -> bool:
    if control.approval is None:
        return False
    authority = control.approval.payload.get("authority")
    return (
        control.approval.payload.get("decision") == "APPROVE"
        and isinstance(authority, Mapping)
        and authority.get("research_start_authorized") is True
    )


def _item_revision_view(item: ResearchLabItem) -> dict[str, Any]:
    return {
        "task_id": item.task_id,
        "research_mode": item.research_mode,
        "stage": item.lifecycle.stage.value,
        "bindings": [(ref.kind, ref.sha256) for ref in item.bindings],
        "run": (
            {
                "run_id": item.run.run_id,
                "state": item.run.state,
                "pid": item.run.pid,
                "generation": item.run.generation,
                "updated_at": item.run.updated_at,
            }
            if item.run is not None
            else None
        ),
        "authority": {
            "a0": item.authority.a0_research_start_authorized,
            "formal": item.authority.formal_validation_status,
            "gate": item.authority.modification_gate_decision,
            "authorized_wiring": item.authority.authorized_wiring,
            "runtime_wiring": item.authority.runtime_wiring,
        },
    }


def _artifact_ref(
    path: Path,
    payload: Mapping[str, Any],
    kind: str,
    repo_root: Path,
) -> ArtifactRef:
    version = payload.get("schema_version")
    identity_field = _EXPECTED_IDS.get(str(version))
    artifact_id = payload.get(identity_field) if identity_field is not None else None
    return ArtifactRef(
        kind=kind,
        locator=_portable(path, repo_root),
        sha256=_sha256_file(path),
        artifact_id=str(artifact_id) if isinstance(artifact_id, str) else None,
    )


def _select_exact_promotion_bundle(
    *,
    task_id: str,
    candidates: Sequence[_LoadedArtifact],
    validations: Sequence[_LoadedArtifact],
    gates: Sequence[_LoadedArtifact],
    receipts: Sequence[_LoadedArtifact],
    warnings: list[PortalWarning],
) -> _PromotionBundle:
    """Select one candidate branch while preserving exact downstream bindings.

    Validation and Gate artifacts live in sibling owner directories and may be
    produced repeatedly.  A timestamp-only zip can therefore combine bytes
    that were never reviewed together.  The portal follows the newest sealed
    candidate, then admits only artifacts whose explicit ids and raw digests
    bind that candidate/evidence round.  Receipt state is selected separately
    because a fresh validation/gate round may legitimately follow the previous
    authorization receipt.
    """

    candidate = _latest(candidates)
    if candidate is None:
        receipt = _latest(receipts)
        if receipt is not None:
            warnings.append(
                PortalWarning(
                    code="PROMOTION_CANDIDATE_MISSING",
                    message=f"promotion receipt has no available candidate bytes for task {task_id}",
                    source="promotion",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )
        return _PromotionBundle(None, None, None, receipt)

    candidate_id = candidate.payload.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        warnings.append(
            PortalWarning(
                code="PROMOTION_CANDIDATE_ID_MISSING",
                message=f"candidate lacks a stable candidate_id: {candidate.ref.locator}",
                source="promotion",
                severity=WarningSeverity.ERROR,
                task_id=task_id,
            )
        )
        return _PromotionBundle(candidate, None, None, None)
    candidate_sha256 = candidate.ref.sha256

    exact_validations = [
        artifact
        for artifact in validations
        if artifact.payload.get("candidate_id") == candidate_id
        and artifact.payload.get("candidate_sha256") == candidate_sha256
    ]
    validation = _latest(exact_validations)
    same_candidate_validations = [
        artifact for artifact in validations if artifact.payload.get("candidate_id") == candidate_id
    ]
    if validation is None and same_candidate_validations:
        warnings.append(
            PortalWarning(
                code="VALIDATION_CANDIDATE_DIGEST_MISMATCH",
                message=f"formal validation binds stale candidate bytes for task {task_id}",
                source="promotion",
                severity=WarningSeverity.ERROR,
                task_id=task_id,
            )
        )

    gate = None
    if validation is not None:
        gate = _latest(
            [
                artifact
                for artifact in gates
                if artifact.payload.get("candidate_id") == candidate_id
                and artifact.payload.get("candidate_sha256") == candidate_sha256
                and artifact.payload.get("validation_sha256") == validation.ref.sha256
            ]
        )
        same_round_gates = [
            artifact
            for artifact in gates
            if artifact.payload.get("candidate_id") == candidate_id
            and artifact.payload.get("candidate_sha256") == candidate_sha256
        ]
        if gate is None and same_round_gates:
            warnings.append(
                PortalWarning(
                    code="GATE_VALIDATION_DIGEST_MISMATCH",
                    message=f"ModificationGate evidence does not bind the current validation bytes for task {task_id}",
                    source="promotion",
                    severity=WarningSeverity.ERROR,
                    task_id=task_id,
                )
            )

    exact_receipts: list[_LoadedArtifact] = []
    for artifact in receipts:
        bindings = artifact.payload.get("bindings")
        if (
            artifact.payload.get("candidate_id") == candidate_id
            and isinstance(bindings, Mapping)
            and bindings.get("candidate_sha256") == candidate_sha256
        ):
            exact_receipts.append(artifact)
    receipt = _latest(exact_receipts)
    same_candidate_receipts = [
        artifact for artifact in receipts if artifact.payload.get("candidate_id") == candidate_id
    ]
    if receipt is None and same_candidate_receipts:
        warnings.append(
            PortalWarning(
                code="RECEIPT_CANDIDATE_DIGEST_MISMATCH",
                message=f"promotion receipt binds stale candidate bytes for task {task_id}",
                source="promotion",
                severity=WarningSeverity.ERROR,
                task_id=task_id,
            )
        )
    return _PromotionBundle(candidate, validation, gate, receipt)


def _latest(artifacts: Sequence[_LoadedArtifact]) -> _LoadedArtifact | None:
    return max(
        artifacts,
        key=lambda artifact: (_created_at(artifact.payload), artifact.ref.locator),
        default=None,
    )


def _created_at(payload: Mapping[str, Any]) -> str:
    value = payload.get("created_at")
    return str(value) if isinstance(value, str) else ""


def _latest_timestamp(*values: str | None) -> str | None:
    present = [value for value in values if value]
    return max(present) if present else None


def _require_str(payload: Mapping[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {key} must be a non-empty string")
    return value


def _require_mapping(payload: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: {key} must be an object")
    return value


def _require_sequence(payload: Mapping[str, Any], key: str, path: Path) -> Sequence[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {key} must be an array")
    return value


def _same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).expanduser().resolve() == expected.expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable(path: Path, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    revision = completed.stdout.strip()
    if not revision:
        raise ValueError("git rev-parse returned an empty revision")
    return revision


def _malformed_warning(path: Path, exc: Exception, *, source: str) -> PortalWarning:
    return PortalWarning(
        code="MALFORMED_ARTIFACT",
        message=f"{path}: {exc}",
        source=source,
        severity=WarningSeverity.ERROR,
    )


def _source_for_version(version: str) -> str:
    if "opportunity" in version:
        return "opportunities"
    if (
        "request" in version
        or "approval" in version
        or "control-event" in version
        or "external-research-handoff" in version
    ):
        return "control"
    if version == "forge-research-task.v1":
        return "tasks"
    return "promotion"


def _source_for_kind(kind: str) -> str:
    lowered = kind.lower()
    if "praxist" in lowered:
        return "praxist"
    if "opportunity" in lowered:
        return "opportunities"
    if "request" in lowered or "approval" in lowered or "event" in lowered or "external handoff" in lowered:
        return "control"
    if "task" in lowered:
        return "tasks"
    return "promotion"
