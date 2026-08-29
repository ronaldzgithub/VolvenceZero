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
    "forge-research-approval.v1": "approval_id",
    "forge-research-control-event.v1": "event_id",
    "forge-research-candidate.v1": "candidate_id",
    "forge-research-promotion-receipt.v1": "receipt_id",
}


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
        for task in tasks:
            task_id = _require_str(task.payload, "task_id", task.path)
            item_warnings: list[PortalWarning] = []
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
            warnings.extend(item_warnings)

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
    ) -> _ControlBundle:
        root = self.repo_root / "artifacts" / "research_control" / task_id
        requests: list[_LoadedArtifact] = []
        for path in sorted(root.glob("*/request.json")):
            artifact = self._load_artifact(path, "forge-research-request.v1", "research request", warnings)
            if artifact is not None:
                requests.append(artifact)
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
        root = self.repo_root / "artifacts" / "research_promotion"
        versions = {
            "forge-research-candidate.v1": "candidate",
            "forge-research-validation.v1": "validation",
            "forge-research-gate.v1": "gate",
            "forge-research-promotion-receipt.v1": "receipt",
        }
        by_task: dict[str, dict[str, list[_LoadedArtifact]]] = {}
        for path in sorted(root.glob("**/*.json")):
            raw = self._load_json_object(path, "research promotion", warnings)
            if raw is None:
                continue
            version = raw.get("schema_version")
            kind = versions.get(version) if isinstance(version, str) else None
            if kind is None:
                warnings.append(
                    PortalWarning(
                        code="UNSUPPORTED_PROMOTION_ARTIFACT",
                        message=f"unsupported schema_version in {_portable(path, self.repo_root)}",
                        source="promotion",
                    )
                )
                continue
            task_id = raw.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                warnings.append(_malformed_warning(path, ValueError("task_id must be a string"), source="promotion"))
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
            result[task_id] = _PromotionBundle(
                candidate=_latest(kinds.get("candidate", [])),
                validation=_latest(kinds.get("validation", [])),
                gate=_latest(kinds.get("gate", [])),
                receipt=_latest(kinds.get("receipt", [])),
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

        authority = _authority(control, promotion, task.payload)
        lifecycle = _lifecycle(
            control=control,
            promotion=promotion,
            active_run=run,
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
                )
                if artifact is not None
            ),
        )
        return ResearchLabItem(
            item_id=f"research-lab-item:{task_id}",
            task_id=task_id,
            claim_id=_require_str(task.payload, "claim_id", task.path),
            title=title,
            objective=_require_str(task.payload, "objective", task.path),
            owner=_require_str(task.payload, "owner", task.path),
            capability_axes=axes,
            release_target=str(release.get("target", "unknown")),
            lifecycle=lifecycle,
            authority=authority,
            evidence=_evidence(control, promotion, run),
            bindings=tuple(refs),
            run=run,
            available_actions=_actions(lifecycle.stage, control, promotion),
            warnings=tuple(warnings),
            updated_at=updated_at,
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
            pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
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
    active_run: PraxistRunSnapshot | None,
    duplicate_active_runs: bool,
) -> LifecycleSnapshot:
    if duplicate_active_runs:
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            "multiple active Praxist runs map to one Task",
            active_run.updated_at if active_run else None,
        )
    if control.approval is not None and control.approval.payload.get("decision") == "REJECT":
        return LifecycleSnapshot(
            LifecycleStage.BLOCKED,
            None,
            "exact A0 review rejected this ResearchRequest",
            _created_at(control.approval.payload),
        )
    if active_run is not None:
        return LifecycleSnapshot(
            LifecycleStage.RESEARCH_RUNNING,
            LifecycleStage.RESEARCH_COMPLETE,
            None,
            active_run.started_at,
        )
    if promotion.receipt is not None:
        payload = promotion.receipt.payload
        if payload.get("outcome") == "BLOCKED":
            reasons = payload.get("blocking_reasons")
            reason = "; ".join(str(value) for value in reasons) if isinstance(reasons, list) else "promotion blocked"
            return LifecycleSnapshot(LifecycleStage.BLOCKED, None, reason, _created_at(payload))
        transition = payload.get("transition")
        action = payload.get("action")
        resulting = transition.get("resulting_wiring") if isinstance(transition, Mapping) else None
        if action == "rollback" and resulting == "disabled":
            return LifecycleSnapshot(LifecycleStage.ROLLED_BACK, None, None, _created_at(payload))
        if resulting == "active":
            return LifecycleSnapshot(
                LifecycleStage.AWAITING_A2,
                LifecycleStage.ACTIVE,
                "target adapter apply receipt is not present",
                _created_at(payload),
            )
        if resulting == "shadow":
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
) -> EvidenceSnapshot:
    development = "running" if run is not None else "baseline_registered"
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
) -> tuple[str, ...]:
    if stage is LifecycleStage.AWAITING_A0 and control.request is not None:
        return ("review_a0",)
    if stage is LifecycleStage.PREFLIGHT and _is_a0_approved(control):
        return ("reconcile",)
    if stage is LifecycleStage.RESEARCH_RUNNING:
        return ("view_run",)
    if stage is LifecycleStage.RESEARCH_COMPLETE:
        return ("inspect_handoff",)
    if stage is LifecycleStage.CANDIDATE_RETAINED:
        return ("run_formal_validation",)
    if stage is LifecycleStage.FORMAL_VALIDATION:
        return ("view_formal_evidence",)
    if stage is LifecycleStage.AWAITING_A1 and promotion.receipt is None:
        return ("authorize_shadow",)
    if stage is LifecycleStage.AWAITING_A2:
        return ("authorize_active",)
    if stage in {LifecycleStage.SHADOW, LifecycleStage.ACTIVE}:
        return ("rollback",)
    if stage is LifecycleStage.BLOCKED:
        return ("inspect_blocker",)
    return ()


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


def _latest(artifacts: Sequence[_LoadedArtifact]) -> _LoadedArtifact | None:
    return max(artifacts, key=lambda artifact: _created_at(artifact.payload), default=None)


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
    if "request" in version or "approval" in version or "control-event" in version:
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
    if "request" in lowered or "approval" in lowered or "event" in lowered:
        return "control"
    if "task" in lowered:
        return "tasks"
    return "promotion"
