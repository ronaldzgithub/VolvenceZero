"""Template + Version CRUD store.

Templates are reusable persona + seed bundles. Each ``PATCH`` to a
template (or an explicit ``snapshot`` call) creates an immutable
:class:`TemplateVersionSpec`. Adoption pins a contract to one
``(template_id, template_version)`` pair forever.

Activation status is tracked on the template row directly because
activation is a property of the **current** state of the persona +
seed bundle; historical versions retain the activation snapshot they
were activated against in their immutable ``snapshot_json``.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    CitationPolicy,
    CoveragePolicy,
    TemplateActivationStatus,
    TemplateSpec,
    TemplateStatus,
    TemplateVersionSpec,
)

from dlaas_platform_registry.db import Registry


class TemplateNotFound(LookupError):
    pass


class TemplateVersionNotFound(LookupError):
    pass


def _fresh_template_id() -> str:
    return f"tpl_{secrets.token_hex(4)}"


def _fresh_version_id() -> str:
    return f"tplv_{secrets.token_hex(4)}"


class TemplateStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        tenant_id: str,
        template_name: str,
        domain: str = "generic",
        description: str = "",
        runtime_template_id: str = "",
        base_persona: Mapping[str, Any] | None = None,
        persona_spec: Mapping[str, Any] | None = None,
        seed_config: Mapping[str, Any] | None = None,
        template_id: str | None = None,
        figure_artifact_id: str = "",
        citation_policy: CitationPolicy = CitationPolicy.DISABLED,
        coverage_policy: CoveragePolicy = CoveragePolicy.PASSTHROUGH,
        figure_time_window: str = "",
    ) -> TemplateSpec:
        template_id = template_id or _fresh_template_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO templates (
                    template_id, tenant_id, template_name, domain,
                    description, runtime_template_id, status,
                    current_version, activation_status,
                    base_persona_json, persona_spec_json,
                    seed_config_json, activation_stats_json,
                    created_at_ms,
                    figure_artifact_id, citation_policy,
                    coverage_policy, figure_time_window
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    template_id,
                    tenant_id,
                    template_name,
                    domain,
                    description,
                    runtime_template_id,
                    TemplateStatus.DRAFT.value,
                    1,
                    TemplateActivationStatus.UNACTIVATED.value,
                    json.dumps(dict(base_persona or {})),
                    json.dumps(dict(persona_spec or {})),
                    json.dumps(dict(seed_config or {})),
                    json.dumps({}),
                    created_at_ms,
                    figure_artifact_id,
                    citation_policy.value,
                    coverage_policy.value,
                    figure_time_window,
                ),
            )
        spec = TemplateSpec(
            template_id=template_id,
            tenant_id=tenant_id,
            template_name=template_name,
            domain=domain,
            description=description,
            runtime_template_id=runtime_template_id,
            status=TemplateStatus.DRAFT,
            current_version=1,
            activation_status=TemplateActivationStatus.UNACTIVATED,
            base_persona=dict(base_persona or {}),
            persona_spec=dict(persona_spec or {}),
            seed_config=dict(seed_config or {}),
            activation_stats={},
            created_at_ms=created_at_ms,
            figure_artifact_id=figure_artifact_id,
            citation_policy=citation_policy,
            coverage_policy=coverage_policy,
            figure_time_window=figure_time_window,
        )
        # Snapshot version 1 immediately so adoption / readiness can refer to it.
        await self._snapshot_locked(spec=spec, version_note="initial")
        return spec

    async def get(self, template_id: str) -> TemplateSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM templates WHERE template_id = ?", (template_id,)
        ).fetchone()
        if row is None:
            raise TemplateNotFound(template_id)
        return _row_to_spec(row)

    async def list_for_tenant(self, *, tenant_id: str) -> tuple[TemplateSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM templates WHERE tenant_id = ? "
            "ORDER BY created_at_ms ASC",
            (tenant_id,),
        ).fetchall()
        return tuple(_row_to_spec(row) for row in rows)

    async def patch(
        self,
        *,
        template_id: str,
        template_name: str | None = None,
        description: str | None = None,
        domain: str | None = None,
        runtime_template_id: str | None = None,
        status: TemplateStatus | None = None,
        base_persona: Mapping[str, Any] | None = None,
        persona_spec: Mapping[str, Any] | None = None,
        seed_config: Mapping[str, Any] | None = None,
        figure_artifact_id: str | None = None,
        citation_policy: CitationPolicy | None = None,
        coverage_policy: CoveragePolicy | None = None,
        figure_time_window: str | None = None,
        version_note: str = "",
    ) -> TemplateSpec:
        """Apply field updates and create a new immutable version.

        ``status`` transitions are validated minimally — moving to
        ``PUBLISHED`` requires a non-empty ``runtime_template_id`` and
        ``activation_status == ACTIVATED``. Slice 3.4 wires the
        readiness gate; this method enforces only the structural
        prerequisites so the registry stays consistent.
        """
        current = await self.get(template_id)
        next_version = current.current_version + 1
        async with self._registry.write_lock:
            new_name = template_name or current.template_name
            new_description = (
                description if description is not None else current.description
            )
            new_domain = domain or current.domain
            new_runtime_template_id = (
                runtime_template_id
                if runtime_template_id is not None
                else current.runtime_template_id
            )
            new_status = status or current.status
            new_base_persona = (
                dict(base_persona)
                if base_persona is not None
                else dict(current.base_persona)
            )
            new_persona_spec = (
                dict(persona_spec)
                if persona_spec is not None
                else dict(current.persona_spec)
            )
            new_seed_config = (
                dict(seed_config)
                if seed_config is not None
                else dict(current.seed_config)
            )
            new_figure_artifact_id = (
                figure_artifact_id
                if figure_artifact_id is not None
                else current.figure_artifact_id
            )
            new_citation_policy = (
                citation_policy
                if citation_policy is not None
                else current.citation_policy
            )
            new_coverage_policy = (
                coverage_policy
                if coverage_policy is not None
                else current.coverage_policy
            )
            new_figure_time_window = (
                figure_time_window
                if figure_time_window is not None
                else current.figure_time_window
            )
            if new_status is TemplateStatus.PUBLISHED:
                if not new_runtime_template_id.strip():
                    raise ValueError(
                        "Cannot publish template without runtime_template_id"
                    )
                if (
                    current.activation_status
                    is not TemplateActivationStatus.ACTIVATED
                ):
                    raise ValueError(
                        "Cannot publish template before activation succeeds"
                    )
            self._registry.conn.execute(
                """
                UPDATE templates SET
                    template_name = ?,
                    description = ?,
                    domain = ?,
                    runtime_template_id = ?,
                    status = ?,
                    current_version = ?,
                    base_persona_json = ?,
                    persona_spec_json = ?,
                    seed_config_json = ?,
                    figure_artifact_id = ?,
                    citation_policy = ?,
                    coverage_policy = ?,
                    figure_time_window = ?
                WHERE template_id = ?
                """,
                (
                    new_name,
                    new_description,
                    new_domain,
                    new_runtime_template_id,
                    new_status.value,
                    next_version,
                    json.dumps(new_base_persona),
                    json.dumps(new_persona_spec),
                    json.dumps(new_seed_config),
                    new_figure_artifact_id,
                    new_citation_policy.value,
                    new_coverage_policy.value,
                    new_figure_time_window,
                    template_id,
                ),
            )
        new_spec = TemplateSpec(
            template_id=template_id,
            tenant_id=current.tenant_id,
            template_name=new_name,
            domain=new_domain,
            description=new_description,
            runtime_template_id=new_runtime_template_id,
            status=new_status,
            current_version=next_version,
            activation_status=current.activation_status,
            base_persona=new_base_persona,
            persona_spec=new_persona_spec,
            seed_config=new_seed_config,
            activation_stats=dict(current.activation_stats),
            created_at_ms=current.created_at_ms,
            figure_artifact_id=new_figure_artifact_id,
            citation_policy=new_citation_policy,
            coverage_policy=new_coverage_policy,
            figure_time_window=new_figure_time_window,
        )
        await self._snapshot_locked(spec=new_spec, version_note=version_note)
        return new_spec

    async def update_activation(
        self,
        *,
        template_id: str,
        activation_status: TemplateActivationStatus,
        activation_stats: Mapping[str, Any] | None = None,
    ) -> TemplateSpec:
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE templates SET activation_status = ?, "
                "activation_stats_json = ? WHERE template_id = ?",
                (
                    activation_status.value,
                    json.dumps(dict(activation_stats or {})),
                    template_id,
                ),
            )
        return await self.get(template_id)

    async def snapshot(
        self, *, template_id: str, version_note: str = ""
    ) -> TemplateVersionSpec:
        spec = await self.get(template_id)
        return await self._snapshot_locked(spec=spec, version_note=version_note)

    async def _snapshot_locked(
        self, *, spec: TemplateSpec, version_note: str
    ) -> TemplateVersionSpec:
        version_id = _fresh_version_id()
        created_at_ms = int(time.time() * 1000.0)
        snapshot = spec.to_json()
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO template_versions (
                    version_id, template_id, version_number,
                    snapshot_json, version_note, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(template_id, version_number) DO UPDATE SET
                    snapshot_json = excluded.snapshot_json,
                    version_note = excluded.version_note
                """,
                (
                    version_id,
                    spec.template_id,
                    spec.current_version,
                    json.dumps(snapshot),
                    version_note,
                    created_at_ms,
                ),
            )
        return TemplateVersionSpec(
            version_id=version_id,
            template_id=spec.template_id,
            version_number=spec.current_version,
            snapshot=snapshot,
            version_note=version_note,
            created_at_ms=created_at_ms,
        )

    async def list_versions(
        self, *, template_id: str
    ) -> tuple[TemplateVersionSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM template_versions WHERE template_id = ? "
            "ORDER BY version_number ASC",
            (template_id,),
        ).fetchall()
        return tuple(
            TemplateVersionSpec(
                version_id=row["version_id"],
                template_id=row["template_id"],
                version_number=int(row["version_number"]),
                snapshot=json.loads(row["snapshot_json"] or "{}"),
                version_note=row["version_note"],
                created_at_ms=int(row["created_at_ms"]),
            )
            for row in rows
        )

    async def get_version(
        self, *, template_id: str, version_number: int
    ) -> TemplateVersionSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM template_versions WHERE template_id = ? "
            "AND version_number = ?",
            (template_id, version_number),
        ).fetchone()
        if row is None:
            raise TemplateVersionNotFound(f"{template_id}@{version_number}")
        return TemplateVersionSpec(
            version_id=row["version_id"],
            template_id=row["template_id"],
            version_number=int(row["version_number"]),
            snapshot=json.loads(row["snapshot_json"] or "{}"),
            version_note=row["version_note"],
            created_at_ms=int(row["created_at_ms"]),
        )


def _row_to_spec(row) -> TemplateSpec:
    keys = row.keys() if hasattr(row, "keys") else ()
    figure_artifact_id = row["figure_artifact_id"] if "figure_artifact_id" in keys else ""
    citation_policy_raw = (
        row["citation_policy"]
        if "citation_policy" in keys
        else CitationPolicy.DISABLED.value
    )
    coverage_policy_raw = (
        row["coverage_policy"]
        if "coverage_policy" in keys
        else CoveragePolicy.PASSTHROUGH.value
    )
    figure_time_window = (
        row["figure_time_window"] if "figure_time_window" in keys else ""
    )
    return TemplateSpec(
        template_id=row["template_id"],
        tenant_id=row["tenant_id"],
        template_name=row["template_name"],
        domain=row["domain"],
        description=row["description"],
        runtime_template_id=row["runtime_template_id"],
        status=TemplateStatus(row["status"]),
        current_version=int(row["current_version"]),
        activation_status=TemplateActivationStatus(row["activation_status"]),
        base_persona=json.loads(row["base_persona_json"] or "{}"),
        persona_spec=json.loads(row["persona_spec_json"] or "{}"),
        seed_config=json.loads(row["seed_config_json"] or "{}"),
        activation_stats=json.loads(row["activation_stats_json"] or "{}"),
        created_at_ms=int(row["created_at_ms"]),
        figure_artifact_id=figure_artifact_id or "",
        citation_policy=CitationPolicy(
            citation_policy_raw or CitationPolicy.DISABLED.value
        ),
        coverage_policy=CoveragePolicy(
            coverage_policy_raw or CoveragePolicy.PASSTHROUGH.value
        ),
        figure_time_window=figure_time_window or "",
    )


__all__ = [
    "TemplateNotFound",
    "TemplateStore",
    "TemplateVersionNotFound",
]
