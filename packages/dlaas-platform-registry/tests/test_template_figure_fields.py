"""Smoke tests for the F4.1 figure-template fields.

Validates:

* :class:`TemplateSpec` round-trips the four new fields through
  ``to_json`` / ``from_json`` symmetrically.
* :class:`TemplateStore.create` persists the figure fields to
  SQLite (in-memory) and ``get`` returns them unchanged.
* :meth:`TemplateStore.patch` updates each field independently and
  bumps the version on every patch.
* Snapshots include the figure fields in their immutable payload.
* Schema migration adds the new columns to a pre-existing v1 DB.
* The :attr:`TemplateSpec.has_figure_artifact` helper reports
  presence accurately.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from dlaas_platform_contracts import (
    CitationPolicy,
    CoveragePolicy,
    TemplateSpec,
    TemplateStatus,
    TemplateActivationStatus,
)
from dlaas_platform_registry.db import (
    Registry,
    SCHEMA_VERSION,
    init_schema,
    open_connection,
)
from dlaas_platform_registry.tenants import TenantStore
from dlaas_platform_registry.templates import TemplateStore


def test_template_spec_round_trips_figure_fields() -> None:
    spec = TemplateSpec(
        template_id="tpl_001",
        tenant_id="tnt_a",
        template_name="Einstein companion",
        figure_artifact_id="figure-bundle:einstein:abc123",
        citation_policy=CitationPolicy.REQUIRED,
        coverage_policy=CoveragePolicy.STRICT_REFUSE,
        figure_time_window="late-1925-1955",
    )
    payload = spec.to_json()
    assert payload["figure_artifact_id"] == "figure-bundle:einstein:abc123"
    assert payload["citation_policy"] == "required"
    assert payload["coverage_policy"] == "strict_refuse"
    assert payload["figure_time_window"] == "late-1925-1955"
    revived = TemplateSpec.from_json(payload)
    assert revived == spec


def test_template_spec_invalid_citation_policy_raises() -> None:
    bad_payload = {
        "tenant_id": "tnt_a",
        "template_name": "x",
        "citation_policy": "not-a-policy",
    }
    with pytest.raises(ValueError, match="citation_policy"):
        TemplateSpec.from_json(bad_payload)


def test_template_spec_invalid_coverage_policy_raises() -> None:
    bad_payload = {
        "tenant_id": "tnt_a",
        "template_name": "x",
        "coverage_policy": "not-a-policy",
    }
    with pytest.raises(ValueError, match="coverage_policy"):
        TemplateSpec.from_json(bad_payload)


def test_template_spec_has_figure_artifact_helper() -> None:
    spec = TemplateSpec(
        template_id="tpl_001",
        tenant_id="tnt_a",
        template_name="x",
        figure_artifact_id="figure-bundle:einstein:abc",
    )
    no_artifact = TemplateSpec(
        template_id="tpl_002",
        tenant_id="tnt_a",
        template_name="y",
    )
    assert spec.has_figure_artifact is True
    assert no_artifact.has_figure_artifact is False


async def _seed_registry(registry: Registry, tenant_id: str) -> None:
    tenants = TenantStore(registry)
    await tenants.create(
        tenant_id=tenant_id,
        tenant_name="Test Tenant",
        contact_email="t@t.com",
    )


def _build_registry() -> Registry:
    return Registry(db_path=":memory:")


def test_template_store_persists_figure_fields() -> None:
    async def run() -> None:
        registry = _build_registry()
        await _seed_registry(registry, "tnt_a")
        store = TemplateStore(registry)
        spec = await store.create(
            tenant_id="tnt_a",
            template_name="Einstein companion",
            figure_artifact_id="figure-bundle:einstein:abc",
            citation_policy=CitationPolicy.REQUIRED,
            coverage_policy=CoveragePolicy.STRICT_REFUSE,
            figure_time_window="late-1925-1955",
        )
        fetched = await store.get(spec.template_id)
        assert fetched.figure_artifact_id == "figure-bundle:einstein:abc"
        assert fetched.citation_policy is CitationPolicy.REQUIRED
        assert fetched.coverage_policy is CoveragePolicy.STRICT_REFUSE
        assert fetched.figure_time_window == "late-1925-1955"
        registry.close()

    asyncio.run(run())


def test_template_store_patch_updates_figure_fields() -> None:
    async def run() -> None:
        registry = _build_registry()
        await _seed_registry(registry, "tnt_a")
        store = TemplateStore(registry)
        spec = await store.create(
            tenant_id="tnt_a",
            template_name="Einstein companion",
        )
        assert spec.figure_artifact_id == ""
        assert spec.citation_policy is CitationPolicy.DISABLED
        patched = await store.patch(
            template_id=spec.template_id,
            figure_artifact_id="figure-bundle:einstein:def",
            citation_policy=CitationPolicy.PREFERRED,
            coverage_policy=CoveragePolicy.SOFT_DISCLAIM,
            figure_time_window="early-1905-1925",
        )
        assert patched.figure_artifact_id == "figure-bundle:einstein:def"
        assert patched.citation_policy is CitationPolicy.PREFERRED
        assert patched.coverage_policy is CoveragePolicy.SOFT_DISCLAIM
        assert patched.figure_time_window == "early-1905-1925"
        assert patched.current_version == spec.current_version + 1
        # Fields not passed in patch must remain unchanged.
        partial = await store.patch(
            template_id=spec.template_id,
            description="figure of foundations of mechanics",
        )
        assert partial.figure_artifact_id == "figure-bundle:einstein:def"
        assert partial.citation_policy is CitationPolicy.PREFERRED
        registry.close()

    asyncio.run(run())


def test_template_snapshot_carries_figure_fields() -> None:
    async def run() -> None:
        registry = _build_registry()
        await _seed_registry(registry, "tnt_a")
        store = TemplateStore(registry)
        spec = await store.create(
            tenant_id="tnt_a",
            template_name="Einstein companion",
            figure_artifact_id="figure-bundle:einstein:abc",
            citation_policy=CitationPolicy.REQUIRED,
            coverage_policy=CoveragePolicy.STRICT_REFUSE,
            figure_time_window="late-1925-1955",
        )
        versions = await store.list_versions(template_id=spec.template_id)
        assert versions
        snapshot = versions[-1].snapshot
        assert snapshot["figure_artifact_id"] == "figure-bundle:einstein:abc"
        assert snapshot["citation_policy"] == "required"
        assert snapshot["coverage_policy"] == "strict_refuse"
        assert snapshot["figure_time_window"] == "late-1925-1955"
        registry.close()

    asyncio.run(run())


def test_schema_migration_adds_figure_columns_to_v1_db(tmp_path) -> None:
    """Forward migration adds the four columns to a pre-existing v1 DB."""

    db_path = tmp_path / "old_v1.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    conn.execute(
        """
        CREATE TABLE tenants (
            tenant_id TEXT PRIMARY KEY,
            tenant_name TEXT NOT NULL,
            contact_email TEXT NOT NULL,
            business_type TEXT NOT NULL DEFAULT 'generic',
            billing_plan TEXT NOT NULL DEFAULT 'pay_as_you_go',
            quota_json TEXT NOT NULL DEFAULT '{}',
            api_key TEXT NOT NULL UNIQUE,
            api_secret_hash TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE templates (
            template_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            template_name TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'generic',
            description TEXT NOT NULL DEFAULT '',
            runtime_template_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft',
            current_version INTEGER NOT NULL DEFAULT 1,
            activation_status TEXT NOT NULL DEFAULT 'unactivated',
            base_persona_json TEXT NOT NULL DEFAULT '{}',
            persona_spec_json TEXT NOT NULL DEFAULT '{}',
            seed_config_json TEXT NOT NULL DEFAULT '{}',
            activation_stats_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1')"
    )
    conn.commit()
    conn.close()

    # Open the v1 DB through the upgraded init_schema; columns must
    # be added without losing existing rows.
    new_conn = open_connection(db_path)
    init_schema(new_conn)
    columns = {row[1] for row in new_conn.execute("PRAGMA table_info(templates)")}
    assert {
        "figure_artifact_id",
        "citation_policy",
        "coverage_policy",
        "figure_time_window",
    }.issubset(columns)
    schema_value = new_conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert schema_value == str(SCHEMA_VERSION)
    new_conn.close()
