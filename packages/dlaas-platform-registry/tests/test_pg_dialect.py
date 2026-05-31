"""D2: SQLite→Postgres dialect translation + backend selection.

The Postgres registry backend reuses the existing store SQL verbatim and
translates it at the connection boundary. These tests pin the *pure*
translation (placeholder + upsert rewrite) and the backend resolver. The
live psycopg path is only exercisable with the driver + a Postgres
instance installed; without them, constructing the backend must fail
loud (asserted here) rather than silently degrade.
"""

from __future__ import annotations

import pytest

from dlaas_platform_registry.db import (
    Registry,
    resolve_registry_backend,
)
from dlaas_platform_registry.pg_dialect import (
    TABLE_PRIMARY_KEYS,
    translate_placeholders,
    translate_statement,
    translate_upsert,
)


# ---------------------------------------------------------------------------
# Placeholder translation
# ---------------------------------------------------------------------------


def test_translate_placeholders_basic() -> None:
    assert (
        translate_placeholders("SELECT * FROM t WHERE a = ? AND b = ?")
        == "SELECT * FROM t WHERE a = %s AND b = %s"
    )


def test_translate_placeholders_preserves_question_mark_in_string_literal() -> None:
    sql = "INSERT INTO t (q) VALUES (?) -- but 'why? really?' stays"
    out = translate_placeholders(sql)
    assert "(%s)" in out
    assert "'why? really?'" in out  # literal '?' untouched


# ---------------------------------------------------------------------------
# Upsert rewrite
# ---------------------------------------------------------------------------


def test_translate_upsert_single_pk() -> None:
    sql = (
        "INSERT OR REPLACE INTO tenants (tenant_id, tenant_name, created_at_ms) "
        "VALUES (?, ?, ?)"
    )
    out = translate_upsert(sql)
    assert out.startswith("INSERT INTO tenants")
    assert "ON CONFLICT (tenant_id)" in out
    assert "tenant_name = EXCLUDED.tenant_name" in out
    assert "created_at_ms = EXCLUDED.created_at_ms" in out
    # PK column is not in the SET clause.
    assert "tenant_id = EXCLUDED.tenant_id" not in out


def test_translate_upsert_composite_pk() -> None:
    sql = (
        "INSERT OR REPLACE INTO training_jobs (job_id, ai_id, status) "
        "VALUES (?, ?, ?)"
    )
    out = translate_upsert(sql)
    assert "ON CONFLICT (ai_id, job_id)" in out
    assert "status = EXCLUDED.status" in out


def test_translate_upsert_passthrough_for_plain_insert() -> None:
    sql = "INSERT INTO tenants (tenant_id) VALUES (?)"
    assert translate_upsert(sql) == sql


def test_translate_upsert_unknown_table_raises() -> None:
    sql = "INSERT OR REPLACE INTO mystery (a, b) VALUES (?, ?)"
    with pytest.raises(ValueError):
        translate_upsert(sql)


def test_translate_statement_combines_upsert_and_placeholders() -> None:
    sql = "INSERT OR REPLACE INTO tenants (tenant_id, tenant_name) VALUES (?, ?)"
    out = translate_statement(sql)
    assert "ON CONFLICT (tenant_id)" in out
    assert "%s" in out
    assert "?" not in out


def test_all_known_tables_have_primary_keys() -> None:
    # Guards against a new INSERT OR REPLACE table being added without a
    # conflict target (which would crash only at runtime on Postgres).
    assert "training_jobs" in TABLE_PRIMARY_KEYS
    assert TABLE_PRIMARY_KEYS["governance_records"] == ("record_kind", "record_id")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_resolve_registry_backend_defaults_sqlite(monkeypatch) -> None:
    monkeypatch.delenv("DLAAS_REGISTRY_BACKEND", raising=False)
    assert resolve_registry_backend() == "sqlite"


def test_resolve_registry_backend_env_postgres(monkeypatch) -> None:
    monkeypatch.setenv("DLAAS_REGISTRY_BACKEND", "postgres")
    assert resolve_registry_backend() == "postgres"


def test_resolve_registry_backend_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_registry_backend("mongo")


def test_sqlite_registry_still_default_and_functional() -> None:
    reg = Registry()
    assert reg.backend == "sqlite"
    reg.conn.execute(
        "INSERT OR REPLACE INTO tenants "
        "(tenant_id, tenant_name, contact_email, api_key, api_secret_hash, created_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("t1", "Tenant One", "e@x.com", "key1", "hash1", 1),
    )
    row = reg.conn.execute(
        "SELECT tenant_name FROM tenants WHERE tenant_id = ?", ("t1",)
    ).fetchone()
    assert row["tenant_name"] == "Tenant One"
    reg.close()


def test_postgres_backend_without_driver_fails_loud() -> None:
    try:
        import psycopg  # type: ignore  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError) as exc:
            Registry(backend="postgres", dsn="postgresql://localhost/x")
        assert "psycopg" in str(exc.value)
    else:  # pragma: no cover - only when a driver is installed
        pytest.skip("psycopg installed; live Postgres path not exercised here")
