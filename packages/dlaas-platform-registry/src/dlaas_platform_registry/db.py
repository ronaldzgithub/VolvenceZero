"""SQLite-backed registry connection + schema bootstrap.

The registry is a thin CRUD layer over the typed resource specs in
``dlaas-platform-contracts``. Every cross-table read returns frozen
spec dataclasses; the platform never exposes raw row tuples past this
boundary.

Backend choice: stdlib :mod:`sqlite3` keeps the dependency footprint
zero and is more than adequate for control-plane CRUD volumes
(tenants / templates / contracts churn slowly). Slice 7+ can swap in
Postgres by replacing this module — the typed store API stays
identical.

Concurrency model: SQLite is opened in WAL mode + ``check_same_thread=False``
so the same connection can be reused from aiohttp event-loop handlers
across coroutines. A single :class:`asyncio.Lock` (held on the
:class:`Registry` instance) serialises writes; reads run without the
lock because SQLite's WAL gives us snapshot consistency. This keeps
the registry safe under aiohttp's single-event-loop concurrency
without paying executor-thread overhead per call.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

from dlaas_platform_registry.pg_dialect import translate_statement

SCHEMA_VERSION = 8


_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tenants (
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
    """,
    """
    CREATE TABLE IF NOT EXISTS shells (
        shell_id TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        shell_kind TEXT NOT NULL,
        shell_type TEXT NOT NULL DEFAULT 'generic',
        display_name TEXT NOT NULL DEFAULT '',
        embodiment_json TEXT NOT NULL DEFAULT '{}',
        channel_json TEXT NOT NULL DEFAULT '{}',
        scene_meta_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        PRIMARY KEY (tenant_id, shell_id),
        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS assets (
        asset_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        asset_type TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        uri TEXT NOT NULL DEFAULT '',
        mime_type TEXT NOT NULL DEFAULT '',
        language TEXT NOT NULL DEFAULT '',
        source_meta_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS templates (
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
        created_at_ms INTEGER NOT NULL,
        figure_artifact_id TEXT NOT NULL DEFAULT '',
        citation_policy TEXT NOT NULL DEFAULT 'disabled',
        coverage_policy TEXT NOT NULL DEFAULT 'passthrough',
        figure_time_window TEXT NOT NULL DEFAULT '',
        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS template_versions (
        version_id TEXT PRIMARY KEY,
        template_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        snapshot_json TEXT NOT NULL,
        version_note TEXT NOT NULL DEFAULT '',
        created_at_ms INTEGER NOT NULL,
        UNIQUE (template_id, version_number),
        FOREIGN KEY (template_id) REFERENCES templates(template_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS template_assets (
        template_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        template_version INTEGER NOT NULL DEFAULT 1,
        role TEXT NOT NULL DEFAULT 'training_material',
        link_meta_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        PRIMARY KEY (template_id, asset_id, template_version),
        FOREIGN KEY (template_id) REFERENCES templates(template_id),
        FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS contracts (
        contract_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        template_id TEXT NOT NULL,
        template_version INTEGER NOT NULL DEFAULT 1,
        shell_id TEXT NOT NULL,
        ai_id TEXT NOT NULL DEFAULT '',
        owner_user_id TEXT NOT NULL DEFAULT '',
        engine_tools_json TEXT NOT NULL DEFAULT '{}',
        tool_policy_snapshot_json TEXT NOT NULL DEFAULT '{}',
        service_contract_json TEXT NOT NULL DEFAULT '{}',
        contract_status TEXT NOT NULL DEFAULT 'created',
        created_at_ms INTEGER NOT NULL,
        plugins_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
        FOREIGN KEY (template_id) REFERENCES templates(template_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS focus_persons (
        contract_id TEXT NOT NULL,
        person_id TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        role TEXT NOT NULL DEFAULT 'user',
        relationship_to_owner TEXT NOT NULL DEFAULT '',
        age INTEGER,
        initial_profile_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        PRIMARY KEY (contract_id, person_id),
        FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS identity_links (
        ai_id TEXT NOT NULL,
        channel_type TEXT NOT NULL,
        channel_ref TEXT NOT NULL,
        canonical_end_user_ref TEXT NOT NULL,
        link_meta_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        PRIMARY KEY (ai_id, channel_type, channel_ref)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS handoff_tickets (
        ticket_id TEXT PRIMARY KEY,
        ai_id TEXT NOT NULL,
        contract_id TEXT NOT NULL,
        end_user_ref TEXT NOT NULL,
        session_id TEXT NOT NULL DEFAULT '',
        trigger_reason TEXT NOT NULL DEFAULT '',
        trigger_details_json TEXT NOT NULL DEFAULT '{}',
        confidence_aggregate REAL NOT NULL DEFAULT 0.0,
        recent_response_ids_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'open',
        operator_id TEXT NOT NULL DEFAULT '',
        human_reply TEXT NOT NULL DEFAULT '',
        resolution_notes TEXT NOT NULL DEFAULT '',
        created_at_ms INTEGER NOT NULL,
        FOREIGN KEY (contract_id) REFERENCES contracts(contract_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS audience_profiles (
        profile_id TEXT PRIMARY KEY,
        template_id TEXT NOT NULL,
        cohort_name TEXT NOT NULL,
        asset_ids_json TEXT NOT NULL DEFAULT '[]',
        common_questions_json TEXT NOT NULL DEFAULT '[]',
        communication_style TEXT NOT NULL DEFAULT '',
        emotion_triggers_json TEXT NOT NULL DEFAULT '[]',
        decision_patterns_json TEXT NOT NULL DEFAULT '[]',
        evidence_stats_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        FOREIGN KEY (template_id) REFERENCES templates(template_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS exam_questions (
        question_id TEXT PRIMARY KEY,
        template_id TEXT NOT NULL,
        scenario_tag TEXT NOT NULL,
        user_prompt TEXT NOT NULL,
        context_json TEXT NOT NULL DEFAULT '{}',
        rubric_json TEXT NOT NULL DEFAULT '[]',
        reference_answer TEXT NOT NULL DEFAULT '',
        tags_json TEXT NOT NULL DEFAULT '[]',
        difficulty TEXT NOT NULL DEFAULT 'medium',
        created_at_ms INTEGER NOT NULL,
        FOREIGN KEY (template_id) REFERENCES templates(template_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS exam_runs (
        run_id TEXT PRIMARY KEY,
        template_id TEXT NOT NULL,
        template_version INTEGER NOT NULL DEFAULT 1,
        run_type TEXT NOT NULL DEFAULT 'launch_gate',
        question_ids_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'pending',
        operator_id TEXT NOT NULL DEFAULT '',
        operator_name TEXT NOT NULL DEFAULT '',
        comment TEXT NOT NULL DEFAULT '',
        ai_id TEXT NOT NULL DEFAULT '',
        contract_id TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL DEFAULT '',
        aggregate_score REAL NOT NULL DEFAULT 0.0,
        pass_threshold REAL NOT NULL DEFAULT 0.6,
        passed INTEGER NOT NULL DEFAULT 0,
        wrong_set_json TEXT NOT NULL DEFAULT '[]',
        submissions_json TEXT NOT NULL DEFAULT '[]',
        created_at_ms INTEGER NOT NULL,
        FOREIGN KEY (template_id) REFERENCES templates(template_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS launch_licenses (
        license_id TEXT PRIMARY KEY,
        template_id TEXT NOT NULL,
        template_version INTEGER NOT NULL DEFAULT 1,
        granted INTEGER NOT NULL DEFAULT 0,
        reason TEXT NOT NULL DEFAULT '',
        granted_by_run_id TEXT NOT NULL DEFAULT '',
        issued_at_ms INTEGER NOT NULL,
        UNIQUE (template_id, template_version),
        FOREIGN KEY (template_id) REFERENCES templates(template_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS governance_records (
        record_kind TEXT NOT NULL,
        record_id TEXT NOT NULL,
        ai_id TEXT NOT NULL DEFAULT '',
        contract_id TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        PRIMARY KEY (record_kind, record_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cultivations (
        cultivation_id TEXT PRIMARY KEY,
        ai_id TEXT NOT NULL DEFAULT '',
        slug TEXT NOT NULL,
        display_name TEXT NOT NULL DEFAULT '',
        domain TEXT NOT NULL DEFAULT '',
        runtime_template_id TEXT NOT NULL DEFAULT '',
        seed_persona_json TEXT NOT NULL DEFAULT '{}',
        curriculum_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'seeding',
        cycles_completed INTEGER NOT NULL DEFAULT 0,
        coherence_score REAL NOT NULL DEFAULT 0.0,
        coherence_detail_json TEXT NOT NULL DEFAULT '{}',
        regime_history_json TEXT NOT NULL DEFAULT '[]',
        dlaas_template_id TEXT NOT NULL DEFAULT '',
        last_exam_run_id TEXT NOT NULL DEFAULT '',
        inducted_template_id TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        package_id TEXT NOT NULL DEFAULT '',
        track_id TEXT NOT NULL DEFAULT '',
        direction_json TEXT NOT NULL DEFAULT '{}',
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS applications (
        application_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT NOT NULL DEFAULT '0.0.0',
        description TEXT NOT NULL DEFAULT '',
        plugins_json TEXT NOT NULL DEFAULT '[]',
        api_key TEXT NOT NULL UNIQUE,
        api_secret_hash TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS org_application_approvals (
        tenant_id TEXT NOT NULL,
        application_id TEXT NOT NULL,
        approved_at_ms INTEGER NOT NULL,
        approved_by_user_id TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (tenant_id, application_id),
        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
        FOREIGN KEY (application_id) REFERENCES applications(application_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS training_jobs (
        job_id TEXT NOT NULL,
        ai_id TEXT NOT NULL,
        contract_id TEXT NOT NULL DEFAULT '',
        tenant_id TEXT NOT NULL DEFAULT '',
        job_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_by TEXT NOT NULL DEFAULT '',
        source_ref TEXT NOT NULL DEFAULT '',
        promotion_gate TEXT NOT NULL DEFAULT '',
        artifact_ref TEXT NOT NULL DEFAULT '',
        gate_evidence_json TEXT NOT NULL DEFAULT '{}',
        notes TEXT NOT NULL DEFAULT '',
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        PRIMARY KEY (ai_id, job_id)
    );
    """,
)


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL + foreign keys enabled.

    Callers should keep the connection alive for the life of the
    registry. ``check_same_thread=False`` lets aiohttp coroutines
    share one connection; the :class:`Registry` lock takes care of
    serialising writes on top of that.
    """
    db_path = Path(db_path)
    if db_path != Path(":memory:"):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path) if db_path != Path(":memory:") else ":memory:",
        check_same_thread=False,
        isolation_level=None,  # autocommit; we use explicit transactions when needed
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if absent, apply forward migrations, and stamp version.

    Idempotent — safe to call on every process start. Forward
    migrations are applied **only** for databases that already exist
    at an earlier schema version; fresh databases get the latest
    columns from the ``CREATE TABLE`` definition above. Each
    migration step is wrapped in a try/except so a column that has
    already been added by a previous run is a no-op.
    """
    for stmt in _SCHEMA_SQL:
        conn.execute(stmt)
    _apply_forward_migrations(conn)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def _apply_forward_migrations(conn: sqlite3.Connection) -> None:
    """Apply forward-only ALTER TABLE deltas for existing databases.

    SQLite's ``ALTER TABLE ... ADD COLUMN`` lacks ``IF NOT EXISTS``,
    so we catch the duplicate-column error and treat it as a no-op.
    Any other ``OperationalError`` is re-raised so contract drift is
    visible immediately rather than masked.
    """

    schema_v2_columns = (
        ("figure_artifact_id", "TEXT NOT NULL DEFAULT ''"),
        ("citation_policy", "TEXT NOT NULL DEFAULT 'disabled'"),
        ("coverage_policy", "TEXT NOT NULL DEFAULT 'passthrough'"),
        ("figure_time_window", "TEXT NOT NULL DEFAULT ''"),
    )
    for column, column_type in schema_v2_columns:
        try:
            conn.execute(
                f"ALTER TABLE templates ADD COLUMN {column} {column_type}"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

    schema_v4_contract_columns = (
        ("plugins_json", "TEXT NOT NULL DEFAULT '[]'"),
    )
    for column, column_type in schema_v4_contract_columns:
        try:
            conn.execute(
                f"ALTER TABLE contracts ADD COLUMN {column} {column_type}"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS governance_records (
            record_kind TEXT NOT NULL,
            record_id TEXT NOT NULL,
            ai_id TEXT NOT NULL DEFAULT '',
            contract_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL,
            PRIMARY KEY (record_kind, record_id)
        );
        """
    )

    # Schema v7: persisted DLaaS training jobs (rare-heavy executor).
    # Forward-create for pre-v7 databases; fresh DBs get it from
    # ``_SCHEMA_SQL`` above.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_jobs (
            job_id TEXT NOT NULL,
            ai_id TEXT NOT NULL,
            contract_id TEXT NOT NULL DEFAULT '',
            tenant_id TEXT NOT NULL DEFAULT '',
            job_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_by TEXT NOT NULL DEFAULT '',
            source_ref TEXT NOT NULL DEFAULT '',
            promotion_gate TEXT NOT NULL DEFAULT '',
            artifact_ref TEXT NOT NULL DEFAULT '',
            gate_evidence_json TEXT NOT NULL DEFAULT '{}',
            notes TEXT NOT NULL DEFAULT '',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (ai_id, job_id)
        );
        """
    )

    # Schema v6: autonomous expert-cultivation records. Forward-create
    # for databases that pre-date the table; fresh DBs already get it
    # from ``_SCHEMA_SQL`` above.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cultivations (
            cultivation_id TEXT PRIMARY KEY,
            ai_id TEXT NOT NULL DEFAULT '',
            slug TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            domain TEXT NOT NULL DEFAULT '',
            runtime_template_id TEXT NOT NULL DEFAULT '',
            seed_persona_json TEXT NOT NULL DEFAULT '{}',
            curriculum_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'seeding',
            cycles_completed INTEGER NOT NULL DEFAULT 0,
            coherence_score REAL NOT NULL DEFAULT 0.0,
            coherence_detail_json TEXT NOT NULL DEFAULT '{}',
            regime_history_json TEXT NOT NULL DEFAULT '[]',
            dlaas_template_id TEXT NOT NULL DEFAULT '',
            last_exam_run_id TEXT NOT NULL DEFAULT '',
            inducted_template_id TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            package_id TEXT NOT NULL DEFAULT '',
            track_id TEXT NOT NULL DEFAULT '',
            direction_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        """
    )

    # Schema v8: multi-direction cultivation packages. Existing
    # single-expert cultivation rows pre-date the package columns; add
    # them forward-only (duplicate-column = no-op) so a seed can fan out
    # into several self-consistent school tracks grouped by package_id.
    cultivation_v8_columns = (
        ("package_id", "TEXT NOT NULL DEFAULT ''"),
        ("track_id", "TEXT NOT NULL DEFAULT ''"),
        ("direction_json", "TEXT NOT NULL DEFAULT '{}'"),
    )
    for column, column_type in cultivation_v8_columns:
        try:
            conn.execute(
                f"ALTER TABLE cultivations ADD COLUMN {column} {column_type}"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


# ---------------------------------------------------------------------------
# Postgres backend (debt D2)
# ---------------------------------------------------------------------------
#
# The registry defaults to SQLite (zero-dependency, unchanged). Setting
# ``DLAAS_REGISTRY_BACKEND=postgres`` + ``DLAAS_REGISTRY_PG_DSN=...`` (or
# passing ``backend="postgres", dsn=...`` to :class:`Registry`) routes the
# *same* store SQL through a translation adapter so the typed store API is
# byte-for-byte identical. The translation (``?`` → ``%s`` and
# ``INSERT OR REPLACE`` → ``ON CONFLICT`` upsert) lives in
# :mod:`dlaas_platform_registry.pg_dialect` and is unit-tested; the live
# psycopg connection is only constructed when the driver is installed.


def resolve_registry_backend(backend: str | None = None) -> str:
    """Return the selected registry backend ('sqlite' | 'postgres')."""

    chosen = (backend or os.environ.get("DLAAS_REGISTRY_BACKEND") or "sqlite").strip().lower()
    if chosen in ("", "sqlite", "sqlite3", "file"):
        return "sqlite"
    if chosen in ("postgres", "postgresql", "pg"):
        return "postgres"
    raise ValueError(
        f"resolve_registry_backend: unknown backend {chosen!r}. "
        "Expected 'sqlite' or 'postgres'."
    )


class _PostgresCursorResult:
    """sqlite3-style result wrapper over a psycopg cursor.

    The stores call ``conn.execute(sql, params).fetchone()/.fetchall()``
    and index rows by column name (``row["col"]``). psycopg's dict-row
    cursor returns ``dict`` rows, which already support ``row["col"]``.
    """

    def __init__(self, rows: list, rowcount: int) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class PostgresConnectionAdapter:
    """Wrap a psycopg connection behind the sqlite3 ``conn.execute`` API.

    Translates each statement via
    :func:`dlaas_platform_registry.pg_dialect.translate_statement` so the
    per-resource stores need no changes. Constructed only when
    ``backend="postgres"``; importing this class never requires psycopg.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Postgres registry backend needs the 'psycopg' driver. "
                "Install it (pip install 'psycopg[binary]') or use the "
                "default SQLite backend (unset DLAAS_REGISTRY_BACKEND)."
            ) from exc
        if not dsn:
            raise ValueError(
                "Postgres registry backend requires a DSN "
                "(DLAAS_REGISTRY_PG_DSN or Registry(dsn=...))."
            )
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)

    def execute(self, sql: str, params: tuple = ()):  # noqa: D401
        translated = translate_statement(sql)
        cur = self._conn.cursor()
        cur.execute(translated, tuple(params))
        rows: list = []
        if cur.description is not None:
            rows = list(cur.fetchall())
        result = _PostgresCursorResult(rows, cur.rowcount)
        cur.close()
        return result

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _init_schema_postgres(adapter: "PostgresConnectionAdapter") -> None:
    """Bootstrap the schema on Postgres (CREATE TABLE IF NOT EXISTS).

    The ``_SCHEMA_SQL`` CREATE statements use portable column types
    (TEXT / INTEGER / REAL) and ``CREATE TABLE IF NOT EXISTS``, so they
    apply cleanly on Postgres after placeholder/upsert translation
    (which is a no-op for DDL). The SQLite-specific forward-migration
    ``ALTER TABLE ADD COLUMN`` deltas are re-expressed with the Postgres
    ``IF NOT EXISTS`` form so re-runs are idempotent.
    """

    for stmt in _SCHEMA_SQL:
        adapter.execute(stmt)
    pg_alters = (
        "ALTER TABLE templates ADD COLUMN IF NOT EXISTS figure_artifact_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE templates ADD COLUMN IF NOT EXISTS citation_policy TEXT NOT NULL DEFAULT 'disabled'",
        "ALTER TABLE templates ADD COLUMN IF NOT EXISTS coverage_policy TEXT NOT NULL DEFAULT 'passthrough'",
        "ALTER TABLE templates ADD COLUMN IF NOT EXISTS figure_time_window TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE contracts ADD COLUMN IF NOT EXISTS plugins_json TEXT NOT NULL DEFAULT '[]'",
    )
    for stmt in pg_alters:
        adapter.execute(stmt)
    adapter.execute(
        "INSERT INTO schema_meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        ("schema_version", str(SCHEMA_VERSION)),
    )


class Registry:
    """Thin facade over the registry connection + per-resource stores.

    Holds a single :class:`asyncio.Lock` that the per-resource stores
    acquire on writes. Reads do not take the lock because WAL mode
    gives them snapshot consistency.

    Backend (debt D2): SQLite by default (``db_path``); pass
    ``backend="postgres"`` + ``dsn=...`` (or set ``DLAAS_REGISTRY_BACKEND``
    / ``DLAAS_REGISTRY_PG_DSN``) to back the same store API with managed
    Postgres instead of SQLite-on-PVC. The store SQL is unchanged — it is
    translated at the connection boundary.

    Slice 3 wires Tenant + Shell + Asset + Template + Contract; Slice
    4 adds focus_persons + identity_links; Slice 5 adds handoff
    tickets.
    """

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        backend: str | None = None,
        dsn: str | None = None,
    ) -> None:
        self._backend = resolve_registry_backend(backend)
        self._write_lock = asyncio.Lock()
        if self._backend == "postgres":
            resolved_dsn = dsn or os.environ.get("DLAAS_REGISTRY_PG_DSN") or ""
            adapter = PostgresConnectionAdapter(resolved_dsn)
            _init_schema_postgres(adapter)
            self._conn = adapter
            self._db_path = f"postgres:{resolved_dsn.split('@')[-1] if '@' in resolved_dsn else 'configured'}"
        else:
            self._conn = open_connection(db_path)
            init_schema(self._conn)
            self._db_path = str(db_path)

    @property
    def conn(self):
        return self._conn

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def write_lock(self) -> asyncio.Lock:
        return self._write_lock

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover - defensive
            pass
        except Exception:  # pragma: no cover - defensive (postgres adapter)
            pass


__all__ = [
    "PostgresConnectionAdapter",
    "Registry",
    "SCHEMA_VERSION",
    "init_schema",
    "open_connection",
    "resolve_registry_backend",
]
