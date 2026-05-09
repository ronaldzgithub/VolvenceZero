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
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2


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


class Registry:
    """Thin facade over the SQLite connection + per-resource stores.

    Holds a single :class:`asyncio.Lock` that the per-resource stores
    acquire on writes. Reads do not take the lock because WAL mode
    gives them snapshot consistency.

    Slice 3 wires Tenant + Shell + Asset + Template + Contract; Slice
    4 adds focus_persons + identity_links; Slice 5 adds handoff
    tickets.
    """

    def __init__(self, *, db_path: str | Path = ":memory:") -> None:
        self._conn = open_connection(db_path)
        init_schema(self._conn)
        self._write_lock = asyncio.Lock()
        self._db_path = str(db_path)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

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


__all__ = [
    "Registry",
    "SCHEMA_VERSION",
    "init_schema",
    "open_connection",
]
