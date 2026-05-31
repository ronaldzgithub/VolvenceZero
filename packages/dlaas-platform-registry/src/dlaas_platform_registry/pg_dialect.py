"""SQLite→Postgres dialect translation for the control-plane registry.

Debt D2: the registry was deliberately written against stdlib
``sqlite3`` (zero-dependency, fine for control-plane CRUD volumes), with
``db.py`` noting *"Slice 7+ can swap in Postgres by replacing this module
— the typed store API stays identical."* This module is that swap layer.

The per-resource stores (``tenants.py`` / ``templates.py`` / ...) issue
SQLite-flavoured SQL:

* ``?`` positional placeholders (Postgres uses ``%s``).
* ``INSERT OR REPLACE INTO <table> (...) VALUES (...)`` upserts
  (Postgres uses ``INSERT ... ON CONFLICT (<pk>) DO UPDATE SET ...``).

Rather than rewrite every store (which would also churn the parallel
kernel agent's surface), we translate the SQL at the connection boundary
so the stores stay byte-for-byte identical and the typed store API is
unchanged. The translation functions here are **pure** and fully
unit-tested; the live psycopg connection adapter is exercised only when
the driver + a Postgres instance are actually present (otherwise the
registry stays on the SQLite default).
"""

from __future__ import annotations

import re

# Per-table primary keys, mirroring the ``PRIMARY KEY`` declarations in
# ``db.py``'s ``_SCHEMA_SQL``. Used to derive the ``ON CONFLICT`` target
# when translating ``INSERT OR REPLACE`` upserts to Postgres.
TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "schema_meta": ("key",),
    "tenants": ("tenant_id",),
    "shells": ("tenant_id", "shell_id"),
    "assets": ("asset_id",),
    "templates": ("template_id",),
    "template_versions": ("version_id",),
    "template_assets": ("template_id", "asset_id", "template_version"),
    "contracts": ("contract_id",),
    "focus_persons": ("contract_id", "person_id"),
    "identity_links": ("ai_id", "channel_type", "channel_ref"),
    "handoff_tickets": ("ticket_id",),
    "audience_profiles": ("profile_id",),
    "exam_questions": ("question_id",),
    "exam_runs": ("run_id",),
    "launch_licenses": ("license_id",),
    "governance_records": ("record_kind", "record_id"),
    "cultivations": ("cultivation_id",),
    "applications": ("application_id",),
    "org_application_approvals": ("tenant_id", "application_id"),
    "training_jobs": ("ai_id", "job_id"),
}


def translate_placeholders(sql: str) -> str:
    """Translate SQLite ``?`` placeholders to Postgres ``%s``.

    Only bare ``?`` tokens are translated; ``?`` characters inside
    single-quoted string literals are preserved so a literal question
    mark in seeded data is never mangled.
    """

    out: list[str] = []
    in_string = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            # Toggle string state, honouring '' escape inside strings.
            if in_string and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            in_string = not in_string
            out.append(ch)
        elif ch == "?" and not in_string:
            out.append("%s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_INSERT_OR_REPLACE_RE = re.compile(
    r"INSERT\s+OR\s+REPLACE\s+INTO\s+(?P<table>\w+)\s*\((?P<cols>[^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)


def _split_columns(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def translate_upsert(sql: str) -> str:
    """Rewrite ``INSERT OR REPLACE`` into a Postgres ``ON CONFLICT`` upsert.

    The conflict target is the table's primary key (from
    :data:`TABLE_PRIMARY_KEYS`); every non-PK column is updated from the
    inserted row (``SET col = EXCLUDED.col``), reproducing SQLite's
    "replace the whole row" semantics. Statements that are not
    ``INSERT OR REPLACE`` are returned unchanged.
    """

    match = _INSERT_OR_REPLACE_RE.search(sql)
    if match is None:
        return sql
    table = match.group("table")
    columns = _split_columns(match.group("cols"))
    pk = TABLE_PRIMARY_KEYS.get(table)
    if not pk:
        raise ValueError(
            f"translate_upsert: no known primary key for table {table!r}; "
            "add it to TABLE_PRIMARY_KEYS before targeting Postgres."
        )
    # Replace the "INSERT OR REPLACE" verb with a plain "INSERT".
    rewritten = (
        sql[: match.start()]
        + re.sub(
            r"INSERT\s+OR\s+REPLACE\s+INTO",
            "INSERT INTO",
            sql[match.start() : match.end()],
            flags=re.IGNORECASE,
        )
        + sql[match.end() :]
    )
    non_pk = [c for c in columns if c not in pk]
    conflict_cols = ", ".join(pk)
    if non_pk:
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk)
        action = f"DO UPDATE SET {set_clause}"
    else:
        action = "DO NOTHING"
    return f"{rewritten.rstrip().rstrip(';')} ON CONFLICT ({conflict_cols}) {action}"


def translate_statement(sql: str) -> str:
    """Full SQLite→Postgres statement translation (upsert + placeholders)."""

    return translate_placeholders(translate_upsert(sql))


__all__ = [
    "TABLE_PRIMARY_KEYS",
    "translate_placeholders",
    "translate_statement",
    "translate_upsert",
]
