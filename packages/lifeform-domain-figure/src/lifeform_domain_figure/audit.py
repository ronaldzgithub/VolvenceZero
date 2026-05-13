"""Audit log for the figure-vertical bake / gate-apply / rollback CLI.

Every CLI subcommand that mutates persisted state (``bake-bundle`` /
``bake-steering`` / ``bake-lora`` / ``rollback``) writes one
:class:`FigureBakeAuditRecord` into ``<audit_root>/`` as a frozen
JSON file. The audit log is **append-only** (R15 byte-level rollback
contract): rollbacks do not delete prior records — they write a
new ``ROLLBACK`` record that names the previous ``record_id`` so an
operator can reconstruct the full timeline.

Mirrors the ``RuntimeAdaptationAudit`` style used in
:mod:`volvence_zero.runtime` (typed dataclass + write helper +
content-addressed audit id) so reviewers see one consistent shape
for "what happened, when, who decided, what to roll back to".

This module is dependency-light on purpose: it imports nothing
from outside the figure wheel + Python stdlib so the CLI can write
audit records even if the heavier figure pipeline modules fail
during the same run.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone


AUDIT_SCHEMA_VERSION = "vz-figure-bake-audit.v1"


class FigureBakeAction(str, enum.Enum):
    """Canonical actions emitted by the figure CLI subcommands."""

    BAKE_BUNDLE = "BAKE_BUNDLE"
    BAKE_STEERING = "BAKE_STEERING"
    BAKE_LORA = "BAKE_LORA"
    ROLLBACK = "ROLLBACK"


class FigureGateDecisionLabel(str, enum.Enum):
    """Gate decision rendering for audit consumers.

    The on-the-wire ``ALLOW`` / ``BLOCK`` come from
    :class:`volvence_zero.credit.gate.GateDecision`; ``NA`` means
    the action did not pass through a gate (currently:
    ``BAKE_BUNDLE`` and ``ROLLBACK``).
    """

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    NA = "NA"


@dataclass(frozen=True)
class FigureBakeAuditRecord:
    """Frozen audit row for one CLI mutation.

    Field meanings:

    * ``audit_id``           — sha256 prefix derived from the rest
                                of the record (deterministic; same
                                payload → same id).
    * ``schema_version``     — :data:`AUDIT_SCHEMA_VERSION`.
    * ``action``             — :class:`FigureBakeAction` value.
    * ``figure_id``          — figure under which the action ran.
    * ``bundle_id``          — id of the bundle the action produced
                                (or rolled back to).
    * ``previous_bundle_id`` — id of the bundle the action replaced
                                (``"absent"`` if none).
    * ``record_id``          — :class:`PersonaLoRAPool` record id
                                produced by ``BAKE_LORA`` /
                                ``ROLLBACK``; ``None`` for actions
                                that do not touch the pool.
    * ``previous_record_id`` — pool record id replaced by the
                                action; ``"absent"`` if none.
    * ``gate_decision``      — :class:`FigureGateDecisionLabel`.
    * ``block_reasons``      — gate's structured ``block_reasons``
                                tuple when ``BLOCK``; ``()``
                                otherwise.
    * ``rollback_evidence``  — non-empty text the operator supplied
                                to identify the rollback target.
                                ``""`` for ``BAKE_BUNDLE`` (no gate).
    * ``validation_delta``   — gate proposal field (NaN-safe via
                                str format on save).
    * ``capacity_cost``      — gate proposal field.
    * ``corpus_mode``        — ``"synthetic"`` / ``"curated"`` for
                                bundle-level bakes; ``""`` for
                                downstream actions.
    * ``backend_id``         — bake backend label (e.g.
                                ``"synthetic-v1"``).
    * ``created_at_iso``     — UTC ISO 8601 second resolution.
    """

    audit_id: str
    schema_version: str
    action: FigureBakeAction
    figure_id: str
    bundle_id: str
    previous_bundle_id: str
    record_id: str | None
    previous_record_id: str
    gate_decision: FigureGateDecisionLabel
    block_reasons: tuple[str, ...]
    rollback_evidence: str
    validation_delta: float
    capacity_cost: float
    corpus_mode: str
    backend_id: str
    created_at_iso: str
    extra: dict[str, str] = field(default_factory=dict)
    # debt #63 (P1 figure-evidence-packet G-D cost回填): per-action
    # cost telemetry for unit-economics actuals回填 (engineer hours /
    # GPU hours / archive fetch wallclock / reviewer hours). Default
    # empty dict for backward-compat; populated by bake CLI when
    # --record-cost flag is on or by external scripts/figure_cost_summary.py
    # when reconstructing from external timing data.
    cost_breakdown: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError(
                f"FigureBakeAuditRecord.schema_version mismatch: "
                f"got {self.schema_version!r}, expected "
                f"{AUDIT_SCHEMA_VERSION!r}"
            )
        if not self.figure_id.strip():
            raise ValueError("FigureBakeAuditRecord.figure_id must be non-empty")
        if not self.bundle_id.strip():
            raise ValueError("FigureBakeAuditRecord.bundle_id must be non-empty")
        if not self.previous_bundle_id.strip():
            raise ValueError(
                "FigureBakeAuditRecord.previous_bundle_id must be "
                "non-empty (use 'absent' if there is no prior bundle)."
            )
        if not self.previous_record_id.strip():
            raise ValueError(
                "FigureBakeAuditRecord.previous_record_id must be "
                "non-empty (use 'absent' if there is no prior record)."
            )
        if not self.created_at_iso.strip():
            raise ValueError(
                "FigureBakeAuditRecord.created_at_iso must be non-empty"
            )
        if not self.audit_id.strip():
            raise ValueError("FigureBakeAuditRecord.audit_id must be non-empty")


def build_audit_record(
    *,
    action: FigureBakeAction,
    figure_id: str,
    bundle_id: str,
    previous_bundle_id: str = "absent",
    record_id: str | None = None,
    previous_record_id: str = "absent",
    gate_decision: FigureGateDecisionLabel = FigureGateDecisionLabel.NA,
    block_reasons: tuple[str, ...] = (),
    rollback_evidence: str = "",
    validation_delta: float = 0.0,
    capacity_cost: float = 0.0,
    corpus_mode: str = "",
    backend_id: str = "",
    created_at_iso: str | None = None,
    extra: dict[str, str] | None = None,
    cost_breakdown: dict[str, float] | None = None,
) -> FigureBakeAuditRecord:
    """Assemble an audit record with a deterministic ``audit_id``.

    The ``audit_id`` is sha256 over (action / figure / bundle /
    record / gate / timestamp / rollback_evidence). Two calls with
    the same input produce the same id; this is the property the
    rollback CLI relies on when it deduplicates audit re-emission.
    """

    timestamp = created_at_iso or _now_iso()
    payload = (
        AUDIT_SCHEMA_VERSION,
        action.value,
        figure_id,
        bundle_id,
        previous_bundle_id,
        record_id or "absent",
        previous_record_id,
        gate_decision.value,
        tuple(block_reasons),
        rollback_evidence,
        f"{validation_delta:.6f}",
        f"{capacity_cost:.6f}",
        corpus_mode,
        backend_id,
        timestamp,
        tuple(sorted((extra or {}).items())),
    )
    audit_id = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return FigureBakeAuditRecord(
        audit_id=audit_id,
        schema_version=AUDIT_SCHEMA_VERSION,
        action=action,
        figure_id=figure_id,
        bundle_id=bundle_id,
        previous_bundle_id=previous_bundle_id,
        record_id=record_id,
        previous_record_id=previous_record_id,
        gate_decision=gate_decision,
        block_reasons=tuple(block_reasons),
        rollback_evidence=rollback_evidence,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        corpus_mode=corpus_mode,
        backend_id=backend_id,
        created_at_iso=timestamp,
        extra=dict(extra or {}),
        cost_breakdown=dict(cost_breakdown or {}),
    )


def write_audit(
    record: FigureBakeAuditRecord,
    *,
    root_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Atomically write ``record`` as JSON under ``root_dir``.

    Path layout::

        <root_dir>/<created_at_safe>_<action>_<figure_id>_<audit_prefix>.json

    Returns the resolved absolute path. ``root_dir`` is created if
    missing.
    """

    root = pathlib.Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_ts = _filename_safe_timestamp(record.created_at_iso)
    safe_figure = _filename_safe_token(record.figure_id)
    target = root / (
        f"{safe_ts}_{record.action.value}_{safe_figure}_"
        f"{record.audit_id[:12]}.json"
    )
    payload = {
        "schema_version": record.schema_version,
        "audit_id": record.audit_id,
        "action": record.action.value,
        "figure_id": record.figure_id,
        "bundle_id": record.bundle_id,
        "previous_bundle_id": record.previous_bundle_id,
        "record_id": record.record_id,
        "previous_record_id": record.previous_record_id,
        "gate_decision": record.gate_decision.value,
        "block_reasons": list(record.block_reasons),
        "rollback_evidence": record.rollback_evidence,
        "validation_delta": record.validation_delta,
        "capacity_cost": record.capacity_cost,
        "corpus_mode": record.corpus_mode,
        "backend_id": record.backend_id,
        "created_at_iso": record.created_at_iso,
        "extra": dict(record.extra),
        "cost_breakdown": dict(record.cost_breakdown),
    }
    _atomic_write_text(
        target,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return target.resolve()


def read_audit_records(
    *,
    root_dir: str | pathlib.Path,
) -> tuple[FigureBakeAuditRecord, ...]:
    """Return every audit record under ``root_dir`` ordered by timestamp.

    Returns ``()`` when ``root_dir`` does not exist (callers may
    treat this as "no audit log yet"; this is intentional and not a
    swallowed error — missing directory is a documented state, not a
    schema violation).
    """

    root = pathlib.Path(root_dir)
    if not root.is_dir():
        return ()
    records: list[FigureBakeAuditRecord] = []
    for path in sorted(root.iterdir()):
        if not (path.is_file() and path.suffix == ".json"):
            continue
        records.append(_read_audit_file(path))
    records.sort(key=lambda r: r.created_at_iso)
    return tuple(records)


def find_previous_audit_for_bundle(
    *,
    root_dir: str | pathlib.Path,
    figure_id: str,
    bundle_id: str,
) -> FigureBakeAuditRecord | None:
    """Return the most recent audit record that produced ``bundle_id``.

    "Produced" = ``record.figure_id == figure_id`` AND
    ``record.bundle_id == bundle_id``. Used by the rollback CLI to
    look up the ``record_id`` and ``previous_*_id`` chain when the
    operator passes ``--to-bundle <id>``.

    Returns ``None`` if no matching record exists.
    """

    if not bundle_id.strip():
        raise ValueError("find_previous_audit_for_bundle: bundle_id must be non-empty")
    if not figure_id.strip():
        raise ValueError("find_previous_audit_for_bundle: figure_id must be non-empty")
    matches = [
        rec
        for rec in read_audit_records(root_dir=root_dir)
        if rec.figure_id == figure_id and rec.bundle_id == bundle_id
    ]
    if not matches:
        return None
    matches.sort(key=lambda r: r.created_at_iso)
    return matches[-1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _filename_safe_token(value: str) -> str:
    cleaned = _FILENAME_SAFE.sub("-", value).strip("-")
    return cleaned or "unknown"


def _filename_safe_timestamp(iso: str) -> str:
    return _filename_safe_token(iso.replace(":", "").replace(".", ""))


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_audit_file(path: pathlib.Path) -> FigureBakeAuditRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            f"_read_audit_file: file at {path} has "
            f"schema_version={schema!r}; this build only loads "
            f"{AUDIT_SCHEMA_VERSION!r}."
        )
    return FigureBakeAuditRecord(
        audit_id=str(payload["audit_id"]),
        schema_version=str(schema),
        action=FigureBakeAction(str(payload["action"])),
        figure_id=str(payload["figure_id"]),
        bundle_id=str(payload["bundle_id"]),
        previous_bundle_id=str(payload["previous_bundle_id"]),
        record_id=(
            str(payload["record_id"])
            if payload.get("record_id") is not None
            else None
        ),
        previous_record_id=str(payload["previous_record_id"]),
        gate_decision=FigureGateDecisionLabel(str(payload["gate_decision"])),
        block_reasons=tuple(str(r) for r in payload.get("block_reasons", ())),
        rollback_evidence=str(payload.get("rollback_evidence", "")),
        validation_delta=float(payload.get("validation_delta", 0.0)),
        capacity_cost=float(payload.get("capacity_cost", 0.0)),
        corpus_mode=str(payload.get("corpus_mode", "")),
        backend_id=str(payload.get("backend_id", "")),
        created_at_iso=str(payload["created_at_iso"]),
        extra={
            str(k): str(v) for k, v in (payload.get("extra") or {}).items()
        },
        cost_breakdown={
            str(k): float(v)
            for k, v in (payload.get("cost_breakdown") or {}).items()
        },
    )


def _atomic_write_text(target: pathlib.Path, payload: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload.encode("utf-8"))
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "FigureBakeAction",
    "FigureBakeAuditRecord",
    "FigureGateDecisionLabel",
    "build_audit_record",
    "find_previous_audit_for_bundle",
    "read_audit_records",
    "write_audit",
]
