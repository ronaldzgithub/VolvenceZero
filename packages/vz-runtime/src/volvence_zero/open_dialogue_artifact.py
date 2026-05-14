"""Open-dialogue artifact export (Rupture-and-Repair M5).

Read-only exporter that writes two files for every session we want to
review or replay:

* ``<out_dir>/<session_id>/turns.jsonl`` — one NDJSON row per dialogue
  action trace (substrate semantic pulls come from the dialogue trace
  store's already-sanitized replay fields; raw user text is never
  written).
* ``<out_dir>/<session_id>/session_summary.json`` — aggregated
  closed-scene / slow-loop / rupture / memory-delta summary.

The exporter NEVER mutates owner state. It reads the dialogue trace
store's public snapshot, the external-outcome module's buffer snapshot
(through the runner's last published snapshots), and the memory store's
durable entries. All failures raise — no swallowed errors.

Matched-control replay (``run_matched_control_session``) constructs a
second :class:`AgentSessionRunner` with a given ``rupture_wiring``
override so callers can compare full-path vs rupture-disabled
trajectories. This is the CLI surface the v0 gate harness uses.

This module lives at ``volvence_zero.open_dialogue_artifact`` (top-level
inside vz-runtime) rather than under ``volvence_zero.agent`` so it can
be imported from integration / evolution without pulling in the whole
agent package (same rationale as
``volvence_zero.dialogue_external_outcome``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import MemoryStratum
from volvence_zero.rupture_state import RuptureStateSnapshot
from volvence_zero.runtime import WiringLevel


EXPORTER_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class OpenDialogueExportReport:
    """Report returned by :func:`export_open_dialogue_session`."""

    session_id: str
    out_dir: Path
    turns_path: Path
    summary_path: Path
    turn_count: int
    durable_rupture_repair_count: int
    description: str


def export_open_dialogue_session(
    runner: AgentSessionRunner,
    *,
    session_id: str,
    out_dir: str | os.PathLike[str],
    user_scope: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> OpenDialogueExportReport:
    """Write turns.jsonl and session_summary.json for the runner's session.

    The exporter reads:

    * ``runner.dialogue_trace_snapshot`` for the per-turn trace rows.
    * ``runner.session_post_queue_state`` for slow-loop counters.
    * ``runner.memory_store.entries_for(DURABLE)`` for rupture-repair
      durable counts (public R8 admin readout from ``vz-memory``).
      Only the tag/content keys needed for the review bundle are
      emitted; raw content strings are not redacted but the schema
      is documented in ``docs/specs/rupture-and-repair.md``.
    * ``runner.upstream_snapshots`` for the latest active and shadow
      snapshots, used to write regime and rupture summaries.

    No owner state is mutated. All failures raise.
    """

    root = Path(out_dir)
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    turns_path = session_dir / "turns.jsonl"
    summary_path = session_dir / "session_summary.json"

    trace_snapshot = runner.dialogue_trace_snapshot
    resolved_scope = user_scope or runner.user_scope
    turn_rows = _build_turn_rows(trace_snapshot)

    with turns_path.open("w", encoding="utf-8") as fh:
        for row in turn_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    durable_rupture_repair_entries = tuple(
        entry
        for entry in runner.memory_store.entries_for(MemoryStratum.DURABLE)
        if "rupture_repair" in entry.tags
    )

    summary = _build_session_summary(
        runner=runner,
        session_id=session_id,
        user_scope=resolved_scope,
        trace_snapshot_turn_count=len(turn_rows),
        durable_rupture_repair_entries=durable_rupture_repair_entries,
        extra_metadata=extra_metadata or {},
    )

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)

    return OpenDialogueExportReport(
        session_id=session_id,
        out_dir=root,
        turns_path=turns_path,
        summary_path=summary_path,
        turn_count=len(turn_rows),
        durable_rupture_repair_count=len(durable_rupture_repair_entries),
        description=(
            f"Open-dialogue artifact for '{session_id}' written to '{session_dir}': "
            f"{len(turn_rows)} turns, "
            f"{len(durable_rupture_repair_entries)} durable rupture_repair entries."
        ),
    )


def _build_turn_rows(trace_snapshot: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in trace_snapshot.traces:
        structured_evidence = [
            {
                "evidence_id": ev.evidence_id,
                "source": ev.source.value,
                "source_owner": ev.source_owner,
                "outcome_kind": ev.outcome_kind.value,
                "confidence": float(ev.confidence),
                "evidence_refs": list(ev.evidence_refs),
            }
            for ev in trace.outcome.structured_evidence
        ]
        rows.append(
            {
                "schema_version": EXPORTER_SCHEMA_VERSION,
                "trace_id": trace.trace_id,
                "event_id": trace.event_id,
                "wave_id": trace.wave_id,
                "turn_index": trace.turn_index,
                "action_kind": trace.action_kind.value,
                "environment_event_kind": trace.environment_event_kind,
                "environment_trigger_kind": trace.environment_trigger_kind,
                "active_regime": trace.active_regime or "",
                "active_abstract_action": trace.active_abstract_action or "",
                "response_rationale": trace.response_rationale,
                "prediction_id": trace.prediction_id or "",
                "response_text_hash": trace.response_text_hash,
                "outcome": {
                    "outcome_id": trace.outcome.outcome_id,
                    "previous_trace_id": trace.outcome.previous_trace_id,
                    "observed_trace_id": trace.outcome.observed_trace_id,
                    "observed_turn_index": trace.outcome.observed_turn_index,
                    "kind": trace.outcome.kind.value,
                    "description": trace.outcome.description,
                    "structured_evidence": structured_evidence,
                },
            }
        )
    return rows


def _build_session_summary(
    *,
    runner: AgentSessionRunner,
    session_id: str,
    user_scope: str,
    trace_snapshot_turn_count: int,
    durable_rupture_repair_entries: tuple,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    queue_state = runner.session_post_queue_state
    last_snapshots = runner.upstream_snapshots
    rupture_summary = _summarize_rupture_state_snapshot(
        last_snapshots.get("rupture_state")
    )
    external_outcome_summary = _summarize_external_outcomes(
        last_snapshots.get("dialogue_external_outcome")
    )
    rupture_repair_memory_summary = _summarize_rupture_repair_entries(
        durable_rupture_repair_entries
    )
    regime_snapshot = last_snapshots.get("regime")
    regime_summary: dict[str, Any] = {}
    if regime_snapshot is not None and hasattr(regime_snapshot.value, "active_regime"):
        regime_value = regime_snapshot.value
        regime_summary = {
            "active_regime_id": regime_value.active_regime.regime_id,
            "candidate_regimes": [
                {"regime_id": r_id, "score": float(score)}
                for r_id, score in regime_value.candidate_regimes
            ],
            "delayed_attributions": [
                {
                    "regime_id": attr.regime_id,
                    "outcome_score": float(attr.outcome_score),
                    "source_turn_index": int(attr.source_turn_index),
                    "source_wave_id": attr.source_wave_id,
                    "abstract_action": attr.abstract_action or "",
                    "action_family_version": int(attr.action_family_version),
                    "resolved_turn_index": int(attr.resolved_turn_index),
                }
                for attr in regime_value.delayed_attributions
            ],
        }
    return {
        "schema_version": EXPORTER_SCHEMA_VERSION,
        "session_id": session_id,
        "user_scope": user_scope,
        "turn_count": trace_snapshot_turn_count,
        "session_post_queue": {
            "pending": queue_state.pending_job_count,
            "running": queue_state.running_job_count,
            "completed": queue_state.completed_job_count,
            "last_completed_job_id": queue_state.last_completed_job_id or "",
            "last_completed_context_session_id": (
                queue_state.last_completed_context_session_id or ""
            ),
        },
        "regime": regime_summary,
        "rupture_state": rupture_summary,
        "external_outcomes": external_outcome_summary,
        "rupture_repair_memory": rupture_repair_memory_summary,
        "extra_metadata": extra_metadata,
    }


def _summarize_rupture_state_snapshot(snapshot: Any) -> dict[str, Any]:
    if snapshot is None or not isinstance(snapshot.value, RuptureStateSnapshot):
        return {}
    value = snapshot.value
    return {
        "rupture_kind": value.rupture_kind.value if value.rupture_kind else None,
        "rupture_signal_strength": float(value.rupture_signal_strength),
        "confidence": float(value.confidence),
        "internal_suspected_only": bool(value.internal_suspected_only),
        "evidence_sources": [src.value for src in value.evidence_sources],
    }


def _summarize_external_outcomes(snapshot: Any) -> dict[str, Any]:
    if snapshot is None or not isinstance(snapshot.value, DialogueExternalOutcomeSnapshot):
        return {}
    entries = [
        {
            "evidence_id": entry.evidence_id,
            "turn_index": int(entry.turn_index),
            "kind": entry.kind.value,
            "source": entry.source.value,
            "confidence": float(entry.confidence),
        }
        for entry in snapshot.value.entries
    ]
    # Also include per-kind counts aggregated across this snapshot for
    # quick-glance review.
    per_kind: dict[str, int] = {}
    for entry in snapshot.value.entries:
        per_kind[entry.kind.value] = per_kind.get(entry.kind.value, 0) + 1
    return {
        "turn_index": int(snapshot.value.turn_index),
        "entries": entries,
        "per_kind_counts": per_kind,
    }


def _summarize_rupture_repair_entries(entries: tuple) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    per_scope: dict[str, int] = {}
    for entry in entries:
        for tag in entry.tags:
            if tag.startswith("rupture_kind:"):
                kinds[tag.split(":", 1)[1]] = kinds.get(tag.split(":", 1)[1], 0) + 1
            elif tag.startswith("repair_outcome:"):
                value = tag.split(":", 1)[1]
                outcomes[value] = outcomes.get(value, 0) + 1
            elif tag.startswith("user_scope:"):
                scope = tag.split(":", 1)[1]
                per_scope[scope] = per_scope.get(scope, 0) + 1
    return {
        "count": len(entries),
        "kinds": kinds,
        "outcomes": outcomes,
        "per_scope": per_scope,
    }


def build_rupture_wiring_config(
    rupture_wiring: WiringLevel | str,
    *,
    base: FinalRolloutConfig | None = None,
) -> FinalRolloutConfig:
    """Return a ``FinalRolloutConfig`` with ``rupture_state`` overridden.

    Used for matched-control replay (Gate D). The rest of the
    configuration is inherited from ``base`` so only the rupture
    wiring changes between control and treatment.
    """

    if isinstance(rupture_wiring, str):
        rupture_wiring = WiringLevel(rupture_wiring.lower())
    cfg = base or FinalRolloutConfig()
    return dataclasses.replace(cfg, rupture_state=rupture_wiring)


__all__ = [
    "EXPORTER_SCHEMA_VERSION",
    "OpenDialogueExportReport",
    "build_rupture_wiring_config",
    "export_open_dialogue_session",
]
