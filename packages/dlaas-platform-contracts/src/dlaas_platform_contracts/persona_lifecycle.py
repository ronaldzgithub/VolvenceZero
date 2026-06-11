"""Unified persona cognitive-training lifecycle contracts.

One platform-owned record tracks a persona (a DLaaS template) through
the full cognitive-training pipeline::

    draft ──▶ pretrained ──▶ studying ──▶ training ──▶ exam ──▶ interview ──▶ inducted
      │            │             │            │          │           │
      └────────────┴─────────────┴────────────┴──────────┴───────────┴──▶ retired

* ``draft``      — lifecycle row created; nothing baked or taught yet.
* ``pretrained`` — offline figure/persona bake completed (corpus →
                   bundle → optional LoRA → persona verification).
                   Evidence: ``figure_bundle_id`` (+ optional
                   ``verification_verdict_ref``).
* ``studying``   — autonomous self-study (cultivation) attached.
                   Evidence: ``cultivation_id``.
* ``training``   — operator/corpus/teach training underway.
                   Evidence: ``training_ref`` (corpus intake id,
                   training job id, or teach session ref).
* ``exam``       — an eval-gate exam run is recorded for this persona.
                   Evidence: ``exam_run_id`` (+ ``passed``).
* ``interview``  — an interactive interview run is recorded.
                   Evidence: ``interview_run_id`` (+ ``passed``).
* ``inducted``   — operator-approved; persona is adoptable / published.
                   Evidence: ``reviewer_id``. Requires passing exam +
                   interview evidence in the history unless an explicit
                   ``waiver_reason`` is supplied (no silent pass).
* ``retired``    — persona withdrawn from service. Evidence:
                   ``reason``.

Ownership boundary (R8 / R12 / R15):

* This is **platform governance state** — it points at evidence
  artifacts (bundles, cultivation rows, exam/interview runs) and never
  mirrors cognition. Cognitive state stays in the kernel behind the
  instance's ``ai_id``.
* Every advance carries evidence and is recorded as an immutable
  event; rollback is an explicit, audited transition (R15), never a
  silent overwrite.
* Stages may be skipped only in the forward direction (some products
  have no offline bake), and the skip itself is recorded in the event
  history — consumers can always tell which gates a persona actually
  passed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PersonaLifecycleStage(str, Enum):
    """Platform-owned cognitive-training stage for one persona."""

    DRAFT = "draft"
    PRETRAINED = "pretrained"
    STUDYING = "studying"
    TRAINING = "training"
    EXAM = "exam"
    INTERVIEW = "interview"
    INDUCTED = "inducted"
    RETIRED = "retired"


_STAGE_ORDER: tuple[PersonaLifecycleStage, ...] = (
    PersonaLifecycleStage.DRAFT,
    PersonaLifecycleStage.PRETRAINED,
    PersonaLifecycleStage.STUDYING,
    PersonaLifecycleStage.TRAINING,
    PersonaLifecycleStage.EXAM,
    PersonaLifecycleStage.INTERVIEW,
    PersonaLifecycleStage.INDUCTED,
)


def stage_order_index(stage: PersonaLifecycleStage) -> int:
    """Position of ``stage`` on the forward pipeline (RETIRED = -1)."""

    try:
        return _STAGE_ORDER.index(stage)
    except ValueError:
        return -1


#: Evidence keys that MUST be present (non-empty) to enter each stage.
#: Entering a stage asserts the pointed artifact exists; pass/fail of
#: gate stages travels in the evidence payload (``passed``).
REQUIRED_EVIDENCE_BY_STAGE: Mapping[PersonaLifecycleStage, tuple[str, ...]] = {
    PersonaLifecycleStage.PRETRAINED: ("figure_bundle_id",),
    PersonaLifecycleStage.STUDYING: ("cultivation_id",),
    PersonaLifecycleStage.TRAINING: ("training_ref",),
    PersonaLifecycleStage.EXAM: ("exam_run_id",),
    PersonaLifecycleStage.INTERVIEW: ("interview_run_id",),
    PersonaLifecycleStage.INDUCTED: ("reviewer_id",),
    PersonaLifecycleStage.RETIRED: ("reason",),
}

#: Gate stages whose evidence must carry ``passed: true`` before the
#: persona can be inducted (unless an explicit waiver is recorded).
INDUCTION_GATE_STAGES: tuple[PersonaLifecycleStage, ...] = (
    PersonaLifecycleStage.EXAM,
    PersonaLifecycleStage.INTERVIEW,
)


class LifecycleTransitionError(ValueError):
    """Raised when an advance/rollback violates the lifecycle contract."""


def validate_stage_advance(
    *,
    current: PersonaLifecycleStage,
    target: PersonaLifecycleStage,
    evidence: Mapping[str, Any],
) -> None:
    """Validate one forward transition. Raises loudly on violation.

    Rules:

    * ``RETIRED`` is reachable from any non-retired stage.
    * Otherwise the target must be strictly later on the forward
      pipeline (skipping intermediate stages is allowed and recorded).
    * ``INDUCTED`` and ``RETIRED`` are terminal for forward advances
      (``inducted → retired`` is the only exit from inducted).
    * The required evidence keys for the target stage must be present
      and non-empty.
    """

    if current is PersonaLifecycleStage.RETIRED:
        raise LifecycleTransitionError(
            "persona lifecycle is retired; no further transitions allowed"
        )
    if target is PersonaLifecycleStage.DRAFT:
        raise LifecycleTransitionError(
            "cannot advance back to 'draft'; use rollback for backwards moves"
        )
    if target is not PersonaLifecycleStage.RETIRED:
        if current is PersonaLifecycleStage.INDUCTED:
            raise LifecycleTransitionError(
                "inducted personas can only transition to 'retired'"
            )
        if stage_order_index(target) <= stage_order_index(current):
            raise LifecycleTransitionError(
                f"advance must move forward: {current.value!r} -> "
                f"{target.value!r} is not a forward transition "
                "(use rollback for backwards moves)"
            )
    missing = [
        key
        for key in REQUIRED_EVIDENCE_BY_STAGE.get(target, ())
        if not str(evidence.get(key, "") or "").strip()
    ]
    if missing:
        raise LifecycleTransitionError(
            f"advance to {target.value!r} requires evidence keys "
            f"{missing!r} (got keys {sorted(evidence.keys())!r})"
        )


@dataclass(frozen=True)
class LifecycleStageEvent:
    """One immutable lifecycle transition (advance or rollback)."""

    event_id: str
    lifecycle_id: str
    event_kind: str  # "created" | "advance" | "rollback"
    from_stage: PersonaLifecycleStage
    to_stage: PersonaLifecycleStage
    evidence: Mapping[str, Any] = field(default_factory=dict)
    actor: str = ""
    recorded_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "lifecycle_id": self.lifecycle_id,
            "event_kind": self.event_kind,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "evidence": dict(self.evidence),
            "actor": self.actor,
            "recorded_at_ms": self.recorded_at_ms,
        }


@dataclass(frozen=True)
class PersonaTrainingLifecycle:
    """Aggregate lifecycle row for one persona (keyed by template)."""

    lifecycle_id: str
    template_id: str
    tenant_id: str = ""
    ai_id: str = ""
    display_name: str = ""
    app_id: str = ""
    stage: PersonaLifecycleStage = PersonaLifecycleStage.DRAFT
    notes: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "lifecycle_id": self.lifecycle_id,
            "template_id": self.template_id,
            "tenant_id": self.tenant_id,
            "ai_id": self.ai_id,
            "display_name": self.display_name,
            "app_id": self.app_id,
            "stage": self.stage.value,
            "notes": self.notes,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }


def gate_summary_from_events(
    events: tuple[LifecycleStageEvent, ...],
) -> dict[str, Any]:
    """Derive the per-gate readout consumers render (latest event wins).

    Returns ``{stage: {"reached": bool, "passed": bool | None,
    "evidence": {...}}}`` for the forward stages. ``passed`` is None
    when the stage carries no pass/fail semantics or no evidence had a
    ``passed`` field.
    """

    summary: dict[str, Any] = {}
    for stage in _STAGE_ORDER[1:]:
        latest = None
        for event in events:
            if event.to_stage is stage and event.event_kind == "advance":
                latest = event
        entry: dict[str, Any] = {"reached": latest is not None}
        if latest is not None:
            entry["evidence"] = dict(latest.evidence)
            raw_passed = latest.evidence.get("passed")
            entry["passed"] = bool(raw_passed) if raw_passed is not None else None
        else:
            entry["passed"] = None
        summary[stage.value] = entry
    return summary


__all__ = [
    "INDUCTION_GATE_STAGES",
    "LifecycleStageEvent",
    "LifecycleTransitionError",
    "PersonaLifecycleStage",
    "PersonaTrainingLifecycle",
    "REQUIRED_EVIDENCE_BY_STAGE",
    "gate_summary_from_events",
    "stage_order_index",
    "validate_stage_advance",
]
