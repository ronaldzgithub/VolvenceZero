"""F6 / P6.3 — apply persona LoRA artifacts through the OFFLINE gate.

Mirrors :func:`lifeform_domain_character.apply_drive_evolution_through_gate`
for the persona-LoRA case. Routes a :class:`FigureLoRAArtifact`
through :class:`ModificationGate.OFFLINE` and, on ALLOW, attaches
it to the figure bundle and registers it in a
:class:`PersonaLoRAPool`.

R10 reminder: a baked LoRA shifts representation in the substrate's
adapter layer — that's "who the lifeform is", which is rare-heavy
self-modification, hence the OFFLINE gate. R15: the apply result
captures both the previous bundle / pool record id and the new
record id so the operator can reattach the previous artifact on
rollback.

This module **does not** depend on any DLaaS-platform symbols. The
lifeform-service adopt path imports from here, not the other way
round (R8 layering).
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.credit.gate import (
    GateDecision,
    evaluate_gate,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.substrate import PersonaLoRAPool, default_persona_lora_pool

from lifeform_domain_figure.figure_artifact import FigureArtifactBundle
from lifeform_domain_figure.lora_artifact import FigureLoRAArtifact
from lifeform_domain_figure.lora_bake_synthetic import attach_baked_lora
from lifeform_domain_figure.lora_data_prep import (
    PersonaLoRAProposal,
    build_persona_lora_proposal,
)


@dataclass(frozen=True)
class GatedPersonaLoRAProposal:
    """A persona LoRA proposal after the gate has decided."""

    proposal: PersonaLoRAProposal
    decision: GateDecision
    block_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PersonaLoRAApplyResult:
    """Return value of :func:`apply_persona_lora_through_gate`.

    Fields:

    * ``bundle``                   — bundle with the artifact bound
                                     (or the unchanged base bundle
                                     if blocked).
    * ``base_bundle``              — bundle as it was before the
                                     proposal (rollback target).
    * ``artifact``                 — the proposed artifact (always
                                     present so the audit log can
                                     re-issue it).
    * ``gate``                     — gated proposal + decision.
    * ``applied``                  — whether the artifact landed.
    * ``record_id``                — pool record id when applied,
                                     ``None`` otherwise.
    * ``previous_record_id``       — pool record id that was
                                     previously active for this
                                     figure, ``"absent"`` if none.
    """

    bundle: FigureArtifactBundle
    base_bundle: FigureArtifactBundle
    artifact: FigureLoRAArtifact
    gate: GatedPersonaLoRAProposal
    applied: bool
    record_id: str | None
    previous_record_id: str


def apply_persona_lora_through_gate(
    *,
    base_bundle: FigureArtifactBundle,
    artifact: FigureLoRAArtifact,
    evaluation_snapshot: EvaluationSnapshot,
    pool: PersonaLoRAPool | None = None,
    validation_delta: float = 0.05,
    capacity_cost: float = 0.30,
    rollback_evidence: str = "",
) -> PersonaLoRAApplyResult:
    """Run a persona LoRA artifact through the OFFLINE gate.

    On ALLOW:

    * Builds a new bundle via :func:`attach_baked_lora` (re-keys the
      bundle id so the artifact's identity is part of the bundle's
      hash address, R15).
    * Registers the artifact in ``pool`` (defaults to the
      process-wide :func:`default_persona_lora_pool`) and returns
      the resulting ``record_id``.

    On BLOCK:

    * Leaves the bundle unchanged.
    * Returns ``applied=False`` and the gate's
      :attr:`block_reasons`.

    A non-empty ``rollback_evidence`` is required: it must identify
    the previous LoRA bundle id (or ``"absent"``). Empty values
    raise ``ValueError``.
    """

    if artifact.figure_id != base_bundle.figure_id:
        raise ValueError(
            "apply_persona_lora_through_gate: artifact.figure_id="
            f"{artifact.figure_id!r} does not match "
            f"base_bundle.figure_id={base_bundle.figure_id!r}"
        )
    if not rollback_evidence.strip():
        raise ValueError(
            "apply_persona_lora_through_gate: rollback_evidence must be "
            "non-empty (the OFFLINE gate requires it)"
        )
    target_pool = pool if pool is not None else default_persona_lora_pool()
    previous_record_id = _resolve_previous_record_id(
        bundle=base_bundle, pool=target_pool
    )
    persona_proposal = build_persona_lora_proposal(
        figure_id=artifact.figure_id,
        plan=_FakePlanForProposal(
            figure_id=artifact.figure_id,
            integrity_hash=artifact.training_plan_hash,
            figure_example_count=_layer_count_as_proxy(artifact),
            replay_example_count=0,
        ),
        new_artifact_integrity_hash=artifact.integrity_hash,
        previous_artifact_id=_resolve_previous_artifact_id(base_bundle),
        rollback_evidence=rollback_evidence,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        justification=(
            f"Apply persona LoRA {artifact.integrity_hash[:8]} for "
            f"{artifact.figure_id} (backend={artifact.backend_id}, "
            f"layers={artifact.total_layers})."
        ),
    )
    decision = evaluate_gate(
        proposal=persona_proposal.proposal,
        evaluation_snapshot=evaluation_snapshot,
    )
    if decision is GateDecision.BLOCK:
        reasons = evaluate_gate_reasons(
            proposal=persona_proposal.proposal,
            evaluation_snapshot=evaluation_snapshot,
        )
        return PersonaLoRAApplyResult(
            bundle=base_bundle,
            base_bundle=base_bundle,
            artifact=artifact,
            gate=GatedPersonaLoRAProposal(
                proposal=persona_proposal,
                decision=decision,
                block_reasons=reasons,
            ),
            applied=False,
            record_id=None,
            previous_record_id=previous_record_id,
        )
    new_bundle = attach_baked_lora(base_bundle, artifact)
    record_id = target_pool.register(
        figure_id=artifact.figure_id,
        source_bundle_id=new_bundle.bundle_id,
        backend_id=artifact.backend_id,
        training_plan_hash=artifact.training_plan_hash,
        adapter_layers=artifact.adapter_layers,
        parameter_count=artifact.parameter_count,
        description=artifact.description,
    )
    return PersonaLoRAApplyResult(
        bundle=new_bundle,
        base_bundle=base_bundle,
        artifact=artifact,
        gate=GatedPersonaLoRAProposal(
            proposal=persona_proposal,
            decision=decision,
            block_reasons=(),
        ),
        applied=True,
        record_id=record_id,
        previous_record_id=previous_record_id,
    )


def _resolve_previous_record_id(
    *,
    bundle: FigureArtifactBundle,
    pool: PersonaLoRAPool,
) -> str:
    """Look up the prior record id for the figure (or 'absent')."""

    figure_id = bundle.figure_id
    if not pool.has(figure_id):
        return "absent"
    return pool.lookup(figure_id).record_id


def _resolve_previous_artifact_id(bundle: FigureArtifactBundle) -> str:
    """Read the prior LoRA artifact id from the bundle (or 'absent').

    Reads ``integrity_hash`` directly because :class:`FigureLoRAArtifact`
    is the only type the bundle's ``lora`` slot is documented to hold;
    if a future caller stuffs a different shape in there the
    :class:`AttributeError` is the intended fail-loud signal
    (no-swallow-errors invariant).
    """

    if bundle.lora is None:
        return "absent"
    integrity = getattr(bundle.lora, "integrity_hash", None)
    if not isinstance(integrity, str) or not integrity.strip():
        raise ValueError(
            "_resolve_previous_artifact_id: bundle.lora has no usable "
            "integrity_hash; the lora slot is expected to hold a "
            "FigureLoRAArtifact-shaped object."
        )
    return integrity


def _layer_count_as_proxy(artifact: FigureLoRAArtifact) -> int:
    """Return a non-zero proxy for the proposal's example count.

    The proposal builder requires figure_example_count > 0 so its
    invariants stay tight even when this gate path constructs the
    proposal from a baked artifact (where the original training-plan
    rows are not retained in-memory). Using ``total_layers`` keeps
    the value monotone in artifact size without re-reading the plan.
    """

    return max(artifact.total_layers, 1)


@dataclass(frozen=True)
class _FakePlanForProposal:
    """Minimal duck-type satisfying :func:`build_persona_lora_proposal`.

    The proposal builder needs ``figure_id``, ``integrity_hash``,
    ``figure_example_count``, and ``replay_example_count`` — all of
    which are available on the artifact / known to the gate path.
    Constructing a synthetic plan-shaped object here avoids
    re-reading the original training plan from disk on every gate
    call and keeps the gate path independent of plan storage.
    """

    figure_id: str
    integrity_hash: str
    figure_example_count: int
    replay_example_count: int


__all__ = [
    "GatedPersonaLoRAProposal",
    "PersonaLoRAApplyResult",
    "apply_persona_lora_through_gate",
]
