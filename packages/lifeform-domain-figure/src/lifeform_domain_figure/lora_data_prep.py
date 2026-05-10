"""F6 / P6.1 — assemble LoRA training data + proposal schema.

Two tightly-coupled responsibilities live here:

1. :func:`build_lora_training_plan` turns the figure's reviewed
   primary-source ingestion envelopes into a frozen
   :class:`LoRATrainingPlan`: a deterministic list of
   :class:`LoRATrainingExample` instances split into a *figure*
   block (the corpus we want the LoRA to internalise) and a
   *replay buffer* block (general-purpose synthetic text we
   interleave to fight catastrophic forgetting on the frozen base).

2. :func:`build_persona_lora_proposal` builds the
   :class:`PersonaLoRAProposal` adapter — a thin wrapper that
   pre-fills :class:`ModificationProposal` for the figure-vertical
   conventions: ``target = "figure.persona_lora[<figure_id>]"``,
   ``desired_gate = OFFLINE``, ``is_reversible = True``, and a
   ``rollback_evidence`` discipline that pins both the previous
   LoRA artifact id and the training-plan integrity hash.

Both pieces are pure / deterministic. The plan integrity hash
encodes every training example, every replay item, and the
hyper-parameters; running the bake (P6.2) twice on the same plan
must produce a byte-for-byte identical artifact (R15).

R10 reminder: a baked LoRA changes who the lifeform is in a
non-trivial way (it shifts representation in the substrate's
adapter layer). It must travel via :class:`ModificationGate.OFFLINE`,
never the online path, even with strong evaluation evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lifeform_ingestion.envelope import IngestionEnvelope
from volvence_zero.credit.gate import (
    ModificationGate,
    ModificationProposal,
)


SCHEMA_VERSION = 1

_DEFAULT_REPLAY_BUFFER: tuple[tuple[str, str], ...] = (
    (
        "replay:wikipedia:water-cycle",
        (
            "Water on the surface of the planet evaporates, condenses into "
            "clouds, and returns as precipitation. The cycle is driven by "
            "incoming solar radiation and gravity."
        ),
    ),
    (
        "replay:wikipedia:photosynthesis",
        (
            "Plants convert light energy, water, and carbon dioxide into "
            "glucose and oxygen via photosynthesis. The reaction occurs in "
            "chloroplasts and underpins most food webs."
        ),
    ),
    (
        "replay:wikipedia:periodic-table",
        (
            "Elements are arranged by atomic number; properties recur in "
            "groups. The periodic law lets chemists predict reactivity and "
            "atomic structure for unfamiliar elements."
        ),
    ),
    (
        "replay:wikipedia:plate-tectonics",
        (
            "Earth's lithosphere is divided into plates that move slowly "
            "over the asthenosphere. Plate boundaries explain earthquakes, "
            "volcanism, and mountain formation."
        ),
    ),
    (
        "replay:wikipedia:cellular-respiration",
        (
            "Cells release energy from glucose through cellular respiration, "
            "producing ATP, carbon dioxide, and water. The process spans "
            "glycolysis, the citric acid cycle, and oxidative phosphorylation."
        ),
    ),
    (
        "replay:wikipedia:newtonian-mechanics",
        (
            "Newtonian mechanics describes motion through three laws: "
            "inertia, force equals mass times acceleration, and equal and "
            "opposite reactions. It is accurate for everyday speeds and "
            "scales."
        ),
    ),
)


@dataclass(frozen=True)
class LoRATrainingExample:
    """One training row consumed by the LoRA bake (P6.2).

    Fields:

    * ``example_id``    — stable identifier (kept across runs).
    * ``text``          — the training text (paragraph-sized).
    * ``locator``       — citation string back to the originating
                          ingestion chunk OR the replay-buffer id.
    * ``source_kind``   — ``"figure"`` (corpus we adapt to) or
                          ``"replay"`` (general-purpose anti-forgetting
                          interleave).
    * ``weight``        — per-row weight for the bake's loss
                          accumulator (figure rows default to 1.0,
                          replay rows default to ``replay_weight``).
    """

    example_id: str
    text: str
    locator: str
    source_kind: str
    weight: float

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("LoRATrainingExample.example_id must be non-empty")
        if not self.text.strip():
            raise ValueError(
                "LoRATrainingExample.text must be non-empty (silent empty "
                f"rows are forbidden); example_id={self.example_id!r}"
            )
        if not self.locator.strip():
            raise ValueError(
                f"LoRATrainingExample.locator must be non-empty; "
                f"example_id={self.example_id!r}"
            )
        if self.source_kind not in {"figure", "replay"}:
            raise ValueError(
                f"LoRATrainingExample.source_kind must be 'figure' or "
                f"'replay', got {self.source_kind!r}"
            )
        if self.weight < 0.0:
            raise ValueError(
                f"LoRATrainingExample.weight must be >= 0, got {self.weight!r}"
            )


@dataclass(frozen=True)
class LoRATrainingPlan:
    """Frozen deterministic LoRA training plan.

    The plan is the only thing the bake (P6.2) consumes; same plan
    in → same baked artifact out (R15). The integrity hash binds
    every example + hyper-parameter so a downstream
    :class:`PersonaLoRAProposal` can pin it as ``new_value_hash``.
    """

    schema_version: int
    figure_id: str
    examples: tuple[LoRATrainingExample, ...]
    rank: int
    target_layer_index: int
    learning_rate: float
    epochs: int
    figure_example_count: int
    replay_example_count: int
    integrity_hash: str
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"LoRATrainingPlan.schema_version mismatch: "
                f"got {self.schema_version!r}, expected {SCHEMA_VERSION!r}"
            )
        if not self.figure_id.strip():
            raise ValueError("LoRATrainingPlan.figure_id must be non-empty")
        if not self.examples:
            raise ValueError(
                "LoRATrainingPlan.examples must be non-empty; refusing to "
                "build a degenerate training plan."
            )
        if self.rank <= 0:
            raise ValueError(
                f"LoRATrainingPlan.rank must be > 0, got {self.rank!r}"
            )
        if self.target_layer_index < 0:
            raise ValueError(
                f"LoRATrainingPlan.target_layer_index must be >= 0, "
                f"got {self.target_layer_index!r}"
            )
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError(
                f"LoRATrainingPlan.learning_rate must be in (0,1), "
                f"got {self.learning_rate!r}"
            )
        if self.epochs <= 0:
            raise ValueError(
                f"LoRATrainingPlan.epochs must be > 0, got {self.epochs!r}"
            )
        if not self.integrity_hash.strip():
            raise ValueError(
                "LoRATrainingPlan.integrity_hash must be non-empty"
            )
        figure_count = sum(
            1 for ex in self.examples if ex.source_kind == "figure"
        )
        replay_count = sum(
            1 for ex in self.examples if ex.source_kind == "replay"
        )
        if figure_count != self.figure_example_count:
            raise ValueError(
                f"LoRATrainingPlan.figure_example_count={self.figure_example_count!r} "
                f"does not match observed figure rows ({figure_count})"
            )
        if replay_count != self.replay_example_count:
            raise ValueError(
                f"LoRATrainingPlan.replay_example_count={self.replay_example_count!r} "
                f"does not match observed replay rows ({replay_count})"
            )
        if figure_count == 0:
            raise ValueError(
                "LoRATrainingPlan: figure example block is empty; refusing to "
                "build a LoRA plan with only replay-buffer rows."
            )

    @property
    def total_examples(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class PersonaLoRAProposal:
    """Figure-vertical adapter over :class:`ModificationProposal`.

    Carries the underlying :class:`ModificationProposal` plus the
    typed fields the audit log needs: figure id, the previous
    artifact id (or ``"absent"``), and the bound training plan
    hash. Use :func:`build_persona_lora_proposal` to construct;
    direct construction is allowed but the post-init guards every
    invariant.
    """

    figure_id: str
    previous_artifact_id: str
    training_plan_hash: str
    proposal: ModificationProposal

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("PersonaLoRAProposal.figure_id must be non-empty")
        if not self.previous_artifact_id.strip():
            raise ValueError(
                "PersonaLoRAProposal.previous_artifact_id must be non-empty "
                "(use 'absent' if there is no prior artifact)."
            )
        if not self.training_plan_hash.strip():
            raise ValueError(
                "PersonaLoRAProposal.training_plan_hash must be non-empty"
            )
        if self.proposal.desired_gate is not ModificationGate.OFFLINE:
            raise ValueError(
                "PersonaLoRAProposal.proposal.desired_gate must be OFFLINE; "
                f"got {self.proposal.desired_gate!r}"
            )
        if not self.proposal.is_reversible:
            raise ValueError(
                "PersonaLoRAProposal.proposal.is_reversible must be True"
            )
        if not self.proposal.rollback_evidence.strip():
            raise ValueError(
                "PersonaLoRAProposal.proposal.rollback_evidence must be "
                "non-empty (the OFFLINE gate requires it)."
            )
        expected_target = f"figure.persona_lora[{self.figure_id}]"
        if self.proposal.target != expected_target:
            raise ValueError(
                f"PersonaLoRAProposal.proposal.target must be "
                f"{expected_target!r}, got {self.proposal.target!r}"
            )


def build_lora_training_plan(
    *,
    figure_id: str,
    envelopes: tuple[IngestionEnvelope, ...],
    rank: int = 8,
    target_layer_index: int = 0,
    learning_rate: float = 1e-4,
    epochs: int = 1,
    replay_weight: float = 0.5,
    extra_replay_examples: tuple[tuple[str, str], ...] = (),
    description: str | None = None,
) -> LoRATrainingPlan:
    """Assemble a deterministic LoRA training plan from corpus envelopes.

    Steps:

    1. Walk every ingestion envelope's ``successful_chunks`` and
       turn each into a ``LoRATrainingExample`` with
       ``source_kind="figure"`` and ``weight = chunk.confidence``.
    2. Append the built-in :data:`_DEFAULT_REPLAY_BUFFER` plus any
       caller-supplied ``extra_replay_examples`` as
       ``source_kind="replay"`` rows with ``weight = replay_weight``.
    3. Compute an integrity hash over (examples + hyper-params) so
       the bake output (P6.2) can pin the plan's identity.
    """

    if not figure_id.strip():
        raise ValueError(
            "build_lora_training_plan: figure_id must be non-empty"
        )
    if not envelopes:
        raise ValueError(
            "build_lora_training_plan: envelopes tuple must be non-empty"
        )
    if not (0.0 <= replay_weight <= 1.0):
        raise ValueError(
            f"build_lora_training_plan: replay_weight must be in [0,1], "
            f"got {replay_weight!r}"
        )
    examples: list[LoRATrainingExample] = []
    figure_count = 0
    for envelope in envelopes:
        for chunk in envelope.successful_chunks:
            example_id = (
                f"figure-row:{envelope.envelope_id}:{chunk.chunk_id}"
            )
            examples.append(
                LoRATrainingExample(
                    example_id=example_id,
                    text=chunk.text,
                    locator=f"{chunk.locator} | {envelope.envelope_id}",
                    source_kind="figure",
                    weight=chunk.confidence,
                )
            )
            figure_count += 1
    if figure_count == 0:
        raise ValueError(
            "build_lora_training_plan: no successful chunks across all "
            "envelopes — refusing to build a LoRA plan with only replay rows."
        )
    replay_examples: list[tuple[str, str]] = list(_DEFAULT_REPLAY_BUFFER)
    seen_replay_ids = {row[0] for row in replay_examples}
    for replay_id, replay_text in extra_replay_examples:
        if replay_id in seen_replay_ids:
            raise ValueError(
                f"build_lora_training_plan: extra_replay_examples has "
                f"duplicate replay_id {replay_id!r}"
            )
        seen_replay_ids.add(replay_id)
        replay_examples.append((replay_id, replay_text))
    replay_count = 0
    for replay_id, replay_text in replay_examples:
        examples.append(
            LoRATrainingExample(
                example_id=replay_id,
                text=replay_text,
                locator=replay_id,
                source_kind="replay",
                weight=replay_weight,
            )
        )
        replay_count += 1
    integrity_hash = _compute_plan_integrity_hash(
        figure_id=figure_id,
        examples=tuple(examples),
        rank=rank,
        target_layer_index=target_layer_index,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    return LoRATrainingPlan(
        schema_version=SCHEMA_VERSION,
        figure_id=figure_id,
        examples=tuple(examples),
        rank=rank,
        target_layer_index=target_layer_index,
        learning_rate=learning_rate,
        epochs=epochs,
        figure_example_count=figure_count,
        replay_example_count=replay_count,
        integrity_hash=integrity_hash,
        description=(
            description
            or (
                f"LoRA training plan for {figure_id}: {figure_count} figure "
                f"rows + {replay_count} replay rows; rank={rank}, "
                f"target_layer={target_layer_index}, lr={learning_rate}, "
                f"epochs={epochs}."
            )
        ),
    )


def build_persona_lora_proposal(
    *,
    figure_id: str,
    plan: LoRATrainingPlan,
    new_artifact_integrity_hash: str,
    previous_artifact_id: str,
    rollback_evidence: str,
    validation_delta: float = 0.05,
    capacity_cost: float = 0.30,
    justification: str | None = None,
) -> PersonaLoRAProposal:
    """Build a ready-to-gate :class:`PersonaLoRAProposal`.

    The underlying :class:`ModificationProposal` pins:

    * ``target = "figure.persona_lora[<figure_id>]"``
    * ``desired_gate = OFFLINE``
    * ``old_value_hash`` = sha256 over the previous artifact id
    * ``new_value_hash`` = sha256 over (training plan hash +
      new artifact integrity hash)
    * ``is_reversible = True``
    * ``rollback_evidence`` = caller-supplied non-empty string
    """

    if not figure_id.strip():
        raise ValueError(
            "build_persona_lora_proposal: figure_id must be non-empty"
        )
    if plan.figure_id != figure_id:
        raise ValueError(
            f"build_persona_lora_proposal: plan.figure_id={plan.figure_id!r} "
            f"does not match figure_id={figure_id!r}"
        )
    if not new_artifact_integrity_hash.strip():
        raise ValueError(
            "build_persona_lora_proposal: new_artifact_integrity_hash "
            "must be non-empty"
        )
    if not previous_artifact_id.strip():
        raise ValueError(
            "build_persona_lora_proposal: previous_artifact_id must be "
            "non-empty (pass 'absent' if there is no prior artifact)."
        )
    if not rollback_evidence.strip():
        raise ValueError(
            "build_persona_lora_proposal: rollback_evidence must be non-empty"
        )
    old_value_repr = repr(
        (
            "figure.persona_lora",
            figure_id,
            previous_artifact_id,
        )
    )
    new_value_repr = repr(
        (
            "figure.persona_lora",
            figure_id,
            plan.integrity_hash,
            new_artifact_integrity_hash,
        )
    )
    proposal = ModificationProposal(
        target=f"figure.persona_lora[{figure_id}]",
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=hashlib.sha256(
            old_value_repr.encode("utf-8")
        ).hexdigest(),
        new_value_hash=hashlib.sha256(
            new_value_repr.encode("utf-8")
        ).hexdigest(),
        justification=(
            justification
            or (
                f"Bake persona LoRA for {figure_id} from training plan "
                f"with {plan.figure_example_count} figure rows + "
                f"{plan.replay_example_count} replay rows."
            )
        ),
        is_reversible=True,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )
    return PersonaLoRAProposal(
        figure_id=figure_id,
        previous_artifact_id=previous_artifact_id,
        training_plan_hash=plan.integrity_hash,
        proposal=proposal,
    )


def _compute_plan_integrity_hash(
    *,
    figure_id: str,
    examples: tuple[LoRATrainingExample, ...],
    rank: int,
    target_layer_index: int,
    learning_rate: float,
    epochs: int,
) -> str:
    payload = (
        SCHEMA_VERSION,
        figure_id,
        rank,
        target_layer_index,
        round(learning_rate, 9),
        epochs,
        tuple(
            (
                ex.example_id,
                ex.locator,
                ex.source_kind,
                round(ex.weight, 6),
                hashlib.sha256(ex.text.encode("utf-8")).hexdigest(),
            )
            for ex in examples
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "LoRATrainingExample",
    "LoRATrainingPlan",
    "PersonaLoRAProposal",
    "build_lora_training_plan",
    "build_persona_lora_proposal",
]
