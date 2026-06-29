"""Apprenticeship-alignment contract surface (R-PE / R7 / R11).

Pure data-only module for the ``apprenticeship_alignment`` owner. Holds
the enums and frozen dataclasses that the owner publishes each turn.

The owner implements *reliable active apprenticeship learning*
(Hanneke, Yang, Wang & Song, ALT 2025) inside the VolvenceZero kernel:
the operator's teaching in an apprentice turn is an expert query/answer
that constrains a *version space* of operator-intended cognition; the
owner compares each piece of guidance against the AI's current public
cognition snapshots (``belief_assumption`` / ``goal_value`` /
``user_model`` / ``boundary_consent``) and reports

* whether the AI's cognition is *pinned* by the guidance (agreement
  region => the AI may act reliably) or not (disagreement region =>
  the AI should defer / surface uncertainty),
* the eluder-style *informativeness* of the guidance
  (``guidance_surprise``),
* per-constraint *mismatches* against current cognition, classified at
  factual vs abstract level using a CLAIRE-style taxonomy, and
* *contradictions* — when no single operator-intent hypothesis can
  satisfy all guidance (version-space collapse), which is the rigorous
  judge of "the training material contradicts itself".

The owner only *publishes* this snapshot. Belief revision goes through
the existing ``SemanticProposal`` pathway (single-writer
``belief_assumption`` / ``goal_value``); the PE owner consumes this
snapshot to fuse a discrete-event apprenticeship PE source. No consumer
rebuilds this owner's internal state (R8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConstraintLevel(str, Enum):
    """Whether an operator-intent constraint is factual or abstract.

    Factual constraints assert something about the world the AI can be
    right or wrong about (a date, a number, an entity, a definition).
    Abstract constraints assert a value priority, principle, or
    behaviour strategy. The two levels surface mismatches the operator
    cares about separately ("you got a fact wrong" vs "your priorities
    diverge from what I'm teaching").
    """

    FACTUAL = "factual"
    ABSTRACT = "abstract"


class MismatchType(str, Enum):
    """CLAIRE-style taxonomy of a guidance-vs-cognition mismatch.

    The factual subset mirrors the WikiCollide discrepancy taxonomy
    (Stanford OVAL, EMNLP 2025); the abstract subset names the
    value / strategy conflicts that matter for a cultivated persona.
    ``*_NOVELTY`` flags guidance the current cognition simply does not
    cover yet (disagreement region) rather than an outright clash.
    """

    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    DEFINITION = "definition"
    DUALITY = "duality"
    CATEGORICAL = "categorical"
    VALUE_CONFLICT = "value_conflict"
    STRATEGY_CONFLICT = "strategy_conflict"
    FACTUAL_NOVELTY = "factual_novelty"
    ABSTRACT_NOVELTY = "abstract_novelty"


class VersionSpaceStatus(str, Enum):
    """State of the operator-intent version space this turn.

    * ``IDLE`` — no guidance this turn (not an apprentice turn / empty
      input); nothing to reconcile.
    * ``CONSISTENT`` — guidance is covered by current cognition; the
      version space is unchanged / barely shrinks.
    * ``SHRINKING`` — informative guidance meaningfully narrows the
      version space (high eluder surprise) but stays satisfiable.
    * ``INCONSISTENT`` — no operator-intent hypothesis satisfies all
      guidance; a contradiction in the material/teaching is detected.
    """

    IDLE = "idle"
    CONSISTENT = "consistent"
    SHRINKING = "shrinking"
    INCONSISTENT = "inconsistent"


class ReliabilityState(str, Enum):
    """Whether the AI may act on its own cognition this turn.

    Reliable-active-apprenticeship guarantee: the learner only acts
    without querying the expert when its action is guaranteed optimal.
    Here ``RELIABLE`` means the current cognition sits in the agreement
    region (pinned by guidance); ``DEFERRING`` means the disagreement
    region (the AI should surface uncertainty / seek guidance rather
    than commit silently); ``IDLE`` means there is no guidance to gate
    on.
    """

    IDLE = "idle"
    RELIABLE = "reliable"
    DEFERRING = "deferring"


@dataclass(frozen=True)
class IntentConstraint:
    """One operator-intent constraint extracted from a teaching turn.

    ``polarity`` encodes assertion (+1) vs negation (-1) vs neutral (0)
    so the version-space solver can detect opposing constraints on the
    same topic without keyword matching (the stance comes from the
    extractor — an LLM in production, explicit in tests). Topic
    similarity (coverage / contradiction matching) is computed from
    ``target_key`` via character-bigram token overlap. ``embedding`` is
    reserved for the production path (a real embedding head) and is
    empty under the current stub.
    """

    constraint_id: str
    statement: str
    level: str
    polarity: int
    target_key: str
    confidence: float
    source_turn: int
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class MismatchRef:
    """A single guidance-vs-cognition mismatch, level + type classified."""

    guidance_constraint_id: str
    level: str
    mismatch_type: str
    belief_ref: str
    severity: float
    description: str


@dataclass(frozen=True)
class ContradictionFinding:
    """A minimal mutually-inconsistent set of operator-intent constraints.

    ``constraint_ids`` are the two (or more) constraints that cannot
    co-exist — the rigorous, locatable evidence that the material /
    teaching contradicts itself ("section X and section Y collide").
    """

    finding_id: str
    constraint_ids: tuple[str, ...]
    level: str
    severity: float
    description: str


@dataclass(frozen=True)
class ApprenticeshipAlignmentSnapshot:
    """Per-turn reliable-apprenticeship readout published by the owner."""

    version_space_status: str
    consistency_margin: float
    reliability: str
    in_agreement_region: bool
    guidance_surprise: float
    active_constraint_count: int
    mismatch_refs: tuple[MismatchRef, ...]
    contradiction_findings: tuple[ContradictionFinding, ...]
    revision_proposal_refs: tuple[str, ...]
    description: str
    memory_retrieval_facets: tuple[str, ...] = field(default=())


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))
