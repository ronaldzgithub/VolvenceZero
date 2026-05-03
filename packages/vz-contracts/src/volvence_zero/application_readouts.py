"""Application readout protocols (R2 / R8).

These ``Protocol`` definitions are the **only** thing the kernel
evaluation layer (``vz-cognition.evaluation``) is allowed to know
about application-tier snapshots. They exist so the kernel can score
application readouts (case memory, domain knowledge, boundary policy,
playbook, experience prior, response assembly, outcome attribution,
sequence payoff) without importing concrete dataclasses from
``vz-application.application.runtime``.

Why this matters:

* R2 (stable substrate + adaptive controllers): the evaluation layer
  is a kernel readout. It must not pull product-tier schema into the
  kernel just to compute scores.
* R8 (snapshot-first / contract-first): cross-tier exchange goes
  through immutable contracts; the consumer never reconstructs the
  producer's internal state and never depends on the producer's
  concrete class hierarchy.

These protocols are deliberately **minimal**: they declare only the
attributes that ``vz-cognition.evaluation`` actually reads. Adding a
new field for evaluation purposes therefore requires updating both
the protocol here and the producing dataclass in ``vz-application``.
A producer dataclass with extra fields is still structurally
compatible: Python ``Protocol`` matching is by attribute presence and
type, not by exhaustive equality.

Slice C (2026-05-03) replaces the previous ``application_types.py``
cycle-break shim (which lived inside ``vz-cognition`` and physically
hosted application-tier dataclasses) with this Protocol surface.
"""

from __future__ import annotations

from typing import Protocol


# ---------------------------------------------------------------------------
# Boundary policy
# ---------------------------------------------------------------------------


class BoundaryDecisionReadout(Protocol):
    """Minimal view of a boundary decision used by evaluation scoring."""

    clarification_required: bool
    refer_out_required: bool
    citation_required: bool


class BoundaryReadout(Protocol):
    """Boundary policy readout consumed by evaluation."""

    active_decision: BoundaryDecisionReadout
    description: str


# ---------------------------------------------------------------------------
# Case memory
# ---------------------------------------------------------------------------


class CaseOutcomeSummaryReadout(Protocol):
    """Per-case outcome readout used to derive delayed-signal evidence."""

    delayed_signal_count: int


class CaseEpisodeHitReadout(Protocol):
    """Per-hit case readout consumed by case-memory scoring."""

    relevance_score: float
    outcome: CaseOutcomeSummaryReadout


class CaseMemoryReadout(Protocol):
    """Case memory readout consumed by evaluation."""

    hits: tuple[CaseEpisodeHitReadout, ...]
    active_problem_patterns: tuple[str, ...]
    active_band_ids: tuple[str, ...]
    mean_continuum_position: float


# ---------------------------------------------------------------------------
# Domain knowledge
# ---------------------------------------------------------------------------


class DomainKnowledgeReadout(Protocol):
    """Domain-knowledge readout consumed by evaluation.

    ``hits`` is intentionally typed as ``tuple[object, ...]``: the
    evaluation layer only consumes the count; per-hit fields are not
    needed for scoring.
    """

    hits: tuple[object, ...]
    citation_required: bool
    unresolved_conflicts: tuple[str, ...]


# ---------------------------------------------------------------------------
# Strategy playbook
# ---------------------------------------------------------------------------


class PlaybookRuleReadout(Protocol):
    """Per-rule playbook readout consumed by evaluation."""

    confidence: float


class StrategyPlaybookReadout(Protocol):
    """Strategy playbook readout consumed by evaluation."""

    matched_rules: tuple[PlaybookRuleReadout, ...]
    matched_problem_patterns: tuple[str, ...]
    active_band_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


class ResponseAssemblyReadout(Protocol):
    """Response-assembly readout consumed by evaluation."""

    answer_depth_limit: str
    max_questions: int
    refer_out_required: bool
    required_disclaimer_phrases: tuple[str, ...]
    ordering_plan: tuple[str, ...]
    prompt_residue_ratio: float
    prompt_residue_summary: str
    support_before_decision_pressure: float
    eta_action_family: str
    description: str


# ---------------------------------------------------------------------------
# Experience fast prior
# ---------------------------------------------------------------------------


class ExperienceFastPriorRegimeBiasReadout(Protocol):
    """Per-regime bias readout from the experience-prior surface."""

    bias: float


class ExperienceFastPriorFamilyBiasReadout(Protocol):
    """Per-family continuation bias readout from the experience-prior surface."""

    continuation_bias: float


class ExperienceFastPriorReadout(Protocol):
    """Experience fast-prior readout consumed by evaluation."""

    source_attribution_ids: tuple[str, ...]
    source_sequence_ids: tuple[str, ...]
    description: str
    regime_biases: tuple[ExperienceFastPriorRegimeBiasReadout, ...]
    family_biases: tuple[ExperienceFastPriorFamilyBiasReadout, ...]
    knowledge_weight_bias: float
    experience_weight_bias: float


# ---------------------------------------------------------------------------
# Application outcome attribution + sequence payoff
# ---------------------------------------------------------------------------


class ApplicationOutcomeAttributionReadout(Protocol):
    """Per-record attribution readout consumed by delayed-evidence scoring."""

    outcome_score: float
    retrieval_mix_alignment: float
    regime_alignment: float
    abstract_action_alignment: float
    continuum_alignment: float


class ApplicationSequencePayoffReadout(Protocol):
    """Per-sequence payoff readout consumed by sequence-evidence scoring."""

    rolling_payoff: float
    mean_continuum_position: float


__all__ = [
    "ApplicationOutcomeAttributionReadout",
    "ApplicationSequencePayoffReadout",
    "BoundaryDecisionReadout",
    "BoundaryReadout",
    "CaseEpisodeHitReadout",
    "CaseMemoryReadout",
    "CaseOutcomeSummaryReadout",
    "DomainKnowledgeReadout",
    "ExperienceFastPriorFamilyBiasReadout",
    "ExperienceFastPriorReadout",
    "ExperienceFastPriorRegimeBiasReadout",
    "PlaybookRuleReadout",
    "ResponseAssemblyReadout",
    "StrategyPlaybookReadout",
]
