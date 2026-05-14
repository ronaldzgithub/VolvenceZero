"""Reviewed growth-advisor profile schema.

This vertical encodes the "long-term private-domain growth-advisor"
archetype: a parental-companion persona that helps families navigate
child height / immunity / nutrition / vision questions across a
multi-day relationship (the LTV path), backed by reviewed knowledge
seeds and explicit anti-sales boundaries.

The schema deliberately mirrors ``lifeform-domain-character`` in spirit
(reviewed structured artifacts → existing kernel application owners)
but does **not** import that wheel. ``PARALLEL_VERTICAL_PAIRS`` in
``tests/contracts/test_import_boundaries.py`` enforces that constraint.
The renamed ``GrowthAdvisor*`` types compile down to the same
``vz-application`` records (``DomainKnowledgeRecord`` / ``CaseMemoryRecord``
/ ``PlaybookRule`` / ``BoundaryPriorHint``) that every other vertical
also uses; ownership is by typed instance, not by schema.

Why a fresh schema here rather than reusing ``CharacterSoulProfile``:

* Verticals are R8 owners of their own reviewed bundles. Two parallel
  verticals sharing one schema would create an implicit second owner.
* The growth-advisor archetype has structurally different reviewed
  artifacts (multi-day playbook days, mining funnels, user archetypes
  for parental anxiety) than fictional-character roleplay. Encoding
  those as first-class fields keeps the compilation path readable.

Nothing in this module decides behavior from raw user text. All
behavior signals reach the kernel via the typed compiled records;
``no-keyword-matching-hacks.mdc`` invariants are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.substrate import SubstrateFingerprint


@dataclass(frozen=True)
class GrowthAdvisorKnowledgeSeed:
    """Reviewed knowledge / value / persona statement.

    Compiles to ``DomainKnowledgeRecord``. Domains used by this
    vertical include ``persona_self`` (人设核心), ``user_archetype``
    (5 类 mom 心态), and ``child_nutrition`` (儿童身高 / 营养 / 抵抗力 /
    视力等领域知识).
    """

    seed_id: str
    domain: str
    title: str
    summary: str
    snippet: str
    evidence_locator: str
    confidence: float
    evidence_strength: str = "medium"
    topic_tags: tuple[str, ...] = ()
    source_type: str = "internal-guide"
    freshness_label: str = "reviewed"


@dataclass(frozen=True)
class GrowthAdvisorSignatureCase:
    """Concrete situation-action-outcome pattern.

    Compiles to ``CaseMemoryRecord``. Used by this vertical for the
    7 onboarding-arc playbook examples (icebreaker → summary; phase
    routing flows through ``BehaviorProtocol.TemporalArc.progression_signals``,
    not calendar days) and the 5 universal-response cases (no-reply /
    asks for a standard / vents / asks for supplements / asks for a
    product).
    """

    case_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    intervention_ordering: tuple[str, ...]
    outcome_label: str
    description: str
    confidence: float
    relevance_score: float = 0.75
    escalation_observed: bool = False
    repair_observed: bool = False


@dataclass(frozen=True)
class GrowthAdvisorStrategyPrior:
    """Reviewed pacing / ordering prior.

    Compiles to ``PlaybookRule``. The 7 onboarding-arc priors carry
    ``applicability_scope`` with funnel / regime tags only; the
    previous calendar-day tags (``growth_advisor:day1`` …
    ``growth_advisor:day7``) were removed on 2026-05-14 — phase
    routing now flows through
    ``BehaviorProtocol.TemporalArc.progression_signals`` (PE-driven)
    in protocol-runtime. The 4 need-mining funnels use
    ``funnel:height``, ``funnel:immunity``, ``funnel:nutrition``,
    ``funnel:vision``.
    """

    rule_id: str
    problem_pattern: str
    recommended_regime: str | None
    recommended_ordering: tuple[str, ...]
    recommended_pacing: str
    avoid_patterns: tuple[str, ...]
    applicability_scope: tuple[str, ...]
    confidence: float
    description: str
    knowledge_weight_hint: float = 0.45
    experience_weight_hint: float = 0.65


@dataclass(frozen=True)
class GrowthAdvisorBoundaryPrior:
    """Reviewed boundary / anti-pattern.

    Compiles to ``BoundaryPriorHint``. The four anchoring boundaries
    for this vertical (``bp-no-hard-sell`` / ``bp-no-overclaim`` /
    ``bp-no-flooding`` / ``bp-no-judgmental``) keep the persona honest
    when the LLM expression layer would otherwise drift toward sales-
    pitch behaviour.
    """

    boundary_id: str
    regime_id: str | None
    trigger_reasons: tuple[str, ...]
    answer_depth_limit_hint: str
    clarification_required: bool
    refer_out_required: bool
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    confidence: float
    description: str


@dataclass(frozen=True)
class GrowthAdvisorDrivePrior:
    """One drive channel for the growth-advisor's homeostatic profile.

    Compiles to ``lifeform_core.DriveSpec`` via the compiler. The four
    canonical drives for this vertical are ``trust_building_drive``,
    ``empathy_response_drive``, ``restraint_against_pitch_drive``, and
    ``kb_share_drive``.
    """

    name: str
    target: float
    homeostatic_band: tuple[float, float]
    decay_per_tick: float
    pe_weight: float
    initial_level: float = 0.5
    recharge_per_turn: float = 0.0
    recharge_per_regime: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class GrowthAdvisorProfile:
    """Reviewed artifact compiled into existing runtime owners.

    The shape is intentionally similar to ``CharacterSoulProfile`` so
    that anyone who has read one vertical can read the other; the
    types are distinct, however, so each vertical owns its own
    bundle.
    """

    profile_id: str
    advisor_name: str
    source_title: str
    version: str
    reviewed_by: str
    source_uri: str
    description: str
    knowledge_seeds: tuple[GrowthAdvisorKnowledgeSeed, ...]
    signature_cases: tuple[GrowthAdvisorSignatureCase, ...]
    strategy_priors: tuple[GrowthAdvisorStrategyPrior, ...]
    boundary_priors: tuple[GrowthAdvisorBoundaryPrior, ...]
    drive_priors: tuple[GrowthAdvisorDrivePrior, ...] = ()
    target_contexts: tuple[str, ...] = (
        "private-domain-growth-advisor",
        "child-nutrition-companion",
    )
    # debt #47 / F-C SHADOW: substrate fingerprints this profile has
    # been validated against. Empty tuple = "untested / generic"
    # (runtime warns on substrate mismatch but does not fail; profile
    # is application-layer typed records, not substrate-bound weights).
    # Non-empty tuple lets ops verify "this reviewed cheng-laoshi was
    # tested on Qwen2.5-1.5B and Llama-3.1-8B" before promoting.
    validated_substrates: tuple[SubstrateFingerprint, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("profile_id", self.profile_id)
        _require_non_empty("advisor_name", self.advisor_name)
        _require_non_empty("source_title", self.source_title)
        _require_non_empty("version", self.version)
        _require_non_empty("reviewed_by", self.reviewed_by)
        _require_non_empty("source_uri", self.source_uri)
        if not self.boundary_priors:
            raise ValueError(
                "GrowthAdvisorProfile.boundary_priors must be non-empty: "
                "the growth-advisor archetype requires explicit anti-sales / "
                "anti-overclaim / anti-flooding / anti-judgmental boundaries"
            )
        _check_unique(
            "knowledge_seeds.seed_id",
            tuple(seed.seed_id for seed in self.knowledge_seeds),
        )
        _check_unique(
            "signature_cases.case_id",
            tuple(case.case_id for case in self.signature_cases),
        )
        _check_unique(
            "strategy_priors.rule_id",
            tuple(rule.rule_id for rule in self.strategy_priors),
        )
        _check_unique(
            "boundary_priors.boundary_id",
            tuple(boundary.boundary_id for boundary in self.boundary_priors),
        )
        _check_unique(
            "drive_priors.name",
            tuple(drive.name for drive in self.drive_priors),
        )


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _check_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique, got {values!r}")


__all__ = [
    "GrowthAdvisorBoundaryPrior",
    "GrowthAdvisorDrivePrior",
    "GrowthAdvisorKnowledgeSeed",
    "GrowthAdvisorProfile",
    "GrowthAdvisorSignatureCase",
    "GrowthAdvisorStrategyPrior",
]
