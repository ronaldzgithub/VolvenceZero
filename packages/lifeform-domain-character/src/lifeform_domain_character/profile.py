"""Reviewed character profile schema.

This package deliberately starts from reviewed, structured artifacts. It does
not inspect arbitrary novel text to decide behavior: extraction can be done by a
separate LLM structured-output or human review process, then this module compiles
the result into the existing Volvence Zero owners.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterKnowledgeSeed:
    """Reviewed value, belief, or world-model statement for the character."""

    seed_id: str
    domain: str
    title: str
    summary: str
    snippet: str
    evidence_locator: str
    confidence: float
    evidence_strength: str = "medium"
    topic_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CharacterSignatureCase:
    """A concrete situation-action-outcome pattern from the source material."""

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
class CharacterStrategyPrior:
    """A reviewed pacing or ordering prior derived from multiple cases."""

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
class CharacterBoundaryPrior:
    """A reviewed boundary or anti-pattern for the character vertical."""

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
class CharacterDrivePrior:
    """One drive channel for the character's always-on pressure profile."""

    name: str
    target: float
    homeostatic_band: tuple[float, float]
    decay_per_tick: float
    pe_weight: float
    initial_level: float = 0.5
    recharge_per_turn: float = 0.0
    recharge_per_regime: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CharacterSoulProfile:
    """Reviewed artifact that can be compiled into existing runtime owners."""

    profile_id: str
    character_name: str
    source_title: str
    version: str
    reviewed_by: str
    source_uri: str
    description: str
    knowledge_seeds: tuple[CharacterKnowledgeSeed, ...]
    signature_cases: tuple[CharacterSignatureCase, ...]
    strategy_priors: tuple[CharacterStrategyPrior, ...]
    boundary_priors: tuple[CharacterBoundaryPrior, ...]
    drive_priors: tuple[CharacterDrivePrior, ...] = ()
    target_contexts: tuple[str, ...] = ("character-companion", "fictional-roleplay")

    def __post_init__(self) -> None:
        _require_non_empty("profile_id", self.profile_id)
        _require_non_empty("character_name", self.character_name)
        _require_non_empty("source_title", self.source_title)
        _require_non_empty("version", self.version)
        _require_non_empty("reviewed_by", self.reviewed_by)
        _require_non_empty("source_uri", self.source_uri)
        if not self.boundary_priors:
            raise ValueError("CharacterSoulProfile.boundary_priors must be non-empty")
        _check_unique("knowledge_seeds.seed_id", tuple(seed.seed_id for seed in self.knowledge_seeds))
        _check_unique("signature_cases.case_id", tuple(case.case_id for case in self.signature_cases))
        _check_unique("strategy_priors.rule_id", tuple(rule.rule_id for rule in self.strategy_priors))
        _check_unique("boundary_priors.boundary_id", tuple(boundary.boundary_id for boundary in self.boundary_priors))
        _check_unique("drive_priors.name", tuple(drive.name for drive in self.drive_priors))


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _check_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique, got {values!r}")


__all__ = [
    "CharacterBoundaryPrior",
    "CharacterDrivePrior",
    "CharacterKnowledgeSeed",
    "CharacterSignatureCase",
    "CharacterSoulProfile",
    "CharacterStrategyPrior",
]
