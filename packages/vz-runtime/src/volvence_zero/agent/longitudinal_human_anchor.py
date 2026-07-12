"""Longitudinal + human-anchor study manifests (plan §14).

The runtime already has benchmark/report builders for dialogue longitudinal
and human ratings. This module freezes the *study shape* required by the
12-month plan so operators can schedule evidence runs without turning a
spreadsheet into the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongitudinalPersonaPlan:
    persona_id: str
    session_count: int
    min_turns_per_session: int
    max_turns_per_session: int
    comparison_arms: tuple[str, ...]
    tracked_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.persona_id.strip():
            raise ValueError("persona_id must be non-empty")
        if self.session_count < 20:
            raise ValueError("session_count must be >= 20")
        if self.min_turns_per_session <= 0:
            raise ValueError("min_turns_per_session must be positive")
        if self.max_turns_per_session < self.min_turns_per_session:
            raise ValueError("max_turns_per_session must be >= min_turns_per_session")
        if not self.comparison_arms:
            raise ValueError("comparison_arms must be non-empty")
        if not self.tracked_metrics:
            raise ValueError("tracked_metrics must be non-empty")


@dataclass(frozen=True)
class HumanAnchorProtocol:
    blinded_rater_count: int
    min_inter_rater_agreement: float
    comparison_arms: tuple[str, ...]
    hidden_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.blinded_rater_count < 3:
            raise ValueError("blinded_rater_count must be >= 3")
        if self.min_inter_rater_agreement < 0.0 or self.min_inter_rater_agreement > 1.0:
            raise ValueError("min_inter_rater_agreement must be in [0, 1]")
        if not self.comparison_arms:
            raise ValueError("comparison_arms must be non-empty")
        if not self.hidden_fields:
            raise ValueError("hidden_fields must be non-empty")


@dataclass(frozen=True)
class LongitudinalHumanAnchorManifest:
    schema_version: str
    persona_plans: tuple[LongitudinalPersonaPlan, ...]
    human_anchor: HumanAnchorProtocol
    description: str

    @property
    def total_sessions(self) -> int:
        return sum(persona.session_count for persona in self.persona_plans)

    def __post_init__(self) -> None:
        if len(self.persona_plans) < 5:
            raise ValueError("at least 5 persona plans are required")


_CORE_LONGITUDINAL_METRICS = (
    "relationship_continuity",
    "callback_correctness",
    "wrong_person_attribution",
    "rupture_repair_lag",
    "commitment_follow_through",
    "open_loop_closure",
    "preference_belief_feeling_intent_contamination",
    "memory_retention_absorption",
    "proactive_followup_precision",
    "consent_violation_rate",
    "policy_drift",
    "regime_churn",
)


def build_longitudinal_human_anchor_manifest() -> LongitudinalHumanAnchorManifest:
    arms = ("volvence", "volvence-cold", "memory-rag", "raw")
    persona_ids = (
        "direct-but-overloaded",
        "slow-trust-repair",
        "boundary-sensitive",
        "preference-conflict",
        "delayed-return",
    )
    persona_plans = tuple(
        LongitudinalPersonaPlan(
            persona_id=persona_id,
            session_count=20,
            min_turns_per_session=8,
            max_turns_per_session=15,
            comparison_arms=("shared-memory-hydration", "default-isolation"),
            tracked_metrics=_CORE_LONGITUDINAL_METRICS,
        )
        for persona_id in persona_ids
    )
    return LongitudinalHumanAnchorManifest(
        schema_version="longitudinal-human-anchor-manifest.v1",
        persona_plans=persona_plans,
        human_anchor=HumanAnchorProtocol(
            blinded_rater_count=3,
            min_inter_rater_agreement=0.6,
            comparison_arms=arms,
            hidden_fields=("profile_label", "system_identity", "expected_label"),
        ),
        description=(
            "Plan §14 longitudinal + human anchor manifest. Produces no claim "
            "until transcripts, ratings, inter-rater agreement, and automatic "
            "judge direction are attached as artifacts."
        ),
    )


__all__ = [
    "HumanAnchorProtocol",
    "LongitudinalHumanAnchorManifest",
    "LongitudinalPersonaPlan",
    "build_longitudinal_human_anchor_manifest",
]
