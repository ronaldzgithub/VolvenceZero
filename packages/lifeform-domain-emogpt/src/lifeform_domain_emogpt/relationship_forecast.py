"""Bounded preference forecast collaborator for relationship actions.

This runtime reads only typed state owned by ``preference_about_other`` and
semantic similarity supplied by the cognition layer.  It emits a proposal;
the preference owner remains the sole publisher of the frozen forecast.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from volvence_zero.social import (
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
    default_summary_similarity,
)
from volvence_zero.social_cognition import (
    OtherMindRecord,
    PreferenceActionOutcomeEvidence,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)


RELATIONSHIP_PREFERENCE_FORECAST_RUNTIME_ID = (
    "relationship-p2-bounded-forecast.v1"
)
_POSITIVE_OUTCOME_IDS = frozenset({"helped", "felt_heard"})
SimilarityFn = Callable[[str, str], float]


class BoundedRelationshipPreferenceForecastRuntime:
    """Bounded semantic readout over owner-persisted typed experiences."""

    runtime_id = RELATIONSHIP_PREFERENCE_FORECAST_RUNTIME_ID

    def __init__(
        self,
        *,
        similarity: SimilarityFn | None = None,
        prior_count: float = 1.0,
        evidence_weight: float = 4.0,
    ) -> None:
        if not math.isfinite(prior_count) or prior_count <= 0.0:
            raise ValueError("prior_count must be finite and > 0")
        if not math.isfinite(evidence_weight) or evidence_weight <= 0.0:
            raise ValueError("evidence_weight must be finite and > 0")
        self._similarity = similarity or default_summary_similarity
        self._prior_count = prior_count
        self._evidence_weight = evidence_weight

    def propose(
        self,
        *,
        request: PreferenceActionForecastRequest,
        records: tuple[OtherMindRecord, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> PreferenceActionForecastProposal | None:
        if not action_outcomes:
            return None
        records_by_id = {record.record_id: record for record in records}
        unknown_evidence = {
            item.evidence_id for item in action_outcomes
        }.difference(records_by_id)
        if unknown_evidence:
            raise ValueError(
                "relationship forecast evidence is not backed by active owner "
                f"records: {sorted(unknown_evidence)!r}"
            )
        action_surface = set(request.candidate_action_ids)
        outcome_surface = set(request.outcome_ids)
        for item in action_outcomes:
            if item.action_id not in action_surface:
                raise ValueError("persisted evidence action is outside request surface")
            if item.observed_outcome_id not in outcome_surface:
                raise ValueError("persisted evidence outcome is outside request surface")

        weighted_evidence: list[tuple[PreferenceActionOutcomeEvidence, float]] = []
        for item in action_outcomes:
            similarity = self._similarity(
                request.current_observation,
                item.observation_summary,
            )
            if (
                isinstance(similarity, bool)
                or not isinstance(similarity, (int, float))
                or not math.isfinite(similarity)
                or not 0.0 <= similarity <= 1.0
            ):
                raise ValueError(
                    "relationship semantic similarity must be finite and in [0, 1]"
                )
            weighted_evidence.append((item, float(similarity) ** 2))

        candidate_predictions: list[SocialActionCandidatePrediction] = []
        positive_mass_by_action: dict[str, float] = {}
        for action_id in request.candidate_action_ids:
            counts = {
                outcome_id: self._prior_count for outcome_id in request.outcome_ids
            }
            for item, weight in weighted_evidence:
                if item.action_id == action_id:
                    counts[item.observed_outcome_id] += self._evidence_weight * weight
            total = math.fsum(counts.values())
            probabilities = tuple(
                SocialActionOutcomeProbability(
                    outcome_id=outcome_id,
                    probability=counts[outcome_id] / total,
                )
                for outcome_id in request.outcome_ids
            )
            candidate_predictions.append(
                SocialActionCandidatePrediction(
                    action_id=action_id,
                    outcomes=probabilities,
                )
            )
            positive_mass_by_action[action_id] = math.fsum(
                item.probability
                for item in probabilities
                if item.outcome_id in _POSITIVE_OUTCOME_IDS
            )

        ranked_actions = sorted(
            request.candidate_action_ids,
            key=lambda action_id: (
                -positive_mass_by_action[action_id],
                request.candidate_action_ids.index(action_id),
            ),
        )
        recommended_action_id = ranked_actions[0]
        margin = (
            positive_mass_by_action[ranked_actions[0]]
            - positive_mass_by_action[ranked_actions[1]]
        )
        support = max(weight for _, weight in weighted_evidence)
        confidence = max(0.0, min(1.0, 0.5 + 0.5 * support * margin))
        source_record_ids = tuple(
            item.evidence_id
            for item, _ in sorted(
                weighted_evidence,
                key=lambda pair: (-pair[1], pair[0].source_turn),
            )
        )
        return PreferenceActionForecastProposal(
            candidate_predictions=tuple(candidate_predictions),
            recommended_action_id=recommended_action_id,
            confidence=confidence,
            source_record_ids=source_record_ids,
            evidence=(
                f"runtime:{self.runtime_id}",
                f"typed_owner_evidence_count:{len(action_outcomes)}",
                "semantic_similarity_only:no_text_routing",
            ),
        )


__all__ = [
    "BoundedRelationshipPreferenceForecastRuntime",
    "RELATIONSHIP_PREFERENCE_FORECAST_RUNTIME_ID",
]
