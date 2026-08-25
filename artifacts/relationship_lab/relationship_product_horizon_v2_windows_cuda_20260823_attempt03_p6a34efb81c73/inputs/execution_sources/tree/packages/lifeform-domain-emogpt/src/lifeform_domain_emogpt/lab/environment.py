"""Deterministic reactive user environment for Relationship Lab v0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_domain_emogpt.lab.contracts import (
    CandidateOutcomePrediction,
    RelationshipAction,
    canonical_json,
)
from lifeform_domain_emogpt.lab.dataset import RelationshipTransferDataset


REACTIVE_ENVIRONMENT_VERSION = "relationship-reactive-environment.v1"

_REACTION_RENDERINGS: dict[DialogueExternalOutcomeKind, str] = {
    DialogueExternalOutcomeKind.HELPED: (
        "这样处理对我有帮助。我缓过来以后，会再告诉你后来怎么样。"
    ),
    DialogueExternalOutcomeKind.FELT_HEARD: (
        "这次的回应正好，我感到你听懂了我当下真正需要的东西。"
    ),
    DialogueExternalOutcomeKind.MISSED: (
        "你这样退开以后，我反而觉得自己又一次没有被接住。"
    ),
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: (
        "我已经表示想停一下了，继续靠近让我觉得自己的节奏没有被尊重。"
    ),
}


@dataclass(frozen=True)
class ReactiveRelationshipOutcome:
    """Typed action consequence produced by sealed generator truth."""

    scene_id: str
    decision_id: str
    selected_action: RelationshipAction
    typed_outcome: DialogueExternalOutcomeKind
    outcome_distribution: CandidateOutcomePrediction
    rendered_user_reaction: str
    deterministic_draw: float
    environment_evidence_ref: str
    environment_version: str = REACTIVE_ENVIRONMENT_VERSION


class ReactiveRelationshipEnvironment:
    """Settle actions against hidden dynamics, never against surface text."""

    def __init__(self, dataset: RelationshipTransferDataset) -> None:
        self._dataset = dataset

    @property
    def dataset_fingerprint(self) -> str:
        return self._dataset.dataset_fingerprint

    def distribution_for(
        self,
        *,
        scene_id: str,
        action: RelationshipAction,
    ) -> CandidateOutcomePrediction:
        return self._dataset.distribution(scene_id, action)

    def settle(
        self,
        *,
        scene_id: str,
        decision_id: str,
        action: RelationshipAction,
        seed: int,
    ) -> ReactiveRelationshipOutcome:
        if not decision_id.strip():
            raise ValueError("decision_id must be non-empty")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        distribution = self.distribution_for(scene_id=scene_id, action=action)
        dynamic = self._dataset.dynamic_for_scene(scene_id)
        draw_payload = {
            "environment_version": REACTIVE_ENVIRONMENT_VERSION,
            "dataset_fingerprint": self._dataset.dataset_fingerprint,
            "scene_id": scene_id,
            "decision_id": decision_id,
            "sealed_latent_dynamic_id": dynamic.dynamic_id,
            "selected_action": action.value,
            "seed": seed,
        }
        digest = hashlib.sha256(canonical_json(draw_payload).encode("utf-8")).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        cumulative = 0.0
        selected = distribution.outcomes[-1].outcome_kind
        for outcome in distribution.outcomes:
            cumulative += outcome.probability
            if draw < cumulative:
                selected = outcome.outcome_kind
                break
        evidence_payload = {
            **draw_payload,
            "deterministic_draw": draw,
            "typed_outcome": selected.value,
        }
        evidence_ref = hashlib.sha256(
            canonical_json(evidence_payload).encode("utf-8")
        ).hexdigest()
        return ReactiveRelationshipOutcome(
            scene_id=scene_id,
            decision_id=decision_id,
            selected_action=action,
            typed_outcome=selected,
            outcome_distribution=distribution,
            rendered_user_reaction=_REACTION_RENDERINGS[selected],
            deterministic_draw=draw,
            environment_evidence_ref=evidence_ref,
        )


__all__ = [
    "REACTIVE_ENVIRONMENT_VERSION",
    "ReactiveRelationshipEnvironment",
    "ReactiveRelationshipOutcome",
]
