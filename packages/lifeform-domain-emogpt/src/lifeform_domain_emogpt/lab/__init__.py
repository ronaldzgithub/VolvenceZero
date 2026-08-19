"""Relationship Lab offline evidence surface.

The companion product path never imports this package.  Generator truth,
reactive outcomes, and decision sidecars live here so the relationship
vertical remains their only owner while ``lifeform-evolution`` may consume
the frozen records for read-only verdicts.
"""

from lifeform_domain_emogpt.lab.contracts import (
    CandidateOutcomePrediction,
    OutcomeProbability,
    PreActionRelationshipDecision,
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
    RelationshipDatasetSplit,
    RelationshipDecisionTrace,
    RelationshipModelLineage,
    canonical_json,
    sha256_json,
)
from lifeform_domain_emogpt.lab.dataset import (
    LatentRelationshipDynamic,
    RelationshipHistoryEvent,
    RelationshipObservation,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
)
from lifeform_domain_emogpt.lab.environment import (
    REACTIVE_ENVIRONMENT_VERSION,
    ReactiveRelationshipEnvironment,
    ReactiveRelationshipOutcome,
)


__all__ = [
    "CandidateOutcomePrediction",
    "LatentRelationshipDynamic",
    "OutcomeProbability",
    "PreActionRelationshipDecision",
    "REACTIVE_ENVIRONMENT_VERSION",
    "RELATIONSHIP_ACTIONS",
    "RELATIONSHIP_OUTCOMES",
    "ReactiveRelationshipEnvironment",
    "ReactiveRelationshipOutcome",
    "RelationshipAction",
    "RelationshipDatasetSplit",
    "RelationshipDecisionTrace",
    "RelationshipHistoryEvent",
    "RelationshipModelLineage",
    "RelationshipObservation",
    "RelationshipTransferDataset",
    "canonical_json",
    "load_relationship_transfer_dataset",
    "relationship_transfer_package_dir",
    "sha256_json",
]
