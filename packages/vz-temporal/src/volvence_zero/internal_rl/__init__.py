from volvence_zero.internal_rl.environment import (
    InternalRLDelayedCreditAssignment,
    InternalRLEnvStep,
    InternalRLEnvironment,
    InternalRLProofEpisode,
    InternalRLProofProgress,
    InternalRLProofSubgoal,
    InternalRLRewardSource,
)
from volvence_zero.internal_rl.proof_environment import (
    HierarchicalEpisodeState,
    HierarchicalLocation,
    HierarchicalObservation,
    HierarchicalRouteSpec,
    HierarchicalStepFeedback,
    HierarchicalStepResult,
    HierarchicalTransition,
    MiniHierarchicalCase,
    MiniHierarchicalEnvironment,
)
from volvence_zero.internal_rl.sandbox import (
    CausalPolicyCheckpoint,
    CausalPolicyParameters,
    CausalPolicyState,
    DualTrackRollout,
    DualTrackOptimizationReport,
    InternalRLSandbox,
    OptimizationReport,
    PolicyOptimizationResult,
    ZRollout,
    ZTransition,
    derive_abstract_action_credit,
)


def load_torch_internal_rl():
    """First-class lazy entry to the torch Internal RL backend (#88).

    The facade stays torch-free; this loader is the sanctioned way to
    reach ``TorchCausalZPolicy`` / ``TorchInternalRLConfig`` and the PPO
    trainer without a module-level torch import. Raises ImportError with
    install guidance when torch is unavailable.
    """

    from volvence_zero.internal_rl import torch_internal_rl

    return torch_internal_rl

__all__ = [
    "load_torch_internal_rl",
    "CausalPolicyCheckpoint",
    "CausalPolicyParameters",
    "CausalPolicyState",
    "DualTrackRollout",
    "DualTrackOptimizationReport",
    "HierarchicalEpisodeState",
    "HierarchicalLocation",
    "HierarchicalObservation",
    "HierarchicalRouteSpec",
    "HierarchicalStepFeedback",
    "HierarchicalStepResult",
    "HierarchicalTransition",
    "InternalRLDelayedCreditAssignment",
    "InternalRLEnvStep",
    "InternalRLEnvironment",
    "InternalRLProofEpisode",
    "InternalRLProofProgress",
    "InternalRLProofSubgoal",
    "InternalRLRewardSource",
    "InternalRLSandbox",
    "MiniHierarchicalCase",
    "MiniHierarchicalEnvironment",
    "OptimizationReport",
    "PolicyOptimizationResult",
    "ZRollout",
    "ZTransition",
    "derive_abstract_action_credit",
]
