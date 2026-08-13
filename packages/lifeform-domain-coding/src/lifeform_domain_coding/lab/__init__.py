"""coding-lab: controlled evolving-repo evidence environment (Packet 0).

Evidence-lane subpackage of the coding vertical. Public surface is
re-exported here; nothing in ``lab`` is imported by the vertical's
product path, and ``lab`` never imports brain owners — Packet 0 is
brain-free by design (the SHADOW observer arrives with Packet 1).
"""

from lifeform_domain_coding.lab.calibration import (
    HAND_API,
    HAND_SCRIPTED,
    CalibrationConfig,
    check_environment_determinism,
    run_calibration,
)
from lifeform_domain_coding.lab.episode import EpisodeBudget, EpisodeResult, run_episode
from lifeform_domain_coding.lab.generation import (
    ALL_INVARIANT_IDS,
    GENERATOR_VERSION,
    EnvSpec,
    GeneratedEnvironment,
    LatentInvariant,
    compute_tree_hash,
    generate_environment,
    latent_invariants,
)
from lifeform_domain_coding.lab.hands import (
    APIHandConfig,
    Hand,
    HandAction,
    HandContext,
    HandDecision,
    OpenAICompatHand,
    ScriptedHand,
)
from lifeform_domain_coding.lab.heldout import (
    SealedVariant,
    seal_heldout_variants,
    verify_sealed_variant,
)
from lifeform_domain_coding.lab.junctions import (
    JUNCTION_ACTIONS,
    ContrastiveJunction,
    IncompleteTrajectoryError,
    JunctionRecord,
    build_contrastive_corpus,
    collect_junctions,
    corpus_manifest,
    extract_junctions,
    split_corpus,
)
from lifeform_domain_coding.lab.oracle import OracleOutcome, evaluate_episode
from lifeform_domain_coding.lab.tasks import (
    ChainTask,
    FileEdit,
    FunctionReplace,
    generate_task_chain,
)
from lifeform_domain_coding.lab.trajectory import TrajectoryRecord, TrajectoryWriter, read_trajectory
from lifeform_domain_coding.lab.workspace import ChainWorkspace, apply_edit, directory_bytes

__all__ = [
    "ALL_INVARIANT_IDS",
    "APIHandConfig",
    "CalibrationConfig",
    "ChainTask",
    "ChainWorkspace",
    "ContrastiveJunction",
    "EnvSpec",
    "EpisodeBudget",
    "EpisodeResult",
    "FileEdit",
    "FunctionReplace",
    "GENERATOR_VERSION",
    "GeneratedEnvironment",
    "HAND_API",
    "HAND_SCRIPTED",
    "Hand",
    "HandAction",
    "HandContext",
    "HandDecision",
    "IncompleteTrajectoryError",
    "JUNCTION_ACTIONS",
    "JunctionRecord",
    "LatentInvariant",
    "OpenAICompatHand",
    "OracleOutcome",
    "ScriptedHand",
    "SealedVariant",
    "TrajectoryRecord",
    "TrajectoryWriter",
    "apply_edit",
    "build_contrastive_corpus",
    "check_environment_determinism",
    "collect_junctions",
    "compute_tree_hash",
    "corpus_manifest",
    "directory_bytes",
    "evaluate_episode",
    "extract_junctions",
    "generate_environment",
    "generate_task_chain",
    "latent_invariants",
    "read_trajectory",
    "run_calibration",
    "run_episode",
    "seal_heldout_variants",
    "split_corpus",
    "verify_sealed_variant",
]
