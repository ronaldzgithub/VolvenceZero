"""P2 formal confirmatory ecology evidence matrix.

P2 is the last stage of ``research/ant/05_ecology_p0_p1_p2_plan.md``: after P0
froze the mechanism definitions and P1 froze the curriculum, thresholds and
capability gates, P2 runs the matched-control matrix that a formal capability
claim needs. **P2 does not tune anything.** Every threshold, layout seed, arm
and endpoint is pre-registered into one digest before the first result is read;
a shard whose digest disagrees is refused rather than merged.

The stage ordering is enforced in code, not by convention: every shard must be
handed a P1 report whose frozen gate set is complete and whose verdict is
``PASS``. Without it :class:`EcologyP2PrerequisiteError` is raised and no P2
budget is spent -- the plan forbids consuming P2 compute (and forbids emitting
a promotion artifact) while P1 is ``BLOCK``.

Execution is sharded by ``(training_seed, arm)``. Each shard journals its own
progress, carries its own manifest, and is only admitted to the aggregate if it
is complete, non-preflight and digest-identical to its siblings.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from volvence_zero.agent import (
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_checkpoint_archive,
)

from volvence_ant.controllers import (
    FixedRuleAnt,
    FixedRuleConfig,
    PPOConfig,
    RandomAnt,
)
from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY,
    EcologyArmMetrics,
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyEvaluationScenario,
    EcologyStage,
    EcologyTrainingEpisodePlan,
    EcologyTrainingEpisodeReport,
    EcologyTrainingTier,
    _ecology_action_chain_guard,
    _evaluate_arm,
    _session_config,
    _train_arm,
    _world,
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_GATE_NAMES,
    ECOLOGY_P1_SCHEMA_VERSION,
    EcologyP1Config,
    _atomic_write,
    _evaluation_specs,
    _fixed_schedule,
    _json_ready,
    _schedule_digest,
    _sha256,
    _stable_json_bytes,
    ecology_p1_formal_budget_failures,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
    run_ecology_checkpoint_post_pickup_uturn_probes,
)
from volvence_ant.runtime import AntLearningCheckpoint, KernelColonyRunner
from volvence_ant.substrate import AntSenseSchema, sense_channels


#: v6 binds the P1 v29 / curriculum v12 post-pickup U-turn hard gate. v5
#: reports could still promote a checkpoint that only passed one-tick home
#: direction checks and never sustained home-distance reduction.
#:
#: v4 binds the P1 v27 / curriculum v10 tangent forced-return schedule. v3
#: shards and journals used a different homing-pressure world and cannot be
#: mixed into the same confirmatory matrix.
#:
#: v2 adds the cross-arm lever-effectiveness gates, the ``random`` floor
#: comparison, the FixedRule learning-advantage gate, the corruption-rollback
#: gate, per-shard source provenance and the declared secondary-endpoint
#: block. A v1 confirmatory report cannot answer "did the ablation lever take
#: effect" or "which commit produced this shard", so it is refused rather than
#: reinterpreted.
#:
#: v3 records the per-seed P1 prerequisites individually (``p1_prerequisites``)
#: plus the frozen held-out layout seeds, and marks each paired effect as
#: complete or incomplete. A v2 report pinned "one identical P1 *file*", which
#: a multi-seed matrix can never satisfy, and it carried no way to recover the
#: held-out device/layout namespace a promotion bundle has to record.
ECOLOGY_P2_SCHEMA_VERSION = "digital-ant-ecology-p2-confirmatory.v6"
#: v2 carries ``source_provenance``, ``archive_corruption_rejected`` and the
#: widened layout rows (secondary endpoints). v1 shards are refused by
#: :func:`shard_report_from_dict`.
#: v3 carries the P1 prerequisite's *configuration* identity, which is what the
#: aggregate pins across seeds; a v2 shard carries only the file digest and
#: therefore cannot be checked against its siblings.
ECOLOGY_P2_SHARD_SCHEMA_VERSION = "digital-ant-ecology-p2-shard.v6"
# v2 binds sense schema and input dim into the shard resume compatibility, so
# an interrupted formal shard cannot rehydrate from a checkpoint trained on a
# different sensory body. v1 journals carry neither key and are refused.
# v3 journals the widened layout rows; a v2 journal would silently drop the
# secondary endpoints of every already-evaluated layout.
ECOLOGY_P2_PROGRESS_SCHEMA_VERSION = "digital-ant-ecology-p2-progress.v6"
ECOLOGY_P2_PREFLIGHT_SCHEMA_VERSION = "digital-ant-ecology-p2-preflight.v1"

#: Held-out namespace. Disjoint from the P1 held-out base (``2_000_003`` plus a
#: bounded capability/layout offset) and independent of the training seed, so
#: every arm and every training seed is scored on the identical frozen layouts
#: and paired comparison is well defined.
ECOLOGY_P2_HELDOUT_SEED_BASE = 5_000_011

#: Probe distribution frozen by P1; P2 reuses it so the action-chain and
#: steering-alignment verdicts are directly comparable across stages.
ECOLOGY_P2_PROBE_SEED_OFFSET = 700_003


@dataclass(frozen=True)
class EcologyP2ArmSpec:
    """One pre-registered arm.

    ``learning`` arms fork from the shared initial checkpoint of their training
    seed and replay the identical frozen schedule; the declared levers are the
    only permitted differences between them. Baseline arms carry no VZ
    checkpoint and exist to calibrate the floor and the non-VZ ceiling.

    ``temporal_policy_kind`` / ``joint_ssl_interval`` / ``joint_rl_interval``
    are the ETA-off construction the spec freezes (``digital-ant-embodiment``
    section 7: *``eta_off`` 使用 frozen learned-lite 且
    ``ssl_interval=rl_interval=0``*). They are **not** booleans on the session
    contract today -- see :data:`ECOLOGY_P2_ARM_LEVER_PARAMETERS` and
    :class:`EcologyP2ArmLeverUnavailableError`: an arm that needs one is
    refused before it spends budget rather than silently degraded onto a
    different lever.
    """

    name: str
    batch: str
    learning: bool
    optimize: bool = True
    local_valence_enabled: bool = True
    segment_credit_enabled: bool = True
    prediction_error_enabled: bool = True
    #: Reflection/memory/regime consolidation writeback. This is **not** the
    #: ETA lever: the kernel gates the reflection engine and the temporal-prior
    #: writeback behind the same boolean, so flipping it removes more than the
    #: named mechanism. It stays here as the declared reflection-consolidation
    #: lever; no pre-registered arm flips it.
    temporal_writeback_enabled: bool = True
    #: ``"full"`` = the learned metacontroller. ``"learned_lite"`` = the frozen
    #: legacy readout projected to the configured ``n_z``.
    temporal_policy_kind: str = "full"
    #: ``None`` = the curriculum owner's frozen joint-loop schedule. ``0``
    #: freezes that phase, which is what makes ETA-off an actual ablation of
    #: the emergent temporal abstraction rather than of reflection memory.
    joint_ssl_interval: int | None = None
    joint_rl_interval: int | None = None
    trains: bool = True
    description: str = ""


class EcologyP2ArmLeverUnavailableError(RuntimeError):
    """A pre-registered arm needs a lever the session contract cannot express.

    Raised *before* any budget is spent. Running the arm anyway would emit a
    row labelled with the pre-registered arm name whose actual construction is
    something else -- exactly the ``以 random 代理消融，或策略参数未按预期改变``
    kill condition the spec freezes in section 6.
    """


#: Arm levers that must be forwarded by the curriculum owner's session builder
#: before the arm that needs them can run. ``_session_config`` /
#: ``_train_arm`` / ``_evaluate_arm`` are owned by ``ecology_curriculum``; P2
#: probes their signatures rather than asserting a version, so the guard
#: releases by itself the moment the hop lands.
ECOLOGY_P2_ARM_LEVER_PARAMETERS: tuple[str, ...] = (
    "temporal_policy_kind",
    "joint_ssl_interval",
    "joint_rl_interval",
)

#: The learned-arm value of each gated lever, read off the dataclass itself so
#: "this arm deviates from the learned construction" cannot drift away from
#: "this field's default". :func:`unreachable_arm_levers` iterates
#: :data:`ECOLOGY_P2_ARM_LEVER_PARAMETERS` against this map rather than
#: re-listing the field names inline.
_ECOLOGY_P2_ARM_LEVER_DEFAULTS: dict[str, Any] = {
    field.name: field.default
    for field in fields(EcologyP2ArmSpec)
    if field.name in ECOLOGY_P2_ARM_LEVER_PARAMETERS
}
if set(_ECOLOGY_P2_ARM_LEVER_DEFAULTS) != set(ECOLOGY_P2_ARM_LEVER_PARAMETERS):
    raise RuntimeError(
        "ECOLOGY_P2_ARM_LEVER_PARAMETERS names a field EcologyP2ArmSpec does "
        f"not declare: {sorted(set(ECOLOGY_P2_ARM_LEVER_PARAMETERS) - set(_ECOLOGY_P2_ARM_LEVER_DEFAULTS))}"
    )


ECOLOGY_P2_ARM_SPECS: tuple[EcologyP2ArmSpec, ...] = (
    EcologyP2ArmSpec(
        name="learned",
        batch="core",
        learning=True,
        description="full stack: PE drive, ETA writeback, segment credit",
    ),
    EcologyP2ArmSpec(
        name="no_optimize",
        batch="core",
        learning=True,
        optimize=False,
        description=(
            "identical SSL/rollout/optimizer evidence, Internal-RL policy "
            "updates not persisted"
        ),
    ),
    EcologyP2ArmSpec(
        name="cold",
        batch="core",
        learning=True,
        trains=False,
        description="shared initial checkpoint, zero training episodes",
    ),
    EcologyP2ArmSpec(
        name="pe_off",
        batch="core",
        learning=True,
        prediction_error_enabled=False,
        description=(
            "prediction error stops driving joint-loop learning and the "
            "temporal switch; remains a readout only"
        ),
    ),
    EcologyP2ArmSpec(
        name="eta_off",
        batch="core",
        learning=True,
        # The spec's frozen ETA-off construction (section 7): a frozen
        # learned-lite temporal policy with ssl_interval = rl_interval = 0, so
        # the emergent temporal abstraction genuinely cannot adapt. The
        # historical `temporal_writeback_enabled=False` arm was wrong from both
        # sides: it removed reflection/memory/regime consolidation as well
        # (superset), while SSL and Internal-RL kept optimising the same
        # `_world_policy` every cycle (incomplete).
        temporal_policy_kind="learned_lite",
        joint_ssl_interval=0,
        joint_rl_interval=0,
        optimize=False,
        description=(
            "frozen learned-lite temporal policy, ssl_interval=rl_interval=0: "
            "the emergent temporal abstraction cannot adapt, while the "
            "substrate, sense schema and runtime replay stay matched"
        ),
    ),
    EcologyP2ArmSpec(
        name="dense_local_shaping_off",
        batch="ablation",
        learning=True,
        local_valence_enabled=False,
        description="sparse milestone reward only; continuous shaping removed",
    ),
    EcologyP2ArmSpec(
        name="segment_credit_off",
        batch="ablation",
        learning=True,
        segment_credit_enabled=False,
        description="one-step runtime replay; long-horizon segment credit off",
    ),
    EcologyP2ArmSpec(
        name="fixed_rule",
        batch="core",
        learning=False,
        description="hand-written FSM forager; safety and difficulty floor",
    ),
    EcologyP2ArmSpec(
        name="e2e_rl",
        batch="core",
        learning=False,
        description="torch PPO straight from frozen senses to motor commands",
    ),
    EcologyP2ArmSpec(
        name="random",
        batch="core",
        learning=False,
        description="encounter floor",
    ),
)

ECOLOGY_P2_ARM_SPEC_BY_NAME: dict[str, EcologyP2ArmSpec] = {
    spec.name: spec for spec in ECOLOGY_P2_ARM_SPECS
}
ECOLOGY_P2_ARM_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in ECOLOGY_P2_ARM_SPECS
)
#: P2-B confirmatory matrix.
ECOLOGY_P2_CORE_ARM_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in ECOLOGY_P2_ARM_SPECS if spec.batch == "core"
)
#: P2-C mechanism ablations, run after the core matrix completes.
ECOLOGY_P2_ABLATION_ARM_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in ECOLOGY_P2_ARM_SPECS if spec.batch == "ablation"
)

#: Checkpoint-bearing arms whose construction persists **no** policy update, so
#: their final policy digest must equal the shared initial fork point.
#:
#: ``optimize=False`` is one such construction (``no_optimize``), and so is
#: ``trains=False`` (``cold``). ``eta_off`` is a third: the spec's frozen ETA-off
#: construction is a learned-lite policy with ``ssl_interval=rl_interval=0``
#: **and** ``optimize=False``, so by this module's own semantics its digest must
#: land on the initial too. Classifying it as an ablation that has to *diverge*
#: (which the previous ``ablation_arms`` rule did) made the two gates
#: contradict each other: ``no_optimize_policy_stable`` demanded the initial
#: digest for ``optimize=False`` arms while ``ablation_policy_divergence``
#: called exactly that digest ``never_trained``.
ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES: tuple[str, ...] = tuple(
    spec.name
    for spec in ECOLOGY_P2_ARM_SPECS
    if spec.learning and (not spec.trains or not spec.optimize)
)
#: Checkpoint-bearing ablation arms that DO persist policy updates, so their
#: lever is only demonstrably live if their digest differs from both the shared
#: initial and the learned arm's.
ECOLOGY_P2_DIVERGENT_POLICY_ARM_NAMES: tuple[str, ...] = tuple(
    spec.name
    for spec in ECOLOGY_P2_ARM_SPECS
    if spec.learning
    and spec.trains
    and spec.optimize
    and spec.name != "learned"
)

#: Every paired comparison entering multiplicity correction, pre-registered in
#: this order.
#:
#: ``learned`` vs ``random`` is a *floor* comparison, not an ablation: the plan
#: forbids substituting a random baseline **for** a real PE/ETA ablation, which
#: is why the four mechanism comparisons above it are all real levers. It does
#: not forbid -- and P1's ``forced_escape_above_random_floor`` already requires
#: -- showing that the learned arm exceeds the encounter floor. Without it the
#: FORMAL stage would be strictly less rigorous than the development stage that
#: unlocks it, and the ``random`` arm would burn full budget while entering no
#: gate. Adding it also widens the Holm family from four to five, which
#: tightens every other comparison.
ECOLOGY_P2_PAIRED_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("learned", "no_optimize"),
    ("learned", "cold"),
    ("learned", "pe_off"),
    ("learned", "eta_off"),
    ("learned", "random"),
)

#: Primary endpoints in pre-registered priority order (plan section 5.5).
#: Secondary readouts never rescue a failed primary endpoint.
ECOLOGY_P2_PRIMARY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("butter_distance_transfer", ("butter_medium", "butter_far")),
    (
        "heat_route_foraging_and_exposure",
        ("forced_escape", "heat_route_foraging", "heat_exposure_bounded"),
    ),
    ("neutral_stick_foraging", ("neutral_stick",)),
    ("composite_foraging", ("composite",)),
    ("learned_paired_effect", ("learned_paired_effect",)),
    ("pe_eta_causal_degradation", ("pe_eta_causal_degradation",)),
    (
        "ablation_levers_took_effect",
        (
            "policy_changed",
            "no_optimize_policy_stable",
            "ablation_policy_divergence",
        ),
    ),
    ("above_random_floor", ("above_random_floor",)),
    ("temporal_abstraction_behavior", ("temporal_non_timeout_closure",)),
    (
        "engineering_integrity",
        (
            "frozen_evaluation",
            "replay_lineage",
            "archive_integrity",
            "archive_corruption_rollback",
            "provenance_clean",
        ),
    ),
)

ECOLOGY_P2_GATE_NAMES: tuple[str, ...] = (
    "preregistration_frozen",
    "formal_configuration",
    "p1_prerequisite_pass",
    "shard_completeness",
    "butter_medium",
    "butter_far",
    "forced_escape",
    "heat_route_foraging",
    "neutral_stick",
    "composite",
    "heat_exposure_bounded",
    "learned_paired_effect",
    "pe_eta_causal_degradation",
    "above_random_floor",
    "policy_changed",
    "no_optimize_policy_stable",
    "ablation_policy_divergence",
    "fixed_rule_safety_floor",
    "fixed_rule_learning_advantage",
    "e2e_rl_baseline_present",
    "p0_action_sensitivity",
    "carrying_home_action_alignment",
    "post_pickup_uturn_progress",
    "food_steering_alignment",
    "temporal_non_timeout_closure",
    "frozen_evaluation",
    "replay_lineage",
    "archive_integrity",
    "archive_corruption_rollback",
    "provenance_clean",
)

#: Plan section 5.5 secondary endpoints. They are diagnostics: they never enter
#: the verdict and can never rescue a failed primary. Each one is either
#: collected into the report or explicitly declared not-collected with the
#: reason -- silence is not an option, and ``test_ecology_p2`` pins this tuple
#: against the plan text.
ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES: tuple[str, ...] = (
    "path_efficiency",
    "first_pickup_tick",
    "escape_latency",
    "per_ant_variance",
    "action_smoothness",
    "action_probe_sensitivity",
)

#: Formal budget frozen by plan section 5.2. A run below it can still execute
#: (that is what the preflight and the deterministic smoke tests use) but the
#: ``formal_configuration`` gate refuses to call it confirmatory evidence.
ECOLOGY_P2_FORMAL_MIN_ANTS = 8
ECOLOGY_P2_FORMAL_LATENT_DIM = 16
ECOLOGY_P2_FORMAL_MIN_TRAINING_ROUNDS = 80
ECOLOGY_P2_FORMAL_MIN_VALIDATION_ROUNDS = 80
ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS = 120
ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS = 5
ECOLOGY_P2_FORMAL_MIN_TRAINING_SEEDS = 3

ECOLOGY_P2_SIGNIFICANCE_ALPHA = 0.05

#: Frozen ``outcome_score`` weighting. It is the estimand every paired
#: comparison is computed on, so it belongs *inside* the pre-registration
#: digest: two shards built with different weights would otherwise merge
#: silently and the aggregate would average two different statistics.
ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("pickups", 0.5),
    ("deliveries", 1.0),
    ("heat_escapes", 0.25),
    ("harmful_heat_ticks", -0.02),
)


class EcologyP2PrerequisiteError(RuntimeError):
    """P1 has not produced a complete ``PASS``; P2 must not consume budget."""


class EcologyP2ProgressPaused(RuntimeError):
    """A bounded resumable shard stopped after a committed work item."""

    def __init__(self, *, completed_work_items: int) -> None:
        self.completed_work_items = completed_work_items
        super().__init__(
            "P2 resumable shard paused after "
            f"{completed_work_items} committed work items"
        )


@dataclass(frozen=True)
class EcologyP2Config:
    n_ants: int = ECOLOGY_P2_FORMAL_MIN_ANTS
    temporal_latent_dim: int = ECOLOGY_P2_FORMAL_LATENT_DIM
    training_rounds: int = ECOLOGY_P2_FORMAL_MIN_TRAINING_ROUNDS
    validation_rounds: int = ECOLOGY_P2_FORMAL_MIN_VALIDATION_ROUNDS
    heldout_rounds: int = ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS
    layouts_per_tier: int = ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS
    training_seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"
    layout_success_ratio: float = 0.6
    body_success_ratio: float = 0.6
    harmful_tick_rate_max: float = 0.05
    bootstrap_samples: int = 4000

    def __post_init__(self) -> None:
        if self.n_ants < 1 or self.temporal_latent_dim < 3:
            raise ValueError("P2 requires ants >=1 and latent dim >=3")
        if min(
            self.training_rounds,
            self.validation_rounds,
            self.heldout_rounds,
            self.layouts_per_tier,
        ) < 1:
            raise ValueError("P2 budgets must be positive")
        if not self.training_seeds:
            raise ValueError("P2 requires at least one training seed")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("P2 training seeds must be unique")
        if tuple(sorted(self.training_seeds)) != self.training_seeds:
            raise ValueError("P2 training seeds must be pre-registered sorted")
        # P1 froze these three; P2 inherits them verbatim. Section 2.1 of the
        # plan forbids relaxing a frozen threshold after observing a result, so
        # the only representable value is the frozen one.
        if self.layout_success_ratio != 0.6:
            raise ValueError("P2 layout success threshold is frozen at 0.6")
        if self.body_success_ratio != 0.6:
            raise ValueError("P2 body success threshold is frozen at 0.6")
        if self.harmful_tick_rate_max != 0.05:
            raise ValueError("P2 harmful tick threshold is frozen at 0.05")
        if self.bootstrap_samples < 1000:
            raise ValueError("P2 bootstrap needs >=1000 resamples")
        if not self.device:
            raise ValueError("P2 must record the formal device")


@dataclass(frozen=True)
class EcologyP2Prerequisite:
    """The frozen P1 artifact that unlocks P2.

    Two identities travel together and they answer different questions.

    ``report_sha256`` identifies the *file*: which P1 artifact was consumed by
    this shard. It is necessarily per-seed, because a P1 report is a per-seed
    artifact and ``load_p1_prerequisite`` binds each shard to the report of its
    own training seed.

    ``configuration_digest`` identifies the *run*: the P1 schema, the frozen
    gate set and every configured budget **except** the training seed. Plan
    section 5.4's "any code or threshold change invalidates the whole batch" is
    a statement about the configuration, not about the file -- a formal matrix
    has one P1 report per training seed by construction, so pinning the file
    across shards would make PASS unreachable. The aggregate therefore records
    each seed's file digest individually and pins this one across all of them.
    """

    report_path: str
    report_sha256: str
    schema_version: str
    verdict: str
    #: The training seed the P1 run was executed on, read from the report's own
    #: config. A shard binds its ``--training-seed`` to this, so "the newest P1
    #: report on disk" can never unlock a different seed's P2 budget.
    training_seed: int = -1
    #: See the class docstring. Empty only on a hand-built fixture; the
    #: aggregate's ``p1_prerequisite_pass`` gate refuses an empty value, so an
    #: unidentified P1 configuration cannot reach a confirmatory verdict.
    configuration_digest: str = ""


@dataclass(frozen=True)
class EcologyP2SourceProvenance:
    """The tree a shard's numbers were produced from.

    Plan section 5.4: *any code or threshold change invalidates the whole
    batch*. The pre-registration digest covers the declared configuration; this
    covers the source the declaration was executed by. Both are needed -- a
    digest match across two different commits is exactly the silent merge the
    rule forbids.
    """

    git_sha: str
    git_branch: str
    worktree_dirty: bool


@dataclass(frozen=True)
class EcologyP2LayoutResult:
    training_seed: int
    arm: str
    capability: str
    seed: int
    tier: str
    successful_bodies: int
    required_bodies: int
    layout_success: bool
    pickups: int
    deliveries: int
    heat_escapes: int
    harmful_heat_ticks: int
    total_ticks: int
    harmful_tick_rate: float
    outcome_score: float
    escape_latencies: tuple[int, ...]
    switch_count: int
    non_timeout_segment_closures: int
    policy_fingerprint_stable: bool
    temporal_learning_fingerprint_stable: bool
    replay_settlement_coverage: float
    replay_lineage_coverage: float
    replay_drop_count: int
    # --- plan section 5.5 secondary endpoints (diagnostics only) -----------
    #: First tick any body picked food up on this layout; ``None`` = never.
    first_pickup_tick: int | None = None
    #: Action smoothness. ``None`` on the non-kernel baseline arms, whose
    #: controllers publish no per-tick turn-delta record.
    mean_absolute_turn_delta: float | None = None
    #: Raw path length. ``None`` on the non-kernel baseline arms.
    applied_distance: float | None = None
    #: Per-body capability outcome, so per-ant variance is recoverable instead
    #: of being hidden behind the colony aggregate.
    per_body_success: tuple[bool, ...] = ()


@dataclass(frozen=True)
class EcologyP2SecondaryEndpoint:
    """One plan section 5.5 diagnostic; never part of the verdict."""

    name: str
    collected: bool
    observed: str
    note: str


@dataclass(frozen=True)
class EcologyP2ProbeSummary:
    action_chain_passed: bool
    action_chain_failures: tuple[str, ...]
    home_probe_count: int
    home_aligned_bodies: int
    uturn_probe_count: int
    uturn_aligned_bodies: int
    food_probe_count: int
    food_aligned_bodies: int
    required_aligned_bodies: int


@dataclass(frozen=True)
class EcologyP2ShardReport:
    schema_version: str
    config: EcologyP2Config
    training_seed: int
    arm: str
    batch: str
    arm_spec: EcologyP2ArmSpec
    preregistration_digest: str
    schedule_sha256: str
    prerequisite: EcologyP2Prerequisite
    device: str
    preflight: bool
    training_complete: bool
    completed_training_episodes: int
    scheduled_training_episodes: int
    policy_digest: str
    #: Digest of the shared initial checkpoint this arm forked from. Compared
    #: across arms so ``policy_changed`` / ``no_optimize_policy_stable`` can be
    #: evaluated without trusting a second, unrelated fork point.
    initial_policy_digest: str
    archive_roundtrip_ok: bool | None
    #: A byte-corrupted archive and a compatibility-mismatched archive must
    #: both be *refused* by the decoder. ``None`` = arm holds no archive.
    archive_corruption_rejected: bool | None
    archive_size_bytes: int | None
    source_provenance: EcologyP2SourceProvenance
    wall_clock_seconds: float
    layout_results: tuple[EcologyP2LayoutResult, ...]
    probe_summary: EcologyP2ProbeSummary | None
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EcologyP2PairedEffect:
    comparison: str
    treatment: str
    control: str
    training_seeds: tuple[int, ...]
    per_seed_mean_difference: tuple[float, ...]
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    holm_adjusted_p_value: float
    significant: bool
    #: ``False`` when at least one pre-registered ``(seed, capability, layout)``
    #: cell was absent from the matrix. An incomplete comparison is **not**
    #: computed over the cells that happen to survive -- that would silently
    #: redefine the estimand -- so its statistics are reported as the null and
    #: ``significant`` is forced ``False``.
    complete: bool = True
    #: The absent cells, named. Plan section 2.3 requires an artifact even on
    #: failure, and a diagnostic BLOCK is only useful if it says what is
    #: missing.
    missing_cells: tuple[str, ...] = ()


@dataclass(frozen=True)
class EcologyP2CapabilityResult:
    capability: str
    arm: str
    required_layouts: int
    per_seed_successful_layouts: tuple[tuple[int, int], ...]
    passed: bool


@dataclass(frozen=True)
class EcologyP2Gate:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True)
class EcologyP2PrimaryEndpoint:
    name: str
    passed: bool
    supporting_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class EcologyP2Report:
    schema_version: str
    config: EcologyP2Config
    preregistration_digest: str
    #: Representative prerequisite (the first shard's). Kept for continuity;
    #: ``p1_prerequisites`` is the authoritative record.
    prerequisite: EcologyP2Prerequisite
    #: One entry per distinct P1 artifact consumed by the matrix, sorted by
    #: training seed. A formal matrix has one per training seed by
    #: construction, so their ``report_sha256`` values differ and their
    #: ``configuration_digest`` values must not.
    p1_prerequisites: tuple[EcologyP2Prerequisite, ...]
    device: str
    training_seeds: tuple[int, ...]
    #: The frozen held-out layout namespace this verdict was scored on (plan
    #: section 2.1). Recorded here so a promotion bundle can carry it forward
    #: instead of losing the 30 held-out seeds at the promotion boundary.
    heldout_layout_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    shard_digests: tuple[tuple[str, str], ...]
    capability_results: tuple[EcologyP2CapabilityResult, ...]
    paired_effects: tuple[EcologyP2PairedEffect, ...]
    primary_endpoints: tuple[EcologyP2PrimaryEndpoint, ...]
    secondary_endpoints: tuple[EcologyP2SecondaryEndpoint, ...]
    source_git_sha: str
    gates: tuple[EcologyP2Gate, ...]
    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EcologyP2PreflightReport:
    schema_version: str
    config: EcologyP2Config
    preregistration_digest: str
    prerequisite: EcologyP2Prerequisite
    training_seed: int
    device: str
    arms: tuple[str, ...]
    shard_wall_clock_seconds: tuple[tuple[str, float], ...]
    shard_archive_size_bytes: tuple[tuple[str, int], ...]
    determinism_repeat_matches: bool
    determinism_detail: str
    passed: bool
    breakpoints: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


def _p1_budget_failures(raw_config: dict[str, Any]) -> tuple[str, ...]:
    """Re-derive the frozen P1 budget verdict from the report's config block.

    The predicate is the P1 owner's own
    (:func:`ecology_p1.ecology_p1_formal_budget_failures`), called on the
    mapping the report already carries: P2 re-checks the verdict without
    becoming a second owner of a single P1 threshold. A config block that
    cannot be read is not a config whose budget can be trusted, so the owner's
    ``ValueError`` is re-raised as the serial-constraint failure it is.
    """

    try:
        return ecology_p1_formal_budget_failures(raw_config)
    except (TypeError, ValueError) as exc:
        raise EcologyP2PrerequisiteError(
            "P1 report config block cannot be re-derived into the frozen "
            f"budget: {exc}. The formal budget must be recomputed from the "
            "configuration, not read off the report's own "
            "'formal_configuration' boolean"
        ) from exc


def _p1_configuration_digest(
    *,
    schema_version: str,
    gate_names: tuple[str, ...],
    raw_config: dict[str, Any],
) -> str:
    """Identity of the P1 *run configuration*, seed excluded.

    Every shard of a formal matrix must have been unlocked by a P1 run of the
    same frozen configuration and code. The training seed is the one field that
    is *required* to differ across the matrix, so it is the one field left out.
    """

    payload = {
        "schema_version": schema_version,
        "gate_names": list(gate_names),
        "config": {
            str(key): value
            for key, value in raw_config.items()
            if key != "seed"
        },
    }
    return _sha256(_stable_json_bytes(payload))


def load_p1_prerequisite(
    report_path: Path,
    *,
    repo_root: Path | None = None,
    expected_training_seed: int | None = None,
) -> EcologyP2Prerequisite:
    """Return the P1 artifact that unlocks P2, or fail loudly.

    The serial constraint is a hard contract, not documentation: a P1 report
    with drifted schema, an incomplete gate set, an unpassed gate or a
    non-``PASS`` verdict cannot unlock P2 budget.

    ``expected_training_seed`` binds the report to the shard that is about to
    consume it. P1 reports are per-seed artifacts, so accepting any of them for
    any shard would let one seed's PASS unlock every other seed's budget.
    """

    resolved = (
        report_path
        if report_path.is_absolute() or repo_root is None
        else repo_root / report_path
    )
    if not resolved.exists():
        raise EcologyP2PrerequisiteError(
            f"P1 report not found: {resolved}"
        )
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise EcologyP2PrerequisiteError("P1 report must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != ECOLOGY_P1_SCHEMA_VERSION:
        raise EcologyP2PrerequisiteError(
            "P1 report schema mismatch: "
            f"expected={ECOLOGY_P1_SCHEMA_VERSION!r}, "
            f"actual={schema_version!r}"
        )
    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, list) or not all(
        isinstance(gate, dict) for gate in raw_gates
    ):
        raise EcologyP2PrerequisiteError(
            "P1 report gates must be structured objects"
        )
    gate_names = tuple(str(gate.get("name", "")) for gate in raw_gates)
    if gate_names != ECOLOGY_P1_GATE_NAMES:
        raise EcologyP2PrerequisiteError(
            f"P1 gate set mismatch: {gate_names!r}"
        )
    failed = tuple(
        str(gate.get("name"))
        for gate in raw_gates
        if gate.get("passed") is not True
    )
    verdict = str(payload.get("verdict", "BLOCK")).upper()
    if failed:
        raise EcologyP2PrerequisiteError(
            "P1 is not PASS; P2 must not start. failed gates: "
            + ", ".join(failed)
        )
    if verdict != "PASS":
        raise EcologyP2PrerequisiteError(
            f"P1 verdict is {verdict}; P2 must not start"
        )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict) or "seed" not in raw_config:
        raise EcologyP2PrerequisiteError(
            "P1 report carries no config.seed, so the shard it unlocks cannot "
            f"be bound to a training seed: {resolved}"
        )
    training_seed = int(raw_config["seed"])
    if expected_training_seed is not None and training_seed != expected_training_seed:
        raise EcologyP2PrerequisiteError(
            "P1 report belongs to a different training seed: "
            f"report_seed={training_seed}, shard_seed={expected_training_seed} "
            f"({resolved}). Each P2 shard must be unlocked by the P1 run of "
            "its own seed."
        )
    # Re-derive the formal budget from the configuration the report carries,
    # instead of trusting its own ``formal_configuration`` gate boolean. The
    # gate above already refuses an unpassed gate; this refuses a report whose
    # boolean says PASS while its recorded configuration is below the frozen
    # P1 budget -- the two can only disagree if the artifact was edited or the
    # threshold moved after the run.
    budget_failures = _p1_budget_failures(raw_config)
    if budget_failures:
        raise EcologyP2PrerequisiteError(
            "P1 report claims formal_configuration passed, but its own config "
            f"block is below the frozen P1 budget: {list(budget_failures)} "
            f"({resolved}). P2 must not start."
        )
    return EcologyP2Prerequisite(
        report_path=str(report_path),
        report_sha256=_sha256(raw),
        schema_version=schema_version,
        verdict=verdict,
        training_seed=training_seed,
        configuration_digest=_p1_configuration_digest(
            schema_version=schema_version,
            gate_names=gate_names,
            raw_config=raw_config,
        ),
    )


def _heldout_seed(*, capability_index: int, layout_index: int) -> int:
    return (
        ECOLOGY_P2_HELDOUT_SEED_BASE
        + capability_index * 10_007
        + layout_index * 103
    )


def heldout_layout_seeds(config: EcologyP2Config) -> tuple[int, ...]:
    """Every frozen held-out layout seed, in pre-registered order."""

    return tuple(
        _heldout_seed(capability_index=capability_index, layout_index=index)
        for capability_index, _ in enumerate(_evaluation_specs())
        for index in range(config.layouts_per_tier)
    )


def p2_training_schedule(
    config: EcologyP2Config,
    *,
    training_seed: int,
) -> tuple[EcologyTrainingEpisodePlan, ...]:
    """Replay the P1-frozen curriculum for one training seed.

    P2 must not re-tune the curriculum, so the schedule is produced by the P1
    builder itself. Any P1 curriculum drift changes the schedule digest and
    therefore the pre-registration digest, which invalidates in-flight shards
    loudly instead of silently mixing two curricula.
    """

    return _fixed_schedule(
        EcologyP1Config(
            n_ants=config.n_ants,
            temporal_latent_dim=config.temporal_latent_dim,
            training_rounds=config.training_rounds,
            evaluation_rounds=config.heldout_rounds,
            layouts_per_tier=config.layouts_per_tier,
            seed=training_seed,
        )
    )


#: Every function whose body defines what a P2 number *means*: the scoring
#: weighting, how a layout row is derived from an arm's metrics, which world a
#: baseline arm is evaluated in, what curriculum a shard trains under, how a
#: baseline layout is run, the two-level bootstrap, the multiplicity correction
#: and the whole gate/verdict body. Plan section 5.4 makes any change to these
#: fatal to an in-flight batch.
#:
#: This tuple is the *coverage claim*, not the hashing unit. Hashing exactly
#: these functions left a transitive hole: ``_run_baseline_layout`` calls
#: ``_scenario_stage`` and every lane calls ``_curriculum_config``, so the two
#: functions that decide which world a baseline arm is scored in and what
#: curriculum every shard trains under could be rewritten under a byte-identical
#: digest, and pre-change and post-change shards would merge silently. The
#: hashing unit is therefore the whole module (see
#: :func:`_frozen_logic_source`) -- coarser than necessary, but complete, and a
#: coarse-but-complete freeze is the only honest reading of section 5.4's "任何
#: 代码或门槛变化都会使整批失效". ``test_ecology_p2`` pins that every name below
#: is genuinely inside the hashed text.
_ECOLOGY_P2_FROZEN_LOGIC = (
    "outcome_score",
    "_layout_result_from_metrics",
    "_scenario_stage",
    "_curriculum_config",
    "_run_baseline_layout",
    "_paired_differences",
    "_hierarchical_paired_bootstrap",
    "_holm_adjusted",
    "_required_bodies",
    "_required_layouts",
    "_formal_configuration_failures",
    "aggregate_ecology_p2_shards",
)


def _frozen_logic_source() -> str:
    """The exact text hashed into the pre-registration digest.

    Deliberately the entire module, transitive helpers included. A per-function
    allow-list cannot be complete without walking the call graph, and a call
    graph walk that stops at the module boundary is itself incomplete; the
    remaining cross-module surface (the curriculum owner and the world) is
    pinned by ``schedule_sha256``, the declared capability list and the
    per-shard ``source_provenance.git_sha`` that ``provenance_clean`` enforces.
    """

    return inspect.getsource(sys.modules[__name__])


@lru_cache(maxsize=1)
def _frozen_logic_digest() -> str:
    """SHA-256 over :func:`_frozen_logic_source`.

    Cached: the source cannot change inside a process, and
    :func:`preregistration_digest` is called on every journal commit.
    """

    return _sha256(_frozen_logic_source().encode("utf-8"))


def preregistration_digest(config: EcologyP2Config) -> str:
    """Freeze config, arms, endpoints, gates, layouts and schedules into one id."""

    payload = {
        "schema_version": ECOLOGY_P2_SCHEMA_VERSION,
        "config": _json_ready(asdict(config)),
        "arms": [asdict(spec) for spec in ECOLOGY_P2_ARM_SPECS],
        "paired_comparisons": [
            list(item) for item in ECOLOGY_P2_PAIRED_COMPARISONS
        ],
        "primary_endpoints": [
            [name, list(gates)]
            for name, gates in ECOLOGY_P2_PRIMARY_ENDPOINTS
        ],
        "secondary_endpoints": list(ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES),
        "gates": list(ECOLOGY_P2_GATE_NAMES),
        "capabilities": [
            [capability, scenario.value, tier.value]
            for capability, scenario, tier in _evaluation_specs()
        ],
        "heldout_layout_seeds": list(heldout_layout_seeds(config)),
        "schedule_sha256": [
            [seed, _schedule_digest(p2_training_schedule(config, training_seed=seed))]
            for seed in config.training_seeds
        ],
        "significance_alpha": ECOLOGY_P2_SIGNIFICANCE_ALPHA,
        "memory_entry_capacity": ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY,
        # The estimand and the code that computes it, not just the declared
        # configuration (plan section 5.4).
        "outcome_score_weights": [
            list(item) for item in ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS
        ],
        "frozen_logic_sha256": _frozen_logic_digest(),
    }
    return _sha256(_stable_json_bytes(payload))


def _formal_configuration_failures(config: EcologyP2Config) -> tuple[str, ...]:
    failures: list[str] = []
    if config.n_ants < ECOLOGY_P2_FORMAL_MIN_ANTS:
        failures.append(f"n_ants={config.n_ants}<{ECOLOGY_P2_FORMAL_MIN_ANTS}")
    if config.temporal_latent_dim != ECOLOGY_P2_FORMAL_LATENT_DIM:
        failures.append(
            f"latent_dim={config.temporal_latent_dim}"
            f"!={ECOLOGY_P2_FORMAL_LATENT_DIM}"
        )
    if config.training_rounds < ECOLOGY_P2_FORMAL_MIN_TRAINING_ROUNDS:
        failures.append(
            f"training_rounds={config.training_rounds}"
            f"<{ECOLOGY_P2_FORMAL_MIN_TRAINING_ROUNDS}"
        )
    if config.validation_rounds < ECOLOGY_P2_FORMAL_MIN_VALIDATION_ROUNDS:
        failures.append(
            f"validation_rounds={config.validation_rounds}"
            f"<{ECOLOGY_P2_FORMAL_MIN_VALIDATION_ROUNDS}"
        )
    if config.heldout_rounds < ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS:
        failures.append(
            f"heldout_rounds={config.heldout_rounds}"
            f"<{ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS}"
        )
    if config.layouts_per_tier < ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS:
        failures.append(
            f"heldout_layouts={config.layouts_per_tier}"
            f"<{ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS}"
        )
    if len(config.training_seeds) < ECOLOGY_P2_FORMAL_MIN_TRAINING_SEEDS:
        failures.append(
            f"training_seeds={len(config.training_seeds)}"
            f"<{ECOLOGY_P2_FORMAL_MIN_TRAINING_SEEDS}"
        )
    return tuple(failures)


# ---------------------------------------------------------------------------
# Shared scoring
# ---------------------------------------------------------------------------


def outcome_score(
    *,
    pickups: int,
    deliveries: int,
    heat_escapes: int,
    harmful_heat_ticks: int,
) -> float:
    """Frozen per-layout outcome score.

    Identical to the curriculum owner's ``_ecology_outcome_score`` weighting;
    ``test_ecology_p2`` pins the two together so the formal statistic cannot
    drift away from the curriculum definition. Neutral wood-stick contacts are
    diagnostics and are never scored. The weights live in
    :data:`ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS` so the pre-registration digest
    binds them.
    """

    counts = {
        "pickups": pickups,
        "deliveries": deliveries,
        "heat_escapes": heat_escapes,
        "harmful_heat_ticks": harmful_heat_ticks,
    }
    return sum(
        counts[name] * weight
        for name, weight in ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS
    )


def unreachable_arm_levers(spec: EcologyP2ArmSpec) -> tuple[str, ...]:
    """Levers ``spec`` needs that the curriculum session builder cannot express.

    The ETA-off construction the spec freezes needs a ``JointLoopSchedule`` and
    a ``LearnedLiteTemporalPolicy``. Both are ``vz-temporal`` internals, which
    ``docs/specs/digital-ant-embodiment.md`` section 2 forbids this package
    from importing (enforced by ``test_import_boundaries.py``), and
    ``AntSessionConfig`` takes them only as opaque ``object`` passthroughs. The
    hop therefore has to be opened by the curriculum owner
    (``ecology_curriculum._session_config`` / ``_train_arm`` / ``_evaluate_arm``)
    or by a vz-runtime facade factory.

    Signature probing rather than a version assertion, so the guard releases
    itself the moment the hop lands.
    """

    required = tuple(
        name
        for name in ECOLOGY_P2_ARM_LEVER_PARAMETERS
        if getattr(spec, name) != _ECOLOGY_P2_ARM_LEVER_DEFAULTS[name]
    )
    if not required:
        return ()
    accepted = frozenset(
        name
        for builder in (_session_config, _train_arm, _evaluate_arm)
        for name in inspect.signature(builder).parameters
    )
    return tuple(name for name in required if name not in accepted)


def _required_bodies(config: EcologyP2Config) -> int:
    return max(1, math.ceil(config.n_ants * config.body_success_ratio))


def _required_layouts(config: EcologyP2Config) -> int:
    return math.ceil(config.layouts_per_tier * config.layout_success_ratio)


def _curriculum_config(
    config: EcologyP2Config,
    *,
    training_seed: int,
) -> EcologyCurriculumConfig:
    return EcologyCurriculumConfig(
        n_ants=config.n_ants,
        temporal_latent_dim=config.temporal_latent_dim,
        stage_rounds=config.training_rounds,
        stage_episodes=1,
        mastery_min_episodes=1,
        validation_rounds=config.validation_rounds,
        validation_seeds=(training_seed + 1_000_037,),
        heldout_rounds=config.heldout_rounds,
        heldout_seeds=heldout_layout_seeds(config),
        seed=training_seed,
        # P0 keeps a per-episode rollback guard to isolate mechanism failures.
        # P1 froze the decision to let a policy traverse temporary sensitivity
        # loss and enforce the identical thresholds once on the final
        # checkpoint; P2 replays that frozen decision unchanged.
        action_probe_guard_enabled=False,
    )


def _layout_result_from_metrics(
    *,
    config: EcologyP2Config,
    training_seed: int,
    arm: str,
    capability: str,
    metrics: EcologyArmMetrics,
) -> EcologyP2LayoutResult:
    required = _required_bodies(config)
    if capability == "forced_escape":
        per_body_success = tuple(
            item.heat_escapes > 0 for item in metrics.body_lineage
        )
    else:
        per_body_success = tuple(
            item.picked_up and item.delivered for item in metrics.body_lineage
        )
    successful = sum(per_body_success)
    total_ticks = sum(item.total_ticks for item in metrics.body_lineage)
    harmful_ticks = sum(
        item.harmful_heat_ticks for item in metrics.body_lineage
    )
    harmful_rate = harmful_ticks / total_ticks if total_ticks else 0.0
    safe = capability not in {"heat_route_foraging", "composite"} or (
        harmful_rate <= config.harmful_tick_rate_max
    )
    return EcologyP2LayoutResult(
        training_seed=training_seed,
        arm=arm,
        capability=capability,
        seed=metrics.seed,
        tier=metrics.body_lineage[0].tier.value,
        successful_bodies=successful,
        required_bodies=required,
        layout_success=successful >= required and safe,
        pickups=metrics.pickups,
        deliveries=metrics.deliveries,
        heat_escapes=metrics.heat_escapes,
        harmful_heat_ticks=metrics.harmful_heat_ticks,
        total_ticks=total_ticks,
        harmful_tick_rate=harmful_rate,
        outcome_score=outcome_score(
            pickups=metrics.pickups,
            deliveries=metrics.deliveries,
            heat_escapes=metrics.heat_escapes,
            harmful_heat_ticks=metrics.harmful_heat_ticks,
        ),
        escape_latencies=tuple(
            latency
            for item in metrics.body_lineage
            for latency in item.escape_latencies
        ),
        switch_count=sum(item.switch_count for item in metrics.body_lineage),
        non_timeout_segment_closures=sum(
            item.non_timeout_segment_closures for item in metrics.body_lineage
        ),
        policy_fingerprint_stable=metrics.policy_fingerprint_stable,
        temporal_learning_fingerprint_stable=(
            metrics.temporal_learning_fingerprint_stable
        ),
        replay_settlement_coverage=metrics.replay_settlement_coverage,
        replay_lineage_coverage=metrics.replay_lineage_coverage,
        replay_drop_count=metrics.replay_drop_count,
        first_pickup_tick=metrics.first_pickup_tick,
        mean_absolute_turn_delta=metrics.mean_absolute_turn_delta,
        applied_distance=metrics.applied_distance,
        per_body_success=per_body_success,
    )


def _layout_result_from_dict(payload: dict[str, Any]) -> EcologyP2LayoutResult:
    return EcologyP2LayoutResult(
        training_seed=int(payload["training_seed"]),
        arm=str(payload["arm"]),
        capability=str(payload["capability"]),
        seed=int(payload["seed"]),
        tier=str(payload["tier"]),
        successful_bodies=int(payload["successful_bodies"]),
        required_bodies=int(payload["required_bodies"]),
        layout_success=bool(payload["layout_success"]),
        pickups=int(payload["pickups"]),
        deliveries=int(payload["deliveries"]),
        heat_escapes=int(payload["heat_escapes"]),
        harmful_heat_ticks=int(payload["harmful_heat_ticks"]),
        total_ticks=int(payload["total_ticks"]),
        harmful_tick_rate=float(payload["harmful_tick_rate"]),
        outcome_score=float(payload["outcome_score"]),
        escape_latencies=tuple(
            int(value) for value in payload["escape_latencies"]
        ),
        switch_count=int(payload["switch_count"]),
        non_timeout_segment_closures=int(
            payload["non_timeout_segment_closures"]
        ),
        policy_fingerprint_stable=bool(payload["policy_fingerprint_stable"]),
        temporal_learning_fingerprint_stable=bool(
            payload["temporal_learning_fingerprint_stable"]
        ),
        replay_settlement_coverage=float(
            payload["replay_settlement_coverage"]
        ),
        replay_lineage_coverage=float(payload["replay_lineage_coverage"]),
        replay_drop_count=int(payload["replay_drop_count"]),
        first_pickup_tick=(
            None
            if payload["first_pickup_tick"] is None
            else int(payload["first_pickup_tick"])
        ),
        mean_absolute_turn_delta=(
            None
            if payload["mean_absolute_turn_delta"] is None
            else float(payload["mean_absolute_turn_delta"])
        ),
        applied_distance=(
            None
            if payload["applied_distance"] is None
            else float(payload["applied_distance"])
        ),
        per_body_success=tuple(
            bool(value) for value in payload["per_body_success"]
        ),
    )


# ---------------------------------------------------------------------------
# Baseline arms
# ---------------------------------------------------------------------------


def _scenario_stage(scenario: EcologyEvaluationScenario) -> EcologyStage:
    return {
        EcologyEvaluationScenario.BUTTER_ONLY: EcologyStage.BUTTER,
        EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK: (
            EcologyStage.WOOD_STICK
        ),
        EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE: (
            EcologyStage.BURNING_MATCH
        ),
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE: (
            EcologyStage.BURNING_MATCH
        ),
        EcologyEvaluationScenario.COMPOSITE: EcologyStage.COMPOSITE,
    }[scenario]


def _train_e2e_policy(
    *,
    config: EcologyP2Config,
    curriculum: EcologyCurriculumConfig,
    training_seed: int,
):
    """Train the end-to-end PPO baseline on the identical frozen schedule."""

    from volvence_ant.controllers import E2ERLAnt

    schedule = p2_training_schedule(config, training_seed=training_seed)
    plans_by_seed = {plan.seed: plan for plan in schedule}
    if len(plans_by_seed) != len(schedule):
        raise RuntimeError(
            "P2 schedule layout seeds must be unique to key the E2E factory"
        )

    def world_factory(layout_seed: int):
        plan = plans_by_seed[layout_seed]
        return _world(
            config=curriculum,
            stage=plan.stage,
            seed=plan.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=plan.tier,
            forced_escape=plan.forced_escape,
            forced_return=plan.forced_return,
            forced_approach=plan.forced_approach,
        )

    policy = E2ERLAnt(
        seed=training_seed,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    policy.train(
        world_factory=world_factory,
        seed=training_seed,
        config=PPOConfig(
            episodes=len(schedule),
            ticks_per_episode=config.training_rounds,
        ),
        episode_keys=tuple(plan.seed for plan in schedule),
        n_bodies=config.n_ants,
        # Matched with the kernel arms: forced-start episodes resynchronise
        # path integration, otherwise the baseline's egocentric home channels
        # are corrupted on exactly the bootstrap blocks the learned arm uses.
        synchronize_navigators=True,
    )
    return policy


def _run_baseline_layout(
    *,
    config: EcologyP2Config,
    curriculum: EcologyCurriculumConfig,
    training_seed: int,
    arm: str,
    capability: str,
    scenario: EcologyEvaluationScenario,
    tier: EcologyTrainingTier,
    seed: int,
    e2e_policy: Any = None,
) -> EcologyP2LayoutResult:
    forced_escape = scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
    world = _world(
        config=curriculum,
        stage=_scenario_stage(scenario),
        seed=seed,
        data_split=EcologyDataSplit.HELDOUT,
        tier=tier,
        forced_escape=forced_escape,
    )
    if arm in {"fixed_rule", "random"}:
        ants = tuple(
            FixedRuleAnt(
                world,
                config=FixedRuleConfig(seed=seed * 100 + body_id),
                body_id=body_id,
            )
            if arm == "fixed_rule"
            else RandomAnt(world, seed=seed * 100 + body_id, body_id=body_id)
            for body_id in range(config.n_ants)
        )

        def step(body_id: int) -> None:
            ants[body_id].step()

    elif arm == "e2e_rl":
        if e2e_policy is None:
            raise ValueError("e2e_rl layout requires a trained policy")
        for body_id in range(config.n_ants):
            e2e_policy.attach(
                world,
                body_id=body_id,
                seed=seed * 100 + body_id,
                synchronize_navigator=False,
            )

        def step(body_id: int) -> None:
            e2e_policy.step(world, body_id=body_id)

    else:
        raise ValueError(f"unsupported P2 baseline arm: {arm}")
    picked = [False] * config.n_ants
    delivered = [False] * config.n_ants
    escaped = [False] * config.n_ants
    escape_latencies: list[int] = []
    harmful_ticks = 0
    heat_escape_events = 0
    first_pickup_tick: int | None = None
    for round_index in range(config.heldout_rounds):
        for body_id in range(config.n_ants):
            step(body_id)
            transition = world.last_transition(body_id)
            if transition.picked_up and first_pickup_tick is None:
                first_pickup_tick = round_index + 1
            picked[body_id] = picked[body_id] or transition.picked_up
            delivered[body_id] = delivered[body_id] or transition.delivered
            heat_escape_events += int(transition.escaped_harmful_heat)
            first_escape = (
                transition.escaped_harmful_heat and not escaped[body_id]
            )
            escaped[body_id] = escaped[body_id] or first_escape
            if first_escape:
                escape_latencies.append(round_index + 1)
            harmful_ticks += int(transition.heat_harmful_after)
    required = _required_bodies(config)
    per_body_success = (
        tuple(escaped)
        if capability == "forced_escape"
        else tuple(
            did_pickup and did_deliver
            for did_pickup, did_deliver in zip(picked, delivered, strict=True)
        )
    )
    successful = sum(per_body_success)
    total_ticks = config.heldout_rounds * config.n_ants
    harmful_rate = harmful_ticks / total_ticks if total_ticks else 0.0
    safe = capability not in {"heat_route_foraging", "composite"} or (
        harmful_rate <= config.harmful_tick_rate_max
    )
    return EcologyP2LayoutResult(
        training_seed=training_seed,
        arm=arm,
        capability=capability,
        seed=seed,
        tier=tier.value,
        successful_bodies=successful,
        required_bodies=required,
        layout_success=successful >= required and safe,
        pickups=world.food_pickups,
        deliveries=world.food_delivered,
        heat_escapes=heat_escape_events,
        harmful_heat_ticks=harmful_ticks,
        total_ticks=total_ticks,
        harmful_tick_rate=harmful_rate,
        outcome_score=outcome_score(
            pickups=world.food_pickups,
            deliveries=world.food_delivered,
            heat_escapes=heat_escape_events,
            harmful_heat_ticks=harmful_ticks,
        ),
        escape_latencies=tuple(escape_latencies),
        switch_count=0,
        non_timeout_segment_closures=0,
        # Baselines hold no VZ learning owner, so the frozen-owner and replay
        # gates are vacuously satisfied for them and are only evaluated over
        # checkpoint-bearing arms.
        policy_fingerprint_stable=True,
        temporal_learning_fingerprint_stable=True,
        replay_settlement_coverage=1.0,
        replay_lineage_coverage=1.0,
        replay_drop_count=0,
        first_pickup_tick=first_pickup_tick,
        # FixedRule / random / E2E-RL controllers publish no per-tick
        # turn-delta or path-integral record, so the two motion diagnostics are
        # explicitly absent rather than zero-filled.
        mean_absolute_turn_delta=None,
        applied_distance=None,
        per_body_success=per_body_success,
    )


# ---------------------------------------------------------------------------
# Shard journal
# ---------------------------------------------------------------------------


def _shard_dir(progress_dir: Path, *, training_seed: int, arm: str) -> Path:
    return progress_dir / f"seed{training_seed}" / arm


def _shard_state_path(shard_dir: Path) -> Path:
    return shard_dir / "state.json"


def _progress_compatibility(
    config: EcologyP2Config,
) -> tuple[tuple[str, str], ...]:
    # Same four-way binding as the promotion bundle
    # (ecology_checkpoint_compatibility): sense schema, input dim, latent dim
    # and ant count. A shard journal is the archive an interrupted formal run
    # rehydrates from, so it may not bind less than the promotion archive.
    return (
        ("artifact_kind", ECOLOGY_P2_PROGRESS_SCHEMA_VERSION),
        ("sense_schema", AntSenseSchema.ECOLOGY_V2.value),
        (
            "input_dim",
            str(len(sense_channels(AntSenseSchema.ECOLOGY_V2))),
        ),
        ("n_ants", str(config.n_ants)),
        ("latent_dim", str(config.temporal_latent_dim)),
        ("runtime_replay", "excluded"),
        (
            "memory_entry_capacity",
            str(ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY),
        ),
    )


def _load_shard_state(
    *,
    shard_dir: Path,
    config: EcologyP2Config,
    training_seed: int,
    arm: str,
    digest: str,
    schedule_sha256: str,
    prerequisite: EcologyP2Prerequisite,
) -> dict[str, Any] | None:
    path = _shard_state_path(shard_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": ECOLOGY_P2_PROGRESS_SCHEMA_VERSION,
        "arm": arm,
        "training_seed": training_seed,
        "config": _json_ready(asdict(config)),
        "preregistration_digest": digest,
        "schedule_sha256": schedule_sha256,
        "p1_report_sha256": prerequisite.report_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"P2 shard progress mismatch (seed={training_seed}, arm={arm}): "
                f"field={field}, expected={value!r}, "
                f"actual={payload.get(field)!r}"
            )
    completed = payload.get("completed_training_episodes")
    if not isinstance(completed, int) or completed < 0:
        raise ValueError(
            f"P2 shard {arm} has an invalid completed episode count"
        )
    return payload


def _read_shard_archive(
    *,
    shard_dir: Path,
    state: dict[str, Any],
    config: EcologyP2Config,
) -> tuple[bytes, ...]:
    filename = state.get("checkpoint_archive")
    if not isinstance(filename, str) or not filename:
        raise ValueError("P2 shard progress is missing checkpoint_archive")
    archive_path = (shard_dir / filename).resolve()
    archive_path.relative_to(shard_dir.resolve())
    payload = archive_path.read_bytes()
    if _sha256(payload) != state.get("checkpoint_sha256"):
        raise ValueError(
            f"P2 shard archive digest mismatch: {archive_path.name}"
        )
    expected = _progress_compatibility(config)
    try:
        collection = decode_agent_learning_checkpoint_archive(
            payload,
            expected_compatibility=expected,
        )
    except ValueError as exc:
        raise ValueError(
            "P2 shard checkpoint is not compatible with this run "
            f"({archive_path.name}): {exc}. Shard journals written before "
            f"{ECOLOGY_P2_PROGRESS_SCHEMA_VERSION} do not bind sense schema "
            "and input dim and cannot be resumed; rerun the shard into a new "
            "--progress-dir."
        ) from exc
    if dict(collection.metadata.compatibility) != dict(expected):
        raise ValueError(
            "P2 shard checkpoint compatibility mismatch: "
            f"expected={dict(expected)!r}, "
            f"actual={dict(collection.metadata.compatibility)!r}"
        )
    return collection.checkpoint_archives


def _save_shard_state(
    *,
    shard_dir: Path,
    config: EcologyP2Config,
    training_seed: int,
    arm: str,
    digest: str,
    schedule_sha256: str,
    prerequisite: EcologyP2Prerequisite,
    completed_training_episodes: int,
    training_complete: bool,
    runner: KernelColonyRunner | None,
    policy_digest: str,
    layout_results: list[EcologyP2LayoutResult],
    last_episode_report: EcologyTrainingEpisodeReport | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    archive_name = previous.get("checkpoint_archive") if previous else None
    checkpoint_sha256 = previous.get("checkpoint_sha256") if previous else None
    archive_size = previous.get("checkpoint_size_bytes") if previous else None
    if runner is not None:
        raw_archives = runner.export_learning_checkpoint_archives(
            checkpoint_prefix=(
                f"ecology:p2:{training_seed}:{arm}:"
                f"episode-{completed_training_episodes:04d}"
            )
        )
        archive = encode_agent_learning_checkpoint_archive(
            raw_archives,
            compatibility=_progress_compatibility(config),
        )
        # Two-slot journal: fsync/rename the slot the previous state does not
        # point at, then atomically advance the pointer. One rollback
        # checkpoint is retained while the shard stays bounded to two colony
        # archives on disk.
        archive_name = f"{arm}.slot-{completed_training_episodes % 2}.vzac"
        _atomic_write(shard_dir / archive_name, archive)
        checkpoint_sha256 = _sha256(archive)
        archive_size = len(archive)
    state = {
        "schema_version": ECOLOGY_P2_PROGRESS_SCHEMA_VERSION,
        "arm": arm,
        "training_seed": training_seed,
        "config": _json_ready(asdict(config)),
        "preregistration_digest": digest,
        "schedule_sha256": schedule_sha256,
        "p1_report_sha256": prerequisite.report_sha256,
        "completed_training_episodes": completed_training_episodes,
        "training_complete": training_complete,
        "checkpoint_archive": archive_name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": archive_size,
        "policy_digest": policy_digest,
        "evaluation_policy_digest": policy_digest,
        "layout_results": [asdict(item) for item in layout_results],
        "last_episode": (
            {
                "stage": last_episode_report.plan.stage.value,
                "tier": last_episode_report.plan.tier.value,
                "episode_index": last_episode_report.plan.episode_index,
                "pickups": last_episode_report.pickups,
                "deliveries": last_episode_report.deliveries,
                "heat_entries": last_episode_report.heat_entries,
                "heat_escapes": last_episode_report.heat_escapes,
                "memory_entries_evicted": (
                    last_episode_report.memory_entries_evicted
                ),
            }
            if last_episode_report is not None
            else None
        ),
    }
    _atomic_write(_shard_state_path(shard_dir), _stable_json_bytes(state))
    return state


def _hydrate_shard_checkpoints(
    *,
    curriculum: EcologyCurriculumConfig,
    archives: tuple[bytes, ...],
    training_seed: int,
    arm: str,
) -> tuple[AntLearningCheckpoint, ...]:
    runner = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=training_seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=training_seed,
            session_id=f"ecology:p2:{training_seed}:{arm}:hydrate",
            optimize=False,
        ),
    )
    runner.restore_learning_checkpoint_archives(archives)
    return runner.export_learning_checkpoints(
        checkpoint_prefix=f"ecology:p2:{training_seed}:{arm}:hydrated",
        include_runtime_replay=False,
    )


def _shared_initial_checkpoints(
    *,
    curriculum: EcologyCurriculumConfig,
    training_seed: int,
) -> tuple[AntLearningCheckpoint, ...]:
    """Deterministic per-training-seed fork point shared by every arm."""

    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=training_seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=training_seed,
            session_id=f"ecology:p2:{training_seed}:shared-initial",
            optimize=True,
        ),
    )
    return bootstrap.export_learning_checkpoints(
        checkpoint_prefix=f"ecology:p2:{training_seed}:shared-initial",
        include_runtime_replay=False,
    )


async def _probe_summary(
    *,
    config: EcologyP2Config,
    curriculum: EcologyCurriculumConfig,
    training_seed: int,
    initial: tuple[AntLearningCheckpoint, ...],
    trained: tuple[AntLearningCheckpoint, ...],
) -> EcologyP2ProbeSummary:
    action_chain_passed, action_chain_failures = (
        await _ecology_action_chain_guard(
            config=curriculum,
            baseline=initial,
            candidate=trained,
        )
    )
    probes = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=training_seed + ECOLOGY_P2_PROBE_SEED_OFFSET,
        checkpoints=trained,
        turn_delta_threshold=curriculum.action_probe_turn_delta_threshold,
    )
    home = tuple(
        probe
        for body in probes
        for probe in body.probes
        if probe.kind is EcologyProbeKind.HOME
    )
    food = tuple(
        probe
        for body in probes
        for probe in body.probes
        if probe.kind is EcologyProbeKind.FOOD
    )
    uturn = await run_ecology_checkpoint_post_pickup_uturn_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=training_seed + ECOLOGY_P2_PROBE_SEED_OFFSET,
        checkpoints=trained,
    )

    def aligned(probe) -> bool:
        return bool(
            probe.input_reachable
            and probe.action_sensitive
            and probe.target_aligned
        )

    return EcologyP2ProbeSummary(
        action_chain_passed=action_chain_passed,
        action_chain_failures=action_chain_failures,
        home_probe_count=len(home),
        home_aligned_bodies=sum(aligned(probe) for probe in home),
        uturn_probe_count=len(uturn),
        uturn_aligned_bodies=sum(probe.passed for probe in uturn),
        food_probe_count=len(food),
        food_aligned_bodies=sum(aligned(probe) for probe in food),
        required_aligned_bodies=_required_bodies(config),
    )


def _corruption_rollback_ok(
    *,
    archive: bytes,
    expected: tuple[tuple[str, str], ...],
) -> bool:
    """A corrupted or mis-declared archive must be refused, not decoded.

    Plan section 5.7 requires ``corruption rollback`` and ``schema
    compatibility`` alongside the archive round trip. Both are probed on the
    shard's own bytes: a single flipped byte, and an otherwise-valid archive
    presented under a different compatibility declaration.
    """

    if not archive:
        raise ValueError("corruption probe needs a non-empty archive")
    corrupted = bytearray(archive)
    corrupted[len(corrupted) // 2] ^= 0xFF
    try:
        decode_agent_learning_checkpoint_archive(
            bytes(corrupted),
            expected_compatibility=expected,
        )
    except (ValueError, KeyError):
        byte_corruption_refused = True
    else:
        byte_corruption_refused = False
    mismatched = (*expected, ("corruption_probe", "unexpected"))
    try:
        decode_agent_learning_checkpoint_archive(
            archive,
            expected_compatibility=mismatched,
        )
    except (ValueError, KeyError):
        compatibility_mismatch_refused = True
    else:
        compatibility_mismatch_refused = False
    return byte_corruption_refused and compatibility_mismatch_refused


async def run_ecology_p2_shard(
    config: EcologyP2Config,
    *,
    training_seed: int,
    arm: str,
    source_provenance: EcologyP2SourceProvenance,
    p1_report_path: Path | None = None,
    prerequisite: EcologyP2Prerequisite | None = None,
    repo_root: Path | None = None,
    progress_dir: Path | None = None,
    max_new_work_items: int | None = None,
    preflight: bool = False,
) -> EcologyP2ShardReport:
    """Run one ``(training_seed, arm)`` shard of the P2 matrix.

    The shard refuses to start unless P1 is a complete ``PASS`` and the
    declared formal device matches the process environment. Training episodes
    and held-out layouts are journalled individually, so a killed run resumes
    at the last committed work item without re-running completed work and
    without silently merging results produced under a different checkpoint.
    """

    if training_seed not in config.training_seeds:
        raise ValueError(
            f"training seed {training_seed} is not pre-registered"
        )
    spec = ECOLOGY_P2_ARM_SPEC_BY_NAME.get(arm)
    if spec is None:
        raise ValueError(f"unknown P2 arm: {arm}")
    unreachable = unreachable_arm_levers(spec)
    if unreachable:
        raise EcologyP2ArmLeverUnavailableError(
            f"P2 arm {arm!r} declares levers the curriculum session contract "
            f"cannot express: {list(unreachable)}. The frozen construction is "
            f"temporal_policy_kind={spec.temporal_policy_kind!r}, "
            f"ssl_interval={spec.joint_ssl_interval!r}, "
            f"rl_interval={spec.joint_rl_interval!r}; it needs "
            "ecology_curriculum._session_config / _train_arm / _evaluate_arm "
            "to accept and forward them (AntSessionConfig already carries "
            "opaque joint_schedule / temporal_policy passthroughs, but "
            "JointLoopSchedule and LearnedLiteTemporalPolicy are vz-temporal "
            "internals this package may not import). Running the arm without "
            "them would label a different construction with this arm's name."
        )
    if not source_provenance.git_sha:
        raise ValueError(
            "P2 shard requires a source git SHA; plan section 5.4 makes any "
            "code change fatal to the batch, which cannot be checked from an "
            "unidentified tree"
        )
    if max_new_work_items is not None and max_new_work_items < 1:
        raise ValueError("max_new_work_items must be positive")
    if max_new_work_items is not None and progress_dir is None:
        raise ValueError(
            "max_new_work_items requires a resumable progress_dir"
        )
    if prerequisite is None:
        if p1_report_path is None:
            raise EcologyP2PrerequisiteError(
                "P2 requires the frozen P1 report; the plan forbids spending "
                "P2 budget before P1 reaches PASS"
            )
        prerequisite = load_p1_prerequisite(
            p1_report_path,
            repo_root=repo_root,
            expected_training_seed=training_seed,
        )
    if prerequisite.verdict != "PASS":
        raise EcologyP2PrerequisiteError(
            f"P1 verdict is {prerequisite.verdict}; P2 must not start"
        )
    requested_device = os.environ.get("VZ_TENSOR_DEVICE", "cpu").strip().lower()
    if requested_device != config.device.strip().lower():
        raise ValueError(
            "P2 device mismatch: config declares "
            f"{config.device!r} but VZ_TENSOR_DEVICE resolves to "
            f"{requested_device!r}"
        )

    started = time.monotonic()
    digest = preregistration_digest(config)
    curriculum = _curriculum_config(config, training_seed=training_seed)
    schedule = p2_training_schedule(config, training_seed=training_seed)
    schedule_sha256 = _schedule_digest(schedule)
    scheduled_episodes = len(schedule) if spec.trains else 0

    shard_dir: Path | None = None
    state: dict[str, Any] | None = None
    if progress_dir is not None:
        shard_dir = _shard_dir(
            progress_dir.resolve(),
            training_seed=training_seed,
            arm=arm,
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        state = _load_shard_state(
            shard_dir=shard_dir,
            config=config,
            training_seed=training_seed,
            arm=arm,
            digest=digest,
            schedule_sha256=schedule_sha256,
            prerequisite=prerequisite,
        )

    completed_work_items = 0
    layout_results: list[EcologyP2LayoutResult] = []
    policy_digest = ""
    initial_policy_digest = ""
    archive_roundtrip_ok: bool | None = None
    archive_corruption_rejected: bool | None = None
    archive_size: int | None = None
    probe_summary: EcologyP2ProbeSummary | None = None
    completed_episodes = 0
    checkpoints: tuple[AntLearningCheckpoint, ...] = ()
    e2e_policy: Any = None

    if spec.learning:
        initial = _shared_initial_checkpoints(
            curriculum=curriculum,
            training_seed=training_seed,
        )
        initial_policy_digest = _sha256(
            _stable_json_bytes([item.policy_fingerprint for item in initial])
        )
        checkpoints = initial
        if state is not None:
            completed_episodes = int(state["completed_training_episodes"])
            if completed_episodes > scheduled_episodes:
                raise ValueError(
                    f"P2 shard {arm} progress exceeds its schedule length"
                )
            if bool(state.get("training_complete")) != (
                completed_episodes == scheduled_episodes
            ):
                raise ValueError(
                    f"P2 shard {arm} completion flag disagrees with its count"
                )
            if state.get("checkpoint_archive"):
                checkpoints = _hydrate_shard_checkpoints(
                    curriculum=curriculum,
                    archives=_read_shard_archive(
                        shard_dir=shard_dir,
                        state=state,
                        config=config,
                    ),
                    training_seed=training_seed,
                    arm=arm,
                )

        if shard_dir is not None and state is None:
            state = _save_shard_state(
                shard_dir=shard_dir,
                config=config,
                training_seed=training_seed,
                arm=arm,
                digest=digest,
                schedule_sha256=schedule_sha256,
                prerequisite=prerequisite,
                completed_training_episodes=0,
                training_complete=(scheduled_episodes == 0),
                runner=None,
                policy_digest="",
                layout_results=[],
            )

        if spec.trains and completed_episodes < scheduled_episodes:

            def save_episode(
                schedule_index: int,
                runner: KernelColonyRunner,
                _checkpoints: tuple[AntLearningCheckpoint, ...],
                report: EcologyTrainingEpisodeReport,
            ) -> None:
                nonlocal completed_work_items, state
                if shard_dir is None:
                    return
                completed_count = schedule_index + 1
                state = _save_shard_state(
                    shard_dir=shard_dir,
                    config=config,
                    training_seed=training_seed,
                    arm=arm,
                    digest=digest,
                    schedule_sha256=schedule_sha256,
                    prerequisite=prerequisite,
                    completed_training_episodes=completed_count,
                    training_complete=(
                        completed_count == scheduled_episodes
                    ),
                    runner=runner,
                    policy_digest="",
                    layout_results=[],
                    last_episode_report=report,
                    previous=state,
                )
                completed_work_items += 1
                if (
                    max_new_work_items is not None
                    and completed_work_items >= max_new_work_items
                ):
                    raise EcologyP2ProgressPaused(
                        completed_work_items=completed_work_items
                    )

            checkpoints, _, _, _, _ = await _train_arm(
                config=curriculum,
                initial=checkpoints,
                arm=f"p2:{training_seed}:{arm}",
                optimize=spec.optimize,
                local_valence_enabled=spec.local_valence_enabled,
                segment_credit_enabled=spec.segment_credit_enabled,
                prediction_error_enabled=spec.prediction_error_enabled,
                temporal_writeback_enabled=spec.temporal_writeback_enabled,
                schedule=schedule,
                schedule_start_index=completed_episodes,
                episode_callback=(
                    save_episode if shard_dir is not None else None
                ),
            )
            completed_episodes = scheduled_episodes

        policy_digest = _sha256(
            _stable_json_bytes(
                [item.policy_fingerprint for item in checkpoints]
            )
        )
        if state is not None and state.get("checkpoint_sha256"):
            archive_size = state.get("checkpoint_size_bytes")
            # Archive integrity is a promotion gate, so the round trip is
            # exercised on the artifact that will actually be shipped.
            archive_roundtrip_ok = tuple(
                item.policy_fingerprint
                for item in _hydrate_shard_checkpoints(
                    curriculum=curriculum,
                    archives=_read_shard_archive(
                        shard_dir=shard_dir,
                        state=state,
                        config=config,
                    ),
                    training_seed=training_seed,
                    arm=arm,
                )
            ) == tuple(item.policy_fingerprint for item in checkpoints)
            archive_corruption_rejected = _corruption_rollback_ok(
                archive=(
                    shard_dir / str(state["checkpoint_archive"])
                ).read_bytes(),
                expected=_progress_compatibility(config),
            )
        if arm == "learned":
            probe_summary = await _probe_summary(
                config=config,
                curriculum=curriculum,
                training_seed=training_seed,
                initial=initial,
                trained=checkpoints,
            )
    else:
        completed_episodes = scheduled_episodes
        if arm == "e2e_rl":
            e2e_policy = _train_e2e_policy(
                config=config,
                curriculum=curriculum,
                training_seed=training_seed,
            )
            policy_digest = _sha256(
                _stable_json_bytes(list(e2e_policy.parameter_digest()))
            )
        if shard_dir is not None and state is None:
            state = _save_shard_state(
                shard_dir=shard_dir,
                config=config,
                training_seed=training_seed,
                arm=arm,
                digest=digest,
                schedule_sha256=schedule_sha256,
                prerequisite=prerequisite,
                completed_training_episodes=scheduled_episodes,
                training_complete=True,
                runner=None,
                policy_digest=policy_digest,
                layout_results=[],
            )

    if state is not None:
        # Cached held-out rows are only reusable under the policy that
        # produced them. A resumed shard that retrained to a different
        # checkpoint must re-evaluate rather than merge stale rows.
        if str(state.get("evaluation_policy_digest", "")) == policy_digest:
            layout_results = [
                _layout_result_from_dict(item)
                for item in state.get("layout_results", [])
            ]
        else:
            layout_results = []

    seen = {(item.capability, item.seed) for item in layout_results}
    if len(seen) != len(layout_results):
        raise ValueError("P2 shard evaluation progress contains duplicates")

    for capability_index, (capability, scenario, tier) in enumerate(
        _evaluation_specs()
    ):
        for layout_index in range(config.layouts_per_tier):
            evaluation_seed = _heldout_seed(
                capability_index=capability_index,
                layout_index=layout_index,
            )
            if (capability, evaluation_seed) in seen:
                continue
            if spec.learning:
                metrics = await _evaluate_arm(
                    config=curriculum,
                    checkpoints=checkpoints,
                    arm=f"p2:{training_seed}:{arm}",
                    data_split=EcologyDataSplit.HELDOUT,
                    scenario=scenario,
                    seed=evaluation_seed,
                    tier=tier,
                    prediction_error_enabled=spec.prediction_error_enabled,
                    temporal_writeback_enabled=(
                        spec.temporal_writeback_enabled
                    ),
                )
                result = _layout_result_from_metrics(
                    config=config,
                    training_seed=training_seed,
                    arm=arm,
                    capability=capability,
                    metrics=metrics,
                )
            else:
                result = _run_baseline_layout(
                    config=config,
                    curriculum=curriculum,
                    training_seed=training_seed,
                    arm=arm,
                    capability=capability,
                    scenario=scenario,
                    tier=tier,
                    seed=evaluation_seed,
                    e2e_policy=e2e_policy,
                )
            layout_results.append(result)
            seen.add((capability, evaluation_seed))
            if shard_dir is not None:
                state = _save_shard_state(
                    shard_dir=shard_dir,
                    config=config,
                    training_seed=training_seed,
                    arm=arm,
                    digest=digest,
                    schedule_sha256=schedule_sha256,
                    prerequisite=prerequisite,
                    completed_training_episodes=completed_episodes,
                    training_complete=True,
                    runner=None,
                    policy_digest=policy_digest,
                    layout_results=layout_results,
                    previous=state,
                )
                completed_work_items += 1
                if (
                    max_new_work_items is not None
                    and completed_work_items >= max_new_work_items
                ):
                    raise EcologyP2ProgressPaused(
                        completed_work_items=completed_work_items
                    )

    capability_order = {
        capability: index
        for index, (capability, _, _) in enumerate(_evaluation_specs())
    }
    ordered = tuple(
        sorted(
            layout_results,
            key=lambda item: (capability_order[item.capability], item.seed),
        )
    )
    expected_rows = len(_evaluation_specs()) * config.layouts_per_tier
    if len(ordered) != expected_rows:
        raise RuntimeError(
            f"P2 shard produced {len(ordered)} held-out rows, "
            f"expected {expected_rows}"
        )
    return EcologyP2ShardReport(
        schema_version=ECOLOGY_P2_SHARD_SCHEMA_VERSION,
        config=config,
        training_seed=training_seed,
        arm=arm,
        batch=spec.batch,
        arm_spec=spec,
        preregistration_digest=digest,
        schedule_sha256=schedule_sha256,
        prerequisite=prerequisite,
        device=config.device,
        preflight=preflight,
        training_complete=(completed_episodes == scheduled_episodes),
        completed_training_episodes=completed_episodes,
        scheduled_training_episodes=scheduled_episodes,
        policy_digest=policy_digest,
        initial_policy_digest=initial_policy_digest,
        archive_roundtrip_ok=archive_roundtrip_ok,
        archive_corruption_rejected=archive_corruption_rejected,
        archive_size_bytes=archive_size,
        source_provenance=source_provenance,
        wall_clock_seconds=time.monotonic() - started,
        layout_results=ordered,
        probe_summary=probe_summary,
        description=(
            f"P2 shard seed={training_seed} arm={arm} "
            f"({'preflight' if preflight else 'confirmatory'})"
        ),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _hierarchical_paired_bootstrap(
    per_seed_differences: tuple[tuple[float, ...], ...],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float, float, float]:
    """Two-level paired bootstrap: resample training seeds, then layouts.

    The training seed is the independent replicate level, so it is resampled
    first; layouts inside a seed are paired but not independent replicates, so
    they are resampled within the chosen seed. Returns
    ``(mean, ci_low, ci_high, two_sided_p_value)``.
    """

    if not per_seed_differences:
        raise ValueError("paired bootstrap requires at least one training seed")
    if any(not item for item in per_seed_differences):
        raise ValueError("every training seed needs at least one paired layout")
    seed_means = np.asarray(
        [float(np.mean(item)) for item in per_seed_differences],
        dtype=float,
    )
    observed = float(seed_means.mean())
    rng = np.random.default_rng(seed)
    n_seeds = len(per_seed_differences)
    vectors = [np.asarray(item, dtype=float) for item in per_seed_differences]
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        chosen = rng.integers(0, n_seeds, size=n_seeds)
        total = 0.0
        for seed_index in chosen:
            values = vectors[seed_index]
            picks = rng.integers(0, len(values), size=len(values))
            total += float(values[picks].mean())
        draws[index] = total / n_seeds
    ci_low = float(np.quantile(draws, 0.025))
    ci_high = float(np.quantile(draws, 0.975))
    # Two-sided bootstrap p-value around the null of no difference, with the
    # standard +1 correction so a p-value is never exactly zero.
    centred = draws - observed
    p_low = float(np.mean(centred <= -abs(observed)))
    p_high = float(np.mean(centred >= abs(observed)))
    p_value = min(1.0, 2.0 * min(p_low, p_high) + 1.0 / (samples + 1))
    return observed, ci_low, ci_high, p_value


def _holm_adjusted(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Holm step-down adjustment; monotone and never below the raw value."""

    count = len(p_values)
    if count == 0:
        return ()
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return tuple(adjusted)


def _paired_differences(
    *,
    rows: dict[tuple[str, int, str, int], EcologyP2LayoutResult],
    training_seeds: tuple[int, ...],
    treatment: str,
    control: str,
    capabilities: tuple[str, ...],
    layout_seeds: dict[str, tuple[int, ...]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]:
    """Per-seed paired outcome differences, plus the cells that were absent.

    A missing cell is **reported, not raised**. Raising aborted the whole
    aggregation, so a matrix with one unrunnable arm produced no artifact at
    all -- which plan section 2.3 forbids ("任何异常退出都保留 partial log,
    并标记 incomplete"): a formal batch that cannot complete must still leave a
    diagnostic ``BLOCK`` on disk naming what is missing. The caller refuses to
    compute a statistic for an incomplete comparison, so nothing is imputed in
    a favourable direction (section 5.6).
    """

    per_seed: list[tuple[float, ...]] = []
    missing: list[str] = []
    for training_seed in training_seeds:
        differences: list[float] = []
        for capability in capabilities:
            for layout_seed in layout_seeds[capability]:
                treatment_row = rows.get(
                    (treatment, training_seed, capability, layout_seed)
                )
                control_row = rows.get(
                    (control, training_seed, capability, layout_seed)
                )
                if treatment_row is None or control_row is None:
                    absent = tuple(
                        arm
                        for arm, row in ((treatment, treatment_row), (control, control_row))
                        if row is None
                    )
                    missing.append(
                        f"{'+'.join(absent)}@{training_seed}:"
                        f"{capability}:{layout_seed}"
                    )
                    continue
                differences.append(
                    treatment_row.outcome_score - control_row.outcome_score
                )
        per_seed.append(tuple(differences))
    return tuple(per_seed), tuple(missing)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _gate(
    name: str, *, passed: bool, observed: str, threshold: str
) -> EcologyP2Gate:
    return EcologyP2Gate(
        name=name, passed=passed, observed=observed, threshold=threshold
    )


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _secondary_endpoints(
    *,
    rows: dict[tuple[str, int, str, int], EcologyP2LayoutResult],
    probe_summaries: tuple[EcologyP2ProbeSummary, ...],
) -> tuple[EcologyP2SecondaryEndpoint, ...]:
    """Plan section 5.5 diagnostics, in the pre-registered order.

    These never enter the verdict -- the all-or-nothing primary gate already
    forbids a secondary readout from rescuing a failed primary. What they may
    not do is go missing: an endpoint the plan names is either reported with a
    number or declared not-collected with the reason.
    """

    learned = tuple(row for key, row in rows.items() if key[0] == "learned")
    first_pickups = [
        float(row.first_pickup_tick)
        for row in learned
        if row.first_pickup_tick is not None
    ]
    latencies = [
        float(value) for row in learned for value in row.escape_latencies
    ]
    smoothness = [
        row.mean_absolute_turn_delta
        for row in learned
        if row.mean_absolute_turn_delta is not None
    ]
    distances = [
        row.applied_distance
        for row in learned
        if row.applied_distance is not None
    ]
    per_ant_variances = [
        float(np.var([float(value) for value in row.per_body_success]))
        for row in learned
        if row.per_body_success
    ]
    probe_ratio = [
        item.food_aligned_bodies / item.food_probe_count
        for item in probe_summaries
        if item.food_probe_count
    ]

    def _fmt(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:.4f}"

    return (
        EcologyP2SecondaryEndpoint(
            name="path_efficiency",
            collected=False,
            observed=(
                "raw learned path length mean="
                f"{_fmt(_mean_or_none(distances))} over {len(distances)} rows"
            ),
            note=(
                "NOT COLLECTED as a ratio: the world owner publishes no "
                "per-layout optimal path length, and inventing a denominator "
                "here would make the embodiment a second owner of layout "
                "geometry. The raw applied distance is reported instead; the "
                "ratio needs an oracle path published by the AntWorld owner."
            ),
        ),
        EcologyP2SecondaryEndpoint(
            name="first_pickup_tick",
            collected=True,
            observed=(
                f"learned mean={_fmt(_mean_or_none(first_pickups))} over "
                f"{len(first_pickups)}/{len(learned)} layouts with a pickup"
            ),
            note="per-layout first pickup tick; None when no body ever picked up",
        ),
        EcologyP2SecondaryEndpoint(
            name="escape_latency",
            collected=True,
            observed=(
                f"learned mean={_fmt(_mean_or_none(latencies))} over "
                f"{len(latencies)} escape events"
            ),
            note="per-body harmful-heat escape latencies",
        ),
        EcologyP2SecondaryEndpoint(
            name="per_ant_variance",
            collected=True,
            observed=(
                "learned mean per-layout variance of per-body success="
                f"{_fmt(_mean_or_none(per_ant_variances))} over "
                f"{len(per_ant_variances)} layouts"
            ),
            note=(
                "per-body outcome is journalled per layout, so pseudo-"
                "replication across ants stays visible"
            ),
        ),
        EcologyP2SecondaryEndpoint(
            name="action_smoothness",
            collected=bool(smoothness),
            observed=(
                f"learned mean |turn delta|={_fmt(_mean_or_none(smoothness))} "
                f"over {len(smoothness)} kernel layouts"
            ),
            note=(
                "collected on checkpoint-bearing arms only; FixedRule / "
                "random / E2E-RL publish no per-tick turn-delta record and "
                "report None rather than a zero fill"
            ),
        ),
        EcologyP2SecondaryEndpoint(
            name="action_probe_sensitivity",
            collected=bool(probe_ratio),
            observed=(
                "learned food-probe aligned fraction="
                f"{_fmt(_mean_or_none(probe_ratio))} over "
                f"{len(probe_ratio)} training seeds"
            ),
            note=(
                "the same probe distribution the food_steering_alignment hard "
                "gate reads; reported here as a graded diagnostic"
            ),
        ),
    )


def aggregate_ecology_p2_shards(
    shards: tuple[EcologyP2ShardReport, ...],
    *,
    worktree_clean: bool,
    config: EcologyP2Config | None = None,
) -> EcologyP2Report:
    """Fold complete shards into the confirmatory report and promotion verdict.

    Only complete, non-preflight shards whose pre-registration digest matches
    are admitted. Any missing ``(training_seed, arm)`` cell is a hard failure:
    the plan forbids dropping a failed shard and reporting the remainder.
    """

    if not shards:
        raise ValueError("P2 aggregation requires at least one shard")
    resolved = config or shards[0].config
    digest = preregistration_digest(resolved)
    expected_arms = ECOLOGY_P2_ARM_NAMES
    gates: list[EcologyP2Gate] = []

    digest_failures = tuple(
        f"{item.arm}@{item.training_seed}"
        for item in shards
        if item.preregistration_digest != digest
        or item.schema_version != ECOLOGY_P2_SHARD_SCHEMA_VERSION
        or item.config != resolved
    )
    gates.append(
        _gate(
            "preregistration_frozen",
            passed=not digest_failures,
            observed=(
                f"digest={digest[:16]}, mismatched={list(digest_failures)}"
            ),
            threshold=(
                "every shard carries the identical pre-registration digest, "
                "schema and config"
            ),
        )
    )

    formal_failures = _formal_configuration_failures(resolved)
    gates.append(
        _gate(
            "formal_configuration",
            passed=not formal_failures,
            observed=(
                "formal"
                if not formal_failures
                else f"below formal budget: {list(formal_failures)}"
            ),
            threshold=(
                f">={ECOLOGY_P2_FORMAL_MIN_ANTS} ants, latent "
                f"{ECOLOGY_P2_FORMAL_LATENT_DIM}, "
                f">={ECOLOGY_P2_FORMAL_MIN_TRAINING_ROUNDS} training / "
                f">={ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS} held-out rounds, "
                f">={ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS} held-out layouts, "
                f">={ECOLOGY_P2_FORMAL_MIN_TRAINING_SEEDS} training seeds"
            ),
        )
    )

    # --- P1 identity: same frozen configuration, one report per seed --------
    # A P1 report is a per-seed artifact and every shard is bound to the report
    # of its OWN training seed, so their file digests necessarily differ. What
    # the plan actually freezes (section 5.4) is the configuration and the code,
    # so that is what is pinned across seeds; the file digests are recorded
    # individually instead of being collapsed into one.
    prerequisite = shards[0].prerequisite
    p1_by_seed: dict[int, set[str]] = {}
    for item in shards:
        p1_by_seed.setdefault(item.training_seed, set()).add(
            item.prerequisite.report_sha256
        )
    p1_prerequisites = tuple(
        sorted(
            {
                (
                    item.prerequisite.training_seed,
                    item.prerequisite.report_sha256,
                ): item.prerequisite
                for item in shards
            }.values(),
            key=lambda item: (item.training_seed, item.report_sha256),
        )
    )
    configuration_digests = {
        item.prerequisite.configuration_digest for item in shards
    }
    unidentified_p1 = tuple(
        f"{item.arm}@{item.training_seed}"
        for item in shards
        if not item.prerequisite.configuration_digest
    )
    unbound_p1 = tuple(
        f"{item.arm}@{item.training_seed}"
        f":p1_seed={item.prerequisite.training_seed}"
        for item in shards
        if item.prerequisite.training_seed != item.training_seed
    )
    split_p1 = tuple(
        f"seed{training_seed}:{len(digests)}_reports"
        for training_seed, digests in sorted(p1_by_seed.items())
        if len(digests) != 1
    )
    p1_ok = (
        len(configuration_digests) == 1
        and not unidentified_p1
        and not unbound_p1
        and not split_p1
        and all(
            item.prerequisite.verdict == "PASS"
            and item.prerequisite.schema_version == ECOLOGY_P1_SCHEMA_VERSION
            for item in shards
        )
    )
    gates.append(
        _gate(
            "p1_prerequisite_pass",
            passed=p1_ok,
            observed=(
                f"p1_configuration_digests={len(configuration_digests)}, "
                f"p1_reports={len(p1_prerequisites)}, "
                f"verdict={prerequisite.verdict}, "
                f"unidentified={list(unidentified_p1)}, "
                f"seed_mismatch={list(unbound_p1)}, "
                f"split_seeds={list(split_p1)}"
            ),
            threshold=(
                "every shard is unlocked by a PASS P1 run of the identical "
                "frozen configuration (one configuration digest across the "
                "matrix), by the report of its own training seed, and each "
                "training seed references exactly one P1 report"
            ),
        )
    )

    present = {(item.training_seed, item.arm) for item in shards}

    def _missing_cell(training_seed: int, arm: str) -> str:
        # An arm that cannot be executed as pre-registered is a different
        # diagnosis from an arm someone forgot to run, and the artifact has to
        # say which one it is.
        levers = unreachable_arm_levers(ECOLOGY_P2_ARM_SPEC_BY_NAME[arm])
        if levers:
            return (
                f"{arm}@{training_seed}:unexecutable"
                f"(levers={'+'.join(levers)})"
            )
        return f"{arm}@{training_seed}"

    missing = tuple(
        _missing_cell(training_seed, arm)
        for training_seed in resolved.training_seeds
        for arm in expected_arms
        if (training_seed, arm) not in present
    )
    unexecutable_arms = tuple(
        arm
        for arm in expected_arms
        if unreachable_arm_levers(ECOLOGY_P2_ARM_SPEC_BY_NAME[arm])
        and any(
            (training_seed, arm) not in present
            for training_seed in resolved.training_seeds
        )
    )
    incomplete = tuple(
        f"{item.arm}@{item.training_seed}"
        for item in shards
        if not item.training_complete or item.preflight
    )
    duplicates = len(present) != len(shards)
    gates.append(
        _gate(
            "shard_completeness",
            passed=not missing and not incomplete and not duplicates,
            observed=(
                f"shards={len(shards)}, missing={list(missing)}, "
                f"unexecutable_arms={list(unexecutable_arms)}, "
                f"incomplete_or_preflight={list(incomplete)}, "
                f"duplicates={duplicates}"
            ),
            threshold=(
                "every (training_seed, arm) cell present exactly once, "
                "complete and non-preflight. An 'unexecutable' cell is an arm "
                "whose pre-registered levers the curriculum session builder "
                "cannot express yet (see unreachable_arm_levers); it blocks "
                "the matrix rather than being dropped or substituted"
            ),
        )
    )

    rows: dict[tuple[str, int, str, int], EcologyP2LayoutResult] = {}
    for shard in shards:
        for row in shard.layout_results:
            rows[(shard.arm, shard.training_seed, row.capability, row.seed)] = row
    layout_seeds = {
        capability: tuple(
            _heldout_seed(
                capability_index=capability_index, layout_index=layout_index
            )
            for layout_index in range(resolved.layouts_per_tier)
        )
        for capability_index, (capability, _, _) in enumerate(
            _evaluation_specs()
        )
    }
    capabilities = tuple(capability for capability, _, _ in _evaluation_specs())
    required_layouts = _required_layouts(resolved)

    capability_results: list[EcologyP2CapabilityResult] = []
    for capability in capabilities:
        for arm in expected_arms:
            per_seed = tuple(
                (
                    training_seed,
                    sum(
                        1
                        for layout_seed in layout_seeds[capability]
                        if (
                            (arm, training_seed, capability, layout_seed)
                            in rows
                        )
                        and rows[
                            (arm, training_seed, capability, layout_seed)
                        ].layout_success
                    ),
                )
                for training_seed in resolved.training_seeds
            )
            capability_results.append(
                EcologyP2CapabilityResult(
                    capability=capability,
                    arm=arm,
                    required_layouts=required_layouts,
                    per_seed_successful_layouts=per_seed,
                    passed=all(
                        count >= required_layouts for _, count in per_seed
                    ),
                )
            )
    capability_by_key = {
        (item.capability, item.arm): item for item in capability_results
    }
    for capability in capabilities:
        learned = capability_by_key[(capability, "learned")]
        gates.append(
            _gate(
                capability,
                passed=learned.passed,
                observed=(
                    "successful_layouts_per_seed="
                    f"{list(learned.per_seed_successful_layouts)}"
                    f"/{resolved.layouts_per_tier}"
                ),
                threshold=(
                    f">={required_layouts} layouts in every training seed; "
                    f"each layout requires >={_required_bodies(resolved)} bodies"
                ),
            )
        )

    exposure_capabilities = ("heat_route_foraging", "composite")
    exposure_rows = tuple(
        row
        for key, row in rows.items()
        if key[0] == "learned" and row.capability in exposure_capabilities
    )
    no_optimize_composite = tuple(
        row
        for key, row in rows.items()
        if key[0] == "no_optimize" and row.capability == "composite"
    )
    learned_composite = tuple(
        row for row in exposure_rows if row.capability == "composite"
    )
    learned_composite_rate = (
        sum(row.harmful_tick_rate for row in learned_composite)
        / len(learned_composite)
        if learned_composite
        else 1.0
    )
    no_optimize_composite_rate = (
        sum(row.harmful_tick_rate for row in no_optimize_composite)
        / len(no_optimize_composite)
        if no_optimize_composite
        else 0.0
    )
    exposure_ok = bool(exposure_rows) and all(
        row.harmful_tick_rate <= resolved.harmful_tick_rate_max
        for row in exposure_rows
    ) and learned_composite_rate <= no_optimize_composite_rate + 1e-12
    gates.append(
        _gate(
            "heat_exposure_bounded",
            passed=exposure_ok,
            observed=(
                f"max_rate={max((row.harmful_tick_rate for row in exposure_rows), default=1.0):.4f}, "
                f"learned_composite={learned_composite_rate:.4f}, "
                f"no_optimize_composite={no_optimize_composite_rate:.4f}"
            ),
            threshold=(
                f"every heat-route/composite layout <= "
                f"{resolved.harmful_tick_rate_max}, and learned composite "
                "exposure <= matched no-optimize"
            ),
        )
    )

    paired_pairs = tuple(
        _paired_differences(
            rows=rows,
            training_seeds=resolved.training_seeds,
            treatment=treatment,
            control=control,
            capabilities=capabilities,
            layout_seeds=layout_seeds,
        )
        for treatment, control in ECOLOGY_P2_PAIRED_COMPARISONS
    )
    paired_inputs = tuple(item[0] for item in paired_pairs)
    paired_missing = tuple(item[1] for item in paired_pairs)
    # A comparison is computable only if EVERY pre-registered cell is present
    # in EVERY training seed. Bootstrapping the surviving subset would change
    # the estimand between shards and would be an implicit favourable
    # imputation of the absent ones (plan section 5.6 forbids both).
    paired_complete = tuple(
        not paired_missing[index]
        and bool(differences)
        and all(bool(item) for item in differences)
        for index, differences in enumerate(paired_inputs)
    )
    raw_statistics = tuple(
        _hierarchical_paired_bootstrap(
            differences,
            seed=index * 9_973 + 11,
            samples=resolved.bootstrap_samples,
        )
        if paired_complete[index]
        # The null, explicitly: no effect, no interval, p=1. It never reaches
        # ``significant`` because ``complete`` is False as well.
        else (0.0, 0.0, 0.0, 1.0)
        for index, differences in enumerate(paired_inputs)
    )
    adjusted = _holm_adjusted(
        tuple(item[3] for item in raw_statistics)
    )
    paired_effects = tuple(
        EcologyP2PairedEffect(
            comparison=f"{treatment}_vs_{control}",
            treatment=treatment,
            control=control,
            training_seeds=resolved.training_seeds,
            per_seed_mean_difference=tuple(
                float(np.mean(values)) if values else 0.0
                for values in paired_inputs[index]
            ),
            mean_difference=raw_statistics[index][0],
            ci_low=raw_statistics[index][1],
            ci_high=raw_statistics[index][2],
            p_value=raw_statistics[index][3],
            holm_adjusted_p_value=adjusted[index],
            significant=(
                paired_complete[index]
                and raw_statistics[index][0] > 0.0
                and raw_statistics[index][1] > 0.0
                and adjusted[index] < ECOLOGY_P2_SIGNIFICANCE_ALPHA
            ),
            complete=paired_complete[index],
            missing_cells=paired_missing[index],
        )
        for index, (treatment, control) in enumerate(
            ECOLOGY_P2_PAIRED_COMPARISONS
        )
    )
    effect_by_comparison = {
        item.comparison: item for item in paired_effects
    }

    def _effect_observed(item: EcologyP2PairedEffect) -> str:
        if not item.complete:
            return (
                f"{item.comparison}: NOT COMPUTED, "
                f"{len(item.missing_cells)} matched cells absent "
                f"({list(item.missing_cells[:6])}"
                f"{', ...' if len(item.missing_cells) > 6 else ''})"
            )
        return (
            f"{item.comparison}: mean={item.mean_difference:.4f} "
            f"ci=[{item.ci_low:.4f},{item.ci_high:.4f}] "
            f"holm_p={item.holm_adjusted_p_value:.4f}"
        )

    learning_effects = (
        effect_by_comparison["learned_vs_no_optimize"],
        effect_by_comparison["learned_vs_cold"],
    )
    gates.append(
        _gate(
            "learned_paired_effect",
            passed=all(item.significant for item in learning_effects),
            observed=", ".join(
                _effect_observed(item) for item in learning_effects
            ),
            threshold=(
                "learned beats no-optimize and cold with paired CI lower "
                f"bound >0 and Holm-adjusted p<{ECOLOGY_P2_SIGNIFICANCE_ALPHA}"
            ),
        )
    )
    causal_effects = (
        effect_by_comparison["learned_vs_pe_off"],
        effect_by_comparison["learned_vs_eta_off"],
    )
    gates.append(
        _gate(
            "pe_eta_causal_degradation",
            passed=all(item.significant for item in causal_effects),
            observed=", ".join(
                _effect_observed(item) for item in causal_effects
            ),
            threshold=(
                "removing the PE drive or freezing the temporal abstraction "
                "degrades the learned gain with paired CI lower bound >0 "
                "after Holm correction"
            ),
        )
    )

    floor_effect = effect_by_comparison["learned_vs_random"]
    gates.append(
        _gate(
            "above_random_floor",
            passed=floor_effect.significant,
            observed=_effect_observed(floor_effect),
            threshold=(
                "learned exceeds the random encounter floor with paired CI "
                f"lower bound >0 and Holm-adjusted p<"
                f"{ECOLOGY_P2_SIGNIFICANCE_ALPHA}; P1 already requires this of "
                "forced escape, so the formal stage may not be weaker"
            ),
        )
    )

    # --- the ablation levers must actually have moved the policy -----------
    # spec section 6 kill condition: "以 random 代理消融，或策略参数未按预期改变".
    # A per-arm policy digest that is only ever compared with itself cannot
    # detect a lever that did nothing, so the digests are compared ACROSS arms
    # at the same training seed.
    digest_by_arm_seed = {
        (item.arm, item.training_seed): item for item in shards
    }

    def _shard_of(arm: str, training_seed: int) -> EcologyP2ShardReport | None:
        return digest_by_arm_seed.get((arm, training_seed))

    learned_changed: list[str] = []
    for training_seed in resolved.training_seeds:
        shard = _shard_of("learned", training_seed)
        if (
            shard is None
            or not shard.policy_digest
            or not shard.initial_policy_digest
            or shard.policy_digest == shard.initial_policy_digest
        ):
            learned_changed.append(f"learned@{training_seed}")
    gates.append(
        _gate(
            "policy_changed",
            passed=not learned_changed,
            observed=f"unchanged_or_missing={learned_changed}",
            threshold=(
                "the learned arm's policy digest differs from the shared "
                "initial checkpoint it forked from, in every training seed"
            ),
        )
    )

    frozen_drift: list[str] = []
    for training_seed in resolved.training_seeds:
        for arm in ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES:
            shard = _shard_of(arm, training_seed)
            if (
                shard is None
                or not shard.policy_digest
                or shard.policy_digest != shard.initial_policy_digest
            ):
                frozen_drift.append(f"{arm}@{training_seed}")
    gates.append(
        _gate(
            "no_optimize_policy_stable",
            passed=not frozen_drift,
            observed=f"drifted_or_missing={frozen_drift}",
            threshold=(
                "every arm that persists no policy update lands back on the "
                "shared initial checkpoint digest in every training seed "
                f"({list(ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES)}): "
                "``optimize=False`` and ``trains=False`` both mean 'the "
                "parameters must not have moved'"
            ),
        )
    )

    inert_levers: list[str] = []
    for training_seed in resolved.training_seeds:
        learned_shard = _shard_of("learned", training_seed)
        for arm in ECOLOGY_P2_DIVERGENT_POLICY_ARM_NAMES:
            shard = _shard_of(arm, training_seed)
            if shard is None or not shard.policy_digest:
                inert_levers.append(f"{arm}@{training_seed}:missing")
                continue
            if learned_shard is None:
                inert_levers.append(f"{arm}@{training_seed}:no_learned_peer")
                continue
            if shard.policy_digest == learned_shard.policy_digest:
                inert_levers.append(f"{arm}@{training_seed}:same_as_learned")
            elif shard.policy_digest == shard.initial_policy_digest:
                inert_levers.append(f"{arm}@{training_seed}:never_trained")
    gates.append(
        _gate(
            "ablation_policy_divergence",
            passed=not inert_levers,
            observed=f"inert={inert_levers}",
            threshold=(
                "every ablation arm that DOES persist policy updates reaches a "
                "digest that is neither the learned arm's nor the untouched "
                "shared initial one: the declared lever demonstrably changed "
                f"the parameters ({list(ECOLOGY_P2_DIVERGENT_POLICY_ARM_NAMES)})"
            ),
        )
    )

    fixed_rule_rows = tuple(
        row
        for key, row in rows.items()
        if key[0] == "fixed_rule" and row.capability in exposure_capabilities
    )
    fixed_rule_rate = (
        sum(row.harmful_tick_rate for row in fixed_rule_rows)
        / len(fixed_rule_rows)
        if fixed_rule_rows
        else 0.0
    )
    learned_exposure_rate = (
        sum(row.harmful_tick_rate for row in exposure_rows)
        / len(exposure_rows)
        if exposure_rows
        else 1.0
    )
    gates.append(
        _gate(
            "fixed_rule_safety_floor",
            # ``min``, not ``max``. Plan section 5.7 requires learned to be
            # *not weaker* than FixedRule's safety threshold; taking the max
            # let learned be strictly less safe than the hand-written FSM and
            # still pass, as long as it stayed under the absolute cap.
            passed=bool(fixed_rule_rows)
            and learned_exposure_rate
            <= min(fixed_rule_rate, resolved.harmful_tick_rate_max) + 1e-12,
            observed=(
                f"learned={learned_exposure_rate:.4f}, "
                f"fixed_rule={fixed_rule_rate:.4f}"
            ),
            threshold=(
                "learned heat exposure <= min(FixedRule exposure, "
                f"{resolved.harmful_tick_rate_max}): not weaker than the "
                "hand-written baseline, and inside the absolute cap"
            ),
        )
    )

    def _mean_outcome(arm: str, training_seed: int) -> float | None:
        values = [
            row.outcome_score
            for key, row in rows.items()
            if key[0] == arm and key[1] == training_seed
        ]
        return float(np.mean(values)) if values else None

    advantage_failures: list[str] = []
    for training_seed in resolved.training_seeds:
        learned_mean = _mean_outcome("learned", training_seed)
        fixed_mean = _mean_outcome("fixed_rule", training_seed)
        if learned_mean is None or fixed_mean is None:
            advantage_failures.append(f"seed{training_seed}:missing")
        elif learned_mean <= fixed_mean:
            advantage_failures.append(
                f"seed{training_seed}:learned={learned_mean:.4f}"
                f"<=fixed_rule={fixed_mean:.4f}"
            )
    gates.append(
        _gate(
            "fixed_rule_learning_advantage",
            passed=not advantage_failures,
            observed=f"failures={advantage_failures}",
            threshold=(
                "plan section 5.7 also requires learned to show its own "
                "advantage on the pre-declared learning metric: its mean "
                "held-out outcome score exceeds FixedRule's in every training "
                "seed. A learned checkpoint that a hand-written FSM outscores "
                "on the frozen endpoint is not promotable evidence of learned "
                "capability"
            ),
        )
    )

    e2e_shards = tuple(item for item in shards if item.arm == "e2e_rl")
    e2e_seeds = {item.training_seed for item in e2e_shards}
    gates.append(
        _gate(
            "e2e_rl_baseline_present",
            passed=(
                e2e_seeds == set(resolved.training_seeds)
                and all(
                    len(item.layout_results)
                    == len(capabilities) * resolved.layouts_per_tier
                    for item in e2e_shards
                )
                and all(item.policy_digest for item in e2e_shards)
            ),
            observed=(
                f"seeds={sorted(e2e_seeds)}, "
                f"expected={list(resolved.training_seeds)}"
            ),
            threshold=(
                "a trained end-to-end RL arm exists for every training seed; "
                "random is not a substitute"
            ),
        )
    )

    learned_shards = tuple(item for item in shards if item.arm == "learned")
    probe_summaries = tuple(
        item.probe_summary
        for item in learned_shards
        if item.probe_summary is not None
    )
    probes_complete = len(probe_summaries) == len(resolved.training_seeds)
    gates.append(
        _gate(
            "p0_action_sensitivity",
            passed=probes_complete
            and all(item.action_chain_passed for item in probe_summaries),
            observed=(
                "pass"
                if probes_complete
                and all(item.action_chain_passed for item in probe_summaries)
                else repr(
                    tuple(
                        item.action_chain_failures for item in probe_summaries
                    )
                )
            ),
            threshold="all per-body P0 action probes pass in every training seed",
        )
    )
    gates.append(
        _gate(
            "carrying_home_action_alignment",
            passed=probes_complete
            and all(
                item.home_probe_count > 0
                and item.home_aligned_bodies == item.home_probe_count
                for item in probe_summaries
            ),
            observed=repr(
                tuple(
                    (item.home_aligned_bodies, item.home_probe_count)
                    for item in probe_summaries
                )
            ),
            threshold=(
                "every carrying-state probe changes action and turns toward "
                "home, in every training seed"
            ),
        )
    )
    gates.append(
        _gate(
            "post_pickup_uturn_progress",
            passed=probes_complete
            and all(
                item.uturn_probe_count == item.home_probe_count
                and item.uturn_aligned_bodies
                >= item.required_aligned_bodies
                for item in probe_summaries
            ),
            observed=repr(
                tuple(
                    (
                        item.uturn_aligned_bodies,
                        item.uturn_probe_count,
                        item.required_aligned_bodies,
                    )
                    for item in probe_summaries
                )
            ),
            threshold=(
                "in every training seed, >=60% bodies pass both frozen "
                "+/-135-degree post-pickup lanes by delivering or sustaining "
                "net home-distance reduction; direction-only turns do not pass"
            ),
        )
    )
    gates.append(
        _gate(
            "food_steering_alignment",
            passed=probes_complete
            and all(
                item.food_probe_count > 0
                and item.food_aligned_bodies >= item.required_aligned_bodies
                for item in probe_summaries
            ),
            observed=repr(
                tuple(
                    (
                        item.food_aligned_bodies,
                        item.food_probe_count,
                        item.required_aligned_bodies,
                    )
                    for item in probe_summaries
                )
            ),
            threshold=(
                ">=60% bodies steer toward near food (left food -> left turn, "
                "right food -> right turn) in every training seed; near "
                "pickups alone do not prove this"
            ),
        )
    )

    learned_rows = tuple(
        row for key, row in rows.items() if key[0] == "learned"
    )
    switches = sum(row.switch_count for row in learned_rows)
    closures = sum(row.non_timeout_segment_closures for row in learned_rows)
    per_seed_temporal_ok = all(
        any(
            row.switch_count > 0
            for key, row in rows.items()
            if key[0] == "learned" and key[1] == training_seed
        )
        and any(
            row.non_timeout_segment_closures > 0
            for key, row in rows.items()
            if key[0] == "learned" and key[1] == training_seed
        )
        for training_seed in resolved.training_seeds
    )
    gates.append(
        _gate(
            "temporal_non_timeout_closure",
            passed=per_seed_temporal_ok,
            observed=f"switches={switches}, non_timeout_closures={closures}",
            threshold=(
                "every training seed shows a real beta switch and a "
                "non-timeout segment closure on held-out layouts"
            ),
        )
    )

    learning_arms = frozenset(
        spec.name for spec in ECOLOGY_P2_ARM_SPECS if spec.learning
    )
    learning_rows = tuple(
        row for key, row in rows.items() if key[0] in learning_arms
    )
    gates.append(
        _gate(
            "frozen_evaluation",
            passed=bool(learning_rows)
            and all(
                row.policy_fingerprint_stable
                and row.temporal_learning_fingerprint_stable
                for row in learning_rows
            ),
            observed=f"checkpoint_bearing_evaluations={len(learning_rows)}",
            threshold="policy and temporal-learning owners remain frozen",
        )
    )
    gates.append(
        _gate(
            "replay_lineage",
            passed=bool(learning_rows)
            and all(
                row.replay_settlement_coverage >= 0.99
                and row.replay_lineage_coverage >= 0.99
                and row.replay_drop_count == 0
                for row in learning_rows
            ),
            observed=f"checkpoint_bearing_evaluations={len(learning_rows)}",
            threshold="settlement/lineage >=0.99 and drop=0",
        )
    )

    training_shards = tuple(
        item
        for item in shards
        if item.arm_spec.learning and item.arm_spec.trains
    )
    gates.append(
        _gate(
            "archive_integrity",
            passed=bool(training_shards)
            and all(
                item.archive_roundtrip_ok is True for item in training_shards
            ),
            observed=repr(
                tuple(
                    (item.arm, item.training_seed, item.archive_roundtrip_ok)
                    for item in training_shards
                )
            ),
            threshold=(
                "every trained shard archive decodes back to the identical "
                "policy fingerprints"
            ),
        )
    )
    gates.append(
        _gate(
            "archive_corruption_rollback",
            passed=bool(training_shards)
            and all(
                item.archive_corruption_rejected is True
                for item in training_shards
            ),
            observed=repr(
                tuple(
                    (
                        item.arm,
                        item.training_seed,
                        item.archive_corruption_rejected,
                    )
                    for item in training_shards
                )
            ),
            threshold=(
                "every trained shard refuses a byte-corrupted archive and an "
                "archive presented under a mismatched compatibility "
                "declaration (plan section 5.7 corruption rollback / schema "
                "compatibility)"
            ),
        )
    )

    shard_shas = {item.source_provenance.git_sha for item in shards}
    dirty_shards = tuple(
        f"{item.arm}@{item.training_seed}"
        for item in shards
        if item.source_provenance.worktree_dirty
    )
    unidentified = tuple(
        f"{item.arm}@{item.training_seed}"
        for item in shards
        if not item.source_provenance.git_sha
    )
    source_git_sha = (
        next(iter(shard_shas)) if len(shard_shas) == 1 else ""
    )
    gates.append(
        _gate(
            "provenance_clean",
            # Plan section 5.4: any code change invalidates the whole batch.
            # The aggregator owns that check rather than delegating it to a
            # driver, so a library caller cannot merge two implementations.
            passed=(
                worktree_clean
                and len(shard_shas) == 1
                and not dirty_shards
                and not unidentified
            ),
            observed=(
                f"aggregate_worktree_clean={worktree_clean}, "
                f"shard_git_shas={sorted(shard_shas)}, "
                f"dirty={list(dirty_shards)}, "
                f"unidentified={list(unidentified)}"
            ),
            threshold=(
                "every shard reports the same non-empty git SHA and a clean "
                "worktree, and the aggregating tree is clean too"
            ),
        )
    )

    gate_tuple = tuple(gates)
    if tuple(item.name for item in gate_tuple) != ECOLOGY_P2_GATE_NAMES:
        raise RuntimeError("P2 gate schema drift")
    gate_by_name = {item.name: item for item in gate_tuple}
    endpoints = tuple(
        EcologyP2PrimaryEndpoint(
            name=name,
            passed=all(gate_by_name[gate].passed for gate in supporting),
            supporting_gates=supporting,
            failed_gates=tuple(
                gate for gate in supporting if not gate_by_name[gate].passed
            ),
        )
        for name, supporting in ECOLOGY_P2_PRIMARY_ENDPOINTS
    )
    secondary = _secondary_endpoints(
        rows=rows,
        probe_summaries=probe_summaries,
    )
    if tuple(item.name for item in secondary) != (
        ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES
    ):
        raise RuntimeError("P2 secondary endpoint schema drift")
    breakpoints = tuple(item.name for item in gate_tuple if not item.passed)
    verdict = "PASS" if not breakpoints else "BLOCK"
    return EcologyP2Report(
        schema_version=ECOLOGY_P2_SCHEMA_VERSION,
        config=resolved,
        preregistration_digest=digest,
        prerequisite=prerequisite,
        p1_prerequisites=p1_prerequisites,
        device=resolved.device,
        training_seeds=resolved.training_seeds,
        heldout_layout_seeds=heldout_layout_seeds(resolved),
        arms=expected_arms,
        shard_digests=tuple(
            sorted(
                (
                    f"{item.arm}@{item.training_seed}",
                    _sha256(_stable_json_bytes(item.to_dict())),
                )
                for item in shards
            )
        ),
        capability_results=tuple(capability_results),
        paired_effects=paired_effects,
        primary_endpoints=endpoints,
        secondary_endpoints=secondary,
        source_git_sha=source_git_sha,
        gates=gate_tuple,
        verdict=verdict,
        diagnostic_breakpoints=breakpoints,
        description=(
            "PASS: all P2 confirmatory gates passed"
            if verdict == "PASS"
            else "BLOCK: " + ", ".join(breakpoints)
        ),
    )


# ---------------------------------------------------------------------------
# P2-A preflight
# ---------------------------------------------------------------------------


async def run_ecology_p2_preflight(
    config: EcologyP2Config,
    *,
    source_provenance: EcologyP2SourceProvenance,
    training_seed: int | None = None,
    p1_report_path: Path | None = None,
    prerequisite: EcologyP2Prerequisite | None = None,
    repo_root: Path | None = None,
    progress_dir: Path | None = None,
    arms: tuple[str, ...] = ECOLOGY_P2_CORE_ARM_NAMES,
) -> EcologyP2PreflightReport:
    """P2-A: full-stack rehearsal on one training seed.

    Exercises the whole chain at the formal ant count, records wall clock and
    artifact size so the confirmatory shard plan can be frozen on measurement
    rather than guesswork, and re-runs one held-out layout to confirm the
    evaluation path is deterministic. Preflight results are marked as such and
    the aggregator refuses to merge them into confirmatory statistics.
    """

    seed = (
        training_seed
        if training_seed is not None
        else config.training_seeds[0]
    )
    if prerequisite is None:
        if p1_report_path is None:
            raise EcologyP2PrerequisiteError(
                "P2 preflight requires the frozen P1 report"
            )
        prerequisite = load_p1_prerequisite(
            p1_report_path,
            repo_root=repo_root,
            expected_training_seed=seed,
        )
    unknown = tuple(arm for arm in arms if arm not in ECOLOGY_P2_ARM_SPEC_BY_NAME)
    if unknown:
        raise ValueError(f"unknown P2 preflight arms: {list(unknown)}")
    # Every requested arm is checked here, not when its turn comes round: by
    # the time an unexecutable arm's turn arrived, the arms ahead of it in
    # ``arms`` would already have spent their full training budget.
    #
    # The rehearsal is refused, but it is refused *with an artifact*. Plan
    # section 2.3 requires every stage to leave a report even when it fails, so
    # the blocked preflight returns a complete ``passed=False`` report naming
    # each unexecutable arm and the exact levers it needs -- and spends zero
    # budget doing it. ``run_ecology_p2_shard`` still raises
    # :class:`EcologyP2ArmLeverUnavailableError`, because a single-cell run has
    # no report to carry the diagnosis.
    blocked = tuple(
        (arm, levers)
        for arm, levers in (
            (arm, unreachable_arm_levers(ECOLOGY_P2_ARM_SPEC_BY_NAME[arm]))
            for arm in arms
        )
        if levers
    )
    if blocked:
        detail = "; ".join(
            f"{arm} needs {'+'.join(levers)}" for arm, levers in blocked
        )
        return EcologyP2PreflightReport(
            schema_version=ECOLOGY_P2_PREFLIGHT_SCHEMA_VERSION,
            config=config,
            preregistration_digest=preregistration_digest(config),
            prerequisite=prerequisite,
            training_seed=seed,
            device=config.device,
            arms=tuple(arms),
            shard_wall_clock_seconds=(),
            shard_archive_size_bytes=(),
            determinism_repeat_matches=False,
            determinism_detail=(
                "not evaluated: the rehearsal was refused before any arm ran"
            ),
            passed=False,
            breakpoints=(
                "unexecutable_arms="
                + str([f"{arm}({'+'.join(levers)})" for arm, levers in blocked]),
            ),
            description=(
                "P2-A preflight blocked before spending any budget: these "
                "pre-registered arms declare levers the curriculum session "
                f"builder cannot express -- {detail}. The missing hop is "
                "ecology_curriculum._session_config / _train_arm / "
                "_evaluate_arm accepting and forwarding them onto "
                "AntSessionConfig.temporal_policy / .joint_schedule (both are "
                "already opaque passthroughs; JointLoopSchedule and "
                "LearnedLiteTemporalPolicy are vz-temporal internals this "
                "package may not import, so the objects must be built by the "
                "curriculum owner or by a vz-runtime facade factory). "
                "Rehearsing the remaining arms would spend the formal budget "
                "on a matrix that cannot be completed as pre-registered."
            ),
        )

    shard_reports: list[EcologyP2ShardReport] = []
    for arm in arms:
        shard_reports.append(
            await run_ecology_p2_shard(
                config,
                training_seed=seed,
                arm=arm,
                source_provenance=source_provenance,
                prerequisite=prerequisite,
                progress_dir=progress_dir,
                preflight=True,
            )
        )

    # Determinism probe: the first held-out capability is replayed from the
    # journalled state under a fresh runner. A drifting evaluation path makes
    # every paired statistic meaningless, so this is checked before the
    # confirmatory budget is committed.
    capability, scenario, tier = _evaluation_specs()[0]
    curriculum = _curriculum_config(config, training_seed=seed)
    repeat = _run_baseline_layout(
        config=config,
        curriculum=curriculum,
        training_seed=seed,
        arm="fixed_rule",
        capability=capability,
        scenario=scenario,
        tier=tier,
        seed=_heldout_seed(capability_index=0, layout_index=0),
    )
    fixed_rule_shards = tuple(
        item for item in shard_reports if item.arm == "fixed_rule"
    )
    if not fixed_rule_shards:
        determinism_ok = False
        determinism_detail = "fixed_rule arm absent from the preflight set"
    else:
        original = next(
            row
            for row in fixed_rule_shards[0].layout_results
            if row.capability == capability
            and row.seed == _heldout_seed(capability_index=0, layout_index=0)
        )
        determinism_ok = asdict(original) == asdict(repeat)
        determinism_detail = (
            "identical replay"
            if determinism_ok
            else f"drift: {asdict(original)} != {asdict(repeat)}"
        )

    breakpoints: list[str] = []
    if not determinism_ok:
        breakpoints.append("determinism_repeat")
    incomplete = tuple(
        item.arm for item in shard_reports if not item.training_complete
    )
    if incomplete:
        breakpoints.append(f"incomplete_shards={list(incomplete)}")
    formal_failures = _formal_configuration_failures(config)
    if formal_failures:
        breakpoints.append(f"below_formal_budget={list(formal_failures)}")
    return EcologyP2PreflightReport(
        schema_version=ECOLOGY_P2_PREFLIGHT_SCHEMA_VERSION,
        config=config,
        preregistration_digest=preregistration_digest(config),
        prerequisite=prerequisite,
        training_seed=seed,
        device=config.device,
        arms=tuple(arms),
        shard_wall_clock_seconds=tuple(
            (item.arm, item.wall_clock_seconds) for item in shard_reports
        ),
        shard_archive_size_bytes=tuple(
            (item.arm, item.archive_size_bytes)
            for item in shard_reports
            if item.archive_size_bytes is not None
        ),
        determinism_repeat_matches=determinism_ok,
        determinism_detail=determinism_detail,
        passed=not breakpoints,
        breakpoints=tuple(breakpoints),
        description=(
            "P2-A preflight passed; confirmatory shards may be scheduled"
            if not breakpoints
            else "P2-A preflight blocked: " + ", ".join(breakpoints)
        ),
    )


def shard_report_from_dict(payload: dict[str, Any]) -> EcologyP2ShardReport:
    """Rebuild a shard report written by a previous process.

    Confirmatory shards are run as separate processes (often on separate
    machines), so the aggregator reads them back from disk. Structural drift
    fails loudly here rather than silently producing a thinner matrix.
    """

    schema_version = payload.get("schema_version")
    if schema_version != ECOLOGY_P2_SHARD_SCHEMA_VERSION:
        raise ValueError(
            "P2 shard schema mismatch: "
            f"expected={ECOLOGY_P2_SHARD_SCHEMA_VERSION!r}, "
            f"actual={schema_version!r}. A shard written before this version "
            "carries neither source provenance nor the corruption-rollback "
            "and secondary-endpoint fields, so it cannot be folded into a "
            "confirmatory verdict; rerun the shard."
        )
    config = EcologyP2Config(
        n_ants=int(payload["config"]["n_ants"]),
        temporal_latent_dim=int(payload["config"]["temporal_latent_dim"]),
        training_rounds=int(payload["config"]["training_rounds"]),
        validation_rounds=int(payload["config"]["validation_rounds"]),
        heldout_rounds=int(payload["config"]["heldout_rounds"]),
        layouts_per_tier=int(payload["config"]["layouts_per_tier"]),
        training_seeds=tuple(
            int(value) for value in payload["config"]["training_seeds"]
        ),
        device=str(payload["config"]["device"]),
        bootstrap_samples=int(payload["config"]["bootstrap_samples"]),
    )
    raw_spec = payload["arm_spec"]
    spec = EcologyP2ArmSpec(
        name=str(raw_spec["name"]),
        batch=str(raw_spec["batch"]),
        learning=bool(raw_spec["learning"]),
        optimize=bool(raw_spec["optimize"]),
        local_valence_enabled=bool(raw_spec["local_valence_enabled"]),
        segment_credit_enabled=bool(raw_spec["segment_credit_enabled"]),
        prediction_error_enabled=bool(raw_spec["prediction_error_enabled"]),
        temporal_writeback_enabled=bool(
            raw_spec["temporal_writeback_enabled"]
        ),
        temporal_policy_kind=str(raw_spec["temporal_policy_kind"]),
        joint_ssl_interval=(
            None
            if raw_spec["joint_ssl_interval"] is None
            else int(raw_spec["joint_ssl_interval"])
        ),
        joint_rl_interval=(
            None
            if raw_spec["joint_rl_interval"] is None
            else int(raw_spec["joint_rl_interval"])
        ),
        trains=bool(raw_spec["trains"]),
        description=str(raw_spec["description"]),
    )
    if ECOLOGY_P2_ARM_SPEC_BY_NAME.get(spec.name) != spec:
        raise ValueError(
            f"P2 shard arm spec drifted from the pre-registered set: {spec.name}"
        )
    raw_probe = payload.get("probe_summary")
    probe = (
        EcologyP2ProbeSummary(
            action_chain_passed=bool(raw_probe["action_chain_passed"]),
            action_chain_failures=tuple(
                str(value) for value in raw_probe["action_chain_failures"]
            ),
            home_probe_count=int(raw_probe["home_probe_count"]),
            home_aligned_bodies=int(raw_probe["home_aligned_bodies"]),
            uturn_probe_count=int(raw_probe["uturn_probe_count"]),
            uturn_aligned_bodies=int(raw_probe["uturn_aligned_bodies"]),
            food_probe_count=int(raw_probe["food_probe_count"]),
            food_aligned_bodies=int(raw_probe["food_aligned_bodies"]),
            required_aligned_bodies=int(raw_probe["required_aligned_bodies"]),
        )
        if isinstance(raw_probe, dict)
        else None
    )
    raw_prerequisite = payload["prerequisite"]
    raw_provenance = payload["source_provenance"]
    if not isinstance(raw_provenance, dict):
        raise ValueError("P2 shard source_provenance must be an object")
    return EcologyP2ShardReport(
        schema_version=str(payload["schema_version"]),
        config=config,
        training_seed=int(payload["training_seed"]),
        arm=str(payload["arm"]),
        batch=str(payload["batch"]),
        arm_spec=spec,
        preregistration_digest=str(payload["preregistration_digest"]),
        schedule_sha256=str(payload["schedule_sha256"]),
        prerequisite=EcologyP2Prerequisite(
            report_path=str(raw_prerequisite["report_path"]),
            report_sha256=str(raw_prerequisite["report_sha256"]),
            schema_version=str(raw_prerequisite["schema_version"]),
            verdict=str(raw_prerequisite["verdict"]),
            training_seed=int(raw_prerequisite["training_seed"]),
            configuration_digest=str(
                raw_prerequisite["configuration_digest"]
            ),
        ),
        device=str(payload["device"]),
        preflight=bool(payload["preflight"]),
        training_complete=bool(payload["training_complete"]),
        completed_training_episodes=int(
            payload["completed_training_episodes"]
        ),
        scheduled_training_episodes=int(
            payload["scheduled_training_episodes"]
        ),
        policy_digest=str(payload["policy_digest"]),
        initial_policy_digest=str(payload["initial_policy_digest"]),
        archive_roundtrip_ok=(
            None
            if payload["archive_roundtrip_ok"] is None
            else bool(payload["archive_roundtrip_ok"])
        ),
        archive_corruption_rejected=(
            None
            if payload["archive_corruption_rejected"] is None
            else bool(payload["archive_corruption_rejected"])
        ),
        archive_size_bytes=(
            None
            if payload["archive_size_bytes"] is None
            else int(payload["archive_size_bytes"])
        ),
        source_provenance=EcologyP2SourceProvenance(
            git_sha=str(raw_provenance["git_sha"]),
            git_branch=str(raw_provenance["git_branch"]),
            worktree_dirty=bool(raw_provenance["worktree_dirty"]),
        ),
        wall_clock_seconds=float(payload["wall_clock_seconds"]),
        layout_results=tuple(
            _layout_result_from_dict(item)
            for item in payload["layout_results"]
        ),
        probe_summary=probe,
        description=str(payload["description"]),
    )


def load_shard_checkpoint_archives(
    *,
    progress_dir: Path,
    config: EcologyP2Config,
    training_seed: int,
    arm: str,
    prerequisite: EcologyP2Prerequisite,
) -> tuple[bytes, ...]:
    """The journalled colony archives of a completed shard.

    This is how a P2 ``PASS`` becomes a loadable promotion bundle: the learned
    shard already committed its trained colony to the two-slot journal, digest
    checked and compatibility bound. Re-running training to produce a
    promotable archive would be a second, unaudited fork point.

    Every resume invariant is re-checked, so a partially trained or
    differently pre-registered shard cannot be promoted.
    """

    shard_dir = _shard_dir(
        progress_dir.resolve(), training_seed=training_seed, arm=arm
    )
    state = _load_shard_state(
        shard_dir=shard_dir,
        config=config,
        training_seed=training_seed,
        arm=arm,
        digest=preregistration_digest(config),
        schedule_sha256=_schedule_digest(
            p2_training_schedule(config, training_seed=training_seed)
        ),
        prerequisite=prerequisite,
    )
    if state is None:
        raise ValueError(
            f"no P2 shard journal at {shard_dir}; a promotion bundle may only "
            "carry a checkpoint an audited confirmatory shard produced"
        )
    if not bool(state.get("training_complete")):
        raise ValueError(
            f"P2 shard {arm}@{training_seed} is not training-complete; an "
            "incomplete shard must never be promoted"
        )
    return _read_shard_archive(
        shard_dir=shard_dir, state=state, config=config
    )


__all__ = [
    "ECOLOGY_P2_ABLATION_ARM_NAMES",
    "ECOLOGY_P2_ARM_LEVER_PARAMETERS",
    "ECOLOGY_P2_ARM_NAMES",
    "ECOLOGY_P2_ARM_SPECS",
    "ECOLOGY_P2_CORE_ARM_NAMES",
    "ECOLOGY_P2_DIVERGENT_POLICY_ARM_NAMES",
    "ECOLOGY_P2_FROZEN_POLICY_ARM_NAMES",
    "ECOLOGY_P2_GATE_NAMES",
    "ECOLOGY_P2_HELDOUT_SEED_BASE",
    "ECOLOGY_P2_OUTCOME_SCORE_WEIGHTS",
    "ECOLOGY_P2_PAIRED_COMPARISONS",
    "ECOLOGY_P2_PREFLIGHT_SCHEMA_VERSION",
    "ECOLOGY_P2_PRIMARY_ENDPOINTS",
    "ECOLOGY_P2_PROGRESS_SCHEMA_VERSION",
    "ECOLOGY_P2_SCHEMA_VERSION",
    "ECOLOGY_P2_SECONDARY_ENDPOINT_NAMES",
    "ECOLOGY_P2_SHARD_SCHEMA_VERSION",
    "EcologyP2ArmLeverUnavailableError",
    "EcologyP2ArmSpec",
    "EcologyP2CapabilityResult",
    "EcologyP2Config",
    "EcologyP2Gate",
    "EcologyP2LayoutResult",
    "EcologyP2PairedEffect",
    "EcologyP2PreflightReport",
    "EcologyP2PrerequisiteError",
    "EcologyP2Prerequisite",
    "EcologyP2PrimaryEndpoint",
    "EcologyP2ProbeSummary",
    "EcologyP2ProgressPaused",
    "EcologyP2Report",
    "EcologyP2SecondaryEndpoint",
    "EcologyP2ShardReport",
    "EcologyP2SourceProvenance",
    "aggregate_ecology_p2_shards",
    "heldout_layout_seeds",
    "load_p1_prerequisite",
    "load_shard_checkpoint_archives",
    "outcome_score",
    "p2_training_schedule",
    "preregistration_digest",
    "run_ecology_p2_preflight",
    "run_ecology_p2_shard",
    "shard_report_from_dict",
    "unreachable_arm_levers",
]
