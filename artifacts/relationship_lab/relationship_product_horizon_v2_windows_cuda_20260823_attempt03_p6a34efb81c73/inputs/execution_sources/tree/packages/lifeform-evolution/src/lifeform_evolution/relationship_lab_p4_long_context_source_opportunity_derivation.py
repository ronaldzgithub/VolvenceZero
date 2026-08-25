"""Exact arithmetic for the P4.7 source-opportunity planning contract.

This module is intentionally limited to immutable, in-memory derivations.  It
does not read or write files, materialize a source, call a model, use CUDA, or
emit semantic/scientific claims.  Its return values are arithmetic and finite
inventory values only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math


DEVELOPMENT_SPLIT_ID = "development"
QUALIFICATION_SPLIT_ID = "qualification"
FORMAL_SPLIT_ID = "formal"
ANALYSIS_ROOT_ROLE = "analysis"
DONOR_ROOT_ROLE = "donor"

SURFACE_CAPACITY = 1 << 15
SURFACE_AFFINE_MULTIPLIER = 2085
SURFACE_AFFINE_OFFSET = 21504


@dataclass(frozen=True)
class SurfaceFactorTypedLevel:
    """One LSB-ordered binary axis and its two typed blueprint values."""

    axis_id: str
    value_zero: str
    value_one: str


SURFACE_FACTOR_TYPED_VALUE_REGISTRY = (
    SurfaceFactorTypedLevel(
        "relationship_history_depth",
        "four_prior_session_anchors",
        "sixteen_prior_session_anchors",
    ),
    SurfaceFactorTypedLevel(
        "preference_stability",
        "stable_until_registered_origin",
        "revised_once_at_registered_origin",
    ),
    SurfaceFactorTypedLevel(
        "boundary_strictness",
        "soft_decline_with_revisit_allowed",
        "hard_decline_until_explicit_reconsent",
    ),
    SurfaceFactorTypedLevel(
        "commitment_horizon",
        "single_session_commitment",
        "cross_session_commitment",
    ),
    SurfaceFactorTypedLevel(
        "repair_cost",
        "one_acknowledgement_step",
        "acknowledgement_plus_followthrough_step",
    ),
    SurfaceFactorTypedLevel(
        "absence_duration",
        "one_session_gap",
        "four_session_gap",
    ),
    SurfaceFactorTypedLevel(
        "belief_uncertainty",
        "single_supported_hypothesis",
        "two_live_hypotheses",
    ),
    SurfaceFactorTypedLevel(
        "goal_conflict",
        "aligned_goals",
        "competing_goals",
    ),
    SurfaceFactorTypedLevel(
        "consent_reversibility",
        "reversible_by_single_fact",
        "requires_new_consent_event",
    ),
    SurfaceFactorTypedLevel(
        "execution_dependency",
        "no_external_dependency",
        "one_registered_external_dependency",
    ),
    SurfaceFactorTypedLevel(
        "return_gap_pattern",
        "regular_return_gap",
        "irregular_return_gap",
    ),
    SurfaceFactorTypedLevel(
        "correction_salience",
        "low_salience_correction",
        "high_salience_correction",
    ),
    SurfaceFactorTypedLevel(
        "open_loop_density",
        "one_open_loop",
        "four_open_loops",
    ),
    SurfaceFactorTypedLevel(
        "relationship_regime",
        "steady_continuity",
        "rupture_repair",
    ),
    SurfaceFactorTypedLevel(
        "challenge_order",
        "forward_nondecisive_challenge_order",
        "reverse_nondecisive_challenge_order",
    ),
)
SURFACE_FACTOR_AXES_IN_BIT_ORDER = tuple(item.axis_id for item in SURFACE_FACTOR_TYPED_VALUE_REGISTRY)

ACTION_ORDER = (
    "stay_present_without_probe",
    "respect_space_with_return_option",
    "neutral_noop",
)
INVALID_GENERATED_ACTION_ID = "INVALID_GENERATED_ACTION"
EVALUATION_DISTANCE_BINS = (4096, 8192, 16384, 32768)
EVALUATION_FACT_VALUES = (0, 1)
FACT_ZERO_UTILITY_VECTOR = (1, -1, 0)
FACT_ONE_UTILITY_VECTOR = (-1, 1, 0)

REFERENCE_SUCCESS_PROBABILITY = Fraction(11, 20)
COMPARATOR_SUCCESS_PROBABILITY = Fraction(9, 20)
COMPARATOR_COUNT = 8
SUCCESS_UTILITY = 1
FAILURE_UTILITY = -1

FACT_ORIENTATION_DOMAIN = "volvence.relationship_p4_long_context_v4.source_fact_orientation.v1"
FACT_ORIENTATION_SEED = 20260831
FACT_ORIENTATION_BLOCK_SIZE = 32
FACT_ORIENTATION_COUNT_PER_VALUE = 16
REVERSAL_PAIR_COUNT = 4
FACT_ORIENTATION_RANKING_COUNTER_FIELDS = (
    "domain_tag",
    "seed",
    "split",
    "reversal_pair_index",
    "root_block_ordinal",
    "within_block_root_ordinal",
)
FACT_ORIENTATION_RANKING_EXCLUDED_FIELDS = (
    "candidate_formal_root_count",
    "arm_id",
    "candidate_cell_id",
    "model_id",
    "model_output",
    "outcome",
    "power_result",
    "host_identity",
    "cuda_backend",
)


@dataclass(frozen=True)
class SourceRootNamespaceSpec:
    """One frozen independent-root namespace primitive."""

    split_id: str
    root_role: str
    root_count: int


ROOT_NAMESPACE_SPECS = (
    SourceRootNamespaceSpec(DEVELOPMENT_SPLIT_ID, ANALYSIS_ROOT_ROLE, 32),
    SourceRootNamespaceSpec(DEVELOPMENT_SPLIT_ID, DONOR_ROOT_ROLE, 32),
    SourceRootNamespaceSpec(QUALIFICATION_SPLIT_ID, ANALYSIS_ROOT_ROLE, 64),
    SourceRootNamespaceSpec(QUALIFICATION_SPLIT_ID, DONOR_ROOT_ROLE, 64),
    SourceRootNamespaceSpec(FORMAL_SPLIT_ID, ANALYSIS_ROOT_ROLE, 8192),
    SourceRootNamespaceSpec(FORMAL_SPLIT_ID, DONOR_ROOT_ROLE, 8192),
)


@dataclass(frozen=True)
class SourceRootNamespaceAllocation:
    """One namespace after global and role-local offsets are assigned."""

    split_id: str
    root_role: str
    root_count: int
    global_slot_start: int
    global_slot_stop_exclusive: int
    role_ordinal_start: int
    role_ordinal_stop_exclusive: int


@dataclass(frozen=True)
class SourceRootSurface:
    """One independent analysis or donor root and its affine surface code."""

    split_id: str
    root_role: str
    namespace_ordinal: int
    role_ordinal: int
    global_slot: int
    surface_code: int
    factor_bits: tuple[int, ...]
    typed_blueprint_values: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisDonorRootPair:
    """One split-local, same-ordinal analysis/donor assignment."""

    pair_ordinal: int
    split_id: str
    split_ordinal: int
    analysis_global_slot: int
    analysis_surface_code: int
    donor_global_slot: int
    donor_surface_code: int


@dataclass(frozen=True)
class CounterfactualTwinMapping:
    """A derived twin mapping that consumes no additional independent root."""

    mapping_ordinal: int
    split_id: str
    split_ordinal: int
    analysis_global_slot: int
    analysis_surface_code: int
    decisive_decision_index: int
    reversal_pair_ordinal: int
    distance_bin_lower_bound_tokens: int
    original_decisive_fact_value: int
    counterfactual_decisive_fact_value: int
    utility_oracle_descendants_recomputed: bool
    all_other_exogenous_nodes_unchanged: bool
    independent_root: bool


@dataclass(frozen=True)
class FormalCandidatePrefix:
    """A half-open prefix of the frozen formal analysis/donor namespaces."""

    root_count: int
    analysis_global_slot_start: int
    analysis_global_slot_stop_exclusive: int
    donor_global_slot_start: int
    donor_global_slot_stop_exclusive: int
    formal_pair_ordinal_start: int
    formal_pair_ordinal_stop_exclusive: int


@dataclass(frozen=True)
class SourceRootSurfaceDerivation:
    """Complete finite root/surface inventory derived from frozen primitives."""

    surface_capacity: int
    affine_multiplier: int
    affine_offset: int
    surface_factor_axes_in_bit_order: tuple[str, ...]
    surface_factor_bit_decoding: str
    surface_factor_typed_value_registry: tuple[SurfaceFactorTypedLevel, ...]
    namespaces: tuple[SourceRootNamespaceAllocation, ...]
    roots: tuple[SourceRootSurface, ...]
    analysis_root_count: int
    donor_root_count: int
    analysis_donor_pairs: tuple[AnalysisDonorRootPair, ...]
    counterfactual_twin_mappings: tuple[CounterfactualTwinMapping, ...]
    formal_candidate_prefixes: tuple[FormalCandidatePrefix, ...]


@dataclass(frozen=True)
class SourceEvaluationSlot:
    """One fact-reversal evaluation slot and its ordered utility vector."""

    slot_ordinal: int
    reversal_pair_ordinal: int
    distance_bin_tokens: int
    fact_value: int
    utility_vector: tuple[int, ...]


@dataclass(frozen=True)
class SourceEvaluationDesign:
    """Closed action registry plus the eight exact evaluation slots."""

    action_order: tuple[str, ...]
    invalid_generated_action_id: str
    invalid_generated_action_in_registry: bool
    slots: tuple[SourceEvaluationSlot, ...]


@dataclass(frozen=True)
class SyntheticPlanningAtom:
    """One generic decision in the independent nine-node Bernoulli support."""

    atom_ordinal: int
    reference_success: bool
    comparator_successes: tuple[bool, ...]
    probability: Fraction
    utility_vector: tuple[int, ...]
    contrast_vector: tuple[int, ...]


@dataclass(frozen=True)
class ExactEigenvalueMultiplicity:
    """One exact eigenvalue and its algebraic multiplicity."""

    eigenvalue: Fraction
    multiplicity: int


@dataclass(frozen=True)
class SyntheticPlanningGeneratorDerivation:
    """One-decision atoms and exact moments; these make no semantic claim."""

    reference_success_probability: Fraction
    comparator_success_probabilities: tuple[Fraction, ...]
    atoms: tuple[SyntheticPlanningAtom, ...]
    contrast_means: tuple[Fraction, ...]
    contrast_covariance_matrix: tuple[tuple[Fraction, ...], ...]
    contrast_correlation_matrix: tuple[tuple[Fraction, ...], ...]
    correlation_eigenvalues: tuple[ExactEigenvalueMultiplicity, ...]


@dataclass(frozen=True)
class RootFactOrientationAssignment:
    """One analysis-root/reversal-pair orientation from a block-local rank."""

    assignment_ordinal: int
    split_id: str
    analysis_root_ordinal: int
    analysis_global_slot: int
    analysis_surface_code: int
    reversal_pair_ordinal: int
    block_ordinal: int
    within_block_ordinal: int
    ranking_digest_sha256: str
    rank_within_block: int
    orientation: int
    fact_values_by_decision_position: tuple[int, int]


@dataclass(frozen=True)
class FactOrientationBlockCommitment:
    """Content commitment and exact 16/16 balance for one 32-root block."""

    commitment_ordinal: int
    split_id: str
    reversal_pair_ordinal: int
    block_ordinal: int
    analysis_root_ordinal_start: int
    analysis_root_ordinal_stop_exclusive: int
    orientation_zero_count: int
    orientation_one_count: int
    assignment_digest_sha256: str


@dataclass(frozen=True)
class FormalCandidateFactPositionBalance:
    """Mechanical balance audit outside the orientation-ranking commitment."""

    formal_root_count: int
    reversal_pair_ordinal: int
    decision_position: int
    fact_zero_count: int
    fact_one_count: int


@dataclass(frozen=True)
class RootFactOrientationDerivation:
    """Frozen orientation inventory and its ranking/commitment contract."""

    ranking_domain: str
    ranking_seed: int
    ranking_hash_algorithm: str
    ranking_payload_contract: str
    ranking_counter_fields: tuple[str, ...]
    ranking_tie_break_fields: tuple[str, ...]
    ranking_excluded_fields: tuple[str, ...]
    analysis_root_count: int
    reversal_pair_count: int
    block_size: int
    orientation_count_per_value_per_block: int
    assignments: tuple[RootFactOrientationAssignment, ...]
    block_commitments: tuple[FactOrientationBlockCommitment, ...]
    assignment_inventory_digest_sha256: str
    formal_candidate_position_balances: tuple[FormalCandidateFactPositionBalance, ...]


def derive_formal_candidate_root_counts(
    *,
    first: int = 192,
    step: int = 64,
    last_inclusive: int = 8192,
) -> tuple[int, ...]:
    """Return the frozen ascending formal-root candidate grid."""

    _require_exact_int(first, "first")
    _require_exact_int(step, "step")
    _require_exact_int(last_inclusive, "last_inclusive")
    if (first, step, last_inclusive) != (192, 64, 8192):
        raise ValueError("formal candidate root-count primitives drift")
    if first <= 0 or step <= 0 or first > last_inclusive:
        raise ValueError("formal candidate root-count bounds are invalid")
    if (last_inclusive - first) % step:
        raise ValueError("formal candidate root-count grid has a partial tail")
    return tuple(range(first, last_inclusive + 1, step))


def derive_source_root_surface_layout(
    *,
    namespace_specs: Sequence[SourceRootNamespaceSpec] = ROOT_NAMESPACE_SPECS,
    surface_capacity: int = SURFACE_CAPACITY,
    affine_multiplier: int = SURFACE_AFFINE_MULTIPLIER,
    affine_offset: int = SURFACE_AFFINE_OFFSET,
    formal_candidate_root_counts: Sequence[int] | None = None,
) -> SourceRootSurfaceDerivation:
    """Assign all independent roots, twins, pairs, and strict formal prefixes."""

    specs = tuple(namespace_specs)
    if any(type(item) is not SourceRootNamespaceSpec for item in specs):
        raise TypeError("namespace_specs must contain SourceRootNamespaceSpec values")
    for index, spec in enumerate(specs):
        _require_nonempty_text(spec.split_id, f"namespace_specs[{index}].split_id")
        _require_nonempty_text(spec.root_role, f"namespace_specs[{index}].root_role")
        _require_positive_int(spec.root_count, f"namespace_specs[{index}].root_count")
    if specs != ROOT_NAMESPACE_SPECS:
        raise ValueError("root namespace order or cardinality drift")

    _require_positive_int(surface_capacity, "surface_capacity")
    _require_exact_int(affine_multiplier, "affine_multiplier")
    _require_exact_int(affine_offset, "affine_offset")
    if surface_capacity != SURFACE_CAPACITY:
        raise ValueError("surface capacity must remain exactly 2^15")
    if affine_multiplier != SURFACE_AFFINE_MULTIPLIER:
        raise ValueError("surface affine multiplier drift")
    if affine_offset != SURFACE_AFFINE_OFFSET:
        raise ValueError("surface affine offset drift")
    if not 0 <= affine_offset < surface_capacity:
        raise ValueError("surface affine offset leaves the surface domain")
    if not 0 < affine_multiplier < surface_capacity:
        raise ValueError("surface affine multiplier leaves the surface domain")
    if math.gcd(affine_multiplier, surface_capacity) != 1:
        raise ValueError("surface affine multiplier is not invertible")
    _validate_surface_factor_registry()

    candidates = (
        derive_formal_candidate_root_counts()
        if formal_candidate_root_counts is None
        else tuple(formal_candidate_root_counts)
    )
    if any(type(value) is not int for value in candidates):
        raise TypeError("formal candidate root counts must be exact integers")
    if candidates != derive_formal_candidate_root_counts():
        raise ValueError("formal candidate root-count grid drift")

    allocations: list[SourceRootNamespaceAllocation] = []
    roots: list[SourceRootSurface] = []
    global_slot = 0
    role_offsets = {ANALYSIS_ROOT_ROLE: 0, DONOR_ROOT_ROLE: 0}
    for spec in specs:
        if spec.root_role not in role_offsets:
            raise ValueError("unsupported independent-root role")
        global_start = global_slot
        role_start = role_offsets[spec.root_role]
        for namespace_ordinal in range(spec.root_count):
            surface_code = (affine_multiplier * global_slot + affine_offset) % surface_capacity
            factor_bits = tuple(
                (surface_code >> bit_ordinal) & 1 for bit_ordinal in range(len(SURFACE_FACTOR_TYPED_VALUE_REGISTRY))
            )
            typed_blueprint_values = tuple(
                level.value_zero if bit == 0 else level.value_one
                for level, bit in zip(
                    SURFACE_FACTOR_TYPED_VALUE_REGISTRY,
                    factor_bits,
                    strict=True,
                )
            )
            roots.append(
                SourceRootSurface(
                    split_id=spec.split_id,
                    root_role=spec.root_role,
                    namespace_ordinal=namespace_ordinal,
                    role_ordinal=role_start + namespace_ordinal,
                    global_slot=global_slot,
                    surface_code=surface_code,
                    factor_bits=factor_bits,
                    typed_blueprint_values=typed_blueprint_values,
                )
            )
            global_slot += 1
        role_offsets[spec.root_role] += spec.root_count
        allocations.append(
            SourceRootNamespaceAllocation(
                split_id=spec.split_id,
                root_role=spec.root_role,
                root_count=spec.root_count,
                global_slot_start=global_start,
                global_slot_stop_exclusive=global_slot,
                role_ordinal_start=role_start,
                role_ordinal_stop_exclusive=role_offsets[spec.root_role],
            )
        )

    expected_total = 16576
    if global_slot != expected_total:
        raise AssertionError("independent-root global-slot total drift")
    if tuple(root.global_slot for root in roots) != tuple(range(expected_total)):
        raise AssertionError("independent-root global slots are not contiguous")
    surface_codes = tuple(root.surface_code for root in roots)
    if len(set(surface_codes)) != expected_total:
        raise AssertionError("affine surface assignment is not injective")
    if any(not 0 <= code < surface_capacity for code in surface_codes):
        raise AssertionError("affine surface code leaves the frozen domain")
    for root in roots:
        if sum(bit << bit_ordinal for bit_ordinal, bit in enumerate(root.factor_bits)) != root.surface_code:
            raise AssertionError("LSB factor-bit decoding does not recover surface code")
        expected_blueprint = tuple(
            level.value_zero if bit == 0 else level.value_one
            for level, bit in zip(
                SURFACE_FACTOR_TYPED_VALUE_REGISTRY,
                root.factor_bits,
                strict=True,
            )
        )
        if root.typed_blueprint_values != expected_blueprint:
            raise AssertionError("surface code to typed-blueprint mapping drift")
    if role_offsets != {ANALYSIS_ROOT_ROLE: 8288, DONOR_ROOT_ROLE: 8288}:
        raise AssertionError("analysis/donor independent-root totals drift")

    roots_by_split_role = {
        (split_id, root_role): tuple(
            root for root in roots if root.split_id == split_id and root.root_role == root_role
        )
        for split_id in (DEVELOPMENT_SPLIT_ID, QUALIFICATION_SPLIT_ID, FORMAL_SPLIT_ID)
        for root_role in (ANALYSIS_ROOT_ROLE, DONOR_ROOT_ROLE)
    }
    pairs: list[AnalysisDonorRootPair] = []
    pair_ordinal = 0
    for split_id in (DEVELOPMENT_SPLIT_ID, QUALIFICATION_SPLIT_ID, FORMAL_SPLIT_ID):
        analysis_roots = roots_by_split_role[(split_id, ANALYSIS_ROOT_ROLE)]
        donor_roots = roots_by_split_role[(split_id, DONOR_ROOT_ROLE)]
        if len(analysis_roots) != len(donor_roots):
            raise AssertionError("split analysis/donor cardinality drift")
        for split_ordinal, (analysis, donor) in enumerate(zip(analysis_roots, donor_roots, strict=True)):
            if analysis.namespace_ordinal != split_ordinal or donor.namespace_ordinal != split_ordinal:
                raise AssertionError("split-local analysis/donor ordinal drift")
            pairs.append(
                AnalysisDonorRootPair(
                    pair_ordinal=pair_ordinal,
                    split_id=split_id,
                    split_ordinal=split_ordinal,
                    analysis_global_slot=analysis.global_slot,
                    analysis_surface_code=analysis.surface_code,
                    donor_global_slot=donor.global_slot,
                    donor_surface_code=donor.surface_code,
                )
            )
            pair_ordinal += 1
    twins: list[CounterfactualTwinMapping] = []
    for split_id in (
        DEVELOPMENT_SPLIT_ID,
        QUALIFICATION_SPLIT_ID,
        FORMAL_SPLIT_ID,
    ):
        analysis_roots = roots_by_split_role[(split_id, ANALYSIS_ROOT_ROLE)]
        for block_ordinal in range(len(analysis_roots) // FACT_ORIENTATION_BLOCK_SIZE):
            orientation_by_within = _derive_fact_orientation_block(
                domain=FACT_ORIENTATION_DOMAIN,
                seed=FACT_ORIENTATION_SEED,
                split_id=split_id,
                pair_ordinal=3,
                block_ordinal=block_ordinal,
            )
            block_start = block_ordinal * FACT_ORIENTATION_BLOCK_SIZE
            for within_block_ordinal in range(FACT_ORIENTATION_BLOCK_SIZE):
                analysis = analysis_roots[block_start + within_block_ordinal]
                _digest, _rank, orientation = orientation_by_within[within_block_ordinal]
                fact_values = (0, 1) if orientation == 0 else (1, 0)
                original_fact_value = fact_values[1]
                twins.append(
                    CounterfactualTwinMapping(
                        mapping_ordinal=len(twins),
                        split_id=split_id,
                        split_ordinal=analysis.namespace_ordinal,
                        analysis_global_slot=analysis.global_slot,
                        analysis_surface_code=analysis.surface_code,
                        decisive_decision_index=7,
                        reversal_pair_ordinal=3,
                        distance_bin_lower_bound_tokens=32768,
                        original_decisive_fact_value=original_fact_value,
                        counterfactual_decisive_fact_value=(1 - original_fact_value),
                        utility_oracle_descendants_recomputed=True,
                        all_other_exogenous_nodes_unchanged=True,
                        independent_root=False,
                    )
                )
    if len(pairs) != 8288 or len(twins) != 8288:
        raise AssertionError("analysis/donor pair or twin mapping total drift")
    if any(twin.independent_root for twin in twins):
        raise AssertionError("counterfactual twin consumed an independent root")
    if any(
        twin.decisive_decision_index != 7
        or twin.reversal_pair_ordinal != 3
        or twin.distance_bin_lower_bound_tokens != 32768
        or twin.counterfactual_decisive_fact_value != 1 - twin.original_decisive_fact_value
        or not twin.utility_oracle_descendants_recomputed
        or not twin.all_other_exogenous_nodes_unchanged
        for twin in twins
    ):
        raise AssertionError("counterfactual twin single-node flip contract drift")
    if (
        tuple(twin.mapping_ordinal for twin in twins) != tuple(range(8288))
        or len({(twin.split_id, twin.split_ordinal) for twin in twins}) != 8288
    ):
        raise AssertionError("counterfactual twin mapping is not unique")

    formal_analysis = roots_by_split_role[(FORMAL_SPLIT_ID, ANALYSIS_ROOT_ROLE)]
    formal_donors = roots_by_split_role[(FORMAL_SPLIT_ID, DONOR_ROOT_ROLE)]
    formal_pairs = tuple(pair for pair in pairs if pair.split_id == FORMAL_SPLIT_ID)
    prefixes = tuple(
        FormalCandidatePrefix(
            root_count=root_count,
            analysis_global_slot_start=formal_analysis[0].global_slot,
            analysis_global_slot_stop_exclusive=(formal_analysis[0].global_slot + root_count),
            donor_global_slot_start=formal_donors[0].global_slot,
            donor_global_slot_stop_exclusive=formal_donors[0].global_slot + root_count,
            formal_pair_ordinal_start=0,
            formal_pair_ordinal_stop_exclusive=root_count,
        )
        for root_count in candidates
    )
    _validate_formal_prefixes(prefixes, formal_pairs)
    return SourceRootSurfaceDerivation(
        surface_capacity=surface_capacity,
        affine_multiplier=affine_multiplier,
        affine_offset=affine_offset,
        surface_factor_axes_in_bit_order=SURFACE_FACTOR_AXES_IN_BIT_ORDER,
        surface_factor_bit_decoding=(
            "axis_at_zero_based_index_i_reads_bit_i_with_axis_zero_as_the_least_significant_bit"
        ),
        surface_factor_typed_value_registry=(SURFACE_FACTOR_TYPED_VALUE_REGISTRY),
        namespaces=tuple(allocations),
        roots=tuple(roots),
        analysis_root_count=role_offsets[ANALYSIS_ROOT_ROLE],
        donor_root_count=role_offsets[DONOR_ROOT_ROLE],
        analysis_donor_pairs=tuple(pairs),
        counterfactual_twin_mappings=tuple(twins),
        formal_candidate_prefixes=prefixes,
    )


def derive_source_evaluation_design(
    *,
    action_order: Sequence[str] = ACTION_ORDER,
    distance_bins: Sequence[int] = EVALUATION_DISTANCE_BINS,
    fact_values: Sequence[int] = EVALUATION_FACT_VALUES,
    fact_zero_utility_vector: Sequence[int] = FACT_ZERO_UTILITY_VECTOR,
    fact_one_utility_vector: Sequence[int] = FACT_ONE_UTILITY_VECTOR,
    invalid_generated_action_id: str = INVALID_GENERATED_ACTION_ID,
) -> SourceEvaluationDesign:
    """Construct the exact closed registry and four fact-reversal pairs."""

    actions = tuple(action_order)
    distances = tuple(distance_bins)
    facts = tuple(fact_values)
    fact_zero = tuple(fact_zero_utility_vector)
    fact_one = tuple(fact_one_utility_vector)
    if any(type(action) is not str or not action for action in actions):
        raise TypeError("action order must contain nonempty text")
    if actions != ACTION_ORDER:
        raise ValueError("action registry order drift")
    if invalid_generated_action_id != INVALID_GENERATED_ACTION_ID:
        raise ValueError("invalid generated-action id drift")
    if invalid_generated_action_id in actions:
        raise ValueError("invalid generated action entered the action registry")
    if any(type(distance) is not int for distance in distances):
        raise TypeError("distance bins must be exact integers")
    if distances != EVALUATION_DISTANCE_BINS:
        raise ValueError("evaluation distance-bin order drift")
    if any(type(fact) is not int for fact in facts):
        raise TypeError("fact values must be exact integers")
    if facts != EVALUATION_FACT_VALUES:
        raise ValueError("evaluation fact-value order drift")
    for vector, expected, label in (
        (fact_zero, FACT_ZERO_UTILITY_VECTOR, "fact-zero utility vector"),
        (fact_one, FACT_ONE_UTILITY_VECTOR, "fact-one utility vector"),
    ):
        if any(type(utility) is not int for utility in vector):
            raise TypeError(f"{label} must contain exact integers")
        if vector != expected or len(vector) != len(actions):
            raise ValueError(f"{label} drift")

    utilities = {0: fact_zero, 1: fact_one}
    slots = tuple(
        SourceEvaluationSlot(
            slot_ordinal=pair_ordinal * len(facts) + fact_ordinal,
            reversal_pair_ordinal=pair_ordinal,
            distance_bin_tokens=distance,
            fact_value=fact,
            utility_vector=utilities[fact],
        )
        for pair_ordinal, distance in enumerate(distances)
        for fact_ordinal, fact in enumerate(facts)
    )
    if len(slots) != 8:
        raise AssertionError("evaluation slot count drift")
    for pair_ordinal in range(4):
        pair = tuple(slot for slot in slots if slot.reversal_pair_ordinal == pair_ordinal)
        if tuple(slot.fact_value for slot in pair) != (0, 1):
            raise AssertionError("fact-reversal pair is incomplete or reordered")
        if len({slot.distance_bin_tokens for slot in pair}) != 1:
            raise AssertionError("fact-reversal pair crosses distance bins")
    return SourceEvaluationDesign(
        action_order=actions,
        invalid_generated_action_id=invalid_generated_action_id,
        invalid_generated_action_in_registry=False,
        slots=slots,
    )


def derive_exact_synthetic_planning_generator(
    *,
    reference_success_probability: Fraction = REFERENCE_SUCCESS_PROBABILITY,
    comparator_success_probabilities: Sequence[Fraction] = ((COMPARATOR_SUCCESS_PROBABILITY,) * COMPARATOR_COUNT),
    success_utility: int = SUCCESS_UTILITY,
    failure_utility: int = FAILURE_UTILITY,
) -> SyntheticPlanningGeneratorDerivation:
    """Enumerate a single generic decision's 512 exact nine-arm atoms.

    The scalar utility vector has no cross-decision latent or temporal
    interpretation.  In particular, its ``99/50`` contrast variance is not a
    root-mean tuple witness.
    """

    _require_probability(reference_success_probability, "reference probability")
    comparator_probabilities = tuple(comparator_success_probabilities)
    if any(type(value) is not Fraction for value in comparator_probabilities):
        raise TypeError("comparator probabilities must be Fractions")
    for index, probability in enumerate(comparator_probabilities):
        _require_probability(probability, f"comparator probability {index}")
    if reference_success_probability != REFERENCE_SUCCESS_PROBABILITY:
        raise ValueError("reference success probability drift")
    if comparator_probabilities != (COMPARATOR_SUCCESS_PROBABILITY,) * COMPARATOR_COUNT:
        raise ValueError("comparator success probabilities drift")
    _require_exact_int(success_utility, "success_utility")
    _require_exact_int(failure_utility, "failure_utility")
    if (success_utility, failure_utility) != (SUCCESS_UTILITY, FAILURE_UTILITY):
        raise ValueError("success/failure utility mapping drift")

    probabilities = (reference_success_probability, *comparator_probabilities)
    node_count = len(probabilities)
    atom_count = 1 << node_count
    atoms: list[SyntheticPlanningAtom] = []
    for atom_ordinal in range(atom_count):
        successes = tuple(
            bool((atom_ordinal >> (node_count - node_ordinal - 1)) & 1) for node_ordinal in range(node_count)
        )
        probability = math.prod(
            source_probability if success else 1 - source_probability
            for success, source_probability in zip(successes, probabilities, strict=True)
        )
        utility_vector = tuple(success_utility if success else failure_utility for success in successes)
        contrast_vector = tuple(utility_vector[0] - comparator_utility for comparator_utility in utility_vector[1:])
        atoms.append(
            SyntheticPlanningAtom(
                atom_ordinal=atom_ordinal,
                reference_success=successes[0],
                comparator_successes=successes[1:],
                probability=probability,
                utility_vector=utility_vector,
                contrast_vector=contrast_vector,
            )
        )
    if atom_count != 512 or len(atoms) != 512:
        raise AssertionError("full planning tensor atom count drift")
    if sum((atom.probability for atom in atoms), Fraction(0, 1)) != 1:
        raise AssertionError("planning atom probabilities do not sum exactly to one")
    if len({(atom.reference_success, atom.comparator_successes) for atom in atoms}) != 512:
        raise AssertionError("planning atom state inventory contains duplicates")

    contrast_values = tuple(
        tuple(atom.contrast_vector[contrast] for atom in atoms) for contrast in range(COMPARATOR_COUNT)
    )
    masses = tuple(atom.probability for atom in atoms)
    means = tuple(_weighted_mean(values, masses) for values in contrast_values)
    covariance_matrix = tuple(
        tuple(_weighted_covariance(left, right, masses) for right in contrast_values) for left in contrast_values
    )
    correlation_matrix = tuple(
        tuple(_weighted_correlation(left, right, masses) for right in contrast_values) for left in contrast_values
    )
    expected_covariance_matrix = tuple(
        tuple(Fraction(99, 50) if row == column else Fraction(99, 100) for column in range(COMPARATOR_COUNT))
        for row in range(COMPARATOR_COUNT)
    )
    expected_correlation_matrix = tuple(
        tuple(Fraction(1, 1) if row == column else Fraction(1, 2) for column in range(COMPARATOR_COUNT))
        for row in range(COMPARATOR_COUNT)
    )
    if means != (Fraction(1, 5),) * COMPARATOR_COUNT:
        raise AssertionError("planning contrast mean derivation drift")
    if covariance_matrix != expected_covariance_matrix:
        raise AssertionError("planning contrast covariance derivation drift")
    if correlation_matrix != expected_correlation_matrix:
        raise AssertionError("planning contrast correlation derivation drift")
    off_diagonal = correlation_matrix[0][1]
    repeated_eigenvalue = correlation_matrix[0][0] - off_diagonal
    shared_eigenvalue = correlation_matrix[0][0] + (COMPARATOR_COUNT - 1) * off_diagonal
    eigenvalues = (
        ExactEigenvalueMultiplicity(repeated_eigenvalue, COMPARATOR_COUNT - 1),
        ExactEigenvalueMultiplicity(shared_eigenvalue, 1),
    )
    if eigenvalues != (
        ExactEigenvalueMultiplicity(Fraction(1, 2), 7),
        ExactEigenvalueMultiplicity(Fraction(9, 2), 1),
    ):
        raise AssertionError("planning correlation PSD eigenvalue derivation drift")
    if any(item.eigenvalue < 0 for item in eigenvalues):
        raise AssertionError("planning correlation matrix is not positive semidefinite")

    return SyntheticPlanningGeneratorDerivation(
        reference_success_probability=reference_success_probability,
        comparator_success_probabilities=comparator_probabilities,
        atoms=tuple(atoms),
        contrast_means=means,
        contrast_covariance_matrix=covariance_matrix,
        contrast_correlation_matrix=correlation_matrix,
        correlation_eigenvalues=eigenvalues,
    )


def derive_root_fact_orientation_inventory(
    *,
    root_layout: SourceRootSurfaceDerivation | None = None,
    ranking_domain: str = FACT_ORIENTATION_DOMAIN,
    ranking_seed: int = FACT_ORIENTATION_SEED,
    reversal_pair_count: int = REVERSAL_PAIR_COUNT,
    block_size: int = FACT_ORIENTATION_BLOCK_SIZE,
    formal_candidate_root_counts: Sequence[int] | None = None,
) -> RootFactOrientationDerivation:
    """Rank each analysis-root/pair inside 32-root blocks and orient 16/16.

    Only ``domain, seed, split, pair, block, within-block`` enter the ranking
    digest.  Candidate N is consumed later solely to audit already-committed
    formal prefixes; arms, models, outputs, power, and CUDA are absent.
    """

    _require_nonempty_text(ranking_domain, "ranking_domain")
    _require_exact_int(ranking_seed, "ranking_seed")
    _require_positive_int(reversal_pair_count, "reversal_pair_count")
    _require_positive_int(block_size, "block_size")
    if ranking_domain != FACT_ORIENTATION_DOMAIN:
        raise ValueError("fact-orientation ranking domain drift")
    if ranking_seed != FACT_ORIENTATION_SEED:
        raise ValueError("fact-orientation ranking seed drift")
    if reversal_pair_count != REVERSAL_PAIR_COUNT:
        raise ValueError("fact-orientation reversal-pair count drift")
    if block_size != FACT_ORIENTATION_BLOCK_SIZE:
        raise ValueError("fact-orientation block size drift")

    layout = derive_source_root_surface_layout() if root_layout is None else root_layout
    if type(layout) is not SourceRootSurfaceDerivation:
        raise TypeError("root_layout must be a SourceRootSurfaceDerivation")
    _validate_orientation_root_layout(layout)
    candidates = (
        derive_formal_candidate_root_counts()
        if formal_candidate_root_counts is None
        else tuple(formal_candidate_root_counts)
    )
    if any(type(value) is not int for value in candidates):
        raise TypeError("formal candidate root counts must be exact integers")
    if candidates != derive_formal_candidate_root_counts():
        raise ValueError("formal candidate root-count grid drift")

    expected_split_counts = (
        (DEVELOPMENT_SPLIT_ID, 32),
        (QUALIFICATION_SPLIT_ID, 64),
        (FORMAL_SPLIT_ID, 8192),
    )
    analysis_by_split = {
        split_id: tuple(
            root for root in layout.roots if root.root_role == ANALYSIS_ROOT_ROLE and root.split_id == split_id
        )
        for split_id, _ in expected_split_counts
    }
    assignments: list[RootFactOrientationAssignment] = []
    commitments: list[FactOrientationBlockCommitment] = []
    for split_id, split_root_count in expected_split_counts:
        split_roots = analysis_by_split[split_id]
        if len(split_roots) != split_root_count or split_root_count % block_size:
            raise AssertionError("fact-orientation split/block cardinality drift")
        for pair_ordinal in range(reversal_pair_count):
            for block_ordinal in range(split_root_count // block_size):
                root_start = block_ordinal * block_size
                block_roots = split_roots[root_start : root_start + block_size]
                orientation_by_within = _derive_fact_orientation_block(
                    domain=ranking_domain,
                    seed=ranking_seed,
                    split_id=split_id,
                    pair_ordinal=pair_ordinal,
                    block_ordinal=block_ordinal,
                )
                block_assignments: list[RootFactOrientationAssignment] = []
                for within_block_ordinal, root in enumerate(block_roots):
                    digest_bytes, rank, orientation = orientation_by_within[within_block_ordinal]
                    assignment = RootFactOrientationAssignment(
                        assignment_ordinal=len(assignments),
                        split_id=split_id,
                        analysis_root_ordinal=root.namespace_ordinal,
                        analysis_global_slot=root.global_slot,
                        analysis_surface_code=root.surface_code,
                        reversal_pair_ordinal=pair_ordinal,
                        block_ordinal=block_ordinal,
                        within_block_ordinal=within_block_ordinal,
                        ranking_digest_sha256=digest_bytes.hex(),
                        rank_within_block=rank,
                        orientation=orientation,
                        fact_values_by_decision_position=((0, 1) if orientation == 0 else (1, 0)),
                    )
                    assignments.append(assignment)
                    block_assignments.append(assignment)
                orientation_counts = tuple(
                    sum(item.orientation == orientation for item in block_assignments) for orientation in (0, 1)
                )
                if orientation_counts != (
                    FACT_ORIENTATION_COUNT_PER_VALUE,
                    FACT_ORIENTATION_COUNT_PER_VALUE,
                ):
                    raise AssertionError("fact-orientation block lost exact 16/16 balance")
                block_digest = canonical_mapping_digest(
                    {
                        "ranking_domain": ranking_domain,
                        "ranking_seed": ranking_seed,
                        "split_id": split_id,
                        "reversal_pair_ordinal": pair_ordinal,
                        "block_ordinal": block_ordinal,
                        "assignments": [_fact_orientation_assignment_mapping(item) for item in block_assignments],
                    }
                )
                commitments.append(
                    FactOrientationBlockCommitment(
                        commitment_ordinal=len(commitments),
                        split_id=split_id,
                        reversal_pair_ordinal=pair_ordinal,
                        block_ordinal=block_ordinal,
                        analysis_root_ordinal_start=root_start,
                        analysis_root_ordinal_stop_exclusive=root_start + block_size,
                        orientation_zero_count=orientation_counts[0],
                        orientation_one_count=orientation_counts[1],
                        assignment_digest_sha256=block_digest,
                    )
                )

    if len(assignments) != 8288 * reversal_pair_count:
        raise AssertionError("root/pair fact-orientation assignment total drift")
    expected_commitment_count = (1 + 2 + 256) * reversal_pair_count
    if len(commitments) != expected_commitment_count:
        raise AssertionError("fact-orientation block commitment total drift")
    assignment_inventory_digest = canonical_mapping_digest(
        {
            "ranking_domain": ranking_domain,
            "ranking_seed": ranking_seed,
            "ranking_payload_contract": "ascii_pipe_join_v1",
            "ranking_counter_fields": list(FACT_ORIENTATION_RANKING_COUNTER_FIELDS),
            "ranking_tie_break_fields": [
                "sha256_digest_bytes",
                "within_block_ordinal",
            ],
            "block_size": block_size,
            "orientation_count_per_value_per_block": (FACT_ORIENTATION_COUNT_PER_VALUE),
            "block_commitments": [
                {
                    "commitment_ordinal": item.commitment_ordinal,
                    "split_id": item.split_id,
                    "reversal_pair_ordinal": item.reversal_pair_ordinal,
                    "block_ordinal": item.block_ordinal,
                    "analysis_root_ordinal_start": (item.analysis_root_ordinal_start),
                    "analysis_root_ordinal_stop_exclusive": (item.analysis_root_ordinal_stop_exclusive),
                    "assignment_digest_sha256": item.assignment_digest_sha256,
                }
                for item in commitments
            ],
        }
    )

    formal_assignments_by_pair = {
        pair_ordinal: tuple(
            item
            for item in assignments
            if item.split_id == FORMAL_SPLIT_ID and item.reversal_pair_ordinal == pair_ordinal
        )
        for pair_ordinal in range(reversal_pair_count)
    }
    balances: list[FormalCandidateFactPositionBalance] = []
    for root_count in candidates:
        for pair_ordinal in range(reversal_pair_count):
            prefix = formal_assignments_by_pair[pair_ordinal][:root_count]
            if len(prefix) != root_count or tuple(item.analysis_root_ordinal for item in prefix) != tuple(
                range(root_count)
            ):
                raise AssertionError("formal fact-orientation inventory is not a prefix")
            for decision_position in (0, 1):
                fact_zero_count = sum(item.fact_values_by_decision_position[decision_position] == 0 for item in prefix)
                fact_one_count = root_count - fact_zero_count
                if fact_zero_count != root_count // 2 or fact_one_count != root_count // 2:
                    raise AssertionError("formal candidate fact-position balance drift")
                balances.append(
                    FormalCandidateFactPositionBalance(
                        formal_root_count=root_count,
                        reversal_pair_ordinal=pair_ordinal,
                        decision_position=decision_position,
                        fact_zero_count=fact_zero_count,
                        fact_one_count=fact_one_count,
                    )
                )
    return RootFactOrientationDerivation(
        ranking_domain=ranking_domain,
        ranking_seed=ranking_seed,
        ranking_hash_algorithm="sha256",
        ranking_payload_contract="ascii_pipe_join_v1",
        ranking_counter_fields=FACT_ORIENTATION_RANKING_COUNTER_FIELDS,
        ranking_tie_break_fields=(
            "sha256_digest_bytes",
            "within_block_ordinal",
        ),
        ranking_excluded_fields=FACT_ORIENTATION_RANKING_EXCLUDED_FIELDS,
        analysis_root_count=8288,
        reversal_pair_count=reversal_pair_count,
        block_size=block_size,
        orientation_count_per_value_per_block=(FACT_ORIENTATION_COUNT_PER_VALUE),
        assignments=tuple(assignments),
        block_commitments=tuple(commitments),
        assignment_inventory_digest_sha256=assignment_inventory_digest,
        formal_candidate_position_balances=tuple(balances),
    )


def canonical_mapping_digest(value: Mapping[str, object]) -> str:
    """Hash an exact JSON mapping using sorted UTF-8 JSON plus one newline."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical digest input must be a mapping")
    normalized = _normalize_exact_json(value, path="$")
    payload = (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_surface_factor_registry() -> None:
    registry = SURFACE_FACTOR_TYPED_VALUE_REGISTRY
    if len(registry) != 15 or SURFACE_CAPACITY != 1 << len(registry):
        raise AssertionError("surface factor registry/capacity cardinality drift")
    if any(type(item) is not SurfaceFactorTypedLevel for item in registry):
        raise TypeError("surface factor registry contains an invalid entry")
    axes = tuple(item.axis_id for item in registry)
    if axes != SURFACE_FACTOR_AXES_IN_BIT_ORDER or len(set(axes)) != len(axes):
        raise AssertionError("surface factor axis order or uniqueness drift")
    for index, item in enumerate(registry):
        _require_nonempty_text(item.axis_id, f"surface factor {index} axis_id")
        _require_nonempty_text(item.value_zero, f"surface factor {index} value_zero")
        _require_nonempty_text(item.value_one, f"surface factor {index} value_one")
        if item.value_zero == item.value_one:
            raise ValueError("surface factor typed levels must be distinct")


def _fact_orientation_ranking_payload(
    *,
    domain: str,
    seed: int,
    split_id: str,
    pair_ordinal: int,
    block_ordinal: int,
    within_block_ordinal: int,
) -> bytes:
    components = (
        domain,
        str(seed),
        split_id,
        str(pair_ordinal),
        str(block_ordinal),
        str(within_block_ordinal),
    )
    if any("|" in component for component in components):
        raise ValueError("fact-orientation ranking component contains delimiter")
    try:
        return "|".join(components).encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("fact-orientation ranking payload must be ASCII") from error


def _derive_fact_orientation_block(
    *,
    domain: str,
    seed: int,
    split_id: str,
    pair_ordinal: int,
    block_ordinal: int,
) -> tuple[tuple[bytes, int, int], ...]:
    if domain != FACT_ORIENTATION_DOMAIN or seed != FACT_ORIENTATION_SEED:
        raise ValueError("fact-orientation domain or seed drift")
    split_counts = {
        DEVELOPMENT_SPLIT_ID: 32,
        QUALIFICATION_SPLIT_ID: 64,
        FORMAL_SPLIT_ID: 8192,
    }
    if split_id not in split_counts:
        raise ValueError("fact-orientation split id drift")
    _require_exact_int(pair_ordinal, "pair_ordinal")
    _require_exact_int(block_ordinal, "block_ordinal")
    if not 0 <= pair_ordinal < REVERSAL_PAIR_COUNT:
        raise ValueError("fact-orientation pair ordinal leaves the registry")
    if not 0 <= block_ordinal < split_counts[split_id] // FACT_ORIENTATION_BLOCK_SIZE:
        raise ValueError("fact-orientation block ordinal leaves the split")
    ranked = []
    for within_block_ordinal in range(FACT_ORIENTATION_BLOCK_SIZE):
        payload = _fact_orientation_ranking_payload(
            domain=domain,
            seed=seed,
            split_id=split_id,
            pair_ordinal=pair_ordinal,
            block_ordinal=block_ordinal,
            within_block_ordinal=within_block_ordinal,
        )
        ranked.append((hashlib.sha256(payload).digest(), within_block_ordinal))
    ranked.sort(key=lambda item: (item[0], item[1]))
    rank_by_within = {within: (digest, rank) for rank, (digest, within) in enumerate(ranked)}
    result = tuple(
        (
            rank_by_within[within][0],
            rank_by_within[within][1],
            int(rank_by_within[within][1] >= FACT_ORIENTATION_COUNT_PER_VALUE),
        )
        for within in range(FACT_ORIENTATION_BLOCK_SIZE)
    )
    if tuple(sorted(item[1] for item in result)) != tuple(range(FACT_ORIENTATION_BLOCK_SIZE)):
        raise AssertionError("fact-orientation block ranks are not a permutation")
    if tuple(sum(item[2] == value for item in result) for value in (0, 1)) != (
        FACT_ORIENTATION_COUNT_PER_VALUE,
        FACT_ORIENTATION_COUNT_PER_VALUE,
    ):
        raise AssertionError("fact-orientation block is not exactly balanced")
    return result


def _fact_orientation_assignment_mapping(
    assignment: RootFactOrientationAssignment,
) -> dict[str, object]:
    return {
        "assignment_ordinal": assignment.assignment_ordinal,
        "split_id": assignment.split_id,
        "analysis_root_ordinal": assignment.analysis_root_ordinal,
        "analysis_global_slot": assignment.analysis_global_slot,
        "analysis_surface_code": assignment.analysis_surface_code,
        "reversal_pair_ordinal": assignment.reversal_pair_ordinal,
        "block_ordinal": assignment.block_ordinal,
        "within_block_ordinal": assignment.within_block_ordinal,
        "ranking_digest_sha256": assignment.ranking_digest_sha256,
        "rank_within_block": assignment.rank_within_block,
        "orientation": assignment.orientation,
        "fact_values_by_decision_position": list(assignment.fact_values_by_decision_position),
    }


def _validate_orientation_root_layout(layout: SourceRootSurfaceDerivation) -> None:
    if (
        layout.surface_capacity != SURFACE_CAPACITY
        or layout.affine_multiplier != SURFACE_AFFINE_MULTIPLIER
        or layout.affine_offset != SURFACE_AFFINE_OFFSET
        or layout.surface_factor_axes_in_bit_order != SURFACE_FACTOR_AXES_IN_BIT_ORDER
        or layout.surface_factor_typed_value_registry != SURFACE_FACTOR_TYPED_VALUE_REGISTRY
        or layout.analysis_root_count != 8288
        or layout.donor_root_count != 8288
        or len(layout.roots) != 16576
    ):
        raise ValueError("fact-orientation root layout primitive drift")
    expected_analysis = (
        (DEVELOPMENT_SPLIT_ID, 32, 0),
        (QUALIFICATION_SPLIT_ID, 64, 64),
        (FORMAL_SPLIT_ID, 8192, 192),
    )
    for split_id, count, global_start in expected_analysis:
        split_roots = tuple(
            root for root in layout.roots if root.root_role == ANALYSIS_ROOT_ROLE and root.split_id == split_id
        )
        if len(split_roots) != count:
            raise ValueError("fact-orientation analysis split count drift")
        for ordinal, root in enumerate(split_roots):
            if type(root) is not SourceRootSurface:
                raise TypeError("fact-orientation root has an invalid type")
            expected_code = (
                SURFACE_AFFINE_MULTIPLIER * (global_start + ordinal) + SURFACE_AFFINE_OFFSET
            ) % SURFACE_CAPACITY
            expected_bits = tuple(
                (expected_code >> index) & 1 for index in range(len(SURFACE_FACTOR_TYPED_VALUE_REGISTRY))
            )
            expected_values = tuple(
                level.value_zero if bit == 0 else level.value_one
                for level, bit in zip(
                    SURFACE_FACTOR_TYPED_VALUE_REGISTRY,
                    expected_bits,
                    strict=True,
                )
            )
            if (
                root.namespace_ordinal != ordinal
                or root.global_slot != global_start + ordinal
                or root.surface_code != expected_code
                or root.factor_bits != expected_bits
                or root.typed_blueprint_values != expected_values
            ):
                raise ValueError("fact-orientation analysis root mapping drift")

    twins = layout.counterfactual_twin_mappings
    if len(twins) != 8288 or any(type(item) is not CounterfactualTwinMapping for item in twins):
        raise ValueError("counterfactual twin inventory drift")
    twins_by_key = {(item.split_id, item.split_ordinal): item for item in twins}
    if len(twins_by_key) != 8288:
        raise ValueError("counterfactual twin keys are not unique")
    for split_id, count, _global_start in expected_analysis:
        for block_ordinal in range(count // FACT_ORIENTATION_BLOCK_SIZE):
            block = _derive_fact_orientation_block(
                domain=FACT_ORIENTATION_DOMAIN,
                seed=FACT_ORIENTATION_SEED,
                split_id=split_id,
                pair_ordinal=3,
                block_ordinal=block_ordinal,
            )
            for within_block_ordinal, (_digest, _rank, orientation) in enumerate(block):
                ordinal = block_ordinal * FACT_ORIENTATION_BLOCK_SIZE + within_block_ordinal
                twin = twins_by_key[(split_id, ordinal)]
                expected_original = 1 if orientation == 0 else 0
                if (
                    twin.decisive_decision_index != 7
                    or twin.reversal_pair_ordinal != 3
                    or twin.distance_bin_lower_bound_tokens != 32768
                    or twin.original_decisive_fact_value != expected_original
                    or twin.counterfactual_decisive_fact_value != 1 - expected_original
                    or not twin.utility_oracle_descendants_recomputed
                    or not twin.all_other_exogenous_nodes_unchanged
                    or twin.independent_root
                ):
                    raise ValueError("counterfactual twin fact flip drift")


def _validate_formal_prefixes(
    prefixes: tuple[FormalCandidatePrefix, ...],
    formal_pairs: tuple[AnalysisDonorRootPair, ...],
) -> None:
    if not prefixes or len(formal_pairs) != 8192:
        raise AssertionError("formal prefix or pair inventory drift")
    previous_count = 0
    first_analysis_slot = formal_pairs[0].analysis_global_slot
    first_donor_slot = formal_pairs[0].donor_global_slot
    for prefix in prefixes:
        if prefix.root_count <= previous_count:
            raise AssertionError("formal candidates are not strict prefixes")
        expected_pairs = formal_pairs[: prefix.root_count]
        if len(expected_pairs) != prefix.root_count:
            raise AssertionError("formal candidate exceeds root inventory")
        if tuple(pair.split_ordinal for pair in expected_pairs) != tuple(range(prefix.root_count)):
            raise AssertionError("formal donor mapping is not same-ordinal one-to-one")
        if tuple(pair.analysis_global_slot for pair in expected_pairs) != tuple(
            range(first_analysis_slot, first_analysis_slot + prefix.root_count)
        ):
            raise AssertionError("formal analysis roots do not form a prefix")
        if tuple(pair.donor_global_slot for pair in expected_pairs) != tuple(
            range(first_donor_slot, first_donor_slot + prefix.root_count)
        ):
            raise AssertionError("formal donor roots do not form a prefix")
        if (
            prefix.analysis_global_slot_start != first_analysis_slot
            or prefix.analysis_global_slot_stop_exclusive != first_analysis_slot + prefix.root_count
            or prefix.donor_global_slot_start != first_donor_slot
            or prefix.donor_global_slot_stop_exclusive != first_donor_slot + prefix.root_count
            or prefix.formal_pair_ordinal_start != 0
            or prefix.formal_pair_ordinal_stop_exclusive != prefix.root_count
        ):
            raise AssertionError("formal prefix boundary metadata drift")
        previous_count = prefix.root_count


def _weighted_mean(
    values: Sequence[int],
    masses: Sequence[Fraction],
) -> Fraction:
    if len(values) != len(masses) or not values:
        raise ValueError("weighted-mean input lengths drift")
    return sum(
        (mass * value for value, mass in zip(values, masses, strict=True)),
        Fraction(0, 1),
    )


def _weighted_covariance(
    left: Sequence[int],
    right: Sequence[int],
    masses: Sequence[Fraction],
) -> Fraction:
    if len(left) != len(right) or len(left) != len(masses) or not left:
        raise ValueError("weighted-covariance input lengths drift")
    left_mean = _weighted_mean(left, masses)
    right_mean = _weighted_mean(right, masses)
    return sum(
        (
            mass * (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value, mass in zip(left, right, masses, strict=True)
        ),
        Fraction(0, 1),
    )


def _weighted_correlation(
    left: Sequence[int],
    right: Sequence[int],
    masses: Sequence[Fraction],
) -> Fraction:
    covariance = _weighted_covariance(left, right, masses)
    left_variance = _weighted_covariance(left, left, masses)
    right_variance = _weighted_covariance(right, right, masses)
    if left_variance <= 0 or right_variance <= 0:
        raise ValueError("correlation requires positive variance")
    if left_variance != right_variance:
        raise ValueError("exact rational correlation requires equal variances")
    return covariance / left_variance


def _normalize_exact_json(value: object, *, path: str) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"canonical mapping key at {path} must be text")
            normalized[key] = _normalize_exact_json(item, path=f"{path}.{key}")
        return normalized
    if type(value) in (list, tuple):
        return [_normalize_exact_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"canonical mapping value at {path} must be exact JSON data, not {type(value).__name__}")


def _require_probability(value: object, label: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"{label} must be a Fraction")
    if not 0 < value < 1:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return value


def _require_nonempty_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    if not value:
        raise ValueError(f"{label} must be nonempty")
    return value


def _require_exact_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    _require_exact_int(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


__all__ = [
    "ACTION_ORDER",
    "ANALYSIS_ROOT_ROLE",
    "AnalysisDonorRootPair",
    "COMPARATOR_COUNT",
    "COMPARATOR_SUCCESS_PROBABILITY",
    "CounterfactualTwinMapping",
    "DEVELOPMENT_SPLIT_ID",
    "DONOR_ROOT_ROLE",
    "EVALUATION_DISTANCE_BINS",
    "EVALUATION_FACT_VALUES",
    "ExactEigenvalueMultiplicity",
    "FACT_ORIENTATION_BLOCK_SIZE",
    "FACT_ORIENTATION_COUNT_PER_VALUE",
    "FACT_ORIENTATION_DOMAIN",
    "FACT_ORIENTATION_RANKING_COUNTER_FIELDS",
    "FACT_ORIENTATION_RANKING_EXCLUDED_FIELDS",
    "FACT_ORIENTATION_SEED",
    "FACT_ONE_UTILITY_VECTOR",
    "FACT_ZERO_UTILITY_VECTOR",
    "FAILURE_UTILITY",
    "FactOrientationBlockCommitment",
    "FORMAL_SPLIT_ID",
    "FormalCandidateFactPositionBalance",
    "FormalCandidatePrefix",
    "INVALID_GENERATED_ACTION_ID",
    "QUALIFICATION_SPLIT_ID",
    "REFERENCE_SUCCESS_PROBABILITY",
    "ROOT_NAMESPACE_SPECS",
    "REVERSAL_PAIR_COUNT",
    "RootFactOrientationAssignment",
    "RootFactOrientationDerivation",
    "SUCCESS_UTILITY",
    "SURFACE_AFFINE_MULTIPLIER",
    "SURFACE_AFFINE_OFFSET",
    "SURFACE_CAPACITY",
    "SURFACE_FACTOR_AXES_IN_BIT_ORDER",
    "SURFACE_FACTOR_TYPED_VALUE_REGISTRY",
    "SourceEvaluationDesign",
    "SourceEvaluationSlot",
    "SourceRootNamespaceAllocation",
    "SourceRootNamespaceSpec",
    "SourceRootSurface",
    "SourceRootSurfaceDerivation",
    "SurfaceFactorTypedLevel",
    "SyntheticPlanningAtom",
    "SyntheticPlanningGeneratorDerivation",
    "canonical_mapping_digest",
    "derive_exact_synthetic_planning_generator",
    "derive_formal_candidate_root_counts",
    "derive_root_fact_orientation_inventory",
    "derive_source_evaluation_design",
    "derive_source_root_surface_layout",
]
