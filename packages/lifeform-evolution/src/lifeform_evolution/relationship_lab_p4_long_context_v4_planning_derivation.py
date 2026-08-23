"""Pure derivations for the P4.7 v4 zero-output planning contract.

The functions in this module have no filesystem, model, CUDA, or artifact-ID
surface.  They turn frozen primitives into independently recomputable values;
callers must never feed protocol ``derived_expected`` fields back as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class V4SentinelDerivation:
    """Exact consequences of one shared-reference finite-support sentinel."""

    contrast_support: tuple[Fraction, ...]
    complete_data_mean: Fraction
    root_difference_variance: Fraction
    temporal_pair_correlations: tuple[tuple[Fraction, ...], ...]
    cross_contrast_correlations: tuple[Fraction, ...]


@dataclass(frozen=True)
class V4NecessaryScreenResult:
    """One exact point-gate screen for a candidate formal root count."""

    root_count: int
    minimum_plus_count: int
    power: Fraction
    passed: bool


@dataclass(frozen=True)
class V4CandidateScheduleBlock:
    """One abstract development block; it contains no model or subject data."""

    root_ordinal: int
    session_index: int
    global_block_ordinal: int
    ordered_cell_ids: tuple[str, ...]


def derive_shared_reference_sentinel(
    *,
    state_masses: Sequence[Fraction],
    state_reference_utilities: Sequence[int],
    state_comparator_utilities: Sequence[int],
    arm_count: int,
    decision_count: int,
    utility_domain: Sequence[int],
) -> V4SentinelDerivation:
    """Expand primitive scalars to 9x8 tensors and derive every pair."""

    _require_positive_int(arm_count, "arm_count")
    _require_positive_int(decision_count, "decision_count")
    if arm_count < 2:
        raise ValueError("arm_count must include a reference and comparator")
    masses = tuple(state_masses)
    reference = tuple(state_reference_utilities)
    comparator = tuple(state_comparator_utilities)
    if not masses or len(masses) != len(reference) or len(masses) != len(comparator):
        raise ValueError("sentinel state primitive lengths drift")
    if any(type(item) is not Fraction or item <= 0 for item in masses):
        raise TypeError("sentinel masses must be positive Fractions")
    if sum(masses, Fraction(0, 1)) != 1:
        raise ValueError("sentinel masses must sum exactly to one")
    exact_domain = tuple(utility_domain)
    if not exact_domain or any(type(item) is not int for item in exact_domain):
        raise TypeError("utility domain must contain exact integers")
    if len(set(exact_domain)) != len(exact_domain):
        raise ValueError("utility domain contains duplicates")
    if any(type(item) is not int or item not in exact_domain for item in reference + comparator):
        raise ValueError("sentinel utility leaves the typed domain")

    state_tensors: list[tuple[tuple[int, ...], ...]] = []
    state_contrasts: list[tuple[tuple[int, ...], ...]] = []
    for reference_utility, comparator_utility in zip(reference, comparator, strict=True):
        tensor = (
            (reference_utility,) * decision_count,
            *((comparator_utility,) * decision_count for _ in range(arm_count - 1)),
        )
        if len(tensor) != arm_count or any(len(row) != decision_count for row in tensor):
            raise AssertionError("constructed sentinel tensor shape drift")
        contrasts = tuple(tuple(tensor[0][turn] - arm[turn] for turn in range(decision_count)) for arm in tensor[1:])
        state_tensors.append(tensor)
        state_contrasts.append(contrasts)

    root_means_by_state = tuple(
        tuple(Fraction(sum(row), decision_count) for row in contrasts) for contrasts in state_contrasts
    )
    if any(len(row) != arm_count - 1 for row in root_means_by_state):
        raise AssertionError("constructed sentinel contrast shape drift")
    per_contrast_state_values = tuple(
        tuple(root_means_by_state[state][contrast] for state in range(len(masses))) for contrast in range(arm_count - 1)
    )
    if len(set(per_contrast_state_values)) != 1:
        raise ValueError("sentinel contrasts do not share the registered distribution")
    contrast_support = per_contrast_state_values[0]
    mean = _weighted_mean(contrast_support, masses)
    variance = _weighted_covariance(contrast_support, contrast_support, masses)
    if variance <= 0:
        raise ValueError("sentinel contrast variance must be positive")

    temporal_by_contrast: list[tuple[Fraction, ...]] = []
    for contrast in range(arm_count - 1):
        pair_correlations: list[Fraction] = []
        for left_turn in range(decision_count):
            for right_turn in range(left_turn + 1, decision_count):
                left = tuple(state_contrasts[state][contrast][left_turn] for state in range(len(masses)))
                right = tuple(state_contrasts[state][contrast][right_turn] for state in range(len(masses)))
                pair_correlations.append(_weighted_correlation(left, right, masses))
        temporal_by_contrast.append(tuple(pair_correlations))

    cross_correlations: list[Fraction] = []
    for left_contrast in range(arm_count - 1):
        for right_contrast in range(left_contrast + 1, arm_count - 1):
            cross_correlations.append(
                _weighted_correlation(
                    per_contrast_state_values[left_contrast],
                    per_contrast_state_values[right_contrast],
                    masses,
                )
            )
    return V4SentinelDerivation(
        contrast_support=contrast_support,
        complete_data_mean=mean,
        root_difference_variance=variance,
        temporal_pair_correlations=tuple(temporal_by_contrast),
        cross_contrast_correlations=tuple(cross_correlations),
    )


def derive_candidate_root_counts(*, first: int, step: int, last_inclusive: int) -> tuple[int, ...]:
    """Return the exact ordered candidate grid without assuming power monotonicity."""

    _require_positive_int(first, "first")
    _require_positive_int(step, "step")
    _require_positive_int(last_inclusive, "last_inclusive")
    if first > last_inclusive:
        raise ValueError("candidate root-count range is empty")
    if (last_inclusive - first) % step != 0:
        raise ValueError("candidate root-count range has a partial final block")
    return tuple(range(first, last_inclusive + 1, step))


def derive_necessary_point_screens(
    *,
    candidate_root_counts: Sequence[int],
    mass_at_plus_two: Fraction,
    practical_gate: Fraction,
    required_power: Fraction,
) -> tuple[V4NecessaryScreenResult, ...]:
    """Evaluate every candidate independently under the exact two-point sentinel."""

    candidates = tuple(candidate_root_counts)
    if not candidates or any(type(item) is not int or item <= 0 for item in candidates):
        raise ValueError("candidate root counts must be positive exact integers")
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("candidate root counts must be unique and ascending")
    for value, label in (
        (mass_at_plus_two, "mass_at_plus_two"),
        (practical_gate, "practical_gate"),
        (required_power, "required_power"),
    ):
        if type(value) is not Fraction:
            raise TypeError(f"{label} must be a Fraction")
    if not 0 < mass_at_plus_two < 1:
        raise ValueError("mass_at_plus_two must lie strictly inside zero and one")
    if not Fraction(-2, 1) < practical_gate < Fraction(2, 1):
        raise ValueError("practical gate must lie inside sentinel support")
    if not 0 < required_power < 1:
        raise ValueError("required power must lie strictly inside zero and one")

    threshold_fraction = (practical_gate + 2) / 4
    results: list[V4NecessaryScreenResult] = []
    for root_count in candidates:
        minimum_plus_count = _ceil_fraction(root_count * threshold_fraction)
        power = exact_binomial_upper_tail(
            trials=root_count,
            success_probability=mass_at_plus_two,
            minimum_successes=minimum_plus_count,
        )
        results.append(
            V4NecessaryScreenResult(
                root_count=root_count,
                minimum_plus_count=minimum_plus_count,
                power=power,
                passed=power >= required_power,
            )
        )
    return tuple(results)


def derive_first_passing_necessary_screen(
    *,
    candidate_root_counts: Sequence[int],
    mass_at_plus_two: Fraction,
    practical_gate: Fraction,
    required_power: Fraction,
) -> tuple[V4NecessaryScreenResult, ...]:
    """Evaluate ascending candidates only through the first exact screen pass."""

    candidates = tuple(candidate_root_counts)
    evaluated: list[V4NecessaryScreenResult] = []
    for root_count in candidates:
        current = derive_necessary_point_screens(
            candidate_root_counts=(root_count,),
            mass_at_plus_two=mass_at_plus_two,
            practical_gate=practical_gate,
            required_power=required_power,
        )[0]
        evaluated.append(current)
        if current.passed:
            break
    return tuple(evaluated)


def exact_binomial_upper_tail(
    *,
    trials: int,
    success_probability: Fraction,
    minimum_successes: int,
) -> Fraction:
    """Return an exact rational binomial upper tail."""

    _require_positive_int(trials, "trials")
    if type(success_probability) is not Fraction or not 0 < success_probability < 1:
        raise ValueError("success_probability must be a strict Fraction probability")
    if type(minimum_successes) is not int:
        raise TypeError("minimum_successes must be an exact integer")
    if minimum_successes <= 0:
        return Fraction(1, 1)
    if minimum_successes > trials:
        return Fraction(0, 1)
    numerator_probability = success_probability.numerator
    denominator_probability = success_probability.denominator
    failure_numerator = denominator_probability - numerator_probability
    common_denominator = denominator_probability**trials
    successes = minimum_successes
    term = (
        math.comb(trials, successes)
        * numerator_probability**successes
        * failure_numerator ** (trials - successes)
    )
    numerator = term
    while successes < trials:
        recurrence_numerator = term * (trials - successes) * numerator_probability
        recurrence_denominator = (successes + 1) * failure_numerator
        if recurrence_numerator % recurrence_denominator:
            raise ArithmeticError("exact binomial recurrence lost integrality")
        term = recurrence_numerator // recurrence_denominator
        numerator += term
        successes += 1
    return Fraction(numerator, common_denominator)


def derive_candidate_cells(
    *,
    baseline_families: Sequence[str],
    candidate_indices: Sequence[int],
) -> tuple[str, ...]:
    """Build the ordered family-by-index Cartesian cell inventory."""

    families = tuple(baseline_families)
    indices = tuple(candidate_indices)
    if not families or any(type(item) is not str or not item for item in families):
        raise ValueError("baseline families must be nonempty text")
    if len(set(families)) != len(families):
        raise ValueError("baseline families contain duplicates")
    if not indices or any(type(item) is not int or item < 0 for item in indices):
        raise ValueError("candidate indices must be nonnegative exact integers")
    if len(set(indices)) != len(indices):
        raise ValueError("candidate indices contain duplicates")
    return tuple(f"{family}::candidate_{index}" for family in families for index in indices)


def derive_williams_candidate_schedule(
    *,
    cell_ids: Sequence[str],
    root_count: int,
    sessions_per_root: int,
    seed: int,
) -> tuple[V4CandidateScheduleBlock, ...]:
    """Apply seeded labels to an even-order Williams carryover design."""

    cells = tuple(cell_ids)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("candidate cells must be nonempty and unique")
    if any(type(item) is not str or not item for item in cells):
        raise TypeError("candidate cell ids must be nonempty text")
    if len(cells) != 6:
        raise ValueError("the frozen P4.7 candidate design requires exactly six cells")
    _require_positive_int(root_count, "root_count")
    _require_positive_int(sessions_per_root, "sessions_per_root")
    if type(seed) is not int:
        raise TypeError("schedule seed must be an exact integer")
    labels = _sha256_fisher_yates(cells, seed=seed)
    first_row_indices = (0, 1, 5, 2, 4, 3)
    williams_rows = tuple(
        tuple(labels[(index + row_ordinal) % len(labels)] for index in first_row_indices)
        for row_ordinal in range(len(labels))
    )
    blocks: list[V4CandidateScheduleBlock] = []
    for session_index in range(sessions_per_root):
        for root_ordinal in range(root_count):
            global_ordinal = session_index * root_count + root_ordinal
            order = williams_rows[global_ordinal % len(williams_rows)]
            blocks.append(
                V4CandidateScheduleBlock(
                    root_ordinal=root_ordinal,
                    session_index=session_index,
                    global_block_ordinal=global_ordinal,
                    ordered_cell_ids=order,
                )
            )
    return tuple(blocks)


def classify_generated_action_case(case: Mapping[str, object]) -> tuple[str, str]:
    """Apply the receipt-first exclusive classification tree."""

    required = {
        "generation_attempt_completed",
        "authenticated_technical_failure_receipt_valid",
        "lineage_receipt_chain_valid",
        "utility_commitment_valid",
        "independent_reobserver_valid",
        "generated_bytes_receipt_valid",
        "delivered_bytes_equal_generated_bytes",
        "parser_artifact_hash_valid",
        "parser_completed_without_internal_error",
        "candidate_count",
        "candidate_in_closed_domain",
        "generation_parent_receipt_valid",
    }
    if set(case) != required:
        raise ValueError("generated-action primitive keys drift")
    completed = _nullable_bool(case["generation_attempt_completed"], "generation_attempt_completed")
    technical_receipt = _nullable_bool(
        case["authenticated_technical_failure_receipt_valid"],
        "authenticated_technical_failure_receipt_valid",
    )
    foundational_integrity_fields = (
        "lineage_receipt_chain_valid",
        "utility_commitment_valid",
        "independent_reobserver_valid",
    )
    if any(
        _nullable_bool(case[key], key) is not True
        for key in foundational_integrity_fields
    ):
        return "integrity_failure", "invalid_attempt_no_claim"
    if completed is not True:
        if (
            completed is False
            and technical_receipt is True
            and all(
                case[key] is None
                for key in required
                - {
                    "generation_attempt_completed",
                    "authenticated_technical_failure_receipt_valid",
                    "lineage_receipt_chain_valid",
                    "utility_commitment_valid",
                    "independent_reobserver_valid",
                }
            )
        ):
            return "technical_missingness", "contrast_specific_worst_case_itt"
        return "integrity_failure", "invalid_attempt_no_claim"
    if technical_receipt is not False:
        return "integrity_failure", "invalid_attempt_no_claim"

    integrity_fields = (
        "generated_bytes_receipt_valid",
        "delivered_bytes_equal_generated_bytes",
        "parser_artifact_hash_valid",
        "parser_completed_without_internal_error",
        "generation_parent_receipt_valid",
    )
    if any(_nullable_bool(case[key], key) is not True for key in integrity_fields):
        return "integrity_failure", "invalid_attempt_no_claim"
    candidate_count = case["candidate_count"]
    if type(candidate_count) is not int or candidate_count < 0:
        return "integrity_failure", "invalid_attempt_no_claim"
    candidate_in_domain = _nullable_bool(case["candidate_in_closed_domain"], "candidate_in_closed_domain")
    if candidate_count == 1 and candidate_in_domain is True:
        return "valid_generated_action", "lookup_committed_utility_vector"
    if candidate_in_domain is None:
        return "integrity_failure", "invalid_attempt_no_claim"
    return (
        "substantive_invalid_generated_action",
        "typed_invalid_action_utility_minus_one_in_itt",
    )


def derive_cartesian_tuple_count(axes: Mapping[str, Sequence[object]]) -> int:
    """Multiply nonempty, unique axis cardinalities without trusting a count literal."""

    if not axes:
        raise ValueError("Cartesian axes are empty")
    count = 1
    for name, values in axes.items():
        if type(name) is not str or not name:
            raise TypeError("Cartesian axis name must be nonempty text")
        items = tuple(values)
        if not items or len(set(items)) != len(items):
            raise ValueError(f"Cartesian axis {name} is empty or contains duplicates")
        count *= len(items)
    return count


def equicorrelation_psd_certificate(*, dimension: int, rho: Fraction) -> tuple[Fraction, Fraction]:
    """Return the two exact eigenvalues for a constant-correlation matrix."""

    _require_positive_int(dimension, "dimension")
    if dimension < 2:
        raise ValueError("equicorrelation dimension must be at least two")
    if type(rho) is not Fraction:
        raise TypeError("rho must be a Fraction")
    repeated = 1 - rho
    shared = 1 + (dimension - 1) * rho
    if repeated < 0 or shared < 0:
        raise ValueError("equicorrelation matrix is not positive semidefinite")
    return repeated, shared


def hoeffding_bonferroni_certificate() -> tuple[Fraction, Fraction]:
    """Prove ln(160) < 1269/250 using a finite positive exp series."""

    log_upper_bound = Fraction(1269, 250)
    exponential_lower_bound = sum((log_upper_bound**exponent) / math.factorial(exponent) for exponent in range(14))
    expected_difference = Fraction(
        1294120199914486134364636005563418813,
        381851196289062500000000000000000000000,
    )
    if exponential_lower_bound - 160 != expected_difference or exponential_lower_bound <= 160:
        raise AssertionError("Hoeffding rational certificate drift")
    return log_upper_bound, exponential_lower_bound


def minimum_candidate_for_exact_positive_mean_gate(
    *,
    candidate_root_counts: Sequence[int],
    practical_gate: Fraction,
    log_upper_bound: Fraction,
) -> int | None:
    """Find the first candidate where the strict bounded-mean gate can pass at 0.15."""

    if type(practical_gate) is not Fraction or practical_gate <= 0:
        raise ValueError("practical gate must be a positive Fraction")
    if type(log_upper_bound) is not Fraction or log_upper_bound <= 0:
        raise ValueError("log upper bound must be a positive Fraction")
    candidates = tuple(candidate_root_counts)
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("candidate root counts must be unique and ascending")
    threshold = 8 * log_upper_bound
    return next(
        (root_count for root_count in candidates if root_count * practical_gate * practical_gate > threshold),
        None,
    )


def _sha256_fisher_yates(items: tuple[str, ...], *, seed: int) -> tuple[str, ...]:
    result = list(items)
    draw_ordinal = 0
    for upper_index in range(len(result) - 1, 0, -1):
        selected = _sha256_draw_below(
            seed=seed,
            draw_ordinal=draw_ordinal,
            upper_exclusive=upper_index + 1,
        )
        result[upper_index], result[selected] = result[selected], result[upper_index]
        draw_ordinal += 1
    return tuple(result)


def _sha256_draw_below(*, seed: int, draw_ordinal: int, upper_exclusive: int) -> int:
    _require_positive_int(upper_exclusive, "upper_exclusive")
    limit = (1 << 64) - ((1 << 64) % upper_exclusive)
    rejection = 0
    while True:
        payload = f"p4.7-v4-candidate-schedule|{seed}|{draw_ordinal}|{rejection}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        if value < limit:
            return value % upper_exclusive
        rejection += 1


def _weighted_mean(values: Sequence[Fraction | int], masses: Sequence[Fraction]) -> Fraction:
    return sum(
        (mass * Fraction(value) for mass, value in zip(masses, values, strict=True)),
        Fraction(0, 1),
    )


def _weighted_covariance(
    left: Sequence[Fraction | int],
    right: Sequence[Fraction | int],
    masses: Sequence[Fraction],
) -> Fraction:
    left_mean = _weighted_mean(left, masses)
    right_mean = _weighted_mean(right, masses)
    return sum(
        (
            mass * (Fraction(left_value) - left_mean) * (Fraction(right_value) - right_mean)
            for mass, left_value, right_value in zip(masses, left, right, strict=True)
        ),
        Fraction(0, 1),
    )


def _weighted_correlation(
    left: Sequence[Fraction | int],
    right: Sequence[Fraction | int],
    masses: Sequence[Fraction],
) -> Fraction:
    covariance = _weighted_covariance(left, right, masses)
    left_variance = _weighted_covariance(left, left, masses)
    right_variance = _weighted_covariance(right, right, masses)
    if left_variance <= 0 or right_variance <= 0:
        raise ValueError("correlation requires positive variance")
    if left_variance != right_variance:
        raise ValueError("exact rational correlation requires the frozen common variance")
    return covariance / left_variance


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _nullable_bool(value: object, label: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise TypeError(f"{label} must be bool or null")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


__all__ = [
    "V4CandidateScheduleBlock",
    "V4NecessaryScreenResult",
    "V4SentinelDerivation",
    "classify_generated_action_case",
    "derive_candidate_cells",
    "derive_candidate_root_counts",
    "derive_cartesian_tuple_count",
    "derive_first_passing_necessary_screen",
    "derive_williams_candidate_schedule",
    "derive_necessary_point_screens",
    "derive_shared_reference_sentinel",
    "equicorrelation_psd_certificate",
    "exact_binomial_upper_tail",
    "hoeffding_bonferroni_certificate",
    "minimum_candidate_for_exact_positive_mean_gate",
]
