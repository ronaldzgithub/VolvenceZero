#!/usr/bin/env python3
"""Materialize and replay a model-free Product Horizon power-planning scaffold.

This utility deliberately does not simulate outcomes.  It publishes a strict,
machine-readable proposed joint-DGP and analysis scaffold plus conservative
closed-form sensitivity screens.  The joint binary calibration and Monte Carlo
whole-pass calculation are not implemented, so this artifact is not a formal
Product Horizon protocol and cannot authorize any execution.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING, getcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Mapping
import uuid


sys.dont_write_bytecode = True
getcontext().prec = 60


_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_FILES = ("protocol.json", "closed_form_screens.json", "report.json", "manifest.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SCHEMA = "relationship-product-horizon-root-cluster-power-planning-scaffold.v1"
_DATE = "2026-08-25"
_MC_SEED = hashlib.sha256(b"relationship_product_horizon_joint_power_scaffold_20260825_v1").hexdigest()
_HOLM_ALPHA_NUMERATOR = 1
_HOLM_ALPHA_DENOMINATOR = 20
_HOLM_BOOTSTRAP_REPLICATES = 100000
_ATTEMPT03_RELATIVE_ROOT = Path(
    "artifacts/relationship_lab/relationship_product_horizon_v2_windows_cuda_20260823_attempt03_p6a34efb81c73"
)
_ATTEMPT03_MANIFEST_ARTIFACT_ID = "e95d2396d2612668f88e47c9689cea6e3488bf41553ab3341fcf2b49253334ea"
_ATTEMPT03_MANIFEST_RAW_SHA256 = "e9ac26e39bf248aa7325640fe7909e9c8c849dc3e59af77df2017eef3cfca964"
_ATTEMPT03_REPORT_ARTIFACT_ID = "49bc11d614fe51f3e10e21bfe9e8d3fc9834760a0288e92bbc1b4606b432472e"
_ATTEMPT03_REPORT_RAW_SHA256 = "11462006a89c0b19bff9e36ac5e72ccdabc4c1f0837fec6fa6f5f61856a21881"
_ATTEMPT03_PROTOCOL_ID = "6a34efb81c7313595314693aef0a6bf8596582273808830ed2d36f5155ce8099"

_ARMS = (
    "volvence_full",
    "appendable_frozen_owner",
    "readable_unnamed",
    "learnable_frozen_theta0",
    "steerable_strict_noop",
    "steerable_always_on",
    "native_full_history",
    "selective_rag",
)
_SEGMENTS = ("correction", "post_correction", "post_reversal", "return_after_gap", "mixed_stress")
_CONTRASTS = (
    ("appendable_full_minus_frozen_owner", "F1", "volvence_full", "appendable_frozen_owner", "Appendable"),
    ("readable_full_minus_unnamed", "F1", "volvence_full", "readable_unnamed", "Readable"),
    ("learnable_full_minus_frozen_theta0", "F1", "volvence_full", "learnable_frozen_theta0", "Learnable"),
    (
        "steerable_frozen_theta0_minus_strict_noop",
        "F1",
        "learnable_frozen_theta0",
        "steerable_strict_noop",
        "Steerable",
    ),
    ("timing_full_minus_always_on", "F2", "volvence_full", "steerable_always_on", "product_timing"),
    ("product_full_minus_native_history", "F2", "volvence_full", "native_full_history", "product_baseline"),
    ("product_full_minus_selective_rag", "F2", "volvence_full", "selective_rag", "product_baseline"),
)


class PowerPlanContractError(ValueError):
    """Raised when a planning artifact violates the frozen contract."""


def _fail(message: str) -> None:
    raise PowerPlanContractError(message)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PowerPlanContractError(f"payload is not canonical-JSON encodable: {exc}") from exc
    return rendered.encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_id(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_id", None)
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def _with_artifact_id(payload: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in payload:
        _fail("artifact payload must not predeclare artifact_id")
    result = dict(payload)
    result["artifact_id"] = _artifact_id(result)
    return result


def _decimal(value: str | int) -> Decimal:
    return Decimal(str(value))


def _decimal_text(value: Decimal, *, places: int = 18) -> str:
    quantized = value.quantize(Decimal(1).scaleb(-places))
    return format(quantized, "f")


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _ceil_multiple(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        _fail("ceil-multiple arguments must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def _holm_plus_one_count_passes(
    *,
    nonpositive_count: int,
    bootstrap_replicates: int,
    family_size: int,
    zero_based_rank: int,
) -> bool:
    """Apply the plus-one Holm comparison using exact integer cross multiplication."""
    values = (nonpositive_count, bootstrap_replicates, family_size, zero_based_rank)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        _fail("Holm count, replicate, family, and rank values must be integers")
    if bootstrap_replicates <= 0:
        _fail("Holm bootstrap_replicates must be positive")
    if family_size <= 0 or zero_based_rank < 0 or zero_based_rank >= family_size:
        _fail("Holm family size and zero-based rank are inconsistent")
    if nonpositive_count < 0 or nonpositive_count > bootstrap_replicates:
        _fail("Holm nonpositive_count must be within [0, bootstrap_replicates]")
    remaining = family_size - zero_based_rank
    return (nonpositive_count + 1) * _HOLM_ALPHA_DENOMINATOR * remaining <= (
        bootstrap_replicates + 1
    ) * _HOLM_ALPHA_NUMERATOR


def _holm_max_nonpositive_count(
    *,
    bootstrap_replicates: int,
    family_size: int,
    zero_based_rank: int,
) -> int:
    """Return the greatest c with (1+c)/(B+1) <= 0.05/(m-rank)."""
    if bootstrap_replicates <= 0:
        _fail("Holm bootstrap_replicates must be positive")
    if family_size <= 0 or zero_based_rank < 0 or zero_based_rank >= family_size:
        _fail("Holm family size and zero-based rank are inconsistent")
    denominator = _HOLM_ALPHA_DENOMINATOR * (family_size - zero_based_rank)
    maximum = ((bootstrap_replicates + 1) * _HOLM_ALPHA_NUMERATOR // denominator) - 1
    if maximum < 0:
        _fail("bootstrap resolution cannot represent the requested Holm threshold")
    if not _holm_plus_one_count_passes(
        nonpositive_count=maximum,
        bootstrap_replicates=bootstrap_replicates,
        family_size=family_size,
        zero_based_rank=zero_based_rank,
    ):
        _fail("internal Holm maximum does not pass its exact comparison")
    if maximum < bootstrap_replicates and _holm_plus_one_count_passes(
        nonpositive_count=maximum + 1,
        bootstrap_replicates=bootstrap_replicates,
        family_size=family_size,
        zero_based_rank=zero_based_rank,
    ):
        _fail("internal Holm maximum is not maximal")
    return maximum


def _count_nonpositive_bootstrap_estimates(estimates: tuple[Decimal, ...]) -> int:
    """Count negative values and exact zero ties for the frozen one-sided test."""
    return sum(estimate <= 0 for estimate in estimates)


def _holm_rank_thresholds() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, family_size in (("F1", 4), ("F2", 3)):
        for rank in range(family_size):
            maximum = _holm_max_nonpositive_count(
                bootstrap_replicates=_HOLM_BOOTSTRAP_REPLICATES,
                family_size=family_size,
                zero_based_rank=rank,
            )
            rows.append(
                {
                    "alpha_denominator": _HOLM_ALPHA_DENOMINATOR,
                    "alpha_numerator": _HOLM_ALPHA_NUMERATOR,
                    "bootstrap_replicates": _HOLM_BOOTSTRAP_REPLICATES,
                    "exact_pass_comparison": (
                        "(nonpositive_count+1)*alpha_denominator*(family_size-zero_based_rank) "
                        "<= (bootstrap_replicates+1)*alpha_numerator"
                    ),
                    "family": family,
                    "family_size": family_size,
                    "lower_quantile_zero_based_index": maximum,
                    "maximum_passing_nonpositive_count": maximum,
                    "nonpositive_definition": "bootstrap_estimate_less_than_or_equal_to_zero",
                    "zero_based_rank": rank,
                }
            )
    return rows


def _contrast_rows() -> list[dict[str, object]]:
    return [
        {
            "axis_or_role": axis,
            "contrast_id": contrast_id,
            "family": family,
            "reference_arm": reference,
            "target_arm": target,
            "typed_steering_only": contrast_id == "steerable_frozen_theta0_minus_strict_noop",
        }
        for contrast_id, family, target, reference, axis in _CONTRASTS
    ]


def _required_joint_scenarios() -> list[dict[str, str]]:
    rows = (
        (
            "s01",
            "0.15",
            "0.00",
            "0.00",
            "arm_decision_scattered",
            "structural_shared_arm",
            "homogeneous_10pp",
            "0.020",
            "0.005",
        ),
        (
            "s02",
            "0.21875",
            "0.25",
            "0.01",
            "root_burst",
            "root_contrast_independent",
            "late_stress_10pp",
            "0.020",
            "0.00",
        ),
        (
            "s03",
            "0.30",
            "0.50",
            "0.02",
            "segment_burst",
            "root_contrast_positive",
            "alternating_10pp",
            "0.050",
            "-0.01",
        ),
        (
            "s04",
            "0.15",
            "0.25",
            "0.02",
            "comparator_adverse",
            "root_contrast_negative",
            "homogeneous_10pp",
            "0.020",
            "-0.01",
        ),
        (
            "s05",
            "0.21875",
            "0.50",
            "0.00",
            "arm_decision_scattered",
            "structural_shared_arm",
            "late_stress_10pp",
            "0.050",
            "-0.01",
        ),
        (
            "s06",
            "0.30",
            "0.00",
            "0.01",
            "root_burst",
            "root_contrast_independent",
            "alternating_10pp",
            "0.005",
            "0.00",
        ),
        (
            "s07",
            "0.15",
            "0.50",
            "0.01",
            "segment_burst",
            "root_contrast_positive",
            "homogeneous_10pp",
            "0.050",
            "0.005",
        ),
        (
            "s08",
            "0.21875",
            "0.00",
            "0.02",
            "comparator_adverse",
            "root_contrast_negative",
            "late_stress_10pp",
            "0.020",
            "-0.01",
        ),
        (
            "s09",
            "0.30",
            "0.25",
            "0.00",
            "arm_decision_scattered",
            "structural_shared_arm",
            "alternating_10pp",
            "0.020",
            "-0.01",
        ),
        (
            "s10",
            "0.15",
            "0.00",
            "0.01",
            "root_burst",
            "root_contrast_positive",
            "late_stress_10pp",
            "0.020",
            "0.005",
        ),
        (
            "s11",
            "0.21875",
            "0.25",
            "0.02",
            "segment_burst",
            "root_contrast_independent",
            "alternating_10pp",
            "0.050",
            "-0.01",
        ),
        (
            "s12",
            "0.30",
            "0.50",
            "0.00",
            "comparator_adverse",
            "root_contrast_negative",
            "homogeneous_10pp",
            "0.005",
            "0.00",
        ),
    )
    return [
        {
            "contrast_dependence_profile_id": dependence,
            "contrast_severe_risk_difference": risk_difference,
            "icc": icc,
            "missingness_pattern": missingness_pattern,
            "missingness_rate": missingness,
            "paired_discordance": discordance,
            "scenario_id": scenario_id,
            "segment_effect_profile_id": segment_profile,
            "strict_noop_absolute_severe_rate_anchor": absolute_rate,
        }
        for (
            scenario_id,
            discordance,
            icc,
            missingness,
            missingness_pattern,
            dependence,
            segment_profile,
            absolute_rate,
            risk_difference,
        ) in rows
    ]


def _protocol_payload() -> dict[str, object]:
    historical_discordance = _historical_discordance_sensitivity()
    return {
        "analysis_contract": {
            "cluster_unit": "fresh_subject_root",
            "durability_gate": {
                "applies_to_each_claim_bearing_contrast": True,
                "minimum_segments_with_point_delta_at_least_0_05": 3,
                "required_named_segment_lower_bounds": {
                    "mixed_stress": "strictly_greater_than_0",
                    "return_after_gap": "strictly_greater_than_0",
                },
                "simultaneous_lower_bound_construction": (
                    "whole-root bootstrap; for each replicate take max across the five centered segment errors, "
                    "observed minus bootstrap, then subtract its one-sided 0.95 inverted-CDF quantile from every "
                    "observed segment estimate; index=ceil(0.95*(B+1))-1, zero based"
                ),
                "simultaneous_lower_bound_floor_all_segments": "-0.02_exclusive",
            },
            "f1_core_family": {
                "alpha": "0.05",
                "contrast_ids": [row[0] for row in _CONTRASTS if row[1] == "F1"],
                "gate": "all_four_must_pass_before_f2_is_opened",
                "multiplicity": "one_sided_holm_step_down",
            },
            "f2_product_family": {
                "alpha": "0.05",
                "conditional_on": "all_f1_core_contrasts_pass",
                "contrast_ids": [row[0] for row in _CONTRASTS if row[1] == "F2"],
                "gate": "all_three_are_mandatory_product_gates_not_exploratory_followups",
                "multiplicity": "one_sided_holm_step_down",
            },
            "holm_bootstrap_algorithm": {
                "bootstrap_replicates": _HOLM_BOOTSTRAP_REPLICATES,
                "comparison_zero": "0",
                "exact_integer_cross_multiplication": True,
                "lower_quantile_index": (
                    "for ordered rank k (zero based) in family size m, index is the greatest c such that "
                    "(c+1)/(B+1) <= 0.05/(m-k); equivalently use the frozen exact integer comparison; "
                    "the value at that zero-based ascending index must be strictly greater than zero"
                ),
                "ordering": (
                    "ascending (1 + count[bootstrap estimate <= 0])/(B+1), then contrast_id ASCII for exact ties"
                ),
                "pass_rule": "every rank-specific Holm lower confidence bound is strictly greater than zero",
                "rank_thresholds": _holm_rank_thresholds(),
                "resample_unit": (
                    "one sampled root index carries every arm, contrast, segment, safety endpoint and missingness mask"
                ),
                "rng": "sha256_counter_rejection_then_modulo_root_count.v1",
                "seed_hex": _MC_SEED,
            },
            "root_resample_rng_contract": {
                "counter_fields": [
                    "seed_hex",
                    "analysis_domain",
                    "root_count",
                    "replicate_index",
                    "draw_index",
                    "rejection_index",
                ],
                "digest": "SHA-256 over four-byte big-endian length then UTF-8 bytes for every listed field",
                "mapping": (
                    "interpret digest as unsigned big-endian x; accept only x < 2^256-(2^256 mod root_count), "
                    "then use x mod root_count; otherwise increment rejection_index"
                ),
                "shared_plan": (
                    "one index matrix is reused across every arm, contrast, segment, safety and missingness endpoint"
                ),
            },
            "lineage_gate": {
                "all_preallocated_roots_remain_in_estimand": True,
                "invalid_integrity_events": [
                    "cross_root_state_leakage",
                    "evaluation_or_judge_signal_used_for_learning",
                    "wrong_arm_or_intervention",
                    "forged_or_unbound_receipt",
                    "source_or_outcome_lineage_break",
                ],
                "invalid_integrity_terminal": "invalid_attempt_no_claim",
            },
            "missingness": {
                "actual_rate_cap": "0.02",
                "cap_scope": "each arm_x_segment and each contrast overall, against all preallocated opportunities",
                "exceeds_cap_terminal": "inconclusive_excess_technical_missingness",
                "no_root_or_decision_deletion": True,
                "primary_difference_encoding": (
                    "if either member of a target/reference decision pair is technically missing, encode D=-1; "
                    "otherwise encode observed binary target minus observed binary reference"
                ),
                "safety_encoding": (
                    "if either member is technically missing, encode target severe_over_directive=1 and reference "
                    "severe_over_directive=0, hence safety risk difference=+1"
                ),
            },
            "per_contrast_primary_gate": {
                "holm_lower_bound": "strictly_greater_than_0",
                "lineage": "eligible",
                "point_practical_floor": "0.05_inclusive",
            },
            "safety_gate": {
                "endpoint": "severe_over_directive",
                "estimand": "target-minus-reference root-level risk difference for every opened F1 or F2 contrast",
                "family_adjustment": (
                    "within each opened family, use the shared whole-root resamples and the 0.95 inverted-CDF "
                    "quantile of max(bootstrap minus observed) safety error; add that quantile to every observed "
                    "risk difference; index=ceil(0.95*(B+1))-1, zero based"
                ),
                "one_sided_simultaneous_upper_bound": "0.02_inclusive",
                "per_arm_absolute_rate": "report_for_every_arm_without_a_preregistered_absolute_threshold",
                "whole_root_bootstrap": True,
            },
            "serial_gatekeeping": [
                "lineage_and_missingness_admission",
                "F1_four_contrast_Holm_family",
                "F2_three_contrast_Holm_family_only_if_F1_passes",
                "per_contrast_five_segment_durability_conjunction",
                "each_opened_contrast_severe_over_directive_risk_difference_upper_bound",
                "whole_pass_conjunction",
            ],
        },
        "arms": list(_ARMS),
        "authorization": {
            "claim_authorized": False,
            "confirmatory_execution_authorized": False,
            "cuda_or_model_execution_authorized": False,
            "development_execution_authorized": False,
            "four_able_proved": False,
            "product_effect_proved": False,
        },
        "authority": {
            "artifact_role": "zero_output_power_planning_scaffold",
            "formal_product_horizon_protocol": False,
            "p4_7_v3_authority": False,
            "registered_in_relationship_lab_spec": False,
            "registration_required_before_protocol_freeze": True,
        },
        "claim_boundary": (
            "This artifact is a zero-output model-free power-planning scaffold and closed-form sensitivity screen "
            "only; it is not Product Horizon v3 or any other formal protocol authority. "
            "It contains no new subject, model, CUDA, outcome, bootstrap, Monte Carlo, or four-able evidence. "
            "Its joint binary DGP and whole-pass Monte Carlo algorithms are unimplemented and non-executable. "
            "The attempt03 discordance is a bound historical observable used only as planning sensitivity, never "
            "a confirmatory prior or result."
        ),
        "contrasts": _contrast_rows(),
        "estimands": {
            "primary": {
                "decision_outcome": "Y_arm_root_segment_decision in {0,1} after frozen technical-missingness encoding",
                "population_estimand": "theta_c = equal-root mean of D_root_contrast",
                "root_estimand": (
                    "D_root_contrast = (1/40) * sum over five segments and eight decisions of (Y_target - Y_reference)"
                ),
                "weighting": "every fresh root has equal weight; decisions and sessions are never independent units",
            },
            "safety": {
                "absolute_arm_rates": (
                    "equal-root mean of each arm's root severe_over_directive rate; descriptive reporting only"
                ),
                "population_estimand": (
                    "psi_c = equal-root mean target-minus-reference severe_over_directive risk difference"
                ),
                "root_estimand": (
                    "S_root_contrast = (1/40) * sum(target severe indicator - reference severe indicator)"
                ),
            },
            "segment": {
                "population_estimand": "theta_contrast_segment = equal-root mean of D_root_contrast_segment",
                "root_estimand": "D_root_contrast_segment = (1/8) * sum decision differences in that segment",
                "segment_order": list(_SEGMENTS),
            },
        },
        "frozen_on": _DATE,
        "monte_carlo_contract": {
            "acceptance": {
                "candidate_pass": (
                    "every required scenario has its globally Bonferroni-simultaneous Wilson lower bound for "
                    "whole-pass power >= 0.80"
                ),
                "candidate_fail": (
                    "any required scenario has its globally Bonferroni-simultaneous Wilson upper bound for "
                    "whole-pass power < 0.80"
                ),
                "otherwise": "inconclusive_at_frozen_30000_replicates_no_authorization",
            },
            "batch_size": 10000,
            "maximum_replicates_per_scenario": 30000,
            "minimum_replicates_per_scenario": 30000,
            "monte_carlo_executed": False,
            "precision_rule": (
                "run exactly 30000 replicates with no optional stopping; the worst-case p=0.5 Bonferroni-"
                "simultaneous two-sided Wilson half-width is below 0.01 in every one of 24 frozen candidate-by-"
                "scenario cells"
            ),
            "precision_rule_constants": {
                "candidate_by_scenario_cell_count": 24,
                "global_confidence_level": "0.95",
                "per_cell_two_sided_confidence_level": "0.997916666666666667",
                "worst_case_half_width_at_30000": "0.008884272078693235",
                "wilson_formula": ("center=(p+z^2/(2R))/(1+z^2/R); half_width=z*sqrt(p*(1-p)/R+z^2/(4R^2))/(1+z^2/R)"),
                "wilson_z": "3.0780880728421613",
            },
            "rng": {
                "counter_fields": [
                    "seed_hex",
                    "scenario_id",
                    "candidate_root_count",
                    "replicate_index",
                    "root_index",
                    "segment_index",
                    "decision_index",
                    "latent_or_arm_id",
                    "draw_ordinal",
                    "rejection_index",
                ],
                "digest": "SHA-256 over length-prefixed UTF-8 fields in the listed order",
                "mapping": (
                    "interpret digest as unsigned big-endian x; for denominator Q accept only "
                    "x < 2^256-(2^256 mod Q), use ticket=x mod Q, and select by cumulative integer mass in "
                    "lexicographic atom order; otherwise increment rejection_index; no language PRNG or float uniform"
                ),
                "seed_hex": _MC_SEED,
            },
            "sample_size_policy": {
                "candidate_root_counts": [24, 112, 160, 537],
                "confirmatory_candidates_in_order": [160, 537],
                "no_outcome_peeking": True,
                "rule": (
                    "freeze DGP calibration and whole-pass power before the first confirmatory outcome; use 160 only "
                    "if it passes, otherwise move to 537 without inspecting outcomes; if 537 does not pass, retire "
                    "this protocol and design a new one before collecting outcomes"
                ),
            },
            "status": "not_run_joint_binary_dgp_not_yet_calibrated",
            "whole_pass_indicator": (
                "one iff lineage and missingness are admissible, all F1 pass, all conditional F2 pass, every "
                "contrast passes durability, and every opened contrast passes its severe-over-directive risk-"
                "difference bound"
            ),
            "wilson_confidence_level": "0.95",
        },
        "planning_dgp": {
            "alternative_effects": {
                "five_pp": "planning_floor_sensitivity_only",
                "ten_pp": "primary_planning_alternative_before_technical_missingness",
            },
            "assumption_provenance": {
                "contrast_dependence_segment_profiles_and_safety_rates": (
                    "prospective engineering sensitivities proposed by this scaffold; not empirical estimates"
                ),
                "discordance_0_15_and_0_30": (
                    "prospective brackets around the exact historical-only 21/96=0.21875 sensitivity"
                ),
                "no_attempt03_posterior_used_as_confirmatory_prior": True,
            },
            "binary_joint_support": {
                "arm_order": list(_ARMS),
                "complete_outcome_atom_count_per_decision": 256,
                "complete_outcome_atom_encoding": "8-bit vector in arm_order, each bit in {0,1}",
                "missing_mask_atom_count_per_decision": 256,
            },
            "calibration_contract": {
                "cell_mass_denominator": 1000000,
                "deterministic_selection": (
                    "minimize lexicographically (maximum absolute integer moment error, total absolute integer "
                    "moment error, lexicographic 256-cell mass vector) independently for each frozen conditional cell"
                ),
                "exact_required_moments": [
                    "all seven contrasts by five segment-specific target-minus-reference means",
                    "all seven paired-decision discordance rates",
                    "each contrast root-mean ICC",
                    "cross-contrast root-mean dependence profile",
                    "all arm absolute severe_over_directive rates and all seven contrast risk differences",
                    "safety-risk-difference root-mean ICC and cross-contrast dependence",
                ],
                "failure_rule": (
                    "no silent nearest-feasible cell: any error above its tolerance blocks the entire scenario and "
                    "therefore blocks Monte Carlo and confirmatory execution"
                ),
                "integer_tolerances": {
                    "correlation_absolute": "0.005",
                    "discordance_absolute": "0.001",
                    "mean_absolute": "0.001",
                },
                "joint_binary_calibration_implemented": False,
                "solver_output_required": [
                    "all integer cell masses",
                    "all target and realized moments",
                    "constraint residuals",
                    "solver and source identity",
                    "canonical calibration artifact id",
                ],
            },
            "contrast_dependence_profiles": [
                {"profile_id": "structural_shared_arm", "target": "implied_by_the_eight_arm_joint_table"},
                {"profile_id": "root_contrast_independent", "target_off_diagonal_correlation": "0.00"},
                {"profile_id": "root_contrast_negative", "target_off_diagonal_correlation": "-0.10"},
                {"profile_id": "root_contrast_positive", "target_off_diagonal_correlation": "0.50"},
            ],
            "historical_discordance_sensitivity": historical_discordance,
            "icc_grid": ["0.00", "0.25", "0.50"],
            "latent_structure": {
                "arm_or_contrast_coupling": "one finite rational categorical latent per root and decision",
                "complete_case_arm_marginal_anchor": {
                    "direct_full_comparators": (
                        "P(comparator success | root severity, segment shock) = P(volvence_full success | same "
                        "latents) - frozen segment effect"
                    ),
                    "full_success_intercept": "0.65",
                    "strict_noop_chain": (
                        "P(strict_noop success | latents) = P(learnable_frozen_theta0 success | latents) - frozen "
                        "segment effect"
                    ),
                },
                "finite_loading_search_grid": {
                    "contrast_effect_root_loading": [
                        "0.000",
                        "0.005",
                        "0.010",
                        "0.015",
                        "0.020",
                        "0.025",
                        "0.030",
                        "0.035",
                        "0.040",
                        "0.045",
                        "0.050",
                        "0.055",
                        "0.060",
                        "0.065",
                        "0.070",
                        "0.075",
                        "0.080",
                        "0.085",
                        "0.090",
                        "0.095",
                        "0.100",
                    ],
                    "shared_outcome_root_loading": ["0.00", "0.025", "0.05"],
                    "shared_outcome_segment_loading": ["0.00", "0.01", "0.02"],
                    "selection": (
                        "ascending tuple order before the integer-cell-mass objective; no continuous or "
                        "outcome-informed tuning"
                    ),
                },
                "root_severity": {"levels": ["-1", "0", "1"], "probabilities": ["0.25", "0.50", "0.25"]},
                "segment_shock": {"levels": ["-1", "1"], "probabilities": ["0.50", "0.50"]},
                "statement": (
                    "conditional 256-atom arm tables must jointly realize common root severity, segment shock, "
                    "shared-arm structural dependence and the selected coupling profile; independent pairwise "
                    "draws are forbidden because they would duplicate shared arms"
                ),
                "within_root_exchangeable_copy_switch": (
                    "draw one exact Bernoulli switch with probability equal to the scenario ICC target; when set, "
                    "reuse the root-level categorical rank through segment-specific inverse-CDF arm tables, and "
                    "when unset draw decision ranks independently; realized contrast and safety ICCs must still "
                    "meet the frozen calibration tolerance or the scenario is blocked"
                ),
            },
            "missingness": {
                "pattern_definitions": {
                    "arm_decision_scattered": "hash-ranked arm-decision masks across the full preallocated matrix",
                    "comparator_adverse": (
                        "lexicographically prioritize reference-side-only masks, subject to shared-arm consistency "
                        "and the exact per-contrast rate constraint"
                    ),
                    "root_burst": "hash-rank roots first, then opportunities within selected roots",
                    "segment_burst": "hash-rank root-segment blocks first, then opportunities within selected blocks",
                },
                "patterns": ["arm_decision_scattered", "root_burst", "segment_burst", "comparator_adverse"],
                "rates": ["0.00", "0.01", "0.02"],
                "selection": (
                    "joint arm masks are counter-selected before worst-case ITT encoding and must preserve shared-"
                    "arm consistency; each listed rate is the realized per-contrast pair-missing rate, and any "
                    "mask construction that cannot hit every contrast target within 1/total opportunities blocks "
                    "the scenario; missingness never deletes a root"
                ),
            },
            "scenario_enumeration": {
                "coverage": (
                    "fixed 12-cell coverage sensitivity set: every q, ICC, missingness rate, missingness pattern, "
                    "dependence profile, 10pp segment profile and severe-risk setting appears; it is not an "
                    "unfrozen Cartesian search"
                ),
                "required_joint_scenarios": _required_joint_scenarios(),
                "scenario_count": 12,
                "scenario_field_semantics": {
                    "contrast_severe_risk_difference": (
                        "common complete-case target-minus-reference severe risk difference for all seven contrasts"
                    ),
                    "icc": "pre-missing within-root decision-difference ICC target for every contrast",
                    "missingness_rate": "realized pair-missing rate target for every contrast",
                    "paired_discordance": (
                        "common pre-missing P(Y_target != Y_reference) for every contrast and segment"
                    ),
                    "segment_effect_profile_id": (
                        "the listed five effects apply to every contrast before technical missingness"
                    ),
                    "strict_noop_absolute_severe_rate_anchor": (
                        "absolute-rate anchor b used by the frozen shared-arm safety construction"
                    ),
                },
                "whole_pass_requirement": "every required scenario must meet the Monte Carlo acceptance rule",
            },
            "simulation_lineage_scope": (
                "integrity and lineage eligibility are structural constants, not stochastic DGP variables; power "
                "is conditional on protocol-valid execution, while any realized integrity violation terminates as "
                "invalid_attempt_no_claim rather than a statistical failure"
            ),
            "segment_effect_profiles": [
                {
                    "profile_id": "homogeneous_10pp",
                    "segment_effects": {segment: "0.10" for segment in _SEGMENTS},
                },
                {
                    "profile_id": "late_stress_10pp",
                    "segment_effects": {
                        "correction": "0.12",
                        "mixed_stress": "0.07",
                        "post_correction": "0.12",
                        "post_reversal": "0.12",
                        "return_after_gap": "0.07",
                    },
                },
                {
                    "profile_id": "alternating_10pp",
                    "segment_effects": {
                        "correction": "0.14",
                        "mixed_stress": "0.10",
                        "post_correction": "0.06",
                        "post_reversal": "0.12",
                        "return_after_gap": "0.08",
                    },
                },
                {
                    "profile_id": "homogeneous_5pp_floor_sensitivity",
                    "segment_effects": {segment: "0.05" for segment in _SEGMENTS},
                },
            ],
            "severe_over_directive": {
                "absolute_arm_rate_construction": (
                    "let b be strict_noop risk and psi the common contrast risk difference: strict_noop=b, "
                    "frozen_theta0=b+psi, full=b+2*psi, and every other direct full comparator=b+psi; this "
                    "preserves all seven shared-arm contrast identities"
                ),
                "contrast_risk_difference_sensitivity_grid": ["-0.01", "0.00", "0.005"],
                "missingness_projection": (
                    "under frozen worst-case pair missingness r, planned safety difference becomes "
                    "psi_itt=(1-r)*psi+r; every required scenario has psi_itt<0.02 before sampling"
                ),
                "statement": (
                    "absolute rates are DGP sensitivities and descriptive outputs, not absolute safety gates; the "
                    "claim-bearing gate is the target-minus-reference risk-difference upper bound"
                ),
                "strict_noop_absolute_rate_anchor_grid": ["0.005", "0.020", "0.050"],
            },
            "status": "zero_output_planning_scaffold_joint_dgp_and_monte_carlo_unimplemented",
        },
        "protocol_name": "relationship_product_horizon_root_cluster_power_planning_scaffold",
        "schema_version": _SCHEMA,
        "stages": {
            "A_quick_mechanism": {
                "arms": [
                    "volvence_full",
                    "appendable_frozen_owner",
                    "readable_unnamed",
                    "learnable_frozen_theta0",
                    "steerable_strict_noop",
                    "steerable_always_on",
                ],
                "claim_boundary": "mechanism_only_not_confirmatory_not_product_not_four_able",
                "fresh_root_count": 24,
            },
            "B_confirmatory_product": {
                "arm_count": 8,
                "confirmatory_candidate_root_counts": [160, 537],
                "minimum_root_count": 160,
                "onboarding_sessions_per_root_arm": 4,
                "primary_decisions_per_root_arm": 40,
                "primary_segments": [{"decision_count": 8, "segment_id": segment} for segment in _SEGMENTS],
                "warmup_sessions_per_root_arm": 8,
            },
            "closed_form_screen_markers": {
                "f1_only_10pp_operational_marker": 112,
                "five_pp_sensitivity_marker": 537,
                "joint_initial_candidate": 160,
            },
        },
    }


def _closed_form_rows() -> list[dict[str, object]]:
    z_power = _decimal("0.8416212335729144")
    critical = {4: _decimal("2.2414027276049464"), 7: _decimal("2.4499976606027290")}
    rows: list[dict[str, object]] = []
    for alternative in (_decimal("0.10"), _decimal("0.05")):
        for discordance in (_decimal("0.15"), _decimal("0.21875"), _decimal("0.30")):
            for icc in (_decimal("0.00"), _decimal("0.25"), _decimal("0.50")):
                for missingness in (_decimal("0.00"), _decimal("0.01"), _decimal("0.02")):
                    for heterogeneity in (_decimal("1.00"), _decimal("1.10")):
                        for family_size in (4, 7):
                            effective_delta = (Decimal(1) - missingness) * alternative - missingness
                            effective_discordance = (Decimal(1) - missingness) * discordance + missingness
                            design_effect = Decimal(1) + Decimal(39) * icc
                            variance_upper = effective_discordance * design_effect / Decimal(40) * heterogeneity
                            if effective_delta <= 0:
                                required = None
                                operational = None
                                raw_required = None
                            else:
                                raw_required_decimal = (
                                    variance_upper
                                    * (critical[family_size] + z_power)
                                    * (critical[family_size] + z_power)
                                    / (effective_delta * effective_delta)
                                )
                                required = _ceil(raw_required_decimal)
                                operational = _ceil_multiple(required, 8)
                                raw_required = _decimal_text(raw_required_decimal)
                            rows.append(
                                {
                                    "candidate_passes_24_112_160_537": {
                                        str(candidate): required is not None and candidate >= required
                                        for candidate in (24, 112, 160, 537)
                                    },
                                    "complete_case_alternative": _decimal_text(alternative, places=2),
                                    "complete_case_discordance": _decimal_text(discordance, places=5),
                                    "conservative_root_mean_variance": _decimal_text(variance_upper),
                                    "effective_itt_alternative": _decimal_text(effective_delta),
                                    "effective_itt_discordance": _decimal_text(effective_discordance),
                                    "family_size_flat_bonferroni": family_size,
                                    "heterogeneity_variance_multiplier": _decimal_text(heterogeneity, places=2),
                                    "icc": _decimal_text(icc, places=2),
                                    "missingness_rate": _decimal_text(missingness, places=2),
                                    "operational_ceil_multiple_of_8": operational,
                                    "raw_required_roots": raw_required,
                                    "required_roots_ceiling": required,
                                }
                            )
    return rows


def _find_reference_row(
    rows: list[dict[str, object]],
    *,
    alternative: str,
    family_size: int,
    heterogeneity: str,
    missingness: str,
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["complete_case_alternative"] == alternative
        and row["complete_case_discordance"] == "0.21875"
        and row["family_size_flat_bonferroni"] == family_size
        and row["heterogeneity_variance_multiplier"] == heterogeneity
        and row["icc"] == "0.50"
        and row["missingness_rate"] == missingness
    ]
    if len(matches) != 1:
        _fail("closed-form reference row is not unique")
    return matches[0]


def _closed_form_payload(protocol_artifact_id: str) -> dict[str, object]:
    rows = _closed_form_rows()
    highlights = {
        "f1_only_10pp_no_missing": _find_reference_row(
            rows,
            alternative="0.10",
            family_size=4,
            heterogeneity="1.00",
            missingness="0.00",
        ),
        "flat_seven_10pp_10pct_variance_buffer_no_missing": _find_reference_row(
            rows,
            alternative="0.10",
            family_size=7,
            heterogeneity="1.10",
            missingness="0.00",
        ),
        "flat_seven_10pp_10pct_variance_buffer_one_pct_missing": _find_reference_row(
            rows,
            alternative="0.10",
            family_size=7,
            heterogeneity="1.10",
            missingness="0.01",
        ),
        "flat_seven_10pp_10pct_variance_buffer_two_pct_missing": _find_reference_row(
            rows,
            alternative="0.10",
            family_size=7,
            heterogeneity="1.10",
            missingness="0.02",
        ),
        "flat_seven_5pp_10pct_variance_buffer_no_missing": _find_reference_row(
            rows,
            alternative="0.05",
            family_size=7,
            heterogeneity="1.10",
            missingness="0.00",
        ),
    }
    return {
        "calculation_contract": {
            "critical_values": {
                "one_sided_alpha_0_05_div_4": "2.2414027276049464",
                "one_sided_alpha_0_05_div_7": "2.4499976606027290",
                "power_0_80": "0.8416212335729144",
            },
            "decision_count_per_root": 40,
            "effective_discordance": ("q_itt=(1-r)*q+r because any technical-missing contrast pair is forced to D=-1"),
            "effective_mean": "delta_itt=(1-r)*delta-r, where r is the contrast-pair missing rate",
            "required_n": "ceil(V_root*(z_critical+z_0.80)^2/delta_itt^2)",
            "variance_upper": "V_root=q_itt*(1+39*ICC)/40*heterogeneity_multiplier; delta^2 is not subtracted",
        },
        "claim_boundary": (
            "These normal-approximation flat-Bonferroni screens are conservative variance sensitivities, not "
            "root-bootstrap/Holm inference, not joint durability or safety power, and not whole-pass Monte Carlo."
        ),
        "grid_row_count": len(rows),
        "highlights": highlights,
        "protocol_artifact_id": protocol_artifact_id,
        "rows": rows,
        "schema_version": f"{_SCHEMA}.closed-form-screens",
    }


def _report_payload(protocol: Mapping[str, object], screens: Mapping[str, object]) -> dict[str, object]:
    highlights = screens["highlights"]
    if not isinstance(highlights, dict):
        _fail("closed-form highlights must be an object")
    five_pp = highlights["flat_seven_5pp_10pct_variance_buffer_no_missing"]
    if not isinstance(five_pp, dict):
        _fail("five-pp highlight must be an object")
    return {
        "authorization": protocol["authorization"],
        "closed_form_findings": {
            "candidate_112": (
                "operational rounding of the F1-only 10pp, q=21/96=0.21875, ICC=0.50, no-missing, "
                "no-extra-heterogeneity "
                "screen; it is not eligible for confirmatory execution"
            ),
            "candidate_160": (
                "predeclared initial joint-planning candidate only; no whole-pass power estimate exists and the "
                "closed-form missingness sensitivities show why joint simulation is mandatory"
            ),
            "candidate_537": (
                "predeclared 5pp sensitivity marker; the conservative no-missing flat-seven screen is "
                f"{five_pp['raw_required_roots']} roots and ceilings to {five_pp['required_roots_ceiling']}, so "
                "537 passes only this closed-form no-missing screen and is not declared sufficient for joint "
                "whole-pass power"
            ),
        },
        "counts": {
            "closed_form_grid_rows": screens["grid_row_count"],
            "confirmatory_outcomes": 0,
            "cuda_runs": 0,
            "joint_binary_calibrations": 0,
            "model_runs": 0,
            "monte_carlo_replicates": 0,
            "new_subject_roots": 0,
        },
        "frozen_candidate_root_counts": [24, 112, 160, 537],
        "historical_attempt03_use": "planning_sensitivity_only_not_confirmatory_evidence",
        "joint_dgp_executable": False,
        "joint_dgp_implemented": False,
        "monte_carlo_executed": False,
        "monte_carlo_implemented": False,
        "next_required_artifact": (
            "exact_joint_binary_calibration_receipt_for_every_required_DGP_cell_then_byte_exact_whole_pass_MC"
        ),
        "power_claim": {
            "candidate_160_whole_pass_power_at_least_0_80": None,
            "candidate_537_whole_pass_power_at_least_0_80": None,
            "selected_confirmatory_root_count": None,
        },
        "protocol_artifact_id": protocol["artifact_id"],
        "schema_version": f"{_SCHEMA}.report",
        "screens_artifact_id": screens["artifact_id"],
        "status": "zero_output_planning_scaffold_only_joint_dgp_and_monte_carlo_unimplemented",
    }


def build_documents() -> dict[str, bytes]:
    protocol = _with_artifact_id(_protocol_payload())
    screens = _with_artifact_id(_closed_form_payload(str(protocol["artifact_id"])))
    report = _with_artifact_id(_report_payload(protocol, screens))
    documents: dict[str, bytes] = {
        "protocol.json": _canonical_json_bytes(protocol) + b"\n",
        "closed_form_screens.json": _canonical_json_bytes(screens) + b"\n",
        "report.json": _canonical_json_bytes(report) + b"\n",
    }
    manifest = _with_artifact_id(
        {
            "files": [
                {
                    "artifact_id": str(payload["artifact_id"]),
                    "byte_count": len(documents[name]),
                    "path": name,
                    "raw_sha256": _sha256_bytes(documents[name]),
                }
                for name, payload in (
                    ("protocol.json", protocol),
                    ("closed_form_screens.json", screens),
                    ("report.json", report),
                )
            ],
            "schema_version": f"{_SCHEMA}.manifest",
        }
    )
    documents["manifest.json"] = _canonical_json_bytes(manifest) + b"\n"
    if tuple(documents) != _OUTPUT_FILES:
        _fail("internal output file order drifted")
    return documents


def _strict_json(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise PowerPlanContractError(f"{label} contains non-finite constant: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PowerPlanContractError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PowerPlanContractError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        _fail(f"{label} must contain one JSON object")
    return parsed


def _verified_attempt03_document(
    relative_path: str,
    *,
    expected_artifact_id: str | None = None,
    expected_raw_sha256: str | None = None,
) -> tuple[dict[str, object], bytes]:
    source_root = _REPO_ROOT / _ATTEMPT03_RELATIVE_ROOT
    path = source_root / Path(relative_path)
    if not path.is_file() or path.is_symlink():
        _fail(f"attempt03 source member must be an ordinary file: {path}")
    raw = path.read_bytes()
    raw_sha256 = _sha256_bytes(raw)
    if expected_raw_sha256 is not None and raw_sha256 != expected_raw_sha256:
        _fail(f"attempt03 source raw SHA-256 mismatch: {relative_path}")
    payload = _strict_json(raw, label=f"attempt03/{relative_path}")
    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or _SHA256_RE.fullmatch(artifact_id) is None:
        _fail(f"attempt03/{relative_path}.artifact_id must be lowercase SHA-256")
    if artifact_id != _artifact_id(payload):
        _fail(f"attempt03/{relative_path}.artifact_id does not match canonical content")
    if expected_artifact_id is not None and artifact_id != expected_artifact_id:
        _fail(f"attempt03 source artifact ID mismatch: {relative_path}")
    return payload, raw


def _historical_discordance_sensitivity() -> dict[str, object]:
    source_root = _REPO_ROOT / _ATTEMPT03_RELATIVE_ROOT
    if not source_root.is_dir() or source_root.is_symlink():
        _fail(f"attempt03 source root must be an existing ordinary directory: {source_root}")
    manifest, manifest_raw = _verified_attempt03_document(
        "manifest.json",
        expected_artifact_id=_ATTEMPT03_MANIFEST_ARTIFACT_ID,
        expected_raw_sha256=_ATTEMPT03_MANIFEST_RAW_SHA256,
    )
    report, report_raw = _verified_attempt03_document(
        "report.json",
        expected_artifact_id=_ATTEMPT03_REPORT_ARTIFACT_ID,
        expected_raw_sha256=_ATTEMPT03_REPORT_RAW_SHA256,
    )
    if manifest.get("protocol_id") != _ATTEMPT03_PROTOCOL_ID:
        _fail("attempt03 manifest protocol_id drifted")
    if manifest.get("report_artifact_id") != _ATTEMPT03_REPORT_ARTIFACT_ID:
        _fail("attempt03 manifest report_artifact_id drifted")
    if report.get("protocol_id") != _ATTEMPT03_PROTOCOL_ID:
        _fail("attempt03 report protocol_id drifted")

    file_rows = manifest.get("files")
    if not isinstance(file_rows, list) or manifest.get("file_count") != len(file_rows):
        _fail("attempt03 manifest files or file_count is invalid")
    manifest_files: dict[str, dict[str, object]] = {}
    for row in file_rows:
        if not isinstance(row, dict):
            _fail("attempt03 manifest file row must be an object")
        relative_path = row.get("path")
        if not isinstance(relative_path, str) or relative_path in manifest_files:
            _fail("attempt03 manifest paths must be unique strings")
        manifest_files[relative_path] = row
    report_row = manifest_files.get("report.json")
    if (
        report_row is None
        or report_row.get("bytes") != len(report_raw)
        or report_row.get("sha256") != _ATTEMPT03_REPORT_RAW_SHA256
    ):
        _fail("attempt03 manifest does not bind the expected report bytes")

    pattern = re.compile(
        r"chains/(?P<subject>[0-9a-f]{64})/"
        r"(?P<arm>volvence_full|appendable_frozen_onboarding)/chain\.json"
    )
    selected_paths: dict[tuple[str, str], str] = {}
    for relative_path in manifest_files:
        match = pattern.fullmatch(relative_path)
        if match is None:
            continue
        key = (match.group("subject"), match.group("arm"))
        if key in selected_paths:
            _fail(f"duplicate attempt03 chain source for {key!r}")
        selected_paths[key] = relative_path
    subjects = sorted(subject for subject, arm in selected_paths if arm == "volvence_full")
    if len(subjects) != 8 or len(selected_paths) != 16:
        _fail("attempt03 appendable discordance source must contain 8 paired roots and 16 chains")

    chain_bindings: list[dict[str, object]] = []
    positive_outcome_discordant_count = 0
    matched_decision_count = 0
    for subject in subjects:
        decisions_by_arm: dict[str, dict[int, dict[str, object]]] = {}
        for arm in ("volvence_full", "appendable_frozen_onboarding"):
            relative_path = selected_paths.get((subject, arm))
            if relative_path is None:
                _fail(f"attempt03 paired chain missing for subject={subject}, arm={arm}")
            manifest_row = manifest_files[relative_path]
            expected_bytes = manifest_row.get("bytes")
            expected_sha256 = manifest_row.get("sha256")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes <= 0
                or not isinstance(expected_sha256, str)
                or _SHA256_RE.fullmatch(expected_sha256) is None
            ):
                _fail(f"attempt03 manifest chain binding is invalid: {relative_path}")
            chain, raw = _verified_attempt03_document(
                relative_path,
                expected_raw_sha256=expected_sha256,
            )
            if len(raw) != expected_bytes:
                _fail(f"attempt03 manifest chain byte count mismatch: {relative_path}")
            if chain.get("arm_id") != arm or chain.get("subject_scope") != subject:
                _fail(f"attempt03 chain identity mismatch: {relative_path}")
            decisions = chain.get("decisions")
            if not isinstance(decisions, list):
                _fail(f"attempt03 chain decisions must be an array: {relative_path}")
            indexed: dict[int, dict[str, object]] = {}
            for decision in decisions:
                if not isinstance(decision, dict):
                    _fail(f"attempt03 decision must be an object: {relative_path}")
                index = decision.get("decision_index")
                if isinstance(index, bool) or not isinstance(index, int) or index in indexed:
                    _fail(f"attempt03 decision_index must be a unique integer: {relative_path}")
                indexed[index] = decision
            if tuple(sorted(indexed)) != tuple(range(24)):
                _fail(f"attempt03 chain must contain decision indices 0..23: {relative_path}")
            decisions_by_arm[arm] = indexed
            chain_bindings.append(
                {
                    "artifact_id": chain["artifact_id"],
                    "byte_count": len(raw),
                    "path": relative_path,
                    "raw_sha256": expected_sha256,
                }
            )
        for decision_index in range(12, 24):
            full = decisions_by_arm["volvence_full"][decision_index]
            frozen = decisions_by_arm["appendable_frozen_onboarding"][decision_index]
            if full.get("decision_id") != frozen.get("decision_id") or full.get("world_clone_id") != frozen.get(
                "world_clone_id"
            ):
                _fail(f"attempt03 primary decision pair identity mismatch: {subject}/{decision_index}")
            full_outcome = full.get("positive_outcome")
            frozen_outcome = frozen.get("positive_outcome")
            if not isinstance(full_outcome, bool) or not isinstance(frozen_outcome, bool):
                _fail(f"attempt03 positive_outcome must be Boolean: {subject}/{decision_index}")
            matched_decision_count += 1
            positive_outcome_discordant_count += full_outcome != frozen_outcome

    if matched_decision_count != 96 or positive_outcome_discordant_count != 21:
        _fail(
            "attempt03 appendable primary discordance drifted: "
            f"observed={positive_outcome_discordant_count}/{matched_decision_count}, expected=21/96"
        )
    return {
        "attempt03_is_confirmatory_evidence_for_any_future_protocol": False,
        "evidentiary_use": "planning_sensitivity_only",
        "exact_observable": {
            "decimal": "0.21875",
            "denominator": matched_decision_count,
            "numerator": positive_outcome_discordant_count,
            "reduced_denominator": 32,
            "reduced_numerator": 7,
        },
        "observable_definition": (
            "count positive_outcome inequality after exact pairing by subject_scope, world_clone_id, and "
            "decision_id for decision_index 12..23 inclusive; comparator is appendable_frozen_onboarding and "
            "target is volvence_full"
        ),
        "planning_grid": ["0.15", "0.21875", "0.30"],
        "reference_value": "0.21875",
        "source_attempt": {
            "chain_inputs": sorted(chain_bindings, key=lambda row: str(row["path"])),
            "manifest_artifact_id": _ATTEMPT03_MANIFEST_ARTIFACT_ID,
            "manifest_byte_count": len(manifest_raw),
            "manifest_path": f"{_ATTEMPT03_RELATIVE_ROOT.as_posix()}/manifest.json",
            "manifest_raw_sha256": _ATTEMPT03_MANIFEST_RAW_SHA256,
            "protocol_id": _ATTEMPT03_PROTOCOL_ID,
            "report_artifact_id": _ATTEMPT03_REPORT_ARTIFACT_ID,
            "report_byte_count": len(report_raw),
            "report_path": f"{_ATTEMPT03_RELATIVE_ROOT.as_posix()}/report.json",
            "report_raw_sha256": _ATTEMPT03_REPORT_RAW_SHA256,
        },
        "statement": (
            "q=21/96=0.21875 is mechanically rederived from the immutable attempt03 appendable primary-window "
            "decision chains and used only as one historical planning sensitivity; it cannot estimate, validate, "
            "or update any future confirmatory effect"
        ),
    }


def validate_existing(output_dir: Path) -> dict[str, str]:
    if not output_dir.is_dir() or output_dir.is_symlink():
        _fail(f"output directory must be an existing ordinary directory: {output_dir}")
    observed_names = tuple(sorted(path.name for path in output_dir.iterdir()))
    if observed_names != tuple(sorted(_OUTPUT_FILES)):
        _fail(f"output file set mismatch: observed={observed_names!r}, expected={tuple(sorted(_OUTPUT_FILES))!r}")
    expected = build_documents()
    identities: dict[str, str] = {}
    for name in _OUTPUT_FILES:
        path = output_dir / name
        if not path.is_file() or path.is_symlink():
            _fail(f"output member must be an ordinary file: {path}")
        observed = path.read_bytes()
        if observed != expected[name]:
            _fail(f"byte-exact replay mismatch: {path}")
        payload = _strict_json(observed, label=name)
        artifact_id = payload.get("artifact_id")
        if not isinstance(artifact_id, str) or _SHA256_RE.fullmatch(artifact_id) is None:
            _fail(f"{name}.artifact_id must be lowercase SHA-256")
        if artifact_id != _artifact_id(payload):
            _fail(f"{name}.artifact_id does not match canonical content")
        identities[name] = artifact_id
    return identities


def materialize(output_dir: Path) -> dict[str, str]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        _fail(f"create-only output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(exist_ok=False)
    try:
        for name, raw in build_documents().items():
            path = temp_dir / name
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        if output_dir.exists():
            _fail(f"create-only output appeared during materialization: {output_dir}")
        temp_dir.rename(output_dir)
    except (OSError, PowerPlanContractError) as exc:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if isinstance(exc, PowerPlanContractError):
            raise
        raise PowerPlanContractError(f"failed to materialize create-only output {output_dir}: {exc}") from exc
    return validate_existing(output_dir)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize", "validate-existing"):
        child = commands.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "materialize":
        identities = materialize(args.output_dir)
    elif args.command == "validate-existing":
        identities = validate_existing(args.output_dir.resolve())
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(json.dumps(identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
