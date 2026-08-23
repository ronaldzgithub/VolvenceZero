from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
from typing import Any, Callable

import pytest

import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign as owner
from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
    prepare_relationship_p4_long_context_power_admission_certificate,
    prepare_relationship_p4_long_context_power_failure_certificate,
    relationship_p4_long_context_protocol_path,
    validate_relationship_p4_long_context_power_admission_certificate,
    validate_relationship_p4_long_context_power_failure_certificate,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_OWNER_SOURCE = (
    _REPO_ROOT
    / "packages"
    / "lifeform-evolution"
    / "src"
    / "lifeform_evolution"
    / "relationship_lab_p4_long_context_causal_campaign.py"
)
_PROTOCOL_DIRECTORY = _OWNER_SOURCE.parent / "protocols"
_POWER_PROTOCOL_V1 = _PROTOCOL_DIRECTORY / "relationship_p4_long_context_power_bound_fail_v1.json"
_POWER_PROTOCOL_V2 = _PROTOCOL_DIRECTORY / "relationship_p4_long_context_power_admission_v2.json"
_CLI_SOURCE = _REPO_ROOT / "scripts" / "run_relationship_lab_p4_long_context_power_bound.py"
_SCIENTIFIC_PREPARATION_V2 = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_causal_campaign_design_prereg_v2_20260823"
)
_SCIENTIFIC_PREPARATION_V3 = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_causal_campaign_design_prereg_v3_20260823"
)
_POWER_ARTIFACT_V1 = (
    _REPO_ROOT / "artifacts" / "relationship_lab" / "p4_independent_long_context_power_preflight_v3_fail_20260823"
)
_POWER_ARTIFACT_V2 = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_power_admission_v2_under_specified_20260823"
)
_V1_CERTIFICATE_FILE = "necessary_power_failure_certificate.json"
_V2_CERTIFICATE_FILE = "power_admission_certificate.json"
_MANIFEST_FILE = "manifest.json"

_EXPECTED_POWER_PROTOCOL_ID_V1 = "735b20a137b03176cf889c0cbe116e29f973c18d4cef4bf38cd42df288dff3fa"
_EXPECTED_POWER_PROTOCOL_RAW_V1 = "1bb8d21ce3a0dca332324d2e35e3bc2c63ec77fc9bd3917b35d018ebd85559f6"
_EXPECTED_POWER_ARTIFACT_ID_V1 = "fad6c105b7c64a6b4ab89bf6e933ecdf4c8f1b1170679d918c2dd77c27809518"
_EXPECTED_POWER_CERTIFICATE_ID_V1 = "682efba886b002db849a83ff086963921a173391a4d5e3c050b3d472d17ee70e"
_EXPECTED_POWER_CERTIFICATE_RAW_V1 = "543bad00793aabce0869b1f7b2780310ea50a3a957c96b22ae4a677d5ad10de8"
_EXPECTED_POWER_MANIFEST_RAW_V1 = "f96750169c8613b8c434a9ab80df4fa5f7bf8fee1e4759a0f64ccf73aa7d70e3"

_EXPECTED_POWER_PROTOCOL_ID_V2 = "67d294faf9209c9d05334f4c0e87371676c9821b7c12e603f3e289f33f566bc9"
_EXPECTED_POWER_PROTOCOL_RAW_V2 = "130f766787ec0b02bd5857344e58b371d996aa51fabe421cbfbde05347fd0e04"
_EXPECTED_POWER_ARTIFACT_ID_V2 = "9883e10784a06260a220a6fdbf72141b1300c21e97faee6e84a401c40a144ee9"
_EXPECTED_POWER_CERTIFICATE_ID_V2 = "cd6ceca086a1d8a311c75bdacd70c976e05b90dff2cde55b3ad41c00d29936b3"
_EXPECTED_POWER_CERTIFICATE_RAW_V2 = "0f20e47da67e5ebaed39e63d274805d5783cc588e0f8aa5fa2a0450b079d18ba"
_EXPECTED_POWER_MANIFEST_RAW_V2 = "f6fb9d482c8eb7f7e5e8dd92546a12bbae6f22ea76730c205f41fba4e14b4972"

_EXPECTED_SCIENTIFIC_PROTOCOL_ID_V3 = "9f352778e128a9573790762222a05225740bdaeb732800dec0eec124116a282d"
_EXPECTED_SCIENTIFIC_PROTOCOL_RAW_V3 = "ea8a17a14a68802d3b60586bf520c9137e6920be4112c951ec8c69f5e6ea359e"
_EXPECTED_SCIENTIFIC_PREPARATION_ID_V3 = "c5a708ae5e68261fddbade165b45579e66e4bbe7db1be1f4a83056561a17f42e"
_EXPECTED_SCIENTIFIC_PREPARATION_RAW_V3 = "a4b2f3ee920e398ae0f7eab5757b7988dc3fa4f7db7b4769599c837bc656bcd6"
_EXPECTED_SCIENTIFIC_PREPARATION_MANIFEST_RAW_V3 = "4aaf10d76b80a780e62a17b62803906cc34dd34d64bb4d0f8b96d60dff1e2663"
_EXPECTED_SCIENTIFIC_PROTOCOL_ID_V2 = "666d2e8546cd4b4cf55ece06354310e10b4dc07298241b94ef9593e4b5f63baf"

_EXPECTED_TAIL_NUMERATOR = int(
    "778949458858892235748447380312021738820275060582978334512217397924398674340078952648157847946225"
    "490252874679879917686927822970664614655961432314618662088464993507158297941512894452204126009227"
    "07927209627064220570521391366669147754170951541079882737"
)
_EXPECTED_TAIL_DENOMINATOR = int(
    "12554203470773361527671578846415332832204710888928069025792000000000000000000000000"
    "00000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "00000000000000000000000000000000000000000000000000000000000000000000000000000000000"
)


def test_v1_and_v2_protocol_artifact_identity_ladders_are_literal_pins() -> None:
    _assert_protocol_identity(
        _POWER_PROTOCOL_V1,
        expected_id=_EXPECTED_POWER_PROTOCOL_ID_V1,
        expected_raw=_EXPECTED_POWER_PROTOCOL_RAW_V1,
    )
    _assert_protocol_identity(
        _POWER_PROTOCOL_V2,
        expected_id=_EXPECTED_POWER_PROTOCOL_ID_V2,
        expected_raw=_EXPECTED_POWER_PROTOCOL_RAW_V2,
    )
    assert owner.P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1 == _EXPECTED_POWER_PROTOCOL_ID_V1
    assert owner.P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1 == _EXPECTED_POWER_PROTOCOL_RAW_V1
    assert owner.P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2 == _EXPECTED_POWER_PROTOCOL_ID_V2
    assert owner.P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2 == _EXPECTED_POWER_PROTOCOL_RAW_V2
    assert owner.P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID == _EXPECTED_POWER_PROTOCOL_ID_V2
    assert owner.P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2 == _EXPECTED_POWER_ARTIFACT_ID_V2

    _assert_artifact_identity(
        _POWER_ARTIFACT_V1,
        certificate_file=_V1_CERTIFICATE_FILE,
        expected_artifact_id=_EXPECTED_POWER_ARTIFACT_ID_V1,
        expected_certificate_id=_EXPECTED_POWER_CERTIFICATE_ID_V1,
        expected_certificate_raw=_EXPECTED_POWER_CERTIFICATE_RAW_V1,
        expected_manifest_raw=_EXPECTED_POWER_MANIFEST_RAW_V1,
    )
    _assert_artifact_identity(
        _POWER_ARTIFACT_V2,
        certificate_file=_V2_CERTIFICATE_FILE,
        expected_artifact_id=_EXPECTED_POWER_ARTIFACT_ID_V2,
        expected_certificate_id=_EXPECTED_POWER_CERTIFICATE_ID_V2,
        expected_certificate_raw=_EXPECTED_POWER_CERTIFICATE_RAW_V2,
        expected_manifest_raw=_EXPECTED_POWER_MANIFEST_RAW_V2,
    )

    historical = validate_relationship_p4_long_context_power_failure_certificate(
        output_dir=_POWER_ARTIFACT_V1,
        preparation_dir=_SCIENTIFIC_PREPARATION_V3,
    )
    assert historical.artifact_id == _EXPECTED_POWER_ARTIFACT_ID_V1
    assert historical.decisive_failure is True
    assert historical.scientific_admission is False
    assert historical.development_authorized is False
    assert historical.formal_authorized is False

    current = validate_relationship_p4_long_context_power_admission_certificate(
        output_dir=_POWER_ARTIFACT_V2,
        preparation_dir=_SCIENTIFIC_PREPARATION_V3,
    )
    assert current.artifact_id == _EXPECTED_POWER_ARTIFACT_ID_V2
    assert current.admission_protocol_id == _EXPECTED_POWER_PROTOCOL_ID_V2
    assert current.scientific_protocol_id == _EXPECTED_SCIENTIFIC_PROTOCOL_ID_V3
    assert current.preparation_artifact_id == _EXPECTED_SCIENTIFIC_PREPARATION_ID_V3
    assert current.status == "power_contract_under_specified_no_development_authorization"
    assert current.power_contract_determinate is False
    assert current.v1_unconditional_scientific_admission_valid is False
    assert current.development_authorized is False
    assert current.formal_authorized is False


def test_v1_math_is_reproducible_but_only_a_conditional_v2_diagnostic() -> None:
    lower = Fraction(-2, 1)
    upper = Fraction(2, 1)
    planning_mean = Fraction(1, 5)
    threshold = Fraction(3, 20)
    required_power = Fraction(4, 5)
    formal_roots = 192

    mass_at_upper = (planning_mean - lower) / (upper - lower)
    mass_at_lower = 1 - mass_at_upper
    maximum_variance = (upper - planning_mean) * (planning_mean - lower)
    fractional_count = formal_roots * (threshold - lower) / (upper - lower)
    minimum_success_count = -(-fractional_count.numerator // fractional_count.denominator)
    exact_tail = sum(
        Fraction(math.comb(formal_roots, k) * 11**k * 9 ** (formal_roots - k), 20**formal_roots)
        for k in range(minimum_success_count, formal_roots + 1)
    )

    assert mass_at_upper == Fraction(11, 20)
    assert mass_at_lower == Fraction(9, 20)
    assert maximum_variance == Fraction(99, 25)
    assert minimum_success_count == 104
    assert exact_tail == Fraction(_EXPECTED_TAIL_NUMERATOR, _EXPECTED_TAIL_DENOMINATOR)
    assert 5 * exact_tail.numerator < 4 * exact_tail.denominator
    assert exact_tail < required_power

    historical = _strict_json(_POWER_ARTIFACT_V1 / _V1_CERTIFICATE_FILE)
    historical_tail = historical["exact_point_gate_enumeration"]["exact_tail_probability"]
    assert int(historical_tail["numerator"]) == exact_tail.numerator
    assert int(historical_tail["denominator"]) == exact_tail.denominator
    assert historical["exact_point_gate_enumeration"]["minimum_success_count"] == 104
    assert historical["maximum_variance_proof"]["maximum_variance"] == {
        "numerator": "99",
        "denominator": "25",
    }

    current = _strict_json(_POWER_ARTIFACT_V2 / _V2_CERTIFICATE_FILE)
    conditional = current["conditional_numeric_bound"]
    assert int(conditional["exact_tail_numerator"]) == exact_tail.numerator
    assert int(conditional["exact_tail_denominator"]) == exact_tail.denominator
    assert conditional["maximum_variance"] == {"numerator": "99", "denominator": "25"}
    assert conditional["minimum_success_count"] == 104
    assert conditional["authority"] == "conditional_diagnostic_only"
    assert conditional["conditional_bound_is_below_required"] is True
    assert conditional["actual_v3_grid_power_estimated"] is False
    assert conditional["may_be_used_as_v3_decisive_failure"] is False
    assert current["posthoc_semantics"]["v1_numeric_bound_valid"] is True
    assert current["posthoc_semantics"]["v1_numeric_bound_conditional_only"] is True
    assert current["posthoc_semantics"]["v1_scientific_admission"] is False
    assert current["posthoc_semantics"]["decisive_v3_power_failure"] is False


def test_v2_mechanically_anchors_the_two_opposite_v3_interpretations() -> None:
    scientific_path = relationship_p4_long_context_protocol_path(_EXPECTED_SCIENTIFIC_PROTOCOL_ID_V3)
    scientific_raw = _strict_json(scientific_path)
    power = scientific_raw["analysis"]["power"]

    variance_scenario = "maximum_feasible_bounded_difference_variance_at_planning_mean"
    expected_icc_labels = ["0.00", "0.25", "0.50"]
    expected_cross_labels = [
        "source_structural_covariance",
        "independent_contrasts",
        "equicorrelation_negative_0_10",
        "equicorrelation_positive_0_50",
    ]
    assert variance_scenario in power["mandatory_variance_scenarios"]
    assert power["mandatory_within_root_icc_decimals"] == expected_icc_labels
    assert power["mandatory_cross_contrast_dependence"] == expected_cross_labels
    assert power["all_feasible_cartesian_scenario_combinations_must_pass"] is True

    plus_reference = (1,) * 8
    minus_reference = (-1,) * 8
    plus_comparators = ((-1,) * 8,) * 8
    minus_comparators = ((1,) * 8,) * 8
    plus_differences = tuple(_root_mean_difference(plus_reference, arm) for arm in plus_comparators)
    minus_differences = tuple(_root_mean_difference(minus_reference, arm) for arm in minus_comparators)
    assert set(plus_reference + minus_reference).issubset({-1, 0, 1})
    assert all(set(arm).issubset({-1, 0, 1}) for arm in plus_comparators + minus_comparators)
    assert plus_differences == (Fraction(2, 1),) * 8
    assert minus_differences == (Fraction(-2, 1),) * 8

    probability_plus = Fraction(11, 20)
    probability_minus = Fraction(9, 20)
    contrast_mean = probability_plus * plus_differences[0] + probability_minus * minus_differences[0]
    contrast_variance = (
        probability_plus * (plus_differences[0] - contrast_mean) ** 2
        + probability_minus * (minus_differences[0] - contrast_mean) ** 2
    )
    assert contrast_mean == Fraction(1, 5)
    assert contrast_variance == Fraction(99, 25)
    assert all(item == plus_differences[0] for item in plus_differences)
    assert all(item == minus_differences[0] for item in minus_differences)
    state_weights = (probability_plus, probability_minus)
    first_contrast_first_decision = (
        Fraction(plus_reference[0] - plus_comparators[0][0], 1),
        Fraction(minus_reference[0] - minus_comparators[0][0], 1),
    )
    first_contrast_second_decision = (
        Fraction(plus_reference[1] - plus_comparators[0][1], 1),
        Fraction(minus_reference[1] - minus_comparators[0][1], 1),
    )
    first_contrast_root_mean = (plus_differences[0], minus_differences[0])
    second_contrast_root_mean = (plus_differences[1], minus_differences[1])
    derived_within_root_temporal_correlation = _exact_positive_unit_correlation(
        first_contrast_first_decision,
        first_contrast_second_decision,
        state_weights,
    )
    derived_cross_contrast_correlation = _exact_positive_unit_correlation(
        first_contrast_root_mean,
        second_contrast_root_mean,
        state_weights,
    )

    fixed_icc_match = "1.00" in expected_icc_labels
    fixed_cross_match = "equicorrelation_positive_1_00" in expected_cross_labels
    source_structural_mapping_established = False
    cartesian_first_admission = fixed_icc_match and (fixed_cross_match or source_structural_mapping_established)
    sentinel_first_admission = variance_scenario in power["mandatory_variance_scenarios"]
    absent_precedence_keys = {
        "mandatory_global_joint_sentinels",
        "sentinel_before_cartesian_filtering",
        "mechanical_infeasibility_witness_for_every_skipped_tuple",
        "within_root_icc_random_variable_and_aggregation",
        "cross_contrast_dependence_matrix_mapping",
        "source_structural_covariance_matrix",
    }
    assert absent_precedence_keys.isdisjoint(power)
    assert derived_within_root_temporal_correlation == 1
    assert derived_cross_contrast_correlation == 1
    assert cartesian_first_admission is False
    assert sentinel_first_admission is True

    certificate = _strict_json(_POWER_ARTIFACT_V2 / _V2_CERTIFICATE_FILE)
    derived = certificate["derived_witness_properties"]
    grid = certificate["grid_membership"]
    ambiguity = certificate["ambiguity_witness"]
    assert derived["typed_nine_arm_eight_decision_joint_feasible"] is True
    assert derived["every_contrast_mean"] == {"numerator": "1", "denominator": "5"}
    assert derived["every_contrast_variance"] == {"numerator": "99", "denominator": "25"}
    assert derived["within_root_temporal_correlation"] == {"numerator": "1", "denominator": "1"}
    assert derived["cross_contrast_correlation"] == {"numerator": "1", "denominator": "1"}
    assert derived["source_joint_feasibility"] == "not_evaluated"
    assert grid == {
        "fixed_icc_label_match": False,
        "fixed_cross_dependence_label_match": False,
        "source_structural_mapping_established": False,
        "membership_under_cartesian_first": False,
        "membership_under_sentinel_first": True,
        "membership_identified_by_v3": False,
    }
    assert ambiguity["interpretation_a"]["plus_one_temporal_and_cross_correlation_witness_admitted"] is False
    assert ambiguity["interpretation_b"]["plus_one_temporal_and_cross_correlation_witness_admitted"] is True
    assert ambiguity["both_interpretations_preserve_all_v3_literal_lists"] is True
    assert ambiguity["interpretations_produce_opposite_numeric_bound_applicability"] is True
    assert ambiguity["mechanical_resolution_from_v3_alone_possible"] is False


def test_v2_lineage_terminal_and_zero_output_firewall_are_explicit() -> None:
    certificate = _strict_json(_POWER_ARTIFACT_V2 / _V2_CERTIFICATE_FILE)
    manifest = _strict_json(_POWER_ARTIFACT_V2 / _MANIFEST_FILE)

    scientific_path = relationship_p4_long_context_protocol_path(_EXPECTED_SCIENTIFIC_PROTOCOL_ID_V3)
    assert hashlib.sha256(scientific_path.read_bytes()).hexdigest() == _EXPECTED_SCIENTIFIC_PROTOCOL_RAW_V3
    assert (
        hashlib.sha256((_SCIENTIFIC_PREPARATION_V3 / "scientific_prereg_preparation.json").read_bytes()).hexdigest()
        == _EXPECTED_SCIENTIFIC_PREPARATION_RAW_V3
    )
    assert hashlib.sha256((_SCIENTIFIC_PREPARATION_V3 / _MANIFEST_FILE).read_bytes()).hexdigest() == (
        _EXPECTED_SCIENTIFIC_PREPARATION_MANIFEST_RAW_V3
    )
    assert certificate["identity"] == {
        "power_admission_protocol_id": _EXPECTED_POWER_PROTOCOL_ID_V2,
        "power_admission_protocol_raw_sha256": _EXPECTED_POWER_PROTOCOL_RAW_V2,
        "scientific_protocol_id": _EXPECTED_SCIENTIFIC_PROTOCOL_ID_V3,
        "scientific_protocol_raw_sha256": _EXPECTED_SCIENTIFIC_PROTOCOL_RAW_V3,
        "scientific_preparation_artifact_id": _EXPECTED_SCIENTIFIC_PREPARATION_ID_V3,
        "historical_power_bound_protocol_id": _EXPECTED_POWER_PROTOCOL_ID_V1,
        "historical_power_bound_protocol_raw_sha256": _EXPECTED_POWER_PROTOCOL_RAW_V1,
        "historical_power_failure_artifact_id": _EXPECTED_POWER_ARTIFACT_ID_V1,
        "historical_power_failure_certificate_id": _EXPECTED_POWER_CERTIFICATE_ID_V1,
        "historical_power_failure_certificate_raw_sha256": _EXPECTED_POWER_CERTIFICATE_RAW_V1,
        "historical_power_failure_manifest_raw_sha256": _EXPECTED_POWER_MANIFEST_RAW_V1,
    }

    terminal = certificate["terminal"]
    assert terminal["certificate_valid"] is True
    assert terminal["v1_numeric_calculation_valid_conditionally"] is True
    assert terminal["v1_unconditional_scientific_admission_valid"] is False
    assert terminal["v3_power_passed"] is False
    assert terminal["v3_power_failed_under_frozen_grid"] is None
    assert terminal["v3_power_contract_resolved"] is False
    assert terminal["v3_prior_power_admission_satisfied"] is False
    assert terminal["v3_development_authorized"] is False
    assert terminal["model_output_authorized"] is False
    assert terminal["stopping_basis"] == "unresolved_prior_power_contract_not_numeric_fail"
    admission = certificate["admission_logic"]
    assert admission["v3_power_contract_is_mechanically_determinate"] is False
    assert admission["numeric_bound_unconditionally_applies_to_v3_frozen_grid"] is False
    assert admission["under_specification_may_authorize_development"] is False

    zero_count_fields = (
        "model_output_count",
        "subject_materialization_count",
        "source_materialization_count",
        "donor_bank_materialization_count",
        "counterfactual_twin_materialization_count",
        "cuda_formal_run_count",
        "simulation_replicate_count",
        "full_joint_dgp_artifact_count",
    )
    for field in zero_count_fields:
        assert certificate["zero_output_firewall"][field] == 0
        assert manifest[field] == 0
    for field in (
        "execution_enabled",
        "development_authorized",
        "qualification_authorized",
        "formal_authorized",
    ):
        assert certificate["zero_output_firewall"][field] is False
        assert manifest[field] is False
    assert manifest["conditional_numeric_bound_only"] is True
    assert manifest["power_contract_determinate"] is False
    assert manifest["v1_unconditional_scientific_admission_valid"] is False
    assert manifest["v3_power_passed"] is False
    assert manifest["v3_power_failed_under_frozen_grid"] is None


def test_v2_prepare_is_deterministic_create_only_and_v1_cannot_republish(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "power admission"
    result = prepare_relationship_p4_long_context_power_admission_certificate(
        output_dir=output,
        preparation_dir=_SCIENTIFIC_PREPARATION_V3,
    )
    assert result.artifact_id == _EXPECTED_POWER_ARTIFACT_ID_V2
    assert hashlib.sha256((output / _V2_CERTIFICATE_FILE).read_bytes()).hexdigest() == (
        _EXPECTED_POWER_CERTIFICATE_RAW_V2
    )
    assert hashlib.sha256((output / _MANIFEST_FILE).read_bytes()).hexdigest() == _EXPECTED_POWER_MANIFEST_RAW_V2
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_power_admission_certificate(
            output_dir=output,
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
        )

    preexisting = tmp_path / "preexisting"
    preexisting.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_power_admission_certificate(
            output_dir=preexisting,
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
        )

    forbidden_v1_output = tmp_path / "v1-republication"
    with pytest.raises(ValueError, match="superseded and cannot be republished"):
        prepare_relationship_p4_long_context_power_failure_certificate(
            output_dir=forbidden_v1_output,
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
        )
    assert not forbidden_v1_output.exists()


def test_falsey_paths_ids_cross_version_and_missing_roots_fail_loudly(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="unregistered protocol id"):
        relationship_p4_long_context_protocol_path("")
    with pytest.raises(TypeError, match="protocol id must be text"):
        relationship_p4_long_context_protocol_path(False)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="regular file"):
        validate_relationship_p4_long_context_power_admission_certificate(
            output_dir=_POWER_ARTIFACT_V2,
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
            protocol_path="",  # type: ignore[arg-type]
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_relationship_p4_long_context_power_admission_certificate(
            output_dir=_POWER_ARTIFACT_V2,
            preparation_dir="",  # type: ignore[arg-type]
        )
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_power_admission_certificate(
            output_dir="",  # type: ignore[arg-type]
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
        )

    scientific_v2 = relationship_p4_long_context_protocol_path(_EXPECTED_SCIENTIFIC_PROTOCOL_ID_V2)
    with pytest.raises(ValueError, match="defined only for v3"):
        validate_relationship_p4_long_context_power_admission_certificate(
            output_dir=_POWER_ARTIFACT_V2,
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
            protocol_path=scientific_v2,
        )
    with pytest.raises(ValueError, match="keys drift|protocol id mismatch"):
        validate_relationship_p4_long_context_power_admission_certificate(
            output_dir=_POWER_ARTIFACT_V2,
            preparation_dir=_SCIENTIFIC_PREPARATION_V2,
        )
    with pytest.raises(FileNotFoundError):
        validate_relationship_p4_long_context_power_admission_certificate(
            output_dir=tmp_path / "missing",
            preparation_dir=_SCIENTIFIC_PREPARATION_V3,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("identity", "scientific_protocol_id"), _EXPECTED_SCIENTIFIC_PROTOCOL_ID_V2),
        (("identity", "historical_power_failure_artifact_id"), "a" * 64),
        (
            (
                "ambiguity_witness",
                "interpretation_a",
                "plus_one_temporal_and_cross_correlation_witness_admitted",
            ),
            True,
        ),
        (("terminal", "v3_power_failed_under_frozen_grid"), True),
        (("terminal", "v1_unconditional_scientific_admission_valid"), True),
        (("zero_output_firewall", "model_output_count"), 1),
        (("conditional_numeric_bound", "minimum_success_count"), True),
    ],
)
def test_v2_validator_rejects_splice_semantic_tamper_and_bool_as_int_after_resigning(
    field_path: tuple[str, ...],
    replacement: object,
    tmp_path: pathlib.Path,
) -> None:
    artifact = _copy_v2_artifact(tmp_path, "resigned")
    _resign_certificate(
        artifact,
        lambda payload: _set_nested(payload, field_path, replacement),
    )

    with pytest.raises((TypeError, ValueError)):
        _validate_v2(artifact)


def test_v2_validator_rejects_raw_tamper_manifest_resign_and_format_attacks(tmp_path: pathlib.Path) -> None:
    raw_tamper = _copy_v2_artifact(tmp_path, "raw-tamper")
    certificate_path = raw_tamper / _V2_CERTIFICATE_FILE
    certificate = _strict_json(certificate_path)
    certificate["terminal"]["v3_development_authorized"] = True
    certificate_path.write_bytes(_canonical_bytes(certificate))
    with pytest.raises(ValueError, match="certificate id drift"):
        _validate_v2(raw_tamper)

    manifest_resign = _copy_v2_artifact(tmp_path, "manifest-resign")
    manifest_path = manifest_resign / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest["development_authorized"] = True
    _resign_manifest(manifest_path, manifest)
    with pytest.raises((TypeError, ValueError), match="development_authorized|manifest"):
        _validate_v2(manifest_resign)

    extra = _copy_v2_artifact(tmp_path, "extra")
    (extra / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set drift"):
        _validate_v2(extra)

    noncanonical = _copy_v2_artifact(tmp_path, "noncanonical")
    noncanonical_path = noncanonical / _V2_CERTIFICATE_FILE
    noncanonical_value = _strict_json(noncanonical_path)
    noncanonical_path.write_text(
        json.dumps(noncanonical_value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        _validate_v2(noncanonical)

    reordered = _copy_v2_artifact(tmp_path, "reordered")
    reordered_path = reordered / _MANIFEST_FILE
    reordered_manifest = _strict_json(reordered_path)
    reordered_path.write_text(
        json.dumps(dict(reversed(tuple(reordered_manifest.items()))), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        _validate_v2(reordered)

    bom = _copy_v2_artifact(tmp_path, "bom")
    bom_path = bom / _V2_CERTIFICATE_FILE
    bom_path.write_bytes(b"\xef\xbb\xbf" + bom_path.read_bytes())
    with pytest.raises(ValueError, match="must not carry a UTF-8 BOM"):
        _validate_v2(bom)

    duplicate = _copy_v2_artifact(tmp_path, "duplicate")
    duplicate_path = duplicate / _V2_CERTIFICATE_FILE
    duplicate_path.write_text(
        duplicate_path.read_text(encoding="utf-8").replace(
            "{",
            '{"schema_version":"duplicate",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _validate_v2(duplicate)


def test_v2_protocol_raw_pin_rejects_key_reordering_and_content_resign(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    protocol = _strict_json(_POWER_PROTOCOL_V2)
    reordered = tmp_path / "reordered-protocol.json"
    reordered.write_text(
        json.dumps(dict(reversed(tuple(protocol.items()))), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(owner, "_POWER_BOUND_PROTOCOL_PATH_V2", reordered)
    with pytest.raises(ValueError, match="raw bytes drift"):
        _validate_v2(_POWER_ARTIFACT_V2)

    drifted = tmp_path / "drifted-protocol.json"
    protocol["admission_logic"]["under_specification_may_authorize_development"] = True
    drifted.write_bytes(_canonical_bytes(protocol))
    monkeypatch.setattr(owner, "_POWER_BOUND_PROTOCOL_PATH_V2", drifted)
    with pytest.raises(ValueError, match="raw bytes drift"):
        _validate_v2(_POWER_ARTIFACT_V2)


def test_v2_artifact_reparse_roots_and_file_symlinks_fail(tmp_path: pathlib.Path) -> None:
    artifact_alias = tmp_path / "artifact-alias"
    _create_directory_alias_or_skip(artifact_alias, _POWER_ARTIFACT_V2)
    try:
        with pytest.raises(ValueError, match="symlink|reparse point"):
            _validate_v2(artifact_alias)
    finally:
        _remove_directory_alias(artifact_alias)

    file_alias_root = _copy_v2_artifact(tmp_path, "file-alias")
    file_alias = file_alias_root / _V2_CERTIFICATE_FILE
    file_alias.unlink()
    try:
        file_alias.symlink_to(_POWER_ARTIFACT_V2 / _V2_CERTIFICATE_FILE)
    except OSError as exc:
        pytest.skip(f"file symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="regular files"):
        _validate_v2(file_alias_root)


def test_v2_dangling_output_reparse_point_fails_without_publishing(tmp_path: pathlib.Path) -> None:
    missing_target = tmp_path / "missing-target"
    output_alias = tmp_path / "output-alias"
    _create_directory_alias_or_skip(output_alias, missing_target)
    try:
        with pytest.raises(ValueError, match="symlink|reparse point"):
            prepare_relationship_p4_long_context_power_admission_certificate(
                output_dir=output_alias,
                preparation_dir=_SCIENTIFIC_PREPARATION_V3,
            )
    finally:
        _remove_directory_alias(output_alias)
    assert not missing_target.exists()


def test_owner_and_cli_have_no_heavy_runtime_or_execution_override_surface() -> None:
    forbidden_roots = {"torch", "transformers", "numpy", "subprocess", "cuda", "cupy"}
    for source_path in (_OWNER_SOURCE, _CLI_SOURCE):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert _import_roots(tree).isdisjoint(forbidden_roots)
        assert not any(isinstance(node, ast.Name) and node.id.lower() in forbidden_roots for node in ast.walk(tree))

    cli_source = _CLI_SOURCE.read_text(encoding="utf-8")
    cli_tree = ast.parse(cli_source)
    option_strings = {
        argument.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("--")
    }
    parser_commands = {
        node.args[0].value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert option_strings == {"--output-dir", "--preparation-dir"}
    assert parser_commands == {"prepare", "validate-existing", "validate-v1-existing"}
    assert "--force" not in cli_source
    assert "override" not in cli_source.lower()
    assert "run-session" not in cli_source


def _assert_protocol_identity(path: pathlib.Path, *, expected_id: str, expected_raw: str) -> None:
    payload = path.read_bytes()
    value = _strict_json(path)
    assert hashlib.sha256(payload).hexdigest() == expected_raw
    assert hashlib.sha256(_canonical_bytes(value)).hexdigest() == expected_id


def _assert_artifact_identity(
    root: pathlib.Path,
    *,
    certificate_file: str,
    expected_artifact_id: str,
    expected_certificate_id: str,
    expected_certificate_raw: str,
    expected_manifest_raw: str,
) -> None:
    certificate_path = root / certificate_file
    certificate_bytes = certificate_path.read_bytes()
    certificate = _strict_json(certificate_path)
    assert certificate_bytes == _canonical_bytes(certificate)
    assert hashlib.sha256(certificate_bytes).hexdigest() == expected_certificate_raw
    certificate_core = dict(certificate)
    assert certificate_core.pop("certificate_id") == expected_certificate_id
    assert hashlib.sha256(_canonical_bytes(certificate_core)).hexdigest() == expected_certificate_id

    manifest_path = root / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_json(manifest_path)
    assert manifest_bytes == _canonical_bytes(manifest)
    assert hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_raw
    manifest_core = dict(manifest)
    assert manifest_core.pop("artifact_id") == expected_artifact_id
    assert hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest() == expected_artifact_id
    assert manifest["certificate_id"] == expected_certificate_id
    assert manifest["files"] == [
        {
            "path": certificate_file,
            "byte_count": len(certificate_bytes),
            "sha256": expected_certificate_raw,
        }
    ]


def _root_mean_difference(reference: tuple[int, ...], comparator: tuple[int, ...]) -> Fraction:
    assert len(reference) == len(comparator) == 8
    return Fraction(sum(reference) - sum(comparator), len(reference))


def _exact_positive_unit_correlation(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
    weights: tuple[Fraction, Fraction],
) -> Fraction:
    assert sum(weights) == 1
    left_mean = sum(weight * value for weight, value in zip(weights, left, strict=True))
    right_mean = sum(weight * value for weight, value in zip(weights, right, strict=True))
    covariance = sum(
        weight * (left_value - left_mean) * (right_value - right_mean)
        for weight, left_value, right_value in zip(weights, left, right, strict=True)
    )
    left_variance = sum(weight * (value - left_mean) ** 2 for weight, value in zip(weights, left, strict=True))
    right_variance = sum(weight * (value - right_mean) ** 2 for weight, value in zip(weights, right, strict=True))
    assert covariance == left_variance == right_variance
    assert covariance > 0
    return covariance / left_variance


def _validate_v2(path: pathlib.Path) -> object:
    return validate_relationship_p4_long_context_power_admission_certificate(
        output_dir=path,
        preparation_dir=_SCIENTIFIC_PREPARATION_V3,
    )


def _copy_v2_artifact(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    destination = tmp_path / name
    shutil.copytree(_POWER_ARTIFACT_V2, destination)
    return destination


def _set_nested(payload: dict[str, Any], field_path: tuple[str, ...], replacement: object) -> None:
    current: dict[str, Any] = payload
    for field in field_path[:-1]:
        current = current[field]
    current[field_path[-1]] = replacement


def _resign_certificate(root: pathlib.Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    certificate_path = root / _V2_CERTIFICATE_FILE
    certificate = _strict_json(certificate_path)
    mutator(certificate)
    certificate.pop("certificate_id")
    certificate["certificate_id"] = hashlib.sha256(_canonical_bytes(certificate)).hexdigest()
    certificate_bytes = _canonical_bytes(certificate)
    certificate_path.write_bytes(certificate_bytes)

    manifest_path = root / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest["certificate_id"] = certificate["certificate_id"]
    manifest["files"] = [
        {
            "path": _V2_CERTIFICATE_FILE,
            "byte_count": len(certificate_bytes),
            "sha256": hashlib.sha256(certificate_bytes).hexdigest(),
        }
    ]
    _resign_manifest(manifest_path, manifest)


def _resign_manifest(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    manifest.pop("artifact_id")
    manifest["artifact_id"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    path.write_bytes(_canonical_bytes(manifest))


def _strict_json(path: pathlib.Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=reject_duplicates)
    assert type(value) is dict
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _import_roots(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            result.add(node.module.split(".", 1)[0])
    return result


def _create_directory_alias_or_skip(alias: pathlib.Path, target: pathlib.Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"directory junction creation is unavailable: {completed.stderr.strip()}")
        return
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")


def _remove_directory_alias(alias: pathlib.Path) -> None:
    if not os.path.lexists(alias):
        return
    if os.name == "nt":
        alias.rmdir()
    else:
        alias.unlink()
