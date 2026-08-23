from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
import pathlib
import shutil
import subprocess
from typing import Any, Callable

import pytest

import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign as owner
import lifeform_evolution.relationship_lab_p4_long_context_v4_planning_derivation as derivation
from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
    load_relationship_p4_long_context_v4_planning_protocol,
    prepare_relationship_p4_long_context_v4_zero_output_plan,
    validate_relationship_p4_long_context_v4_zero_output_plan,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "packages" / "lifeform-evolution"
_OWNER_SOURCE = _PACKAGE_ROOT / "src" / "lifeform_evolution" / "relationship_lab_p4_long_context_causal_campaign.py"
_HELPER_SOURCE = (
    _PACKAGE_ROOT / "src" / "lifeform_evolution" / "relationship_lab_p4_long_context_v4_planning_derivation.py"
)
_PROTOCOL_PATH = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "protocols"
    / "relationship_p4_long_context_v4_planning_contract_v1.json"
)
_V3_PROTOCOL_PATH = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "protocols"
    / "relationship_p4_independent_long_context_causal_campaign_v3.json"
)
_V2_ADMISSION_PROTOCOL_PATH = (
    _PACKAGE_ROOT / "src" / "lifeform_evolution" / "protocols" / "relationship_p4_long_context_power_admission_v2.json"
)
_CLI_SOURCE = _REPO_ROOT / "scripts" / "run_relationship_lab_p4_long_context_v4_planning.py"
_V3_PREPARATION = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_causal_campaign_design_prereg_v3_20260823"
)
_V2_ADMISSION = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_power_admission_v2_under_specified_20260823"
)
_V4_ARTIFACT = (
    _REPO_ROOT / "artifacts" / "relationship_lab" / "p4_independent_long_context_v4a_zero_output_planning_20260823"
)

_PLAN_FILE = "v4_zero_output_plan.json"
_SCHEDULE_FILE = "development_candidate_cell_schedule.json"
_SCREEN_TABLE_FILE = "sentinel_necessary_point_screens.json"
_MANIFEST_FILE = "manifest.json"

_EXPECTED_PROTOCOL_ID = "63e007b7d43bb152e5891162d6567c4edd4396af99cf1c5525c28d0be4c08753"
_EXPECTED_PROTOCOL_RAW = "d06b07101624b3996bd712c98d3c633b7b00af7a878912817b5149a199c00e0a"
_EXPECTED_HELPER_RAW = "bf38e7ab89c56bdae8844f533cac077443d157a793c698adbb11a9591e32a0ef"
_EXPECTED_ARTIFACT_ID = "082454002260db90b7236a1104311a5d92cc3959171bb3190e7a30f8387e56c1"
_EXPECTED_CERTIFICATE_ID = "b7e95f149afe77b283bf135f7cb5d76eb4f4edee4594c8649a778acb4186c764"
_EXPECTED_PLAN_RAW = "9e17383f416eea555799d7e603996a34d526c7c20e9e65e53af25196a700064f"
_EXPECTED_SCREEN_TABLE_RAW = "d8f0f6b4fa1927138007bac77b687f3507b09ca0f000c6549b584ba2d33b01ba"
_EXPECTED_SCHEDULE_RAW = "df426477209d0e99c74cf62938fcf3700554c6242f9439c2e51ebdd20edf1d6f"
_EXPECTED_MANIFEST_RAW = "26b46683260dc01f632ff9c1874839760f4b075c53eb5cd0298c7fc025633e3e"

_EXPECTED_V3_PROTOCOL_ID = "9f352778e128a9573790762222a05225740bdaeb732800dec0eec124116a282d"
_EXPECTED_V3_PROTOCOL_RAW = "ea8a17a14a68802d3b60586bf520c9137e6920be4112c951ec8c69f5e6ea359e"
_EXPECTED_V3_PREPARATION_ID = "c5a708ae5e68261fddbade165b45579e66e4bbe7db1be1f4a83056561a17f42e"
_EXPECTED_V3_PREPARATION_RAW = "a4b2f3ee920e398ae0f7eab5757b7988dc3fa4f7db7b4769599c837bc656bcd6"
_EXPECTED_V3_MANIFEST_RAW = "4aaf10d76b80a780e62a17b62803906cc34dd34d64bb4d0f8b96d60dff1e2663"
_EXPECTED_V2_PROTOCOL_ID = "67d294faf9209c9d05334f4c0e87371676c9821b7c12e603f3e289f33f566bc9"
_EXPECTED_V2_PROTOCOL_RAW = "130f766787ec0b02bd5857344e58b371d996aa51fabe421cbfbde05347fd0e04"
_EXPECTED_V2_ARTIFACT_ID = "9883e10784a06260a220a6fdbf72141b1300c21e97faee6e84a401c40a144ee9"
_EXPECTED_V2_CERTIFICATE_ID = "cd6ceca086a1d8a311c75bdacd70c976e05b90dff2cde55b3ad41c00d29936b3"
_EXPECTED_V2_CERTIFICATE_RAW = "0f20e47da67e5ebaed39e63d274805d5783cc588e0f8aa5fa2a0450b079d18ba"
_EXPECTED_V2_MANIFEST_RAW = "f6fb9d482c8eb7f7e5e8dd92546a12bbae6f22ea76730c205f41fba4e14b4972"

_EXPECTED_CELLS = (
    "qwen_steelman_full_history::candidate_0",
    "qwen_steelman_full_history::candidate_1",
    "qwen_steelman_full_history::candidate_2",
    "qwen_steelman_selective_rag::candidate_0",
    "qwen_steelman_selective_rag::candidate_1",
    "qwen_steelman_selective_rag::candidate_2",
)
_EXPECTED_CLASSIFICATIONS = {
    "one_valid_candidate": (
        "valid_generated_action",
        "lookup_committed_utility_vector",
    ),
    "authentic_zero_candidate_bytes": (
        "substantive_invalid_generated_action",
        "typed_invalid_action_utility_minus_one_in_itt",
    ),
    "authentic_multiple_candidates": (
        "substantive_invalid_generated_action",
        "typed_invalid_action_utility_minus_one_in_itt",
    ),
    "authentic_single_out_of_domain_candidate": (
        "substantive_invalid_generated_action",
        "typed_invalid_action_utility_minus_one_in_itt",
    ),
    "generation_attempt_did_not_complete": (
        "technical_missingness",
        "contrast_specific_worst_case_itt",
    ),
    "incomplete_generation_without_authenticated_failure_receipt": (
        "integrity_failure",
        "invalid_attempt_no_claim",
    ),
    "completed_attempt_without_valid_generated_bytes_receipt": (
        "integrity_failure",
        "invalid_attempt_no_claim",
    ),
    "delivered_bytes_tampered": (
        "integrity_failure",
        "invalid_attempt_no_claim",
    ),
    "parser_artifact_or_execution_drift": (
        "integrity_failure",
        "invalid_attempt_no_claim",
    ),
    "missing_or_fake_generation_parent": (
        "integrity_failure",
        "invalid_attempt_no_claim",
    ),
    "utility_commitment_or_reobserver_drift": (
        "integrity_failure",
        "invalid_attempt_no_claim",
    ),
}


def test_v4_protocol_helper_and_published_artifact_are_independent_literal_pins() -> None:
    protocol_bytes = _PROTOCOL_PATH.read_bytes()
    protocol = _strict_json(_PROTOCOL_PATH)
    assert hashlib.sha256(protocol_bytes).hexdigest() == _EXPECTED_PROTOCOL_RAW
    assert hashlib.sha256(_canonical_bytes(protocol)).hexdigest() == _EXPECTED_PROTOCOL_ID
    assert hashlib.sha256(_HELPER_SOURCE.read_bytes()).hexdigest() == _EXPECTED_HELPER_RAW
    assert owner.P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1 == _EXPECTED_PROTOCOL_ID
    assert owner.P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1 == _EXPECTED_PROTOCOL_RAW
    assert owner.P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1 == _EXPECTED_ARTIFACT_ID

    plan_path = _V4_ARTIFACT / _PLAN_FILE
    schedule_path = _V4_ARTIFACT / _SCHEDULE_FILE
    screen_table_path = _V4_ARTIFACT / _SCREEN_TABLE_FILE
    manifest_path = _V4_ARTIFACT / _MANIFEST_FILE
    plan_bytes = plan_path.read_bytes()
    schedule_bytes = schedule_path.read_bytes()
    screen_table_bytes = screen_table_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    plan = _strict_json(plan_path)
    schedule = _strict_json(schedule_path)
    screen_table = _strict_json(screen_table_path)
    manifest = _strict_json(manifest_path)

    assert plan_bytes == _canonical_bytes(plan)
    assert schedule_bytes == _canonical_bytes(schedule)
    assert screen_table_bytes == _canonical_bytes(screen_table)
    assert manifest_bytes == _canonical_bytes(manifest)
    assert hashlib.sha256(plan_bytes).hexdigest() == _EXPECTED_PLAN_RAW
    assert hashlib.sha256(schedule_bytes).hexdigest() == _EXPECTED_SCHEDULE_RAW
    assert hashlib.sha256(screen_table_bytes).hexdigest() == _EXPECTED_SCREEN_TABLE_RAW
    assert hashlib.sha256(manifest_bytes).hexdigest() == _EXPECTED_MANIFEST_RAW

    plan_core = dict(plan)
    assert plan_core.pop("certificate_id") == _EXPECTED_CERTIFICATE_ID
    assert hashlib.sha256(_canonical_bytes(plan_core)).hexdigest() == _EXPECTED_CERTIFICATE_ID
    manifest_core = dict(manifest)
    assert manifest_core.pop("artifact_id") == _EXPECTED_ARTIFACT_ID
    assert hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest() == _EXPECTED_ARTIFACT_ID
    assert manifest["certificate_id"] == _EXPECTED_CERTIFICATE_ID
    assert manifest["files"] == [
        {
            "path": _PLAN_FILE,
            "byte_count": len(plan_bytes),
            "sha256": _EXPECTED_PLAN_RAW,
        },
        {
            "path": _SCHEDULE_FILE,
            "byte_count": len(schedule_bytes),
            "sha256": _EXPECTED_SCHEDULE_RAW,
        },
        {
            "path": _SCREEN_TABLE_FILE,
            "byte_count": len(screen_table_bytes),
            "sha256": _EXPECTED_SCREEN_TABLE_RAW,
        },
    ]

    loaded = load_relationship_p4_long_context_v4_planning_protocol()
    assert loaded.protocol_id == _EXPECTED_PROTOCOL_ID
    assert loaded.candidate_root_counts == tuple(range(192, 8193, 64))
    assert loaded.first_necessary_screen_passing_root_count == 1088
    assert loaded.first_positive_mean_gate_capable_root_count == 1856
    assert loaded.cartesian_candidate_tuple_count == 576
    assert loaded.candidate_cell_ids == _EXPECTED_CELLS
    assert loaded.schedule_block_count == 640
    assert loaded.power_contract_determinate is True
    assert loaded.source_grid_resolved is False
    assert loaded.selected_formal_root_count is None

    validated = _validate_v4(_V4_ARTIFACT)
    assert validated.artifact_id == _EXPECTED_ARTIFACT_ID
    assert validated.protocol_id == _EXPECTED_PROTOCOL_ID
    assert validated.scientific_v3_protocol_id == _EXPECTED_V3_PROTOCOL_ID
    assert validated.power_admission_v2_artifact_id == _EXPECTED_V2_ARTIFACT_ID
    assert validated.first_necessary_screen_passing_root_count == 1088
    assert validated.first_positive_mean_gate_capable_root_count == 1856
    assert validated.cartesian_candidate_tuple_count == 576
    assert validated.candidate_schedule_block_count == 640
    assert validated.power_contract_determinate is True
    assert validated.source_grid_resolved is False
    assert validated.selected_formal_root_count is None
    assert validated.source_materialization_authorized is False
    assert validated.development_authorized is False
    assert validated.model_output_authorized is False
    assert validated.formal_authorized is False


def test_v4_lineage_and_zero_output_firewalls_bind_real_tracked_inputs() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    plan = _strict_json(_V4_ARTIFACT / _PLAN_FILE)
    schedule = _strict_json(_V4_ARTIFACT / _SCHEDULE_FILE)
    manifest = _strict_json(_V4_ARTIFACT / _MANIFEST_FILE)
    lineage = protocol["input_lineage"]

    assert lineage == {
        "scientific_v3_protocol_id": _EXPECTED_V3_PROTOCOL_ID,
        "scientific_v3_protocol_raw_sha256": _EXPECTED_V3_PROTOCOL_RAW,
        "scientific_v3_preparation_artifact_id": _EXPECTED_V3_PREPARATION_ID,
        "scientific_v3_preparation_raw_sha256": _EXPECTED_V3_PREPARATION_RAW,
        "scientific_v3_preparation_manifest_raw_sha256": _EXPECTED_V3_MANIFEST_RAW,
        "power_admission_v2_protocol_id": _EXPECTED_V2_PROTOCOL_ID,
        "power_admission_v2_protocol_raw_sha256": _EXPECTED_V2_PROTOCOL_RAW,
        "power_admission_v2_artifact_id": _EXPECTED_V2_ARTIFACT_ID,
        "power_admission_v2_certificate_id": _EXPECTED_V2_CERTIFICATE_ID,
        "power_admission_v2_certificate_raw_sha256": _EXPECTED_V2_CERTIFICATE_RAW,
        "power_admission_v2_manifest_raw_sha256": _EXPECTED_V2_MANIFEST_RAW,
        "v4_planning_derivation_helper_raw_sha256": _EXPECTED_HELPER_RAW,
        "v3_terminal_status": "power_contract_under_specified_no_development_authorization",
        "v3_power_failed_under_frozen_grid": None,
        "v3_power_passed": False,
    }
    assert hashlib.sha256(_V3_PROTOCOL_PATH.read_bytes()).hexdigest() == _EXPECTED_V3_PROTOCOL_RAW
    assert hashlib.sha256(_V2_ADMISSION_PROTOCOL_PATH.read_bytes()).hexdigest() == _EXPECTED_V2_PROTOCOL_RAW
    assert (
        hashlib.sha256((_V3_PREPARATION / "scientific_prereg_preparation.json").read_bytes()).hexdigest()
        == _EXPECTED_V3_PREPARATION_RAW
    )
    assert hashlib.sha256((_V3_PREPARATION / _MANIFEST_FILE).read_bytes()).hexdigest() == _EXPECTED_V3_MANIFEST_RAW
    assert hashlib.sha256((_V2_ADMISSION / "power_admission_certificate.json").read_bytes()).hexdigest() == (
        _EXPECTED_V2_CERTIFICATE_RAW
    )
    assert hashlib.sha256((_V2_ADMISSION / _MANIFEST_FILE).read_bytes()).hexdigest() == _EXPECTED_V2_MANIFEST_RAW

    assert plan["identity"] == {
        "v4_planning_protocol_id": _EXPECTED_PROTOCOL_ID,
        "v4_planning_protocol_raw_sha256": _EXPECTED_PROTOCOL_RAW,
        "v4_planning_derivation_helper_raw_sha256": _EXPECTED_HELPER_RAW,
        "scientific_v3_protocol_id": _EXPECTED_V3_PROTOCOL_ID,
        "scientific_v3_preparation_artifact_id": _EXPECTED_V3_PREPARATION_ID,
        "power_admission_v2_protocol_id": _EXPECTED_V2_PROTOCOL_ID,
        "power_admission_v2_artifact_id": _EXPECTED_V2_ARTIFACT_ID,
    }
    assert schedule["model_output_count"] == 0
    assert schedule["selected_candidate_id"] is None
    assert manifest["source_materialization_authorized"] is False
    assert manifest["development_authorized"] is False
    assert manifest["model_output_authorized"] is False
    assert manifest["qualification_authorized"] is False
    assert manifest["formal_authorized"] is False
    assert manifest["selected_formal_root_count"] is None
    for field in (
        "source_structural_artifact_count",
        "full_joint_dgp_artifact_count",
        "power_search_replicate_count",
        "power_confirmation_replicate_count",
        "subject_materialization_count",
        "baseline_output_count",
        "model_output_count",
        "cuda_planner_run_count",
        "cuda_formal_run_count",
        "empirical_outcome_count",
    ):
        assert manifest[field] == 0

    firewall = plan["zero_output_firewall"]
    assert firewall["planning_protocol_count"] == 1
    for field, value in firewall.items():
        if field.endswith("_count") and field != "planning_protocol_count":
            assert value == 0
        elif field.endswith("_supported") or field.endswith("_authorized") or field.endswith("_claimed"):
            assert value is False
    source = plan["source_preflight_contract"]
    assert source["source_structural_inventory_materialization_authorized_by_this_protocol"] is False
    assert source["concrete_subject_pack_materialization_authorized"] is False
    assert source["donor_bank_materialization_authorized"] is False
    assert source["counterfactual_twin_materialization_authorized"] is False
    assert source["baseline_or_model_generation_authorized"] is False
    assert source["source_preflight_completed"] is False
    assert source["full_joint_dgp_artifact_completed"] is False
    assert source["cuda_planner_execution_authorized_by_this_protocol"] is False


def test_global_sentinel_and_126_candidate_screen_are_independently_recomputed() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    plan = _strict_json(_V4_ARTIFACT / _PLAN_FILE)
    units = protocol["scientific_units"]
    sentinel = protocol["mandatory_global_joint_sentinels"]["sentinels"][0]
    primitive = sentinel["primitive_joint_distribution"]

    masses = (Fraction(11, 20), Fraction(9, 20))
    state_reference = (
        primitive["plus_state_reference_utility_each_decision"],
        primitive["minus_state_reference_utility_each_decision"],
    )
    state_comparator = (
        primitive["plus_state_every_comparator_utility_each_decision"],
        primitive["minus_state_every_comparator_utility_each_decision"],
    )
    arm_count = units["arm_count"]
    decision_count = units["evaluation_decisions_per_arm"]
    state_tensors = tuple(
        (
            (reference,) * decision_count,
            *((comparator,) * decision_count for _ in range(arm_count - 1)),
        )
        for reference, comparator in zip(state_reference, state_comparator, strict=True)
    )
    state_contrasts = tuple(
        tuple(tuple(tensor[0][turn] - comparator[turn] for turn in range(decision_count)) for comparator in tensor[1:])
        for tensor in state_tensors
    )
    root_contrast_values = tuple(
        tuple(Fraction(sum(contrast), decision_count) for contrast in contrasts) for contrasts in state_contrasts
    )
    one_contrast = tuple(row[0] for row in root_contrast_values)
    mean = sum(mass * value for mass, value in zip(masses, one_contrast, strict=True))
    variance = sum(mass * (value - mean) ** 2 for mass, value in zip(masses, one_contrast, strict=True))
    assert set(itertools.chain.from_iterable(itertools.chain.from_iterable(state_tensors))) == {-1, 1}
    assert len(state_tensors) == 2
    assert all(len(tensor) == 9 and all(len(arm) == 8 for arm in tensor) for tensor in state_tensors)
    assert one_contrast == (Fraction(2), Fraction(-2))
    assert mean == Fraction(1, 5)
    assert variance == Fraction(99, 25)

    temporal_correlations = []
    for contrast_index in range(8):
        for left_turn in range(8):
            for right_turn in range(left_turn + 1, 8):
                left = tuple(state_contrasts[state][contrast_index][left_turn] for state in range(2))
                right = tuple(state_contrasts[state][contrast_index][right_turn] for state in range(2))
                temporal_correlations.append(_weighted_correlation(left, right, masses))
    cross_correlations = []
    for left_contrast in range(8):
        for right_contrast in range(left_contrast + 1, 8):
            left = tuple(row[left_contrast] for row in root_contrast_values)
            right = tuple(row[right_contrast] for row in root_contrast_values)
            cross_correlations.append(_weighted_correlation(left, right, masses))
    assert temporal_correlations == [Fraction(1)] * (8 * math.comb(8, 2))
    assert cross_correlations == [Fraction(1)] * math.comb(8, 2)

    candidate_spec = protocol["full_joint_power_planner"]["candidate_formal_root_counts"]
    candidates = tuple(range(candidate_spec["first"], candidate_spec["last_inclusive"] + 1, candidate_spec["step"]))
    assert len(candidates) == 126
    assert candidates[0] == 192
    assert candidates[-1] == 8192
    screens = []
    for root_count in candidates:
        minimum_plus_count = _ceil_fraction(Fraction(43 * root_count, 80))
        exact_power = _exact_binomial_upper_tail(
            trials=root_count,
            success_numerator=11,
            probability_denominator=20,
            minimum_successes=minimum_plus_count,
        )
        screens.append((root_count, minimum_plus_count, exact_power))
    assert len(screens) == 126
    helper_screens = derivation.derive_necessary_point_screens(
        candidate_root_counts=candidates,
        mass_at_plus_two=Fraction(11, 20),
        practical_gate=Fraction(3, 20),
        required_power=Fraction(4, 5),
    )
    assert tuple(
        (item.root_count, item.minimum_plus_count, item.power, item.passed) for item in helper_screens
    ) == tuple(
        (root_count, minimum_plus_count, power, power >= Fraction(4, 5))
        for root_count, minimum_plus_count, power in screens
    )
    screen_table_path = _V4_ARTIFACT / _SCREEN_TABLE_FILE
    screen_table = _strict_json(screen_table_path)
    assert screen_table_path.read_bytes() == _canonical_bytes(screen_table)
    assert screen_table["protocol_id"] == _EXPECTED_PROTOCOL_ID
    assert screen_table["candidate_count"] == 126
    assert screen_table["required_power"] == {"numerator": "4", "denominator": "5"}
    assert screen_table["exact_fraction_encoding"] == (
        "reduced_positive_numerator_and_denominator_as_canonical_lowercase_hex_without_prefix_or_leading_zero"
    )
    assert screen_table["each_candidate_evaluated_independently"] is True
    assert screen_table["monotonicity_shortcut_permitted"] is False
    assert screen_table["model_output_count"] == 0
    assert screen_table["power_simulation_replicate_count"] == 0
    assert len(screen_table["screens"]) == 126
    for encoded, (root_count, minimum_plus_count, exact_power) in zip(screen_table["screens"], screens, strict=True):
        assert encoded["root_count"] == root_count
        assert encoded["minimum_plus_count"] == minimum_plus_count
        assert encoded["passed"] is (exact_power >= Fraction(4, 5))
        encoded_fraction = encoded["exact_power_hex"]
        assert set(encoded_fraction) == {"numerator_hex", "denominator_hex"}
        for value in encoded_fraction.values():
            assert type(value) is str and value
            assert value == value.lower()
            assert not value.startswith("0x")
            assert not value.startswith("0")
            assert set(value) <= set("0123456789abcdef")
        parsed = Fraction(
            int(encoded_fraction["numerator_hex"], 16),
            int(encoded_fraction["denominator_hex"], 16),
        )
        assert parsed == exact_power
        assert (
            math.gcd(
                int(encoded_fraction["numerator_hex"], 16),
                int(encoded_fraction["denominator_hex"], 16),
            )
            == 1
        )
    first_passing_index = next(index for index, (_, _, power) in enumerate(screens) if power >= Fraction(4, 5))
    assert all(power < Fraction(4, 5) for _, _, power in screens[:first_passing_index])
    root_count, minimum_plus_count, exact_power = screens[first_passing_index]
    assert root_count == 1088
    assert minimum_plus_count == 585
    assert exact_power >= Fraction(4, 5)
    next_root_count, next_minimum_plus_count, next_power = screens[first_passing_index + 1]
    assert next_root_count == 1152
    assert next_minimum_plus_count == 620
    assert next_power < Fraction(4, 5)
    assert [item.passed for item in helper_screens[first_passing_index : first_passing_index + 2]] == [
        True,
        False,
    ]
    frozen = plan["derived_global_sentinel"]
    assert frozen["role"] == "necessary_screen_only_not_full_power_pass"
    assert frozen["first_candidate_passing_exact_point_screen"] == root_count
    assert frozen["minimum_plus_count_at_first_pass"] == minimum_plus_count
    assert frozen["exact_power"] == {
        "numerator": str(exact_power.numerator),
        "denominator": str(exact_power.denominator),
    }
    assert frozen["source_filterable"] is False
    assert frozen["full_decision_rule_power_completed"] is False
    assert frozen["candidate_screen_count"] == 126
    assert frozen["screen_table_file"] == _SCREEN_TABLE_FILE
    assert frozen["screen_table_raw_sha256"] == hashlib.sha256(screen_table_path.read_bytes()).hexdigest()
    assert frozen["screen_table_byte_count"] == len(screen_table_path.read_bytes())
    assert frozen["nonmonotonic_witness"] == {
        "earlier_candidate_N": 1088,
        "earlier_candidate_passed": True,
        "later_candidate_N": 1152,
        "later_candidate_exact_power_hex": screen_table["screens"][first_passing_index + 1]["exact_power_hex"],
        "later_candidate_passed": False,
    }


def test_hoeffding_taylor_certificate_and_1856_gate_are_exact() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    plan = _strict_json(_V4_ARTIFACT / _PLAN_FILE)
    decision = protocol["full_joint_power_planner"]["decision_rule"]
    candidates = tuple(range(192, 8193, 64))
    practical_gate = Fraction(3, 20)
    log_upper_bound = Fraction(1269, 250)
    exponential_lower_bound = sum(log_upper_bound**exponent / math.factorial(exponent) for exponent in range(14))
    exact_difference = Fraction(
        1294120199914486134364636005563418813,
        381851196289062500000000000000000000000,
    )
    assert exponential_lower_bound > 160
    assert exponential_lower_bound - 160 == exact_difference
    assert decision["log_160_strict_rational_upper_bound"] == {
        "numerator": "1269",
        "denominator": "250",
    }
    assert decision["log_upper_bound_exact_certificate"]["exact_sum_minus_160"] == {
        "numerator": str(exact_difference.numerator),
        "denominator": str(exact_difference.denominator),
    }

    threshold = 8 * log_upper_bound
    capable = tuple(root_count for root_count in candidates if root_count * practical_gate**2 > threshold)
    assert capable[0] == 1856
    assert Fraction(1792) * practical_gate**2 <= threshold
    assert Fraction(1856) * practical_gate**2 > threshold
    assert plan["derived_decision_rule"]["first_candidate_capable_at_the_practical_boundary"] == 1856
    assert plan["derived_decision_rule"]["bootstrap_inner_loop_count"] == 0
    assert decision["bootstrap_inner_loop_count"] == 0

    # mean = S/(8N): mean >= 3/20 gives 5S >= 6N.  The one-sided
    # Hoeffding/Bonferroni requirement gives 125S^2 > 324864N.
    assert 8 * 3 == 24
    assert Fraction(3, 20) * 8 == Fraction(6, 5)
    assert 512 * log_upper_bound == Fraction(324864, 125)
    assert decision["exact_practical_gate"] == "5_times_S_c_greater_than_or_equal_to_6_times_N"
    assert decision["exact_positive_mean_gate"] == (
        "S_c_positive_and_125_times_S_c_squared_strictly_greater_than_324864_times_N"
    )


def test_power_rng_bytes_rejection_sampling_and_monte_carlo_integer_gate_are_frozen() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    plan = _strict_json(_V4_ARTIFACT / _PLAN_FILE)
    estimation = protocol["full_joint_power_planner"]["power_estimation"]
    rng = estimation["rng_contract"]
    monte_carlo = estimation["monte_carlo_certification"]

    assert rng["algorithm"] == "sha256_multiblock_counter_exact_rational_categorical_v1"
    assert rng["domain_tag_ascii"] == "volvence.relationship_p4_long_context_v4.power_rng.v1"
    assert rng["stream_labels"] == {"search": "search", "confirmation": "confirmation"}
    assert rng["counter_fields_in_order"] == [
        "domain_tag",
        "protocol_id",
        "stream",
        "seed",
        "scenario_id",
        "replicate_index",
        "root_ordinal",
        "generator_node_id",
        "draw_index",
        "rejection_ordinal",
        "block_ordinal",
    ]
    assert rng["counter_field_types_in_order"] == [
        "text",
        "text",
        "text",
        "integer",
        "text",
        "integer",
        "integer",
        "text",
        "integer",
        "integer",
        "integer",
    ]
    assert "candidate_N" not in rng["counter_fields_in_order"]
    assert rng["counter_text_field_ascii_regex"] == "^[a-z0-9_.:-]+$"
    assert rng["counter_integer_fields_are_zero_based_nonnegative_canonical_base10"] is True
    assert rng["canonical_counter_bytes"] == (
        "utf8_without_bom_of_the_JSON_array_of_counter_fields_in_order_using_double_quoted_ASCII_strings_"
        "commas_without_whitespace_nonnegative_base10_integers_without_leading_zero_and_one_final_0x0a_byte"
    )
    assert rng["sha256_preimage"] == "exactly_the_canonical_counter_bytes"
    assert rng["sha256_block_digest"] == "32_raw_bytes_not_hex"
    assert rng["atom_order"] == "ascending_unique_lowercase_sha256_hex_atom_id_by_unsigned_ASCII_byte_order"
    assert rng["atom_probabilities"] == ("canonical_reduced_strictly_positive_exact_rationals_summing_exactly_to_one")
    assert rng["integer_mass_derivation"] == (
        "Q_is_lcm_of_all_probability_denominators_and_weight_i_is_numerator_i_times_Q_divided_by_"
        "denominator_i_with_sum_of_weights_exactly_Q"
    )
    assert rng["single_atom_Q_equals_one_rule"] == "select_the_only_atom_without_hashing"
    assert rng["bit_and_block_counts"] == (
        "for_Q_greater_than_one_b_is_bit_length_of_Q_minus_one_and_h_is_ceiling_b_divided_by_256"
    )
    assert rng["multiblock_integer"] == (
        "concatenate_h_SHA256_block_digests_in_ascending_block_ordinal_as_one_unsigned_big_endian_integer_Z"
    )
    assert rng["candidate_ticket"] == "u_equals_Z_modulo_two_pow_b_selecting_the_least_significant_b_bits"
    assert rng["rejection_rule"] == (
        "start_rejection_ordinal_at_zero_and_if_u_is_greater_than_or_equal_to_Q_increment_rejection_ordinal_"
        "then_recompute_all_h_blocks"
    )
    assert rng["selected_atom"] == (
        "first_atom_in_frozen_atom_order_whose_strict_cumulative_integer_weight_is_greater_than_u"
    )
    assert rng["acceptance_probability_strictly_greater_than_one_half_for_Q_greater_than_one"] is True
    assert (
        rng["candidate_N_is_absent_from_counter_so_ascending_search_candidates_use_the_same_replicate_root_prefixes"]
        is True
    )
    assert rng["search_and_confirmation_domains_differ_in_the_stream_field_and_seed_field"] is True
    assert rng["every_used_counter_tuple_is_unique_and_duplicate_use_invalidates_the_planner"] is True
    assert rng["generator_node_inventory_and_each_draw_count_frozen_in_source_generator_before_power"] is True
    assert rng["cpu_or_cuda_backend_must_match_frozen_per_draw_or_aggregate_digest_equivalence_receipt"] is True

    counter_bytes = _power_counter_bytes(
        domain_tag="volvence.relationship_p4_long_context_v4.power_rng.v1",
        protocol_id="a" * 64,
        stream="search",
        seed=20260824,
        scenario_id="b" * 64,
        replicate_index=7,
        root_ordinal=3,
        generator_node_id="joint_atom",
        draw_index=2,
        rejection_ordinal=0,
        block_ordinal=0,
    )
    assert counter_bytes == (
        b'["volvence.relationship_p4_long_context_v4.power_rng.v1",'
        b'"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"search",20260824,'
        b'"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        b'7,3,"joint_atom",2,0,0]\n'
    )
    digest = hashlib.sha256(counter_bytes).digest()
    assert len(digest) == 32
    assert digest.hex() == "f14f72a36cb50d9c6f64907693f308f8e14f180427aa84e152b111fcab384d15"
    confirmation_bytes = _power_counter_bytes(
        domain_tag="volvence.relationship_p4_long_context_v4.power_rng.v1",
        protocol_id="a" * 64,
        stream="confirmation",
        seed=20260827,
        scenario_id="b" * 64,
        replicate_index=7,
        root_ordinal=3,
        generator_node_id="joint_atom",
        draw_index=2,
        rejection_ordinal=0,
        block_ordinal=0,
    )
    rejected_retry_bytes = _power_counter_bytes(
        domain_tag="volvence.relationship_p4_long_context_v4.power_rng.v1",
        protocol_id="a" * 64,
        stream="search",
        seed=20260824,
        scenario_id="b" * 64,
        replicate_index=7,
        root_ordinal=3,
        generator_node_id="joint_atom",
        draw_index=2,
        rejection_ordinal=1,
        block_ordinal=0,
    )
    second_block_bytes = _power_counter_bytes(
        domain_tag="volvence.relationship_p4_long_context_v4.power_rng.v1",
        protocol_id="a" * 64,
        stream="search",
        seed=20260824,
        scenario_id="b" * 64,
        replicate_index=7,
        root_ordinal=3,
        generator_node_id="joint_atom",
        draw_index=2,
        rejection_ordinal=0,
        block_ordinal=1,
    )
    assert len({counter_bytes, confirmation_bytes, rejected_retry_bytes, second_block_bytes}) == 4
    with pytest.raises(TypeError, match="exact nonnegative integer"):
        _power_counter_bytes(
            domain_tag="volvence.relationship_p4_long_context_v4.power_rng.v1",
            protocol_id="a" * 64,
            stream="search",
            seed=True,
            scenario_id="b" * 64,
            replicate_index=7,
            root_ordinal=3,
            generator_node_id="joint_atom",
            draw_index=2,
            rejection_ordinal=0,
            block_ordinal=0,
        )

    probabilities = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
    common_denominator = math.lcm(*(item.denominator for item in probabilities))
    integer_masses = [item.numerator * (common_denominator // item.denominator) for item in probabilities]
    integer_weights = tuple(integer_masses)
    assert common_denominator == sum(integer_weights) == 6
    assert integer_weights == (3, 2, 1)
    total_mass = common_denominator
    bit_count = (total_mass - 1).bit_length()
    block_count = _ceil_fraction(Fraction(bit_count, 256))
    assert (bit_count, block_count) == (3, 1)
    assert Fraction(total_mass, 2**bit_count) > Fraction(1, 2)
    synthetic_candidates = (6, 5)
    accepted_ticket = None
    accepted_rejection_ordinal = None
    for rejection_ordinal, candidate_ticket in enumerate(synthetic_candidates):
        if candidate_ticket >= total_mass:
            continue
        accepted_rejection_ordinal = rejection_ordinal
        accepted_ticket = candidate_ticket
        break
    assert accepted_rejection_ordinal == 1
    assert accepted_ticket == 5
    atom_ids = ("0" * 64, "8" * 64, "f" * 64)
    selected_by_ticket = []
    for ticket in range(total_mass):
        cumulative = 0
        for atom_id, mass in zip(atom_ids, integer_weights, strict=True):
            cumulative += mass
            if cumulative > ticket:
                selected_by_ticket.append(atom_id)
                break
    assert selected_by_ticket == [atom_ids[0]] * 3 + [atom_ids[1]] * 2 + [atom_ids[2]]

    multiblock_q = 2**300 + 123
    assert (multiblock_q - 1).bit_length() == 301
    assert _ceil_fraction(Fraction((multiblock_q - 1).bit_length(), 256)) == 2
    multiblock_ticket, accepted_rejection, accepted_digests, rejected_candidates = _independent_multiblock_ticket(
        total_mass=multiblock_q,
        protocol_id="a" * 64,
        stream="search",
        seed=20260824,
        scenario_id="b" * 64,
        replicate_index=7,
        root_ordinal=3,
        generator_node_id="joint_atom",
        draw_index=2,
    )
    assert 0 <= multiblock_ticket < multiblock_q
    assert len(accepted_digests) == 2
    assert all(len(block) == 32 for block in accepted_digests)
    assert all(candidate >= multiblock_q for candidate in rejected_candidates)
    assert accepted_rejection == len(rejected_candidates)
    single_ticket, single_rejection, single_digests, single_rejected = _independent_multiblock_ticket(
        total_mass=1,
        protocol_id="a" * 64,
        stream="search",
        seed=20260824,
        scenario_id="b" * 64,
        replicate_index=7,
        root_ordinal=3,
        generator_node_id="single_atom",
        draw_index=0,
    )
    assert (single_ticket, single_rejection, single_digests, single_rejected) == (0, 0, (), ())

    assert estimation["search_joint_dgp_replicates"] == 8192
    assert estimation["search_seed"] == 20260824
    assert estimation["search_proposal_gate_exact"] == {"numerator": "41", "denominator": "50"}
    assert estimation["search_exact_integer_pass_rule"] == (
        "50_times_X_search_is_greater_than_or_equal_to_41_times_8192"
    )
    assert estimation["confirmation_joint_dgp_replicates"] == 100000
    assert estimation["confirmation_seed"] == 20260827
    assert estimation["search_and_confirmation_counter_domains_disjoint"] is True
    assert estimation["confirmation_failure_may_increment_or_research_N"] is False
    assert monte_carlo["scenario_count_symbol"] == ("M_equals_1_plus_A_where_A_is_the_admitted_grid_tuple_count")
    assert monte_carlo["M_is_frozen_by_the_pre_power_tuple_feasibility_index"] is True
    assert monte_carlo["replicate_count"] == 100000
    assert monte_carlo["familywise_alpha"] == {"numerator": "1", "denominator": "100"}
    assert monte_carlo["null_boundary_probability"] == {"numerator": "4", "denominator": "5"}
    assert monte_carlo["exact_upper_tail_T"] == (
        "sum_k_from_X_to_100000_of_binomial_100000_choose_k_times_4_pow_k_divided_by_5_pow_100000"
    )
    assert monte_carlo["exact_integer_pass_rule"] == (
        "100_times_M_times_sum_k_from_X_to_100000_of_binomial_100000_choose_k_times_4_pow_k_is_less_"
        "than_or_equal_to_5_pow_100000"
    )
    assert monte_carlo["all_M_scenarios_must_pass"] is True
    assert monte_carlo["pass_rule_equality_passes"] is True

    for successes in range(41):
        weighted_tail = sum(math.comb(40, k) * 4**k for k in range(successes, 41))
        exact_tail = Fraction(weighted_tail, 5**40)
        assert _monte_carlo_integer_pass(
            family_size=3,
            weighted_tail=weighted_tail,
            denominator=5**40,
        ) == (exact_tail <= Fraction(1, 300))
    assert _monte_carlo_integer_pass(family_size=577, weighted_tail=4**100000, denominator=5**100000)
    assert not _monte_carlo_integer_pass(
        family_size=577,
        weighted_tail=5**100000,
        denominator=5**100000,
    )
    assert _monte_carlo_integer_pass(family_size=577, weighted_tail=1, denominator=100 * 577)

    frozen = plan["frozen_power_estimation_contract"]
    assert frozen == {
        "search_joint_dgp_replicates": 8192,
        "search_seed": 20260824,
        "search_proposal_gate_exact": {"numerator": "41", "denominator": "50"},
        "search_exact_integer_pass_rule": ("50_times_X_search_is_greater_than_or_equal_to_41_times_8192"),
        "confirmation_joint_dgp_replicates": 100000,
        "confirmation_seed": 20260827,
        "rng_contract": rng,
        "monte_carlo_certification": monte_carlo,
        "confirmation_failure_may_increment_or_research_N": False,
        "point_estimate_without_monte_carlo_uncertainty_may_authorize": False,
    }


def test_cartesian_576_grid_missingness_and_anti_vacuity_remain_unresolved() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    plan = _strict_json(_V4_ARTIFACT / _PLAN_FILE)
    grid = protocol["source_conditioned_cartesian_grid"]
    axes = grid["axes"]
    expected_axes = {
        "paired_root_difference_variance_scenarios": (
            "source_structural_covariance_upper_bound",
            "paired_root_difference_variance_0_25",
            "paired_root_difference_variance_0_50",
            "paired_root_difference_variance_1_00",
        ),
        "within_root_icc_decimals": ("0.00", "0.25", "0.50"),
        "cross_contrast_dependence_labels": (
            "source_structural_covariance",
            "independent_contrasts",
            "equicorrelation_negative_0_10",
            "equicorrelation_positive_0_50",
        ),
        "technical_missingness_rate_decimals": ("0.00", "0.01", "0.02"),
        "technical_missingness_patterns": (
            "balanced_uniform_single_evaluation_slot",
            "reference_root_correlated",
            "all_comparators_root_correlated",
            "reference_and_all_comparators_root_correlated",
        ),
    }
    assert {name: tuple(values) for name, values in axes.items()} == expected_axes
    tuples = tuple(itertools.product(*(expected_axes[name] for name in expected_axes)))
    assert len(tuples) == 4 * 3 * 4 * 3 * 4 == 576
    assert len(set(tuples)) == 576
    assert grid["candidate_tuple_count_before_feasibility"] == 576
    assert grid["resource_timeout_nonconvergence_or_missing_solver_may_skip_tuple"] is False
    assert grid["infeasibility_witness"]["absence_of_a_constructive_search_result_is_a_proof"] is False

    repeated, shared = _equicorrelation_eigenvalues(8, Fraction(-1, 10))
    assert (repeated, shared) == (Fraction(11, 10), Fraction(3, 10))
    repeated, shared = _equicorrelation_eigenvalues(8, Fraction(1, 2))
    assert (repeated, shared) == (Fraction(1, 2), Fraction(9, 2))

    anti_vacuity = grid["anti_vacuity"]
    assert anti_vacuity["at_least_one_source_reference_tuple_must_be_admitted"] is True
    assert anti_vacuity["every_level_of_every_mandatory_axis_appears_in_at_least_one_admitted_tuple"] is True
    assert anti_vacuity["admitted_plus_skipped_plus_unresolved_equals_576"] is True
    assert anti_vacuity["unresolved_tuple_count_must_equal_zero_before_power_search"] is True
    assert anti_vacuity["declaring_all_difficult_tuples_infeasible_may_authorize"] is False

    frozen = plan["derived_grid_contract"]
    assert frozen == {
        "candidate_tuple_count_before_feasibility": 576,
        "global_sentinel_count": 1,
        "source_grid_resolved": False,
        "feasible_tuple_count": None,
        "skipped_tuple_count": None,
        "unresolved_tuple_count": 576,
        "grid_digest": None,
        "sample_size_selected": False,
        "selected_formal_root_count": None,
    }
    missingness = protocol["missingness_semantics"]
    assert missingness["planning_alternative_mean_stage"] == (
        "latent_complete_data_after_substantive_malformed_mapping_and_before_technical_missingness_or_itt_imputation"
    )
    assert missingness["substantive_malformed_utility_minus_one_is_included_in_the_one_fifth_alternative"] is True
    assert missingness["technical_missingness_is_applied_after_complete_utility_tensor_generation"] is True
    assert missingness["contrast_specific_worst_case_itt"] == {
        "missing_reference_utility": -1,
        "missing_comparator_utility": 1,
        "both_missing_contrast_value_applied_directly": -2,
        "imputation_occurs_per_decision_before_root_mean": True,
        "all_preallocated_roots_remain_in_each_contrast": True,
    }
    assert _ceil_fraction(Fraction(5 * 192, 6)) == 160
    assert missingness["minimum_globally_complete_roots"] == "ceiling_five_sixths_of_candidate_N"


def test_receipt_first_classifier_recomputes_all_eleven_frozen_cases() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    plan = _strict_json(_V4_ARTIFACT / _PLAN_FILE)
    classification = protocol["generated_action_classification"]
    cases = classification["classification_cases"]
    assert len(cases) == 11
    assert classification["precedence"] == [
        "integrity_lineage_checks",
        "generation_attempt_completion",
        "frozen_parser_execution",
        "candidate_cardinality_and_domain",
        "utility_lookup",
    ]
    actual: dict[str, tuple[str, str]] = {}
    for case in cases:
        primitives = {
            key: value for key, value in case.items() if key not in {"case_id", "classification", "consequence"}
        }
        independently_derived = _independent_classification(primitives)
        helper_derived = derivation.classify_generated_action_case(primitives)
        frozen = (case["classification"], case["consequence"])
        assert independently_derived == helper_derived == frozen
        actual[case["case_id"]] = independently_derived
    assert actual == _EXPECTED_CLASSIFICATIONS
    assert {
        item["case_id"]: (item["classification"], item["consequence"])
        for item in plan["derived_generated_action_cases"]
    } == _EXPECTED_CLASSIFICATIONS

    valid = next(case for case in cases if case["case_id"] == "one_valid_candidate")
    primitive_valid = {
        key: value for key, value in valid.items() if key not in {"case_id", "classification", "consequence"}
    }
    invalid_receipt = dict(primitive_valid, generated_bytes_receipt_valid=False)
    assert _independent_classification(invalid_receipt) == (
        "integrity_failure",
        "invalid_attempt_no_claim",
    )
    assert derivation.classify_generated_action_case(invalid_receipt) == (
        "integrity_failure",
        "invalid_attempt_no_claim",
    )
    bool_count = dict(primitive_valid, candidate_count=True)
    assert derivation.classify_generated_action_case(bool_count) == (
        "integrity_failure",
        "invalid_attempt_no_claim",
    )
    incomplete_with_payload = {key: None for key in primitive_valid}
    incomplete_with_payload.update(
        generation_attempt_completed=False,
        authenticated_technical_failure_receipt_valid=True,
        lineage_receipt_chain_valid=True,
        utility_commitment_valid=True,
        independent_reobserver_valid=True,
        candidate_count=0,
    )
    assert derivation.classify_generated_action_case(incomplete_with_payload) == (
        "integrity_failure",
        "invalid_attempt_no_claim",
    )


def test_six_cell_williams_schedule_is_independently_rebuilt_for_all_640_blocks() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    schedule = _strict_json(_V4_ARTIFACT / _SCHEDULE_FILE)
    candidates = protocol["development_candidate_cells"]
    counterbalance = candidates["counterbalance"]
    expected_cells = tuple(
        f"{family}::candidate_{index}"
        for family in (
            "qwen_steelman_full_history",
            "qwen_steelman_selective_rag",
        )
        for index in (0, 1, 2)
    )
    assert expected_cells == _EXPECTED_CELLS
    assert candidates["baseline_families"] == [
        "qwen_steelman_full_history",
        "qwen_steelman_selective_rag",
    ]
    assert candidates["candidate_indices"] == [0, 1, 2]
    assert sum(counterbalance["session_phases"].values()) == 4 + 8 + 8 == 20
    assert counterbalance["block_enumeration"] == (
        "session_major_then_root_ordinal_so_all_roots_finish_session_s_before_any_root_starts_session_s_plus_one"
    )

    expected_blocks = _independent_williams_schedule(
        expected_cells,
        root_count=32,
        sessions_per_root=20,
        seed=20260826,
    )
    assert len(expected_blocks) == 32 * 20 == 640
    assert schedule["candidate_cell_ids"] == list(expected_cells)
    assert schedule["development_root_count"] == 32
    assert schedule["sessions_per_root"] == 20
    assert schedule["seed"] == 20260826
    assert schedule["block_count"] == 640
    assert schedule["blocks"] == expected_blocks
    assert schedule["model_output_count"] == 0
    assert schedule["selected_candidate_id"] is None
    for ordinal, block in enumerate(expected_blocks):
        assert block["global_block_ordinal"] == ordinal
        assert block["session_index"] == ordinal // 32
        assert block["root_ordinal"] == ordinal % 32

    all_pairs = {(left, right) for left in expected_cells for right in expected_cells if left != right}
    block_orders = tuple(tuple(block["ordered_cell_ids"]) for block in expected_blocks)
    for start in range(len(block_orders) - 5):
        window = block_orders[start : start + 6]
        for position in range(6):
            assert {row[position] for row in window} == set(expected_cells)
        carryovers = {(row[position], row[position + 1]) for row in window for position in range(5)}
        assert carryovers == all_pairs
    for position in range(6):
        counts = Counter(row[position] for row in block_orders)
        assert set(counts) == set(expected_cells)
        assert max(counts.values()) - min(counts.values()) <= 1
    for root_ordinal in range(32):
        root_blocks = tuple(
            tuple(block["ordered_cell_ids"]) for block in expected_blocks if block["root_ordinal"] == root_ordinal
        )
        assert len(root_blocks) == 20
        counts = Counter(cell for row in root_blocks for cell in row)
        assert counts == Counter({cell: 20 for cell in expected_cells})


def test_create_only_reproduction_is_byte_identical_and_refuses_existing_outputs(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "reproduced-v4a"
    result = prepare_relationship_p4_long_context_v4_zero_output_plan(
        output_dir=output,
        v3_preparation_dir=_V3_PREPARATION,
        v2_admission_dir=_V2_ADMISSION,
    )
    assert result.artifact_id == _EXPECTED_ARTIFACT_ID
    for filename, expected_raw in (
        (_PLAN_FILE, _EXPECTED_PLAN_RAW),
        (_SCHEDULE_FILE, _EXPECTED_SCHEDULE_RAW),
        (_SCREEN_TABLE_FILE, _EXPECTED_SCREEN_TABLE_RAW),
        (_MANIFEST_FILE, _EXPECTED_MANIFEST_RAW),
    ):
        generated = (output / filename).read_bytes()
        assert generated == (_V4_ARTIFACT / filename).read_bytes()
        assert hashlib.sha256(generated).hexdigest() == expected_raw

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_v4_zero_output_plan(
            output_dir=output,
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir=_V2_ADMISSION,
        )
    preexisting = tmp_path / "preexisting"
    preexisting.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_v4_zero_output_plan(
            output_dir=preexisting,
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir=_V2_ADMISSION,
        )
    assert not tuple(tmp_path.glob(".reproduced-v4a.tmp-*"))
    assert not tuple(tmp_path.glob(".preexisting.tmp-*"))


def test_falsey_missing_and_cross_spliced_paths_fail_loudly(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="regular file"):
        load_relationship_p4_long_context_v4_planning_protocol("")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        load_relationship_p4_long_context_v4_planning_protocol(False)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="root is missing"):
        _validate_v4(tmp_path / "missing")
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_relationship_p4_long_context_v4_zero_output_plan(
            output_dir=_V4_ARTIFACT,
            v3_preparation_dir="",  # type: ignore[arg-type]
            v2_admission_dir=_V2_ADMISSION,
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_relationship_p4_long_context_v4_zero_output_plan(
            output_dir=_V4_ARTIFACT,
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir="",  # type: ignore[arg-type]
        )
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_v4_zero_output_plan(
            output_dir="",  # type: ignore[arg-type]
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir=_V2_ADMISSION,
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_relationship_p4_long_context_v4_zero_output_plan(
            output_dir=_V4_ARTIFACT,
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir=_V3_PREPARATION,
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_relationship_p4_long_context_v4_zero_output_plan(
            output_dir=_V4_ARTIFACT,
            v3_preparation_dir=_V2_ADMISSION,
            v2_admission_dir=_V2_ADMISSION,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("identity", "scientific_v3_preparation_artifact_id"), "a" * 64),
        (("derived_global_sentinel", "first_candidate_passing_exact_point_screen"), True),
        (("derived_grid_contract", "source_grid_resolved"), True),
        (("zero_output_firewall", "cuda_formal_run_count"), 1),
        (("terminal", "development_authorized"), True),
    ],
)
def test_validator_rejects_semantic_tamper_after_plan_and_manifest_resigning(
    field_path: tuple[str, ...],
    replacement: object,
    tmp_path: pathlib.Path,
) -> None:
    artifact = _copy_v4_artifact(tmp_path, "resigned-plan")
    _resign_plan(artifact, lambda payload: _set_nested(payload, field_path, replacement))
    with pytest.raises((TypeError, ValueError)):
        _validate_v4(artifact)


def test_validator_rejects_schedule_and_manifest_semantic_tamper_after_resigning(
    tmp_path: pathlib.Path,
) -> None:
    schedule_tamper = _copy_v4_artifact(tmp_path, "resigned-schedule")
    schedule_path = schedule_tamper / _SCHEDULE_FILE
    schedule = _strict_json(schedule_path)
    schedule["blocks"][0]["ordered_cell_ids"][0:2] = reversed(schedule["blocks"][0]["ordered_cell_ids"][0:2])
    schedule_bytes = _canonical_bytes(schedule)
    schedule_path.write_bytes(schedule_bytes)
    _resign_plan(
        schedule_tamper,
        lambda payload: payload["derived_candidate_schedule"].update(
            schedule_raw_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
            schedule_byte_count=len(schedule_bytes),
        ),
    )
    with pytest.raises((TypeError, ValueError), match="candidate schedule"):
        _validate_v4(schedule_tamper)

    screen_tamper = _copy_v4_artifact(tmp_path, "resigned-screen-table")
    screen_path = screen_tamper / _SCREEN_TABLE_FILE
    screen_table = _strict_json(screen_path)
    first_numerator = screen_table["screens"][0]["exact_power_hex"]["numerator_hex"]
    screen_table["screens"][0]["exact_power_hex"]["numerator_hex"] = f"0{first_numerator}"
    screen_bytes = _canonical_bytes(screen_table)
    screen_path.write_bytes(screen_bytes)
    _resign_plan(
        screen_tamper,
        lambda payload: payload["derived_global_sentinel"].update(
            screen_table_raw_sha256=hashlib.sha256(screen_bytes).hexdigest(),
            screen_table_byte_count=len(screen_bytes),
        ),
    )
    with pytest.raises((TypeError, ValueError), match="screen table|exact power|leading zero"):
        _validate_v4(screen_tamper)

    manifest_tamper = _copy_v4_artifact(tmp_path, "resigned-manifest")
    manifest_path = manifest_tamper / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest["development_authorized"] = True
    _resign_manifest(manifest_path, manifest)
    with pytest.raises((TypeError, ValueError), match="manifest|development_authorized"):
        _validate_v4(manifest_tamper)


def test_validator_rejects_extra_noncanonical_bom_duplicate_and_raw_tamper(
    tmp_path: pathlib.Path,
) -> None:
    extra = _copy_v4_artifact(tmp_path, "extra")
    (extra / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set drift"):
        _validate_v4(extra)

    noncanonical = _copy_v4_artifact(tmp_path, "noncanonical")
    plan_path = noncanonical / _PLAN_FILE
    plan_path.write_text(
        json.dumps(_strict_json(plan_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        _validate_v4(noncanonical)

    reordered = _copy_v4_artifact(tmp_path, "key-reordered")
    manifest_path = reordered / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest_path.write_text(
        json.dumps(dict(reversed(tuple(manifest.items()))), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        _validate_v4(reordered)

    bom = _copy_v4_artifact(tmp_path, "bom")
    bom_path = bom / _PLAN_FILE
    bom_path.write_bytes(b"\xef\xbb\xbf" + bom_path.read_bytes())
    with pytest.raises(ValueError, match="must not carry a UTF-8 BOM"):
        _validate_v4(bom)

    duplicate = _copy_v4_artifact(tmp_path, "duplicate")
    duplicate_path = duplicate / _PLAN_FILE
    duplicate_path.write_text(
        duplicate_path.read_text(encoding="utf-8").replace(
            "{",
            '{"artifact_sequence":"duplicate",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _validate_v4(duplicate)

    raw_tamper = _copy_v4_artifact(tmp_path, "raw-tamper")
    raw_path = raw_tamper / _PLAN_FILE
    raw = _strict_json(raw_path)
    raw["terminal"]["model_output_authorized"] = True
    raw_path.write_bytes(_canonical_bytes(raw))
    with pytest.raises(ValueError, match="certificate id drift"):
        _validate_v4(raw_tamper)


def test_protocol_and_derivation_helper_raw_pins_reject_format_or_content_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    reordered = tmp_path / "reordered-protocol.json"
    reordered.write_text(
        json.dumps(dict(reversed(tuple(protocol.items()))), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_V4_PLANNING_PROTOCOL_PATH_V1", reordered)
        with pytest.raises(ValueError, match="raw bytes drift"):
            load_relationship_p4_long_context_v4_planning_protocol()

    drifted_helper = tmp_path / "drifted-helper.py"
    drifted_helper.write_bytes(_HELPER_SOURCE.read_bytes() + b"\n")
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_V4_PLANNING_DERIVATION_HELPER_PATH", drifted_helper)
        with pytest.raises(ValueError, match="derivation helper hash"):
            load_relationship_p4_long_context_v4_planning_protocol()


def test_validator_rejects_hardlinked_artifact_files(tmp_path: pathlib.Path) -> None:
    artifact = _copy_v4_artifact(tmp_path, "hardlinked")
    independent_source = tmp_path / "plan-source.json"
    shutil.copy2(artifact / _PLAN_FILE, independent_source)
    (artifact / _PLAN_FILE).unlink()
    try:
        os.link(independent_source, artifact / _PLAN_FILE)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="exactly one hard link"):
        _validate_v4(artifact)


def test_validator_rejects_symlinked_artifact_files_when_supported(tmp_path: pathlib.Path) -> None:
    artifact = _copy_v4_artifact(tmp_path, "symlinked-file")
    linked = artifact / _PLAN_FILE
    linked.unlink()
    try:
        linked.symlink_to(_V4_ARTIFACT / _PLAN_FILE)
    except OSError as exc:
        pytest.skip(f"file symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="regular files"):
        _validate_v4(artifact)


def test_reparse_artifact_roots_and_dangling_output_aliases_fail(
    tmp_path: pathlib.Path,
) -> None:
    artifact_alias = tmp_path / "artifact-alias"
    _create_directory_alias_or_skip(artifact_alias, _V4_ARTIFACT)
    try:
        with pytest.raises(ValueError, match="symlink|reparse point"):
            _validate_v4(artifact_alias)
    finally:
        _remove_directory_alias(artifact_alias)

    missing_target = tmp_path / "missing-target"
    output_alias = tmp_path / "output-alias"
    _create_directory_alias_or_skip(output_alias, missing_target)
    try:
        with pytest.raises(ValueError, match="symlink|reparse point"):
            prepare_relationship_p4_long_context_v4_zero_output_plan(
                output_dir=output_alias,
                v3_preparation_dir=_V3_PREPARATION,
                v2_admission_dir=_V2_ADMISSION,
            )
    finally:
        _remove_directory_alias(output_alias)
    assert not missing_target.exists()


def test_owner_helper_and_cli_have_no_heavy_dependency_or_execution_surface() -> None:
    forbidden_import_roots = {
        "accelerate",
        "cupy",
        "cuda",
        "numpy",
        "onnxruntime",
        "subprocess",
        "torch",
        "transformers",
    }
    for source_path in (_OWNER_SOURCE, _HELPER_SOURCE, _CLI_SOURCE):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert _import_roots(tree).isdisjoint(forbidden_import_roots)
        assert not any(
            isinstance(node, ast.Name) and node.id.lower() in forbidden_import_roots for node in ast.walk(tree)
        )

    helper_tree = ast.parse(_HELPER_SOURCE.read_text(encoding="utf-8"))
    helper_imports = _import_roots(helper_tree)
    assert helper_imports <= {"__future__", "dataclasses", "fractions", "hashlib", "math", "typing"}
    forbidden_helper_attributes = {
        "exists",
        "glob",
        "is_dir",
        "is_file",
        "mkdir",
        "open",
        "read_bytes",
        "read_text",
        "rename",
        "replace",
        "resolve",
        "unlink",
        "write_bytes",
        "write_text",
    }
    assert not {node.attr for node in ast.walk(helper_tree) if isinstance(node, ast.Attribute)}.intersection(
        forbidden_helper_attributes
    )

    cli_source = _CLI_SOURCE.read_text(encoding="utf-8")
    cli_tree = ast.parse(cli_source)
    option_strings = {
        argument.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("--")
    }
    literal_commands = {
        node.args[0].value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    loop_commands = {
        element.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "command"
        and isinstance(node.iter, (ast.Tuple, ast.List))
        for element in node.iter.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    assert option_strings == {"--output-dir", "--v3-preparation-dir", "--v2-admission-dir"}
    assert literal_commands | loop_commands == {"prepare", "validate-existing"}
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "command"
        for node in ast.walk(cli_tree)
    )
    for forbidden in (
        "--force",
        "--override",
        "--run-session",
        "--model",
        "--cuda",
        "--source-materialization",
        "--development",
        "--formal",
    ):
        assert forbidden not in cli_source.lower()


def _validate_v4(path: pathlib.Path) -> object:
    return validate_relationship_p4_long_context_v4_zero_output_plan(
        output_dir=path,
        v3_preparation_dir=_V3_PREPARATION,
        v2_admission_dir=_V2_ADMISSION,
    )


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


def _weighted_correlation(
    left: tuple[int | Fraction, ...],
    right: tuple[int | Fraction, ...],
    masses: tuple[Fraction, ...],
) -> Fraction:
    left_mean = sum((mass * value for mass, value in zip(masses, left, strict=True)), Fraction())
    right_mean = sum((mass * value for mass, value in zip(masses, right, strict=True)), Fraction())
    covariance = sum(
        (
            mass * (left_value - left_mean) * (right_value - right_mean)
            for mass, left_value, right_value in zip(masses, left, right, strict=True)
        ),
        Fraction(),
    )
    left_variance = sum(
        (mass * (value - left_mean) ** 2 for mass, value in zip(masses, left, strict=True)),
        Fraction(),
    )
    right_variance = sum(
        (mass * (value - right_mean) ** 2 for mass, value in zip(masses, right, strict=True)),
        Fraction(),
    )
    assert left_variance > 0 and right_variance > 0
    squared = covariance * covariance / (left_variance * right_variance)
    assert squared == 1
    return Fraction(1 if covariance > 0 else -1)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _exact_binomial_upper_tail(
    *,
    trials: int,
    success_numerator: int,
    probability_denominator: int,
    minimum_successes: int,
) -> Fraction:
    failure_numerator = probability_denominator - success_numerator
    successes = trials
    term = success_numerator**trials
    numerator = term
    while successes > minimum_successes:
        recurrence_numerator = term * successes * failure_numerator
        recurrence_denominator = (trials - successes + 1) * success_numerator
        assert recurrence_numerator % recurrence_denominator == 0
        term = recurrence_numerator // recurrence_denominator
        numerator += term
        successes -= 1
    return Fraction(numerator, probability_denominator**trials)


def _equicorrelation_eigenvalues(dimension: int, rho: Fraction) -> tuple[Fraction, Fraction]:
    return 1 - rho, 1 + (dimension - 1) * rho


def _independent_classification(primitives: dict[str, object]) -> tuple[str, str]:
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
    assert set(primitives) == required
    completed = primitives["generation_attempt_completed"]
    technical_receipt = primitives["authenticated_technical_failure_receipt_valid"]
    foundational_integrity = (
        "lineage_receipt_chain_valid",
        "utility_commitment_valid",
        "independent_reobserver_valid",
    )
    if any(primitives[field] is not True for field in foundational_integrity):
        return "integrity_failure", "invalid_attempt_no_claim"
    if completed is not True:
        other_fields = required - {
            "generation_attempt_completed",
            "authenticated_technical_failure_receipt_valid",
            *foundational_integrity,
        }
        if completed is False and technical_receipt is True and all(primitives[key] is None for key in other_fields):
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
    if any(primitives[field] is not True for field in integrity_fields):
        return "integrity_failure", "invalid_attempt_no_claim"
    count = primitives["candidate_count"]
    in_domain = primitives["candidate_in_closed_domain"]
    if type(count) is not int or count < 0 or type(in_domain) is not bool:
        return "integrity_failure", "invalid_attempt_no_claim"
    if count == 1 and in_domain:
        return "valid_generated_action", "lookup_committed_utility_vector"
    return (
        "substantive_invalid_generated_action",
        "typed_invalid_action_utility_minus_one_in_itt",
    )


def _power_counter_bytes(
    *,
    domain_tag: str,
    protocol_id: str,
    stream: str,
    seed: int,
    scenario_id: str,
    replicate_index: int,
    root_ordinal: int,
    generator_node_id: str,
    draw_index: int,
    rejection_ordinal: int,
    block_ordinal: int,
) -> bytes:
    allowed_text = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.:-")
    texts = (domain_tag, protocol_id, stream, scenario_id, generator_node_id)
    if any(type(value) is not str or not value or not set(value) <= allowed_text for value in texts):
        raise ValueError("counter text is outside the frozen ASCII domain")
    integers = (seed, replicate_index, root_ordinal, draw_index, rejection_ordinal, block_ordinal)
    if any(type(value) is not int for value in integers):
        raise TypeError("counter integer must be an exact nonnegative integer")
    if any(value < 0 for value in integers):
        raise ValueError("counter integer must be an exact nonnegative integer")
    return (
        json.dumps(
            [
                domain_tag,
                protocol_id,
                stream,
                seed,
                scenario_id,
                replicate_index,
                root_ordinal,
                generator_node_id,
                draw_index,
                rejection_ordinal,
                block_ordinal,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _independent_multiblock_ticket(
    *,
    total_mass: int,
    protocol_id: str,
    stream: str,
    seed: int,
    scenario_id: str,
    replicate_index: int,
    root_ordinal: int,
    generator_node_id: str,
    draw_index: int,
) -> tuple[int, int, tuple[bytes, ...], tuple[int, ...]]:
    assert type(total_mass) is int and total_mass > 0
    if total_mass == 1:
        return 0, 0, (), ()
    bit_count = (total_mass - 1).bit_length()
    block_count = _ceil_fraction(Fraction(bit_count, 256))
    rejected: list[int] = []
    rejection_ordinal = 0
    while True:
        digests = tuple(
            hashlib.sha256(
                _power_counter_bytes(
                    domain_tag="volvence.relationship_p4_long_context_v4.power_rng.v1",
                    protocol_id=protocol_id,
                    stream=stream,
                    seed=seed,
                    scenario_id=scenario_id,
                    replicate_index=replicate_index,
                    root_ordinal=root_ordinal,
                    generator_node_id=generator_node_id,
                    draw_index=draw_index,
                    rejection_ordinal=rejection_ordinal,
                    block_ordinal=block_ordinal,
                )
            ).digest()
            for block_ordinal in range(block_count)
        )
        multiblock_integer = int.from_bytes(b"".join(digests), "big")
        candidate = multiblock_integer % (2**bit_count)
        if candidate < total_mass:
            return candidate, rejection_ordinal, digests, tuple(rejected)
        rejected.append(candidate)
        rejection_ordinal += 1


def _monte_carlo_integer_pass(*, family_size: int, weighted_tail: int, denominator: int) -> bool:
    assert type(family_size) is int and family_size > 0
    assert type(weighted_tail) is int and weighted_tail >= 0
    assert type(denominator) is int and denominator > 0
    return 100 * family_size * weighted_tail <= denominator


def _independent_williams_schedule(
    cells: tuple[str, ...],
    *,
    root_count: int,
    sessions_per_root: int,
    seed: int,
) -> list[dict[str, object]]:
    labels = list(cells)
    for draw_ordinal, upper_index in enumerate(range(len(labels) - 1, 0, -1)):
        selected = _sha256_draw_below(
            seed=seed,
            draw_ordinal=draw_ordinal,
            upper_exclusive=upper_index + 1,
        )
        labels[upper_index], labels[selected] = labels[selected], labels[upper_index]
    first_row = (0, 1, 5, 2, 4, 3)
    rows = tuple(tuple(labels[(index + row_ordinal) % 6] for index in first_row) for row_ordinal in range(6))
    result = []
    for session_index in range(sessions_per_root):
        for root_ordinal in range(root_count):
            global_ordinal = session_index * root_count + root_ordinal
            result.append(
                {
                    "root_ordinal": root_ordinal,
                    "session_index": session_index,
                    "global_block_ordinal": global_ordinal,
                    "ordered_cell_ids": list(rows[global_ordinal % 6]),
                }
            )
    return result


def _sha256_draw_below(*, seed: int, draw_ordinal: int, upper_exclusive: int) -> int:
    limit = (1 << 64) - ((1 << 64) % upper_exclusive)
    rejection = 0
    while True:
        payload = f"p4.7-v4-candidate-schedule|{seed}|{draw_ordinal}|{rejection}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        if value < limit:
            return value % upper_exclusive
        rejection += 1


def _copy_v4_artifact(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    destination = tmp_path / name
    shutil.copytree(_V4_ARTIFACT, destination)
    return destination


def _set_nested(payload: dict[str, Any], field_path: tuple[str, ...], replacement: object) -> None:
    current: dict[str, Any] = payload
    for field in field_path[:-1]:
        current = current[field]
    current[field_path[-1]] = replacement


def _resign_plan(root: pathlib.Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    plan_path = root / _PLAN_FILE
    plan = _strict_json(plan_path)
    plan.pop("certificate_id")
    mutator(plan)
    plan["certificate_id"] = hashlib.sha256(_canonical_bytes(plan)).hexdigest()
    plan_bytes = _canonical_bytes(plan)
    plan_path.write_bytes(plan_bytes)

    schedule_bytes = (root / _SCHEDULE_FILE).read_bytes()
    screen_table_bytes = (root / _SCREEN_TABLE_FILE).read_bytes()
    manifest_path = root / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest["certificate_id"] = plan["certificate_id"]
    manifest["files"] = [
        {
            "path": _PLAN_FILE,
            "byte_count": len(plan_bytes),
            "sha256": hashlib.sha256(plan_bytes).hexdigest(),
        },
        {
            "path": _SCHEDULE_FILE,
            "byte_count": len(schedule_bytes),
            "sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        },
        {
            "path": _SCREEN_TABLE_FILE,
            "byte_count": len(screen_table_bytes),
            "sha256": hashlib.sha256(screen_table_bytes).hexdigest(),
        },
    ]
    _resign_manifest(manifest_path, manifest)


def _resign_manifest(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    manifest.pop("artifact_id")
    manifest["artifact_id"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    path.write_bytes(_canonical_bytes(manifest))


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
