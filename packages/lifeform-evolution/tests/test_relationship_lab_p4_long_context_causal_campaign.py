from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

import pytest

import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign as owner
from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
    P4_LONG_CONTEXT_PREPARATION_STATUS,
    P4_LONG_CONTEXT_PROTOCOL_ID,
    P4_LONG_CONTEXT_PROTOCOL_ID_V1,
    P4_LONG_CONTEXT_PROTOCOL_ID_V2,
    P4_LONG_CONTEXT_PROTOCOL_ID_V3,
    load_relationship_p4_long_context_scientific_prereg,
    prepare_relationship_p4_long_context_scientific_prereg,
    relationship_p4_long_context_protocol_path,
    validate_relationship_p4_long_context_scientific_prereg,
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
_CLI_SOURCE = _REPO_ROOT / "scripts" / "run_relationship_lab_p4_long_context_causal_campaign.py"
_V1_ARTIFACT = (
    _REPO_ROOT / "artifacts" / "relationship_lab" / "p4_independent_long_context_causal_campaign_design_prereg_20260823"
)
_V2_ARTIFACT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_causal_campaign_design_prereg_v2_20260823"
)
_V3_ARTIFACT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_causal_campaign_design_prereg_v3_20260823"
)

_EXPECTED_PROTOCOL_IDS = {
    "v1": "5387516a803940a738e13bb47acc8a40b837c3f033797e09dbfaa23c6cda6d2e",
    "v2": "666d2e8546cd4b4cf55ece06354310e10b4dc07298241b94ef9593e4b5f63baf",
    "v3": "9f352778e128a9573790762222a05225740bdaeb732800dec0eec124116a282d",
}
_EXPECTED_PROTOCOL_RAW_SHA256 = {
    "v1": "c9b8f828ddd39caa36272865cb1fdfb556eb6fa0e9a214b8692c5eb4f158417a",
    "v2": "24a748dde5ea2ba33943b7f66f35755acfcbdd87b36f45f48ee596f95b132d44",
    "v3": "ea8a17a14a68802d3b60586bf520c9137e6920be4112c951ec8c69f5e6ea359e",
}
_EXPECTED_PUBLISHED_ARTIFACT_IDS = {
    "v1": "899b7b0adc395186e108dc0a90c28c0d25ce67cd5445f61636cc2775d09b6901",
    "v2": "795dea07eabda98c964ca50ee84694fd93bb6ee27fdd0370db7e6cb3ef01a8bd",
    "v3": "c5a708ae5e68261fddbade165b45579e66e4bbe7db1be1f4a83056561a17f42e",
}
_EXPECTED_PUBLISHED_ARTIFACT_FILE_SHA256 = {
    "v1": {
        "scientific_prereg_preparation.json": ("ec215d1b0b02850b5d3d863138eb8dcbe27007316e6c8453b270555e8c2d4c6d"),
        "manifest.json": "203afb32373fee663bc6d0f8875ea38430b5e4ecc47bc26a9421a28c18d3f960",
    },
    "v2": {
        "scientific_prereg_preparation.json": ("cb9699434fcc96faa812a2a75fb92ef9b033d0705b2ec99c5a16b32598aa81bf"),
        "manifest.json": "d5704d38e057661181b37443a0d11cd025e8bb342ab46cf53e024bf599848dce",
    },
    "v3": {
        "scientific_prereg_preparation.json": ("a4b2f3ee920e398ae0f7eab5757b7988dc3fa4f7db7b4769599c837bc656bcd6"),
        "manifest.json": "4aaf10d76b80a780e62a17b62803906cc34dd34d64bb4d0f8b96d60dff1e2663",
    },
}
_EXPECTED_V3_SECTION_SHA256 = {
    "schema_version": "d4d8d181fe24687a4beb3a2684dbf47e464605030d2314f157a52d8caaac094c",
    "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
    "frozen_at_utc": "0752843cebc4b3410c1994ef50f634e6cd42e52b1453fb8e13155710086a543c",
    "owner": "24daca4641b8ee6658d6bb9a397b3967d4ad979492c30a7e7d47f399297d38ac",
    "supersession": "02511072e2b4623f40e767d7d42e8487bef55d3af211c543b5baa93bc8388e03",
    "question": "e67e01e5b339114c0ad0eeb6700b59348cfb931edd5fbc347385bdae204385fc",
    "lineage": "67173347b470a2fff89f0650c4be2e99f0819bd9ecff1804ac09093998638478",
    "cohort": "65cf7ce9ed0050bb947db6d2e0b5029a7b9bead317f286306a10f3d2b3239710",
    "longitudinal_design": "e68e4ed323a4686cace32a14426a0a0f049d4d5e32ab14968767538d446f2d1d",
    "baseline_admission": "f76bb4999b6818e3c14d741c638d9c399aacf2e1fe2f7fb646b2fdf9a54ae4fe",
    "integrated_arms": "efd2622f4da07c94994cf76ec320858d298124e71e0fb157f34612933ac8924e",
    "arm_matrix": "ab772f89cc6d0d53db4a929905ef4de0e1e8b7002e65dfdf247d72b5398f4725",
    "axis_contrasts": "8f964f87bb0677b3e1f1970a7a0e6724e1b00c576f245b16d316ca2361df27e5",
    "intervention_integrity": "cfff340cca1e605748526ccd848e30f6b9cc84f56e6db99bdcfb1db72a9545f9",
    "shared_exposure_schedule": "e0276f0f3582a0cdab2727d8d4228370c7081385bda291cc5bfd959de6e075ef",
    "causal_execution": "094938c2e21be9ecbf21b76de04f0ef68f3f6d77688daa26a28446ad5b09f0cc",
    "analysis": "478dc3cea458d50a85548fd161a02dd252cfb9ed576ce65cbc721a20c1ffed3e",
    "execution_admission": "5d7e913a3405eec20482b229854b8d95111151a9e486ca70a465f104530361ba",
    "stopping_rules": "c80a8e540408f367a1d510f59b59b91bac697acc0debb363c74632f19ac406f7",
    "evidence_firewall": "4c0a7fdbd1c6e820b7af3b3cd6807f9f92aad50ac8956cccd52e7bcce9def31b",
    "claim_boundary": "8a5ede78ac23b40f6f93d90c530e047ac9eeca24f68ed3896a383eeacb140a46",
}
_EXPECTED_V3_ARM_MATRIX = (
    "volvence_closed_loop",
    "appendable_empty_prior",
    "appendable_swapped_same_stage_prior",
    "readable_label_permuted",
    "learnable_credit_withheld",
    "steerable_strict_noop",
    "steerable_sensor_off_matched",
    "qwen_steelman_full_history",
    "qwen_steelman_selective_rag",
)


def test_default_protocol_is_v3_large_independent_longitudinal_design() -> None:
    protocol = load_relationship_p4_long_context_scientific_prereg()

    assert P4_LONG_CONTEXT_PROTOCOL_ID == _EXPECTED_PROTOCOL_IDS["v3"]
    assert P4_LONG_CONTEXT_PROTOCOL_ID_V3 == _EXPECTED_PROTOCOL_IDS["v3"]
    assert protocol.protocol_id == _EXPECTED_PROTOCOL_IDS["v3"]
    assert protocol.schema_version == ("relationship-p4-independent-long-context-causal-campaign-scientific-prereg.v3")
    assert protocol.superseded is False
    assert protocol.development_subject_count == 32
    assert protocol.qualification_subject_count == 64
    assert protocol.formal_subject_count == 192
    assert protocol.minimum_complete_paired_subjects == 160
    assert (
        protocol.onboarding_sessions_per_subject,
        protocol.learning_sessions_per_subject,
        protocol.evaluation_sessions_per_subject,
    ) == (4, 8, 8)
    assert protocol.minimum_public_history_tokens == 32_768
    assert protocol.minimum_native_context_window_tokens == 65_536
    assert protocol.minimum_generation_headroom_tokens == 1_024
    assert len(protocol.arm_matrix) == 9
    assert "learnable_credit_withheld" in protocol.arm_matrix
    assert tuple(item.axis for item in protocol.axis_contrasts) == (
        "appendable",
        "readable",
        "learnable",
        "steerable",
    )
    assert protocol.bootstrap_replicates == 100_000
    assert protocol.minimum_practical_mean_delta == 0.15


def test_protocol_and_published_artifact_anchors_are_independent_literals() -> None:
    imported_ids = {
        "v1": P4_LONG_CONTEXT_PROTOCOL_ID_V1,
        "v2": P4_LONG_CONTEXT_PROTOCOL_ID_V2,
        "v3": P4_LONG_CONTEXT_PROTOCOL_ID_V3,
    }
    artifact_roots = {
        "v1": _V1_ARTIFACT,
        "v2": _V2_ARTIFACT,
        "v3": _V3_ARTIFACT,
    }

    assert imported_ids == _EXPECTED_PROTOCOL_IDS
    for version, expected_protocol_id in _EXPECTED_PROTOCOL_IDS.items():
        protocol_path = relationship_p4_long_context_protocol_path(expected_protocol_id)
        protocol_bytes = protocol_path.read_bytes()
        protocol_raw = json.loads(protocol_bytes.decode("utf-8"))
        assert hashlib.sha256(protocol_bytes).hexdigest() == _EXPECTED_PROTOCOL_RAW_SHA256[version]
        assert hashlib.sha256(_canonical_bytes(protocol_raw)).hexdigest() == expected_protocol_id
        assert load_relationship_p4_long_context_scientific_prereg(protocol_path).protocol_id == expected_protocol_id

        artifact_root = artifact_roots[version]
        for filename, expected_sha256 in _EXPECTED_PUBLISHED_ARTIFACT_FILE_SHA256[version].items():
            assert hashlib.sha256((artifact_root / filename).read_bytes()).hexdigest() == expected_sha256
        manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["protocol_id"] == expected_protocol_id
        assert manifest["artifact_id"] == _EXPECTED_PUBLISHED_ARTIFACT_IDS[version]
        validated = validate_relationship_p4_long_context_scientific_prereg(output_dir=artifact_root)
        assert validated.protocol_id == expected_protocol_id
        assert validated.artifact_id == _EXPECTED_PUBLISHED_ARTIFACT_IDS[version]


def test_v3_section_hashes_are_independently_frozen() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    actual = {key: hashlib.sha256(_canonical_bytes(value)).hexdigest() for key, value in raw.items()}

    assert actual == _EXPECTED_V3_SECTION_SHA256
    assert dict(owner._V3_FROZEN_SECTION_SHA256) == _EXPECTED_V3_SECTION_SHA256


def test_v3_freezes_disjoint_nonanalysis_donor_banks() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    donor_banks = raw["cohort"]["appendable_donor_banks"]
    donor_contract = raw["axis_contrasts"]["appendable"]["donor_contract"]

    assert donor_banks["bank_scope"] == "one_disjoint_bank_per_split"
    assert (
        donor_banks["development_donor_root_count"],
        donor_banks["qualification_donor_root_count"],
        donor_banks["formal_donor_root_count"],
    ) == (32, 64, 192)
    assert donor_banks["one_unique_donor_root_per_analysis_root"]
    assert donor_banks["donor_root_reuse_permitted"] is False
    assert donor_banks["donor_roots_disjoint_from_all_analysis_roots"]
    assert donor_banks["donor_roots_disjoint_across_splits"]
    assert donor_banks["donor_roots_enter_estimand_or_bootstrap"] is False
    assert donor_banks["assignment_uses_arm_output_or_outcome"] is False
    assert donor_banks["donor_failure_replacement_permitted"] is False
    assert donor_banks["donor_failure_makes_only_the_target_swapped_arm_decisions_missing"]

    assert donor_contract["assignment_type"] == "one_to_one_from_disjoint_nonanalysis_split_donor_bank"
    assert donor_contract["donor_root_must_not_be_any_analysis_root"]
    assert donor_contract["donor_root_may_be_reused_by_another_target"] is False
    assert donor_contract["donor_root_enters_estimand_or_bootstrap"] is False
    assert donor_contract["donor_selected_from_outcome_or_arm_output"] is False
    assert {
        "donor_bank_split_id",
        "donor_bank_root_id",
        "donor_assignment_artifact_id",
        "source_session_completion_receipt_id",
        "hydrated_owner_snapshot_id",
    }.issubset(donor_contract["owner_hydration_receipt_required_fields"])


def test_v3_freezes_typed_utility_and_decision_level_missingness() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    endpoint = raw["causal_execution"]["typed_outcome_endpoint_contract"]
    missingness = raw["analysis"]["missingness_and_attrition"]

    assert endpoint["utility_closed_integer_domain"] == [-1, 0, 1]
    assert endpoint["one_utility_slot_per_frozen_policy_evaluation_decision"]
    assert endpoint["utility_vector_owner"] == "frozen_reactive_environment_oracle"
    assert endpoint["utility_vector_covers_every_closed_world_typed_action"]
    assert endpoint["actual_utility_rule"] == "lookup_actual_typed_action_in_committed_utility_vector"
    assert endpoint["malformed_missing_or_out_of_domain_observed_action_utility"] == -1
    assert endpoint["unique_optimal_action_has_utility"] == 1
    assert endpoint["nonoptimal_safe_action_maximum_utility"] == 0
    assert endpoint["harmful_action_utility"] == -1
    assert endpoint["scale_shift_rescore_or_posthoc_mapping_permitted"] is False
    assert endpoint["utility_vector_commitment_precedes_model_input"]
    assert endpoint["utility_vector_hidden_until_environment_transition"]
    assert endpoint["independent_reobserver_recomputes_utility_from_typed_settlement"]

    assert missingness["primary_population"] == "all_192_preallocated_subject_roots_intention_to_treat"
    assert missingness["imputation_unit"] == "each_of_8_frozen_policy_evaluation_decisions_before_root_mean"
    assert missingness["every_root_arm_has_exactly_8_preallocated_evaluation_slots"]
    assert missingness["missing_reference_utility"] == -1
    assert missingness["missing_comparator_utility"] == 1
    assert missingness["both_missing_contrast_value"] == -2
    assert missingness["both_missing_value_is_applied_directly_per_decision"]
    assert missingness["whole_root_or_whole_arm_imputation_before_decision_averaging_permitted"] is False
    assert missingness["all_192_roots_remain_in_each_contrast"]


def test_v3_freezes_deterministic_baseline_selection_and_qualification_math() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    baseline = raw["baseline_admission"]
    selection = baseline["development_candidate_selection"]

    assert baseline["candidate_budget"] == 3
    assert baseline["required_baselines"] == [
        "qwen_steelman_full_history",
        "qwen_steelman_selective_rag",
    ]
    assert selection["candidate_unit"] == "paired_full_history_and_selective_rag_configuration"
    assert selection["all_three_candidates_run_on_same_32_roots_tape_and_opportunities"]
    assert selection["selection_key_in_priority_order"] == [
        "descending_minimum_exact_one_action_rate_across_two_baselines",
        "descending_minimum_root_mean_accuracy_across_two_baselines",
        "descending_minimum_root_mean_pair_flip_across_two_baselines",
        "descending_mean_root_accuracy_across_two_baselines",
        "ascending_frozen_candidate_inventory_index",
    ]
    assert selection["selection_targets_qualification_informative_band"] is False
    assert selection["manual_selection_or_tie_break_permitted"] is False
    assert selection["all_candidate_outputs_and_selection_receipt_published"]

    assert baseline["root_accuracy_statistic"] == "sum_of_8_mechanical_correct_indicators_divided_by_8"
    assert baseline["root_pair_flip_statistic"] == (
        "sum_of_4_preregistered_reversal_pair_success_indicators_divided_by_4"
    )
    assert baseline["qualification_point_estimator"] == "unweighted_mean_of_64_root_statistics"
    assert baseline["qualification_bootstrap_replicates"] == 100_000
    assert baseline["qualification_bootstrap_seed"] == 20_260_825
    assert baseline["qualification_bootstrap_index_plan"] == "sha256_seed_replicate_draw_u64_mod_64"
    assert baseline["qualification_bootstrap_jointly_resamples_all_decisions_and_pairs_for_a_root"]
    assert baseline["qualification_quantile_method"] == "inverted_cdf"
    assert baseline["qualification_lower_order_statistic_zero_based_index"] == 4_999
    assert baseline["qualification_arithmetic"] == "exact_integer_counts_and_rational_root_means"
    assert baseline["both_baselines_must_qualify_independently"]
    assert baseline["qualification_data_may_reenter_development"] is False


def test_v3_freezes_balanced_fact_assignment_and_unexecuted_twin_audit() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    relevance = raw["longitudinal_design"]["long_context_causal_relevance"]
    assignment = relevance["fact_value_assignment"]
    twin = relevance["counterfactual_twin_proxy_audit"]

    assert relevance["minimum_decisive_long_range_facts_per_subject"] == 8
    assert relevance["final_decision_fact_to_query_distance_tokens_minimum"] == 32_768
    assert relevance["fact_restatement_after_origin_permitted"] is False
    assert relevance["counterfactual_fact_value_must_change_unique_optimal_typed_action"]
    assert assignment["closed_typed_domain"] == "binary_relationship_preference_value"
    assert assignment["counter_based_rng_artifact_frozen_before_source_materialization"]
    assert assignment["independent_of_root_surface_tape_arm_and_outcome"]
    assert assignment["balanced_within_each_split_and_distance_bin"]
    assert assignment["maximum_value_count_imbalance"] == 1

    assert twin["one_unexecuted_source_twin_per_analysis_root"]
    assert twin["twin_is_not_an_analysis_arm_or_model_run"]
    assert twin["twin_differs_only_in_fact_origin_typed_value_and_oracle_descendants"]
    assert twin["masking_fact_origin_makes_all_later_exogenous_tape_bytes_exactly_equal"]
    assert twin["forced_learning_actions_and_pre_evaluation_endogenous_receipts_exactly_equal"]
    assert twin["post_origin_public_event_or_reaction_may_encode_fact_value"] is False
    assert twin["source_dependency_graph_allows_fact_value_parents_only_for_origin_and_evaluation_utility_oracle"]
    assert twin["semantic_proxy_scan_artifact_and_independent_reobserver_required"]
    assert twin["proxy_audit_failure_status"] == "invalid_attempt_no_claim"


def test_v3_freezes_all_nine_generated_action_lineage_and_execution_order() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    global_action = raw["causal_execution"]["global_generated_evaluation_action_contract"]
    schedule = raw["causal_execution"]["counterbalanced_execution_schedule"]

    assert tuple(raw["arm_matrix"]) == _EXPECTED_V3_ARM_MATRIX
    assert raw["integrated_arms"] == [
        "qwen_steelman_full_history",
        "qwen_steelman_selective_rag",
        "volvence_closed_loop",
    ]
    assert global_action["applies_to_all_nine_arms"]
    assert global_action["applies_to_all_frozen_policy_evaluation_decisions"]
    assert global_action["actual_action_source"] == "model_generated_tokens_only"
    assert global_action["parser_type"] == "frozen_deterministic_closed_world_parser"
    assert global_action["repair_reprompt_or_second_parse_permitted"] is False
    assert global_action["hidden_direct_action_channel_permitted"] is False
    assert global_action["delivered_assistant_bytes_equal_generated_bytes"]
    assert global_action["parsed_candidate_count_must_equal"] == 1
    assert global_action["environment_accepts_only_generation_parented_actual_action"]
    assert global_action["only_exception"] == "preallocated_forced_actions_in_matched_learning_collection"
    assert global_action["exception_is_excluded_from_steerable_and_evaluation_endpoints"]

    assert schedule["block_unit"] == "subject_root_x_session_index"
    assert schedule["each_block_contains_each_of_nine_arms_exactly_once"]
    assert schedule["arm_order_algorithm"] == "sha256_counter_fisher_yates_with_rejection_sampling"
    assert schedule["each_arm_ordinal_position_count_differs_by_at_most_one_per_split"]
    assert schedule["root_block_order_interleaved_by_frozen_schedule"]
    assert schedule["arm_or_root_is_never_systematically_earlier_in_host_time"]
    assert schedule["schedule_uses_no_arm_output_outcome_or_failure"]
    assert schedule["same_host_boot_identity_required_within_each_block"]
    assert schedule["donor_bank_stage_snapshots_sealed_before_target_blocks"]


def test_v3_freezes_exact_bootstrap_and_full_joint_power_dgp() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    decision = raw["analysis"]["decision_rule"]
    power = raw["analysis"]["power"]

    assert decision["bootstrap_replicates"] == 100_000
    assert decision["bootstrap_seed"] == 20_260_823
    assert decision["bootstrap_index_algorithm"] == "sha256_counter_u64_big_endian_rejection_then_modulo_192"
    assert decision["bootstrap_index_domain"] == "replicate_index_x_draw_index_x_rejection_counter"
    assert decision["bootstrap_index_plan_artifact_required_before_first_development_output"]
    assert decision["joint_resample_preserves_all_arm_session_values_for_each_root"]
    assert decision["quantile_method"] == "inverted_cdf"
    assert decision["lower_order_statistic_zero_based_index"] == 4_999
    assert decision["primary_arithmetic"] == "exact_integer_utilities_and_rational_means_counts_and_p_values"
    assert decision["holm_comparisons_use_exact_cross_multiplied_rationals"]
    assert decision["all_eight_unique_contrasts_must_pass"]

    assert power["full_decision_rule_power_target_decimal"] == "0.80"
    assert power["prior_power_artifact_required_before_first_development_output"]
    assert power["qualification_power_recheck_required_before_formal_unseal"]
    assert power["qualification_effect_means_used"] is False
    assert power["sample_size_adaptation_permitted"] is False
    assert power["threshold_or_family_adaptation_permitted"] is False
    assert power["power_simulation_replicates"] == 100_000
    assert power["power_simulation_seed"] == 20_260_824
    assert power["power_simulation_index_algorithm"] == "sha256_counter_based_exact_rational_categorical_draws"
    assert power["joint_dgp_artifact_required_before_first_development_output"]
    assert power["joint_dgp_required_contents"] == [
        "all_9_arm_x_8_evaluation_discrete_utility_probability_generator",
        "all_8_contrast_marginal_means_variances_and_cross_contrast_covariance",
        "within_root_temporal_dependence_and_challenge_strata",
        "source_exact_mechanical_opportunity_mix_and_structural_covariance_bounds",
        "technical_missingness_rate_and_reference_comparator_both_missing_patterns",
        "worst_case_itt_imputation_before_root_mean",
        "full_practical_bootstrap_holm_and_claim_conjunction_decision_rule",
        "rational_probability_masses_rng_index_plan_and_source_hashes",
    ]
    assert power["mandatory_variance_scenarios"] == [
        "source_structural_covariance_upper_bound",
        "maximum_feasible_bounded_difference_variance_at_planning_mean",
        "paired_root_difference_variance_0_25",
        "paired_root_difference_variance_0_50",
        "paired_root_difference_variance_1_00",
    ]
    assert power["mandatory_within_root_icc_decimals"] == ["0.00", "0.25", "0.50"]
    assert power["mandatory_cross_contrast_dependence"] == [
        "source_structural_covariance",
        "independent_contrasts",
        "equicorrelation_negative_0_10",
        "equicorrelation_positive_0_50",
    ]
    assert power["mandatory_technical_missingness_rate_decimals"] == ["0.00", "0.01", "0.02"]
    assert power["mandatory_missingness_patterns"] == [
        "balanced_independent",
        "reference_only_correlated_worst_case",
        "comparator_only_correlated_worst_case",
        "both_arm_correlated_worst_case",
    ]
    assert power["all_feasible_cartesian_scenario_combinations_must_pass"]
    assert power["zero_missing_or_arbitrarily_low_variance_scenario_alone_may_authorize"] is False
    assert power["each_unique_contrast_axis_conjunction_and_integrated_joint_power_at_least_decimal"] == "0.80"


def test_v2_freezes_unique_contrasts_practical_gate_and_itt() -> None:
    raw = _protocol_raw(P4_LONG_CONTEXT_PROTOCOL_ID_V2)
    contrasts = raw["analysis"]["confirmatory_contrasts"]
    ids = [item["contrast_id"] for item in contrasts]
    pairs = [(item["reference_arm"], item["comparator_arm"]) for item in contrasts]

    assert len(contrasts) == len(set(ids)) == len(set(pairs)) == 8
    assert all(item["holm_member"] for item in contrasts)
    assert raw["analysis"]["decision_rule"]["minimum_practical_point_estimate_delta_decimal"] == "0.15"
    assert raw["analysis"]["decision_rule"]["contrast_pass_requires_lineage_practical_ci_and_holm"]
    missingness = raw["analysis"]["missingness_and_attrition"]
    assert missingness["all_192_roots_remain_in_each_contrast"]
    assert missingness["complete_case_primary_analysis_permitted"] is False
    assert missingness["missing_reference_utility"] == -1
    assert missingness["missing_comparator_utility"] == 1
    assert missingness["both_missing_contrast_value"] == -2
    assert missingness["integrity_or_lineage_violation_status"] == "invalid_attempt_no_claim"


def test_v2_qualification_and_power_are_one_shot_nonadaptive() -> None:
    raw = _protocol_raw(P4_LONG_CONTEXT_PROTOCOL_ID_V2)
    qualification = raw["cohort"]["splits"]["qualification"]
    formal = raw["cohort"]["splits"]["formal"]
    power = raw["analysis"]["power"]

    assert qualification["one_shot"]
    assert qualification["campaign_retry_count"] == 0
    assert qualification["session_retry_count"] == 0
    assert qualification["subject_replacement_permitted"] is False
    assert formal["one_shot"]
    assert formal["campaign_retry_count"] == 0
    assert formal["session_retry_count"] == 0
    assert formal["intention_to_treat_subject_count"] == 192
    assert power["prior_power_artifact_required_before_first_development_output"]
    assert power["qualification_power_recheck_required_before_formal_unseal"]
    assert power["qualification_effect_means_used"] is False
    assert power["sample_size_adaptation_permitted"] is False
    assert power["threshold_or_family_adaptation_permitted"] is False


def test_v2_freezes_realized_single_variable_and_reactive_lineage() -> None:
    raw = _protocol_raw(P4_LONG_CONTEXT_PROTOCOL_ID_V2)
    axes = raw["axis_contrasts"]
    pointers = raw["intervention_integrity"]["allowed_exogenous_difference_json_pointers"]

    assert pointers["readable_label_permuted"] == ["/components/condition_label_map/artifact_id"]
    assert axes["readable"]["reader_artifact_id_same_across_pair"]
    assert axes["readable"]["raw_logits_dtype_shape_and_bytes_same_at_first_intervention_boundary"]
    assert axes["readable"]["consumer_may_access_raw_logits_or_class_index"] is False

    learnable = axes["learnable"]
    assert learnable["gate_mutation_during_collection"] is False
    assert learnable["credit_computed_and_published_in_both_arms"]
    assert learnable["update_timing"] == ("single_atomic_batch_after_all_matched_exposures")
    assert learnable["pre_update_command_pair_diff_exact_json_pointers"] == ["/intervention/apply_exact_pe_credit"]
    assert learnable["withheld_parameter_delta_exact_zero"]

    steerable = axes["steerable"]
    assert steerable["free_bias_present"] is False
    assert steerable["same_turn_manipulation_diagnostic"]["diagnostic_is_not_longitudinal_primary_endpoint"]
    assert steerable["longitudinal_cross_arm_residual_equality_after_first_action_required"] is False
    assert steerable["generated_action_lineage"]["actual_action_source"] == "model_generated_tokens_only"
    assert steerable["sensor_off_formal_capacity_gate_required"]

    shared = raw["shared_exposure_schedule"]
    assert shared["shared_exogenous_tape"]["realized_user_reaction_is_part_of_tape"] is False
    assert shared["endogenous_reaction"]["may_diverge_across_arms_as_action_descendant"]
    assert shared["endogenous_reaction"]["learnable_collection_pair_must_be_exactly_equal"]


def test_v2_long_context_is_causally_relevant_and_bound_to_actual_inputs() -> None:
    raw = _protocol_raw(P4_LONG_CONTEXT_PROTOCOL_ID_V2)
    longitudinal = raw["longitudinal_design"]
    relevance = longitudinal["long_context_causal_relevance"]
    receipt = longitudinal["model_input_receipt"]

    assert relevance["minimum_decisive_long_range_facts_per_subject"] == 8
    assert relevance["final_decision_fact_to_query_distance_tokens_minimum"] == 32768
    assert relevance["fact_restatement_after_origin_permitted"] is False
    assert relevance["counterfactual_fact_value_must_change_unique_optimal_typed_action"]
    assert receipt["required_for_every_arm_and_evaluation_decision"]
    assert "rendered_input_token_ids_sha256" in receipt["required_fields"]
    assert "attention_mask_sha256" in receipt["required_fields"]
    assert receipt["truncation_must_be_false"]
    assert receipt["total_input_plus_headroom_must_not_exceed_native_window"]
    assert longitudinal["runtime_context_extension_or_rope_scaling_permitted"] is False


def test_protocol_separates_development_cuda_from_formal_authorization() -> None:
    raw = _protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"])
    admission = raw["execution_admission"]

    assert admission["development_cuda_diagnostics_allowed"] is True
    assert admission["development_cuda_diagnostics_are_formal_evidence"] is False
    assert admission["user_development_cuda_consent_is_formal_authorization"] is False
    assert admission["manual_override_permitted"] is False
    assert admission["environment_override_permitted"] is False
    assert admission["ignore_microcode_override_permitted"] is False
    assert admission["force_cli_permitted"] is False
    assert admission["existing_p4_6_fit_native_window_tokens"] == 32_768
    assert admission["existing_p4_6_fit_may_authorize_long_context_formal"] is False
    assert raw["evidence_firewall"]["integrated_four_axis_supported"] is False


@pytest.mark.parametrize(
    ("version", "artifact_root"),
    [
        ("v1", _V1_ARTIFACT),
        ("v2", _V2_ARTIFACT),
    ],
)
def test_superseded_protocols_are_preserved_auto_validatable_and_cannot_be_republished(
    version: str,
    artifact_root: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    protocol_id = _EXPECTED_PROTOCOL_IDS[version]
    protocol_path = relationship_p4_long_context_protocol_path(protocol_id)
    protocol = load_relationship_p4_long_context_scientific_prereg(protocol_path)
    assert protocol.protocol_id == protocol_id
    assert protocol.superseded

    validated = validate_relationship_p4_long_context_scientific_prereg(output_dir=artifact_root)
    assert validated.artifact_id == _EXPECTED_PUBLISHED_ARTIFACT_IDS[version]
    with pytest.raises(ValueError, match="superseded protocol cannot publish"):
        prepare_relationship_p4_long_context_scientific_prereg(
            output_dir=tmp_path / f"{version} forbidden",
            protocol_path=protocol_path,
        )


def test_prepare_v3_is_create_only_zero_output_and_auto_validatable(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "p4.7 v3 prereg"
    prepared = prepare_relationship_p4_long_context_scientific_prereg(output_dir=output)

    assert prepared.status == P4_LONG_CONTEXT_PREPARATION_STATUS
    assert prepared.status == "scientific_prereg_v3_frozen_execution_envelope_absent"
    assert prepared.protocol_id == _EXPECTED_PROTOCOL_IDS["v3"]
    assert prepared.artifact_id == _EXPECTED_PUBLISHED_ARTIFACT_IDS["v3"]
    assert prepared.execution_enabled is False
    assert prepared.formal_run_authorized is False
    preparation = json.loads((output / "scientific_prereg_preparation.json").read_text(encoding="utf-8"))
    assert preparation["model_output_count"] == 0
    assert preparation["subject_pack_materialization_count"] == 0
    assert preparation["cuda_formal_execution_count"] == 0
    assert preparation["formal_result_count"] == 0
    assert preparation["donor_bank_materialization_count"] == 0
    assert preparation["counterfactual_twin_materialization_count"] == 0
    assert preparation["power_dgp_artifact_count"] == 0

    validated = validate_relationship_p4_long_context_scientific_prereg(output_dir=output)
    assert validated.artifact_id == prepared.artifact_id
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_relationship_p4_long_context_scientific_prereg(output_dir=output)


@pytest.mark.parametrize(
    "protocol_id",
    [
        _EXPECTED_PROTOCOL_IDS["v1"],
        _EXPECTED_PROTOCOL_IDS["v2"],
        _EXPECTED_PROTOCOL_IDS["v3"],
    ],
)
def test_canonical_object_key_reordering_preserves_acceptance(
    protocol_id: str,
    tmp_path: pathlib.Path,
) -> None:
    raw = _protocol_raw(protocol_id)
    canonical = tmp_path / f"{protocol_id}.json"
    canonical.write_bytes(_canonical_bytes(raw))

    loaded = load_relationship_p4_long_context_scientific_prereg(canonical)
    assert loaded.protocol_id == protocol_id


@pytest.mark.parametrize(
    ("protocol_id", "exception"),
    [
        ("", ValueError),
        (0, TypeError),
        (False, TypeError),
    ],
)
def test_falsey_protocol_ids_fail_loudly(protocol_id: object, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        relationship_p4_long_context_protocol_path(protocol_id)  # type: ignore[arg-type]


def test_falsey_protocol_paths_do_not_select_the_default() -> None:
    with pytest.raises(FileNotFoundError, match="must be a regular file"):
        load_relationship_p4_long_context_scientific_prereg("")  # type: ignore[arg-type]


def test_protocol_rejects_content_drift_duplicate_keys_and_bom(
    tmp_path: pathlib.Path,
) -> None:
    source = relationship_p4_long_context_protocol_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["cohort"]["splits"]["formal"]["preallocated_subject_count"] = 20
    drifted = tmp_path / "drifted.json"
    drifted.write_bytes(_canonical_bytes(payload))
    with pytest.raises(ValueError, match="unregistered or drifted protocol id"):
        load_relationship_p4_long_context_scientific_prereg(drifted)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        source.read_text(encoding="utf-8").replace(
            "{",
            '{"schema_version":"duplicate",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_relationship_p4_long_context_scientific_prereg(duplicate)

    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + source.read_bytes())
    with pytest.raises(ValueError, match="must not carry a UTF-8 BOM"):
        load_relationship_p4_long_context_scientific_prereg(bom)


@pytest.mark.parametrize(
    ("section", "mutator"),
    [
        (
            "axis_contrasts",
            lambda raw: raw["axis_contrasts"]["steerable"].__setitem__("free_bias_present", True),
        ),
        (
            "cohort",
            lambda raw: raw["cohort"]["splits"]["formal"].__setitem__("outcome_based_replacement_permitted", True),
        ),
        (
            "cohort",
            lambda raw: raw["cohort"]["splits"]["qualification"].__setitem__("one_shot", False),
        ),
        (
            "analysis",
            lambda raw: raw["analysis"]["decision_rule"].__setitem__(
                "minimum_practical_point_estimate_delta_decimal", "0.01"
            ),
        ),
        (
            "analysis",
            lambda raw: raw["analysis"]["confirmatory_contrasts"][5].__setitem__("status", "optional"),
        ),
        (
            "execution_admission",
            lambda raw: raw["execution_admission"]["future_execution_envelope_required_fields"].__setitem__(
                0, "arbitrary"
            ),
        ),
        (
            "stopping_rules",
            lambda raw: raw["stopping_rules"].__setitem__(0, "arbitrary"),
        ),
        (
            "evidence_firewall",
            lambda raw: raw["evidence_firewall"].pop("integrated_four_axis_supported"),
        ),
    ],
)
def test_v3_semantic_section_anchors_reject_drift(
    section: str,
    mutator: object,
) -> None:
    raw = copy.deepcopy(_protocol_raw(_EXPECTED_PROTOCOL_IDS["v3"]))
    mutator(raw)  # type: ignore[operator]

    with pytest.raises(ValueError, match=f"P4.7 v3 frozen section drift: {section}"):
        owner._validate_v3_frozen_sections(raw)


def test_artifact_root_symlink_and_dangling_output_symlink_fail_loudly(
    tmp_path: pathlib.Path,
) -> None:
    artifact_alias = tmp_path / "artifact alias"
    _create_directory_alias_or_skip(artifact_alias, _V3_ARTIFACT)
    try:
        with pytest.raises(ValueError, match="symlink|reparse point"):
            validate_relationship_p4_long_context_scientific_prereg(output_dir=artifact_alias)
    finally:
        _remove_directory_alias(artifact_alias)

    missing_target = tmp_path / "must remain absent"
    dangling_output = tmp_path / "dangling output"
    _create_directory_alias_or_skip(dangling_output, missing_target)
    try:
        with pytest.raises(ValueError, match="symlink|reparse point"):
            prepare_relationship_p4_long_context_scientific_prereg(output_dir=dangling_output)
    finally:
        _remove_directory_alias(dangling_output)
    assert not missing_target.exists()


def test_validator_rejects_forged_unregistered_lineage_and_cross_version_splice(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source"
    prepare_relationship_p4_long_context_scientific_prereg(output_dir=source)

    forged = tmp_path / "forged"
    shutil.copytree(source, forged)
    preparation_path = forged / "scientific_prereg_preparation.json"
    manifest_path = forged / "manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_id = "a" * 64
    preparation["protocol_id"] = forged_id
    preparation_bytes = _canonical_bytes(preparation)
    preparation_path.write_bytes(preparation_bytes)
    manifest["protocol_id"] = forged_id
    manifest["files"][0]["byte_count"] = len(preparation_bytes)
    manifest["files"][0]["sha256"] = hashlib.sha256(preparation_bytes).hexdigest()
    del manifest["artifact_id"]
    manifest["artifact_id"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path.write_bytes(_canonical_bytes(manifest))
    with pytest.raises(ValueError, match="unregistered protocol id"):
        validate_relationship_p4_long_context_scientific_prereg(output_dir=forged)

    spliced = tmp_path / "spliced"
    shutil.copytree(source, spliced)
    spliced_manifest_path = spliced / "manifest.json"
    spliced_manifest = json.loads(spliced_manifest_path.read_text(encoding="utf-8"))
    spliced_manifest["protocol_id"] = P4_LONG_CONTEXT_PROTOCOL_ID_V1
    spliced_manifest_path.write_bytes(_canonical_bytes(spliced_manifest))
    with pytest.raises(ValueError, match="protocol id mismatch"):
        validate_relationship_p4_long_context_scientific_prereg(output_dir=spliced)


def test_validator_rejects_resealed_output_claim_extra_file_and_reordered_artifact(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source"
    prepare_relationship_p4_long_context_scientific_prereg(output_dir=source)

    tampered = tmp_path / "tampered"
    shutil.copytree(source, tampered)
    preparation_path = tampered / "scientific_prereg_preparation.json"
    manifest_path = tampered / "manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["model_output_count"] = 1
    preparation_bytes = _canonical_bytes(preparation)
    preparation_path.write_bytes(preparation_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["byte_count"] = len(preparation_bytes)
    manifest["files"][0]["sha256"] = hashlib.sha256(preparation_bytes).hexdigest()
    del manifest["artifact_id"]
    manifest["artifact_id"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path.write_bytes(_canonical_bytes(manifest))
    with pytest.raises(ValueError, match="model_output_count value drift"):
        validate_relationship_p4_long_context_scientific_prereg(output_dir=tampered)

    extra = tmp_path / "extra"
    shutil.copytree(source, extra)
    (extra / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set drift"):
        validate_relationship_p4_long_context_scientific_prereg(output_dir=extra)

    reordered = tmp_path / "reordered"
    shutil.copytree(source, reordered)
    reordered_preparation = reordered / "scientific_prereg_preparation.json"
    value = json.loads(reordered_preparation.read_text(encoding="utf-8"))
    reordered_preparation.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_relationship_p4_long_context_scientific_prereg(output_dir=reordered)


def test_owner_has_no_cognition_internal_import_or_execution_bypass() -> None:
    tree = ast.parse(_OWNER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(
        name.startswith(
            (
                "volvence_zero.credit",
                "volvence_zero.prediction_error",
                "volvence_zero.social",
                "volvence_zero.steering_sensor",
            )
        )
        for name in imported
    )

    owner_source = _OWNER_SOURCE.read_text(encoding="utf-8")
    cli = _CLI_SOURCE.read_text(encoding="utf-8")
    assert "run-session" not in cli
    assert "--force" not in cli
    assert "ignore-host" not in cli
    assert "os.environ" not in cli
    assert 'validate.add_argument("--protocol"' not in cli
    assert "allow-unregistered" not in owner_source


def _protocol_raw(protocol_id: str) -> dict[str, object]:
    path = relationship_p4_long_context_protocol_path(protocol_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


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
