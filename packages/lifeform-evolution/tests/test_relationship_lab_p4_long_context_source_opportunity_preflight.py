from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Callable

import pytest

import lifeform_evolution.relationship_lab_p4_long_context_causal_campaign as owner
import lifeform_evolution.relationship_lab_p4_long_context_source_opportunity_derivation as derivation
from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
    load_relationship_p4_long_context_source_opportunity_preflight_protocol,
    prepare_relationship_p4_long_context_source_opportunity_preflight,
    validate_relationship_p4_long_context_source_opportunity_preflight,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "packages" / "lifeform-evolution"
_OWNER_SOURCE = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "relationship_lab_p4_long_context_causal_campaign.py"
)
_HELPER_SOURCE = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "relationship_lab_p4_long_context_source_opportunity_derivation.py"
)
_PROTOCOL_PATH = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "protocols"
    / "relationship_p4_long_context_v4_source_opportunity_preflight_v1.json"
)
_ACTION_SCHEMA_PATH = (
    _PACKAGE_ROOT
    / "src"
    / "lifeform_evolution"
    / "schemas"
    / "relationship_action_choice.schema.json"
)
_ACTION_OWNER_SOURCE = (
    _REPO_ROOT
    / "packages"
    / "lifeform-domain-emogpt"
    / "src"
    / "lifeform_domain_emogpt"
    / "relationship_action_contracts.py"
)
_CLI_SOURCE = _REPO_ROOT / "scripts" / "run_relationship_lab_p4_long_context_source_opportunity_preflight.py"
_V4A_PLANNING = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_v4a_zero_output_planning_20260823"
)
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
_SOURCE_PREFLIGHT_ARTIFACT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_independent_long_context_source_opportunity_preflight_v1_20260823"
)

_PROJECTION_FILE = "source_opportunity_contract_projection.json"
_CERTIFICATE_FILE = "source_opportunity_preflight_certificate.json"
_MANIFEST_FILE = "manifest.json"

_EXPECTED_PROTOCOL_ID = "47bcf6561be1ace0698cc0f96e2e7e35701f46d15baac9eb87ad1d662576494a"
_EXPECTED_PROTOCOL_RAW = "9d4d3ab5cb683d8ff5827e5047e5b176800fe5c4e86ad6a07217b7a2040c40b0"
_EXPECTED_HELPER_RAW = "72efc093b815c2ca07872f6cb6a78f53a4d4d5ada5975222b36cf90c640746f8"
_EXPECTED_ACTION_SCHEMA_RAW = "764309ff7b1d4aa6e9001a73a8c72407a1fabfd1e9d5c89e7cdf37360054efea"
_EXPECTED_ACTION_OWNER_RAW = "dc1907cc67d76536b88894f5e06c907ec4651a76acab0fe28531ffe14db2b526"
_EXPECTED_ACTION_REGISTRY_ID = "5b6250960c43401d7a14f463f0cc32c7518735c0aba6bf0e855e0e55f8a45fcc"
_EXPECTED_V4A_ARTIFACT_ID = "082454002260db90b7236a1104311a5d92cc3959171bb3190e7a30f8387e56c1"
_EXPECTED_V4A_CERTIFICATE_ID = "b7e95f149afe77b283bf135f7cb5d76eb4f4edee4594c8649a778acb4186c764"
_EXPECTED_V3_PREPARATION_ID = "c5a708ae5e68261fddbade165b45579e66e4bbe7db1be1f4a83056561a17f42e"
_EXPECTED_V2_ADMISSION_ID = "9883e10784a06260a220a6fdbf72141b1300c21e97faee6e84a401c40a144ee9"
_EXPECTED_PROJECTION_ID = "b8b7823a6fd2c7ad706c4ffa143438b730da667c26a925f0be87df14212e6f1b"
_EXPECTED_CERTIFICATE_ID = "64d879c4f41ca873f8e40f0344234771343f6efee229b668914b61d31c96c95a"
_EXPECTED_ARTIFACT_ID = "8a36d2de9077bb5550db8018338eded27b6ce30d77eea17739ffe35b73e00a99"
_EXPECTED_PROJECTION_RAW = "ee33fa32a3829cbaaa1c92022016c184197b4cd97e08818af19b98024b4866b2"
_EXPECTED_CERTIFICATE_RAW = "f9089ce08e6868d402a753ccd3247024a0170a96e8cad9f411f659e913300736"
_EXPECTED_MANIFEST_RAW = "2829b16d674ae9efe971eaa668610f80a20799e45ac47e208ec4e7a3261760a6"

_EXPECTED_ACTIONS = (
    "stay_present_without_probe",
    "respect_space_with_return_option",
    "neutral_noop",
)
_EXPECTED_PROJECTION_KEYS = frozenset(
    {
        "claim_boundary",
        "contract_projection_id",
        "contract_projection_id_contract",
        "frozen_contract_sections",
        "identity",
        "mechanical_projection",
        "schema_version",
        "stage_boundary",
        "terminal",
        "upstream_raw_lineage",
        "zero_output_firewall",
    }
)
_EXPECTED_CERTIFICATE_KEYS = frozenset(
    {
        "certificate_id",
        "certificate_id_contract",
        "claim_boundary",
        "identity",
        "schema_version",
        "stage",
        "status",
        "terminal",
        "validation_receipts",
        "zero_output_firewall",
    }
)
_EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "artifact_id",
        "certificate_id",
        "claim_boundary",
        "contract_projection_id",
        "counterfactual_twin_materialization_count",
        "cuda_planner_authorized",
        "cuda_planner_run_count",
        "current_source_execution_authorized",
        "development_authorized",
        "donor_bank_materialization_count",
        "empirical_outcome_count",
        "external_publication_anchor_present",
        "files",
        "formal_authorized",
        "model_output_authorized",
        "model_output_count",
        "planning_atom_materialization_count",
        "qualification_authorized",
        "schema_version",
        "selected_formal_root_count",
        "source_grid_resolved",
        "source_opportunity_constraint_row_count",
        "source_opportunity_stage_completed",
        "source_preflight_protocol_id",
        "source_structural_inventory_artifact_count",
        "source_structural_inventory_artifact_id",
        "source_structural_inventory_materialized",
        "status",
        "subject_materialization_count",
        "tuple_feasibility_authorized",
        "unresolved_tuple_count",
        "v4a_planning_artifact_id",
        "zero_output_preflight_contract_frozen",
    }
)
_EXPECTED_FIREWALL_KEYS = frozenset(
    {
        "NTFS_alternate_data_stream_absence_claimed",
        "administrator_level_concurrent_reparse_or_WORM_resistance_claimed",
        "all_materialization_counts_refer_to_persisted_or_published_artifact_rows_not_ephemeral_in_memory_exact_derivation_objects",
        "appendable_formal_supported",
        "baseline_output_count",
        "counterfactual_twin_materialization_count",
        "cuda_formal_run_count",
        "cuda_planner_run_count",
        "donor_bank_materialization_count",
        "empirical_outcome_count",
        "ephemeral_in_memory_structural_derivation_objects_are_not_source_content_or_persistent_materialization",
        "full_nine_arm_planning_atom_materialization_count",
        "integrated_four_axis_supported",
        "learnable_formal_supported",
        "machine_global_history_without_source_claimed",
        "malicious_operator_seed_grinding_resistance_claimed",
        "model_output_count",
        "power_confirmation_replicate_count",
        "power_search_replicate_count",
        "production_active_authorized",
        "public_tape_entry_count",
        "readable_formal_supported",
        "real_human_product_value_claimed",
        "source_opportunity_constraint_row_count",
        "source_structural_inventory_artifact_count",
        "source_text_surface_count",
        "steerable_formal_supported",
        "subject_materialization_count",
        "tuple_feasibility_membership_count",
    }
)


def test_literal_pins_and_real_published_artifact_replay() -> None:
    protocol_bytes = _PROTOCOL_PATH.read_bytes()
    protocol_payload = _strict_json(_PROTOCOL_PATH)
    helper_bytes = _HELPER_SOURCE.read_bytes()
    schema_bytes = _ACTION_SCHEMA_PATH.read_bytes()
    action_owner_bytes = _ACTION_OWNER_SOURCE.read_bytes()

    assert hashlib.sha256(protocol_bytes).hexdigest() == _EXPECTED_PROTOCOL_RAW
    assert hashlib.sha256(_canonical_bytes(protocol_payload)).hexdigest() == _EXPECTED_PROTOCOL_ID
    assert hashlib.sha256(helper_bytes).hexdigest() == _EXPECTED_HELPER_RAW
    assert hashlib.sha256(schema_bytes).hexdigest() == _EXPECTED_ACTION_SCHEMA_RAW
    assert hashlib.sha256(action_owner_bytes).hexdigest() == _EXPECTED_ACTION_OWNER_RAW
    assert owner.P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1 == _EXPECTED_PROTOCOL_ID
    assert owner.P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1 == (
        _EXPECTED_PROTOCOL_RAW
    )
    assert owner.P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1 == _EXPECTED_ARTIFACT_ID
    assert owner.P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1 == (
        _EXPECTED_CERTIFICATE_ID
    )

    action_registry_core = {
        "owner_module": "lifeform_domain_emogpt.relationship_action_contracts",
        "owner_module_raw_sha256": _EXPECTED_ACTION_OWNER_RAW,
        "schema_id": "https://volvence.local/schemas/relationship-action-choice.v1.json",
        "schema_raw_sha256": _EXPECTED_ACTION_SCHEMA_RAW,
        "action_ids": list(_EXPECTED_ACTIONS),
    }
    assert hashlib.sha256(_canonical_bytes(action_registry_core)).hexdigest() == _EXPECTED_ACTION_REGISTRY_ID

    projection_path = _SOURCE_PREFLIGHT_ARTIFACT / _PROJECTION_FILE
    certificate_path = _SOURCE_PREFLIGHT_ARTIFACT / _CERTIFICATE_FILE
    manifest_path = _SOURCE_PREFLIGHT_ARTIFACT / _MANIFEST_FILE
    projection_bytes = projection_path.read_bytes()
    certificate_bytes = certificate_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    projection = _strict_json(projection_path)
    certificate = _strict_json(certificate_path)
    manifest = _strict_json(manifest_path)
    assert projection_bytes == _canonical_bytes(projection)
    assert certificate_bytes == _canonical_bytes(certificate)
    assert manifest_bytes == _canonical_bytes(manifest)
    assert hashlib.sha256(projection_bytes).hexdigest() == _EXPECTED_PROJECTION_RAW
    assert hashlib.sha256(certificate_bytes).hexdigest() == _EXPECTED_CERTIFICATE_RAW
    assert hashlib.sha256(manifest_bytes).hexdigest() == _EXPECTED_MANIFEST_RAW

    projection_core = dict(projection)
    assert projection_core.pop("contract_projection_id") == _EXPECTED_PROJECTION_ID
    assert hashlib.sha256(_canonical_bytes(projection_core)).hexdigest() == _EXPECTED_PROJECTION_ID
    certificate_core = dict(certificate)
    assert certificate_core.pop("certificate_id") == _EXPECTED_CERTIFICATE_ID
    assert hashlib.sha256(_canonical_bytes(certificate_core)).hexdigest() == _EXPECTED_CERTIFICATE_ID
    manifest_core = dict(manifest)
    assert manifest_core.pop("artifact_id") == _EXPECTED_ARTIFACT_ID
    assert hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest() == _EXPECTED_ARTIFACT_ID

    registry_module = "lifeform_domain_emogpt.relationship_action_contracts"
    registry_was_imported = registry_module in sys.modules
    loaded = load_relationship_p4_long_context_source_opportunity_preflight_protocol()
    assert (registry_module in sys.modules) is registry_was_imported
    assert loaded.protocol_id == _EXPECTED_PROTOCOL_ID
    assert loaded.v4a_planning_artifact_id == _EXPECTED_V4A_ARTIFACT_ID
    assert loaded.action_registry_id == _EXPECTED_ACTION_REGISTRY_ID
    assert loaded.action_ids == _EXPECTED_ACTIONS
    assert loaded.independent_root_slot_count == 16576
    assert loaded.counterfactual_twin_mapping_count == 8288
    assert loaded.formal_candidate_prefix_count == 126
    assert loaded.generic_decision_atom_count == 512
    assert loaded.zero_output_preflight_contract_frozen is True
    assert loaded.source_opportunity_stage_completed is False
    assert loaded.source_structural_inventory_materialized is False
    assert loaded.future_structural_inventory_scope_defined is True
    assert loaded.future_structural_inventory_materialization_authorized is False

    replayed = validate_relationship_p4_long_context_source_opportunity_preflight(
        output_dir=_SOURCE_PREFLIGHT_ARTIFACT,
        v4a_planning_dir=_V4A_PLANNING,
        v3_preparation_dir=_V3_PREPARATION,
        v2_admission_dir=_V2_ADMISSION,
    )
    assert replayed.artifact_id == _EXPECTED_ARTIFACT_ID
    assert replayed.certificate_id == _EXPECTED_CERTIFICATE_ID
    assert replayed.contract_projection_id == _EXPECTED_PROJECTION_ID
    assert replayed.protocol_id == _EXPECTED_PROTOCOL_ID
    assert replayed.v4a_planning_artifact_id == _EXPECTED_V4A_ARTIFACT_ID
    assert replayed.action_registry_id == _EXPECTED_ACTION_REGISTRY_ID
    assert replayed.zero_output_preflight_contract_frozen is True
    assert replayed.source_opportunity_stage_completed is False
    assert replayed.source_structural_inventory_materialized is False
    assert replayed.unresolved_tuple_count == 576
    assert replayed.selected_formal_root_count is None
    assert replayed.current_source_execution_authorized is False
    assert replayed.tuple_feasibility_authorized is False
    assert replayed.model_output_authorized is False
    assert replayed.development_authorized is False
    assert replayed.qualification_authorized is False
    assert replayed.formal_authorized is False
    assert replayed.cuda_planner_authorized is False


def test_zero_output_artifact_has_exact_file_key_and_materialization_counts() -> None:
    assert {item.name for item in _SOURCE_PREFLIGHT_ARTIFACT.iterdir()} == {
        _PROJECTION_FILE,
        _CERTIFICATE_FILE,
        _MANIFEST_FILE,
    }
    projection = _strict_json(_SOURCE_PREFLIGHT_ARTIFACT / _PROJECTION_FILE)
    certificate = _strict_json(_SOURCE_PREFLIGHT_ARTIFACT / _CERTIFICATE_FILE)
    manifest = _strict_json(_SOURCE_PREFLIGHT_ARTIFACT / _MANIFEST_FILE)
    assert set(projection) == _EXPECTED_PROJECTION_KEYS
    assert len(projection) == 11
    assert set(certificate) == _EXPECTED_CERTIFICATE_KEYS
    assert len(certificate) == 10
    assert set(manifest) == _EXPECTED_MANIFEST_KEYS
    assert len(manifest) == 33
    assert set(projection["mechanical_projection"]) == {
        "action_and_opportunity_layout",
        "balanced_fact_orientation",
        "generic_decision_planning_generator",
        "root_surface_layout",
    }

    assert manifest["files"] == [
        {
            "byte_count": 72969,
            "path": _PROJECTION_FILE,
            "sha256": _EXPECTED_PROJECTION_RAW,
        },
        {
            "byte_count": 8036,
            "path": _CERTIFICATE_FILE,
            "sha256": _EXPECTED_CERTIFICATE_RAW,
        },
    ]
    assert manifest["source_preflight_protocol_id"] == _EXPECTED_PROTOCOL_ID
    assert manifest["v4a_planning_artifact_id"] == _EXPECTED_V4A_ARTIFACT_ID
    assert manifest["source_structural_inventory_artifact_id"] is None
    assert manifest["selected_formal_root_count"] is None
    assert manifest["unresolved_tuple_count"] == 576
    for key in (
        "source_opportunity_constraint_row_count",
        "source_structural_inventory_artifact_count",
        "subject_materialization_count",
        "donor_bank_materialization_count",
        "counterfactual_twin_materialization_count",
        "planning_atom_materialization_count",
        "model_output_count",
        "cuda_planner_run_count",
        "empirical_outcome_count",
    ):
        assert type(manifest[key]) is int
        assert manifest[key] == 0
    assert manifest["zero_output_preflight_contract_frozen"] is True
    for key in (
        "source_opportunity_stage_completed",
        "source_structural_inventory_materialized",
        "source_grid_resolved",
        "external_publication_anchor_present",
        "current_source_execution_authorized",
        "tuple_feasibility_authorized",
        "model_output_authorized",
        "development_authorized",
        "qualification_authorized",
        "formal_authorized",
        "cuda_planner_authorized",
    ):
        assert manifest[key] is False

    firewall = certificate["zero_output_firewall"]
    assert set(firewall) == _EXPECTED_FIREWALL_KEYS
    assert len(firewall) == 29
    for key, value in firewall.items():
        if key.endswith("_count"):
            assert type(value) is int
            assert value == 0
        elif key in {
            "all_materialization_counts_refer_to_persisted_or_published_artifact_rows_not_ephemeral_in_memory_exact_derivation_objects",
            "ephemeral_in_memory_structural_derivation_objects_are_not_source_content_or_persistent_materialization",
        }:
            assert value is True
        else:
            assert type(value) is bool
            assert value is False
    stage = certificate["stage"]
    assert stage == {
        "future_structural_inventory_materialization_authorized": False,
        "future_structural_inventory_scope_defined": True,
        "source_opportunity_stage_completed": False,
        "source_structural_inventory_materialized": False,
        "zero_output_preflight_contract_frozen": True,
    }
    receipts = certificate["validation_receipts"]
    assert receipts["generic_decision_atoms_accepted_as_temporal_or_tuple_witness"] is False
    assert receipts["external_publication_anchor_present"] is False
    assert projection["mechanical_projection"]["generic_decision_planning_generator"][
        "temporal_joint_or_tuple_witness_derived"
    ] is False

    forbidden_materialized_keys = {
        "analysis_donor_pairs",
        "assignments",
        "atoms",
        "counterfactual_twins",
        "roots",
        "source_rows",
        "source_text_surfaces",
        "subjects",
    }
    assert _all_mapping_keys(projection).isdisjoint(forbidden_materialized_keys)


def test_root_surface_and_fact_orientation_arithmetic_is_independently_rebuilt() -> None:
    layout = _derived_root_layout()
    orientations = _derived_fact_orientations()
    expected_namespaces = (
        ("development", "analysis", 32),
        ("development", "donor", 32),
        ("qualification", "analysis", 64),
        ("qualification", "donor", 64),
        ("formal", "analysis", 8192),
        ("formal", "donor", 8192),
    )
    assert tuple((item.split_id, item.root_role, item.root_count) for item in layout.namespaces) == (
        expected_namespaces
    )
    assert layout.surface_capacity == 32768
    assert layout.affine_multiplier == 2085
    assert layout.affine_offset == 21504
    assert len(layout.surface_factor_axes_in_bit_order) == 15
    assert len(layout.roots) == 16576
    assert layout.analysis_root_count == 8288
    assert layout.donor_root_count == 8288
    assert tuple(root.global_slot for root in layout.roots) == tuple(range(16576))
    assert len({root.surface_code for root in layout.roots}) == 16576
    typed_registry = tuple(
        (item.value_zero, item.value_one) for item in layout.surface_factor_typed_value_registry
    )
    for root in layout.roots:
        expected_surface = (2085 * root.global_slot + 21504) % 32768
        expected_bits = tuple((expected_surface >> bit) & 1 for bit in range(15))
        expected_typed = tuple(typed_registry[bit][value] for bit, value in enumerate(expected_bits))
        assert root.surface_code == expected_surface
        assert root.factor_bits == expected_bits
        assert root.typed_blueprint_values == expected_typed

    assert len(layout.analysis_donor_pairs) == 8288
    assert len(layout.counterfactual_twin_mappings) == 8288
    assert tuple(item.root_count for item in layout.formal_candidate_prefixes) == tuple(range(192, 8193, 64))
    assert len(layout.formal_candidate_prefixes) == 126

    assignment_by_key = {
        (
            item.split_id,
            item.reversal_pair_ordinal,
            item.block_ordinal,
            item.within_block_ordinal,
        ): item
        for item in orientations.assignments
    }
    assert len(assignment_by_key) == 33152
    assert orientations.ranking_domain == (
        "volvence.relationship_p4_long_context_v4.source_fact_orientation.v1"
    )
    assert orientations.ranking_seed == 20260831
    assert orientations.block_size == 32
    assert orientations.orientation_count_per_value_per_block == 16
    assert orientations.ranking_excluded_fields == (
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
    for split_id, root_count in (
        ("development", 32),
        ("qualification", 64),
        ("formal", 8192),
    ):
        for pair_ordinal in range(4):
            for block_ordinal in range(root_count // 32):
                independently_ranked = []
                for within in range(32):
                    payload = (
                        f"{orientations.ranking_domain}|20260831|{split_id}|"
                        f"{pair_ordinal}|{block_ordinal}|{within}"
                    ).encode("ascii")
                    independently_ranked.append((hashlib.sha256(payload).digest(), within))
                independently_ranked.sort(key=lambda item: (item[0], item[1]))
                rank_by_within = {
                    within: (digest.hex(), rank)
                    for rank, (digest, within) in enumerate(independently_ranked)
                }
                for within in range(32):
                    assignment = assignment_by_key[(split_id, pair_ordinal, block_ordinal, within)]
                    digest, rank = rank_by_within[within]
                    expected_orientation = int(rank >= 16)
                    assert assignment.ranking_digest_sha256 == digest
                    assert assignment.rank_within_block == rank
                    assert assignment.orientation == expected_orientation
                    assert assignment.fact_values_by_decision_position == (
                        (0, 1) if expected_orientation == 0 else (1, 0)
                    )

    assert len(orientations.block_commitments) == 1036
    assert all(
        (item.orientation_zero_count, item.orientation_one_count) == (16, 16)
        for item in orientations.block_commitments
    )
    balances = {
        (item.formal_root_count, item.reversal_pair_ordinal, item.decision_position): (
            item.fact_zero_count,
            item.fact_one_count,
        )
        for item in orientations.formal_candidate_position_balances
    }
    assert len(balances) == 1008
    for root_count in range(192, 8193, 64):
        for pair_ordinal in range(4):
            for decision_position in range(2):
                assert balances[(root_count, pair_ordinal, decision_position)] == (
                    root_count // 2,
                    root_count // 2,
                )


def test_latin_rotation_and_final_32k_twin_are_exact_for_every_root_and_prefix() -> None:
    protocol = _strict_json(_PROTOCOL_PATH)
    layout = _derived_root_layout()
    orientations = _derived_fact_orientations()
    opportunity = protocol["opportunity_layout_and_utility_vectors"]
    assert opportunity["stratum_distance_latin_rotation_contract"] == {
        "stratum_index_for_analysis_root_and_pair": (
            "open_parenthesis_analysis_root_ordinal_plus_reversal_pair_index_close_parenthesis_modulo_4"
        ),
        "each_consecutive_four_root_block_assigns_each_semantic_stratum_once_to_each_distance_bin": True,
        "every_development_qualification_and_formal_candidate_prefix_is_exactly_balanced": True,
        "distance_bin_does_not_identify_semantic_stratum": True,
        "candidate_N_arm_candidate_cell_model_output_outcome_and_power_result_are_absent": True,
    }
    strata = opportunity["semantic_stratum_registry_in_canonical_order"]
    assert len(strata) == 4
    assert len({item["source_stratum_id"] for item in strata}) == 4
    assert len({item["challenge_stratum_id"] for item in strata}) == 4

    analysis_roots_by_split = {
        split_id: tuple(
            root
            for root in layout.roots
            if root.root_role == "analysis" and root.split_id == split_id
        )
        for split_id in ("development", "qualification", "formal")
    }
    for split_id, expected_count in (
        ("development", 32),
        ("qualification", 64),
        ("formal", 8192),
    ):
        roots = analysis_roots_by_split[split_id]
        assert len(roots) == expected_count
        assert tuple(root.namespace_ordinal for root in roots) == tuple(range(expected_count))
        for pair_ordinal in range(4):
            latin = tuple((root.namespace_ordinal + pair_ordinal) % 4 for root in roots)
            assert Counter(latin) == Counter({index: expected_count // 4 for index in range(4)})
            for start in range(0, expected_count, 4):
                assert set(latin[start : start + 4]) == {0, 1, 2, 3}
    for root_count in range(192, 8193, 64):
        for pair_ordinal in range(4):
            latin = tuple((ordinal + pair_ordinal) % 4 for ordinal in range(root_count))
            assert Counter(latin) == Counter({index: root_count // 4 for index in range(4)})

    orientation_by_root = {
        (item.split_id, item.analysis_root_ordinal): item
        for item in orientations.assignments
        if item.reversal_pair_ordinal == 3
    }
    assert len(orientation_by_root) == 8288
    assert len(layout.counterfactual_twin_mappings) == 8288
    for twin in layout.counterfactual_twin_mappings:
        assignment = orientation_by_root[(twin.split_id, twin.split_ordinal)]
        assert twin.decisive_decision_index == 7
        assert twin.reversal_pair_ordinal == 3
        assert twin.distance_bin_lower_bound_tokens == 32768
        assert twin.original_decisive_fact_value == assignment.fact_values_by_decision_position[1]
        assert twin.counterfactual_decisive_fact_value == 1 - twin.original_decisive_fact_value
        assert twin.utility_oracle_descendants_recomputed is True
        assert twin.all_other_exogenous_nodes_unchanged is True
        assert twin.independent_root is False


def test_action_slots_and_generic_decision_dgp_are_exactly_recomputed() -> None:
    evaluation = derivation.derive_source_evaluation_design(action_order=_EXPECTED_ACTIONS)
    assert evaluation.action_order == _EXPECTED_ACTIONS
    assert evaluation.invalid_generated_action_id == "INVALID_GENERATED_ACTION"
    assert evaluation.invalid_generated_action_in_registry is False
    assert tuple(
        (
            slot.slot_ordinal,
            slot.reversal_pair_ordinal,
            slot.distance_bin_tokens,
            slot.fact_value,
            slot.utility_vector,
        )
        for slot in evaluation.slots
    ) == tuple(
        (
            2 * pair + fact,
            pair,
            distance,
            fact,
            (1, -1, 0) if fact == 0 else (-1, 1, 0),
        )
        for pair, distance in enumerate((4096, 8192, 16384, 32768))
        for fact in (0, 1)
    )

    generator = _derived_planning_generator()
    assert generator.reference_success_probability == Fraction(11, 20)
    assert generator.comparator_success_probabilities == (Fraction(9, 20),) * 8
    assert len(generator.atoms) == 512
    states = {(atom.reference_success, atom.comparator_successes) for atom in generator.atoms}
    assert len(states) == 512
    for atom in generator.atoms:
        successes = (atom.reference_success, *atom.comparator_successes)
        expected_probability = (
            Fraction(11, 20) if atom.reference_success else Fraction(9, 20)
        )
        for comparator_success in atom.comparator_successes:
            expected_probability *= Fraction(9, 20) if comparator_success else Fraction(11, 20)
        expected_utilities = tuple(1 if success else -1 for success in successes)
        assert atom.probability == expected_probability
        assert atom.utility_vector == expected_utilities
        assert atom.contrast_vector == tuple(
            expected_utilities[0] - comparator for comparator in expected_utilities[1:]
        )
    assert sum((atom.probability for atom in generator.atoms), Fraction()) == 1

    masses = tuple(atom.probability for atom in generator.atoms)
    contrasts = tuple(
        tuple(atom.contrast_vector[index] for atom in generator.atoms) for index in range(8)
    )
    independently_computed_means = tuple(_weighted_mean(values, masses) for values in contrasts)
    independently_computed_covariance = tuple(
        tuple(_weighted_covariance(left, right, masses) for right in contrasts) for left in contrasts
    )
    reference_utility_mean = 2 * Fraction(11, 20) - 1
    comparator_utility_mean = 2 * Fraction(9, 20) - 1
    reference_utility_variance = 1 - reference_utility_mean**2
    comparator_utility_variance = 1 - comparator_utility_mean**2
    assert reference_utility_mean == Fraction(1, 10)
    assert comparator_utility_mean == Fraction(-1, 10)
    assert reference_utility_variance == comparator_utility_variance == Fraction(99, 100)
    assert independently_computed_means == (Fraction(1, 5),) * 8
    assert independently_computed_covariance == tuple(
        tuple(Fraction(99, 50) if row == column else Fraction(99, 100) for column in range(8))
        for row in range(8)
    )
    assert generator.contrast_means == independently_computed_means
    assert generator.contrast_covariance_matrix == independently_computed_covariance
    assert generator.contrast_correlation_matrix == tuple(
        tuple(Fraction(1) if row == column else Fraction(1, 2) for column in range(8))
        for row in range(8)
    )
    assert tuple((item.eigenvalue, item.multiplicity) for item in generator.correlation_eigenvalues) == (
        (Fraction(1, 2), 7),
        (Fraction(9, 2), 1),
    )


def test_protocol_helper_action_schema_and_action_owner_raw_drift_fail_loudly(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_drift = tmp_path / "protocol.json"
    protocol_drift.write_bytes(_PROTOCOL_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="protocol raw bytes drift"):
        load_relationship_p4_long_context_source_opportunity_preflight_protocol(protocol_drift)

    helper_drift = tmp_path / "helper.py"
    helper_drift.write_bytes(_HELPER_SOURCE.read_bytes() + b"\n")
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH", helper_drift)
        with pytest.raises(ValueError, match="input lineage.*drift"):
            load_relationship_p4_long_context_source_opportunity_preflight_protocol()

    schema_drift = tmp_path / "action-schema.json"
    schema_drift.write_bytes(_ACTION_SCHEMA_PATH.read_bytes() + b"\n")
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_RELATIONSHIP_ACTION_CHOICE_SCHEMA_PATH", schema_drift)
        with pytest.raises(ValueError, match="action schema raw bytes drift"):
            load_relationship_p4_long_context_source_opportunity_preflight_protocol()

    action_owner_drift = tmp_path / "relationship_action_contracts.py"
    action_owner_drift.write_bytes(_ACTION_OWNER_SOURCE.read_bytes() + b"\n")
    original_find_spec = owner.importlib.machinery.PathFinder.find_spec

    def find_spec(
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> object:
        if fullname == owner._RELATIONSHIP_ACTION_REGISTRY_MODULE:
            return importlib.util.spec_from_file_location(fullname, action_owner_drift)
        return original_find_spec(fullname, path, target)

    with monkeypatch.context() as patch:
        patch.setattr(
            owner.importlib.machinery.PathFinder,
            "find_spec",
            staticmethod(find_spec),
        )
        with pytest.raises(ValueError, match="action registry raw bytes drift"):
            load_relationship_p4_long_context_source_opportunity_preflight_protocol()


def test_full_hash_chain_resign_cannot_hide_semantic_or_key_tampering(tmp_path: pathlib.Path) -> None:
    semantic = _copy_source_artifact(tmp_path, "semantic")
    projection_id, certificate_id, artifact_id = _resign_source_artifact(
        semantic,
        lambda payload: payload["mechanical_projection"]["generic_decision_planning_generator"].__setitem__(
            "temporal_joint_or_tuple_witness_derived",
            True,
        ),
    )
    _assert_internal_hash_chain(
        semantic,
        projection_id=projection_id,
        certificate_id=certificate_id,
        artifact_id=artifact_id,
    )
    with pytest.raises(ValueError, match="temporal_joint_or_tuple_witness_derived value drift"):
        _validate_source_fast(semantic)

    extra_key = _copy_source_artifact(tmp_path, "extra-key")
    _resign_source_artifact(extra_key, lambda payload: payload.__setitem__("unexpected", False))
    with pytest.raises(ValueError, match="keys drift.*extra=.*unexpected"):
        _validate_source_fast(extra_key)

    missing_key = _copy_source_artifact(tmp_path, "missing-key")
    _resign_source_artifact(missing_key, lambda payload: payload.pop("claim_boundary"))
    with pytest.raises(ValueError, match="keys drift.*missing=.*claim_boundary"):
        _validate_source_fast(missing_key)


def test_artifact_rejects_extra_missing_noncanonical_bom_and_duplicate_json(
    tmp_path: pathlib.Path,
) -> None:
    extra_file = _copy_source_artifact(tmp_path, "extra-file")
    (extra_file / "unexpected.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="file set drift"):
        _validate_source_fast(extra_file)

    missing_file = _copy_source_artifact(tmp_path, "missing-file")
    (missing_file / _CERTIFICATE_FILE).unlink()
    with pytest.raises(ValueError, match="file set drift"):
        _validate_source_fast(missing_file)

    noncanonical = _copy_source_artifact(tmp_path, "noncanonical")
    projection_path = noncanonical / _PROJECTION_FILE
    projection_path.write_bytes(
        (json.dumps(_strict_json(projection_path), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        _validate_source_fast(noncanonical)

    bom = _copy_source_artifact(tmp_path, "bom")
    projection_path = bom / _PROJECTION_FILE
    projection_path.write_bytes(b"\xef\xbb\xbf" + projection_path.read_bytes())
    with pytest.raises(ValueError, match="must not carry a UTF-8 BOM"):
        _validate_source_fast(bom)

    duplicate = _copy_source_artifact(tmp_path, "duplicate")
    projection_path = duplicate / _PROJECTION_FILE
    original = projection_path.read_bytes()
    assert original.startswith(b"{")
    projection_path.write_bytes(b'{"schema_version":"duplicate",' + original[1:])
    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        _validate_source_fast(duplicate)


def test_artifact_rejects_hardlinks_symlinks_and_reparse_roots_when_supported(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardlinked = _copy_source_artifact(tmp_path, "hardlinked")
    independent = tmp_path / "independent-projection.json"
    shutil.copy2(hardlinked / _PROJECTION_FILE, independent)
    (hardlinked / _PROJECTION_FILE).unlink()
    try:
        os.link(independent, hardlinked / _PROJECTION_FILE)
    except OSError as exc:
        pytest.skip(f"hardlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="one hard link"):
        _validate_source_fast(hardlinked)

    symlinked = _copy_source_artifact(tmp_path, "symlinked")
    linked_projection = symlinked / _PROJECTION_FILE
    linked_projection.unlink()
    try:
        linked_projection.symlink_to(_SOURCE_PREFLIGHT_ARTIFACT / _PROJECTION_FILE)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match="regular files"):
            _validate_source_fast(symlinked)

    cached_inputs = _source_inputs()
    artifact_alias = tmp_path / "artifact-alias"
    _create_directory_alias_or_skip(artifact_alias, _SOURCE_PREFLIGHT_ARTIFACT)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_validated_v4_source_opportunity_inputs", lambda **_kwargs: cached_inputs)
            with pytest.raises(ValueError, match="symlink|reparse point"):
                validate_relationship_p4_long_context_source_opportunity_preflight(
                    output_dir=artifact_alias,
                    v4a_planning_dir=_V4A_PLANNING,
                    v3_preparation_dir=_V3_PREPARATION,
                    v2_admission_dir=_V2_ADMISSION,
                )
    finally:
        _remove_directory_alias(artifact_alias)

    output_target = tmp_path / "output-target"
    output_target.mkdir()
    output_alias = tmp_path / "output-alias"
    _create_directory_alias_or_skip(output_alias, output_target)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_validated_v4_source_opportunity_inputs", lambda **_kwargs: cached_inputs)
            with pytest.raises(ValueError, match="symlink|reparse point"):
                prepare_relationship_p4_long_context_source_opportunity_preflight(
                    output_dir=output_alias,
                    v4a_planning_dir=_V4A_PLANNING,
                    v3_preparation_dir=_V3_PREPARATION,
                    v2_admission_dir=_V2_ADMISSION,
                )
    finally:
        _remove_directory_alias(output_alias)
    assert not tuple(output_target.iterdir())


def test_prepare_is_create_only_and_bytes_ignore_output_host_and_cuda_environment(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_inputs = _source_inputs()
    output_a = tmp_path / "preflight-a"
    output_b = tmp_path / "different-parent" / "preflight-b"
    environment_keys = (
        "COMPUTERNAME",
        "HOSTNAME",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
    )
    first_environment = (
        "host-alpha-preflight-sentinel",
        "host-alpha-preflight-sentinel",
        "cuda-visible-alpha-preflight-sentinel",
        "gpu-alpha-preflight-sentinel",
        "cuda-order-alpha-preflight-sentinel",
    )
    second_environment = (
        "host-beta-preflight-sentinel",
        "host-beta-preflight-sentinel",
        "cuda-visible-beta-preflight-sentinel",
        "gpu-beta-preflight-sentinel",
        "cuda-order-beta-preflight-sentinel",
    )
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_validated_v4_source_opportunity_inputs", lambda **_kwargs: cached_inputs)
        for key, value in zip(environment_keys, first_environment, strict=True):
            patch.setenv(key, value)
        result_a = prepare_relationship_p4_long_context_source_opportunity_preflight(
            output_dir=output_a,
            v4a_planning_dir=_V4A_PLANNING,
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir=_V2_ADMISSION,
        )
        for key, value in zip(environment_keys, second_environment, strict=True):
            patch.setenv(key, value)
        result_b = prepare_relationship_p4_long_context_source_opportunity_preflight(
            output_dir=output_b,
            v4a_planning_dir=_V4A_PLANNING,
            v3_preparation_dir=_V3_PREPARATION,
            v2_admission_dir=_V2_ADMISSION,
        )

        bytes_a = {name: (output_a / name).read_bytes() for name in _artifact_filenames()}
        bytes_b = {name: (output_b / name).read_bytes() for name in _artifact_filenames()}
        published_bytes = {
            name: (_SOURCE_PREFLIGHT_ARTIFACT / name).read_bytes() for name in _artifact_filenames()
        }
        assert bytes_a == bytes_b == published_bytes
        assert result_a.artifact_id == result_b.artifact_id == _EXPECTED_ARTIFACT_ID
        assert result_a.output_dir != result_b.output_dir
        combined = b"".join(bytes_a.values())
        for sentinel in (*first_environment, *second_environment):
            assert sentinel.encode("utf-8") not in combined
        assert str(output_a).encode("utf-8") not in combined
        assert str(output_b).encode("utf-8") not in combined

        before = dict(bytes_a)
        with pytest.raises(FileExistsError, match="output already exists"):
            prepare_relationship_p4_long_context_source_opportunity_preflight(
                output_dir=output_a,
                v4a_planning_dir=_V4A_PLANNING,
                v3_preparation_dir=_V3_PREPARATION,
                v2_admission_dir=_V2_ADMISSION,
            )
        assert {name: (output_a / name).read_bytes() for name in _artifact_filenames()} == before
        assert not tuple(tmp_path.glob(".preflight-a.tmp-*"))


def test_cli_and_ast_expose_no_source_execution_or_heavy_dependency_surface(
    tmp_path: pathlib.Path,
) -> None:
    cli_source = _CLI_SOURCE.read_text(encoding="utf-8")
    cli_tree = ast.parse(cli_source)
    option_strings = {
        argument.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
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
    assert option_strings == {
        "--output-dir",
        "--v4a-planning-dir",
        "--v3-preparation-dir",
        "--v2-admission-dir",
    }
    assert loop_commands == {"prepare", "validate-existing"}
    assert option_strings.isdisjoint(
        {
            "--source",
            "--source-dir",
            "--source-path",
            "--seed",
            "--n",
            "--sample-size",
            "--model",
            "--cuda",
            "--device",
            "--force",
            "--override",
            "--protocol-path",
            "--host",
        }
    )

    forbidden_import_roots = {
        "accelerate",
        "asyncio",
        "concurrent",
        "cupy",
        "cuda",
        "httpx",
        "multiprocessing",
        "numpy",
        "onnxruntime",
        "requests",
        "socket",
        "subprocess",
        "threading",
        "torch",
        "transformers",
    }
    forbidden_execution_calls = {
        "Popen",
        "call",
        "check_call",
        "check_output",
        "execv",
        "execve",
        "popen",
        "run",
        "spawnl",
        "spawnle",
        "spawnv",
        "spawnve",
        "startfile",
        "system",
    }
    for source_path in (_OWNER_SOURCE, _HELPER_SOURCE, _CLI_SOURCE):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert _import_roots(tree).isdisjoint(forbidden_import_roots)
        assert not {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }.intersection(forbidden_execution_calls)

    helper_tree = ast.parse(_HELPER_SOURCE.read_text(encoding="utf-8"))
    assert _import_roots(helper_tree) <= {
        "__future__",
        "collections",
        "dataclasses",
        "fractions",
        "hashlib",
        "json",
        "math",
    }
    forbidden_helper_io = {
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
    assert not {
        node.attr for node in ast.walk(helper_tree) if isinstance(node, ast.Attribute)
    }.intersection(forbidden_helper_io)

    prepared = tmp_path / "cli-prepared"
    common_arguments = (
        "--output-dir",
        str(prepared),
        "--v4a-planning-dir",
        str(_V4A_PLANNING),
        "--v3-preparation-dir",
        str(_V3_PREPARATION),
        "--v2-admission-dir",
        str(_V2_ADMISSION),
    )
    assert _run_cli_import_closure(("--help",)) == []
    assert _run_cli_import_closure(("prepare", *common_arguments)) == []
    assert {item.name for item in prepared.iterdir()} == set(_artifact_filenames())
    assert _run_cli_import_closure(("validate-existing", *common_arguments)) == []


@lru_cache(maxsize=1)
def _derived_root_layout() -> derivation.SourceRootSurfaceDerivation:
    return derivation.derive_source_root_surface_layout()


@lru_cache(maxsize=1)
def _derived_fact_orientations() -> derivation.RootFactOrientationDerivation:
    return derivation.derive_root_fact_orientation_inventory(root_layout=_derived_root_layout())


@lru_cache(maxsize=1)
def _derived_planning_generator() -> derivation.SyntheticPlanningGeneratorDerivation:
    return derivation.derive_exact_synthetic_planning_generator()


@lru_cache(maxsize=1)
def _source_inputs() -> tuple[Any, Any, Any, Any]:
    return owner._validated_v4_source_opportunity_inputs(
        v4a_planning_dir=_V4A_PLANNING,
        v3_preparation_dir=_V3_PREPARATION,
        v2_admission_dir=_V2_ADMISSION,
        protocol_path=None,
    )


def _validate_source_fast(path: pathlib.Path) -> object:
    protocol, raw, derived, upstream = _source_inputs()
    return owner._validate_v4_source_opportunity_preflight_root(
        path,
        protocol=protocol,
        raw=raw,
        derived=derived,
        upstream=upstream,
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


def _all_mapping_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        result.update(value)
        for child in value.values():
            result.update(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_all_mapping_keys(child))
    return result


def _weighted_mean(values: tuple[int, ...], masses: tuple[Fraction, ...]) -> Fraction:
    return sum(
        (mass * value for mass, value in zip(masses, values, strict=True)),
        Fraction(),
    )


def _weighted_covariance(
    left: tuple[int, ...],
    right: tuple[int, ...],
    masses: tuple[Fraction, ...],
) -> Fraction:
    left_mean = _weighted_mean(left, masses)
    right_mean = _weighted_mean(right, masses)
    return sum(
        (
            mass * (left_value - left_mean) * (right_value - right_mean)
            for mass, left_value, right_value in zip(masses, left, right, strict=True)
        ),
        Fraction(),
    )


def _copy_source_artifact(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    destination = tmp_path / name
    shutil.copytree(_SOURCE_PREFLIGHT_ARTIFACT, destination)
    return destination


def _resign_source_artifact(
    root: pathlib.Path,
    projection_mutator: Callable[[dict[str, Any]], object],
) -> tuple[str, str, str]:
    projection_path = root / _PROJECTION_FILE
    projection = _strict_json(projection_path)
    projection.pop("contract_projection_id")
    projection_mutator(projection)
    projection_id = hashlib.sha256(_canonical_bytes(projection)).hexdigest()
    projection["contract_projection_id"] = projection_id
    projection_bytes = _canonical_bytes(projection)
    projection_path.write_bytes(projection_bytes)

    certificate_path = root / _CERTIFICATE_FILE
    certificate = _strict_json(certificate_path)
    certificate.pop("certificate_id")
    identity = certificate["identity"]
    assert isinstance(identity, dict)
    identity["contract_projection_id"] = projection_id
    identity["contract_projection_raw_sha256"] = hashlib.sha256(projection_bytes).hexdigest()
    identity["contract_projection_byte_count"] = len(projection_bytes)
    certificate_id = hashlib.sha256(_canonical_bytes(certificate)).hexdigest()
    certificate["certificate_id"] = certificate_id
    certificate_bytes = _canonical_bytes(certificate)
    certificate_path.write_bytes(certificate_bytes)

    manifest_path = root / _MANIFEST_FILE
    manifest = _strict_json(manifest_path)
    manifest.pop("artifact_id")
    manifest["contract_projection_id"] = projection_id
    manifest["certificate_id"] = certificate_id
    manifest["files"] = [
        {
            "path": _PROJECTION_FILE,
            "byte_count": len(projection_bytes),
            "sha256": hashlib.sha256(projection_bytes).hexdigest(),
        },
        {
            "path": _CERTIFICATE_FILE,
            "byte_count": len(certificate_bytes),
            "sha256": hashlib.sha256(certificate_bytes).hexdigest(),
        },
    ]
    artifact_id = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest["artifact_id"] = artifact_id
    manifest_path.write_bytes(_canonical_bytes(manifest))
    return projection_id, certificate_id, artifact_id


def _assert_internal_hash_chain(
    root: pathlib.Path,
    *,
    projection_id: str,
    certificate_id: str,
    artifact_id: str,
) -> None:
    projection = _strict_json(root / _PROJECTION_FILE)
    projection_core = dict(projection)
    assert projection_core.pop("contract_projection_id") == projection_id
    assert hashlib.sha256(_canonical_bytes(projection_core)).hexdigest() == projection_id
    certificate = _strict_json(root / _CERTIFICATE_FILE)
    certificate_core = dict(certificate)
    assert certificate_core.pop("certificate_id") == certificate_id
    assert hashlib.sha256(_canonical_bytes(certificate_core)).hexdigest() == certificate_id
    manifest = _strict_json(root / _MANIFEST_FILE)
    manifest_core = dict(manifest)
    assert manifest_core.pop("artifact_id") == artifact_id
    assert hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest() == artifact_id


def _artifact_filenames() -> tuple[str, str, str]:
    return _PROJECTION_FILE, _CERTIFICATE_FILE, _MANIFEST_FILE


def _import_roots(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            result.add(node.module.split(".", 1)[0])
    return result


def _run_cli_import_closure(arguments: tuple[str, ...]) -> list[str]:
    probe = """
import importlib.util
import json
import pathlib
import sys

cli_path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_source_preflight_cli_probe", cli_path)
if spec is None or spec.loader is None:
    raise RuntimeError("CLI probe could not create an import spec")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    return_code = module.main(sys.argv[2:])
except SystemExit as exc:
    return_code = exc.code
if return_code != 0:
    raise RuntimeError(f"CLI returned {return_code!r}")
forbidden_roots = (
    "lifeform_core",
    "lifeform_domain_emogpt",
    "volvence_zero.substrate",
    "torch",
    "transformers",
    "vllm",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
)
print("IMPORT_CLOSURE=" + json.dumps(loaded, separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(_CLI_SOURCE), *arguments],
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    marker = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("IMPORT_CLOSURE=")),
        None,
    )
    assert marker is not None, completed.stdout
    loaded = json.loads(marker.removeprefix("IMPORT_CLOSURE="))
    assert isinstance(loaded, list)
    assert all(isinstance(item, str) for item in loaded)
    return loaded


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
