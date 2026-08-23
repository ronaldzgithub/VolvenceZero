"""Zero-output P4.7 independent long-context causal-campaign preregistration.

This owner freezes the scientific design and may publish an exact, zero-output
necessary-condition power failure.  It deliberately has no model, CUDA,
subject-generation, session-worker, resume, or formal-run entry point.  Those
capabilities require a later content-addressed execution envelope that may add
execution lineage but may not change a protocol's frozen scientific contract.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import importlib.machinery
import json
import math
import os
import pathlib
import shutil
import stat
import sys
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, Mapping
import uuid

if TYPE_CHECKING:
    from lifeform_evolution.relationship_lab_p4_long_context_source_opportunity_derivation import (
        RootFactOrientationDerivation,
        SourceEvaluationDesign,
        SourceRootSurfaceDerivation,
        SyntheticPlanningGeneratorDerivation,
    )
    from lifeform_evolution.relationship_lab_p4_long_context_v4_planning_derivation import (
        V4CandidateScheduleBlock,
        V4NecessaryScreenResult,
    )


P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-causal-campaign-scientific-prereg.v1"
)
P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V2 = (
    "relationship-p4-independent-long-context-causal-campaign-scientific-prereg.v2"
)
P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V3 = (
    "relationship-p4-independent-long-context-causal-campaign-scientific-prereg.v3"
)
P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION = P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V3
P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-causal-campaign-preparation.v1"
)
P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V2 = (
    "relationship-p4-independent-long-context-causal-campaign-preparation.v2"
)
P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V3 = (
    "relationship-p4-independent-long-context-causal-campaign-preparation.v3"
)
P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION = P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V3
P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V1 = "relationship-p4-independent-long-context-causal-campaign-manifest.v1"
P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V2 = "relationship-p4-independent-long-context-causal-campaign-manifest.v2"
P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V3 = "relationship-p4-independent-long-context-causal-campaign-manifest.v3"
P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION = P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V3
P4_LONG_CONTEXT_PROTOCOL_ID_V1 = "5387516a803940a738e13bb47acc8a40b837c3f033797e09dbfaa23c6cda6d2e"
P4_LONG_CONTEXT_PROTOCOL_ID_V2 = "666d2e8546cd4b4cf55ece06354310e10b4dc07298241b94ef9593e4b5f63baf"
P4_LONG_CONTEXT_PROTOCOL_ID_V3 = "9f352778e128a9573790762222a05225740bdaeb732800dec0eec124116a282d"
P4_LONG_CONTEXT_PROTOCOL_ID = P4_LONG_CONTEXT_PROTOCOL_ID_V3
P4_LONG_CONTEXT_PREPARATION_STATUS_V1 = "scientific_prereg_frozen_execution_envelope_absent"
P4_LONG_CONTEXT_PREPARATION_STATUS_V2 = "scientific_prereg_v2_frozen_execution_envelope_absent"
P4_LONG_CONTEXT_PREPARATION_STATUS_V3 = "scientific_prereg_v3_frozen_execution_envelope_absent"
P4_LONG_CONTEXT_PREPARATION_STATUS = P4_LONG_CONTEXT_PREPARATION_STATUS_V3
P4_LONG_CONTEXT_POWER_FAILURE_SCHEMA_VERSION = (
    "relationship-p4-independent-long-context-causal-campaign-necessary-power-failure.v1"
)
P4_LONG_CONTEXT_POWER_FAILURE_MANIFEST_SCHEMA_VERSION = (
    "relationship-p4-independent-long-context-causal-campaign-necessary-power-failure-manifest.v1"
)
P4_LONG_CONTEXT_POWER_FAILURE_STATUS = "prior_power_gate_failed_exact_necessary_condition_before_development_output"
P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID_V1: str | None = (
    "fad6c105b7c64a6b4ab89bf6e933ecdf4c8f1b1170679d918c2dd77c27809518"
)
P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID = P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID_V1
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-power-bound-fail-protocol.v1"
)
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1 = "735b20a137b03176cf889c0cbe116e29f973c18d4cef4bf38cd42df288dff3fa"
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1 = "1bb8d21ce3a0dca332324d2e35e3bc2c63ec77fc9bd3917b35d018ebd85559f6"
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V2 = (
    "relationship-p4-independent-long-context-power-admission-protocol.v2"
)
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2 = "67d294faf9209c9d05334f4c0e87371676c9821b7c12e603f3e289f33f566bc9"
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2 = "130f766787ec0b02bd5857344e58b371d996aa51fabe421cbfbde05347fd0e04"
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION = P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V2
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID = P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2
P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256 = P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2
P4_LONG_CONTEXT_POWER_ADMISSION_SCHEMA_VERSION_V2 = (
    "relationship-p4-independent-long-context-power-admission-certificate.v2"
)
P4_LONG_CONTEXT_POWER_ADMISSION_MANIFEST_SCHEMA_VERSION_V2 = (
    "relationship-p4-independent-long-context-power-admission-manifest.v2"
)
P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2 = "power_contract_under_specified_no_development_authorization"
P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2: str | None = (
    "9883e10784a06260a220a6fdbf72141b1300c21e97faee6e84a401c40a144ee9"
)
P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-zero-output-planning-protocol.v1"
)
P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1 = "63e007b7d43bb152e5891162d6567c4edd4396af99cf1c5525c28d0be4c08753"
P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1 = "d06b07101624b3996bd712c98d3c633b7b00af7a878912817b5149a199c00e0a"
P4_LONG_CONTEXT_V4_PLANNING_FREEZE_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-zero-output-planning-freeze.v1"
)
P4_LONG_CONTEXT_V4_PLANNING_MANIFEST_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-zero-output-planning-manifest.v1"
)
P4_LONG_CONTEXT_V4_CANDIDATE_SCHEDULE_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-development-candidate-schedule.v1"
)
P4_LONG_CONTEXT_V4_SENTINEL_SCREEN_TABLE_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-sentinel-necessary-point-screen-table.v1"
)
P4_LONG_CONTEXT_V4_PLANNING_STATUS_V1 = "v4_planning_contract_frozen_full_joint_planner_pending"
P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1: str | None = (
    "082454002260db90b7236a1104311a5d92cc3959171bb3190e7a30f8387e56c1"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-source-opportunity-preflight-protocol.v1"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1 = (
    "47bcf6561be1ace0698cc0f96e2e7e35701f46d15baac9eb87ad1d662576494a"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1 = (
    "9d4d3ab5cb683d8ff5827e5047e5b176800fe5c4e86ad6a07217b7a2040c40b0"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_CONTRACT_SCHEMA_VERSION_V1 = (
    "relationship-p4-long-context-source-opportunity-contract-projection.v1"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_SCHEMA_VERSION_V1 = (
    "relationship-p4-long-context-source-opportunity-preflight-certificate.v1"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_MANIFEST_SCHEMA_VERSION_V1 = (
    "relationship-p4-long-context-source-opportunity-preflight-manifest.v1"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_STATUS_V1 = (
    "source_opportunity_preflight_contract_frozen_zero_output_inventory_materializer_not_run"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1: str | None = (
    "8a36d2de9077bb5550db8018338eded27b6ce30d77eea17739ffe35b73e00a99"
)
P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1: str | None = (
    "64d879c4f41ca873f8e40f0344234771343f6efee229b668914b61d31c96c95a"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SCHEMA_VERSION_V1 = (
    "relationship-p4-independent-long-context-v4-external-publication-anchor-request-protocol.v1"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1 = (
    "dedfc7ff42f1be0030cdfbe64fd6b1d6dc868adf9db6a9f1150883a9a96a4bee"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_RAW_SHA256_V1 = (
    "38ce85d479c4359c252de8e5293ca1c15d886c5e1757610435aad136feeca8c6"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_SCHEMA_VERSION_V1 = (
    "relationship-p4-long-context-external-publication-anchor-request.v1"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_MANIFEST_SCHEMA_VERSION_V1 = (
    "relationship-p4-long-context-external-publication-anchor-request-manifest.v1"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_STATUS_V1 = (
    "external_publication_anchor_request_frozen_publication_not_observed_no_authority"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1 = (
    "7897e3285299eac33385f69fb560a7d68e9f3316fdaf200f27cfa9bbfda489d1"
)
P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1 = (
    "5496fa80bba07c6b2234e0e2ca9293111d7ed6edf0a676ee4f561a7893c22900"
)

_MODULE_FILE = pathlib.Path(os.path.abspath(__file__))
_MODULE_DIRECTORY = _MODULE_FILE.parent
_PROTOCOL_DIRECTORY = _MODULE_DIRECTORY / "protocols"
_REPOSITORY_ROOT = _MODULE_FILE.parents[4]
_V1_PROTOCOL_PATH = _PROTOCOL_DIRECTORY / "relationship_p4_independent_long_context_causal_campaign_v1.json"
_V2_PROTOCOL_PATH = _PROTOCOL_DIRECTORY / "relationship_p4_independent_long_context_causal_campaign_v2.json"
_V3_PROTOCOL_PATH = _PROTOCOL_DIRECTORY / "relationship_p4_independent_long_context_causal_campaign_v3.json"
_POWER_BOUND_PROTOCOL_PATH_V1 = _PROTOCOL_DIRECTORY / "relationship_p4_long_context_power_bound_fail_v1.json"
_POWER_BOUND_PROTOCOL_PATH_V2 = _PROTOCOL_DIRECTORY / "relationship_p4_long_context_power_admission_v2.json"
_V4_PLANNING_PROTOCOL_PATH_V1 = _PROTOCOL_DIRECTORY / "relationship_p4_long_context_v4_planning_contract_v1.json"
_V4_PLANNING_DERIVATION_HELPER_PATH = _MODULE_DIRECTORY / "relationship_lab_p4_long_context_v4_planning_derivation.py"
_V4_PLANNING_DERIVATION_HELPER_RAW_SHA256_V1 = "bf38e7ab89c56bdae8844f533cac077443d157a793c698adbb11a9591e32a0ef"
_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_PATH_V1 = (
    _PROTOCOL_DIRECTORY / "relationship_p4_long_context_v4_source_opportunity_preflight_v1.json"
)
_V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH = (
    _MODULE_DIRECTORY / "relationship_lab_p4_long_context_source_opportunity_derivation.py"
)
_V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_RAW_SHA256_V1 = (
    "72efc093b815c2ca07872f6cb6a78f53a4d4d5ada5975222b36cf90c640746f8"
)
_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_PATH_V1 = (
    _PROTOCOL_DIRECTORY / "relationship_p4_long_context_v4_external_publication_anchor_v1.json"
)
_RELATIONSHIP_ACTION_CHOICE_SCHEMA_PATH = (
    _PROTOCOL_DIRECTORY.parent / "schemas" / "relationship_action_choice.schema.json"
)
_RELATIONSHIP_ACTION_CHOICE_SCHEMA_RAW_SHA256_V1 = "764309ff7b1d4aa6e9001a73a8c72407a1fabfd1e9d5c89e7cdf37360054efea"
_RELATIONSHIP_ACTION_REGISTRY_MODULE = "lifeform_domain_emogpt.relationship_action_contracts"
_RELATIONSHIP_ACTION_REGISTRY_MODULE_RAW_SHA256_V1 = "dc1907cc67d76536b88894f5e06c907ec4651a76acab0fe28531ffe14db2b526"
_RELATIONSHIP_ACTION_IDS_V1 = (
    "stay_present_without_probe",
    "respect_space_with_return_option",
    "neutral_noop",
)
_DEFAULT_PROTOCOL_PATH = _V3_PROTOCOL_PATH
_PREPARATION_FILE = "scientific_prereg_preparation.json"
_MANIFEST_FILE = "manifest.json"
_POWER_FAILURE_CERTIFICATE_FILE = "necessary_power_failure_certificate.json"
_POWER_ADMISSION_CERTIFICATE_FILE = "power_admission_certificate.json"
_V4_PLANNING_FREEZE_FILE = "v4_zero_output_plan.json"
_V4_CANDIDATE_SCHEDULE_FILE = "development_candidate_cell_schedule.json"
_V4_SENTINEL_SCREEN_TABLE_FILE = "sentinel_necessary_point_screens.json"
_V4_SOURCE_OPPORTUNITY_CONTRACT_FILE = "source_opportunity_contract_projection.json"
_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE = "source_opportunity_preflight_certificate.json"
_V4_EXTERNAL_ANCHOR_REQUEST_FILE = "external_publication_anchor_request.json"
_HEX_DIGITS = frozenset("0123456789abcdef")

_POWER_BOUND_PROTOCOL_SECTION_SHA256_V1 = MappingProxyType(
    {
        "schema_version": "e27791c053d0ad0a4c36ccff8345640aa01b3bb6cd167c911f58833b7195535a",
        "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
        "frozen_at_utc": "38caef62bf3ca7c330e3779501eda0861046b21dc3597f918771c24c2c866790",
        "owner": "d9776b5cd834ea0931bbd661f1e3dfbaaf11a9eb20f50bac8af59f23960e0cb1",
        "input_lineage": "f217211c77a18db1221404540dd485ac8863712087ee71cedb0c50f7c4ba2b60",
        "question": "3b68cb9dafc6cff47a55d46401659452c0eb8b949b2f30eb49c7ae3b559eb2e0",
        "mandatory_scenario": "5a6a6c9922b00913f5ef7e4090c29bfc5d3f17cb1860245de31eb008cca14ad5",
        "feasibility_witness": "69f58fb31b7886ec4e25832b76280117417d02894cb25b6516b8d1de7b9efdd2",
        "exact_calculation": "26da5d79d5da1bf370dc47d98fc2bd0d4148ef877b810be8d5d1898107a6b9ed",
        "certificate_contract": "b777c69edfafc4c5f6a2a5dac16b70abf6f02b3a0cd6bad731ce7e34df5b1cd1",
        "zero_output_firewall": "7bf53f249b34c1f86aede84dbc27e01a238770ed395feff37aa93100100dfb89",
        "terminal": "811a7c2775eb190211a7de9408a50a2debaa5094557096f93e75d637993f4eb5",
        "claim_boundary": "34cc47b96181afe27285b7ecd8958062dcc986d9bbc516ef0f612925712dac9c",
    }
)

_POWER_BOUND_PROTOCOL_SECTION_SHA256_V2 = MappingProxyType(
    {
        "schema_version": "048588aca0138b549faaa4b50a68aab3dd0c2ddfeb2ba10eedf4b0717e21b94d",
        "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
        "frozen_at_utc": "93232a8b1d579f7081485ab4175995cb30ca786bf159823c60256b9753e6343f",
        "owner": "fe124926273a4004e6496460a6a47c8e49a998a4fe4ed9f72a3c95c97f109979",
        "supersession": "895af77ce7dc21e8feb5411a9b0a96ca37aa720507c070e8d5ad37be41f47c10",
        "input_lineage": "f217211c77a18db1221404540dd485ac8863712087ee71cedb0c50f7c4ba2b60",
        "question": "d92f3e5cf11137a8fdd6f0abbfdf7677004fedc8ce50aa34ae82e88623b065e6",
        "frozen_v3_grid_facts": "fbf3832b99daa3d4b34541595261535665cf53c9514598f0572695dd3c3394ad",
        "historical_v1_witness_facts": "f2ef111c74d9f3b9c267ebb6567e01f0a9a4452af151a2dc2864278c6e6070bb",
        "primitive_joint_witness": "202b8bfc6300465e9c11658e1c9e25ece7321441916bc9192a4aa90ef65ba614",
        "derived_witness_properties": "6137a2599d2baee048f5ef25f180f42c634c607a64ccf87efd5ca9d44534bb12",
        "grid_membership": "c410bfdd9fbd90f92d829b95c5486226a251def9bbf0bb2aec9dad35b5c6513a",
        "ambiguity_witness": "31359504a29a1e163da500e68deb0169b770335349f221faff16baa9b0ff8a11",
        "posthoc_semantics": "fae525dc96a65728f5b2865abb7f956d6ca3fe5ec02c5902f27a8cde61be04b4",
        "admission_logic": "1e6a698ea99bc6d8a4f4b9a5ff155d0cedc436bca951a213c60ebf7116e1fe7f",
        "conditional_numeric_bound": "07438498d5c6522e8e1edfbd99c81dc955ccfd6b35faaccf9107e34c0a5e347c",
        "certificate_contract": "20974dd99f9600e0ef630a21026a00f9229016f21c078fa94b7bc61c3bda2e24",
        "zero_output_firewall": "81f47477aa971826ef904776399f9b7c7ae8bed0a1672de45ea9a0d532fe9d60",
        "v4_requirements": "51de4aad20e81f98ea10d50819b076c18df124ccaa36d5a62862da7d579c19f8",
        "terminal": "20052bb7df6e47c535cf3aa0fe1818160d2f1867a3a5e21d3c3fff51a0de835d",
        "claim_boundary": "48bf5ed942e23802d584e25c936e50b33477a9ff93a37c18b9294e3d0a5951a4",
    }
)

_V4_PLANNING_PROTOCOL_SECTION_SHA256_V1 = MappingProxyType(
    {
        "schema_version": "f6ffd60e649e51d21590d08dc8bc7908d23d09d612ddc7ef97e0979a11fbfebb",
        "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
        "frozen_at_utc": "0726b80c6725888bd2111266798b1bcec7518c9ebbf734c1cf38fb402a1d5a41",
        "owner": "ca6d8069b241946909560dede52aa50517fccd4ffccfc605450e9d225769aa4a",
        "supersession": "f2711b99db3d3ecde6906df7e1701c7b0a64b7499d1c87d23d4471058092ca66",
        "input_lineage": "3ce0dd328ce2d59d6bec5a70d33b48384485c2663f81f22936a868a58a361d99",
        "question": "257ed3cbcc97cd195b2f5c4f8b1249c9c8b46a58ae6cc1ebecd40f1867139991",
        "artifact_sequence": "0efd999a2d0234a2635e383463cc235b6f022c0ef6894389345bab135c90ef78",
        "scientific_units": "4594197ef0f9dfe5b310dd8f6bbd7bcde318a7680dfcfc3db825f340f9720609",
        "mandatory_global_joint_sentinels": ("9ed04320e18c7f68a0a190c60b0ce4b48aab1dca2ba7e09c922f588a422b6cac"),
        "source_conditioned_cartesian_grid": ("f742ae41b93266233c98913bff99cf87aedaf580e20a4584e6db8c4d96eb5e30"),
        "missingness_semantics": "3a9bfeb0c96705eb999d58fc69dba96b34b02bd3c40dccd882c571f0c6e4ae74",
        "generated_action_classification": ("c6bf1727a4c977658e0674694d6170f22895c9a337d7abc33fd18ae5356d72da"),
        "development_candidate_cells": ("8becf7ce5c1db43a22e934e10d2e92a226fa8779dc58840c08c08bcc2ac009eb"),
        "full_joint_power_planner": "690ef40435ba16eb501d21858b065feea8453337ce17052ded715e5f0f0bb6e9",
        "sample_size_freeze": "e98560c105e508b49cc1ae59950f4b83dfda23e5a49bc6427e16509f3f609bed",
        "source_preflight_contract": ("0d8081ef7fab569f2bc25ad9cd7e8b86e6f1f60c88fd2daa23ae55f3bfd6648f"),
        "zero_output_firewall": "767ddd72836c19a31306c49761cdba5635b21fee6955fc583355a930e0d91614",
        "terminal": "ebefb057ff825f18f5e4021a4418ec0fd2bb87eee5438c09b74616a973919104",
        "claim_boundary": "2f558535230c3bba14daa4e6b86bf946e9fed806dfd7b0cfbe41b89f6f3edb00",
    }
)

_V4_PLANNING_TOP_LEVEL_KEYS = frozenset(_V4_PLANNING_PROTOCOL_SECTION_SHA256_V1)

_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SECTION_SHA256_V1 = MappingProxyType(
    {
        "schema_version": "40ab3f6088314f92c51dabd9ac722d2f937dca99b6a2b501996f2cda681b1be3",
        "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
        "frozen_at_utc": "14517d0e57f1b809405da64bdd216bb41d4e573ef51bcf5cbe6cbe8bf1a0a8d2",
        "owner": "b96cdf36843f4a60eaedfc7b970070c04a090ec6447bbb4a4b9b81d1f5d2e1c9",
        "supersession": "d297e6696e8d2bcb1e89ad1996ed04290da3dfe5b77f6ab49d9f6f68323dc0cd",
        "input_lineage": "be67309a77fa7b95a5c48faa17122155ec1dde0cbe670a7aafd5d71fa12ff4ac",
        "question": "c7db1fd99f6dce5fe08e957b6a671af42ce03dcb4688c4598c921d8e0b8f67aa",
        "stage_boundary": "eacc25acc511e61073a95bc84edb06197c79231fbbd57e944f7975d327108e2d",
        "source_opportunity_unit": "985ef8f68d913013926bf7e2393cbe11a57e15bc84572ee54c38601a023f4508",
        "sampling_frame_contract": "9785ce913833e418ec9f551f9317bfe578bb35d00259f06769166573081fa91b",
        "root_independence_and_capacity": "e91dd9eebf5abb4de3c0c738e8bea5ddefd08f383a8253dbfbadf49c00dd17c1",
        "opportunity_layout_and_utility_vectors": "b5b8d81ef3d65d6bc665c3ead829d0371e4bcb33405482a8b95b00e21f446e59",
        "exact_source_planning_generator": "4c9d48515bb7bd3fd67c9662eb81d86b0bcc0152d3c81eb7430251061099a58c",
        "truth_twin_and_leakage_firewall": "db333eac8ccff571836dcea7d6d75360eb9bd36bc4ccc5db62b1c3840bb778a5",
        "future_materialization_envelope": "f28ba81515ccacb8a5b8856ffafb8b5fa2e92fd0e7a4b535734d40631b285b19",
        "zero_output_firewall": "912e45f629b7fe4c0807144c694bf6d631111118925144e8e13e477b11dfee01",
        "terminal": "504f5a3298962934c8c681d3f3cd1564d35214b3a6e28d5d965aef8699454998",
        "claim_boundary": "126fa82098e56c9618a57952f63dc6fb0556b5cd0ae429a7310cc9e0752d8192",
    }
)
_V4_SOURCE_OPPORTUNITY_PREFLIGHT_TOP_LEVEL_KEYS = frozenset(_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SECTION_SHA256_V1)

_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SECTION_SHA256_V1 = MappingProxyType(
    {
        "schema_version": "5ce63abdc943432e02d932d8f38036d2c630982d8dcab493438ce530a038747f",
        "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
        "frozen_at_utc": "cf35b37fdcb58ba82055529a45fa9cadff04f15877d3bf810ab4f058eb137be1",
        "owner": "9d3c16023430115cb2ee14603d6af3209e6e7fc25912dfcb1e408b296b8f34a4",
        "supersession": "d356c00957a170fabb20a218302f5fa21b3e2299984d6ac48f49dcd2d9257046",
        "question": "3bf6fc3e45a1f723e647cc06a3e5e253d09b17819af4c536bb6f6059d1c6c0d6",
        "input_lineage": "ae1a18aeaf917fcd8a4c6dbb873f4bd06f3814f576e0a185309e3d50fb4c4920",
        "anchor_stage": "baa251cbcceb889089939838a3bd77a4bfd1a28eb2d6432e7130babad7759a76",
        "publication_subject_contract": "bfa95ccb8f5dd836269c7b4c84ac1727010d60c42c3a1228c073a87bcf7ec112",
        "publication_target_contract": "41798c349218be5e58e324e434f35f51379854503e12e00f654f9983ad0eab7d",
        "self_publication_binding": "06e28786b66320232f93d4634b2fee9bba6c5a6bf3fe13034ea83a54fd51769f",
        "future_receipt_requirements": "5b5ec66c404c20674abdd74dc4170ac87a71fa338e01bcf6b42ff70e7a02bd68",
        "authorization_firewall": "9740ae863e28a3028ec13e3d5c4e6989afd1c3abb1cda8b635e5fc9635fb3bb2",
        "zero_output_firewall": "a1874b50d34dfea446bddbad68c6887842ab4a96974d335ff02c6938283cf959",
        "terminal": "02137f0a4009feb3982d5aac2e378e679bcdae8e08a0deeed20031a00db7ce7c",
        "claim_boundary": "1bda30c5cd3efad160b41455ed6af1dd1fd570a0fe49ac365aac846623e5f883",
    }
)
_V4_EXTERNAL_ANCHOR_REQUEST_TOP_LEVEL_KEYS = frozenset(_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SECTION_SHA256_V1)

_TOP_LEVEL_KEYS_V1 = frozenset(
    {
        "schema_version",
        "protocol_id_contract",
        "frozen_at_utc",
        "owner",
        "question",
        "lineage",
        "cohort",
        "longitudinal_design",
        "baseline_admission",
        "integrated_arms",
        "arm_matrix",
        "axis_contrasts",
        "intervention_integrity",
        "shared_exposure_schedule",
        "causal_execution",
        "analysis",
        "execution_admission",
        "stopping_rules",
        "evidence_firewall",
        "claim_boundary",
    }
)
_TOP_LEVEL_KEYS_V2 = frozenset(set(_TOP_LEVEL_KEYS_V1) | {"supersession"})
_TOP_LEVEL_KEYS_V3 = _TOP_LEVEL_KEYS_V2
_ARM_MATRIX_V1 = (
    "volvence_closed_loop",
    "appendable_empty_prior",
    "appendable_swapped_same_stage_prior",
    "readable_label_permuted",
    "volvence_typed_noop_control",
    "steerable_strict_noop",
    "steerable_sensor_off_matched",
    "qwen_steelman_full_history",
    "qwen_steelman_selective_rag",
)
_ARM_MATRIX_V2 = (
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
_ARM_MATRIX_V3 = _ARM_MATRIX_V2
_INTEGRATED_ARMS_V1 = (
    "qwen_steelman_full_history",
    "qwen_steelman_selective_rag",
    "volvence_closed_loop",
    "volvence_typed_noop_control",
)
_INTEGRATED_ARMS_V2 = (
    "qwen_steelman_full_history",
    "qwen_steelman_selective_rag",
    "volvence_closed_loop",
)
_INTEGRATED_ARMS_V3 = _INTEGRATED_ARMS_V2
_AXIS_CONTROLS_V1 = {
    "appendable": (
        "appendable_empty_prior",
        "appendable_swapped_same_stage_prior",
    ),
    "readable": ("readable_label_permuted",),
    "learnable": ("volvence_typed_noop_control",),
    "steerable": ("steerable_strict_noop",),
}
_AXIS_CONTROLS_V2 = {
    "appendable": (
        "appendable_empty_prior",
        "appendable_swapped_same_stage_prior",
    ),
    "readable": ("readable_label_permuted",),
    "learnable": ("learnable_credit_withheld",),
    "steerable": ("steerable_strict_noop",),
}
_AXIS_CONTROLS_V3 = _AXIS_CONTROLS_V2
_ALLOWED_INTERVENTION_POINTERS_V1 = {
    "appendable_empty_prior": ("/hydration/prior_state_selector",),
    "appendable_swapped_same_stage_prior": ("/hydration/prior_state_selector",),
    "readable_label_permuted": ("/components/condition_reader/artifact_id",),
    "volvence_typed_noop_control": ("/learning/apply_exact_pe_credit",),
    "steerable_strict_noop": ("/steering/executor_action",),
    "steerable_sensor_off_matched": ("/steering/executor_artifact_id",),
}
_ALLOWED_INTERVENTION_POINTERS_V2 = {
    "appendable_empty_prior": ("/hydration/prior_state_selector",),
    "appendable_swapped_same_stage_prior": ("/hydration/prior_state_selector",),
    "readable_label_permuted": ("/components/condition_label_map/artifact_id",),
    "learnable_credit_withheld": ("/learning/final_batch_update/apply_exact_pe_credit",),
    "steerable_strict_noop": ("/steering/executor_action",),
    "steerable_sensor_off_matched": ("/steering/sensor_condition_delivery",),
}
_ALLOWED_INTERVENTION_POINTERS_V3 = _ALLOWED_INTERVENTION_POINTERS_V2

_ARM_MATRIX_BY_PROTOCOL_ID = MappingProxyType(
    {
        P4_LONG_CONTEXT_PROTOCOL_ID_V1: _ARM_MATRIX_V1,
        P4_LONG_CONTEXT_PROTOCOL_ID_V2: _ARM_MATRIX_V2,
        P4_LONG_CONTEXT_PROTOCOL_ID_V3: _ARM_MATRIX_V3,
    }
)
_INTEGRATED_ARMS_BY_PROTOCOL_ID = MappingProxyType(
    {
        P4_LONG_CONTEXT_PROTOCOL_ID_V1: _INTEGRATED_ARMS_V1,
        P4_LONG_CONTEXT_PROTOCOL_ID_V2: _INTEGRATED_ARMS_V2,
        P4_LONG_CONTEXT_PROTOCOL_ID_V3: _INTEGRATED_ARMS_V3,
    }
)
_AXIS_CONTROLS_BY_PROTOCOL_ID = MappingProxyType(
    {
        P4_LONG_CONTEXT_PROTOCOL_ID_V1: _AXIS_CONTROLS_V1,
        P4_LONG_CONTEXT_PROTOCOL_ID_V2: _AXIS_CONTROLS_V2,
        P4_LONG_CONTEXT_PROTOCOL_ID_V3: _AXIS_CONTROLS_V3,
    }
)


@dataclass(frozen=True)
class _ProtocolDescriptor:
    version: str
    protocol_id: str
    protocol_schema: str
    protocol_path: pathlib.Path
    bundled_raw_sha256: str
    preparation_schema: str
    manifest_schema: str
    status: str
    published_artifact_id: str | None
    superseded_by: str | None
    preparation_allowed: bool


_PROTOCOLS_BY_ID = MappingProxyType(
    {
        P4_LONG_CONTEXT_PROTOCOL_ID_V1: _ProtocolDescriptor(
            version="v1",
            protocol_id=P4_LONG_CONTEXT_PROTOCOL_ID_V1,
            protocol_schema=P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V1,
            protocol_path=_V1_PROTOCOL_PATH,
            bundled_raw_sha256=("c9b8f828ddd39caa36272865cb1fdfb556eb6fa0e9a214b8692c5eb4f158417a"),
            preparation_schema=P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V1,
            manifest_schema=P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V1,
            status=P4_LONG_CONTEXT_PREPARATION_STATUS_V1,
            published_artifact_id=("899b7b0adc395186e108dc0a90c28c0d25ce67cd5445f61636cc2775d09b6901"),
            superseded_by=P4_LONG_CONTEXT_PROTOCOL_ID_V2,
            preparation_allowed=False,
        ),
        P4_LONG_CONTEXT_PROTOCOL_ID_V2: _ProtocolDescriptor(
            version="v2",
            protocol_id=P4_LONG_CONTEXT_PROTOCOL_ID_V2,
            protocol_schema=P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V2,
            protocol_path=_V2_PROTOCOL_PATH,
            bundled_raw_sha256=("24a748dde5ea2ba33943b7f66f35755acfcbdd87b36f45f48ee596f95b132d44"),
            preparation_schema=P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V2,
            manifest_schema=P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V2,
            status=P4_LONG_CONTEXT_PREPARATION_STATUS_V2,
            published_artifact_id=("795dea07eabda98c964ca50ee84694fd93bb6ee27fdd0370db7e6cb3ef01a8bd"),
            superseded_by=P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            preparation_allowed=False,
        ),
        P4_LONG_CONTEXT_PROTOCOL_ID_V3: _ProtocolDescriptor(
            version="v3",
            protocol_id=P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            protocol_schema=P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V3,
            protocol_path=_V3_PROTOCOL_PATH,
            bundled_raw_sha256=("ea8a17a14a68802d3b60586bf520c9137e6920be4112c951ec8c69f5e6ea359e"),
            preparation_schema=P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V3,
            manifest_schema=P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V3,
            status=P4_LONG_CONTEXT_PREPARATION_STATUS_V3,
            published_artifact_id=("c5a708ae5e68261fddbade165b45579e66e4bbe7db1be1f4a83056561a17f42e"),
            superseded_by=None,
            preparation_allowed=True,
        ),
    }
)
_PROTOCOLS_BY_SCHEMA = MappingProxyType({item.protocol_schema: item for item in _PROTOCOLS_BY_ID.values()})

# Independent per-section anchors make changing only a protocol and its global
# ID insufficient to loosen a frozen scientific or safety literal.  Object
# key order is intentionally erased by canonicalization; array order is not.
_V2_FROZEN_SECTION_SHA256 = MappingProxyType(
    {
        "schema_version": "4ffd83575e954fc11c5b4a2740b147938d7c89c603945cdb7c940788f8cd4c0e",
        "protocol_id_contract": "34c79b317bf8a9de446937cbfe2255d9409a7d831323787de601a2001142f2fe",
        "frozen_at_utc": "0b7125f04ef6def8a3a55fe03e4ad2ab7b2a5d33e663b22ebce81e88cc84037d",
        "owner": "24daca4641b8ee6658d6bb9a397b3967d4ad979492c30a7e7d47f399297d38ac",
        "supersession": "0b391bc8eb886b3987e0794028624ff69fc5a8ee66e84af30890d166ebc50e7d",
        "question": "c9d24a58a92527173f9435a7bab9127d86f3480b483d1e92d1c5cdd389dba369",
        "lineage": "67173347b470a2fff89f0650c4be2e99f0819bd9ecff1804ac09093998638478",
        "cohort": "7c3b44e5bd481d43493c579f5cbcbdc22e45f76ab2b2024b5801afd4aeb65bff",
        "longitudinal_design": "ab2c8a5a4366b32d8d0593c2f5053b0af2298df3a5ed05c7cda76aa93919d97b",
        "baseline_admission": "674e414919828c6ae48a8e3cacac720be99ea1adca58a98afcd66aece8fddf6a",
        "integrated_arms": "efd2622f4da07c94994cf76ec320858d298124e71e0fb157f34612933ac8924e",
        "arm_matrix": "ab772f89cc6d0d53db4a929905ef4de0e1e8b7002e65dfdf247d72b5398f4725",
        "axis_contrasts": "b1019b60bae657ea64bb9dc8e1edabdaae9b39e57f3bd56383db0544b5b47bd7",
        "intervention_integrity": "cfff340cca1e605748526ccd848e30f6b9cc84f56e6db99bdcfb1db72a9545f9",
        "shared_exposure_schedule": "e0276f0f3582a0cdab2727d8d4228370c7081385bda291cc5bfd959de6e075ef",
        "causal_execution": "de63b89ba4b72e63acd582d09c27b8341286284428d293ebed8dcf350dfdb9dd",
        "analysis": "279dced2209b6b5861bddd7b5f357a63fbcda0e4c726ebdc02f2954f7201fbe3",
        "execution_admission": "6a769b5ef7aebd43762d58190fedc8c41cf511430f56a0606f9b9176795e6896",
        "stopping_rules": "01f09be095689ab311cce7cd855e80153c21f989c380248155cb6a54b8f65ac3",
        "evidence_firewall": "b1a197a7f3f4ccf9baeb8bcdb94761a4ece11aec474c10075e8996cbdfaa4e49",
        "claim_boundary": "e2fdc8172d5d6087704843ee6bcc6bb844c87a31cf7ab226618deb7ad34347af",
    }
)
_V3_FROZEN_SECTION_SHA256 = MappingProxyType(
    {
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
)


@dataclass(frozen=True)
class P4LongContextAxisContrast:
    """One frozen subject-level causal contrast."""

    axis: str
    reference_arm: str
    control_arms: tuple[str, ...]
    primary_effect: str


@dataclass(frozen=True)
class RelationshipP4LongContextScientificPrereg:
    """Deeply selected immutable view of the source-controlled protocol."""

    protocol_id: str
    schema_version: str
    superseded: bool
    frozen_at_utc: str
    development_subject_count: int
    qualification_subject_count: int
    formal_subject_count: int
    minimum_complete_paired_subjects: int
    onboarding_sessions_per_subject: int
    learning_sessions_per_subject: int
    evaluation_sessions_per_subject: int
    minimum_public_history_tokens: int
    minimum_native_context_window_tokens: int
    minimum_generation_headroom_tokens: int
    arm_matrix: tuple[str, ...]
    integrated_arms: tuple[str, ...]
    axis_contrasts: tuple[P4LongContextAxisContrast, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    minimum_practical_mean_delta: float
    execution_enabled: bool
    formal_run_authorized: bool
    model_output_count_before_freeze: int
    subject_pack_materialization_count_before_freeze: int
    claim_boundary: str

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_id, "P4.7 protocol id")
        _require_timestamp(self.frozen_at_utc, "P4.7 frozen_at_utc")
        descriptor = _protocol_descriptor(self.protocol_id)
        if self.schema_version != descriptor.protocol_schema:
            raise ValueError("P4.7 protocol schema/id registry drift")
        if self.superseded is not (descriptor.superseded_by is not None):
            raise ValueError("P4.7 supersession view drift")
        if self.arm_matrix != _ARM_MATRIX_BY_PROTOCOL_ID[self.protocol_id]:
            raise ValueError("P4.7 arm matrix drift")
        if self.integrated_arms != _INTEGRATED_ARMS_BY_PROTOCOL_ID[self.protocol_id]:
            raise ValueError("P4.7 integrated arm order drift")
        axis_controls = _AXIS_CONTROLS_BY_PROTOCOL_ID[self.protocol_id]
        if tuple(item.axis for item in self.axis_contrasts) != tuple(axis_controls):
            raise ValueError("P4.7 axis contrast order drift")
        if any(
            item.reference_arm != "volvence_closed_loop" or item.control_arms != axis_controls[item.axis]
            for item in self.axis_contrasts
        ):
            raise ValueError("P4.7 axis control drift")
        if self.development_subject_count != 32:
            raise ValueError("P4.7 development cohort drift")
        if self.qualification_subject_count != 64:
            raise ValueError("P4.7 qualification cohort drift")
        if self.formal_subject_count != 192:
            raise ValueError("P4.7 formal cohort drift")
        if self.minimum_complete_paired_subjects != 160:
            raise ValueError("P4.7 minimum paired cohort drift")
        if (
            self.onboarding_sessions_per_subject,
            self.learning_sessions_per_subject,
            self.evaluation_sessions_per_subject,
        ) != (4, 8, 8):
            raise ValueError("P4.7 longitudinal horizon drift")
        if self.minimum_public_history_tokens != 32_768:
            raise ValueError("P4.7 public-history floor drift")
        if self.minimum_native_context_window_tokens < 65_536:
            raise ValueError("P4.7 native context window is too small")
        if self.minimum_generation_headroom_tokens < 1_024:
            raise ValueError("P4.7 generation headroom is too small")
        if (
            self.minimum_public_history_tokens + self.minimum_generation_headroom_tokens
            >= self.minimum_native_context_window_tokens
        ):
            raise ValueError("P4.7 context feasibility has no overhead room")
        if self.bootstrap_replicates != 100_000:
            raise ValueError("P4.7 bootstrap count drift")
        if self.bootstrap_seed != 20_260_823:
            raise ValueError("P4.7 bootstrap seed drift")
        if self.minimum_practical_mean_delta != 0.15:
            raise ValueError("P4.7 minimum effect drift")
        if self.execution_enabled or self.formal_run_authorized:
            raise ValueError("P4.7 design prereg cannot authorize execution")
        if self.model_output_count_before_freeze != 0:
            raise ValueError("P4.7 must freeze before model output")
        if self.subject_pack_materialization_count_before_freeze != 0:
            raise ValueError("P4.7 must freeze before subject materialization")
        if not self.claim_boundary.strip():
            raise ValueError("P4.7 claim boundary is empty")


@dataclass(frozen=True)
class RelationshipP4LongContextPreparation:
    """Validated create-only zero-output preparation artifact."""

    artifact_id: str
    protocol_id: str
    status: str
    execution_enabled: bool
    formal_run_authorized: bool
    output_dir: pathlib.Path

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "P4.7 preparation artifact id")
        _require_sha256(self.protocol_id, "P4.7 preparation protocol id")
        descriptor = _protocol_descriptor(self.protocol_id)
        if self.status != descriptor.status:
            raise ValueError("P4.7 preparation status drift")
        if descriptor.published_artifact_id is not None and self.artifact_id != descriptor.published_artifact_id:
            raise ValueError("P4.7 published preparation artifact id drift")
        if self.execution_enabled or self.formal_run_authorized:
            raise ValueError("P4.7 preparation firewall is open")


@dataclass(frozen=True)
class RelationshipP4LongContextPowerFailureCertificate:
    """Preserved v1 numeric certificate rejected for v3 grid admission."""

    artifact_id: str
    protocol_id: str
    preparation_artifact_id: str
    status: str
    point_gate_power_numerator: int
    point_gate_power_denominator: int
    point_gate_power_display_decimal: str
    decisive_failure: bool
    full_joint_grid_completed: bool
    development_authorized: bool
    formal_authorized: bool
    output_dir: pathlib.Path

    @property
    def scientific_admission(self) -> bool:
        """V1 arithmetic is reproducible but its v3 applicability was not frozen."""

        return False

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "P4.7 power failure artifact id")
        if (
            P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID is not None
            and self.artifact_id != P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID
        ):
            raise ValueError("P4.7 published power failure artifact id drift")
        if self.protocol_id != P4_LONG_CONTEXT_PROTOCOL_ID_V3:
            raise ValueError("P4.7 power failure must bind v3")
        descriptor = _protocol_descriptor(self.protocol_id)
        if self.preparation_artifact_id != descriptor.published_artifact_id:
            raise ValueError("P4.7 power failure preparation lineage drift")
        if self.status != P4_LONG_CONTEXT_POWER_FAILURE_STATUS:
            raise ValueError("P4.7 power failure status drift")
        exact_tail = _maximum_variance_point_gate_probability(
            formal_root_count=192,
            mass_at_upper=Fraction(11, 20),
            minimum_upper_count=104,
        )
        if self.point_gate_power_numerator != exact_tail.numerator:
            raise ValueError("P4.7 power failure numerator drift")
        if self.point_gate_power_denominator != exact_tail.denominator:
            raise ValueError("P4.7 power failure denominator drift")
        if self.point_gate_power_display_decimal != _fraction_display_decimal(
            exact_tail,
            places=20,
        ):
            raise ValueError("P4.7 power failure display decimal drift")
        if not self.decisive_failure:
            raise ValueError("P4.7 power failure must be decisive")
        if self.full_joint_grid_completed:
            raise ValueError("P4.7 necessary-condition certificate cannot claim full grid")
        if self.development_authorized or self.formal_authorized:
            raise ValueError("P4.7 failed power gate cannot authorize execution")


@dataclass(frozen=True)
class RelationshipP4LongContextPowerAdmissionCertificate:
    """Current zero-output finding that v3 power-grid admission is ambiguous."""

    artifact_id: str
    admission_protocol_id: str
    scientific_protocol_id: str
    preparation_artifact_id: str
    status: str
    conditional_bound_numerator: int
    conditional_bound_denominator: int
    conditional_bound_display_decimal: str
    power_contract_determinate: bool
    v1_unconditional_scientific_admission_valid: bool
    development_authorized: bool
    formal_authorized: bool
    output_dir: pathlib.Path

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "P4.7 power admission artifact id")
        if (
            P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2 is not None
            and self.artifact_id != P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2
        ):
            raise ValueError("P4.7 published power admission artifact id drift")
        if self.admission_protocol_id != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2:
            raise ValueError("P4.7 power admission protocol lineage drift")
        if self.scientific_protocol_id != P4_LONG_CONTEXT_PROTOCOL_ID_V3:
            raise ValueError("P4.7 power admission must bind scientific v3")
        descriptor = _protocol_descriptor(self.scientific_protocol_id)
        if self.preparation_artifact_id != descriptor.published_artifact_id:
            raise ValueError("P4.7 power admission preparation lineage drift")
        if self.status != P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2:
            raise ValueError("P4.7 power admission status drift")
        conditional_tail = _maximum_variance_point_gate_probability(
            formal_root_count=192,
            mass_at_upper=Fraction(11, 20),
            minimum_upper_count=104,
        )
        if self.conditional_bound_numerator != conditional_tail.numerator:
            raise ValueError("P4.7 conditional bound numerator drift")
        if self.conditional_bound_denominator != conditional_tail.denominator:
            raise ValueError("P4.7 conditional bound denominator drift")
        if self.conditional_bound_display_decimal != _fraction_display_decimal(
            conditional_tail,
            places=20,
        ):
            raise ValueError("P4.7 conditional bound display drift")
        if self.power_contract_determinate:
            raise ValueError("P4.7 v3 power contract cannot be marked determinate")
        if self.v1_unconditional_scientific_admission_valid:
            raise ValueError("P4.7 v1 unconditional admission cannot be restored")
        if self.development_authorized or self.formal_authorized:
            raise ValueError("P4.7 under-specified power contract cannot authorize execution")


@dataclass(frozen=True)
class RelationshipP4LongContextV4PlanningProtocol:
    """Selected immutable view of the v4a zero-output planning primitives."""

    protocol_id: str
    schema_version: str
    frozen_at_utc: str
    candidate_root_counts: tuple[int, ...]
    first_necessary_screen_passing_root_count: int
    first_positive_mean_gate_capable_root_count: int
    cartesian_candidate_tuple_count: int
    candidate_cell_ids: tuple[str, ...]
    schedule_block_count: int
    power_contract_determinate: bool
    source_grid_resolved: bool
    selected_formal_root_count: int | None
    claim_boundary: str

    def __post_init__(self) -> None:
        if self.protocol_id != P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1:
            raise ValueError("P4.7 v4 planning protocol id drift")
        if self.schema_version != P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_SCHEMA_VERSION_V1:
            raise ValueError("P4.7 v4 planning schema drift")
        _require_timestamp(self.frozen_at_utc, "P4.7 v4 planning frozen_at_utc")
        if len(self.candidate_root_counts) != 126:
            raise ValueError("P4.7 v4 candidate root-count grid drift")
        if self.first_necessary_screen_passing_root_count != 1088:
            raise ValueError("P4.7 v4 necessary point-screen lower candidate drift")
        if self.first_positive_mean_gate_capable_root_count != 1856:
            raise ValueError("P4.7 v4 positive-mean gate lower candidate drift")
        if self.cartesian_candidate_tuple_count != 576:
            raise ValueError("P4.7 v4 Cartesian candidate count drift")
        if len(self.candidate_cell_ids) != 6 or len(set(self.candidate_cell_ids)) != 6:
            raise ValueError("P4.7 v4 candidate cell inventory drift")
        if self.schedule_block_count != 640:
            raise ValueError("P4.7 v4 candidate schedule block count drift")
        if not self.power_contract_determinate or self.source_grid_resolved:
            raise ValueError("P4.7 v4 planning terminal state drift")
        if self.selected_formal_root_count is not None:
            raise ValueError("P4.7 v4a cannot select a formal root count")
        if not self.claim_boundary.strip():
            raise ValueError("P4.7 v4 planning claim boundary is empty")


@dataclass(frozen=True)
class RelationshipP4LongContextV4PlanningFreeze:
    """Validated create-only v4a plan and abstract six-cell schedule."""

    artifact_id: str
    protocol_id: str
    scientific_v3_protocol_id: str
    power_admission_v2_artifact_id: str
    status: str
    first_necessary_screen_passing_root_count: int
    first_positive_mean_gate_capable_root_count: int
    cartesian_candidate_tuple_count: int
    candidate_schedule_block_count: int
    power_contract_determinate: bool
    source_grid_resolved: bool
    selected_formal_root_count: int | None
    source_materialization_authorized: bool
    development_authorized: bool
    model_output_authorized: bool
    formal_authorized: bool
    output_dir: pathlib.Path

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "P4.7 v4 planning artifact id")
        if (
            P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1 is not None
            and self.artifact_id != P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1
        ):
            raise ValueError("P4.7 published v4 planning artifact id drift")
        if self.protocol_id != P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1:
            raise ValueError("P4.7 v4 planning artifact protocol lineage drift")
        if self.scientific_v3_protocol_id != P4_LONG_CONTEXT_PROTOCOL_ID_V3:
            raise ValueError("P4.7 v4 planning scientific lineage drift")
        if self.power_admission_v2_artifact_id != P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2:
            raise ValueError("P4.7 v4 planning admission lineage drift")
        if self.status != P4_LONG_CONTEXT_V4_PLANNING_STATUS_V1:
            raise ValueError("P4.7 v4 planning status drift")
        if self.first_necessary_screen_passing_root_count != 1088:
            raise ValueError("P4.7 v4 planning necessary screen drift")
        if self.first_positive_mean_gate_capable_root_count != 1856:
            raise ValueError("P4.7 v4 planning bounded-mean screen drift")
        if self.cartesian_candidate_tuple_count != 576:
            raise ValueError("P4.7 v4 planning tuple count drift")
        if self.candidate_schedule_block_count != 640:
            raise ValueError("P4.7 v4 planning schedule count drift")
        if not self.power_contract_determinate or self.source_grid_resolved:
            raise ValueError("P4.7 v4 planning resolution state drift")
        if self.selected_formal_root_count is not None:
            raise ValueError("P4.7 v4 planning artifact cannot select N")
        if (
            self.source_materialization_authorized
            or self.development_authorized
            or self.model_output_authorized
            or self.formal_authorized
        ):
            raise ValueError("P4.7 v4 planning artifact opened an execution firewall")


@dataclass(frozen=True)
class RelationshipP4LongContextSourceOpportunityPreflightProtocol:
    """Validated view of the source-opportunity zero-output contract."""

    protocol_id: str
    schema_version: str
    frozen_at_utc: str
    v4a_planning_protocol_id: str
    v4a_planning_artifact_id: str
    action_registry_id: str
    action_ids: tuple[str, ...]
    independent_root_slot_count: int
    counterfactual_twin_mapping_count: int
    formal_candidate_prefix_count: int
    generic_decision_atom_count: int
    zero_output_preflight_contract_frozen: bool
    source_opportunity_stage_completed: bool
    source_structural_inventory_materialized: bool
    future_structural_inventory_scope_defined: bool
    future_structural_inventory_materialization_authorized: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if self.protocol_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1:
            raise ValueError("P4.7 source-opportunity preflight protocol id drift")
        if self.schema_version != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SCHEMA_VERSION_V1:
            raise ValueError("P4.7 source-opportunity preflight schema drift")
        _require_timestamp(self.frozen_at_utc, "P4.7 source preflight frozen_at_utc")
        if self.v4a_planning_protocol_id != P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1:
            raise ValueError("P4.7 source preflight v4a protocol lineage drift")
        if self.v4a_planning_artifact_id != P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1:
            raise ValueError("P4.7 source preflight v4a artifact lineage drift")
        _require_sha256(self.action_registry_id, "P4.7 source preflight action registry id")
        if self.action_ids != _RELATIONSHIP_ACTION_IDS_V1:
            raise ValueError("P4.7 source preflight action order drift")
        if (
            self.independent_root_slot_count,
            self.counterfactual_twin_mapping_count,
            self.formal_candidate_prefix_count,
            self.generic_decision_atom_count,
        ) != (16576, 8288, 126, 512):
            raise ValueError("P4.7 source preflight finite inventory drift")
        if not self.zero_output_preflight_contract_frozen:
            raise ValueError("P4.7 source preflight contract was not frozen")
        if self.source_opportunity_stage_completed or self.source_structural_inventory_materialized:
            raise ValueError("P4.7 zero-output preflight claimed source-stage completion")
        if not self.future_structural_inventory_scope_defined:
            raise ValueError("P4.7 future structural inventory scope is not defined")
        if self.future_structural_inventory_materialization_authorized:
            raise ValueError("P4.7 unanchored preflight authorized materialization")
        if not self.claim_boundary.strip():
            raise ValueError("P4.7 source preflight claim boundary is empty")


@dataclass(frozen=True)
class RelationshipP4LongContextSourceOpportunityPreflightCertificate:
    """Validated create-only source-opportunity preflight artifact."""

    artifact_id: str
    certificate_id: str
    contract_projection_id: str
    protocol_id: str
    v4a_planning_artifact_id: str
    action_registry_id: str
    status: str
    zero_output_preflight_contract_frozen: bool
    source_opportunity_stage_completed: bool
    source_structural_inventory_materialized: bool
    unresolved_tuple_count: int
    selected_formal_root_count: int | None
    current_source_execution_authorized: bool
    tuple_feasibility_authorized: bool
    model_output_authorized: bool
    development_authorized: bool
    qualification_authorized: bool
    formal_authorized: bool
    cuda_planner_authorized: bool
    output_dir: pathlib.Path

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "P4.7 source preflight artifact id")
        _require_sha256(self.certificate_id, "P4.7 source preflight certificate id")
        _require_sha256(self.contract_projection_id, "P4.7 source contract projection id")
        _require_sha256(self.action_registry_id, "P4.7 source preflight action registry id")
        if (
            P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1 is not None
            and self.artifact_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1
        ):
            raise ValueError("P4.7 published source preflight artifact id drift")
        if (
            P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1 is not None
            and self.certificate_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1
        ):
            raise ValueError("P4.7 published source preflight certificate id drift")
        if self.protocol_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1:
            raise ValueError("P4.7 source preflight protocol lineage drift")
        if self.v4a_planning_artifact_id != P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1:
            raise ValueError("P4.7 source preflight v4a artifact lineage drift")
        if self.status != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_STATUS_V1:
            raise ValueError("P4.7 source preflight terminal status drift")
        if not self.zero_output_preflight_contract_frozen:
            raise ValueError("P4.7 source preflight certificate is not frozen")
        if self.source_opportunity_stage_completed or self.source_structural_inventory_materialized:
            raise ValueError("P4.7 source preflight certificate claimed source output")
        if self.unresolved_tuple_count != 576 or self.selected_formal_root_count is not None:
            raise ValueError("P4.7 source preflight changed tuple or sample-size state")
        if any(
            (
                self.current_source_execution_authorized,
                self.tuple_feasibility_authorized,
                self.model_output_authorized,
                self.development_authorized,
                self.qualification_authorized,
                self.formal_authorized,
                self.cuda_planner_authorized,
            )
        ):
            raise ValueError("P4.7 source preflight opened an execution firewall")


@dataclass(frozen=True)
class RelationshipP4LongContextExternalAnchorRequestProtocol:
    """Validated view of the A0 public-anchor request contract."""

    protocol_id: str
    schema_version: str
    frozen_at_utc: str
    source_preflight_protocol_id: str
    source_preflight_artifact_id: str
    publication_subject_count: int
    provider: str
    expected_owner_login: str
    required_filename: str
    publication_request_contract_frozen: bool
    external_publication_anchor_present: bool
    structural_inventory_materialization_authorized: bool
    status: str
    claim_boundary: str

    def __post_init__(self) -> None:
        if self.protocol_id != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1:
            raise ValueError("P4.7 A0 anchor-request protocol id drift")
        if self.schema_version != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SCHEMA_VERSION_V1:
            raise ValueError("P4.7 A0 anchor-request schema drift")
        _require_timestamp(self.frozen_at_utc, "P4.7 A0 anchor-request frozen_at_utc")
        if self.source_preflight_protocol_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1:
            raise ValueError("P4.7 A0 source protocol lineage drift")
        if self.source_preflight_artifact_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1:
            raise ValueError("P4.7 A0 source artifact lineage drift")
        if self.publication_subject_count != 5:
            raise ValueError("P4.7 A0 publication subject count drift")
        if self.provider != "github_public_gist_first_revision_v1":
            raise ValueError("P4.7 A0 publication provider drift")
        if self.expected_owner_login != "ronaldzgithub":
            raise ValueError("P4.7 A0 expected Gist owner drift")
        if self.required_filename != "volvence_p4_7_source_opportunity_a0_anchor_request.json":
            raise ValueError("P4.7 A0 required Gist filename drift")
        if not self.publication_request_contract_frozen:
            raise ValueError("P4.7 A0 request contract is not frozen")
        if self.external_publication_anchor_present or self.structural_inventory_materialization_authorized:
            raise ValueError("P4.7 A0 request opened downstream authority")
        if self.status != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_STATUS_V1:
            raise ValueError("P4.7 A0 request status drift")
        if not self.claim_boundary.strip():
            raise ValueError("P4.7 A0 request claim boundary is empty")


@dataclass(frozen=True)
class RelationshipP4LongContextExternalAnchorRequest:
    """Validated create-only A0 request artifact with no external action."""

    artifact_id: str
    request_id: str
    protocol_id: str
    status: str
    publication_request_contract_frozen: bool
    external_request_dispatched: bool
    publication_performed: bool
    external_publication_anchor_present: bool
    external_anchor_admitted: bool
    structural_inventory_materialization_authorized: bool
    source_execution_authorized: bool
    tuple_feasibility_authorized: bool
    model_output_authorized: bool
    cuda_planner_authorized: bool
    output_dir: pathlib.Path

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_id, "P4.7 A0 request artifact id")
        _require_sha256(self.request_id, "P4.7 A0 request id")
        if (
            P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1 is not None
            and self.artifact_id != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1
        ):
            raise ValueError("P4.7 published A0 request artifact id drift")
        if (
            P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1 is not None
            and self.request_id != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1
        ):
            raise ValueError("P4.7 published A0 request id drift")
        if self.protocol_id != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1:
            raise ValueError("P4.7 A0 request protocol lineage drift")
        if self.status != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_STATUS_V1:
            raise ValueError("P4.7 A0 request artifact status drift")
        if not self.publication_request_contract_frozen:
            raise ValueError("P4.7 A0 request artifact is not frozen")
        if any(
            (
                self.external_request_dispatched,
                self.publication_performed,
                self.external_publication_anchor_present,
                self.external_anchor_admitted,
                self.structural_inventory_materialization_authorized,
                self.source_execution_authorized,
                self.tuple_feasibility_authorized,
                self.model_output_authorized,
                self.cuda_planner_authorized,
            )
        ):
            raise ValueError("P4.7 A0 request artifact opened an authorization firewall")


@dataclass(frozen=True)
class _V4SourceActionRegistry:
    owner_module_raw_sha256: str
    schema_raw_sha256: str
    schema_id: str
    action_ids: tuple[str, ...]
    registry_id: str


@dataclass(frozen=True)
class _V4SourceOpportunityDerived:
    action_registry: _V4SourceActionRegistry
    root_layout: SourceRootSurfaceDerivation
    evaluation_design: SourceEvaluationDesign
    fact_orientations: RootFactOrientationDerivation
    planning_generator: SyntheticPlanningGeneratorDerivation


@dataclass(frozen=True)
class _V4PlanningDerived:
    candidate_root_counts: tuple[int, ...]
    necessary_point_screens: tuple[V4NecessaryScreenResult, ...]
    first_necessary_screen_root_count: int
    first_necessary_screen_minimum_plus_count: int
    first_necessary_screen_power: Fraction
    first_positive_mean_gate_root_count: int
    cartesian_candidate_tuple_count: int
    candidate_cell_ids: tuple[str, ...]
    candidate_schedule: tuple[V4CandidateScheduleBlock, ...]
    action_classifications: tuple[tuple[str, str, str], ...]


def relationship_p4_long_context_protocol_path(
    protocol_id: str | None = None,
) -> pathlib.Path:
    """Return the bundled path for a registered protocol, v3 by default."""

    selected_id = P4_LONG_CONTEXT_PROTOCOL_ID if protocol_id is None else protocol_id
    return _protocol_descriptor(selected_id).protocol_path


def load_relationship_p4_long_context_scientific_prereg(
    path: pathlib.Path | None = None,
) -> RelationshipP4LongContextScientificPrereg:
    """Load and fully validate the immutable zero-output design protocol."""

    protocol_path = pathlib.Path(_DEFAULT_PROTOCOL_PATH if path is None else path)
    raw = _load_json_object(protocol_path)
    schema = _require_text(raw.get("schema_version"), "P4.7 protocol schema")
    descriptor = _PROTOCOLS_BY_SCHEMA.get(schema)
    if descriptor is None:
        raise ValueError(f"P4.7 unregistered protocol schema: {schema}")
    if raw["protocol_id_contract"] != "sha256_canonical_json_utf8_newline_v1":
        raise ValueError("P4.7 protocol id contract drift")
    protocol_id = _sha256_bytes(_canonical_bytes(raw))
    if protocol_id != descriptor.protocol_id:
        raise ValueError("P4.7 unregistered or drifted protocol id")
    if protocol_path.resolve() == descriptor.protocol_path.resolve():
        raw_sha256 = _sha256_bytes(protocol_path.read_bytes())
        if raw_sha256 != descriptor.bundled_raw_sha256:
            raise ValueError("P4.7 bundled protocol raw bytes drift")
    if descriptor.version in {"v2", "v3"}:
        return _parse_v2_protocol(raw, descriptor)
    return _parse_v1_protocol(raw, descriptor)


def load_relationship_p4_long_context_v4_planning_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP4LongContextV4PlanningProtocol:
    """Load and mechanically derive the v4a planning-only contract."""

    raw = _load_v4_planning_protocol_raw(path)
    derived = _validate_v4_planning_derivation(raw)
    return _v4_planning_protocol_view(raw, derived)


def _v4_planning_protocol_view(
    raw: Mapping[str, Any],
    derived: _V4PlanningDerived,
) -> RelationshipP4LongContextV4PlanningProtocol:
    terminal = _require_mapping(raw["terminal"], "P4.7 v4 planning terminal")
    return RelationshipP4LongContextV4PlanningProtocol(
        protocol_id=P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1,
        schema_version=_require_text(raw["schema_version"], "P4.7 v4 planning schema"),
        frozen_at_utc=_require_text(raw["frozen_at_utc"], "P4.7 v4 planning frozen_at_utc"),
        candidate_root_counts=derived.candidate_root_counts,
        first_necessary_screen_passing_root_count=(derived.first_necessary_screen_root_count),
        first_positive_mean_gate_capable_root_count=(derived.first_positive_mean_gate_root_count),
        cartesian_candidate_tuple_count=derived.cartesian_candidate_tuple_count,
        candidate_cell_ids=derived.candidate_cell_ids,
        schedule_block_count=len(derived.candidate_schedule),
        power_contract_determinate=_require_bool(
            terminal["power_contract_determinate"],
            "P4.7 v4 power contract determinate",
        ),
        source_grid_resolved=_require_bool(
            terminal["full_joint_grid_completed"],
            "P4.7 v4 source grid resolved",
        ),
        selected_formal_root_count=terminal["selected_formal_root_count"],
        claim_boundary=_require_text(raw["claim_boundary"], "P4.7 v4 claim boundary"),
    )


def load_relationship_p4_long_context_source_opportunity_preflight_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP4LongContextSourceOpportunityPreflightProtocol:
    """Load and mechanically validate the zero-output source contract."""

    raw = _load_v4_source_opportunity_preflight_protocol_raw(path)
    derived = _validate_v4_source_opportunity_derivation(raw)
    return _v4_source_opportunity_protocol_view(raw, derived)


def _v4_source_opportunity_protocol_view(
    raw: Mapping[str, Any],
    derived: _V4SourceOpportunityDerived,
) -> RelationshipP4LongContextSourceOpportunityPreflightProtocol:
    lineage = _require_mapping(raw["input_lineage"], "P4.7 source preflight input lineage")
    stage = _require_mapping(raw["stage_boundary"], "P4.7 source preflight stage boundary")
    capacity = _require_mapping(
        raw["root_independence_and_capacity"],
        "P4.7 source preflight capacity",
    )
    generator = _require_mapping(
        raw["exact_source_planning_generator"],
        "P4.7 source preflight generator",
    )
    return RelationshipP4LongContextSourceOpportunityPreflightProtocol(
        protocol_id=P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1,
        schema_version=_require_text(raw["schema_version"], "P4.7 source preflight schema"),
        frozen_at_utc=_require_text(
            raw["frozen_at_utc"],
            "P4.7 source preflight frozen_at_utc",
        ),
        v4a_planning_protocol_id=_require_sha256(
            lineage["v4a_planning_protocol_id"],
            "P4.7 source preflight v4a protocol id",
        ),
        v4a_planning_artifact_id=_require_sha256(
            lineage["v4a_planning_artifact_id"],
            "P4.7 source preflight v4a artifact id",
        ),
        action_registry_id=derived.action_registry.registry_id,
        action_ids=derived.action_registry.action_ids,
        independent_root_slot_count=_require_int(
            capacity["preallocated_independent_root_slot_count"],
            "P4.7 source preflight root slot count",
        ),
        counterfactual_twin_mapping_count=_require_int(
            capacity["deterministic_counterfactual_twin_mapping_count"],
            "P4.7 source preflight twin mapping count",
        ),
        formal_candidate_prefix_count=_require_int(
            _require_mapping(
                capacity["formal_candidate_root_counts"],
                "P4.7 source preflight formal candidates",
            )["count"],
            "P4.7 source preflight formal candidate count",
        ),
        generic_decision_atom_count=_require_int(
            generator["full_nine_arm_scalar_atom_count"],
            "P4.7 source preflight atom count",
        ),
        zero_output_preflight_contract_frozen=_require_bool(
            stage["zero_output_preflight_contract_frozen_by_this_artifact"],
            "P4.7 source preflight contract frozen",
        ),
        source_opportunity_stage_completed=_require_bool(
            stage["source_opportunity_stage_completed_by_this_artifact"],
            "P4.7 source opportunity stage completed",
        ),
        source_structural_inventory_materialized=_require_bool(
            stage["source_structural_inventory_materialized_by_this_artifact"],
            "P4.7 source inventory materialized",
        ),
        future_structural_inventory_scope_defined=_require_bool(
            stage["future_structural_inventory_scope_is_defined_by_this_contract"],
            "P4.7 future source scope defined",
        ),
        future_structural_inventory_materialization_authorized=_require_bool(
            stage["future_single_create_only_structural_inventory_attempt_authorized_by_this_contract"],
            "P4.7 future source materialization authorization",
        ),
        claim_boundary=_require_text(
            raw["claim_boundary"],
            "P4.7 source preflight claim boundary",
        ),
    )


def load_relationship_p4_long_context_external_anchor_request_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP4LongContextExternalAnchorRequestProtocol:
    """Load the frozen local A0 request contract without external I/O."""

    raw = _load_v4_external_anchor_request_protocol_raw(path)
    return _v4_external_anchor_request_protocol_view(raw)


def _v4_external_anchor_request_protocol_view(
    raw: Mapping[str, Any],
) -> RelationshipP4LongContextExternalAnchorRequestProtocol:
    lineage = _require_mapping(raw["input_lineage"], "P4.7 A0 request lineage")
    subjects = _require_mapping(raw["publication_subject_contract"], "P4.7 A0 subjects")
    target = _require_mapping(raw["publication_target_contract"], "P4.7 A0 target")
    firewall = _require_mapping(raw["authorization_firewall"], "P4.7 A0 authorization firewall")
    terminal = _require_mapping(raw["terminal"], "P4.7 A0 terminal")
    return RelationshipP4LongContextExternalAnchorRequestProtocol(
        protocol_id=P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1,
        schema_version=_require_text(raw["schema_version"], "P4.7 A0 request schema"),
        frozen_at_utc=_require_text(raw["frozen_at_utc"], "P4.7 A0 request frozen_at_utc"),
        source_preflight_protocol_id=_require_sha256(
            lineage["source_preflight_protocol_id"],
            "P4.7 A0 source protocol id",
        ),
        source_preflight_artifact_id=_require_sha256(
            lineage["source_preflight_artifact_id"],
            "P4.7 A0 source artifact id",
        ),
        publication_subject_count=_require_int(subjects["subject_count"], "P4.7 A0 subject count"),
        provider=_require_text(target["provider"], "P4.7 A0 provider"),
        expected_owner_login=_require_text(
            target["expected_owner_login"],
            "P4.7 A0 expected owner",
        ),
        required_filename=_require_text(target["required_filename"], "P4.7 A0 filename"),
        publication_request_contract_frozen=_require_bool(
            firewall["publication_request_contract_frozen"],
            "P4.7 A0 request frozen",
        ),
        external_publication_anchor_present=_require_bool(
            firewall["external_publication_anchor_present"],
            "P4.7 A0 external anchor present",
        ),
        structural_inventory_materialization_authorized=_require_bool(
            firewall["structural_inventory_materialization_authorized"],
            "P4.7 A0 materialization authorization",
        ),
        status=_require_text(terminal["status"], "P4.7 A0 status"),
        claim_boundary=_require_text(raw["claim_boundary"], "P4.7 A0 claim boundary"),
    )


def _load_power_bound_protocol_v1() -> Mapping[str, Any]:
    raw_bytes = _POWER_BOUND_PROTOCOL_PATH_V1.read_bytes()
    if _sha256_bytes(raw_bytes) != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1:
        raise ValueError("P4.7 power-bound v1 protocol raw bytes drift")
    raw = _load_json_object(_POWER_BOUND_PROTOCOL_PATH_V1)
    if _sha256_bytes(_canonical_bytes(raw)) != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1:
        raise ValueError("P4.7 power-bound v1 protocol id drift")
    _validate_frozen_sections(
        raw,
        expected_keys=frozenset(_POWER_BOUND_PROTOCOL_SECTION_SHA256_V1),
        expected_sha256=_POWER_BOUND_PROTOCOL_SECTION_SHA256_V1,
        label="P4.7 power-bound v1",
    )
    if raw["schema_version"] != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V1:
        raise ValueError("P4.7 power-bound v1 protocol schema drift")
    if raw["protocol_id_contract"] != "sha256_canonical_json_utf8_newline_v1":
        raise ValueError("P4.7 power-bound v1 protocol id contract drift")
    return raw


def _load_power_bound_protocol_v2() -> Mapping[str, Any]:
    raw_bytes = _POWER_BOUND_PROTOCOL_PATH_V2.read_bytes()
    if _sha256_bytes(raw_bytes) != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2:
        raise ValueError("P4.7 power-bound v2 protocol raw bytes drift")
    raw = _load_json_object(_POWER_BOUND_PROTOCOL_PATH_V2)
    if _sha256_bytes(_canonical_bytes(raw)) != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2:
        raise ValueError("P4.7 power-bound v2 protocol id drift")
    _validate_frozen_sections(
        raw,
        expected_keys=frozenset(_POWER_BOUND_PROTOCOL_SECTION_SHA256_V2),
        expected_sha256=_POWER_BOUND_PROTOCOL_SECTION_SHA256_V2,
        label="P4.7 power-bound v2",
    )
    if raw["schema_version"] != P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V2:
        raise ValueError("P4.7 power-bound v2 protocol schema drift")
    if raw["protocol_id_contract"] != "sha256_canonical_json_utf8_newline_v1":
        raise ValueError("P4.7 power-bound v2 protocol id contract drift")
    return raw


def _parse_v1_protocol(
    raw: Mapping[str, Any],
    descriptor: _ProtocolDescriptor,
) -> RelationshipP4LongContextScientificPrereg:
    _require_exact_keys(raw, _TOP_LEVEL_KEYS_V1, "P4.7 v1 protocol")

    owner = _require_mapping(raw["owner"], "P4.7 owner")
    _require_literal(
        owner,
        {
            "wheel": "lifeform-evolution",
            "module": ("lifeform_evolution.relationship_lab_p4_long_context_causal_campaign"),
            "role": "offline_scientific_prereg_owner",
            "new_runtime_slot_added": False,
            "execution_implemented": False,
        },
        "P4.7 owner",
    )
    _require_text(raw["question"], "P4.7 question")
    _validate_lineage(_require_mapping(raw["lineage"], "P4.7 lineage"))
    cohort = _require_mapping(raw["cohort"], "P4.7 cohort")
    longitudinal = _require_mapping(
        raw["longitudinal_design"],
        "P4.7 longitudinal design",
    )
    _validate_cohort(cohort)
    _validate_longitudinal(longitudinal)
    _validate_baseline(_require_mapping(raw["baseline_admission"], "P4.7 baseline"))

    arm_matrix = _require_text_tuple(raw["arm_matrix"], "P4.7 arm matrix")
    integrated_arms = _require_text_tuple(
        raw["integrated_arms"],
        "P4.7 integrated arms",
    )
    if arm_matrix != _ARM_MATRIX_V1:
        raise ValueError("P4.7 arm matrix content drift")
    if integrated_arms != _INTEGRATED_ARMS_V1:
        raise ValueError("P4.7 integrated arms content drift")
    contrasts = _parse_axis_contrasts_v1(_require_mapping(raw["axis_contrasts"], "P4.7 axis contrasts"))
    _validate_intervention_integrity(
        _require_mapping(
            raw["intervention_integrity"],
            "P4.7 intervention integrity",
        )
    )
    _validate_shared_exposure(
        _require_mapping(
            raw["shared_exposure_schedule"],
            "P4.7 shared exposure",
        )
    )
    _validate_causal_execution(_require_mapping(raw["causal_execution"], "P4.7 causal execution"))
    analysis = _require_mapping(raw["analysis"], "P4.7 analysis")
    _validate_analysis(analysis)
    admission = _require_mapping(
        raw["execution_admission"],
        "P4.7 execution admission",
    )
    _validate_execution_admission(admission)
    stopping_rules = _require_text_tuple(
        raw["stopping_rules"],
        "P4.7 stopping rules",
    )
    if len(stopping_rules) != 9 or len(set(stopping_rules)) != 9:
        raise ValueError("P4.7 stopping-rule inventory drift")
    _validate_evidence_firewall(_require_mapping(raw["evidence_firewall"], "P4.7 firewall"))

    splits = _require_mapping(cohort["splits"], "P4.7 cohort splits")
    development = _require_mapping(splits["development"], "development split")
    qualification = _require_mapping(
        splits["qualification"],
        "qualification split",
    )
    formal = _require_mapping(splits["formal"], "formal split")
    return RelationshipP4LongContextScientificPrereg(
        protocol_id=descriptor.protocol_id,
        schema_version=descriptor.protocol_schema,
        superseded=True,
        frozen_at_utc=_require_timestamp(
            raw["frozen_at_utc"],
            "P4.7 frozen_at_utc",
        ),
        development_subject_count=_require_int(
            development["subject_count"],
            "development subject count",
        ),
        qualification_subject_count=_require_int(
            qualification["subject_count"],
            "qualification subject count",
        ),
        formal_subject_count=_require_int(
            formal["preallocated_subject_count"],
            "formal subject count",
        ),
        minimum_complete_paired_subjects=_require_int(
            formal["minimum_complete_paired_subjects"],
            "minimum paired subjects",
        ),
        onboarding_sessions_per_subject=_require_int(
            longitudinal["onboarding_sessions_per_subject"],
            "onboarding sessions",
        ),
        learning_sessions_per_subject=_require_int(
            longitudinal["matched_learning_exposure_sessions_per_subject"],
            "learning sessions",
        ),
        evaluation_sessions_per_subject=_require_int(
            longitudinal["frozen_policy_evaluation_sessions_per_subject"],
            "evaluation sessions",
        ),
        minimum_public_history_tokens=_require_int(
            longitudinal["final_decision_public_history_tokens_minimum"],
            "public history tokens",
        ),
        minimum_native_context_window_tokens=_require_int(
            longitudinal["minimum_native_context_window_tokens"],
            "native context tokens",
        ),
        minimum_generation_headroom_tokens=_require_int(
            longitudinal["minimum_generation_headroom_tokens"],
            "generation headroom tokens",
        ),
        arm_matrix=arm_matrix,
        integrated_arms=integrated_arms,
        axis_contrasts=contrasts,
        bootstrap_replicates=_require_int(
            analysis["bootstrap_replicates"],
            "bootstrap replicates",
        ),
        bootstrap_seed=_require_int(
            analysis["bootstrap_seed"],
            "bootstrap seed",
        ),
        minimum_practical_mean_delta=_require_float(
            analysis["minimum_practical_mean_delta"],
            "minimum practical delta",
        ),
        execution_enabled=_require_bool(
            admission["execution_enabled"],
            "execution enabled",
        ),
        formal_run_authorized=_require_bool(
            admission["formal_run_authorized"],
            "formal run authorized",
        ),
        model_output_count_before_freeze=_require_int(
            admission["model_output_count_before_freeze"],
            "model output count before freeze",
        ),
        subject_pack_materialization_count_before_freeze=_require_int(
            admission["subject_pack_materialization_count_before_freeze"],
            "subject materialization count before freeze",
        ),
        claim_boundary=_require_text(
            raw["claim_boundary"],
            "P4.7 claim boundary",
        ),
    )


def _parse_v2_protocol(
    raw: Mapping[str, Any],
    descriptor: _ProtocolDescriptor,
) -> RelationshipP4LongContextScientificPrereg:
    _validate_frozen_current_sections(raw, descriptor)
    cohort = _require_mapping(raw["cohort"], "P4.7 v2 cohort")
    splits = _require_mapping(cohort["splits"], "P4.7 v2 cohort splits")
    _require_exact_keys(
        splits,
        {"development", "qualification", "formal"},
        "P4.7 v2 cohort splits",
    )
    development = _require_mapping(splits["development"], "development split")
    qualification = _require_mapping(
        splits["qualification"],
        "qualification split",
    )
    formal = _require_mapping(splits["formal"], "formal split")
    longitudinal = _require_mapping(
        raw["longitudinal_design"],
        "P4.7 v2 longitudinal design",
    )
    analysis = _require_mapping(raw["analysis"], "P4.7 v2 analysis")
    decision_rule = _require_mapping(
        analysis["decision_rule"],
        "P4.7 v2 decision rule",
    )
    admission = _require_mapping(
        raw["execution_admission"],
        "P4.7 v2 execution admission",
    )
    contrasts = _parse_axis_contrasts_v2(_require_mapping(raw["axis_contrasts"], "P4.7 v2 axis contrasts"))
    arm_matrix = _require_text_tuple(raw["arm_matrix"], "P4.7 v2 arm matrix")
    integrated_arms = _require_text_tuple(
        raw["integrated_arms"],
        "P4.7 v2 integrated arms",
    )
    return RelationshipP4LongContextScientificPrereg(
        protocol_id=descriptor.protocol_id,
        schema_version=descriptor.protocol_schema,
        superseded=descriptor.superseded_by is not None,
        frozen_at_utc=_require_timestamp(
            raw["frozen_at_utc"],
            "P4.7 v2 frozen_at_utc",
        ),
        development_subject_count=_require_int(
            development["subject_count"],
            "development subject count",
        ),
        qualification_subject_count=_require_int(
            qualification["subject_count"],
            "qualification subject count",
        ),
        formal_subject_count=_require_int(
            formal["preallocated_subject_count"],
            "formal subject count",
        ),
        minimum_complete_paired_subjects=_require_int(
            formal["minimum_globally_complete_paired_subjects"],
            "minimum globally complete subjects",
        ),
        onboarding_sessions_per_subject=_require_int(
            longitudinal["onboarding_sessions_per_subject"],
            "onboarding sessions",
        ),
        learning_sessions_per_subject=_require_int(
            longitudinal["matched_learning_exposure_sessions_per_subject"],
            "learning sessions",
        ),
        evaluation_sessions_per_subject=_require_int(
            longitudinal["frozen_policy_evaluation_sessions_per_subject"],
            "evaluation sessions",
        ),
        minimum_public_history_tokens=_require_int(
            longitudinal["final_decision_public_history_tokens_minimum"],
            "public history tokens",
        ),
        minimum_native_context_window_tokens=_require_int(
            longitudinal["minimum_native_context_window_tokens"],
            "native context tokens",
        ),
        minimum_generation_headroom_tokens=_require_int(
            longitudinal["minimum_generation_headroom_tokens"],
            "generation headroom tokens",
        ),
        arm_matrix=arm_matrix,
        integrated_arms=integrated_arms,
        axis_contrasts=contrasts,
        bootstrap_replicates=_require_int(
            decision_rule["bootstrap_replicates"],
            "bootstrap replicates",
        ),
        bootstrap_seed=_require_int(
            decision_rule["bootstrap_seed"],
            "bootstrap seed",
        ),
        minimum_practical_mean_delta=_require_decimal_float(
            decision_rule["minimum_practical_point_estimate_delta_decimal"],
            "minimum practical delta",
        ),
        execution_enabled=_require_bool(
            admission["execution_enabled"],
            "execution enabled",
        ),
        formal_run_authorized=_require_bool(
            admission["formal_run_authorized"],
            "formal run authorized",
        ),
        model_output_count_before_freeze=_require_int(
            admission["model_output_count_before_freeze"],
            "model output count before freeze",
        ),
        subject_pack_materialization_count_before_freeze=_require_int(
            admission["subject_pack_materialization_count_before_freeze"],
            "subject materialization count before freeze",
        ),
        claim_boundary=_require_text(
            raw["claim_boundary"],
            "P4.7 v2 claim boundary",
        ),
    )


def _validate_v2_frozen_sections(raw: Mapping[str, Any]) -> None:
    """Freeze every historical v2 section independently of its global ID."""

    _validate_frozen_sections(
        raw,
        expected_keys=_TOP_LEVEL_KEYS_V2,
        expected_sha256=_V2_FROZEN_SECTION_SHA256,
        label="P4.7 v2",
    )


def _validate_v3_frozen_sections(raw: Mapping[str, Any]) -> None:
    """Freeze every current v3 section independently of its global ID."""

    _validate_frozen_sections(
        raw,
        expected_keys=_TOP_LEVEL_KEYS_V3,
        expected_sha256=_V3_FROZEN_SECTION_SHA256,
        label="P4.7 v3",
    )


def _validate_frozen_current_sections(
    raw: Mapping[str, Any],
    descriptor: _ProtocolDescriptor,
) -> None:
    if descriptor.version == "v2":
        _validate_v2_frozen_sections(raw)
        return
    if descriptor.version == "v3":
        _validate_v3_frozen_sections(raw)
        return
    raise AssertionError(f"unexpected current protocol descriptor: {descriptor.version}")


def _validate_frozen_sections(
    raw: Mapping[str, Any],
    *,
    expected_keys: frozenset[str],
    expected_sha256: Mapping[str, str],
    label: str,
) -> None:
    """Validate a complete independently anchored protocol section inventory."""

    _require_exact_keys(raw, expected_keys, f"{label} protocol")
    _require_exact_keys(
        expected_sha256,
        expected_keys,
        f"{label} section anchor inventory",
    )
    for key in sorted(expected_keys):
        actual = _sha256_bytes(_canonical_bytes(raw[key]))
        if actual != expected_sha256[key]:
            raise ValueError(f"{label} frozen section drift: {key}")


def _parse_axis_contrasts_v2(
    payload: Mapping[str, Any],
) -> tuple[P4LongContextAxisContrast, ...]:
    _require_exact_keys(
        payload,
        set(_AXIS_CONTROLS_V2),
        "P4.7 v2 axis contrasts",
    )
    result: list[P4LongContextAxisContrast] = []
    for axis, expected_controls in _AXIS_CONTROLS_V2.items():
        value = _require_mapping(payload[axis], f"P4.7 v2 {axis} contrast")
        controls = _require_text_tuple(
            value["control_arms"],
            f"P4.7 v2 {axis} controls",
        )
        if value["reference_arm"] != "volvence_closed_loop":
            raise ValueError(f"P4.7 v2 {axis} reference arm drift")
        if controls != expected_controls:
            raise ValueError(f"P4.7 v2 {axis} controls drift")
        result.append(
            P4LongContextAxisContrast(
                axis=axis,
                reference_arm="volvence_closed_loop",
                control_arms=controls,
                primary_effect=_require_text(
                    value["primary_effect"],
                    f"P4.7 v2 {axis} primary effect",
                ),
            )
        )
    return tuple(result)


def prepare_relationship_p4_long_context_scientific_prereg(
    *,
    output_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextPreparation:
    """Publish a deterministic create-only zero-output preparation root."""

    protocol = load_relationship_p4_long_context_scientific_prereg(protocol_path)
    descriptor = _protocol_descriptor(protocol.protocol_id)
    if not descriptor.preparation_allowed:
        raise ValueError(f"P4.7 superseded protocol cannot publish preparation: {protocol.protocol_id}")
    output = _absolute_without_resolving(output_dir)
    _reject_reparse_components(output, "P4.7 requested output")
    if os.path.lexists(output):
        raise FileExistsError(f"P4.7 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False
    try:
        preparation_payload = _preparation_payload(protocol)
        preparation_bytes = _canonical_bytes(preparation_payload)
        _write_create_bytes(temporary / _PREPARATION_FILE, preparation_bytes)
        manifest_core = _manifest_core(protocol, preparation_bytes)
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        result = _validate_preparation_root(temporary, protocol)
        if os.path.lexists(output):
            raise FileExistsError("P4.7 output appeared during create-only publication")
        temporary.rename(output)
        published = True
        return RelationshipP4LongContextPreparation(
            artifact_id=result.artifact_id,
            protocol_id=result.protocol_id,
            status=result.status,
            execution_enabled=result.execution_enabled,
            formal_run_authorized=result.formal_run_authorized,
            output_dir=output,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_long_context_scientific_prereg(
    *,
    output_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextPreparation:
    """Validate a preparation without importing a model, torch, or CUDA."""

    source = _absolute_without_resolving(output_dir)
    _reject_reparse_components(source, "P4.7 artifact root")
    if protocol_path is None:
        artifact_protocol_id = _artifact_protocol_id(source)
        protocol = load_relationship_p4_long_context_scientific_prereg(
            _protocol_descriptor(artifact_protocol_id).protocol_path
        )
    else:
        protocol = load_relationship_p4_long_context_scientific_prereg(protocol_path)
    return _validate_preparation_root(source, protocol)


def prepare_relationship_p4_long_context_power_failure_certificate(
    *,
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextPowerFailureCertificate:
    """Reject republication of the preserved, superseded v1 interpretation."""

    raise ValueError("P4.7 v1 power-failure interpretation is superseded and cannot be republished")


def validate_relationship_p4_long_context_power_failure_certificate(
    *,
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextPowerFailureCertificate:
    """Validate a published exact failure without model, source, or CUDA."""

    protocol, preparation = _validated_v3_power_failure_inputs(
        preparation_dir=preparation_dir,
        protocol_path=protocol_path,
    )
    source = _absolute_without_resolving(output_dir)
    _reject_reparse_components(source, "P4.7 power failure artifact root")
    return _validate_power_failure_root(
        source,
        protocol=protocol,
        preparation=preparation,
    )


def prepare_relationship_p4_long_context_power_admission_certificate(
    *,
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextPowerAdmissionCertificate:
    """Publish the current zero-output v3 under-specification finding."""

    protocol, preparation = _validated_v3_power_failure_inputs(
        preparation_dir=preparation_dir,
        protocol_path=protocol_path,
    )
    output = _absolute_without_resolving(output_dir)
    _reject_reparse_components(output, "P4.7 power admission requested output")
    if os.path.lexists(output):
        raise FileExistsError(f"P4.7 power admission output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False
    try:
        certificate_core = _power_admission_certificate_core(protocol, preparation)
        certificate_id = _sha256_bytes(_canonical_bytes(certificate_core))
        certificate_bytes = _canonical_bytes({**certificate_core, "certificate_id": certificate_id})
        _write_create_bytes(
            temporary / _POWER_ADMISSION_CERTIFICATE_FILE,
            certificate_bytes,
        )
        manifest_core = _power_admission_manifest_core(
            protocol=protocol,
            preparation=preparation,
            certificate_id=certificate_id,
            certificate_bytes=certificate_bytes,
        )
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        result = _validate_power_admission_root(
            temporary,
            protocol=protocol,
            preparation=preparation,
        )
        if os.path.lexists(output):
            raise FileExistsError("P4.7 power admission output appeared during create-only publication")
        temporary.rename(output)
        published = True
        return RelationshipP4LongContextPowerAdmissionCertificate(
            artifact_id=result.artifact_id,
            admission_protocol_id=result.admission_protocol_id,
            scientific_protocol_id=result.scientific_protocol_id,
            preparation_artifact_id=result.preparation_artifact_id,
            status=result.status,
            conditional_bound_numerator=result.conditional_bound_numerator,
            conditional_bound_denominator=result.conditional_bound_denominator,
            conditional_bound_display_decimal=(result.conditional_bound_display_decimal),
            power_contract_determinate=result.power_contract_determinate,
            v1_unconditional_scientific_admission_valid=(result.v1_unconditional_scientific_admission_valid),
            development_authorized=result.development_authorized,
            formal_authorized=result.formal_authorized,
            output_dir=output,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_long_context_power_admission_certificate(
    *,
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextPowerAdmissionCertificate:
    """Validate the current under-specification artifact without execution."""

    protocol, preparation = _validated_v3_power_failure_inputs(
        preparation_dir=preparation_dir,
        protocol_path=protocol_path,
    )
    source = _absolute_without_resolving(output_dir)
    _reject_reparse_components(source, "P4.7 power admission artifact root")
    return _validate_power_admission_root(
        source,
        protocol=protocol,
        preparation=preparation,
    )


def prepare_relationship_p4_long_context_v4_zero_output_plan(
    *,
    output_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextV4PlanningFreeze:
    """Publish v4a primitives and an abstract six-cell schedule, with no execution."""

    protocol, preparation, admission, raw, derived = _validated_v4_planning_inputs(
        v3_preparation_dir=v3_preparation_dir,
        v2_admission_dir=v2_admission_dir,
        protocol_path=protocol_path,
    )
    output = _absolute_without_resolving(output_dir)
    _reject_reparse_components(output, "P4.7 v4 planning requested output")
    if os.path.lexists(output):
        raise FileExistsError(f"P4.7 v4 planning output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False
    try:
        schedule_payload = _v4_candidate_schedule_payload(protocol, raw, derived)
        schedule_bytes = _canonical_bytes(schedule_payload)
        _write_create_bytes(temporary / _V4_CANDIDATE_SCHEDULE_FILE, schedule_bytes)
        screen_table_payload = _v4_sentinel_screen_table_payload(protocol, raw, derived)
        screen_table_bytes = _canonical_bytes(screen_table_payload)
        _write_create_bytes(temporary / _V4_SENTINEL_SCREEN_TABLE_FILE, screen_table_bytes)
        plan_core = _v4_planning_freeze_core(
            protocol=protocol,
            preparation=preparation,
            admission=admission,
            raw=raw,
            derived=derived,
            schedule_bytes=schedule_bytes,
            screen_table_bytes=screen_table_bytes,
        )
        certificate_id = _sha256_bytes(_canonical_bytes(plan_core))
        plan_bytes = _canonical_bytes({**plan_core, "certificate_id": certificate_id})
        _write_create_bytes(temporary / _V4_PLANNING_FREEZE_FILE, plan_bytes)
        manifest_core = _v4_planning_manifest_core(
            protocol=protocol,
            preparation=preparation,
            admission=admission,
            certificate_id=certificate_id,
            plan_bytes=plan_bytes,
            schedule_bytes=schedule_bytes,
            screen_table_bytes=screen_table_bytes,
        )
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        result = _validate_v4_planning_root(
            temporary,
            protocol=protocol,
            preparation=preparation,
            admission=admission,
            raw=raw,
            derived=derived,
        )
        if os.path.lexists(output):
            raise FileExistsError("P4.7 v4 planning output appeared during create-only publication")
        temporary.rename(output)
        published = True
        return RelationshipP4LongContextV4PlanningFreeze(
            artifact_id=result.artifact_id,
            protocol_id=result.protocol_id,
            scientific_v3_protocol_id=result.scientific_v3_protocol_id,
            power_admission_v2_artifact_id=result.power_admission_v2_artifact_id,
            status=result.status,
            first_necessary_screen_passing_root_count=(result.first_necessary_screen_passing_root_count),
            first_positive_mean_gate_capable_root_count=(result.first_positive_mean_gate_capable_root_count),
            cartesian_candidate_tuple_count=result.cartesian_candidate_tuple_count,
            candidate_schedule_block_count=result.candidate_schedule_block_count,
            power_contract_determinate=result.power_contract_determinate,
            source_grid_resolved=result.source_grid_resolved,
            selected_formal_root_count=result.selected_formal_root_count,
            source_materialization_authorized=result.source_materialization_authorized,
            development_authorized=result.development_authorized,
            model_output_authorized=result.model_output_authorized,
            formal_authorized=result.formal_authorized,
            output_dir=output,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_long_context_v4_zero_output_plan(
    *,
    output_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextV4PlanningFreeze:
    """Rebuild and validate a v4a plan without source, model, CUDA, or power runs."""

    protocol, preparation, admission, raw, derived = _validated_v4_planning_inputs(
        v3_preparation_dir=v3_preparation_dir,
        v2_admission_dir=v2_admission_dir,
        protocol_path=protocol_path,
    )
    source = _absolute_without_resolving(output_dir)
    _reject_reparse_components(source, "P4.7 v4 planning artifact root")
    return _validate_v4_planning_root(
        source,
        protocol=protocol,
        preparation=preparation,
        admission=admission,
        raw=raw,
        derived=derived,
    )


def prepare_relationship_p4_long_context_source_opportunity_preflight(
    *,
    output_dir: pathlib.Path,
    v4a_planning_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextSourceOpportunityPreflightCertificate:
    """Publish the content-addressed preflight contract with no source output."""

    protocol, raw, derived, upstream = _validated_v4_source_opportunity_inputs(
        v4a_planning_dir=v4a_planning_dir,
        v3_preparation_dir=v3_preparation_dir,
        v2_admission_dir=v2_admission_dir,
        protocol_path=protocol_path,
    )
    output = _absolute_without_resolving(output_dir)
    _reject_reparse_components(output, "P4.7 source preflight requested output")
    if os.path.lexists(output):
        raise FileExistsError(f"P4.7 source preflight output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False
    try:
        projection_core = _v4_source_opportunity_contract_projection_core(
            protocol=protocol,
            raw=raw,
            derived=derived,
            upstream=upstream,
        )
        projection_id = _sha256_bytes(_canonical_bytes(projection_core))
        projection_bytes = _canonical_bytes({**projection_core, "contract_projection_id": projection_id})
        _write_create_bytes(
            temporary / _V4_SOURCE_OPPORTUNITY_CONTRACT_FILE,
            projection_bytes,
        )
        certificate_core = _v4_source_opportunity_preflight_certificate_core(
            protocol=protocol,
            raw=raw,
            derived=derived,
            upstream=upstream,
            projection_id=projection_id,
            projection_bytes=projection_bytes,
        )
        certificate_id = _sha256_bytes(_canonical_bytes(certificate_core))
        certificate_bytes = _canonical_bytes({**certificate_core, "certificate_id": certificate_id})
        _write_create_bytes(
            temporary / _V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE,
            certificate_bytes,
        )
        manifest_core = _v4_source_opportunity_preflight_manifest_core(
            protocol=protocol,
            raw=raw,
            projection_id=projection_id,
            projection_bytes=projection_bytes,
            certificate_id=certificate_id,
            certificate_bytes=certificate_bytes,
        )
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        result = _validate_v4_source_opportunity_preflight_root(
            temporary,
            protocol=protocol,
            raw=raw,
            derived=derived,
            upstream=upstream,
        )
        if os.path.lexists(output):
            raise FileExistsError("P4.7 source preflight output appeared during create-only publication")
        temporary.rename(output)
        published = True
        return replace(result, output_dir=output)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_long_context_source_opportunity_preflight(
    *,
    output_dir: pathlib.Path,
    v4a_planning_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextSourceOpportunityPreflightCertificate:
    """Validate an existing preflight without source, model, power, or CUDA."""

    protocol, raw, derived, upstream = _validated_v4_source_opportunity_inputs(
        v4a_planning_dir=v4a_planning_dir,
        v3_preparation_dir=v3_preparation_dir,
        v2_admission_dir=v2_admission_dir,
        protocol_path=protocol_path,
    )
    source = _absolute_without_resolving(output_dir)
    _reject_reparse_components(source, "P4.7 source preflight artifact root")
    return _validate_v4_source_opportunity_preflight_root(
        source,
        protocol=protocol,
        raw=raw,
        derived=derived,
        upstream=upstream,
    )


def prepare_relationship_p4_long_context_external_anchor_request(
    *,
    output_dir: pathlib.Path,
    source_preflight_dir: pathlib.Path,
    v4a_planning_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextExternalAnchorRequest:
    """Materialize the canonical local A0 request without network, Git, source, or CUDA."""

    protocol_source = _require_local_default_stream_path(
        _V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_PATH_V1 if protocol_path is None else protocol_path,
        "P4.7 A0 request protocol",
    )
    raw_before_inputs = _load_v4_external_anchor_request_protocol_raw(protocol_source)
    output = _require_local_default_stream_path(output_dir, "P4.7 A0 request output")
    expected_output = _v4_external_anchor_request_canonical_output(raw_before_inputs)
    if output != expected_output:
        raise ValueError(
            f"P4.7 A0 request preparation must target its frozen canonical repository path: {expected_output}"
        )
    source_preflight_source = _require_local_default_stream_path(
        source_preflight_dir,
        "P4.7 A0 source-preflight input",
    )
    _require_v4_external_anchor_canonical_subject_sources(
        raw=raw_before_inputs,
        source_preflight_dir=source_preflight_source,
    )
    v4a_source = _require_local_default_stream_path(v4a_planning_dir, "P4.7 A0 v4a input")
    v3_source = _require_local_default_stream_path(v3_preparation_dir, "P4.7 A0 v3 input")
    v2_source = _require_local_default_stream_path(v2_admission_dir, "P4.7 A0 v2 input")
    expected_upstream_roots = _v4_external_anchor_canonical_upstream_roots(raw_before_inputs)
    actual_upstream_roots = {
        "source_preflight": source_preflight_source,
        "v4a_planning": v4a_source,
        "v3_preparation": v3_source,
        "v2_admission": v2_source,
    }
    for field, actual_root in actual_upstream_roots.items():
        if actual_root != expected_upstream_roots[field]:
            raise ValueError(
                "P4.7 A0 request preparation must read every upstream artifact from its "
                f"frozen canonical repository root: {expected_upstream_roots[field]}"
            )
    protocol, raw, source_certificate, subjects = _validated_v4_external_anchor_request_inputs(
        source_preflight_dir=source_preflight_source,
        v4a_planning_dir=v4a_source,
        v3_preparation_dir=v3_source,
        v2_admission_dir=v2_source,
        protocol_path=protocol_source,
    )
    _reject_reparse_components(output, "P4.7 A0 request output")
    if os.path.lexists(output):
        raise FileExistsError(f"P4.7 A0 request output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    published = False
    try:
        request_core = _v4_external_anchor_request_core(
            protocol=protocol,
            raw=raw,
            source_certificate=source_certificate,
            subjects=subjects,
        )
        request_id = _sha256_bytes(_canonical_bytes(request_core))
        request_bytes = _canonical_bytes({**request_core, "request_id": request_id})
        _write_create_bytes(temporary / _V4_EXTERNAL_ANCHOR_REQUEST_FILE, request_bytes)
        manifest_core = _v4_external_anchor_request_manifest_core(
            protocol=protocol,
            request_id=request_id,
            request_bytes=request_bytes,
        )
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        result = _validate_v4_external_anchor_request_root(
            temporary,
            protocol=protocol,
            raw=raw,
            source_certificate=source_certificate,
            subjects=subjects,
        )
        if os.path.lexists(output):
            raise FileExistsError("P4.7 A0 request output appeared during create-only publication")
        temporary.rename(output)
        published = True
        return replace(result, output_dir=output)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_long_context_external_anchor_request(
    *,
    output_dir: pathlib.Path,
    source_preflight_dir: pathlib.Path,
    v4a_planning_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongContextExternalAnchorRequest:
    """Replay an A0 request artifact without external publication or execution."""

    protocol_source = _require_local_default_stream_path(
        _V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_PATH_V1 if protocol_path is None else protocol_path,
        "P4.7 A0 request protocol",
    )
    source = _require_local_default_stream_path(output_dir, "P4.7 A0 request artifact root")
    source_preflight_source = _require_local_default_stream_path(
        source_preflight_dir,
        "P4.7 A0 source-preflight input",
    )
    v4a_source = _require_local_default_stream_path(v4a_planning_dir, "P4.7 A0 v4a input")
    v3_source = _require_local_default_stream_path(v3_preparation_dir, "P4.7 A0 v3 input")
    v2_source = _require_local_default_stream_path(v2_admission_dir, "P4.7 A0 v2 input")
    protocol, raw, source_certificate, subjects = _validated_v4_external_anchor_request_inputs(
        source_preflight_dir=source_preflight_source,
        v4a_planning_dir=v4a_source,
        v3_preparation_dir=v3_source,
        v2_admission_dir=v2_source,
        protocol_path=protocol_source,
    )
    _reject_reparse_components(source, "P4.7 A0 request artifact root")
    return _validate_v4_external_anchor_request_root(
        source,
        protocol=protocol,
        raw=raw,
        source_certificate=source_certificate,
        subjects=subjects,
    )


def _validated_v3_power_failure_inputs(
    *,
    preparation_dir: pathlib.Path,
    protocol_path: pathlib.Path | None,
) -> tuple[
    RelationshipP4LongContextScientificPrereg,
    RelationshipP4LongContextPreparation,
]:
    protocol = load_relationship_p4_long_context_scientific_prereg(protocol_path)
    if protocol.protocol_id != P4_LONG_CONTEXT_PROTOCOL_ID_V3:
        raise ValueError("P4.7 necessary power failure is defined only for v3")
    preparation = validate_relationship_p4_long_context_scientific_prereg(
        output_dir=preparation_dir,
        protocol_path=relationship_p4_long_context_protocol_path(P4_LONG_CONTEXT_PROTOCOL_ID_V3),
    )
    return protocol, preparation


def _power_failure_certificate_core(
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation: RelationshipP4LongContextPreparation,
) -> dict[str, object]:
    descriptor = _protocol_descriptor(protocol.protocol_id)
    if descriptor.version != "v3":
        raise ValueError("P4.7 necessary power failure must consume v3")
    power_bound_protocol = _load_power_bound_protocol_v1()
    power_bound_lineage = _require_mapping(
        power_bound_protocol["input_lineage"],
        "P4.7 power-bound input lineage",
    )
    _require_literal(
        power_bound_lineage,
        {
            "scientific_protocol_id": P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            "scientific_protocol_raw_sha256": descriptor.bundled_raw_sha256,
            "scientific_preparation_artifact_id": preparation.artifact_id,
            "scientific_preparation_raw_sha256": ("a4b2f3ee920e398ae0f7eab5757b7988dc3fa4f7db7b4769599c837bc656bcd6"),
            "scientific_preparation_manifest_raw_sha256": (
                "4aaf10d76b80a780e62a17b62803906cc34dd34d64bb4d0f8b96d60dff1e2663"
            ),
        },
        "P4.7 power-bound input lineage",
    )
    if (
        _sha256_bytes((preparation.output_dir / _PREPARATION_FILE).read_bytes())
        != power_bound_lineage["scientific_preparation_raw_sha256"]
    ):
        raise ValueError("P4.7 power-bound preparation raw lineage drift")
    if (
        _sha256_bytes((preparation.output_dir / _MANIFEST_FILE).read_bytes())
        != power_bound_lineage["scientific_preparation_manifest_raw_sha256"]
    ):
        raise ValueError("P4.7 power-bound preparation manifest raw lineage drift")
    raw = _load_json_object(descriptor.protocol_path)
    _validate_v3_frozen_sections(raw)
    analysis = _require_mapping(raw["analysis"], "P4.7 v3 analysis")
    power = _require_mapping(analysis["power"], "P4.7 v3 power")
    decision_rule = _require_mapping(
        analysis["decision_rule"],
        "P4.7 v3 decision rule",
    )
    causal_execution = _require_mapping(
        raw["causal_execution"],
        "P4.7 v3 causal execution",
    )
    endpoint = _require_mapping(
        causal_execution["typed_outcome_endpoint_contract"],
        "P4.7 v3 typed endpoint",
    )
    _require_literal(
        endpoint["utility_closed_integer_domain"],
        [-1, 0, 1],
        "P4.7 v3 utility domain",
    )
    _require_literal(
        power["mandatory_variance_scenarios"],
        [
            "source_structural_covariance_upper_bound",
            "maximum_feasible_bounded_difference_variance_at_planning_mean",
            "paired_root_difference_variance_0_25",
            "paired_root_difference_variance_0_50",
            "paired_root_difference_variance_1_00",
        ],
        "P4.7 v3 mandatory variance scenarios",
    )
    if power["planning_alternative_mean_delta_for_every_contrast_decimal"] != "0.20":
        raise ValueError("P4.7 v3 planning mean drift")
    if power["full_decision_rule_power_target_decimal"] != "0.80":
        raise ValueError("P4.7 v3 power target drift")
    if decision_rule["minimum_practical_point_estimate_delta_decimal"] != "0.15":
        raise ValueError("P4.7 v3 practical threshold drift")

    lower = Fraction(-2, 1)
    upper = Fraction(2, 1)
    planning_mean = Fraction(1, 5)
    practical_threshold = Fraction(3, 20)
    required_power = Fraction(4, 5)
    mass_at_upper = (planning_mean - lower) / (upper - lower)
    mass_at_lower = Fraction(1, 1) - mass_at_upper
    maximum_variance = (upper - planning_mean) * (planning_mean - lower)
    minimum_upper_count = _minimum_upper_count_for_point_gate(
        formal_root_count=protocol.formal_subject_count,
        lower=lower,
        upper=upper,
        practical_threshold=practical_threshold,
    )
    exact_tail = _maximum_variance_point_gate_probability(
        formal_root_count=protocol.formal_subject_count,
        mass_at_upper=mass_at_upper,
        minimum_upper_count=minimum_upper_count,
    )
    if not exact_tail < required_power:
        raise AssertionError("P4.7 v3 exact necessary-condition failure disappeared")
    comparator_arms = tuple(arm for arm in protocol.arm_matrix if arm != "volvence_closed_loop")
    if len(comparator_arms) != 8:
        raise ValueError("P4.7 v3 comparator arm count drift")
    claim_boundary = (
        "This content-addressed zero-model-output certificate proves only that "
        "P4.7 v3 with 192 formal roots fails one mandatory exact necessary "
        "power condition. The full contrast rule is a subset of the practical "
        "point-gate event, whose exact maximum-variance power is below 0.80. "
        "It does not complete the v3 joint DGP or source preflight, estimate "
        "actual claim power, run a model or CUDA formal workload, or provide "
        "Appendable, Readable, Learnable, Steerable, integrated, human, "
        "product, or production ACTIVE evidence. V3 remains closed to every "
        "development, qualification, and formal model output."
    )
    return {
        "schema_version": P4_LONG_CONTEXT_POWER_FAILURE_SCHEMA_VERSION,
        "certificate_id_contract": ("sha256_canonical_json_utf8_newline_without_certificate_id_v1"),
        "identity": {
            "power_bound_protocol_id": P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1,
            "power_bound_protocol_raw_sha256": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1),
            "scientific_protocol_id": protocol.protocol_id,
            "scientific_protocol_raw_sha256": descriptor.bundled_raw_sha256,
            "scientific_preparation_artifact_id": preparation.artifact_id,
            "owner": ("lifeform_evolution.relationship_lab_p4_long_context_causal_campaign"),
            "calculation_contract": ("exact_integer_binomial_tail_and_rational_cross_multiplication_v1"),
        },
        "zero_output_attestation": {
            "model_output_count": 0,
            "subject_materialization_count": 0,
            "donor_bank_materialization_count": 0,
            "counterfactual_twin_materialization_count": 0,
            "source_materialization_count": 0,
            "cuda_formal_run_count": 0,
            "outer_power_simulation_count": 0,
            "inner_bootstrap_simulation_count": 0,
            "full_joint_dgp_artifact_count": 0,
        },
        "mandatory_scenario": {
            "scenario_id": ("maximum_feasible_bounded_difference_variance_at_planning_mean"),
            "scenario_scope": "mandatory_marginal_joint_sentinel_before_cartesian_filtering",
            "formal_root_count": protocol.formal_subject_count,
            "evaluation_decisions_per_root_arm": protocol.evaluation_sessions_per_subject,
            "typed_utility_domain": [-1, 0, 1],
            "root_difference_support": [-2, 2],
            "planning_mean": _fraction_payload(planning_mean),
            "practical_threshold": _fraction_payload(practical_threshold),
            "required_power": _fraction_payload(required_power),
            "technical_missingness": _fraction_payload(Fraction(0, 1)),
        },
        "maximum_variance_proof": {
            "method": "bhatia_davis_bounded_mean_variance_equality",
            "mass_at_plus_two": _fraction_payload(mass_at_upper),
            "mass_at_minus_two": _fraction_payload(mass_at_lower),
            "maximum_variance": _fraction_payload(maximum_variance),
            "mean_recomputed": _fraction_payload(mass_at_upper * upper + mass_at_lower * lower),
            "variance_recomputed": _fraction_payload(
                mass_at_upper * (upper - planning_mean) ** 2 + mass_at_lower * (lower - planning_mean) ** 2
            ),
            "equality_attained": True,
        },
        "nine_arm_eight_decision_joint_feasibility_witness": {
            "root_latent": "iid_bernoulli_11_over_20_across_analysis_roots",
            "reference_arm": "volvence_closed_loop",
            "comparator_arms": list(comparator_arms),
            "upper_latent_state": {
                "probability": _fraction_payload(mass_at_upper),
                "reference_utility_at_each_of_8_decisions": 1,
                "every_comparator_utility_at_each_of_8_decisions": -1,
                "every_registered_contrast_root_difference": 2,
            },
            "lower_latent_state": {
                "probability": _fraction_payload(mass_at_lower),
                "reference_utility_at_each_of_8_decisions": -1,
                "every_comparator_utility_at_each_of_8_decisions": 1,
                "every_registered_contrast_root_difference": -2,
            },
            "registered_contrast_count": 8,
            "all_nine_arm_utilities_are_in_frozen_domain": True,
            "all_eight_contrasts_have_planning_mean_and_maximum_variance": True,
            "analysis_roots_are_independent": True,
            "derived_within_root_temporal_correlation": "1.0",
            "derived_cross_contrast_correlation": "1.0",
            "mandatory_sentinel_is_constructively_feasible": True,
            "cartesian_filter_may_remove_mandatory_sentinel": False,
            "rejection_of_sentinel_requires_new_protocol_not_silent_infeasibility": True,
            "witness_is_not_a_source_structural_claim": True,
        },
        "exact_point_gate_enumeration": {
            "success_count_variable": "K",
            "success_count_distribution": "binomial_192_11_over_20",
            "minimum_success_count": minimum_upper_count,
            "point_gate_equivalence": "sample_mean_delta_ge_3_over_20_iff_K_ge_104",
            "exact_tail_expression": (
                "sum_k_104_to_192_comb_192_k_times_11_pow_k_times_9_pow_192_minus_k_over_20_pow_192"
            ),
            "exact_tail_probability": _fraction_payload(exact_tail),
            "display_decimal_half_even_20_places": _fraction_display_decimal(
                exact_tail,
                places=20,
            ),
            "exact_target_comparison": "5_times_numerator_lt_4_times_denominator",
            "exact_target_comparison_passed": (5 * exact_tail.numerator < 4 * exact_tail.denominator),
        },
        "logical_upper_bound": {
            "full_contrast_pass_implies_point_gate_pass": True,
            "axis_pass_implies_all_registered_contrasts_pass": True,
            "integrated_pass_implies_all_eight_contrasts_pass": True,
            "full_rule_power_upper_bound": _fraction_payload(exact_tail),
            "required_power": _fraction_payload(required_power),
            "full_rule_power_upper_bound_below_required": True,
            "bootstrap_ci_holm_lineage_or_missingness_can_increase_upper_bound": False,
            "necessary_condition_short_circuit_is_decisive": True,
        },
        "terminal": {
            "verdict": P4_LONG_CONTEXT_POWER_FAILURE_STATUS,
            "decisive_failure": True,
            "full_joint_grid_completed": False,
            "actual_contrast_power_estimated": False,
            "source_preflight_completed": False,
            "development_authorized": False,
            "qualification_authorized": False,
            "formal_authorized": False,
            "sample_size_may_change_within_v3": False,
            "next_required_action": ("freeze_new_zero_output_protocol_before_any_source_or_model_output"),
            "unresolved_before_v4": [
                "malformed_generated_action_vs_integrity_failure_classification",
                "six_development_candidate_cell_counterbalance_and_state_isolation",
                "full_joint_power_planner_for_new_sample_size",
            ],
        },
        "claim_boundary": claim_boundary,
    }


def _power_failure_manifest_core(
    *,
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation: RelationshipP4LongContextPreparation,
    certificate_id: str,
    certificate_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": P4_LONG_CONTEXT_POWER_FAILURE_MANIFEST_SCHEMA_VERSION,
        "power_bound_protocol_id": P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1,
        "power_bound_protocol_raw_sha256": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1),
        "protocol_id": protocol.protocol_id,
        "preparation_artifact_id": preparation.artifact_id,
        "certificate_id": certificate_id,
        "status": P4_LONG_CONTEXT_POWER_FAILURE_STATUS,
        "files": [
            {
                "path": _POWER_FAILURE_CERTIFICATE_FILE,
                "byte_count": len(certificate_bytes),
                "sha256": _sha256_bytes(certificate_bytes),
            }
        ],
        "decisive_failure": True,
        "power_fail_certificate_count": 1,
        "power_dgp_artifact_count": 0,
        "full_joint_grid_completed": False,
        "execution_enabled": False,
        "development_authorized": False,
        "qualification_authorized": False,
        "formal_authorized": False,
        "model_output_count": 0,
        "subject_materialization_count": 0,
        "source_materialization_count": 0,
        "donor_bank_materialization_count": 0,
        "counterfactual_twin_materialization_count": 0,
        "cuda_formal_run_count": 0,
        "simulation_replicate_count": 0,
        "claim_boundary": (
            "Exact necessary-condition FAIL for P4.7 v3 N=192; not a full joint DGP, "
            "source preflight, empirical result, execution authorization, or four-axis claim."
        ),
    }


def _validate_power_failure_root(
    output: pathlib.Path,
    *,
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation: RelationshipP4LongContextPreparation,
) -> RelationshipP4LongContextPowerFailureCertificate:
    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 power failure root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    expected_entries = tuple(sorted((_POWER_FAILURE_CERTIFICATE_FILE, _MANIFEST_FILE)))
    if entries != expected_entries:
        raise ValueError("P4.7 power failure file set drift")
    for name in entries:
        candidate = output / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("P4.7 power failure files must be regular files")

    certificate_path = output / _POWER_FAILURE_CERTIFICATE_FILE
    certificate_bytes = certificate_path.read_bytes()
    certificate = _load_json_object(certificate_path)
    if certificate_bytes != _canonical_bytes(certificate):
        raise ValueError("P4.7 power failure certificate is not canonical JSON")
    certificate_id = _require_sha256(
        certificate.get("certificate_id"),
        "P4.7 power failure certificate id",
    )
    certificate_core = dict(certificate)
    del certificate_core["certificate_id"]
    if certificate_id != _sha256_bytes(_canonical_bytes(certificate_core)):
        raise ValueError("P4.7 power failure certificate id drift")
    expected_certificate_core = _power_failure_certificate_core(protocol, preparation)
    _require_literal(
        certificate_core,
        expected_certificate_core,
        "P4.7 power failure certificate",
    )

    manifest_path = output / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_path)
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("P4.7 power failure manifest is not canonical JSON")
    artifact_id = _require_sha256(
        manifest.get("artifact_id"),
        "P4.7 power failure artifact id",
    )
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4.7 power failure manifest artifact id drift")
    expected_manifest_core = _power_failure_manifest_core(
        protocol=protocol,
        preparation=preparation,
        certificate_id=certificate_id,
        certificate_bytes=certificate_bytes,
    )
    _require_literal(
        manifest_core,
        expected_manifest_core,
        "P4.7 power failure manifest",
    )
    if (
        P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID is not None
        and artifact_id != P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID
    ):
        raise ValueError("P4.7 published power failure artifact id drift")

    point_gate = _require_mapping(
        certificate["exact_point_gate_enumeration"],
        "P4.7 power failure point gate",
    )
    exact_tail = _require_mapping(
        point_gate["exact_tail_probability"],
        "P4.7 power failure exact tail",
    )
    terminal = _require_mapping(
        certificate["terminal"],
        "P4.7 power failure terminal",
    )
    return RelationshipP4LongContextPowerFailureCertificate(
        artifact_id=artifact_id,
        protocol_id=protocol.protocol_id,
        preparation_artifact_id=preparation.artifact_id,
        status=_require_text(terminal["verdict"], "P4.7 power failure verdict"),
        point_gate_power_numerator=int(_require_text(exact_tail["numerator"], "P4.7 power numerator")),
        point_gate_power_denominator=int(_require_text(exact_tail["denominator"], "P4.7 power denominator")),
        point_gate_power_display_decimal=_require_text(
            point_gate["display_decimal_half_even_20_places"],
            "P4.7 power display decimal",
        ),
        decisive_failure=_require_bool(
            terminal["decisive_failure"],
            "P4.7 decisive failure",
        ),
        full_joint_grid_completed=_require_bool(
            terminal["full_joint_grid_completed"],
            "P4.7 full joint grid completed",
        ),
        development_authorized=_require_bool(
            terminal["development_authorized"],
            "P4.7 development authorized",
        ),
        formal_authorized=_require_bool(
            terminal["formal_authorized"],
            "P4.7 formal authorized",
        ),
        output_dir=output,
    )


def _validate_power_admission_derivation(
    admission_protocol: Mapping[str, Any],
    scientific_raw: Mapping[str, Any],
) -> None:
    """Derive the v2 ambiguity from primitive utilities and frozen v3 literals."""

    primitive = _require_mapping(
        admission_protocol["primitive_joint_witness"],
        "P4.7 primitive joint witness",
    )
    plus_mass = _fraction_from_payload(
        primitive["root_latent_plus_probability"],
        "P4.7 plus-state mass",
    )
    minus_mass = _fraction_from_payload(
        primitive["root_latent_minus_probability"],
        "P4.7 minus-state mass",
    )
    if plus_mass + minus_mass != 1:
        raise ValueError("P4.7 primitive latent masses do not sum to one")
    if primitive["root_latents_iid"] is not True:
        raise ValueError("P4.7 primitive roots must be iid")
    decision_count = _require_int(
        primitive["evaluation_decisions_per_arm"],
        "P4.7 primitive decision count",
    )
    comparator_count = _require_int(
        primitive["comparator_arm_count"],
        "P4.7 primitive comparator count",
    )
    arm_matrix = _require_text_tuple(scientific_raw["arm_matrix"], "P4.7 v3 arms")
    if primitive["reference_arm"] != "volvence_closed_loop":
        raise ValueError("P4.7 primitive reference arm drift")
    if decision_count != 8 or comparator_count != len(arm_matrix) - 1:
        raise ValueError("P4.7 primitive 9-arm x 8-decision shape drift")
    endpoint = _require_mapping(
        _require_mapping(
            scientific_raw["causal_execution"],
            "P4.7 v3 causal execution",
        )["typed_outcome_endpoint_contract"],
        "P4.7 v3 typed endpoint",
    )
    utility_domain = tuple(endpoint["utility_closed_integer_domain"])
    state_inputs = (
        (
            plus_mass,
            _require_int(
                primitive["plus_state_reference_utility_each_decision"],
                "P4.7 plus reference utility",
            ),
            _require_int(
                primitive["plus_state_every_comparator_utility_each_decision"],
                "P4.7 plus comparator utility",
            ),
        ),
        (
            minus_mass,
            _require_int(
                primitive["minus_state_reference_utility_each_decision"],
                "P4.7 minus reference utility",
            ),
            _require_int(
                primitive["minus_state_every_comparator_utility_each_decision"],
                "P4.7 minus comparator utility",
            ),
        ),
    )
    state_contrasts: list[tuple[Fraction, tuple[tuple[int, ...], ...]]] = []
    for mass, reference_utility, comparator_utility in state_inputs:
        utilities = (
            (reference_utility,) * decision_count,
            *((comparator_utility,) * decision_count for _ in range(comparator_count)),
        )
        if len(utilities) != 9 or any(len(row) != 8 for row in utilities):
            raise ValueError("P4.7 primitive joint utility shape drift")
        if any(value not in utility_domain for row in utilities for value in row):
            raise ValueError("P4.7 primitive utility leaves frozen domain")
        contrasts = tuple(
            tuple(utilities[0][turn] - comparator[turn] for turn in range(decision_count))
            for comparator in utilities[1:]
        )
        state_contrasts.append((mass, contrasts))

    root_differences_by_contrast: list[tuple[Fraction, Fraction]] = []
    for contrast_index in range(comparator_count):
        plus_values = state_contrasts[0][1][contrast_index]
        minus_values = state_contrasts[1][1][contrast_index]
        plus_root_mean = Fraction(sum(plus_values), decision_count)
        minus_root_mean = Fraction(sum(minus_values), decision_count)
        root_differences_by_contrast.append((plus_root_mean, minus_root_mean))
    if len(set(root_differences_by_contrast)) != 1:
        raise ValueError("P4.7 primitive contrasts do not share the frozen witness")
    plus_difference, minus_difference = root_differences_by_contrast[0]
    mean = plus_mass * plus_difference + minus_mass * minus_difference
    variance = plus_mass * (plus_difference - mean) ** 2 + minus_mass * (minus_difference - mean) ** 2
    if variance <= 0:
        raise ValueError("P4.7 primitive witness variance must be positive")

    plus_temporal = state_contrasts[0][1][0]
    minus_temporal = state_contrasts[1][1][0]
    temporal_covariance = (
        plus_mass * plus_temporal[0] * plus_temporal[1] + minus_mass * minus_temporal[0] * minus_temporal[1] - mean**2
    )
    cross_covariance = (
        plus_mass * state_contrasts[0][1][0][0] * state_contrasts[0][1][1][0]
        + minus_mass * state_contrasts[1][1][0][0] * state_contrasts[1][1][1][0]
        - mean**2
    )
    temporal_correlation = temporal_covariance / variance
    cross_correlation = cross_covariance / variance
    expected_derived = {
        "typed_nine_arm_eight_decision_joint_feasible": True,
        "every_contrast_plus_state_difference": plus_difference.numerator,
        "every_contrast_minus_state_difference": minus_difference.numerator,
        "every_contrast_mean": _fraction_payload(mean),
        "every_contrast_variance": _fraction_payload(variance),
        "within_root_temporal_correlation": _fraction_payload(temporal_correlation),
        "cross_contrast_correlation": _fraction_payload(cross_correlation),
        "source_joint_feasibility": "not_evaluated",
    }
    _require_literal(
        admission_protocol["derived_witness_properties"],
        expected_derived,
        "P4.7 derived witness properties",
    )

    scientific_power = _require_mapping(
        _require_mapping(scientific_raw["analysis"], "P4.7 v3 analysis")["power"],
        "P4.7 v3 power",
    )
    icc_labels = {
        Fraction(_require_text(value, "P4.7 ICC label"))
        for value in scientific_power["mandatory_within_root_icc_decimals"]
    }
    cross_label_values = {
        "independent_contrasts": Fraction(0, 1),
        "equicorrelation_negative_0_10": Fraction(-1, 10),
        "equicorrelation_positive_0_50": Fraction(1, 2),
    }
    configured_cross_values = {
        cross_label_values[label]
        for label in scientific_power["mandatory_cross_contrast_dependence"]
        if label in cross_label_values
    }
    source_mapping_established = "source_structural_covariance_matrix" in scientific_power
    expected_membership = {
        "fixed_icc_label_match": temporal_correlation in icc_labels,
        "fixed_cross_dependence_label_match": (cross_correlation in configured_cross_values),
        "source_structural_mapping_established": source_mapping_established,
        "membership_under_cartesian_first": False,
        "membership_under_sentinel_first": True,
        "membership_identified_by_v3": False,
    }
    if (
        expected_membership["fixed_icc_label_match"]
        or expected_membership["fixed_cross_dependence_label_match"]
        or source_mapping_established
    ):
        raise ValueError("P4.7 ambiguity witness unexpectedly entered the v3 grid")
    _require_literal(
        admission_protocol["grid_membership"],
        expected_membership,
        "P4.7 grid membership",
    )

    precedence_present = any(
        key in scientific_power
        for key in (
            "mandatory_global_joint_sentinels",
            "sentinel_before_cartesian_filtering",
            "sentinel_survives_cartesian_filtering",
        )
    )
    expected_ambiguity = {
        "interpretation_a": {
            "name": "cartesian_product_then_feasibility_filter",
            "rule": (
                "variance_x_icc_x_cross_dependence_x_missingness_x_pattern_labels_define_all_candidate_tuples_before_feasibility"
            ),
            "plus_one_temporal_and_cross_correlation_witness_admitted": False,
            "conditional_numeric_bound_applies": False,
        },
        "interpretation_b": {
            "name": "mandatory_variance_sentinel_before_cartesian_product",
            "rule": (
                "maximum_feasible_bounded_variance_is_a_global_sentinel_evaluated_before_labeled_cartesian_tuples"
            ),
            "plus_one_temporal_and_cross_correlation_witness_admitted": True,
            "conditional_numeric_bound_applies": True,
        },
        "both_interpretations_preserve_all_v3_literal_lists": True,
        "v3_contains_precedence_or_survival_rule_selecting_one_interpretation": (precedence_present),
        "interpretations_produce_opposite_numeric_bound_applicability": True,
        "mechanical_resolution_from_v3_alone_possible": precedence_present,
    }
    _require_literal(
        admission_protocol["ambiguity_witness"],
        expected_ambiguity,
        "P4.7 ambiguity witness",
    )
    if precedence_present:
        raise ValueError("P4.7 v3 no longer supports the frozen ambiguity verdict")

    historical_v1 = _load_power_bound_protocol_v1()
    historical_scenario = _require_mapping(
        historical_v1["mandatory_scenario"],
        "P4.7 historical sentinel",
    )
    v1_sentinel_present = (
        historical_scenario["application_order"] == "mandatory_marginal_joint_sentinel_before_cartesian_filtering"
    )
    expected_posthoc = {
        "v1_sentinel_order_rule_present": v1_sentinel_present,
        "same_rule_frozen_in_target_v3": precedence_present,
        "retroactive_application_to_v3_permitted": False,
        "v1_numeric_bound_valid": True,
        "v1_numeric_bound_conditional_only": True,
        "v1_scientific_admission": False,
        "scientific_admission_false_means": ("not_proven_admitted_not_proven_infeasible"),
        "decisive_v3_power_failure": False,
    }
    _require_literal(
        admission_protocol["posthoc_semantics"],
        expected_posthoc,
        "P4.7 posthoc semantics",
    )

    expected_admission = {
        "post_freeze_rule_may_expand_v3_scenario_membership": False,
        "absence_of_membership_rule_may_be_treated_as_feasible": False,
        "absence_of_membership_rule_may_be_treated_as_infeasible": False,
        "numeric_bound_unconditionally_applies_to_v3_frozen_grid": False,
        "v3_power_contract_is_mechanically_determinate": False,
        "under_specification_may_authorize_development": False,
        "required_prior_full_power_artifact_exists": False,
        "required_source_preflight_exists": False,
        "admission_result": P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2,
    }
    _require_literal(
        admission_protocol["admission_logic"],
        expected_admission,
        "P4.7 derived admission logic",
    )
    expected_terminal = {
        "status": P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2,
        "certificate_valid": True,
        "v1_numeric_calculation_valid_conditionally": True,
        "v1_unconditional_scientific_admission_valid": False,
        "v3_power_passed": False,
        "v3_power_failed_under_frozen_grid": None,
        "v3_power_contract_resolved": False,
        "v3_prior_power_admission_satisfied": False,
        "v3_development_authorized": False,
        "v3_retired_without_model_output": True,
        "stopping_basis": "unresolved_prior_power_contract_not_numeric_fail",
        "v4_zero_output_planning_authorized": True,
        "model_output_authorized": False,
        "next_action": ("publish_v4_zero_output_protocol_and_full_power_planner_before_any_source_or_model_output"),
    }
    _require_literal(
        admission_protocol["terminal"],
        expected_terminal,
        "P4.7 derived terminal",
    )


def _power_admission_certificate_core(
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation: RelationshipP4LongContextPreparation,
) -> dict[str, object]:
    descriptor = _protocol_descriptor(protocol.protocol_id)
    if descriptor.version != "v3":
        raise ValueError("P4.7 power admission must consume scientific v3")
    admission_protocol = _load_power_bound_protocol_v2()
    lineage = _require_mapping(
        admission_protocol["input_lineage"],
        "P4.7 power admission input lineage",
    )
    _require_literal(
        lineage,
        {
            "scientific_protocol_id": P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            "scientific_protocol_raw_sha256": descriptor.bundled_raw_sha256,
            "scientific_preparation_artifact_id": preparation.artifact_id,
            "scientific_preparation_raw_sha256": ("a4b2f3ee920e398ae0f7eab5757b7988dc3fa4f7db7b4769599c837bc656bcd6"),
            "scientific_preparation_manifest_raw_sha256": (
                "4aaf10d76b80a780e62a17b62803906cc34dd34d64bb4d0f8b96d60dff1e2663"
            ),
        },
        "P4.7 power admission input lineage",
    )
    if (
        _sha256_bytes((preparation.output_dir / _PREPARATION_FILE).read_bytes())
        != lineage["scientific_preparation_raw_sha256"]
    ):
        raise ValueError("P4.7 power admission preparation raw lineage drift")
    if (
        _sha256_bytes((preparation.output_dir / _MANIFEST_FILE).read_bytes())
        != lineage["scientific_preparation_manifest_raw_sha256"]
    ):
        raise ValueError("P4.7 power admission preparation manifest lineage drift")

    scientific_raw = _load_json_object(descriptor.protocol_path)
    _validate_v3_frozen_sections(scientific_raw)
    _validate_power_admission_derivation(admission_protocol, scientific_raw)
    analysis = _require_mapping(scientific_raw["analysis"], "P4.7 v3 analysis")
    power = _require_mapping(analysis["power"], "P4.7 v3 power")
    frozen_grid = _require_mapping(
        admission_protocol["frozen_v3_grid_facts"],
        "P4.7 frozen v3 grid facts",
    )
    _require_literal(
        power["mandatory_within_root_icc_decimals"],
        frozen_grid["mandatory_within_root_icc_labels"],
        "P4.7 frozen ICC labels",
    )
    _require_literal(
        power["mandatory_cross_contrast_dependence"],
        frozen_grid["mandatory_cross_contrast_dependence_labels"],
        "P4.7 frozen cross-contrast labels",
    )
    if frozen_grid["mandatory_variance_scenario"] not in power["mandatory_variance_scenarios"]:
        raise ValueError("P4.7 maximum-variance scenario disappeared")
    if power["all_feasible_cartesian_scenario_combinations_must_pass"] is not True:
        raise ValueError("P4.7 Cartesian requirement drift")
    for absent_key in (
        "mandatory_global_joint_sentinels",
        "sentinel_before_cartesian_filtering",
        "mechanical_infeasibility_witness_for_every_skipped_tuple",
        "within_root_icc_random_variable_and_aggregation",
        "cross_contrast_dependence_matrix_mapping",
        "source_structural_covariance_matrix",
    ):
        if absent_key in power:
            raise ValueError(f"P4.7 power ambiguity premise drift: {absent_key}")

    conditional = _require_mapping(
        admission_protocol["conditional_numeric_bound"],
        "P4.7 conditional numeric bound",
    )
    exact_tail = _maximum_variance_point_gate_probability(
        formal_root_count=protocol.formal_subject_count,
        mass_at_upper=Fraction(11, 20),
        minimum_upper_count=104,
    )
    _require_literal(
        conditional,
        {
            "authority": "conditional_diagnostic_only",
            "condition": (
                "a_future_protocol_explicitly_registers_the_maximum_variance_witness_as_a_mandatory_pre_filter_sentinel"
            ),
            "formal_root_count": 192,
            "root_difference_support": [-2, 2],
            "planning_mean": _fraction_payload(Fraction(1, 5)),
            "maximum_variance": _fraction_payload(Fraction(99, 25)),
            "mass_at_plus_two": _fraction_payload(Fraction(11, 20)),
            "mass_at_minus_two": _fraction_payload(Fraction(9, 20)),
            "practical_threshold": _fraction_payload(Fraction(3, 20)),
            "minimum_success_count": 104,
            "exact_tail_numerator": str(exact_tail.numerator),
            "exact_tail_denominator": str(exact_tail.denominator),
            "display_decimal_half_even_20_places": _fraction_display_decimal(
                exact_tail,
                places=20,
            ),
            "required_power": _fraction_payload(Fraction(4, 5)),
            "conditional_bound_is_below_required": True,
            "actual_v3_grid_power_estimated": False,
            "may_be_used_as_v3_decisive_failure": False,
        },
        "P4.7 conditional numeric bound",
    )
    return {
        "schema_version": P4_LONG_CONTEXT_POWER_ADMISSION_SCHEMA_VERSION_V2,
        "certificate_id_contract": ("sha256_canonical_json_utf8_newline_without_certificate_id_v1"),
        "identity": {
            "power_admission_protocol_id": P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2,
            "power_admission_protocol_raw_sha256": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2),
            "scientific_protocol_id": protocol.protocol_id,
            "scientific_protocol_raw_sha256": descriptor.bundled_raw_sha256,
            "scientific_preparation_artifact_id": preparation.artifact_id,
            "historical_power_bound_protocol_id": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1),
            "historical_power_bound_protocol_raw_sha256": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1),
            "historical_power_failure_artifact_id": (P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID_V1),
            "historical_power_failure_certificate_id": (
                "682efba886b002db849a83ff086963921a173391a4d5e3c050b3d472d17ee70e"
            ),
            "historical_power_failure_certificate_raw_sha256": (
                "543bad00793aabce0869b1f7b2780310ea50a3a957c96b22ae4a677d5ad10de8"
            ),
            "historical_power_failure_manifest_raw_sha256": (
                "f96750169c8613b8c434a9ab80df4fa5f7bf8fee1e4759a0f64ccf73aa7d70e3"
            ),
        },
        "supersession": admission_protocol["supersession"],
        "frozen_v3_grid_facts": admission_protocol["frozen_v3_grid_facts"],
        "historical_v1_witness_facts": admission_protocol["historical_v1_witness_facts"],
        "primitive_joint_witness": admission_protocol["primitive_joint_witness"],
        "derived_witness_properties": admission_protocol["derived_witness_properties"],
        "grid_membership": admission_protocol["grid_membership"],
        "ambiguity_witness": admission_protocol["ambiguity_witness"],
        "posthoc_semantics": admission_protocol["posthoc_semantics"],
        "admission_logic": admission_protocol["admission_logic"],
        "conditional_numeric_bound": conditional,
        "zero_output_firewall": admission_protocol["zero_output_firewall"],
        "v4_requirements": admission_protocol["v4_requirements"],
        "terminal": admission_protocol["terminal"],
        "claim_boundary": admission_protocol["claim_boundary"],
    }


def _power_admission_manifest_core(
    *,
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation: RelationshipP4LongContextPreparation,
    certificate_id: str,
    certificate_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": P4_LONG_CONTEXT_POWER_ADMISSION_MANIFEST_SCHEMA_VERSION_V2,
        "power_admission_protocol_id": P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2,
        "power_admission_protocol_raw_sha256": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2),
        "scientific_protocol_id": protocol.protocol_id,
        "preparation_artifact_id": preparation.artifact_id,
        "certificate_id": certificate_id,
        "status": P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2,
        "files": [
            {
                "path": _POWER_ADMISSION_CERTIFICATE_FILE,
                "byte_count": len(certificate_bytes),
                "sha256": _sha256_bytes(certificate_bytes),
            }
        ],
        "certificate_valid": True,
        "power_contract_determinate": False,
        "conditional_numeric_bound_only": True,
        "v1_unconditional_scientific_admission_valid": False,
        "v3_power_passed": False,
        "v3_power_failed_under_frozen_grid": None,
        "power_admission_certificate_count": 1,
        "full_joint_dgp_artifact_count": 0,
        "simulation_replicate_count": 0,
        "execution_enabled": False,
        "development_authorized": False,
        "qualification_authorized": False,
        "formal_authorized": False,
        "model_output_count": 0,
        "subject_materialization_count": 0,
        "source_materialization_count": 0,
        "donor_bank_materialization_count": 0,
        "counterfactual_twin_materialization_count": 0,
        "cuda_formal_run_count": 0,
        "historical_v1_artifact_preserved": True,
        "claim_boundary": (
            "P4.7 v3 power-grid applicability is under-specified; the v1 numeric bound "
            "is conditional only, and no empirical or execution authorization is granted."
        ),
    }


def _validate_power_admission_root(
    output: pathlib.Path,
    *,
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation: RelationshipP4LongContextPreparation,
) -> RelationshipP4LongContextPowerAdmissionCertificate:
    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 power admission root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    expected_entries = tuple(sorted((_POWER_ADMISSION_CERTIFICATE_FILE, _MANIFEST_FILE)))
    if entries != expected_entries:
        raise ValueError("P4.7 power admission file set drift")
    for name in entries:
        candidate = output / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("P4.7 power admission files must be regular files")

    certificate_path = output / _POWER_ADMISSION_CERTIFICATE_FILE
    certificate_bytes = certificate_path.read_bytes()
    certificate = _load_json_object(certificate_path)
    if certificate_bytes != _canonical_bytes(certificate):
        raise ValueError("P4.7 power admission certificate is not canonical JSON")
    certificate_id = _require_sha256(
        certificate.get("certificate_id"),
        "P4.7 power admission certificate id",
    )
    certificate_core = dict(certificate)
    del certificate_core["certificate_id"]
    if certificate_id != _sha256_bytes(_canonical_bytes(certificate_core)):
        raise ValueError("P4.7 power admission certificate id drift")
    _require_literal(
        certificate_core,
        _power_admission_certificate_core(protocol, preparation),
        "P4.7 power admission certificate",
    )

    manifest_path = output / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_path)
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("P4.7 power admission manifest is not canonical JSON")
    artifact_id = _require_sha256(
        manifest.get("artifact_id"),
        "P4.7 power admission artifact id",
    )
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4.7 power admission manifest artifact id drift")
    expected_manifest_core = _power_admission_manifest_core(
        protocol=protocol,
        preparation=preparation,
        certificate_id=certificate_id,
        certificate_bytes=certificate_bytes,
    )
    _require_literal(
        manifest_core,
        expected_manifest_core,
        "P4.7 power admission manifest",
    )
    if (
        P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2 is not None
        and artifact_id != P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2
    ):
        raise ValueError("P4.7 published power admission artifact id drift")

    conditional = _require_mapping(
        certificate["conditional_numeric_bound"],
        "P4.7 conditional numeric bound",
    )
    admission_logic = _require_mapping(
        certificate["admission_logic"],
        "P4.7 power admission logic",
    )
    terminal = _require_mapping(
        certificate["terminal"],
        "P4.7 power admission terminal",
    )
    return RelationshipP4LongContextPowerAdmissionCertificate(
        artifact_id=artifact_id,
        admission_protocol_id=P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2,
        scientific_protocol_id=protocol.protocol_id,
        preparation_artifact_id=preparation.artifact_id,
        status=_require_text(terminal["status"], "P4.7 power admission status"),
        conditional_bound_numerator=int(
            _require_text(
                conditional["exact_tail_numerator"],
                "P4.7 conditional numerator",
            )
        ),
        conditional_bound_denominator=int(
            _require_text(
                conditional["exact_tail_denominator"],
                "P4.7 conditional denominator",
            )
        ),
        conditional_bound_display_decimal=_require_text(
            conditional["display_decimal_half_even_20_places"],
            "P4.7 conditional display",
        ),
        power_contract_determinate=_require_bool(
            admission_logic["v3_power_contract_is_mechanically_determinate"],
            "P4.7 power contract determinate",
        ),
        v1_unconditional_scientific_admission_valid=_require_bool(
            terminal["v1_unconditional_scientific_admission_valid"],
            "P4.7 v1 unconditional admission",
        ),
        development_authorized=_require_bool(
            terminal["v3_development_authorized"],
            "P4.7 v3 development authorized",
        ),
        formal_authorized=False,
        output_dir=output,
    )


def _load_verified_derivation_helper(
    *,
    path: pathlib.Path,
    expected_raw_sha256: str,
    module_label: str,
) -> ModuleType:
    _validate_v4_source_plain_file(path, module_label)
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_raw_sha256:
        raise ValueError(f"{module_label} raw bytes drift before import")
    module_name = f"_volvence_verified_{module_label}_{expected_raw_sha256}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if not isinstance(module, ModuleType):
            raise TypeError(f"{module_label} verified module cache is not a module")
        if module.__file__ is None or (
            _absolute_without_resolving(pathlib.Path(module.__file__)) != _absolute_without_resolving(path)
        ):
            raise ValueError(f"{module_label} verified module path drift")
        return module
    code = compile(payload, str(path), "exec", dont_inherit=True)
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def _load_verified_v4_planning_derivation_helper() -> ModuleType:
    return _load_verified_derivation_helper(
        path=_V4_PLANNING_DERIVATION_HELPER_PATH,
        expected_raw_sha256=_V4_PLANNING_DERIVATION_HELPER_RAW_SHA256_V1,
        module_label="p4_v4_planning_derivation",
    )


def _load_verified_v4_source_opportunity_derivation_helper() -> ModuleType:
    return _load_verified_derivation_helper(
        path=_V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH,
        expected_raw_sha256=_V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_RAW_SHA256_V1,
        module_label="p4_v4_source_opportunity_derivation",
    )


def _load_v4_planning_protocol_raw(
    path: pathlib.Path | None = None,
) -> Mapping[str, Any]:
    protocol_path = pathlib.Path(_V4_PLANNING_PROTOCOL_PATH_V1 if path is None else path)
    raw = _load_json_object(protocol_path)
    _require_exact_keys(raw, _V4_PLANNING_TOP_LEVEL_KEYS, "P4.7 v4 planning protocol")
    _require_literal(
        raw["schema_version"],
        P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_SCHEMA_VERSION_V1,
        "P4.7 v4 planning schema",
    )
    _require_literal(
        raw["protocol_id_contract"],
        "sha256_canonical_json_utf8_newline_v1",
        "P4.7 v4 planning protocol id contract",
    )
    protocol_id = _sha256_bytes(_canonical_bytes(raw))
    if protocol_id != P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1:
        raise ValueError("P4.7 v4 planning protocol id drift")
    if protocol_path.resolve() == _V4_PLANNING_PROTOCOL_PATH_V1.resolve():
        if _sha256_bytes(protocol_path.read_bytes()) != P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1:
            raise ValueError("P4.7 bundled v4 planning protocol raw bytes drift")
    for section, expected_hash in _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1.items():
        if _sha256_bytes(_canonical_bytes(raw[section])) != expected_hash:
            raise ValueError(f"P4.7 v4 planning frozen section drift: {section}")
    lineage = _require_mapping(raw["input_lineage"], "P4.7 v4 planning input lineage")
    helper_raw_sha256 = _sha256_bytes(_V4_PLANNING_DERIVATION_HELPER_PATH.read_bytes())
    _require_literal(
        lineage["v4_planning_derivation_helper_raw_sha256"],
        helper_raw_sha256,
        "P4.7 v4 planning derivation helper hash",
    )
    return raw


def _validate_v4_planning_derivation(raw: Mapping[str, Any]) -> _V4PlanningDerived:
    derivation = _load_verified_v4_planning_derivation_helper()
    scientific_raw = _load_json_object(_V3_PROTOCOL_PATH)
    _validate_v3_frozen_sections(scientific_raw)
    scientific = load_relationship_p4_long_context_scientific_prereg(_V3_PROTOCOL_PATH)
    lineage = _require_mapping(raw["input_lineage"], "P4.7 v4 planning input lineage")
    _require_literal(
        lineage,
        {
            "scientific_v3_protocol_id": P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            "scientific_v3_protocol_raw_sha256": _protocol_descriptor(
                P4_LONG_CONTEXT_PROTOCOL_ID_V3
            ).bundled_raw_sha256,
            "scientific_v3_preparation_artifact_id": _protocol_descriptor(
                P4_LONG_CONTEXT_PROTOCOL_ID_V3
            ).published_artifact_id,
            "scientific_v3_preparation_raw_sha256": (
                "a4b2f3ee920e398ae0f7eab5757b7988dc3fa4f7db7b4769599c837bc656bcd6"
            ),
            "scientific_v3_preparation_manifest_raw_sha256": (
                "4aaf10d76b80a780e62a17b62803906cc34dd34d64bb4d0f8b96d60dff1e2663"
            ),
            "power_admission_v2_protocol_id": P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2,
            "power_admission_v2_protocol_raw_sha256": (P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2),
            "power_admission_v2_artifact_id": P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2,
            "power_admission_v2_certificate_id": ("cd6ceca086a1d8a311c75bdacd70c976e05b90dff2cde55b3ad41c00d29936b3"),
            "power_admission_v2_certificate_raw_sha256": (
                "0f20e47da67e5ebaed39e63d274805d5783cc588e0f8aa5fa2a0450b079d18ba"
            ),
            "power_admission_v2_manifest_raw_sha256": (
                "f6fb9d482c8eb7f7e5e8dd92546a12bbae6f22ea76730c205f41fba4e14b4972"
            ),
            "v4_planning_derivation_helper_raw_sha256": _sha256_bytes(_V4_PLANNING_DERIVATION_HELPER_PATH.read_bytes()),
            "v3_terminal_status": P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2,
            "v3_power_failed_under_frozen_grid": None,
            "v3_power_passed": False,
        },
        "P4.7 v4 planning input lineage",
    )

    units = _require_mapping(raw["scientific_units"], "P4.7 v4 scientific units")
    arm_matrix = _require_text_tuple(scientific_raw["arm_matrix"], "P4.7 v3 arm matrix")
    if units["reference_arm"] != arm_matrix[0] or units["arm_count"] != len(arm_matrix):
        raise ValueError("P4.7 v4 arm primitive drift")
    if units["evaluation_decisions_per_arm"] != scientific.evaluation_sessions_per_subject:
        raise ValueError("P4.7 v4 decision primitive drift")
    analysis = _require_mapping(scientific_raw["analysis"], "P4.7 v3 analysis")
    contrast_registry = analysis["confirmatory_contrasts"]
    if type(contrast_registry) is not list or units["contrast_count"] != len(contrast_registry):
        raise ValueError("P4.7 v4 contrast primitive drift")
    endpoint = _require_mapping(
        _require_mapping(scientific_raw["causal_execution"], "P4.7 v3 causal execution")[
            "typed_outcome_endpoint_contract"
        ],
        "P4.7 v3 typed endpoint",
    )
    _require_literal(
        units["typed_utility_closed_integer_domain"],
        endpoint["utility_closed_integer_domain"],
        "P4.7 v4 typed utility domain",
    )
    complete_mean = _fraction_from_payload(
        units["complete_data_planning_mean_each_contrast"],
        "P4.7 v4 complete-data planning mean",
    )
    practical_gate = _fraction_from_payload(
        units["practical_observed_mean_gate"],
        "P4.7 v4 practical gate",
    )
    required_power = _fraction_from_payload(
        units["required_power_each_contrast_axis_and_integrated"],
        "P4.7 v4 required power",
    )
    if (complete_mean, practical_gate, required_power) != (
        Fraction(1, 5),
        Fraction(3, 20),
        Fraction(4, 5),
    ):
        raise ValueError("P4.7 v4 planning constants drift")
    if units["formal_sample_size_selected"] is not None:
        raise ValueError("P4.7 v4a scientific units selected N")

    sentinels = _require_mapping(
        raw["mandatory_global_joint_sentinels"],
        "P4.7 v4 mandatory sentinels",
    )
    if sentinels["application_order"] != "before_source_conditioned_cartesian_enumeration_and_filtering":
        raise ValueError("P4.7 v4 sentinel precedence drift")
    if sentinels["filterable_by_source_or_grid_labels"] is not False:
        raise ValueError("P4.7 v4 sentinel became filterable")
    sentinel_items = sentinels["sentinels"]
    if type(sentinel_items) is not list or len(sentinel_items) != 1:
        raise ValueError("P4.7 v4 sentinel inventory drift")
    sentinel = _require_mapping(sentinel_items[0], "P4.7 v4 sentinel")
    primitive = _require_mapping(
        sentinel["primitive_joint_distribution"],
        "P4.7 v4 sentinel primitive",
    )
    sentinel_derived = derivation.derive_shared_reference_sentinel(
        state_masses=(
            _fraction_from_payload(
                primitive["plus_state_probability"],
                "P4.7 v4 sentinel plus mass",
            ),
            _fraction_from_payload(
                primitive["minus_state_probability"],
                "P4.7 v4 sentinel minus mass",
            ),
        ),
        state_reference_utilities=(
            _require_int(
                primitive["plus_state_reference_utility_each_decision"],
                "P4.7 v4 sentinel plus reference utility",
            ),
            _require_int(
                primitive["minus_state_reference_utility_each_decision"],
                "P4.7 v4 sentinel minus reference utility",
            ),
        ),
        state_comparator_utilities=(
            _require_int(
                primitive["plus_state_every_comparator_utility_each_decision"],
                "P4.7 v4 sentinel plus comparator utility",
            ),
            _require_int(
                primitive["minus_state_every_comparator_utility_each_decision"],
                "P4.7 v4 sentinel minus comparator utility",
            ),
        ),
        arm_count=len(arm_matrix),
        decision_count=scientific.evaluation_sessions_per_subject,
        utility_domain=tuple(endpoint["utility_closed_integer_domain"]),
    )
    sentinel_completion = _require_mapping(
        primitive["technical_completion"],
        "P4.7 v4 sentinel technical completion",
    )
    for probability_field in (
        "per_root_probability_of_any_authenticated_technical_failure_across_20_sessions_x_9_arms",
        "substantive_malformed_action_probability",
        "integrity_failure_probability",
    ):
        if (
            _fraction_from_payload(
                sentinel_completion[probability_field],
                f"P4.7 v4 sentinel {probability_field}",
            )
            != 0
        ):
            raise ValueError(f"P4.7 v4 sentinel completion probability drift: {probability_field}")
    for field in (
        "all_20_sessions_x_9_arms_and_evaluation_receipts_complete_and_authenticated",
        "every_generated_root_is_globally_complete",
    ):
        if sentinel_completion[field] is not True:
            raise ValueError(f"P4.7 v4 sentinel completion contract drift: {field}")
    if sentinel_completion["worst_case_itt_imputation_invoked"] is not False:
        raise ValueError("P4.7 v4 sentinel unexpectedly invokes ITT imputation")
    temporal_correlations = tuple(item for contrast in sentinel_derived.temporal_pair_correlations for item in contrast)
    if len(temporal_correlations) != 8 * 28 or set(temporal_correlations) != {Fraction(1, 1)}:
        raise ValueError("P4.7 v4 sentinel temporal derivation drift")
    if len(sentinel_derived.cross_contrast_correlations) != 28 or set(sentinel_derived.cross_contrast_correlations) != {
        Fraction(1, 1)
    }:
        raise ValueError("P4.7 v4 sentinel cross-contrast derivation drift")
    if any(value.denominator != 1 for value in sentinel_derived.contrast_support):
        raise ValueError("P4.7 v4 sentinel support is not integral")
    expected_sentinel_derived = {
        "typed_nine_arm_eight_decision_joint_feasible": True,
        "every_contrast_root_difference_support": sorted(
            value.numerator for value in sentinel_derived.contrast_support
        ),
        "every_contrast_complete_data_mean": _fraction_payload(sentinel_derived.complete_data_mean),
        "every_contrast_root_difference_variance": _fraction_payload(sentinel_derived.root_difference_variance),
        "every_contrast_each_temporal_pair_correlation": _fraction_payload(Fraction(1, 1)),
        "every_distinct_contrast_root_mean_pair_correlation": _fraction_payload(Fraction(1, 1)),
        "globally_complete_root_count_equals_candidate_N": True,
        "ceiling_five_sixths_completeness_gate_always_passes": True,
        "maximum_variance_proof": ("bhatia_davis_equality_for_support_minus_two_plus_two_and_mean_one_fifth"),
    }
    _require_literal(
        sentinel["derived_expected"],
        expected_sentinel_derived,
        "P4.7 v4 sentinel derived audit values",
    )
    if (
        sentinel_derived.complete_data_mean != complete_mean
        or sentinel_derived.root_difference_variance != Fraction(99, 25)
        or sentinel_derived.contrast_support != (Fraction(2, 1), Fraction(-2, 1))
    ):
        raise ValueError("P4.7 v4 sentinel exact moments drift")

    planner = _require_mapping(raw["full_joint_power_planner"], "P4.7 v4 power planner")
    candidate_spec = _require_mapping(
        planner["candidate_formal_root_counts"],
        "P4.7 v4 candidate root counts",
    )
    candidate_root_counts = derivation.derive_candidate_root_counts(
        first=_require_int(candidate_spec["first"], "P4.7 v4 first candidate N"),
        step=_require_int(candidate_spec["step"], "P4.7 v4 candidate N step"),
        last_inclusive=_require_int(
            candidate_spec["last_inclusive"],
            "P4.7 v4 last candidate N",
        ),
    )
    if candidate_spec["derived_count"] != len(candidate_root_counts):
        raise ValueError("P4.7 v4 candidate N derived count drift")
    necessary_screens = derivation.derive_necessary_point_screens(
        candidate_root_counts=candidate_root_counts,
        mass_at_plus_two=Fraction(11, 20),
        practical_gate=practical_gate,
        required_power=required_power,
    )
    if len(necessary_screens) != len(candidate_root_counts):
        raise ValueError("P4.7 v4 necessary screen did not enumerate every candidate")
    first_passing_screen = next((item for item in necessary_screens if item.passed), None)
    if first_passing_screen is None:
        raise ValueError("P4.7 v4 necessary screen has no passing candidate")
    screen_by_root_count = {item.root_count: item for item in necessary_screens}
    if (
        first_passing_screen.root_count != 1088
        or not screen_by_root_count[1088].passed
        or screen_by_root_count[1152].passed
    ):
        raise ValueError("P4.7 v4 necessary screen nonmonotonic witness drift")
    screen_contract = _require_mapping(
        sentinel["exact_necessary_point_screen"],
        "P4.7 v4 sentinel point screen contract",
    )
    if screen_contract["evaluated_independently_for_every_candidate_N"] is not True:
        raise ValueError("P4.7 v4 sentinel screen permits partial candidate enumeration")
    if screen_contract["published_table_contains_every_candidate_N_exact_fraction"] is not True:
        raise ValueError("P4.7 v4 sentinel screen table may omit exact candidate powers")
    _require_literal(
        screen_contract["published_table_exact_fraction_encoding"],
        ("reduced_positive_numerator_and_denominator_as_canonical_lowercase_hex_without_prefix_or_leading_zero"),
        "P4.7 v4 sentinel screen fraction encoding",
    )
    if screen_contract["published_table_file"] != _V4_SENTINEL_SCREEN_TABLE_FILE:
        raise ValueError("P4.7 v4 sentinel screen table filename drift")
    if screen_contract["monotonicity_shortcut_permitted"] is not False:
        raise ValueError("P4.7 v4 sentinel screen permits a monotonicity shortcut")
    if screen_contract["first_pass_may_be_used_as_a_lower_bound_for_later_candidates"] is not False:
        raise ValueError("P4.7 v4 sentinel first pass may be treated as a lower bound")
    _require_literal(
        screen_contract["nonmonotonic_witness"],
        {
            "earlier_candidate_N": 1088,
            "earlier_candidate_passed": True,
            "later_candidate_N": 1152,
            "later_candidate_passed": False,
        },
        "P4.7 v4 sentinel nonmonotonic witness",
    )

    grid = _require_mapping(
        raw["source_conditioned_cartesian_grid"],
        "P4.7 v4 source-conditioned grid",
    )
    axes = _require_mapping(grid["axes"], "P4.7 v4 Cartesian axes")
    cartesian_count = derivation.derive_cartesian_tuple_count(axes)
    if cartesian_count != grid["candidate_tuple_count_before_feasibility"]:
        raise ValueError("P4.7 v4 Cartesian tuple count drift")
    cross_labels = tuple(axes["cross_contrast_dependence_labels"])
    if cross_labels != (
        "source_structural_covariance",
        "independent_contrasts",
        "equicorrelation_negative_0_10",
        "equicorrelation_positive_0_50",
    ):
        raise ValueError("P4.7 v4 cross-contrast label order drift")
    if derivation.equicorrelation_psd_certificate(dimension=8, rho=Fraction(-1, 10)) != (
        Fraction(11, 10),
        Fraction(3, 10),
    ):
        raise ValueError("P4.7 v4 negative equicorrelation certificate drift")
    if derivation.equicorrelation_psd_certificate(dimension=8, rho=Fraction(1, 2)) != (
        Fraction(1, 2),
        Fraction(9, 2),
    ):
        raise ValueError("P4.7 v4 positive equicorrelation certificate drift")
    anti_vacuity = _require_mapping(grid["anti_vacuity"], "P4.7 v4 anti-vacuity")
    for field in (
        "at_least_one_source_reference_tuple_must_be_admitted",
        "every_level_of_every_mandatory_axis_appears_in_at_least_one_admitted_tuple",
        "unresolved_tuple_count_must_equal_zero_before_power_search",
    ):
        if anti_vacuity[field] is not True:
            raise ValueError(f"P4.7 v4 anti-vacuity gate drift: {field}")
    if anti_vacuity["declaring_all_difficult_tuples_infeasible_may_authorize"] is not False:
        raise ValueError("P4.7 v4 vacuous feasibility authorization drift")
    infeasibility = _require_mapping(
        grid["infeasibility_witness"],
        "P4.7 v4 infeasibility witness contract",
    )
    if infeasibility["absence_of_a_constructive_search_result_is_a_proof"] is not False:
        raise ValueError("P4.7 v4 permits search failure as infeasibility proof")

    missingness = _require_mapping(raw["missingness_semantics"], "P4.7 v4 missingness")
    if missingness["planning_alternative_mean_stage"] != (
        "latent_complete_data_after_substantive_malformed_mapping_and_before_technical_missingness_or_itt_imputation"
    ):
        raise ValueError("P4.7 v4 planning mean/missingness order drift")
    if (
        _fraction_from_payload(
            missingness["complete_data_mean_each_contrast"],
            "P4.7 v4 missingness complete-data mean",
        )
        != complete_mean
    ):
        raise ValueError("P4.7 v4 missingness planning mean drift")
    missing_rates = tuple(Fraction(value) for value in axes["technical_missingness_rate_decimals"])
    if missing_rates != (Fraction(0, 1), Fraction(1, 100), Fraction(1, 50)):
        raise ValueError("P4.7 v4 missingness rate grid drift")
    if missingness["minimum_globally_complete_roots"] != "ceiling_five_sixths_of_candidate_N":
        raise ValueError("P4.7 v4 globally-complete gate drift")

    classifier = _require_mapping(
        raw["generated_action_classification"],
        "P4.7 v4 generated-action classifier",
    )
    cases = classifier["classification_cases"]
    if type(cases) is not list or len(cases) != 11:
        raise ValueError("P4.7 v4 generated-action case inventory drift")
    action_classifications: list[tuple[str, str, str]] = []
    seen_case_ids: set[str] = set()
    for item in cases:
        case = _require_mapping(item, "P4.7 v4 generated-action case")
        case_id = _require_text(case["case_id"], "P4.7 v4 generated-action case id")
        if case_id in seen_case_ids:
            raise ValueError("P4.7 v4 duplicate generated-action case id")
        seen_case_ids.add(case_id)
        primitive_case = {
            key: value for key, value in case.items() if key not in {"case_id", "classification", "consequence"}
        }
        derived_classification, derived_consequence = derivation.classify_generated_action_case(primitive_case)
        _require_literal(
            (case["classification"], case["consequence"]),
            (derived_classification, derived_consequence),
            f"P4.7 v4 generated-action case {case_id}",
        )
        action_classifications.append((case_id, derived_classification, derived_consequence))

    candidate_cells = _require_mapping(
        raw["development_candidate_cells"],
        "P4.7 v4 development candidate cells",
    )
    candidate_cell_ids = derivation.derive_candidate_cells(
        baseline_families=_require_text_tuple(
            candidate_cells["baseline_families"],
            "P4.7 v4 baseline families",
        ),
        candidate_indices=tuple(candidate_cells["candidate_indices"]),
    )
    if candidate_cells["candidate_cell_count"] != len(candidate_cell_ids):
        raise ValueError("P4.7 v4 candidate cell count drift")
    counterbalance = _require_mapping(
        candidate_cells["counterbalance"],
        "P4.7 v4 candidate counterbalance",
    )
    phases = _require_mapping(counterbalance["session_phases"], "P4.7 v4 candidate phases")
    sessions_per_cell = _require_int(
        counterbalance["sessions_per_candidate_cell"],
        "P4.7 v4 candidate sessions",
    )
    if sum(_require_int(value, "P4.7 v4 candidate phase count") for value in phases.values()) != sessions_per_cell:
        raise ValueError("P4.7 v4 candidate phase horizon drift")
    candidate_schedule = derivation.derive_williams_candidate_schedule(
        cell_ids=candidate_cell_ids,
        root_count=scientific.development_subject_count,
        sessions_per_root=sessions_per_cell,
        seed=_require_int(counterbalance["seed"], "P4.7 v4 candidate schedule seed"),
    )
    _validate_v4_williams_schedule(candidate_schedule, candidate_cell_ids)

    decision_rule = _require_mapping(planner["decision_rule"], "P4.7 v4 decision rule")
    log_upper_bound, exponential_lower_bound = derivation.hoeffding_bonferroni_certificate()
    if (
        _fraction_from_payload(
            decision_rule["log_160_strict_rational_upper_bound"],
            "P4.7 v4 log upper bound",
        )
        != log_upper_bound
    ):
        raise ValueError("P4.7 v4 Hoeffding log bound drift")
    log_certificate = _require_mapping(
        decision_rule["log_upper_bound_exact_certificate"],
        "P4.7 v4 log certificate",
    )
    if (
        _fraction_from_payload(
            log_certificate["exact_sum_minus_160"],
            "P4.7 v4 log certificate difference",
        )
        != exponential_lower_bound - 160
    ):
        raise ValueError("P4.7 v4 Hoeffding certificate drift")
    if 64 * (8 * log_upper_bound).numerator != 324864 or (8 * log_upper_bound).denominator != 125:
        raise ValueError("P4.7 v4 exact integer decision coefficient drift")
    first_positive_mean_gate = derivation.minimum_candidate_for_exact_positive_mean_gate(
        candidate_root_counts=candidate_root_counts,
        practical_gate=practical_gate,
        log_upper_bound=log_upper_bound,
    )
    if first_positive_mean_gate is None:
        raise ValueError("P4.7 v4 bounded-mean gate cannot pass inside candidate range")
    estimation = _require_mapping(planner["power_estimation"], "P4.7 v4 power estimation")
    if _require_int(estimation["search_joint_dgp_replicates"], "P4.7 v4 search replicates") != 8192:
        raise ValueError("P4.7 v4 search replicate count drift")
    if _require_int(estimation["confirmation_joint_dgp_replicates"], "P4.7 v4 confirmation replicates") != 100000:
        raise ValueError("P4.7 v4 confirmation replicate count drift")
    if _require_int(estimation["search_seed"], "P4.7 v4 search seed") != 20260824:
        raise ValueError("P4.7 v4 search seed drift")
    if _require_int(estimation["confirmation_seed"], "P4.7 v4 confirmation seed") != 20260827:
        raise ValueError("P4.7 v4 confirmation seed drift")
    if _fraction_from_payload(estimation["search_proposal_gate_exact"], "P4.7 v4 search point gate") != Fraction(
        41, 50
    ):
        raise ValueError("P4.7 v4 search point gate drift")
    _require_literal(
        estimation["search_exact_integer_pass_rule"],
        "50_times_X_search_is_greater_than_or_equal_to_41_times_8192",
        "P4.7 v4 search integer gate",
    )
    if estimation["search_seed"] == estimation["confirmation_seed"]:
        raise ValueError("P4.7 v4 search and confirmation streams overlap")
    if estimation["search_and_confirmation_counter_domains_disjoint"] is not True:
        raise ValueError("P4.7 v4 search and confirmation counter domains may overlap")
    if estimation["confirmation_failure_may_increment_or_research_N"] is not False:
        raise ValueError("P4.7 v4 confirmation failure may retune N")
    if estimation["only_integrated_confirmation_indicator_authorizes"] is not True:
        raise ValueError("P4.7 v4 confirmation authority drift")
    if (
        estimation["confirmation_replicate_indicators_are_iid_within_each_scenario_under_the_frozen_counter_rng_model"]
        is not True
    ):
        raise ValueError("P4.7 v4 confirmation IID contract drift")
    if estimation["preallocated_roots_are_independent_across_root_ids"] is not True:
        raise ValueError("P4.7 v4 root independence drift")
    if estimation["technical_missingness_dependence_is_confined_within_one_root"] is not True:
        raise ValueError("P4.7 v4 missingness crosses independent roots")
    rng_contract = _require_mapping(estimation["rng_contract"], "P4.7 v4 RNG contract")
    _require_literal(
        rng_contract["algorithm"],
        "sha256_multiblock_counter_exact_rational_categorical_v1",
        "P4.7 v4 RNG algorithm",
    )
    _require_literal(
        rng_contract["domain_tag_ascii"],
        "volvence.relationship_p4_long_context_v4.power_rng.v1",
        "P4.7 v4 RNG domain tag",
    )
    _require_literal(
        rng_contract["stream_labels"],
        {"search": "search", "confirmation": "confirmation"},
        "P4.7 v4 RNG stream labels",
    )
    _require_literal(
        tuple(rng_contract["counter_fields_in_order"]),
        (
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
        ),
        "P4.7 v4 RNG counter field order",
    )
    _require_literal(
        tuple(rng_contract["counter_field_types_in_order"]),
        (
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
        ),
        "P4.7 v4 RNG counter field types",
    )
    _require_literal(
        rng_contract["counter_text_field_ascii_regex"],
        "^[a-z0-9_.:-]+$",
        "P4.7 v4 RNG counter text domain",
    )
    _require_literal(
        rng_contract["canonical_counter_bytes"],
        (
            "utf8_without_bom_of_the_JSON_array_of_counter_fields_in_order_using_double_quoted_ASCII_"
            "strings_commas_without_whitespace_nonnegative_base10_integers_without_leading_zero_and_one_"
            "final_0x0a_byte"
        ),
        "P4.7 v4 RNG counter byte encoding",
    )
    _require_literal(
        rng_contract["sha256_preimage"],
        "exactly_the_canonical_counter_bytes",
        "P4.7 v4 RNG SHA-256 preimage",
    )
    _require_literal(
        rng_contract["sha256_block_digest"],
        "32_raw_bytes_not_hex",
        "P4.7 v4 RNG block digest",
    )
    _require_literal(
        rng_contract["atom_order"],
        "ascending_unique_lowercase_sha256_hex_atom_id_by_unsigned_ASCII_byte_order",
        "P4.7 v4 RNG atom order",
    )
    _require_literal(
        rng_contract["integer_mass_derivation"],
        (
            "Q_is_lcm_of_all_probability_denominators_and_weight_i_is_numerator_i_times_Q_divided_by_"
            "denominator_i_with_sum_of_weights_exactly_Q"
        ),
        "P4.7 v4 RNG integer-mass derivation",
    )
    _require_literal(
        rng_contract["bit_and_block_counts"],
        "for_Q_greater_than_one_b_is_bit_length_of_Q_minus_one_and_h_is_ceiling_b_divided_by_256",
        "P4.7 v4 RNG bit and block counts",
    )
    _require_literal(
        rng_contract["multiblock_integer"],
        "concatenate_h_SHA256_block_digests_in_ascending_block_ordinal_as_one_unsigned_big_endian_integer_Z",
        "P4.7 v4 RNG multiblock integer",
    )
    _require_literal(
        rng_contract["candidate_ticket"],
        "u_equals_Z_modulo_two_pow_b_selecting_the_least_significant_b_bits",
        "P4.7 v4 RNG candidate ticket",
    )
    _require_literal(
        rng_contract["selected_atom"],
        ("first_atom_in_frozen_atom_order_whose_strict_cumulative_integer_weight_is_greater_than_u"),
        "P4.7 v4 RNG categorical mapping",
    )
    for field in (
        "counter_integer_fields_are_zero_based_nonnegative_canonical_base10",
        "generator_node_inventory_and_each_draw_count_frozen_in_source_generator_before_power",
        "acceptance_probability_strictly_greater_than_one_half_for_Q_greater_than_one",
        "candidate_N_is_absent_from_counter_so_ascending_search_candidates_use_the_same_replicate_root_prefixes",
        "search_and_confirmation_domains_differ_in_the_stream_field_and_seed_field",
        "every_used_counter_tuple_is_unique_and_duplicate_use_invalidates_the_planner",
        "cpu_or_cuda_backend_must_match_frozen_per_draw_or_aggregate_digest_equivalence_receipt",
    ):
        if rng_contract[field] is not True:
            raise ValueError(f"P4.7 v4 RNG contract drift: {field}")
    monte_carlo = _require_mapping(
        estimation["monte_carlo_certification"],
        "P4.7 v4 Monte Carlo certification",
    )
    if _require_int(monte_carlo["replicate_count"], "P4.7 v4 MC replicate count") != 100000:
        raise ValueError("P4.7 v4 MC replicate count drift")
    _require_literal(
        monte_carlo["scenario_count_symbol"],
        "M_equals_1_plus_A_where_A_is_the_admitted_grid_tuple_count",
        "P4.7 v4 MC family size",
    )
    if monte_carlo["M_is_frozen_by_the_pre_power_tuple_feasibility_index"] is not True:
        raise ValueError("P4.7 v4 MC family size is not frozen before power")
    if _fraction_from_payload(monte_carlo["familywise_alpha"], "P4.7 v4 MC familywise alpha") != Fraction(1, 100):
        raise ValueError("P4.7 v4 MC familywise alpha drift")
    if _fraction_from_payload(
        monte_carlo["null_boundary_probability"],
        "P4.7 v4 MC null boundary",
    ) != Fraction(4, 5):
        raise ValueError("P4.7 v4 MC null boundary drift")
    _require_literal(
        monte_carlo["exact_upper_tail_T"],
        ("sum_k_from_X_to_100000_of_binomial_100000_choose_k_times_4_pow_k_divided_by_5_pow_100000"),
        "P4.7 v4 MC exact upper tail",
    )
    _require_literal(
        monte_carlo["exact_integer_pass_rule"],
        (
            "100_times_M_times_sum_k_from_X_to_100000_of_binomial_100000_choose_k_times_4_pow_k_is_less_"
            "than_or_equal_to_5_pow_100000"
        ),
        "P4.7 v4 MC exact integer pass rule",
    )
    for field in ("all_M_scenarios_must_pass", "pass_rule_equality_passes"):
        if monte_carlo[field] is not True:
            raise ValueError(f"P4.7 v4 MC certification drift: {field}")
    if estimation["exact_enumeration_may_replace_monte_carlo_only_with_equivalence_receipt"] is not True:
        raise ValueError("P4.7 v4 exact-enumeration substitution contract drift")
    if estimation["point_estimate_without_monte_carlo_uncertainty_may_authorize"] is not False:
        raise ValueError("P4.7 v4 raw Monte Carlo point estimate may authorize")

    sample_size = _require_mapping(raw["sample_size_freeze"], "P4.7 v4 sample-size freeze")
    terminal = _require_mapping(raw["terminal"], "P4.7 v4 planning terminal")
    if sample_size["selected_formal_root_count"] is not None or terminal["selected_formal_root_count"] is not None:
        raise ValueError("P4.7 v4a selected a formal root count")
    if terminal["power_contract_determinate"] is not True or terminal["full_joint_grid_completed"] is not False:
        raise ValueError("P4.7 v4 planning terminal resolution drift")
    for field in (
        "model_output_authorized",
        "development_authorized",
        "qualification_authorized",
        "formal_authorized",
        "source_structural_inventory_authorized",
    ):
        if terminal[field] is not False:
            raise ValueError(f"P4.7 v4 planning terminal opened {field}")
    firewall = _require_mapping(raw["zero_output_firewall"], "P4.7 v4 zero-output firewall")
    for field in (
        "source_structural_artifact_count",
        "full_joint_dgp_artifact_count",
        "power_simulation_replicate_count",
        "subject_materialization_count",
        "donor_bank_materialization_count",
        "counterfactual_twin_materialization_count",
        "baseline_output_count",
        "model_output_count",
        "cuda_formal_run_count",
        "empirical_outcome_count",
    ):
        if _require_int(firewall[field], f"P4.7 v4 firewall {field}") != 0:
            raise ValueError(f"P4.7 v4 zero-output count drift: {field}")
    for field in (
        "appendable_formal_supported",
        "readable_formal_supported",
        "learnable_formal_supported",
        "steerable_formal_supported",
        "integrated_four_axis_supported",
        "production_active_authorized",
        "product_value_claimed",
    ):
        if firewall[field] is not False:
            raise ValueError(f"P4.7 v4 claim firewall opened: {field}")

    return _V4PlanningDerived(
        candidate_root_counts=candidate_root_counts,
        necessary_point_screens=necessary_screens,
        first_necessary_screen_root_count=first_passing_screen.root_count,
        first_necessary_screen_minimum_plus_count=(first_passing_screen.minimum_plus_count),
        first_necessary_screen_power=first_passing_screen.power,
        first_positive_mean_gate_root_count=first_positive_mean_gate,
        cartesian_candidate_tuple_count=cartesian_count,
        candidate_cell_ids=candidate_cell_ids,
        candidate_schedule=candidate_schedule,
        action_classifications=tuple(action_classifications),
    )


def _validate_v4_williams_schedule(
    schedule: tuple[V4CandidateScheduleBlock, ...],
    cell_ids: tuple[str, ...],
) -> None:
    if len(schedule) != 32 * 20:
        raise ValueError("P4.7 v4 Williams schedule block count drift")
    if any(
        block.global_block_ordinal != ordinal
        or block.session_index != ordinal // 32
        or block.root_ordinal != ordinal % 32
        or len(block.ordered_cell_ids) != 6
        or set(block.ordered_cell_ids) != set(cell_ids)
        for ordinal, block in enumerate(schedule)
    ):
        raise ValueError("P4.7 v4 Williams schedule block drift")
    for start in range(0, len(schedule) - 5, 6):
        group = schedule[start : start + 6]
        for position in range(6):
            if {block.ordered_cell_ids[position] for block in group} != set(cell_ids):
                raise ValueError("P4.7 v4 Williams ordinal balance drift")
        carryovers = {
            (order[left], order[left + 1])
            for block in group
            for order in (block.ordered_cell_ids,)
            for left in range(5)
        }
        expected_carryovers = {(left, right) for left in cell_ids for right in cell_ids if left != right}
        if carryovers != expected_carryovers:
            raise ValueError("P4.7 v4 Williams carryover balance drift")


def _validated_v4_planning_inputs(
    *,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None,
) -> tuple[
    RelationshipP4LongContextV4PlanningProtocol,
    RelationshipP4LongContextPreparation,
    RelationshipP4LongContextPowerAdmissionCertificate,
    Mapping[str, Any],
    _V4PlanningDerived,
]:
    raw = _load_v4_planning_protocol_raw(protocol_path)
    derived = _validate_v4_planning_derivation(raw)
    protocol = _v4_planning_protocol_view(raw, derived)
    preparation = validate_relationship_p4_long_context_scientific_prereg(
        output_dir=v3_preparation_dir,
        protocol_path=_V3_PROTOCOL_PATH,
    )
    admission = validate_relationship_p4_long_context_power_admission_certificate(
        output_dir=v2_admission_dir,
        preparation_dir=v3_preparation_dir,
        protocol_path=_V3_PROTOCOL_PATH,
    )
    if preparation.protocol_id != P4_LONG_CONTEXT_PROTOCOL_ID_V3:
        raise ValueError("P4.7 v4 planning requires scientific v3 preparation")
    if admission.artifact_id != P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2:
        raise ValueError("P4.7 v4 planning requires power-admission v2")
    lineage = _require_mapping(raw["input_lineage"], "P4.7 v4 planning input lineage")
    preparation_root = _absolute_without_resolving(v3_preparation_dir)
    admission_root = _absolute_without_resolving(v2_admission_dir)
    actual_hashes = {
        "scientific_v3_preparation_raw_sha256": _sha256_bytes((preparation_root / _PREPARATION_FILE).read_bytes()),
        "scientific_v3_preparation_manifest_raw_sha256": _sha256_bytes(
            (preparation_root / _MANIFEST_FILE).read_bytes()
        ),
        "power_admission_v2_certificate_raw_sha256": _sha256_bytes(
            (admission_root / _POWER_ADMISSION_CERTIFICATE_FILE).read_bytes()
        ),
        "power_admission_v2_manifest_raw_sha256": _sha256_bytes((admission_root / _MANIFEST_FILE).read_bytes()),
    }
    for field, actual in actual_hashes.items():
        _require_literal(lineage[field], actual, f"P4.7 v4 planning {field}")
    return protocol, preparation, admission, raw, derived


def _v4_candidate_schedule_payload(
    protocol: RelationshipP4LongContextV4PlanningProtocol,
    raw: Mapping[str, Any],
    derived: _V4PlanningDerived,
) -> dict[str, object]:
    counterbalance = _require_mapping(
        _require_mapping(
            raw["development_candidate_cells"],
            "P4.7 v4 candidate cells",
        )["counterbalance"],
        "P4.7 v4 candidate counterbalance",
    )
    return {
        "schema_version": P4_LONG_CONTEXT_V4_CANDIDATE_SCHEDULE_SCHEMA_VERSION_V1,
        "protocol_id": protocol.protocol_id,
        "algorithm": counterbalance["order_algorithm"],
        "seed": counterbalance["seed"],
        "development_root_count": 32,
        "sessions_per_root": 20,
        "candidate_cell_ids": list(derived.candidate_cell_ids),
        "block_count": len(derived.candidate_schedule),
        "blocks": [
            {
                "root_ordinal": block.root_ordinal,
                "session_index": block.session_index,
                "global_block_ordinal": block.global_block_ordinal,
                "ordered_cell_ids": list(block.ordered_cell_ids),
            }
            for block in derived.candidate_schedule
        ],
        "model_output_count": 0,
        "selected_candidate_id": None,
    }


def _v4_sentinel_screen_table_payload(
    protocol: RelationshipP4LongContextV4PlanningProtocol,
    raw: Mapping[str, Any],
    derived: _V4PlanningDerived,
) -> dict[str, object]:
    sentinel_items = _require_mapping(
        raw["mandatory_global_joint_sentinels"],
        "P4.7 v4 sentinel table source",
    )["sentinels"]
    if type(sentinel_items) is not list or len(sentinel_items) != 1:
        raise ValueError("P4.7 v4 sentinel table source inventory drift")
    sentinel = _require_mapping(sentinel_items[0], "P4.7 v4 sentinel table source item")
    return {
        "schema_version": P4_LONG_CONTEXT_V4_SENTINEL_SCREEN_TABLE_SCHEMA_VERSION_V1,
        "protocol_id": protocol.protocol_id,
        "sentinel_id": sentinel["sentinel_id"],
        "candidate_count": len(derived.necessary_point_screens),
        "required_power": _fraction_payload(Fraction(4, 5)),
        "exact_fraction_encoding": (
            "reduced_positive_numerator_and_denominator_as_canonical_lowercase_hex_without_prefix_or_leading_zero"
        ),
        "each_candidate_evaluated_independently": True,
        "monotonicity_shortcut_permitted": False,
        "screens": [
            {
                "root_count": screen.root_count,
                "minimum_plus_count": screen.minimum_plus_count,
                "exact_power_hex": _fraction_hex_payload(screen.power),
                "passed": screen.passed,
            }
            for screen in derived.necessary_point_screens
        ],
        "model_output_count": 0,
        "power_simulation_replicate_count": 0,
    }


def _v4_planning_freeze_core(
    *,
    protocol: RelationshipP4LongContextV4PlanningProtocol,
    preparation: RelationshipP4LongContextPreparation,
    admission: RelationshipP4LongContextPowerAdmissionCertificate,
    raw: Mapping[str, Any],
    derived: _V4PlanningDerived,
    schedule_bytes: bytes,
    screen_table_bytes: bytes,
) -> dict[str, object]:
    derivation = _load_verified_v4_planning_derivation_helper()
    terminal = _require_mapping(raw["terminal"], "P4.7 v4 planning terminal")
    firewall = _require_mapping(raw["zero_output_firewall"], "P4.7 v4 firewall")
    return {
        "schema_version": P4_LONG_CONTEXT_V4_PLANNING_FREEZE_SCHEMA_VERSION_V1,
        "certificate_id_contract": "sha256_canonical_json_utf8_newline_without_certificate_id_v1",
        "identity": {
            "v4_planning_protocol_id": protocol.protocol_id,
            "v4_planning_protocol_raw_sha256": P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1,
            "v4_planning_derivation_helper_raw_sha256": _sha256_bytes(_V4_PLANNING_DERIVATION_HELPER_PATH.read_bytes()),
            "scientific_v3_protocol_id": preparation.protocol_id,
            "scientific_v3_preparation_artifact_id": preparation.artifact_id,
            "power_admission_v2_protocol_id": admission.admission_protocol_id,
            "power_admission_v2_artifact_id": admission.artifact_id,
        },
        "artifact_sequence": raw["artifact_sequence"],
        "derived_global_sentinel": {
            "role": "necessary_screen_only_not_full_power_pass",
            "candidate_screen_count": len(derived.necessary_point_screens),
            "screen_table_file": _V4_SENTINEL_SCREEN_TABLE_FILE,
            "screen_table_raw_sha256": _sha256_bytes(screen_table_bytes),
            "screen_table_byte_count": len(screen_table_bytes),
            "first_candidate_passing_exact_point_screen": (derived.first_necessary_screen_root_count),
            "minimum_plus_count_at_first_pass": (derived.first_necessary_screen_minimum_plus_count),
            "exact_power": _fraction_payload(derived.first_necessary_screen_power),
            "display_decimal_half_even_20_places": _fraction_display_decimal(
                derived.first_necessary_screen_power,
                places=20,
            ),
            "source_filterable": False,
            "nonmonotonic_witness": {
                "earlier_candidate_N": 1088,
                "earlier_candidate_passed": True,
                "later_candidate_N": 1152,
                "later_candidate_exact_power_hex": _fraction_hex_payload(
                    next(screen.power for screen in derived.necessary_point_screens if screen.root_count == 1152)
                ),
                "later_candidate_passed": False,
            },
            "full_decision_rule_power_completed": False,
        },
        "derived_decision_rule": {
            "method": "paired_root_Hoeffding_Bonferroni_exact_integer_gate",
            "log_160_upper_bound": _fraction_payload(Fraction(1269, 250)),
            "log_certificate_sum_minus_160": _fraction_payload(derivation.hoeffding_bonferroni_certificate()[1] - 160),
            "practical_gate_integer_rule": "5_times_S_c_greater_than_or_equal_to_6_times_N",
            "positive_mean_integer_rule": (
                "S_c_positive_and_125_times_S_c_squared_strictly_greater_than_324864_times_N"
            ),
            "first_candidate_capable_at_the_practical_boundary": (derived.first_positive_mean_gate_root_count),
            "bootstrap_inner_loop_count": 0,
        },
        "frozen_power_estimation_contract": {
            "search_joint_dgp_replicates": raw["full_joint_power_planner"]["power_estimation"][
                "search_joint_dgp_replicates"
            ],
            "search_seed": raw["full_joint_power_planner"]["power_estimation"]["search_seed"],
            "search_proposal_gate_exact": raw["full_joint_power_planner"]["power_estimation"][
                "search_proposal_gate_exact"
            ],
            "search_exact_integer_pass_rule": raw["full_joint_power_planner"]["power_estimation"][
                "search_exact_integer_pass_rule"
            ],
            "confirmation_joint_dgp_replicates": raw["full_joint_power_planner"]["power_estimation"][
                "confirmation_joint_dgp_replicates"
            ],
            "confirmation_seed": raw["full_joint_power_planner"]["power_estimation"]["confirmation_seed"],
            "rng_contract": raw["full_joint_power_planner"]["power_estimation"]["rng_contract"],
            "monte_carlo_certification": raw["full_joint_power_planner"]["power_estimation"][
                "monte_carlo_certification"
            ],
            "confirmation_failure_may_increment_or_research_N": raw["full_joint_power_planner"]["power_estimation"][
                "confirmation_failure_may_increment_or_research_N"
            ],
            "point_estimate_without_monte_carlo_uncertainty_may_authorize": raw["full_joint_power_planner"][
                "power_estimation"
            ]["point_estimate_without_monte_carlo_uncertainty_may_authorize"],
        },
        "derived_grid_contract": {
            "candidate_tuple_count_before_feasibility": (derived.cartesian_candidate_tuple_count),
            "global_sentinel_count": 1,
            "source_grid_resolved": False,
            "feasible_tuple_count": None,
            "skipped_tuple_count": None,
            "unresolved_tuple_count": derived.cartesian_candidate_tuple_count,
            "grid_digest": None,
            "sample_size_selected": False,
            "selected_formal_root_count": None,
        },
        "derived_generated_action_cases": [
            {
                "case_id": case_id,
                "classification": classification,
                "consequence": consequence,
            }
            for case_id, classification, consequence in derived.action_classifications
        ],
        "derived_candidate_schedule": {
            "candidate_cell_ids": list(derived.candidate_cell_ids),
            "block_count": len(derived.candidate_schedule),
            "schedule_file": _V4_CANDIDATE_SCHEDULE_FILE,
            "schedule_raw_sha256": _sha256_bytes(schedule_bytes),
            "schedule_byte_count": len(schedule_bytes),
            "selected_candidate_id": None,
        },
        "source_preflight_contract": raw["source_preflight_contract"],
        "zero_output_firewall": firewall,
        "terminal": terminal,
        "claim_boundary": raw["claim_boundary"],
    }


def _v4_planning_manifest_core(
    *,
    protocol: RelationshipP4LongContextV4PlanningProtocol,
    preparation: RelationshipP4LongContextPreparation,
    admission: RelationshipP4LongContextPowerAdmissionCertificate,
    certificate_id: str,
    plan_bytes: bytes,
    schedule_bytes: bytes,
    screen_table_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": P4_LONG_CONTEXT_V4_PLANNING_MANIFEST_SCHEMA_VERSION_V1,
        "v4_planning_protocol_id": protocol.protocol_id,
        "scientific_v3_protocol_id": preparation.protocol_id,
        "scientific_v3_preparation_artifact_id": preparation.artifact_id,
        "power_admission_v2_artifact_id": admission.artifact_id,
        "certificate_id": certificate_id,
        "status": P4_LONG_CONTEXT_V4_PLANNING_STATUS_V1,
        "files": [
            {
                "path": _V4_PLANNING_FREEZE_FILE,
                "byte_count": len(plan_bytes),
                "sha256": _sha256_bytes(plan_bytes),
            },
            {
                "path": _V4_CANDIDATE_SCHEDULE_FILE,
                "byte_count": len(schedule_bytes),
                "sha256": _sha256_bytes(schedule_bytes),
            },
            {
                "path": _V4_SENTINEL_SCREEN_TABLE_FILE,
                "byte_count": len(screen_table_bytes),
                "sha256": _sha256_bytes(screen_table_bytes),
            },
        ],
        "power_contract_determinate": True,
        "source_grid_resolved": False,
        "full_joint_grid_completed": False,
        "sample_size_selected": False,
        "selected_formal_root_count": None,
        "source_materialization_authorized": False,
        "development_authorized": False,
        "model_output_authorized": False,
        "qualification_authorized": False,
        "formal_authorized": False,
        "source_structural_artifact_count": 0,
        "full_joint_dgp_artifact_count": 0,
        "power_search_replicate_count": 0,
        "power_confirmation_replicate_count": 0,
        "subject_materialization_count": 0,
        "baseline_output_count": 0,
        "model_output_count": 0,
        "cuda_planner_run_count": 0,
        "cuda_formal_run_count": 0,
        "empirical_outcome_count": 0,
        "claim_boundary": (
            "v4a freezes planning primitives and an abstract candidate schedule only; "
            "source, power, sample-size, model, CUDA, and four-axis evidence remain absent."
        ),
    }


def _validate_v4_planning_root(
    output: pathlib.Path,
    *,
    protocol: RelationshipP4LongContextV4PlanningProtocol,
    preparation: RelationshipP4LongContextPreparation,
    admission: RelationshipP4LongContextPowerAdmissionCertificate,
    raw: Mapping[str, Any],
    derived: _V4PlanningDerived,
) -> RelationshipP4LongContextV4PlanningFreeze:
    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 v4 planning root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    expected_entries = tuple(
        sorted(
            (
                _V4_PLANNING_FREEZE_FILE,
                _V4_CANDIDATE_SCHEDULE_FILE,
                _V4_SENTINEL_SCREEN_TABLE_FILE,
                _MANIFEST_FILE,
            )
        )
    )
    if entries != expected_entries:
        raise ValueError("P4.7 v4 planning file set drift")
    for name in entries:
        candidate = output / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("P4.7 v4 planning files must be regular files")
        if os.stat(candidate, follow_symlinks=False).st_nlink != 1:
            raise ValueError("P4.7 v4 planning files must have exactly one hard link")

    schedule_path = output / _V4_CANDIDATE_SCHEDULE_FILE
    schedule_bytes = schedule_path.read_bytes()
    schedule = _load_json_object(schedule_path)
    if schedule_bytes != _canonical_bytes(schedule):
        raise ValueError("P4.7 v4 candidate schedule is not canonical JSON")
    _require_literal(
        schedule,
        _v4_candidate_schedule_payload(protocol, raw, derived),
        "P4.7 v4 candidate schedule",
    )

    screen_table_path = output / _V4_SENTINEL_SCREEN_TABLE_FILE
    screen_table_bytes = screen_table_path.read_bytes()
    screen_table = _load_json_object(screen_table_path)
    if screen_table_bytes != _canonical_bytes(screen_table):
        raise ValueError("P4.7 v4 sentinel screen table is not canonical JSON")
    encoded_screens = screen_table.get("screens")
    if type(encoded_screens) is not list or len(encoded_screens) != len(derived.necessary_point_screens):
        raise ValueError("P4.7 v4 sentinel screen table row count drift")
    for encoded, expected in zip(encoded_screens, derived.necessary_point_screens, strict=True):
        encoded_row = _require_mapping(encoded, "P4.7 v4 sentinel screen row")
        if (
            _fraction_from_hex_payload(
                encoded_row["exact_power_hex"],
                "P4.7 v4 sentinel exact power",
            )
            != expected.power
        ):
            raise ValueError("P4.7 v4 sentinel screen exact power drift")
    _require_literal(
        screen_table,
        _v4_sentinel_screen_table_payload(protocol, raw, derived),
        "P4.7 v4 sentinel screen table",
    )

    plan_path = output / _V4_PLANNING_FREEZE_FILE
    plan_bytes = plan_path.read_bytes()
    plan = _load_json_object(plan_path)
    if plan_bytes != _canonical_bytes(plan):
        raise ValueError("P4.7 v4 planning freeze is not canonical JSON")
    certificate_id = _require_sha256(plan.get("certificate_id"), "P4.7 v4 certificate id")
    plan_core = dict(plan)
    del plan_core["certificate_id"]
    if certificate_id != _sha256_bytes(_canonical_bytes(plan_core)):
        raise ValueError("P4.7 v4 planning certificate id drift")
    _require_literal(
        plan_core,
        _v4_planning_freeze_core(
            protocol=protocol,
            preparation=preparation,
            admission=admission,
            raw=raw,
            derived=derived,
            schedule_bytes=schedule_bytes,
            screen_table_bytes=screen_table_bytes,
        ),
        "P4.7 v4 planning freeze",
    )

    manifest_path = output / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_path)
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("P4.7 v4 planning manifest is not canonical JSON")
    artifact_id = _require_sha256(manifest.get("artifact_id"), "P4.7 v4 artifact id")
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4.7 v4 planning manifest artifact id drift")
    _require_literal(
        manifest_core,
        _v4_planning_manifest_core(
            protocol=protocol,
            preparation=preparation,
            admission=admission,
            certificate_id=certificate_id,
            plan_bytes=plan_bytes,
            schedule_bytes=schedule_bytes,
            screen_table_bytes=screen_table_bytes,
        ),
        "P4.7 v4 planning manifest",
    )
    if (
        P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1 is not None
        and artifact_id != P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1
    ):
        raise ValueError("P4.7 published v4 planning artifact id drift")
    terminal = _require_mapping(plan["terminal"], "P4.7 v4 planning terminal")
    sentinel = _require_mapping(
        plan["derived_global_sentinel"],
        "P4.7 v4 derived sentinel",
    )
    decision = _require_mapping(plan["derived_decision_rule"], "P4.7 v4 decision")
    grid = _require_mapping(plan["derived_grid_contract"], "P4.7 v4 derived grid")
    schedule_summary = _require_mapping(
        plan["derived_candidate_schedule"],
        "P4.7 v4 schedule summary",
    )
    return RelationshipP4LongContextV4PlanningFreeze(
        artifact_id=artifact_id,
        protocol_id=protocol.protocol_id,
        scientific_v3_protocol_id=preparation.protocol_id,
        power_admission_v2_artifact_id=admission.artifact_id,
        status=_require_text(terminal["status"], "P4.7 v4 planning status"),
        first_necessary_screen_passing_root_count=_require_int(
            sentinel["first_candidate_passing_exact_point_screen"],
            "P4.7 v4 first necessary candidate",
        ),
        first_positive_mean_gate_capable_root_count=_require_int(
            decision["first_candidate_capable_at_the_practical_boundary"],
            "P4.7 v4 first bounded-mean candidate",
        ),
        cartesian_candidate_tuple_count=_require_int(
            grid["candidate_tuple_count_before_feasibility"],
            "P4.7 v4 tuple count",
        ),
        candidate_schedule_block_count=_require_int(
            schedule_summary["block_count"],
            "P4.7 v4 schedule block count",
        ),
        power_contract_determinate=_require_bool(
            terminal["power_contract_determinate"],
            "P4.7 v4 power contract determinate",
        ),
        source_grid_resolved=_require_bool(
            terminal["full_joint_grid_completed"],
            "P4.7 v4 source grid resolved",
        ),
        selected_formal_root_count=terminal["selected_formal_root_count"],
        source_materialization_authorized=_require_bool(
            terminal["source_structural_inventory_authorized"],
            "P4.7 v4 source authorization",
        ),
        development_authorized=_require_bool(
            terminal["development_authorized"],
            "P4.7 v4 development authorization",
        ),
        model_output_authorized=_require_bool(
            terminal["model_output_authorized"],
            "P4.7 v4 model output authorization",
        ),
        formal_authorized=_require_bool(
            terminal["formal_authorized"],
            "P4.7 v4 formal authorization",
        ),
        output_dir=output,
    )


def _load_v4_source_opportunity_preflight_protocol_raw(
    path: pathlib.Path | None = None,
) -> Mapping[str, Any]:
    protocol_path = pathlib.Path(_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_PATH_V1 if path is None else path)
    _reject_reparse_components(protocol_path, "P4.7 source preflight protocol")
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise FileNotFoundError("P4.7 source preflight protocol is missing")
    if os.stat(protocol_path, follow_symlinks=False).st_nlink != 1:
        raise ValueError("P4.7 source preflight protocol must have one hard link")
    raw_bytes = protocol_path.read_bytes()
    if _sha256_bytes(raw_bytes) != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1:
        raise ValueError("P4.7 source preflight protocol raw bytes drift")
    raw = _strict_json_object_from_bytes(raw_bytes, "P4.7 source preflight protocol")
    _require_exact_keys(
        raw,
        _V4_SOURCE_OPPORTUNITY_PREFLIGHT_TOP_LEVEL_KEYS,
        "P4.7 source preflight protocol",
    )
    _require_literal(
        raw["schema_version"],
        P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SCHEMA_VERSION_V1,
        "P4.7 source preflight schema",
    )
    _require_literal(
        raw["protocol_id_contract"],
        "sha256_canonical_json_utf8_newline_v1",
        "P4.7 source preflight protocol id contract",
    )
    if _sha256_bytes(_canonical_bytes(raw)) != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1:
        raise ValueError("P4.7 source preflight protocol id drift")
    for section, expected_hash in _V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SECTION_SHA256_V1.items():
        if _sha256_bytes(_canonical_bytes(raw[section])) != expected_hash:
            raise ValueError(f"P4.7 source preflight frozen section drift: {section}")
    helper_path = _V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH
    _reject_reparse_components(helper_path, "P4.7 source preflight derivation helper")
    if helper_path.is_symlink() or not helper_path.is_file():
        raise FileNotFoundError("P4.7 source preflight derivation helper is missing")
    if os.stat(helper_path, follow_symlinks=False).st_nlink != 1:
        raise ValueError("P4.7 source preflight derivation helper must have one hard link")
    helper_raw_sha256 = _sha256_bytes(helper_path.read_bytes())
    lineage = _require_mapping(raw["input_lineage"], "P4.7 source preflight lineage")
    _require_literal(
        lineage,
        {
            "v4a_planning_protocol_id": P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1,
            "v4a_planning_protocol_raw_sha256": P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1,
            "v4a_derivation_helper_raw_sha256": _sha256_bytes(_V4_PLANNING_DERIVATION_HELPER_PATH.read_bytes()),
            "v4a_planning_artifact_id": P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1,
            "v4a_planning_certificate_id": ("b7e95f149afe77b283bf135f7cb5d76eb4f4edee4594c8649a778acb4186c764"),
            "v4a_plan_raw_sha256": ("9e17383f416eea555799d7e603996a34d526c7c20e9e65e53af25196a700064f"),
            "v4a_screen_table_raw_sha256": ("d8f0f6b4fa1927138007bac77b687f3507b09ca0f000c6549b584ba2d33b01ba"),
            "v4a_candidate_schedule_raw_sha256": ("df426477209d0e99c74cf62938fcf3700554c6242f9439c2e51ebdd20edf1d6f"),
            "v4a_manifest_raw_sha256": ("26b46683260dc01f632ff9c1874839760f4b075c53eb5cd0298c7fc025633e3e"),
            "v4a_artifact_sequence_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["artifact_sequence"],
            "v4a_scientific_units_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["scientific_units"],
            "v4a_source_grid_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1[
                "source_conditioned_cartesian_grid"
            ],
            "v4a_source_preflight_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["source_preflight_contract"],
            "scientific_v3_protocol_id": P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            "scientific_v3_preparation_artifact_id": _protocol_descriptor(
                P4_LONG_CONTEXT_PROTOCOL_ID_V3
            ).published_artifact_id,
            "power_admission_v2_artifact_id": P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2,
            "relationship_action_registry_module": _RELATIONSHIP_ACTION_REGISTRY_MODULE,
            "relationship_action_registry_raw_sha256": (_RELATIONSHIP_ACTION_REGISTRY_MODULE_RAW_SHA256_V1),
            "relationship_action_choice_schema_id": (
                "https://volvence.local/schemas/relationship-action-choice.v1.json"
            ),
            "relationship_action_choice_schema_raw_sha256": (_RELATIONSHIP_ACTION_CHOICE_SCHEMA_RAW_SHA256_V1),
            "source_opportunity_derivation_helper_raw_sha256": helper_raw_sha256,
        },
        "P4.7 source preflight input lineage",
    )
    return raw


def _strict_json_object_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not carry a UTF-8 BOM")
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not strict UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def _load_v4_source_action_registry() -> _V4SourceActionRegistry:
    schema_path = _RELATIONSHIP_ACTION_CHOICE_SCHEMA_PATH
    _reject_reparse_components(schema_path, "P4.7 source action schema")
    if schema_path.is_symlink() or not schema_path.is_file():
        raise FileNotFoundError("P4.7 source action schema is missing")
    if os.stat(schema_path, follow_symlinks=False).st_nlink != 1:
        raise ValueError("P4.7 source action schema must have one hard link")
    schema_bytes = schema_path.read_bytes()
    if _sha256_bytes(schema_bytes) != _RELATIONSHIP_ACTION_CHOICE_SCHEMA_RAW_SHA256_V1:
        raise ValueError("P4.7 source action schema raw bytes drift")
    schema = _strict_json_object_from_bytes(schema_bytes, "P4.7 source action schema")
    schema_id = "https://volvence.local/schemas/relationship-action-choice.v1.json"
    _require_literal(
        schema,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": schema_id,
            "title": "Relationship Lab stateless action choice",
            "type": "object",
            "additionalProperties": False,
            "required": ["action_id"],
            "properties": {
                "action_id": {
                    "type": "string",
                    "enum": list(_RELATIONSHIP_ACTION_IDS_V1),
                }
            },
        },
        "P4.7 source action schema",
    )

    package_spec = importlib.machinery.PathFinder.find_spec("lifeform_domain_emogpt")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise ModuleNotFoundError("lifeform_domain_emogpt package is unavailable")
    module_spec = importlib.machinery.PathFinder.find_spec(
        _RELATIONSHIP_ACTION_REGISTRY_MODULE,
        package_spec.submodule_search_locations,
    )
    if module_spec is None or module_spec.origin is None:
        raise ModuleNotFoundError("relationship action registry module is unavailable")
    module_path = pathlib.Path(module_spec.origin)
    _reject_reparse_components(module_path, "P4.7 relationship action registry module")
    if module_path.is_symlink() or not module_path.is_file():
        raise FileNotFoundError("P4.7 relationship action registry module is missing")
    if os.stat(module_path, follow_symlinks=False).st_nlink != 1:
        raise ValueError("P4.7 relationship action registry module must have one hard link")
    module_bytes = module_path.read_bytes()
    module_raw_sha256 = _sha256_bytes(module_bytes)
    if module_raw_sha256 != _RELATIONSHIP_ACTION_REGISTRY_MODULE_RAW_SHA256_V1:
        raise ValueError("P4.7 relationship action registry raw bytes drift")
    try:
        module_text = module_bytes.decode("utf-8")
        module_tree = ast.parse(module_text, filename=str(module_path))
    except UnicodeDecodeError as exc:
        raise ValueError("P4.7 relationship action registry is not strict UTF-8") from exc
    relationship_classes = [
        node for node in module_tree.body if isinstance(node, ast.ClassDef) and node.name == "RelationshipAction"
    ]
    if len(relationship_classes) != 1:
        raise ValueError("P4.7 RelationshipAction owner class drift")
    action_values = tuple(
        node.value.value
        for node in relationship_classes[0].body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and type(node.value.value) is str
    )
    if action_values != _RELATIONSHIP_ACTION_IDS_V1:
        raise ValueError("P4.7 RelationshipAction enum order drift")
    registry_id = _sha256_bytes(
        _canonical_bytes(
            {
                "owner_module": _RELATIONSHIP_ACTION_REGISTRY_MODULE,
                "owner_module_raw_sha256": module_raw_sha256,
                "schema_id": schema_id,
                "schema_raw_sha256": _RELATIONSHIP_ACTION_CHOICE_SCHEMA_RAW_SHA256_V1,
                "action_ids": list(action_values),
            }
        )
    )
    return _V4SourceActionRegistry(
        owner_module_raw_sha256=module_raw_sha256,
        schema_raw_sha256=_RELATIONSHIP_ACTION_CHOICE_SCHEMA_RAW_SHA256_V1,
        schema_id=schema_id,
        action_ids=action_values,
        registry_id=registry_id,
    )


def _validate_v4_source_opportunity_derivation(
    raw: Mapping[str, Any],
) -> _V4SourceOpportunityDerived:
    derivation = _load_verified_v4_source_opportunity_derivation_helper()
    action_registry = _load_v4_source_action_registry()
    root_layout = derivation.derive_source_root_surface_layout()
    evaluation_design = derivation.derive_source_evaluation_design(action_order=action_registry.action_ids)
    fact_orientations = derivation.derive_root_fact_orientation_inventory(root_layout=root_layout)
    planning_generator = derivation.derive_exact_synthetic_planning_generator()

    sampling = _require_mapping(
        raw["sampling_frame_contract"],
        "P4.7 source sampling frame",
    )
    _require_literal(
        sampling["surface_factor_axes_in_bit_order"],
        list(root_layout.surface_factor_axes_in_bit_order),
        "P4.7 source factor axis order",
    )
    _require_literal(
        sampling["surface_factor_bit_decoding"],
        root_layout.surface_factor_bit_decoding,
        "P4.7 source factor bit decoding",
    )
    _require_literal(
        sampling["surface_factor_typed_value_registry"],
        [
            {
                "axis_id": item.axis_id,
                "value_0": item.value_zero,
                "value_1": item.value_one,
            }
            for item in root_layout.surface_factor_typed_value_registry
        ],
        "P4.7 source typed factor registry",
    )
    _require_literal(
        sampling["root_namespaces_in_global_order"],
        [
            {
                "namespace_id": f"{item.split_id}_{item.root_role}",
                "split": item.split_id,
                "role": item.root_role,
                "root_count": item.root_count,
                "global_slot_start": item.global_slot_start,
                "global_slot_end_inclusive": item.global_slot_stop_exclusive - 1,
            }
            for item in root_layout.namespaces
        ],
        "P4.7 source root namespaces",
    )
    if sampling["surface_family_capacity"] != root_layout.surface_capacity or sampling[
        "surface_factor_axis_count"
    ] != len(root_layout.surface_factor_axes_in_bit_order):
        raise ValueError("P4.7 source surface capacity or axis count drift")

    capacity = _require_mapping(
        raw["root_independence_and_capacity"],
        "P4.7 source root capacity",
    )
    expected_capacity = {
        "analysis_root_count_at_maximum_candidate": root_layout.analysis_root_count,
        "donor_root_count_at_maximum_candidate": root_layout.donor_root_count,
        "preallocated_independent_root_slot_count": len(root_layout.roots),
        "deterministic_counterfactual_twin_mapping_count": len(root_layout.counterfactual_twin_mappings),
        "unused_surface_family_capacity": root_layout.surface_capacity - len(root_layout.roots),
    }
    for field, expected in expected_capacity.items():
        _require_literal(capacity[field], expected, f"P4.7 source capacity {field}")
    formal_candidates = _require_mapping(
        capacity["formal_candidate_root_counts"],
        "P4.7 source formal candidates",
    )
    candidate_counts = tuple(item.root_count for item in root_layout.formal_candidate_prefixes)
    _require_literal(
        formal_candidates,
        {
            "first": candidate_counts[0],
            "step": candidate_counts[1] - candidate_counts[0],
            "last_inclusive": candidate_counts[-1],
            "count": len(candidate_counts),
        },
        "P4.7 source formal candidate roots",
    )

    opportunity = _require_mapping(
        raw["opportunity_layout_and_utility_vectors"],
        "P4.7 source opportunity layout",
    )
    _require_literal(
        opportunity["ordered_legal_action_ids"],
        list(evaluation_design.action_order),
        "P4.7 source action order",
    )
    _require_literal(
        opportunity["utility_vectors_by_fact_value"],
        {
            "0": list(derivation.FACT_ZERO_UTILITY_VECTOR),
            "1": list(derivation.FACT_ONE_UTILITY_VECTOR),
        },
        "P4.7 source utility vectors",
    )
    orientation = _require_mapping(
        opportunity["fact_orientation_contract"],
        "P4.7 source fact orientation",
    )
    _require_literal(
        orientation["domain_tag_ascii"],
        fact_orientations.ranking_domain,
        "P4.7 source orientation domain",
    )
    _require_literal(
        orientation["seed"],
        fact_orientations.ranking_seed,
        "P4.7 source orientation seed",
    )
    _require_literal(
        orientation["ranking_payload_contract"],
        fact_orientations.ranking_payload_contract,
        "P4.7 source orientation payload contract",
    )
    _require_literal(
        orientation["ranking_key_fields_in_order"],
        list(fact_orientations.ranking_counter_fields),
        "P4.7 source orientation counter fields",
    )
    _require_literal(
        orientation["excluded_fields"],
        list(fact_orientations.ranking_excluded_fields),
        "P4.7 source orientation excluded fields",
    )
    if (
        orientation["root_block_size"] != fact_orientations.block_size
        or orientation["orientation_zero_count_per_block_pair"]
        != fact_orientations.orientation_count_per_value_per_block
        or orientation["orientation_one_count_per_block_pair"]
        != fact_orientations.orientation_count_per_value_per_block
    ):
        raise ValueError("P4.7 source fact orientation balance drift")
    _validate_v4_source_latin_rotation(raw, root_layout)
    _validate_v4_source_planning_generator(raw, planning_generator)

    stage = _require_mapping(raw["stage_boundary"], "P4.7 source stage boundary")
    terminal = _require_mapping(raw["terminal"], "P4.7 source terminal")
    firewall = _require_mapping(raw["zero_output_firewall"], "P4.7 source firewall")
    if (
        stage["source_opportunity_stage_completed_by_this_artifact"] is not False
        or stage["source_structural_inventory_materialized_by_this_artifact"] is not False
        or stage["future_single_create_only_structural_inventory_attempt_authorized_by_this_contract"] is not False
    ):
        raise ValueError("P4.7 source preflight stage boundary opened")
    if terminal["status"] != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_STATUS_V1:
        raise ValueError("P4.7 source preflight status drift")
    for field, value in terminal.items():
        if field.endswith("_authorized") and value is not False:
            raise ValueError(f"P4.7 source preflight authorization opened: {field}")
    for field, value in firewall.items():
        if field.endswith("_count") and value != 0:
            raise ValueError(f"P4.7 source preflight nonzero output count: {field}")
    for field in (
        "all_materialization_counts_refer_to_persisted_or_published_artifact_rows_not_ephemeral_in_memory_exact_derivation_objects",
        "ephemeral_in_memory_structural_derivation_objects_are_not_source_content_or_persistent_materialization",
    ):
        if firewall[field] is not True:
            raise ValueError(f"P4.7 source preflight materialization semantics drift: {field}")
    return _V4SourceOpportunityDerived(
        action_registry=action_registry,
        root_layout=root_layout,
        evaluation_design=evaluation_design,
        fact_orientations=fact_orientations,
        planning_generator=planning_generator,
    )


def _validate_v4_source_latin_rotation(
    raw: Mapping[str, Any],
    root_layout: SourceRootSurfaceDerivation,
) -> None:
    opportunity = _require_mapping(
        raw["opportunity_layout_and_utility_vectors"],
        "P4.7 source opportunity layout",
    )
    strata = opportunity["semantic_stratum_registry_in_canonical_order"]
    if type(strata) is not list or len(strata) != 4:
        raise ValueError("P4.7 source semantic stratum registry drift")
    if len({_canonical_bytes(item) for item in strata}) != 4:
        raise ValueError("P4.7 source semantic strata contain duplicates")
    analysis_roots = tuple(root for root in root_layout.roots if root.root_role == "analysis")
    for split_id, root_count in (
        ("development", 32),
        ("qualification", 64),
        ("formal", 8192),
    ):
        split_roots = tuple(root for root in analysis_roots if root.split_id == split_id)
        if len(split_roots) != root_count:
            raise AssertionError("P4.7 source Latin split cardinality drift")
        for pair_index in range(4):
            counts = tuple(
                sum((root.namespace_ordinal + pair_index) % 4 == stratum for root in split_roots)
                for stratum in range(4)
            )
            if counts != (root_count // 4,) * 4:
                raise AssertionError("P4.7 source Latin stratum balance drift")
    formal_roots = tuple(root for root in analysis_roots if root.split_id == "formal")
    for prefix in root_layout.formal_candidate_prefixes:
        for pair_index in range(4):
            counts = tuple(
                sum((root.namespace_ordinal + pair_index) % 4 == stratum for root in formal_roots[: prefix.root_count])
                for stratum in range(4)
            )
            if counts != (prefix.root_count // 4,) * 4:
                raise AssertionError("P4.7 source formal Latin prefix balance drift")


def _validate_v4_source_planning_generator(
    raw: Mapping[str, Any],
    generator: SyntheticPlanningGeneratorDerivation,
) -> None:
    source = _require_mapping(
        raw["exact_source_planning_generator"],
        "P4.7 source planning generator",
    )
    if source["full_nine_arm_scalar_atom_count"] != len(generator.atoms):
        raise ValueError("P4.7 source planning atom count drift")
    expected = _require_mapping(source["derived_expected"], "P4.7 source derived moments")
    for encoded, actual, label in (
        (expected["every_contrast_generic_decision_mean"], generator.contrast_means[0], "mean"),
        (
            expected["every_contrast_generic_decision_variance"],
            generator.contrast_covariance_matrix[0][0],
            "variance",
        ),
        (
            expected["every_distinct_contrast_pair_covariance"],
            generator.contrast_covariance_matrix[0][1],
            "covariance",
        ),
        (
            expected["every_distinct_contrast_pair_correlation"],
            generator.contrast_correlation_matrix[0][1],
            "correlation",
        ),
    ):
        if _fraction_from_payload(encoded, f"P4.7 source {label}") != actual:
            raise ValueError(f"P4.7 source planning {label} drift")
    eigen_payload = expected["correlation_matrix_eigenvalue_multiplicities"]
    if type(eigen_payload) is not list:
        raise TypeError("P4.7 source eigenvalue registry must be an array")
    eigenvalues = tuple(
        (
            _fraction_from_payload(item["eigenvalue"], "P4.7 source eigenvalue"),
            _require_int(item["multiplicity"], "P4.7 source eigenvalue multiplicity"),
        )
        for item in eigen_payload
    )
    if eigenvalues != tuple((item.eigenvalue, item.multiplicity) for item in generator.correlation_eigenvalues):
        raise ValueError("P4.7 source planning PSD certificate drift")
    dbar_target = _require_mapping(
        source["future_Dbar_source_structural_correlation_target"],
        "P4.7 source Dbar target",
    )
    if (
        _fraction_from_payload(dbar_target["diagonal"], "P4.7 source Dbar diagonal") != 1
        or _fraction_from_payload(
            dbar_target["off_diagonal"],
            "P4.7 source Dbar off diagonal",
        )
        != Fraction(1, 2)
        or dbar_target["status"] != "preregistered_target_constraint_not_derived_from_the_generic_decision_atoms"
    ):
        raise ValueError("P4.7 source Dbar target constraint drift")


def _validate_v4_source_plain_file(path: pathlib.Path, label: str) -> None:
    candidate = _absolute_without_resolving(path)
    _reject_reparse_components(candidate, label)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    if os.stat(candidate, follow_symlinks=False).st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")


def _validate_v4_source_plain_file_closure(
    root_path: pathlib.Path,
    expected_file_names: tuple[str, ...],
    label: str,
) -> pathlib.Path:
    root = _absolute_without_resolving(root_path)
    _reject_reparse_components(root, label)
    if not root.is_dir():
        raise FileNotFoundError(f"{label} root is missing: {root}")
    entries = tuple(sorted(item.name for item in root.iterdir()))
    if entries != tuple(sorted(expected_file_names)):
        raise ValueError(f"{label} file set drift")
    for name in expected_file_names:
        _validate_v4_source_plain_file(root / name, f"{label} input")
    return root


def _validated_v4_source_opportunity_inputs(
    *,
    v4a_planning_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None,
) -> tuple[
    RelationshipP4LongContextSourceOpportunityPreflightProtocol,
    Mapping[str, Any],
    _V4SourceOpportunityDerived,
    Mapping[str, Any],
]:
    for source_path, label in (
        (_V3_PROTOCOL_PATH, "P4.7 source preflight scientific v3 protocol"),
        (_POWER_BOUND_PROTOCOL_PATH_V2, "P4.7 source preflight power-admission v2 protocol"),
        (_V4_PLANNING_PROTOCOL_PATH_V1, "P4.7 source preflight v4a planning protocol"),
        (_V4_PLANNING_DERIVATION_HELPER_PATH, "P4.7 source preflight v4a derivation helper"),
    ):
        _validate_v4_source_plain_file(source_path, label)
    _validate_v4_source_plain_file_closure(
        v3_preparation_dir,
        (_PREPARATION_FILE, _MANIFEST_FILE),
        "P4.7 source preflight v3 preparation",
    )
    _validate_v4_source_plain_file_closure(
        v2_admission_dir,
        (_POWER_ADMISSION_CERTIFICATE_FILE, _MANIFEST_FILE),
        "P4.7 source preflight v2 power admission",
    )
    v4_root = _validate_v4_source_plain_file_closure(
        v4a_planning_dir,
        (
            _V4_PLANNING_FREEZE_FILE,
            _V4_CANDIDATE_SCHEDULE_FILE,
            _V4_SENTINEL_SCREEN_TABLE_FILE,
            _MANIFEST_FILE,
        ),
        "P4.7 source preflight v4a artifact",
    )
    raw = _load_v4_source_opportunity_preflight_protocol_raw(protocol_path)
    derived = _validate_v4_source_opportunity_derivation(raw)
    protocol = _v4_source_opportunity_protocol_view(raw, derived)
    v4_protocol, preparation, admission, v4_raw, v4_derived = _validated_v4_planning_inputs(
        v3_preparation_dir=v3_preparation_dir,
        v2_admission_dir=v2_admission_dir,
        protocol_path=None,
    )
    v4_freeze = _validate_v4_planning_root(
        v4_root,
        protocol=v4_protocol,
        preparation=preparation,
        admission=admission,
        raw=v4_raw,
        derived=v4_derived,
    )
    plan_bytes = (v4_root / _V4_PLANNING_FREEZE_FILE).read_bytes()
    screen_bytes = (v4_root / _V4_SENTINEL_SCREEN_TABLE_FILE).read_bytes()
    schedule_bytes = (v4_root / _V4_CANDIDATE_SCHEDULE_FILE).read_bytes()
    manifest_bytes = (v4_root / _MANIFEST_FILE).read_bytes()
    plan = _strict_json_object_from_bytes(plan_bytes, "P4.7 source preflight v4a plan")
    upstream = {
        "v4a_planning_protocol_id": v4_protocol.protocol_id,
        "v4a_planning_protocol_raw_sha256": P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1,
        "v4a_derivation_helper_raw_sha256": _sha256_bytes(_V4_PLANNING_DERIVATION_HELPER_PATH.read_bytes()),
        "v4a_planning_artifact_id": v4_freeze.artifact_id,
        "v4a_planning_certificate_id": _require_sha256(
            plan["certificate_id"],
            "P4.7 source preflight v4a certificate id",
        ),
        "v4a_plan_raw_sha256": _sha256_bytes(plan_bytes),
        "v4a_screen_table_raw_sha256": _sha256_bytes(screen_bytes),
        "v4a_candidate_schedule_raw_sha256": _sha256_bytes(schedule_bytes),
        "v4a_manifest_raw_sha256": _sha256_bytes(manifest_bytes),
        "v4a_artifact_sequence_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["artifact_sequence"],
        "v4a_scientific_units_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["scientific_units"],
        "v4a_source_grid_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["source_conditioned_cartesian_grid"],
        "v4a_source_preflight_section_sha256": _V4_PLANNING_PROTOCOL_SECTION_SHA256_V1["source_preflight_contract"],
        "scientific_v3_protocol_id": preparation.protocol_id,
        "scientific_v3_preparation_artifact_id": preparation.artifact_id,
        "power_admission_v2_artifact_id": admission.artifact_id,
        "relationship_action_registry_module": _RELATIONSHIP_ACTION_REGISTRY_MODULE,
        "relationship_action_registry_raw_sha256": (derived.action_registry.owner_module_raw_sha256),
        "relationship_action_choice_schema_id": derived.action_registry.schema_id,
        "relationship_action_choice_schema_raw_sha256": (derived.action_registry.schema_raw_sha256),
        "source_opportunity_derivation_helper_raw_sha256": _sha256_bytes(
            _V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH.read_bytes()
        ),
    }
    _require_literal(
        raw["input_lineage"],
        upstream,
        "P4.7 source preflight validated upstream lineage",
    )
    sequence = _require_mapping(v4_raw["artifact_sequence"], "P4.7 v4a artifact sequence")
    ordered_stages = sequence["ordered_stages"]
    if type(ordered_stages) is not list or ordered_stages[:3] != [
        "v4a_zero_output_planning_freeze",
        "source_opportunity_preflight",
        "tuple_feasibility_index",
    ]:
        raise ValueError("P4.7 source preflight v4a stage sequence drift")
    return protocol, raw, derived, upstream


def _v4_source_opportunity_mechanical_projection(
    derived: _V4SourceOpportunityDerived,
) -> dict[str, object]:
    derivation = _load_verified_v4_source_opportunity_derivation_helper()
    layout = derived.root_layout
    orientations = derived.fact_orientations
    generator = derived.planning_generator
    root_inventory_digest = derivation.canonical_mapping_digest(
        {
            "roots": [
                {
                    "split_id": root.split_id,
                    "root_role": root.root_role,
                    "namespace_ordinal": root.namespace_ordinal,
                    "role_ordinal": root.role_ordinal,
                    "global_slot": root.global_slot,
                    "surface_code": root.surface_code,
                    "factor_bits": list(root.factor_bits),
                    "typed_blueprint_values": list(root.typed_blueprint_values),
                }
                for root in layout.roots
            ]
        }
    )
    pair_inventory_digest = derivation.canonical_mapping_digest(
        {
            "analysis_donor_pairs": [
                {
                    "pair_ordinal": pair.pair_ordinal,
                    "split_id": pair.split_id,
                    "split_ordinal": pair.split_ordinal,
                    "analysis_global_slot": pair.analysis_global_slot,
                    "analysis_surface_code": pair.analysis_surface_code,
                    "donor_global_slot": pair.donor_global_slot,
                    "donor_surface_code": pair.donor_surface_code,
                }
                for pair in layout.analysis_donor_pairs
            ]
        }
    )
    twin_inventory_digest = derivation.canonical_mapping_digest(
        {
            "counterfactual_twins": [
                {
                    "mapping_ordinal": twin.mapping_ordinal,
                    "split_id": twin.split_id,
                    "split_ordinal": twin.split_ordinal,
                    "analysis_global_slot": twin.analysis_global_slot,
                    "analysis_surface_code": twin.analysis_surface_code,
                    "decisive_decision_index": twin.decisive_decision_index,
                    "reversal_pair_ordinal": twin.reversal_pair_ordinal,
                    "distance_bin_lower_bound_tokens": (twin.distance_bin_lower_bound_tokens),
                    "original_decisive_fact_value": twin.original_decisive_fact_value,
                    "counterfactual_decisive_fact_value": (twin.counterfactual_decisive_fact_value),
                    "utility_oracle_descendants_recomputed": (twin.utility_oracle_descendants_recomputed),
                    "all_other_exogenous_nodes_unchanged": (twin.all_other_exogenous_nodes_unchanged),
                    "independent_root": twin.independent_root,
                }
                for twin in layout.counterfactual_twin_mappings
            ]
        }
    )
    orientation_balance_digest = derivation.canonical_mapping_digest(
        {
            "formal_candidate_position_balances": [
                {
                    "formal_root_count": item.formal_root_count,
                    "reversal_pair_ordinal": item.reversal_pair_ordinal,
                    "decision_position": item.decision_position,
                    "fact_zero_count": item.fact_zero_count,
                    "fact_one_count": item.fact_one_count,
                }
                for item in orientations.formal_candidate_position_balances
            ]
        }
    )
    atom_inventory_digest = derivation.canonical_mapping_digest(
        {
            "atoms": [
                {
                    "atom_ordinal": atom.atom_ordinal,
                    "reference_success": atom.reference_success,
                    "comparator_successes": list(atom.comparator_successes),
                    "probability": _fraction_payload(atom.probability),
                    "utility_vector": list(atom.utility_vector),
                    "contrast_vector": list(atom.contrast_vector),
                }
                for atom in generator.atoms
            ]
        }
    )
    return {
        "root_surface_layout": {
            "surface_capacity": layout.surface_capacity,
            "affine_multiplier": layout.affine_multiplier,
            "affine_offset": layout.affine_offset,
            "surface_factor_axes_in_bit_order": list(layout.surface_factor_axes_in_bit_order),
            "surface_factor_typed_value_registry": [
                {
                    "axis_id": item.axis_id,
                    "value_0": item.value_zero,
                    "value_1": item.value_one,
                }
                for item in layout.surface_factor_typed_value_registry
            ],
            "namespaces": [
                {
                    "split_id": item.split_id,
                    "root_role": item.root_role,
                    "root_count": item.root_count,
                    "global_slot_start": item.global_slot_start,
                    "global_slot_stop_exclusive": item.global_slot_stop_exclusive,
                    "role_ordinal_start": item.role_ordinal_start,
                    "role_ordinal_stop_exclusive": item.role_ordinal_stop_exclusive,
                }
                for item in layout.namespaces
            ],
            "independent_root_slot_count": len(layout.roots),
            "analysis_root_count": layout.analysis_root_count,
            "donor_root_count": layout.donor_root_count,
            "root_inventory_digest_sha256": root_inventory_digest,
            "analysis_donor_pair_count": len(layout.analysis_donor_pairs),
            "analysis_donor_pair_inventory_digest_sha256": pair_inventory_digest,
            "counterfactual_twin_mapping_count": len(layout.counterfactual_twin_mappings),
            "counterfactual_twin_inventory_digest_sha256": twin_inventory_digest,
            "formal_candidate_prefixes": [
                {
                    "root_count": item.root_count,
                    "analysis_global_slot_start": item.analysis_global_slot_start,
                    "analysis_global_slot_stop_exclusive": (item.analysis_global_slot_stop_exclusive),
                    "donor_global_slot_start": item.donor_global_slot_start,
                    "donor_global_slot_stop_exclusive": (item.donor_global_slot_stop_exclusive),
                    "formal_pair_ordinal_start": item.formal_pair_ordinal_start,
                    "formal_pair_ordinal_stop_exclusive": (item.formal_pair_ordinal_stop_exclusive),
                }
                for item in layout.formal_candidate_prefixes
            ],
        },
        "action_and_opportunity_layout": {
            "action_registry_id": derived.action_registry.registry_id,
            "action_ids": list(derived.evaluation_design.action_order),
            "invalid_generated_action_id": (derived.evaluation_design.invalid_generated_action_id),
            "invalid_generated_action_in_registry": (derived.evaluation_design.invalid_generated_action_in_registry),
            "canonical_slot_templates": [
                {
                    "slot_ordinal": item.slot_ordinal,
                    "reversal_pair_ordinal": item.reversal_pair_ordinal,
                    "distance_bin_tokens": item.distance_bin_tokens,
                    "canonical_fact_value": item.fact_value,
                    "utility_vector": list(item.utility_vector),
                }
                for item in derived.evaluation_design.slots
            ],
            "canonical_slots_are_not_realized_root_orientations": True,
        },
        "balanced_fact_orientation": {
            "ranking_domain": orientations.ranking_domain,
            "ranking_seed": orientations.ranking_seed,
            "ranking_hash_algorithm": orientations.ranking_hash_algorithm,
            "ranking_payload_contract": orientations.ranking_payload_contract,
            "ranking_counter_fields": list(orientations.ranking_counter_fields),
            "ranking_tie_break_fields": list(orientations.ranking_tie_break_fields),
            "ranking_excluded_fields": list(orientations.ranking_excluded_fields),
            "analysis_root_count": orientations.analysis_root_count,
            "reversal_pair_count": orientations.reversal_pair_count,
            "assignment_count": len(orientations.assignments),
            "block_size": orientations.block_size,
            "block_commitment_count": len(orientations.block_commitments),
            "orientation_count_per_value_per_block": (orientations.orientation_count_per_value_per_block),
            "assignment_inventory_digest_sha256": (orientations.assignment_inventory_digest_sha256),
            "formal_candidate_position_balance_audit_count": len(orientations.formal_candidate_position_balances),
            "formal_candidate_position_balance_digest_sha256": (orientation_balance_digest),
        },
        "generic_decision_planning_generator": {
            "atom_count": len(generator.atoms),
            "atom_inventory_digest_sha256": atom_inventory_digest,
            "atom_mass_sum": _fraction_payload(sum((atom.probability for atom in generator.atoms), Fraction(0, 1))),
            "contrast_means": [_fraction_payload(value) for value in generator.contrast_means],
            "contrast_covariance_matrix": [
                [_fraction_payload(value) for value in row] for row in generator.contrast_covariance_matrix
            ],
            "contrast_correlation_matrix": [
                [_fraction_payload(value) for value in row] for row in generator.contrast_correlation_matrix
            ],
            "correlation_eigenvalue_multiplicities": [
                {
                    "eigenvalue": _fraction_payload(item.eigenvalue),
                    "multiplicity": item.multiplicity,
                }
                for item in generator.correlation_eigenvalues
            ],
            "temporal_joint_or_tuple_witness_derived": False,
        },
    }


def _v4_source_opportunity_contract_projection_core(
    *,
    protocol: RelationshipP4LongContextSourceOpportunityPreflightProtocol,
    raw: Mapping[str, Any],
    derived: _V4SourceOpportunityDerived,
    upstream: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_CONTRACT_SCHEMA_VERSION_V1,
        "contract_projection_id_contract": ("sha256_canonical_json_utf8_newline_without_contract_projection_id_v1"),
        "identity": {
            "source_preflight_protocol_id": protocol.protocol_id,
            "source_preflight_protocol_raw_sha256": (
                P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1
            ),
            "source_derivation_helper_raw_sha256": upstream["source_opportunity_derivation_helper_raw_sha256"],
            "v4a_planning_protocol_id": upstream["v4a_planning_protocol_id"],
            "v4a_planning_artifact_id": upstream["v4a_planning_artifact_id"],
            "v4a_planning_certificate_id": upstream["v4a_planning_certificate_id"],
            "scientific_v3_protocol_id": upstream["scientific_v3_protocol_id"],
            "scientific_v3_preparation_artifact_id": upstream["scientific_v3_preparation_artifact_id"],
            "power_admission_v2_artifact_id": upstream["power_admission_v2_artifact_id"],
            "action_registry_id": derived.action_registry.registry_id,
            "action_registry_owner_module": _RELATIONSHIP_ACTION_REGISTRY_MODULE,
            "action_registry_owner_module_raw_sha256": (derived.action_registry.owner_module_raw_sha256),
            "action_choice_schema_id": derived.action_registry.schema_id,
            "action_choice_schema_raw_sha256": (derived.action_registry.schema_raw_sha256),
        },
        "upstream_raw_lineage": dict(upstream),
        "stage_boundary": raw["stage_boundary"],
        "frozen_contract_sections": {
            "source_opportunity_unit": raw["source_opportunity_unit"],
            "sampling_frame_contract": raw["sampling_frame_contract"],
            "root_independence_and_capacity": raw["root_independence_and_capacity"],
            "opportunity_layout_and_utility_vectors": raw["opportunity_layout_and_utility_vectors"],
            "exact_source_planning_generator": raw["exact_source_planning_generator"],
            "truth_twin_and_leakage_firewall": raw["truth_twin_and_leakage_firewall"],
            "future_materialization_envelope": raw["future_materialization_envelope"],
        },
        "mechanical_projection": _v4_source_opportunity_mechanical_projection(derived),
        "zero_output_firewall": raw["zero_output_firewall"],
        "terminal": raw["terminal"],
        "claim_boundary": raw["claim_boundary"],
    }


def _v4_source_opportunity_preflight_certificate_core(
    *,
    protocol: RelationshipP4LongContextSourceOpportunityPreflightProtocol,
    raw: Mapping[str, Any],
    derived: _V4SourceOpportunityDerived,
    upstream: Mapping[str, Any],
    projection_id: str,
    projection_bytes: bytes,
) -> dict[str, object]:
    terminal = _require_mapping(raw["terminal"], "P4.7 source certificate terminal")
    stage = _require_mapping(raw["stage_boundary"], "P4.7 source certificate stage")
    return {
        "schema_version": (P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_SCHEMA_VERSION_V1),
        "certificate_id_contract": ("sha256_canonical_json_utf8_newline_without_certificate_id_v1"),
        "identity": {
            "source_preflight_protocol_id": protocol.protocol_id,
            "source_preflight_protocol_raw_sha256": (
                P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1
            ),
            "source_derivation_helper_raw_sha256": upstream["source_opportunity_derivation_helper_raw_sha256"],
            "v4a_planning_artifact_id": protocol.v4a_planning_artifact_id,
            "action_registry_id": derived.action_registry.registry_id,
            "contract_projection_id": projection_id,
            "contract_projection_raw_sha256": _sha256_bytes(projection_bytes),
            "contract_projection_byte_count": len(projection_bytes),
        },
        "validation_receipts": {
            "protocol_section_sha256": dict(_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SECTION_SHA256_V1),
            "protocol_and_helper_raw_pins_validated": True,
            "v4a_v3_and_v2_artifacts_independently_validated": True,
            "action_owner_module_and_choice_schema_validated_without_importing_runtime_owner": (True),
            "root_surface_layout_mechanically_rebuilt": True,
            "typed_blueprint_mapping_mechanically_rebuilt": True,
            "fact_orientation_and_all_candidate_prefix_balances_rebuilt": True,
            "semantic_stratum_latin_rotation_balances_rebuilt": True,
            "final_32K_counterfactual_twin_transform_rebuilt": True,
            "generic_decision_512_atom_distribution_and_exact_moments_rebuilt": (True),
            "generic_decision_atoms_accepted_as_temporal_or_tuple_witness": False,
            "external_publication_anchor_present": False,
            "ordinary_default_stream_validation_only": True,
        },
        "stage": {
            "zero_output_preflight_contract_frozen": stage["zero_output_preflight_contract_frozen_by_this_artifact"],
            "source_opportunity_stage_completed": stage["source_opportunity_stage_completed_by_this_artifact"],
            "source_structural_inventory_materialized": stage[
                "source_structural_inventory_materialized_by_this_artifact"
            ],
            "future_structural_inventory_scope_defined": stage[
                "future_structural_inventory_scope_is_defined_by_this_contract"
            ],
            "future_structural_inventory_materialization_authorized": stage[
                "future_single_create_only_structural_inventory_attempt_authorized_by_this_contract"
            ],
        },
        "zero_output_firewall": raw["zero_output_firewall"],
        "terminal": terminal,
        "status": terminal["status"],
        "claim_boundary": raw["claim_boundary"],
    }


def _v4_source_opportunity_preflight_manifest_core(
    *,
    protocol: RelationshipP4LongContextSourceOpportunityPreflightProtocol,
    raw: Mapping[str, Any],
    projection_id: str,
    projection_bytes: bytes,
    certificate_id: str,
    certificate_bytes: bytes,
) -> dict[str, object]:
    terminal = _require_mapping(raw["terminal"], "P4.7 source manifest terminal")
    return {
        "schema_version": (P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_MANIFEST_SCHEMA_VERSION_V1),
        "source_preflight_protocol_id": protocol.protocol_id,
        "v4a_planning_artifact_id": protocol.v4a_planning_artifact_id,
        "contract_projection_id": projection_id,
        "certificate_id": certificate_id,
        "status": terminal["status"],
        "files": [
            {
                "path": _V4_SOURCE_OPPORTUNITY_CONTRACT_FILE,
                "byte_count": len(projection_bytes),
                "sha256": _sha256_bytes(projection_bytes),
            },
            {
                "path": _V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE,
                "byte_count": len(certificate_bytes),
                "sha256": _sha256_bytes(certificate_bytes),
            },
        ],
        "zero_output_preflight_contract_frozen": True,
        "source_opportunity_stage_completed": False,
        "source_structural_inventory_materialized": False,
        "source_structural_inventory_artifact_id": None,
        "source_grid_resolved": False,
        "unresolved_tuple_count": 576,
        "selected_formal_root_count": None,
        "external_publication_anchor_present": False,
        "current_source_execution_authorized": False,
        "tuple_feasibility_authorized": False,
        "model_output_authorized": False,
        "development_authorized": False,
        "qualification_authorized": False,
        "formal_authorized": False,
        "cuda_planner_authorized": False,
        "source_opportunity_constraint_row_count": 0,
        "source_structural_inventory_artifact_count": 0,
        "subject_materialization_count": 0,
        "donor_bank_materialization_count": 0,
        "counterfactual_twin_materialization_count": 0,
        "planning_atom_materialization_count": 0,
        "model_output_count": 0,
        "cuda_planner_run_count": 0,
        "empirical_outcome_count": 0,
        "claim_boundary": (
            "Content-addressed zero-output source-opportunity contract only; "
            "external anchor, structural inventory, source rows, tuple feasibility, "
            "model, CUDA, and four-axis evidence remain absent."
        ),
    }


def _validate_v4_source_opportunity_preflight_root(
    output: pathlib.Path,
    *,
    protocol: RelationshipP4LongContextSourceOpportunityPreflightProtocol,
    raw: Mapping[str, Any],
    derived: _V4SourceOpportunityDerived,
    upstream: Mapping[str, Any],
) -> RelationshipP4LongContextSourceOpportunityPreflightCertificate:
    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 source preflight root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    expected_entries = tuple(
        sorted(
            (
                _V4_SOURCE_OPPORTUNITY_CONTRACT_FILE,
                _V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE,
                _MANIFEST_FILE,
            )
        )
    )
    if entries != expected_entries:
        raise ValueError("P4.7 source preflight file set drift")
    for name in entries:
        candidate = output / name
        _reject_reparse_components(candidate, "P4.7 source preflight artifact file")
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("P4.7 source preflight files must be regular files")
        if os.stat(candidate, follow_symlinks=False).st_nlink != 1:
            raise ValueError("P4.7 source preflight files must have one hard link")

    projection_path = output / _V4_SOURCE_OPPORTUNITY_CONTRACT_FILE
    projection_bytes = projection_path.read_bytes()
    projection = _strict_json_object_from_bytes(
        projection_bytes,
        "P4.7 source contract projection",
    )
    if projection_bytes != _canonical_bytes(projection):
        raise ValueError("P4.7 source contract projection is not canonical JSON")
    projection_id = _require_sha256(
        projection.get("contract_projection_id"),
        "P4.7 source contract projection id",
    )
    projection_core = dict(projection)
    del projection_core["contract_projection_id"]
    if projection_id != _sha256_bytes(_canonical_bytes(projection_core)):
        raise ValueError("P4.7 source contract projection id drift")
    _require_literal(
        projection_core,
        _v4_source_opportunity_contract_projection_core(
            protocol=protocol,
            raw=raw,
            derived=derived,
            upstream=upstream,
        ),
        "P4.7 source contract projection",
    )

    certificate_path = output / _V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE
    certificate_bytes = certificate_path.read_bytes()
    certificate = _strict_json_object_from_bytes(
        certificate_bytes,
        "P4.7 source preflight certificate",
    )
    if certificate_bytes != _canonical_bytes(certificate):
        raise ValueError("P4.7 source preflight certificate is not canonical JSON")
    certificate_id = _require_sha256(
        certificate.get("certificate_id"),
        "P4.7 source preflight certificate id",
    )
    certificate_core = dict(certificate)
    del certificate_core["certificate_id"]
    if certificate_id != _sha256_bytes(_canonical_bytes(certificate_core)):
        raise ValueError("P4.7 source preflight certificate id drift")
    _require_literal(
        certificate_core,
        _v4_source_opportunity_preflight_certificate_core(
            protocol=protocol,
            raw=raw,
            derived=derived,
            upstream=upstream,
            projection_id=projection_id,
            projection_bytes=projection_bytes,
        ),
        "P4.7 source preflight certificate",
    )
    if (
        P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1 is not None
        and certificate_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1
    ):
        raise ValueError("P4.7 published source preflight certificate id drift")

    manifest_path = output / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_json_object_from_bytes(
        manifest_bytes,
        "P4.7 source preflight manifest",
    )
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("P4.7 source preflight manifest is not canonical JSON")
    artifact_id = _require_sha256(
        manifest.get("artifact_id"),
        "P4.7 source preflight artifact id",
    )
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4.7 source preflight artifact id drift")
    _require_literal(
        manifest_core,
        _v4_source_opportunity_preflight_manifest_core(
            protocol=protocol,
            raw=raw,
            projection_id=projection_id,
            projection_bytes=projection_bytes,
            certificate_id=certificate_id,
            certificate_bytes=certificate_bytes,
        ),
        "P4.7 source preflight manifest",
    )
    if (
        P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1 is not None
        and artifact_id != P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1
    ):
        raise ValueError("P4.7 published source preflight artifact id drift")

    terminal = _require_mapping(
        certificate["terminal"],
        "P4.7 source preflight certificate terminal",
    )
    stage = _require_mapping(
        certificate["stage"],
        "P4.7 source preflight certificate stage",
    )
    identity = _require_mapping(
        certificate["identity"],
        "P4.7 source preflight certificate identity",
    )
    return RelationshipP4LongContextSourceOpportunityPreflightCertificate(
        artifact_id=artifact_id,
        certificate_id=certificate_id,
        contract_projection_id=projection_id,
        protocol_id=protocol.protocol_id,
        v4a_planning_artifact_id=protocol.v4a_planning_artifact_id,
        action_registry_id=_require_sha256(
            identity["action_registry_id"],
            "P4.7 source preflight action registry id",
        ),
        status=_require_text(terminal["status"], "P4.7 source preflight status"),
        zero_output_preflight_contract_frozen=_require_bool(
            stage["zero_output_preflight_contract_frozen"],
            "P4.7 source preflight contract frozen",
        ),
        source_opportunity_stage_completed=_require_bool(
            stage["source_opportunity_stage_completed"],
            "P4.7 source opportunity stage completed",
        ),
        source_structural_inventory_materialized=_require_bool(
            stage["source_structural_inventory_materialized"],
            "P4.7 source inventory materialized",
        ),
        unresolved_tuple_count=_require_int(
            terminal["unresolved_tuple_count"],
            "P4.7 source unresolved tuple count",
        ),
        selected_formal_root_count=terminal["selected_formal_root_count"],
        current_source_execution_authorized=_require_bool(
            terminal["current_source_execution_authorized"],
            "P4.7 current source execution authorization",
        ),
        tuple_feasibility_authorized=_require_bool(
            terminal["tuple_feasibility_authorized"],
            "P4.7 tuple feasibility authorization",
        ),
        model_output_authorized=_require_bool(
            terminal["model_output_authorized"],
            "P4.7 model output authorization",
        ),
        development_authorized=_require_bool(
            terminal["development_authorized"],
            "P4.7 development authorization",
        ),
        qualification_authorized=_require_bool(
            terminal["qualification_authorized"],
            "P4.7 qualification authorization",
        ),
        formal_authorized=_require_bool(
            terminal["formal_authorized"],
            "P4.7 formal authorization",
        ),
        cuda_planner_authorized=_require_bool(
            terminal["CUDA_planner_authorized"],
            "P4.7 CUDA planner authorization",
        ),
        output_dir=output,
    )


def _load_v4_external_anchor_request_protocol_raw(
    path: pathlib.Path | None = None,
) -> Mapping[str, Any]:
    protocol_path = _require_local_default_stream_path(
        _V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_PATH_V1 if path is None else path,
        "P4.7 A0 request protocol",
    )
    _reject_reparse_components(protocol_path, "P4.7 A0 request protocol")
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise FileNotFoundError("P4.7 A0 request protocol is missing")
    if os.stat(protocol_path, follow_symlinks=False).st_nlink != 1:
        raise ValueError("P4.7 A0 request protocol must have exactly one hard link")
    raw_bytes = protocol_path.read_bytes()
    raw = _strict_json_object_from_bytes(raw_bytes, "P4.7 A0 request protocol")
    _require_exact_keys(raw, _V4_EXTERNAL_ANCHOR_REQUEST_TOP_LEVEL_KEYS, "P4.7 A0 request protocol")
    _require_literal(
        raw["schema_version"],
        P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SCHEMA_VERSION_V1,
        "P4.7 A0 request schema",
    )
    _require_literal(
        raw["protocol_id_contract"],
        "sha256_canonical_json_utf8_newline_v1",
        "P4.7 A0 request protocol id contract",
    )
    if _sha256_bytes(_canonical_bytes(raw)) != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1:
        raise ValueError("P4.7 A0 request protocol id drift")
    if _sha256_bytes(raw_bytes) != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_RAW_SHA256_V1:
        raise ValueError("P4.7 A0 request protocol raw bytes drift")
    for section, expected_hash in _V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SECTION_SHA256_V1.items():
        if _sha256_bytes(_canonical_bytes(raw[section])) != expected_hash:
            raise ValueError(f"P4.7 A0 frozen protocol section drift: {section}")
    _validate_v4_external_anchor_request_protocol_semantics(raw)
    return raw


def _validate_v4_external_anchor_request_protocol_semantics(raw: Mapping[str, Any]) -> None:
    _require_literal(
        raw["owner"],
        {
            "wheel": "lifeform-evolution",
            "module": "lifeform_evolution.relationship_lab_p4_long_context_causal_campaign",
            "data_owner": "relationship_p4_long_context_external_publication_anchor",
            "wiring_level": "OFFLINE_READOUT_ONLY",
            "runtime_slot_registered": False,
            "second_artifact_publisher_created": False,
        },
        "P4.7 A0 owner",
    )
    lineage = _require_mapping(raw["input_lineage"], "P4.7 A0 lineage")
    _require_literal(
        lineage,
        {
            "source_preflight_protocol_id": P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1,
            "source_preflight_protocol_raw_sha256": (
                P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1
            ),
            "source_preflight_protocol_byte_count": 34883,
            "source_derivation_helper_raw_sha256": ("72efc093b815c2ca07872f6cb6a78f53a4d4d5ada5975222b36cf90c640746f8"),
            "source_derivation_helper_byte_count": 59810,
            "source_preflight_artifact_id": P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1,
            "source_preflight_contract_projection_id": (
                "b8b7823a6fd2c7ad706c4ffa143438b730da667c26a925f0be87df14212e6f1b"
            ),
            "source_preflight_certificate_id": (P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1),
            "source_preflight_projection_raw_sha256": (
                "ee33fa32a3829cbaaa1c92022016c184197b4cd97e08818af19b98024b4866b2"
            ),
            "source_preflight_projection_byte_count": 72969,
            "source_preflight_certificate_raw_sha256": (
                "f9089ce08e6868d402a753ccd3247024a0170a96e8cad9f411f659e913300736"
            ),
            "source_preflight_certificate_byte_count": 8036,
            "source_preflight_manifest_raw_sha256": (
                "2829b16d674ae9efe971eaa668610f80a20799e45ac47e208ec4e7a3261760a6"
            ),
            "source_preflight_manifest_byte_count": 2036,
            "v4a_planning_artifact_id": P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1,
            "relationship_action_registry_id": ("5b6250960c43401d7a14f463f0cc32c7518735c0aba6bf0e855e0e55f8a45fcc"),
        },
        "P4.7 A0 input lineage",
    )
    anchor_stage = _require_mapping(raw["anchor_stage"], "P4.7 A0 stage")
    if (
        anchor_stage["stage_id"] != "A0_source_opportunity_contract_before_materializer_implementation"
        or anchor_stage["A0_request_freeze_precedes_publication"] is not True
        or anchor_stage["A0_publication_precedes_materializer_implementation"] is not True
        or anchor_stage["A1_is_required_after_materializer_implementation_and_before_materialization"] is not True
        or anchor_stage["A0_receipt_alone_may_authorize_materialization"] is not False
        or anchor_stage["request_id_seed_or_remote_identity_may_affect_source_generation"] is not False
    ):
        raise ValueError("P4.7 A0 stage boundary drift")
    subjects = _require_mapping(raw["publication_subject_contract"], "P4.7 A0 subject contract")
    ordered_subjects = subjects["ordered_subjects"]
    if type(ordered_subjects) is not list or len(ordered_subjects) != 5 or subjects["subject_count"] != 5:
        raise ValueError("P4.7 A0 publication subject inventory drift")
    seen_paths: set[str] = set()
    seen_casefolded: set[str] = set()
    for index, item in enumerate(ordered_subjects):
        subject = _require_mapping(item, f"P4.7 A0 subject {index}")
        _require_exact_keys(
            subject,
            {
                "role",
                "repo_relative_posix_path",
                "byte_count",
                "raw_sha256",
                "expected_git_blob_oid_sha1",
                "expected_git_mode",
                "expected_git_object_type",
                "semantic_identity_kind",
                "semantic_identity",
            },
            f"P4.7 A0 subject {index}",
        )
        path = _require_safe_repo_relative_posix_path(
            subject["repo_relative_posix_path"],
            f"P4.7 A0 subject path {index}",
        )
        if path in seen_paths or path.casefold() in seen_casefolded:
            raise ValueError("P4.7 A0 subject paths collide")
        seen_paths.add(path)
        seen_casefolded.add(path.casefold())
        _require_int(subject["byte_count"], f"P4.7 A0 subject bytes {index}")
        _require_sha256(subject["raw_sha256"], f"P4.7 A0 subject SHA-256 {index}")
        _require_lower_hex(subject["expected_git_blob_oid_sha1"], 40, f"P4.7 A0 Git blob {index}")
        if subject["expected_git_mode"] != "100644" or subject["expected_git_object_type"] != "blob":
            raise ValueError("P4.7 A0 subject Git object contract drift")
    target = _require_mapping(raw["publication_target_contract"], "P4.7 A0 publication target")
    if (
        target["provider"] != "github_public_gist_first_revision_v1"
        or target["provider_host"] != "github.com"
        or target["api_host"] != "api.github.com"
        or target["raw_host"] != "gist.githubusercontent.com"
        or target["expected_owner_login"] != "ronaldzgithub"
        or target["required_description"] != ""
        or target["required_visibility"] != "public"
        or target["HTTPS_required"] is not True
        or target["URL_userinfo_forbidden"] is not True
        or target["nondefault_port_forbidden"] is not True
        or target["unauthenticated_read_required"] is not True
        or target["new_gist_required"] is not True
        or target["first_revision_required"] is not True
        or target["required_parent_count"] != 0
        or target["required_exact_file_count"] != 1
        or target["full_lowercase_40_hex_revision_oid_required"] is not True
        or target["mutable_latest_URL_is_authority"] is not False
        or target["current_private_origin_is_acceptable_as_publication_anchor"] is not False
    ):
        raise ValueError("P4.7 A0 public Gist target drift")
    for field in ("actual_gist_id", "actual_revision_oid", "actual_raw_permalink", "actual_HTML_permalink"):
        if target[field] is not None:
            raise ValueError(f"P4.7 A0 request prefilled future remote identity: {field}")
    binding = _require_mapping(raw["self_publication_binding"], "P4.7 A0 self-publication binding")
    for field in ("request_payload_repo_relative_path", "request_manifest_repo_relative_path"):
        _require_safe_repo_relative_posix_path(binding[field], f"P4.7 A0 {field}")
    upstream_roots = _require_mapping(
        binding["prepare_upstream_repo_relative_roots"],
        "P4.7 A0 canonical upstream roots",
    )
    _require_exact_keys(
        upstream_roots,
        frozenset(("source_preflight", "v4a_planning", "v3_preparation", "v2_admission")),
        "P4.7 A0 canonical upstream roots",
    )
    for field, value in upstream_roots.items():
        _require_safe_repo_relative_posix_path(value, f"P4.7 A0 canonical upstream root {field}")
    if (
        binding["prepare_must_read_request_and_all_publication_subjects_from_their_frozen_canonical_repository_paths"]
        is not True
        or binding["prepare_must_read_all_upstream_lineage_artifacts_from_their_frozen_canonical_repository_roots"]
        is not True
        or binding["validation_of_byte_identical_relocated_replicas_does_not_rebind_canonical_paths"] is not True
        or binding["public_Gist_content_must_equal_request_payload_default_stream_bytes"] is not True
        or binding["request_manifest_must_bind_the_published_payload_raw_SHA256_and_byte_count"] is not True
        or binding["actual_publication_revision_is_intentionally_unknown_before_publication"] is not True
        or binding["future_revision_oid_must_not_be_embedded_in_the_request_payload"] is not True
    ):
        raise ValueError("P4.7 A0 self-publication binding drift")
    receipt = _require_mapping(raw["future_receipt_requirements"], "P4.7 A0 receipt requirements")
    _require_literal(
        receipt["schema_version"],
        "relationship-p4-long-context-github-public-gist-anchor-receipt.v1",
        "P4.7 A0 future receipt schema",
    )
    for field in (
        "separate_create_only_artifact_required",
        "independent_observer_process_required",
        "request_artifact_id_required",
        "request_payload_raw_SHA256_and_byte_count_required",
        "GitHub_gist_id_node_id_owner_and_public_visibility_required",
        "observer_requests_must_send_no_Authorization_header_and_no_Cookie_header",
        "publisher_creation_authentication_requires_separate_explicit_user_authority",
        "publisher_credentials_headers_cookies_and_tokens_must_never_be_serialized",
        "all_observed_HTTP_URLs_must_use_HTTPS_without_userinfo_or_nondefault_port",
        "required_empty_Gist_description_must_be_observed",
        "API_version_final_URL_HTTP_status_Date_ETag_and_request_id_recorded",
        "HTTP_Date_ETag_request_id_and_Git_commit_time_are_observation_metadata_not_trusted_time",
        "exact_first_revision_oid_and_zero_parents_required",
        "exact_one_file_filename_size_raw_permalink_and_raw_SHA256_required",
        "same_gist_owner_id_revision_and_filename_identity_join_required",
        "revision_commit_must_be_loaded_from_the_same_gist_and_have_zero_parents",
        "revision_tree_must_have_exactly_one_entry_with_mode_100644_and_required_filename",
        "tree_entry_blob_OID_must_equal_Git_blob_SHA1_of_observed_request_bytes",
        "revision_pinned_raw_final_URL_API_identity_and_HTML_identity_must_resolve_to_the_same_gist_revision_and_filename",
        "observed_request_bytes_must_equal_local_request_payload_bytes_with_exact_SHA256_and_byte_count",
        "redirect_outside_the_frozen_host_allowlist_rejected",
        "HTTP_403_404_429_5xx_timeout_or_network_failure_is_NO_GO",
        "fresh_unauthenticated_online_reobservation_required_before_A0_admission",
    ):
        if receipt[field] is not True:
            raise ValueError(f"P4.7 A0 future receipt requirement drift: {field}")
    if receipt["receipt_alone_may_authorize_source_materialization"] is not False:
        raise ValueError("P4.7 A0 future receipt boundary drift")
    authorization = _require_mapping(raw["authorization_firewall"], "P4.7 A0 authorization firewall")
    if authorization["publication_request_contract_frozen"] is not True:
        raise ValueError("P4.7 A0 request contract is not frozen")
    for field, value in authorization.items():
        if field == "publication_request_contract_frozen":
            continue
        if value is not False:
            raise ValueError(f"P4.7 A0 authorization opened: {field}")
    zero = _require_mapping(raw["zero_output_firewall"], "P4.7 A0 zero-output firewall")
    for field, value in zero.items():
        if field.endswith("_count") and value != 0:
            raise ValueError(f"P4.7 A0 nonzero output count: {field}")
        if field.endswith("_claimed") and value is not False:
            raise ValueError(f"P4.7 A0 overclaim opened: {field}")
        if field.endswith("_supported") and value is not False:
            raise ValueError(f"P4.7 A0 four-axis claim opened: {field}")
    for field in (
        "all_materialization_counts_refer_to_persisted_or_published_artifact_rows_not_ephemeral_in_memory_exact_derivation_objects",
        "ephemeral_in_memory_structural_derivation_objects_are_not_source_content_or_persistent_materialization",
    ):
        if zero[field] is not True:
            raise ValueError(f"P4.7 A0 materialization semantics drift: {field}")
    terminal = _require_mapping(raw["terminal"], "P4.7 A0 terminal")
    if (
        terminal["status"] != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_STATUS_V1
        or terminal["publication_request_contract_frozen"] is not True
        or terminal["external_publication_anchor_present"] is not False
        or terminal["A1_required_before_materialization"] is not True
        or terminal["structural_inventory_materialization_authorized"] is not False
    ):
        raise ValueError("P4.7 A0 terminal drift")
    for field in (
        "publication_request_artifact_id",
        "GitHub_gist_id",
        "GitHub_revision_oid",
        "external_receipt_artifact_id",
        "A0_admission_artifact_id",
    ):
        if terminal[field] is not None:
            raise ValueError(f"P4.7 A0 terminal prefilled future identity: {field}")


def _require_safe_repo_relative_posix_path(value: object, label: str) -> str:
    text = _require_text(value, label)
    path = pathlib.PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
        or ":" in text
        or "%" in text
    ):
        raise ValueError(f"{label} must be an unambiguous repository-relative POSIX path")
    return text


def _v4_external_anchor_request_canonical_output(raw: Mapping[str, Any]) -> pathlib.Path:
    binding = _require_mapping(raw["self_publication_binding"], "P4.7 A0 self-publication binding")
    request_path = pathlib.PurePosixPath(
        _require_safe_repo_relative_posix_path(
            binding["request_payload_repo_relative_path"],
            "P4.7 A0 request payload path",
        )
    )
    manifest_path = pathlib.PurePosixPath(
        _require_safe_repo_relative_posix_path(
            binding["request_manifest_repo_relative_path"],
            "P4.7 A0 request manifest path",
        )
    )
    if request_path.name != _V4_EXTERNAL_ANCHOR_REQUEST_FILE:
        raise ValueError("P4.7 A0 request payload filename drift")
    if manifest_path.name != _MANIFEST_FILE:
        raise ValueError("P4.7 A0 request manifest filename drift")
    if request_path.parent != manifest_path.parent:
        raise ValueError("P4.7 A0 request payload and manifest roots diverge")
    return _absolute_without_resolving(_REPOSITORY_ROOT.joinpath(*request_path.parent.parts))


def _v4_external_anchor_canonical_upstream_roots(
    raw: Mapping[str, Any],
) -> Mapping[str, pathlib.Path]:
    binding = _require_mapping(raw["self_publication_binding"], "P4.7 A0 self-publication binding")
    roots = _require_mapping(
        binding["prepare_upstream_repo_relative_roots"],
        "P4.7 A0 canonical upstream roots",
    )
    return MappingProxyType(
        {
            field: _absolute_without_resolving(
                _REPOSITORY_ROOT.joinpath(
                    *pathlib.PurePosixPath(
                        _require_safe_repo_relative_posix_path(
                            value,
                            f"P4.7 A0 canonical upstream root {field}",
                        )
                    ).parts
                )
            )
            for field, value in roots.items()
        }
    )


def _require_lower_hex(value: object, length: int, label: str) -> str:
    text = _require_text(value, label)
    if len(text) != length or any(character not in _HEX_DIGITS for character in text):
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def _git_blob_oid_sha1(payload: bytes) -> str:
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _v4_external_anchor_subject_source_paths(
    source_preflight_dir: pathlib.Path,
) -> tuple[pathlib.Path, ...]:
    source_root = _absolute_without_resolving(source_preflight_dir)
    return (
        _V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_PATH_V1,
        _V4_SOURCE_OPPORTUNITY_DERIVATION_HELPER_PATH,
        source_root / _V4_SOURCE_OPPORTUNITY_CONTRACT_FILE,
        source_root / _V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE,
        source_root / _MANIFEST_FILE,
    )


def _require_v4_external_anchor_canonical_subject_sources(
    *,
    raw: Mapping[str, Any],
    source_preflight_dir: pathlib.Path,
) -> None:
    subject_contract = _require_mapping(
        raw["publication_subject_contract"],
        "P4.7 A0 subject contract",
    )
    expected_items = subject_contract["ordered_subjects"]
    actual_paths = _v4_external_anchor_subject_source_paths(source_preflight_dir)
    if type(expected_items) is not list or len(expected_items) != len(actual_paths):
        raise ValueError("P4.7 A0 subject source cardinality drift")
    for index, (expected_value, actual_path) in enumerate(zip(expected_items, actual_paths, strict=True)):
        expected = _require_mapping(expected_value, f"P4.7 A0 subject {index}")
        relative = pathlib.PurePosixPath(
            _require_safe_repo_relative_posix_path(
                expected["repo_relative_posix_path"],
                f"P4.7 A0 subject path {index}",
            )
        )
        canonical_path = _absolute_without_resolving(_REPOSITORY_ROOT.joinpath(*relative.parts))
        if _absolute_without_resolving(actual_path) != canonical_path:
            raise ValueError(
                "P4.7 A0 request preparation must read each publication subject from its "
                f"frozen canonical repository path: {canonical_path}"
            )


def _validated_v4_external_anchor_request_inputs(
    *,
    source_preflight_dir: pathlib.Path,
    v4a_planning_dir: pathlib.Path,
    v3_preparation_dir: pathlib.Path,
    v2_admission_dir: pathlib.Path,
    protocol_path: pathlib.Path | None,
) -> tuple[
    RelationshipP4LongContextExternalAnchorRequestProtocol,
    Mapping[str, Any],
    RelationshipP4LongContextSourceOpportunityPreflightCertificate,
    tuple[Mapping[str, Any], ...],
]:
    raw = _load_v4_external_anchor_request_protocol_raw(protocol_path)
    protocol = _v4_external_anchor_request_protocol_view(raw)
    source_certificate = validate_relationship_p4_long_context_source_opportunity_preflight(
        output_dir=source_preflight_dir,
        v4a_planning_dir=v4a_planning_dir,
        v3_preparation_dir=v3_preparation_dir,
        v2_admission_dir=v2_admission_dir,
    )
    source_root = _validate_v4_source_plain_file_closure(
        source_preflight_dir,
        (
            _V4_SOURCE_OPPORTUNITY_CONTRACT_FILE,
            _V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_FILE,
            _MANIFEST_FILE,
        ),
        "P4.7 A0 source preflight artifact",
    )
    subject_sources = _v4_external_anchor_subject_source_paths(source_root)
    expected_items = _require_mapping(
        raw["publication_subject_contract"],
        "P4.7 A0 subject contract",
    )["ordered_subjects"]
    if type(expected_items) is not list or len(expected_items) != len(subject_sources):
        raise ValueError("P4.7 A0 subject source cardinality drift")
    actual_items: list[Mapping[str, Any]] = []
    for index, (expected_value, source_path) in enumerate(zip(expected_items, subject_sources, strict=True)):
        expected = _require_mapping(expected_value, f"P4.7 A0 subject {index}")
        _validate_v4_source_plain_file(source_path, f"P4.7 A0 subject file {index}")
        payload = source_path.read_bytes()
        if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("P4.7 A0 subject must not be a Git LFS pointer")
        actual = {
            **expected,
            "byte_count": len(payload),
            "raw_sha256": _sha256_bytes(payload),
            "expected_git_blob_oid_sha1": _git_blob_oid_sha1(payload),
        }
        _require_literal(actual, expected, f"P4.7 A0 subject bytes {index}")
        actual_items.append(MappingProxyType(dict(actual)))
    lineage = _require_mapping(raw["input_lineage"], "P4.7 A0 lineage")
    if (
        source_certificate.artifact_id != lineage["source_preflight_artifact_id"]
        or source_certificate.certificate_id != lineage["source_preflight_certificate_id"]
        or source_certificate.contract_projection_id != lineage["source_preflight_contract_projection_id"]
        or source_certificate.v4a_planning_artifact_id != lineage["v4a_planning_artifact_id"]
        or source_certificate.action_registry_id != lineage["relationship_action_registry_id"]
    ):
        raise ValueError("P4.7 A0 validated source certificate lineage drift")
    return protocol, raw, source_certificate, tuple(actual_items)


def _v4_external_anchor_request_core(
    *,
    protocol: RelationshipP4LongContextExternalAnchorRequestProtocol,
    raw: Mapping[str, Any],
    source_certificate: RelationshipP4LongContextSourceOpportunityPreflightCertificate,
    subjects: tuple[Mapping[str, Any], ...],
) -> dict[str, object]:
    return {
        "schema_version": P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_SCHEMA_VERSION_V1,
        "request_id_contract": "sha256_canonical_json_utf8_newline_without_request_id_v1",
        "frozen_at_utc": protocol.frozen_at_utc,
        "identity": {
            "anchor_request_protocol_id": protocol.protocol_id,
            "anchor_request_protocol_raw_sha256": (P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_RAW_SHA256_V1),
            "source_preflight_protocol_id": protocol.source_preflight_protocol_id,
            "source_preflight_artifact_id": source_certificate.artifact_id,
            "source_preflight_contract_projection_id": source_certificate.contract_projection_id,
            "source_preflight_certificate_id": source_certificate.certificate_id,
        },
        "anchor_stage": raw["anchor_stage"],
        "publication_subjects": [dict(subject) for subject in subjects],
        "publication_target": raw["publication_target_contract"],
        "self_publication_binding": raw["self_publication_binding"],
        "future_receipt_requirements": raw["future_receipt_requirements"],
        "authorization_firewall": raw["authorization_firewall"],
        "zero_output_firewall": raw["zero_output_firewall"],
        "terminal": raw["terminal"],
        "claim_boundary": protocol.claim_boundary,
    }


def _v4_external_anchor_request_manifest_core(
    *,
    protocol: RelationshipP4LongContextExternalAnchorRequestProtocol,
    request_id: str,
    request_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_MANIFEST_SCHEMA_VERSION_V1,
        "anchor_request_protocol_id": protocol.protocol_id,
        "request_id": request_id,
        "status": protocol.status,
        "files": [
            {
                "path": _V4_EXTERNAL_ANCHOR_REQUEST_FILE,
                "byte_count": len(request_bytes),
                "sha256": _sha256_bytes(request_bytes),
            }
        ],
        "publication_request_contract_frozen": True,
        "external_request_dispatched": False,
        "publication_performed": False,
        "external_publication_anchor_present": False,
        "external_anchor_admitted": False,
        "structural_inventory_materialization_authorized": False,
        "source_execution_authorized": False,
        "tuple_feasibility_authorized": False,
        "model_output_authorized": False,
        "CUDA_planner_authorized": False,
        "network_request_count": 0,
        "Git_commit_count": 0,
        "Git_push_count": 0,
        "source_structural_inventory_artifact_count": 0,
        "model_output_count": 0,
        "CUDA_run_count": 0,
        "empirical_outcome_count": 0,
        "claim_boundary": (
            "Local A0 publication request only; no external observation, admission, source, model, "
            "CUDA, or four-axis evidence exists."
        ),
    }


def _validate_v4_external_anchor_request_root(
    output: pathlib.Path,
    *,
    protocol: RelationshipP4LongContextExternalAnchorRequestProtocol,
    raw: Mapping[str, Any],
    source_certificate: RelationshipP4LongContextSourceOpportunityPreflightCertificate,
    subjects: tuple[Mapping[str, Any], ...],
) -> RelationshipP4LongContextExternalAnchorRequest:
    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 A0 request root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    if entries != tuple(sorted((_V4_EXTERNAL_ANCHOR_REQUEST_FILE, _MANIFEST_FILE))):
        raise ValueError("P4.7 A0 request file set drift")
    for name in entries:
        candidate = output / name
        _validate_v4_source_plain_file(candidate, "P4.7 A0 request artifact file")
    request_path = output / _V4_EXTERNAL_ANCHOR_REQUEST_FILE
    request_bytes = request_path.read_bytes()
    request = _strict_json_object_from_bytes(request_bytes, "P4.7 A0 request payload")
    if request_bytes != _canonical_bytes(request):
        raise ValueError("P4.7 A0 request payload is not canonical JSON")
    request_id = _require_sha256(request.get("request_id"), "P4.7 A0 request id")
    request_core = dict(request)
    del request_core["request_id"]
    if request_id != _sha256_bytes(_canonical_bytes(request_core)):
        raise ValueError("P4.7 A0 request id drift")
    _require_literal(
        request_core,
        _v4_external_anchor_request_core(
            protocol=protocol,
            raw=raw,
            source_certificate=source_certificate,
            subjects=subjects,
        ),
        "P4.7 A0 request payload",
    )
    if P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1 is not None and (
        request_id != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1
    ):
        raise ValueError("P4.7 published A0 request id drift")
    manifest_path = output / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_json_object_from_bytes(manifest_bytes, "P4.7 A0 request manifest")
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("P4.7 A0 request manifest is not canonical JSON")
    artifact_id = _require_sha256(manifest.get("artifact_id"), "P4.7 A0 request artifact id")
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4.7 A0 request artifact id drift")
    _require_literal(
        manifest_core,
        _v4_external_anchor_request_manifest_core(
            protocol=protocol,
            request_id=request_id,
            request_bytes=request_bytes,
        ),
        "P4.7 A0 request manifest",
    )
    if P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1 is not None and (
        artifact_id != P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1
    ):
        raise ValueError("P4.7 published A0 request artifact id drift")
    firewall = _require_mapping(request["authorization_firewall"], "P4.7 A0 request firewall")
    return RelationshipP4LongContextExternalAnchorRequest(
        artifact_id=artifact_id,
        request_id=request_id,
        protocol_id=protocol.protocol_id,
        status=protocol.status,
        publication_request_contract_frozen=_require_bool(
            firewall["publication_request_contract_frozen"],
            "P4.7 A0 request frozen",
        ),
        external_request_dispatched=_require_bool(
            firewall["external_request_dispatched"],
            "P4.7 A0 request dispatched",
        ),
        publication_performed=_require_bool(
            firewall["publication_performed"],
            "P4.7 A0 publication performed",
        ),
        external_publication_anchor_present=_require_bool(
            firewall["external_publication_anchor_present"],
            "P4.7 A0 external anchor present",
        ),
        external_anchor_admitted=_require_bool(
            firewall["external_anchor_admitted"],
            "P4.7 A0 anchor admitted",
        ),
        structural_inventory_materialization_authorized=_require_bool(
            firewall["structural_inventory_materialization_authorized"],
            "P4.7 A0 materialization authorized",
        ),
        source_execution_authorized=_require_bool(
            firewall["source_execution_authorized"],
            "P4.7 A0 source execution authorized",
        ),
        tuple_feasibility_authorized=_require_bool(
            firewall["tuple_feasibility_authorized"],
            "P4.7 A0 tuple feasibility authorized",
        ),
        model_output_authorized=_require_bool(
            firewall["model_output_authorized"],
            "P4.7 A0 model output authorized",
        ),
        cuda_planner_authorized=_require_bool(
            firewall["CUDA_planner_authorized"],
            "P4.7 A0 CUDA authorized",
        ),
        output_dir=output,
    )


def _preparation_payload(
    protocol: RelationshipP4LongContextScientificPrereg,
) -> dict[str, object]:
    descriptor = _protocol_descriptor(protocol.protocol_id)
    payload: dict[str, object] = {
        "schema_version": descriptor.preparation_schema,
        "protocol_id": protocol.protocol_id,
        "frozen_at_utc": protocol.frozen_at_utc,
        "status": descriptor.status,
        "scientific_design_frozen": True,
        "execution_envelope_present": False,
        "execution_enabled": False,
        "formal_run_authorized": False,
        "model_output_count": 0,
        "subject_pack_materialization_count": 0,
        "cuda_formal_execution_count": 0,
        "formal_result_count": 0,
        "development_cuda_diagnostics_allowed": True,
        "development_cuda_diagnostics_are_formal_evidence": False,
        "next_required_artifact": ("content_addressed_execution_envelope_without_scientific_design_changes"),
        "claim_boundary": protocol.claim_boundary,
    }
    if descriptor.version == "v3":
        payload.update(
            {
                "donor_bank_materialization_count": 0,
                "counterfactual_twin_materialization_count": 0,
                "power_dgp_artifact_count": 0,
            }
        )
    return payload


def _manifest_core(
    protocol: RelationshipP4LongContextScientificPrereg,
    preparation_bytes: bytes,
) -> dict[str, object]:
    descriptor = _protocol_descriptor(protocol.protocol_id)
    payload: dict[str, object] = {
        "schema_version": descriptor.manifest_schema,
        "protocol_id": protocol.protocol_id,
        "status": descriptor.status,
        "files": [
            {
                "path": _PREPARATION_FILE,
                "byte_count": len(preparation_bytes),
                "sha256": _sha256_bytes(preparation_bytes),
            }
        ],
        "execution_enabled": False,
        "formal_run_authorized": False,
        "model_output_count": 0,
        "subject_pack_materialization_count": 0,
        "claim_boundary": protocol.claim_boundary,
    }
    if descriptor.version == "v3":
        payload.update(
            {
                "donor_bank_materialization_count": 0,
                "counterfactual_twin_materialization_count": 0,
                "power_dgp_artifact_count": 0,
            }
        )
    return payload


def _validate_preparation_root(
    output: pathlib.Path,
    protocol: RelationshipP4LongContextScientificPrereg,
) -> RelationshipP4LongContextPreparation:
    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 preparation root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    if entries != tuple(sorted((_PREPARATION_FILE, _MANIFEST_FILE))):
        raise ValueError("P4.7 preparation file set drift")
    for name in entries:
        candidate = output / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("P4.7 preparation files must be regular files")

    preparation_path = output / _PREPARATION_FILE
    preparation_bytes = preparation_path.read_bytes()
    preparation = _load_json_object(preparation_path)
    if preparation_bytes != _canonical_bytes(preparation):
        raise ValueError("P4.7 preparation payload is not canonical JSON")
    expected_preparation = _preparation_payload(protocol)
    _require_literal(
        preparation,
        expected_preparation,
        "P4.7 preparation payload",
    )

    manifest_path = output / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_path)
    if manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("P4.7 manifest is not canonical JSON")
    expected_manifest_core = _manifest_core(protocol, preparation_bytes)
    manifest_keys = set(expected_manifest_core) | {"artifact_id"}
    _require_exact_keys(manifest, manifest_keys, "P4.7 manifest")
    artifact_id = _require_sha256(manifest["artifact_id"], "manifest artifact id")
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4.7 manifest artifact id drift")
    _require_literal(
        manifest_core,
        expected_manifest_core,
        "P4.7 manifest core",
    )
    descriptor = _protocol_descriptor(protocol.protocol_id)
    if descriptor.published_artifact_id is not None and artifact_id != descriptor.published_artifact_id:
        raise ValueError("P4.7 published artifact id drift")
    return RelationshipP4LongContextPreparation(
        artifact_id=artifact_id,
        protocol_id=protocol.protocol_id,
        status=descriptor.status,
        execution_enabled=False,
        formal_run_authorized=False,
        output_dir=output,
    )


def _artifact_protocol_id(output: pathlib.Path) -> str:
    """Read only the redundant artifact lineage needed for registry dispatch."""

    if not output.is_dir():
        raise FileNotFoundError(f"P4.7 preparation root is missing: {output}")
    entries = tuple(sorted(item.name for item in output.iterdir()))
    if entries != tuple(sorted((_PREPARATION_FILE, _MANIFEST_FILE))):
        raise ValueError("P4.7 preparation file set drift")
    preparation = _load_json_object(output / _PREPARATION_FILE)
    manifest = _load_json_object(output / _MANIFEST_FILE)
    preparation_id = _require_sha256(
        preparation.get("protocol_id"),
        "P4.7 preparation protocol id",
    )
    manifest_id = _require_sha256(
        manifest.get("protocol_id"),
        "P4.7 manifest protocol id",
    )
    if preparation_id != manifest_id:
        raise ValueError("P4.7 preparation/manifest protocol id mismatch")
    _protocol_descriptor(preparation_id)
    return preparation_id


def _validate_lineage(lineage: Mapping[str, Any]) -> None:
    _require_sha256(
        lineage["p4_1_seen_canary_protocol_sha256"],
        "P4.7 P4.1 lineage",
    )
    evidence = _require_mapping(
        lineage["development_evidence_only"],
        "P4.7 development evidence",
    )
    for field in (
        "readable_named_reader_artifact_id",
        "learnable_pe_credit_artifact_id",
        "appendable_cross_process_artifact_id",
        "steerable_fit_artifact_id",
    ):
        _require_sha256(evidence[field], f"P4.7 {field}")
    if evidence["may_authorize_formal_execution"] is not False:
        raise ValueError("P4.7 development evidence cannot authorize formal")
    negative = _require_text_tuple(
        lineage["negative_or_blocking_results_preserved"],
        "P4.7 negative lineage",
    )
    if len(negative) != 5 or len(set(negative)) != 5:
        raise ValueError("P4.7 negative lineage inventory drift")
    if lineage["old_results_may_be_relabelled_as_p4_7"] is not False:
        raise ValueError("P4.7 old results cannot be relabelled")


def _validate_cohort(cohort: Mapping[str, Any]) -> None:
    if cohort["subject_type"] != "independently_seeded_reactive_simulated_companion":
        raise ValueError("P4.7 subject type drift")
    if cohort["independent_unit"] != "subject_root":
        raise ValueError("P4.7 independent unit drift")
    if cohort["real_human_subject_count"] != 0:
        raise ValueError("P4.7 synthetic cohort cannot claim human subjects")
    independence = _require_mapping(
        cohort["independence_requirements"],
        "P4.7 independence requirements",
    )
    if not independence or any(value is not True for value in independence.values()):
        raise ValueError("P4.7 independence requirements are open")
    splits = _require_mapping(cohort["splits"], "P4.7 cohort splits")
    _require_exact_keys(
        splits,
        {"development", "qualification", "formal"},
        "P4.7 cohort splits",
    )
    if cohort["split_root_seeds_disjoint"] is not True:
        raise ValueError("P4.7 split seeds must be disjoint")
    if cohort["surface_families_disjoint_across_splits"] is not True:
        raise ValueError("P4.7 split surfaces must be disjoint")
    if cohort["formal_subject_materialization_before_prereg"] != 0:
        raise ValueError("P4.7 formal subjects were materialized before freeze")
    if cohort["formal_outcomes_before_prereg"] != 0:
        raise ValueError("P4.7 formal outcomes existed before freeze")


def _validate_longitudinal(longitudinal: Mapping[str, Any]) -> None:
    total = (
        _require_int(longitudinal["onboarding_sessions_per_subject"], "onboarding")
        + _require_int(
            longitudinal["matched_learning_exposure_sessions_per_subject"],
            "learning exposure",
        )
        + _require_int(
            longitudinal["frozen_policy_evaluation_sessions_per_subject"],
            "policy evaluation",
        )
    )
    if total != longitudinal["total_sessions_per_subject"] or total != 20:
        raise ValueError("P4.7 session horizon does not add up")
    required_true = (
        "fresh_os_process_per_session",
        "process_restart_is_observed_not_inferred",
        "per_arm_reactive_trajectory",
        "same_subject_seed_schedule_across_matched_arms",
        "fail_on_truncation",
        "full_history_baseline_receives_all_public_history",
        "context_feasibility_preflight_before_first_model_output",
    )
    if any(longitudinal[field] is not True for field in required_true):
        raise ValueError("P4.7 longitudinal hard gate is open")
    if longitudinal["raw_history_replayed_to_volvence_owner"] is not False:
        raise ValueError("P4.7 cannot replay raw history to the owner")
    if longitudinal["padding_or_filler_tokens_count_toward_minimum"] is not False:
        raise ValueError("P4.7 filler cannot satisfy long context")
    guards = _require_mapping(
        longitudinal["substantive_history_guards"],
        "P4.7 substantive history guards",
    )
    for field, value in guards.items():
        if field in {
            "minimum_distinct_public_turns",
            "minimum_distinct_typed_settlements",
        }:
            continue
        if value is not True:
            raise ValueError(f"P4.7 substantive history guard is open: {field}")
    if guards["minimum_distinct_public_turns"] < 96:
        raise ValueError("P4.7 public-turn floor is too small")
    if guards["minimum_distinct_typed_settlements"] < 16:
        raise ValueError("P4.7 settlement floor is too small")


def _validate_baseline(baseline: Mapping[str, Any]) -> None:
    if baseline["candidate_budget"] != 3:
        raise ValueError("P4.7 baseline candidate budget drift")
    if tuple(baseline["required_baselines"]) != _INTEGRATED_ARMS_V1[:2]:
        raise ValueError("P4.7 baseline inventory drift")
    if baseline["exact_one_typed_action_rate"] != 1.0:
        raise ValueError("P4.7 baseline validity threshold drift")
    if not (
        baseline["accuracy_point_estimate_minimum"] == 0.55
        and baseline["accuracy_point_estimate_maximum"] == 0.85
        and baseline["one_sided_wilson_accuracy_lower_strictly_above"] == 0.5
        and baseline["one_sided_wilson_pair_flip_lower_strictly_above"] == 0.35
    ):
        raise ValueError("P4.7 baseline informative band drift")
    required_true = (
        "candidate_inventory_and_order_frozen_before_first_development_output",
        "prompt_and_retrieval_search_uses_development_only",
        "selected_model_revision_weights_generation_and_prompt_frozen_before_qualification",
        "both_baselines_must_qualify_independently",
    )
    if any(baseline[field] is not True for field in required_true):
        raise ValueError("P4.7 baseline freeze gate is open")
    required_false = (
        "qualification_feedback_may_change_selected_baseline",
        "baseline_scores_are_system_learning_signals",
        "baseline_scores_are_four_axis_evidence",
    )
    if any(baseline[field] is not False for field in required_false):
        raise ValueError("P4.7 baseline feedback firewall is open")


def _parse_axis_contrasts_v1(
    payload: Mapping[str, Any],
) -> tuple[P4LongContextAxisContrast, ...]:
    _require_exact_keys(
        payload,
        set(_AXIS_CONTROLS_V1),
        "P4.7 v1 axis contrasts",
    )
    result: list[P4LongContextAxisContrast] = []
    for axis, expected_controls in _AXIS_CONTROLS_V1.items():
        value = _require_mapping(payload[axis], f"P4.7 {axis} contrast")
        if value["reference_arm"] != "volvence_closed_loop":
            raise ValueError(f"P4.7 {axis} reference arm drift")
        controls = _require_text_tuple(
            value["control_arms"],
            f"P4.7 {axis} controls",
        )
        if controls != expected_controls:
            raise ValueError(f"P4.7 {axis} controls drift")
        if axis == "learnable":
            if (
                value["learning_phase_action_exposure"] != "arm_independent_preregistered_schedule"
                or value["learning_phase_outcome_history_identical_for_primary_pair"] is not True
                or value["gate_updates_frozen_before_evaluation_phase"] is not True
                or value["actual_exposure_receipt_required_before_settlement"] is not True
                or value["shadow_suggestion_may_be_settled"] is not False
            ):
                raise ValueError("P4.7 Learnable identification drift")
        if axis == "steerable":
            if (
                value["conditionality_control_arm"] != "steerable_sensor_off_matched"
                or value["conditionality_control_is_pure_single_variable_claim"] is not False
                or value["strict_noop_delta_and_effect_exact_zero"] is not True
                or value["user_visible_generation_required"] is not True
            ):
                raise ValueError("P4.7 Steerable control drift")
        result.append(
            P4LongContextAxisContrast(
                axis=axis,
                reference_arm="volvence_closed_loop",
                control_arms=controls,
                primary_effect=_require_text(
                    value["primary_effect"],
                    f"P4.7 {axis} primary effect",
                ),
            )
        )
    return tuple(result)


def _validate_intervention_integrity(payload: Mapping[str, Any]) -> None:
    if payload["reference_arm"] != "volvence_closed_loop":
        raise ValueError("P4.7 intervention reference drift")
    pointers = _require_mapping(
        payload["allowed_exogenous_difference_json_pointers"],
        "P4.7 intervention pointers",
    )
    _require_exact_keys(
        pointers,
        set(_ALLOWED_INTERVENTION_POINTERS_V1),
        "P4.7 v1 intervention pointers",
    )
    for arm, expected in _ALLOWED_INTERVENTION_POINTERS_V1.items():
        if _require_text_tuple(pointers[arm], arm) != expected:
            raise ValueError(f"P4.7 intervention pointer drift: {arm}")
    if payload["config_hash_matches_after_removing_allowed_paths"] is not True:
        raise ValueError("P4.7 matched-config gate is open")
    if payload["matched_invariant_receipt_required_per_subject_arm"] is not True:
        raise ValueError("P4.7 matched receipt is optional")


def _validate_shared_exposure(payload: Mapping[str, Any]) -> None:
    required_true = (
        "same_exposure_id_payload_hidden_step_and_random_tape_across_arms",
        "exogenous_future_exposure_is_action_independent",
        "fixed_session_count_across_arms",
        "inventory_hashes_required_before_first_development_output",
    )
    if any(payload[field] is not True for field in required_true):
        raise ValueError("P4.7 shared exposure contract is open")
    if payload["arm_or_prior_action_may_change_future_exposure_id"] is not False:
        raise ValueError("P4.7 arm may not change future exogenous exposure")
    if payload["dropout_or_rerouting_based_on_arm_or_outcome"] is not False:
        raise ValueError("P4.7 arm-dependent dropout is forbidden")


def _validate_causal_execution(payload: Mapping[str, Any]) -> None:
    required_true = (
        "same_frozen_substrate_across_all_arms",
        "same_generation_config_across_all_arms",
        "pre_action_record_before_environment",
        "typed_reactive_outcome_is_mechanical",
        "future_outcome_hidden_from_sut",
        "generator_truth_hidden_from_sut",
        "judge_and_evaluation_hidden_from_memory_pe_credit_gate_and_steering",
    )
    if any(payload[field] is not True for field in required_true):
        raise ValueError("P4.7 causal execution gate is open")
    if payload["human_annotation_role"] != "validation_anchor_only":
        raise ValueError("P4.7 human annotation role drift")
    if payload["formal_result_retry_count"] != 0:
        raise ValueError("P4.7 formal retries are forbidden")
    if payload["interim_efficacy_looks"] != 0:
        raise ValueError("P4.7 interim efficacy looks are forbidden")


def _validate_analysis(payload: Mapping[str, Any]) -> None:
    if payload["primary_analysis_unit"] != "subject_root":
        raise ValueError("P4.7 analysis unit drift")
    if payload["minimum_practical_mean_delta"] != 0.15:
        raise ValueError("P4.7 effect threshold drift")
    if payload["bootstrap_replicates"] != 100_000:
        raise ValueError("P4.7 bootstrap count drift")
    if payload["formal_preallocated_subject_count"] != 192:
        raise ValueError("P4.7 formal analysis cohort drift")
    if payload["minimum_complete_paired_subjects"] != 160:
        raise ValueError("P4.7 paired analysis floor drift")
    required_true = (
        "all_axis_primary_contrasts_must_pass",
        "confidence_lower_bound_must_be_strictly_positive",
        "power_simulation_artifact_required_before_first_development_output",
    )
    if any(payload[field] is not True for field in required_true):
        raise ValueError("P4.7 analysis gate is open")
    if payload["power_target"] < 0.8:
        raise ValueError("P4.7 power target is too low")
    if payload["secondary_measures_are_promotion_inputs"] is not False:
        raise ValueError("P4.7 secondary measures cannot decide promotion")


def _validate_execution_admission(payload: Mapping[str, Any]) -> None:
    if payload["current_stage"] != "scientific_prereg_only":
        raise ValueError("P4.7 stage drift")
    false_fields = (
        "execution_enabled",
        "formal_run_authorized",
        "execution_envelope_present",
        "development_cuda_diagnostics_are_formal_evidence",
        "user_development_cuda_consent_is_formal_authorization",
        "manual_override_permitted",
        "environment_override_permitted",
        "ignore_microcode_override_permitted",
        "force_cli_permitted",
        "future_execution_envelope_may_change_scientific_design",
        "host_qualification_terminal_self_report_authoritative",
        "existing_p4_6_fit_may_authorize_long_context_formal",
    )
    if any(payload[field] is not False for field in false_fields):
        raise ValueError("P4.7 execution admission firewall is open")
    if payload["development_cuda_diagnostics_allowed"] is not True:
        raise ValueError("P4.7 development CUDA permission drift")
    if payload["model_output_count_before_freeze"] != 0:
        raise ValueError("P4.7 model output preceded freeze")
    if payload["subject_pack_materialization_count_before_freeze"] != 0:
        raise ValueError("P4.7 subject pack preceded freeze")
    legacy = _require_mapping(payload["legacy_host_block"], "legacy host block")
    _require_sha256(legacy["physical_actuation_protocol_id"], "P4.6 protocol")
    _require_sha256(
        legacy["host_block_receipt_raw_sha256"],
        "P4.6 host block",
    )
    if legacy["automatically_replaced_by_short_cuda_success"] is not False:
        raise ValueError("P4.7 short CUDA success cannot replace host block")
    acquisition = _require_mapping(
        payload["current_acquisition_v2"],
        "P4.7 current acquisition",
    )
    _require_sha256(acquisition["protocol_id"], "acquisition protocol id")
    _require_sha256(
        acquisition["protocol_raw_sha256"],
        "acquisition protocol raw hash",
    )
    if acquisition["may_authorize_execution"] is not False:
        raise ValueError("P4.7 acquisition v2 cannot authorize execution")
    required = _require_text_tuple(
        payload["future_execution_envelope_required_fields"],
        "P4.7 execution-envelope fields",
    )
    if len(required) != 14 or len(set(required)) != 14:
        raise ValueError("P4.7 execution-envelope field inventory drift")
    if payload["production_host_qualification_real_observation_required"] is not True:
        raise ValueError("P4.7 requires real host observation")
    if payload["production_host_qualification_validated_eligible_required"] is not True:
        raise ValueError("P4.7 requires independently validated host eligibility")
    if payload["minimum_microcode_integer"] != 303:
        raise ValueError("P4.7 formal microcode threshold drift")
    if payload["existing_p4_6_fit_native_window_tokens"] != 32_768:
        raise ValueError("P4.7 P4.6 context disclosure drift")


def _validate_evidence_firewall(payload: Mapping[str, Any]) -> None:
    if payload["scientific_design_frozen"] is not True:
        raise ValueError("P4.7 scientific design is not frozen")
    for field, value in payload.items():
        if field == "scientific_design_frozen":
            continue
        if value is not False:
            raise ValueError(f"P4.7 evidence firewall is open: {field}")


def _protocol_descriptor(protocol_id: str) -> _ProtocolDescriptor:
    if type(protocol_id) is not str:
        raise TypeError("P4.7 protocol id must be text")
    descriptor = _PROTOCOLS_BY_ID.get(protocol_id)
    if descriptor is None:
        raise ValueError(f"P4.7 unregistered protocol id: {protocol_id}")
    return descriptor


def _absolute_without_resolving(path: os.PathLike[str] | str) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _require_local_default_stream_path(
    path: os.PathLike[str] | str,
    label: str,
) -> pathlib.Path:
    raw_path = os.fspath(path)
    if type(raw_path) is not str or not raw_path or "\x00" in raw_path:
        raise ValueError(f"{label} must be a non-empty local text path")
    if os.name == "nt":
        windows_path = raw_path.replace("/", "\\")
        drive, remainder = os.path.splitdrive(windows_path)
        if windows_path.startswith("\\\\") or drive.startswith("\\\\"):
            raise ValueError(f"{label} must not use a UNC or Windows device namespace")
        if ":" in remainder:
            raise ValueError(f"{label} must address only the NTFS default data stream")
    absolute = _absolute_without_resolving(raw_path)
    if os.name == "nt" and str(absolute).replace("/", "\\").startswith("\\\\"):
        raise ValueError(f"{label} must remain on a local filesystem path")
    return absolute


def _reject_reparse_components(path: pathlib.Path, label: str) -> None:
    """Reject symlinks and Windows reparse points without resolving them."""

    for candidate in (path, *path.parents):
        if not os.path.lexists(candidate):
            continue
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")
        if os.name == "nt":
            attributes = os.lstat(candidate).st_file_attributes
            if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"{label} must not traverse a Windows reparse point: {candidate}")


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"JSON source must be a regular file: {source}")
    payload = source.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"JSON source must not carry a UTF-8 BOM: {source}")
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON source is not strict UTF-8: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON source is invalid: {source}") from exc
    if type(value) is not dict:
        raise ValueError(f"JSON source root must be an object: {source}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _fraction_payload(value: Fraction) -> dict[str, str]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
    }


def _fraction_hex_payload(value: Fraction) -> dict[str, str]:
    if type(value) is not Fraction or value <= 0:
        raise ValueError("hex fraction payload requires a positive Fraction")
    return {
        "numerator_hex": format(value.numerator, "x"),
        "denominator_hex": format(value.denominator, "x"),
    }


def _fraction_from_hex_payload(value: object, label: str) -> Fraction:
    payload = _require_mapping(value, label)
    _require_exact_keys(payload, {"numerator_hex", "denominator_hex"}, label)
    numerator_hex = _require_text(payload["numerator_hex"], f"{label} numerator hex")
    denominator_hex = _require_text(payload["denominator_hex"], f"{label} denominator hex")
    for text, component in ((numerator_hex, "numerator"), (denominator_hex, "denominator")):
        if any(character not in _HEX_DIGITS for character in text):
            raise ValueError(f"{label} {component} must be lowercase hexadecimal")
        if len(text) > 1 and text.startswith("0"):
            raise ValueError(f"{label} {component} must not contain a leading zero")
    numerator = int(numerator_hex, 16)
    denominator = int(denominator_hex, 16)
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"{label} must be a positive fraction")
    result = Fraction(numerator, denominator)
    if _fraction_hex_payload(result) != payload:
        raise ValueError(f"{label} must be reduced canonical lowercase hexadecimal")
    return result


def _fraction_from_payload(value: object, label: str) -> Fraction:
    payload = _require_mapping(value, label)
    _require_exact_keys(payload, {"numerator", "denominator"}, label)
    numerator_text = _require_text(payload["numerator"], f"{label} numerator")
    denominator_text = _require_text(
        payload["denominator"],
        f"{label} denominator",
    )
    if not numerator_text.lstrip("-").isdigit() or not denominator_text.isdigit():
        raise ValueError(f"{label} must contain canonical integer strings")
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator <= 0:
        raise ValueError(f"{label} denominator must be positive")
    result = Fraction(numerator, denominator)
    if _fraction_payload(result) != payload:
        raise ValueError(f"{label} must be reduced with a positive denominator")
    return result


def _minimum_upper_count_for_point_gate(
    *,
    formal_root_count: int,
    lower: Fraction,
    upper: Fraction,
    practical_threshold: Fraction,
) -> int:
    required = Fraction(formal_root_count, 1) * (practical_threshold - lower) / (upper - lower)
    return -(-required.numerator // required.denominator)


def _maximum_variance_point_gate_probability(
    *,
    formal_root_count: int,
    mass_at_upper: Fraction,
    minimum_upper_count: int,
) -> Fraction:
    mass_at_lower = Fraction(1, 1) - mass_at_upper
    return sum(
        (
            Fraction(math.comb(formal_root_count, upper_count), 1)
            * mass_at_upper**upper_count
            * mass_at_lower ** (formal_root_count - upper_count)
        )
        for upper_count in range(minimum_upper_count, formal_root_count + 1)
    )


def _fraction_display_decimal(value: Fraction, *, places: int) -> str:
    quantum = Decimal(1).scaleb(-places)
    with localcontext() as context:
        context.prec = max(places + 40, 80)
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        return format(
            decimal_value.quantize(quantum, rounding=ROUND_HALF_EVEN),
            "f",
        )


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


def _write_create_bytes(path: pathlib.Path, payload: bytes) -> None:
    with pathlib.Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ValueError(f"{label} keys drift; missing={missing}, extra={extra}")


def _require_literal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(f"{label} type drift")
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        _require_exact_keys(actual, set(expected), label)
        for key, value in expected.items():
            _require_literal(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            raise ValueError(f"{label} length drift")
        for index, value in enumerate(expected):
            _require_literal(actual[index], value, f"{label}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{label} value drift")


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{label} must be non-empty text")
    return value


def _require_text_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an array")
    result = tuple(_require_text(item, f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates")
    return result


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _require_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _require_decimal_float(value: object, label: str) -> float:
    text = _require_text(value, label)
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a finite decimal string") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite decimal string")
    return float(result)


def _require_timestamp(value: object, label: str) -> str:
    text = _require_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def _require_sha256(value: object, label: str) -> str:
    text = _require_text(value, label)
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION",
    "P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V2",
    "P4_LONG_CONTEXT_MANIFEST_SCHEMA_VERSION_V3",
    "P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION",
    "P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V2",
    "P4_LONG_CONTEXT_PREPARATION_SCHEMA_VERSION_V3",
    "P4_LONG_CONTEXT_PREPARATION_STATUS",
    "P4_LONG_CONTEXT_PREPARATION_STATUS_V1",
    "P4_LONG_CONTEXT_PREPARATION_STATUS_V2",
    "P4_LONG_CONTEXT_PREPARATION_STATUS_V3",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V1",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_ID_V2",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V1",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_RAW_SHA256_V2",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_POWER_BOUND_PROTOCOL_SCHEMA_VERSION_V2",
    "P4_LONG_CONTEXT_POWER_ADMISSION_ARTIFACT_ID_V2",
    "P4_LONG_CONTEXT_POWER_ADMISSION_MANIFEST_SCHEMA_VERSION_V2",
    "P4_LONG_CONTEXT_POWER_ADMISSION_SCHEMA_VERSION_V2",
    "P4_LONG_CONTEXT_POWER_ADMISSION_STATUS_V2",
    "P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID",
    "P4_LONG_CONTEXT_POWER_FAILURE_ARTIFACT_ID_V1",
    "P4_LONG_CONTEXT_POWER_FAILURE_MANIFEST_SCHEMA_VERSION",
    "P4_LONG_CONTEXT_POWER_FAILURE_SCHEMA_VERSION",
    "P4_LONG_CONTEXT_POWER_FAILURE_STATUS",
    "P4_LONG_CONTEXT_V4_CANDIDATE_SCHEDULE_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_ARTIFACT_ID_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_FREEZE_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_MANIFEST_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_ID_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_RAW_SHA256_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_PROTOCOL_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_PLANNING_STATUS_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_CONTRACT_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_ARTIFACT_ID_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_ID_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_CERTIFICATE_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_MANIFEST_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_ID_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_RAW_SHA256_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_PROTOCOL_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_SOURCE_OPPORTUNITY_PREFLIGHT_STATUS_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ARTIFACT_ID_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_ID_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_MANIFEST_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_ID_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_RAW_SHA256_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_PROTOCOL_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_V4_EXTERNAL_ANCHOR_REQUEST_STATUS_V1",
    "P4_LONG_CONTEXT_PROTOCOL_ID",
    "P4_LONG_CONTEXT_PROTOCOL_ID_V1",
    "P4_LONG_CONTEXT_PROTOCOL_ID_V2",
    "P4_LONG_CONTEXT_PROTOCOL_ID_V3",
    "P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION",
    "P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V1",
    "P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V2",
    "P4_LONG_CONTEXT_PROTOCOL_SCHEMA_VERSION_V3",
    "P4LongContextAxisContrast",
    "RelationshipP4LongContextPreparation",
    "RelationshipP4LongContextPowerAdmissionCertificate",
    "RelationshipP4LongContextPowerFailureCertificate",
    "RelationshipP4LongContextScientificPrereg",
    "RelationshipP4LongContextExternalAnchorRequest",
    "RelationshipP4LongContextExternalAnchorRequestProtocol",
    "RelationshipP4LongContextSourceOpportunityPreflightCertificate",
    "RelationshipP4LongContextSourceOpportunityPreflightProtocol",
    "RelationshipP4LongContextV4PlanningFreeze",
    "RelationshipP4LongContextV4PlanningProtocol",
    "load_relationship_p4_long_context_scientific_prereg",
    "load_relationship_p4_long_context_external_anchor_request_protocol",
    "load_relationship_p4_long_context_source_opportunity_preflight_protocol",
    "load_relationship_p4_long_context_v4_planning_protocol",
    "prepare_relationship_p4_long_context_scientific_prereg",
    "prepare_relationship_p4_long_context_external_anchor_request",
    "prepare_relationship_p4_long_context_source_opportunity_preflight",
    "prepare_relationship_p4_long_context_power_admission_certificate",
    "prepare_relationship_p4_long_context_power_failure_certificate",
    "prepare_relationship_p4_long_context_v4_zero_output_plan",
    "relationship_p4_long_context_protocol_path",
    "validate_relationship_p4_long_context_scientific_prereg",
    "validate_relationship_p4_long_context_external_anchor_request",
    "validate_relationship_p4_long_context_source_opportunity_preflight",
    "validate_relationship_p4_long_context_power_admission_certificate",
    "validate_relationship_p4_long_context_power_failure_certificate",
    "validate_relationship_p4_long_context_v4_zero_output_plan",
)
