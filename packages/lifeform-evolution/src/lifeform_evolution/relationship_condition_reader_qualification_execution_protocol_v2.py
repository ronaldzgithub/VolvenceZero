"""V2 freeze and public-anchor contract for reader qualification execution.

V2 keeps every V1 byte immutable.  It wraps the V1 preflight, source-tree,
BGE snapshot, process-firewall, gate, and claim mechanisms while extending
repository/runtime closure, child-origin attestation, and the external-
publication proof.  Formal observation performs six bounded unauthenticated
HTTPS GETs itself: the base Gist, its first page of commits (with
``per_page=2``), the sole revision, the base API raw URL, the canonical
revision raw URL, and a final commits reobservation that serves as the
observation linearization point.

The retired V1 Gist remains immutable lineage and can never authorize V2.
Neither importing this module nor freezing/observing a protocol imports a
model runtime or executes CUDA.
"""

from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib.metadata
import io
import os
import pathlib
import re
import ssl
import stat
import sys
from typing import Mapping, Protocol
import urllib.error
import urllib.request

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads

from . import relationship_condition_reader_qualification_execution_protocol as v1
from .relationship_condition_reader_qualification_repository_runtime_coverage import (
    build_relationship_condition_reader_repository_runtime_coverage,
    validate_relationship_condition_reader_repository_runtime_coverage,
)


RELATIONSHIP_READER_EXECUTION_PROTOCOL_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-protocol.v2"
)
RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-public-anchor-receipt.v2"
)
RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-runtime-identity.v2"
)
RELATIONSHIP_READER_EXECUTION_RUNTIME_RAW_TREE_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-runtime-raw-tree.v2"
)
RELATIONSHIP_READER_EXECUTION_SITE_PACKAGES_COVERAGE_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-site-packages-coverage.v2"
)
RELATIONSHIP_READER_EXECUTION_PYTHON_STDLIB_ZIP_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-python-stdlib-zip.v2"
)
RELATIONSHIP_READER_EXECUTION_PYTHON_HOME_TOP_LEVEL_TREE_V2_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-python-home-top-level-tree.v2"
)

V2_GIST_FILENAME = "relationship_condition_reader_qualification_execution_v2.json"
_WINDOWS_INVALID_EXECUTION_ROOT_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_EXECUTION_ROOT_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)
GITHUB_API_VERSION = "2026-03-10"

_GIST_OWNER = "ronaldzgithub"
_GITHUB_ACCEPT = "application/vnd.github+json"
_GITHUB_USER_AGENT = "VolvenceQualificationAnchorV2/1.0"
_OBSERVATION_TRANSPORT = "unauthenticated_github_rest_api_v2026_03_10_and_raw_https"
_MAX_GITHUB_JSON_BYTES = 8_000_000
_MAX_PROTOCOL_BYTES = 8_000_000
_INTEGRITY_PHASES = (
    "post_anchor_pre_execution",
    "pre_prediction_child_1",
    "post_prediction_child_1",
    "pre_prediction_child_2",
    "post_prediction_child_2",
    "pre_scorer",
    "post_scorer",
    "final_validation",
)
_FORBIDDEN_MODEL_MODULE_PREFIXES = ("torch", "sentence_transformers")
_STDLIB_TREE_ROLE = "stdlib_lib_excluding_site_packages_and_cache_bytecode"
_DLLS_TREE_ROLE = "python_dlls_excluding_cache_bytecode"
_LIBRARY_BIN_TREE_ROLE = "python_library_bin_excluding_cache_bytecode"
_PYTHON_HOME_TOP_LEVEL_TREE_ROLE = "python_home_application_directory_top_level_files"
_RUNTIME_TREE_CACHE_DIRECTORY_NAMES = ("__pycache__",)
_RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES = (".pyc",)
_SITE_PACKAGES_EXCLUDED_SITE_STARTUP_SUFFIXES = (".pth",)
_CHILD_IMPORT_CONTRACT_V2 = {
    "schema_version": "relationship-condition-reader-qualification-child-import-contract.v2",
    "python_flag_order": [
        "-P",
        "-S",
        "-B",
        "-u",
        "-X",
        "utf8",
        "-X",
        "pycache_prefix=<canonical-per-child-capsule-path>",
        "-c",
    ],
    "ambient_pythonpath_inherited": False,
    "pythonpath_built_from_frozen_runtime": True,
    "site_startup_enabled": False,
    "current_working_directory_on_sys_path": False,
    "adjacent_bytecode_cache_read_allowed": False,
    "per_child_isolated_pycache_prefix_required": True,
    "child_sys_path_exact_validation_required": True,
    "child_module_origin_exact_validation_required": True,
    "ambient_path_inherited": False,
    "controlled_path_order": [
        "<python_environment_root>",
        "<python_environment_root>\\DLLs",
        "<python_environment_root>\\Library\\bin",
        "<SystemRoot>\\System32",
        "<SystemRoot>",
    ],
    "child_path_exact_validation_required": True,
    "ambient_cuda_path_inherited": False,
    "predictor_cuda_visible_devices": "0",
    "predictor_v1_physical_gpu_index": 0,
    "scorer_cuda_visible_devices": None,
    "scorer_cuda_execution_allowed": False,
}
_V2_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_execution_protocol_v2.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_runtime_binding.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_repository_runtime_coverage.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_execution_v2.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_execution_cli_v2.py",
    }
)

_RETIRED_V1_PREDECESSOR = {
    "schema_version": ("relationship-condition-reader-qualification-retired-execution-predecessor.v1"),
    "execution_protocol_schema_version": ("relationship-condition-reader-qualification-execution-protocol.v1"),
    "execution_protocol_id": ("0ab8543a69f3ff5a270ada9038cc326ace9b00c408c29d54c5f8012e70aaf1ab"),
    "protocol_raw_sha256": ("02dd24e68efdd7c988c84ac250d48116d4bba637fbf7dad3add5d9c491614572"),
    "protocol_raw_bytes": 186_131,
    "gist_owner": _GIST_OWNER,
    "gist_id": "5f506d8dcb5a9ed68e10274597ba56e3",
    "gist_url": ("https://gist.github.com/ronaldzgithub/5f506d8dcb5a9ed68e10274597ba56e3"),
    "filename": "relationship_condition_reader_qualification_execution_v1.json",
    "sole_version": "63ebb32de5703d6e77c21c13e0f52c9dc2c38560",
    "created_at": "2026-08-24T12:21:08Z",
    "updated_at": "2026-08-24T12:21:09Z",
    "retired": True,
    "retirement_reason": ("v1_required_created_at_equal_updated_at_but_github_reported_one_second_drift"),
    "anchor_receipt_created": False,
    "qualification_execution_started": False,
    "model_or_cuda_execution_used": False,
    "can_authorize_v2": False,
}


class _HttpsFetcher(Protocol):
    def __call__(self, *, url: str, max_bytes: int) -> bytes: ...


class _UtcClock(Protocol):
    def __call__(self) -> dt.datetime: ...


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: object,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        raise RuntimeError("V2 GitHub observer refuses HTTP redirects")


@dataclass(frozen=True)
class RelationshipConditionReaderQualificationIntegrityGuardV2:
    """Reobserve the frozen V2 source/model/runtime at each V1 stage phase."""

    execution_protocol: Mapping[str, object]
    expected_execution_protocol_id: str
    repository_root: pathlib.Path
    bge_snapshot_root: pathlib.Path

    def __call__(
        self,
        *,
        phase: str,
        previous_integrity_receipt_artifact_id: str | None,
    ) -> Mapping[str, object]:
        return build_relationship_condition_reader_qualification_integrity_receipt_v2(
            execution_protocol=self.execution_protocol,
            expected_execution_protocol_id=self.expected_execution_protocol_id,
            repository_root=self.repository_root,
            bge_snapshot_root=self.bge_snapshot_root,
            phase=phase,
            previous_integrity_receipt_artifact_id=(previous_integrity_receipt_artifact_id),
        )


def retired_relationship_condition_reader_qualification_v1_predecessor() -> Mapping[str, object]:
    """Return immutable failed-V1 publication lineage; it has no authority."""

    return _json_copy(_RETIRED_V1_PREDECESSOR)


def build_relationship_condition_reader_qualification_import_runtime_identity_v2(
    *,
    base_runtime_identity: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Observe every Python import domain used by the isolated V2 children.

    The V1 runtime remains an immutable component of this larger identity.  Its
    four distribution pins are reused after their RECORD contents have already
    been verified; every other installed distribution under the same sole
    ``site-packages`` root is verified here.  ``Lib`` and ``DLLs`` are pinned as
    raw trees, while startup-only ``site-packages`` and adjacent bytecode caches
    are excluded because the child contract uses ``-S`` and an isolated cache
    prefix.
    """

    _assert_model_free()
    observed_base = v1.build_relationship_condition_reader_qualification_runtime_identity()
    if base_runtime_identity is not None and not _json_type_exact_equal(
        _mapping(base_runtime_identity, "supplied V1 runtime identity"),
        observed_base,
    ):
        raise ValueError("supplied V1 runtime identity differs from the current exact runtime")
    site_packages_root = _sole_site_packages_root(observed_base)
    environment_root = _python_environment_root(
        base_runtime_identity=observed_base,
        site_packages_root=site_packages_root,
    )
    distribution_inventory = _observe_installed_distribution_inventory(
        site_packages_root=site_packages_root,
        environment_root=environment_root,
        base_runtime_identity=observed_base,
    )
    site_packages_coverage = _observe_site_packages_coverage(
        site_packages_root=site_packages_root,
        environment_root=environment_root,
        distribution_inventory=distribution_inventory,
    )
    python_stdlib_zip = _observe_python_stdlib_zip(
        environment_root=environment_root,
        python_identity=_mapping(observed_base["python"], "V1 Python identity"),
    )
    python_home_top_level_tree = _observe_python_home_top_level_tree(
        environment_root=environment_root,
    )
    stdlib_tree = _observe_runtime_raw_tree(
        root=environment_root / "Lib",
        tree_role=_STDLIB_TREE_ROLE,
        excluded_top_level_directories=("site-packages",),
    )
    dlls_tree = _observe_runtime_raw_tree(
        root=environment_root / "DLLs",
        tree_role=_DLLS_TREE_ROLE,
        excluded_top_level_directories=(),
    )
    library_bin_tree = _observe_runtime_raw_tree(
        root=environment_root / "Library" / "bin",
        tree_role=_LIBRARY_BIN_TREE_ROLE,
        excluded_top_level_directories=(),
    )
    core = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_V2_SCHEMA_VERSION,
        "platform": observed_base["platform"],
        "gpu": observed_base["gpu"],
        "python": observed_base["python"],
        "distributions": observed_base["distributions"],
        "v1_runtime_identity_artifact_id": observed_base["artifact_id"],
        "child_import_contract": _json_copy(_CHILD_IMPORT_CONTRACT_V2),
        "python_environment_root": str(environment_root),
        "site_packages_root": str(site_packages_root),
        "installed_distributions": distribution_inventory,
        "installed_distribution_count": len(distribution_inventory),
        "site_packages_coverage": site_packages_coverage,
        "python_stdlib_zip": python_stdlib_zip,
        "python_home_top_level_tree": python_home_top_level_tree,
        "stdlib_lib_tree": stdlib_tree,
        "dlls_tree": dlls_tree,
        "python_library_bin_tree": library_bin_tree,
    }
    payload = _with_artifact_id(core)
    _validate_import_runtime_identity_v2(payload)
    _assert_model_free()
    return payload


def validate_relationship_condition_reader_qualification_import_runtime_identity_v2(
    payload: Mapping[str, object],
    *,
    reobserve_current_runtime: bool = False,
) -> str:
    """Validate a V2 import-runtime snapshot and optionally reobserve disk."""

    runtime = _mapping(payload, "V2 import runtime identity")
    _validate_import_runtime_identity_v2(runtime)
    if reobserve_current_runtime:
        observed = build_relationship_condition_reader_qualification_import_runtime_identity_v2()
        if not _json_type_exact_equal(runtime, observed):
            raise ValueError("V2 import runtime identity differs from the current exact runtime")
    return _digest(runtime["artifact_id"], "V2 import runtime artifact id")


def build_relationship_condition_reader_qualification_execution_protocol_v2(
    *,
    preflight_binding: Mapping[str, object],
    source_tree_manifest: Mapping[str, object],
    repository_runtime_coverage: Mapping[str, object],
    bge_snapshot_tree_manifest: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    proposed_execution_root: pathlib.Path,
    anchor_receipt_relative_path: str,
) -> Mapping[str, object]:
    """Compose a static V2 protocol; this never observes or authorizes a Gist."""

    execution_root_text = canonical_relationship_condition_reader_qualification_execution_root_v2(
        proposed_execution_root,
        "proposed execution root",
    )
    import_runtime = build_relationship_condition_reader_qualification_import_runtime_identity_v2(
        base_runtime_identity=runtime_identity,
    )
    v1_runtime = _v1_runtime_identity_from_v2(import_runtime)
    base = v1.build_relationship_condition_reader_qualification_execution_protocol(
        preflight_binding=preflight_binding,
        source_tree_manifest=source_tree_manifest,
        bge_snapshot_tree_manifest=bge_snapshot_tree_manifest,
        runtime_identity=v1_runtime,
        proposed_execution_root=pathlib.Path(execution_root_text),
        anchor_receipt_relative_path=anchor_receipt_relative_path,
    )
    receipt_path = _relative_receipt_path(anchor_receipt_relative_path)
    payload = {
        **dict(base),
        "schema_version": RELATIONSHIP_READER_EXECUTION_PROTOCOL_V2_SCHEMA_VERSION,
        "repository_runtime_coverage": dict(repository_runtime_coverage),
        "runtime_identity": dict(import_runtime),
        "external_public_anchor": _external_anchor_contract_v2(receipt_path),
        "retired_predecessor": _json_copy(_RETIRED_V1_PREDECESSOR),
    }
    _validate_protocol_shape(payload)
    return payload


def relationship_condition_reader_qualification_execution_protocol_id_v2(
    payload: Mapping[str, object],
) -> str:
    """Return the semantic V2 protocol ID after strict validation."""

    protocol = _mapping(payload, "V2 execution protocol")
    _validate_protocol_shape(protocol)
    return _sha256_json(protocol)


def validate_relationship_condition_reader_qualification_execution_protocol_v2(
    payload: Mapping[str, object],
    *,
    expected_protocol_id: str,
    repository_root: pathlib.Path | None = None,
    preflight_root: pathlib.Path | None = None,
    bge_snapshot_root: pathlib.Path | None = None,
) -> str:
    """Validate V2 identity and reobserve its runtime plus optional disk domains."""

    protocol = _mapping(payload, "V2 execution protocol")
    _validate_protocol_shape(protocol)
    observed_id = _sha256_json(protocol)
    if observed_id != _digest(expected_protocol_id, "expected V2 protocol id"):
        raise ValueError("V2 execution protocol id differs from the external expected id")
    observed_runtime = build_relationship_condition_reader_qualification_import_runtime_identity_v2()
    if not _json_type_exact_equal(protocol["runtime_identity"], observed_runtime):
        raise ValueError("V2 execution protocol runtime differs from the current exact runtime")
    source_tree = _mapping(protocol["execution_source_tree"], "V2 source tree")
    v1.validate_relationship_condition_reader_execution_source_tree_manifest(
        source_tree,
        repository_root=repository_root,
    )
    repository_coverage = _mapping(
        protocol["repository_runtime_coverage"],
        "V2 repository runtime coverage",
    )
    validate_relationship_condition_reader_repository_runtime_coverage(
        repository_coverage,
        execution_source_tree=source_tree,
    )
    if repository_root is not None:
        observed_repository_coverage = build_relationship_condition_reader_repository_runtime_coverage(
            repository_root=repository_root,
            execution_source_tree=source_tree,
        )
        if not _json_type_exact_equal(repository_coverage, observed_repository_coverage):
            raise ValueError("V2 repository runtime coverage differs from current exact source roots")
    v1.validate_bge_m3_snapshot_tree_manifest(
        _mapping(protocol["bge_snapshot_tree"], "V2 BGE tree"),
        snapshot_root=bge_snapshot_root,
    )
    v1.validate_relationship_condition_reader_execution_preflight_binding(
        _mapping(protocol["qualification_preflight"], "V2 preflight binding"),
        preflight_root=preflight_root,
    )
    return observed_id


def load_relationship_condition_reader_qualification_execution_protocol_v2(
    path: pathlib.Path,
    *,
    expected_protocol_id: str,
) -> tuple[Mapping[str, object], bytes]:
    """Load canonical LF V2 protocol bytes with an external expected ID."""

    source = _absolute_file(path, "V2 execution protocol")
    raw = _read_stable_regular_file(
        source,
        field_name="V2 execution protocol",
        max_bytes=_MAX_PROTOCOL_BYTES,
    )
    payload = _parse_json_object(raw, "V2 execution protocol", _MAX_PROTOCOL_BYTES)
    if raw != canonical_json_bytes(dict(payload)) + b"\n":
        raise ValueError("V2 execution protocol must be canonical LF-terminated JSON")
    validate_relationship_condition_reader_qualification_execution_protocol_v2(
        payload,
        expected_protocol_id=expected_protocol_id,
    )
    return payload, raw


def observe_relationship_condition_reader_qualification_public_anchor_v2(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    expected_execution_root: pathlib.Path,
    gist_id: str,
    timeout_seconds: int = 30,
) -> Mapping[str, object]:
    """Perform the fixed unauthenticated six-GET V2 anchor observation."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
        or timeout_seconds > 300
    ):
        raise ValueError("timeout_seconds must be an integer in [1, 300]")

    def fetcher(*, url: str, max_bytes: int) -> bytes:
        return _unauthenticated_github_https_get(
            url=url,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )

    return _observe_relationship_condition_reader_qualification_public_anchor_v2_with_fetcher(
        execution_protocol_payload=execution_protocol_payload,
        execution_protocol_raw=execution_protocol_raw,
        expected_execution_protocol_id=expected_execution_protocol_id,
        expected_execution_root=expected_execution_root,
        gist_id=gist_id,
        fetcher=fetcher,
    )


def _observe_relationship_condition_reader_qualification_public_anchor_v2_with_fetcher(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    expected_execution_root: pathlib.Path,
    gist_id: str,
    fetcher: _HttpsFetcher,
    clock: _UtcClock | None = None,
) -> Mapping[str, object]:
    """Injectable six-GET orchestration used only by deterministic tests."""

    if not callable(fetcher):
        raise TypeError("V2 GitHub fetcher must be callable")
    observed_gist_id = _hex_text(gist_id, "gist id", lengths={32})
    base_url = f"https://api.github.com/gists/{observed_gist_id}"
    commits_url = f"{base_url}/commits?per_page=2&page=1"
    base_raw = fetcher(url=base_url, max_bytes=_MAX_GITHUB_JSON_BYTES)
    commits_raw = fetcher(url=commits_url, max_bytes=_MAX_GITHUB_JSON_BYTES)
    base = _parse_json_object(base_raw, "base Gist API response", _MAX_GITHUB_JSON_BYTES)
    commits = _parse_json_array(
        commits_raw,
        "Gist commits API response",
        _MAX_GITHUB_JSON_BYTES,
    )
    if len(commits) != 1:
        raise ValueError("V2 anchor requires exactly one item on commits page 1")
    commit = _mapping(commits[0], "sole Gist commit")
    version = _hex_text(commit.get("version"), "sole Gist version", lengths={40})
    revision_url = f"{base_url}/{version}"
    revision_raw = fetcher(url=revision_url, max_bytes=_MAX_GITHUB_JSON_BYTES)
    _parse_json_object(
        revision_raw,
        "sole Gist revision API response",
        _MAX_GITHUB_JSON_BYTES,
    )
    api_raw_url = _prevalidate_api_raw_url(
        base=base,
        gist_id=observed_gist_id,
    )
    canonical_revision_raw_url = _canonical_revision_raw_url(
        gist_id=observed_gist_id,
        version=version,
    )
    base_api_protocol_raw = fetcher(
        url=api_raw_url,
        max_bytes=_MAX_PROTOCOL_BYTES,
    )
    revision_protocol_raw = fetcher(
        url=canonical_revision_raw_url,
        max_bytes=_MAX_PROTOCOL_BYTES,
    )
    commits_reobservation_raw = fetcher(
        url=commits_url,
        max_bytes=_MAX_GITHUB_JSON_BYTES,
    )
    observed_at_utc = _observer_clock_timestamp(clock)
    return _build_relationship_condition_reader_qualification_public_anchor_receipt_v2(
        execution_protocol_payload=execution_protocol_payload,
        execution_protocol_raw=execution_protocol_raw,
        expected_execution_protocol_id=expected_execution_protocol_id,
        expected_execution_root=expected_execution_root,
        gist_id=observed_gist_id,
        observed_at_utc=observed_at_utc,
        base_gist_api_raw=base_raw,
        commits_api_raw=commits_raw,
        commits_reobservation_api_raw=commits_reobservation_raw,
        revision_api_raw=revision_raw,
        base_api_protocol_raw=base_api_protocol_raw,
        revision_protocol_raw=revision_protocol_raw,
    )


def _build_relationship_condition_reader_qualification_public_anchor_receipt_v2(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    expected_execution_root: pathlib.Path,
    gist_id: str,
    observed_at_utc: str,
    base_gist_api_raw: bytes,
    commits_api_raw: bytes,
    commits_reobservation_api_raw: bytes,
    revision_api_raw: bytes,
    base_api_protocol_raw: bytes,
    revision_protocol_raw: bytes,
) -> Mapping[str, object]:
    """Build one receipt strictly from the six raw HTTP response bodies."""

    protocol = _mapping(execution_protocol_payload, "V2 execution protocol")
    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol_v2(
        protocol,
        expected_protocol_id=expected_execution_protocol_id,
    )
    _validate_protocol_raw_binding(protocol, execution_protocol_raw)
    expected_root = canonical_relationship_condition_reader_qualification_execution_root_v2(
        expected_execution_root,
        "expected execution root",
    )
    if protocol["proposed_execution_root"] != expected_root:
        raise ValueError("V2 public anchor execution-root lineage mismatch")
    if pathlib.Path(expected_execution_root).exists():
        raise FileExistsError("V2 public anchor requires an absent execution root")
    if base_api_protocol_raw != execution_protocol_raw:
        raise ValueError("base Gist API raw bytes differ from the frozen V2 protocol")
    if revision_protocol_raw != execution_protocol_raw:
        raise ValueError("canonical revision raw bytes differ from the frozen V2 protocol")

    observed_gist_id = _hex_text(gist_id, "gist id", lengths={32})
    base = _parse_json_object(
        base_gist_api_raw,
        "base Gist API response",
        _MAX_GITHUB_JSON_BYTES,
    )
    commits = _parse_json_array(
        commits_api_raw,
        "Gist commits API response",
        _MAX_GITHUB_JSON_BYTES,
    )
    commits_reobservation = _parse_json_array(
        commits_reobservation_api_raw,
        "Gist final commits API response",
        _MAX_GITHUB_JSON_BYTES,
    )
    revision = _parse_json_object(
        revision_api_raw,
        "sole Gist revision API response",
        _MAX_GITHUB_JSON_BYTES,
    )
    if len(commits) != 1:
        raise ValueError("V2 anchor requires exactly one item on commits page 1")
    if len(commits_reobservation) != 1:
        raise ValueError("V2 anchor final commits reobservation requires exactly one item")
    commit = _mapping(commits[0], "sole Gist commit")
    metadata = _validate_cross_endpoint_gist_observation(
        base=base,
        commit=commit,
        revision=revision,
        gist_id=observed_gist_id,
        execution_protocol_raw=execution_protocol_raw,
    )
    _validate_final_commits_reobservation(
        _mapping(commits_reobservation[0], "final sole Gist commit"),
        metadata=metadata,
    )
    observed_at = _github_utc_timestamp(observed_at_utc, "observed_at_utc")

    core = {
        "schema_version": (RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_V2_SCHEMA_VERSION),
        "execution_protocol_id": protocol_id,
        "protocol_raw_sha256": hashlib.sha256(execution_protocol_raw).hexdigest(),
        "protocol_raw_bytes": len(execution_protocol_raw),
        "gist_owner": _GIST_OWNER,
        "gist_id": observed_gist_id,
        "gist_url": metadata["gist_url"],
        "filename": V2_GIST_FILENAME,
        "public": True,
        "created_at": metadata["created_at"],
        "updated_at": metadata["updated_at"],
        "revision_created_at": metadata["revision_created_at"],
        "revision_updated_at": metadata["revision_updated_at"],
        "created_equals_updated_required": False,
        "timestamp_fields_format_validated_only": True,
        "timestamp_order_used_as_revision_authority": False,
        "observed_at_caller_supplied": False,
        "observed_at_recorded_after_final_commits_get": True,
        "sole_version": metadata["sole_version"],
        "sole_committed_at": metadata["sole_committed_at"],
        "base_gist_api_url": metadata["base_gist_api_url"],
        "commits_api_url": metadata["commits_api_url"],
        "revision_api_url": metadata["revision_api_url"],
        "base_api_raw_url": metadata["base_api_raw_url"],
        "revision_api_file_raw_url": metadata["revision_api_file_raw_url"],
        "canonical_revision_raw_url": metadata["canonical_revision_raw_url"],
        "base_gist_response_raw_sha256": hashlib.sha256(base_gist_api_raw).hexdigest(),
        "base_gist_response_raw_bytes": len(base_gist_api_raw),
        "commits_response_raw_sha256": hashlib.sha256(commits_api_raw).hexdigest(),
        "commits_response_raw_bytes": len(commits_api_raw),
        "commits_reobservation_response_raw_sha256": hashlib.sha256(commits_reobservation_api_raw).hexdigest(),
        "commits_reobservation_response_raw_bytes": len(commits_reobservation_api_raw),
        "revision_response_raw_sha256": hashlib.sha256(revision_api_raw).hexdigest(),
        "revision_response_raw_bytes": len(revision_api_raw),
        "base_api_protocol_raw_sha256": hashlib.sha256(base_api_protocol_raw).hexdigest(),
        "base_api_protocol_raw_bytes": len(base_api_protocol_raw),
        "revision_protocol_raw_sha256": hashlib.sha256(revision_protocol_raw).hexdigest(),
        "revision_protocol_raw_bytes": len(revision_protocol_raw),
        "commits_page_item_count": 1,
        "final_commits_page_item_count": 1,
        "commits_get_count": 2,
        "commits_reobserved_after_raw_gets": True,
        "initial_and_final_sole_commit_identity_match": True,
        "observation_linearization_point": "final_commits_reobservation",
        "sole_revision_endpoint_bound": True,
        "base_history_used_as_revision_authority": False,
        "api_file_raw_url_revision_used_as_commit_authority": False,
        "current_file_count": 1,
        "revision_file_count": 1,
        "current_and_revision_public": True,
        "sole_commit_version_bound_to_revision_endpoint": True,
        "sole_commit_timestamp_source": "dedicated_commits_endpoint",
        "base_and_revision_raw_exact_protocol_match": True,
        "github_api_version": GITHUB_API_VERSION,
        "request_headers": {
            "Accept": _GITHUB_ACCEPT,
            "User-Agent": _GITHUB_USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        "authorization_or_cookie_header_used": False,
        "observation_transport": _OBSERVATION_TRANSPORT,
        "observed_at_utc": observed_at,
        "execution_root": expected_root,
        "execution_root_existed_at_observation": False,
        "model_output_count_at_observation": 0,
        "qualification_report_existed_at_observation": False,
        "retired_predecessor_execution_protocol_id": _RETIRED_V1_PREDECESSOR["execution_protocol_id"],
        "retired_predecessor_authorized": False,
    }
    receipt = _with_artifact_id(core)
    _validate_anchor_receipt_shape(receipt)
    return receipt


def validate_relationship_condition_reader_qualification_public_anchor_receipt_v2(
    receipt_payload: Mapping[str, object],
    *,
    expected_receipt_artifact_id: str,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    expected_execution_root: pathlib.Path,
) -> str:
    """Validate a V2 receipt with an independently supplied artifact ID."""

    protocol = _mapping(execution_protocol_payload, "V2 execution protocol")
    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol_v2(
        protocol,
        expected_protocol_id=expected_execution_protocol_id,
    )
    _validate_protocol_raw_binding(protocol, execution_protocol_raw)
    receipt = _mapping(receipt_payload, "V2 public anchor receipt")
    _validate_anchor_receipt_shape(receipt)
    expected_id = _digest(expected_receipt_artifact_id, "expected V2 receipt artifact id")
    if receipt["artifact_id"] != expected_id:
        raise ValueError("V2 receipt differs from the external expected artifact id")
    expected_root = canonical_relationship_condition_reader_qualification_execution_root_v2(
        expected_execution_root,
        "expected execution root",
    )
    if pathlib.Path(expected_execution_root).exists():
        raise FileExistsError("V2 public anchor admission requires an absent execution root")
    raw_sha256 = hashlib.sha256(execution_protocol_raw).hexdigest()
    if (
        receipt["execution_protocol_id"] != protocol_id
        or receipt["protocol_raw_sha256"] != raw_sha256
        or receipt["protocol_raw_bytes"] != len(execution_protocol_raw)
        or receipt["base_api_protocol_raw_sha256"] != raw_sha256
        or receipt["base_api_protocol_raw_bytes"] != len(execution_protocol_raw)
        or receipt["revision_protocol_raw_sha256"] != raw_sha256
        or receipt["revision_protocol_raw_bytes"] != len(execution_protocol_raw)
        or receipt["execution_root"] != expected_root
        or protocol["proposed_execution_root"] != expected_root
    ):
        raise ValueError("V2 receipt protocol/raw/execution-root lineage mismatch")
    anchor = _mapping(protocol["external_public_anchor"], "V2 anchor contract")
    if receipt["gist_owner"] != anchor["gist_owner"] or receipt["filename"] != anchor["filename"]:
        raise ValueError("V2 receipt owner or filename differs from protocol")
    return expected_id


def relationship_condition_reader_qualification_integrity_guard_v2(
    *,
    execution_protocol: Mapping[str, object],
    expected_execution_protocol_id: str,
    repository_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
) -> RelationshipConditionReaderQualificationIntegrityGuardV2:
    """Return a V2-ID-bound guard compatible with the existing stage core."""

    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol_v2(
        execution_protocol,
        expected_protocol_id=expected_execution_protocol_id,
        repository_root=repository_root,
        bge_snapshot_root=bge_snapshot_root,
    )
    frozen = _parse_json_object(
        canonical_json_bytes(dict(execution_protocol)),
        "V2 guard protocol copy",
        _MAX_PROTOCOL_BYTES,
    )
    return RelationshipConditionReaderQualificationIntegrityGuardV2(
        execution_protocol=frozen,
        expected_execution_protocol_id=protocol_id,
        repository_root=pathlib.Path(repository_root).absolute(),
        bge_snapshot_root=pathlib.Path(bge_snapshot_root).absolute(),
    )


def build_relationship_condition_reader_qualification_integrity_receipt_v2(
    *,
    execution_protocol: Mapping[str, object],
    expected_execution_protocol_id: str,
    repository_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
    phase: str,
    previous_integrity_receipt_artifact_id: str | None,
) -> Mapping[str, object]:
    """Reobserve V1-owned domains while binding the receipt to the V2 ID."""

    _assert_model_free()
    if phase not in _INTEGRITY_PHASES:
        raise ValueError(f"unknown V2 integrity phase: {phase}")
    ordinal = _INTEGRITY_PHASES.index(phase)
    if ordinal == 0:
        if previous_integrity_receipt_artifact_id is not None:
            raise ValueError("first V2 integrity phase must not have a previous receipt")
        previous_id = None
    else:
        previous_id = _digest(
            previous_integrity_receipt_artifact_id,
            "previous V2 integrity receipt artifact id",
        )
    protocol = _mapping(execution_protocol, "V2 execution protocol")
    protocol_id = relationship_condition_reader_qualification_execution_protocol_id_v2(protocol)
    if protocol_id != _digest(expected_execution_protocol_id, "expected V2 protocol id"):
        raise ValueError("V2 execution protocol id differs from the external expected id")
    observed_source = v1.build_relationship_condition_reader_execution_source_tree_manifest(
        repository_root=repository_root,
    )
    observed_repository_coverage = build_relationship_condition_reader_repository_runtime_coverage(
        repository_root=repository_root,
        execution_source_tree=observed_source,
    )
    observed_bge = v1.build_bge_m3_snapshot_tree_manifest(
        snapshot_root=bge_snapshot_root,
    )
    observed_runtime = build_relationship_condition_reader_qualification_import_runtime_identity_v2()
    if observed_source != protocol["execution_source_tree"]:
        raise ValueError("V2 integrity guard observed source-tree drift")
    if not _json_type_exact_equal(
        observed_repository_coverage,
        protocol["repository_runtime_coverage"],
    ):
        raise ValueError("V2 integrity guard observed repository runtime coverage drift")
    if observed_bge != protocol["bge_snapshot_tree"]:
        raise ValueError("V2 integrity guard observed BGE snapshot drift")
    if not _json_type_exact_equal(observed_runtime, protocol["runtime_identity"]):
        raise ValueError("V2 integrity guard observed runtime drift")
    _assert_model_free()
    return _with_artifact_id(
        {
            # The existing private stage core freezes this transport schema.
            "schema_version": (v1.RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION),
            "execution_protocol_id": protocol_id,
            "phase": phase,
            "phase_ordinal": ordinal,
            "previous_integrity_receipt_artifact_id": previous_id,
            "source_tree_artifact_id": observed_source["artifact_id"],
            "source_tree_entry_count": observed_source["entry_count"],
            "bge_snapshot_tree_artifact_id": observed_bge["artifact_id"],
            "bge_snapshot_entry_count": observed_bge["entry_count"],
            "runtime_identity_artifact_id": observed_runtime["artifact_id"],
            "source_tree_exact": True,
            "bge_snapshot_tree_exact": True,
            "runtime_identity_exact": True,
            "observer_model_or_cuda_execution_used": False,
            "torch_imported": False,
            "sentence_transformers_imported": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )


def _validate_protocol_shape(protocol: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "evidence_role",
        "qualification_preflight",
        "execution_source_tree",
        "repository_runtime_coverage",
        "bge_snapshot_tree",
        "runtime_identity",
        "proposed_execution_root",
        "external_public_anchor",
        "process_firewall",
        "execution_order",
        "qualification_gates",
        "claims",
        "retired_predecessor",
    }
    _exact_keys(protocol, expected_keys, "V2 execution protocol")
    if protocol["schema_version"] != RELATIONSHIP_READER_EXECUTION_PROTOCOL_V2_SCHEMA_VERSION:
        raise ValueError("V2 execution protocol schema drifted")
    canonical_relationship_condition_reader_qualification_execution_root_v2(
        protocol["proposed_execution_root"],
        "V2 proposed execution root",
    )
    anchor = _mapping(protocol["external_public_anchor"], "V2 anchor contract")
    receipt_path = _relative_receipt_path(anchor.get("receipt_path"))
    if not _json_type_exact_equal(anchor, _external_anchor_contract_v2(receipt_path)):
        raise ValueError("V2 external anchor contract drifted")
    retired = _mapping(protocol["retired_predecessor"], "retired V1 predecessor")
    if not _json_type_exact_equal(retired, _RETIRED_V1_PREDECESSOR):
        raise ValueError("retired V1 predecessor lineage drifted")
    runtime = _mapping(protocol["runtime_identity"], "V2 runtime")
    _validate_import_runtime_identity_v2(runtime)
    validate_relationship_condition_reader_repository_runtime_coverage(
        _mapping(
            protocol["repository_runtime_coverage"],
            "V2 repository runtime coverage",
        ),
        execution_source_tree=_mapping(protocol["execution_source_tree"], "V2 source"),
    )
    # The public V1 composer is the SSOT for all unchanged mechanism fields.
    base = v1.build_relationship_condition_reader_qualification_execution_protocol(
        preflight_binding=_mapping(protocol["qualification_preflight"], "V2 preflight"),
        source_tree_manifest=_mapping(protocol["execution_source_tree"], "V2 source"),
        bge_snapshot_tree_manifest=_mapping(protocol["bge_snapshot_tree"], "V2 BGE"),
        runtime_identity=_v1_runtime_identity_from_v2(runtime),
        proposed_execution_root=pathlib.Path(str(protocol["proposed_execution_root"])),
        anchor_receipt_relative_path=receipt_path,
    )
    for field_name in (
        "evidence_role",
        "qualification_preflight",
        "execution_source_tree",
        "bge_snapshot_tree",
        "proposed_execution_root",
        "process_firewall",
        "execution_order",
        "qualification_gates",
        "claims",
    ):
        if not _json_type_exact_equal(protocol[field_name], base[field_name]):
            raise ValueError(f"V2 reused V1 mechanism drifted: {field_name}")
    _validate_v2_source_closure(_mapping(protocol["execution_source_tree"], "V2 source tree"))


def _validate_v2_source_closure(source_tree: Mapping[str, object]) -> None:
    entries = source_tree.get("entries")
    if not isinstance(entries, list):
        raise ValueError("V2 execution source entries must be a list")
    observed_paths = {
        _text(_mapping(row, "V2 execution source entry").get("path"), "V2 source path") for row in entries
    }
    missing = sorted(_V2_REQUIRED_SOURCE_PATHS - observed_paths)
    if missing:
        raise ValueError(f"V2 execution source tree omits V2 composition sources: {missing}")


def _sole_site_packages_root(base_runtime_identity: Mapping[str, object]) -> pathlib.Path:
    distributions = base_runtime_identity.get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("V1 runtime distributions must be a non-empty list")
    roots: dict[str, pathlib.Path] = {}
    for index, value in enumerate(distributions):
        distribution = _mapping(value, f"V1 runtime distribution {index}")
        dist_info_path = pathlib.Path(
            _canonical_windows_path(
                distribution.get("dist_info_path"),
                f"V1 runtime distribution {index} dist-info path",
            )
        )
        root = dist_info_path.parent
        roots[os.path.normcase(str(root))] = root
    if len(roots) != 1:
        raise ValueError("V2 import runtime requires one shared site-packages root")
    root = next(iter(roots.values()))
    return v1._absolute_directory(root, "V2 site-packages root")


def _python_environment_root(
    *,
    base_runtime_identity: Mapping[str, object],
    site_packages_root: pathlib.Path,
) -> pathlib.Path:
    site_root = pathlib.Path(site_packages_root)
    if site_root.name.casefold() != "site-packages" or site_root.parent.name.casefold() != "lib":
        raise ValueError("V2 site-packages root must be the environment Lib/site-packages directory")
    environment_root = v1._absolute_directory(site_root.parent.parent, "V2 Python environment root")
    python_identity = _mapping(base_runtime_identity.get("python"), "V1 Python identity")
    executable = pathlib.Path(_canonical_windows_path(python_identity.get("executable"), "V1 Python executable"))
    if os.path.normcase(str(executable.parent)) != os.path.normcase(str(environment_root)):
        raise ValueError("V2 Python executable must be directly under its frozen environment root")
    expected_site_root = environment_root / "Lib" / "site-packages"
    if os.path.normcase(str(site_root)) != os.path.normcase(str(expected_site_root)):
        raise ValueError("V2 site-packages root disagrees with the frozen Python environment")
    return environment_root


def _observe_installed_distribution_inventory(
    *,
    site_packages_root: pathlib.Path,
    environment_root: pathlib.Path,
    base_runtime_identity: Mapping[str, object],
) -> list[dict[str, object]]:
    discovered = _discover_dist_info_directories(site_packages_root)
    base_rows = {
        os.path.normcase(
            _canonical_windows_path(
                _mapping(value, f"V1 runtime distribution {index}").get("dist_info_path"),
                f"V1 runtime distribution {index} dist-info path",
            )
        ): _mapping(value, f"V1 runtime distribution {index}")
        for index, value in enumerate(_runtime_distribution_list(base_runtime_identity.get("distributions")))
    }
    rows: list[dict[str, object]] = []
    observed_dist_info_keys: list[str] = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages_root)]):
        dist_info_path = _distribution_dist_info_path(
            distribution,
            site_packages_root=site_packages_root,
        )
        key = os.path.normcase(str(dist_info_path))
        observed_dist_info_keys.append(key)
        row = _observe_distribution_inventory_row(
            distribution,
            dist_info_path=dist_info_path,
            site_packages_root=site_packages_root,
            environment_root=environment_root,
        )
        if key in base_rows:
            expected = _inventory_row_from_v1_distribution_pin(
                base_rows[key],
                site_packages_root=str(site_packages_root),
            )
            _validate_inventory_row_contains_exact_v1_pin(row, expected)
        rows.append(row)
    if len(observed_dist_info_keys) != len(set(observed_dist_info_keys)):
        raise ValueError("installed distribution metadata resolves to duplicate dist-info paths")
    if set(observed_dist_info_keys) != set(discovered):
        missing = sorted(set(discovered) - set(observed_dist_info_keys))
        extra = sorted(set(observed_dist_info_keys) - set(discovered))
        raise ValueError(
            "installed distribution discovery differs from site-packages dist-info directories; "
            f"missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda row: str(row["normalized_name"]).encode("utf-8"))
    normalized_names = [str(row["normalized_name"]) for row in rows]
    if not rows or len(normalized_names) != len(set(normalized_names)):
        raise ValueError("installed distributions must have unique normalized names")
    return rows


def _discover_dist_info_directories(site_packages_root: pathlib.Path) -> dict[str, pathlib.Path]:
    discovered: dict[str, pathlib.Path] = {}
    with os.scandir(site_packages_root) as entries:
        for entry in entries:
            folded = entry.name.casefold()
            if folded.endswith(".egg-info"):
                raise ValueError("V2 import runtime refuses installed distributions without RECORD")
            if not folded.endswith(".dist-info"):
                continue
            value = entry.stat(follow_symlinks=False)
            path = pathlib.Path(entry.path)
            if stat.S_ISLNK(value.st_mode) or v1._is_reparse(value) or not stat.S_ISDIR(value.st_mode):
                raise ValueError(f"installed dist-info must be a non-reparse directory: {path}")
            key = os.path.normcase(str(path))
            if key in discovered:
                raise ValueError("installed dist-info directories collide after path normalization")
            discovered[key] = path
    if not discovered:
        raise ValueError("V2 import runtime found no installed dist-info directories")
    return discovered


def _distribution_dist_info_path(
    distribution: importlib.metadata.Distribution,
    *,
    site_packages_root: pathlib.Path,
) -> pathlib.Path:
    files = distribution.files
    if files is None:
        raise ValueError("installed distribution does not publish a RECORD file list")
    metadata_candidates = sorted(
        (
            item
            for item in files
            if len(pathlib.PurePosixPath(item.as_posix()).parts) == 2
            and pathlib.PurePosixPath(item.as_posix()).parts[0].casefold().endswith(".dist-info")
            and pathlib.PurePosixPath(item.as_posix()).parts[1] == "METADATA"
        ),
        key=lambda item: item.as_posix().encode("utf-8"),
    )
    if len(metadata_candidates) != 1:
        raise ValueError("installed distribution has ambiguous dist-info metadata")
    relative = pathlib.PurePosixPath(metadata_candidates[0].as_posix()).parent
    dist_info_path = pathlib.Path(os.path.abspath(distribution.locate_file(str(relative))))
    v1._assert_directory_without_reparse(
        dist_info_path,
        root=site_packages_root,
        field_name="installed distribution dist-info",
    )
    if os.path.normcase(str(dist_info_path.parent)) != os.path.normcase(str(site_packages_root)):
        raise ValueError("installed distribution dist-info escaped the sole site-packages root")
    return dist_info_path


def _observe_distribution_inventory_row(
    distribution: importlib.metadata.Distribution,
    *,
    dist_info_path: pathlib.Path,
    site_packages_root: pathlib.Path,
    environment_root: pathlib.Path,
) -> dict[str, object]:
    record_path = dist_info_path / "RECORD"
    record_raw = v1._read_stable_regular_file(
        record_path,
        root=site_packages_root,
        field_name="installed distribution RECORD",
        require_single_hardlink=False,
    )
    distribution_name = _text(distribution.metadata["Name"], "installed distribution name")
    verification = _verify_installed_distribution_record_entries_v2(
        record_raw,
        site_packages_root=site_packages_root,
        environment_root=environment_root,
        distribution_name=distribution_name,
        normalized_distribution_name=_normalized_distribution_name(distribution_name),
        inside_entries_preverified=False,
    )
    dist_info_relative = v1._relative_to_root(
        dist_info_path,
        root=site_packages_root,
        field_name="installed distribution dist-info",
    ).as_posix()
    record_relative = f"{dist_info_relative}/RECORD"
    unhashed_non_pyc = verification["record_unhashed_non_pyc_paths"]
    if unhashed_non_pyc != [record_relative]:
        raise ValueError("installed distribution RECORD self path is not canonical")
    total = _positive_integer(verification["record_entry_count"], "RECORD entry count")
    hashed = _positive_integer(verification["record_hashed_entry_count"], "RECORD hashed entry count")
    return {
        "normalized_name": _normalized_distribution_name(distribution_name),
        "distribution_name": distribution_name,
        "version": _text(distribution.version, "installed distribution version"),
        "dist_info_relative_path": dist_info_relative,
        "record_relative_path": record_relative,
        "record_raw_sha256": hashlib.sha256(record_raw).hexdigest(),
        "record_raw_bytes": len(record_raw),
        "record_entry_count": total,
        "record_hashed_entry_count": hashed,
        "record_unhashed_pyc_entry_count": verification["record_unhashed_pyc_entry_count"],
        "record_hashed_site_packages_entry_count": verification["record_hashed_site_packages_entry_count"],
        "record_hashed_environment_external_entry_count": verification[
            "record_hashed_environment_external_entry_count"
        ],
        "record_hashed_absolute_environment_entry_count": verification[
            "record_hashed_absolute_environment_entry_count"
        ],
        "record_hashed_absolute_environment_entries": verification["record_hashed_absolute_environment_entries"],
        "record_hashed_absent_outside_environment_entry_count": verification[
            "record_hashed_absent_outside_environment_entry_count"
        ],
        "record_absent_outside_environment_entries": verification["record_absent_outside_environment_entries"],
        "record_hashed_pinned_identity_mismatch_entry_count": verification[
            "record_hashed_pinned_identity_mismatch_entry_count"
        ],
        "record_pinned_identity_mismatch_entries": verification["record_pinned_identity_mismatch_entries"],
        "record_in_environment_hashed_entries_verified_or_explicitly_pinned": verification[
            "record_in_environment_hashed_entries_verified_or_explicitly_pinned"
        ],
        "record_absent_outside_environment_entries_attested": verification[
            "record_absent_outside_environment_entries_attested"
        ],
    }


def _inventory_row_from_v1_distribution_pin(
    value: Mapping[str, object],
    *,
    site_packages_root: str,
) -> dict[str, object]:
    distribution = _mapping(value, "V1 distribution pin")
    dist_info_path = pathlib.PureWindowsPath(
        _canonical_windows_path(distribution.get("dist_info_path"), "V1 dist-info path")
    )
    site_root = pathlib.PureWindowsPath(_canonical_windows_path(site_packages_root, "site-packages root"))
    try:
        dist_info_relative_windows = dist_info_path.relative_to(site_root)
    except ValueError as exc:
        raise ValueError("V1 dist-info path escaped the sole site-packages root") from exc
    dist_info_relative = pathlib.PurePosixPath(*dist_info_relative_windows.parts).as_posix()
    record = _mapping(distribution.get("record"), "V1 distribution RECORD pin")
    record_pin_relative = _relative_posix_runtime_path(record.get("path"), "V1 RECORD pin path")
    record_relative = f"{dist_info_relative}/{record_pin_relative}"
    unhashed_non_pyc = distribution.get("record_unhashed_non_pyc_paths")
    if not isinstance(unhashed_non_pyc, list) or unhashed_non_pyc != [record_relative]:
        raise ValueError("V1 distribution RECORD self path disagrees with its dist-info path")
    total = _positive_integer(distribution.get("record_entry_count"), "V1 RECORD entry count")
    hashed = _positive_integer(distribution.get("record_hashed_entry_count"), "V1 RECORD hashed entry count")
    distribution_name = _text(distribution.get("distribution_name"), "V1 distribution name")
    return {
        "normalized_name": _normalized_distribution_name(distribution_name),
        "distribution_name": distribution_name,
        "version": _text(distribution.get("version"), "V1 distribution version"),
        "dist_info_relative_path": dist_info_relative,
        "record_relative_path": record_relative,
        "record_raw_sha256": _digest(record.get("raw_sha256"), "V1 RECORD raw sha256"),
        "record_raw_bytes": _positive_integer(record.get("raw_bytes"), "V1 RECORD raw bytes"),
        "record_entry_count": total,
        "record_hashed_entry_count": hashed,
        "record_unhashed_pyc_entry_count": _record_unhashed_pyc_count(total=total, hashed=hashed),
        "record_hashed_entries_verified": True,
    }


def _validate_inventory_row_contains_exact_v1_pin(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Require every V1-pinned field while permitting the stricter V2 closure."""

    shared_fields = (
        "normalized_name",
        "distribution_name",
        "version",
        "dist_info_relative_path",
        "record_relative_path",
        "record_raw_sha256",
        "record_raw_bytes",
        "record_entry_count",
        "record_hashed_entry_count",
        "record_unhashed_pyc_entry_count",
    )
    if any(not _json_type_exact_equal(observed.get(field), expected.get(field)) for field in shared_fields):
        raise ValueError("V2 installed distribution inventory disagrees with an exact V1 pin")


def _record_unhashed_pyc_count(*, total: int, hashed: int) -> int:
    count = total - hashed - 1
    if count < 0:
        raise ValueError("distribution RECORD counts cannot account for its unhashed self entry")
    return count


def _verify_installed_distribution_record_entries_v2(
    record_raw: bytes,
    *,
    site_packages_root: pathlib.Path,
    environment_root: pathlib.Path,
    distribution_name: str,
    normalized_distribution_name: str,
    inside_entries_preverified: bool,
) -> Mapping[str, object]:
    try:
        rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"), newline="")))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{distribution_name} RECORD is not UTF-8") from exc
    if not rows:
        raise ValueError(f"{distribution_name} RECORD is empty")
    site_root = pathlib.Path(os.path.abspath(site_packages_root))
    environment = pathlib.Path(os.path.abspath(environment_root))
    retained_rows: list[list[str]] = []
    observed_paths: list[str] = []
    unhashed_non_pyc: list[str] = []
    unhashed_pyc_count = 0
    hashed_count = 0
    hashed_site_count = 0
    hashed_environment_external_count = 0
    absolute_environment_entries: list[dict[str, object]] = []
    absent_outside_entries: list[dict[str, object]] = []
    pinned_identity_mismatch_entries: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(f"{distribution_name} RECORD row {index} must contain three fields")
        relative_text, hash_field, size_field = row
        relative, candidate, is_absolute_windows_path = _resolve_record_entry_path_v2(
            relative_text,
            site_packages_root=site_root,
            field_name=f"{distribution_name} RECORD path {index}",
        )
        observed_paths.append(relative)
        if not hash_field:
            if size_field:
                raise ValueError(f"{distribution_name} unhashed RECORD row unexpectedly declares a size")
            if relative.casefold().endswith(".pyc"):
                unhashed_pyc_count += 1
            else:
                unhashed_non_pyc.append(relative)
            retained_rows.append(row)
            continue
        declared_sha256, declared_bytes = _record_declared_raw_identity(
            hash_field=hash_field,
            size_field=size_field,
            field_name=f"{distribution_name} RECORD row {index}",
        )
        hashed_count += 1
        try:
            environment_relative = candidate.relative_to(environment)
        except ValueError:
            absent_outside_entries.append(
                _absent_pip_console_script_record_exclusion(
                    normalized_distribution_name=normalized_distribution_name,
                    record_path=relative,
                    resolved_target=candidate,
                    environment_root=environment,
                    declared_sha256=declared_sha256,
                    declared_bytes=declared_bytes,
                )
            )
            continue
        actual_sha256, actual_bytes = v1._hash_stable_regular_file(
            candidate,
            root=environment,
            field_name=f"{distribution_name} RECORD entry {relative}",
            require_single_hardlink=False,
        )
        if actual_sha256 != declared_sha256 or actual_bytes != declared_bytes:
            pinned_identity_mismatch_entries.append(
                _pinned_nonimportable_record_identity_mismatch(
                    normalized_distribution_name=normalized_distribution_name,
                    record_path=relative,
                    resolved_target=candidate,
                    site_packages_root=site_root,
                    environment_root=environment,
                    declared_sha256=declared_sha256,
                    declared_bytes=declared_bytes,
                    observed_sha256=actual_sha256,
                    observed_bytes=actual_bytes,
                )
            )
            continue
        try:
            candidate.relative_to(site_root)
        except ValueError:
            _validate_record_environment_external_path(
                environment_relative,
                distribution_name=distribution_name,
            )
            hashed_environment_external_count += 1
            if is_absolute_windows_path:
                absolute_environment_entries.append(
                    {
                        "record_path": relative,
                        "resolved_target": str(candidate),
                        "declared_raw_sha256": declared_sha256,
                        "declared_raw_bytes": declared_bytes,
                        "observed_raw_sha256": actual_sha256,
                        "observed_raw_bytes": actual_bytes,
                        "target_within_permitted_environment_external_root": True,
                    }
                )
        else:
            hashed_site_count += 1
        if not is_absolute_windows_path:
            retained_rows.append(row)
    if len(observed_paths) != len(set(observed_paths)) or len(observed_paths) != len(
        {path.casefold() for path in observed_paths}
    ):
        raise ValueError(f"{distribution_name} RECORD contains duplicate or case-colliding paths")
    if len(unhashed_non_pyc) != 1 or not unhashed_non_pyc[0].endswith(".dist-info/RECORD"):
        raise ValueError(f"{distribution_name} RECORD has unexpected unhashed non-pyc paths")
    absent_outside_entries.sort(key=lambda row: str(row["record_path"]).encode("utf-8"))
    pinned_identity_mismatch_entries.sort(key=lambda row: str(row["record_path"]).encode("utf-8"))
    absolute_environment_entries.sort(key=lambda row: str(row["record_path"]).encode("utf-8"))
    if not inside_entries_preverified:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(retained_rows)
        verified = v1._verify_record_entries(
            stream.getvalue().encode("utf-8"),
            record_base=site_root,
            environment_root=environment,
            field_name=distribution_name,
        )
        if (
            verified["record_entry_count"] != len(retained_rows)
            or verified["record_hashed_entry_count"]
            != hashed_count
            - len(absent_outside_entries)
            - len(pinned_identity_mismatch_entries)
            - len(absolute_environment_entries)
            or verified["record_unhashed_non_pyc_paths"] != unhashed_non_pyc
        ):
            raise RuntimeError("V2 RECORD verification result disagrees with its closure classifier")
    return {
        "record_entry_count": len(rows),
        "record_hashed_entry_count": hashed_count,
        "record_unhashed_non_pyc_paths": unhashed_non_pyc,
        "record_unhashed_pyc_entry_count": unhashed_pyc_count,
        "record_hashed_site_packages_entry_count": hashed_site_count,
        "record_hashed_environment_external_entry_count": (hashed_environment_external_count),
        "record_hashed_absolute_environment_entry_count": len(absolute_environment_entries),
        "record_hashed_absolute_environment_entries": absolute_environment_entries,
        "record_hashed_absent_outside_environment_entry_count": len(absent_outside_entries),
        "record_absent_outside_environment_entries": absent_outside_entries,
        "record_hashed_pinned_identity_mismatch_entry_count": len(pinned_identity_mismatch_entries),
        "record_pinned_identity_mismatch_entries": pinned_identity_mismatch_entries,
        "record_in_environment_hashed_entries_verified_or_explicitly_pinned": True,
        "record_absent_outside_environment_entries_attested": True,
    }


def _record_declared_raw_identity(
    *,
    hash_field: str,
    size_field: str,
    field_name: str,
) -> tuple[str, int]:
    if not size_field:
        raise ValueError(f"{field_name} hashed entry is missing a size")
    try:
        declared_bytes = int(size_field)
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid size") from exc
    if declared_bytes < 0:
        raise ValueError(f"{field_name} has a negative size")
    algorithm, separator, encoded = hash_field.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise ValueError(f"{field_name} permits only sha256 hashes")
    try:
        declared_sha256 = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        ).hex()
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} has invalid URL-safe base64") from exc
    if len(declared_sha256) != 64:
        raise ValueError(f"{field_name} does not contain a SHA-256 digest")
    return declared_sha256, declared_bytes


def _resolve_record_entry_path_v2(
    value: str,
    *,
    site_packages_root: pathlib.Path,
    field_name: str,
) -> tuple[str, pathlib.Path, bool]:
    """Resolve either a canonical RECORD-relative path or a Windows absolute path."""

    if re.match(r"^[A-Za-z]:", value) is not None:
        pure = _canonical_absolute_windows_record_path(value, field_name)
        candidate = pathlib.Path(os.path.abspath(str(pure)))
        if pathlib.PureWindowsPath(str(candidate)).as_posix() != value:
            raise ValueError(f"{field_name} Windows absolute path changed under canonicalization")
        if os.path.lexists(candidate):
            final_path = candidate.resolve(strict=True)
            if os.path.normcase(str(final_path)) != os.path.normcase(str(candidate)):
                raise ValueError(f"{field_name} Windows absolute path is a filesystem alias")
        return value, candidate, True
    relative = v1._record_relative_posix_path(value, field_name)
    candidate = pathlib.Path(os.path.abspath(site_packages_root / pathlib.PurePosixPath(relative)))
    return relative, candidate, False


def _canonical_absolute_windows_record_path(
    value: object,
    field_name: str,
) -> pathlib.PureWindowsPath:
    text = _text(value, field_name)
    if (
        re.fullmatch(
            r'[A-Za-z]:/[^\x00-\x1f<>:"/\\|?*]+(?:/[^\x00-\x1f<>:"/\\|?*]+)*',
            text,
        )
        is None
    ):
        raise ValueError(f"{field_name} Windows absolute path contains an invalid component")
    path = pathlib.PureWindowsPath(text)
    reserved = {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    components = path.parts[1:]
    if (
        not path.is_absolute()
        or path.as_posix() != text
        or any(
            component in {"", ".", ".."}
            or component.endswith((" ", "."))
            or component.split(".", maxsplit=1)[0].casefold() in reserved
            for component in components
        )
    ):
        raise ValueError(f"{field_name} Windows absolute path is not canonical")
    return path


def _validate_record_environment_external_path(
    relative: pathlib.Path,
    *,
    distribution_name: str,
) -> None:
    parts = relative.parts
    allowed = bool(parts) and (
        parts[0].casefold() in {"scripts", "bin"}
        or (len(parts) >= 3 and parts[0].casefold() == "share" and parts[1].casefold() == "man")
    )
    if not allowed:
        raise ValueError(
            f"{distribution_name} RECORD entry escaped permitted Python-environment external roots: {relative}"
        )


def _absent_pip_console_script_record_exclusion(
    *,
    normalized_distribution_name: str,
    record_path: str,
    resolved_target: pathlib.Path,
    environment_root: pathlib.Path,
    declared_sha256: str,
    declared_bytes: int,
) -> dict[str, object]:
    match = re.fullmatch(r"\.\./\.\./\.\./bin/(pip|pip3|pip3\.[0-9]+)", record_path)
    if normalized_distribution_name != "pip" or match is None:
        raise ValueError("installed distribution RECORD entry escaped the frozen Python environment")
    expected_parent = pathlib.Path(os.path.abspath(environment_root.parent / "bin"))
    _reject_reparse_components(expected_parent, "pip outside-environment console-script parent")
    if os.path.lexists(expected_parent):
        parent_stat = os.lstat(expected_parent)
        if stat.S_ISLNK(parent_stat.st_mode) or v1._is_reparse(parent_stat) or not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("pip outside-environment console-script parent must be a regular directory")
    expected_target = expected_parent / match.group(1)
    if os.path.normcase(str(resolved_target)) != os.path.normcase(str(expected_target)):
        raise ValueError("pip outside-environment console-script target is not canonical exact")
    if os.path.lexists(resolved_target):
        raise ValueError("pip outside-environment console-script target must remain absent")
    return {
        "record_path": record_path,
        "resolved_target": str(expected_target),
        "declared_raw_sha256": declared_sha256,
        "declared_raw_bytes": declared_bytes,
        "target_absent": True,
        "child_import_or_controlled_path_reachable": False,
        "exclusion_reason": ("absent_pip_console_script_declared_outside_frozen_python_environment.v1"),
    }


def _pinned_nonimportable_record_identity_mismatch(
    *,
    normalized_distribution_name: str,
    record_path: str,
    resolved_target: pathlib.Path,
    site_packages_root: pathlib.Path,
    environment_root: pathlib.Path,
    declared_sha256: str,
    declared_bytes: int,
    observed_sha256: str,
    observed_bytes: int,
) -> dict[str, object]:
    if normalized_distribution_name == "pip" and record_path.casefold().endswith(".dist-info/installer"):
        expected_target = pathlib.Path(os.path.abspath(site_packages_root / pathlib.PurePosixPath(record_path)))
        if os.path.normcase(str(resolved_target)) != os.path.normcase(str(expected_target)):
            raise ValueError("pip INSTALLER mismatch target is not canonical exact")
        raw = v1._read_stable_regular_file(
            expected_target,
            root=site_packages_root,
            field_name="conda-rewritten pip INSTALLER metadata",
            require_single_hardlink=True,
        )
        if raw != b"conda":
            raise ValueError("only the exact conda-rewritten pip INSTALLER metadata is permitted")
        exclusion_reason = "conda_rewritten_pip_installer_metadata_not_importable.v1"
    elif normalized_distribution_name == "wheel":
        expected_target = pathlib.Path(os.path.abspath(environment_root / "Scripts" / "wheel.exe"))
        expected_record_path = pathlib.PureWindowsPath(str(expected_target)).as_posix()
        if record_path != expected_record_path:
            raise ValueError("wheel console-script mismatch RECORD path is not canonical exact")
        if os.path.normcase(str(resolved_target)) != os.path.normcase(str(expected_target)):
            raise ValueError("wheel console-script mismatch target is not canonical exact")
        raw = v1._read_stable_regular_file(
            expected_target,
            root=environment_root,
            field_name="conda-rewritten wheel console-script launcher",
            require_single_hardlink=True,
        )
        exclusion_reason = "raw_pinned_wheel_console_script_not_in_controlled_path.v1"
    else:
        raise ValueError(f"{normalized_distribution_name} RECORD entry {record_path} identity mismatch")
    if hashlib.sha256(raw).hexdigest() != observed_sha256 or len(raw) != observed_bytes:
        raise RuntimeError("nonimportable RECORD mismatch raw pin changed during verification")
    return {
        "record_path": record_path,
        "resolved_target": str(expected_target),
        "declared_raw_sha256": declared_sha256,
        "declared_raw_bytes": declared_bytes,
        "observed_raw_sha256": observed_sha256,
        "observed_raw_bytes": observed_bytes,
        "target_raw_pinned": True,
        "child_import_or_controlled_path_reachable": False,
        "exclusion_reason": exclusion_reason,
    }


def _observe_site_packages_coverage(
    *,
    site_packages_root: pathlib.Path,
    environment_root: pathlib.Path,
    distribution_inventory: list[dict[str, object]],
) -> Mapping[str, object]:
    root = v1._absolute_directory(site_packages_root, "V2 site-packages coverage root")
    ownership_rows, record_self_paths = _site_packages_record_ownership(
        site_packages_root=root,
        environment_root=environment_root,
        distribution_inventory=distribution_inventory,
    )
    ownership_by_path = {str(row["path"]).casefold(): row for row in ownership_rows}
    record_self_by_path = {path.casefold(): path for path in record_self_paths}
    seen_paths: dict[str, str] = {}
    unowned_files: list[dict[str, object]] = []
    total_regular_files = 0
    owned_hashed_files = 0
    observed_record_self_files = 0
    excluded_bytecode_files = 0
    excluded_site_startup_files = 0
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = pathlib.Path(current)
        v1._assert_directory_without_reparse(
            current_path,
            root=root,
            field_name="V2 site-packages coverage directory",
        )
        retained_directories: list[str] = []
        for dirname in sorted(dirnames, key=lambda name: name.encode("utf-8")):
            candidate = current_path / dirname
            v1._assert_directory_without_reparse(
                candidate,
                root=root,
                field_name="V2 site-packages coverage directory",
            )
            retained_directories.append(dirname)
        dirnames[:] = retained_directories
        for filename in sorted(filenames, key=lambda name: name.encode("utf-8")):
            path = current_path / filename
            value = os.lstat(path)
            if stat.S_ISLNK(value.st_mode) or v1._is_reparse(value) or not stat.S_ISREG(value.st_mode):
                raise ValueError(f"site-packages coverage requires regular non-reparse files: {path}")
            if value.st_nlink != 1:
                raise ValueError(f"site-packages coverage refuses a hard-linked file: {path}")
            relative_path = path.relative_to(root).as_posix()
            folded_path = relative_path.casefold()
            if folded_path in seen_paths:
                raise ValueError("site-packages files collide after case-insensitive normalization")
            seen_paths[folded_path] = relative_path
            total_regular_files += 1
            parts = pathlib.PurePosixPath(relative_path).parts
            is_in_bytecode_cache = any(
                part.casefold() in {value.casefold() for value in _RUNTIME_TREE_CACHE_DIRECTORY_NAMES}
                for part in parts[:-1]
            )
            is_bytecode_file = relative_path.casefold().endswith(
                tuple(value.casefold() for value in _RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES)
            )
            if is_bytecode_file and not is_in_bytecode_cache:
                raise ValueError("site-packages coverage refuses adjacent bytecode outside __pycache__")
            if is_bytecode_file:
                excluded_bytecode_files += 1
            elif relative_path.casefold().endswith(
                tuple(value.casefold() for value in _SITE_PACKAGES_EXCLUDED_SITE_STARTUP_SUFFIXES)
            ):
                excluded_site_startup_files += 1
            elif folded_path in ownership_by_path:
                owned_hashed_files += 1
            elif folded_path in record_self_by_path:
                observed_record_self_files += 1
            else:
                raw_sha256, raw_bytes = v1._hash_stable_regular_file(
                    path,
                    root=root,
                    field_name=f"unowned site-packages file {relative_path}",
                    require_single_hardlink=True,
                )
                unowned_files.append(
                    {
                        "path": relative_path,
                        "raw_sha256": raw_sha256,
                        "raw_bytes": raw_bytes,
                    }
                )
    missing_owned = sorted(set(ownership_by_path) - set(seen_paths))
    missing_records = sorted(set(record_self_by_path) - set(seen_paths))
    if missing_owned or missing_records:
        raise ValueError(
            f"site-packages RECORD ownership contains absent files; owned={missing_owned}, records={missing_records}"
        )
    unowned_files.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    core = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_SITE_PACKAGES_COVERAGE_V2_SCHEMA_VERSION,
        "root": str(root),
        "coverage_contract": ("every_nonstartup_noncache_bytecode_file_is_record_owned_or_raw_pinned.v1"),
        "link_contract": "regular_non_symlink_non_reparse_single_link_files_only.v1",
        "cache_directory_names": list(_RUNTIME_TREE_CACHE_DIRECTORY_NAMES),
        "excluded_cache_bytecode_suffixes": list(_RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES),
        "excluded_site_startup_suffixes": list(_SITE_PACKAGES_EXCLUDED_SITE_STARTUP_SUFFIXES),
        "record_ownership_join_sha256": hashlib.sha256(canonical_json_bytes({"entries": ownership_rows})).hexdigest(),
        "record_hashed_site_path_count": len(ownership_rows),
        "record_ownership_overlap_count": sum(
            1 for row in ownership_rows if len(_string_list(row["record_paths"], "record paths")) > 1
        ),
        "total_regular_file_count": total_regular_files,
        "owned_hashed_record_file_count": owned_hashed_files,
        "record_self_file_count": observed_record_self_files,
        "excluded_bytecode_file_count": excluded_bytecode_files,
        "excluded_site_startup_file_count": excluded_site_startup_files,
        "unowned_regular_files": unowned_files,
        "unowned_regular_file_count": len(unowned_files),
        "symlink_or_reparse_entry_count": 0,
        "non_single_link_regular_file_count": 0,
        "all_nonexcluded_regular_files_record_owned_or_raw_pinned": True,
    }
    payload = _with_artifact_id(core)
    _validate_site_packages_coverage(
        payload,
        expected_root=str(root),
        expected_distribution_count=len(distribution_inventory),
    )
    return payload


def _site_packages_record_ownership(
    *,
    site_packages_root: pathlib.Path,
    environment_root: pathlib.Path,
    distribution_inventory: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    ownership: dict[str, dict[str, object]] = {}
    record_self_paths: list[str] = []
    environment = pathlib.Path(os.path.abspath(environment_root))
    for index, raw_row in enumerate(distribution_inventory):
        row = _mapping(raw_row, f"installed distribution coverage row {index}")
        raw_exclusions = row.get("record_absent_outside_environment_entries")
        if not isinstance(raw_exclusions, list):
            raise ValueError("installed distribution coverage row lacks outside-environment exclusions")
        excluded_record_paths = {
            _text(
                _mapping(value, f"installed distribution exclusion {index}:{exclusion_index}").get("record_path"),
                f"installed distribution exclusion record path {index}:{exclusion_index}",
            )
            for exclusion_index, value in enumerate(raw_exclusions)
        }
        raw_mismatches = row.get("record_pinned_identity_mismatch_entries")
        if not isinstance(raw_mismatches, list):
            raise ValueError("installed distribution coverage row lacks pinned RECORD mismatches")
        mismatched_record_paths = {
            _text(
                _mapping(value, f"installed distribution mismatch {index}:{mismatch_index}").get("record_path"),
                f"installed distribution mismatch record path {index}:{mismatch_index}",
            )
            for mismatch_index, value in enumerate(raw_mismatches)
        }
        record_relative = _relative_posix_runtime_path(
            row["record_relative_path"],
            f"installed distribution RECORD path {index}",
        )
        record_path = site_packages_root / pathlib.PurePosixPath(record_relative)
        record_raw = v1._read_stable_regular_file(
            record_path,
            root=site_packages_root,
            field_name=f"installed distribution RECORD coverage {index}",
            require_single_hardlink=True,
        )
        if (
            hashlib.sha256(record_raw).hexdigest() != row["record_raw_sha256"]
            or len(record_raw) != row["record_raw_bytes"]
        ):
            raise ValueError("site-packages coverage RECORD disagrees with distribution inventory")
        try:
            record_rows = list(csv.reader(io.StringIO(record_raw.decode("utf-8"), newline="")))
        except UnicodeDecodeError as exc:
            raise ValueError("site-packages coverage RECORD is not UTF-8") from exc
        record_self_observed = False
        for record_index, record_row in enumerate(record_rows):
            if len(record_row) != 3:
                raise ValueError("site-packages coverage RECORD row must contain three fields")
            relative_text, candidate, _is_absolute_windows_path = _resolve_record_entry_path_v2(
                record_row[0],
                site_packages_root=site_packages_root,
                field_name=f"site-packages coverage RECORD path {index}:{record_index}",
            )
            try:
                candidate_relative = candidate.relative_to(site_packages_root).as_posix()
            except ValueError:
                try:
                    candidate.relative_to(environment)
                except ValueError:
                    if relative_text not in excluded_record_paths:
                        raise ValueError(
                            "site-packages coverage observed an unregistered RECORD path "
                            "outside the frozen Python environment"
                        ) from None
                continue
            if not record_row[1]:
                if candidate_relative.casefold() == record_relative.casefold():
                    record_self_observed = True
                continue
            if relative_text in mismatched_record_paths:
                continue
            folded = candidate_relative.casefold()
            existing = ownership.get(folded)
            if existing is None:
                ownership[folded] = {
                    "path": candidate_relative,
                    "record_paths": {record_relative},
                }
            else:
                if existing["path"] != candidate_relative:
                    raise ValueError("RECORD-owned paths collide after case normalization")
                record_paths = existing["record_paths"]
                if not isinstance(record_paths, set):
                    raise TypeError("internal RECORD ownership paths must be a set")
                record_paths.add(record_relative)
        if not record_self_observed:
            raise ValueError("site-packages coverage did not observe the inventory RECORD self row")
        record_self_paths.append(record_relative)
    rows = [
        {
            "path": str(value["path"]),
            "record_paths": sorted(
                _string_set(value["record_paths"], "RECORD ownership paths"),
                key=lambda item: item.encode("utf-8"),
            ),
        }
        for value in ownership.values()
    ]
    rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    record_self_paths.sort(key=lambda item: item.encode("utf-8"))
    if len(record_self_paths) != len({path.casefold() for path in record_self_paths}):
        raise ValueError("installed distribution RECORD self paths must be unique")
    return rows, record_self_paths


def _observe_python_stdlib_zip(
    *,
    environment_root: pathlib.Path,
    python_identity: Mapping[str, object],
) -> Mapping[str, object]:
    root = v1._absolute_directory(environment_root, "V2 Python stdlib zip environment root")
    major, minor = _python_major_minor(python_identity.get("version"))
    path = root / f"python{major}{minor}.zip"
    if os.path.lexists(path):
        raw_sha256, raw_bytes = v1._hash_stable_regular_file(
            path,
            root=root,
            field_name="V2 Python stdlib zip",
            require_single_hardlink=True,
        )
        exists = True
    else:
        raw_sha256 = None
        raw_bytes = None
        exists = False
    payload = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_EXECUTION_PYTHON_STDLIB_ZIP_V2_SCHEMA_VERSION,
            "path": str(path),
            "exists": exists,
            "raw_sha256": raw_sha256,
            "raw_bytes": raw_bytes,
            "precedence_contract": "python_home_zip_precedes_dlls_and_lib_on_minus_P_minus_S_sys_path.v1",
            "link_contract": "absent_or_regular_non_symlink_non_reparse_single_link_file.v1",
        }
    )
    _validate_python_stdlib_zip(
        payload,
        expected_environment_root=str(root),
        python_identity=python_identity,
    )
    return payload


def _observe_python_home_top_level_tree(
    *,
    environment_root: pathlib.Path,
) -> Mapping[str, object]:
    """Pin the Python application directory used before controlled ``PATH`` roots."""

    root = v1._absolute_directory(environment_root, "V2 Python-home top-level root")
    entries: list[dict[str, object]] = []
    directories: list[str] = []
    seen_names: set[str] = set()
    with os.scandir(root) as iterator:
        scanned = sorted(iterator, key=lambda entry: entry.name.encode("utf-8"))
    for entry in scanned:
        folded = entry.name.casefold()
        if folded in seen_names:
            raise ValueError("Python-home top-level entries collide after case normalization")
        seen_names.add(folded)
        path = pathlib.Path(entry.path)
        value = os.lstat(path)
        if stat.S_ISLNK(value.st_mode) or v1._is_reparse(value):
            raise ValueError(f"Python-home top-level entry must not be a link or reparse point: {path}")
        if stat.S_ISDIR(value.st_mode):
            directories.append(entry.name)
            continue
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"Python-home top-level entry must be a regular file or directory: {path}")
        if value.st_nlink != 1:
            raise ValueError(f"Python-home top-level file must have one hard link: {path}")
        if entry.name.casefold().endswith(".pyc"):
            raise ValueError(f"Python-home top-level refuses sourceless adjacent bytecode: {path}")
        raw_sha256, raw_bytes = v1._hash_stable_regular_file(
            path,
            root=root,
            field_name=f"V2 Python-home top-level file {entry.name}",
            require_single_hardlink=True,
        )
        entries.append(
            {
                "path": entry.name,
                "raw_sha256": raw_sha256,
                "raw_bytes": raw_bytes,
            }
        )
    core = {
        "schema_version": (RELATIONSHIP_READER_EXECUTION_PYTHON_HOME_TOP_LEVEL_TREE_V2_SCHEMA_VERSION),
        "tree_role": _PYTHON_HOME_TOP_LEVEL_TREE_ROLE,
        "root": str(root),
        "content_contract": "exact_raw_bytes_no_eol_normalization.v1",
        "link_contract": "regular_non_symlink_non_reparse_single_link_files_only.v1",
        "directory_contract": "direct_children_non_symlink_non_reparse_directories_names_frozen.v1",
        "directories": directories,
        "directory_count": len(directories),
        "entries": entries,
        "entry_count": len(entries),
        "total_raw_bytes": sum(int(row["raw_bytes"]) for row in entries),
    }
    payload = _with_artifact_id(core)
    _validate_python_home_top_level_tree(payload, expected_root=str(root))
    return payload


def _validate_python_home_top_level_tree(
    payload: Mapping[str, object],
    *,
    expected_root: str,
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "tree_role",
            "root",
            "content_contract",
            "link_contract",
            "directory_contract",
            "directories",
            "directory_count",
            "entries",
            "entry_count",
            "total_raw_bytes",
            "artifact_id",
        },
        "V2 Python-home top-level tree",
    )
    required = {
        "schema_version": (RELATIONSHIP_READER_EXECUTION_PYTHON_HOME_TOP_LEVEL_TREE_V2_SCHEMA_VERSION),
        "tree_role": _PYTHON_HOME_TOP_LEVEL_TREE_ROLE,
        "root": expected_root,
        "content_contract": "exact_raw_bytes_no_eol_normalization.v1",
        "link_contract": "regular_non_symlink_non_reparse_single_link_files_only.v1",
        "directory_contract": ("direct_children_non_symlink_non_reparse_directories_names_frozen.v1"),
    }
    if any(payload[key] != value for key, value in required.items()):
        raise ValueError("V2 Python-home top-level tree contract drifted")
    directories = payload["directories"]
    if not isinstance(directories, list) or not all(
        isinstance(value, str) and value and "/" not in value and "\\" not in value and value not in {".", ".."}
        for value in directories
    ):
        raise ValueError("V2 Python-home top-level directories must be simple names")
    if directories != sorted(directories, key=lambda value: value.encode("utf-8")) or len(
        {value.casefold() for value in directories}
    ) != len(directories):
        raise ValueError("V2 Python-home top-level directories are not canonical unique order")
    if _nonnegative_integer(
        payload["directory_count"],
        "V2 Python-home top-level directory count",
    ) != len(directories):
        raise ValueError("V2 Python-home top-level directory count mismatch")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("V2 Python-home top-level entries must be a list")
    entries: list[Mapping[str, object]] = []
    paths: list[str] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"V2 Python-home top-level entry {index}")
        _exact_keys(
            entry,
            {"path", "raw_sha256", "raw_bytes"},
            f"V2 Python-home top-level entry {index}",
        )
        path = _text(entry["path"], f"V2 Python-home top-level path {index}")
        if path in {".", ".."} or "/" in path or "\\" in path or path.casefold().endswith(".pyc"):
            raise ValueError("V2 Python-home top-level file path is not an allowed simple name")
        _digest(entry["raw_sha256"], f"V2 Python-home top-level digest {index}")
        _nonnegative_integer(entry["raw_bytes"], f"V2 Python-home top-level bytes {index}")
        entries.append(entry)
        paths.append(path)
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")) or len(
        {value.casefold() for value in paths}
    ) != len(paths):
        raise ValueError("V2 Python-home top-level files are not canonical unique order")
    if set(value.casefold() for value in paths) & set(value.casefold() for value in directories):
        raise ValueError("V2 Python-home top-level file and directory names collide")
    if _nonnegative_integer(
        payload["entry_count"],
        "V2 Python-home top-level entry count",
    ) != len(entries):
        raise ValueError("V2 Python-home top-level entry count mismatch")
    total = sum(int(entry["raw_bytes"]) for entry in entries)
    if (
        _nonnegative_integer(
            payload["total_raw_bytes"],
            "V2 Python-home top-level total bytes",
        )
        != total
    ):
        raise ValueError("V2 Python-home top-level total byte count mismatch")
    _validate_artifact_id(payload, "V2 Python-home top-level tree")


def _python_major_minor(value: object) -> tuple[int, int]:
    version = _text(value, "Python version")
    components = version.split(".")
    if len(components) < 2 or not components[0].isdigit() or not components[1].isdigit():
        raise ValueError("Python version must begin with numeric major.minor components")
    return int(components[0]), int(components[1])


def _normalized_distribution_name(value: object) -> str:
    name = _text(value, "distribution name")
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if not normalized or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized) is None:
        raise ValueError("distribution name cannot be canonically normalized")
    return normalized


def _observe_runtime_raw_tree(
    *,
    root: pathlib.Path,
    tree_role: str,
    excluded_top_level_directories: tuple[str, ...],
) -> Mapping[str, object]:
    tree_root = v1._absolute_directory(root, f"V2 {tree_role} root")
    paths: list[pathlib.Path] = []
    excluded_cache_bytecode_file_count = 0
    for current, dirnames, filenames in os.walk(tree_root, topdown=True, followlinks=False):
        current_path = pathlib.Path(current)
        v1._assert_directory_without_reparse(
            current_path,
            root=tree_root,
            field_name=f"V2 {tree_role} directory",
        )
        relative = current_path.relative_to(tree_root)
        retained_directories: list[str] = []
        for dirname in sorted(dirnames, key=lambda name: name.encode("utf-8")):
            candidate_directory = current_path / dirname
            v1._assert_directory_without_reparse(
                candidate_directory,
                root=tree_root,
                field_name=f"V2 {tree_role} directory",
            )
            folded_dirname = dirname.casefold()
            if not relative.parts and folded_dirname in {value.casefold() for value in excluded_top_level_directories}:
                continue
            retained_directories.append(dirname)
        dirnames[:] = retained_directories
        for filename in sorted(filenames, key=lambda name: name.encode("utf-8")):
            is_cache_bytecode = filename.casefold().endswith(
                tuple(value.casefold() for value in _RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES)
            )
            inside_cache = any(
                part.casefold() in {value.casefold() for value in _RUNTIME_TREE_CACHE_DIRECTORY_NAMES}
                for part in relative.parts
            )
            if is_cache_bytecode:
                if not inside_cache:
                    raise ValueError(
                        f"V2 {tree_role} refuses adjacent bytecode outside __pycache__: {current_path / filename}"
                    )
                excluded_cache_bytecode_file_count += 1
                continue
            paths.append(current_path / filename)
    entries: list[dict[str, object]] = []
    for path in sorted(
        paths,
        key=lambda candidate: candidate.relative_to(tree_root).as_posix().encode("utf-8"),
    ):
        relative_path = path.relative_to(tree_root).as_posix()
        raw_sha256, raw_bytes = v1._hash_stable_regular_file(
            path,
            root=tree_root,
            field_name=f"V2 {tree_role} {relative_path}",
            require_single_hardlink=False,
        )
        entries.append(
            {
                "path": relative_path,
                "raw_sha256": raw_sha256,
                "raw_bytes": raw_bytes,
            }
        )
    core = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_RUNTIME_RAW_TREE_V2_SCHEMA_VERSION,
        "tree_role": tree_role,
        "root": str(tree_root),
        "content_contract": "exact_raw_bytes_no_eol_normalization.v1",
        "link_contract": "regular_non_symlink_non_reparse_hardlinks_allowed.v1",
        "excluded_top_level_directories": list(excluded_top_level_directories),
        "cache_directory_names": list(_RUNTIME_TREE_CACHE_DIRECTORY_NAMES),
        "excluded_cache_bytecode_suffixes": list(_RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES),
        "excluded_cache_bytecode_file_count": excluded_cache_bytecode_file_count,
        "entries": entries,
        "entry_count": len(entries),
        "total_raw_bytes": sum(int(row["raw_bytes"]) for row in entries),
    }
    payload = _with_artifact_id(core)
    _validate_runtime_raw_tree(
        payload,
        expected_role=tree_role,
        expected_root=str(tree_root),
        expected_excluded_top_level_directories=excluded_top_level_directories,
    )
    return payload


def _validate_import_runtime_identity_v2(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "platform",
            "gpu",
            "python",
            "distributions",
            "v1_runtime_identity_artifact_id",
            "child_import_contract",
            "python_environment_root",
            "site_packages_root",
            "installed_distributions",
            "installed_distribution_count",
            "site_packages_coverage",
            "python_stdlib_zip",
            "python_home_top_level_tree",
            "stdlib_lib_tree",
            "dlls_tree",
            "python_library_bin_tree",
            "artifact_id",
        },
        "V2 import runtime identity",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_V2_SCHEMA_VERSION:
        raise ValueError("V2 import runtime identity schema drifted")
    base_runtime = _v1_runtime_identity_from_v2(payload)
    v1.validate_relationship_condition_reader_qualification_runtime_identity(base_runtime)
    contract = _mapping(payload["child_import_contract"], "V2 child import contract")
    if not _json_type_exact_equal(contract, _CHILD_IMPORT_CONTRACT_V2):
        raise ValueError("V2 child import contract drifted")
    environment_root = _canonical_windows_path(
        payload["python_environment_root"],
        "V2 Python environment root",
    )
    site_packages_root = _canonical_windows_path(payload["site_packages_root"], "V2 site-packages root")
    environment_path = pathlib.PureWindowsPath(environment_root)
    site_path = pathlib.PureWindowsPath(site_packages_root)
    if str(site_path).casefold() != str(environment_path / "Lib" / "site-packages").casefold():
        raise ValueError("V2 site-packages root does not belong to the Python environment")
    python_identity = _mapping(payload["python"], "V2 Python identity")
    python_executable = pathlib.PureWindowsPath(
        _canonical_windows_path(python_identity["executable"], "V2 Python executable")
    )
    if str(python_executable.parent).casefold() != str(environment_path).casefold():
        raise ValueError("V2 Python executable escaped its environment root")
    distributions = payload["installed_distributions"]
    if not isinstance(distributions, list):
        raise ValueError("V2 installed distributions must be a list")
    if _positive_integer(
        payload["installed_distribution_count"],
        "V2 installed distribution count",
    ) != len(distributions):
        raise ValueError("V2 installed distribution count mismatch")
    rows = [
        _validate_distribution_inventory_row(
            _mapping(value, f"V2 installed distribution {index}"),
            index=index,
            environment_root=environment_root,
        )
        for index, value in enumerate(distributions)
    ]
    normalized_names = [str(row["normalized_name"]) for row in rows]
    if normalized_names != sorted(normalized_names, key=lambda name: name.encode("utf-8")):
        raise ValueError("V2 installed distributions are not in canonical name order")
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("V2 installed distribution normalized names must be unique")
    dist_info_paths = [str(row["dist_info_relative_path"]).casefold() for row in rows]
    if len(dist_info_paths) != len(set(dist_info_paths)):
        raise ValueError("V2 installed dist-info paths must be case-insensitively unique")
    rows_by_path = {str(row["dist_info_relative_path"]).casefold(): row for row in rows}
    for index, value in enumerate(_runtime_distribution_list(payload["distributions"])):
        expected = _inventory_row_from_v1_distribution_pin(
            _mapping(value, f"V1 runtime distribution {index}"),
            site_packages_root=site_packages_root,
        )
        observed = rows_by_path.get(str(expected["dist_info_relative_path"]).casefold())
        if observed is None:
            raise ValueError("V2 installed distribution inventory does not contain an exact V1 pin")
        _validate_inventory_row_contains_exact_v1_pin(observed, expected)
    _validate_site_packages_coverage(
        _mapping(payload["site_packages_coverage"], "V2 site-packages coverage"),
        expected_root=site_packages_root,
        expected_distribution_count=len(distributions),
    )
    _validate_python_stdlib_zip(
        _mapping(payload["python_stdlib_zip"], "V2 Python stdlib zip"),
        expected_environment_root=environment_root,
        python_identity=python_identity,
    )
    _validate_python_home_top_level_tree(
        _mapping(
            payload["python_home_top_level_tree"],
            "V2 Python-home top-level tree",
        ),
        expected_root=environment_root,
    )
    _validate_runtime_raw_tree(
        _mapping(payload["stdlib_lib_tree"], "V2 stdlib Lib tree"),
        expected_role=_STDLIB_TREE_ROLE,
        expected_root=str(environment_path / "Lib"),
        expected_excluded_top_level_directories=("site-packages",),
    )
    _validate_runtime_raw_tree(
        _mapping(payload["dlls_tree"], "V2 DLLs tree"),
        expected_role=_DLLS_TREE_ROLE,
        expected_root=str(environment_path / "DLLs"),
        expected_excluded_top_level_directories=(),
    )
    _validate_runtime_raw_tree(
        _mapping(payload["python_library_bin_tree"], "V2 Python Library bin tree"),
        expected_role=_LIBRARY_BIN_TREE_ROLE,
        expected_root=str(environment_path / "Library" / "bin"),
        expected_excluded_top_level_directories=(),
    )
    _validate_artifact_id(payload, "V2 import runtime identity")


def _v1_runtime_identity_from_v2(payload: Mapping[str, object]) -> Mapping[str, object]:
    runtime = _mapping(payload, "V2 import runtime identity")
    return {
        "schema_version": v1.RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "platform": runtime["platform"],
        "gpu": runtime["gpu"],
        "python": runtime["python"],
        "distributions": runtime["distributions"],
        "artifact_id": _digest(
            runtime["v1_runtime_identity_artifact_id"],
            "V1 runtime identity artifact id",
        ),
    }


def _validate_distribution_inventory_row(
    row: Mapping[str, object],
    *,
    index: int,
    environment_root: str,
) -> Mapping[str, object]:
    _exact_keys(
        row,
        {
            "normalized_name",
            "distribution_name",
            "version",
            "dist_info_relative_path",
            "record_relative_path",
            "record_raw_sha256",
            "record_raw_bytes",
            "record_entry_count",
            "record_hashed_entry_count",
            "record_unhashed_pyc_entry_count",
            "record_hashed_site_packages_entry_count",
            "record_hashed_environment_external_entry_count",
            "record_hashed_absolute_environment_entry_count",
            "record_hashed_absolute_environment_entries",
            "record_hashed_absent_outside_environment_entry_count",
            "record_absent_outside_environment_entries",
            "record_hashed_pinned_identity_mismatch_entry_count",
            "record_pinned_identity_mismatch_entries",
            "record_in_environment_hashed_entries_verified_or_explicitly_pinned",
            "record_absent_outside_environment_entries_attested",
        },
        f"V2 installed distribution {index}",
    )
    distribution_name = _text(row["distribution_name"], "installed distribution name")
    if row["normalized_name"] != _normalized_distribution_name(distribution_name):
        raise ValueError("installed distribution normalized name drifted")
    _text(row["version"], "installed distribution version")
    dist_info_path = _relative_posix_runtime_path(
        row["dist_info_relative_path"],
        "installed dist-info relative path",
    )
    if not dist_info_path.casefold().endswith(".dist-info") or "/" in dist_info_path:
        raise ValueError("installed dist-info must be a direct site-packages child")
    record_path = _relative_posix_runtime_path(
        row["record_relative_path"],
        "installed RECORD relative path",
    )
    if record_path != f"{dist_info_path}/RECORD":
        raise ValueError("installed RECORD path disagrees with its dist-info directory")
    _digest(row["record_raw_sha256"], "installed RECORD raw sha256")
    _positive_integer(row["record_raw_bytes"], "installed RECORD raw bytes")
    total = _positive_integer(row["record_entry_count"], "installed RECORD entry count")
    hashed = _positive_integer(row["record_hashed_entry_count"], "installed RECORD hashed entry count")
    pyc = _nonnegative_integer(
        row["record_unhashed_pyc_entry_count"],
        "installed RECORD unhashed pyc count",
    )
    if hashed + pyc + 1 != total:
        raise ValueError("installed RECORD counts do not close exactly")
    site_hashed = _nonnegative_integer(
        row["record_hashed_site_packages_entry_count"],
        "installed RECORD site-packages hashed count",
    )
    environment_external = _nonnegative_integer(
        row["record_hashed_environment_external_entry_count"],
        "installed RECORD environment-external hashed count",
    )
    absolute_environment_count = _nonnegative_integer(
        row["record_hashed_absolute_environment_entry_count"],
        "installed RECORD absolute-environment hashed count",
    )
    if absolute_environment_count > environment_external:
        raise ValueError("installed RECORD absolute entries exceed environment-external entries")
    absent_outside = _nonnegative_integer(
        row["record_hashed_absent_outside_environment_entry_count"],
        "installed RECORD absent-outside-environment hashed count",
    )
    pinned_mismatch_count = _nonnegative_integer(
        row["record_hashed_pinned_identity_mismatch_entry_count"],
        "installed RECORD pinned identity-mismatch count",
    )
    if site_hashed + environment_external + absent_outside + pinned_mismatch_count != hashed:
        raise ValueError("installed RECORD hashed-domain counts do not close exactly")
    raw_absolute_entries = row["record_hashed_absolute_environment_entries"]
    if not isinstance(raw_absolute_entries, list) or len(raw_absolute_entries) != absolute_environment_count:
        raise ValueError("installed RECORD absolute-environment list count mismatch")
    environment_path = pathlib.PureWindowsPath(environment_root)
    observed_absolute_paths: list[str] = []
    for absolute_index, raw_absolute in enumerate(raw_absolute_entries):
        absolute = _mapping(
            raw_absolute,
            f"installed RECORD absolute-environment entry {index}:{absolute_index}",
        )
        _exact_keys(
            absolute,
            {
                "record_path",
                "resolved_target",
                "declared_raw_sha256",
                "declared_raw_bytes",
                "observed_raw_sha256",
                "observed_raw_bytes",
                "target_within_permitted_environment_external_root",
            },
            f"installed RECORD absolute-environment entry {index}:{absolute_index}",
        )
        absolute_path = _text(
            absolute["record_path"],
            f"installed RECORD absolute path {index}:{absolute_index}",
        )
        pure_absolute = _canonical_absolute_windows_record_path(
            absolute_path,
            f"installed RECORD absolute path {index}:{absolute_index}",
        )
        target = _canonical_windows_path(
            absolute["resolved_target"],
            f"installed RECORD absolute target {index}:{absolute_index}",
        )
        if target.casefold() != str(pure_absolute).casefold():
            raise ValueError("installed RECORD absolute target disagrees with its record path")
        try:
            environment_relative = pure_absolute.relative_to(environment_path)
        except ValueError as exc:
            raise ValueError("installed RECORD absolute target escaped the Python environment") from exc
        _validate_record_environment_external_path(
            pathlib.Path(*environment_relative.parts),
            distribution_name=distribution_name,
        )
        declared_digest = _digest(
            absolute["declared_raw_sha256"],
            f"installed RECORD absolute declared digest {index}:{absolute_index}",
        )
        observed_digest = _digest(
            absolute["observed_raw_sha256"],
            f"installed RECORD absolute observed digest {index}:{absolute_index}",
        )
        declared_size = _nonnegative_integer(
            absolute["declared_raw_bytes"],
            f"installed RECORD absolute declared bytes {index}:{absolute_index}",
        )
        observed_size = _nonnegative_integer(
            absolute["observed_raw_bytes"],
            f"installed RECORD absolute observed bytes {index}:{absolute_index}",
        )
        if (
            declared_digest != observed_digest
            or declared_size != observed_size
            or absolute["target_within_permitted_environment_external_root"] is not True
        ):
            raise ValueError("installed RECORD absolute-environment identity did not verify exactly")
        observed_absolute_paths.append(absolute_path)
    if observed_absolute_paths != sorted(
        observed_absolute_paths,
        key=lambda value: value.encode("utf-8"),
    ) or len(observed_absolute_paths) != len(set(observed_absolute_paths)):
        raise ValueError("installed RECORD absolute-environment entries are not canonical unique order")
    exclusions = row["record_absent_outside_environment_entries"]
    if not isinstance(exclusions, list) or len(exclusions) != absent_outside:
        raise ValueError("installed RECORD absent-outside-environment list count mismatch")
    expected_parent = pathlib.PureWindowsPath(environment_root).parent / "bin"
    observed_exclusion_paths: list[str] = []
    for exclusion_index, raw_exclusion in enumerate(exclusions):
        exclusion = _mapping(
            raw_exclusion,
            f"installed RECORD absent-outside exclusion {index}:{exclusion_index}",
        )
        _exact_keys(
            exclusion,
            {
                "record_path",
                "resolved_target",
                "declared_raw_sha256",
                "declared_raw_bytes",
                "target_absent",
                "child_import_or_controlled_path_reachable",
                "exclusion_reason",
            },
            f"installed RECORD absent-outside exclusion {index}:{exclusion_index}",
        )
        record_path_value = v1._record_relative_posix_path(
            exclusion["record_path"],
            f"installed RECORD absent-outside path {index}:{exclusion_index}",
        )
        match = re.fullmatch(r"\.\./\.\./\.\./bin/(pip|pip3|pip3\.[0-9]+)", record_path_value)
        if row["normalized_name"] != "pip" or match is None:
            raise ValueError("only exact pip console-script paths may be absent outside the environment")
        target = _canonical_windows_path(
            exclusion["resolved_target"],
            f"installed RECORD absent-outside target {index}:{exclusion_index}",
        )
        if target.casefold() != str(expected_parent / match.group(1)).casefold():
            raise ValueError("installed RECORD absent-outside target is not canonical exact")
        _digest(
            exclusion["declared_raw_sha256"],
            f"installed RECORD absent-outside digest {index}:{exclusion_index}",
        )
        _nonnegative_integer(
            exclusion["declared_raw_bytes"],
            f"installed RECORD absent-outside bytes {index}:{exclusion_index}",
        )
        if (
            exclusion["target_absent"] is not True
            or exclusion["child_import_or_controlled_path_reachable"] is not False
            or exclusion["exclusion_reason"]
            != "absent_pip_console_script_declared_outside_frozen_python_environment.v1"
        ):
            raise ValueError("installed RECORD absent-outside exclusion contract drifted")
        observed_exclusion_paths.append(record_path_value)
    if observed_exclusion_paths != sorted(
        observed_exclusion_paths,
        key=lambda value: value.encode("utf-8"),
    ) or len(observed_exclusion_paths) != len(set(observed_exclusion_paths)):
        raise ValueError("installed RECORD absent-outside exclusions are not canonical unique order")
    raw_mismatches = row["record_pinned_identity_mismatch_entries"]
    if not isinstance(raw_mismatches, list) or len(raw_mismatches) != pinned_mismatch_count:
        raise ValueError("installed RECORD pinned identity-mismatch list count mismatch")
    observed_mismatch_paths: list[str] = []
    for mismatch_index, raw_mismatch in enumerate(raw_mismatches):
        mismatch = _mapping(
            raw_mismatch,
            f"installed RECORD pinned mismatch {index}:{mismatch_index}",
        )
        _exact_keys(
            mismatch,
            {
                "record_path",
                "resolved_target",
                "declared_raw_sha256",
                "declared_raw_bytes",
                "observed_raw_sha256",
                "observed_raw_bytes",
                "target_raw_pinned",
                "child_import_or_controlled_path_reachable",
                "exclusion_reason",
            },
            f"installed RECORD pinned mismatch {index}:{mismatch_index}",
        )
        target = _canonical_windows_path(
            mismatch["resolved_target"],
            f"installed RECORD pinned mismatch target {index}:{mismatch_index}",
        )
        raw_mismatch_path = _text(
            mismatch["record_path"],
            f"installed RECORD pinned mismatch path {index}:{mismatch_index}",
        )
        if row["normalized_name"] == "pip":
            mismatch_path = _relative_posix_runtime_path(
                raw_mismatch_path,
                f"installed RECORD pinned mismatch path {index}:{mismatch_index}",
            )
            expected_mismatch_path = f"{dist_info_path}/INSTALLER"
            if mismatch_path.casefold() != expected_mismatch_path.casefold():
                raise ValueError("only the exact pip INSTALLER metadata mismatch may be pinned")
            expected_target = (
                pathlib.PureWindowsPath(environment_root)
                / "Lib"
                / "site-packages"
                / pathlib.PureWindowsPath(*pathlib.PurePosixPath(mismatch_path).parts)
            )
            expected_reason = "conda_rewritten_pip_installer_metadata_not_importable.v1"
        elif row["normalized_name"] == "wheel":
            record_windows_path = pathlib.PureWindowsPath(raw_mismatch_path)
            expected_target = pathlib.PureWindowsPath(environment_root) / "Scripts" / "wheel.exe"
            expected_mismatch_path = expected_target.as_posix()
            if (
                not record_windows_path.is_absolute()
                or record_windows_path.as_posix() != raw_mismatch_path
                or raw_mismatch_path != expected_mismatch_path
            ):
                raise ValueError("only the exact wheel console-script mismatch may be pinned")
            mismatch_path = raw_mismatch_path
            expected_reason = "raw_pinned_wheel_console_script_not_in_controlled_path.v1"
        else:
            raise ValueError("installed RECORD identity mismatch is not an admitted inert artifact")
        if target.casefold() != str(expected_target).casefold():
            raise ValueError("installed RECORD pinned mismatch target is not canonical exact")
        declared_digest = _digest(
            mismatch["declared_raw_sha256"],
            f"installed RECORD pinned mismatch declared digest {index}:{mismatch_index}",
        )
        declared_bytes = _nonnegative_integer(
            mismatch["declared_raw_bytes"],
            f"installed RECORD pinned mismatch declared bytes {index}:{mismatch_index}",
        )
        observed_digest = _digest(
            mismatch["observed_raw_sha256"],
            f"installed RECORD pinned mismatch observed digest {index}:{mismatch_index}",
        )
        observed_bytes = _nonnegative_integer(
            mismatch["observed_raw_bytes"],
            f"installed RECORD pinned mismatch observed bytes {index}:{mismatch_index}",
        )
        if declared_digest == observed_digest and declared_bytes == observed_bytes:
            raise ValueError("installed RECORD pinned mismatch must preserve the observed inequality")
        if row["normalized_name"] == "pip" and (
            observed_digest != hashlib.sha256(b"conda").hexdigest() or observed_bytes != len(b"conda")
        ):
            raise ValueError("pip INSTALLER observed raw pin is not the exact conda marker")
        if (
            mismatch["target_raw_pinned"] is not True
            or mismatch["child_import_or_controlled_path_reachable"] is not False
            or mismatch["exclusion_reason"] != expected_reason
        ):
            raise ValueError("installed RECORD pinned identity-mismatch contract drifted")
        observed_mismatch_paths.append(mismatch_path)
    if observed_mismatch_paths != sorted(
        observed_mismatch_paths,
        key=lambda value: value.encode("utf-8"),
    ) or len(observed_mismatch_paths) != len(set(observed_mismatch_paths)):
        raise ValueError("installed RECORD pinned mismatches are not canonical unique order")
    if (
        row["record_in_environment_hashed_entries_verified_or_explicitly_pinned"] is not True
        or row["record_absent_outside_environment_entries_attested"] is not True
    ):
        raise ValueError("installed distribution RECORD closure must be explicitly attested")
    return row


def _validate_site_packages_coverage(
    payload: Mapping[str, object],
    *,
    expected_root: str,
    expected_distribution_count: int,
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "root",
            "coverage_contract",
            "link_contract",
            "cache_directory_names",
            "excluded_cache_bytecode_suffixes",
            "excluded_site_startup_suffixes",
            "record_ownership_join_sha256",
            "record_hashed_site_path_count",
            "record_ownership_overlap_count",
            "total_regular_file_count",
            "owned_hashed_record_file_count",
            "record_self_file_count",
            "excluded_bytecode_file_count",
            "excluded_site_startup_file_count",
            "unowned_regular_files",
            "unowned_regular_file_count",
            "symlink_or_reparse_entry_count",
            "non_single_link_regular_file_count",
            "all_nonexcluded_regular_files_record_owned_or_raw_pinned",
            "artifact_id",
        },
        "V2 site-packages coverage",
    )
    required = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_SITE_PACKAGES_COVERAGE_V2_SCHEMA_VERSION,
        "root": _canonical_windows_path(expected_root, "expected site-packages coverage root"),
        "coverage_contract": ("every_nonstartup_noncache_bytecode_file_is_record_owned_or_raw_pinned.v1"),
        "link_contract": "regular_non_symlink_non_reparse_single_link_files_only.v1",
        "cache_directory_names": list(_RUNTIME_TREE_CACHE_DIRECTORY_NAMES),
        "excluded_cache_bytecode_suffixes": list(_RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES),
        "excluded_site_startup_suffixes": list(_SITE_PACKAGES_EXCLUDED_SITE_STARTUP_SUFFIXES),
    }
    if any(not _json_type_exact_equal(payload[field], value) for field, value in required.items()):
        raise ValueError("V2 site-packages coverage contract drifted")
    _digest(payload["record_ownership_join_sha256"], "site-packages RECORD ownership join")
    hashed_site_paths = _positive_integer(
        payload["record_hashed_site_path_count"],
        "site-packages hashed RECORD site path count",
    )
    overlap_count = _nonnegative_integer(
        payload["record_ownership_overlap_count"],
        "site-packages RECORD ownership overlap count",
    )
    total = _positive_integer(
        payload["total_regular_file_count"],
        "site-packages total regular file count",
    )
    owned = _nonnegative_integer(
        payload["owned_hashed_record_file_count"],
        "site-packages owned hashed RECORD file count",
    )
    record_self = _positive_integer(
        payload["record_self_file_count"],
        "site-packages RECORD self file count",
    )
    excluded_bytecode = _nonnegative_integer(
        payload["excluded_bytecode_file_count"],
        "site-packages excluded bytecode file count",
    )
    excluded_startup = _nonnegative_integer(
        payload["excluded_site_startup_file_count"],
        "site-packages excluded site-startup file count",
    )
    if record_self != expected_distribution_count:
        raise ValueError("site-packages RECORD self count differs from installed distributions")
    if owned > hashed_site_paths or overlap_count > hashed_site_paths:
        raise ValueError("site-packages RECORD ownership counts are inconsistent")
    unowned_files = payload["unowned_regular_files"]
    if not isinstance(unowned_files, list):
        raise ValueError("site-packages unowned regular files must be a list")
    unowned_paths: list[str] = []
    for index, value in enumerate(unowned_files):
        row = _mapping(value, f"site-packages unowned regular file {index}")
        _exact_keys(
            row,
            {"path", "raw_sha256", "raw_bytes"},
            f"site-packages unowned regular file {index}",
        )
        path = _relative_posix_runtime_path(
            row["path"],
            f"site-packages unowned regular file path {index}",
        )
        if path.casefold().endswith(
            tuple(
                value.casefold()
                for value in (
                    *_RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES,
                    *_SITE_PACKAGES_EXCLUDED_SITE_STARTUP_SUFFIXES,
                )
            )
        ):
            raise ValueError("site-packages unowned raw pin uses an excluded path")
        _digest(row["raw_sha256"], f"site-packages unowned raw sha256 {index}")
        _nonnegative_integer(row["raw_bytes"], f"site-packages unowned raw bytes {index}")
        unowned_paths.append(path)
    if unowned_paths != sorted(unowned_paths, key=lambda path: path.encode("utf-8")):
        raise ValueError("site-packages unowned raw pins are not in canonical path order")
    if len(unowned_paths) != len({path.casefold() for path in unowned_paths}):
        raise ValueError("site-packages unowned raw pins contain duplicate paths")
    unowned_count = _nonnegative_integer(
        payload["unowned_regular_file_count"],
        "site-packages unowned regular file count",
    )
    if unowned_count != len(unowned_files):
        raise ValueError("site-packages unowned regular file count mismatch")
    if total != owned + record_self + excluded_bytecode + excluded_startup + unowned_count:
        raise ValueError("site-packages coverage counts do not close exactly")
    if (
        _nonnegative_integer(
            payload["symlink_or_reparse_entry_count"],
            "site-packages symlink or reparse entry count",
        )
        != 0
        or _nonnegative_integer(
            payload["non_single_link_regular_file_count"],
            "site-packages non-single-link regular file count",
        )
        != 0
    ):
        raise ValueError("site-packages coverage must reject links instead of recording them")
    if payload["all_nonexcluded_regular_files_record_owned_or_raw_pinned"] is not True:
        raise ValueError("site-packages coverage closure must be explicitly true")
    _validate_artifact_id(payload, "V2 site-packages coverage")


def _validate_python_stdlib_zip(
    payload: Mapping[str, object],
    *,
    expected_environment_root: str,
    python_identity: Mapping[str, object],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "path",
            "exists",
            "raw_sha256",
            "raw_bytes",
            "precedence_contract",
            "link_contract",
            "artifact_id",
        },
        "V2 Python stdlib zip",
    )
    root = pathlib.PureWindowsPath(
        _canonical_windows_path(expected_environment_root, "expected Python environment root")
    )
    major, minor = _python_major_minor(python_identity.get("version"))
    required = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_PYTHON_STDLIB_ZIP_V2_SCHEMA_VERSION,
        "path": str(root / f"python{major}{minor}.zip"),
        "precedence_contract": "python_home_zip_precedes_dlls_and_lib_on_minus_P_minus_S_sys_path.v1",
        "link_contract": "absent_or_regular_non_symlink_non_reparse_single_link_file.v1",
    }
    if any(not _json_type_exact_equal(payload[field], value) for field, value in required.items()):
        raise ValueError("V2 Python stdlib zip contract drifted")
    exists = payload["exists"]
    if not isinstance(exists, bool):
        raise ValueError("V2 Python stdlib zip exists must be a boolean")
    if exists:
        _digest(payload["raw_sha256"], "V2 Python stdlib zip raw sha256")
        _nonnegative_integer(payload["raw_bytes"], "V2 Python stdlib zip raw bytes")
    elif payload["raw_sha256"] is not None or payload["raw_bytes"] is not None:
        raise ValueError("absent V2 Python stdlib zip must not publish raw identity")
    _validate_artifact_id(payload, "V2 Python stdlib zip")


def _validate_runtime_raw_tree(
    tree: Mapping[str, object],
    *,
    expected_role: str,
    expected_root: str,
    expected_excluded_top_level_directories: tuple[str, ...],
) -> None:
    _exact_keys(
        tree,
        {
            "schema_version",
            "tree_role",
            "root",
            "content_contract",
            "link_contract",
            "excluded_top_level_directories",
            "cache_directory_names",
            "excluded_cache_bytecode_suffixes",
            "excluded_cache_bytecode_file_count",
            "entries",
            "entry_count",
            "total_raw_bytes",
            "artifact_id",
        },
        f"V2 {expected_role} tree",
    )
    required = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_RUNTIME_RAW_TREE_V2_SCHEMA_VERSION,
        "tree_role": expected_role,
        "root": _canonical_windows_path(expected_root, f"V2 {expected_role} expected root"),
        "content_contract": "exact_raw_bytes_no_eol_normalization.v1",
        "link_contract": "regular_non_symlink_non_reparse_hardlinks_allowed.v1",
        "excluded_top_level_directories": list(expected_excluded_top_level_directories),
        "cache_directory_names": list(_RUNTIME_TREE_CACHE_DIRECTORY_NAMES),
        "excluded_cache_bytecode_suffixes": list(_RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES),
    }
    if any(not _json_type_exact_equal(tree[field], value) for field, value in required.items()):
        raise ValueError(f"V2 {expected_role} tree contract drifted")
    entries = tree["entries"]
    if not isinstance(entries, list):
        raise ValueError(f"V2 {expected_role} entries must be a list")
    paths: list[str] = []
    total_raw_bytes = 0
    for index, value in enumerate(entries):
        row = _mapping(value, f"V2 {expected_role} entry {index}")
        _exact_keys(row, {"path", "raw_sha256", "raw_bytes"}, f"V2 {expected_role} entry {index}")
        path = _relative_posix_runtime_path(row["path"], f"V2 {expected_role} path {index}")
        path_parts = pathlib.PurePosixPath(path).parts
        if path.casefold().endswith(tuple(value.casefold() for value in _RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES)):
            raise ValueError(f"V2 {expected_role} contains an excluded bytecode file")
        if path_parts and path_parts[0].casefold() in {
            value.casefold() for value in expected_excluded_top_level_directories
        }:
            raise ValueError(f"V2 {expected_role} contains an excluded top-level directory")
        paths.append(path)
        _digest(row["raw_sha256"], f"V2 {expected_role} raw sha256 {index}")
        total_raw_bytes += _nonnegative_integer(
            row["raw_bytes"],
            f"V2 {expected_role} raw bytes {index}",
        )
    if paths != sorted(paths, key=lambda path: path.encode("utf-8")):
        raise ValueError(f"V2 {expected_role} entries are not in canonical path order")
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise ValueError(f"V2 {expected_role} entries contain duplicate paths")
    _nonnegative_integer(
        tree["excluded_cache_bytecode_file_count"],
        f"V2 {expected_role} excluded cache bytecode file count",
    )
    if _positive_integer(tree["entry_count"], f"V2 {expected_role} entry count") != len(entries):
        raise ValueError(f"V2 {expected_role} entry count mismatch")
    if (
        _nonnegative_integer(
            tree["total_raw_bytes"],
            f"V2 {expected_role} total raw bytes",
        )
        != total_raw_bytes
    ):
        raise ValueError(f"V2 {expected_role} total raw bytes mismatch")
    _validate_artifact_id(tree, f"V2 {expected_role} tree")


def _runtime_distribution_list(value: object) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError("runtime distributions must be a non-empty list")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings")
    return value


def _string_set(value: object, field_name: str) -> set[str]:
    if not isinstance(value, set) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a set of strings")
    return value


def _relative_posix_runtime_path(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    path = pathlib.PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be a canonical relative POSIX path")
    return text


def _external_anchor_contract_v2(receipt_path: str) -> dict[str, object]:
    return {
        "required": True,
        "receipt_schema_version": (RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_V2_SCHEMA_VERSION),
        "receipt_path": receipt_path,
        "gist_owner": _GIST_OWNER,
        "filename": V2_GIST_FILENAME,
        "visibility": "public",
        "new_gist_required": True,
        "retired_v1_gist_accepted": False,
        "base_gist_get_required": True,
        "commits_get_required": True,
        "commits_get_count": 2,
        "commits_query": "per_page=2&page=1",
        "final_commits_reobservation_after_raw_gets_required": True,
        "observation_linearization_point": "final_commits_reobservation",
        "sole_revision_get_required": True,
        "base_api_raw_get_required": True,
        "canonical_revision_raw_get_required": True,
        "github_api_version": GITHUB_API_VERSION,
        "authorization_or_cookie_header_allowed": False,
        "single_commit_required": True,
        "base_or_revision_history_field_required": False,
        "single_exact_file_required": True,
        "created_equals_updated_required": False,
        "timestamp_fields_format_validated_only": True,
        "timestamp_order_used_as_revision_authority": False,
        "observed_at_caller_supplied_allowed": False,
        "observed_at_recorded_after_final_commits_get": True,
        "sole_commit_version_bound_to_revision_endpoint_required": True,
        "sole_commit_timestamp_source": "dedicated_commits_endpoint",
        "api_file_raw_url_revision_used_as_commit_authority": False,
        "both_raw_responses_exact_protocol_required": True,
        "expected_receipt_artifact_id_must_be_external": True,
        "protocol_alone_authorizes_execution": False,
    }


def _validate_cross_endpoint_gist_observation(
    *,
    base: Mapping[str, object],
    commit: Mapping[str, object],
    revision: Mapping[str, object],
    gist_id: str,
    execution_protocol_raw: bytes,
) -> Mapping[str, object]:
    base_url = f"https://api.github.com/gists/{gist_id}"
    gist_url = f"https://gist.github.com/{_GIST_OWNER}/{gist_id}"
    if base.get("id") != gist_id or revision.get("id") != gist_id:
        raise ValueError("V2 Gist id differs across base and revision endpoints")
    if base.get("public") is not True or revision.get("public") is not True:
        raise ValueError("V2 current and revision Gist must both be public")
    if base.get("url") != base_url:
        raise ValueError("V2 base Gist API URL is not canonical")
    if base.get("commits_url") != f"{base_url}/commits":
        raise ValueError("V2 base Gist commits URL is not canonical")
    if base.get("html_url") != gist_url or revision.get("html_url") != gist_url:
        raise ValueError("V2 Gist HTML URL is not canonical")
    if base.get("truncated") is not False or revision.get("truncated") is not False:
        raise ValueError("V2 base and revision Gist responses must not be truncated")
    _validate_owner(base, "base Gist")
    _validate_owner(revision, "revision Gist")

    version = _hex_text(commit.get("version"), "sole commit version", lengths={40})
    committed_at = _github_utc_timestamp(
        commit.get("committed_at"),
        "sole commit committed_at",
    )
    revision_url = f"{base_url}/{version}"
    if commit.get("url") != revision_url or revision.get("url") != revision_url:
        raise ValueError("V2 commit/revision API URL lineage drifted")
    created_at = _github_utc_timestamp(base.get("created_at"), "base created_at")
    updated_at = _github_utc_timestamp(base.get("updated_at"), "base updated_at")
    revision_created_at = _github_utc_timestamp(
        revision.get("created_at"),
        "revision created_at",
    )
    revision_updated_at = _github_utc_timestamp(
        revision.get("updated_at"),
        "revision updated_at",
    )
    base_file = _single_gist_file(base, "base Gist", execution_protocol_raw)
    revision_file = _single_gist_file(
        revision,
        "revision Gist",
        execution_protocol_raw,
    )
    base_raw_url = _canonical_api_file_raw_url(
        base_file.get("raw_url"),
        gist_id=gist_id,
        field_name="base API raw URL",
    )
    revision_file_raw_url = _canonical_api_file_raw_url(
        revision_file.get("raw_url"),
        gist_id=gist_id,
        field_name="revision API file raw URL",
    )
    canonical_revision_raw_url = _canonical_revision_raw_url(
        gist_id=gist_id,
        version=version,
    )
    return {
        "gist_url": gist_url,
        "created_at": created_at,
        "updated_at": updated_at,
        "revision_created_at": revision_created_at,
        "revision_updated_at": revision_updated_at,
        "sole_version": version,
        "sole_committed_at": committed_at,
        "base_gist_api_url": base_url,
        "commits_api_url": f"{base_url}/commits?per_page=2&page=1",
        "revision_api_url": revision_url,
        "base_api_raw_url": base_raw_url,
        "revision_api_file_raw_url": revision_file_raw_url,
        "canonical_revision_raw_url": canonical_revision_raw_url,
    }


def _validate_final_commits_reobservation(
    commit: Mapping[str, object],
    *,
    metadata: Mapping[str, object],
) -> None:
    version = _hex_text(commit.get("version"), "final sole commit version", lengths={40})
    committed_at = _github_utc_timestamp(
        commit.get("committed_at"),
        "final sole commit committed_at",
    )
    commit_url = _text(commit.get("url"), "final sole commit URL")
    if (
        version != metadata["sole_version"]
        or committed_at != metadata["sole_committed_at"]
        or commit_url != metadata["revision_api_url"]
    ):
        raise ValueError("V2 final commits reobservation changed sole-commit identity")


def _single_gist_file(
    payload: Mapping[str, object],
    field_name: str,
    execution_protocol_raw: bytes,
) -> Mapping[str, object]:
    files = _mapping(payload.get("files"), f"{field_name} files")
    if set(files) != {V2_GIST_FILENAME}:
        raise ValueError(f"{field_name} must contain the sole exact V2 filename")
    row = _mapping(files[V2_GIST_FILENAME], f"{field_name} file")
    if row.get("filename") != V2_GIST_FILENAME:
        raise ValueError(f"{field_name} file filename drifted")
    if row.get("truncated") is not False:
        raise ValueError(f"{field_name} file must not be truncated")
    if _positive_integer(row.get("size"), f"{field_name} file size") != len(execution_protocol_raw):
        raise ValueError(f"{field_name} file size differs from protocol raw bytes")
    content = row.get("content")
    if not isinstance(content, str) or content.encode("utf-8") != execution_protocol_raw:
        raise ValueError(f"{field_name} embedded content differs from protocol raw bytes")
    return row


def _validate_owner(payload: Mapping[str, object], field_name: str) -> None:
    owner = _mapping(payload.get("owner"), f"{field_name} owner")
    if owner.get("login") != _GIST_OWNER:
        raise ValueError(f"{field_name} owner differs from frozen owner")


def _prevalidate_api_raw_url(
    *,
    base: Mapping[str, object],
    gist_id: str,
) -> str:
    if base.get("id") != gist_id:
        raise ValueError("base Gist id differs from requested id")
    _validate_owner(base, "base Gist")
    files = _mapping(base.get("files"), "base Gist files")
    if set(files) != {V2_GIST_FILENAME}:
        raise ValueError("base Gist must contain the sole exact V2 filename")
    row = _mapping(files[V2_GIST_FILENAME], "base Gist file")
    return _canonical_api_file_raw_url(
        row.get("raw_url"),
        gist_id=gist_id,
        field_name="base API raw URL",
    )


def _canonical_api_file_raw_url(
    value: object,
    *,
    gist_id: str,
    field_name: str,
) -> str:
    text = _text(value, field_name)
    prefix = f"https://gist.githubusercontent.com/{_GIST_OWNER}/{gist_id}/raw/"
    suffix = f"/{V2_GIST_FILENAME}"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError(f"{field_name} is not canonical")
    middle = text[len(prefix) : -len(suffix)]
    _hex_text(middle, f"{field_name} content revision", lengths={40})
    if text != f"{prefix}{middle}{suffix}":
        raise ValueError(f"{field_name} contains noncanonical URL syntax")
    return text


def _canonical_revision_raw_url(*, gist_id: str, version: str) -> str:
    return f"https://gist.githubusercontent.com/{_GIST_OWNER}/{gist_id}/raw/{version}/{V2_GIST_FILENAME}"


def _validate_anchor_receipt_shape(receipt: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "execution_protocol_id",
        "protocol_raw_sha256",
        "protocol_raw_bytes",
        "gist_owner",
        "gist_id",
        "gist_url",
        "filename",
        "public",
        "created_at",
        "updated_at",
        "revision_created_at",
        "revision_updated_at",
        "created_equals_updated_required",
        "timestamp_fields_format_validated_only",
        "timestamp_order_used_as_revision_authority",
        "observed_at_caller_supplied",
        "observed_at_recorded_after_final_commits_get",
        "sole_version",
        "sole_committed_at",
        "base_gist_api_url",
        "commits_api_url",
        "revision_api_url",
        "base_api_raw_url",
        "revision_api_file_raw_url",
        "canonical_revision_raw_url",
        "base_gist_response_raw_sha256",
        "base_gist_response_raw_bytes",
        "commits_response_raw_sha256",
        "commits_response_raw_bytes",
        "commits_reobservation_response_raw_sha256",
        "commits_reobservation_response_raw_bytes",
        "revision_response_raw_sha256",
        "revision_response_raw_bytes",
        "base_api_protocol_raw_sha256",
        "base_api_protocol_raw_bytes",
        "revision_protocol_raw_sha256",
        "revision_protocol_raw_bytes",
        "commits_page_item_count",
        "final_commits_page_item_count",
        "commits_get_count",
        "commits_reobserved_after_raw_gets",
        "initial_and_final_sole_commit_identity_match",
        "observation_linearization_point",
        "sole_revision_endpoint_bound",
        "base_history_used_as_revision_authority",
        "api_file_raw_url_revision_used_as_commit_authority",
        "current_file_count",
        "revision_file_count",
        "current_and_revision_public",
        "sole_commit_version_bound_to_revision_endpoint",
        "sole_commit_timestamp_source",
        "base_and_revision_raw_exact_protocol_match",
        "github_api_version",
        "request_headers",
        "authorization_or_cookie_header_used",
        "observation_transport",
        "observed_at_utc",
        "execution_root",
        "execution_root_existed_at_observation",
        "model_output_count_at_observation",
        "qualification_report_existed_at_observation",
        "retired_predecessor_execution_protocol_id",
        "retired_predecessor_authorized",
        "artifact_id",
    }
    _exact_keys(receipt, expected_keys, "V2 public anchor receipt")
    if receipt["schema_version"] != RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_V2_SCHEMA_VERSION:
        raise ValueError("V2 public anchor receipt schema drifted")
    _validate_artifact_id(receipt, "V2 public anchor receipt")
    for name in (
        "execution_protocol_id",
        "protocol_raw_sha256",
        "base_gist_response_raw_sha256",
        "commits_response_raw_sha256",
        "commits_reobservation_response_raw_sha256",
        "revision_response_raw_sha256",
        "base_api_protocol_raw_sha256",
        "revision_protocol_raw_sha256",
    ):
        _digest(receipt[name], name)
    for name in (
        "protocol_raw_bytes",
        "base_gist_response_raw_bytes",
        "commits_response_raw_bytes",
        "commits_reobservation_response_raw_bytes",
        "revision_response_raw_bytes",
        "base_api_protocol_raw_bytes",
        "revision_protocol_raw_bytes",
    ):
        _positive_integer(receipt[name], name)
    if receipt["gist_owner"] != _GIST_OWNER or receipt["filename"] != V2_GIST_FILENAME:
        raise ValueError("V2 receipt owner or filename drifted")
    gist_id = _hex_text(receipt["gist_id"], "gist id", lengths={32})
    version = _hex_text(receipt["sole_version"], "sole version", lengths={40})
    if receipt["gist_url"] != f"https://gist.github.com/{_GIST_OWNER}/{gist_id}":
        raise ValueError("V2 receipt Gist URL is not canonical")
    base_url = f"https://api.github.com/gists/{gist_id}"
    revision_url = f"{base_url}/{version}"
    if (
        receipt["base_gist_api_url"] != base_url
        or receipt["commits_api_url"] != f"{base_url}/commits?per_page=2&page=1"
        or receipt["revision_api_url"] != revision_url
        or receipt["canonical_revision_raw_url"] != _canonical_revision_raw_url(gist_id=gist_id, version=version)
    ):
        raise ValueError("V2 receipt endpoint URL lineage drifted")
    _canonical_api_file_raw_url(
        receipt["base_api_raw_url"],
        gist_id=gist_id,
        field_name="receipt base API raw URL",
    )
    _canonical_api_file_raw_url(
        receipt["revision_api_file_raw_url"],
        gist_id=gist_id,
        field_name="receipt revision API file raw URL",
    )
    _github_utc_timestamp(receipt["created_at"], "receipt created_at")
    _github_utc_timestamp(receipt["updated_at"], "receipt updated_at")
    _github_utc_timestamp(
        receipt["revision_created_at"],
        "receipt revision_created_at",
    )
    _github_utc_timestamp(
        receipt["revision_updated_at"],
        "receipt revision_updated_at",
    )
    _github_utc_timestamp(
        receipt["sole_committed_at"],
        "receipt sole_committed_at",
    )
    _github_utc_timestamp(receipt["observed_at_utc"], "receipt observed_at")
    required_booleans = {
        "public": True,
        "created_equals_updated_required": False,
        "timestamp_fields_format_validated_only": True,
        "timestamp_order_used_as_revision_authority": False,
        "observed_at_caller_supplied": False,
        "observed_at_recorded_after_final_commits_get": True,
        "sole_revision_endpoint_bound": True,
        "base_history_used_as_revision_authority": False,
        "api_file_raw_url_revision_used_as_commit_authority": False,
        "commits_reobserved_after_raw_gets": True,
        "initial_and_final_sole_commit_identity_match": True,
        "current_and_revision_public": True,
        "sole_commit_version_bound_to_revision_endpoint": True,
        "base_and_revision_raw_exact_protocol_match": True,
        "authorization_or_cookie_header_used": False,
        "execution_root_existed_at_observation": False,
        "qualification_report_existed_at_observation": False,
        "retired_predecessor_authorized": False,
    }
    for key, expected in required_booleans.items():
        if receipt[key] is not expected:
            raise ValueError(f"V2 receipt boolean boundary drifted: {key}")
    for key in (
        "commits_page_item_count",
        "final_commits_page_item_count",
        "current_file_count",
        "revision_file_count",
    ):
        if _positive_integer(receipt[key], f"receipt {key}") != 1:
            raise ValueError(f"V2 receipt count boundary drifted: {key}")
    if _positive_integer(receipt["commits_get_count"], "receipt commits_get_count") != 2:
        raise ValueError("V2 receipt commits GET count drifted")
    if (
        _nonnegative_integer(
            receipt["model_output_count_at_observation"],
            "receipt model_output_count_at_observation",
        )
        != 0
    ):
        raise ValueError("V2 receipt model output count drifted")
    required_text = {
        "sole_commit_timestamp_source": "dedicated_commits_endpoint",
        "github_api_version": GITHUB_API_VERSION,
        "observation_transport": _OBSERVATION_TRANSPORT,
        "observation_linearization_point": "final_commits_reobservation",
        "retired_predecessor_execution_protocol_id": _RETIRED_V1_PREDECESSOR["execution_protocol_id"],
    }
    if any(receipt[key] != value for key, value in required_text.items()):
        raise ValueError("V2 receipt textual honesty boundary drifted")
    if receipt["request_headers"] != {
        "Accept": _GITHUB_ACCEPT,
        "User-Agent": _GITHUB_USER_AGENT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }:
        raise ValueError("V2 receipt request-header contract drifted")
    canonical_relationship_condition_reader_qualification_execution_root_v2(
        receipt["execution_root"],
        "receipt execution root",
    )


def _unauthenticated_github_https_get(
    *,
    url: str,
    max_bytes: int,
    timeout_seconds: int,
) -> bytes:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("GitHub response max_bytes must be positive")
    if not (
        url.startswith("https://api.github.com/gists/")
        or url.startswith(f"https://gist.githubusercontent.com/{_GIST_OWNER}/")
    ):
        raise ValueError("V2 observer refused a noncanonical GitHub HTTPS origin")
    headers = {
        "Accept": _GITHUB_ACCEPT,
        "User-Agent": _GITHUB_USER_AGENT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if any(name.lower() in {"authorization", "cookie"} for name in headers):
        raise RuntimeError("V2 observer must not send authorization or cookie headers")
    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError(f"GitHub GET returned HTTP {response.status}: {url}")
            if response.geturl() != url:
                raise RuntimeError("GitHub GET redirected away from the frozen URL")
            length_text = response.headers.get("Content-Length")
            if length_text is not None:
                try:
                    content_length = int(length_text)
                except ValueError as exc:
                    raise RuntimeError("GitHub Content-Length is not an integer") from exc
                if content_length < 1 or content_length > max_bytes:
                    raise RuntimeError("GitHub response Content-Length exceeds its bound")
            raw = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub GET failed with HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub GET transport failed: {url}: {exc.reason}") from exc
    if not raw or len(raw) > max_bytes:
        raise RuntimeError("GitHub response is empty or exceeds its byte bound")
    return raw


def _validate_protocol_raw_binding(protocol: Mapping[str, object], raw: bytes) -> None:
    if not isinstance(raw, bytes):
        raise TypeError("V2 execution protocol raw must be bytes")
    parsed = _parse_json_object(raw, "V2 execution protocol raw", _MAX_PROTOCOL_BYTES)
    if parsed != protocol:
        raise ValueError("V2 execution protocol raw differs from supplied payload")
    if raw != canonical_json_bytes(dict(protocol)) + b"\n":
        raise ValueError("V2 execution protocol raw must be canonical LF JSON")


def _relative_receipt_path(value: object) -> str:
    text = _text(value, "V2 anchor receipt path")
    path = pathlib.PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or not text.startswith("artifacts/relationship_lab/")
        or not text.endswith(".json")
    ):
        raise ValueError("V2 anchor receipt path must be canonical under artifacts/relationship_lab")
    return text


def canonical_relationship_condition_reader_qualification_execution_root_v2(
    value: object,
    field_name: str = "qualification execution root",
) -> str:
    text = str(value) if isinstance(value, pathlib.Path) else _text(value, field_name)
    windows_path = pathlib.PureWindowsPath(text)
    is_windows_drive_path = (
        len(windows_path.drive) == 2
        and windows_path.drive[0].isalpha()
        and windows_path.drive[1] == ":"
        and windows_path.root == "\\"
    )
    if not is_windows_drive_path:
        raise ValueError(f"{field_name} must be a canonical absolute local-drive path")
    components = windows_path.parts[1:]
    if not windows_path.is_absolute() or str(windows_path) != text or not components:
        raise ValueError(f"{field_name} must be a canonical absolute local-drive path")
    for component in components:
        base_name = component.split(".", maxsplit=1)[0].rstrip(" .").casefold()
        if (
            component in {"", ".", ".."}
            or component.endswith((" ", "."))
            or base_name in _WINDOWS_RESERVED_EXECUTION_ROOT_NAMES
            or any(
                ord(character) < 32
                or character in _WINDOWS_INVALID_EXECUTION_ROOT_CHARACTERS
                for character in component
            )
        ):
            raise ValueError(f"{field_name} contains an invalid Windows path component")
    return text


def _canonical_windows_path(value: object, field_name: str) -> str:
    text = str(value) if isinstance(value, pathlib.Path) else _text(value, field_name)
    path = pathlib.PureWindowsPath(text)
    if not path.is_absolute() or str(path) != text:
        raise ValueError(f"{field_name} must be a canonical absolute Windows path")
    return text


def _github_utc_timestamp(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", text) is None:
        raise ValueError(f"{field_name} must be canonical whole-second UTC")
    try:
        parsed = dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise ValueError(f"{field_name} must be canonical whole-second UTC")
    return text


def _observer_clock_timestamp(clock: _UtcClock | None) -> str:
    observed = dt.datetime.now(dt.timezone.utc) if clock is None else clock()
    if not isinstance(observed, dt.datetime):
        raise TypeError("V2 observer clock must return a datetime")
    if observed.tzinfo is None or observed.utcoffset() != dt.timedelta(0):
        raise ValueError("V2 observer clock must return an aware UTC datetime")
    return observed.astimezone(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_json_object(raw: bytes, source: str, max_bytes: int) -> Mapping[str, object]:
    parsed = strict_json_loads(raw, max_bytes=max_bytes)
    return _mapping(parsed, source)


def _parse_json_array(raw: bytes, source: str, max_bytes: int) -> list[object]:
    parsed = strict_json_loads(raw, max_bytes=max_bytes)
    if not isinstance(parsed, list):
        raise ValueError(f"{source} must contain a JSON array")
    return parsed


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a string-keyed mapping")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _hex_text(value: object, field_name: str, *, lengths: set[int]) -> str:
    text = _text(value, field_name)
    if len(text) not in lengths or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be lowercase hexadecimal with length in {sorted(lengths)}")
    return text


def _digest(value: object, field_name: str) -> str:
    return _hex_text(value, field_name, lengths={64})


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _exact_keys(payload: Mapping[str, object], expected: set[str], field_name: str) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{field_name} keys differ: missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in core:
        raise ValueError("artifact core must not contain artifact_id")
    return {**dict(core), "artifact_id": _sha256_json(core)}


def _validate_artifact_id(payload: Mapping[str, object], field_name: str) -> None:
    artifact_id = _digest(payload.get("artifact_id"), f"{field_name} artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if artifact_id != _sha256_json(core):
        raise ValueError(f"{field_name} artifact_id mismatch")


def _json_copy(payload: Mapping[str, object]) -> Mapping[str, object]:
    return _parse_json_object(
        canonical_json_bytes(dict(payload)),
        "JSON copy",
        _MAX_PROTOCOL_BYTES,
    )


def _json_type_exact_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            return False
        return all(_json_type_exact_equal(observed[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _json_type_exact_equal(observed_item, expected_item)
                for observed_item, expected_item in zip(observed, expected, strict=True)
            )
        )
    if expected is None:
        return observed is None
    return type(observed) is type(expected) and observed == expected


def _assert_model_free() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_MODEL_MODULE_PREFIXES)
    )
    if imported:
        raise RuntimeError(f"V2 integrity observer imported model module: {imported[0]}")


def _absolute_file(path: pathlib.Path, field_name: str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{field_name} path must be absolute")
    normalized = pathlib.Path(os.path.abspath(candidate))
    if os.path.normcase(str(candidate)) != os.path.normcase(str(normalized)):
        raise ValueError(f"{field_name} path must be lexically canonical")
    return normalized


def _read_stable_regular_file(
    path: pathlib.Path,
    *,
    field_name: str,
    max_bytes: int,
) -> bytes:
    _reject_reparse_components(path, field_name)
    if not os.path.lexists(path):
        raise FileNotFoundError(f"{field_name} is absent: {path}")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{field_name} must be a single-link regular file")
    if before.st_size > max_bytes:
        raise ValueError(f"{field_name} exceeds its byte bound")
    with path.open("rb") as handle:
        during = os.fstat(handle.fileno())
        raw = handle.read(max_bytes + 1)
    after = os.lstat(path)
    if not (_file_identity(before) == _file_identity(during) == _file_identity(after)):
        raise ValueError(f"{field_name} changed identity while being read")
    if len(raw) > max_bytes or len(raw) != before.st_size:
        raise ValueError(f"{field_name} changed size while being read")
    return raw


def _reject_reparse_components(path: pathlib.Path, field_name: str) -> None:
    candidate = path
    while True:
        if os.path.lexists(candidate):
            value = os.lstat(candidate)
            is_reparse = bool(
                os.name == "nt" and getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            )
            if stat.S_ISLNK(value.st_mode) or is_reparse:
                raise ValueError(f"{field_name} traverses a symlink or reparse point")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink


__all__ = [
    "GITHUB_API_VERSION",
    "RELATIONSHIP_READER_EXECUTION_PROTOCOL_V2_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_V2_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_PYTHON_STDLIB_ZIP_V2_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_V2_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_RUNTIME_RAW_TREE_V2_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_SITE_PACKAGES_COVERAGE_V2_SCHEMA_VERSION",
    "RelationshipConditionReaderQualificationIntegrityGuardV2",
    "V2_GIST_FILENAME",
    "canonical_relationship_condition_reader_qualification_execution_root_v2",
    "build_relationship_condition_reader_qualification_execution_protocol_v2",
    "build_relationship_condition_reader_qualification_import_runtime_identity_v2",
    "build_relationship_condition_reader_qualification_integrity_receipt_v2",
    "load_relationship_condition_reader_qualification_execution_protocol_v2",
    "observe_relationship_condition_reader_qualification_public_anchor_v2",
    "relationship_condition_reader_qualification_execution_protocol_id_v2",
    "relationship_condition_reader_qualification_integrity_guard_v2",
    "retired_relationship_condition_reader_qualification_v1_predecessor",
    "validate_relationship_condition_reader_qualification_execution_protocol_v2",
    "validate_relationship_condition_reader_qualification_import_runtime_identity_v2",
    "validate_relationship_condition_reader_qualification_public_anchor_receipt_v2",
]
