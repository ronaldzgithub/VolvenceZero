from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
import pathlib
from typing import Mapping
import urllib.request

import pytest

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads

import lifeform_evolution.relationship_condition_reader_qualification_execution_protocol_v2 as protocol_v2
import lifeform_evolution.relationship_condition_reader_qualification_repository_runtime_coverage as repository_coverage


_GIST_ID = "1234567890abcdef1234567890abcdef"
_VERSION = "c" * 40
_OTHER_VERSION = "d" * 40
_BASE_CONTENT_REVISION = "e" * 40
_REVISION_CONTENT_REVISION = "f" * 40
_CREATED_AT = "2026-08-24T12:21:08Z"
_UPDATED_AT = "2026-08-24T12:21:09Z"
_COMMITTED_AT = "2026-08-24T12:21:10Z"
_OBSERVED_AT = "2026-08-24T12:21:11Z"


@pytest.mark.parametrize(
    "value",
    (
        r"C:\qualification:stream",
        r"C:\CON",
        r"C:\COM¹.txt",
        r"C:\LPT².log",
        r"C:\trailing.",
        r"\\?\C:\qualification",
        r"\\server\share\qualification",
        "/tmp/qualification",
        "//?/C:/qualification",
        "//server/share/qualification",
    ),
)
def test_execution_root_owner_rejects_noncanonical_windows_paths(value: str) -> None:
    with pytest.raises(ValueError):
        protocol_v2.canonical_relationship_condition_reader_qualification_execution_root_v2(
            value
        )


@dataclass(frozen=True)
class _FrozenProtocol:
    payload: Mapping[str, object]
    raw: bytes
    protocol_id: str
    execution_root: pathlib.Path


@dataclass
class _GitHubObservation:
    base: dict[str, object]
    commits: list[object]
    commits_reobservation: list[object]
    revision: dict[str, object]
    base_api_protocol_raw: bytes
    revision_protocol_raw: bytes


def _host_test_execution_root(tmp_path: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return tmp_path / "relationship-condition-reader-v2-execution"
    return pathlib.Path(r"D:\relationship-condition-reader-v2-execution")


def _source_manifest_with_v2_composition_sources(
    source_manifest: object,
) -> Mapping[str, object]:
    assert isinstance(source_manifest, dict)
    manifest = copy.deepcopy(source_manifest)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    repository_root = pathlib.Path(__file__).parents[3]
    existing_paths = {str(row["path"]) for row in entries if isinstance(row, dict)}
    for relative_path in sorted(protocol_v2._V2_REQUIRED_SOURCE_PATHS, key=lambda item: item.encode("utf-8")):
        if relative_path in existing_paths:
            continue
        raw = (repository_root / pathlib.PurePosixPath(relative_path)).read_bytes()
        entries.append(
            {
                "path": relative_path,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
            }
        )
    entries.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    manifest["entry_count"] = len(entries)
    manifest["total_raw_bytes"] = sum(int(row["raw_bytes"]) for row in entries)
    del manifest["artifact_id"]
    manifest["artifact_id"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


def _synthetic_repository_runtime_coverage(
    execution_source_tree: Mapping[str, object],
) -> Mapping[str, object]:
    raw_entries = execution_source_tree["entries"]
    assert isinstance(raw_entries, list)
    entries = [
        copy.deepcopy(row)
        for row in raw_entries
        if isinstance(row, dict) and str(row["path"]).startswith("packages/") and str(row["path"]).endswith(".py")
    ]
    source_roots = sorted(
        {pathlib.PurePosixPath(*pathlib.PurePosixPath(str(row["path"])).parts[:3]).as_posix() for row in entries},
        key=lambda value: value.encode("utf-8"),
    )
    core = {
        "schema_version": (repository_coverage.RELATIONSHIP_READER_REPOSITORY_RUNTIME_COVERAGE_SCHEMA_VERSION),
        "repository_root": str(pathlib.Path(__file__).parents[3]),
        "execution_source_tree_schema_version": execution_source_tree["schema_version"],
        "execution_source_tree_artifact_id": execution_source_tree["artifact_id"],
        "execution_source_tree_entry_count": execution_source_tree["entry_count"],
        "path_contract": repository_coverage._PATH_CONTRACT,
        "content_contract": repository_coverage._CONTENT_CONTRACT,
        "link_contract": repository_coverage._LINK_CONTRACT,
        "excluded_cache_directory_names": [repository_coverage._EXCLUDED_CACHE_DIRECTORY_NAME],
        "forbidden_native_suffixes": list(repository_coverage._FORBIDDEN_NATIVE_SUFFIXES),
        "source_roots": source_roots,
        "source_root_count": len(source_roots),
        "entries": entries,
        "entry_count": len(entries),
        "total_raw_bytes": sum(int(row["raw_bytes"]) for row in entries),
        "excluded_cache_directory_count": 0,
        "excluded_bytecode_file_count": 0,
        "frozen_python_source_entry_count": len(entries),
        "frozen_python_source_join_sha256": hashlib.sha256(canonical_json_bytes({"entries": entries})).hexdigest(),
    }
    payload = repository_coverage._with_artifact_id(core)
    repository_coverage.validate_relationship_condition_reader_repository_runtime_coverage(
        payload,
        execution_source_tree=execution_source_tree,
    )
    return payload


def _runtime_raw_tree(
    *,
    root: pathlib.PureWindowsPath,
    tree_role: str,
    excluded_top_level_directories: tuple[str, ...],
    entry_path: str,
) -> Mapping[str, object]:
    core = {
        "schema_version": protocol_v2.RELATIONSHIP_READER_EXECUTION_RUNTIME_RAW_TREE_V2_SCHEMA_VERSION,
        "tree_role": tree_role,
        "root": str(root),
        "content_contract": "exact_raw_bytes_no_eol_normalization.v1",
        "link_contract": "regular_non_symlink_non_reparse_hardlinks_allowed.v1",
        "excluded_top_level_directories": list(excluded_top_level_directories),
        "cache_directory_names": list(protocol_v2._RUNTIME_TREE_CACHE_DIRECTORY_NAMES),
        "excluded_cache_bytecode_suffixes": list(protocol_v2._RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES),
        "excluded_cache_bytecode_file_count": 0,
        "entries": [
            {
                "path": entry_path,
                "raw_sha256": hashlib.sha256(entry_path.encode("utf-8")).hexdigest(),
                "raw_bytes": len(entry_path.encode("utf-8")),
            }
        ],
        "entry_count": 1,
        "total_raw_bytes": len(entry_path.encode("utf-8")),
    }
    return protocol_v2._with_artifact_id(core)


def _site_packages_coverage(
    *,
    root: pathlib.PureWindowsPath,
    installed_distribution_count: int,
) -> Mapping[str, object]:
    core = {
        "schema_version": protocol_v2.RELATIONSHIP_READER_EXECUTION_SITE_PACKAGES_COVERAGE_V2_SCHEMA_VERSION,
        "root": str(root),
        "coverage_contract": ("every_nonstartup_noncache_bytecode_file_is_record_owned_or_raw_pinned.v1"),
        "link_contract": "regular_non_symlink_non_reparse_single_link_files_only.v1",
        "cache_directory_names": list(protocol_v2._RUNTIME_TREE_CACHE_DIRECTORY_NAMES),
        "excluded_cache_bytecode_suffixes": list(protocol_v2._RUNTIME_TREE_CACHE_BYTECODE_SUFFIXES),
        "excluded_site_startup_suffixes": list(protocol_v2._SITE_PACKAGES_EXCLUDED_SITE_STARTUP_SUFFIXES),
        "record_ownership_join_sha256": hashlib.sha256(b"synthetic ownership join").hexdigest(),
        "record_hashed_site_path_count": installed_distribution_count,
        "record_ownership_overlap_count": 0,
        "total_regular_file_count": installed_distribution_count * 2,
        "owned_hashed_record_file_count": installed_distribution_count,
        "record_self_file_count": installed_distribution_count,
        "excluded_bytecode_file_count": 0,
        "excluded_site_startup_file_count": 0,
        "unowned_regular_files": [],
        "unowned_regular_file_count": 0,
        "symlink_or_reparse_entry_count": 0,
        "non_single_link_regular_file_count": 0,
        "all_nonexcluded_regular_files_record_owned_or_raw_pinned": True,
    }
    return protocol_v2._with_artifact_id(core)


def _absent_python_stdlib_zip(
    *,
    environment_root: pathlib.PureWindowsPath,
    python_version: str,
) -> Mapping[str, object]:
    major, minor = python_version.split(".")[:2]
    core = {
        "schema_version": protocol_v2.RELATIONSHIP_READER_EXECUTION_PYTHON_STDLIB_ZIP_V2_SCHEMA_VERSION,
        "path": str(environment_root / f"python{major}{minor}.zip"),
        "exists": False,
        "raw_sha256": None,
        "raw_bytes": None,
        "precedence_contract": "python_home_zip_precedes_dlls_and_lib_on_minus_P_minus_S_sys_path.v1",
        "link_contract": "absent_or_regular_non_symlink_non_reparse_single_link_file.v1",
    }
    return protocol_v2._with_artifact_id(core)


def _python_home_top_level_tree(
    *,
    environment_root: pathlib.PureWindowsPath,
) -> Mapping[str, object]:
    entry_path = "python.exe"
    core = {
        "schema_version": (protocol_v2.RELATIONSHIP_READER_EXECUTION_PYTHON_HOME_TOP_LEVEL_TREE_V2_SCHEMA_VERSION),
        "tree_role": protocol_v2._PYTHON_HOME_TOP_LEVEL_TREE_ROLE,
        "root": str(environment_root),
        "content_contract": "exact_raw_bytes_no_eol_normalization.v1",
        "link_contract": "regular_non_symlink_non_reparse_single_link_files_only.v1",
        "directory_contract": ("direct_children_non_symlink_non_reparse_directories_names_frozen.v1"),
        "directories": ["DLLs", "Lib", "Library"],
        "directory_count": 3,
        "entries": [
            {
                "path": entry_path,
                "raw_sha256": hashlib.sha256(entry_path.encode("utf-8")).hexdigest(),
                "raw_bytes": len(entry_path.encode("utf-8")),
            }
        ],
        "entry_count": 1,
        "total_raw_bytes": len(entry_path.encode("utf-8")),
    }
    return protocol_v2._with_artifact_id(core)


def _synthetic_import_runtime(base_runtime: Mapping[str, object]) -> Mapping[str, object]:
    python_identity = base_runtime["python"]
    assert isinstance(python_identity, dict)
    environment_root = pathlib.PureWindowsPath(str(python_identity["executable"])).parent
    site_packages_root = environment_root / "Lib" / "site-packages"
    base_distributions = base_runtime["distributions"]
    assert isinstance(base_distributions, list)
    installed_distributions = [
        protocol_v2._inventory_row_from_v1_distribution_pin(
            value,
            site_packages_root=str(site_packages_root),
        )
        for value in base_distributions
        if isinstance(value, dict)
    ]
    for row in installed_distributions:
        row.pop("record_hashed_entries_verified")
        row["record_hashed_site_packages_entry_count"] = row["record_hashed_entry_count"]
        row["record_hashed_environment_external_entry_count"] = 0
        row["record_hashed_absolute_environment_entry_count"] = 0
        row["record_hashed_absolute_environment_entries"] = []
        row["record_hashed_absent_outside_environment_entry_count"] = 0
        row["record_absent_outside_environment_entries"] = []
        row["record_hashed_pinned_identity_mismatch_entry_count"] = 0
        row["record_pinned_identity_mismatch_entries"] = []
        row["record_in_environment_hashed_entries_verified_or_explicitly_pinned"] = True
        row["record_absent_outside_environment_entries_attested"] = True
    installed_distributions.sort(key=lambda row: str(row["normalized_name"]).encode("utf-8"))
    core = {
        "schema_version": protocol_v2.RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_V2_SCHEMA_VERSION,
        "platform": copy.deepcopy(base_runtime["platform"]),
        "gpu": copy.deepcopy(base_runtime["gpu"]),
        "python": copy.deepcopy(base_runtime["python"]),
        "distributions": copy.deepcopy(base_distributions),
        "v1_runtime_identity_artifact_id": base_runtime["artifact_id"],
        "child_import_contract": copy.deepcopy(protocol_v2._CHILD_IMPORT_CONTRACT_V2),
        "python_environment_root": str(environment_root),
        "site_packages_root": str(site_packages_root),
        "installed_distributions": installed_distributions,
        "installed_distribution_count": len(installed_distributions),
        "site_packages_coverage": _site_packages_coverage(
            root=site_packages_root,
            installed_distribution_count=len(installed_distributions),
        ),
        "python_stdlib_zip": _absent_python_stdlib_zip(
            environment_root=environment_root,
            python_version=str(python_identity["version"]),
        ),
        "python_home_top_level_tree": _python_home_top_level_tree(
            environment_root=environment_root,
        ),
        "stdlib_lib_tree": _runtime_raw_tree(
            root=environment_root / "Lib",
            tree_role=protocol_v2._STDLIB_TREE_ROLE,
            excluded_top_level_directories=("site-packages",),
            entry_path="encodings/__init__.py",
        ),
        "dlls_tree": _runtime_raw_tree(
            root=environment_root / "DLLs",
            tree_role=protocol_v2._DLLS_TREE_ROLE,
            excluded_top_level_directories=(),
            entry_path="_ssl.pyd",
        ),
        "python_library_bin_tree": _runtime_raw_tree(
            root=environment_root / "Library" / "bin",
            tree_role=protocol_v2._LIBRARY_BIN_TREE_ROLE,
            excluded_top_level_directories=(),
            entry_path="cublas64_12.dll",
        ),
    }
    payload = protocol_v2._with_artifact_id(core)
    protocol_v2.validate_relationship_condition_reader_qualification_import_runtime_identity_v2(payload)
    return payload


def _rehash_artifact(payload: dict[str, object]) -> None:
    payload.pop("artifact_id", None)
    payload["artifact_id"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record_hash_field(raw: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode("ascii")


@pytest.fixture
def frozen_protocol(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _FrozenProtocol:
    v1_path = (
        pathlib.Path(__file__).parents[1]
        / "src"
        / "lifeform_evolution"
        / "protocols"
        / "relationship_condition_reader_qualification_execution_v1.json"
    )
    v1_payload = strict_json_loads(v1_path.read_bytes(), max_bytes=8_000_000)
    assert isinstance(v1_payload, dict)
    v1_runtime = v1_payload["runtime_identity"]
    assert isinstance(v1_runtime, dict)
    import_runtime = _synthetic_import_runtime(v1_runtime)

    def observe_import_runtime(
        *,
        base_runtime_identity: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        if base_runtime_identity is not None:
            assert protocol_v2._json_type_exact_equal(base_runtime_identity, v1_runtime)
        return copy.deepcopy(import_runtime)

    monkeypatch.setattr(
        protocol_v2,
        "build_relationship_condition_reader_qualification_import_runtime_identity_v2",
        observe_import_runtime,
    )
    source_tree = _source_manifest_with_v2_composition_sources(v1_payload["execution_source_tree"])
    repository_runtime = _synthetic_repository_runtime_coverage(source_tree)

    def observe_repository_runtime(
        *,
        repository_root: pathlib.Path,
        execution_source_tree: Mapping[str, object],
    ) -> Mapping[str, object]:
        del repository_root
        assert protocol_v2._json_type_exact_equal(execution_source_tree, source_tree)
        return copy.deepcopy(repository_runtime)

    monkeypatch.setattr(
        protocol_v2,
        "build_relationship_condition_reader_repository_runtime_coverage",
        observe_repository_runtime,
    )
    execution_root = _host_test_execution_root(tmp_path)
    payload = protocol_v2.build_relationship_condition_reader_qualification_execution_protocol_v2(
        preflight_binding=v1_payload["qualification_preflight"],
        source_tree_manifest=source_tree,
        repository_runtime_coverage=repository_runtime,
        bge_snapshot_tree_manifest=v1_payload["bge_snapshot_tree"],
        runtime_identity=v1_payload["runtime_identity"],
        proposed_execution_root=execution_root,
        anchor_receipt_relative_path=(
            "artifacts/relationship_lab/relationship_condition_reader_qualification_"
            "execution_public_gist_anchor_v2_test.json"
        ),
    )
    raw = canonical_json_bytes(dict(payload)) + b"\n"
    protocol_id = protocol_v2.relationship_condition_reader_qualification_execution_protocol_id_v2(payload)
    return _FrozenProtocol(
        payload=payload,
        raw=raw,
        protocol_id=protocol_id,
        execution_root=execution_root,
    )


def _gist_file(*, raw_url: str, protocol_raw: bytes) -> dict[str, object]:
    return {
        "filename": protocol_v2.V2_GIST_FILENAME,
        "type": "application/json",
        "language": "JSON",
        "raw_url": raw_url,
        "size": len(protocol_raw),
        "truncated": False,
        "content": protocol_raw.decode("utf-8"),
        "future_file_field": {"accepted": True},
    }


def _github_observation(protocol_raw: bytes) -> _GitHubObservation:
    base_url = f"https://api.github.com/gists/{_GIST_ID}"
    gist_url = f"https://gist.github.com/ronaldzgithub/{_GIST_ID}"
    base_raw_url = (
        "https://gist.githubusercontent.com/ronaldzgithub/"
        f"{_GIST_ID}/raw/{_BASE_CONTENT_REVISION}/{protocol_v2.V2_GIST_FILENAME}"
    )
    revision_file_raw_url = (
        "https://gist.githubusercontent.com/ronaldzgithub/"
        f"{_GIST_ID}/raw/{_REVISION_CONTENT_REVISION}/{protocol_v2.V2_GIST_FILENAME}"
    )
    common = {
        "id": _GIST_ID,
        "public": True,
        "html_url": gist_url,
        "created_at": _CREATED_AT,
        "updated_at": _UPDATED_AT,
        "owner": {
            "login": "ronaldzgithub",
            "id": 12345,
            "type": "User",
            "future_owner_field": "accepted",
        },
        "truncated": False,
        "description": "V2 frozen qualification protocol",
        "comments": 0,
    }
    base = {
        **copy.deepcopy(common),
        "url": base_url,
        "commits_url": f"{base_url}/commits",
        "comments_url": f"{base_url}/comments",
        "files": {
            protocol_v2.V2_GIST_FILENAME: _gist_file(
                raw_url=base_raw_url,
                protocol_raw=protocol_raw,
            )
        },
        "future_base_field": ["additive", "accepted"],
    }
    revision = {
        **copy.deepcopy(common),
        "url": f"{base_url}/{_VERSION}",
        "files": {
            protocol_v2.V2_GIST_FILENAME: _gist_file(
                raw_url=revision_file_raw_url,
                protocol_raw=protocol_raw,
            )
        },
        "future_revision_field": {"additive": True},
    }
    commits: list[object] = [
        {
            "version": _VERSION,
            "committed_at": _COMMITTED_AT,
            "url": f"{base_url}/{_VERSION}",
            "user": {"login": "ronaldzgithub", "id": 12345},
            "change_status": {"total": 1, "additions": 1, "deletions": 0},
            "future_commit_field": "accepted",
        }
    ]
    return _GitHubObservation(
        base=base,
        commits=commits,
        commits_reobservation=copy.deepcopy(commits),
        revision=revision,
        base_api_protocol_raw=protocol_raw,
        revision_protocol_raw=protocol_raw,
    )


def test_historical_v1_protocol_raw_bytes_remain_exact() -> None:
    v1_path = (
        pathlib.Path(__file__).parents[1]
        / "src"
        / "lifeform_evolution"
        / "protocols"
        / "relationship_condition_reader_qualification_execution_v1.json"
    )
    raw = v1_path.read_bytes()

    assert len(raw) == 186_131
    assert hashlib.sha256(raw).hexdigest() == ("02dd24e68efdd7c988c84ac250d48116d4bba637fbf7dad3add5d9c491614572")


def test_v2_protocol_preserves_v1_and_admits_registered_v2_preflight_lineage(
    frozen_protocol: _FrozenProtocol,
) -> None:
    original_binding = frozen_protocol.payload["qualification_preflight"]
    assert isinstance(original_binding, dict)
    assert (
        protocol_v2._qualification_protocol_schema_version_from_preflight_binding(
            original_binding
        )
        == protocol_v2.RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1
    )

    payload = copy.deepcopy(frozen_protocol.payload)
    binding = payload["qualification_preflight"]
    assert isinstance(binding, dict)
    rows = binding["files"]
    assert isinstance(rows, list)
    protocol_row = next(row for row in rows if row["path"] == "protocol.json")
    protocol_row["schema_version"] = (
        protocol_v2.RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2
    )
    _rehash_artifact(binding)

    protocol_v2._validate_protocol_shape(payload)
    assert (
        protocol_v2._qualification_protocol_schema_version_from_preflight_binding(
            binding
        )
        == protocol_v2.RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2
    )


def test_v2_preflight_lineage_rejects_unknown_duplicate_or_missing_protocol_row(
    frozen_protocol: _FrozenProtocol,
) -> None:
    binding = copy.deepcopy(frozen_protocol.payload["qualification_preflight"])
    assert isinstance(binding, dict)
    rows = binding["files"]
    assert isinstance(rows, list)
    protocol_row = next(row for row in rows if row["path"] == "protocol.json")
    protocol_row["schema_version"] = "relationship-condition-reader-qualification-protocol.v3"
    with pytest.raises(ValueError, match="unsupported V2 qualification protocol schema"):
        protocol_v2._qualification_protocol_schema_version_from_preflight_binding(
            binding
        )

    protocol_row["schema_version"] = (
        protocol_v2.RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1
    )
    rows.append(copy.deepcopy(protocol_row))
    with pytest.raises(ValueError, match="exactly one protocol.json row"):
        protocol_v2._qualification_protocol_schema_version_from_preflight_binding(
            binding
        )

    binding["files"] = [row for row in rows if row["path"] != "protocol.json"]
    with pytest.raises(ValueError, match="exactly one protocol.json row"):
        protocol_v2._qualification_protocol_schema_version_from_preflight_binding(
            binding
        )


def test_v2_runtime_identity_freezes_complete_child_import_domains(
    frozen_protocol: _FrozenProtocol,
) -> None:
    runtime = frozen_protocol.payload["runtime_identity"]
    assert isinstance(runtime, dict)
    assert set(runtime) == {
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
    }
    environment_root = pathlib.PureWindowsPath(str(runtime["python_environment_root"]))
    assert runtime["site_packages_root"] == str(environment_root / "Lib" / "site-packages")
    installed = runtime["installed_distributions"]
    assert isinstance(installed, list)
    assert runtime["installed_distribution_count"] == len(installed)
    assert len(installed) >= len(runtime["distributions"])
    assert all(row["record_in_environment_hashed_entries_verified_or_explicitly_pinned"] is True for row in installed)
    assert all(row["record_absent_outside_environment_entries_attested"] is True for row in installed)
    child_contract = runtime["child_import_contract"]
    assert isinstance(child_contract, dict)
    assert child_contract["ambient_pythonpath_inherited"] is False
    assert child_contract["site_startup_enabled"] is False
    assert child_contract["adjacent_bytecode_cache_read_allowed"] is False
    assert child_contract["schema_version"] == ("relationship-condition-reader-qualification-child-import-contract.v3")
    assert child_contract["ambient_torchinductor_cache_dir_inherited"] is False
    assert child_contract["per_child_isolated_torchinductor_cache_required"] is True
    assert child_contract["torchinductor_cache_direct_child_of_predictor_capsule"] is True
    assert child_contract["torchinductor_cache_absent_before_launch_required"] is True
    assert child_contract["torchinductor_cache_materialized_after_child_required"] is True
    assert child_contract["torchinductor_cache_exact_empty_after_child_required"] is True
    assert child_contract["torchinductor_cache_non_reparse_required"] is True
    assert child_contract["torchinductor_cache_outer_reobservation_required"] is True
    assert child_contract["predictor_username_required"] is True
    assert child_contract["predictor_kmp_duplicate_lib_ok"] == "True"
    assert child_contract["predictor_kmp_init_at_fork"] == "FALSE"
    assert child_contract["child_sys_path_exact_validation_required"] is True
    assert child_contract["child_module_origin_exact_validation_required"] is True
    assert child_contract["ambient_path_inherited"] is False
    assert child_contract["controlled_path_order"] == [
        "<python_environment_root>",
        "<python_environment_root>\\DLLs",
        "<python_environment_root>\\Library\\bin",
        "<SystemRoot>\\System32",
        "<SystemRoot>",
    ]
    assert child_contract["ambient_cuda_path_inherited"] is False
    assert child_contract["predictor_cuda_visible_devices"] == "0"
    assert child_contract["predictor_v1_physical_gpu_index"] == 0
    assert child_contract["scorer_cuda_visible_devices"] is None
    assert child_contract["scorer_cuda_execution_allowed"] is False
    coverage = runtime["site_packages_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["root"] == runtime["site_packages_root"]
    assert coverage["record_self_file_count"] == runtime["installed_distribution_count"]
    assert coverage["all_nonexcluded_regular_files_record_owned_or_raw_pinned"] is True
    stdlib_zip = runtime["python_stdlib_zip"]
    assert isinstance(stdlib_zip, dict)
    runtime_python = runtime["python"]
    assert isinstance(runtime_python, dict)
    major, minor = str(runtime_python["version"]).split(".")[:2]
    assert stdlib_zip["path"] == str(environment_root / f"python{major}{minor}.zip")
    assert stdlib_zip["exists"] is False
    assert stdlib_zip["raw_sha256"] is None
    assert stdlib_zip["raw_bytes"] is None
    stdlib_tree = runtime["stdlib_lib_tree"]
    dlls_tree = runtime["dlls_tree"]
    library_bin_tree = runtime["python_library_bin_tree"]
    assert isinstance(stdlib_tree, dict)
    assert isinstance(dlls_tree, dict)
    assert isinstance(library_bin_tree, dict)
    assert stdlib_tree["excluded_top_level_directories"] == ["site-packages"]
    assert stdlib_tree["cache_directory_names"] == ["__pycache__"]
    assert stdlib_tree["excluded_cache_bytecode_suffixes"] == [".pyc"]
    assert stdlib_tree["excluded_cache_bytecode_file_count"] == 0
    assert dlls_tree["excluded_top_level_directories"] == []
    assert library_bin_tree["root"] == str(environment_root / "Library" / "bin")


@pytest.mark.parametrize(
    "target",
    [
        "installed_distribution_count",
        "record_in_environment_hashed_entries_verified_or_explicitly_pinned",
        "stdlib_entry_count",
        "ambient_pythonpath_inherited",
        "site_packages_coverage_closure",
        "python_stdlib_zip_exists",
        "library_bin_entry_count",
    ],
)
def test_v2_runtime_identity_rejects_json_shape_and_type_confusion(
    frozen_protocol: _FrozenProtocol,
    target: str,
) -> None:
    runtime = copy.deepcopy(frozen_protocol.payload["runtime_identity"])
    assert isinstance(runtime, dict)
    if target == "installed_distribution_count":
        runtime[target] = float(runtime[target])
    elif target == "record_in_environment_hashed_entries_verified_or_explicitly_pinned":
        installed = runtime["installed_distributions"]
        assert isinstance(installed, list) and isinstance(installed[0], dict)
        installed[0][target] = 1
    elif target == "stdlib_entry_count":
        stdlib_tree = runtime["stdlib_lib_tree"]
        assert isinstance(stdlib_tree, dict)
        stdlib_tree["entry_count"] = float(stdlib_tree["entry_count"])
    elif target == "ambient_pythonpath_inherited":
        child_contract = runtime["child_import_contract"]
        assert isinstance(child_contract, dict)
        child_contract[target] = 0
    elif target == "site_packages_coverage_closure":
        coverage = runtime["site_packages_coverage"]
        assert isinstance(coverage, dict)
        coverage["all_nonexcluded_regular_files_record_owned_or_raw_pinned"] = 1
    elif target == "python_stdlib_zip_exists":
        stdlib_zip = runtime["python_stdlib_zip"]
        assert isinstance(stdlib_zip, dict)
        stdlib_zip["exists"] = 0
    else:
        library_bin_tree = runtime["python_library_bin_tree"]
        assert isinstance(library_bin_tree, dict)
        library_bin_tree["entry_count"] = float(library_bin_tree["entry_count"])

    with pytest.raises(ValueError):
        protocol_v2.validate_relationship_condition_reader_qualification_import_runtime_identity_v2(runtime)


def test_v2_runtime_identity_rejects_missing_site_packages_root(
    frozen_protocol: _FrozenProtocol,
) -> None:
    runtime = copy.deepcopy(frozen_protocol.payload["runtime_identity"])
    assert isinstance(runtime, dict)
    del runtime["site_packages_root"]

    with pytest.raises(ValueError, match="keys differ"):
        protocol_v2.validate_relationship_condition_reader_qualification_import_runtime_identity_v2(runtime)


def test_protocol_validator_reobserves_complete_runtime_and_rejects_drift(
    frozen_protocol: _FrozenProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = copy.deepcopy(frozen_protocol.payload["runtime_identity"])
    assert isinstance(drifted, dict)
    stdlib_zip = drifted["python_stdlib_zip"]
    assert isinstance(stdlib_zip, dict)
    stdlib_zip["exists"] = True
    stdlib_zip["raw_sha256"] = "f" * 64
    stdlib_zip["raw_bytes"] = 1
    _rehash_artifact(stdlib_zip)
    _rehash_artifact(drifted)
    protocol_v2.validate_relationship_condition_reader_qualification_import_runtime_identity_v2(drifted)
    monkeypatch.setattr(
        protocol_v2,
        "build_relationship_condition_reader_qualification_import_runtime_identity_v2",
        lambda: copy.deepcopy(drifted),
    )

    with pytest.raises(ValueError, match="current exact runtime"):
        protocol_v2.validate_relationship_condition_reader_qualification_execution_protocol_v2(
            frozen_protocol.payload,
            expected_protocol_id=frozen_protocol.protocol_id,
        )


def test_runtime_raw_tree_allows_pycache_bytecode_but_rejects_adjacent_bytecode(
    tmp_path: pathlib.Path,
) -> None:
    tree_root = tmp_path / "Lib"
    (tree_root / "site-packages").mkdir(parents=True)
    (tree_root / "site-packages" / "startup.pth").write_bytes(b"import startup")
    (tree_root / "__pycache__").mkdir()
    (tree_root / "__pycache__" / "cached.pyc").write_bytes(b"cache")
    (tree_root / "__pycache__" / "metadata.json").write_bytes(b'{"raw_pinned":true}\n')
    (tree_root / "encodings").mkdir()
    (tree_root / "encodings" / "__init__.py").write_bytes(b"# frozen stdlib\n")

    tree = protocol_v2._observe_runtime_raw_tree(
        root=tree_root,
        tree_role=protocol_v2._STDLIB_TREE_ROLE,
        excluded_top_level_directories=("site-packages",),
    )

    assert tree["entry_count"] == 2
    assert tree["excluded_cache_bytecode_file_count"] == 1
    assert tree["entries"] == [
        {
            "path": "__pycache__/metadata.json",
            "raw_sha256": hashlib.sha256(b'{"raw_pinned":true}\n').hexdigest(),
            "raw_bytes": len(b'{"raw_pinned":true}\n'),
        },
        {
            "path": "encodings/__init__.py",
            "raw_sha256": hashlib.sha256(b"# frozen stdlib\n").hexdigest(),
            "raw_bytes": len(b"# frozen stdlib\n"),
        },
    ]
    (tree_root / "BYTECODE.PYC").write_bytes(b"sourceless shadow")
    with pytest.raises(ValueError, match="adjacent bytecode outside __pycache__"):
        protocol_v2._observe_runtime_raw_tree(
            root=tree_root,
            tree_role=protocol_v2._STDLIB_TREE_ROLE,
            excluded_top_level_directories=("site-packages",),
        )


def test_distribution_record_closes_exact_pip_absence_and_conda_installer_pin(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "python-env"
    site_root = environment_root / "Lib" / "site-packages"
    dist_info = site_root / "pip-26.1.2.dist-info"
    package = site_root / "pip"
    dist_info.mkdir(parents=True)
    package.mkdir()
    package_raw = b"version = '26.1.2'\n"
    installer_declared_raw = b"pip\n"
    (package / "__init__.py").write_bytes(package_raw)
    (dist_info / "INSTALLER").write_bytes(b"conda")
    record_raw = (
        f"../../../bin/pip,{_record_hash_field(b'console script')},14\n"
        f"pip/__init__.py,{_record_hash_field(package_raw)},{len(package_raw)}\n"
        f"pip-26.1.2.dist-info/INSTALLER,{_record_hash_field(installer_declared_raw)},"
        f"{len(installer_declared_raw)}\n"
        "pip-26.1.2.dist-info/RECORD,,\n"
    ).encode("utf-8")
    (dist_info / "RECORD").write_bytes(record_raw)

    verification = protocol_v2._verify_installed_distribution_record_entries_v2(
        record_raw,
        site_packages_root=site_root,
        environment_root=environment_root,
        distribution_name="pip",
        normalized_distribution_name="pip",
        inside_entries_preverified=False,
    )

    assert verification["record_hashed_entry_count"] == 3
    assert verification["record_hashed_site_packages_entry_count"] == 1
    assert verification["record_hashed_environment_external_entry_count"] == 0
    assert verification["record_hashed_absent_outside_environment_entry_count"] == 1
    assert verification["record_hashed_pinned_identity_mismatch_entry_count"] == 1
    assert verification["record_absent_outside_environment_entries"][0]["target_absent"] is True
    mismatch = verification["record_pinned_identity_mismatch_entries"][0]
    assert mismatch["record_path"] == "pip-26.1.2.dist-info/INSTALLER"
    assert mismatch["observed_raw_sha256"] == hashlib.sha256(b"conda").hexdigest()
    assert mismatch["observed_raw_bytes"] == 5
    assert mismatch["target_raw_pinned"] is True


def test_distribution_record_rejects_present_or_unregistered_outside_environment_target(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "python-env"
    site_root = environment_root / "Lib" / "site-packages"
    dist_info = site_root / "pip-26.1.2.dist-info"
    dist_info.mkdir(parents=True)
    outside_bin = environment_root.parent / "bin"
    outside_bin.mkdir()
    (outside_bin / "pip").write_bytes(b"console script")
    record_raw = (
        f"../../../bin/pip,{_record_hash_field(b'console script')},14\npip-26.1.2.dist-info/RECORD,,\n"
    ).encode("utf-8")
    (dist_info / "RECORD").write_bytes(record_raw)

    with pytest.raises(ValueError, match="must remain absent"):
        protocol_v2._verify_installed_distribution_record_entries_v2(
            record_raw,
            site_packages_root=site_root,
            environment_root=environment_root,
            distribution_name="pip",
            normalized_distribution_name="pip",
            inside_entries_preverified=False,
        )

    (outside_bin / "pip").unlink()
    unregistered = record_raw.replace(b"../../../bin/pip", b"../../../bin/demo")
    (dist_info / "RECORD").write_bytes(unregistered)
    with pytest.raises(ValueError, match="escaped the frozen Python environment"):
        protocol_v2._verify_installed_distribution_record_entries_v2(
            unregistered,
            site_packages_root=site_root,
            environment_root=environment_root,
            distribution_name="pip",
            normalized_distribution_name="pip",
            inside_entries_preverified=False,
        )


def test_distribution_record_verifies_only_permitted_environment_external_roots(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "python-env"
    site_root = environment_root / "Lib" / "site-packages"
    dist_info = site_root / "demo-1.0.dist-info"
    scripts = environment_root / "Scripts"
    dist_info.mkdir(parents=True)
    scripts.mkdir()
    script_raw = b"launcher"
    (scripts / "demo.exe").write_bytes(script_raw)
    record_raw = (
        f"../../Scripts/demo.exe,{_record_hash_field(script_raw)},{len(script_raw)}\ndemo-1.0.dist-info/RECORD,,\n"
    ).encode("utf-8")
    (dist_info / "RECORD").write_bytes(record_raw)

    verification = protocol_v2._verify_installed_distribution_record_entries_v2(
        record_raw,
        site_packages_root=site_root,
        environment_root=environment_root,
        distribution_name="demo",
        normalized_distribution_name="demo",
        inside_entries_preverified=False,
    )
    assert verification["record_hashed_environment_external_entry_count"] == 1

    other = environment_root / "Other"
    other.mkdir()
    (other / "demo.exe").write_bytes(script_raw)
    unpermitted = record_raw.replace(b"../../Scripts/demo.exe", b"../../Other/demo.exe")
    (dist_info / "RECORD").write_bytes(unpermitted)
    with pytest.raises(ValueError, match="escaped permitted Python-environment external roots"):
        protocol_v2._verify_installed_distribution_record_entries_v2(
            unpermitted,
            site_packages_root=site_root,
            environment_root=environment_root,
            distribution_name="demo",
            normalized_distribution_name="demo",
            inside_entries_preverified=False,
        )


def test_distribution_record_verifies_canonical_absolute_windows_path_only_inside_environment(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "python-env"
    site_root = environment_root / "Lib" / "site-packages"
    dist_info = site_root / "wheel-0.47.0.dist-info"
    package = site_root / "wheel"
    scripts = environment_root / "Scripts"
    dist_info.mkdir(parents=True)
    package.mkdir()
    scripts.mkdir()
    package_raw = b"version = '0.47.0'\n"
    launcher_raw = b"wheel launcher"
    (package / "__init__.py").write_bytes(package_raw)
    launcher = scripts / "wheel.exe"
    launcher.write_bytes(launcher_raw)
    absolute_record_path = launcher.resolve().as_posix()
    record_raw = (
        f"wheel/__init__.py,{_record_hash_field(package_raw)},{len(package_raw)}\n"
        f"{absolute_record_path},{_record_hash_field(launcher_raw)},{len(launcher_raw)}\n"
        "wheel-0.47.0.dist-info/RECORD,,\n"
    ).encode("utf-8")
    (dist_info / "RECORD").write_bytes(record_raw)

    verification = protocol_v2._verify_installed_distribution_record_entries_v2(
        record_raw,
        site_packages_root=site_root,
        environment_root=environment_root,
        distribution_name="wheel",
        normalized_distribution_name="wheel",
        inside_entries_preverified=False,
    )

    assert verification["record_hashed_environment_external_entry_count"] == 1
    assert verification["record_hashed_absolute_environment_entry_count"] == 1
    absolute = verification["record_hashed_absolute_environment_entries"][0]
    assert absolute["record_path"] == absolute_record_path
    assert absolute["declared_raw_sha256"] == absolute["observed_raw_sha256"]
    assert absolute["declared_raw_bytes"] == absolute["observed_raw_bytes"]

    declared_launcher_raw = b"pip-generated wheel launcher"
    mismatched_record_raw = (
        f"wheel/__init__.py,{_record_hash_field(package_raw)},{len(package_raw)}\n"
        f"{absolute_record_path},{_record_hash_field(declared_launcher_raw)},"
        f"{len(declared_launcher_raw)}\n"
        "wheel-0.47.0.dist-info/RECORD,,\n"
    ).encode("utf-8")
    (dist_info / "RECORD").write_bytes(mismatched_record_raw)
    pinned = protocol_v2._verify_installed_distribution_record_entries_v2(
        mismatched_record_raw,
        site_packages_root=site_root,
        environment_root=environment_root,
        distribution_name="wheel",
        normalized_distribution_name="wheel",
        inside_entries_preverified=False,
    )
    assert pinned["record_hashed_absolute_environment_entry_count"] == 0
    assert pinned["record_hashed_pinned_identity_mismatch_entry_count"] == 1
    mismatch = pinned["record_pinned_identity_mismatch_entries"][0]
    assert mismatch["record_path"] == absolute_record_path
    assert mismatch["observed_raw_sha256"] == hashlib.sha256(launcher_raw).hexdigest()
    assert mismatch["observed_raw_bytes"] == len(launcher_raw)
    assert mismatch["exclusion_reason"] == ("raw_pinned_wheel_console_script_not_in_controlled_path.v1")

    for alias_suffix in ("::$DATA", ".", " "):
        aliased = record_raw.replace(
            absolute_record_path.encode("utf-8"),
            f"{absolute_record_path}{alias_suffix}".encode("utf-8"),
        )
        with pytest.raises(ValueError, match="invalid component|not canonical"):
            protocol_v2._verify_installed_distribution_record_entries_v2(
                aliased,
                site_packages_root=site_root,
                environment_root=environment_root,
                distribution_name="wheel",
                normalized_distribution_name="wheel",
                inside_entries_preverified=False,
            )

    outside = tmp_path / "outside" / "wheel.exe"
    outside.parent.mkdir()
    outside.write_bytes(launcher_raw)
    escaped = record_raw.replace(absolute_record_path.encode("utf-8"), outside.resolve().as_posix().encode("utf-8"))
    (dist_info / "RECORD").write_bytes(escaped)
    with pytest.raises(ValueError, match="escaped the frozen Python environment"):
        protocol_v2._verify_installed_distribution_record_entries_v2(
            escaped,
            site_packages_root=site_root,
            environment_root=environment_root,
            distribution_name="wheel",
            normalized_distribution_name="wheel",
            inside_entries_preverified=False,
        )


def test_python_home_top_level_tree_pins_files_directories_and_refuses_bytecode(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "python-env"
    (environment_root / "Lib").mkdir(parents=True)
    (environment_root / "DLLs").mkdir()
    (environment_root / "python.exe").write_bytes(b"python executable")
    (environment_root / "vcruntime140.dll").write_bytes(b"runtime dll")

    frozen = protocol_v2._observe_python_home_top_level_tree(
        environment_root=environment_root,
    )

    assert frozen["directories"] == ["DLLs", "Lib"]
    assert [row["path"] for row in frozen["entries"]] == ["python.exe", "vcruntime140.dll"]
    protocol_v2._validate_python_home_top_level_tree(
        frozen,
        expected_root=str(environment_root.resolve()),
    )
    (environment_root / "sentinel.dll").write_bytes(b"shadow")
    observed = protocol_v2._observe_python_home_top_level_tree(
        environment_root=environment_root,
    )
    assert observed["artifact_id"] != frozen["artifact_id"]
    (environment_root / "SENTINEL.PYC").write_bytes(b"sourceless")
    with pytest.raises(ValueError, match="refuses sourceless adjacent bytecode"):
        protocol_v2._observe_python_home_top_level_tree(
            environment_root=environment_root,
        )


def test_distribution_inventory_validator_accepts_only_exact_inert_wheel_launcher_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = (tmp_path / "python-env").resolve()
    launcher = environment_root / "Scripts" / "wheel.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"conda wheel launcher")
    launcher_record_path = launcher.as_posix()
    row = {
        "normalized_name": "wheel",
        "distribution_name": "wheel",
        "version": "0.47.0",
        "dist_info_relative_path": "wheel-0.47.0.dist-info",
        "record_relative_path": "wheel-0.47.0.dist-info/RECORD",
        "record_raw_sha256": hashlib.sha256(b"record").hexdigest(),
        "record_raw_bytes": len(b"record"),
        "record_entry_count": 3,
        "record_hashed_entry_count": 2,
        "record_unhashed_pyc_entry_count": 0,
        "record_hashed_site_packages_entry_count": 1,
        "record_hashed_environment_external_entry_count": 0,
        "record_hashed_absolute_environment_entry_count": 0,
        "record_hashed_absolute_environment_entries": [],
        "record_hashed_absent_outside_environment_entry_count": 0,
        "record_absent_outside_environment_entries": [],
        "record_hashed_pinned_identity_mismatch_entry_count": 1,
        "record_pinned_identity_mismatch_entries": [
            {
                "record_path": launcher_record_path,
                "resolved_target": str(launcher),
                "declared_raw_sha256": hashlib.sha256(b"pip wheel launcher").hexdigest(),
                "declared_raw_bytes": len(b"pip wheel launcher"),
                "observed_raw_sha256": hashlib.sha256(b"conda wheel launcher").hexdigest(),
                "observed_raw_bytes": len(b"conda wheel launcher"),
                "target_raw_pinned": True,
                "child_import_or_controlled_path_reachable": False,
                "exclusion_reason": ("raw_pinned_wheel_console_script_not_in_controlled_path.v1"),
            }
        ],
        "record_in_environment_hashed_entries_verified_or_explicitly_pinned": True,
        "record_absent_outside_environment_entries_attested": True,
    }

    protocol_v2._validate_distribution_inventory_row(
        row,
        index=0,
        environment_root=str(environment_root),
    )
    escaped = copy.deepcopy(row)
    escaped["record_pinned_identity_mismatch_entries"][0]["record_path"] = (
        (tmp_path / "outside" / "wheel.exe").resolve().as_posix()
    )
    with pytest.raises(ValueError, match="only the exact wheel console-script mismatch"):
        protocol_v2._validate_distribution_inventory_row(
            escaped,
            index=0,
            environment_root=str(environment_root),
        )
    ads_alias = copy.deepcopy(row)
    ads_alias["record_pinned_identity_mismatch_entries"][0]["record_path"] += "::$DATA"
    with pytest.raises(ValueError, match="only the exact wheel console-script mismatch"):
        protocol_v2._validate_distribution_inventory_row(
            ads_alias,
            index=0,
            environment_root=str(environment_root),
        )


def test_site_packages_coverage_joins_record_owned_files_and_raw_pins_strays(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "python-env"
    site_root = environment_root / "Lib" / "site-packages"
    dist_info = site_root / "demo-1.0.dist-info"
    cache = site_root / "__pycache__"
    dist_info.mkdir(parents=True)
    cache.mkdir()
    owned_raw = b"owned = True\n"
    (site_root / "owned.py").write_bytes(owned_raw)
    (site_root / "stray_shadow.py").write_bytes(b"shadow = True\n")
    (site_root / "startup.pth").write_bytes(b"import startup\n")
    (cache / "owned.cpython-311.pyc").write_bytes(b"bytecode")
    (cache / "metadata.json").write_bytes(b'{"raw_pinned":true}\n')
    encoded_digest = base64.urlsafe_b64encode(hashlib.sha256(owned_raw).digest()).rstrip(b"=")
    record_raw = (
        b"owned.py,sha256="
        + encoded_digest
        + b","
        + str(len(owned_raw)).encode("ascii")
        + b"\ndemo-1.0.dist-info/RECORD,,\n"
    )
    (dist_info / "RECORD").write_bytes(record_raw)
    inventory = [
        {
            "normalized_name": "demo",
            "distribution_name": "demo",
            "version": "1.0",
            "dist_info_relative_path": "demo-1.0.dist-info",
            "record_relative_path": "demo-1.0.dist-info/RECORD",
            "record_raw_sha256": hashlib.sha256(record_raw).hexdigest(),
            "record_raw_bytes": len(record_raw),
            "record_entry_count": 2,
            "record_hashed_entry_count": 1,
            "record_unhashed_pyc_entry_count": 0,
            "record_hashed_site_packages_entry_count": 1,
            "record_hashed_environment_external_entry_count": 0,
            "record_hashed_absolute_environment_entry_count": 0,
            "record_hashed_absolute_environment_entries": [],
            "record_hashed_absent_outside_environment_entry_count": 0,
            "record_absent_outside_environment_entries": [],
            "record_hashed_pinned_identity_mismatch_entry_count": 0,
            "record_pinned_identity_mismatch_entries": [],
            "record_in_environment_hashed_entries_verified_or_explicitly_pinned": True,
            "record_absent_outside_environment_entries_attested": True,
        }
    ]

    coverage = protocol_v2._observe_site_packages_coverage(
        site_packages_root=site_root,
        environment_root=environment_root,
        distribution_inventory=inventory,
    )

    assert coverage["total_regular_file_count"] == 6
    assert coverage["owned_hashed_record_file_count"] == 1
    assert coverage["record_self_file_count"] == 1
    assert coverage["excluded_bytecode_file_count"] == 1
    assert coverage["excluded_site_startup_file_count"] == 1
    assert coverage["unowned_regular_file_count"] == 2
    assert coverage["unowned_regular_files"] == [
        {
            "path": "__pycache__/metadata.json",
            "raw_sha256": hashlib.sha256(b'{"raw_pinned":true}\n').hexdigest(),
            "raw_bytes": len(b'{"raw_pinned":true}\n'),
        },
        {
            "path": "stray_shadow.py",
            "raw_sha256": hashlib.sha256(b"shadow = True\n").hexdigest(),
            "raw_bytes": len(b"shadow = True\n"),
        },
    ]
    (site_root / "json.pyc").write_bytes(b"sourceless shadow")
    with pytest.raises(ValueError, match="adjacent bytecode outside __pycache__"):
        protocol_v2._observe_site_packages_coverage(
            site_packages_root=site_root,
            environment_root=environment_root,
            distribution_inventory=inventory,
        )


def _json_raw(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _build_receipt(
    frozen: _FrozenProtocol,
    observation: _GitHubObservation,
    *,
    observed_at_utc: str = _OBSERVED_AT,
) -> Mapping[str, object]:
    return protocol_v2._build_relationship_condition_reader_qualification_public_anchor_receipt_v2(
        execution_protocol_payload=frozen.payload,
        execution_protocol_raw=frozen.raw,
        expected_execution_protocol_id=frozen.protocol_id,
        expected_execution_root=frozen.execution_root,
        gist_id=_GIST_ID,
        observed_at_utc=observed_at_utc,
        base_gist_api_raw=_json_raw(observation.base),
        commits_api_raw=_json_raw(observation.commits),
        commits_reobservation_api_raw=_json_raw(observation.commits_reobservation),
        revision_api_raw=_json_raw(observation.revision),
        base_api_protocol_raw=observation.base_api_protocol_raw,
        revision_protocol_raw=observation.revision_protocol_raw,
    )


def test_observer_accepts_real_endpoint_shapes_drift_and_additive_fields(
    frozen_protocol: _FrozenProtocol,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    assert "history" not in observation.base
    assert "history" not in observation.revision
    base_url = f"https://api.github.com/gists/{_GIST_ID}"
    canonical_revision_raw_url = (
        f"https://gist.githubusercontent.com/ronaldzgithub/{_GIST_ID}/raw/{_VERSION}/{protocol_v2.V2_GIST_FILENAME}"
    )
    routes = {
        base_url: _json_raw(observation.base),
        f"{base_url}/commits?per_page=2&page=1": _json_raw(observation.commits),
        f"{base_url}/{_VERSION}": _json_raw(observation.revision),
        str(observation.base["files"][protocol_v2.V2_GIST_FILENAME]["raw_url"]): (frozen_protocol.raw),
        canonical_revision_raw_url: frozen_protocol.raw,
    }
    calls: list[str] = []

    def fetcher(*, url: str, max_bytes: int) -> bytes:
        calls.append(url)
        raw = routes[url]
        assert len(raw) <= max_bytes
        return raw

    def clock() -> dt.datetime:
        calls.append("observer-clock")
        return dt.datetime.strptime(
            _OBSERVED_AT,
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=dt.timezone.utc)

    receipt = protocol_v2._observe_relationship_condition_reader_qualification_public_anchor_v2_with_fetcher(
        execution_protocol_payload=frozen_protocol.payload,
        execution_protocol_raw=frozen_protocol.raw,
        expected_execution_protocol_id=frozen_protocol.protocol_id,
        expected_execution_root=frozen_protocol.execution_root,
        gist_id=_GIST_ID,
        fetcher=fetcher,
        clock=clock,
    )

    assert calls == [
        base_url,
        f"{base_url}/commits?per_page=2&page=1",
        f"{base_url}/{_VERSION}",
        str(observation.base["files"][protocol_v2.V2_GIST_FILENAME]["raw_url"]),
        canonical_revision_raw_url,
        f"{base_url}/commits?per_page=2&page=1",
        "observer-clock",
    ]
    assert receipt["created_at"] == _CREATED_AT
    assert receipt["updated_at"] == _UPDATED_AT
    assert receipt["sole_committed_at"] == _COMMITTED_AT
    assert receipt["created_equals_updated_required"] is False
    assert receipt["timestamp_fields_format_validated_only"] is True
    assert receipt["timestamp_order_used_as_revision_authority"] is False
    assert receipt["observed_at_caller_supplied"] is False
    assert receipt["observed_at_recorded_after_final_commits_get"] is True
    assert receipt["sole_revision_endpoint_bound"] is True
    assert receipt["sole_commit_version_bound_to_revision_endpoint"] is True
    assert receipt["sole_commit_timestamp_source"] == "dedicated_commits_endpoint"
    assert receipt["base_history_used_as_revision_authority"] is False
    assert receipt["api_file_raw_url_revision_used_as_commit_authority"] is False
    assert receipt["commits_reobserved_after_raw_gets"] is True
    assert receipt["initial_and_final_sole_commit_identity_match"] is True
    assert receipt["observation_linearization_point"] == "final_commits_reobservation"
    assert receipt["base_api_raw_url"].split("/")[-2] == _BASE_CONTENT_REVISION
    assert receipt["revision_api_file_raw_url"].split("/")[-2] == _REVISION_CONTENT_REVISION
    assert receipt["canonical_revision_raw_url"].split("/")[-2] == _VERSION


@pytest.mark.parametrize("commit_count", [0, 2])
def test_receipt_rejects_zero_or_second_revision(
    frozen_protocol: _FrozenProtocol,
    commit_count: int,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    sole_commit = copy.deepcopy(observation.commits[0])
    observation.commits = [] if commit_count == 0 else [sole_commit, copy.deepcopy(sole_commit)]

    with pytest.raises(ValueError, match="exactly one item"):
        _build_receipt(frozen_protocol, observation)


def test_receipt_rejects_revision_created_after_initial_commits_get(
    frozen_protocol: _FrozenProtocol,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    second_commit = copy.deepcopy(observation.commits_reobservation[0])
    assert isinstance(second_commit, dict)
    second_commit["version"] = _OTHER_VERSION
    second_commit["url"] = f"https://api.github.com/gists/{_GIST_ID}/{_OTHER_VERSION}"
    observation.commits_reobservation = [
        second_commit,
        *observation.commits_reobservation,
    ]

    with pytest.raises(ValueError, match="final commits reobservation requires exactly one item"):
        _build_receipt(frozen_protocol, observation)


@pytest.mark.parametrize(
    "case",
    ["commit_version", "commit_url", "revision_url"],
)
def test_receipt_rejects_cross_endpoint_version_or_url_drift(
    frozen_protocol: _FrozenProtocol,
    case: str,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    commit = observation.commits[0]
    assert isinstance(commit, dict)
    base_url = f"https://api.github.com/gists/{_GIST_ID}"
    if case == "commit_version":
        commit["version"] = _OTHER_VERSION
    elif case == "commit_url":
        commit["url"] = f"{base_url}/{_OTHER_VERSION}"
    else:
        observation.revision["url"] = f"{base_url}/{_OTHER_VERSION}"

    with pytest.raises(ValueError, match="lineage"):
        _build_receipt(frozen_protocol, observation)


@pytest.mark.parametrize("endpoint", ["base", "revision"])
def test_receipt_rejects_wrong_raw_protocol_bytes(
    frozen_protocol: _FrozenProtocol,
    endpoint: str,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    if endpoint == "base":
        observation.base_api_protocol_raw = b"{}\n"
    else:
        observation.revision_protocol_raw = b"{}\n"

    with pytest.raises(ValueError, match="raw bytes differ"):
        _build_receipt(frozen_protocol, observation)


def test_receipt_rejects_multiple_files(frozen_protocol: _FrozenProtocol) -> None:
    observation = _github_observation(frozen_protocol.raw)
    files = observation.base["files"]
    assert isinstance(files, dict)
    files["unexpected.json"] = copy.deepcopy(files[protocol_v2.V2_GIST_FILENAME])

    with pytest.raises(ValueError, match="sole exact V2 filename"):
        _build_receipt(frozen_protocol, observation)


@pytest.mark.parametrize("endpoint", ["base", "revision"])
def test_receipt_rejects_nonpublic_endpoint(
    frozen_protocol: _FrozenProtocol,
    endpoint: str,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    getattr(observation, endpoint)["public"] = False

    with pytest.raises(ValueError, match="must both be public"):
        _build_receipt(frozen_protocol, observation)


@pytest.mark.parametrize(
    "location",
    ["base_response", "revision_response", "base_file", "revision_file"],
)
def test_receipt_rejects_truncated_response_or_file(
    frozen_protocol: _FrozenProtocol,
    location: str,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    if location == "base_response":
        observation.base["truncated"] = True
    elif location == "revision_response":
        observation.revision["truncated"] = True
    else:
        endpoint = observation.base if location == "base_file" else observation.revision
        files = endpoint["files"]
        assert isinstance(files, dict)
        file_row = files[protocol_v2.V2_GIST_FILENAME]
        assert isinstance(file_row, dict)
        file_row["truncated"] = True

    with pytest.raises(ValueError, match="must not be truncated"):
        _build_receipt(frozen_protocol, observation)


@pytest.mark.parametrize("endpoint", ["base", "revision"])
def test_receipt_rejects_float_typed_file_size(
    frozen_protocol: _FrozenProtocol,
    endpoint: str,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    files = getattr(observation, endpoint)["files"]
    assert isinstance(files, dict)
    file_row = files[protocol_v2.V2_GIST_FILENAME]
    assert isinstance(file_row, dict)
    file_row["size"] = float(len(frozen_protocol.raw))

    with pytest.raises(ValueError, match="must be a positive integer"):
        _build_receipt(frozen_protocol, observation)


@pytest.mark.parametrize("endpoint", ["base", "revision"])
def test_receipt_records_timestamp_order_without_using_it_as_revision_evidence(
    frozen_protocol: _FrozenProtocol,
    endpoint: str,
) -> None:
    observation = _github_observation(frozen_protocol.raw)
    getattr(observation, endpoint)["updated_at"] = "2026-08-24T12:21:07Z"

    receipt = _build_receipt(
        frozen_protocol,
        observation,
        observed_at_utc="2026-08-24T12:21:06Z",
    )

    assert receipt["timestamp_fields_format_validated_only"] is True
    assert receipt["timestamp_order_used_as_revision_authority"] is False


@pytest.mark.parametrize(
    ("field_name", "wrongly_equal_value"),
    [
        ("public", 1),
        ("created_equals_updated_required", 0),
        ("commits_page_item_count", 1.0),
        ("model_output_count_at_observation", False),
    ],
)
def test_receipt_rejects_bool_and_integer_type_confusion(
    frozen_protocol: _FrozenProtocol,
    field_name: str,
    wrongly_equal_value: object,
) -> None:
    receipt = dict(
        _build_receipt(
            frozen_protocol,
            _github_observation(frozen_protocol.raw),
        )
    )
    receipt[field_name] = wrongly_equal_value
    del receipt["artifact_id"]
    receipt["artifact_id"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()

    with pytest.raises(ValueError):
        protocol_v2._validate_anchor_receipt_shape(receipt)


def test_protocol_rejects_source_manifest_without_v2_composition_sources(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_path = (
        pathlib.Path(__file__).parents[1]
        / "src"
        / "lifeform_evolution"
        / "protocols"
        / "relationship_condition_reader_qualification_execution_v1.json"
    )
    v1_payload = strict_json_loads(v1_path.read_bytes(), max_bytes=8_000_000)
    assert isinstance(v1_payload, dict)
    v1_runtime = v1_payload["runtime_identity"]
    assert isinstance(v1_runtime, dict)
    import_runtime = _synthetic_import_runtime(v1_runtime)
    monkeypatch.setattr(
        protocol_v2,
        "build_relationship_condition_reader_qualification_import_runtime_identity_v2",
        lambda *, base_runtime_identity=None: copy.deepcopy(import_runtime),
    )
    repository_runtime = _synthetic_repository_runtime_coverage(v1_payload["execution_source_tree"])

    with pytest.raises(ValueError, match="omits V2 composition sources"):
        protocol_v2.build_relationship_condition_reader_qualification_execution_protocol_v2(
            preflight_binding=v1_payload["qualification_preflight"],
            source_tree_manifest=v1_payload["execution_source_tree"],
            repository_runtime_coverage=repository_runtime,
            bge_snapshot_tree_manifest=v1_payload["bge_snapshot_tree"],
            runtime_identity=v1_payload["runtime_identity"],
            proposed_execution_root=_host_test_execution_root(tmp_path),
            anchor_receipt_relative_path="artifacts/relationship_lab/v2-anchor-test.json",
        )


@pytest.mark.parametrize(
    ("section", "field_name", "wrongly_equal_value"),
    [
        ("external_public_anchor", "required", 1),
        ("external_public_anchor", "commits_get_count", 2.0),
        ("retired_predecessor", "can_authorize_v2", 0),
        ("process_firewall", "predictor_process_count", 2.0),
    ],
)
def test_protocol_rejects_json_bool_and_number_type_confusion(
    frozen_protocol: _FrozenProtocol,
    section: str,
    field_name: str,
    wrongly_equal_value: object,
) -> None:
    payload = copy.deepcopy(frozen_protocol.payload)
    section_payload = payload[section]
    assert isinstance(section_payload, dict)
    section_payload[field_name] = wrongly_equal_value

    with pytest.raises(ValueError, match="drifted"):
        protocol_v2.relationship_condition_reader_qualification_execution_protocol_id_v2(payload)


def test_receipt_validator_requires_external_artifact_id(
    frozen_protocol: _FrozenProtocol,
) -> None:
    receipt = _build_receipt(
        frozen_protocol,
        _github_observation(frozen_protocol.raw),
    )
    artifact_id = str(receipt["artifact_id"])

    assert (
        protocol_v2.validate_relationship_condition_reader_qualification_public_anchor_receipt_v2(
            receipt,
            expected_receipt_artifact_id=artifact_id,
            execution_protocol_payload=frozen_protocol.payload,
            execution_protocol_raw=frozen_protocol.raw,
            expected_execution_protocol_id=frozen_protocol.protocol_id,
            expected_execution_root=frozen_protocol.execution_root,
        )
        == artifact_id
    )
    with pytest.raises(ValueError, match="external expected artifact id"):
        protocol_v2.validate_relationship_condition_reader_qualification_public_anchor_receipt_v2(
            receipt,
            expected_receipt_artifact_id="0" * 64,
            execution_protocol_payload=frozen_protocol.payload,
            execution_protocol_raw=frozen_protocol.raw,
            expected_execution_protocol_id=frozen_protocol.protocol_id,
            expected_execution_root=frozen_protocol.execution_root,
        )


def test_v2_integrity_guard_reuses_frozen_domains_and_chains_v2_id(
    frozen_protocol: _FrozenProtocol,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = frozen_protocol.payload["execution_source_tree"]
    bge = frozen_protocol.payload["bge_snapshot_tree"]
    runtime = frozen_protocol.payload["runtime_identity"]
    monkeypatch.setattr(
        protocol_v2.v1,
        "build_relationship_condition_reader_execution_source_tree_manifest",
        lambda *, repository_root, execution_cli_relative_path=None: source,
    )
    monkeypatch.setattr(
        protocol_v2.v1,
        "build_bge_m3_snapshot_tree_manifest",
        lambda *, snapshot_root: bge,
    )
    monkeypatch.setattr(
        protocol_v2,
        "build_relationship_condition_reader_qualification_import_runtime_identity_v2",
        lambda *, base_runtime_identity=None: runtime,
    )
    guard = protocol_v2.relationship_condition_reader_qualification_integrity_guard_v2(
        execution_protocol=frozen_protocol.payload,
        expected_execution_protocol_id=frozen_protocol.protocol_id,
        repository_root=tmp_path / "repository",
        bge_snapshot_root=tmp_path / "bge",
    )

    first = guard(
        phase="post_anchor_pre_execution",
        previous_integrity_receipt_artifact_id=None,
    )
    second = guard(
        phase="pre_prediction_child_1",
        previous_integrity_receipt_artifact_id=str(first["artifact_id"]),
    )

    assert first["schema_version"] == (protocol_v2.v1.RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION)
    assert first["execution_protocol_id"] == frozen_protocol.protocol_id
    assert first["phase_ordinal"] == 0
    assert second["execution_protocol_id"] == frozen_protocol.protocol_id
    assert second["phase_ordinal"] == 1
    assert second["previous_integrity_receipt_artifact_id"] == first["artifact_id"]
    assert second["source_tree_artifact_id"] == source["artifact_id"]
    assert second["bge_snapshot_tree_artifact_id"] == bge["artifact_id"]
    assert second["runtime_identity_artifact_id"] == runtime["artifact_id"]


def test_v2_integrity_guard_reobserves_and_rejects_runtime_drift(
    frozen_protocol: _FrozenProtocol,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source = frozen_protocol.payload["execution_source_tree"]
    bge = frozen_protocol.payload["bge_snapshot_tree"]
    monkeypatch.setattr(
        protocol_v2.v1,
        "build_relationship_condition_reader_execution_source_tree_manifest",
        lambda *, repository_root, execution_cli_relative_path=None: source,
    )
    monkeypatch.setattr(
        protocol_v2.v1,
        "build_bge_m3_snapshot_tree_manifest",
        lambda *, snapshot_root: bge,
    )
    guard = protocol_v2.relationship_condition_reader_qualification_integrity_guard_v2(
        execution_protocol=frozen_protocol.payload,
        expected_execution_protocol_id=frozen_protocol.protocol_id,
        repository_root=tmp_path / "repository",
        bge_snapshot_root=tmp_path / "bge",
    )
    drifted = copy.deepcopy(frozen_protocol.payload["runtime_identity"])
    assert isinstance(drifted, dict)
    coverage = drifted["site_packages_coverage"]
    assert isinstance(coverage, dict)
    unowned = coverage["unowned_regular_files"]
    assert isinstance(unowned, list)
    unowned.append(
        {
            "path": "stray_shadow.py",
            "raw_sha256": hashlib.sha256(b"shadow").hexdigest(),
            "raw_bytes": len(b"shadow"),
        }
    )
    coverage["unowned_regular_file_count"] = int(coverage["unowned_regular_file_count"]) + 1
    coverage["total_regular_file_count"] = int(coverage["total_regular_file_count"]) + 1
    _rehash_artifact(coverage)
    _rehash_artifact(drifted)
    protocol_v2.validate_relationship_condition_reader_qualification_import_runtime_identity_v2(drifted)
    monkeypatch.setattr(
        protocol_v2,
        "build_relationship_condition_reader_qualification_import_runtime_identity_v2",
        lambda: drifted,
    )

    with pytest.raises(ValueError, match="runtime drift"):
        guard(
            phase="post_anchor_pre_execution",
            previous_integrity_receipt_artifact_id=None,
        )


def test_retired_v1_predecessor_cannot_authorize_v2(
    frozen_protocol: _FrozenProtocol,
) -> None:
    retired = protocol_v2.retired_relationship_condition_reader_qualification_v1_predecessor()
    assert retired["retired"] is True
    assert retired["can_authorize_v2"] is False
    assert retired["anchor_receipt_created"] is False
    assert retired["qualification_execution_started"] is False
    assert retired["execution_protocol_id"] != frozen_protocol.protocol_id

    with pytest.raises(ValueError, match="external expected id"):
        protocol_v2.validate_relationship_condition_reader_qualification_execution_protocol_v2(
            frozen_protocol.payload,
            expected_protocol_id=str(retired["execution_protocol_id"]),
        )

    tampered = copy.deepcopy(frozen_protocol.payload)
    retired_lineage = tampered["retired_predecessor"]
    assert isinstance(retired_lineage, dict)
    retired_lineage["can_authorize_v2"] = True
    with pytest.raises(ValueError, match="retired V1 predecessor lineage drifted"):
        protocol_v2.relationship_condition_reader_qualification_execution_protocol_id_v2(tampered)


def test_production_transport_uses_only_fixed_headers_and_disables_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"https://api.github.com/gists/{_GIST_ID}"
    captured: dict[str, object] = {}

    class Response:
        status = 200
        headers = {"Content-Length": "4"}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def geturl(self) -> str:
            return url

        def read(self, size: int) -> bytes:
            assert size == 5
            return b"body"

    class Opener:
        def open(
            self,
            request: urllib.request.Request,
            *,
            timeout: int,
        ) -> Response:
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    def build_opener(*handlers: object) -> Opener:
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(protocol_v2.urllib.request, "build_opener", build_opener)

    assert (
        protocol_v2._unauthenticated_github_https_get(
            url=url,
            max_bytes=4,
            timeout_seconds=17,
        )
        == b"body"
    )
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.get_method() == "GET"
    assert {name.lower(): value for name, value in request.header_items()} == {
        "accept": "application/vnd.github+json",
        "user-agent": "VolvenceQualificationAnchorV2/1.0",
        "x-github-api-version": "2026-03-10",
    }
    assert captured["timeout"] == 17
    handlers = captured["handlers"]
    assert isinstance(handlers, tuple)
    proxy = next(value for value in handlers if isinstance(value, urllib.request.ProxyHandler))
    assert proxy.proxies == {}
    assert any(isinstance(value, protocol_v2._RejectRedirectHandler) for value in handlers)
    assert any(isinstance(value, urllib.request.HTTPSHandler) for value in handlers)


def test_production_transport_redirect_handler_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="refuses HTTP redirects"):
        protocol_v2._RejectRedirectHandler().redirect_request(
            object(),
            object(),
            302,
            "Found",
            object(),
            "https://example.invalid/redirected",
        )
