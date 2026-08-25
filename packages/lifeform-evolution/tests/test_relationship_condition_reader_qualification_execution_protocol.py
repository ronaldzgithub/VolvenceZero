from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import pathlib

import pytest

from volvence_zero.canonical_json import canonical_json_bytes

import lifeform_evolution.relationship_condition_reader_qualification_execution_protocol as protocol_module
from lifeform_evolution.relationship_condition_reader_qualification_execution_protocol import (
    BGE_M3_MODEL_REVISION,
    DEFAULT_EXECUTION_CLI_RELATIVE_PATH,
    RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION,
    build_bge_m3_snapshot_tree_manifest,
    build_relationship_condition_reader_execution_preflight_binding,
    build_relationship_condition_reader_execution_source_tree_manifest,
    build_relationship_condition_reader_qualification_execution_protocol,
    build_relationship_condition_reader_qualification_public_anchor_receipt,
    relationship_condition_reader_qualification_integrity_guard,
    relationship_condition_reader_qualification_execution_protocol_id,
    validate_bge_m3_snapshot_tree_manifest,
    validate_relationship_condition_reader_execution_preflight_binding,
    validate_relationship_condition_reader_execution_source_tree_manifest,
    validate_relationship_condition_reader_qualification_execution_protocol,
    validate_relationship_condition_reader_qualification_public_anchor_receipt,
    validate_relationship_condition_reader_qualification_runtime_identity,
)


_BGE_PATHS = (
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_PREFLIGHT_SCHEMAS = {
    "protocol.json": "relationship-condition-reader-qualification-protocol.v1",
    "public/predictor_request.json": ("relationship-condition-reader-qualification-predictor-request.v1"),
    "public/public_corpus.json": ("relationship-condition-reader-qualification-public-corpus.v1"),
    "public/publication_request.json": ("relationship-condition-reader-qualification-publication-request.v1"),
    "sealed/challenge_labels.json": ("relationship-condition-reader-qualification-challenge-labels.v1"),
    "sealed/condition_training_labels.json": ("relationship-condition-reader-qualification-training-labels.v1"),
    "sealed/group_split.json": ("relationship-condition-reader-qualification-group-split.v1"),
}


def _artifact(core: dict[str, object]) -> dict[str, object]:
    return {**core, "artifact_id": hashlib.sha256(canonical_json_bytes(core)).hexdigest()}


def _reartifact(payload: dict[str, object]) -> dict[str, object]:
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    return _artifact(core)


def _write_artifact(path: pathlib.Path, payload: dict[str, object]) -> bytes:
    raw = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _make_repository(root: pathlib.Path) -> pathlib.Path:
    files = {
        "packages/alpha/src/alpha/__init__.py": b"VALUE = 1\n",
        "packages/alpha/src/alpha/worker.py": b"def work():\n    return 2\n",
        "packages/beta/src/beta.py": b"BETA = True\r\n",
        DEFAULT_EXECUTION_CLI_RELATIVE_PATH: b"#!/usr/bin/env python3\n",
    }
    for relative_path, raw in files.items():
        path = root / pathlib.PurePosixPath(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root


def _make_bge_snapshot(
    root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    for index, relative_path in enumerate(_BGE_PATHS):
        path = root / pathlib.PurePosixPath(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"resource-{index}\n".encode())
    weight_raw = (root / "pytorch_model.bin").read_bytes()
    monkeypatch.setattr(
        protocol_module,
        "BGE_M3_WEIGHTS_SHA256",
        hashlib.sha256(weight_raw).hexdigest(),
    )
    return root


def _make_preflight(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    protocol_payload = {
        "schema_version": _PREFLIGHT_SCHEMAS["protocol.json"],
        "evidence_role": "test-fixture-only",
    }
    protocol_raw = json.dumps(protocol_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    protocol_path = root / "protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes(protocol_raw)
    qualification_protocol_id = hashlib.sha256(canonical_json_bytes(protocol_payload)).hexdigest()

    receipts: list[dict[str, object]] = [
        {
            "path": "protocol.json",
            "raw_sha256": hashlib.sha256(protocol_raw).hexdigest(),
            "raw_bytes": len(protocol_raw),
            "artifact_id": None,
        }
    ]
    for relative_path, schema_version in _PREFLIGHT_SCHEMAS.items():
        if relative_path == "protocol.json":
            continue
        payload = _artifact(
            {
                "schema_version": schema_version,
                "protocol_id": qualification_protocol_id,
                "fixture_path": relative_path,
            }
        )
        raw = _write_artifact(root / pathlib.PurePosixPath(relative_path), payload)
        receipts.append(
            {
                "path": relative_path,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
                "artifact_id": payload["artifact_id"],
            }
        )
    manifest = _artifact(
        {
            "schema_version": "relationship-condition-reader-qualification-preflight-manifest.v1",
            "protocol_id": qualification_protocol_id,
            "files": receipts,
            "file_count": len(receipts),
            "model_output_count": 0,
            "external_public_anchor_created": False,
            "qualification_execution_authorized": False,
        }
    )
    _write_artifact(root / "manifest.json", manifest)
    return root, qualification_protocol_id


def _file_pin(path: str, marker: str) -> dict[str, object]:
    return {
        "path": path,
        "raw_sha256": hashlib.sha256(marker.encode()).hexdigest(),
        "raw_bytes": len(marker),
    }


def _runtime_identity() -> dict[str, object]:
    distributions = []
    versions = {
        "huggingface-hub": "1.16.0",
        "sentence-transformers": "5.6.0",
        "torch": "2.12.0+cu126",
        "transformers": "5.9.0",
    }
    for lookup_name in sorted(versions):
        normalized = lookup_name.replace("-", "_")
        distributions.append(
            {
                "lookup_name": lookup_name,
                "distribution_name": normalized,
                "version": versions[lookup_name],
                "dist_info_path": f"C:\\qualification\\site-packages\\{normalized}.dist-info",
                "metadata": _file_pin("METADATA", f"{lookup_name}-metadata"),
                "record": _file_pin("RECORD", f"{lookup_name}-record"),
                "wheel": _file_pin("WHEEL", f"{lookup_name}-wheel"),
                "record_entry_count": 10,
                "record_hashed_entry_count": 9,
                "record_hashed_entries_verified_at_freeze": True,
                "record_unhashed_non_pyc_paths": [f"{normalized}.dist-info/RECORD"],
            }
        )
    return _artifact(
        {
            "schema_version": RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION,
            "platform": "windows",
            "gpu": {
                "index": 0,
                "uuid": "GPU-test",
                "name": "NVIDIA GeForce RTX 4090",
                "pci_bus_id": "00000000:01:00.0",
                "driver_version": "560.94",
                "vbios_version": "test-vbios",
                "memory_total_mib": 24564,
                "compute_capability": "8.9",
                "nvidia_smi_cuda_max_version": "12.6",
                "nvidia_smi_binary": _file_pin("C:\\Windows\\System32\\nvidia-smi.exe", "smi"),
                "nvcuda_binary": _file_pin("C:\\Windows\\System32\\nvcuda.dll", "cuda"),
            },
            "python": {
                "implementation": "CPython",
                "version": "3.11.15",
                "version_full": "3.11.15 test build",
                "architecture": "AMD64",
                "pointer_bits": 64,
                "platform_version": "Windows-10-10.0.26200-SP0",
                "executable": "C:\\qualification\\python.exe",
                "executable_raw_sha256": hashlib.sha256(b"python").hexdigest(),
                "executable_raw_bytes": 6,
                "runtime_dlls": [
                    _file_pin("C:\\qualification\\python3.dll", "python3"),
                    _file_pin("C:\\qualification\\python311.dll", "python311"),
                ],
            },
            "distributions": distributions,
        }
    )


def _protocol_fixture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], pathlib.Path]:
    repository = _make_repository(tmp_path / "repository")
    snapshot = _make_bge_snapshot(tmp_path / "snapshot", monkeypatch)
    preflight, qualification_protocol_id = _make_preflight(tmp_path / "preflight")
    source_tree = build_relationship_condition_reader_execution_source_tree_manifest(repository_root=repository)
    bge_tree = build_bge_m3_snapshot_tree_manifest(snapshot_root=snapshot)
    preflight_binding = build_relationship_condition_reader_execution_preflight_binding(
        preflight_root=preflight,
        expected_qualification_protocol_id=qualification_protocol_id,
    )
    execution_root = tmp_path / "qualification-output"
    protocol = build_relationship_condition_reader_qualification_execution_protocol(
        preflight_binding=preflight_binding,
        source_tree_manifest=source_tree,
        bge_snapshot_tree_manifest=bge_tree,
        runtime_identity=_runtime_identity(),
        proposed_execution_root=execution_root,
        anchor_receipt_relative_path=("artifacts/relationship_lab/reader_qualification_execution_anchor.json"),
    )
    return dict(protocol), execution_root


def test_source_tree_manifest_reobserves_exact_raw_tree(tmp_path: pathlib.Path) -> None:
    repository = _make_repository(tmp_path / "repository")
    manifest = build_relationship_condition_reader_execution_source_tree_manifest(repository_root=repository)

    assert manifest["entry_count"] == 4
    assert [row["path"] for row in manifest["entries"]] == sorted(
        [row["path"] for row in manifest["entries"]],
        key=lambda value: value.encode("utf-8"),
    )
    assert (
        validate_relationship_condition_reader_execution_source_tree_manifest(
            manifest,
            repository_root=repository,
        )
        == manifest["artifact_id"]
    )

    extra = repository / "packages/alpha/src/alpha/extra.py"
    extra.write_bytes(b"EXTRA = True\n")
    with pytest.raises(ValueError, match="does not match"):
        validate_relationship_condition_reader_execution_source_tree_manifest(
            manifest,
            repository_root=repository,
        )


def test_source_tree_rejects_hardlinks_and_casefold_collision(tmp_path: pathlib.Path) -> None:
    repository = _make_repository(tmp_path / "repository")
    cli = repository / pathlib.PurePosixPath(DEFAULT_EXECUTION_CLI_RELATIVE_PATH)
    hardlink = repository / "cli-hardlink.txt"
    os.link(cli, hardlink)
    with pytest.raises(ValueError, match="hard link"):
        build_relationship_condition_reader_execution_source_tree_manifest(repository_root=repository)

    hardlink.unlink()
    manifest = dict(build_relationship_condition_reader_execution_source_tree_manifest(repository_root=repository))
    rows = [dict(row) for row in manifest["entries"]]
    duplicate = dict(rows[0])
    duplicate["path"] = str(duplicate["path"]).upper()
    rows.append(duplicate)
    rows.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    manifest["entries"] = rows
    manifest["entry_count"] = len(rows)
    manifest["total_raw_bytes"] = sum(int(row["raw_bytes"]) for row in rows)
    manifest = _reartifact(manifest)
    with pytest.raises(ValueError, match="casefold"):
        validate_relationship_condition_reader_execution_source_tree_manifest(manifest)


def test_source_tree_rejects_symlink_when_host_can_create_one(tmp_path: pathlib.Path) -> None:
    repository = _make_repository(tmp_path / "repository")
    target = repository / "outside.py"
    target.write_bytes(b"OUTSIDE = True\n")
    link = repository / "packages/alpha/src/alpha/link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("host does not permit creating symlinks")
    with pytest.raises(ValueError, match="symlink|reparse"):
        build_relationship_condition_reader_execution_source_tree_manifest(repository_root=repository)


def test_bge_snapshot_tree_rejects_drift_and_extra_resource(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _make_bge_snapshot(tmp_path / "snapshot", monkeypatch)
    manifest = build_bge_m3_snapshot_tree_manifest(snapshot_root=snapshot)

    assert manifest["entry_count"] == 11
    assert manifest["model_revision"] == BGE_M3_MODEL_REVISION
    assert (
        validate_bge_m3_snapshot_tree_manifest(
            manifest,
            snapshot_root=snapshot,
        )
        == manifest["artifact_id"]
    )

    (snapshot / "tokenizer.json").write_bytes(b"drifted\n")
    with pytest.raises(ValueError, match="does not match"):
        validate_bge_m3_snapshot_tree_manifest(manifest, snapshot_root=snapshot)
    (snapshot / "unexpected.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="file set mismatch"):
        build_bge_m3_snapshot_tree_manifest(snapshot_root=snapshot)


def test_preflight_binding_covers_all_eight_files_and_detects_tampering(
    tmp_path: pathlib.Path,
) -> None:
    preflight, qualification_protocol_id = _make_preflight(tmp_path / "preflight")
    binding = build_relationship_condition_reader_execution_preflight_binding(
        preflight_root=preflight,
        expected_qualification_protocol_id=qualification_protocol_id,
    )

    assert binding["file_count"] == 8
    assert {row["path"] for row in binding["files"]} == {
        "manifest.json",
        *_PREFLIGHT_SCHEMAS,
    }
    assert (
        validate_relationship_condition_reader_execution_preflight_binding(
            binding,
            preflight_root=preflight,
        )
        == binding["artifact_id"]
    )

    challenge_path = preflight / "sealed/challenge_labels.json"
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    challenge["fixture_path"] = "tampered"
    _write_artifact(challenge_path, _reartifact(challenge))
    with pytest.raises(ValueError, match="receipt identity mismatch"):
        build_relationship_condition_reader_execution_preflight_binding(
            preflight_root=preflight,
            expected_qualification_protocol_id=qualification_protocol_id,
        )


def test_execution_protocol_requires_external_id_and_honest_claims(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _execution_root = _protocol_fixture(tmp_path, monkeypatch)
    protocol_id = relationship_condition_reader_qualification_execution_protocol_id(protocol)

    assert (
        validate_relationship_condition_reader_qualification_execution_protocol(
            protocol,
            expected_protocol_id=protocol_id,
        )
        == protocol_id
    )
    with pytest.raises(ValueError, match="external expected id"):
        validate_relationship_condition_reader_qualification_execution_protocol(
            protocol,
            expected_protocol_id="0" * 64,
        )

    overclaim = copy.deepcopy(protocol)
    overclaim["claims"]["readable_product_effect"] = True
    with pytest.raises(ValueError, match="claim ceiling"):
        validate_relationship_condition_reader_qualification_execution_protocol(
            overclaim,
            expected_protocol_id=hashlib.sha256(canonical_json_bytes(overclaim)).hexdigest(),
        )

    unsafe = copy.deepcopy(protocol)
    unsafe["process_firewall"]["os_security_boundary"] = True
    with pytest.raises(ValueError, match="firewall"):
        validate_relationship_condition_reader_qualification_execution_protocol(
            unsafe,
            expected_protocol_id=hashlib.sha256(canonical_json_bytes(unsafe)).hexdigest(),
        )


def test_execution_protocol_requires_runtime_distribution_and_new_anchor_contract(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _execution_root = _protocol_fixture(tmp_path, monkeypatch)

    missing_distribution = copy.deepcopy(protocol)
    missing_distribution["runtime_identity"]["distributions"].pop()
    with pytest.raises(ValueError, match="four load-critical distributions"):
        relationship_condition_reader_qualification_execution_protocol_id(missing_distribution)

    old_anchor = copy.deepcopy(protocol)
    old_anchor["external_public_anchor"]["existing_product_horizon_anchor_accepted"] = True
    with pytest.raises(ValueError, match="public-anchor"):
        relationship_condition_reader_qualification_execution_protocol_id(old_anchor)


def test_public_anchor_requires_external_receipt_id_exact_first_revision_and_absent_root(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, execution_root = _protocol_fixture(tmp_path, monkeypatch)
    protocol_id = relationship_condition_reader_qualification_execution_protocol_id(protocol)
    protocol_raw = json.dumps(protocol, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    gist_id = "1" * 32
    history_version = "2" * 40
    filename = "relationship_condition_reader_qualification_execution_v1.json"
    revision_raw_url = f"https://gist.githubusercontent.com/ronaldzgithub/{gist_id}/raw/{history_version}/{filename}"
    receipt = build_relationship_condition_reader_qualification_public_anchor_receipt(
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=protocol_id,
        expected_execution_root=execution_root,
        gist_owner="ronaldzgithub",
        gist_id=gist_id,
        gist_url=f"https://gist.github.com/ronaldzgithub/{gist_id}",
        filename=filename,
        public=True,
        history_version=history_version,
        history_revision_count=1,
        first_revision=True,
        created_at="2026-08-24T12:00:00Z",
        updated_at="2026-08-24T12:00:00Z",
        api_raw_url=revision_raw_url,
        revision_raw_url=revision_raw_url,
        observation_transport="unauthenticated_github_rest_api_and_raw_http",
        observed_at_utc="2026-08-24T12:01:00Z",
        observed_protocol_raw=protocol_raw,
    )

    assert (
        validate_relationship_condition_reader_qualification_public_anchor_receipt(
            receipt,
            expected_receipt_artifact_id=str(receipt["artifact_id"]),
            execution_protocol_payload=protocol,
            execution_protocol_raw=protocol_raw,
            expected_execution_protocol_id=protocol_id,
            expected_execution_root=execution_root,
        )
        == receipt["artifact_id"]
    )

    with pytest.raises(ValueError, match="external expected artifact id"):
        validate_relationship_condition_reader_qualification_public_anchor_receipt(
            receipt,
            expected_receipt_artifact_id="0" * 64,
            execution_protocol_payload=protocol,
            execution_protocol_raw=protocol_raw,
            expected_execution_protocol_id=protocol_id,
            expected_execution_root=execution_root,
        )

    with pytest.raises(ValueError, match="exactly match protocol raw bytes"):
        build_relationship_condition_reader_qualification_public_anchor_receipt(
            execution_protocol_payload=protocol,
            execution_protocol_raw=protocol_raw,
            expected_execution_protocol_id=protocol_id,
            expected_execution_root=execution_root,
            gist_owner="ronaldzgithub",
            gist_id=gist_id,
            gist_url=f"https://gist.github.com/ronaldzgithub/{gist_id}",
            filename=filename,
            public=True,
            history_version=history_version,
            history_revision_count=1,
            first_revision=True,
            created_at="2026-08-24T12:00:00Z",
            updated_at="2026-08-24T12:00:00Z",
            api_raw_url=revision_raw_url,
            revision_raw_url=revision_raw_url,
            observation_transport=("unauthenticated_github_rest_api_and_raw_http"),
            observed_at_utc="2026-08-24T12:01:00Z",
            observed_protocol_raw=b"{}\n",
        )

    second_revision = copy.deepcopy(receipt)
    second_revision["history_revision_count"] = 2
    second_revision = _reartifact(second_revision)
    with pytest.raises(ValueError, match="exactly one first revision"):
        validate_relationship_condition_reader_qualification_public_anchor_receipt(
            second_revision,
            expected_receipt_artifact_id=str(second_revision["artifact_id"]),
            execution_protocol_payload=protocol,
            execution_protocol_raw=protocol_raw,
            expected_execution_protocol_id=protocol_id,
            expected_execution_root=execution_root,
        )

    execution_root.mkdir()
    with pytest.raises(FileExistsError, match="absent execution root"):
        validate_relationship_condition_reader_qualification_public_anchor_receipt(
            receipt,
            expected_receipt_artifact_id=str(receipt["artifact_id"]),
            execution_protocol_payload=protocol,
            execution_protocol_raw=protocol_raw,
            expected_execution_protocol_id=protocol_id,
            expected_execution_root=execution_root,
        )


def test_runtime_identity_requires_content_addressed_full_record_verification() -> None:
    runtime = _runtime_identity()

    assert validate_relationship_condition_reader_qualification_runtime_identity(runtime) == runtime["artifact_id"]

    incomplete = copy.deepcopy(runtime)
    incomplete["distributions"][0]["record_hashed_entries_verified_at_freeze"] = False
    incomplete = _reartifact(incomplete)
    with pytest.raises(ValueError, match="all hashed RECORD entries"):
        validate_relationship_condition_reader_qualification_runtime_identity(incomplete)

    extra_unhashed = copy.deepcopy(runtime)
    extra_unhashed["distributions"][0]["record_unhashed_non_pyc_paths"].append("package/data.bin")
    extra_unhashed = _reartifact(extra_unhashed)
    with pytest.raises(ValueError, match="only the RECORD file itself"):
        validate_relationship_condition_reader_qualification_runtime_identity(extra_unhashed)


def test_record_verification_hashes_prefix_confined_parent_entries(
    tmp_path: pathlib.Path,
) -> None:
    environment_root = tmp_path / "runtime"
    record_base = environment_root / "Lib/site-packages"
    package_file = record_base / "package/module.py"
    script_file = environment_root / "Scripts/tool.exe"
    package_file.parent.mkdir(parents=True)
    script_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"MODULE = True\n")
    script_file.write_bytes(b"tool-binary")

    def record_hash(path: pathlib.Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    record_raw = (
        f"package/module.py,sha256={record_hash(package_file)},{package_file.stat().st_size}\n"
        f"../../Scripts/tool.exe,sha256={record_hash(script_file)},{script_file.stat().st_size}\n"
        "demo.dist-info/RECORD,,\n"
    ).encode("utf-8")
    verification = protocol_module._verify_record_entries(
        record_raw,
        record_base=record_base,
        environment_root=environment_root,
        field_name="demo",
    )

    assert verification == {
        "record_entry_count": 3,
        "record_hashed_entry_count": 2,
        "record_unhashed_non_pyc_paths": ["demo.dist-info/RECORD"],
    }

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    escaped = (
        f"../../../outside.bin,sha256={record_hash(outside)},{outside.stat().st_size}\ndemo.dist-info/RECORD,,\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="within its declared root"):
        protocol_module._verify_record_entries(
            escaped,
            record_base=record_base,
            environment_root=environment_root,
            field_name="demo",
        )

    mismatched = (
        f"package/module.py,sha256={record_hash(script_file)},{package_file.stat().st_size}\ndemo.dist-info/RECORD,,\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="entry identity mismatch"):
        protocol_module._verify_record_entries(
            mismatched,
            record_base=record_base,
            environment_root=environment_root,
            field_name="demo",
        )


def test_integrity_guard_reobserves_all_frozen_domains_and_chains_receipts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _execution_root = _protocol_fixture(tmp_path, monkeypatch)
    protocol_id = relationship_condition_reader_qualification_execution_protocol_id(protocol)
    frozen_runtime = copy.deepcopy(protocol["runtime_identity"])
    monkeypatch.setattr(
        protocol_module,
        "build_relationship_condition_reader_qualification_runtime_identity",
        lambda: copy.deepcopy(frozen_runtime),
    )
    guard = relationship_condition_reader_qualification_integrity_guard(
        execution_protocol=protocol,
        expected_execution_protocol_id=protocol_id,
        repository_root=tmp_path / "repository",
        bge_snapshot_root=tmp_path / "snapshot",
    )
    phases = (
        "post_anchor_pre_execution",
        "pre_prediction_child_1",
        "post_prediction_child_1",
        "pre_prediction_child_2",
        "post_prediction_child_2",
        "pre_scorer",
        "post_scorer",
        "final_validation",
    )
    previous_id: str | None = None
    receipts = []
    for ordinal, phase in enumerate(phases):
        receipt = guard(
            phase=phase,
            previous_integrity_receipt_artifact_id=previous_id,
        )
        assert receipt["phase_ordinal"] == ordinal
        assert receipt["previous_integrity_receipt_artifact_id"] == previous_id
        assert receipt["source_tree_exact"] is True
        assert receipt["bge_snapshot_tree_exact"] is True
        assert receipt["runtime_identity_exact"] is True
        assert receipt["observer_model_or_cuda_execution_used"] is False
        receipts.append(receipt)
        previous_id = str(receipt["artifact_id"])

    assert len({receipt["artifact_id"] for receipt in receipts}) == len(phases)
    (tmp_path / "repository/packages/alpha/src/alpha/worker.py").write_bytes(b"def work():\n    return 999\n")
    with pytest.raises(ValueError, match="source-tree drift"):
        guard(
            phase="final_validation",
            previous_integrity_receipt_artifact_id=previous_id,
        )
