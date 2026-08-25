from __future__ import annotations

import hashlib
import importlib
import os
import pathlib
from types import ModuleType, SimpleNamespace

import pytest

from volvence_zero.canonical_json import canonical_json_bytes


_CLI_MODULE_NAME = "lifeform_evolution.relationship_condition_reader_qualification_execution_cli_v2"
_PROTOCOL_ID = "a" * 64
_RECEIPT_ID = "b" * 64


@pytest.fixture
def cli_module() -> ModuleType:
    return importlib.import_module(_CLI_MODULE_NAME)


def _remove_option(argv: list[str], option: str) -> list[str]:
    reduced = list(argv)
    index = reduced.index(option)
    del reduced[index : index + 2]
    return reduced


def test_help_exposes_five_commands_without_importing_execution_or_models(
    cli_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "_install_workspace_source_roots", lambda: ())

    def forbidden_lazy_import() -> object:
        raise AssertionError("parser/help must not import protocol or execution modules")

    monkeypatch.setattr(cli_module, "_load_v1_protocol_module", forbidden_lazy_import)
    monkeypatch.setattr(cli_module, "_load_v2_protocol_module", forbidden_lazy_import)
    monkeypatch.setattr(cli_module, "_load_v2_execution_module", forbidden_lazy_import)
    forbidden_before = {
        name
        for name in cli_module.sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "sentence_transformers"
        or name.startswith("sentence_transformers.")
        or "relationship_condition_reader_predictor" in name
        or "relationship_condition_reader_scorer" in name
        or "relationship_condition_reader_qualification_executor" in name
    }

    with pytest.raises(SystemExit) as help_exit:
        cli_module.main(["--help"])

    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "freeze-protocol" in help_text
    assert "show-protocol" in help_text
    assert "record-anchor" in help_text
    assert "validate-anchor" in help_text
    assert "execute" in help_text
    assert "--force" not in help_text
    forbidden_after = {
        name
        for name in cli_module.sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "sentence_transformers"
        or name.startswith("sentence_transformers.")
        or "relationship_condition_reader_predictor" in name
        or "relationship_condition_reader_scorer" in name
        or "relationship_condition_reader_qualification_executor" in name
    }
    assert forbidden_after == forbidden_before


def test_record_anchor_parser_has_no_caller_supplied_observation_escape_hatch(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    parser = cli_module._build_parser()
    protocol_path = tmp_path / "protocol.json"
    execution_root = tmp_path / "execution"
    argv = [
        "record-anchor",
        "--execution-protocol-path",
        str(protocol_path),
        "--expected-execution-protocol-id",
        _PROTOCOL_ID,
        "--gist-id",
        "c" * 32,
        "--expected-execution-root",
        str(execution_root),
        "--timeout-seconds",
        "17",
    ]

    parsed = parser.parse_args(argv)

    assert vars(parsed) == {
        "command": "record-anchor",
        "execution_protocol_path": protocol_path,
        "expected_execution_protocol_id": _PROTOCOL_ID,
        "gist_id": "c" * 32,
        "expected_execution_root": execution_root,
        "timeout_seconds": 17,
    }

    forbidden_options = (
        "--force",
        "--observed-at-utc",
        "--observed-protocol-raw-path",
        "--observation-json-path",
        "--local-observation-json",
        "--history-version",
        "--history-revision-count",
        "--committed-at",
        "--public",
        "--first-revision",
        "--created-at",
        "--updated-at",
        "--api-raw-url",
        "--revision-raw-url",
        "--raw-url",
        "--anchor-receipt-path",
    )
    for forbidden_option in forbidden_options:
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, forbidden_option, "forged-caller-value"])


def test_protocol_output_must_remain_outside_frozen_source_roots(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repository").resolve()
    source_root = repository / "packages" / "demo" / "src"
    source_root.mkdir(parents=True)
    artifact_path = repository / "artifacts" / "relationship_lab" / "protocol.json"
    artifact_path.parent.mkdir(parents=True)
    monkeypatch.setattr(cli_module, "_REPOSITORY_ROOT", repository)
    coverage = {"source_roots": ["packages/demo/src"]}

    cli_module._require_protocol_output_outside_frozen_source_roots(
        artifact_path,
        repository_runtime_coverage=coverage,
    )
    with pytest.raises(ValueError, match="protocol-to-coverage self-reference"):
        cli_module._require_protocol_output_outside_frozen_source_roots(
            source_root / "protocols" / "protocol.json",
            repository_runtime_coverage=coverage,
        )


def test_freeze_requires_publication_request_and_protocol_execution_roots_to_match(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_module = cli_module._load_v2_protocol_module()
    preflight_root = (tmp_path / "preflight").resolve()
    publication_path = preflight_root / "public" / "publication_request.json"
    publication_path.parent.mkdir(parents=True)
    frozen_execution_root = (tmp_path / "frozen-execution").resolve()
    publication_core = {
        "schema_version": "relationship-condition-reader-qualification-publication-request.v1",
        "proposed_execution_root": str(frozen_execution_root),
        "proposed_execution_root_exists_at_prepare": False,
    }
    publication = {
        **publication_core,
        "artifact_id": hashlib.sha256(canonical_json_bytes(publication_core)).hexdigest(),
    }
    publication_raw = canonical_json_bytes(publication) + b"\n"
    publication_path.write_bytes(publication_raw)
    preflight_binding = {
        "files": [
            {
                "path": "public/publication_request.json",
                "raw_sha256": hashlib.sha256(publication_raw).hexdigest(),
                "raw_bytes": len(publication_raw),
                "artifact_id": publication["artifact_id"],
            }
        ]
    }

    cli_module._require_preflight_publication_execution_root(
        protocol_module=protocol_module,
        preflight_root=preflight_root,
        preflight_binding=preflight_binding,
        proposed_execution_root=frozen_execution_root,
    )
    with pytest.raises(
        ValueError,
        match="publication request proposed execution root differs",
    ):
        cli_module._require_preflight_publication_execution_root(
            protocol_module=protocol_module,
            preflight_root=preflight_root,
            preflight_binding=preflight_binding,
            proposed_execution_root=(tmp_path / "different-execution").resolve(),
        )

    if os.name == "nt":
        case_variant_root = frozen_execution_root.with_name(
            frozen_execution_root.name.upper()
        )
        case_variant_core = {
            **publication_core,
            "proposed_execution_root": str(case_variant_root),
        }
        case_variant_publication = {
            **case_variant_core,
            "artifact_id": hashlib.sha256(
                canonical_json_bytes(case_variant_core)
            ).hexdigest(),
        }
        case_variant_raw = canonical_json_bytes(case_variant_publication) + b"\n"
        publication_path.write_bytes(case_variant_raw)
        preflight_binding = {
            "files": [
                {
                    "path": "public/publication_request.json",
                    "raw_sha256": hashlib.sha256(case_variant_raw).hexdigest(),
                    "raw_bytes": len(case_variant_raw),
                    "artifact_id": case_variant_publication["artifact_id"],
                }
            ]
        }
        with pytest.raises(
            ValueError,
            match="publication request proposed execution root differs",
        ):
            cli_module._require_preflight_publication_execution_root(
                protocol_module=protocol_module,
                preflight_root=preflight_root,
                preflight_binding=preflight_binding,
                proposed_execution_root=frozen_execution_root,
            )

    def forbidden_source_scan(**_kwargs: object) -> object:
        raise AssertionError("root mismatch must fail before the expensive source scan")

    fake_v1_protocol = SimpleNamespace(
        build_relationship_condition_reader_execution_preflight_binding=(
            lambda **_kwargs: preflight_binding
        ),
        build_relationship_condition_reader_execution_source_tree_manifest=forbidden_source_scan,
    )
    monkeypatch.setattr(cli_module, "_load_v1_protocol_module", lambda: fake_v1_protocol)
    monkeypatch.setattr(
        cli_module,
        "_load_v2_protocol_module",
        lambda: SimpleNamespace(
            canonical_relationship_condition_reader_qualification_execution_root_v2=(
                protocol_module.canonical_relationship_condition_reader_qualification_execution_root_v2
            )
        ),
    )
    monkeypatch.setattr(cli_module, "_assert_model_free", lambda: None)
    protocol_output_path = (tmp_path / "artifacts" / "protocol.json").resolve()
    protocol_output_path.parent.mkdir()
    freeze_args = SimpleNamespace(
        preflight_root=preflight_root,
        expected_qualification_protocol_id=_PROTOCOL_ID,
        bge_snapshot_root=(tmp_path / "bge").resolve(),
        proposed_execution_root=(tmp_path / "different-execution").resolve(),
        anchor_receipt_relative_path="artifacts/anchor.json",
        protocol_output_path=protocol_output_path,
    )
    with pytest.raises(
        ValueError,
        match="publication request proposed execution root differs",
    ):
        cli_module._freeze_protocol(freeze_args)
    assert not protocol_output_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows local-drive path contract")
def test_execution_root_rejects_windows_alias_and_invalid_component_forms(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    protocol_module = cli_module._load_v2_protocol_module()
    resolved = tmp_path.resolve()
    invalid_roots = (
        pathlib.Path(f"{resolved}:stream"),
        resolved.parent / "CON",
        resolved.parent / "trailing.",
        pathlib.Path(f"\\\\?\\{resolved}"),
    )

    for invalid_root in invalid_roots:
        with pytest.raises(ValueError):
            protocol_module.canonical_relationship_condition_reader_qualification_execution_root_v2(
                invalid_root,
                "test execution root",
            )


@pytest.mark.skipif(os.name != "nt", reason="Windows local-drive path contract")
def test_freeze_execution_root_rejects_existing_file_ancestor(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    parent_file = tmp_path.resolve() / "parent-file"
    parent_file.write_bytes(b"not a directory\n")

    with pytest.raises(ValueError, match="nearest existing ancestor must be a directory"):
        cli_module._require_absent_canonical_directory_target(
            parent_file / "child",
            "test execution root",
        )


def test_validate_and_execute_parser_require_both_external_ids_paths_and_root(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    parser = cli_module._build_parser()
    protocol_path = tmp_path / "protocol.json"
    receipt_path = tmp_path / "receipt.json"
    execution_root = tmp_path / "execution"
    shared = [
        "--execution-protocol-path",
        str(protocol_path),
        "--expected-execution-protocol-id",
        _PROTOCOL_ID,
        "--anchor-receipt-path",
        str(receipt_path),
        "--expected-anchor-receipt-artifact-id",
        _RECEIPT_ID,
        "--expected-execution-root",
        str(execution_root),
    ]
    required_external_options = (
        "--execution-protocol-path",
        "--expected-execution-protocol-id",
        "--anchor-receipt-path",
        "--expected-anchor-receipt-artifact-id",
        "--expected-execution-root",
    )

    validate_argv = ["validate-anchor", *shared]
    validate_args = parser.parse_args(validate_argv)
    assert validate_args.expected_execution_protocol_id == _PROTOCOL_ID
    assert validate_args.expected_anchor_receipt_artifact_id == _RECEIPT_ID
    for required_option in required_external_options:
        with pytest.raises(SystemExit):
            parser.parse_args(_remove_option(validate_argv, required_option))

    execute_argv = [
        "execute",
        *shared,
        "--preflight-root",
        str(tmp_path / "preflight"),
        "--bge-snapshot-root",
        str(tmp_path / "bge"),
        "--run-nonce",
        "d" * 64,
    ]
    execute_args = parser.parse_args(execute_argv)
    assert execute_args.expected_execution_protocol_id == _PROTOCOL_ID
    assert execute_args.expected_anchor_receipt_artifact_id == _RECEIPT_ID
    assert execute_args.run_nonce == "d" * 64
    for required_option in required_external_options:
        with pytest.raises(SystemExit):
            parser.parse_args(_remove_option(execute_argv, required_option))
    invalid_nonce_argv = list(execute_argv)
    invalid_nonce_argv[invalid_nonce_argv.index("--run-nonce") + 1] = "not-a-digest"
    with pytest.raises(SystemExit):
        parser.parse_args(invalid_nonce_argv)


def test_record_anchor_calls_live_observer_and_writes_only_pre_registered_receipt(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    anchor_relative_path = "artifacts/relationship_lab/v2-anchor-receipt.json"
    anchor_path = repository / pathlib.PurePosixPath(anchor_relative_path)
    anchor_path.parent.mkdir(parents=True)
    execution_root = tmp_path / "execution"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(b"fixture protocol path\n")
    protocol = {
        "schema_version": "fixture-protocol.v2",
        "external_public_anchor": {"receipt_path": anchor_relative_path},
    }
    protocol_raw = canonical_json_bytes(protocol) + b"\n"
    receipt = {
        "schema_version": "fixture-public-anchor-receipt.v2",
        "artifact_id": _RECEIPT_ID,
    }
    observer_calls: list[dict[str, object]] = []
    validation_calls: list[dict[str, object]] = []

    def observe(**kwargs: object) -> dict[str, object]:
        assert not os.path.lexists(anchor_path)
        observer_calls.append(kwargs)
        return receipt

    def validate(payload: object, **kwargs: object) -> str:
        assert payload == receipt
        validation_calls.append(kwargs)
        return _RECEIPT_ID

    protocol_owner = cli_module._load_v2_protocol_module()
    fake_protocol_module = SimpleNamespace(
        canonical_relationship_condition_reader_qualification_execution_root_v2=(
            protocol_owner.canonical_relationship_condition_reader_qualification_execution_root_v2
        ),
        load_relationship_condition_reader_qualification_execution_protocol_v2=(
            lambda path, **kwargs: (protocol, protocol_raw)
        ),
        observe_relationship_condition_reader_qualification_public_anchor_v2=observe,
        validate_relationship_condition_reader_qualification_public_anchor_receipt_v2=(validate),
    )
    model_free_checks: list[bool] = []
    monkeypatch.setattr(cli_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        cli_module,
        "_load_v2_protocol_module",
        lambda: fake_protocol_module,
    )
    monkeypatch.setattr(
        cli_module,
        "_assert_model_free",
        lambda: model_free_checks.append(True),
    )
    args = cli_module._build_parser().parse_args(
        [
            "record-anchor",
            "--execution-protocol-path",
            str(protocol_path),
            "--expected-execution-protocol-id",
            _PROTOCOL_ID,
            "--gist-id",
            "c" * 32,
            "--expected-execution-root",
            str(execution_root),
            "--timeout-seconds",
            "19",
        ]
    )

    summary = cli_module._record_anchor(args)

    assert len(observer_calls) == 1
    observation = observer_calls[0]
    assert set(observation) == {
        "execution_protocol_payload",
        "execution_protocol_raw",
        "expected_execution_protocol_id",
        "expected_execution_root",
        "gist_id",
        "timeout_seconds",
    }
    assert observation == {
        "execution_protocol_payload": protocol,
        "execution_protocol_raw": protocol_raw,
        "expected_execution_protocol_id": _PROTOCOL_ID,
        "expected_execution_root": execution_root,
        "gist_id": "c" * 32,
        "timeout_seconds": 19,
    }
    assert all(
        forbidden_fragment not in key
        for key in observation
        for forbidden_fragment in (
            "history",
            "public",
            "first_revision",
            "created",
            "updated",
            "observed_at",
            "raw_url",
            "observation_json",
            "observed_protocol_raw_path",
        )
    )
    assert validation_calls == [
        {
            "expected_receipt_artifact_id": _RECEIPT_ID,
            "execution_protocol_payload": protocol,
            "execution_protocol_raw": protocol_raw,
            "expected_execution_protocol_id": _PROTOCOL_ID,
            "expected_execution_root": execution_root,
        }
    ]
    assert anchor_path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert summary["anchor_receipt_path"] == str(anchor_path)
    assert summary["anchor_receipt_artifact_id"] == _RECEIPT_ID
    assert summary["github_reobservation_performed_by_cli"] is True
    assert summary["caller_supplied_github_observation_metadata_used"] is False
    assert model_free_checks == [True, True]

    with pytest.raises(FileExistsError, match="already exists"):
        cli_module._record_anchor(args)
    assert len(observer_calls) == 1


def test_execute_calls_v2_outer_with_independently_supplied_ids(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    protocol_path = tmp_path / "protocol.json"
    receipt_path = tmp_path / "receipt.json"
    execution_root = tmp_path / "execution"
    preflight_root = tmp_path / "preflight"
    bge_root = tmp_path / "bge"
    for directory in (execution_root, preflight_root, bge_root):
        directory.mkdir()
    protocol = {"schema_version": "fixture-protocol.v2"}
    protocol_raw = canonical_json_bytes(protocol) + b"\n"
    receipt = {"schema_version": "fixture-receipt.v2", "artifact_id": _RECEIPT_ID}
    receipt_raw = canonical_json_bytes(receipt) + b"\n"
    outer_calls: list[dict[str, object]] = []

    def execute_v2(**kwargs: object) -> dict[str, object]:
        outer_calls.append(kwargs)
        return {
            "schema_version": "fixture-authorized-execution-manifest.v2",
            "artifact_id": "f" * 64,
        }

    fake_execution_module = SimpleNamespace(
        execute_authorized_relationship_condition_reader_qualification_execution_v2=(execute_v2)
    )
    monkeypatch.setattr(cli_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        cli_module,
        "_load_and_validate_anchor",
        lambda _args: (
            protocol,
            protocol_raw,
            protocol_path,
            receipt,
            receipt_raw,
            receipt_path,
            execution_root,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_load_v2_execution_module",
        lambda: fake_execution_module,
    )
    args = cli_module._build_parser().parse_args(
        [
            "execute",
            "--execution-protocol-path",
            str(protocol_path),
            "--expected-execution-protocol-id",
            _PROTOCOL_ID,
            "--anchor-receipt-path",
            str(receipt_path),
            "--expected-anchor-receipt-artifact-id",
            _RECEIPT_ID,
            "--expected-execution-root",
            str(execution_root),
            "--preflight-root",
            str(preflight_root),
            "--bge-snapshot-root",
            str(bge_root),
            "--run-nonce",
            "d" * 64,
            "--prediction-timeout-seconds",
            "73",
            "--scorer-timeout-seconds",
            "19",
        ]
    )

    result = cli_module._execute(args)

    assert result["artifact_id"] == "f" * 64
    assert len(outer_calls) == 1
    call = outer_calls[0]
    assert call["execution_protocol_payload"] == protocol
    assert call["execution_protocol_raw"] == protocol_raw
    assert call["expected_execution_protocol_id"] == _PROTOCOL_ID
    assert call["public_anchor_receipt_payload"] == receipt
    assert call["expected_public_anchor_receipt_artifact_id"] == _RECEIPT_ID
    assert call["repository_root"] == repository
    assert call["preflight_root"] == preflight_root
    assert call["bge_snapshot_root"] == bge_root
    assert call["execution_root"] == execution_root
    assert call["run_nonce"] == "d" * 64
    assert call["python_executable"] is None
    assert call["prediction_timeout_seconds"] == 73
    assert call["scorer_timeout_seconds"] == 19
