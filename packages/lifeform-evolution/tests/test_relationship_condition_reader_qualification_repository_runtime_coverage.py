from __future__ import annotations

import os
import pathlib
from typing import Mapping

import pytest

from lifeform_evolution.relationship_condition_reader_qualification_execution_protocol import (
    build_relationship_condition_reader_execution_source_tree_manifest,
)
from lifeform_evolution.relationship_condition_reader_qualification_repository_runtime_coverage import (
    build_relationship_condition_reader_repository_runtime_coverage,
    validate_relationship_condition_reader_repository_runtime_coverage,
)


def _repository(tmp_path: pathlib.Path) -> pathlib.Path:
    root = (tmp_path / "repository").resolve()
    package = root / "packages/demo/src/demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""fixture package."""\n', encoding="utf-8")
    (package / "reader.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "packages/demo/src/top_level.py").write_text("TOP_LEVEL = True\n", encoding="utf-8")
    resources = package / "resources"
    resources.mkdir()
    (resources / "schema.json").write_bytes(b'{"version":1}\n')
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "run_relationship_condition_reader_qualification_execution.py").write_text(
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    return root


def _source_tree(root: pathlib.Path) -> Mapping[str, object]:
    return build_relationship_condition_reader_execution_source_tree_manifest(repository_root=root)


def _coverage(root: pathlib.Path, source_tree: Mapping[str, object]) -> Mapping[str, object]:
    return build_relationship_condition_reader_repository_runtime_coverage(
        repository_root=root,
        execution_source_tree=source_tree,
    )


def test_repository_runtime_coverage_pins_complete_tree_and_binds_source_manifest(
    tmp_path: pathlib.Path,
) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)

    observed = _coverage(root, source_tree)

    paths = [row["path"] for row in observed["entries"]]  # type: ignore[index]
    assert observed["source_roots"] == ["packages/demo/src"]
    assert paths == [
        "packages/demo/src/demo/__init__.py",
        "packages/demo/src/demo/reader.py",
        "packages/demo/src/demo/resources/schema.json",
        "packages/demo/src/top_level.py",
    ]
    assert observed["execution_source_tree_schema_version"] == source_tree["schema_version"]
    assert observed["execution_source_tree_artifact_id"] == source_tree["artifact_id"]
    assert observed["execution_source_tree_entry_count"] == source_tree["entry_count"]
    assert validate_relationship_condition_reader_repository_runtime_coverage(observed) == observed["artifact_id"]
    assert (
        validate_relationship_condition_reader_repository_runtime_coverage(
            observed,
            execution_source_tree=source_tree,
        )
        == observed["artifact_id"]
    )
    assert (
        validate_relationship_condition_reader_repository_runtime_coverage(
            observed,
            repository_root=root,
            execution_source_tree=source_tree,
        )
        == observed["artifact_id"]
    )


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("packages/demo/src/json.PyC", "bytecode outside __pycache__"),
        ("packages/demo/src/sentinel.PYD", "native shadow"),
        ("packages/demo/src/demo/sentinel.DlL", "native shadow"),
        ("packages/demo/src/demo/__pycache__/sentinel.pyd", "native shadow"),
    ],
)
def test_repository_runtime_coverage_rejects_unobserved_import_shadows(
    tmp_path: pathlib.Path,
    relative_path: str,
    message: str,
) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    poison = root / pathlib.PurePosixPath(relative_path)
    poison.parent.mkdir(parents=True, exist_ok=True)
    poison.write_bytes(b"poison")

    with pytest.raises(ValueError, match=message):
        _coverage(root, source_tree)


def test_repository_runtime_coverage_validates_cache_and_excludes_only_its_bytecode(
    tmp_path: pathlib.Path,
) -> None:
    root = _repository(tmp_path)
    cache = root / "packages/demo/src/demo/__pycache__"
    nested = cache / "nested"
    nested.mkdir(parents=True)
    (cache / "reader.CPYTHON-311.PYC").write_bytes(b"ignored bytecode")
    (nested / "metadata.json").write_bytes(b'{"pinned":true}\n')
    source_tree = _source_tree(root)

    observed = _coverage(root, source_tree)

    paths = [row["path"] for row in observed["entries"]]  # type: ignore[index]
    assert observed["excluded_cache_directory_count"] == 1
    assert observed["excluded_bytecode_file_count"] == 1
    assert "packages/demo/src/demo/__pycache__/reader.CPYTHON-311.PYC" not in paths
    assert "packages/demo/src/demo/__pycache__/nested/metadata.json" in paths


@pytest.mark.parametrize("mutation", ["modify", "add"])
def test_repository_runtime_coverage_requires_exact_frozen_python_join(
    tmp_path: pathlib.Path,
    mutation: str,
) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    package = root / "packages/demo/src/demo"
    if mutation == "modify":
        (package / "reader.py").write_text("VALUE = 2\n", encoding="utf-8")
    else:
        (package / "late.py").write_text("LATE = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen Python source entries do not exactly match"):
        _coverage(root, source_tree)


def test_repository_runtime_coverage_reobservation_rejects_package_data_drift(
    tmp_path: pathlib.Path,
) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    frozen = _coverage(root, source_tree)
    (root / "packages/demo/src/demo/resources/schema.json").write_bytes(b'{"version":2}\n')

    with pytest.raises(ValueError, match="does not match the current exact source roots"):
        validate_relationship_condition_reader_repository_runtime_coverage(
            frozen,
            repository_root=root,
            execution_source_tree=source_tree,
        )


def test_repository_runtime_coverage_rejects_artifact_tampering(tmp_path: pathlib.Path) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    tampered = dict(_coverage(root, source_tree))
    tampered["artifact_id"] = "0" * 64

    with pytest.raises(ValueError, match="artifact_id mismatch"):
        validate_relationship_condition_reader_repository_runtime_coverage(tampered)


def test_repository_runtime_coverage_rejects_source_manifest_splicing(tmp_path: pathlib.Path) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    frozen = _coverage(root, source_tree)
    other_root = _repository(tmp_path / "other")
    (other_root / "packages/demo/src/demo/reader.py").write_text("VALUE = 99\n", encoding="utf-8")
    other_source_tree = _source_tree(other_root)

    with pytest.raises(ValueError, match="bound to a different execution source tree"):
        validate_relationship_condition_reader_repository_runtime_coverage(
            frozen,
            execution_source_tree=other_source_tree,
        )


def test_repository_runtime_coverage_requires_complete_reobservation_arguments(
    tmp_path: pathlib.Path,
) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    frozen = _coverage(root, source_tree)

    with pytest.raises(ValueError, match="required for repository reobservation"):
        validate_relationship_condition_reader_repository_runtime_coverage(
            frozen,
            repository_root=root,
        )


def test_repository_runtime_coverage_rejects_directory_symlink_or_reparse(
    tmp_path: pathlib.Path,
) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    target = tmp_path / "outside"
    target.mkdir()
    link = root / "packages/demo/src/demo/linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlink or reparse point"):
        _coverage(root, source_tree)


def test_repository_runtime_coverage_rejects_hardlinked_file(tmp_path: pathlib.Path) -> None:
    root = _repository(tmp_path)
    source_tree = _source_tree(root)
    source = root / "packages/demo/src/demo/resources/schema.json"
    alias = source.with_name("schema-alias.json")
    try:
        os.link(source, alias)
    except OSError as error:
        pytest.skip(f"hardlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="exactly one hard link"):
        _coverage(root, source_tree)
