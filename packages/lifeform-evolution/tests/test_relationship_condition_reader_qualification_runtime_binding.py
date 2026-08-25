from __future__ import annotations

import _imp
from dataclasses import replace
import hashlib
import importlib.machinery
import ntpath
import os
import pathlib
import platform
import sys
import types
from typing import Mapping

import _socket
import pytest

from lifeform_evolution.relationship_condition_reader_qualification_runtime_binding import (
    QualificationChildImportBinding,
    build_qualification_child_import_binding,
    controlled_child_path,
    expected_child_sys_path,
    snapshot_file_backed_module_origins,
    validate_child_file_backed_module_origin_attestation,
)


_MODULE_PATHS = {
    "lifeform_evolution.relationship_condition_reader_qualification_predictor": (
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_predictor.py"
    ),
    "volvence_zero.social_cognition": ("packages/vz-contracts/src/volvence_zero/social_cognition.py"),
}
_EXTRA_SOURCE_PATH = "packages/vz-temporal/src/volvence_zero/temporal/__init__.py"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _binding() -> QualificationChildImportBinding:
    repository_root = pathlib.Path(__file__).resolve().parents[3]
    frozen_entries: dict[str, Mapping[str, object]] = {}
    for relative in sorted({*_MODULE_PATHS.values(), _EXTRA_SOURCE_PATH}):
        raw = (repository_root / pathlib.PurePosixPath(relative)).read_bytes()
        frozen_entries[relative] = {
            "raw_sha256": _sha(raw),
            "raw_bytes": len(raw),
        }
    return build_qualification_child_import_binding(
        python_executable=pathlib.Path(sys.executable).resolve(),
        repository_root=repository_root,
        repository_source_roots=tuple(
            repository_root / pathlib.PurePosixPath(relative)
            for relative in (
                "packages/lifeform-evolution/src",
                "packages/vz-contracts/src",
                "packages/vz-temporal/src",
            )
        ),
        frozen_source_entries=frozen_entries,
        frozen_site_packages_root=(pathlib.Path(sys.executable).resolve().parent / "Lib" / "site-packages").resolve(),
    )


def _origin_rows(binding: QualificationChildImportBinding) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for module_name, relative in sorted(_MODULE_PATHS.items()):
        origin = binding.repository_root / pathlib.PurePosixPath(relative)
        rows.append(
            {
                "module_name": module_name,
                "origin": str(origin),
            }
        )
    return rows


def _namespace_locations(binding: QualificationChildImportBinding) -> list[str]:
    return [str(path) for path in binding.volvence_zero_namespace_search_locations]


def _temporary_binding(
    tmp_path: pathlib.Path,
) -> tuple[QualificationChildImportBinding, pathlib.Path]:
    repository_root = (tmp_path / "repository").resolve()
    source_root = repository_root / "packages/example/src"
    namespace_root = source_root / "volvence_zero"
    namespace_root.mkdir(parents=True)
    origin = namespace_root / "example.py"
    raw = b"VALUE = 1\n"
    origin.write_bytes(raw)
    binding = build_qualification_child_import_binding(
        python_executable=pathlib.Path(sys.executable).resolve(),
        repository_root=repository_root,
        repository_source_roots=(source_root,),
        frozen_source_entries={
            "packages/example/src/volvence_zero/example.py": {
                "raw_sha256": _sha(raw),
                "raw_bytes": len(raw),
            }
        },
        frozen_site_packages_root=(pathlib.Path(sys.executable).resolve().parent / "Lib/site-packages").resolve(),
    )
    return binding, origin


def test_runtime_binding_derives_complete_python_path_and_namespace_order() -> None:
    binding = _binding()

    observed = expected_child_sys_path(
        binding,
        python_version=platform.python_version(),
    )

    major, minor, *_rest = platform.python_version().split(".")
    home = binding.python_executable.parent
    assert observed == (
        *(str(path) for path in binding.import_roots),
        str(home / f"python{major}{minor}.zip"),
        str(home / "DLLs"),
        str(home / "Lib"),
        str(home),
    )
    assert binding.volvence_zero_namespace_search_locations == (
        binding.repository_root / "packages/vz-contracts/src/volvence_zero",
        binding.repository_root / "packages/vz-temporal/src/volvence_zero",
    )


def test_runtime_binding_derives_controlled_native_path_without_ambient_entries() -> None:
    binding = _binding()
    windows_root = pathlib.Path("C:/Windows").resolve()

    observed = controlled_child_path(binding, system_root=windows_root)

    home = binding.python_executable.parent
    assert observed == (
        home,
        (home / "DLLs").resolve(),
        (home / "Library/bin").resolve(),
        (windows_root / "System32").resolve(),
        windows_root,
    )
    assert all(path.is_dir() for path in observed)


def test_runtime_binding_validates_exact_module_path_hash_and_namespace_join() -> None:
    binding = _binding()

    validate_child_file_backed_module_origin_attestation(
        loaded_module_origins=_origin_rows(binding),
        volvence_zero_namespace_search_locations=[
            str(path) for path in binding.volvence_zero_namespace_search_locations
        ],
        binding=binding,
        required_module_names=frozenset(_MODULE_PATHS),
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("module_path", "module name does not match"),
        ("namespace_order", "namespace search locations drifted"),
    ],
)
def test_runtime_binding_rejects_source_or_namespace_attestation_drift(
    tamper: str,
    message: str,
) -> None:
    binding = _binding()
    rows = [dict(row) for row in _origin_rows(binding)]
    locations = [str(path) for path in binding.volvence_zero_namespace_search_locations]
    if tamper == "module_path":
        rows[0]["module_name"] = "volvence_zero.social_cognition"
    elif tamper == "namespace_order":
        locations.reverse()
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(tamper)

    with pytest.raises(ValueError, match=message):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=rows,
            volvence_zero_namespace_search_locations=locations,
            binding=binding,
            required_module_names=frozenset(_MODULE_PATHS),
        )


def test_runtime_binding_rejects_source_root_not_derived_from_frozen_entries() -> None:
    binding = _binding()

    with pytest.raises(ValueError, match="do not exactly match"):
        build_qualification_child_import_binding(
            python_executable=binding.python_executable,
            repository_root=binding.repository_root,
            repository_source_roots=binding.repository_source_roots[:-1],
            frozen_source_entries={
                entry.path: {
                    "raw_sha256": entry.raw_sha256,
                    "raw_bytes": entry.raw_bytes,
                }
                for entry in binding.frozen_source_entries
            },
            frozen_site_packages_root=binding.frozen_site_packages_root,
        )


def test_snapshot_includes_all_file_backed_modules_and_omits_builtin_frozen_and_no_file(
    tmp_path: pathlib.Path,
) -> None:
    origin = (tmp_path / "module.py").resolve()
    origin.write_text("VALUE = 1\n", encoding="utf-8")

    alpha = types.ModuleType("alpha")
    alpha.__file__ = str(origin)
    alpha.__spec__ = importlib.machinery.ModuleSpec("alpha", loader=None, origin=str(origin))
    beta = types.ModuleType("beta")
    beta.__file__ = str(origin)
    beta.__spec__ = importlib.machinery.ModuleSpec("beta", loader=None, origin=str(origin))
    builtin = types.ModuleType("builtin_probe")
    builtin.__file__ = str(origin)
    builtin.__spec__ = importlib.machinery.ModuleSpec("builtin_probe", loader=None, origin="built-in")
    frozen = types.ModuleType("frozen_probe")
    frozen.__file__ = str(origin)
    frozen.__spec__ = importlib.machinery.ModuleSpec("frozen_probe", loader=None, origin="frozen")
    no_file = types.ModuleType("namespace_probe")

    observed = snapshot_file_backed_module_origins(
        {
            "beta": beta,
            "frozen_probe": frozen,
            "namespace_probe": no_file,
            "builtin_probe": builtin,
            "alpha": alpha,
            "ntpath": ntpath,
            "none_probe": None,
        }
    )

    assert _imp.is_frozen("ntpath")
    assert observed == [
        {"module_name": "alpha", "origin": str(origin)},
        {"module_name": "beta", "origin": str(origin)},
        {"module_name": "builtin_probe", "origin": str(origin)},
        {"module_name": "frozen_probe", "origin": str(origin)},
    ]

    different_origin = (tmp_path / "different.py").resolve()
    different_origin.write_text("VALUE = 2\n", encoding="utf-8")
    drifted = types.ModuleType("drifted")
    drifted.__file__ = str(origin)
    drifted.__spec__ = importlib.machinery.ModuleSpec(
        "drifted",
        loader=None,
        origin=str(different_origin),
    )
    with pytest.raises(ValueError, match="__file__ and __spec__.origin drifted"):
        snapshot_file_backed_module_origins({"drifted": drifted})

    fake_frozen_key = types.ModuleType("ntpath")
    fake_frozen_key.__file__ = str(origin)
    fake_frozen_key.__spec__ = importlib.machinery.ModuleSpec(
        "ntpath",
        loader=None,
        origin=str(origin),
    )
    with pytest.raises(ValueError, match="interpreter frozen module metadata drifted"):
        snapshot_file_backed_module_origins({"ntpath": fake_frozen_key})

    fake_builtin_key = types.ModuleType("sys")
    fake_builtin_key.__file__ = str(origin)
    fake_builtin_key.__spec__ = importlib.machinery.ModuleSpec(
        "sys",
        loader=None,
        origin=str(origin),
    )
    with pytest.raises(ValueError, match="interpreter built-in module metadata drifted"):
        snapshot_file_backed_module_origins({"sys": fake_builtin_key})


def test_expected_child_sys_path_rejects_present_python_archive(tmp_path: pathlib.Path) -> None:
    binding = _binding()
    python_home = (tmp_path / "python").resolve()
    python_home.mkdir()
    executable = python_home / "python.exe"
    executable.write_bytes(b"fixture executable")
    major, minor, *_rest = platform.python_version().split(".")
    (python_home / f"python{major}{minor}.zip").write_bytes(b"fixture archive")

    with pytest.raises(ValueError, match="does not admit zip-imported"):
        expected_child_sys_path(
            replace(binding, python_executable=executable),
            python_version=platform.python_version(),
        )


def test_all_origin_validator_accepts_repository_site_stdlib_dll_home_and_library_bin() -> None:
    binding = _binding()
    assert pytest.__file__ is not None
    assert pathlib.__file__ is not None
    assert _socket.__file__ is not None
    library_bin_origin = sorted(
        path for path in (binding.python_executable.parent / "Library/bin").iterdir() if path.is_file()
    )[0].resolve()
    rows = [
        *_origin_rows(binding),
        {"module_name": "pytest", "origin": str(pathlib.Path(pytest.__file__).resolve())},
        {"module_name": "pathlib", "origin": str(pathlib.Path(pathlib.__file__).resolve())},
        {"module_name": "_socket", "origin": str(pathlib.Path(_socket.__file__).resolve())},
        {"module_name": "python_runtime_probe", "origin": str(binding.python_executable)},
        {"module_name": "library_bin_probe", "origin": str(library_bin_origin)},
    ]
    rows.sort(key=lambda row: str(row["module_name"]).encode("utf-8"))

    validate_child_file_backed_module_origin_attestation(
        loaded_module_origins=rows,
        volvence_zero_namespace_search_locations=_namespace_locations(binding),
        binding=binding,
        required_module_names=frozenset(_MODULE_PATHS),
    )


def test_all_origin_validator_allows_stdlib_aliases_with_the_same_origin() -> None:
    binding = _binding()
    assert ntpath.__file__ is not None
    assert os.path.__file__ == ntpath.__file__
    origin = str(pathlib.Path(ntpath.__file__).resolve())

    validate_child_file_backed_module_origin_attestation(
        loaded_module_origins=[
            {"module_name": "ntpath", "origin": origin},
            {"module_name": "os.path", "origin": origin},
        ],
        volvence_zero_namespace_search_locations=_namespace_locations(binding),
        binding=binding,
        required_module_names=frozenset(),
    )


@pytest.mark.parametrize("suffix", [".pyc", ".pyd", ".dll"])
def test_all_origin_validator_rejects_repository_binary_or_bytecode_shadow(
    tmp_path: pathlib.Path,
    suffix: str,
) -> None:
    binding, frozen_origin = _temporary_binding(tmp_path)
    shadow = frozen_origin.parents[1] / f"json{suffix}"
    shadow.write_bytes(b"shadow")

    with pytest.raises(ValueError, match=r"frozen \.py source"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[{"module_name": "json", "origin": str(shadow)}],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )


def test_all_origin_validator_rejects_uncontrolled_origin(tmp_path: pathlib.Path) -> None:
    binding, _frozen_origin = _temporary_binding(tmp_path)
    outside = (tmp_path / "outside.py").resolve()
    outside.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside every controlled import domain"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[{"module_name": "outside", "origin": str(outside)}],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )


def test_all_origin_validator_rejects_noncanonical_origin_text(tmp_path: pathlib.Path) -> None:
    binding, origin = _temporary_binding(tmp_path)
    noncanonical = origin.parent / ".." / origin.parent.name / origin.name

    with pytest.raises(ValueError, match="exact canonical path text"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[
                {"module_name": "volvence_zero.example", "origin": str(noncanonical)},
            ],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )


def test_all_origin_validator_rejects_duplicate_or_noncanonical_module_order() -> None:
    binding = _binding()
    origin = str(binding.python_executable)

    with pytest.raises(ValueError, match="canonical module-name order"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[
                {"module_name": "zeta", "origin": origin},
                {"module_name": "alpha", "origin": origin},
            ],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )
    with pytest.raises(ValueError, match="names must be unique"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[
                {"module_name": "alpha", "origin": origin},
                {"module_name": "alpha", "origin": origin},
            ],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )


@pytest.mark.parametrize(
    ("loaded_origins", "error_type"),
    [
        (({"module_name": "probe", "origin": "C:/probe.py"},), TypeError),
        ([[]], TypeError),
        ([{"module_name": "probe", "origin": "C:/probe.py", "extra": False}], ValueError),
        ([{"module_name": 1, "origin": "C:/probe.py"}], TypeError),
        ([{"module_name": "probe", "origin": 1}], TypeError),
    ],
)
def test_all_origin_validator_rejects_shape_and_scalar_type_confusion(
    loaded_origins: object,
    error_type: type[Exception],
) -> None:
    binding = _binding()

    with pytest.raises(error_type):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=loaded_origins,
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )


def test_all_origin_validator_rejects_site_shadow_of_required_repository_module() -> None:
    binding = _binding()
    assert pytest.__file__ is not None

    with pytest.raises(ValueError, match="shadowed outside repository source roots"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[
                {
                    "module_name": "lifeform_evolution.relationship_condition_reader_qualification_predictor",
                    "origin": str(pathlib.Path(pytest.__file__).resolve()),
                }
            ],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(
                {"lifeform_evolution.relationship_condition_reader_qualification_predictor"}
            ),
        )


def test_all_origin_validator_reobserves_repository_identity_and_frozen_membership(
    tmp_path: pathlib.Path,
) -> None:
    binding, origin = _temporary_binding(tmp_path)
    row = {"module_name": "volvence_zero.example", "origin": str(origin)}
    origin.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="raw identity drifted"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[row],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )

    unfrozen = origin.with_name("unfrozen.py")
    unfrozen.write_bytes(b"VALUE = 1\n")
    with pytest.raises(ValueError, match="absent from frozen source entries"):
        validate_child_file_backed_module_origin_attestation(
            loaded_module_origins=[
                {"module_name": "volvence_zero.unfrozen", "origin": str(unfrozen)},
            ],
            volvence_zero_namespace_search_locations=_namespace_locations(binding),
            binding=binding,
            required_module_names=frozenset(),
        )
