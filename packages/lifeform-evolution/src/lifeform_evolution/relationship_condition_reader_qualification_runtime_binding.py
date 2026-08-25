"""Frozen Python import binding shared by reader-qualification child processes."""

from __future__ import annotations

import _imp
from dataclasses import dataclass
import hashlib
import importlib.machinery
import os
import pathlib
import stat
import sys
from typing import Mapping


@dataclass(frozen=True)
class FrozenSourceEntry:
    path: str
    raw_sha256: str
    raw_bytes: int


@dataclass(frozen=True)
class QualificationChildImportBinding:
    python_executable: pathlib.Path
    repository_root: pathlib.Path
    repository_source_roots: tuple[pathlib.Path, ...]
    frozen_source_entries: tuple[FrozenSourceEntry, ...]
    frozen_site_packages_root: pathlib.Path
    import_roots: tuple[pathlib.Path, ...]
    volvence_zero_namespace_search_locations: tuple[pathlib.Path, ...]


def build_qualification_child_import_binding(
    *,
    python_executable: pathlib.Path,
    repository_root: pathlib.Path,
    repository_source_roots: tuple[pathlib.Path, ...],
    frozen_source_entries: Mapping[str, Mapping[str, object]],
    frozen_site_packages_root: pathlib.Path,
) -> QualificationChildImportBinding:
    """Build an exact import boundary from already-frozen protocol fields."""

    executable = _existing_canonical_file(python_executable, "child Python executable")
    root = _existing_canonical_directory(repository_root, "repository root")
    if not isinstance(repository_source_roots, tuple) or not repository_source_roots:
        raise TypeError("repository_source_roots must be a non-empty tuple of canonical paths")
    source_roots = tuple(
        _existing_canonical_directory(value, f"repository source root {index}")
        for index, value in enumerate(repository_source_roots)
    )
    site_packages = _existing_canonical_directory(
        frozen_site_packages_root,
        "frozen site-packages root",
    )
    expected_site_packages = (executable.parent / "Lib" / "site-packages").resolve(strict=True)
    if _normalized_path(site_packages) != _normalized_path(expected_site_packages):
        raise ValueError("frozen site-packages root does not belong to the frozen Python executable")
    if not isinstance(frozen_source_entries, Mapping) or not frozen_source_entries:
        raise TypeError("frozen_source_entries must be a non-empty mapping")
    entries: list[FrozenSourceEntry] = []
    for path, raw_row in frozen_source_entries.items():
        if not isinstance(path, str) or not path:
            raise TypeError("frozen source entry paths must be non-empty text")
        relative = pathlib.PurePosixPath(path)
        if (
            relative.is_absolute()
            or relative.as_posix() != path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"frozen source entry path is not canonical POSIX relative text: {path!r}")
        row = _mapping(raw_row, f"frozen source entry {path}")
        if set(row) != {"raw_sha256", "raw_bytes"}:
            raise ValueError(f"frozen source entry {path} must contain exact raw identity fields")
        entries.append(
            FrozenSourceEntry(
                path=path,
                raw_sha256=_digest(row["raw_sha256"], f"frozen source entry {path} raw_sha256"),
                raw_bytes=_nonnegative_integer(
                    row["raw_bytes"],
                    f"frozen source entry {path} raw_bytes",
                ),
            )
        )
    entries.sort(key=lambda item: item.path.encode("utf-8"))
    if len({item.path.casefold() for item in entries}) != len(entries):
        raise ValueError("frozen source entry paths contain a casefold collision")

    expected_source_relatives = sorted(
        {
            pathlib.PurePosixPath(*parts[:3]).as_posix()
            for item in entries
            if (parts := pathlib.PurePosixPath(item.path).parts)
            and len(parts) >= 5
            and parts[0] == "packages"
            and parts[2] == "src"
            and pathlib.PurePosixPath(item.path).suffix == ".py"
        },
        key=lambda value: value.encode("utf-8"),
    )
    observed_source_relatives: list[str] = []
    for source_root in source_roots:
        try:
            relative = source_root.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("repository source roots must remain below repository_root") from error
        observed_source_relatives.append(relative)
    if observed_source_relatives != expected_source_relatives:
        raise ValueError("repository source roots do not exactly match the roots derived from frozen source entries")

    import_roots = (*source_roots, site_packages)
    if len({_normalized_path(path) for path in import_roots}) != len(import_roots):
        raise ValueError("child import roots must be unique")
    namespace_locations = tuple(
        source_root / "volvence_zero"
        for source_root, relative_root in zip(
            source_roots,
            observed_source_relatives,
            strict=True,
        )
        if any(item.path.startswith(f"{relative_root}/volvence_zero/") for item in entries)
    )
    if not namespace_locations:
        raise ValueError("frozen source entries do not contain the volvence_zero namespace")
    for index, location in enumerate(namespace_locations):
        canonical = _existing_canonical_directory(location, f"volvence_zero namespace location {index}")
        if canonical != location:
            raise ValueError("volvence_zero namespace locations must be canonical")
    return QualificationChildImportBinding(
        python_executable=executable,
        repository_root=root,
        repository_source_roots=source_roots,
        frozen_source_entries=tuple(entries),
        frozen_site_packages_root=site_packages,
        import_roots=import_roots,
        volvence_zero_namespace_search_locations=namespace_locations,
    )


def expected_child_sys_path(
    binding: QualificationChildImportBinding,
    *,
    python_version: str,
) -> tuple[str, ...]:
    """Return the complete ``-P -S`` path after controlled PYTHONPATH loading."""

    if not isinstance(binding, QualificationChildImportBinding):
        raise TypeError("binding must be a QualificationChildImportBinding")
    if not isinstance(python_version, str) or not python_version:
        raise TypeError("python_version must be non-empty text")
    components = python_version.split(".")
    if len(components) < 2 or not components[0].isdigit() or not components[1].isdigit():
        raise ValueError("python_version must begin with numeric major.minor components")
    major = int(components[0])
    minor = int(components[1])
    if major != 3 or minor < 11:
        raise ValueError("reader qualification requires the frozen Python 3.11+ runtime")
    home = binding.python_executable.parent
    python_archive = home / f"python{major}{minor}.zip"
    if python_archive.exists():
        raise ValueError("reader qualification does not admit zip-imported Python runtime modules")
    return (
        *(str(path) for path in binding.import_roots),
        str(python_archive),
        str(home / "DLLs"),
        str(home / "Lib"),
        str(home),
    )


def controlled_child_path(
    binding: QualificationChildImportBinding,
    *,
    system_root: pathlib.Path,
) -> tuple[pathlib.Path, ...]:
    """Return the only native executable/DLL search directories for a child."""

    if not isinstance(binding, QualificationChildImportBinding):
        raise TypeError("binding must be a QualificationChildImportBinding")
    windows_root = _existing_canonical_directory(system_root, "child SystemRoot")
    python_home = binding.python_executable.parent
    candidates = (
        python_home,
        python_home / "DLLs",
        python_home / "Library" / "bin",
        windows_root / "System32",
        windows_root,
    )
    controlled: list[pathlib.Path] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        canonical = _existing_canonical_directory(
            candidate,
            f"controlled child PATH directory {index}",
        )
        normalized = _normalized_path(canonical)
        if normalized in seen:
            continue
        seen.add(normalized)
        controlled.append(canonical)
    return tuple(controlled)


def snapshot_file_backed_module_origins(
    modules: Mapping[str, object],
) -> list[dict[str, str]]:
    """Snapshot every non-builtin, non-frozen ``sys.modules`` file origin."""

    if not isinstance(modules, Mapping):
        raise TypeError("modules must be a mapping")
    origins: list[dict[str, str]] = []
    for raw_module_name, module in list(modules.items()):
        module_name = _text(raw_module_name, "loaded module name")
        if module is None:
            continue
        try:
            namespace = vars(module)
        except TypeError as error:
            raise TypeError(f"loaded module {module_name} has no module namespace") from error
        if _is_interpreter_non_file_module(
            module_name,
            namespace=namespace,
        ):
            continue
        module_spec = namespace.get("__spec__")
        spec_origin: str | None = None
        if module_spec is not None:
            try:
                spec_origin = module_spec.origin
            except AttributeError as error:
                raise TypeError(f"loaded module {module_name} has a malformed __spec__") from error
            if spec_origin is not None and not isinstance(spec_origin, str):
                raise TypeError(f"loaded module {module_name} has a non-text __spec__.origin")
        raw_origin = namespace.get("__file__")
        if raw_origin is None:
            continue
        origin_text = _text(raw_origin, f"loaded module origin {module_name}")
        origin = _existing_canonical_file(
            pathlib.Path(origin_text),
            f"loaded module origin {module_name}",
        )
        if module_spec is not None and spec_origin is not None and spec_origin not in {"built-in", "frozen"}:
            spec_origin_path = pathlib.Path(spec_origin)
            if not spec_origin_path.is_absolute():
                raise ValueError(f"loaded module {module_name} has a non-absolute __spec__.origin")
            if spec_origin_path.resolve(strict=True) != origin:
                raise ValueError(f"loaded module {module_name} __file__ and __spec__.origin drifted")
        origins.append(
            {
                "module_name": module_name,
                "origin": str(origin),
            }
        )
    origins.sort(key=lambda row: row["module_name"].encode("utf-8"))
    if len({row["module_name"] for row in origins}) != len(origins):
        raise ValueError("loaded file-backed module names must be unique")
    return origins


def _is_interpreter_non_file_module(
    module_name: str,
    *,
    namespace: Mapping[str, object],
) -> bool:
    if module_name in sys.builtin_module_names:
        expected_origin = "built-in"
        expected_loader = importlib.machinery.BuiltinImporter
        kind = "built-in"
    elif _imp.is_frozen(module_name):
        expected_origin = "frozen"
        expected_loader = importlib.machinery.FrozenImporter
        kind = "frozen"
    else:
        return False
    module_spec = namespace.get("__spec__")
    if module_spec is None:
        raise ValueError(f"interpreter {kind} module {module_name} is missing __spec__")
    try:
        spec_name = module_spec.name
        spec_origin = module_spec.origin
        spec_loader = module_spec.loader
    except AttributeError as error:
        raise TypeError(f"interpreter {kind} module {module_name} has a malformed __spec__") from error
    registered_spec_name = isinstance(spec_name, str) and (
        spec_name in sys.builtin_module_names if kind == "built-in" else _imp.is_frozen(spec_name)
    )
    if (
        not registered_spec_name
        or spec_origin != expected_origin
        or spec_loader is not expected_loader
        or namespace.get("__loader__") is not expected_loader
    ):
        raise ValueError(f"interpreter {kind} module metadata drifted: {module_name}")
    return True


def validate_child_file_backed_module_origin_attestation(
    *,
    loaded_module_origins: object,
    volvence_zero_namespace_search_locations: object,
    binding: QualificationChildImportBinding,
    required_module_names: frozenset[str],
) -> None:
    """Validate all reported file-backed origins and close repository source joins."""

    if not isinstance(binding, QualificationChildImportBinding):
        raise TypeError("binding must be a QualificationChildImportBinding")
    if not isinstance(required_module_names, frozenset) or not all(
        isinstance(value, str) and _is_python_module_name(value) for value in required_module_names
    ):
        raise TypeError("required_module_names must be a frozenset of Python module names")
    if not isinstance(loaded_module_origins, list):
        raise TypeError("loaded_module_origins must be a list")
    frozen_by_path = {entry.path: entry for entry in binding.frozen_source_entries}
    repository_top_levels = _repository_top_level_module_names(binding.frozen_source_entries)
    observed_names: list[str] = []
    observed_repository_names: set[str] = set()
    observed_repository_paths: list[str] = []
    for index, raw_row in enumerate(loaded_module_origins):
        row = _mapping(raw_row, f"loaded file-backed module origin {index}")
        if set(row) != {"module_name", "origin"}:
            raise ValueError(f"loaded file-backed module origin {index} fields drifted")
        module_name = _text(row["module_name"], f"loaded module name {index}")
        if not _is_python_module_name(module_name):
            raise ValueError(f"loaded module name is not canonical dotted Python text: {module_name}")
        origin_text = _text(row["origin"], f"loaded module origin {module_name}")
        origin = _existing_canonical_file(pathlib.Path(origin_text), f"loaded module origin {module_name}")
        if origin_text != str(origin):
            raise ValueError(f"loaded module origin must use exact canonical path text: {module_name}")
        domain = _controlled_origin_domain(origin, binding=binding)
        if domain == "repository_source":
            if origin.suffix != ".py":
                raise ValueError(f"loaded repository module origin must be frozen .py source: {module_name}")
            relative = origin.relative_to(binding.repository_root).as_posix()
            expected_module_name = _module_name_for_repository_path(relative)
            if module_name != expected_module_name:
                raise ValueError(
                    "loaded repository module name does not match its frozen repository path: "
                    f"{module_name} != {expected_module_name}"
                )
            frozen = frozen_by_path.get(relative)
            if frozen is None:
                raise ValueError(f"loaded repository module is absent from frozen source entries: {module_name}")
            observed_sha256, observed_bytes = _hash_stable_regular_file(origin)
            if observed_sha256 != frozen.raw_sha256 or observed_bytes != frozen.raw_bytes:
                raise ValueError(f"loaded repository module raw identity drifted: {module_name}")
            observed_repository_names.add(module_name)
            observed_repository_paths.append(relative)
        elif module_name.split(".", 1)[0] in repository_top_levels:
            raise ValueError(f"repository module name was shadowed outside repository source roots: {module_name}")
        observed_names.append(module_name)
    if observed_names != sorted(observed_names, key=lambda value: value.encode("utf-8")):
        raise ValueError("loaded file-backed module origins must use canonical module-name order")
    if len(set(observed_names)) != len(observed_names):
        raise ValueError("loaded file-backed module names must be unique")
    if len(set(observed_repository_paths)) != len(observed_repository_paths):
        raise ValueError("loaded repository module paths must map one-to-one to module names")
    missing = sorted(required_module_names - observed_repository_names)
    if missing:
        raise ValueError(f"required repository modules are absent from child attestation: {missing}")

    if not isinstance(volvence_zero_namespace_search_locations, list) or not all(
        isinstance(value, str) and value for value in volvence_zero_namespace_search_locations
    ):
        raise TypeError("volvence_zero namespace search locations must be a list of paths")
    expected_locations = [str(path) for path in binding.volvence_zero_namespace_search_locations]
    if volvence_zero_namespace_search_locations != expected_locations:
        raise ValueError("volvence_zero namespace search locations drifted")
    for index, value in enumerate(volvence_zero_namespace_search_locations):
        _existing_canonical_directory(
            pathlib.Path(value),
            f"volvence_zero namespace search location {index}",
        )


def _controlled_origin_domain(
    origin: pathlib.Path,
    *,
    binding: QualificationChildImportBinding,
) -> str:
    repository_matches = [
        source_root for source_root in binding.repository_source_roots if _is_relative_to(origin, source_root)
    ]
    if len(repository_matches) > 1:
        raise ValueError(f"loaded module origin belongs to multiple repository source roots: {origin}")

    python_home = binding.python_executable.parent
    site_packages = binding.frozen_site_packages_root
    python_lib = python_home / "Lib"
    python_dlls = python_home / "DLLs"
    python_library_bin = python_home / "Library" / "bin"
    domains: list[str] = []
    if repository_matches:
        domains.append("repository_source")
    if _is_relative_to(origin, site_packages):
        domains.append("site_packages")
    if _is_relative_to(origin, python_lib) and not _is_relative_to(origin, site_packages):
        domains.append("python_lib")
    if _is_relative_to(origin, python_dlls):
        domains.append("python_dlls")
    if _is_relative_to(origin, python_library_bin):
        domains.append("python_library_bin")
    if origin.parent == python_home:
        domains.append("python_home")
    if len(domains) != 1:
        if _is_relative_to(origin, binding.repository_root):
            raise ValueError(f"loaded module origin is inside the repository but outside frozen source roots: {origin}")
        raise ValueError(f"loaded module origin is outside every controlled import domain: {origin}")
    return domains[0]


def _hash_stable_regular_file(path: pathlib.Path) -> tuple[str, int]:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"repository module origin must be a regular non-symlink file: {path}")
    if before.st_nlink != 1:
        raise ValueError(f"repository module origin must have exactly one hardlink: {path}")
    raw = path.read_bytes()
    after = os.lstat(path)
    if _file_identity(before) != _file_identity(after):
        raise RuntimeError(f"repository module origin changed during read: {path}")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _existing_canonical_file(value: pathlib.Path, field_name: str) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    resolved = candidate.resolve(strict=True)
    if _normalized_path(candidate) != _normalized_path(resolved):
        raise ValueError(f"{field_name} must be canonical: {candidate}")
    value_stat = os.lstat(candidate)
    if not stat.S_ISREG(value_stat.st_mode) or stat.S_ISLNK(value_stat.st_mode):
        raise ValueError(f"{field_name} must be a regular non-symlink file: {candidate}")
    return resolved


def _existing_canonical_directory(value: pathlib.Path, field_name: str) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    resolved = candidate.resolve(strict=True)
    if _normalized_path(candidate) != _normalized_path(resolved):
        raise ValueError(f"{field_name} must be canonical: {candidate}")
    value_stat = os.lstat(candidate)
    if not stat.S_ISDIR(value_stat.st_mode) or stat.S_ISLNK(value_stat.st_mode):
        raise ValueError(f"{field_name} must be a non-symlink directory: {candidate}")
    return resolved


def _repository_top_level_module_names(
    entries: tuple[FrozenSourceEntry, ...],
) -> frozenset[str]:
    names: set[str] = set()
    for entry in entries:
        path = pathlib.PurePosixPath(entry.path)
        if (
            path.suffix == ".py"
            and len(path.parts) >= 5
            and path.parts[0] == "packages"
            and path.parts[2] == "src"
            and path.parts[3].isidentifier()
        ):
            names.add(path.parts[3])
    return frozenset(names)


def _is_python_module_name(value: str) -> bool:
    return bool(value) and all(part.isidentifier() for part in value.split("."))


def _is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _module_name_for_repository_path(relative_path: str) -> str:
    parts = pathlib.PurePosixPath(relative_path).parts
    if len(parts) < 5 or parts[0] != "packages" or parts[2] != "src":
        raise ValueError(f"repository module path is outside package source roots: {relative_path}")
    module_parts = list(parts[3:])
    filename = pathlib.PurePosixPath(module_parts[-1])
    if filename.suffix != ".py":
        raise ValueError(f"repository module origin is not Python source: {relative_path}")
    if filename.name == "__init__.py":
        module_parts.pop()
    else:
        module_parts[-1] = filename.stem
    if not module_parts or any(not part.isidentifier() for part in module_parts):
        raise ValueError(f"repository module path cannot map to a Python module: {relative_path}")
    return ".".join(module_parts)


def _normalized_path(value: pathlib.Path) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be non-empty text")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field_name} must be a non-negative integer")
    return value


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink


__all__ = [
    "FrozenSourceEntry",
    "QualificationChildImportBinding",
    "build_qualification_child_import_binding",
    "controlled_child_path",
    "expected_child_sys_path",
    "snapshot_file_backed_module_origins",
    "validate_child_file_backed_module_origin_attestation",
]
