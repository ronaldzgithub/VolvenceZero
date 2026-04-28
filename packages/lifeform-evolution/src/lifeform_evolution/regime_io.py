"""Persistence for ``RegimeBootstrap`` artifacts.

Sister module of ``snapshot_io.py``. Same magic-header + schema-version
discipline so we can later switch to a portable JSON encoding without
silently loading old files.
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

from volvence_zero.regime import RegimeBootstrap


MAGIC = b"VZ-REGIMEBS\x00"
SCHEMA_VERSION = "vz-regimebs.v1"


@dataclass(frozen=True)
class RegimeBootstrapArtifact:
    schema_version: str
    bootstrap: RegimeBootstrap
    metadata: dict[str, object]


def save_regime_bootstrap(
    bootstrap: RegimeBootstrap,
    path: str | pathlib.Path,
    *,
    metadata: dict[str, object] | None = None,
) -> pathlib.Path:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = RegimeBootstrapArtifact(
        schema_version=SCHEMA_VERSION,
        bootstrap=bootstrap,
        metadata=dict(metadata or {}),
    )
    with out.open("wb") as handle:
        handle.write(MAGIC)
        pickle.dump(artifact, handle)
    return out.resolve()


def load_regime_bootstrap(path: str | pathlib.Path) -> RegimeBootstrapArtifact:
    in_path = pathlib.Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Regime bootstrap artifact not found: {in_path}")
    with in_path.open("rb") as handle:
        header = handle.read(len(MAGIC))
        if header != MAGIC:
            raise ValueError(
                f"File at {in_path} is not a Volvence Zero regime bootstrap "
                f"artifact (missing magic header)."
            )
        artifact = pickle.load(handle)
    if not isinstance(artifact, RegimeBootstrapArtifact):
        raise ValueError(
            f"File at {in_path} did not deserialize to RegimeBootstrapArtifact "
            f"(got {type(artifact).__name__})."
        )
    if artifact.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Regime bootstrap at {in_path} has schema_version="
            f"{artifact.schema_version!r}; this build only loads "
            f"{SCHEMA_VERSION!r}."
        )
    return artifact


def load_regime_bootstrap_only(path: str | pathlib.Path) -> RegimeBootstrap:
    return load_regime_bootstrap(path).bootstrap
