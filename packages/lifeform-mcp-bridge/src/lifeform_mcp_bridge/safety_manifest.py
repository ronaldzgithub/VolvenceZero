"""``.vzbridge.yaml`` safety manifest — bridge-side compatibility facade.

The schema (:class:`SafetyManifest` / :class:`SafetyManifestEntry`), the
validation SSOT (:func:`build_safety_manifest`), and the file loader
(:func:`load_safety_manifest`) now live in ``vz-contracts``
(``volvence_zero.mcp_safety_manifest``) so the platform tier can consume
the same schema without reaching across the wheel boundary into this
lifeform-side wheel. See that module + ``docs/specs/mcp-bridge.md``.

This module keeps the historical bridge surface stable:

* re-exports the schema dataclasses,
* exposes :func:`load_manifest`, which wraps the contracts loader and
  re-raises the tier-neutral :class:`SafetyManifestSchemaError` as the
  bridge's :class:`MCPSafetyManifestSchemaError` (an ``MCPBridgeError``
  subclass) so bridge bringup code can still ``except MCPBridgeError``.
"""

from __future__ import annotations

import os

from volvence_zero.mcp_safety_manifest import (
    SafetyManifest,
    SafetyManifestEntry,
    SafetyManifestSchemaError,
    build_safety_manifest,
    load_safety_manifest,
)

from lifeform_mcp_bridge.errors import MCPSafetyManifestSchemaError


def load_manifest(
    *, path: str | os.PathLike[str], expected_server_name: str
) -> SafetyManifest:
    """Read + validate the ``.vzbridge.yaml`` at ``path``.

    Thin wrapper over :func:`volvence_zero.mcp_safety_manifest.load_safety_manifest`
    that re-raises the tier-neutral :class:`SafetyManifestSchemaError`
    as :class:`MCPSafetyManifestSchemaError` so callers that handle any
    bridge failure via ``except MCPBridgeError`` keep working.
    """
    try:
        return load_safety_manifest(
            path=path, expected_server_name=expected_server_name
        )
    except SafetyManifestSchemaError as exc:
        raise MCPSafetyManifestSchemaError(str(exc)) from exc


__all__ = [
    "SafetyManifest",
    "SafetyManifestEntry",
    "build_safety_manifest",
    "load_manifest",
]
