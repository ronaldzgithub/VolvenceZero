"""``.vzbridge.yaml`` safety-manifest schema + validation SSOT.

This is the cross-tier contract for the per-server MCP / plugin safety
manifest. It lives in ``vz-contracts`` (the foundation wheel) so that
**both** the lifeform-side bridge (``lifeform-mcp-bridge``) and the
platform tier (``dlaas-platform-*``) can consume one canonical schema +
one validation implementation, instead of the platform reaching across
the wheel boundary into a lifeform implementation wheel (which inverted
the one-way ``platform -> lifeform -> kernel`` dependency direction and
broke the split-ready promise; see
``tests/contracts/test_import_boundaries.py``).

Per ``docs/specs/mcp-bridge.md`` § "Safety Manifest Schema", every
external MCP server / HTTP plugin **must** ship a reviewed YAML manifest
declaring ``safety_model`` / ``cost_model`` / ``when_to_use`` /
``when_not_to_use`` / ``affordance_tags`` for each tool the server
exposes. Without an entry, the bridge refuses to register the tool
(R10: safety must never default to "unsafe-but-quiet").

Schema version 1 layout::

    schema_version: 1
    server:
      name: "<MCPServerSpec.name>"
      description: "..."
    tools:
      - name: "<tool_name>"
        when_to_use: |
          ... >= 50 chars ...
        when_not_to_use: |
          ... >= 50 chars ...
        cost_model:
          latency_class: instant | fast | slow | very_slow
          monetary_class: free | low | medium | high
          rate_limit_per_minute: <int|null>
        safety_model:
          requires_user_confirmation: <bool>
          irreversible: <bool>
          requires_consent_grant: [<str>, ...]
          blocked_in_regimes: [<str>, ...]
          audit_required: <bool>
        affordance_tags: [<str>, ...]
        excluded: <bool>           # optional, default false
    resources:
      default_compliance_profile: forced | consultative
    prompts:
      enabled: <bool>

Dependency note: this module imports ``yaml`` **inside**
:func:`load_safety_manifest` (deferred), so importing the schema /
:func:`build_safety_manifest` validation surface keeps ``vz-contracts``
free of any hard third-party dependency. Callers that use the
file-loading convenience (:func:`load_safety_manifest`) must have
``pyyaml`` installed; the bridge and ``dlaas-platform-api`` already
declare it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from volvence_zero.affordance import (
    MIN_SELECTION_HINT_CHARS,
    AffordanceCost,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)

_SCHEMA_VERSION = 1
_VALID_LATENCY = {item.value for item in AffordanceLatencyClass}
_VALID_MONETARY = {item.value for item in AffordanceMonetaryClass}
_VALID_COMPLIANCE = {"forced", "consultative"}


class SafetyManifestSchemaError(Exception):
    """The ``.vzbridge.yaml`` file is structurally invalid.

    Canonical, tier-neutral schema error raised by
    :func:`build_safety_manifest` / :func:`load_safety_manifest`.

    Typical causes: ``schema_version`` mismatch, missing required field
    on a tool entry, ``when_to_use`` < 50 chars, illegal enum value for
    ``cost_model.latency_class``, server name does not match the
    ``MCPServerSpec.name`` configured on the BrainConfig.

    The error message includes the offending key path so the external
    repo author can fix the YAML directly. The lifeform-side bridge
    re-exports a subclass (``MCPSafetyManifestSchemaError``) that also
    inherits ``MCPBridgeError`` so bridge bringup code can catch any
    bridge failure uniformly.
    """


@dataclass(frozen=True)
class SafetyManifestEntry:
    """One per-tool entry from the manifest, validated.

    Bridge-side counterpart to ``AffordanceDescriptor`` invariants:
    ``when_to_use`` / ``when_not_to_use`` already enforced >=
    ``MIN_SELECTION_HINT_CHARS`` here so the descriptor build cannot
    fail later on the same invariant (would be a wasted RPC).
    """

    tool_name: str
    when_to_use: str
    when_not_to_use: str
    cost_model: AffordanceCost
    safety_model: AffordanceSafety
    affordance_tags: tuple[str, ...] = ()
    excluded: bool = False


@dataclass(frozen=True)
class SafetyManifest:
    """Loaded + validated ``.vzbridge.yaml``.

    Holds:

    * ``server_name`` — must equal the ``MCPServerSpec.name`` the
      caller is loading the manifest for.
    * ``tool_entries`` — keyed by ``tool_name`` for O(1) lookup
      during ``MCPAffordanceAdapter.fetch_tools``.
    * ``resources_default_compliance_profile`` — applied to ingestion
      envelopes the resource adapter emits.
    * ``prompts_enabled`` — gates the prompt adapter at the manifest
      level (in addition to ``MCPServerSpec.enable_prompts``).
    """

    server_name: str
    server_description: str
    tool_entries: Mapping[str, SafetyManifestEntry]
    resources_default_compliance_profile: str
    prompts_enabled: bool
    manifest_path: str

    def lookup(self, tool_name: str) -> SafetyManifestEntry | None:
        """Return the manifest entry for ``tool_name`` or ``None``.

        Callers MUST treat ``None`` as a missing-manifest condition;
        this helper does not raise so call sites can attach richer
        context.
        """
        return self.tool_entries.get(tool_name)


def load_safety_manifest(
    *, path: str | os.PathLike[str], expected_server_name: str
) -> SafetyManifest:
    """Read + validate the ``.vzbridge.yaml`` at ``path``.

    Raises :class:`SafetyManifestSchemaError` on:

    * missing file / unreadable yaml
    * ``schema_version`` mismatch
    * ``server.name`` not equal to ``expected_server_name``
    * any tool entry missing required field
    * ``when_to_use`` / ``when_not_to_use`` < 50 chars
    * illegal enum value for cost_model

    ``pyyaml`` is imported lazily so importing this module does not pull
    a third-party dependency into ``vz-contracts``.
    """
    import yaml

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SafetyManifestSchemaError(
            f"safety manifest not found at {str(manifest_path)!r}; "
            f"every MCPServerSpec MUST point at a reviewed "
            f".vzbridge.yaml file."
        )
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SafetyManifestSchemaError(
            f"could not read safety manifest {str(manifest_path)!r}: {exc}"
        ) from exc
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SafetyManifestSchemaError(
            f"safety manifest {str(manifest_path)!r} is not valid YAML: "
            f"{exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise SafetyManifestSchemaError(
            f"safety manifest {str(manifest_path)!r}: top-level YAML "
            f"document must be a mapping; got "
            f"{type(document).__name__}."
        )
    return build_safety_manifest(
        document=document,
        path=str(manifest_path),
        expected_server_name=expected_server_name,
    )


def build_safety_manifest(
    *,
    document: Mapping[str, Any],
    path: str,
    expected_server_name: str,
) -> SafetyManifest:
    """Validate an already-parsed manifest mapping into a SafetyManifest.

    Pure (no file IO, no yaml) so it is the single validation SSOT for
    every caller regardless of where the bytes came from.
    """
    schema_version = document.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: schema_version must be "
            f"{_SCHEMA_VERSION}; got {schema_version!r}."
        )
    server_block = document.get("server")
    if not isinstance(server_block, Mapping):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: 'server' block missing or not a "
            f"mapping."
        )
    server_name = server_block.get("name", "")
    if not isinstance(server_name, str) or not server_name.strip():
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: server.name must be a non-empty "
            f"string."
        )
    if server_name != expected_server_name:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: server.name={server_name!r} does "
            f"not match the BrainConfig MCPServerSpec.name="
            f"{expected_server_name!r}. The two must agree so an operator "
            f"cannot accidentally apply one server's safety model to "
            f"another's tools."
        )
    server_description = str(server_block.get("description", ""))
    tools_block = document.get("tools", [])
    if not isinstance(tools_block, list):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: 'tools' must be a list."
        )
    tool_entries: dict[str, SafetyManifestEntry] = {}
    for index, raw_tool in enumerate(tools_block):
        if not isinstance(raw_tool, Mapping):
            raise SafetyManifestSchemaError(
                f"safety manifest {path!r}: tools[{index}] must be a "
                f"mapping; got {type(raw_tool).__name__}."
            )
        entry = _build_tool_entry(raw_tool, path=path, index=index)
        if entry.tool_name in tool_entries:
            raise SafetyManifestSchemaError(
                f"safety manifest {path!r}: duplicate tool entry for "
                f"name={entry.tool_name!r} (tools[{index}])."
            )
        tool_entries[entry.tool_name] = entry
    resources_block = document.get("resources", {}) or {}
    if not isinstance(resources_block, Mapping):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: 'resources' must be a mapping."
        )
    compliance = resources_block.get(
        "default_compliance_profile", "forced"
    )
    if compliance not in _VALID_COMPLIANCE:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: "
            f"resources.default_compliance_profile must be one of "
            f"{sorted(_VALID_COMPLIANCE)!r}; got {compliance!r}."
        )
    prompts_block = document.get("prompts", {}) or {}
    if not isinstance(prompts_block, Mapping):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: 'prompts' must be a mapping."
        )
    prompts_enabled = bool(prompts_block.get("enabled", False))
    return SafetyManifest(
        server_name=server_name,
        server_description=server_description,
        tool_entries=tool_entries,
        resources_default_compliance_profile=str(compliance),
        prompts_enabled=prompts_enabled,
        manifest_path=path,
    )


def _build_tool_entry(
    raw_tool: Mapping[str, Any],
    *,
    path: str,
    index: int,
) -> SafetyManifestEntry:
    tool_name = raw_tool.get("name", "")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}].name must be a "
            f"non-empty string."
        )
    when_to_use = raw_tool.get("when_to_use", "")
    when_not_to_use = raw_tool.get("when_not_to_use", "")
    if not isinstance(when_to_use, str) or len(when_to_use) < MIN_SELECTION_HINT_CHARS:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] "
            f"({tool_name!r}).when_to_use must be a string >= "
            f"{MIN_SELECTION_HINT_CHARS} chars; got "
            f"len={len(when_to_use) if isinstance(when_to_use, str) else 'n/a'}."
        )
    if not isinstance(when_not_to_use, str) or len(when_not_to_use) < MIN_SELECTION_HINT_CHARS:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] "
            f"({tool_name!r}).when_not_to_use must be a string >= "
            f"{MIN_SELECTION_HINT_CHARS} chars; got "
            f"len={len(when_not_to_use) if isinstance(when_not_to_use, str) else 'n/a'}."
        )
    cost_model = _build_cost_model(raw_tool, path=path, index=index, tool_name=tool_name)
    safety_model = _build_safety_model(raw_tool, path=path, index=index, tool_name=tool_name)
    raw_tags = raw_tool.get("affordance_tags", []) or []
    if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".affordance_tags must be a list of strings."
        )
    excluded = bool(raw_tool.get("excluded", False))
    return SafetyManifestEntry(
        tool_name=tool_name,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        cost_model=cost_model,
        safety_model=safety_model,
        affordance_tags=tuple(raw_tags),
        excluded=excluded,
    )


def _build_cost_model(
    raw_tool: Mapping[str, Any],
    *,
    path: str,
    index: int,
    tool_name: str,
) -> AffordanceCost:
    raw = raw_tool.get("cost_model", {})
    if not isinstance(raw, Mapping):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".cost_model must be a mapping."
        )
    latency_class = raw.get("latency_class")
    if latency_class not in _VALID_LATENCY:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".cost_model.latency_class must be one of "
            f"{sorted(_VALID_LATENCY)!r}; got {latency_class!r}."
        )
    monetary_class = raw.get("monetary_class", "free")
    if monetary_class not in _VALID_MONETARY:
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".cost_model.monetary_class must be one of "
            f"{sorted(_VALID_MONETARY)!r}; got {monetary_class!r}."
        )
    rate_limit = raw.get("rate_limit_per_minute")
    if rate_limit is not None:
        if not isinstance(rate_limit, int) or isinstance(rate_limit, bool) or rate_limit <= 0:
            raise SafetyManifestSchemaError(
                f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
                f".cost_model.rate_limit_per_minute must be a positive "
                f"integer or null; got {rate_limit!r}."
            )
    return AffordanceCost(
        latency_class=AffordanceLatencyClass(latency_class),
        monetary_class=AffordanceMonetaryClass(monetary_class),
        rate_limit_per_minute=rate_limit,
    )


def _build_safety_model(
    raw_tool: Mapping[str, Any],
    *,
    path: str,
    index: int,
    tool_name: str,
) -> AffordanceSafety:
    raw = raw_tool.get("safety_model", {}) or {}
    if not isinstance(raw, Mapping):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".safety_model must be a mapping."
        )
    requires_consent_grant = raw.get("requires_consent_grant", []) or []
    if not isinstance(requires_consent_grant, list) or not all(
        isinstance(g, str) for g in requires_consent_grant
    ):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".safety_model.requires_consent_grant must be a list of "
            f"strings."
        )
    blocked_in_regimes = raw.get("blocked_in_regimes", []) or []
    if not isinstance(blocked_in_regimes, list) or not all(
        isinstance(r, str) for r in blocked_in_regimes
    ):
        raise SafetyManifestSchemaError(
            f"safety manifest {path!r}: tools[{index}] ({tool_name!r})"
            f".safety_model.blocked_in_regimes must be a list of "
            f"strings."
        )
    return AffordanceSafety(
        requires_user_confirmation=bool(
            raw.get("requires_user_confirmation", False)
        ),
        irreversible=bool(raw.get("irreversible", False)),
        requires_consent_grant=tuple(requires_consent_grant),
        blocked_in_regimes=tuple(blocked_in_regimes),
        audit_required=bool(raw.get("audit_required", False)),
    )


__all__ = [
    "SafetyManifest",
    "SafetyManifestEntry",
    "SafetyManifestSchemaError",
    "build_safety_manifest",
    "load_safety_manifest",
]
