"""``MCPServerSpec`` — frozen dataclass describing one external MCP server.

This is the configuration surface a ``BrainConfig`` consumer touches
to declare which MCP servers the lifeform should attach. Everything
on it is validated at construction time so a misconfigured spec
fails before the bridge even tries to spawn the server.

See ``docs/specs/mcp-bridge.md`` § "BrainConfig Extension" for the
canonical contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


_VALID_TRANSPORTS = frozenset({"stdio", "http"})
_VALID_RESTART_POLICIES = frozenset({"never", "on_crash", "always"})


@dataclass(frozen=True)
class MCPServerSpec:
    """Immutable spec for one MCP server connection.

    Attributes:
        name: Stable server id. Used as the prefix for every
            ``AffordanceDescriptor.name`` produced by this server
            (``<name>.<tool_name>``) AND as the lookup key into the
            ``.vzbridge.yaml`` safety manifest. Must be a non-empty
            ASCII identifier-like string (letters / digits /
            underscore / hyphen) so it is safe inside a descriptor
            name.
        transport: ``"stdio"`` (subprocess JSON-RPC, the only
            transport currently implemented) or ``"http"`` (planning
            stub — raises at pool startup if selected).
        command: For stdio transport only. Tuple of argv to spawn
            (e.g. ``("python", "-m", "vz_bundle.server")``). Empty
            tuple is invalid for stdio.
        url: For http transport only. Full URL of the MCP HTTP+SSE
            endpoint.
        env: Extra environment variables to inject into the spawned
            stdio subprocess. Merged on top of ``os.environ`` (this
            mapping wins on conflict). Frozen as a ``dict`` snapshot
            captured at spec construction time.
        safety_manifest_path: Required path (absolute or
            repo-relative) to the per-server ``.vzbridge.yaml``.
            Empty string is invalid — manifest is mandatory per
            R10. The bridge resolves it relative to CWD if
            relative; downstream loaders may add their own anchor.
        autostart: When ``True`` (default), ``MCPClientPool`` will
            spawn the server on first reference. When ``False``, the
            caller is responsible for ``pool.ensure_started(spec)``.
        restart_policy: How the pool reacts to subprocess exit.
            ``"never"`` — leave the server down; affordances stay
            blocked until the operator restarts manually.
            ``"on_crash"`` (default) — restart only if exit code is
            non-zero. ``"always"`` — restart on any exit.
        call_timeout_seconds: Per-RPC timeout. Each ``tools/call``,
            ``resources/read``, ``prompts/get`` call must return
            within this many seconds or the bridge raises
            ``MCPCallTimeoutError``. Default 30.0; set lower for
            interactive tools, higher for long-running computations.
        enable_resources: When ``False``, skip ``resources/list`` /
            ``resources/read`` discovery. Use this to register only
            the server's tools without ingesting its resources.
        enable_prompts: When ``True``, ingest the server's prompts
            as low-confidence reviewed knowledge events. Default
            ``False`` because prompts are a less stable MCP surface
            and most servers do not expose useful prompts yet.
    """

    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: tuple[str, ...] = ()
    url: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    safety_manifest_path: str = ""
    autostart: bool = True
    restart_policy: Literal["never", "on_crash", "always"] = "on_crash"
    call_timeout_seconds: float = 30.0
    enable_resources: bool = True
    enable_prompts: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError(
                "MCPServerSpec.name must be a non-empty string."
            )
        if not _is_safe_server_name(self.name):
            raise ValueError(
                f"MCPServerSpec.name={self.name!r} contains characters "
                f"that are not allowed in an affordance descriptor name "
                f"prefix; allowed: ASCII letters / digits / underscore / "
                f"hyphen."
            )
        if self.transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"MCPServerSpec.transport must be one of "
                f"{sorted(_VALID_TRANSPORTS)!r}; got {self.transport!r}."
            )
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                f"MCPServerSpec(name={self.name!r}): stdio transport "
                f"requires a non-empty 'command' tuple."
            )
        if self.transport == "http" and not self.url.strip():
            raise ValueError(
                f"MCPServerSpec(name={self.name!r}): http transport "
                f"requires a non-empty 'url'."
            )
        if not self.safety_manifest_path or not self.safety_manifest_path.strip():
            raise ValueError(
                f"MCPServerSpec(name={self.name!r}): safety_manifest_path "
                f"is required (R10: safety is never derived from MCP "
                f"server alone)."
            )
        if self.restart_policy not in _VALID_RESTART_POLICIES:
            raise ValueError(
                f"MCPServerSpec.restart_policy must be one of "
                f"{sorted(_VALID_RESTART_POLICIES)!r}; got "
                f"{self.restart_policy!r}."
            )
        if self.call_timeout_seconds <= 0:
            raise ValueError(
                f"MCPServerSpec.call_timeout_seconds must be > 0; "
                f"got {self.call_timeout_seconds!r}."
            )
        # Freeze env into a true mapping to defend against caller
        # passing a mutable dict and later mutating it.
        if not isinstance(self.env, Mapping):
            raise TypeError(
                f"MCPServerSpec.env must be a Mapping; got "
                f"{type(self.env).__name__}."
            )
        for key, value in dict(self.env).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    f"MCPServerSpec.env keys and values must be strings; "
                    f"got key={key!r} ({type(key).__name__}) value="
                    f"{value!r} ({type(value).__name__})."
                )


def _is_safe_server_name(name: str) -> bool:
    """Server name must be safe to embed in an AffordanceDescriptor name.

    Allowed: ASCII letters, digits, underscore, hyphen. We
    explicitly forbid '.' so descriptors keyed with the
    ``<server>.<tool>`` convention have an unambiguous boundary.
    """
    if not name:
        return False
    return all(
        ch.isalnum() or ch in {"_", "-"} for ch in name
    )


__all__ = ["MCPServerSpec"]
