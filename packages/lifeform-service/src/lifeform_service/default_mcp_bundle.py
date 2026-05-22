"""Default external MCP bundle wiring for the service layer.

The main repo carries ``external/vz-bundle`` as a submodule. The
service attaches it to every newly-created lifeform unless that
lifeform already declares explicit ``mcp_server_specs``. Missing
submodule contents are treated as "not installed in this deployment";
once a spec is emitted, the MCP bridge itself still fail-louds on
manifest / server errors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from lifeform_core import Lifeform
from lifeform_mcp_bridge import MCPServerSpec


_DISABLE_ENV = "VZ_DISABLE_DEFAULT_MCP_BUNDLE"
_SERVER_NAME = "vz-bundle"
_BUNDLE_RELATIVE_PATH = Path("external") / "vz-bundle"


def with_default_mcp_bundle(lifeform: Lifeform) -> Lifeform:
    """Attach the default ``external/vz-bundle`` spec when appropriate."""

    if lifeform.config.mcp_server_specs:
        return lifeform
    specs = default_mcp_server_specs()
    if not specs:
        return lifeform
    return lifeform.with_mcp_server_specs(specs)


def default_mcp_server_specs() -> tuple[MCPServerSpec, ...]:
    """Return the default service MCP specs for this checkout."""

    if _default_bundle_disabled():
        return ()
    bundle_root = _default_bundle_root()
    if bundle_root is None:
        return ()
    manifest_path = bundle_root / ".vzbridge.yaml"
    src_path = bundle_root / "src"
    env = _bundle_subprocess_env(src_path=src_path)
    return (
        MCPServerSpec(
            name=_SERVER_NAME,
            transport="stdio",
            command=(sys.executable, "-m", "vz_bundle.server"),
            env=env,
            safety_manifest_path=str(manifest_path),
            autostart=True,
        ),
    )


def _default_bundle_disabled() -> bool:
    raw = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _default_bundle_root() -> Path | None:
    for root in _candidate_repo_roots():
        candidate = root / _BUNDLE_RELATIVE_PATH
        if (candidate / ".vzbridge.yaml").is_file() and (candidate / "src").is_dir():
            return candidate
    return None


def _candidate_repo_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    cwd = Path.cwd().resolve()
    roots.append(cwd)
    for parent in Path(__file__).resolve().parents:
        roots.append(parent)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return tuple(deduped)


def _bundle_subprocess_env(*, src_path: Path) -> dict[str, str]:
    env: dict[str, str] = {
        "VZ_BUNDLE_SANDBOX_ROOT": str(Path.cwd().resolve()),
    }
    pythonpath = os.environ.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{str(src_path)}{os.pathsep}{pythonpath}" if pythonpath else str(src_path)
    )
    return env


__all__ = [
    "default_mcp_server_specs",
    "with_default_mcp_bundle",
]
