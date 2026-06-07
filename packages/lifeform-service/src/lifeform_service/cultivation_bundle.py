"""Portable cultivation cognition bundle (R8 SSOT + R15 portability).

When an autonomously-cultivated expert converges, its *learned* school is
the set of approved :class:`BehaviorProtocol` instances the kernel formed
(Identity Core + researched theory protocols) — NOT just the persona
metadata recorded on the template. This module serialises that converged
set into a portable, lossless envelope so it can be carried on the
inducted DLaaS template's ``seed_config`` and re-hydrated into any
adopting instance.

Reflow contract
---------------

* **induct -> freeze**: at graduation the bundle is captured from the
  source ai_id's approved protocol set (a reviewed-frozen snapshot).
* **adopt -> hydrate**: on wake of a template carrying the bundle, the
  protocols seed the adopting instance's :class:`ProtocolUptakeService`
  via :meth:`InstanceManager.set_protocol_uptake_service`. The adopted
  instance *starts* from the cultivated school AND keeps learning online
  (its own per-session α/β PE mixing + a fresh ``revision_log``).
* **re-induct -> new version**: the adopted runtime never writes back to
  the published bundle (so the bundle is not a second cognition owner);
  improving the published expert means re-inducting, which produces a new
  template version with a new bundle (R15 rollback via versioned
  templates).

Ownership boundary (R8): this module owns only the *envelope* shape. The
per-protocol serialisation contract is owned by
``lifeform_protocol_runtime.protocol_to_payload`` / ``protocol_from_payload``;
the in-memory registry is owned by :class:`ProtocolUptakeService`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lifeform_protocol_runtime import (
    protocol_from_payload,
    protocol_to_payload,
)
from lifeform_service.openai_compat_client import build_client_from_env
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from volvence_zero.behavior_protocol import BehaviorProtocol
from volvence_zero.protocol_runtime import ProtocolRegistry

CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION = "cultivation.protocol_bundle.v1"


def export_protocol_bundle(
    protocols: Iterable[BehaviorProtocol],
    *,
    source_ai_id: str,
    cultivation_id: str = "",
    package_id: str = "",
    track_id: str = "",
) -> dict[str, Any]:
    """Serialise the converged approved protocol set into a portable bundle.

    ``protocols`` is the source instance's approved set (typically
    ``ProtocolUptakeService.loaded_approved_snapshot()``). Each protocol
    is round-tripped through the canonical ``protocol_to_payload`` so the
    bundle is lossless. Provenance fields tie the bundle back to the
    cultivation it came from.
    """

    payloads = [protocol_to_payload(p) for p in protocols]
    return {
        "schema_version": CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION,
        "source_ai_id": source_ai_id,
        "cultivation_id": cultivation_id,
        "package_id": package_id,
        "track_id": track_id,
        "protocol_count": len(payloads),
        "protocols": payloads,
    }


def read_protocol_bundle(payload: Any) -> tuple[BehaviorProtocol, ...]:
    """Deserialise a bundle envelope back into BehaviorProtocols.

    Fails loudly (``ValueError``) on a malformed envelope or schema
    mismatch — a contract violation must not be silently swallowed.
    """

    if not isinstance(payload, dict):
        raise ValueError(
            "read_protocol_bundle: bundle payload must be an object, got "
            f"{type(payload).__name__}"
        )
    schema_version = payload.get("schema_version")
    if schema_version != CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "read_protocol_bundle: unsupported schema_version "
            f"{schema_version!r} (expected "
            f"{CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION!r})"
        )
    raw = payload.get("protocols")
    if not isinstance(raw, list):
        raise ValueError(
            "read_protocol_bundle: 'protocols' must be a list, got "
            f"{type(raw).__name__}"
        )
    return tuple(protocol_from_payload(item) for item in raw)


def build_uptake_service_from_bundle(
    payload: Any,
    *,
    llm_client_factory: Any = None,
) -> ProtocolUptakeService:
    """Hydrate a :class:`ProtocolUptakeService` from a bundle envelope.

    Loads every protocol in the bundle into a fresh registry so the
    adopting instance's SessionManager seeds them into each session (and
    continues online α/β PE mixing). The ``llm_client_factory`` defaults
    to ``build_client_from_env`` so the hydrated service can still extract
    new protocols later if the deployment configures an LLM.
    """

    protocols = read_protocol_bundle(payload)
    registry = ProtocolRegistry()
    for protocol in protocols:
        registry.load(protocol)
    return ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            llm_client_factory=llm_client_factory or build_client_from_env,
        ),
        registry=registry,
    )


__all__ = [
    "CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION",
    "build_uptake_service_from_bundle",
    "export_protocol_bundle",
    "read_protocol_bundle",
]
