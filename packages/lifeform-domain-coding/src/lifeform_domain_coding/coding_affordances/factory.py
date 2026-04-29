"""Factory helpers that wire descriptors + backends together.

Typical host usage:

    from lifeform_domain_coding import (
        build_coding_affordance_invoker,
        build_coding_lifeform,
    )

    lifeform = build_coding_lifeform()
    session = lifeform.create_session()
    invoker = build_coding_affordance_invoker(sandbox_root="/path/to/workspace")
    result = await invoker.invoke(
        "read_file",
        {"path": "src/main.py"},
        session=session.brain_session,
        event_id="call-001",
        granted_consents=frozenset({"filesystem_read"}),
        active_regime_id="problem_solving",
    )
"""

from __future__ import annotations

import pathlib

from lifeform_affordance import (
    AffordanceInvoker,
    AffordanceRegistry,
    BoundaryPolicy,
)

from lifeform_domain_coding.coding_affordances.backends import (
    build_coding_affordance_backends,
)
from lifeform_domain_coding.coding_affordances.descriptors import (
    CODING_AFFORDANCE_DESCRIPTORS,
)


def build_coding_affordance_registry(
    *, seal: bool = True,
) -> AffordanceRegistry:
    """Return an ``AffordanceRegistry`` populated with the coding
    vertical's descriptors.

    ``seal=True`` (default) locks the registry so mid-session
    registrations raise. Hosts that explicitly want to extend the
    registry at runtime pass ``seal=False``; they become responsible
    for calling ``registry.seal()`` when they are done.
    """
    registry = AffordanceRegistry()
    registry.register_all(CODING_AFFORDANCE_DESCRIPTORS)
    if seal:
        registry.seal()
    return registry


def build_coding_affordance_invoker(
    *,
    sandbox_root: pathlib.Path | str,
    registry: AffordanceRegistry | None = None,
    boundary_policy: BoundaryPolicy | None = None,
) -> AffordanceInvoker:
    """Build an ``AffordanceInvoker`` with coding backends bound to ``sandbox_root``.

    Construction steps:

    1. If no ``registry`` supplied, build a freshly-sealed one with
       ``CODING_AFFORDANCE_DESCRIPTORS``. Passing in a registry lets
       hosts register their own additional affordances side-by-side.
    2. Validate ``sandbox_root`` exists and is a directory (fail-loud
       at construction; better than discovering it on the first
       invocation).
    3. Build backends closed over the resolved sandbox_root.
    4. Register each backend on the invoker.

    The returned invoker is ready for ``invoke(...)`` calls. The
    caller is responsible for supplying ``granted_consents`` (must
    include ``filesystem_read``) and ``active_regime_id`` per call.
    """
    effective_registry = (
        registry if registry is not None else build_coding_affordance_registry()
    )
    invoker = AffordanceInvoker(
        registry=effective_registry,
        boundary_policy=boundary_policy,
    )
    backends = build_coding_affordance_backends(sandbox_root)
    for name, backend in backends.items():
        if name in effective_registry:
            invoker.register_backend(name, backend)
    return invoker


__all__ = [
    "build_coding_affordance_invoker",
    "build_coding_affordance_registry",
]
