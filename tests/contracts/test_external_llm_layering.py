"""Pin the architectural layering for the external LLM client.

Asserts:

* The canonical implementation lives in ``lifeform-core``
  (so any ``lifeform-*`` wheel can use it without depending
  on protocol-runtime / domain wheels).
* The kernel (``vz-*``) does NOT import the client (R8 / tier
  invariant: external LLM is a lifeform-side concern).
* The ``lifeform-protocol-runtime`` re-export is a back-compat
  shim, NOT a second implementation.
* ``lifeform-service`` exposes the shared instance via
  ``app["external_llm_client"]`` so all verticals can opt into
  the same config / API key / quota.
"""

from __future__ import annotations

import importlib

from lifeform_core import (
    LlmJsonClient,
    OpenAiCompatConfig,
    OpenAiCompatJsonClient,
)
from lifeform_core import external_llm as core_module


def test_canonical_implementation_lives_in_lifeform_core() -> None:
    assert OpenAiCompatJsonClient.__module__.startswith("lifeform_core.")
    assert OpenAiCompatConfig.__module__.startswith("lifeform_core.")


def test_protocol_runtime_re_exports_lifeform_core_class() -> None:
    """Same Python object — back-compat shim, no second copy."""
    from lifeform_protocol_runtime import (
        OpenAiCompatConfig as PrConfig,
        OpenAiCompatJsonClient as PrClient,
    )
    assert PrClient is OpenAiCompatJsonClient
    assert PrConfig is OpenAiCompatConfig


def test_service_layer_imports_from_lifeform_core() -> None:
    """The deployment-layer wrapper now sources the client from core."""
    from lifeform_service import openai_compat_client as svc_module

    src = importlib.util.find_spec(svc_module.__name__).origin
    assert src is not None
    with open(src, encoding="utf-8") as f:
        source = f.read()
    assert "from lifeform_core import" in source
    assert "OpenAiCompatJsonClient" in source


def test_kernel_does_not_import_external_llm() -> None:
    """vz-* kernel must NOT import the lifeform-side LLM client."""
    import pkgutil
    import volvence_zero  # vz wheels share this namespace

    forbidden = {"lifeform_core.external_llm"}
    visited: set[str] = set()
    for module_info in pkgutil.walk_packages(
        volvence_zero.__path__, prefix="volvence_zero."
    ):
        name = module_info.name
        if name in visited:
            continue
        visited.add(name)
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for forbidden_name in forbidden:
            # Module attribute would mean the kernel imported it.
            attr_name = forbidden_name.rsplit(".", 1)[-1]
            assert getattr(mod, attr_name, None) is None or not (
                getattr(mod, attr_name).__module__.startswith("lifeform_core.")
            ), (
                f"kernel module {name} imports {forbidden_name} — "
                "external LLM is a lifeform-side concern"
            )


def test_service_app_exposes_shared_external_llm_client() -> None:
    """app["external_llm_client"] is set (possibly None) so any
    vertical handler can read the shared client without depending
    on the service module."""
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    verticals = discover_verticals()
    if not verticals:
        return  # nothing to test on a bare install
    default_name = next(iter(verticals))
    app = create_app(
        verticals=verticals,
        default_vertical=default_name,
        external_llm_client=None,
    )
    assert "external_llm_client" in app
    assert app["external_llm_client"] is None


def test_get_shared_helper_round_trips() -> None:
    from lifeform_service.app import create_app
    from lifeform_service.openai_compat_client import (
        get_shared_external_llm_client,
    )
    from lifeform_service.verticals import discover_verticals

    verticals = discover_verticals()
    if not verticals:
        return
    default_name = next(iter(verticals))

    sentinel = OpenAiCompatJsonClient(
        OpenAiCompatConfig(
            base_url="https://example.com/v1",
            api_key="test-key",
            model="test-model",
        )
    )
    app = create_app(
        verticals=verticals,
        default_vertical=default_name,
        external_llm_client=sentinel,
    )
    assert get_shared_external_llm_client(app) is sentinel
