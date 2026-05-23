"""Contract test for U2 — adopt path binds bundle per ai_id.

The family-memorial product depends on multiple memorials living in
the same dlaas-platform process. Each memorial has its own ``ai_id``
and its own baked ``FigureArtifactBundle``. Without per-``ai_id``
binding, every memorial would inherit the process-wide default
bundle (Einstein) and citations / refusals would cross-contaminate
across families.

The DLaaS adopt path now calls
``instance_manager.get(ai_id).bind_figure_bundle(bundle)`` after
``lookup_figure_bundle`` returns. This test locks the load-bearing
invariant of that call: **the SAME process can hold two ai_ids with
two distinct bound bundles, and querying one's ``figure_bundle``
does NOT return the other's.**

We exercise :class:`InstanceManager` directly rather than spinning
up the full DLaaS HTTP server — the adopt handler is a thin caller
of :meth:`InstanceManager.acquire` followed by
:meth:`SessionManager.bind_figure_bundle`, both of which are typed
public methods. The full HTTP-level smoke runs in
``tests/service/test_dlaas_chat_smoke.py``; this contract guards the
specific wiring change.
"""

from __future__ import annotations

import pytest

from dlaas_platform_launcher.instance_manager import (
    InstanceManager,
    default_vertical_resolver,
)
from lifeform_service import (
    FigureBundleStore,
    lookup_figure_bundle,
)


class _FakeMemorialBundle:
    def __init__(self, bundle_id: str, figure_id: str) -> None:
        self.bundle_id = bundle_id
        self.figure_id = figure_id


@pytest.mark.asyncio
async def test_two_ai_ids_bind_distinct_bundles_no_crosstalk() -> None:
    """Acquire two ai_ids on the einstein vertical, then bind a
    DIFFERENT bundle to each via the SessionManager that
    InstanceManager.get returns. Each ai_id must independently report
    its own bundle on `.figure_bundle`."""

    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_alpha", runtime_template_id="einstein-bundle"
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_beta", runtime_template_id="einstein-bundle"
    )

    # Resolve the default-shipped Einstein bundle as bundle A. Use a
    # fake "family memorial" bundle for B so we can be sure they are
    # distinct objects with distinct bundle_ids.
    bundle_a = lookup_figure_bundle(bundle_id="einstein")
    bundle_b = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_grandpa:0123456789abcdef",
        figure_id="family_grandpa",
    )

    manager_a = instance_manager.get("memorial_ai_alpha")
    manager_b = instance_manager.get("memorial_ai_beta")

    assert manager_a is not manager_b
    assert manager_a.figure_bundle is None
    assert manager_b.figure_bundle is None

    manager_a.bind_figure_bundle(bundle_a)
    manager_b.bind_figure_bundle(bundle_b)

    # Key invariant: per-ai_id binding does not leak.
    assert manager_a.figure_bundle is bundle_a
    assert manager_b.figure_bundle is bundle_b
    assert manager_a.figure_bundle is not bundle_b
    assert manager_b.figure_bundle is not bundle_a


@pytest.mark.asyncio
async def test_rebinding_an_ai_id_does_not_affect_the_other() -> None:
    """Rebinding a bundle on ai_id_A (e.g. after a re-bake) must not
    drag ai_id_B along. Critical for "re-bake a memorial without
    bouncing the platform process" workflows."""

    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_alpha", runtime_template_id="einstein-bundle"
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_beta", runtime_template_id="einstein-bundle"
    )

    bundle_a_v1 = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_a:v1aaaaaaaaaaaaaa", figure_id="family_a"
    )
    bundle_a_v2 = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_a:v2bbbbbbbbbbbbbb", figure_id="family_a"
    )
    bundle_b = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_b:0000000000000000", figure_id="family_b"
    )

    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(bundle_a_v1)
    instance_manager.get("memorial_ai_beta").bind_figure_bundle(bundle_b)

    # Re-bind ai_id_A to a new bundle version. ai_id_B must be untouched.
    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(bundle_a_v2)

    assert instance_manager.get("memorial_ai_alpha").figure_bundle is bundle_a_v2
    assert instance_manager.get("memorial_ai_beta").figure_bundle is bundle_b


@pytest.mark.asyncio
async def test_unbinding_one_ai_id_does_not_affect_others() -> None:
    """Passing ``None`` clears one ai_id's bundle without affecting siblings."""

    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_alpha", runtime_template_id="einstein-bundle"
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_beta", runtime_template_id="einstein-bundle"
    )
    bundle_a = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_a:aaaa", figure_id="family_a"
    )
    bundle_b = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_b:bbbb", figure_id="family_b"
    )
    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(bundle_a)
    instance_manager.get("memorial_ai_beta").bind_figure_bundle(bundle_b)

    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(None)

    assert instance_manager.get("memorial_ai_alpha").figure_bundle is None
    assert instance_manager.get("memorial_ai_beta").figure_bundle is bundle_b
