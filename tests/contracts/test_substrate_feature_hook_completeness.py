"""Substrate feature hook completeness contract test (B2 阶段 1).

Architecture-uplift B2 (see [`docs/moving forward/experiment-arch-uplift.md`](../../docs/moving%20forward/experiment-arch-uplift.md)
§3 B2). COG-3 (Persona / Regime geometry readout) depends on
``SubstrateSnapshot.feature_surface`` + ``residual_activations`` being
populated by the active substrate backend.

**Phase 1 (this T13 packet)**: substrate-side fields are marked
*recommended*, not yet abstract. This contract test serves as a
visibility layer:

- Every SubstrateAdapter subclass discovered in the kernel is enumerated
- For each, we record whether its ``SurfaceKind`` is permitted to emit
  empty ``feature_surface`` / ``residual_activations`` (placeholder) or
  is expected to populate them (production)
- Production adapters with empty hook fields produce a recorded gap (NOT
  yet a FAIL, just a snapshot of where we stand)

The "promote to FAIL" switch is documented inline; flipping it is the
B2 阶段 2 work.
"""

from __future__ import annotations

import pytest

from volvence_zero.substrate.adapter import (
    FeatureSurfaceSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SubstrateAdapter,
    SurfaceKind,
)


# B2 阶段 2 promotion switch: flip to True when every production adapter
# populates both hook fields. Until then, gaps surface as informational
# messages, not test failures.
_FAIL_ON_PRODUCTION_GAPS = False

# SurfaceKind values that may legitimately publish empty hook fields.
_PLACEHOLDER_SURFACES: set[SurfaceKind] = {SurfaceKind.PLACEHOLDER}


def _all_substrate_adapter_subclasses() -> tuple[type, ...]:
    seen: set[type] = set()
    to_visit: list[type] = list(SubstrateAdapter.__subclasses__())
    while to_visit:
        cls = to_visit.pop()
        if cls in seen:
            continue
        seen.add(cls)
        to_visit.extend(cls.__subclasses__())
    return tuple(sorted(seen, key=lambda c: c.__name__))


def test_substrate_adapter_subclass_enumeration_is_non_empty() -> None:
    """Sanity: at least the placeholder + feature-surface adapters exist."""
    subclasses = _all_substrate_adapter_subclasses()
    names = {cls.__name__ for cls in subclasses}
    assert "PlaceholderSubstrateAdapter" in names
    assert "FeatureSurfaceSubstrateAdapter" in names


def test_placeholder_adapter_emits_unavailable_fields_marker() -> None:
    """The placeholder adapter must explicitly mark its empty hooks as
    unavailable — silent empty fields would mask backend bugs."""
    import asyncio

    adapter = PlaceholderSubstrateAdapter(model_id="test-placeholder")
    snapshot = asyncio.run(adapter.capture(source_text=None))

    assert snapshot.feature_surface == ()
    assert snapshot.residual_activations == ()
    declared_unavailable = {u.field_name for u in snapshot.unavailable_fields}
    assert "feature_surface" in declared_unavailable
    assert "residual_activations" in declared_unavailable


def test_feature_surface_adapter_populates_feature_surface() -> None:
    """Production feature-surface adapter must fill feature_surface even
    when residual_activations stays empty."""
    import asyncio

    from volvence_zero.substrate.adapter import FeatureSignal

    adapter = FeatureSurfaceSubstrateAdapter(
        model_id="test-feature-surface",
        feature_surface=(
            FeatureSignal(name="signal_a", values=(0.0,), source="test"),
        ),
    )
    snapshot = asyncio.run(adapter.capture(source_text=None))
    assert len(snapshot.feature_surface) == 1
    assert snapshot.surface_kind is not SurfaceKind.PLACEHOLDER


def test_production_adapter_hook_population_report() -> None:
    """Diagnostic test: report which production adapters skip the feature
    hook fields. Currently informational; will FAIL after B2 阶段 2
    promotion flip (``_FAIL_ON_PRODUCTION_GAPS = True``)."""
    import asyncio

    gaps: list[str] = []
    for cls in _all_substrate_adapter_subclasses():
        # PlaceholderSubstrateAdapter is explicitly allowed empty hooks.
        if cls is PlaceholderSubstrateAdapter:
            continue
        # Skip abstract base.
        if cls is SubstrateAdapter:
            continue
        # We cannot generically construct every subclass without knowing
        # its required ctor args (e.g. residual backend objects). Skip
        # subclasses we cannot smoke-instantiate; B2 阶段 2 will replace
        # this with a structured factory.
        try:
            adapter = cls(  # type: ignore[call-arg]
                model_id=f"smoke-{cls.__name__}",
                feature_surface=(),
            )
        except TypeError:
            continue

        snapshot = asyncio.run(adapter.capture(source_text=None))
        if snapshot.surface_kind in _PLACEHOLDER_SURFACES:
            continue

        has_features = bool(snapshot.feature_surface)
        has_residuals = bool(snapshot.residual_activations)
        # Production adapters should at least fill feature_surface; residual
        # backends fill both.
        if not has_features:
            gaps.append(
                f"{cls.__name__} (surface={snapshot.surface_kind.value}): "
                f"feature_surface empty"
            )

    if gaps and _FAIL_ON_PRODUCTION_GAPS:
        pytest.fail(
            "Production substrate adapters have empty feature_surface hooks:\n"
            + "\n".join(f"  - {g}" for g in gaps)
        )
    # Informational mode: still record the report in test output for visibility.
    if gaps:
        pytest.skip(
            "B2 阶段 1 (informational): production adapters with empty hooks:\n"
            + "\n".join(f"  - {g}" for g in gaps)
        )
