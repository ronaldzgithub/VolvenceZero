"""ACTIVE behaviour for ``ProtocolRegistryModule`` (packet 4.0).

Originally a SHADOW dual-run gate (packets 1.0–1.5a'), this file
now pins the post-promotion invariants for the ``active_mixture``
slot now that the default rollout flipped to ACTIVE:

* ``active_mixture`` lives in ``active_snapshots`` (default).
* Default config still produces a valid snapshot (empty registry
  → empty mixture; loaded fixture → 1 entry, etc.).
* Published value's shape is correct (``ActiveMixtureSnapshot``).
* Loading a fixture does not widen *value* drift on any other
  active slot beyond the baseline run-to-run drift (matched-control
  invariant — proves no kernel module silently took
  ``active_mixture`` as a dependency without a contract test).

The SHADOW behaviour is still exercised explicitly by passing a
``ProtocolRegistryModule(wiring_level=SHADOW)`` instance — see
the ``test_shadow_*`` cases below — which preserves backwards
compatibility for any caller that wants to keep ``protocol_runtime``
quarantined.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
)
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter


_MODEL_ID = "protocol-runtime-shadow-test-model"


def _run_turn(
    *,
    protocol_registry_module: ProtocolRegistryModule | None = None,
) -> dict[str, Any]:
    config = FinalRolloutConfig()
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=PlaceholderSubstrateAdapter(model_id=_MODEL_ID),
            session_id="protocol-runtime-shadow-session",
            wave_id="protocol-runtime-shadow-wave",
            protocol_registry_module=protocol_registry_module,
        )
    )
    return {
        "active": result.active_snapshots,
        "shadow": result.shadow_snapshots,
    }


def _active_mixture_snapshot(
    snapshots: dict[str, Snapshot[Any]],
) -> ActiveMixtureSnapshot:
    snapshot = snapshots["active_mixture"]
    assert isinstance(snapshot.value, ActiveMixtureSnapshot)
    return snapshot.value


# ---------------------------------------------------------------------------
# ACTIVE publishing (packet 4.0 default): active_snapshots
# ---------------------------------------------------------------------------


def test_active_default_publishes_to_active_snapshots_only() -> None:
    snapshots = _run_turn()
    assert "active_mixture" in snapshots["active"], list(snapshots["active"])
    assert "active_mixture" not in snapshots["shadow"], list(snapshots["shadow"])


def test_active_published_value_is_active_mixture_snapshot() -> None:
    snapshots = _run_turn()
    am = _active_mixture_snapshot(snapshots["active"])
    assert isinstance(am, ActiveMixtureSnapshot)


# ---------------------------------------------------------------------------
# Empty registry publishes empty snapshot
# ---------------------------------------------------------------------------


def test_empty_registry_publishes_empty_active_protocols() -> None:
    snapshots = _run_turn()
    am = _active_mixture_snapshot(snapshots["active"])
    assert am.active_protocols == ()
    assert am.boundary_union_ids == ()


def test_empty_registry_fingerprint_is_empty_string() -> None:
    """Stable empty fingerprint so consumers can detect 'not loaded yet'."""
    snapshots = _run_turn()
    am = _active_mixture_snapshot(snapshots["active"])
    assert am.revision_fingerprint == ""


# ---------------------------------------------------------------------------
# Loading cheng_laoshi: registry → active_mixture has 1 entry + 4 boundaries
# ---------------------------------------------------------------------------


def _build_module_with_cheng_laoshi() -> ProtocolRegistryModule:
    """Build an ACTIVE-wired module with cheng_laoshi pre-loaded.

    Packet 4.0: explicitly request ACTIVE so the loaded mixture
    appears in ``active_snapshots`` (matching the production
    default rollout). The module's per-instance wiring_level
    overrides the rollout config's default in
    ``run_final_wiring_turn``.
    """

    module = ProtocolRegistryModule(wiring_level=WiringLevel.ACTIVE)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return module


def test_loaded_protocol_appears_in_active_mixture() -> None:
    module = _build_module_with_cheng_laoshi()
    snapshots = _run_turn(protocol_registry_module=module)
    am = _active_mixture_snapshot(snapshots["active"])
    assert len(am.active_protocols) == 1
    entry = am.active_protocols[0]
    assert isinstance(entry, ActiveProtocolEntry)
    assert entry.protocol_id == "growth_advisor:cheng-laoshi"
    # Equal-weight fallback: single protocol → weight 1.0
    assert entry.activation_weight == 1.0


def test_loaded_protocol_boundary_union_ids_match_protocol_boundaries() -> None:
    """Packet 1.2 (Choice A): the snapshot publishes IDs only.

    Canonical ``BoundaryPriorHint`` content lives in
    ``ApplicationRareHeavyState`` (populated via the protocol
    compile path); this slot publishes references so consumers
    never become a second owner.
    """
    module = _build_module_with_cheng_laoshi()
    snapshots = _run_turn(protocol_registry_module=module)
    am = _active_mixture_snapshot(snapshots["active"])
    # cheng_laoshi has 4 boundary priors → 4 IDs in the union
    assert len(am.boundary_union_ids) == 4
    assert set(am.boundary_union_ids) == {
        "bp-no-hard-sell",
        "bp-no-overclaim",
        "bp-no-flooding",
        "bp-no-judgmental",
    }


def test_loaded_protocol_phase_id_is_placeholder() -> None:
    module = _build_module_with_cheng_laoshi()
    snapshots = _run_turn(protocol_registry_module=module)
    am = _active_mixture_snapshot(snapshots["active"])
    entry = am.active_protocols[0]
    assert entry.current_phase_id == "long_term_companion"


def test_loaded_protocol_fingerprint_is_nonempty() -> None:
    module = _build_module_with_cheng_laoshi()
    snapshots = _run_turn(protocol_registry_module=module)
    am = _active_mixture_snapshot(snapshots["active"])
    assert am.revision_fingerprint != ""
    assert len(am.revision_fingerprint) == 64  # sha256 hex digest


# ---------------------------------------------------------------------------
# Explicit SHADOW path still works (back-compat for opt-in callers)
# ---------------------------------------------------------------------------


def test_explicit_shadow_module_still_publishes_to_shadow() -> None:
    """A caller that explicitly constructs a SHADOW
    ``ProtocolRegistryModule`` keeps the legacy quarantine.

    Packet 4.0 promoted the *default* to ACTIVE but did not
    remove SHADOW as a valid wiring level. Tests / dual-run
    harnesses that want to keep ``active_mixture`` out of the
    main snapshot stream can still pass an explicit SHADOW
    module instance.
    """

    shadow_module = ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)
    snapshots = _run_turn(protocol_registry_module=shadow_module)
    assert "active_mixture" in snapshots["shadow"], list(snapshots["shadow"])
    assert "active_mixture" not in snapshots["active"], list(snapshots["active"])


# ---------------------------------------------------------------------------
# Matched-control: SHADOW protocol_runtime introduces no new active drift
# ---------------------------------------------------------------------------


def _drifted_slot_names(
    a: dict[str, Snapshot[Any]], b: dict[str, Snapshot[Any]]
) -> set[str]:
    shared = set(a) & set(b)
    return {name for name in shared if a[name].value != b[name].value}


def test_loading_protocol_does_not_change_active_slot_set() -> None:
    """Loading a fixture into the registry must not move any
    OTHER slot's membership between active / shadow / disabled.

    Post packet 4.0 ACTIVE default: ``active_mixture`` itself
    appears in active for both empty and loaded runs (because
    the owner is wired ACTIVE in both). The invariant that loading
    a protocol doesn't *spread* ``active_mixture`` consumption to
    new slots is what this test pins.
    """

    empty_run = _run_turn()
    loaded_run = _run_turn(
        protocol_registry_module=_build_module_with_cheng_laoshi()
    )

    only_in_loaded = set(loaded_run["active"]) - set(empty_run["active"])
    only_in_empty = set(empty_run["active"]) - set(loaded_run["active"])
    assert only_in_loaded == set(), only_in_loaded
    assert only_in_empty == set(), only_in_empty


def test_loading_protocol_does_not_widen_active_value_drift() -> None:
    """Loading a fixture must not introduce additional drift on
    OTHER active slots beyond the baseline run-to-run drift inherent in
    timestamp / counter state.

    Post packet 4.0 (ACTIVE default): ``active_mixture`` itself
    legitimately drifts (empty → loaded). The matched-control
    invariant we care about is that no OTHER slot silently took
    ``active_mixture`` as a hidden dependency — which would show
    up as that slot drifting in the loaded-vs-empty set but not
    in baseline. We therefore exclude ``active_mixture`` itself
    from the drift comparison.
    """

    baseline_a = _run_turn()
    baseline_b = _run_turn()
    baseline_drift = _drifted_slot_names(
        baseline_a["active"], baseline_b["active"]
    )

    empty_run = _run_turn()
    loaded_run = _run_turn(
        protocol_registry_module=_build_module_with_cheng_laoshi()
    )
    load_drift = _drifted_slot_names(
        empty_run["active"], loaded_run["active"]
    )

    new_drift = (load_drift - baseline_drift) - {"active_mixture"}
    assert not new_drift, (
        "Loading a BehaviorProtocol introduced new drift on active "
        f"slots that are stable across baseline runs: {sorted(new_drift)}. "
        "Either a kernel module silently took active_mixture as a "
        "dependency, or fixture loading is mutating shared state."
    )
