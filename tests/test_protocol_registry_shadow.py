"""SHADOW behaviour for ``ProtocolRegistryModule`` (packet 1.0).

Asserts the SHADOW invariant for the ``active_mixture`` slot:

* ``active_mixture`` lives in ``shadow_snapshots`` and is absent
  from ``active_snapshots`` when wiring level is SHADOW (default).
* The published value's shape is correct (``ActiveMixtureSnapshot``).
* Empty registry publishes empty snapshot; loaded fixture publishes
  one ``ActiveProtocolEntry``.
* Loading the cheng_laoshi fixture does not introduce new value
  drift on any other active slot beyond the baseline drift inherent
  in re-running the kernel turn (matched-control invariant).

These tests are the SHADOW dual-run gate. Promotion to ACTIVE in
packet 1.2+ requires passing this contract test plus a new
matched-control test for each consumer.
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
# SHADOW publishing: shadow_snapshots only, never active_snapshots
# ---------------------------------------------------------------------------


def test_shadow_default_publishes_to_shadow_snapshots_only() -> None:
    snapshots = _run_turn()
    assert "active_mixture" in snapshots["shadow"], list(snapshots["shadow"])
    assert "active_mixture" not in snapshots["active"], list(snapshots["active"])


def test_shadow_published_value_is_active_mixture_snapshot() -> None:
    snapshots = _run_turn()
    am = _active_mixture_snapshot(snapshots["shadow"])
    assert isinstance(am, ActiveMixtureSnapshot)


# ---------------------------------------------------------------------------
# Empty registry publishes empty snapshot
# ---------------------------------------------------------------------------


def test_empty_registry_publishes_empty_active_protocols() -> None:
    snapshots = _run_turn()
    am = _active_mixture_snapshot(snapshots["shadow"])
    assert am.active_protocols == ()
    assert am.boundary_union_ids == ()


def test_empty_registry_fingerprint_is_empty_string() -> None:
    """Stable empty fingerprint so consumers can detect 'not loaded yet'."""
    snapshots = _run_turn()
    am = _active_mixture_snapshot(snapshots["shadow"])
    assert am.revision_fingerprint == ""


# ---------------------------------------------------------------------------
# Loading cheng_laoshi: registry → active_mixture has 1 entry + 4 boundaries
# ---------------------------------------------------------------------------


def _build_module_with_cheng_laoshi() -> ProtocolRegistryModule:
    module = ProtocolRegistryModule()
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return module


def test_loaded_protocol_appears_in_active_mixture() -> None:
    module = _build_module_with_cheng_laoshi()
    snapshots = _run_turn(protocol_registry_module=module)
    am = _active_mixture_snapshot(snapshots["shadow"])
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
    am = _active_mixture_snapshot(snapshots["shadow"])
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
    am = _active_mixture_snapshot(snapshots["shadow"])
    entry = am.active_protocols[0]
    assert entry.current_phase_id == "long_term_companion"


def test_loaded_protocol_fingerprint_is_nonempty() -> None:
    module = _build_module_with_cheng_laoshi()
    snapshots = _run_turn(protocol_registry_module=module)
    am = _active_mixture_snapshot(snapshots["shadow"])
    assert am.revision_fingerprint != ""
    assert len(am.revision_fingerprint) == 64  # sha256 hex digest


# ---------------------------------------------------------------------------
# Matched-control: SHADOW protocol_runtime introduces no new active drift
# ---------------------------------------------------------------------------


def _drifted_slot_names(
    a: dict[str, Snapshot[Any]], b: dict[str, Snapshot[Any]]
) -> set[str]:
    shared = set(a) & set(b)
    return {name for name in shared if a[name].value != b[name].value}


def test_loading_protocol_does_not_change_active_slot_set() -> None:
    """Loading a fixture into the SHADOW owner must not move any
    other slot from ``shadow`` to ``active`` or vice versa.

    No kernel module currently declares ``active_mixture`` as a
    dependency. This test pins that invariant: adding such a
    dependency requires an explicit contract review (and a new
    matched-control test for the consumer).
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
    active slots beyond the baseline run-to-run drift inherent in
    timestamp / counter state.

    If a kernel module silently took ``active_mixture`` as a
    dependency, its slot would appear in the loaded-vs-empty drift
    set but not in the empty-vs-empty baseline drift set.
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

    new_drift = load_drift - baseline_drift
    assert not new_drift, (
        "Loading a BehaviorProtocol introduced new drift on active "
        f"slots that are stable across baseline runs: {sorted(new_drift)}. "
        "Either a kernel module silently took active_mixture as a "
        "dependency, or fixture loading is mutating shared state."
    )
