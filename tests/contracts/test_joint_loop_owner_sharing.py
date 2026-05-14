"""Contract test: joint_loop / runtime main-chain owner sharing (debt #8).

Restates and tightens the architectural invariant from
``tests/test_phase2_eta_nl.py::test_joint_loop_shares_owner_instances_with_runtime_by_design``
so it lives alongside the rest of the
``tests/contracts/`` suite (CI defaults to running the whole
contracts directory; the legacy phase-2 test was easy to miss when
hunting the SSOT for a particular debt).

Two assertions:

1. **Sharing whitelist (positive)** — the documented owners that
   the joint-loop intentionally shares with ``AgentSessionRunner``
   (so on-turn training updates are visible to next-turn serving)
   are identical instances on both sides.
2. **Sharing blacklist (negative)** — owners that must stay
   joint-loop-private (e.g. ``_regime_module``) remain distinct.

Pinned name set guards against a silent third orchestrator slipping
in: if anyone adds a new owner that shares state across the boundary,
they must extend ``_SHARED_OWNERS`` here AND update the joint-loop
docstring (debt #8 mitigation chain).

Refs:

* docs/known-debts.md #8
* packages/vz-temporal/src/volvence_zero/joint_loop/runtime.py
"""

from __future__ import annotations

import pytest


# Owners explicitly shared by design (debt #8 docstring + spec
# §"TRAINING WRITEBACK PHASE"). Each entry is a (runner_attr,
# joint_attr) pair so we can assert ``runner.X is joint.Y``.
#
# The joint-loop exposes several aliases for the same shared instance
# (``memory_store`` + ``_memory_store``, ``world_temporal_policy`` +
# ``_world_policy`` + ``temporal_policy``, etc.). Each alias gets an
# entry so the silent-orchestrator scan below recognises it.
_SHARED_OWNERS: tuple[tuple[str, str], ...] = (
    ("_memory_store", "memory_store"),
    ("_memory_store", "_memory_store"),
    ("_evaluation_backbone", "_evaluation_backbone"),
    ("_world_temporal_policy", "world_temporal_policy"),
    ("_world_temporal_policy", "_world_policy"),
    ("_world_temporal_policy", "temporal_policy"),
    ("_temporal_policy", "_world_policy"),
    ("_temporal_policy", "temporal_policy"),
    ("_self_temporal_policy", "self_temporal_policy"),
    ("_self_temporal_policy", "_self_policy"),
    ("_default_residual_runtime", "residual_runtime"),
    ("_default_residual_runtime", "_residual_runtime"),
)


# Owners that MUST stay joint-loop-private (assertion: NOT shared).
_PRIVATE_OWNERS: tuple[tuple[str, str], ...] = (
    ("_regime_module", "_regime_module"),
)


def test_joint_loop_shares_documented_owners() -> None:
    """All entries in ``_SHARED_OWNERS`` are the same instance on both sides."""

    from lifeform_domain_emogpt import build_companion_lifeform

    life = build_companion_lifeform(
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
    )
    session = life.create_session(session_id="joint-loop-shared-contract")
    runner = session.brain_session.runner
    joint = runner._joint_loop
    for runner_attr, joint_attr in _SHARED_OWNERS:
        runner_owner = getattr(runner, runner_attr)
        joint_owner = getattr(joint, joint_attr)
        assert runner_owner is joint_owner, (
            f"joint-loop / runtime owner sharing drift: "
            f"runner.{runner_attr} ({type(runner_owner).__name__}) "
            f"is NOT joint.{joint_attr} ({type(joint_owner).__name__}). "
            "Either the contract changed (update _SHARED_OWNERS + "
            "joint-loop docstring) or there's a duplicate owner bug."
        )


def test_joint_loop_owners_stay_private_when_required() -> None:
    """Pinned ``_PRIVATE_OWNERS`` MUST be distinct instances."""

    from lifeform_domain_emogpt import build_companion_lifeform

    life = build_companion_lifeform(
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
    )
    session = life.create_session(session_id="joint-loop-private-contract")
    runner = session.brain_session.runner
    joint = runner._joint_loop
    for runner_attr, joint_attr in _PRIVATE_OWNERS:
        runner_owner = getattr(runner, runner_attr)
        joint_owner = getattr(joint, joint_attr)
        assert runner_owner is not joint_owner, (
            f"joint-loop owner that should stay private now leaks: "
            f"runner.{runner_attr} is the SAME instance as joint.{joint_attr}. "
            "RegimeModule (and any other owner in _PRIVATE_OWNERS) must stay "
            "joint-loop-private; see ETANLJointLoop docstring."
        )


def test_no_new_silent_orchestrator_owners_introduced() -> None:
    """Catch a silent third-orchestrator regression early.

    If anyone adds a new long-lived owner instance to ``ETANLJointLoop``
    that ALSO appears as an attribute on ``AgentSessionRunner`` and
    happens to be the same instance, this test catches the addition
    by enumerating the joint-loop's public attributes and cross-
    checking each against the runner. The whitelist
    ``_SHARED_OWNERS`` lists the only ones permitted to share.
    """

    from lifeform_domain_emogpt import build_companion_lifeform

    life = build_companion_lifeform(
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
    )
    session = life.create_session(session_id="joint-loop-no-silent-contract")
    runner = session.brain_session.runner
    joint = runner._joint_loop

    documented_joint_attrs = {pair[1] for pair in _SHARED_OWNERS} | {
        pair[1] for pair in _PRIVATE_OWNERS
    }
    runner_attr_lookup = {pair[1]: pair[0] for pair in _SHARED_OWNERS}

    suspicious_shared: list[str] = []
    for joint_attr in dir(joint):
        if joint_attr.startswith("__") or joint_attr in documented_joint_attrs:
            continue
        try:
            joint_owner = getattr(joint, joint_attr)
        except AttributeError:
            continue
        if not _looks_like_owner_instance(joint_owner):
            continue
        # Walk all runner attrs; if any one is the SAME instance, flag.
        for runner_attr in dir(runner):
            if runner_attr.startswith("__") or runner_attr in runner_attr_lookup:
                continue
            try:
                runner_owner = getattr(runner, runner_attr)
            except AttributeError:
                continue
            if joint_owner is runner_owner:
                suspicious_shared.append(
                    f"joint.{joint_attr} is runner.{runner_attr}"
                )
    assert not suspicious_shared, (
        "Detected joint-loop owner sharing NOT in _SHARED_OWNERS:\n  - "
        + "\n  - ".join(suspicious_shared)
        + "\nIf intentional, extend _SHARED_OWNERS + ETANLJointLoop "
        "docstring per debt #8."
    )


def _looks_like_owner_instance(obj: object) -> bool:
    """Heuristic: an object is an "owner" if it's a custom class instance.

    Excludes primitives / collections / None / callables. Used by the
    silent-orchestrator scan above to avoid flagging inert state
    (counters, dicts, etc.) as suspicious shares.
    """

    if obj is None:
        return False
    if isinstance(obj, (str, bytes, int, float, bool, complex)):
        return False
    if isinstance(obj, (list, tuple, set, dict, frozenset)):
        return False
    if callable(obj) and not hasattr(type(obj), "__dict__"):
        return False
    cls = type(obj)
    # Module-level functions / lambdas have stdlib types we want to skip.
    if cls.__module__ == "builtins":
        return False
    return True
