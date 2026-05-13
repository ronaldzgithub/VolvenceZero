"""Multi-party scenario package framework contract test (B1 / T14).

Verifies the COG-2 起跑面 fixture set:

- ``MultiPartyScenarioCase`` / ``MultiPartyInterlocutor`` /
  ``EnvironmentEventFrame`` / ``ScenarioPackageDescriptor`` are frozen
- Wrapped legacy ``ScriptedDialogueCase`` remains unchanged (B1 关键不变量 1)
- All three reference scenarios (wrong_person / witness / private_leakage)
  declare at least 2 interlocutors with distinct roles
- ``DEFAULT_MULTI_PARTY_SCENARIOS`` exposes exactly 3 canonical cases
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.agent.dialogue.multi_party_scenarios import (
    DEFAULT_MULTI_PARTY_SCENARIOS,
    EnvironmentEventFrame,
    InterlocutorRole,
    MultiPartyInterlocutor,
    MultiPartyScenarioCase,
    ScenarioPackageDescriptor,
    private_leakage_case,
    witness_case,
    wrong_person_case,
)
from volvence_zero.agent.dialogue.types import ScriptedDialogueCase


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_all_dataclasses_frozen() -> None:
    for cls in (
        InterlocutorRole,
        MultiPartyInterlocutor,
        EnvironmentEventFrame,
        ScenarioPackageDescriptor,
        MultiPartyScenarioCase,
    ):
        params = getattr(cls, "__dataclass_params__", None)
        assert params is not None and params.frozen, (
            f"{cls.__name__} must be a frozen dataclass"
        )


def test_legacy_scripted_case_unchanged() -> None:
    """B1 关键不变量 1: ScriptedDialogueCase 字段集不变 (no new required
    fields, no renamed existing fields)."""
    expected = (
        "case_id",
        "description",
        "user_inputs",
        "expected_pressure_turns",
        "expected_delayed_signals",
    )
    actual = tuple(f.name for f in dataclasses.fields(ScriptedDialogueCase))
    assert actual == expected, (
        "ScriptedDialogueCase fields changed; multi-party wrapper must "
        "remain non-invasive."
    )


# ---------------------------------------------------------------------------
# Reference fixtures
# ---------------------------------------------------------------------------


def test_default_set_contains_three_canonical_scenarios() -> None:
    scenarios = DEFAULT_MULTI_PARTY_SCENARIOS()
    assert len(scenarios) == 3
    kinds = {s.scenario_kind for s in scenarios}
    assert kinds == {"wrong_person", "witness", "private_leakage"}


@pytest.mark.parametrize(
    ("builder", "expected_kind"),
    [
        (wrong_person_case, "wrong_person"),
        (witness_case, "witness"),
        (private_leakage_case, "private_leakage"),
    ],
)
def test_each_reference_scenario_declares_two_interlocutors(builder, expected_kind):
    case = builder()
    assert case.scenario_kind == expected_kind
    assert len(case.interlocutors) >= 2, (
        f"{expected_kind}: multi-party scenarios must declare ≥2 interlocutors"
    )
    roles = {p.role.role for p in case.interlocutors}
    assert len(roles) >= 2, (
        f"{expected_kind}: roles must be distinct ({roles}); otherwise "
        f"the scenario isn't actually multi-party"
    )


def test_scenarios_carry_legacy_script_verbatim() -> None:
    """The wrapped ScriptedDialogueCase must be a real instance with non-empty
    user_inputs — B1 wrapper does not produce malformed legacy cases."""
    for scenario in DEFAULT_MULTI_PARTY_SCENARIOS():
        assert isinstance(scenario.script, ScriptedDialogueCase)
        assert len(scenario.script.user_inputs) >= 1
        # case_id is exposed via property for benchmark dispatch
        assert scenario.case_id == scenario.script.case_id


def test_environment_events_reference_known_turn_indices() -> None:
    """EnvironmentEventFrame.turn_index must point to a real user_input index."""
    for scenario in DEFAULT_MULTI_PARTY_SCENARIOS():
        num_turns = len(scenario.script.user_inputs)
        for event in scenario.environment_events:
            assert 0 <= event.turn_index < num_turns, (
                f"{scenario.scenario_kind}: event {event.event_id!r} "
                f"turn_index={event.turn_index} out of range "
                f"[0, {num_turns})"
            )


def test_scenarios_declare_qd_axis_tags() -> None:
    """B1 + EVO-3: scenarios must surface (regime, intensity, rupture_kind)
    for QD scenario archive consumers."""
    for scenario in DEFAULT_MULTI_PARTY_SCENARIOS():
        desc = scenario.package_descriptor
        assert isinstance(desc, ScenarioPackageDescriptor)
        assert desc.regime
        assert 0.0 <= desc.intensity <= 1.0
        # rupture_kind may be empty string when no rupture is staged
        assert isinstance(desc.rupture_kind, str)


def test_witness_role_present_when_required() -> None:
    """witness + private_leakage scenarios must include a 'witness' role,
    otherwise the scenario doesn't actually test witness handling."""
    for builder in (witness_case, private_leakage_case):
        case = builder()
        roles = {p.role.role for p in case.interlocutors}
        assert "witness" in roles, (
            f"{case.scenario_kind}: missing 'witness' interlocutor role"
        )
