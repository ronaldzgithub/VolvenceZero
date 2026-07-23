"""Small-budget P1 schema and fixed-gate smoke."""

from __future__ import annotations

from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_ARM_NAMES,
    ECOLOGY_P1_GATE_NAMES,
    ECOLOGY_P1_SCHEMA_VERSION,
    EcologyP1Config,
    run_ecology_p1,
    run_ecology_p1_diagnostics,
)


def test_p1_diagnostics_are_checkpoint_free_and_structured() -> None:
    report = run_ecology_p1_diagnostics(
        EcologyP1Config(
            n_ants=1,
            temporal_latent_dim=4,
            training_rounds=1,
            evaluation_rounds=3,
            layouts_per_tier=1,
            seed=7,
        )
    )

    assert report.schema_version == "digital-ant-ecology-p1-diagnostics.v2"
    assert len(report.results) == 18
    assert len(report.oracle_success_by_capability) == 6


async def test_p1_uses_fixed_schedule_per_body_mastery() -> None:
    report = await run_ecology_p1(
        EcologyP1Config(
            n_ants=1,
            temporal_latent_dim=4,
            training_rounds=1,
            evaluation_rounds=3,
            layouts_per_tier=1,
            seed=7,
        )
    )

    assert report.schema_version == ECOLOGY_P1_SCHEMA_VERSION
    assert report.verdict in {"PASS", "BLOCK"}
    assert tuple(gate.name for gate in report.gates) == ECOLOGY_P1_GATE_NAMES
    assert {item.arm for item in report.layout_results} == set(
        ECOLOGY_P1_ARM_NAMES
    )
    assert len(report.layout_results) == len(ECOLOGY_P1_ARM_NAMES) * 6
    assert len(report.diagnostic_results) == 3 * 6
    assert {item.controller for item in report.diagnostic_results} == {
        "oracle_steering",
        "fixed_rule",
        "random",
    }
    assert all(item.required_bodies == 1 for item in report.layout_results)
    assert all(item.policy_fingerprint_stable for item in report.layout_results)
    assert report.schedule[:3] and all(
        item.tier.value == "near" for item in report.schedule[:3]
    )
