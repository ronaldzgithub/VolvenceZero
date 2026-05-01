"""Companion regime convergence guard.

This is the behavioral gate that caught the regression where the
companion vertical collapsed every scenario into ``emotional_support``.
It runs the shipped companion scenario pack through the public benchmark
runner and asserts broad regime differentiation without requiring every
single seed scenario to pass.
"""

from __future__ import annotations


async def test_companion_scenario_pack_does_not_collapse_to_support() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform, scenarios_dir
    from lifeform_evolution import load_scenarios, run_benchmark_async

    lifeform = build_companion_lifeform()
    reports = tuple(
        [
            await run_benchmark_async(scenario=scenario, lifeform=lifeform)
            for scenario in load_scenarios(scenarios_dir())
        ]
    )
    passed = tuple(report for report in reports if report.passed())
    regimes = {
        turn.active_regime
        for report in reports
        for turn in report.turn_reports
        if turn.active_regime is not None
    }
    by_id = {report.scenario_id: report for report in reports}

    assert len(passed) >= 7, (
        f"Expected >=7 companion scenarios to pass; got {len(passed)}/{len(reports)}: "
        f"{[(report.scenario_id, report.regime_match_rate) for report in reports]}"
    )
    assert "emotional_support" in regimes
    assert "guided_exploration" in regimes
    assert "acquaintance_building" in regimes
    assert by_id["trust-rupture-repair"].passed()
    assert by_id["task-troubleshoot"].passed()
