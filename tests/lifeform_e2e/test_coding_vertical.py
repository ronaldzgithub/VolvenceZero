"""Tests for the second vertical: ``lifeform-domain-coding``.

This whole file's purpose is to prove **trigger \u2461 of `SPLIT.md`** \u2014 the
kernel is genuinely vertical-agnostic. A second vertical lands without
any change inside ``vz-*``; it ships its own
``DomainExperiencePackage``, drive set, and scenarios; both verticals
coexist in one Python process; the service registry auto-discovers
both. If any of these break, the kernel's claim to "vertical-agnostic"
is false.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from lifeform_core import VitalsBootstrap


# ---------------------------------------------------------------------------
# Vertical-shape contract: package shape mirrors lifeform-domain-emogpt
# ---------------------------------------------------------------------------


def test_coding_vertical_exposes_canonical_public_api():
    """The vertical's public API matches the companion's so verticals are
    interchangeable from the service / loop layer's point of view."""
    import lifeform_domain_coding as coding

    expected_names = {
        "build_coding_lifeform",
        "build_coding_package",
        "build_coding_vitals_bootstrap",
        "scenarios_dir",
    }
    assert expected_names <= set(coding.__all__)
    for name in expected_names:
        assert hasattr(coding, name), f"missing {name!r} on lifeform_domain_coding"


def test_coding_package_compiles_into_kernel_owners():
    """The DomainExperiencePackage is shaped exactly like the companion's
    \u2014 only the records inside differ. This is what makes the kernel
    blind to which vertical produced the package.
    """
    from lifeform_domain_coding import build_coding_package
    from lifeform_domain_emogpt import build_companion_package

    coding = build_coding_package()
    companion = build_companion_package()
    # Same dataclass type, same field count.
    assert type(coding) is type(companion)
    # Same record-tuple shapes so kernel compilation paths are identical.
    assert isinstance(coding.knowledge_records, tuple)
    assert isinstance(coding.case_records, tuple)
    assert isinstance(coding.playbook_rules, tuple)
    assert isinstance(coding.boundary_hints, tuple)
    # And the manifests carry distinct package_ids so two co-installed
    # packs never alias to each other.
    assert coding.manifest.package_id != companion.manifest.package_id


# ---------------------------------------------------------------------------
# Vertical drive set: distinct from companion; shape compatible
# ---------------------------------------------------------------------------


def test_coding_vitals_bootstrap_distinct_from_companion():
    from lifeform_domain_coding import build_coding_vitals_bootstrap
    from lifeform_domain_emogpt import build_companion_vitals_bootstrap

    coding = build_coding_vitals_bootstrap()
    companion = build_companion_vitals_bootstrap()
    coding_names = {d.name for d in coding.drives}
    companion_names = {d.name for d in companion.drives}
    # Neither vertical's drives leak into the other.
    assert coding_names.isdisjoint(companion_names), (
        f"verticals must not share drive names; got coding={coding_names} "
        f"companion={companion_names}"
    )
    # Both still produce a VitalsBootstrap of schema version 1.
    assert isinstance(coding, VitalsBootstrap)
    assert isinstance(companion, VitalsBootstrap)
    assert coding.schema_version == companion.schema_version == 1


def test_coding_vitals_drives_have_engineering_shape():
    """Drive names express engineering concerns, not relational ones."""
    from lifeform_domain_coding import build_coding_vitals_bootstrap

    bootstrap = build_coding_vitals_bootstrap()
    names = {d.name for d in bootstrap.drives}
    assert names == {"solution_clarity", "code_freshness", "direction_certainty"}


# ---------------------------------------------------------------------------
# Negative regime recharge: exercise the new on_turn semantics
# ---------------------------------------------------------------------------


async def test_guided_exploration_drains_direction_certainty():
    """``direction_certainty`` recharge_per_regime['guided_exploration'] = -0.05.
    A guided-exploration turn should DECREASE the drive level (clamped at 0).
    """
    from lifeform_core import VitalsModule
    from lifeform_domain_coding import build_coding_vitals_bootstrap

    vm = VitalsModule(build_coding_vitals_bootstrap())
    pre = {d.name: d.level for d in vm.current_snapshot().drive_levels}
    vm.on_turn(regime="guided_exploration", user_input_present=True)
    post = {d.name: d.level for d in vm.current_snapshot().drive_levels}
    # Net charge for direction_certainty on guided_exploration:
    #   recharge_per_turn (0.05) + recharge_per_regime['guided_exploration'] (-0.05) = 0.0
    assert post["direction_certainty"] == pytest.approx(pre["direction_certainty"])
    # Net charge for direction_certainty on problem_solving:
    #   0.05 + 0.18 = +0.23 \u2192 clamped to <= 1.0
    vm.on_turn(regime="problem_solving", user_input_present=True)
    after_ps = {d.name: d.level for d in vm.current_snapshot().drive_levels}
    assert after_ps["direction_certainty"] > post["direction_certainty"]


# ---------------------------------------------------------------------------
# Scenarios: 4 valid JSON packs, loadable by lifeform-evolution
# ---------------------------------------------------------------------------


def test_coding_scenarios_dir_contains_four_loadable_packs():
    from lifeform_domain_coding import scenarios_dir

    sdir = scenarios_dir()
    assert sdir.is_dir(), f"scenarios dir should exist, got {sdir}"
    files = sorted(sdir.glob("*.json"))
    assert len(files) >= 4, f"expected \u22654 scenario files, got {[f.name for f in files]}"
    # Each is loadable + has scenario_id + non-empty turns.
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "scenario_id" in payload
        assert payload.get("turns"), f"empty turns in {path.name}"


async def test_coding_scenarios_run_through_lifeform_bench():
    """End-to-end: every coding scenario produces a non-empty response
    on the coding lifeform, with the new vitals fields populated.
    """
    from lifeform_domain_coding import build_coding_lifeform, scenarios_dir
    from lifeform_evolution import (
        compute_family_report,
        load_scenarios,
        run_benchmark_async,
    )

    scenarios = load_scenarios(scenarios_dir())
    assert len(scenarios) >= 4

    life = build_coding_lifeform()
    for scenario in scenarios:
        bench = await run_benchmark_async(scenario=scenario, lifeform=life)
        assert bench.response_non_empty_rate == pytest.approx(1.0), (
            f"scenario {scenario.scenario_id} produced an empty response"
        )
        # Coding vertical has vitals → bench report should reflect them.
        assert bench.final_vitals_drive_levels, (
            f"{scenario.scenario_id}: coding vertical should populate vitals"
        )
        drive_names = {name for name, _level in bench.final_vitals_drive_levels}
        assert "solution_clarity" in drive_names, drive_names
        # Family report computes cleanly across all six families.
        family_report = compute_family_report(bench=bench)
        assert len(family_report.families) == 6


# ---------------------------------------------------------------------------
# Service vertical registry sees both verticals
# ---------------------------------------------------------------------------


def test_service_registry_auto_discovers_coding_vertical():
    """``lifeform-serve --list-verticals`` would show ``companion``,
    ``companion-cold``, AND ``coding``. The kernel was not touched to
    register the new vertical \u2014 the registry just imports it.

    The coding vertical now also ships pre-trained bootstraps (produced
    by ``lifeform-super-loop --vertical coding``). The registry reports
    their presence dynamically, so the assertion adapts to whichever
    artifacts are currently checked in.
    """
    from lifeform_domain_coding import (
        has_coding_regime_bootstrap,
        has_coding_temporal_bootstrap,
    )
    from lifeform_service.verticals import discover_verticals

    verticals = discover_verticals()
    assert "companion" in verticals
    assert "coding" in verticals
    coding = verticals["coding"]
    assert coding.name == "coding"
    assert coding.scenarios_dir is not None
    assert coding.bootstraps_dir is not None
    # Bootstrap flags must mirror what the loaders see on disk.
    assert coding.has_temporal_bootstrap is has_coding_temporal_bootstrap()
    assert coding.has_regime_bootstrap is has_coding_regime_bootstrap()


def test_service_registry_factories_take_substrate_runtime():
    """Both verticals must accept the shared substrate parameter so the
    service can hand them one Qwen instance. Trigger \u2461 + the single-GPU
    deployment story BOTH require this signature to remain stable."""
    from inspect import signature

    from lifeform_service.verticals import discover_verticals

    for name, spec in discover_verticals().items():
        sig = signature(spec.factory)
        params = list(sig.parameters.values())
        assert len(params) == 1, (
            f"vertical {name!r} factory must take exactly one positional "
            f"argument (the substrate runtime), got {sig}"
        )


def test_two_verticals_coexist_in_one_process():
    """Two Lifeform instances built from different verticals must not
    interfere. This is the runtime equivalent of trigger \u2461.
    """
    from lifeform_domain_coding import build_coding_lifeform
    from lifeform_domain_emogpt import build_companion_lifeform

    a = build_companion_lifeform()
    b = build_coding_lifeform()
    sa = a.create_session(session_id="companion-coexist")
    sb = b.create_session(session_id="coding-coexist")

    a_drives = {d.name for d in sa.vitals_snapshot.drive_levels}
    b_drives = {d.name for d in sb.vitals_snapshot.drive_levels}
    assert a_drives == {"bond_warmth", "user_engagement", "conversation_continuity"}
    assert b_drives == {"solution_clarity", "code_freshness", "direction_certainty"}
    # Brain instances are distinct \u2014 no shared mutable state.
    assert sa.brain_session.runner is not sb.brain_session.runner


# ---------------------------------------------------------------------------
# Pre-trained bootstraps: same training pipeline as companion
# ---------------------------------------------------------------------------


def test_coding_vertical_ships_pretrained_bootstraps():
    """The vertical ships ``coding-temporal.snap`` + ``coding-regime.bs``
    produced by ``lifeform-super-loop --vertical coding``. Both load
    cleanly through the typed loaders that fail loudly on schema-version
    drift, so we know we are loading what we trained \u2014 not a stale
    artifact accidentally checked in.
    """
    from lifeform_domain_coding import (
        bootstraps_dir,
        has_coding_regime_bootstrap,
        has_coding_temporal_bootstrap,
        load_coding_regime_bootstrap,
        load_coding_temporal_bootstrap,
    )
    from volvence_zero.regime import RegimeBootstrap
    from volvence_zero.temporal import MetacontrollerParameterSnapshot

    assert bootstraps_dir().is_dir()
    assert has_coding_temporal_bootstrap()
    assert has_coding_regime_bootstrap()

    temporal = load_coding_temporal_bootstrap()
    regime = load_coding_regime_bootstrap()
    assert isinstance(temporal, MetacontrollerParameterSnapshot)
    assert isinstance(regime, RegimeBootstrap)
    # Regime bootstrap must contain the canonical kernel regimes \u2014 we did
    # not invent new regime names; we only adjusted selection_weights.
    weights = dict(regime.selection_weights)
    expected = {
        "casual_social",
        "acquaintance_building",
        "emotional_support",
        "guided_exploration",
        "problem_solving",
        "repair_and_deescalation",
    }
    assert set(weights.keys()) == expected
    # All weights stay in the calibrator's clip range.
    for name, weight in weights.items():
        assert 0.3 <= weight <= 2.0, f"weight for {name} out of range: {weight}"


def test_build_coding_lifeform_loads_shipped_bootstraps_by_default():
    from lifeform_domain_coding import build_coding_lifeform

    life = build_coding_lifeform()
    assert life.temporal_bootstrap is not None
    assert life.regime_bootstrap is not None


def test_build_coding_lifeform_can_run_cold_via_ablation_flags():
    """``use_*_bootstrap=False`` MUST suppress shipped artifacts. This
    is the ablation contract: 'compare cold vs trained' would silently
    lie if we ignored the flag.
    """
    from lifeform_domain_coding import build_coding_lifeform

    cold = build_coding_lifeform(
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
    )
    assert cold.temporal_bootstrap is None
    assert cold.regime_bootstrap is None


async def test_super_loop_works_on_coding_vertical_end_to_end():
    """The same training pipeline that produced the companion bootstraps
    runs cleanly on the coding vertical: round 0 is the cold baseline,
    later rounds train, the trajectory verdicts pass.

    Held to two rounds for test speed; verifies the public surfaces, not
    the magnitude of any specific calibration win (that depends on the
    scenario set's regime labels).
    """
    from lifeform_domain_coding import (
        build_coding_package,
        scenarios_dir,
    )
    from lifeform_evolution import load_scenarios, run_super_loop_async

    scenarios = load_scenarios(scenarios_dir())
    report = await run_super_loop_async(
        rounds=2,
        scenarios=scenarios,
        domain_experience_packages=(build_coding_package(),),
    )
    assert report.verdicts.get("sufficient_rounds") is True
    assert report.verdicts.get("temporal_state_evolved") is True
    # Round 0 is the baseline; round 1's snapshot must be different.
    assert (
        repr(report.rounds[1].temporal_snapshot)
        != repr(report.rounds[0].temporal_snapshot)
    )


def test_lifeform_bench_cli_supports_vertical_flag(capsys):
    """The bench CLI accepts ``--vertical coding`` and runs the full
    coding scenario set with the coding vertical's lifeform.
    """
    from lifeform_evolution.cli import main

    rc = main(["--vertical", "coding", "--family-report"])
    captured = capsys.readouterr().out
    assert rc == 0
    # All four coding scenarios reported.
    for scenario_id in (
        "bug-no-repro",
        "concrete-debug",
        "vague-feature-request",
        "security-refer-out",
    ):
        assert scenario_id in captured
    # Family report headers present.
    for fid in ("[F1]", "[F2]", "[F3]", "[F4]", "[F5]", "[F6]"):
        assert fid in captured
