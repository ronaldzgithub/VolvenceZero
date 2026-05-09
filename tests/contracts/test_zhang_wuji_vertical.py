"""Contract tests for the zhang_wuji service vertical.

Validates the wiring added to ``lifeform_service.verticals`` so the
browser-chat path (``start_browser_chat_qwen.{sh,ps1}``) can serve
the trained 张无忌 character end-to-end:

* The vertical is discovered alongside companion / coding.
* Both the synthetic factory and the alpha factory return a
  ``Lifeform`` whose first session is constructible.
* When ``ZHANG_WUJI_TEMPLATE_PATH`` points at a saved template
  produced by ``examples/train_zhang_wuji_template.py``, the alpha
  factory routes through ``give_birth(skip_memory_restore=True)``
  so the per-user filesystem-scoped memory is preserved while the
  template's drives / evolved profile still seed the lifeform.
* The alpha lifeform built with a template carries the saved
  drive levels (the "trained" bit) on its vitals bootstrap, not
  the bootstrap defaults.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import tempfile

import pytest

from lifeform_core import Lifeform
from lifeform_domain_character import (
    apply_drive_evolution_through_gate,
    build_zhang_wuji_demo_arc,
    build_zhang_wuji_lifeform,
    compute_drive_shape_evolution,
    ExperientialReplayDriver,
    LifeformTemplate,
    save_lifeform_template,
    vitals_drive_levels_from_session,
)
from lifeform_service.alpha import AlphaIdentityProvider
from lifeform_service.verticals import VerticalSpec, discover_verticals
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import build_default_memory_store


def _healthy_eval() -> EvaluationSnapshot:
    score = lambda name, value: EvaluationScore(
        family="zhang_wuji_test",
        metric_name=name,
        value=value,
        confidence=1.0,
        evidence="contract test",
    )
    return EvaluationSnapshot(
        turn_scores=(
            score("contract_integrity", 1.0),
            score("rollback_resilience", 0.95),
            score("fallback_reliance", 0.10),
        ),
        session_scores=(),
        alerts=(),
        description="contract test healthy eval",
    )


def test_zhang_wuji_vertical_is_discovered():
    verticals = discover_verticals()
    assert "zhang_wuji" in verticals
    spec = verticals["zhang_wuji"]
    assert isinstance(spec, VerticalSpec)
    assert spec.factory is not None
    assert spec.alpha_factory is not None


def test_zhang_wuji_vertical_factory_builds_synthetic_lifeform(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ZHANG_WUJI_TEMPLATE_PATH", raising=False)
    spec = discover_verticals()["zhang_wuji"]
    lifeform = spec.factory(None)
    assert isinstance(lifeform, Lifeform)
    session = lifeform.create_session(session_id="probe-synthetic")
    assert session is not None


def test_zhang_wuji_vertical_alpha_factory_builds_alpha_lifeform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    monkeypatch.delenv("ZHANG_WUJI_TEMPLATE_PATH", raising=False)
    spec = discover_verticals()["zhang_wuji"]
    identity = AlphaIdentityProvider(allowed_users=())
    lifeform = spec.alpha_factory(None, identity, str(tmp_path))
    assert isinstance(lifeform, Lifeform)


def _train_minimal_template(tmp_path: pathlib.Path) -> pathlib.Path:
    """Run a minimal 1-scene replay + save to produce a real template
    for the give_birth path tests. Subset of
    ``examples/train_zhang_wuji_template.py`` to keep test cost low.
    """
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    profile = bundle.profile
    arc = build_zhang_wuji_demo_arc()
    # Trim to the first 5 scenes (the schema minimum) so the test
    # stays under ~30s on the synthetic substrate while still
    # producing a real PE signal. We have to drop life-phase
    # boundaries that point past the trimmed range.
    from lifeform_domain_character import NarrativeArc

    trimmed_count = 5
    trimmed_phase_boundaries = tuple(
        (scene_index, phase_label)
        for scene_index, phase_label in arc.life_phase_boundaries
        if scene_index < trimmed_count
    )
    short_arc = NarrativeArc(
        arc_id=arc.arc_id + "-test-trimmed",
        character_id=arc.character_id,
        scenes=arc.scenes[:trimmed_count],
        life_phase_boundaries=trimmed_phase_boundaries,
        reviewed_by=arc.reviewed_by,
        source_provenance=arc.source_provenance + " (test trimmed)",
    )

    driver = ExperientialReplayDriver()
    replay_report = asyncio.run(
        driver.run_arc_async(arc=short_arc, lifeform=bundle.lifeform)
    )
    evolution = compute_drive_shape_evolution(
        replay_report=replay_report, base_profile=profile
    )
    apply_result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_healthy_eval(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence=(
            f"base_profile.version={profile.version}; "
            f"arc_id={short_arc.arc_id}"
        ),
    )
    snapshot_session = bundle.lifeform.create_session(
        session_id="vitals-capture"
    )
    asyncio.run(snapshot_session.run_turn("test snapshot turn."))
    saved_levels = vitals_drive_levels_from_session(snapshot_session)
    save_result = save_lifeform_template(
        profile=profile,
        evolved_profile=apply_result.evolved_profile
        if apply_result.allowed
        else None,
        template_id="zhang-wuji-contract-test",
        output_dir=tmp_path,
        memory_store=memory_store,
        vitals_drive_levels=saved_levels,
        replay_report=replay_report,
        source_arc_id=short_arc.arc_id,
        replay_provenance=(
            f"contract-test (allowed={len(apply_result.allowed)},"
            f"blocked={len(apply_result.blocked)})"
        ),
        overwrite_existing=True,
    )
    return save_result.template_path


def test_zhang_wuji_alpha_factory_with_template_uses_give_birth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    template_path = _train_minimal_template(tmp_path)
    monkeypatch.setenv("ZHANG_WUJI_TEMPLATE_PATH", str(template_path))

    spec = discover_verticals()["zhang_wuji"]
    identity = AlphaIdentityProvider(allowed_users=())
    alpha_root = tmp_path / "alpha_memory"
    alpha_root.mkdir(parents=True, exist_ok=True)

    lifeform = spec.alpha_factory(None, identity, str(alpha_root))
    assert isinstance(lifeform, Lifeform)

    template = LifeformTemplate.from_json_bytes(template_path.read_bytes())

    saved_drive_names = {drive_name for drive_name, _ in template.vitals_drive_levels}

    # The lifeform's vitals bootstrap should reflect the saved drive
    # levels (the "trained" bit). Read the drive specs through the
    # public LifeformConfig surface — never reach into private
    # owners (R8).
    config = lifeform._config
    bootstrap = config.vitals_bootstrap
    assert bootstrap is not None
    bootstrap_names = {drive.name for drive in bootstrap.drives}
    # All saved drive names must show up in the bootstrap (each
    # drive in the saved levels has been patched into the bootstrap
    # by give_birth's _patch_vitals_initial_levels).
    assert saved_drive_names.issubset(bootstrap_names)
    # And at least one drive's initial_level must equal the saved
    # value (otherwise give_birth's drive-level injection is a
    # no-op).
    saved_level_map = dict(template.vitals_drive_levels)
    matched = any(
        abs(drive.initial_level - saved_level_map.get(drive.name, drive.initial_level))
        < 1e-9
        and drive.name in saved_level_map
        for drive in bootstrap.drives
    )
    assert matched, (
        "expected at least one drive's initial_level to match the "
        "saved-template level after alpha give_birth(skip_memory_restore=True)"
    )


def test_zhang_wuji_factory_with_template_uses_give_birth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    template_path = _train_minimal_template(tmp_path)
    monkeypatch.setenv("ZHANG_WUJI_TEMPLATE_PATH", str(template_path))

    spec = discover_verticals()["zhang_wuji"]
    lifeform = spec.factory(None)
    assert isinstance(lifeform, Lifeform)

    # In the non-alpha path give_birth restores the template's
    # MemoryStore checkpoint into the lifeform — observable by the
    # presence of an injected memory_store kwarg.
    assert lifeform._init_kwargs.get("memory_store") is not None
