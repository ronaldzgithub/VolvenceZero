"""Wave T5 e2e — save_lifeform_template extracts a lived state.

End-to-end:

1. Build a 张无忌 lifeform with a shared ``MemoryStore``.
2. Run a tiny scripted dialogue (so memory has at least one entry).
3. Call ``save_lifeform_template``.
4. Verify the on-disk JSON file exists, parses, and the round-tripped
   template carries the expected manifest / profile / vitals.

Also verify failure modes (missing template_id, file already exists
without overwrite).
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

from lifeform_domain_character import (
    ExperientialReplayDriver,
    LifeformTemplate,
    TEMPLATE_SCHEMA_VERSION,
    build_zhang_wuji_demo_arc,
    build_zhang_wuji_lifeform,
    save_lifeform_template,
    vitals_drive_levels_from_session,
)
from volvence_zero.memory import build_default_memory_store


def _drain_a_few_turns(lifeform, *, session_id: str, n: int = 2):
    session = lifeform.create_session(session_id=session_id)

    async def _go():
        for i in range(n):
            await session.run_turn(
                f"测试输入 {i}",
            )
        return session

    return asyncio.run(_go())


def test_save_creates_on_disk_template_file(tmp_path: pathlib.Path) -> None:
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    session = _drain_a_few_turns(
        bundle.lifeform, session_id="save-e2e-1", n=2
    )

    result = save_lifeform_template(
        profile=bundle.profile,
        template_id="t-save-e2e-1",
        output_dir=tmp_path,
        memory_store=memory_store,
        vitals_drive_levels=vitals_drive_levels_from_session(session),
        replay_provenance="test-fixture",
    )

    assert result.template_path.exists()
    assert result.template_path.suffix == ".json"
    # Reload from disk; verify schema version matches.
    raw = json.loads(result.template_path.read_text(encoding="utf-8"))
    assert raw["manifest"]["schema_version"] == TEMPLATE_SCHEMA_VERSION
    assert raw["manifest"]["template_id"] == "t-save-e2e-1"
    assert raw["manifest"]["character_id"] == "zhang-wuji"


def test_save_then_load_round_trip_recovers_profile(tmp_path: pathlib.Path) -> None:
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    session = _drain_a_few_turns(
        bundle.lifeform, session_id="save-e2e-roundtrip", n=2
    )

    result = save_lifeform_template(
        profile=bundle.profile,
        template_id="t-save-roundtrip",
        output_dir=tmp_path,
        memory_store=memory_store,
        vitals_drive_levels=vitals_drive_levels_from_session(session),
        replay_provenance="roundtrip-test",
    )
    blob = result.template_path.read_bytes()
    loaded = LifeformTemplate.from_json_bytes(blob)
    assert loaded.profile.profile_id == bundle.profile.profile_id
    assert loaded.manifest.integrity_hash == result.template.manifest.integrity_hash
    # Drive levels should be carried verbatim.
    assert loaded.vitals_drive_levels == result.template.vitals_drive_levels
    # Memory checkpoint round-tripped (we don't byte-compare because
    # the memory checkpoint uses dynamic checkpoint_ids; just check
    # that the field is present after round-trip).
    assert loaded.memory_checkpoint is not None or memory_store is None


def test_save_after_replay_includes_replay_report(tmp_path: pathlib.Path) -> None:
    """When a replay report is passed, the template carries it."""
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    arc = build_zhang_wuji_demo_arc()
    driver = ExperientialReplayDriver()
    report = driver.run_arc(arc=arc, lifeform=bundle.lifeform)

    result = save_lifeform_template(
        profile=bundle.profile,
        template_id="t-save-after-replay",
        output_dir=tmp_path,
        memory_store=memory_store,
        replay_report=report,
        source_arc_id=arc.arc_id,
        replay_provenance="zhang-wuji-demo-arc + wave T5 e2e",
    )
    assert result.template.replay_report is not None
    assert result.template.replay_report.arc_id == arc.arc_id
    assert result.template.replay_report.character_id == arc.character_id
    assert result.template.manifest.source_arc_id == arc.arc_id

    # On-disk round-trip preserves replay report.
    loaded = LifeformTemplate.from_json_bytes(
        result.template_path.read_bytes()
    )
    assert loaded.replay_report is not None
    assert loaded.replay_report.arc_id == arc.arc_id
    assert (
        loaded.replay_report.scenes_processed == report.scenes_processed
    )


def test_save_rejects_existing_path_without_overwrite(
    tmp_path: pathlib.Path,
) -> None:
    bundle = build_zhang_wuji_lifeform()
    save_lifeform_template(
        profile=bundle.profile,
        template_id="t-overwrite",
        output_dir=tmp_path,
        replay_provenance="initial-write",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        save_lifeform_template(
            profile=bundle.profile,
            template_id="t-overwrite",
            output_dir=tmp_path,
            replay_provenance="second-write",
        )


def test_save_overwrite_explicit(tmp_path: pathlib.Path) -> None:
    bundle = build_zhang_wuji_lifeform()
    first = save_lifeform_template(
        profile=bundle.profile,
        template_id="t-overwrite-yes",
        output_dir=tmp_path,
        replay_provenance="initial-write",
    )
    second = save_lifeform_template(
        profile=bundle.profile,
        template_id="t-overwrite-yes",
        output_dir=tmp_path,
        replay_provenance="second-write",
        overwrite_existing=True,
    )
    assert first.template_path == second.template_path
    # Provenance string differs, so integrity hashes must differ.
    assert (
        first.template.manifest.integrity_hash
        != second.template.manifest.integrity_hash
    )


def test_save_rejects_empty_template_id(tmp_path: pathlib.Path) -> None:
    bundle = build_zhang_wuji_lifeform()
    with pytest.raises(ValueError, match="template_id"):
        save_lifeform_template(
            profile=bundle.profile,
            template_id="   ",
            output_dir=tmp_path,
            replay_provenance="empty-id",
        )
