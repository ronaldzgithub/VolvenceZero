"""Wave C6 (Tier 1) — 张无忌 character lifeform e2e.

Builds a full ``Lifeform`` from the reviewed 张无忌 profile, runs a
small deterministic dialogue, and verifies that the four
application owners (`domain_knowledge` / `case_memory` /
`strategy_playbook` / `boundary_policy`) plus `vitals` are wired
with the character's seeded records.

This test exercises the canonical "novel-character → lifeform"
construct-time path documented in
``docs/specs/character-soul-bootstrap.md``. It does NOT exercise:

* Tier 2 multi-chunk ingestion (covered by
  ``test_zhang_wuji_ingestion_e2e.py`` in Wave C7)
* Tier 3 cross-session emergence (covered by
  ``test_zhang_wuji_longitudinal.py`` in Wave C8)

We use the synthetic substrate (Brain default) so the test runs
without torch.
"""

from __future__ import annotations

import asyncio

from lifeform_core import Lifeform, LifeformConfig
from lifeform_domain_character import (
    build_character_lifeform,
    build_zhang_wuji_lifeform,
    build_zhang_wuji_profile,
)


def _zhang_wuji_lifeform() -> Lifeform:
    """Convenience: a Zhang Wuji lifeform with companion-style defaults
    suitable for a fast deterministic e2e test.
    """
    bundle = build_zhang_wuji_lifeform()
    return bundle.lifeform


def _zhang_wuji_question_arc() -> tuple[str, ...]:
    """Three deterministic prompts that probe the character's signature
    drives / boundaries / pacing priors without requiring any LLM
    behaviour to validate a typed claim.
    """
    return (
        "我是来求教的，最近遇到一件事不知道怎么办。",
        "对方已经投降了，我应该追击吗？",
        "面对道德两难时，你通常怎么决定？",
    )


def test_build_zhang_wuji_lifeform_returns_bundle() -> None:
    bundle = build_zhang_wuji_lifeform()
    assert isinstance(bundle.lifeform, Lifeform)
    assert bundle.profile.profile_id == "zhang-wuji"
    # No excerpt was passed -> envelope is None (Tier 2 path is opt-in).
    assert bundle.ingestion_envelope is None


def test_build_character_lifeform_with_excerpt_emits_envelope() -> None:
    bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        novel_excerpt=(
            "山道上传来呼救声。\n\n他停步，回头，向声响走去。\n\n"
            "对方是个陌生人，他依旧伸出手。"
        ),
    )
    assert bundle.ingestion_envelope is not None
    assert bundle.ingestion_envelope.envelope_id == (
        "character-ingestion:zhang-wuji"
    )
    assert len(bundle.ingestion_envelope.chunks) >= 1


def test_zhang_wuji_lifeform_first_turn_runs_without_raise() -> None:
    lifeform = _zhang_wuji_lifeform()
    session = lifeform.create_session(session_id="zhang-wuji-tier1-first")

    async def _go() -> None:
        result = await session.run_turn(_zhang_wuji_question_arc()[0])
        # Sanity: the kernel committed to a regime and produced a
        # non-empty response.
        assert result.active_regime, (
            "regime should be non-empty after first turn"
        )
        assert result.response.text.strip(), "response was empty"

    asyncio.run(_go())


def test_zhang_wuji_application_owners_publish_seeded_records() -> None:
    lifeform = _zhang_wuji_lifeform()
    session = lifeform.create_session(
        session_id="zhang-wuji-tier1-application"
    )

    async def _go():
        results = []
        for prompt in _zhang_wuji_question_arc():
            results.append(await session.run_turn(prompt))
        return results

    results = asyncio.run(_go())
    assert len(results) == 3

    # Pull the four application owner snapshots from the latest turn.
    last = results[-1]
    snapshots = last.active_snapshots
    for slot in (
        "domain_knowledge",
        "case_memory",
        "strategy_playbook",
        "boundary_policy",
    ):
        assert slot in snapshots, (
            f"slot {slot!r} missing from active_snapshots; the wired "
            f"FinalRolloutConfig should publish all four application owners"
        )

    # ``case_memory`` and ``strategy_playbook`` are the load-bearing
    # surface for the character package: they index by problem
    # pattern, which is character-specific (e.g.
    # ``lineage-loyalty-conflicts-with-mercy``). At least one of the
    # ``rid-character:`` ids must appear in their hits / matched
    # rules across the three turns.
    #
    # ``domain_knowledge`` is intentionally NOT asserted: its ranking
    # combines a generic knowledge corpus with the package and the
    # specific user inputs above do not necessarily surface character
    # knowledge as top hits (this is owner-internal ranking, not a
    # contract violation). The package's knowledge records are still
    # in the store and reachable by other queries; we don't pin
    # ranking output per turn.
    case = snapshots["case_memory"].value
    playbook = snapshots["strategy_playbook"].value
    boundary = snapshots["boundary_policy"].value
    domain = snapshots["domain_knowledge"].value

    case_hits_seen: set[str] = set()
    rule_hits_seen: set[str] = set()
    for r in results:
        case_snap = r.active_snapshots["case_memory"].value
        playbook_snap = r.active_snapshots["strategy_playbook"].value
        for hit in getattr(case_snap, "hits", ()):
            cid = getattr(hit, "case_id", "")
            if isinstance(cid, str):
                case_hits_seen.add(cid)
        for rule in getattr(playbook_snap, "matched_rules", ()):
            rid = getattr(rule, "rule_id", "")
            if isinstance(rid, str):
                rule_hits_seen.add(rid)

    assert any("rid-character:case" in cid for cid in case_hits_seen), (
        f"case_memory owner did not surface any character case across "
        f"all turns; case_hits_seen={case_hits_seen!r}"
    )
    assert any("rid-character:playbook" in rid for rid in rule_hits_seen), (
        f"strategy_playbook owner did not surface any character rule "
        f"across all turns; rule_hits_seen={rule_hits_seen!r}"
    )

    # boundary_policy: just verify it published something non-trivial.
    # Specific trigger reasons depend on which prompt the owner
    # matched, which is owner-internal.
    assert boundary is not None
    # domain_knowledge: smoke (owner ran).
    assert domain is not None


def test_zhang_wuji_vitals_carry_signature_drives_at_runtime() -> None:
    lifeform = _zhang_wuji_lifeform()
    session = lifeform.create_session(session_id="zhang-wuji-tier1-vitals")

    async def _go():
        await session.run_turn(_zhang_wuji_question_arc()[0])
        return session.vitals_snapshot

    vitals = asyncio.run(_go())
    assert vitals is not None, "vitals snapshot should be populated"
    drive_names = {drive.name for drive in vitals.drive_levels}
    expected = {
        "compassion_active",
        "decisive_under_crisis",
        "loyalty_to_kin",
        "martial_curiosity",
        "self_sacrifice_pull",
    }
    missing = expected - drive_names
    assert not missing, (
        f"vitals owner missing 张无忌 drives at runtime: missing={missing}, "
        f"got={drive_names}"
    )


def test_zhang_wuji_lifeform_default_uses_character_package_only() -> None:
    """Vertical isolation: a Zhang Wuji lifeform must NOT default-include
    the companion package. Mixing verticals would leak companion drives
    or cases into the character lifeform; this test pins that the
    facade is per-vertical clean.
    """
    bundle = build_zhang_wuji_lifeform()
    config = bundle.lifeform.config
    pkgs = config.brain_config.domain_experience_packages
    assert len(pkgs) == 1, (
        f"expected exactly 1 domain experience package, got {len(pkgs)}"
    )
    assert pkgs[0].manifest.package_id == "lifeform-character:zhang-wuji"
    # Verify no companion package id anywhere.
    package_ids = [pkg.manifest.package_id for pkg in pkgs]
    assert not any(
        "companion" in pid.lower() for pid in package_ids
    ), f"companion package leaked into character lifeform: {package_ids}"


def test_zhang_wuji_lifeform_with_custom_config_preserves_overrides() -> None:
    """When the caller supplies ``config``, the facade must layer the
    character package + vitals on top of the supplied config rather
    than discarding it.
    """
    base = LifeformConfig(idle_close_after_system_ticks=42)
    bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        config=base,
    )
    assert bundle.lifeform.config.idle_close_after_system_ticks == 42
    # Character package must still be applied on top of the override.
    pkgs = bundle.lifeform.config.brain_config.domain_experience_packages
    assert len(pkgs) == 1
    assert pkgs[0].manifest.package_id == "lifeform-character:zhang-wuji"
    # Vitals bootstrap is on by default.
    assert bundle.lifeform.config.vitals_bootstrap is not None
