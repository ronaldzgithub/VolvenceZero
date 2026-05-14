"""Contract test: Companion Bench i18n / language schema (debt #55).

Validates:

1. ``ScenarioSpec.language`` defaults to ``"en"`` (initial 24
   reviewer-curated yaml are English).
2. Loader rejects unknown language values.
3. ``language`` is **not** in ``to_canonical()`` so cross-language
   scenarios with otherwise identical semantics share a hash (RFC §3
   P3 reproducibility preserved).
4. The shipped public scenario set has at least 6 zh demo scenarios
   covering F1-F6 (i18n roadmap Batch 1).
5. zh demo scenario YAMLs load cleanly + carry the language tag.

Refs:

* docs/external/companion-bench-i18n-roadmap.md
* docs/known-debts.md #55
"""

from __future__ import annotations

import dataclasses
import importlib.resources as res
import pathlib

import pytest

from companion_bench.spec import (
    FamilyId,
    ScenarioSpec,
    load_scenarios_dir,
    load_scenario_yaml,
    scenario_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_dir() -> pathlib.Path:
    return pathlib.Path(
        str(res.files("companion_bench") / "scenarios" / "public")
    )


def _public_specs() -> tuple[ScenarioSpec, ...]:
    return tuple(load_scenarios_dir(_public_dir(), include_held_out=False))


# ---------------------------------------------------------------------------
# Defaults + loader rejection
# ---------------------------------------------------------------------------


def test_scenario_spec_language_default_is_en() -> None:
    """Initial 24 reviewer-curated scenarios are English; default reflects that."""

    fields = {f.name: f for f in dataclasses.fields(ScenarioSpec)}
    assert fields["language"].default == "en", (
        f"ScenarioSpec.language default drifted to {fields['language'].default!r}; "
        "initial 24 yaml ship with English simulator content (see "
        "docs/external/companion-bench-i18n-roadmap.md §3)."
    )


def test_loader_rejects_unknown_language(tmp_path: pathlib.Path) -> None:
    """``language`` must be one of ``zh`` / ``en`` / ``bilingual``."""
    yaml_text = """
scenario_id: F1-bad-lang-001
language: klingon
family: F1
arc_length_sessions: 2
session_turn_range: [4, 6]
inter_session_gap_days: [1]
user_simulator:
  persona: "test"
  goals: ["test"]
  perturbation_seed: 1
expected_axes:
  primary: [A3]
  secondary: []
  hard_constraint: A6
disqualifiers: []
public_test: true
held_out: false
paraphrase_seed_count: 1
""".strip()
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="language must be one of"):
        load_scenario_yaml(bad)


# ---------------------------------------------------------------------------
# Hash stability across languages
# ---------------------------------------------------------------------------


def test_language_not_in_canonical_hash() -> None:
    """``language`` field is publishing metadata, not part of canonical."""
    base = _public_specs()[0]
    probe = ScenarioSpec(
        scenario_id="probe",
        family=FamilyId.F1_CONTINUITY,
        arc_length_sessions=2,
        session_turn_range=(4, 6),
        inter_session_gap_days=(1,),
        user_simulator=base.user_simulator,
        expected_axes=base.expected_axes,
        disqualifiers=(),
        public_test=True,
        held_out=False,
        paraphrase_seed_count=1,
        language="zh",
    )
    canonical_keys = probe.to_canonical().keys()
    assert "language" not in canonical_keys, (
        "ScenarioSpec.language must not enter to_canonical(); adding it would "
        "invalidate every existing scenario_hash."
    )


def test_zh_and_en_variants_with_same_semantics_share_hash() -> None:
    """Two specs that differ ONLY in language must share a scenario_hash."""
    base_spec = _public_specs()[0]
    en_variant = dataclasses.replace(base_spec, language="en")
    zh_variant = dataclasses.replace(base_spec, language="zh")
    assert scenario_hash(en_variant) == scenario_hash(zh_variant)


# ---------------------------------------------------------------------------
# zh demo coverage (Batch 1)
# ---------------------------------------------------------------------------


_EXPECTED_ZH_DEMO_IDS = frozenset(
    {
        "F1-continuity-zh-001",
        "F2-repair-zh-001",
        "F3-personalization-zh-001",
        "F4-long-absence-zh-001",
        "F5-boundary-zh-001",
        "F6-goal-drift-zh-001",
    }
)


def test_six_zh_demo_scenarios_loaded() -> None:
    """Each of F1-F6 has at least 1 zh demo scenario (Batch 1 floor)."""
    specs = _public_specs()
    zh_specs = {s.scenario_id: s for s in specs if s.language == "zh"}
    missing = _EXPECTED_ZH_DEMO_IDS - zh_specs.keys()
    assert not missing, (
        f"i18n Batch 1 zh demo scenarios missing: {sorted(missing)}; "
        "see docs/external/companion-bench-i18n-roadmap.md §3."
    )


def test_zh_demo_covers_each_family() -> None:
    """zh demo set covers all 6 scenario families."""
    specs = _public_specs()
    zh_families = {s.family for s in specs if s.language == "zh"}
    assert zh_families == {f for f in FamilyId}, (
        f"zh demo missing family coverage; got {sorted(f.value for f in zh_families)}, "
        f"want all 6 of {sorted(f.value for f in FamilyId)}."
    )


def test_public_set_remains_at_or_above_24_en() -> None:
    """English public set must not silently shrink; v1.0 floor is 24."""
    specs = _public_specs()
    en_count = sum(1 for s in specs if s.language == "en")
    assert en_count >= 24, (
        f"English public set dropped below 24 (got {en_count}); "
        "i18n addition must not regress v1.0 baseline."
    )


def test_public_set_total_count_matches_roadmap() -> None:
    """Total public scenarios match the i18n roadmap §3 status table."""
    specs = _public_specs()
    counts = {"en": 0, "zh": 0, "bilingual": 0}
    for s in specs:
        counts[s.language] = counts.get(s.language, 0) + 1
    # Roadmap §3: 24 en + 6 zh + 0 bilingual = 30 (Batch 1 state).
    assert counts["en"] == 24, f"en count = {counts['en']}, expected 24"
    assert counts["zh"] == 6, f"zh count = {counts['zh']}, expected 6"
    assert counts["bilingual"] == 0, (
        f"bilingual count = {counts['bilingual']}, expected 0 (reserved for v1.x)"
    )
