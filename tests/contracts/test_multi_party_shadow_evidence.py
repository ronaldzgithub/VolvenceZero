"""Wave E4 multi-party SHADOW evidence faceting contract tests.

These tests pin the new BenchmarkReport fields:

* ``per_interlocutor_record_counts`` aggregates ToM owner records by
  ``OtherMindRecord.interlocutor_id``. Verifies keying does not leak
  between buckets.
* ``wrong_person_pe_events_total`` counts cumulative
  ``SocialPredictionError`` events whose typed kind is identity /
  relationship attribution. Verifies the diagnostic counter is wired
  through to the bench surface.

Wave E4 deliberately does NOT activate any of the multi-party 5
owners — it just gives the existing dyad-only chain a multi-party
probe so a future Wave can build on the readout.

The 3-party scenario lives at
``packages/lifeform-domain-emogpt/.../scenarios/long-form-three-party-arc.json``.
We pin its presence here so a future cleanup that drops it without
updating the evidence-bundle script gets caught.
"""

from __future__ import annotations

import json
import pathlib

from lifeform_evolution.benchmark import BenchmarkReport
from lifeform_evolution.family_report import FamilyId, compute_family_report


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCENARIOS_DIR = (
    _REPO_ROOT
    / "packages"
    / "lifeform-domain-emogpt"
    / "src"
    / "lifeform_domain_emogpt"
    / "scenarios"
)


def _bench(
    *,
    scenario_id: str,
    per_interlocutor_record_counts: tuple[tuple[str, int], ...] = (),
    wrong_person_pe_events_total: int = 0,
) -> BenchmarkReport:
    return BenchmarkReport(
        scenario_id=scenario_id,
        turn_reports=(),
        regime_match_rate=1.0,
        pe_threshold_match_rate=1.0,
        response_non_empty_rate=1.0,
        closed_scene_count=1,
        per_interlocutor_record_counts=per_interlocutor_record_counts,
        wrong_person_pe_events_total=wrong_person_pe_events_total,
    )


def test_default_per_interlocutor_counts_empty() -> None:
    bench = _bench(scenario_id="legacy")
    assert bench.per_interlocutor_record_counts == ()
    assert bench.wrong_person_pe_events_total == 0


def test_per_interlocutor_counts_round_trip_into_family_report() -> None:
    bench = _bench(
        scenario_id="three-party",
        per_interlocutor_record_counts=(("alice", 4), ("sam", 3)),
        wrong_person_pe_events_total=2,
    )
    family = compute_family_report(bench=bench)
    f3 = family.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    distinct_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.distinct_interlocutor_count"
    )
    wrong_person_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.wrong_person_pe_events_total"
    )
    assert distinct_metric.value == 2.0  # alice + sam
    assert distinct_metric.threshold is None
    assert wrong_person_metric.value == 2.0
    assert wrong_person_metric.threshold is None
    assert "alice=4" in distinct_metric.note
    assert "sam=3" in distinct_metric.note


def test_per_interlocutor_counts_dyad_default_only_primary() -> None:
    """Single-interlocutor scenarios (the companion vertical default)
    should produce exactly one bucket whose key is ``"primary"``.
    """
    bench = _bench(
        scenario_id="dyad-default",
        per_interlocutor_record_counts=(("primary", 7),),
        wrong_person_pe_events_total=0,
    )
    family = compute_family_report(bench=bench)
    f3 = family.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    distinct_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.distinct_interlocutor_count"
    )
    assert distinct_metric.value == 1.0
    assert "primary=7" in distinct_metric.note


def test_three_party_scenario_present_on_disk() -> None:
    path = _SCENARIOS_DIR / "long-form-three-party-arc.json"
    assert path.exists(), f"{path} is missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "long-form-three-party-arc"
    assert len(payload["turns"]) >= 30, (
        "long-form-three-party-arc.json should have at least 30 turns "
        "to fill the PE distribution window AND give both Alice and "
        "Sam enough turns to seed records; got "
        f"{len(payload['turns'])}"
    )
    text_blob = " ".join(turn["user_input"] for turn in payload["turns"])
    # Sanity-check the named parties appear in the script. Used later
    # by an evidence-running probe to verify the resolved interlocutor
    # ids include "alice" / "sam" once the multi-party owner is wired
    # to consume the conversational-role frame.
    assert "Alice" in text_blob
    assert "Sam" in text_blob
    # The deliberate-misattribution segment should be detectable from
    # a quick scan: at least one turn must reference "you mixed us
    # up" / "got that backwards" so the Wave E4 evidence run can
    # explain why wrong_person_pe_events_total > 0 is expected here.
    assert any(
        phrase in turn["user_input"].lower()
        for turn in payload["turns"]
        for phrase in (
            "mixed us up",
            "got that backwards",
            "wrong-person",
            "wrong person",
        )
    ), (
        "long-form-three-party-arc.json must include explicit "
        "wrong-person callout turns so the social PE probe has "
        "evidence to react to."
    )
