from __future__ import annotations

from lifeform_domain_emogpt.lab import load_relationship_consumer_training_view
from lifeform_evolution.relationship_lab_packet1l import (
    RELATIONSHIP_P1L_REQUIRED_UNITS,
    RelationshipP1lRating,
    RelationshipP1lVerdict,
    assess_relationship_p1l_ratings,
    freeze_relationship_p1l_protocol,
    load_relationship_p1l_protocol,
    load_relationship_p1l_sealed_key,
    write_relationship_p1l_packet,
)


_FROZEN_AT = "2026-08-21T08:30:00+00:00"


def test_p1l_packet_hides_labels_and_round_trips(tmp_path) -> None:
    dataset = load_relationship_consumer_training_view().training_dataset
    protocol, units = freeze_relationship_p1l_protocol(
        dataset=dataset,
        frozen_at_iso=_FROZEN_AT,
    )
    assert len(units) == RELATIONSHIP_P1L_REQUIRED_UNITS
    assert {item.evidence_kind for item in units} == {"history", "probe"}
    forbidden = (
        "latent_condition_",
        "preferred_action",
        "latent_policy_",
        "generator_truth",
    )
    for unit in units:
        blob = " ".join(unit.rater_payload().values())
        assert all(token not in blob for token in forbidden)
        assert unit.expected_option in {"A", "B"}
        expected_summary = unit.option_a if unit.expected_option == "A" else unit.option_b
        matching = tuple(
            item
            for item in dataset.abstract_conditions
            if item.hidden_summary == expected_summary
        )
        assert len(matching) == 1

    paths = write_relationship_p1l_packet(
        protocol=protocol,
        units=units,
        output_dir=tmp_path,
    )
    loaded = load_relationship_p1l_protocol(paths[0])
    assert loaded == protocol
    sealed = load_relationship_p1l_sealed_key(paths[3])
    assert len(sealed) == RELATIONSHIP_P1L_REQUIRED_UNITS
    csv_text = paths[2].read_text(encoding="utf-8")
    assert "expected_option" not in csv_text
    assert "latent_condition_" not in csv_text


def test_p1l_scores_majority_and_keeps_learning_closed() -> None:
    dataset = load_relationship_consumer_training_view().training_dataset
    protocol, units = freeze_relationship_p1l_protocol(
        dataset=dataset,
        frozen_at_iso=_FROZEN_AT,
    )
    pending = assess_relationship_p1l_ratings(
        protocol=protocol,
        units=units,
        ratings=(),
        created_at_iso=_FROZEN_AT,
    )
    assert pending.verdict is RelationshipP1lVerdict.RATINGS_PENDING

    perfect = tuple(
        RelationshipP1lRating(
            rater_id=f"rater-{index}",
            unit_id=unit.unit_id,
            chosen_option=unit.expected_option,
        )
        for index in range(1, 4)
        for unit in units
    )
    passed = assess_relationship_p1l_ratings(
        protocol=protocol,
        units=units,
        ratings=perfect,
        created_at_iso=_FROZEN_AT,
    )
    assert passed.verdict is RelationshipP1lVerdict.PASSED
    assert passed.majority_agreement == 1.0
    assert passed.majority_accuracy == 1.0
    assert not passed.to_payload()["experiment_guards"]["p2_enabled"]
    assert not passed.to_payload()["experiment_guards"][
        "evaluation_feedback_to_pe_credit_reward_or_steering"
    ]

    inverted = tuple(
        RelationshipP1lRating(
            rater_id=f"rater-{index}",
            unit_id=unit.unit_id,
            chosen_option="B" if unit.expected_option == "A" else "A",
        )
        for index in range(1, 4)
        for unit in units
    )
    failed = assess_relationship_p1l_ratings(
        protocol=protocol,
        units=units,
        ratings=inverted,
        created_at_iso=_FROZEN_AT,
    )
    assert failed.verdict is RelationshipP1lVerdict.FAILED
    assert failed.majority_accuracy == 0.0
