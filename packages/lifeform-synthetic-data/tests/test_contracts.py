from __future__ import annotations

import dataclasses
import json

import pytest

from lifeform_synthetic_data.canonical import canonical_json, stable_hash
from lifeform_synthetic_data.conformance import (
    assert_v1_conformance,
    build_conformance_trajectory,
)
from lifeform_synthetic_data.contracts import (
    AnnotationRecord,
    AnnotationSource,
    TrainingUse,
)
from lifeform_synthetic_data.schema import build_json_schema


def test_v1_conformance_round_trip_and_hash_are_stable() -> None:
    first = assert_v1_conformance()
    second = assert_v1_conformance()

    assert first == second
    assert len(first) == 64


def test_contracts_are_frozen() -> None:
    trajectory = build_conformance_trajectory()

    with pytest.raises(dataclasses.FrozenInstanceError):
        trajectory.family = "mutated"  # type: ignore[misc]


def test_model_and_evaluation_labels_cannot_be_training_targets() -> None:
    original = build_conformance_trajectory().annotations[0]

    for source in (
        AnnotationSource.MODEL_PREDICTION,
        AnnotationSource.EVALUATION_READOUT,
    ):
        with pytest.raises(ValueError, match="cannot be training targets"):
            dataclasses.replace(
                original,
                source=source,
                training_use=TrainingUse.TARGET,
            )


def test_human_annotation_cannot_be_fabricated_without_identity_and_evidence() -> None:
    original = build_conformance_trajectory().annotations[0]

    with pytest.raises(ValueError, match="requires annotator_id"):
        AnnotationRecord(
            **{
                **dataclasses.asdict(original),
                "source": AnnotationSource.HUMAN_ANNOTATION,
                "annotator_id": None,
            }
        )


def test_json_schema_has_strict_master_roots() -> None:
    schema = build_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["oneOf"] == [
        {"$ref": "#/$defs/ExperienceTrajectory"},
        {"$ref": "#/$defs/CorpusManifest"},
    ]
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    trajectory_schema = definitions["ExperienceTrajectory"]
    assert isinstance(trajectory_schema, dict)
    assert trajectory_schema["additionalProperties"] is False


def test_canonical_json_is_key_order_independent_and_rejects_nan() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})

    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_canonical_payload_is_valid_json() -> None:
    payload = canonical_json(build_conformance_trajectory())
    decoded = json.loads(payload)

    assert decoded["schema_version"] == "synthetic-experience.v1"
