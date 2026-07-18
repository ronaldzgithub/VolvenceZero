# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""JSON Schema (draft 2020-12) export for non-Python consumers.

Python consumers validate through the frozen dataclasses; everyone else
(TypeScript, Go, data warehouses) can validate trajectory documents with
this exported JSON Schema. The Python types remain the SSOT — this module
derives enum vocabularies directly from them, and the drift guard test
(``tests/contracts/test_companion_standard_conformance.py``) asserts the
committed ``docs/external`` export matches this function's output.
"""

from __future__ import annotations

import json

from companion_standard.trajectory import (
    SCHEMA_VERSION,
    LabelSource,
    RelationshipPhase,
    TrajectorySource,
    TurnRole,
)


def _enum_values(enum_class: type) -> list[str]:
    return [member.value for member in enum_class]


def trajectory_json_schema() -> dict:
    """JSON Schema for the canonical InteractionTrajectory document."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://companionbench.com/schemas/relationship-representation-trajectory.v1.json",
        "title": "InteractionTrajectory",
        "description": (
            "Canonical multi-session human-AI interaction trajectory with "
            "per-segment relationship-state labels (Relationship "
            "Representation Standard, schema_version 1)."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "trajectory_id",
            "schema_version",
            "source",
            "family",
            "scenario_ref",
            "sessions",
            "labels",
            "metadata",
        ],
        "properties": {
            "trajectory_id": {"type": "string", "minLength": 1},
            "schema_version": {"const": SCHEMA_VERSION},
            "source": {"enum": _enum_values(TrajectorySource)},
            "family": {"type": "string", "minLength": 1},
            "scenario_ref": {"type": "string", "minLength": 1},
            "sessions": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/session"},
            },
            "labels": {
                "type": "array",
                "items": {"$ref": "#/$defs/label"},
            },
            "metadata": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}, {"type": "string"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "$defs": {
            "session": {
                "type": "object",
                "additionalProperties": False,
                "required": ["session_index", "gap_days_before", "turns"],
                "properties": {
                    "session_index": {"type": "integer", "minimum": 0},
                    "gap_days_before": {"type": "integer", "minimum": 0},
                    "turns": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/turn"},
                    },
                },
            },
            "turn": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_index", "role", "text"],
                "properties": {
                    "turn_index": {"type": "integer", "minimum": 0},
                    "role": {"enum": _enum_values(TurnRole)},
                    "text": {"type": "string", "minLength": 1},
                },
            },
            "label": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "session_index",
                    "turn_index",
                    "phase",
                    "trust_level",
                    "continuity_level",
                    "repair_pressure",
                    "source",
                    "evidence",
                ],
                "properties": {
                    "session_index": {"type": "integer", "minimum": 0},
                    "turn_index": {"type": "integer", "minimum": 0},
                    "phase": {"enum": _enum_values(RelationshipPhase)},
                    "trust_level": {"type": "number", "minimum": 0, "maximum": 1},
                    "continuity_level": {"type": "number", "minimum": 0, "maximum": 1},
                    "repair_pressure": {"type": "number", "minimum": 0, "maximum": 1},
                    "source": {"enum": _enum_values(LabelSource)},
                    "evidence": {"type": "string"},
                },
            },
        },
    }


def trajectory_json_schema_text() -> str:
    """Pretty-printed, key-sorted schema text (what ships in docs/external)."""
    return json.dumps(trajectory_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


__all__ = [
    "trajectory_json_schema",
    "trajectory_json_schema_text",
]
