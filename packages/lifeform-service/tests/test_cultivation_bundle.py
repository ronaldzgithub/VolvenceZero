"""Unit tests for the portable cultivation protocol bundle.

Covers the export -> read round-trip and hydration into a
ProtocolUptakeService, using a real Identity Core BehaviorProtocol
built by the cultivation wheel (no live kernel / LLM needed).
"""

from __future__ import annotations

import pytest

from lifeform_cultivation import CultivationSeed, build_identity_core_protocol

from lifeform_service.cultivation_bundle import (
    CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION,
    build_uptake_service_from_bundle,
    export_protocol_bundle,
    read_protocol_bundle,
)


def _seed() -> CultivationSeed:
    return CultivationSeed(
        display_name="儿童心理专家",
        domain="儿童心理",
        role_archetype="临床儿童心理学家",
        value_boundaries=("不做医疗诊断", "不替代危机干预"),
    )


def _protocol():
    return build_identity_core_protocol(_seed())


def test_export_bundle_envelope_fields():
    proto = _protocol()
    bundle = export_protocol_bundle(
        [proto],
        source_ai_id="cultivation:child-psych",
        cultivation_id="cult_abcd",
        package_id="cpkg_1",
        track_id="pd",
    )
    assert bundle["schema_version"] == CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION
    assert bundle["source_ai_id"] == "cultivation:child-psych"
    assert bundle["cultivation_id"] == "cult_abcd"
    assert bundle["package_id"] == "cpkg_1"
    assert bundle["track_id"] == "pd"
    assert bundle["protocol_count"] == 1
    assert isinstance(bundle["protocols"], list) and len(bundle["protocols"]) == 1


def test_export_read_round_trip_preserves_protocol_id():
    proto = _protocol()
    bundle = export_protocol_bundle([proto], source_ai_id="ai:x")
    restored = read_protocol_bundle(bundle)
    assert len(restored) == 1
    assert restored[0].protocol_id == proto.protocol_id
    assert restored[0].review_status == proto.review_status


def test_read_bundle_rejects_malformed_envelope():
    with pytest.raises(ValueError):
        read_protocol_bundle("not a dict")
    with pytest.raises(ValueError):
        read_protocol_bundle({"schema_version": "wrong", "protocols": []})
    with pytest.raises(ValueError):
        read_protocol_bundle(
            {
                "schema_version": CULTIVATION_PROTOCOL_BUNDLE_SCHEMA_VERSION,
                "protocols": "not a list",
            }
        )


def test_build_uptake_service_hydrates_approved_set():
    proto = _protocol()
    bundle = export_protocol_bundle([proto], source_ai_id="ai:x")
    service = build_uptake_service_from_bundle(bundle)
    loaded = service.loaded_approved_snapshot()
    assert proto.protocol_id in {p.protocol_id for p in loaded}


def test_empty_bundle_round_trips_to_no_protocols():
    bundle = export_protocol_bundle([], source_ai_id="ai:x")
    assert bundle["protocol_count"] == 0
    assert read_protocol_bundle(bundle) == ()
    service = build_uptake_service_from_bundle(bundle)
    assert service.loaded_approved_snapshot() == ()
