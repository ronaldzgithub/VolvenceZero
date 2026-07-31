from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from volvence_zero.agent.gate811_human_anchor import (
    build_gate811_human_anchor_preregistration,
)
from volvence_zero.agent.gate811_human_anchor_tooling import (
    GATE811_CAPTURE_SCHEMA_VERSION,
    build_gate811_pilot_packet,
    export_gate811_pilot_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION_SHA256 = "1" * 64


def _sessions(arm: str) -> list[dict[str, object]]:
    blinded_variant = hashlib.sha256(arm.encode("utf-8")).hexdigest()[:8]
    sessions = []
    for session_index in range(3):
        turns = []
        for turn_index in range(10):
            speaker = "user" if turn_index % 2 == 0 else "assistant"
            text = (
                f"shared user turn {session_index}-{turn_index}"
                if speaker == "user"
                else f"response {blinded_variant} {session_index}-{turn_index}"
            )
            turns.append({"speaker": speaker, "text": text})
        sessions.append({"session_index": session_index, "turns": turns})
    return sessions


def _capture() -> dict[str, object]:
    records = []
    contrasts = (
        (
            "gate11-correct-state-vs-stateless",
            "correct-user-state",
            "stateless",
        ),
        (
            "gate8-sleep-vs-no-sleep",
            "sleep-consolidation",
            "no-sleep",
        ),
    )
    for contrast_index, (contrast_id, experimental, control) in enumerate(
        contrasts
    ):
        for pair_index in range(24):
            pair_key = f"pair-{contrast_index}-{pair_index:02d}"
            lineage = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()
            for arm in (experimental, control):
                records.append(
                    {
                        "record_id": f"{pair_key}-{arm}",
                        "contrast_id": contrast_id,
                        "pair_key": pair_key,
                        "arm_label": arm,
                        "capture_seed": (1401, 1413, 1427)[pair_index % 3],
                        "source_lineage": lineage,
                        "persona_ref": f"persona-{pair_index % 6}",
                        "model_and_adapter_fingerprint": "2" * 64,
                        "sessions": _sessions(arm),
                        "deidentification_attestation": {
                            "consent_scope_sha256": "3" * 64,
                            "pii_scan_artifact_sha256": "4" * 64,
                            "deidentified_by": "human:privacy-reviewer",
                            "callback_event_present": True,
                            "emotional_event_present": True,
                            "boundary_event_present": True,
                        },
                    }
                )
    return {
        "schema_version": GATE811_CAPTURE_SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "records": records,
    }


def _preregistration() -> dict[str, object]:
    return build_gate811_human_anchor_preregistration(
        repo_root=REPO_ROOT,
        created_at_unix_ms=1_785_520_000_000,
    )


def test_packet_is_matched_blinded_and_deterministic() -> None:
    first = build_gate811_pilot_packet(
        capture=_capture(),
        preregistration=_preregistration(),
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    second = build_gate811_pilot_packet(
        capture=_capture(),
        preregistration=_preregistration(),
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    assert first == second
    packet = first["packet"]
    assert packet["pair_count"] == 48
    assert packet["pilot_only"] is True
    packet_text = json.dumps(packet, sort_keys=True)
    for hidden_value in (
        "correct-user-state",
        "stateless",
        "sleep-consolidation",
        "no-sleep",
        "gate11-correct-state-vs-stateless",
        "gate8-sleep-vs-no-sleep",
    ):
        assert hidden_value not in packet_text
    assert first["internal_key"]["do_not_distribute_to_raters"] is True


def test_rating_template_assigns_three_slots_per_pair() -> None:
    bundle = build_gate811_pilot_packet(
        capture=_capture(),
        preregistration=_preregistration(),
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    rows = list(csv.DictReader(io.StringIO(bundle["rating_template_csv"])))
    assert len(rows) == 48 * 3
    assert {row["forced_preference"] for row in rows} == {""}
    pair_counts: dict[str, int] = {}
    for row in rows:
        pair_counts[row["pair_id"]] = pair_counts.get(row["pair_id"], 0) + 1
    assert set(pair_counts.values()) == {3}


def test_pair_mismatch_fails_loudly() -> None:
    capture = _capture()
    capture["records"][1]["persona_ref"] = "different-persona"
    with pytest.raises(ValueError, match="persona_ref mismatch"):
        build_gate811_pilot_packet(
            capture=capture,
            preregistration=_preregistration(),
            preregistration_sha256=PREREGISTRATION_SHA256,
        )


def test_missing_privacy_or_event_attestation_fails_loudly() -> None:
    capture = deepcopy(_capture())
    capture["records"][0]["deidentification_attestation"][
        "boundary_event_present"
    ] = False
    with pytest.raises(ValueError, match="boundary_event_present"):
        build_gate811_pilot_packet(
            capture=capture,
            preregistration=_preregistration(),
            preregistration_sha256=PREREGISTRATION_SHA256,
        )


def test_export_hashes_blinded_packet_and_internal_key(tmp_path: Path) -> None:
    bundle = build_gate811_pilot_packet(
        capture=_capture(),
        preregistration=_preregistration(),
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    manifest = export_gate811_pilot_packet(
        bundle=bundle,
        output_dir=tmp_path,
    )
    assert manifest["human_anchor_claim_allowed"] is False
    assert manifest["production_promotion_authorized"] is False
    for relative in manifest["required_files"]:
        assert (tmp_path / relative).is_file()
