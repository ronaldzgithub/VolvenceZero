from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from volvence_zero.agent.gate811_human_anchor_analysis import (
    GATE811_PILOT_ANALYSIS_SCHEMA_VERSION,
    GATE811_RATER_ROSTER_SCHEMA_VERSION,
    analyze_gate811_pilot_ratings,
    build_gate811_analysis_preregistration,
    validate_gate811_analysis_preregistration,
)
from volvence_zero.agent.gate811_human_anchor_tooling import (
    GATE811_CAPTURE_SCHEMA_VERSION,
    build_gate811_pilot_packet,
    export_gate811_pilot_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
HUMAN_PREREGISTRATION_PATH = (
    REPO_ROOT / "artifacts/gate811_human_anchor_prereg_20260731T180225Z.json"
)
ANALYSIS_PREREGISTRATION_PATH = (
    REPO_ROOT
    / "artifacts/gate811_human_anchor_analysis_prereg_20260731T182514Z.json"
)


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


def _capture(preregistration_sha256: str) -> dict[str, object]:
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
        "preregistration_sha256": preregistration_sha256,
        "records": records,
    }


def _completed_ratings(
    *,
    template: str,
    internal_key: dict[str, object],
) -> str:
    key_by_pair = {
        entry["pair_id"]: entry for entry in internal_key["entries"]
    }
    source = io.StringIO(template)
    rows = list(csv.DictReader(source))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0])
    writer.writeheader()
    for row in rows:
        pair_id = row["pair_id"]
        key = key_by_pair[pair_id]
        experimental_side = (
            "A"
            if key["side_a_arm"]
            in ("correct-user-state", "sleep-consolidation")
            else "B"
        )
        control_side = "B" if experimental_side == "A" else "A"
        row["rater_id"] = row["rater_slot"].replace("slot", "external")
        for dimension in (
            "rememberedness",
            "relationship_continuity",
        ):
            row[f"{experimental_side.lower()}_{dimension}"] = "6"
            row[f"{control_side.lower()}_{dimension}"] = "3"
        row["a_boundary_respect"] = "6"
        row["b_boundary_respect"] = "6"
        row["forced_preference"] = experimental_side
        writer.writerow(row)
    return output.getvalue()


def _analysis_inputs(tmp_path: Path) -> dict[str, object]:
    human_bytes = HUMAN_PREREGISTRATION_PATH.read_bytes()
    human = json.loads(human_bytes)
    human_sha256 = hashlib.sha256(human_bytes).hexdigest()
    bundle = build_gate811_pilot_packet(
        capture=_capture(human_sha256),
        preregistration=human,
        preregistration_sha256=human_sha256,
    )
    packet_dir = tmp_path / "packet"
    manifest = export_gate811_pilot_packet(
        bundle=bundle,
        output_dir=packet_dir,
    )
    analysis = build_gate811_analysis_preregistration(
        repo_root=REPO_ROOT,
        human_anchor_preregistration_path=(
            "artifacts/gate811_human_anchor_prereg_20260731T180225Z.json"
        ),
        created_at_unix_ms=1_785_520_900_000,
    )
    analysis_bytes = (
        json.dumps(
            analysis,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    packet_bytes = (packet_dir / "pilot_packet_blinded.json").read_bytes()
    key_bytes = (packet_dir / "pilot_key_internal.json").read_bytes()
    analysis_sha256 = hashlib.sha256(analysis_bytes).hexdigest()
    rating_template = bundle["rating_template_csv"]
    assert isinstance(rating_template, str)
    rater_roster = {
        "schema_version": GATE811_RATER_ROSTER_SCHEMA_VERSION,
        "human_anchor_preregistration_sha256": human_sha256,
        "analysis_preregistration_sha256": analysis_sha256,
        "entries": [
            {
                "rater_id": f"rater-external-{index:02d}",
                "human_rater_attested": True,
                "non_project_member_attested": True,
                "eligibility_review_artifact_sha256": "5" * 64,
                "attested_by": "human:study-operator",
            }
            for index in range(1, 7)
        ],
    }
    rater_roster_bytes = (
        json.dumps(
            rater_roster,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return {
        "human_anchor_preregistration": human,
        "human_anchor_preregistration_bytes": human_bytes,
        "human_anchor_preregistration_sha256": human_sha256,
        "analysis_preregistration": analysis,
        "analysis_preregistration_bytes": analysis_bytes,
        "analysis_preregistration_sha256": analysis_sha256,
        "packet": json.loads(packet_bytes),
        "packet_bytes": packet_bytes,
        "internal_key": json.loads(key_bytes),
        "internal_key_bytes": key_bytes,
        "packet_manifest": manifest,
        "rating_template_csv": rating_template,
        "rating_csv": _completed_ratings(
            template=rating_template,
            internal_key=bundle["internal_key"],
        ),
        "rater_roster": rater_roster,
        "rater_roster_bytes": rater_roster_bytes,
    }


def test_pilot_analysis_freezes_power_without_making_claim(
    tmp_path: Path,
) -> None:
    report = analyze_gate811_pilot_ratings(**_analysis_inputs(tmp_path))
    assert report["schema_version"] == GATE811_PILOT_ANALYSIS_SCHEMA_VERSION
    assert report["pair_count"] == 48
    assert report["rating_count"] == 144
    assert report["unique_rater_count"] == 6
    assert report["formal_capture_authorized"] is True
    assert report["human_anchor_claim_allowed"] is False
    assert report["rating_may_enter_reward_or_credit"] is False
    assert report["production_promotion_authorized"] is False
    assert len(report["contrasts"]) == 2
    for result in report["contrasts"].values():
        assert result["ordinal_krippendorff_alpha"] == pytest.approx(1.0)
        assert result["power"]["recommended_pairs"] == 60
        assert result["preference"]["formal_gate_evaluated"] is False


def test_formal_analysis_preregistration_is_current() -> None:
    payload = json.loads(ANALYSIS_PREREGISTRATION_PATH.read_bytes())
    validate_gate811_analysis_preregistration(payload, repo_root=REPO_ROOT)


def test_completed_rating_layout_must_match_frozen_template(
    tmp_path: Path,
) -> None:
    inputs = _analysis_inputs(tmp_path)
    rows = list(csv.DictReader(io.StringIO(inputs["rating_csv"])))
    rows[0]["pair_id"] = rows[3]["pair_id"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    inputs["rating_csv"] = output.getvalue()
    with pytest.raises(ValueError, match="frozen template"):
        analyze_gate811_pilot_ratings(**inputs)


def test_malformed_row_blocks_power_freeze(tmp_path: Path) -> None:
    inputs = _analysis_inputs(tmp_path)
    rows = list(csv.DictReader(io.StringIO(inputs["rating_csv"])))
    rows[0]["forced_preference"] = "MALFORMED"
    rows[0]["malformed_reason"] = "transcript rendering failure"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    inputs["rating_csv"] = output.getvalue()
    with pytest.raises(ValueError, match="malformed pilot rows"):
        analyze_gate811_pilot_ratings(**inputs)


def test_packet_manifest_hash_drift_fails_loudly(tmp_path: Path) -> None:
    inputs = _analysis_inputs(tmp_path)
    inputs["packet_bytes"] += b" "
    with pytest.raises(ValueError, match="manifest hash drift"):
        analyze_gate811_pilot_ratings(**inputs)


def test_non_project_human_attestation_is_required(tmp_path: Path) -> None:
    inputs = _analysis_inputs(tmp_path)
    inputs["rater_roster"]["entries"][0][
        "non_project_member_attested"
    ] = False
    inputs["rater_roster_bytes"] = (
        json.dumps(
            inputs["rater_roster"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ValueError, match="non_project_member_attested"):
        analyze_gate811_pilot_ratings(**inputs)
