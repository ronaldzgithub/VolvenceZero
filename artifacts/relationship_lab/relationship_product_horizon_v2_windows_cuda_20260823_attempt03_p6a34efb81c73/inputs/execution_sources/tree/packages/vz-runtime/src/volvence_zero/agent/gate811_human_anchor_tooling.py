"""Build blinded Gate 8/11 human-anchor pilot packets.

The tooling consumes typed, already-deidentified transcript captures.  It
does not infer consent, PII, event type, or comparison arms from natural
language.  Pairing is exact on typed lineage and byte-identical user turns.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import random
from typing import Mapping, Sequence


GATE811_CAPTURE_SCHEMA_VERSION = "gate811-human-anchor-capture.v1"
GATE811_PACKET_SCHEMA_VERSION = "gate811-human-anchor-packet.v1"
GATE811_RATING_TEMPLATE_SCHEMA_VERSION = (
    "gate811-human-anchor-rating-template.v1"
)
_RATING_COLUMNS = (
    "rater_slot",
    "rater_id",
    "pair_id",
    "a_rememberedness",
    "b_rememberedness",
    "a_relationship_continuity",
    "b_relationship_continuity",
    "a_boundary_respect",
    "b_boundary_respect",
    "forced_preference",
    "malformed_reason",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 digest") from exc
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _validated_sessions(
    value: object,
    *,
    expected_session_count: int,
    expected_total_turns: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or len(value) != expected_session_count:
        raise ValueError("transcript session count drift")
    sessions = []
    turn_count = 0
    for expected_index, session in enumerate(value):
        if not isinstance(session, Mapping):
            raise ValueError("transcript session must be an object")
        if session.get("session_index") != expected_index:
            raise ValueError("transcript session order drift")
        turns = session.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("transcript session must contain turns")
        clean_turns = []
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("transcript turn must be an object")
            speaker = turn.get("speaker")
            if speaker not in ("user", "assistant"):
                raise ValueError("transcript speaker must be user or assistant")
            text = _require_string(turn.get("text"), field="turn.text")
            clean_turns.append({"speaker": speaker, "text": text})
        turn_count += len(clean_turns)
        sessions.append(
            {"session_index": expected_index, "turns": clean_turns}
        )
    if turn_count != expected_total_turns:
        raise ValueError("transcript total turn count drift")
    return tuple(sessions)


def _user_turn_digest(sessions: Sequence[Mapping[str, object]]) -> str:
    user_turns = []
    for session in sessions:
        turns = session["turns"]
        assert isinstance(turns, list)
        for turn in turns:
            assert isinstance(turn, Mapping)
            if turn["speaker"] == "user":
                user_turns.append(turn["text"])
    return _sha256_bytes(_canonical_bytes(user_turns))


def _validate_attestation(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("deidentification_attestation must be an object")
    attestation = {
        "consent_scope_sha256": _require_sha256(
            value.get("consent_scope_sha256"),
            field="consent_scope_sha256",
        ),
        "pii_scan_artifact_sha256": _require_sha256(
            value.get("pii_scan_artifact_sha256"),
            field="pii_scan_artifact_sha256",
        ),
        "deidentified_by": _require_string(
            value.get("deidentified_by"), field="deidentified_by"
        ),
    }
    for event_field in (
        "callback_event_present",
        "emotional_event_present",
        "boundary_event_present",
    ):
        if value.get(event_field) is not True:
            raise ValueError(f"{event_field} must be attested true")
        attestation[event_field] = True
    return attestation


def _contrast_contracts(
    preregistration: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    contrasts = preregistration.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != 2:
        raise ValueError("human-anchor preregistration lacks two contrasts")
    resolved = {}
    for contrast in contrasts:
        if not isinstance(contrast, Mapping):
            raise ValueError("human-anchor contrast must be an object")
        contrast_id = _require_string(
            contrast.get("contrast_id"), field="contrast_id"
        )
        if contrast_id in resolved:
            raise ValueError("duplicate human-anchor contrast_id")
        resolved[contrast_id] = contrast
    return resolved


def _validate_capture_records(
    *,
    capture: Mapping[str, object],
    preregistration: Mapping[str, object],
    preregistration_sha256: str,
) -> list[dict[str, object]]:
    if capture.get("schema_version") != GATE811_CAPTURE_SCHEMA_VERSION:
        raise ValueError("Gate 8/11 capture schema drift")
    if capture.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("capture preregistration binding drift")
    capture_contract = preregistration.get("capture")
    if not isinstance(capture_contract, Mapping):
        raise ValueError("preregistration lacks capture contract")
    expected_seeds = capture_contract.get("capture_seeds")
    if not isinstance(expected_seeds, list):
        raise ValueError("preregistration lacks capture seeds")
    session_count = capture_contract.get("session_count_per_transcript")
    total_turns = capture_contract.get("total_turns_per_transcript")
    if not isinstance(session_count, int) or not isinstance(total_turns, int):
        raise ValueError("preregistration transcript shape drift")
    contracts = _contrast_contracts(preregistration)
    records = capture.get("records")
    if not isinstance(records, list):
        raise ValueError("capture records must be a list")
    resolved = []
    record_ids = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("capture record must be an object")
        record_id = _require_string(record.get("record_id"), field="record_id")
        if record_id in record_ids:
            raise ValueError("duplicate capture record_id")
        record_ids.add(record_id)
        contrast_id = _require_string(
            record.get("contrast_id"), field="contrast_id"
        )
        contract = contracts.get(contrast_id)
        if contract is None:
            raise ValueError("capture record uses unknown contrast_id")
        arm_label = _require_string(
            record.get("arm_label"), field="arm_label"
        )
        allowed_arms = {
            contract["experimental_arm"],
            contract["control_arm"],
        }
        if arm_label not in allowed_arms:
            raise ValueError("capture record uses unregistered arm")
        capture_seed = record.get("capture_seed")
        if capture_seed not in expected_seeds:
            raise ValueError("capture record uses unregistered pilot seed")
        sessions = _validated_sessions(
            record.get("sessions"),
            expected_session_count=session_count,
            expected_total_turns=total_turns,
        )
        resolved.append(
            {
                "record_id": record_id,
                "contrast_id": contrast_id,
                "pair_key": _require_string(
                    record.get("pair_key"), field="pair_key"
                ),
                "arm_label": arm_label,
                "capture_seed": capture_seed,
                "source_lineage": _require_sha256(
                    record.get("source_lineage"), field="source_lineage"
                ),
                "persona_ref": _require_string(
                    record.get("persona_ref"), field="persona_ref"
                ),
                "model_and_adapter_fingerprint": _require_sha256(
                    record.get("model_and_adapter_fingerprint"),
                    field="model_and_adapter_fingerprint",
                ),
                "sessions": list(sessions),
                "user_turn_digest": _user_turn_digest(sessions),
                "deidentification_attestation": _validate_attestation(
                    record.get("deidentification_attestation")
                ),
            }
        )
    return resolved


def _rating_template_csv(
    *,
    pairs: Sequence[Mapping[str, object]],
    minimum_unique_raters: int,
    ratings_per_pair: int,
) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=_RATING_COLUMNS)
    writer.writeheader()
    for pair_index, pair in enumerate(pairs):
        for offset in range(ratings_per_pair):
            slot = (pair_index + offset) % minimum_unique_raters + 1
            writer.writerow(
                {
                    "rater_slot": f"rater-slot-{slot:02d}",
                    "rater_id": "",
                    "pair_id": pair["pair_id"],
                    "a_rememberedness": "",
                    "b_rememberedness": "",
                    "a_relationship_continuity": "",
                    "b_relationship_continuity": "",
                    "a_boundary_respect": "",
                    "b_boundary_respect": "",
                    "forced_preference": "",
                    "malformed_reason": "",
                }
            )
    return handle.getvalue()


def build_gate811_pilot_packet(
    *,
    capture: Mapping[str, object],
    preregistration: Mapping[str, object],
    preregistration_sha256: str,
) -> dict[str, object]:
    """Select, pair, blind, and format the frozen L4-A pilot."""

    _require_sha256(
        preregistration_sha256, field="preregistration_sha256"
    )
    records = _validate_capture_records(
        capture=capture,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
    )
    pilot = preregistration.get("pilot")
    blinding = preregistration.get("blinding")
    rating = preregistration.get("rating")
    if not all(isinstance(value, Mapping) for value in (pilot, blinding, rating)):
        raise ValueError("preregistration lacks pilot/blinding/rating contract")
    assert isinstance(pilot, Mapping)
    assert isinstance(blinding, Mapping)
    assert isinstance(rating, Mapping)
    pairs_per_contrast = pilot.get("pairs_per_contrast")
    ratings_per_pair = pilot.get("ratings_per_pair")
    minimum_unique_raters = pilot.get("minimum_unique_raters")
    if not all(
        isinstance(value, int)
        for value in (
            pairs_per_contrast,
            ratings_per_pair,
            minimum_unique_raters,
        )
    ):
        raise ValueError("pilot integer contract drift")
    assert isinstance(pairs_per_contrast, int)
    assert isinstance(ratings_per_pair, int)
    assert isinstance(minimum_unique_raters, int)
    contrasts = _contrast_contracts(preregistration)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["contrast_id"]), str(record["pair_key"]))
        grouped.setdefault(key, []).append(record)
    matched_by_contrast: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {
        contrast_id: [] for contrast_id in contrasts
    }
    for (contrast_id, _pair_key), pair_records in grouped.items():
        if len(pair_records) != 2:
            raise ValueError("each pilot pair must contain exactly two arms")
        contract = contrasts[contrast_id]
        by_arm = {str(record["arm_label"]): record for record in pair_records}
        if set(by_arm) != {
            contract["experimental_arm"],
            contract["control_arm"],
        }:
            raise ValueError("pilot pair arm coverage drift")
        experimental = by_arm[str(contract["experimental_arm"])]
        control = by_arm[str(contract["control_arm"])]
        for matched_field in (
            "capture_seed",
            "source_lineage",
            "persona_ref",
            "model_and_adapter_fingerprint",
            "user_turn_digest",
        ):
            if experimental[matched_field] != control[matched_field]:
                raise ValueError(f"pilot pair {matched_field} mismatch")
        for attestation_field in (
            "consent_scope_sha256",
            "pii_scan_artifact_sha256",
        ):
            if (
                experimental["deidentification_attestation"][
                    attestation_field
                ]
                != control["deidentification_attestation"][
                    attestation_field
                ]
            ):
                raise ValueError(
                    f"pilot pair {attestation_field} mismatch"
                )
        matched_by_contrast[contrast_id].append((experimental, control))
    selection_rng = random.Random(blinding["pilot_selection_seed"])
    selected = []
    for contrast_id in sorted(contrasts):
        candidates = sorted(
            matched_by_contrast[contrast_id],
            key=lambda pair: str(pair[0]["pair_key"]),
        )
        if len(candidates) < pairs_per_contrast:
            raise ValueError("insufficient matched pairs for frozen pilot")
        selected.extend(
            (contrast_id, pair)
            for pair in selection_rng.sample(candidates, pairs_per_contrast)
        )
    orientation_rng = random.Random(blinding["orientation_and_order_seed"])
    packet_pairs = []
    key_entries = []
    for contrast_id, (experimental, control) in selected:
        pair_id = hashlib.sha256(
            (
                f"{preregistration_sha256}:{contrast_id}:"
                f"{experimental['pair_key']}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        if orientation_rng.randrange(2) == 0:
            side_a, side_b = experimental, control
        else:
            side_a, side_b = control, experimental
        packet_pairs.append(
            {
                "pair_id": pair_id,
                "transcript_a": side_a["sessions"],
                "transcript_b": side_b["sessions"],
                "rating_dimensions": list(rating["dimensions"]),
                "forced_preference": rating["forced_preference_prompt"],
            }
        )
        key_entries.append(
            {
                "pair_id": pair_id,
                "contrast_id": contrast_id,
                "gate_id": contrasts[contrast_id]["gate_id"],
                "side_a_arm": side_a["arm_label"],
                "side_b_arm": side_b["arm_label"],
                "side_a_record_id": side_a["record_id"],
                "side_b_record_id": side_b["record_id"],
                "source_lineage": experimental["source_lineage"],
                "capture_seed": experimental["capture_seed"],
                "consent_scope_sha256": experimental[
                    "deidentification_attestation"
                ]["consent_scope_sha256"],
                "pii_scan_artifact_sha256": experimental[
                    "deidentification_attestation"
                ]["pii_scan_artifact_sha256"],
            }
        )
    orientation_rng.shuffle(packet_pairs)
    key_by_id = {entry["pair_id"]: entry for entry in key_entries}
    ordered_key_entries = [key_by_id[pair["pair_id"]] for pair in packet_pairs]
    packet = {
        "schema_version": GATE811_PACKET_SCHEMA_VERSION,
        "preregistration_sha256": preregistration_sha256,
        "pilot_only": True,
        "human_anchor_claim_allowed": False,
        "scale": {"min": rating["scale_min"], "max": rating["scale_max"]},
        "pair_count": len(packet_pairs),
        "pairs": packet_pairs,
    }
    internal_key = {
        "schema_version": GATE811_PACKET_SCHEMA_VERSION,
        "preregistration_sha256": preregistration_sha256,
        "do_not_distribute_to_raters": True,
        "entries": ordered_key_entries,
    }
    rating_csv = _rating_template_csv(
        pairs=packet_pairs,
        minimum_unique_raters=minimum_unique_raters,
        ratings_per_pair=ratings_per_pair,
    )
    return {
        "packet": packet,
        "internal_key": internal_key,
        "rating_template_csv": rating_csv,
    }


def export_gate811_pilot_packet(
    *,
    bundle: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    packet_bytes = _canonical_bytes(bundle["packet"])
    key_bytes = _canonical_bytes(bundle["internal_key"])
    rating_csv = bundle["rating_template_csv"]
    if not isinstance(rating_csv, str):
        raise ValueError("rating_template_csv must be text")
    files = {
        "pilot_packet_blinded.json": packet_bytes,
        "pilot_key_internal.json": key_bytes,
        "pilot_rating_template.csv": rating_csv.encode("utf-8"),
    }
    for relative, content in files.items():
        (output / relative).write_bytes(content)
    manifest = {
        "schema_version": GATE811_PACKET_SCHEMA_VERSION,
        "rating_template_schema_version": (
            GATE811_RATING_TEMPLATE_SCHEMA_VERSION
        ),
        "required_files": sorted(files),
        "sha256": {
            relative: _sha256_bytes(content)
            for relative, content in sorted(files.items())
        },
        "pilot_only": True,
        "human_anchor_claim_allowed": False,
        "production_promotion_authorized": False,
    }
    (output / "manifest.json").write_bytes(_canonical_bytes(manifest))
    return manifest


__all__ = [
    "GATE811_CAPTURE_SCHEMA_VERSION",
    "GATE811_PACKET_SCHEMA_VERSION",
    "GATE811_RATING_TEMPLATE_SCHEMA_VERSION",
    "build_gate811_pilot_packet",
    "export_gate811_pilot_packet",
]
