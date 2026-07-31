"""Build Gate 8/11 pilot captures from seven-day simulated-user runs.

The frozen v1 preregistration constrains transcript matching and requires
human *raters*, but does not require the chatting party to be human.  This
adapter therefore keeps v1 intact and labels every resulting claim as
human-rated simulated-user evidence, never real-user product evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

from volvence_zero.agent.gate811_human_anchor_tooling import (
    GATE811_CAPTURE_SCHEMA_VERSION,
    build_gate811_pilot_packet,
    export_gate811_pilot_packet,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayRunEnvelope,
)


GATE811_SIMULATED_COMPATIBILITY_SCHEMA_VERSION = (
    "gate811-simulated-capture-compatibility.v1"
)
_CONTRASTS = (
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
_REQUIRED_EVENTS = frozenset({"callback", "emotion", "boundary"})
_STATE_POLICY_BY_ARM = {
    "correct-user-state": "correct-user-state",
    "stateless": "stateless",
    "sleep-consolidation": "correct-user-state",
    "no-sleep": "correct-user-state",
}


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


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


@dataclass(frozen=True)
class Gate811SimulatedCompatibility:
    schema_version: str
    compatible_with_frozen_v1: bool
    requires_v2_preregistration: bool
    capture_source_scope: str
    resulting_claim_scope: str
    human_raters_still_required: bool
    production_promotion_authorized: bool
    reasons: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "compatible_with_frozen_v1": self.compatible_with_frozen_v1,
            "requires_v2_preregistration": self.requires_v2_preregistration,
            "capture_source_scope": self.capture_source_scope,
            "resulting_claim_scope": self.resulting_claim_scope,
            "human_raters_still_required": self.human_raters_still_required,
            "production_promotion_authorized": (
                self.production_promotion_authorized
            ),
            "reasons": list(self.reasons),
        }


def audit_gate811_simulated_capture_compatibility(
    preregistration: Mapping[str, object],
) -> Gate811SimulatedCompatibility:
    """Audit frozen v1 without weakening or rewriting it."""

    if preregistration.get("schema_version") != (
        "gate811-human-anchor-prereg.v1"
    ):
        raise ValueError("Gate 8/11 human-anchor preregistration schema drift")
    capture = _require_mapping(
        preregistration.get("capture"), field="capture"
    )
    authorization = _require_mapping(
        preregistration.get("authorization"), field="authorization"
    )
    source_population = capture.get("source_population")
    explicitly_real_only = source_population in {
        "real-user-only",
        "human-chat-participant-only",
    }
    required = {
        "matched_variables",
        "only_manipulated_variable",
        "session_count_per_transcript",
        "total_turns_per_transcript",
        "capture_seeds",
    }
    missing = sorted(required - set(capture))
    if missing:
        raise ValueError(f"frozen v1 capture contract lacks {missing!r}")
    compatible = not explicitly_real_only
    return Gate811SimulatedCompatibility(
        schema_version=GATE811_SIMULATED_COMPATIBILITY_SCHEMA_VERSION,
        compatible_with_frozen_v1=compatible,
        requires_v2_preregistration=not compatible,
        capture_source_scope="synthetic-user-no-human-chat-participant",
        resulting_claim_scope=(
            "human-rated-simulated-user-transcripts-only"
            if compatible
            else "not-admissible-under-v1"
        ),
        human_raters_still_required=(
            authorization.get("human_recruitment_required") is True
        ),
        production_promotion_authorized=False,
        reasons=(
            "v1 freezes transcript matching and shape, not a real-user-only source",
            "human_recruitment_required applies to blinded raters",
            "pilot authorization remains non-claim and non-production",
        ),
    )


def _days(run: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_days = run.get("days")
    if not isinstance(raw_days, (list, tuple)) or len(raw_days) != 7:
        raise ValueError("capture source must contain seven days")
    return tuple(_require_mapping(day, field="day") for day in raw_days)


def _validate_source_run(run: Mapping[str, object], *, arm: str) -> None:
    if run.get("schema_version") != "seven-day-companion-run.v1":
        raise ValueError("capture source run schema drift")
    if run.get("process_restart_count") != 6 or run.get(
        "all_restarts_exact"
    ) is not True:
        raise ValueError("capture source lacks six exact restarts")
    if run.get("simulated_longitudinal_only") is not True:
        raise ValueError("capture source claim scope drift")
    if run.get("external_human_value_claim_allowed") is not False:
        raise ValueError("capture source may not claim human value")
    if run.get("production_promotion_authorized") is not False:
        raise ValueError("capture source may not authorize production")
    expected_policy = _STATE_POLICY_BY_ARM[arm]
    for day_index, day in enumerate(_days(run), start=1):
        restart = day.get("restart_after_day")
        if day_index == 7:
            if restart is not None:
                raise ValueError("capture source day seven may not restart")
            continue
        restart_payload = _require_mapping(
            restart, field="restart_after_day"
        )
        intervention = _require_mapping(
            restart_payload.get("state_intervention"),
            field="state_intervention",
        )
        if (
            intervention.get("experiment_arm_label") != arm
            or intervention.get("state_loading_policy") != expected_policy
            or intervention.get("after_day_index") != day_index
        ):
            raise ValueError("capture source state intervention drift")


def _events(day: Mapping[str, object]) -> frozenset[str]:
    turns = day.get("turns")
    if not isinstance(turns, (list, tuple)):
        raise ValueError("capture source day lacks turns")
    tags = set()
    for turn in turns:
        typed = _require_mapping(turn, field="turn")
        raw_tags = typed.get("event_tags")
        if not isinstance(raw_tags, (list, tuple)):
            raise ValueError("capture source turn lacks typed event_tags")
        tags.update(str(tag) for tag in raw_tags)
    return frozenset(tags)


def _eligible_windows(
    run: Mapping[str, object],
) -> tuple[tuple[int, int, int], ...]:
    days = _days(run)
    windows = []
    for indices in itertools.combinations(range(7), 3):
        coverage = set()
        for index in indices:
            coverage.update(_events(days[index]))
        if _REQUIRED_EVENTS.issubset(coverage):
            windows.append(indices)
    if len(windows) < 2:
        raise ValueError(
            "capture source needs two distinct three-session event-complete windows"
        )
    return tuple(windows[:2])


def _sessions(
    run: Mapping[str, object],
    indices: Sequence[int],
) -> list[dict[str, object]]:
    days = _days(run)
    sessions = []
    for session_index, day_index in enumerate(indices):
        turns = days[day_index].get("turns")
        if not isinstance(turns, (list, tuple)) or len(turns) != 5:
            raise ValueError("capture session must contain five exchanges")
        transcript = []
        for turn in turns:
            typed = _require_mapping(turn, field="turn")
            transcript.extend(
                (
                    {
                        "speaker": "user",
                        "text": _require_string(
                            typed.get("user_text"), field="user_text"
                        ),
                    },
                    {
                        "speaker": "assistant",
                        "text": _require_string(
                            typed.get("assistant_text"),
                            field="assistant_text",
                        ),
                    },
                )
            )
        sessions.append(
            {"session_index": session_index, "turns": transcript}
        )
    return sessions


def _user_turns(sessions: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    values = []
    for session in sessions:
        turns = session["turns"]
        assert isinstance(turns, list)
        for turn in turns:
            assert isinstance(turn, Mapping)
            if turn["speaker"] == "user":
                values.append(str(turn["text"]))
    return tuple(values)


def build_gate811_simulated_capture(
    *,
    runs: Sequence[SevenDayRunEnvelope],
    preregistration: Mapping[str, object],
    preregistration_sha256: str,
    deidentified_by: str = "automated:synthetic-source-attestor",
) -> dict[str, object]:
    """Convert exact-matched seven-day runs into frozen v1 capture records."""

    compatibility = audit_gate811_simulated_capture_compatibility(
        preregistration
    )
    if not compatibility.compatible_with_frozen_v1:
        raise ValueError("simulated capture is not compatible with frozen v1")
    if len(preregistration_sha256) != 64:
        raise ValueError("preregistration_sha256 must be a SHA-256 digest")
    _require_string(deidentified_by, field="deidentified_by")
    capture_contract = _require_mapping(
        preregistration.get("capture"), field="capture"
    )
    capture_seeds = capture_contract.get("capture_seeds")
    if not isinstance(capture_seeds, list):
        raise ValueError("frozen v1 capture seeds are missing")
    keyed = {(item.case.case_id, item.arm_label): item.run for item in runs}
    if len(keyed) != len(runs):
        raise ValueError("capture source has duplicate case/arm records")
    case_ids = tuple(sorted({item.case.case_id for item in runs}))
    case_by_id = {item.case.case_id: item.case for item in runs}
    records = []
    for case_id in case_ids:
        case = case_by_id[case_id]
        if case.paraphrase_seed not in capture_seeds:
            raise ValueError("capture source uses unregistered capture seed")
        required_arms = {
            arm for _contrast, experimental, control in _CONTRASTS
            for arm in (experimental, control)
        }
        missing = sorted(
            arm for arm in required_arms if (case_id, arm) not in keyed
        )
        if missing:
            raise ValueError(f"capture source lacks arms {missing!r}")
        reference = keyed[(case_id, "correct-user-state")]
        windows = _eligible_windows(reference)
        for contrast_id, experimental, control in _CONTRASTS:
            for window_index, indices in enumerate(windows):
                pair_key = (
                    f"{contrast_id}:{case_id}:window-{window_index + 1}"
                )
                source_lineage = _sha256(
                    {
                        "scenario_id": case.scenario_id,
                        "paraphrase_seed": case.paraphrase_seed,
                        "day_indices": [index + 1 for index in indices],
                    }
                )
                pair_sessions = {}
                for arm in (experimental, control):
                    run = keyed[(case_id, arm)]
                    _validate_source_run(run, arm=arm)
                    if run.get("scenario_id") != case.scenario_id:
                        raise ValueError("capture source scenario drift")
                    sessions = _sessions(run, indices)
                    pair_sessions[arm] = sessions
                    attestation = _require_mapping(
                        run.get("source_attestation"),
                        field="source_attestation",
                    )
                    consent_scope = _require_string(
                        attestation.get("consent_scope"),
                        field="consent_scope",
                    )
                    records.append(
                        {
                            "record_id": _sha256(
                                {
                                    "pair_key": pair_key,
                                    "arm_label": arm,
                                }
                            ),
                            "contrast_id": contrast_id,
                            "pair_key": pair_key,
                            "arm_label": arm,
                            "capture_seed": case.paraphrase_seed,
                            "source_lineage": source_lineage,
                            "persona_ref": _require_string(
                                run.get("persona_ref"), field="persona_ref"
                            ),
                            "model_and_adapter_fingerprint": _require_string(
                                attestation.get(
                                    "model_and_adapter_fingerprint"
                                ),
                                field="model_and_adapter_fingerprint",
                            ),
                            "sessions": sessions,
                            "deidentification_attestation": {
                                "consent_scope_sha256": hashlib.sha256(
                                    consent_scope.encode("utf-8")
                                ).hexdigest(),
                                "pii_scan_artifact_sha256": _require_string(
                                    attestation.get(
                                        "pii_scan_artifact_sha256"
                                    ),
                                    field="pii_scan_artifact_sha256",
                                ),
                                "deidentified_by": deidentified_by,
                                "callback_event_present": True,
                                "emotional_event_present": True,
                                "boundary_event_present": True,
                            },
                        }
                    )
                if _user_turns(pair_sessions[experimental]) != _user_turns(
                    pair_sessions[control]
                ):
                    raise ValueError(
                        "capture pair user turns are not byte-identical"
                    )
    return {
        "schema_version": GATE811_CAPTURE_SCHEMA_VERSION,
        "preregistration_sha256": preregistration_sha256,
        "capture_source_scope": (
            "synthetic-user-no-human-chat-participant"
        ),
        "resulting_claim_scope": (
            "human-rated-simulated-user-transcripts-only"
        ),
        "real_user_product_value_claim_allowed": False,
        "records": records,
    }


def export_gate811_simulated_pilot(
    *,
    runs: Sequence[SevenDayRunEnvelope],
    preregistration: Mapping[str, object],
    preregistration_sha256: str,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write the source capture and existing-tooling blinded packet."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    compatibility = audit_gate811_simulated_capture_compatibility(
        preregistration
    )
    capture = build_gate811_simulated_capture(
        runs=runs,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
    )
    (target / "simulated_capture.json").write_bytes(
        _canonical_bytes(capture)
    )
    (target / "compatibility_audit.json").write_bytes(
        _canonical_bytes(compatibility.to_json())
    )
    bundle = build_gate811_pilot_packet(
        capture=capture,
        preregistration=preregistration,
        preregistration_sha256=preregistration_sha256,
    )
    manifest = export_gate811_pilot_packet(
        bundle=bundle,
        output_dir=target,
    )
    return {
        **manifest,
        "capture_record_count": len(capture["records"]),
        "capture_source_scope": capture["capture_source_scope"],
        "real_user_product_value_claim_allowed": False,
        "human_ratings_pending": True,
    }


__all__ = [
    "GATE811_SIMULATED_COMPATIBILITY_SCHEMA_VERSION",
    "Gate811SimulatedCompatibility",
    "audit_gate811_simulated_capture_compatibility",
    "build_gate811_simulated_capture",
    "export_gate811_simulated_pilot",
]
