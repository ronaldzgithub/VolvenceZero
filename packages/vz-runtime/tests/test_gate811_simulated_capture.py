from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from volvence_zero.agent.gate811_human_anchor_tooling import (
    build_gate811_pilot_packet,
)
from volvence_zero.agent.gate811_simulated_capture import (
    audit_gate811_simulated_capture_compatibility,
    build_gate811_simulated_capture,
    export_gate811_simulated_pilot,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayExperimentCase,
    SevenDayRunEnvelope,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PREREG_PATH = (
    REPO_ROOT
    / "artifacts/gate811_human_anchor_prereg_20260731T180225Z.json"
)
ARMS = (
    "correct-user-state",
    "stateless",
    "sleep-consolidation",
    "no-sleep",
)


def _prereg() -> dict[str, object]:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _run(case: SevenDayExperimentCase, arm: str) -> dict[str, object]:
    days = []
    for day_index in range(1, 8):
        turns = []
        for exchange_index in range(1, 6):
            tags = []
            if day_index == 1 and exchange_index == 1:
                tags.append("emotion")
            if day_index == 4 and exchange_index == 1:
                tags.append("boundary")
            if day_index in (5, 7) and exchange_index == 1:
                tags.append("callback")
            turns.append(
                {
                    "user_text": (
                        f"{case.scenario_id}:{case.paraphrase_seed}:"
                        f"{day_index}:{exchange_index}"
                    ),
                    "assistant_text": f"{arm}:{day_index}:{exchange_index}",
                    "event_tags": tags,
                }
            )
        policy = {
            "correct-user-state": "correct-user-state",
            "stateless": "stateless",
            "sleep-consolidation": "correct-user-state",
            "no-sleep": "correct-user-state",
        }[arm]
        stateful = policy != "stateless"
        restart = (
            {
                "after_day_index": day_index,
                "state_intervention": {
                    "experiment_arm_label": arm,
                    "state_loading_policy": policy,
                    "after_day_index": day_index,
                    "archived_state_sha256": "4" * 64,
                    "measurement_checkpoint_sha256": "6" * 64,
                    "next_day_source_day_index": (
                        day_index if stateful else None
                    ),
                    "next_day_loaded_state_sha256": (
                        "5" * 64 if stateful else None
                    ),
                },
            }
            if day_index < 7
            else None
        )
        days.append(
            {
                "day_index": day_index,
                "turns": turns,
                "restart_after_day": restart,
            }
        )
    return {
        "schema_version": "seven-day-companion-run.v1",
        "scenario_id": case.scenario_id,
        "paraphrase_seed": case.paraphrase_seed,
        "persona_ref": case.scenario_id.split("-")[-1],
        "process_restart_count": 6,
        "all_restarts_exact": True,
        "simulated_longitudinal_only": True,
        "external_human_value_claim_allowed": False,
        "production_promotion_authorized": False,
        "source_attestation": {
            "consent_scope": "synthetic-no-human-subject",
            "pii_scan_artifact_sha256": "2" * 64,
            "model_and_adapter_fingerprint": "3" * 64,
        },
        "days": days,
    }


def _runs() -> list[SevenDayRunEnvelope]:
    scenarios = tuple(
        f"{family}-seven-day-{persona}"
        for persona in ("researcher", "nurse", "designer")
        for family in ("F1", "F2")
    )
    cases = tuple(
        SevenDayExperimentCase(scenario, seed)
        for scenario in scenarios
        for seed in (1401, 1413, 1427)
    )
    return [
        SevenDayRunEnvelope(
            case=case,
            arm_label=arm,
            run=_run(case, arm),
        )
        for case in cases
        for arm in ARMS
    ]


def test_frozen_v1_accepts_simulated_chat_source_but_not_real_user_claim() -> None:
    audit = audit_gate811_simulated_capture_compatibility(_prereg())
    assert audit.compatible_with_frozen_v1 is True
    assert audit.requires_v2_preregistration is False
    assert audit.human_raters_still_required is True
    assert audit.production_promotion_authorized is False
    assert audit.resulting_claim_scope == (
        "human-rated-simulated-user-transcripts-only"
    )


def test_capture_supplies_enough_exact_pairs_for_frozen_pilot() -> None:
    prereg = _prereg()
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    capture = build_gate811_simulated_capture(
        runs=_runs(),
        preregistration=prereg,
        preregistration_sha256=prereg_sha,
    )
    assert len(capture["records"]) == 144
    assert capture["real_user_product_value_claim_allowed"] is False
    bundle = build_gate811_pilot_packet(
        capture=capture,
        preregistration=prereg,
        preregistration_sha256=prereg_sha,
    )
    assert bundle["packet"]["pair_count"] == 48
    assert bundle["packet"]["human_anchor_claim_allowed"] is False


def test_capture_rejects_arm_specific_user_turns() -> None:
    runs = _runs()
    target = next(
        index
        for index, envelope in enumerate(runs)
        if envelope.arm_label == "stateless"
    )
    mutated = deepcopy(runs[target].run)
    mutated["days"][0]["turns"][0]["user_text"] = "mismatched"
    runs[target] = SevenDayRunEnvelope(
        case=runs[target].case,
        arm_label=runs[target].arm_label,
        run=mutated,
    )
    with pytest.raises(ValueError, match="byte-identical"):
        build_gate811_simulated_capture(
            runs=runs,
            preregistration=_prereg(),
            preregistration_sha256="1" * 64,
        )


def test_explicit_real_user_only_clause_requires_new_preregistration() -> None:
    prereg = deepcopy(_prereg())
    prereg["capture"]["source_population"] = "real-user-only"
    audit = audit_gate811_simulated_capture_compatibility(prereg)
    assert audit.compatible_with_frozen_v1 is False
    assert audit.requires_v2_preregistration is True


def test_export_writes_capture_audit_and_blinded_packet(tmp_path: Path) -> None:
    prereg = _prereg()
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    manifest = export_gate811_simulated_pilot(
        runs=_runs(),
        preregistration=prereg,
        preregistration_sha256=prereg_sha,
        output_dir=tmp_path,
    )
    assert manifest["capture_record_count"] == 144
    assert manifest["human_ratings_pending"] is True
    for relative in (
        "simulated_capture.json",
        "compatibility_audit.json",
        "pilot_packet_blinded.json",
        "pilot_key_internal.json",
        "pilot_rating_template.csv",
        "manifest.json",
    ):
        assert (tmp_path / relative).is_file()
