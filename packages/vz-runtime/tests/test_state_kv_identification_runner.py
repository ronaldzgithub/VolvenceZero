"""Acceptance for the State-KV smoke/P1 command-line evidence runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from volvence_zero.agent.response import LLMResponseSynthesizer
from volvence_zero.state_kv_identification import (
    C5Grade,
    IDENTIFICATION_ARM_LABELS,
    SubstrateEvidenceKind,
    run_identification_smoke,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "run_state_kv_identification.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_state_kv_identification",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load State-KV runner from {_SCRIPT_PATH}")
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)

DeterministicFakeSubstrate = _RUNNER.DeterministicFakeSubstrate
P2_PERSONA_PAIRS = _RUNNER.P2_PERSONA_PAIRS
P2_PROBE_SENTENCES = _RUNNER.P2_PROBE_SENTENCES
PERSONAS = _RUNNER.PERSONAS
PROBE_SENTENCES = _RUNNER.PROBE_SENTENCES
RecordingSynthesizer = _RUNNER.RecordingSynthesizer
_base_context = _RUNNER._base_context
_fingerprint_weights = _RUNNER._fingerprint_weights
_judge_materials_from_cases = _RUNNER._judge_materials_from_cases
build_p2_probe_cases = _RUNNER.build_p2_probe_cases
build_probe_cases = _RUNNER.build_probe_cases
main = _RUNNER.main


def test_strict_probe_cases_close_decode_carrier() -> None:
    cases = build_probe_cases(strict_carriers=True)
    runtime = DeterministicFakeSubstrate()
    recording = RecordingSynthesizer(LLMResponseSynthesizer(runtime=runtime))

    verdict = run_identification_smoke(
        cases=cases,
        synthesizer=recording,
        base_context=_base_context(),
        substrate_kind=SubstrateEvidenceKind.TRACE_ONLY,
        substrate_fingerprint=runtime.fingerprint,
        arm_labels=IDENTIFICATION_ARM_LABELS,
    )

    assert verdict.c5_grade is C5Grade.DECODE_MATCHED
    assert {case.assembly for case in cases} == {cases[0].assembly}
    assert len({case.conditioning.state_vector for case in cases}) == 2


def test_probe_personas_are_semantically_counterfactual_not_scalar_fills() -> None:
    cases = build_probe_cases(strict_carriers=True)
    states = {
        case.user_id: dict(
            zip(
                case.conditioning.vector_labels,
                case.conditioning.state_vector,
                strict=True,
            )
        )
        for case in cases
    }

    assert (
        states["persona-a"]["relationship_repair_need"]
        > states["persona-b"]["relationship_repair_need"]
    )
    assert (
        states["persona-a"]["boundary_autonomy_risk"]
        > states["persona-b"]["boundary_autonomy_risk"]
    )
    assert (
        states["persona-a"]["goal_decision_readiness"]
        < states["persona-b"]["goal_decision_readiness"]
    )
    assert (
        states["persona-a"]["relationship_trust"]
        < states["persona-b"]["relationship_trust"]
    )


def test_blind_judge_material_uses_owner_rendered_state() -> None:
    cases = build_probe_cases(strict_carriers=True)
    materials = _judge_materials_from_cases(cases)

    assert [material.user_id for material in materials] == [
        "persona-a",
        "persona-b",
    ]
    assert {material.material_kind for material in materials} == {
        "rendered-state-statement"
    }
    by_user = {case.user_id: case.conditioning.rendered_statement for case in cases}
    assert {material.summary for material in materials} == set(by_user.values())


def test_p2_probe_cases_are_held_out_and_pairwise() -> None:
    cases = build_p2_probe_cases(pair_id="repair-vs-execute")

    assert len(cases) == 2 * len(P2_PROBE_SENTENCES)
    assert {case.user_id for case in cases} == {
        "heldout-repair",
        "heldout-execute",
    }
    assert {sentence for _, sentence in P2_PROBE_SENTENCES}.isdisjoint(
        {sentence for _, sentence in PROBE_SENTENCES}
    )
    assert {
        persona[1]
        for pair in P2_PERSONA_PAIRS.values()
        for persona in pair
    }.isdisjoint({persona[1] for persona in PERSONAS})
    assert {case.assembly for case in cases} == {cases[0].assembly}


def test_p2_unknown_pair_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown P2 persona pair"):
        build_p2_probe_cases(pair_id="missing")


def test_smoke_cli_writes_verdict_transcript_and_fingerprint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "verdict_identification.json"

    assert main(["--lane", "smoke", "--output", str(output)]) == 0

    transcript = output.with_name("transcript_identification.json")
    fingerprint = output.with_name("substrate_fingerprint.json")
    assert output.is_file()
    assert transcript.is_file()
    assert fingerprint.is_file()
    transcript_payload = json.loads(transcript.read_text(encoding="utf-8"))
    assert transcript_payload["lane"] == "smoke"
    assert len(transcript_payload["turns"]) == (
        len(IDENTIFICATION_ARM_LABELS) * len(build_probe_cases())
    )
    assert all("rationale_tags" in turn for turn in transcript_payload["turns"])


def test_smoke_cli_can_resume_from_turn_cache(tmp_path: Path) -> None:
    output = tmp_path / "verdict_identification.json"

    assert (
        main(
            [
                "--lane",
                "smoke",
                "--resume-turn-cache",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    cache = output.with_name("turn_cache_identification.jsonl")
    first_lines = cache.read_text(encoding="utf-8").splitlines()
    assert len(first_lines) == len(IDENTIFICATION_ARM_LABELS) * len(build_probe_cases())

    assert (
        main(
            [
                "--lane",
                "smoke",
                "--resume-turn-cache",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    second_lines = cache.read_text(encoding="utf-8").splitlines()
    assert second_lines == first_lines

    fingerprint = json.loads(
        output.with_name("substrate_fingerprint.json").read_text(encoding="utf-8")
    )
    material = fingerprint["identification_material"]
    assert material["turn_cache_schema_version"] == "state-kv-turn-cache.v1"
    assert len(material["turn_cache_key"]) == 64


@pytest.mark.parametrize(
    "argv",
    [
        # Arm G with no artifact could only run by falling back to another
        # carrier, which would publish a mislabelled arm.
        ["--lane", "p3"],
        ["--lane", "p2"],
        # The prefix artifact has no meaning on a lane that never runs arm G.
        ["--lane", "p1", "--prefix-kv-artifact", "prefix.json"],
        ["--lane", "smoke", "--prefix-kv-artifact", "prefix.json"],
        # Sampling on a frozen lane needs an aligned rollout seed.
        ["--lane", "p3", "--prefix-kv-artifact", "p.json", "--temperature", "0.7"],
        ["--lane", "p2", "--prefix-kv-artifact", "p.json", "--temperature", "0.7"],
        ["--lane", "p3", "--prefix-kv-artifact", "p.json", "--sampling-seed", "7"],
        ["--lane", "smoke", "--sampling-seed", "7"],
        # A trace-only fake substrate cannot satisfy the cross-family rule.
        ["--lane", "smoke", "--judge-model-id", "TinyLlama/test"],
    ],
)
def test_lane_argument_combinations_fail_loudly(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 2


def test_weight_fingerprint_is_content_addressed(tmp_path: Path) -> None:
    first = tmp_path / "model.safetensors"
    second = tmp_path / "shard.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    before = _fingerprint_weights(model_id="Qwen/test", weights_root=tmp_path)
    second.write_bytes(b"changed")
    after = _fingerprint_weights(model_id="Qwen/test", weights_root=tmp_path)

    assert before["weight_file_count"] == 2
    assert before["weights_sha256"] != after["weights_sha256"]
