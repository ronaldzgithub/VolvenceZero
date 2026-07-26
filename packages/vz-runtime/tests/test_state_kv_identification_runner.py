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
RecordingSynthesizer = _RUNNER.RecordingSynthesizer
_base_context = _RUNNER._base_context
_fingerprint_weights = _RUNNER._fingerprint_weights
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


@pytest.mark.parametrize(
    "argv",
    [
        # Arm G with no artifact could only run by falling back to another
        # carrier, which would publish a mislabelled arm.
        ["--lane", "p3"],
        # The prefix artifact has no meaning on a lane that never runs arm G.
        ["--lane", "p1", "--prefix-kv-artifact", "prefix.json"],
        ["--lane", "smoke", "--prefix-kv-artifact", "prefix.json"],
        # Sampling on a frozen lane would let RNG masquerade as a carrier.
        ["--lane", "p3", "--prefix-kv-artifact", "p.json", "--temperature", "0.7"],
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
