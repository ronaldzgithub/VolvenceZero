"""Gate A / gate B claim logic for the State-KV prefix carrier.

These gates exist to separate two diagnoses that a chance-level wrong-user
control leaves entangled: attention never reads the slots, versus it reads them
and they carry no state. The tests below pin the properties that make that
separation trustworthy -- above all that neither gate can be passed by a
degenerate prefix, and that gate B cannot be passed by leakage.
"""

from __future__ import annotations

import json

import pytest

from volvence_zero.state_kv_carrier_diagnostics import (
    PROBE_R2_FLOOR,
    build_carrier_diagnostics_verdict,
    evaluate_slot_attention_read,
    evaluate_state_linearly_readable,
)
from volvence_zero.state_kv_identification import ClaimState

LAYERS = 24


def _attention(
    *,
    learned: float,
    control: float = 1.15,
    beaten_layers: int = LAYERS,
    state_spread: float = 0.02,
    sentence_spread: float = 0.001,
):
    learned_profile = [
        learned if index < beaten_layers else control * 0.5
        for index in range(LAYERS)
    ]
    return evaluate_slot_attention_read(
        learned_nonuniformity=learned_profile,
        control_nonuniformity=[control] * LAYERS,
        state_spread=state_spread,
        sentence_spread=sentence_spread,
    )


def test_gate_a_passes_when_slots_are_differentiated_and_state_modulated() -> None:
    result = _attention(learned=2.0)

    assert result.state is ClaimState.PASS


def test_gate_a_fails_a_degenerate_prefix() -> None:
    # A zero-content prefix scores exactly 1.0 for non-uniformity: identical
    # slots cannot be told apart. It must not pass on the strength of drawing
    # a lot of attention.
    result = _attention(learned=1.0)

    assert result.state is ClaimState.FAIL
    assert "not differentiated" in result.detail


def test_gate_a_needs_a_majority_of_layers() -> None:
    half = _attention(learned=2.0, beaten_layers=LAYERS // 2)
    just_over = _attention(learned=2.0, beaten_layers=LAYERS // 2 + 1)

    assert half.state is ClaimState.FAIL
    assert just_over.state is ClaimState.PASS


def test_gate_a_fails_when_attention_tracks_the_sentence_not_the_state() -> None:
    result = _attention(learned=2.0, state_spread=0.001, sentence_spread=0.02)

    assert result.state is ClaimState.FAIL
    assert "probe sentence" in result.detail


def test_gate_a_reports_insufficient_data_on_mismatched_profiles() -> None:
    result = evaluate_slot_attention_read(
        learned_nonuniformity=[1.5, 1.5],
        control_nonuniformity=[1.1],
        state_spread=1.0,
        sentence_spread=0.0,
    )

    assert result.state is ClaimState.INSUFFICIENT_DATA


def _readable(
    *,
    held_out: float,
    shuffled: float = -0.03,
    control: float = -0.03,
    identical: bool = True,
):
    return evaluate_state_linearly_readable(
        held_out_r2={0: held_out - 0.5, 1: held_out},
        shuffled_r2={0: shuffled, 1: shuffled},
        control_r2={0: control, 1: control},
        control_hidden_identical=identical,
    )


def test_gate_b_passes_above_the_floor_with_both_controls_dead() -> None:
    result = _readable(held_out=PROBE_R2_FLOOR + 0.5)

    assert result.state is ClaimState.PASS
    assert "layer 1" in result.detail


def test_gate_b_fails_at_or_below_the_floor() -> None:
    assert _readable(held_out=PROBE_R2_FLOOR).state is ClaimState.FAIL


def test_gate_b_fails_when_the_shuffled_null_is_not_cleared() -> None:
    # The null is compared by margin, not against zero: the reported statistic
    # is a maximum over layers and draws, whose null is positively biased.
    result = _readable(held_out=0.25, shuffled=0.2)

    assert result.state is ClaimState.FAIL
    assert "null ceiling" in result.detail


def test_gate_b_tolerates_a_positively_biased_but_dominated_null() -> None:
    result = _readable(held_out=0.86, shuffled=0.11)

    assert result.state is ClaimState.PASS


def test_gate_b_fails_when_the_no_prefix_control_recovers_signal() -> None:
    # If the state is readable without a prefix, whatever the probe found did
    # not come from the carrier under test.
    result = _readable(held_out=0.25, control=0.2)

    assert result.state is ClaimState.FAIL
    assert "not the prefix" in result.detail


def test_gate_b_is_insufficient_when_the_pure_arm_is_not_isolated() -> None:
    result = _readable(held_out=0.8, identical=False)

    assert result.state is ClaimState.INSUFFICIENT_DATA
    assert "carrier-isolated" in result.detail


def _verdict(*, attention_pass: bool, readable_pass: bool):
    return build_carrier_diagnostics_verdict(
        substrate_fingerprint="Qwen/test@deadbeef",
        prefix_artifact_id="a" * 64,
        attention_claim=_attention(learned=2.0 if attention_pass else 1.0),
        readable_claim=_readable(
            held_out=0.8 if readable_pass else 0.0
        ),
        slot_mass_report={"learned": [0.3], "zero": [0.34], "random": [0.33]},
        nonuniformity_report={"learned": [1.2], "zero": [1.0], "random": [1.15]},
        probe_report={"held_out": {1: 0.8}},
    )


def test_carrier_is_live_requires_both_gates() -> None:
    assert _verdict(attention_pass=True, readable_pass=True).carrier_is_live
    assert not _verdict(attention_pass=False, readable_pass=True).carrier_is_live
    assert not _verdict(attention_pass=True, readable_pass=False).carrier_is_live


def test_verdict_carries_the_anti_overclaim_notes() -> None:
    payload = json.loads(_verdict(attention_pass=True, readable_pass=True).to_json())

    joined = " ".join(payload["notes"])
    # The two things a reader must not take away from a passing verdict.
    assert "never asserted on" in joined
    assert "context" in joined
    assert payload["prefix_artifact_id"] == "a" * 64


def test_verdict_requires_attribution() -> None:
    with pytest.raises(ValueError, match="substrate fingerprint"):
        build_carrier_diagnostics_verdict(
            substrate_fingerprint="",
            prefix_artifact_id="a" * 64,
            attention_claim=_attention(learned=2.0),
            readable_claim=_readable(held_out=0.8),
            slot_mass_report={},
            nonuniformity_report={},
            probe_report={},
        )
    with pytest.raises(ValueError, match="prefix artifact id"):
        build_carrier_diagnostics_verdict(
            substrate_fingerprint="Qwen/test@deadbeef",
            prefix_artifact_id="",
            attention_claim=_attention(learned=2.0),
            readable_claim=_readable(held_out=0.8),
            slot_mass_report={},
            nonuniformity_report={},
            probe_report={},
        )
