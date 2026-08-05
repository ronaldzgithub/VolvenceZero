from __future__ import annotations

from dataclasses import replace
import hashlib
import struct

import pytest

from companion_bench.msc_corpus import MSCDyad, MSCSession, MSCUtterance
from companion_bench.prediction_research import (
    MSC_HELDOUT_ID_SHA256,
    PREDICTION_ARMS,
    CapacityObservation,
    PredictionObservation,
    PredictionThresholds,
    SameSubstrateContextAttestation,
    TemporalCapacityObservation,
    adjudicate_capacity_ladder,
    adjudicate_prediction_experiment,
    adjudicate_temporal_capacity_ladder,
    build_msc_next_turn_examples,
    parse_msc_full_runtime_context,
    render_long_context,
)


def _dyad(dyad_id: str = "d1") -> MSCDyad:
    return MSCDyad(
        dyad_id=dyad_id,
        split="heldout",
        sessions=(
            MSCSession(
                session_index=1,
                utterances=(
                    MSCUtterance("speaker_1", "I like tea.", 1),
                    MSCUtterance("speaker_2", "What kind?", 2),
                    MSCUtterance("speaker_1", "Oolong.", 3),
                ),
            ),
            MSCSession(
                session_index=2,
                utterances=(
                    MSCUtterance("speaker_2", "Still drinking it?", 1),
                    MSCUtterance("speaker_1", "Every morning.", 2),
                ),
            ),
        ),
        initial_personas=(("I like tea.",), ("I ask questions.",)),
    )


def test_msc_examples_keep_cross_session_history_and_real_target() -> None:
    examples = build_msc_next_turn_examples((_dyad(),))
    assert tuple(example.target_text for example in examples) == (
        "Oolong.",
        "Every morning.",
    )
    assert examples[-1].history_turns == 4
    assert {turn.session_index for turn in examples[-1].history} == {1, 2}
    context = render_long_context(examples[-1])
    assert "[session 1]" in context and "[session 2]" in context


def test_stateless_latest_text_selects_partner_across_same_speaker_boundary() -> None:
    dyad = MSCDyad(
        dyad_id="same-speaker-boundary",
        split="heldout",
        sessions=(
            MSCSession(
                session_index=1,
                utterances=(
                    MSCUtterance("speaker_2", "partner message", 1),
                    MSCUtterance("speaker_1", "previous target message", 2),
                ),
            ),
            MSCSession(
                session_index=2,
                utterances=(
                    MSCUtterance("speaker_1", "next target message", 1),
                ),
            ),
        ),
        initial_personas=(("persona",), ()),
    )
    examples = build_msc_next_turn_examples((dyad,))
    assert examples[-1].latest_text == "partner message"


def _observations() -> tuple[PredictionObservation, ...]:
    rows = []
    for arm in PREDICTION_ARMS:
        for seed in (0, 1, 2):
            for session in (1, 2, 3, 4, 5):
                for dyad in ("d1", "d2"):
                    control = 0.50 + 0.01 * session
                    cosine = (
                        control + 0.03 * session
                        if arm == "volvence"
                        else control
                    )
                    rows.append(
                        PredictionObservation(
                            arm=arm,
                            seed=seed,
                            sample_id=f"{dyad}:s{session}",
                            dyad_id=dyad,
                            session_index=session,
                            history_turns=session * 4,
                            cosine_similarity=cosine,
                            mean_squared_error=1.0 - cosine,
                            persistence_cosine_similarity=0.4,
                            persistence_mean_squared_error=0.6,
                            context_token_count=(
                                10 if arm == "volvence" else 200
                            ),
                            context_truncated_tokens=0,
                            latency_ms=(1.0 if arm == "volvence" else 4.0),
                        )
                    )
    return tuple(rows)


def test_partial_prediction_evidence_can_never_emit_thesis_win() -> None:
    verdict = adjudicate_prediction_experiment(
        _observations(),
        heldout_sorted_id_sha256=MSC_HELDOUT_ID_SHA256,
        encoder_fingerprint="encoder-sha",
        volvence_full_stack=False,
        thresholds=PredictionThresholds(
            formal_heldout_dyads=501,
            bootstrap_resamples=100,
        ),
    )
    assert verdict.evidence_level == "pilot"
    assert verdict.thesis_exit == "INELIGIBLE_PILOT"
    assert not verdict.quality_condition_met
    assert not verdict.scaling_condition_met
    assert verdict.longest_quality_advantage > 0.02


def test_complete_attested_evidence_can_select_quality_exit() -> None:
    observed_hash = hashlib.sha256(b"d1\nd2\n").hexdigest()
    verdict = adjudicate_prediction_experiment(
        _observations(),
        heldout_sorted_id_sha256=observed_hash,
        encoder_fingerprint="encoder-sha",
        volvence_full_stack=True,
        same_substrate_context=True,
        temporal_controller_capacity=True,
        formal_preregistered=True,
        thresholds=PredictionThresholds(
            formal_heldout_dyads=2,
            formal_heldout_id_sha256=observed_hash,
            bootstrap_resamples=100,
        ),
    )
    assert verdict.evidence_level == "formal"
    assert verdict.quality_condition_met
    assert verdict.thesis_exit == "QUALITY_ADVANTAGE"


def test_same_substrate_attestation_requires_matching_zero_truncation_surface() -> None:
    digest = "a" * 64
    attestation = SameSubstrateContextAttestation(
        context_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        target_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        context_weights_sha256=digest,
        target_weights_sha256=digest,
        context_readout_kind="latest-token-selected-layer-residual-l2.v1",
        target_readout_kind="latest-token-selected-layer-residual-l2.v1",
        context_layer_indices=(11, 12, 13),
        target_layer_indices=(11, 12, 13),
        context_activation_widths=(896, 896, 896),
        target_activation_widths=(896, 896, 896),
        context_limit=32768,
        maximum_observed_tokens=2048,
        truncated_token_count=0,
    )
    assert attestation.passed
    assert not replace(attestation, truncated_token_count=1).passed
    assert not replace(attestation, target_weights_sha256="b" * 64).passed


def test_full_stack_without_r3_attestation_remains_ineligible() -> None:
    observed_hash = hashlib.sha256(b"d1\nd2\n").hexdigest()
    verdict = adjudicate_prediction_experiment(
        _observations(),
        heldout_sorted_id_sha256=observed_hash,
        encoder_fingerprint="encoder-sha",
        volvence_full_stack=True,
        same_substrate_context=False,
        thresholds=PredictionThresholds(
            formal_heldout_dyads=2,
            formal_heldout_id_sha256=observed_hash,
            bootstrap_resamples=100,
        ),
    )
    assert verdict.thesis_exit == "INELIGIBLE_PILOT"
    assert ("same-substrate-zero-truncation-context", False) in (
        verdict.formal_requirements
    )


def _runtime_context_payload() -> dict[str, object]:
    values = (0.6, 0.8)
    values_sha256 = hashlib.sha256(struct.pack("!2d", *values)).hexdigest()
    return {
        "schema_version": "msc-full-runtime-context.v1",
        "volvence_full_stack": True,
        "acceptance_passed": True,
        "propagate_event_count": 12,
        "active_speaker_id": "speaker_2",
        "temporal_n_z": 3,
        "substrate_fallback_active": False,
        "runtime_slot_surface_sha256": "c" * 64,
        "context_lineage": {
            "model_fingerprint": {
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "version": "frozen-snapshot",
                "weights_sha256": "a" * 64,
            },
            "readout_kind": "latest-token-selected-layer-residual-l2.v1",
            "runtime_origin": "hf-local",
            "layer_indices": [11, 12],
            "activation_widths": [1, 1],
        },
        "context_representation": {
            "values": list(values),
            "values_sha256": values_sha256,
            "source_sha256": "b" * 64,
        },
        "input_token_count": 21,
        "output_token_count": 4,
        "total_token_count": 25,
        "generation_latency_ms": 7.0,
        "end_to_end_latency_ms": 8.5,
        "raw_text_retained": False,
        "evaluation_writeback_allowed": False,
    }


def test_full_runtime_context_dto_validates_lineage_cost_and_privacy() -> None:
    parsed = parse_msc_full_runtime_context(
        _runtime_context_payload(), sample_id="heldout:d1:s2:u4"
    )
    assert parsed.sample_id == "heldout:d1:s2:u4"
    assert parsed.values == (0.6, 0.8)
    assert parsed.total_token_count == 25
    assert parsed.temporal_n_z == 3


def test_full_runtime_context_dto_fails_closed_on_tampering() -> None:
    payload = _runtime_context_payload()
    payload["raw_text_retained"] = True
    with pytest.raises(ValueError, match="retained raw text"):
        parse_msc_full_runtime_context(payload, sample_id="sample")

    payload = _runtime_context_payload()
    context = payload["context_representation"]
    assert isinstance(context, dict)
    context["values_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fields are invalid"):
        parse_msc_full_runtime_context(payload, sample_id="sample")


def test_prediction_adjudicator_rejects_unmatched_arm_rows() -> None:
    rows = _observations()
    with pytest.raises(ValueError, match="sample-matched"):
        adjudicate_prediction_experiment(
            rows[:-1],
            heldout_sorted_id_sha256=MSC_HELDOUT_ID_SHA256,
            encoder_fingerprint="encoder-sha",
            volvence_full_stack=False,
        )


def test_capacity_ladder_fail_closes_pilot_and_can_formally_kill() -> None:
    rows = tuple(
        CapacityObservation(
            forward_head_n_z=n_z,
            seed=seed,
            split="validation",
            mean_cosine_similarity=0.5 + n_z / 100_000,
            mean_squared_error=0.4,
        )
        for n_z in (3, 16, 64, 256)
        for seed in (0, 1, 2)
    )
    pilot = adjudicate_capacity_ladder(
        rows, complete_train=False, complete_validation=False
    )
    assert pilot.forward_head_claim_exit == "INELIGIBLE_PILOT"
    formal = adjudicate_capacity_ladder(
        rows, complete_train=True, complete_validation=True
    )
    assert formal.forward_head_capacity_is_flat
    assert formal.forward_head_claim_exit == "KEEP_MINIMAL_FORWARD_HEAD"
    assert formal.chosen_forward_head_n_z == 3


def test_temporal_capacity_ladder_holds_pe_head_fixed_and_chooses_minimal_flat() -> None:
    rows = tuple(
        TemporalCapacityObservation(
            temporal_n_z=n_z,
            forward_head_n_z=3,
            seed=seed,
            split="validation",
            mean_cosine_similarity=0.5 + n_z / 100_000,
            mean_squared_error=0.4,
        )
        for n_z in (3, 16, 64, 256)
        for seed in (0, 1, 2)
    )
    verdict = adjudicate_temporal_capacity_ladder(
        rows,
        complete_train=True,
        complete_validation=True,
    )
    assert verdict.evidence_level == "formal"
    assert verdict.capacity_integrity_passed
    assert verdict.temporal_capacity_is_flat
    assert verdict.chosen_temporal_n_z == 3
    assert verdict.fixed_forward_head_n_z == 3


def test_temporal_capacity_ladder_exposes_zero_norm_collapse() -> None:
    rows = tuple(
        TemporalCapacityObservation(
            temporal_n_z=n_z,
            forward_head_n_z=3,
            seed=seed,
            split="validation",
            mean_cosine_similarity=0.5,
            mean_squared_error=0.4,
            zero_norm_prediction_count=(1 if n_z == 16 and seed == 1 else 0),
        )
        for n_z in (3, 16, 64, 256)
        for seed in (0, 1, 2)
    )
    verdict = adjudicate_temporal_capacity_ladder(
        rows,
        complete_train=True,
        complete_validation=True,
    )
    assert verdict.evidence_level == "formal"
    assert not verdict.capacity_integrity_passed
    assert verdict.zero_norm_prediction_count == 1
    assert verdict.temporal_capacity_claim_exit == "FAIL_ZERO_NORM_PREDICTIONS"


def test_prediction_observation_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown prediction arm"):
        replace(_observations()[0], arm="handwave")
