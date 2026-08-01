from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from companion_bench.msc_corpus import MSCDyad, MSCSession, MSCUtterance
from companion_bench.prediction_research import (
    MSC_HELDOUT_ID_SHA256,
    PREDICTION_ARMS,
    CapacityObservation,
    PredictionObservation,
    PredictionThresholds,
    adjudicate_capacity_ladder,
    adjudicate_prediction_experiment,
    build_msc_next_turn_examples,
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


def _observations() -> tuple[PredictionObservation, ...]:
    rows = []
    for arm in PREDICTION_ARMS:
        for seed in (0, 1, 2):
            for session in (1, 2):
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
        thresholds=PredictionThresholds(
            formal_heldout_dyads=2,
            formal_heldout_id_sha256=observed_hash,
            bootstrap_resamples=100,
        ),
    )
    assert verdict.evidence_level == "formal"
    assert verdict.quality_condition_met
    assert verdict.thesis_exit == "QUALITY_ADVANTAGE"


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
            n_z=n_z,
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
    assert pilot.eta_claim_exit == "INELIGIBLE_PILOT"
    formal = adjudicate_capacity_ladder(
        rows, complete_train=True, complete_validation=True
    )
    assert formal.capacity_is_flat
    assert formal.eta_claim_exit == "KILL_ETA_CAPACITY_CLAIM"


def test_prediction_observation_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown prediction arm"):
        replace(_observations()[0], arm="handwave")
