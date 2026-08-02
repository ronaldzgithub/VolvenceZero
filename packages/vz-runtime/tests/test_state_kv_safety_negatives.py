from volvence_zero.state_kv_safety_negatives import (
    ExtractionAttackNegative,
    StaleStateNegative,
    build_safety_negative_verdict,
)


def _extraction(**overrides):
    values = {
        "sample_count": 24,
        "train_count": 16,
        "test_count": 8,
        "exact_numeric_leak_count": 0,
        "baseline_mae": 0.25,
        "linear_probe_mae": 0.24,
        "extraction_advantage": 0.04,
        "conditioned_generation_applied_count": 24,
        "embedding_model_id": "BAAI/bge-m3",
    }
    values.update(overrides)
    return ExtractionAttackNegative(**values)


def test_safety_negatives_pass_only_with_stale_inert_and_weak_extraction() -> None:
    verdict = build_safety_negative_verdict(
        artifact_id="safety:test",
        stale=StaleStateNegative(True, True, True),
        extraction=_extraction(),
    )

    assert verdict.gate_state == "pass"


def test_numeric_leak_fails_extraction_claim() -> None:
    verdict = build_safety_negative_verdict(
        artifact_id="safety:test",
        stale=StaleStateNegative(True, True, True),
        extraction=_extraction(exact_numeric_leak_count=1),
    )

    assert verdict.extraction_state == "fail"
    assert verdict.gate_state == "fail"


def test_stale_output_change_fails_stale_claim() -> None:
    verdict = build_safety_negative_verdict(
        artifact_id="safety:test",
        stale=StaleStateNegative(False, True, True),
        extraction=_extraction(),
    )

    assert verdict.stale_state == "fail"
