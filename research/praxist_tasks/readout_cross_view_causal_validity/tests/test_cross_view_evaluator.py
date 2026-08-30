from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from evaluations.cross_view import run  # noqa: E402


def test_frozen_corpus_is_balanced_and_cross_view_complete() -> None:
    corpus, samples = run._load_corpus()
    views = tuple(corpus["views"])
    groups = {sample.group_id for sample in samples}

    assert len(groups) == 16
    assert len(samples) == 16 * len(views)
    assert views == (
        "original_en",
        "paraphrase_en",
        "question_en",
        "zh_open",
        "role_order_en",
        "counterfactual_en",
    )
    for split in ("train", "evaluation"):
        counts = {
            label: len(
                {
                    sample.group_id
                    for sample in samples
                    if sample.split == split and sample.label == label
                }
            )
            for label in run.LABEL_TO_INT
        }
        assert counts == {"agency_displacement": 4, "belonging_erasure": 4}


def test_reader_surface_is_exact_and_rejects_extra_files(tmp_path: Path) -> None:
    reader = json.loads(
        (TASK_ROOT / "assets" / "baseline" / "reader.json").read_text(
            encoding="utf-8"
        )
    )
    variant = tmp_path / "valid"
    variant.mkdir()
    (variant / "reader.json").write_text(
        json.dumps(reader, sort_keys=True),
        encoding="utf-8",
    )
    assert run._load_reader(variant) == reader

    (variant / "notes.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(run.EvaluationError, match="only reader.json"):
        run._load_reader(variant)


def test_gate_margin_uses_weakest_preregistered_requirement() -> None:
    thresholds = {
        "same_view_balanced_accuracy": 0.75,
        "cross_view_balanced_accuracy": 0.65,
        "worst_view_balanced_accuracy": 0.5,
        "brier_score_max": 0.22,
        "heldout_cohen_d": 0.8,
        "cross_view_identity_retrieval": 0.15,
        "mean_direction_coherence": 0.0,
        "causal_target_margin_effect": 0.02,
        "random_control_separation": 0.01,
    }
    metrics = {
        "same_view_balanced_accuracy": 0.9,
        "cross_view_balanced_accuracy": 0.8,
        "worst_view_balanced_accuracy": 0.49,
        "brier_score": 0.15,
        "heldout_cohen_d": 1.2,
        "cross_view_identity_retrieval": 0.3,
        "mean_direction_coherence": 0.2,
        "causal_target_margin_effect": 0.08,
        "random_control_separation": 0.04,
    }

    margins = run._metric_margins(metrics, thresholds)

    assert min(margins, key=margins.get) == "worst_view_balanced_accuracy"
    assert margins["worst_view_balanced_accuracy"] == pytest.approx(-0.02)


def test_preliminary_and_complete_summary_authority_remain_distinct() -> None:
    corpus, _ = run._load_corpus()
    reader = json.loads(
        (TASK_ROOT / "assets" / "baseline" / "reader.json").read_text(
            encoding="utf-8"
        )
    )
    measured = {
        "qualification_margin": 0.1,
        "qualification_passed": True,
        "qualification_gate_margins": {"all": 0.1},
        "exit_classification": "PASS",
        "protocol_integrity_passed": True,
        "protocol_integrity_failed": False,
        "random_control": False,
        "domain_local": False,
        "instrument_invalid": False,
        "suspect_protocol": False,
        "suspect_leakage": False,
        "late_after_generation_boundary": False,
    }
    corpus_sha = run._sha256_file(run.CORPUS_PATH)

    preliminary = run._build_summary(
        mode="preliminary",
        reader=reader,
        corpus=corpus,
        metrics=measured,
        corpus_sha256=corpus_sha,
        elapsed=1.0,
    )
    complete = run._build_summary(
        mode="complete",
        reader=reader,
        corpus=corpus,
        metrics=measured,
        corpus_sha256=corpus_sha,
        elapsed=1.0,
    )

    assert preliminary["effort_ratio"] == 0.25
    assert preliminary["promotion_eligible"] is False
    assert preliminary["parent_authorized"] is False
    assert complete["effort_ratio"] == 1.0
    assert complete["promotion_eligible"] is True
    assert complete["protocol_integrity"]["formal_validation_performed"] is False
    assert complete["extra"]["production_promotion_authorized"] is False
