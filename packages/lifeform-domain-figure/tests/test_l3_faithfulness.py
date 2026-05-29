"""Tests for the L3 citation-faithfulness scorer (#59, Full Version Track 3).

`score_l3_faithfulness` quantifies "are the response's verifiable claims
grounded in the cited evidence frames?" from the synthesizer's
``l3_grounded_verify=passed:K;unsupported:M;evidence:N`` tag(s).

A small reference fixture pins the expected faithfulness for canonical
tag shapes so the metric stays stable across refactors.
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure.verification.persona import (
    PersonaCondition,
    aggregate_scores,
    score_l3_faithfulness,
)
from lifeform_domain_figure.verification.persona.records import (
    CognitionScore,
    QuestionScore,
    RefusalScore,
    VoiceScore,
)


# Reference fixture: (rationale_tags, expected_faithfulness, expected_verified)
_FIXTURE = [
    # Fully grounded: every assertion supported.
    (("l3_grounded_verify=passed:3;unsupported:0;evidence:5",), 1.0, True),
    # Partially grounded: 2 of 3 supported.
    (("l3_grounded_verify=passed:2;unsupported:1;evidence:4",), 2 / 3, True),
    # Entirely unsupported claims.
    (("l3_grounded_verify=passed:0;unsupported:2;evidence:0",), 0.0, True),
    # Verifier ran but asserted nothing needing support -> vacuously faithful.
    (("l3_grounded_verify=passed:0;unsupported:0;evidence:0",), 1.0, True),
    # No L3 verify at all -> unverified, faithfulness 0.
    (("voice_ok", "l4_scope_refusal"), 0.0, False),
    # Multiple verifies in one response aggregate.
    (
        (
            "l3_grounded_verify=passed:1;unsupported:1;evidence:2",
            "l3_grounded_verify=passed:3;unsupported:1;evidence:3",
        ),
        4 / 6,
        True,
    ),
]


@pytest.mark.parametrize("tags,expected,verified", _FIXTURE)
def test_score_l3_faithfulness_reference_fixture(
    tags: tuple[str, ...], expected: float, verified: bool
) -> None:
    score = score_l3_faithfulness(tags)
    assert score.verified is verified
    assert score.faithfulness == pytest.approx(expected)
    assert 0.0 <= score.faithfulness <= 1.0


def test_aggregate_counts_evidence_and_totals() -> None:
    score = score_l3_faithfulness(
        ("l3_grounded_verify=passed:2;unsupported:1;evidence:4",)
    )
    assert score.passed_count == 2
    assert score.unsupported_count == 1
    assert score.evidence_count == 4


def _in_corpus_question_score(
    qid: str, tags: tuple[str, ...]
) -> QuestionScore:
    return QuestionScore(
        question_id=qid,
        condition=PersonaCondition.BUNDLE_LORA,
        voice=VoiceScore(0.5, 0.5, 0.5, 0.5),
        cognition=CognitionScore(0.8, True, 3),
        refusal=RefusalScore(False, False, True),
        l3_evidence_count=4,
        l3_faithfulness=score_l3_faithfulness(tags),
    )


def test_aggregate_scores_averages_verified_faithfulness() -> None:
    scores = (
        _in_corpus_question_score(
            "q1", ("l3_grounded_verify=passed:3;unsupported:0;evidence:3",)
        ),  # 1.0
        _in_corpus_question_score(
            "q2", ("l3_grounded_verify=passed:1;unsupported:1;evidence:2",)
        ),  # 0.5
        _in_corpus_question_score("q3", ("no_verify",)),  # unverified -> excluded
    )
    aggs = aggregate_scores(scores=scores)
    agg = next(a for a in aggs if a.condition is PersonaCondition.BUNDLE_LORA)
    # Mean over the two verified responses only.
    assert agg.l3_faithfulness == pytest.approx(0.75)
    assert agg.l3_verified_question_count == 2
    # JSON-serialisable.
    assert agg.to_json()["l3_faithfulness"] == pytest.approx(0.75)
