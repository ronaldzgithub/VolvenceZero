"""D1 contract tests: DLaaS real LLM eval — rubric judge + audience analysis.

Closes the code side of debts #13 / #14:

* #13 ``LLMRubricGrader`` — real per-criterion LLM judging with a fake
  transport (no network): strict JSON parsing, score clamping,
  criterion-coverage enforcement, fail-loud on malformed output
  (never a silent 0.5 fallback), and the env-driven
  ``build_grader_from_env`` seam.
* #14 ``LLMAudienceAnalyzer`` — real corpus-grounded cohort analysis:
  asset content resolution (inline / local file, typed errors for
  everything else), strict JSON profile parsing, env seam, and the
  ``audience/analyze`` route wiring (analysis enriches empty slots,
  caller fields stay authoritative, ``evidence_stats.analyzer`` is
  honest in both modes).

R12 / OA-1: everything here is a readout — no kernel owner is read or
written by grader or analyzer.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from dlaas_platform_contracts import AssetSpec, RubricEntry
from dlaas_platform_eval.audience import (
    AssetCorpusError,
    AudienceAnalysisError,
    LLMAudienceAnalyzer,
    build_audience_analyzer_from_env,
    load_asset_corpus,
)
from dlaas_platform_eval.grader import DefaultRubricGrader
from dlaas_platform_eval.llm_grader import (
    EvalLLMConfig,
    EvalLLMError,
    GraderResponseError,
    LLMRubricGrader,
    build_grader_from_env,
)


_CONFIG = EvalLLMConfig(
    base_url="https://llm.example/v1",
    api_key="test-key",
    model="judge-model",
)

_ENV_VARS = (
    "EVAL_LLM_BASE_URL",
    "EVAL_LLM_API_KEY",
    "EVAL_LLM_MODEL",
    "PROTOCOL_LLM_BASE_URL",
    "PROTOCOL_LLM_API_KEY",
    "PROTOCOL_LLM_MODEL",
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _rubric() -> tuple[RubricEntry, ...]:
    return (
        RubricEntry(
            criterion="warmth",
            max_score=5.0,
            weight=1.0,
            description="warm response",
        ),
        RubricEntry(
            criterion="accuracy",
            max_score=10.0,
            weight=3.0,
            description="accurate response",
        ),
    )


def _transport_returning(payload: object):
    def transport(config, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt.strip()
        assert user_prompt.strip()
        return payload if isinstance(payload, str) else json.dumps(payload)

    return transport


# ---------------------------------------------------------------------------
# #13 LLMRubricGrader
# ---------------------------------------------------------------------------


def test_llm_grader_scores_and_weights_criteria() -> None:
    grader = LLMRubricGrader(
        _CONFIG,
        transport=_transport_returning(
            {
                "scores": [
                    {"criterion": "warmth", "score": 5.0, "justification": "warm"},
                    {"criterion": "accuracy", "score": 5.0, "justification": "half"},
                ]
            }
        ),
    )
    out = grader.grade(
        rubric=_rubric(), ai_response="answer", reference_answer="ref"
    )
    # warmth 5/5 weight 1 + accuracy 5/10 weight 3 → (1*1 + 3*0.5)/4.
    assert abs(out.weighted_score - 0.625) < 1e-9
    assert [row["criterion"] for row in out.rubric_breakdown] == [
        "warmth",
        "accuracy",
    ]
    assert all(
        row["grader_label"] == "llm:judge-model" for row in out.rubric_breakdown
    )


def test_llm_grader_clamps_out_of_range_scores() -> None:
    grader = LLMRubricGrader(
        _CONFIG,
        transport=_transport_returning(
            {
                "scores": [
                    {"criterion": "warmth", "score": 99.0, "justification": ""},
                    {"criterion": "accuracy", "score": -3.0, "justification": ""},
                ]
            }
        ),
    )
    out = grader.grade(
        rubric=_rubric(), ai_response="answer", reference_answer=""
    )
    assert out.rubric_breakdown[0]["score"] == 5.0
    assert out.rubric_breakdown[1]["score"] == 0.0


def test_llm_grader_missing_criterion_fails_loudly() -> None:
    grader = LLMRubricGrader(
        _CONFIG,
        transport=_transport_returning(
            {"scores": [{"criterion": "warmth", "score": 4.0}]}
        ),
    )
    with pytest.raises(GraderResponseError, match="omitted rubric criterion"):
        grader.grade(rubric=_rubric(), ai_response="a", reference_answer="")


def test_llm_grader_non_json_output_fails_loudly() -> None:
    grader = LLMRubricGrader(
        _CONFIG, transport=_transport_returning("I would give this a 7/10.")
    )
    with pytest.raises(GraderResponseError, match="not valid JSON"):
        grader.grade(rubric=_rubric(), ai_response="a", reference_answer="")


def test_llm_grader_transport_failure_is_typed_not_silent() -> None:
    def broken_transport(config, *, system_prompt, user_prompt):
        raise EvalLLMError("eval LLM unreachable")

    grader = LLMRubricGrader(_CONFIG, transport=broken_transport)
    # No silent 0.5 fallback: the typed error propagates.
    with pytest.raises(GraderResponseError, match="unreachable"):
        grader.grade(rubric=_rubric(), ai_response="a", reference_answer="")


def test_build_grader_from_env_selects_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    assert isinstance(build_grader_from_env(), DefaultRubricGrader)

    monkeypatch.setenv("EVAL_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("EVAL_LLM_API_KEY", "k")
    monkeypatch.setenv("EVAL_LLM_MODEL", "judge-model")
    grader = build_grader_from_env()
    assert isinstance(grader, LLMRubricGrader)
    assert grader.grader_label == "llm:judge-model"


# ---------------------------------------------------------------------------
# #14 asset corpus resolution
# ---------------------------------------------------------------------------


def _asset(
    *,
    asset_id: str = "ast-1",
    uri: str = "",
    inline_text: str = "",
) -> AssetSpec:
    return AssetSpec(
        asset_id=asset_id,
        tenant_id="tnt-1",
        asset_type="chat_log",
        title="corpus",
        uri=uri,
        mime_type="text/plain",
        language="en",
        source_meta={"inline_text": inline_text} if inline_text else {},
        created_at_ms=0,
    )


def test_load_asset_corpus_inline_and_local_file(
    tmp_path: pathlib.Path,
) -> None:
    corpus_file = tmp_path / "chatlog.txt"
    corpus_file.write_text("Parents ask about bedtime routines.", encoding="utf-8")
    chunks = load_asset_corpus(
        (
            _asset(asset_id="a-inline", inline_text="Picky eating questions."),
            _asset(asset_id="a-file", uri=str(corpus_file)),
        )
    )
    assert chunks == (
        "Picky eating questions.",
        "Parents ask about bedtime routines.",
    )


def test_load_asset_corpus_caps_per_asset_chars(tmp_path: pathlib.Path) -> None:
    chunks = load_asset_corpus(
        (_asset(inline_text="x" * 5000),), max_chars_per_asset=100
    )
    assert len(chunks[0]) == 100


@pytest.mark.parametrize(
    "asset, match",
    [
        (_asset(uri="https://cdn.example/log.txt"), "scheme 'https'"),
        (_asset(uri="test:persona.md"), "scheme 'test'"),
        (_asset(), "neither"),
        (_asset(uri="/nonexistent/definitely/missing.txt"), "unreadable"),
    ],
)
def test_load_asset_corpus_typed_errors(asset: AssetSpec, match: str) -> None:
    with pytest.raises(AssetCorpusError, match=match):
        load_asset_corpus((asset,))


# ---------------------------------------------------------------------------
# #14 LLMAudienceAnalyzer
# ---------------------------------------------------------------------------


_PROFILE_JSON = {
    "common_questions": ["What's a healthy bedtime?", "Is picky eating normal?"],
    "communication_style": "warm-pragmatic",
    "emotion_triggers": ["comparison with other kids", "guilt"],
    "decision_patterns": ["seek expert confirmation before changing routine"],
    "evidence_notes": "Derived from 2 chat-log chunks.",
}


def test_audience_analyzer_parses_profile() -> None:
    analyzer = LLMAudienceAnalyzer(
        _CONFIG, transport=_transport_returning(_PROFILE_JSON)
    )
    enriched = analyzer.analyze(
        cohort_name="anxious-parents",
        corpus_chunks=("chunk one", "chunk two"),
    )
    assert enriched["common_questions"] == (
        "What's a healthy bedtime?",
        "Is picky eating normal?",
    )
    assert enriched["communication_style"] == "warm-pragmatic"
    assert enriched["emotion_triggers"] == (
        "comparison with other kids",
        "guilt",
    )
    assert enriched["decision_patterns"] == (
        "seek expert confirmation before changing routine",
    )
    stats = enriched["evidence_stats"]
    assert stats["analyzer"] == "llm:judge-model"
    assert stats["chunk_count"] == 2
    assert stats["corpus_truncated"] is False
    assert "chat-log" in stats["evidence_notes"]


def test_audience_analyzer_missing_field_fails_loudly() -> None:
    bad = dict(_PROFILE_JSON)
    del bad["emotion_triggers"]
    analyzer = LLMAudienceAnalyzer(
        _CONFIG, transport=_transport_returning(bad)
    )
    with pytest.raises(AudienceAnalysisError, match="emotion_triggers"):
        analyzer.analyze(cohort_name="c", corpus_chunks=("text",))


def test_audience_analyzer_non_json_fails_loudly() -> None:
    analyzer = LLMAudienceAnalyzer(
        _CONFIG, transport=_transport_returning("The cohort seems anxious.")
    )
    with pytest.raises(AudienceAnalysisError, match="not valid JSON"):
        analyzer.analyze(cohort_name="c", corpus_chunks=("text",))


def test_audience_analyzer_requires_corpus() -> None:
    analyzer = LLMAudienceAnalyzer(
        _CONFIG, transport=_transport_returning(_PROFILE_JSON)
    )
    with pytest.raises(AudienceAnalysisError, match="non-empty corpus"):
        analyzer.analyze(cohort_name="c", corpus_chunks=())


def test_build_audience_analyzer_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    assert build_audience_analyzer_from_env() is None
    monkeypatch.setenv("PROTOCOL_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "k")
    monkeypatch.setenv("PROTOCOL_LLM_MODEL", "shared-model")
    analyzer = build_audience_analyzer_from_env()
    assert isinstance(analyzer, LLMAudienceAnalyzer)
    assert analyzer.analyzer_label == "llm:shared-model"
