"""Contract test: E1 DLaaS placeholder injection (debt #13 / #14 / #15).

Validates:

1. ``DefaultRubricGrader(llm_grader_callable=fake)`` routes every
   ``grade(...)`` call to the injected callable; without it, the
   fallback path runs and emits a deprecation warning.
2. ``EvalStore.upsert_audience_profile(corpus_analyzer_callable=fake)``
   accepts the analyzer; the fake's return enriches empty-only
   slots, never overwrites caller-supplied fields.
3. ``_activation_text(asset_fetcher_callable=fake, template_assets=...)``
   appends the fetched corpus chunks to the persona block; without
   the fetcher, the legacy persona-only path is preserved.

Refs:

* docs/known-debts.md #13 / #14 / #15
"""

from __future__ import annotations

import asyncio
import logging
import pathlib

import pytest

from dlaas_platform_eval.grader import (
    DefaultRubricGrader,
    GradedSubmission,
)
from dlaas_platform_contracts import RubricEntry


# ---------------------------------------------------------------------------
# #13 DefaultRubricGrader llm_grader_callable
# ---------------------------------------------------------------------------


def _rubric() -> tuple[RubricEntry, ...]:
    return (
        RubricEntry(criterion="warmth", max_score=5.0, weight=1.0,
                    description="warm response"),
        RubricEntry(criterion="accuracy", max_score=5.0, weight=2.0,
                    description="accurate response"),
    )


def test_default_rubric_grader_routes_to_injected_callable() -> None:
    """When LLMGraderCallable injected, grade() returns its output verbatim."""
    received: list[tuple] = []

    def fake_judge(
        rubric: tuple[RubricEntry, ...],
        ai_response: str,
        reference_answer: str,
    ) -> GradedSubmission:
        received.append((tuple(r.criterion for r in rubric),
                          ai_response, reference_answer))
        return GradedSubmission(
            weighted_score=0.85,
            rubric_breakdown=tuple(
                {"criterion": r.criterion, "score": 4.5,
                 "max_score": r.max_score, "weight": r.weight}
                for r in rubric
            ),
        )

    grader = DefaultRubricGrader(llm_grader_callable=fake_judge)
    out = grader.grade(
        rubric=_rubric(),
        ai_response="Great answer",
        reference_answer="Reference",
    )
    assert out.weighted_score == 0.85
    assert len(out.rubric_breakdown) == 2
    assert received == [(("warmth", "accuracy"), "Great answer", "Reference")]


def test_default_rubric_grader_warns_when_no_llm_judge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Construction without llm_grader_callable emits a warning."""
    with caplog.at_level(logging.WARNING):
        DefaultRubricGrader()
    assert any("debt #13" in r.getMessage() for r in caplog.records), (
        "expected fallback warning citing debt #13"
    )


def test_default_rubric_grader_fallback_path_unchanged() -> None:
    """No-injection grade() preserves the deterministic 0.5 baseline."""
    grader = DefaultRubricGrader()
    out = grader.grade(
        rubric=_rubric(),
        ai_response="something",
        reference_answer="ref",
    )
    # 0.5 * max for every criterion, weighted average normalised:
    # both criteria score 2.5/5.0 = 0.5 → weighted_score = 0.5.
    assert abs(out.weighted_score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# #14 EvalStore.upsert_audience_profile corpus_analyzer_callable
# ---------------------------------------------------------------------------


def _registry(tmp_path: pathlib.Path):
    from dlaas_platform_registry.db import Registry
    reg = Registry(db_path=str(tmp_path / "registry.db"))
    reg.conn.execute(
        "INSERT INTO tenants (tenant_id, tenant_name, contact_email, "
        "api_key, api_secret_hash, created_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("tnt-001", "Brand A", "ops@brand-a", "k", "h", 0),
    )
    reg.conn.execute(
        "INSERT INTO templates (template_id, tenant_id, template_name, "
        "created_at_ms) VALUES (?, ?, ?, ?)",
        ("tmpl-001", "tnt-001", "tmpl", 0),
    )
    reg.conn.commit()
    return reg


def test_audience_profile_corpus_analyzer_enriches_empty_slots(
    tmp_path: pathlib.Path,
) -> None:
    from dlaas_platform_registry.eval_store import EvalStore
    registry = _registry(tmp_path)
    store = EvalStore(registry)

    def fake_analyzer(
        *, template_id, asset_ids, caller_supplied,
    ):
        return {
            "common_questions": ("What's bedtime?", "Picky eating?"),
            "communication_style": "warm-pragmatic",
            "emotion_triggers": ("comparison", "guilt"),
            "decision_patterns": ("seek-confirmation",),
            "evidence_stats": {"n_chunks": 12, "topics": 3},
        }

    profile = asyncio.run(
        store.upsert_audience_profile(
            template_id="tmpl-001",
            cohort_name="anxious-mom",
            asset_ids=("a-001", "a-002"),
            corpus_analyzer_callable=fake_analyzer,
        )
    )
    # Caller supplied no questions / style / emotions → analyzer fills.
    assert profile.common_questions == ("What's bedtime?", "Picky eating?")
    assert profile.communication_style == "warm-pragmatic"
    assert profile.emotion_triggers == ("comparison", "guilt")
    assert profile.evidence_stats == {"n_chunks": 12, "topics": 3}
    registry.close()


def test_audience_profile_caller_fields_take_precedence(
    tmp_path: pathlib.Path,
) -> None:
    """Caller's explicit fields are authoritative; analyzer only fills empties."""
    from dlaas_platform_registry.eval_store import EvalStore
    registry = _registry(tmp_path)
    store = EvalStore(registry)

    def overwriting_analyzer(
        *, template_id, asset_ids, caller_supplied,
    ):
        return {
            "communication_style": "analyzer-pushed",
            "common_questions": ("analyzer-q",),
        }

    profile = asyncio.run(
        store.upsert_audience_profile(
            template_id="tmpl-001",
            cohort_name="x",
            asset_ids=("a-001",),
            communication_style="caller-explicit",
            corpus_analyzer_callable=overwriting_analyzer,
        )
    )
    assert profile.communication_style == "caller-explicit"
    # common_questions was empty so analyzer filled.
    assert profile.common_questions == ("analyzer-q",)
    registry.close()


def test_audience_profile_warns_when_no_analyzer(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlaas_platform_registry.eval_store import EvalStore
    registry = _registry(tmp_path)
    store = EvalStore(registry)
    with caplog.at_level(logging.WARNING):
        asyncio.run(
            store.upsert_audience_profile(
                template_id="tmpl-001",
                cohort_name="x",
                asset_ids=("a-001",),
            )
        )
    registry.close()
    assert any("debt #14" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# #15 _activation_text asset_fetcher_callable
# ---------------------------------------------------------------------------


def test_activation_text_appends_fetched_corpus_chunks() -> None:
    from dlaas_platform_api.control_plane import _activation_text
    from dlaas_platform_contracts import TemplateSpec

    template = TemplateSpec(
        template_id="tmpl-001",
        tenant_id="tnt-001",
        template_name="cheng-laoshi",
        domain="growth_advisor",
        description="growth-advisor template",
        runtime_template_id="rt-001",
        status="published",
        current_version=1,
        activation_status="activated",
        base_persona={},
        persona_spec={
            "display_name": "Cheng Laoshi",
            "role_archetype": "growth-advisor",
            "speaking_style": "warm",
        },
        seed_config={},
        activation_stats={},
        figure_artifact_id="",
        created_at_ms=0,
    )

    captured: list[tuple] = []

    def fake_fetcher(*, template_id, template_assets):
        captured.append((template_id, template_assets))
        return [
            "Calcium absorption is best supported by vitamin D and weight-bearing exercise.",
            "Standard pediatric height curves are reviewed by WHO every 5 years.",
        ]

    text = _activation_text(
        template=template,
        seed_override={},
        asset_fetcher_callable=fake_fetcher,
        template_assets=("a-001", "a-002"),
    )
    assert "Asset corpus:" in text
    assert "Calcium absorption" in text
    assert captured == [("tmpl-001", ("a-001", "a-002"))]


def test_activation_text_legacy_path_unchanged_without_fetcher() -> None:
    from dlaas_platform_api.control_plane import _activation_text
    from dlaas_platform_contracts import TemplateSpec

    template = TemplateSpec(
        template_id="tmpl-001",
        tenant_id="tnt-001",
        template_name="cheng-laoshi",
        domain="growth_advisor",
        description="growth-advisor template",
        runtime_template_id="rt-001",
        status="published",
        current_version=1,
        activation_status="activated",
        base_persona={},
        persona_spec={"display_name": "Cheng Laoshi"},
        seed_config={},
        activation_stats={},
        figure_artifact_id="",
        created_at_ms=0,
    )
    text = _activation_text(template=template, seed_override={})
    # No "Asset corpus:" header when no fetcher.
    assert "Asset corpus:" not in text
    assert "Cheng Laoshi" in text
