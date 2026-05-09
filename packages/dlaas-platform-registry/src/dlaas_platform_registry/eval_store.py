"""Eval gate persistence (Slice 6).

Stores audience profiles, exam questions, exam runs, and launch
licenses. The :class:`EvalStore` is held by the eval wheel; this
module is here only because the SQL schema lives next to the rest
of the platform registry tables (one DB, one connection, one set
of write locks).
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    AudienceProfileSpec,
    ExamQuestionSpec,
    ExamRunSpec,
    ExamRunStatus,
    ExamSubmissionScore,
    LaunchLicenseSpec,
    RubricEntry,
)

from dlaas_platform_registry.db import Registry


class AudienceProfileNotFound(LookupError):
    pass


class ExamQuestionNotFound(LookupError):
    pass


class ExamRunNotFound(LookupError):
    pass


class LaunchLicenseNotFound(LookupError):
    pass


def _fresh_profile_id() -> str:
    return f"aud_{secrets.token_hex(4)}"


def _fresh_question_id() -> str:
    return f"q_{secrets.token_hex(4)}"


def _fresh_run_id() -> str:
    return f"run_{secrets.token_hex(4)}"


def _fresh_license_id() -> str:
    return f"lic_{secrets.token_hex(4)}"


class EvalStore:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Audience profiles
    # ------------------------------------------------------------------

    async def upsert_audience_profile(
        self,
        *,
        template_id: str,
        cohort_name: str,
        asset_ids: tuple[str, ...],
        common_questions: tuple[str, ...] = (),
        communication_style: str = "",
        emotion_triggers: tuple[str, ...] = (),
        decision_patterns: tuple[str, ...] = (),
        evidence_stats: Mapping[str, Any] | None = None,
        profile_id: str | None = None,
    ) -> AudienceProfileSpec:
        profile_id = profile_id or _fresh_profile_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO audience_profiles (
                    profile_id, template_id, cohort_name,
                    asset_ids_json, common_questions_json,
                    communication_style, emotion_triggers_json,
                    decision_patterns_json, evidence_stats_json,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    template_id,
                    cohort_name,
                    json.dumps(list(asset_ids)),
                    json.dumps(list(common_questions)),
                    communication_style,
                    json.dumps(list(emotion_triggers)),
                    json.dumps(list(decision_patterns)),
                    json.dumps(dict(evidence_stats or {})),
                    created_at_ms,
                ),
            )
        return AudienceProfileSpec(
            profile_id=profile_id,
            template_id=template_id,
            cohort_name=cohort_name,
            asset_ids=asset_ids,
            common_questions=common_questions,
            communication_style=communication_style,
            emotion_triggers=emotion_triggers,
            decision_patterns=decision_patterns,
            evidence_stats=dict(evidence_stats or {}),
            created_at_ms=created_at_ms,
        )

    async def list_audience_profiles_for_template(
        self, *, template_id: str
    ) -> tuple[AudienceProfileSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM audience_profiles WHERE template_id = ? "
            "ORDER BY created_at_ms ASC",
            (template_id,),
        ).fetchall()
        return tuple(_row_to_audience(row) for row in rows)

    async def get_audience_profile(self, profile_id: str) -> AudienceProfileSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM audience_profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise AudienceProfileNotFound(profile_id)
        return _row_to_audience(row)

    # ------------------------------------------------------------------
    # Exam questions
    # ------------------------------------------------------------------

    async def create_exam_question(
        self,
        *,
        template_id: str,
        scenario_tag: str,
        user_prompt: str,
        context: Mapping[str, Any] | None = None,
        rubric: tuple[RubricEntry, ...] = (),
        reference_answer: str = "",
        tags: tuple[str, ...] = (),
        difficulty: str = "medium",
        question_id: str | None = None,
    ) -> ExamQuestionSpec:
        question_id = question_id or _fresh_question_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO exam_questions (
                    question_id, template_id, scenario_tag, user_prompt,
                    context_json, rubric_json, reference_answer,
                    tags_json, difficulty, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    template_id,
                    scenario_tag,
                    user_prompt,
                    json.dumps(dict(context or {})),
                    json.dumps([r.to_json() for r in rubric]),
                    reference_answer,
                    json.dumps(list(tags)),
                    difficulty,
                    created_at_ms,
                ),
            )
        return ExamQuestionSpec(
            question_id=question_id,
            template_id=template_id,
            scenario_tag=scenario_tag,
            user_prompt=user_prompt,
            context=dict(context or {}),
            rubric=rubric,
            reference_answer=reference_answer,
            tags=tags,
            difficulty=difficulty,
            created_at_ms=created_at_ms,
        )

    async def list_exam_questions(
        self,
        *,
        template_id: str | None = None,
        scenario_tag: str | None = None,
    ) -> tuple[ExamQuestionSpec, ...]:
        clauses = []
        params: list[Any] = []
        if template_id:
            clauses.append("template_id = ?")
            params.append(template_id)
        if scenario_tag:
            clauses.append("scenario_tag = ?")
            params.append(scenario_tag)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._registry.conn.execute(
            f"SELECT * FROM exam_questions{where} ORDER BY created_at_ms ASC",
            params,
        ).fetchall()
        return tuple(_row_to_question(row) for row in rows)

    async def get_exam_question(self, question_id: str) -> ExamQuestionSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM exam_questions WHERE question_id = ?", (question_id,)
        ).fetchone()
        if row is None:
            raise ExamQuestionNotFound(question_id)
        return _row_to_question(row)

    # ------------------------------------------------------------------
    # Exam runs
    # ------------------------------------------------------------------

    async def create_exam_run(
        self,
        *,
        template_id: str,
        template_version: int,
        run_type: str,
        question_ids: tuple[str, ...],
        pass_threshold: float = 0.6,
        run_id: str | None = None,
    ) -> ExamRunSpec:
        run_id = run_id or _fresh_run_id()
        created_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO exam_runs (
                    run_id, template_id, template_version, run_type,
                    question_ids_json, status, pass_threshold, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    run_id,
                    template_id,
                    template_version,
                    run_type,
                    json.dumps(list(question_ids)),
                    pass_threshold,
                    created_at_ms,
                ),
            )
        return ExamRunSpec(
            run_id=run_id,
            template_id=template_id,
            template_version=template_version,
            run_type=run_type,
            question_ids=question_ids,
            status=ExamRunStatus.PENDING,
            pass_threshold=pass_threshold,
            created_at_ms=created_at_ms,
        )

    async def get_exam_run(self, run_id: str) -> ExamRunSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM exam_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ExamRunNotFound(run_id)
        return _row_to_run(row)

    async def list_runs_for_template(
        self, *, template_id: str
    ) -> tuple[ExamRunSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM exam_runs WHERE template_id = ? "
            "ORDER BY created_at_ms ASC",
            (template_id,),
        ).fetchall()
        return tuple(_row_to_run(row) for row in rows)

    async def update_exam_run(
        self,
        *,
        run_id: str,
        status: ExamRunStatus,
        operator_id: str = "",
        operator_name: str = "",
        comment: str = "",
        ai_id: str = "",
        contract_id: str = "",
        session_id: str = "",
        aggregate_score: float = 0.0,
        passed: bool = False,
        wrong_set: tuple[str, ...] = (),
        submissions: tuple[ExamSubmissionScore, ...] = (),
    ) -> ExamRunSpec:
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                UPDATE exam_runs SET
                    status = ?,
                    operator_id = ?,
                    operator_name = ?,
                    comment = ?,
                    ai_id = ?,
                    contract_id = ?,
                    session_id = ?,
                    aggregate_score = ?,
                    passed = ?,
                    wrong_set_json = ?,
                    submissions_json = ?
                WHERE run_id = ?
                """,
                (
                    status.value,
                    operator_id,
                    operator_name,
                    comment,
                    ai_id,
                    contract_id,
                    session_id,
                    float(aggregate_score),
                    1 if passed else 0,
                    json.dumps(list(wrong_set)),
                    json.dumps([s.to_json() for s in submissions]),
                    run_id,
                ),
            )
        return await self.get_exam_run(run_id)

    # ------------------------------------------------------------------
    # Launch license
    # ------------------------------------------------------------------

    async def upsert_launch_license(
        self,
        *,
        template_id: str,
        template_version: int,
        granted: bool,
        reason: str = "",
        granted_by_run_id: str = "",
    ) -> LaunchLicenseSpec:
        license_id = _fresh_license_id()
        issued_at_ms = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO launch_licenses (
                    license_id, template_id, template_version,
                    granted, reason, granted_by_run_id, issued_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(template_id, template_version) DO UPDATE SET
                    license_id = excluded.license_id,
                    granted = excluded.granted,
                    reason = excluded.reason,
                    granted_by_run_id = excluded.granted_by_run_id,
                    issued_at_ms = excluded.issued_at_ms
                """,
                (
                    license_id,
                    template_id,
                    template_version,
                    1 if granted else 0,
                    reason,
                    granted_by_run_id,
                    issued_at_ms,
                ),
            )
        return LaunchLicenseSpec(
            license_id=license_id,
            template_id=template_id,
            template_version=template_version,
            granted=granted,
            reason=reason,
            granted_by_run_id=granted_by_run_id,
            issued_at_ms=issued_at_ms,
        )

    async def get_launch_license(
        self, *, template_id: str, template_version: int
    ) -> LaunchLicenseSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM launch_licenses WHERE template_id = ? "
            "AND template_version = ?",
            (template_id, template_version),
        ).fetchone()
        if row is None:
            raise LaunchLicenseNotFound(f"{template_id}@{template_version}")
        return _row_to_license(row)


def _row_to_audience(row) -> AudienceProfileSpec:
    return AudienceProfileSpec(
        profile_id=row["profile_id"],
        template_id=row["template_id"],
        cohort_name=row["cohort_name"],
        asset_ids=tuple(json.loads(row["asset_ids_json"] or "[]")),
        common_questions=tuple(json.loads(row["common_questions_json"] or "[]")),
        communication_style=row["communication_style"],
        emotion_triggers=tuple(json.loads(row["emotion_triggers_json"] or "[]")),
        decision_patterns=tuple(json.loads(row["decision_patterns_json"] or "[]")),
        evidence_stats=json.loads(row["evidence_stats_json"] or "{}"),
        created_at_ms=int(row["created_at_ms"]),
    )


def _row_to_question(row) -> ExamQuestionSpec:
    rubric_data = json.loads(row["rubric_json"] or "[]")
    rubric: list[RubricEntry] = []
    for item in rubric_data:
        try:
            rubric.append(RubricEntry.from_json(item))
        except ValueError:
            continue
    return ExamQuestionSpec(
        question_id=row["question_id"],
        template_id=row["template_id"],
        scenario_tag=row["scenario_tag"],
        user_prompt=row["user_prompt"],
        context=json.loads(row["context_json"] or "{}"),
        rubric=tuple(rubric),
        reference_answer=row["reference_answer"],
        tags=tuple(json.loads(row["tags_json"] or "[]")),
        difficulty=row["difficulty"],
        created_at_ms=int(row["created_at_ms"]),
    )


def _row_to_run(row) -> ExamRunSpec:
    submissions_data = json.loads(row["submissions_json"] or "[]")
    submissions = tuple(
        ExamSubmissionScore(
            question_id=str(item.get("question_id", "")),
            ai_response=str(item.get("ai_response", "")),
            weighted_score=float(item.get("weighted_score", 0.0)),
            rubric_breakdown=tuple(item.get("rubric_breakdown") or ()),
        )
        for item in submissions_data
    )
    return ExamRunSpec(
        run_id=row["run_id"],
        template_id=row["template_id"],
        template_version=int(row["template_version"]),
        run_type=row["run_type"],
        question_ids=tuple(json.loads(row["question_ids_json"] or "[]")),
        status=ExamRunStatus(row["status"]),
        operator_id=row["operator_id"],
        operator_name=row["operator_name"],
        comment=row["comment"],
        ai_id=row["ai_id"],
        contract_id=row["contract_id"],
        session_id=row["session_id"],
        aggregate_score=float(row["aggregate_score"]),
        pass_threshold=float(row["pass_threshold"]),
        passed=bool(row["passed"]),
        wrong_set=tuple(json.loads(row["wrong_set_json"] or "[]")),
        submissions=submissions,
        created_at_ms=int(row["created_at_ms"]),
    )


def _row_to_license(row) -> LaunchLicenseSpec:
    return LaunchLicenseSpec(
        license_id=row["license_id"],
        template_id=row["template_id"],
        template_version=int(row["template_version"]),
        granted=bool(row["granted"]),
        reason=row["reason"],
        granted_by_run_id=row["granted_by_run_id"],
        issued_at_ms=int(row["issued_at_ms"]),
    )


__all__ = [
    "AudienceProfileNotFound",
    "EvalStore",
    "ExamQuestionNotFound",
    "ExamRunNotFound",
    "LaunchLicenseNotFound",
]
