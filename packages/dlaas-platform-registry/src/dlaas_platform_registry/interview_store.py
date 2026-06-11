"""Interview-run persistence (schema v10).

One row per interview run; turns are stored as a JSON array on the row
(turn volume per run is small and turns are only ever appended). The
store mirrors the :class:`EvalStore` shape: writes take the registry
write-lock; reads run lock-free on WAL snapshots and return frozen
contract dataclasses.

R12: interview scores are readouts — nothing in this store flows back
into the kernel.
"""

from __future__ import annotations

import json
import secrets
import time

from dlaas_platform_contracts import (
    InterviewRunSpec,
    InterviewRunStatus,
    InterviewTurn,
    InterviewerKind,
)

from dlaas_platform_registry.db import Registry


class InterviewRunNotFound(LookupError):
    pass


class InterviewRunStateError(ValueError):
    """Raised when an operation violates the interview-run state machine."""


def _fresh_run_id() -> str:
    return f"ivrun_{secrets.token_hex(4)}"


class InterviewRunStore:
    """CRUD over the ``interview_runs`` table."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def create(
        self,
        *,
        template_id: str,
        template_version: int = 1,
        ai_id: str = "",
        session_id: str = "",
        interviewer_kind: InterviewerKind = InterviewerKind.OPERATOR,
        question_plan: tuple[str, ...] = (),
        pass_threshold: float = 0.6,
    ) -> InterviewRunSpec:
        if not 0.0 < pass_threshold <= 1.0:
            raise InterviewRunStateError(
                f"pass_threshold must be in (0, 1], got {pass_threshold!r}"
            )
        run_id = _fresh_run_id()
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                INSERT INTO interview_runs (
                    run_id, template_id, template_version, ai_id, session_id,
                    interviewer_kind, status, question_plan_json, turns_json,
                    aggregate_score, pass_threshold, passed, operator_id,
                    verdict_comment, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, '[]', 0.0, ?, 0,
                          '', '', ?, ?)
                """,
                (
                    run_id,
                    template_id,
                    int(template_version),
                    ai_id,
                    session_id,
                    interviewer_kind.value,
                    json.dumps(list(question_plan), ensure_ascii=False),
                    float(pass_threshold),
                    now,
                    now,
                ),
            )
        return await self.get(run_id)

    async def get(self, run_id: str) -> InterviewRunSpec:
        row = self._registry.conn.execute(
            "SELECT * FROM interview_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise InterviewRunNotFound(run_id)
        return _row_to_run(row)

    async def list_for_template(
        self, template_id: str
    ) -> tuple[InterviewRunSpec, ...]:
        rows = self._registry.conn.execute(
            "SELECT * FROM interview_runs WHERE template_id = ? "
            "ORDER BY created_at_ms DESC",
            (template_id,),
        ).fetchall()
        return tuple(_row_to_run(row) for row in rows)

    async def append_turn(
        self,
        *,
        run_id: str,
        question: str,
        ai_response: str = "",
        asked_by: str = "operator",
        score: float | None = None,
        notes: str = "",
    ) -> InterviewRunSpec:
        run = await self.get(run_id)
        if run.status in (
            InterviewRunStatus.COMPLETED,
            InterviewRunStatus.FAILED,
        ):
            raise InterviewRunStateError(
                f"interview run {run_id!r} is {run.status.value!r}; "
                "no further turns accepted"
            )
        now = int(time.time() * 1000.0)
        turn = InterviewTurn.from_json(
            {
                "turn_index": len(run.turns),
                "question": question,
                "ai_response": ai_response,
                "asked_by": asked_by,
                "score": score,
                "notes": notes,
                "recorded_at_ms": now,
            }
        )
        turns = [t.to_json() for t in run.turns] + [turn.to_json()]
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE interview_runs SET turns_json = ?, status = ?, "
                "updated_at_ms = ? WHERE run_id = ?",
                (
                    json.dumps(turns, ensure_ascii=False),
                    InterviewRunStatus.IN_PROGRESS.value,
                    now,
                    run_id,
                ),
            )
        return await self.get(run_id)

    async def score_turn(
        self,
        *,
        run_id: str,
        turn_index: int,
        score: float,
        notes: str = "",
    ) -> InterviewRunSpec:
        run = await self.get(run_id)
        if run.status in (
            InterviewRunStatus.COMPLETED,
            InterviewRunStatus.FAILED,
        ):
            raise InterviewRunStateError(
                f"interview run {run_id!r} is {run.status.value!r}; "
                "turn scores are frozen"
            )
        if not 0.0 <= score <= 1.0:
            raise InterviewRunStateError(
                f"turn score must be in [0, 1], got {score!r}"
            )
        if turn_index < 0 or turn_index >= len(run.turns):
            raise InterviewRunStateError(
                f"turn_index {turn_index} out of range "
                f"(run has {len(run.turns)} turns)"
            )
        turns = [t.to_json() for t in run.turns]
        turns[turn_index]["score"] = float(score)
        if notes:
            turns[turn_index]["notes"] = notes
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE interview_runs SET turns_json = ?, updated_at_ms = ? "
                "WHERE run_id = ?",
                (json.dumps(turns, ensure_ascii=False), now, run_id),
            )
        return await self.get(run_id)

    async def complete(
        self,
        *,
        run_id: str,
        operator_id: str = "",
        verdict_comment: str = "",
    ) -> InterviewRunSpec:
        """Finalize the run: every turn must carry a score (no silent pass)."""

        run = await self.get(run_id)
        if run.status in (
            InterviewRunStatus.COMPLETED,
            InterviewRunStatus.FAILED,
        ):
            raise InterviewRunStateError(
                f"interview run {run_id!r} is already {run.status.value!r}"
            )
        if not run.turns:
            raise InterviewRunStateError(
                "interview run has no turns; record at least one turn "
                "before completing"
            )
        unscored = [t.turn_index for t in run.turns if t.score is None]
        if unscored:
            raise InterviewRunStateError(
                f"turns {unscored!r} have no score; every turn must be "
                "scored before completion"
            )
        aggregate = sum(t.score for t in run.turns) / len(run.turns)
        passed = aggregate >= run.pass_threshold
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                """
                UPDATE interview_runs SET status = 'completed',
                    aggregate_score = ?, passed = ?, operator_id = ?,
                    verdict_comment = ?, updated_at_ms = ?
                WHERE run_id = ?
                """,
                (
                    float(aggregate),
                    1 if passed else 0,
                    operator_id,
                    verdict_comment,
                    now,
                    run_id,
                ),
            )
        return await self.get(run_id)

    async def mark_failed(
        self, *, run_id: str, verdict_comment: str
    ) -> InterviewRunSpec:
        now = int(time.time() * 1000.0)
        async with self._registry.write_lock:
            self._registry.conn.execute(
                "UPDATE interview_runs SET status = 'failed', "
                "verdict_comment = ?, updated_at_ms = ? WHERE run_id = ?",
                (verdict_comment, now, run_id),
            )
        return await self.get(run_id)


def _row_to_run(row) -> InterviewRunSpec:
    turns = tuple(
        InterviewTurn.from_json(t)
        for t in json.loads(row["turns_json"] or "[]")
    )
    return InterviewRunSpec(
        run_id=row["run_id"],
        template_id=row["template_id"],
        template_version=int(row["template_version"]),
        ai_id=row["ai_id"],
        session_id=row["session_id"],
        interviewer_kind=InterviewerKind(row["interviewer_kind"]),
        status=InterviewRunStatus(row["status"]),
        question_plan=tuple(json.loads(row["question_plan_json"] or "[]")),
        turns=turns,
        aggregate_score=float(row["aggregate_score"]),
        pass_threshold=float(row["pass_threshold"]),
        passed=bool(row["passed"]),
        operator_id=row["operator_id"],
        verdict_comment=row["verdict_comment"],
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


__all__ = [
    "InterviewRunNotFound",
    "InterviewRunStateError",
    "InterviewRunStore",
]
