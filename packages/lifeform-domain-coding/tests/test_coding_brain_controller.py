from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_core import LifeformConfig
from lifeform_domain_coding import (
    CodingBrainConflictError,
    CodingBrainController,
    CodingBrainLineageError,
    CodingContextRequest,
    CodingOutcomeKind,
    CodingOutcomeReport,
    CodingOutcomeRoute,
    CodingOutcomeSource,
    CodingTaskKind,
    build_coding_lifeform,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.memory import StaticIdentityProvider, UserIdentity
from volvence_zero.runtime import WiringLevel


def _request(
    *,
    request_id: str = "request-1",
    task_id: str = "task-1",
    max_context_chars: int = 4_000,
) -> CodingContextRequest:
    return CodingContextRequest(
        request_id=request_id,
        project_id="project-1",
        repository_id="repo-1",
        task_id=task_id,
        task_kind=CodingTaskKind.BUGFIX,
        task_summary="Fix state restoration after a failed checkpoint",
        repository_revision="abc123",
        target_paths=("src/state.py", "tests/test_state.py"),
        max_context_chars=max_context_chars,
    )


def _session(*, session_id: str = "coding-session"):
    lifeform = build_coding_lifeform(
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
    )
    return lifeform.create_session(session_id=session_id)


async def test_context_pack_is_active_advice_is_shadow_and_request_is_idempotent() -> None:
    controller = CodingBrainController()
    session = _session()
    request = _request()

    first = await controller.build_context_pack(
        session=session,
        request=request,
        generated_at_ms=1_000,
    )
    repeated = await controller.build_context_pack(
        session=session,
        request=request,
        generated_at_ms=9_999,
    )

    assert repeated is first
    assert first.wiring_level is WiringLevel.ACTIVE
    assert first.advice.wiring_level is WiringLevel.SHADOW
    assert first.advice.applied is False
    assert first.advice.advice_id not in first.rendered_context
    assert first.source_entry_ids == ()

    with pytest.raises(CodingBrainConflictError, match="reused"):
        await controller.build_context_pack(
            session=session,
            request=replace(request, task_summary="Different immutable request"),
            generated_at_ms=1_001,
        )


async def test_deterministic_outcome_is_recalled_and_settled_by_next_turn_pe() -> None:
    controller = CodingBrainController()
    session = _session()
    context = await controller.build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=1_000,
    )
    report = CodingOutcomeReport(
        outcome_id="outcome-1",
        context_pack_id=context.context_pack_id,
        kind=CodingOutcomeKind.TASK_REGRESSED,
        source=CodingOutcomeSource.CI,
        summary="Checkpoint regression",
        detail="tests/test_state.py::test_restore expected committed state",
        observed_at_ms=1_100,
        evidence_ref="ci:run-17",
        changed_paths=("src/state.py",),
    )

    memory_count_before_outcome = session.memory_entry_count()
    receipt = await controller.record_outcome(session=session, report=report)
    repeated = await controller.record_outcome(session=session, report=report)
    assert repeated is receipt
    assert receipt.learning_route is CodingOutcomeRoute.DIALOGUE_EXTERNAL_OUTCOME
    assert receipt.external_outcome_evidence_id
    assert session.memory_entry_count() == memory_count_before_outcome + 1

    next_context = await controller.build_context_pack(
        session=session,
        request=_request(request_id="request-2", task_id="task-2"),
        generated_at_ms=1_200,
    )
    assert receipt.external_outcome_evidence_id in (
        next_context.settled_outcome_evidence_refs
    )
    assert receipt.memory_entry_id in next_context.source_entry_ids
    assert "expected committed state" in next_context.rendered_context

    conflicting = replace(report, detail="Different evidence under same id")
    with pytest.raises(CodingBrainConflictError, match="reused"):
        await controller.record_outcome(session=session, report=conflicting)


async def test_review_outcome_uses_execution_result_without_fabricating_task_oracle() -> None:
    controller = CodingBrainController()
    session = _session()
    context = await controller.build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=2_000,
    )
    receipt = await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="review-1",
            context_pack_id=context.context_pack_id,
            kind=CodingOutcomeKind.REVIEW_CHANGES_REQUESTED,
            source=CodingOutcomeSource.CODE_REVIEW,
            summary="Review requested changes",
            detail="Restore must preserve the previous transaction boundary",
            observed_at_ms=2_100,
            evidence_ref="review:comment-8",
        ),
    )
    assert receipt.learning_route is CodingOutcomeRoute.EXECUTION_RESULT
    assert receipt.external_outcome_evidence_id == ""

    next_context = await controller.build_context_pack(
        session=session,
        request=_request(request_id="request-2", task_id="task-2"),
        generated_at_ms=2_200,
    )
    assert next_context.settled_outcome_evidence_refs == ()
    assert "previous transaction boundary" in next_context.rendered_context


async def test_outcome_requires_same_live_session_context_lineage() -> None:
    controller = CodingBrainController()
    first_session = _session(session_id="first")
    second_session = _session(session_id="second")
    context = await controller.build_context_pack(
        session=first_session,
        request=_request(),
        generated_at_ms=3_000,
    )
    report = CodingOutcomeReport(
        outcome_id="outcome-cross-session",
        context_pack_id=context.context_pack_id,
        kind=CodingOutcomeKind.MERGED,
        source=CodingOutcomeSource.VCS,
        summary="Merged",
        detail="Commit abc123 reached the integration branch",
        observed_at_ms=3_100,
        evidence_ref="git:abc123",
    )
    with pytest.raises(CodingBrainLineageError, match="this live session"):
        await controller.record_outcome(session=second_session, report=report)


async def test_context_rendering_is_bounded_and_marks_truncation() -> None:
    controller = CodingBrainController()
    session = _session()
    context = await controller.build_context_pack(
        session=session,
        request=_request(),
        generated_at_ms=4_000,
    )
    await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="large-outcome",
            context_pack_id=context.context_pack_id,
            kind=CodingOutcomeKind.TASK_REGRESSED,
            source=CodingOutcomeSource.TEST_SUITE,
            summary="Large failure",
            detail="state mismatch " * 200,
            observed_at_ms=4_100,
            evidence_ref="pytest:run-1",
        ),
    )
    bounded = await controller.build_context_pack(
        session=session,
        request=_request(
            request_id="bounded-request",
            task_id="bounded-task",
            max_context_chars=256,
        ),
        generated_at_ms=4_200,
    )
    assert len(bounded.rendered_context) == 256
    assert bounded.truncated is True
    assert bounded.source_entry_ids


async def test_identity_scoped_memory_is_recalled_by_a_new_session(tmp_path) -> None:
    identity = StaticIdentityProvider(
        identity=UserIdentity(user_id="coder-1", scope_key="coder-1")
    )
    config = LifeformConfig(
        brain_config=BrainConfig(memory_scope_root_dir=str(tmp_path))
    )
    first_lifeform = build_coding_lifeform(
        config=config,
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
        identity_provider=identity,
    )
    first_session = first_lifeform.create_session(session_id="session-a")
    first_controller = CodingBrainController()
    context = await first_controller.build_context_pack(
        session=first_session,
        request=_request(),
        generated_at_ms=5_000,
    )
    receipt = await first_controller.record_outcome(
        session=first_session,
        report=CodingOutcomeReport(
            outcome_id="persistent-outcome",
            context_pack_id=context.context_pack_id,
            kind=CodingOutcomeKind.TASK_VERIFIED,
            source=CodingOutcomeSource.BUILD_GATE,
            summary="State restoration verified",
            detail="Build gate confirmed restart from the committed checkpoint",
            observed_at_ms=5_100,
            evidence_ref="build:run-1",
        ),
    )
    assert receipt.memory_persisted is True

    second_lifeform = build_coding_lifeform(
        config=config,
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
        identity_provider=identity,
    )
    second_session = second_lifeform.create_session(session_id="session-b")
    second_context = await CodingBrainController().build_context_pack(
        session=second_session,
        request=_request(request_id="request-new-session", task_id="task-new"),
        generated_at_ms=5_200,
    )
    assert receipt.memory_entry_id in second_context.source_entry_ids
    assert "restart from the committed checkpoint" in second_context.rendered_context


async def test_content_position_policy_updates_only_after_exact_next_turn_pe() -> None:
    controller = CodingBrainController()
    session = _session(session_id="coding-content-policy")

    first = await controller.build_context_pack(
        session=session,
        request=_request(request_id="policy-request-1", task_id="policy-task-1"),
        generated_at_ms=10_000,
    )
    await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="policy-outcome-1",
            context_pack_id=first.context_pack_id,
            kind=CodingOutcomeKind.TASK_REGRESSED,
            source=CodingOutcomeSource.TEST_SUITE,
            summary="First policy memory",
            detail="The first deterministic failure establishes one recalled entry",
            observed_at_ms=10_100,
            evidence_ref="pytest:policy-1",
        ),
    )
    second = await controller.build_context_pack(
        session=session,
        request=_request(request_id="policy-request-2", task_id="policy-task-2"),
        generated_at_ms=10_200,
    )
    assert second.content_policy_decision is None
    await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="policy-outcome-2",
            context_pack_id=second.context_pack_id,
            kind=CodingOutcomeKind.TASK_VERIFIED,
            source=CodingOutcomeSource.BUILD_GATE,
            summary="Second policy memory",
            detail="The second deterministic result establishes a challenger entry",
            observed_at_ms=10_300,
            evidence_ref="build:policy-2",
        ),
    )

    policy_pack = await controller.build_context_pack(
        session=session,
        request=_request(request_id="policy-request-3", task_id="policy-task-3"),
        generated_at_ms=10_400,
    )
    decision = policy_pack.content_policy_decision
    assert decision is not None
    assert decision.source_prediction_id
    assert set(decision.output_entry_ids) == set(decision.input_entry_ids)
    assert policy_pack.source_entry_ids == decision.output_entry_ids
    with pytest.raises(
        ValueError,
        match="source_entry_ids must match content policy output order",
    ):
        replace(
            policy_pack,
            source_entry_ids=tuple(reversed(policy_pack.source_entry_ids)),
        )
    receipt = await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="policy-outcome-3",
            context_pack_id=policy_pack.context_pack_id,
            kind=CodingOutcomeKind.TASK_REGRESSED,
            source=CodingOutcomeSource.CI,
            summary="Policy-bearing result",
            detail="The applied Context Pack ordering now receives exact PE credit",
            observed_at_ms=10_500,
            evidence_ref="ci:policy-3",
        ),
    )
    assert receipt.source_content_policy_decision_id == decision.policy_decision_id
    assert receipt.content_policy_action_applied is True

    settled = await controller.build_context_pack(
        session=session,
        request=_request(request_id="policy-request-4", task_id="policy-task-4"),
        generated_at_ms=10_600,
    )
    assert len(settled.settled_policy_credits) == 1
    assert len(settled.policy_updates) == 1
    assert (
        settled.settled_policy_credits[0].policy_decision_id
        == decision.policy_decision_id
    )
    assert settled.policy_updates[0].update_count == 1


async def test_content_position_policy_disabled_restores_owner_order() -> None:
    controller = CodingBrainController(
        content_policy_wiring_level=WiringLevel.DISABLED
    )
    session = _session(session_id="coding-content-policy-disabled")
    first = await controller.build_context_pack(
        session=session,
        request=_request(request_id="disabled-request-1", task_id="disabled-task-1"),
        generated_at_ms=20_000,
    )
    await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="disabled-outcome-1",
            context_pack_id=first.context_pack_id,
            kind=CodingOutcomeKind.REVIEW_CHANGES_REQUESTED,
            source=CodingOutcomeSource.CODE_REVIEW,
            summary="First disabled-policy memory",
            detail="Review evidence remains memory-only",
            observed_at_ms=20_100,
            evidence_ref="review:disabled-1",
        ),
    )
    second = await controller.build_context_pack(
        session=session,
        request=_request(request_id="disabled-request-2", task_id="disabled-task-2"),
        generated_at_ms=20_200,
    )
    await controller.record_outcome(
        session=session,
        report=CodingOutcomeReport(
            outcome_id="disabled-outcome-2",
            context_pack_id=second.context_pack_id,
            kind=CodingOutcomeKind.MERGED,
            source=CodingOutcomeSource.VCS,
            summary="Second disabled-policy memory",
            detail="VCS evidence remains memory-only",
            observed_at_ms=20_300,
            evidence_ref="git:disabled-2",
        ),
    )

    third = await controller.build_context_pack(
        session=session,
        request=_request(request_id="disabled-request-3", task_id="disabled-task-3"),
        generated_at_ms=20_400,
    )

    assert third.content_policy_wiring_level is WiringLevel.DISABLED
    assert third.content_policy_decision is None
    assert third.settled_policy_credits == ()
    assert third.policy_updates == ()
    assert len(third.source_entry_ids) == 2
