"""Contract test: submission wiring fixes (debt #73 / #74 / #75).

Validates:

1. ``run_submission`` injects ``manifest.system_prompt`` as the first
   ``role: system`` message in every SUT call (debt #74).
2. ``run_submission`` translates ``manifest.generation_config``
   ``max_tokens`` / ``temperature`` keys into ``ArcRunConfig`` so the
   SUT receives the manifest-declared knobs (debt #74).
3. ``run_submission`` calls ``judge.drain_usage_log()`` on both
   per-turn and arc judges and feeds entries to
   ``CostTracker.record_perturn_judge`` / ``record_arc_judge`` so
   judge cost is attributed end-to-end (debt #75).
4. A single arc raising mid-pipeline does NOT terminate the
   submission; the failed arc is recorded in ``arc_failure.jsonl``
   under ``artifact_dir`` and the remaining arcs aggregate normally
   (debt #73 arc-level isolation).

Refs:

* docs/known-debts.md #73 / #74 / #75
* docs/specs/companion-bench.md
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import importlib.resources as res
from typing import Any, Mapping

from companion_bench.spec import AxisId, load_scenarios_dir
from companion_bench.submission import (
    SubmissionAttestation,
    SubmissionManifest,
    run_submission,
)
from companion_bench.sut_client import EchoFakeSUTClient, SUTResponse
from companion_bench.user_simulator import DeterministicFakeUtteranceClient
from companion_bench.judge_perturn import DeterministicFakePerTurnJudge, CRITERIA
from companion_bench.judge_arc import AXIS_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_specs():
    public_dir = pathlib.Path(
        str(res.files("companion_bench") / "scenarios" / "public")
    )
    return load_scenarios_dir(public_dir, include_held_out=False)


def _attestation() -> SubmissionAttestation:
    return SubmissionAttestation(True, True, True, True)


def _manifest(
    *,
    submission_id: str = "test-sub",
    system_prompt: str = "(none)",
    generation_config: dict | None = None,
) -> SubmissionManifest:
    return SubmissionManifest(
        submission_id=submission_id,
        system_name="Test System",
        model_identifier="fake/model-x",
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt=system_prompt,
        generation_config=generation_config or {},
        attestation=_attestation(),
        leaderboard_category="bespoke",
    )


class _CapturingSUT:
    """SUT client that records every chat() call's args."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> SUTResponse:
        # Capture a snapshot so later mutations to the original list
        # don't change what we recorded.
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "session_id": session_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return SUTResponse(
            text=f"echo:{messages[-1]['content']}",
            model_id="fake/model-x",
            response_headers={},
            usage_prompt_tokens=10,
            usage_completion_tokens=5,
            raw={},
        )


class _UsageLoggingPerTurnJudge:
    """Fake judge that records a fixed usage entry per call."""

    model = "fake/perturn-usage"

    def __init__(self) -> None:
        self._log: list[dict] = []

    def drain_usage_log(self) -> list[dict]:
        out = list(self._log)
        self._log.clear()
        return out

    def score(
        self,
        *,
        prior_context: list[dict[str, str]],
        assistant_text: str,
        session_index: int,
        turn_index: int,
    ) -> Mapping[str, int]:
        self._log.append({"prompt_tokens": 100, "completion_tokens": 20})
        return {c: 3 for c in CRITERIA}


class _UsageLoggingArcJudge:
    model = "fake/arc-usage"

    def __init__(self) -> None:
        self._log: list[dict] = []

    def drain_usage_log(self) -> list[dict]:
        out = list(self._log)
        self._log.clear()
        return out

    def score(self, *, arc, ledger, family) -> Mapping[AxisId, float]:
        self._log.append({"prompt_tokens": 500, "completion_tokens": 80})
        return {a: 50.0 for a in AXIS_ORDER}


class _RaisingArcJudge:
    """Arc judge that raises on the second arc to exercise isolation."""

    model = "fake/arc-raises"

    def __init__(self, raise_on_arc_index: int = 1) -> None:
        self._call_idx = 0
        self._raise_on = raise_on_arc_index
        self._log: list[dict] = []

    def drain_usage_log(self) -> list[dict]:
        out = list(self._log)
        self._log.clear()
        return out

    def score(self, *, arc, ledger, family) -> Mapping[AxisId, float]:
        idx = self._call_idx
        self._call_idx += 1
        if idx == self._raise_on:
            raise RuntimeError("synthetic arc judge failure")
        return {a: 50.0 for a in AXIS_ORDER}


def _two_specs():
    """Two distinct public specs to exercise multi-arc orchestration."""
    return tuple(
        s
        for s in _public_specs()
        if s.scenario_id in {"F1-continuity-001", "F5-boundary-001"}
    )


# ---------------------------------------------------------------------------
# #74 system_prompt + generation_config injection
# ---------------------------------------------------------------------------


def test_run_submission_injects_system_prompt(tmp_path: pathlib.Path) -> None:
    sut = _CapturingSUT()
    manifest = _manifest(system_prompt="You are a careful, kind companion.")
    run_submission(
        manifest=manifest,
        specs=_two_specs(),
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        perturn_judge=DeterministicFakePerTurnJudge(),
        arc_judge=_UsageLoggingArcJudge(),
        paraphrase_seeds=(0,),
        artifact_dir=tmp_path,
    )
    # Every SUT call must carry the manifest system prompt as the head
    # of the messages list.
    assert sut.calls, "SUT was never called"
    for call in sut.calls:
        first = call["messages"][0]
        assert first["role"] == "system"
        assert first["content"] == "You are a careful, kind companion."


def test_run_submission_injects_generation_config(tmp_path: pathlib.Path) -> None:
    sut = _CapturingSUT()
    manifest = _manifest(
        system_prompt="(none)",
        generation_config={"max_tokens": 256, "temperature": 0.7},
    )
    run_submission(
        manifest=manifest,
        specs=_two_specs(),
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        perturn_judge=DeterministicFakePerTurnJudge(),
        arc_judge=_UsageLoggingArcJudge(),
        paraphrase_seeds=(0,),
        artifact_dir=tmp_path,
    )
    for call in sut.calls:
        assert call["max_tokens"] == 256
        assert call["temperature"] == 0.7


def test_run_submission_empty_system_prompt_keeps_no_system_message(
    tmp_path: pathlib.Path,
) -> None:
    """Empty system_prompt must NOT inject a stray ``role:system`` message."""
    sut = _CapturingSUT()
    manifest = _manifest(system_prompt="")
    run_submission(
        manifest=manifest,
        specs=_two_specs(),
        sut_client=sut,
        user_backend=DeterministicFakeUtteranceClient(),
        perturn_judge=DeterministicFakePerTurnJudge(),
        arc_judge=_UsageLoggingArcJudge(),
        paraphrase_seeds=(0,),
        artifact_dir=tmp_path,
    )
    for call in sut.calls:
        assert call["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# #75 judge cost tracking
# ---------------------------------------------------------------------------


def test_run_submission_drains_judge_usage_into_cost_tracker(
    tmp_path: pathlib.Path,
) -> None:
    perturn = _UsageLoggingPerTurnJudge()
    arc = _UsageLoggingArcJudge()
    result = run_submission(
        manifest=_manifest(),
        specs=_two_specs(),
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        perturn_judge=perturn,
        arc_judge=arc,
        paraphrase_seeds=(0,),
        artifact_dir=tmp_path,
    )
    cost_json = result.cost.to_json()
    # Per-turn judge: at least one perturn call per arc, each accumulating
    # input + output tokens.
    assert cost_json["perturn_judge"]["calls"] >= 2
    assert cost_json["perturn_judge"]["input_tokens"] > 0
    assert cost_json["perturn_judge"]["output_tokens"] > 0
    # Arc judge: exactly one call per arc.
    assert cost_json["arc_judge"]["calls"] == 2
    assert cost_json["arc_judge"]["input_tokens"] == 1000  # 2 * 500
    assert cost_json["arc_judge"]["output_tokens"] == 160  # 2 * 80


def test_fake_judges_have_drain_usage_log() -> None:
    """Deterministic fakes must expose ``drain_usage_log`` per Protocol."""
    from companion_bench.judge_arc import DeterministicFakeArcJudge

    perturn = DeterministicFakePerTurnJudge()
    arc = DeterministicFakeArcJudge()
    assert perturn.drain_usage_log() == []
    assert arc.drain_usage_log() == []


# ---------------------------------------------------------------------------
# #73 arc-level failure isolation
# ---------------------------------------------------------------------------


def test_arc_failure_does_not_abort_submission(tmp_path: pathlib.Path) -> None:
    """Single arc raising must not terminate the submission.

    With ``fail_isolated=True`` (default) the surviving arc(s) must
    still aggregate, and ``arc_failure.jsonl`` records the failure.
    """
    perturn = DeterministicFakePerTurnJudge()
    arc_judge = _RaisingArcJudge(raise_on_arc_index=1)
    result = run_submission(
        manifest=_manifest(),
        specs=_two_specs(),
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        perturn_judge=perturn,
        arc_judge=arc_judge,
        paraphrase_seeds=(0,),
        artifact_dir=tmp_path,
    )
    # 2 specs × 1 seed = 2 attempts; one raised, one survived.
    assert len(result.arc_bundles) == 1
    failure_log = tmp_path / "arc_failure.jsonl"
    assert failure_log.exists()
    lines = failure_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["exception_type"] == "RuntimeError"
    assert entry["stage"] == "arc_judge"
    assert entry["scenario_id"] in {"F1-continuity-001", "F5-boundary-001"}


def test_arc_failure_propagates_when_fail_isolated_false(
    tmp_path: pathlib.Path,
) -> None:
    """Tests / debugging mode: ``fail_isolated=False`` re-raises immediately."""
    import pytest

    perturn = DeterministicFakePerTurnJudge()
    arc_judge = _RaisingArcJudge(raise_on_arc_index=0)
    with pytest.raises(RuntimeError, match="synthetic arc judge failure"):
        run_submission(
            manifest=_manifest(),
            specs=_two_specs(),
            sut_client=EchoFakeSUTClient(),
            user_backend=DeterministicFakeUtteranceClient(),
            perturn_judge=perturn,
            arc_judge=arc_judge,
            paraphrase_seeds=(0,),
            artifact_dir=tmp_path,
            fail_isolated=False,
        )
