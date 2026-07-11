# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Submission orchestration: run a SUT against the full Companion Bench scenario set.

This module is the entry point for "score one system on one scenario
set" — it wires arc_runner / callback_ledger / disqualifier /
judge_perturn / judge_arc / aggregator / cost into one driver,
emits an artifact bundle, and reports the per-system aggregate.

The orchestrator is **submission-shaped**, not system-shaped: a
"submission" is one ``YAML manifest`` that declares a name, model
identifier, system prompt, generation config, attestation, and
endpoint. We do not bake in any specific system class — the
abstraction is the OpenAI-compatible HTTP endpoint.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import pathlib
from typing import Iterable

import yaml

from companion_bench.aggregator import (
    CompanionBenchScore,
    SubmissionAggregate,
    aggregate_arc,
    aggregate_submission,
)
from companion_bench.arc_runner import ArcRecord, ArcRunConfig, run_arc, write_arc_record
from companion_bench.callback_ledger import (
    CallbackLedger,
    HeuristicCallbackExtractor,
    build_callback_ledger,
)
from companion_bench.cost import CostBreakdown, CostTracker
from companion_bench.disqualifier import (
    DisqualifierReport,
    axis_for_disqualifier,
    run_disqualifiers,
)
from companion_bench.judge_arc import (
    ArcAxisScores,
    ArcJudge,
    DeterministicFakeArcJudge,
    score_arc_axes,
)
from companion_bench.judge_perturn import (
    ArcRubric,
    DeterministicFakePerTurnJudge,
    PerTurnJudge,
    score_arc_perturn,
)
from companion_bench.spec import AxisId, ScenarioSpec, scenario_hash
from companion_bench.sut_client import SUTClient
from companion_bench.user_simulator import UtteranceClient


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SubmissionAttestation:
    """Required attestation block (RFC §7.2)."""

    no_companionbench_derivative_in_training: bool
    no_scenario_specific_prompt: bool
    no_public_test_set_tuning: bool
    cross_user_memory_isolation: bool

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def all_affirmed(self) -> bool:
        return all(
            (
                self.no_companionbench_derivative_in_training,
                self.no_scenario_specific_prompt,
                self.no_public_test_set_tuning,
                self.cross_user_memory_isolation,
            )
        )


@dataclasses.dataclass(frozen=True)
class SubmissionManifest:
    """Parsed submission manifest YAML."""

    submission_id: str
    system_name: str
    model_identifier: str
    base_url: str
    api_key_env: str
    system_prompt: str
    generation_config: dict
    attestation: SubmissionAttestation
    leaderboard_category: str  # "open-weight" / "closed-api" / "bespoke"

    def to_json(self) -> dict:
        return {
            "submission_id": self.submission_id,
            "system_name": self.system_name,
            "model_identifier": self.model_identifier,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "system_prompt": self.system_prompt,
            "generation_config": dict(self.generation_config),
            "attestation": self.attestation.to_json(),
            "leaderboard_category": self.leaderboard_category,
        }


def load_manifest(path: pathlib.Path | str) -> SubmissionManifest:
    """Parse a submission YAML; raises on schema break (no defensive defaults)."""

    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"submission manifest must be a YAML mapping; got {type(raw).__name__}")
    required = (
        "submission_id",
        "system_name",
        "model_identifier",
        "base_url",
        "api_key_env",
        "system_prompt",
        "attestation",
        "leaderboard_category",
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"submission manifest missing required field {key!r}")
    cat = raw["leaderboard_category"]
    if cat not in {"open-weight", "closed-api", "bespoke"}:
        raise ValueError(
            f"leaderboard_category must be one of "
            f"['open-weight', 'closed-api', 'bespoke']; got {cat!r}"
        )
    att_raw = raw["attestation"]
    if not isinstance(att_raw, dict):
        raise ValueError("attestation block must be a mapping")
    attestation = SubmissionAttestation(
        no_companionbench_derivative_in_training=bool(
            att_raw.get("no_companionbench_derivative_in_training", False)
        ),
        no_scenario_specific_prompt=bool(att_raw.get("no_scenario_specific_prompt", False)),
        no_public_test_set_tuning=bool(att_raw.get("no_public_test_set_tuning", False)),
        cross_user_memory_isolation=bool(att_raw.get("cross_user_memory_isolation", False)),
    )
    if not attestation.all_affirmed():
        raise ValueError(
            "submission manifest must affirm all four attestation fields "
            "(see RFC §7.2). Set every attestation.* field to true."
        )
    return SubmissionManifest(
        submission_id=str(raw["submission_id"]),
        system_name=str(raw["system_name"]),
        model_identifier=str(raw["model_identifier"]),
        base_url=str(raw["base_url"]),
        api_key_env=str(raw["api_key_env"]),
        system_prompt=str(raw["system_prompt"]),
        generation_config=dict(raw.get("generation_config", {})),
        attestation=attestation,
        leaderboard_category=cat,
    )


# ---------------------------------------------------------------------------
# Per-arc bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArcBundle:
    """Everything we produce for one (scenario, paraphrase_seed) pair."""

    arc: ArcRecord
    callback_ledger: CallbackLedger
    disqualifier_report: DisqualifierReport
    perturn_rubric: ArcRubric
    arc_axis_scores: ArcAxisScores
    final_score: CompanionBenchScore

    def to_json(self) -> dict:
        return {
            "arc": self.arc.to_json(),
            "callback_ledger": self.callback_ledger.to_json(),
            "disqualifier_report": self.disqualifier_report.to_json(),
            "perturn_rubric": self.perturn_rubric.to_json(),
            "arc_axis_scores": self.arc_axis_scores.to_json(),
            "final_score": self.final_score.to_json(),
        }


# ---------------------------------------------------------------------------
# Submission run
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SubmissionResult:
    """Top-level result of running one submission against many scenarios."""

    manifest: SubmissionManifest
    aggregate: SubmissionAggregate
    arc_bundles: tuple[ArcBundle, ...]
    cost: CostBreakdown
    started_at: str
    finished_at: str

    def to_json(self) -> dict:
        return {
            "manifest": self.manifest.to_json(),
            "aggregate": self.aggregate.to_json(),
            "cost": self.cost.to_json(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "arc_count": len(self.arc_bundles),
        }


def _arc_run_config_from_manifest(
    manifest: SubmissionManifest,
    *,
    user_simulator_model: str,
) -> ArcRunConfig:
    """Translate manifest declared system_prompt + generation_config into runner config.

    Per debt #74 the manifest's system_prompt + generation_config
    fields must reach the SUT message stream; arc_runner exposes
    them via :class:`ArcRunConfig`. ``max_tokens`` and ``temperature``
    are pulled from ``generation_config`` when present and otherwise
    fall back to the dataclass defaults (512 / 0.0). Unknown keys in
    ``generation_config`` are tolerated so manifests that ship
    extra knobs (top_p / top_k / etc.) do not need a runner upgrade
    in lock-step.
    """

    gen = dict(manifest.generation_config or {})
    raw_max = gen.get("max_tokens")
    max_tokens: int | None
    if raw_max is None:
        max_tokens = 512
    else:
        max_tokens = int(raw_max)
    raw_temp = gen.get("temperature")
    temperature: float | None
    if raw_temp is None:
        temperature = 0.0
    else:
        temperature = float(raw_temp)
    return ArcRunConfig(
        submission_id=manifest.submission_id,
        user_simulator_model=user_simulator_model,
        sut_max_tokens=max_tokens,
        sut_temperature=temperature,
        system_prompt=manifest.system_prompt,
    )


def _drain_judge_usage(
    *,
    judge_obj: object,
    tracker: CostTracker,
    record_callable_name: str,
) -> None:
    """Call ``judge.drain_usage_log()`` and feed entries to the tracker.

    Per the PerTurnJudge / ArcJudge protocols (debt #75) every judge
    implementation exposes ``drain_usage_log() -> list[dict]``. Fakes
    return ``[]``; LLM-backed judges return one entry per LLM call
    accumulated since the last drain. We call the drain after every
    arc so cost is attributed even when an arc later raises.
    """

    drain = judge_obj.drain_usage_log
    record = getattr(tracker, record_callable_name)
    for entry in drain():
        record(
            model=judge_obj.model,
            prompt_tokens=entry.get("prompt_tokens"),
            completion_tokens=entry.get("completion_tokens"),
        )


def run_submission(
    *,
    manifest: SubmissionManifest,
    specs: Iterable[ScenarioSpec],
    sut_client: SUTClient,
    user_backend: UtteranceClient,
    perturn_judge: PerTurnJudge,
    arc_judge: ArcJudge,
    paraphrase_seeds: Iterable[int] = (0, 1, 2),
    artifact_dir: pathlib.Path | str | None = None,
    user_simulator_model: str = "fake/user-sim",
    fail_isolated: bool = True,
) -> SubmissionResult:
    """Run one submission across (specs × paraphrase_seeds) → SubmissionResult.

    Cost telemetry is captured automatically. If ``artifact_dir`` is
    set, every per-arc bundle is written as ``{arc_id}.json`` under it.

    ``fail_isolated`` (debt #73): when True (default), an exception in
    a single arc is logged to ``arc_failure.jsonl`` under
    ``artifact_dir`` and the submission continues with the next arc.
    Set False for tests that want to surface the exception (or when
    re-running a single arc for debugging). The ``arc_failure.jsonl``
    has one line per failed arc:
    ``{"scenario_id": ..., "paraphrase_seed": ..., "exception": ...,
       "stage": ...}``. The remaining ArcBundles aggregate normally,
    so partial submissions still produce a usable summary.
    """

    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    tracker = CostTracker()
    bundles: list[ArcBundle] = []
    arc_dir = pathlib.Path(artifact_dir) if artifact_dir else None
    if arc_dir:
        arc_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    arc_config = _arc_run_config_from_manifest(
        manifest, user_simulator_model=user_simulator_model
    )

    for spec in specs:
        for seed in paraphrase_seeds:
            if seed >= spec.paraphrase_seed_count:
                continue
            stage = "run_arc"
            try:
                arc = run_arc(
                    spec=spec,
                    paraphrase_seed=seed,
                    sut_client=sut_client,
                    user_backend=user_backend,
                    config=arc_config,
                )
                tracker.record_arc_record(arc)
                stage = "callback_ledger"
                ledger = build_callback_ledger(
                    arc=arc, extractor=HeuristicCallbackExtractor()
                )
                stage = "disqualifiers"
                disq = run_disqualifiers(
                    arc=arc, ledger=ledger, declared=spec.disqualifiers
                )
                stage = "perturn_judge"
                perturn = score_arc_perturn(arc=arc, judge=perturn_judge)
                _drain_judge_usage(
                    judge_obj=perturn_judge,
                    tracker=tracker,
                    record_callable_name="record_perturn_judge",
                )
                stage = "arc_judge"
                arc_scores = score_arc_axes(
                    arc=arc,
                    ledger=ledger,
                    family=spec.family.value,
                    judge=arc_judge,
                )
                _drain_judge_usage(
                    judge_obj=arc_judge,
                    tracker=tracker,
                    record_callable_name="record_arc_judge",
                )
                stage = "aggregate"
                # Scoring pipeline (order matters):
                #   1. ledger fabrication penalty on A3 (RFC §4 hard penalty)
                #   2. blend per-turn EQ rubric into A2 (RFC §6.1 signal)
                #   3. disqualifier void of mapped axis (RFC §B.1) — LAST so a
                #      triggered disqualifier is authoritative over the blend.
                penalised = _apply_ledger_penalty(arc_scores, ledger=ledger)
                penalised = _blend_perturn_into_a2(penalised, perturn=perturn)
                penalised = _apply_disqualifier_penalty(penalised, report=disq)
                final = aggregate_arc(penalised)
                bundle = ArcBundle(
                    arc=arc,
                    callback_ledger=ledger,
                    disqualifier_report=disq,
                    perturn_rubric=perturn,
                    arc_axis_scores=penalised,
                    final_score=final,
                )
                bundles.append(bundle)
                if arc_dir is not None:
                    _write_bundle(bundle, arc_dir)
            except Exception as exc:
                if not fail_isolated:
                    raise
                # Drain any partial usage so cost is attributed even on
                # arc failure (judge may have been called before the
                # exception).
                for jobj, name in (
                    (perturn_judge, "record_perturn_judge"),
                    (arc_judge, "record_arc_judge"),
                ):
                    try:
                        _drain_judge_usage(
                            judge_obj=jobj,
                            tracker=tracker,
                            record_callable_name=name,
                        )
                    except AttributeError:
                        # Judge implementations are required to expose
                        # drain_usage_log; if missing we surface it as
                        # part of the original failure.
                        pass
                failures.append(
                    {
                        "scenario_id": spec.scenario_id,
                        "paraphrase_seed": seed,
                        "stage": stage,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                )

    if arc_dir is not None and failures:
        ledger_path = arc_dir / "arc_failure.jsonl"
        with ledger_path.open("a", encoding="utf-8") as fh:
            for entry in failures:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    aggregate = aggregate_submission(
        submission_id=manifest.submission_id,
        per_arc_scores=[b.final_score for b in bundles],
    )
    cost = tracker.freeze()
    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return SubmissionResult(
        manifest=manifest,
        aggregate=aggregate,
        arc_bundles=tuple(bundles),
        cost=cost,
        started_at=started_at,
        finished_at=finished_at,
    )


def _apply_ledger_penalty(
    scores: ArcAxisScores, *, ledger: CallbackLedger,
) -> ArcAxisScores:
    """RFC §4: any fabricated callback caps A3 at the floor.

    We apply a soft hard penalty: if any fabrication exists, A3 is
    capped at 30 (matches the deterministic-fake judge's enforcement
    so production and test paths agree).
    """
    if ledger.fabrication_count == 0:
        return scores
    new_scores = dict(scores.scores)
    new_scores[AxisId.A3_CONTINUITY] = min(new_scores.get(AxisId.A3_CONTINUITY, 0.0), 30.0)
    return ArcAxisScores(
        arc_id=scores.arc_id,
        judge_model=scores.judge_model,
        scores=new_scores,
        rationale=dict(scores.rationale),
    )


# Fraction of the A2 axis contributed by the per-turn EQ rubric. The
# per-turn 8-criterion rubric (judge_perturn.CRITERIA, labelled A2.1-A2.8)
# is an independent EQ signal on the same axis as the arc judge's A2, so we
# average the two 50/50. This gives the final score a second, per-turn
# judge signal instead of letting the single arc judge's A2 dominate.
_PERTURN_A2_WEIGHT: float = 0.5


def _blend_perturn_into_a2(
    scores: ArcAxisScores, *, perturn: ArcRubric,
) -> ArcAxisScores:
    """Blend the arc-level per-turn EQ mean into the A2 axis (RFC §6.1).

    The per-turn rubric scores each turn 0-5 across 8 EQ criteria. We take
    the mean of per-turn averages, scale to 0-100 (x20), and blend with the
    arc judge's A2 using ``_PERTURN_A2_WEIGHT``. With no scored turns the A2
    axis is left unchanged.
    """
    turn_scores = perturn.turn_scores
    if not turn_scores:
        return scores
    perturn_mean_0_5 = sum(ts.average() for ts in turn_scores) / len(turn_scores)
    perturn_mean_0_100 = perturn_mean_0_5 * 20.0
    arc_a2 = scores.scores.get(AxisId.A2_CONVERSATIONAL, 0.0)
    blended = (
        (1.0 - _PERTURN_A2_WEIGHT) * arc_a2
        + _PERTURN_A2_WEIGHT * perturn_mean_0_100
    )
    new_scores = dict(scores.scores)
    new_scores[AxisId.A2_CONVERSATIONAL] = max(0.0, min(100.0, blended))
    return ArcAxisScores(
        arc_id=scores.arc_id,
        judge_model=scores.judge_model,
        scores=new_scores,
        rationale=dict(scores.rationale),
    )


def _apply_disqualifier_penalty(
    scores: ArcAxisScores, *, report: DisqualifierReport,
) -> ArcAxisScores:
    """Void the mapped axis of every triggered disqualifier (RFC §B.1).

    A triggered disqualifier floors its mapped axis (see
    ``disqualifier.DISQUALIFIER_AXIS``) to 0. Voiding A6 also trips the
    A6-cap in aggregation (final <= 50). Applied last so a deterministic
    disqualifier is authoritative over the per-turn blend and judge scores.
    """
    triggered = report.triggered_kinds()
    if not triggered:
        return scores
    new_scores = dict(scores.scores)
    for kind in triggered:
        new_scores[axis_for_disqualifier(kind)] = 0.0
    return ArcAxisScores(
        arc_id=scores.arc_id,
        judge_model=scores.judge_model,
        scores=new_scores,
        rationale=dict(scores.rationale),
    )


def _write_bundle(bundle: ArcBundle, out_dir: pathlib.Path) -> pathlib.Path:
    out_path = out_dir / f"{bundle.arc.arc_id}.bundle.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle.to_json(), fh, indent=2, ensure_ascii=False)
    return out_path


def write_submission_summary(
    result: SubmissionResult, out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Top-level summary JSON consumed by the leaderboard site."""
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_json()
    payload["per_axis_scores"] = result.aggregate.to_json()
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return p


# ---------------------------------------------------------------------------
# Convenience: dry-run end-to-end with fakes
# ---------------------------------------------------------------------------


def dry_run_with_fakes(
    *,
    manifest: SubmissionManifest,
    specs: Iterable[ScenarioSpec],
    sut_client: SUTClient,
    user_backend: UtteranceClient,
    paraphrase_seeds: Iterable[int] = (0,),
    artifact_dir: pathlib.Path | str | None = None,
) -> SubmissionResult:
    """Convenience for tests and CI smoke runs."""
    return run_submission(
        manifest=manifest,
        specs=specs,
        sut_client=sut_client,
        user_backend=user_backend,
        perturn_judge=DeterministicFakePerTurnJudge(),
        arc_judge=DeterministicFakeArcJudge(),
        paraphrase_seeds=paraphrase_seeds,
        artifact_dir=artifact_dir,
    )
