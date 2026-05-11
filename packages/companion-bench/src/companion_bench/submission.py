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
from companion_bench.disqualifier import DisqualifierReport, run_disqualifiers
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

    no_lscb_derivative_in_training: bool
    no_scenario_specific_prompt: bool
    no_public_test_set_tuning: bool
    cross_user_memory_isolation: bool

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def all_affirmed(self) -> bool:
        return all(
            (
                self.no_lscb_derivative_in_training,
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
        no_lscb_derivative_in_training=bool(att_raw.get("no_lscb_derivative_in_training", False)),
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
) -> SubmissionResult:
    """Run one submission across (specs × paraphrase_seeds) → SubmissionResult.

    Cost telemetry is captured automatically. If ``artifact_dir`` is
    set, every per-arc bundle is written as ``{arc_id}.json`` under it.
    """

    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    tracker = CostTracker()
    bundles: list[ArcBundle] = []
    arc_dir = pathlib.Path(artifact_dir) if artifact_dir else None
    if arc_dir:
        arc_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        for seed in paraphrase_seeds:
            if seed >= spec.paraphrase_seed_count:
                continue
            arc = run_arc(
                spec=spec,
                paraphrase_seed=seed,
                sut_client=sut_client,
                user_backend=user_backend,
                config=ArcRunConfig(
                    submission_id=manifest.submission_id,
                    user_simulator_model=user_simulator_model,
                ),
            )
            tracker.record_arc_record(arc)
            ledger = build_callback_ledger(arc=arc, extractor=HeuristicCallbackExtractor())
            disq = run_disqualifiers(arc=arc, ledger=ledger, declared=spec.disqualifiers)
            perturn = score_arc_perturn(arc=arc, judge=perturn_judge)
            arc_scores = score_arc_axes(
                arc=arc, ledger=ledger, family=spec.family.value, judge=arc_judge,
            )
            # Apply ledger fabrication penalty on A3 BEFORE aggregation
            # (RFC §4 hard penalty).
            penalised = _apply_ledger_penalty(arc_scores, ledger=ledger)
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
