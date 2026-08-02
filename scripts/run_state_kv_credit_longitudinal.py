#!/usr/bin/env python3
"""Run matched SHADOW/ACTIVE credit-feedback sessions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_identification import _resolve_local_weights  # noqa: E402

from volvence_zero.agent.dialogue import (  # noqa: E402
    DEFAULT_DIALOGUE_PROOF_CASES,
    build_standard_dialogue_runner,
)
from volvence_zero.dialogue_trace import (  # noqa: E402
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.personal_conditioning_contracts import (  # noqa: E402
    PersonalConditioningSnapshot,
)
from volvence_zero.state_kv_credit_longitudinal import (  # noqa: E402
    CreditOutcomeJudgeResult,
    CreditLongitudinalSample,
    build_credit_longitudinal_verdict,
)
from volvence_zero.state_kv_blind_judge import (  # noqa: E402
    JudgeMaterial,
    JudgeMaterialKind,
    LocalEmbeddingBlindJudge,
)

SUBSTRATE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTCOME_MATERIALS = (
    JudgeMaterial(
        user_id="matched-outcome",
        summary=(
            "The response uses the reported outcome instead of guessing, "
            "preserves the stated boundary and continuity, and proposes one "
            "clear reversible next action with a verification point."
        ),
        material_kind=JudgeMaterialKind.MATCHED_OUTCOME_RUBRIC,
    ),
    JudgeMaterial(
        user_id="counter-outcome",
        summary=(
            "The response guesses beyond the reported outcome, ignores the "
            "stated boundary or continuity, and pushes an irreversible or "
            "unverifiable action."
        ),
        material_kind=JudgeMaterialKind.MATCHED_OUTCOME_RUBRIC,
    ),
)

TURN_INPUTS = (
    "I need a reversible next step and a clear boundary.",
    "That helped; keep the same careful direction.",
    "The decision is getting clearer, but do not overreach.",
    "Please preserve continuity while we test one step.",
    "The small step worked; what should remain protected?",
    "Keep using the outcome instead of guessing.",
    "I am ready for the next bounded action.",
    "Close this with a reversible verification point.",
    "The verification passed; retain the boundary.",
    "Now summarize the next action without losing continuity.",
)


async def _run_profile(profile_label: str) -> tuple[dict[str, object], ...]:
    runner = build_standard_dialogue_runner(
        profile_label=profile_label,
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
    )
    rows = []
    for turn_index, user_input in enumerate(TURN_INPUTS, start=1):
        result = await runner.run_turn(user_input)
        snapshot = result.active_snapshots.get("personal_conditioning")
        if (
            snapshot is None
            or not isinstance(snapshot.value, PersonalConditioningSnapshot)
        ):
            raise RuntimeError(
                f"{profile_label} turn {turn_index} published no Personal bank"
            )
        rows.append(
            {
                "turn_index": turn_index,
                "input": user_input,
                "response": result.response.text,
                "confidence": snapshot.value.confidence,
                "credit_confidence_delta": (
                    snapshot.value.credit_confidence_delta
                ),
            }
        )
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.HELPED,
            source=DialogueExternalOutcomeEvidenceSource.HUMAN_REVIEW,
            confidence=0.95,
            evidence_ref=f"state-kv-credit:{profile_label}:{turn_index}",
            action_turn_index=turn_index,
        )
    return tuple(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/state_kv/verdict_credit_longitudinal.json",
    )
    parser.add_argument(
        "--observation-output",
        default=(
            "artifacts/state_kv/credit-longitudinal-v2/"
            "observations_credit_longitudinal.json"
        ),
    )
    parser.set_defaults(
        output=(
            "artifacts/state_kv/credit-longitudinal-v2/"
            "verdict_credit_longitudinal.json"
        )
    )
    parser.add_argument(
        "--judge-model-id",
        action="append",
        default=["BAAI/bge-m3", "moka-ai/m3e-base"],
    )
    parser.add_argument("--judge-device", default="cpu")
    return parser


def _judge_outcomes(*, model_id: str, device: str, paired):
    judge_source = _resolve_local_weights(
        model_id=model_id,
        model_source=None,
        allow_download=False,
    )
    substrate_source = _resolve_local_weights(
        model_id=SUBSTRATE_MODEL_ID,
        model_source=None,
        allow_download=False,
    )
    judge = LocalEmbeddingBlindJudge(
        judge_model_id=model_id,
        substrate_model_id=SUBSTRATE_MODEL_ID,
        materials=OUTCOME_MATERIALS,
        judge_source=str(judge_source),
        substrate_source=str(substrate_source),
        device=device,
        local_files_only=True,
    )
    shadow_matched = 0
    active_matched = 0
    improvements = 0
    regressions = 0
    for row in paired:
        shadow = judge.match(
            response_text=str(row["shadow_response"]),
            candidate_user_ids=("matched-outcome", "counter-outcome"),
        ) == "matched-outcome"
        active = judge.match(
            response_text=str(row["active_response"]),
            candidate_user_ids=("matched-outcome", "counter-outcome"),
        ) == "matched-outcome"
        shadow_matched += int(shadow)
        active_matched += int(active)
        improvements += int(active and not shadow)
        regressions += int(shadow and not active)
    return CreditOutcomeJudgeResult(
        judge_model_id=model_id,
        judge_family=judge.judge_family,
        shadow_matched_count=shadow_matched,
        active_matched_count=active_matched,
        improvement_count=improvements,
        regression_count=regressions,
        sample_count=len(paired),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.judge_model_id) != 2 or len(set(args.judge_model_id)) != 2:
        raise ValueError("credit outcome lane requires exactly two distinct judges")
    shadow = asyncio.run(_run_profile("state-kv-bank-dual"))
    active = asyncio.run(
        _run_profile("state-kv-bank-dual-credit-active")
    )
    if len(shadow) != len(active):
        raise RuntimeError("matched credit sessions produced unequal turn counts")
    paired = tuple(
        {
            "turn_index": int(shadow_row["turn_index"]),
            "input": shadow_row["input"],
            "shadow_response": shadow_row["response"],
            "active_response": active_row["response"],
            "shadow_confidence": shadow_row["confidence"],
            "active_confidence": active_row["confidence"],
            "shadow_credit_delta": shadow_row["credit_confidence_delta"],
            "active_credit_delta": active_row["credit_confidence_delta"],
            "responses_differ": (
                shadow_row["response"] != active_row["response"]
            ),
        }
        for shadow_row, active_row in zip(shadow, active, strict=True)
    )
    observation_output = (REPO_ROOT / args.observation_output).resolve()
    observation_output.parent.mkdir(parents=True, exist_ok=True)
    observation_output.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-credit-longitudinal-observations.v1",
                "shadow_profile": "state-kv-bank-dual",
                "active_profile": "state-kv-bank-dual-credit-active",
                "typed_outcome": "helped",
                "samples": list(paired),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(observation_output.read_bytes()).hexdigest()
    outcome_judges = tuple(
        _judge_outcomes(model_id=model_id, device=args.judge_device, paired=paired)
        for model_id in args.judge_model_id
    )
    verdict = build_credit_longitudinal_verdict(
        samples=tuple(
            CreditLongitudinalSample(
                turn_index=int(row["turn_index"]),
                shadow_confidence=float(row["shadow_confidence"]),
                active_confidence=float(row["active_confidence"]),
                shadow_credit_delta=float(row["shadow_credit_delta"]),
                active_credit_delta=float(row["active_credit_delta"]),
                responses_differ=bool(row["responses_differ"]),
            )
            for row in paired
        ),
        artifact_id=f"state-kv-credit-longitudinal:{digest}",
        outcome_judges=outcome_judges,
        minimum_net_improvement=1,
    )
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")
    print(f"gate_state = {verdict.gate_state}")
    print(f"mechanism_state = {verdict.mechanism_state}")
    print(f"output = {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
