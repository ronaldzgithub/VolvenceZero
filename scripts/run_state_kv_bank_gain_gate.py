#!/usr/bin/env python3
"""Run the frozen-Qwen four-arm State KV per-bank gain experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in [
    Path(__file__).resolve().parent,
    *sorted((REPO_ROOT / "packages").glob("*/src")),
]:
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_identification import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _fingerprint_weights,
    _resolve_local_weights,
)
from volvence_zero.agent.dialogue import (  # noqa: E402
    DEFAULT_DIALOGUE_PROOF_CASES,
    build_standard_dialogue_runner,
)
from volvence_zero.agent.response import LLMResponseSynthesizer  # noqa: E402
from volvence_zero.conditioning_bank_contracts import (  # noqa: E402
    ConditioningBankReadout,
)
from volvence_zero.personal_conditioning_contracts import (  # noqa: E402
    PersonalConditioningSnapshot,
)
from volvence_zero.owner_hydration import (  # noqa: E402
    OwnerPersistenceSnapshot,
)
from volvence_zero.semantic_embedding import (  # noqa: E402
    reset_semantic_embedding_backend,
    semantic_embedding_backend_status,
    set_semantic_embedding_backend,
)
from volvence_zero.state_kv_bank_gain_gate import (  # noqa: E402
    BANK_GAIN_PROFILE_LABELS,
    IrrelevantBankControlSample,
    PairedBankGainSample,
    build_bank_gain_verdict,
)
from volvence_zero.state_kv_blind_judge import (  # noqa: E402
    JudgeMaterial,
    JudgeMaterialKind,
    LocalEmbeddingBlindJudge,
    resolve_model_family,
)
from volvence_zero.substrate import (  # noqa: E402
    SubstrateTextEncoderBackend,
    TransformersOpenWeightResidualRuntime,
)
from volvence_zero.temporal.conditioning_router import (  # noqa: E402
    TOPK_SEMANTIC_ROUTER_VERSION,
)

PERSONAS: dict[str, tuple[str, ...]] = {
    "repair": (
        "I felt dismissed in our last exchange and trust is still fragile.",
        "Please acknowledge the rupture before suggesting a plan.",
        "I need a reversible next step while we rebuild trust.",
    ),
    "steady": (
        "Our previous exchange helped and I trust the direction we chose.",
        "The decision is approved and I am ready to execute.",
        "Keep the continuity, then help me take the next concrete step.",
    ),
}
GAIN_PROBES: tuple[tuple[str, str], ...] = (
    ("g0", "What should I protect before I act?"),
    ("g1", "How should we approach the next step?"),
    ("g2", "What would a careful response look like?"),
    ("g3", "Should I pause or proceed?"),
)
IRRELEVANT_PROBES: tuple[tuple[str, str], ...] = (
    ("n0", "Explain why a checksum changes when one byte changes."),
    ("n1", "Compare breadth-first and depth-first traversal."),
    ("n2", "What does an HTTP 304 response mean?"),
    ("n3", "Describe how binary search narrows its interval."),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _router_score(result: Any, bank_type: str) -> float:
    trace = result.dialogue_trace
    if trace is None or trace.conditioning_lineage is None:
        raise RuntimeError("dual-bank probe produced no conditioning lineage")
    scores = dict(trace.conditioning_lineage.shadow_router_scores)
    if bank_type not in scores:
        raise RuntimeError(
            f"dual-bank probe has no SHADOW score for {bank_type!r}"
        )
    return float(scores[bank_type])


def _materials(result: Any) -> dict[str, str]:
    personal = result.active_snapshots.get("personal_conditioning")
    relationship = result.active_snapshots.get("relationship_conditioning")
    if (
        personal is None
        or not isinstance(personal.value, PersonalConditioningSnapshot)
        or relationship is None
        or not isinstance(relationship.value, ConditioningBankReadout)
    ):
        raise RuntimeError(
            "dual-bank probe did not publish both conditioning readouts"
        )
    rendered = {
        "personal": personal.value.rendered_statement.strip(),
        "relationship": relationship.value.rendered_statement.strip(),
    }
    if not all(rendered.values()):
        raise RuntimeError("conditioning readout material must be non-empty")
    return rendered


async def _run_turn(
    *,
    profile_label: str,
    persona_id: str,
    probe_id: str,
    user_input: str,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
    semantic_state_snapshot: OwnerPersistenceSnapshot,
) -> dict[str, object]:
    case = replace(
        DEFAULT_DIALOGUE_PROOF_CASES[0],
        case_id=f"state-kv-bank-gain:{persona_id}:{probe_id}",
    )
    runner = build_standard_dialogue_runner(
        profile_label=profile_label,
        case=case,
    )
    runner._semantic_state_store.hydrate_from_persistence(
        semantic_state_snapshot
    )
    # Warm-up uses the deterministic non-LLM response path. The evidence turn
    # is the only turn that reaches the frozen model, so earlier arm-specific
    # model text cannot become an unregistered carrier.
    runner._response_synthesizer = LLMResponseSynthesizer(
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    result = await runner.run_turn(user_input)
    payload: dict[str, object] = {
        "profile": profile_label,
        "persona": persona_id,
        "probe": probe_id,
        "input": user_input,
        "response": result.response.text,
        "rationale_tags": list(result.response.rationale_tags),
    }
    if profile_label == "state-kv-bank-dual":
        payload["materials"] = _materials(result)
        payload["router_scores"] = dict(
            result.dialogue_trace.conditioning_lineage.shadow_router_scores
        )
    return payload


async def _build_persona_snapshots() -> dict[str, OwnerPersistenceSnapshot]:
    snapshots = {}
    for persona_id in sorted(PERSONAS):
        case = replace(
            DEFAULT_DIALOGUE_PROOF_CASES[0],
            case_id=f"state-kv-bank-gain:warmup:{persona_id}",
        )
        runner = build_standard_dialogue_runner(
            profile_label="state-kv-bank-none",
            case=case,
        )
        for warmup in PERSONAS[persona_id]:
            await runner.run_turn(warmup)
        snapshots[persona_id] = (
            runner._semantic_state_store.export_persistence_snapshot()
        )
    return snapshots


async def _collect_observations(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
) -> tuple[dict[str, object], ...]:
    observations = []
    persona_snapshots = await _build_persona_snapshots()
    for profile_label in BANK_GAIN_PROFILE_LABELS:
        for persona_id in sorted(PERSONAS):
            for probe_id, user_input in (*GAIN_PROBES, *IRRELEVANT_PROBES):
                observations.append(
                    await _run_turn(
                        profile_label=profile_label,
                        persona_id=persona_id,
                        probe_id=probe_id,
                        user_input=user_input,
                        runtime=runtime,
                        max_new_tokens=max_new_tokens,
                        semantic_state_snapshot=persona_snapshots[persona_id],
                    )
                )
    return tuple(observations)


def _judge_for(
    *,
    bank_type: str,
    probe_id: str,
    observations: dict[tuple[str, str, str], dict[str, object]],
    judge_model_id: str,
    substrate_model_id: str,
    judge_source: str,
    substrate_source: str,
    judge_model: object,
    judge_tokenizer: object,
    judge_family: str,
    substrate_family: str,
    device: str,
) -> LocalEmbeddingBlindJudge:
    materials = []
    for persona_id in sorted(PERSONAS):
        dual = observations[("state-kv-bank-dual", persona_id, probe_id)]
        rendered = dual["materials"]
        if not isinstance(rendered, dict):
            raise TypeError("dual observation materials must be an object")
        materials.append(
            JudgeMaterial(
                user_id=persona_id,
                summary=str(rendered[bank_type]),
                material_kind=JudgeMaterialKind.RENDERED_STATE,
            )
        )
    return LocalEmbeddingBlindJudge(
        judge_model_id=judge_model_id,
        judge_source=judge_source,
        substrate_model_id=substrate_model_id,
        substrate_source=substrate_source,
        materials=tuple(materials),
        device=device,
        local_files_only=True,
        model=judge_model,
        tokenizer=judge_tokenizer,
        judge_family=judge_family,
        substrate_family=substrate_family,
    )


def _build_samples(
    *,
    raw_observations: tuple[dict[str, object], ...],
    judge_model_id: str,
    judge_source: str,
    substrate_model_id: str,
    substrate_source: str,
    judge_device: str,
) -> tuple[
    tuple[PairedBankGainSample, ...],
    tuple[IrrelevantBankControlSample, ...],
    str,
]:
    from transformers import AutoModel, AutoTokenizer

    judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_source,
        local_files_only=True,
    )
    judge_model = AutoModel.from_pretrained(
        judge_source,
        local_files_only=True,
    )
    judge_family = resolve_model_family(
        model_id=judge_source,
        local_files_only=True,
    )
    substrate_family = resolve_model_family(
        model_id=substrate_source,
        local_files_only=True,
    )
    observations = {
        (str(row["profile"]), str(row["persona"]), str(row["probe"])): row
        for row in raw_observations
    }
    paired = []
    for bank_type, ablated_profile in (
        ("personal", "state-kv-bank-relationship-only"),
        ("relationship", "state-kv-bank-personal-only"),
    ):
        for probe_id, _ in GAIN_PROBES:
            judge = _judge_for(
                bank_type=bank_type,
                probe_id=probe_id,
                observations=observations,
                judge_model_id=judge_model_id,
                substrate_model_id=substrate_model_id,
                judge_source=judge_source,
                substrate_source=substrate_source,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                judge_family=judge_family,
                substrate_family=substrate_family,
                device=judge_device,
            )
            candidates = tuple(sorted(PERSONAS))
            for persona_id in candidates:
                dual = observations[
                    ("state-kv-bank-dual", persona_id, probe_id)
                ]
                ablated = observations[
                    (ablated_profile, persona_id, probe_id)
                ]
                paired.append(
                    PairedBankGainSample(
                        probe_id=f"{persona_id}:{probe_id}",
                        bank_type=bank_type,
                        dual_output=str(dual["response"]),
                        ablated_output=str(ablated["response"]),
                        dual_match_correct=(
                            judge.match(
                                response_text=str(dual["response"]),
                                candidate_user_ids=candidates,
                            )
                            == persona_id
                        ),
                        ablated_match_correct=(
                            judge.match(
                                response_text=str(ablated["response"]),
                                candidate_user_ids=candidates,
                            )
                            == persona_id
                        ),
                    )
                )
    irrelevant = []
    for probe_id, _ in IRRELEVANT_PROBES:
        judge = _judge_for(
            bank_type="relationship",
            probe_id=probe_id,
            observations=observations,
            judge_model_id=judge_model_id,
            substrate_model_id=substrate_model_id,
            judge_source=judge_source,
            substrate_source=substrate_source,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_family=judge_family,
            substrate_family=substrate_family,
            device=judge_device,
        )
        candidates = tuple(sorted(PERSONAS))
        for persona_id in candidates:
            with_bank = observations[
                ("state-kv-bank-dual", persona_id, probe_id)
            ]
            without_bank = observations[
                ("state-kv-bank-personal-only", persona_id, probe_id)
            ]
            scores = with_bank["router_scores"]
            if not isinstance(scores, dict):
                raise TypeError("dual observation router_scores must be an object")
            irrelevant.append(
                IrrelevantBankControlSample(
                    probe_id=f"{persona_id}:{probe_id}",
                    bank_type="relationship",
                    router_score=float(scores["relationship"]),
                    without_bank_match_correct=(
                        judge.match(
                            response_text=str(without_bank["response"]),
                            candidate_user_ids=candidates,
                        )
                        == persona_id
                    ),
                    with_bank_match_correct=(
                        judge.match(
                            response_text=str(with_bank["response"]),
                            candidate_user_ids=candidates,
                        )
                        == persona_id
                    ),
                )
            )
    return tuple(paired), tuple(irrelevant), judge_family


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--judge-model-id", default="BAAI/bge-m3")
    parser.add_argument("--judge-source", default="")
    parser.add_argument("--judge-device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/state_kv/verdict_bank_gain.json",
    )
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--bootstrap-seed", type=int, default=7301)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    weights_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    judge_root = _resolve_local_weights(
        model_id=args.judge_model_id,
        model_source=args.judge_source,
        allow_download=args.allow_download,
    )
    substrate_fingerprint_payload = _fingerprint_weights(
        model_id=args.model_id,
        weights_root=weights_root,
    )
    judge_fingerprint_payload = _fingerprint_weights(
        model_id=args.judge_model_id,
        weights_root=judge_root,
    )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=args.model_id,
        pretrained_source=str(weights_root),
        device=args.device,
        local_files_only=True,
        runtime_origin="hf-local",
    )
    set_semantic_embedding_backend(
        SubstrateTextEncoderBackend(runtime),
        owner=runtime.model_id,
    )
    try:
        raw_observations = asyncio.run(
            _collect_observations(
                runtime=runtime,
                max_new_tokens=args.max_new_tokens,
            )
        )
        paired_samples, irrelevant_controls, judge_family = _build_samples(
            raw_observations=raw_observations,
            judge_model_id=args.judge_model_id,
            judge_source=str(judge_root),
            substrate_model_id=args.model_id,
            substrate_source=str(weights_root),
            judge_device=args.judge_device,
        )
        semantic_backend = ":".join(
            str(value) for value in semantic_embedding_backend_status()
        )
    finally:
        reset_semantic_embedding_backend()

    output = (REPO_ROOT / args.output).resolve()
    observation_path = output.with_name("observations_bank_gain.json")
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-bank-gain-observations.v1",
                "substrate": substrate_fingerprint_payload,
                "judge": judge_fingerprint_payload,
                "profiles": list(BANK_GAIN_PROFILE_LABELS),
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                },
                "personas": {
                    key: list(value) for key, value in PERSONAS.items()
                },
                "gain_probes": list(GAIN_PROBES),
                "irrelevant_probes": list(IRRELEVANT_PROBES),
                "turns": list(raw_observations),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    substrate_fingerprint = (
        f"{args.model_id}@"
        f"{str(substrate_fingerprint_payload['weights_sha256'])[:16]}"
    )
    verdict = build_bank_gain_verdict(
        paired_samples=paired_samples,
        irrelevant_controls=irrelevant_controls,
        artifact_id=_sha256(observation_path),
        substrate_fingerprint=substrate_fingerprint,
        router_version=TOPK_SEMANTIC_ROUTER_VERSION,
        minimum_samples=args.minimum_samples,
        bootstrap_seed=args.bootstrap_seed,
        judge_model_id=args.judge_model_id,
        judge_family=judge_family,
        judge_material_kind=JudgeMaterialKind.RENDERED_STATE,
        observation_artifact_sha256=_sha256(observation_path),
        semantic_backend=semantic_backend,
    )
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")
    print(f"gate_state = {verdict.gate_state}")
    print(f"bank_count_frozen = {verdict.bank_count_frozen}")
    print(f"observations = {observation_path}")
    print(f"output = {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
