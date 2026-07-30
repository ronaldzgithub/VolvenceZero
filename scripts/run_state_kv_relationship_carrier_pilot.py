#!/usr/bin/env python3
"""Run a matched text-vs-residual pilot for the Relationship bank."""

from __future__ import annotations

import argparse
import asyncio
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

from run_state_kv_bank_gain_gate import (  # noqa: E402
    GAIN_PROBES,
    IRRELEVANT_PROBES,
    PERSONAS,
    _build_persona_snapshots,
)
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
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot  # noqa: E402
from volvence_zero.semantic_embedding import (  # noqa: E402
    reset_semantic_embedding_backend,
    set_semantic_embedding_backend,
)
from volvence_zero.state_kv_blind_judge import (  # noqa: E402
    JudgeMaterial,
    JudgeMaterialKind,
    LocalEmbeddingBlindJudge,
    resolve_model_family,
)
from volvence_zero.substrate import (  # noqa: E402
    RelationshipConditioningProjectorArtifact,
    SubstrateTextEncoderBackend,
    TransformersOpenWeightResidualRuntime,
)

PROFILE_NONE = "state-kv-bank-none"
PROFILE_TEXT = "state-kv-bank-relationship-only"
PROFILE_LATENT = "state-kv-bank-relationship-latent-pure"
PROFILES = (PROFILE_NONE, PROFILE_TEXT, PROFILE_LATENT)


def _tag_value(tags: tuple[str, ...], key: str) -> str:
    prefix = f"{key}="
    matches = tuple(tag[len(prefix) :] for tag in tags if tag.startswith(prefix))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {key!r} rationale tag, got {matches!r}"
        )
    return matches[0]


async def _run_arm(
    *,
    profile_label: str,
    persona_id: str,
    probe_id: str,
    user_input: str,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
    semantic_state_snapshot: OwnerPersistenceSnapshot,
) -> dict[str, Any]:
    case = replace(
        DEFAULT_DIALOGUE_PROOF_CASES[0],
        case_id=f"relationship-carrier:{persona_id}:{probe_id}",
    )
    runner = build_standard_dialogue_runner(
        profile_label=profile_label,
        case=case,
        residual_runtime=runtime,
    )
    runner._semantic_state_store.hydrate_from_persistence(
        semantic_state_snapshot
    )
    runner._response_synthesizer = LLMResponseSynthesizer(
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    result = await runner.run_turn(user_input)
    relationship = result.active_snapshots.get("relationship_conditioning")
    readout = (
        relationship.value
        if relationship is not None
        and isinstance(relationship.value, ConditioningBankReadout)
        else None
    )
    lineage = (
        result.dialogue_trace.conditioning_lineage
        if result.dialogue_trace is not None
        else None
    )
    fingerprints = (
        dict(lineage.bank_fingerprints) if lineage is not None else {}
    )
    tags = result.response.rationale_tags
    return {
        "profile": profile_label,
        "persona": persona_id,
        "probe": probe_id,
        "input": user_input,
        "response": result.response.text,
        "prompt_fingerprint": _tag_value(tags, "prompt_fp"),
        "relationship_material": (
            readout.rendered_statement if readout is not None else ""
        ),
        "relationship_source_fingerprint": (
            readout.source_fingerprint if readout is not None else ""
        ),
        "relationship_lineage_fingerprint": fingerprints.get(
            "relationship", ""
        ),
        "lineage_state_encoder_version": (
            lineage.state_encoder_version if lineage is not None else ""
        ),
        "relationship_carrier_tag": next(
            (
                tag
                for tag in tags
                if tag.startswith("relationship_conditioning=")
                or tag.startswith("relationship_conditioning_not_applied=")
            ),
            "",
        ),
    }


async def _collect(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
    probes: tuple[tuple[str, str], ...],
) -> tuple[dict[str, Any], ...]:
    snapshots = await _build_persona_snapshots(runtime=runtime)
    rows = []
    for profile in PROFILES:
        for persona in sorted(PERSONAS):
            for probe_id, user_input in probes:
                rows.append(
                    await _run_arm(
                        profile_label=profile,
                        persona_id=persona,
                        probe_id=probe_id,
                        user_input=user_input,
                        runtime=runtime,
                        max_new_tokens=max_new_tokens,
                        semantic_state_snapshot=snapshots[persona],
                    )
                )
            print(
                f"observations[{profile}:{persona}] = {len(probes)}",
                flush=True,
            )
    return tuple(rows)


def _accuracy(
    *,
    profile: str,
    probe_ids: tuple[str, ...],
    observations: dict[tuple[str, str, str], dict[str, Any]],
    judge_model_id: str,
    judge_source: str,
    substrate_model_id: str,
    substrate_source: str,
    judge_model: object,
    judge_tokenizer: object,
    judge_family: str,
    substrate_family: str,
    device: str,
) -> tuple[int, int]:
    correct = 0
    total = 0
    personas = tuple(sorted(PERSONAS))
    for probe_id in probe_ids:
        materials = tuple(
            JudgeMaterial(
                user_id=persona,
                summary=str(
                    observations[(PROFILE_TEXT, persona, probe_id)][
                        "relationship_material"
                    ]
                ),
                material_kind=JudgeMaterialKind.RENDERED_STATE,
            )
            for persona in personas
        )
        judge = LocalEmbeddingBlindJudge(
            judge_model_id=judge_model_id,
            judge_source=judge_source,
            substrate_model_id=substrate_model_id,
            substrate_source=substrate_source,
            materials=materials,
            device=device,
            local_files_only=True,
            model=judge_model,
            tokenizer=judge_tokenizer,
            judge_family=judge_family,
            substrate_family=substrate_family,
        )
        for persona in personas:
            row = observations[(profile, persona, probe_id)]
            correct += int(
                judge.match(
                    response_text=str(row["response"]),
                    candidate_user_ids=personas,
                )
                == persona
            )
            total += 1
    return correct, total


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--judge-model-id", default="BAAI/bge-m3")
    parser.add_argument("--judge-source", default="")
    parser.add_argument("--judge-device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--relationship-projector",
        default="",
        help="Optional model-derived Relationship projector artifact.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gain-probe-limit", type=int, default=2)
    parser.add_argument("--irrelevant-probe-limit", type=int, default=2)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/state_kv/pilots/relationship-latent/"
            "verdict_relationship_carrier.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not 1 <= args.gain_probe_limit <= len(GAIN_PROBES):
        raise ValueError(
            f"--gain-probe-limit must be within [1, {len(GAIN_PROBES)}]"
        )
    if not 1 <= args.irrelevant_probe_limit <= len(IRRELEVANT_PROBES):
        raise ValueError(
            "--irrelevant-probe-limit must be within "
            f"[1, {len(IRRELEVANT_PROBES)}]"
        )
    gain_probes = GAIN_PROBES[: args.gain_probe_limit]
    irrelevant_probes = IRRELEVANT_PROBES[: args.irrelevant_probe_limit]
    probes = (*gain_probes, *irrelevant_probes)
    substrate_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    judge_root = _resolve_local_weights(
        model_id=args.judge_model_id,
        model_source=args.judge_source,
        allow_download=args.allow_download,
    )
    relationship_projector = (
        RelationshipConditioningProjectorArtifact.from_json(
            (REPO_ROOT / args.relationship_projector)
            .resolve()
            .read_text(encoding="utf-8")
        )
        if args.relationship_projector
        else None
    )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=args.model_id,
        pretrained_source=str(substrate_root),
        device=args.device,
        local_files_only=True,
        runtime_origin="hf-local",
        relationship_conditioning_projector=relationship_projector,
    )
    set_semantic_embedding_backend(
        SubstrateTextEncoderBackend(runtime),
        owner=runtime.model_id,
    )
    try:
        rows = asyncio.run(
            _collect(
                runtime=runtime,
                max_new_tokens=args.max_new_tokens,
                probes=probes,
            )
        )
    finally:
        reset_semantic_embedding_backend()

    observations = {
        (str(row["profile"]), str(row["persona"]), str(row["probe"])): row
        for row in rows
    }
    matched_fingerprints = sum(
        observations[(PROFILE_TEXT, persona, probe_id)][
            "relationship_source_fingerprint"
        ]
        == observations[(PROFILE_LATENT, persona, probe_id)][
            "relationship_source_fingerprint"
        ]
        for persona in sorted(PERSONAS)
        for probe_id, _ in probes
    )
    matched_total = len(PERSONAS) * len(probes)
    latent_applied = sum(
        str(
            observations[(PROFILE_LATENT, persona, probe_id)][
                "relationship_carrier_tag"
            ]
        ).startswith("relationship_conditioning=residual:")
        for persona in sorted(PERSONAS)
        for probe_id, _ in probes
    )
    latent_prompt_identity = sum(
        len(
            {
                observations[(PROFILE_LATENT, persona, probe_id)][
                    "prompt_fingerprint"
                ]
                for persona in sorted(PERSONAS)
            }
        )
        == 1
        for probe_id, _ in probes
    )

    from transformers import AutoModel, AutoTokenizer

    judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_root, local_files_only=True
    )
    judge_model = AutoModel.from_pretrained(
        judge_root, local_files_only=True
    )
    judge_family = resolve_model_family(
        model_id=str(judge_root), local_files_only=True
    )
    substrate_family = resolve_model_family(
        model_id=str(substrate_root), local_files_only=True
    )
    gain_probe_ids = tuple(probe_id for probe_id, _ in gain_probes)
    accuracies = {}
    for profile in PROFILES:
        correct, total = _accuracy(
            profile=profile,
            probe_ids=gain_probe_ids,
            observations=observations,
            judge_model_id=args.judge_model_id,
            judge_source=str(judge_root),
            substrate_model_id=args.model_id,
            substrate_source=str(substrate_root),
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_family=judge_family,
            substrate_family=substrate_family,
            device=args.judge_device,
        )
        accuracies[profile] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }

    persona_divergence = {}
    for profile in PROFILES:
        divergent = sum(
            len(
                {
                    observations[(profile, persona, probe_id)]["response"]
                    for persona in sorted(PERSONAS)
                }
            )
            > 1
            for probe_id, _ in gain_probes
        )
        persona_divergence[profile] = {
            "divergent_probes": divergent,
            "probe_count": len(gain_probes),
            "rate": divergent / len(gain_probes),
        }

    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "state-kv-relationship-carrier-pilot.v1",
        "projector": {
            "version": runtime.relationship_conditioning_projector_version,
            "artifact_id": runtime.relationship_conditioning_projector_id,
            "training_mode": (
                runtime.relationship_conditioning_projector_training_mode
            ),
        },
        "substrate": _fingerprint_weights(
            model_id=args.model_id,
            weights_root=substrate_root,
        ),
        "judge": _fingerprint_weights(
            model_id=args.judge_model_id,
            weights_root=judge_root,
        ),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
        },
        "profiles": list(PROFILES),
        "matched_source_fingerprints": {
            "count": matched_fingerprints,
            "total": matched_total,
            "passed": matched_fingerprints == matched_total,
        },
        "latent_attestation": {
            "applied_count": latent_applied,
            "total": matched_total,
            "passed": latent_applied == matched_total,
        },
        "latent_prompt_identity": {
            "identical_probe_count": latent_prompt_identity,
            "probe_count": len(probes),
            "passed": latent_prompt_identity == len(probes),
        },
        "relationship_match": accuracies,
        "persona_output_divergence": persona_divergence,
        "turns": list(rows),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"matched_source_fingerprints = {matched_fingerprints}/{matched_total}")
    print(f"latent_attestation = {latent_applied}/{matched_total}")
    print(
        "latent_prompt_identity = "
        f"{latent_prompt_identity}/{len(probes)}"
    )
    for profile, metric in accuracies.items():
        print(f"relationship_match[{profile}] = {metric['accuracy']:.3f}")
    print(f"output = {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
