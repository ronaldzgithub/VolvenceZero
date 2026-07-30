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
from volvence_zero.semantic_state import (  # noqa: E402
    ExternalSemanticEventBatch,
    GenericSemanticEvent,
    SemanticProposalOperation,
)
from volvence_zero.state_kv_bank_gain_gate import (  # noqa: E402
    BANK_GAIN_PROFILE_LABELS,
    BankPersonaContrast,
    IrrelevantBankControlSample,
    NonBankPersonaControlSample,
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

PERSONAS: dict[str, tuple[GenericSemanticEvent, ...]] = {
    "repair": (
        GenericSemanticEvent(
            event_id="bank-gain:repair:relationship:rupture",
            target_slot="relationship_state",
            operation=SemanticProposalOperation.BLOCK,
            summary="Reviewed unresolved relational rupture.",
            detail="Trust is fragile and repair must precede forward motion.",
            confidence=0.45,
            evidence="human-review:bank-gain-repair-relationship-1",
            control_signal=0.95,
        ),
        GenericSemanticEvent(
            event_id="bank-gain:repair:relationship:tension",
            target_slot="relationship_state",
            operation=SemanticProposalOperation.BLOCK,
            summary="Reviewed unresolved tension.",
            detail="The dyad requires acknowledgement and stabilization.",
            confidence=0.55,
            evidence="human-review:bank-gain-repair-relationship-2",
            control_signal=0.85,
        ),
        GenericSemanticEvent(
            event_id="bank-gain:repair:boundary:reversible",
            target_slot="boundary_consent",
            operation=SemanticProposalOperation.BLOCK,
            summary="Reviewed boundary requires reversibility.",
            detail="Do not advance beyond a reversible next step.",
            confidence=0.50,
            evidence="human-review:bank-gain-repair-boundary-1",
            control_signal=0.90,
        ),
    ),
    "steady": (
        GenericSemanticEvent(
            event_id="bank-gain:steady:relationship:trust",
            target_slot="relationship_state",
            operation=SemanticProposalOperation.OBSERVE,
            summary="Reviewed stable relational trust.",
            detail="Prior coordination helped and the direction is trusted.",
            confidence=0.95,
            evidence="human-review:bank-gain-steady-relationship-1",
            control_signal=0.05,
        ),
        GenericSemanticEvent(
            event_id="bank-gain:steady:relationship:continuity",
            target_slot="relationship_state",
            operation=SemanticProposalOperation.COMPLETE,
            summary="Reviewed successful continuity checkpoint.",
            detail="The prior exchange completed without unresolved tension.",
            confidence=0.92,
            evidence="human-review:bank-gain-steady-relationship-2",
            control_signal=0.08,
        ),
        GenericSemanticEvent(
            event_id="bank-gain:steady:boundary:granted",
            target_slot="boundary_consent",
            operation=SemanticProposalOperation.OBSERVE,
            summary="Reviewed consent for the bounded next step.",
            detail="Proceed within the already agreed scope.",
            confidence=0.95,
            evidence="human-review:bank-gain-steady-boundary-1",
            control_signal=0.05,
        ),
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


def _semantic_event_as_json(
    event: GenericSemanticEvent,
) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "target_slot": event.target_slot,
        "operation": event.operation.value,
        "summary": event.summary,
        "detail": event.detail,
        "confidence": event.confidence,
        "evidence": event.evidence,
        "control_signal": event.control_signal,
        "requires_confirmation": event.requires_confirmation,
    }


def _contrast_as_json(
    contrast: BankPersonaContrast,
) -> dict[str, object]:
    return {
        "bank_type": contrast.bank_type,
        "probe_count": contrast.probe_count,
        "material_contrast_count": contrast.material_contrast_count,
        "fingerprint_contrast_count": contrast.fingerprint_contrast_count,
        "passed": contrast.passed,
    }


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


def _bank_fingerprints(result: Any) -> dict[str, str]:
    trace = result.dialogue_trace
    if trace is None or trace.conditioning_lineage is None:
        raise RuntimeError("dual-bank probe produced no conditioning lineage")
    fingerprints = dict(trace.conditioning_lineage.bank_fingerprints)
    if not all(bank in fingerprints for bank in ("personal", "relationship")):
        raise RuntimeError(
            "dual-bank probe lineage did not publish both bank fingerprints"
        )
    return fingerprints


async def _run_turn(
    *,
    profile_label: str,
    persona_id: str,
    probe_id: str,
    user_input: str,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
    semantic_state_snapshot: OwnerPersistenceSnapshot,
    use_model_response: bool = True,
) -> dict[str, object]:
    case = replace(
        DEFAULT_DIALOGUE_PROOF_CASES[0],
        case_id=f"state-kv-bank-gain:{persona_id}:{probe_id}",
    )
    runner = build_standard_dialogue_runner(
        profile_label=profile_label,
        case=case,
        residual_runtime=runtime,
    )
    runner._semantic_state_store.hydrate_from_persistence(
        semantic_state_snapshot
    )
    if use_model_response:
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
        payload["bank_fingerprints"] = _bank_fingerprints(result)
        payload["router_scores"] = dict(
            result.dialogue_trace.conditioning_lineage.shadow_router_scores
        )
    return payload


async def _build_persona_snapshots(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
) -> dict[str, OwnerPersistenceSnapshot]:
    snapshots = {}
    for persona_id in sorted(PERSONAS):
        case = replace(
            DEFAULT_DIALOGUE_PROOF_CASES[0],
            case_id=f"state-kv-bank-gain:warmup:{persona_id}",
        )
        runner = build_standard_dialogue_runner(
            profile_label="state-kv-bank-none",
            case=case,
            residual_runtime=runtime,
        )
        runner.enqueue_semantic_events(
            ExternalSemanticEventBatch(
                events=PERSONAS[persona_id],
                source="human-review",
                description=(
                    "Reviewed typed State KV bank-gain persona packet."
                ),
            )
        )
        await runner.run_turn(
            "Apply the reviewed typed relationship state for this case."
        )
        snapshots[persona_id] = (
            runner._semantic_state_store.export_persistence_snapshot()
        )
    return snapshots


def _build_persona_contrasts(
    *,
    rows: tuple[dict[str, object], ...],
    probe_ids: tuple[str, ...],
) -> tuple[BankPersonaContrast, ...]:
    observations = {
        (str(row["persona"]), str(row["probe"])): row
        for row in rows
        if row["profile"] == "state-kv-bank-dual"
    }
    contrasts = []
    persona_ids = tuple(sorted(PERSONAS))
    for bank_type in ("personal", "relationship"):
        material_contrast_count = 0
        fingerprint_contrast_count = 0
        for probe_id in probe_ids:
            probe_rows = [
                observations[(persona_id, probe_id)]
                for persona_id in persona_ids
            ]
            materials = []
            fingerprints = []
            for row in probe_rows:
                rendered = row.get("materials")
                lineage_fingerprints = row.get("bank_fingerprints")
                if not isinstance(rendered, dict):
                    raise TypeError(
                        "dual observation materials must be an object"
                    )
                if not isinstance(lineage_fingerprints, dict):
                    raise TypeError(
                        "dual observation bank_fingerprints must be an object"
                    )
                materials.append(str(rendered[bank_type]))
                fingerprints.append(str(lineage_fingerprints[bank_type]))
            if len(set(materials)) == len(persona_ids):
                material_contrast_count += 1
            if len(set(fingerprints)) == len(persona_ids):
                fingerprint_contrast_count += 1
        contrasts.append(
            BankPersonaContrast(
                bank_type=bank_type,
                probe_count=len(probe_ids),
                material_contrast_count=material_contrast_count,
                fingerprint_contrast_count=fingerprint_contrast_count,
            )
        )
    return tuple(contrasts)


async def _collect_observations(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
    gain_probes: tuple[tuple[str, str], ...],
    irrelevant_probes: tuple[tuple[str, str], ...],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    observations = []
    persona_snapshots = await _build_persona_snapshots(runtime=runtime)
    preflight, _ = await _run_persona_preflight(
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        persona_snapshots=persona_snapshots,
    )
    for profile_label in BANK_GAIN_PROFILE_LABELS:
        for persona_id in sorted(PERSONAS):
            for probe_id, user_input in (*gain_probes, *irrelevant_probes):
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
            print(
                f"observations[{profile_label}:{persona_id}] = "
                f"{len(gain_probes) + len(irrelevant_probes)}",
                flush=True,
            )
    return tuple(observations), preflight


async def _run_persona_preflight(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
    persona_snapshots: dict[str, OwnerPersistenceSnapshot],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[BankPersonaContrast, ...],
]:
    preflight = tuple(
        [
            await _run_turn(
                profile_label="state-kv-bank-dual",
                persona_id=persona_id,
                probe_id="preflight",
                user_input="What should I protect before I act?",
                runtime=runtime,
                max_new_tokens=max_new_tokens,
                semantic_state_snapshot=persona_snapshots[persona_id],
                use_model_response=False,
            )
            for persona_id in sorted(PERSONAS)
        ]
    )
    preflight_contrasts = _build_persona_contrasts(
        rows=preflight,
        probe_ids=("preflight",),
    )
    failed_preflight = tuple(
        contrast.bank_type
        for contrast in preflight_contrasts
        if not contrast.passed
    )
    if failed_preflight:
        raise RuntimeError(
            "bank-gain persona-state preflight collapsed for "
            f"{failed_preflight!r}; refusing to run causal gain statistics"
        )
    return preflight, preflight_contrasts


async def _run_preflight_only(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
    max_new_tokens: int,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[BankPersonaContrast, ...],
]:
    persona_snapshots = await _build_persona_snapshots(runtime=runtime)
    return await _run_persona_preflight(
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        persona_snapshots=persona_snapshots,
    )


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
    gain_probes: tuple[tuple[str, str], ...],
    irrelevant_probes: tuple[tuple[str, str], ...],
) -> tuple[
    tuple[PairedBankGainSample, ...],
    tuple[IrrelevantBankControlSample, ...],
    tuple[NonBankPersonaControlSample, ...],
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
    non_bank_controls = []
    for bank_type, ablated_profile in (
        ("personal", "state-kv-bank-relationship-only"),
        ("relationship", "state-kv-bank-personal-only"),
    ):
        for probe_id, _ in gain_probes:
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
                without_bank = observations[
                    ("state-kv-bank-none", persona_id, probe_id)
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
                non_bank_controls.append(
                    NonBankPersonaControlSample(
                        probe_id=f"{persona_id}:{probe_id}",
                        bank_type=bank_type,
                        match_correct=(
                            judge.match(
                                response_text=str(without_bank["response"]),
                                candidate_user_ids=candidates,
                            )
                            == persona_id
                        ),
                    )
                )
    irrelevant = []
    for probe_id, _ in irrelevant_probes:
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
    return (
        tuple(paired),
        tuple(irrelevant),
        tuple(non_bank_controls),
        judge_family,
    )


def _load_reused_observations(
    *,
    path: Path,
    substrate_fingerprint: dict[str, object],
    judge_fingerprint: dict[str, object],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "state-kv-bank-gain-observations.v2":
        raise ValueError(
            "reused observations must use "
            "state-kv-bank-gain-observations.v2"
        )
    for label, expected in (
        ("substrate", substrate_fingerprint),
        ("judge", judge_fingerprint),
    ):
        observed = payload.get(label)
        if not isinstance(observed, dict):
            raise TypeError(f"reused observations {label} must be an object")
        for field in ("model_id", "weights_sha256"):
            if observed.get(field) != expected[field]:
                raise ValueError(
                    f"reused observations {label}.{field} does not match "
                    "the resolved frozen weights"
                )
    if tuple(payload.get("profiles", ())) != BANK_GAIN_PROFILE_LABELS:
        raise ValueError("reused observations profile matrix does not match")
    turns = payload.get("turns")
    preflight = payload.get("preflight_turns")
    if not isinstance(turns, list) or not all(
        isinstance(row, dict) for row in turns
    ):
        raise TypeError("reused observations turns must be a list of objects")
    if not isinstance(preflight, list) or not all(
        isinstance(row, dict) for row in preflight
    ):
        raise TypeError(
            "reused observations preflight_turns must be a list of objects"
        )
    return tuple(turns), tuple(preflight)


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
    parser.add_argument(
        "--gain-probe-limit",
        type=int,
        default=len(GAIN_PROBES),
    )
    parser.add_argument(
        "--irrelevant-probe-limit",
        type=int,
        default=len(IRRELEVANT_PROBES),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate typed persona bank contrast without running 64 turns",
    )
    parser.add_argument(
        "--reuse-observations",
        action="store_true",
        help="re-adjudicate the existing frozen 64-turn observation artifact",
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
    if args.preflight_only and args.reuse_observations:
        raise ValueError(
            "--preflight-only and --reuse-observations are mutually exclusive"
        )
    output = (REPO_ROOT / args.output).resolve()
    observation_path = output.with_name("observations_bank_gain.json")
    gain_probes = GAIN_PROBES[: args.gain_probe_limit]
    irrelevant_probes = IRRELEVANT_PROBES[: args.irrelevant_probe_limit]
    weights_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    substrate_fingerprint_payload = _fingerprint_weights(
        model_id=args.model_id,
        weights_root=weights_root,
    )
    if args.preflight_only:
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
            preflight_turns, preflight_contrasts = asyncio.run(
                _run_preflight_only(
                    runtime=runtime,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        finally:
            reset_semantic_embedding_backend()
        print(
            json.dumps(
                {
                    "schema_version": "state-kv-bank-gain-preflight.v1",
                    "persona_contrasts": [
                        _contrast_as_json(contrast)
                        for contrast in preflight_contrasts
                    ],
                    "turns": list(preflight_turns),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    judge_root = _resolve_local_weights(
        model_id=args.judge_model_id,
        model_source=args.judge_source,
        allow_download=args.allow_download,
    )
    judge_fingerprint_payload = _fingerprint_weights(
        model_id=args.judge_model_id,
        weights_root=judge_root,
    )
    if args.reuse_observations:
        raw_observations, preflight_observations = (
            _load_reused_observations(
                path=observation_path,
                substrate_fingerprint=substrate_fingerprint_payload,
                judge_fingerprint=judge_fingerprint_payload,
            )
        )
        semantic_backend = (
            "reused:state-kv-bank-gain-observations.v2"
        )
    else:
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
            raw_observations, preflight_observations = asyncio.run(
                _collect_observations(
                    runtime=runtime,
                    max_new_tokens=args.max_new_tokens,
                    gain_probes=gain_probes,
                    irrelevant_probes=irrelevant_probes,
                )
            )
            semantic_backend = ":".join(
                str(value) for value in semantic_embedding_backend_status()
            )
        finally:
            reset_semantic_embedding_backend()
    persona_contrasts = _build_persona_contrasts(
        rows=raw_observations,
        probe_ids=tuple(probe_id for probe_id, _ in gain_probes),
    )
    (
        paired_samples,
        irrelevant_controls,
        non_bank_persona_controls,
        judge_family,
    ) = _build_samples(
        raw_observations=raw_observations,
        judge_model_id=args.judge_model_id,
        judge_source=str(judge_root),
        substrate_model_id=args.model_id,
        substrate_source=str(weights_root),
        judge_device=args.judge_device,
        gain_probes=gain_probes,
        irrelevant_probes=irrelevant_probes,
    )
    if not args.reuse_observations:
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_path.write_text(
            json.dumps(
                {
                    "schema_version": "state-kv-bank-gain-observations.v2",
                    "substrate": substrate_fingerprint_payload,
                    "judge": judge_fingerprint_payload,
                    "profiles": list(BANK_GAIN_PROFILE_LABELS),
                    "generation": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0.0,
                    },
                    "personas": {
                        key: [
                            _semantic_event_as_json(event)
                            for event in value
                        ]
                        for key, value in PERSONAS.items()
                    },
                    "preflight_turns": list(preflight_observations),
                    "persona_contrasts": [
                        _contrast_as_json(contrast)
                        for contrast in persona_contrasts
                    ],
                    "gain_probes": list(gain_probes),
                    "irrelevant_probes": list(irrelevant_probes),
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
        non_bank_persona_controls=non_bank_persona_controls,
        persona_contrasts=persona_contrasts,
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
