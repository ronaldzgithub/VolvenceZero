#!/usr/bin/env python3
"""Run State-KV carrier-identification evidence on smoke or frozen Qwen.

``smoke`` keeps the original deterministic fake lane. ``p1`` resolves one
local weight snapshot, loads one frozen Transformers runtime, and reuses that
same runtime object for all four arms. P1 also gives both personas the exact
same response assembly so carrier C5 is closed by construction; their only
per-user input is the personal-conditioning snapshot. ``p3`` adds the
Prefix-KV arm on the original hand-checked pair. ``p2`` runs the same carrier
test on held-out persona/probe pairs.

The runner writes three independently inspectable artifacts next to each other:

* ``verdict_identification.json`` -- the four computed claim states;
* ``transcript_identification.json`` -- response text and emitted audit tags;
* ``substrate_fingerprint.json`` -- the content hash of the loaded weight files.

No cross-family judge is invented here. Without an explicitly wired judge,
claims 3/4 and the overall verdict remain ``insufficient_data`` even on real
Qwen. That is the honest P1 substrate lane, ready for a later blind-judge pass.

Usage:
    python scripts/run_state_kv_identification.py --lane smoke
    python scripts/run_state_kv_identification.py \
        --lane p1 --device cpu \
        --model-id Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.agent.response import (  # noqa: E402
    LLMResponseSynthesizer,
    ResponseContext,
)
from volvence_zero.application.runtime import (  # noqa: E402
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (  # noqa: E402
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.personal_conditioning_rendering import (  # noqa: E402
    render_personal_conditioning_statement,
)
from volvence_zero.state_kv_blind_judge import (  # noqa: E402
    JudgeMaterial,
    JudgeMaterialKind,
    LocalEmbeddingBlindJudge,
    LocalTransformersBlindJudge,
)
from volvence_zero.state_kv_identification import (  # noqa: E402
    DEFAULT_CANDIDATE_ARM_LABEL,
    IDENTIFICATION_ARM_LABELS,
    PREFIX_ARM_LABEL,
    PREFIX_IDENTIFICATION_ARM_LABELS,
    ProbeCase,
    SubstrateEvidenceKind,
    run_identification_smoke,
)
from volvence_zero.substrate import (  # noqa: E402
    PersonalConditioningProjectorArtifact,
    PrefixKVArtifact,
    TransformersOpenWeightResidualRuntime,
)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth"})

# Probe sentences: identical for both users, so any per-user difference in the
# response must come from state, not from what was asked.
PROBE_SENTENCES: tuple[tuple[str, str], ...] = (
    ("p0", "我又搞砸了"),
    ("p1", "今天还是没睡好"),
    ("p2", "你觉得我该继续吗"),
    ("p3", "我现在有点撑不住了"),
    ("p4", "能不能帮我定一下下一步"),
    ("p5", "我是不是该先停一下"),
)

P2_PROBE_SENTENCES: tuple[tuple[str, str], ...] = (
    ("h0", "我刚刚差点直接放弃"),
    ("h1", "你帮我判断一下现在该不该推进"),
    ("h2", "我怕自己又做错选择"),
    ("h3", "下一步能不能简单一点"),
    ("h4", "我需要有人帮我把风险摆清楚"),
    ("h5", "这件事是不是可以今天就定下来"),
    ("h6", "我现在不确定该相信哪个判断"),
    ("h7", "帮我把这件事收个口"),
    ("h8", "我有点担心这会越界"),
    ("h9", "如果先试一小步可以吗"),
    ("h10", "你觉得我该保护哪个边界"),
    ("h11", "我想今天把计划定下来"),
    ("h12", "这个决定还能不能撤回"),
    ("h13", "帮我把可验证的一步写出来"),
    ("h14", "我现在需要先停还是继续"),
    ("h15", "把选择标准列给我"),
)

# Two coherent, counterfactual relationship states. Coordinates follow
# PERSONAL_CONDITIONING_VECTOR_LABELS exactly. They are not "all high" versus
# "all low": risk/readiness dimensions have opposite valence, so moving every
# coordinate together would describe an internally contradictory person.
PERSONAS: tuple[tuple[str, tuple[float, ...], str, str], ...] = (
    (
        "persona-a",
        (
            0.28,  # user_stability
            0.82,  # user_overwhelm
            0.30,  # user_control
            0.32,  # relationship_trust
            0.68,  # relationship_continuity
            0.90,  # relationship_repair_need
            0.88,  # relationship_emotional_load
            0.76,  # relationship_attunement_gap
            0.36,  # goal_alignment
            0.78,  # goal_value_conflict
            0.22,  # goal_decision_readiness
            0.90,  # goal_reversibility_need
            0.76,  # boundary_compliance
            0.92,  # boundary_autonomy_risk
            0.46,  # boundary_consent_clarity
            0.86,  # boundary_overreach_risk
        ),
        "Carry forward continuity from prior context: her cat died last week.",
        "continuum-support-first",
    ),
    (
        "persona-b",
        (
            0.86,
            0.20,
            0.84,
            0.90,
            0.92,
            0.08,
            0.24,
            0.10,
            0.90,
            0.14,
            0.92,
            0.24,
            0.96,
            0.08,
            0.94,
            0.06,
        ),
        "Carry forward continuity from prior context: he starts a new job Monday.",
        "continuum-structure-first",
    ),
)

P2_PERSONA_PAIRS: dict[
    str, tuple[tuple[str, tuple[float, ...], str, str], ...]
] = {
    "repair-vs-execute": (
        (
            "heldout-repair",
            (
                0.18,  # user_stability
                0.88,  # user_overwhelm
                0.24,  # user_control
                0.38,  # relationship_trust
                0.58,  # relationship_continuity
                0.86,  # relationship_repair_need
                0.84,  # relationship_emotional_load
                0.72,  # relationship_attunement_gap
                0.42,  # goal_alignment
                0.72,  # goal_value_conflict
                0.28,  # goal_decision_readiness
                0.84,  # goal_reversibility_need
                0.70,  # boundary_compliance
                0.88,  # boundary_autonomy_risk
                0.52,  # boundary_consent_clarity
                0.82,  # boundary_overreach_risk
            ),
            "Carry forward continuity from prior context: she postponed a hard conversation twice.",
            "continuum-repair-holdout",
        ),
        (
            "heldout-execute",
            (
                0.90,
                0.16,
                0.88,
                0.86,
                0.88,
                0.12,
                0.20,
                0.16,
                0.86,
                0.16,
                0.88,
                0.18,
                0.92,
                0.12,
                0.90,
                0.10,
            ),
            "Carry forward continuity from prior context: he already got stakeholder approval.",
            "continuum-execute-holdout",
        ),
    ),
    "boundary-vs-commit": (
        (
            "heldout-boundary",
            (
                0.36,
                0.70,
                0.30,
                0.54,
                0.64,
                0.64,
                0.70,
                0.58,
                0.50,
                0.68,
                0.34,
                0.88,
                0.48,
                0.94,
                0.36,
                0.92,
            ),
            "Carry forward continuity from prior context: she was pressured into saying yes before.",
            "continuum-boundary-holdout",
        ),
        (
            "heldout-commit",
            (
                0.82,
                0.24,
                0.82,
                0.78,
                0.82,
                0.18,
                0.28,
                0.20,
                0.88,
                0.22,
                0.86,
                0.20,
                0.88,
                0.18,
                0.86,
                0.14,
            ),
            "Carry forward continuity from prior context: he chose a reversible trial plan.",
            "continuum-commit-holdout",
        ),
    ),
}


class DeterministicFakeSubstrate:
    """Trace-only fake: derives text from the prompt, never injects.

    Mirrors the synthetic runtime's contract
    (``personal_conditioning_applied=False``) so the smoke run exercises the
    honest path -- claim 2 must come back ``insufficient_data`` rather than
    being satisfied by a fake that pretends to inject.
    """

    model_id = "deterministic-fake-substrate"

    def __init__(self, *, applies_conditioning: bool = False) -> None:
        self._applies = applies_conditioning

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.model_id}:applies={self._applies}".encode("utf-8")
        ).hexdigest()[:16]

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        conditioning = kwargs.get("personal_conditioning")
        applied = self._applies and conditioning is not None
        parts = [
            str(kwargs.get("system_context", "")),
            str(kwargs.get("prompt", "")),
        ]
        if applied:
            parts.append(conditioning.source_fingerprint)
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]
        return SimpleNamespace(
            text=f"reply-{digest}",
            token_count=len(parts),
            personal_conditioning_applied=applied,
        )


def _assembly(*, residue: str, ordering_driver: str) -> ResponseAssemblySnapshot:
    return ResponseAssemblySnapshot(
        regime_id="steady",
        regime_name="Steady",
        abstract_action=None,
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="high-level-only",
        citation_mode="none",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary=residue,
        prompt_residue_ratio=0.4,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="carrier-identification probe assembly",
        ordering_driver=ordering_driver,
    )


def _conditioning(
    *,
    user_id: str,
    state_vector: tuple[float, ...],
) -> PersonalConditioningSnapshot:
    if len(state_vector) != len(PERSONAL_CONDITIONING_VECTOR_LABELS):
        raise ValueError(
            f"probe persona {user_id!r} has {len(state_vector)} coordinates; "
            f"expected {len(PERSONAL_CONDITIONING_VECTOR_LABELS)}"
        )
    statement = render_personal_conditioning_statement(
        state_vector=state_vector,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        confidence=0.72,
        is_cold_start=False,
    )
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=state_vector,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 3), ("relationship_state", 2)),
        source_fingerprint=hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16],
        confidence=0.72,
        is_cold_start=False,
        description=f"probe state for {user_id}",
        rendered_statement=statement,
    )


def build_probe_cases(
    *,
    strict_carriers: bool = False,
    personas: tuple[tuple[str, tuple[float, ...], str, str], ...] = PERSONAS,
    probe_sentences: tuple[tuple[str, str], ...] = PROBE_SENTENCES,
) -> tuple[ProbeCase, ...]:
    """Build probe cases for smoke or strict real-substrate evidence.

    Smoke deliberately keeps divergent assemblies to prove
    ``prompt_state_delivery="suppressed"`` closes a genuine leak opportunity.
    P1 uses one shared assembly for both personas: otherwise
    ``GenerationConstraints`` would carry per-user response shaping through C5
    even when the prompt bytes match.
    """

    cases: list[ProbeCase] = []
    shared_assembly = (
        _assembly(residue="", ordering_driver="playbook-only")
        if strict_carriers
        else None
    )
    if len(personas) != 2:
        raise ValueError(
            "identification matching is two-alternative; build_probe_cases "
            f"requires exactly two personas, got {len(personas)}"
        )
    if not probe_sentences:
        raise ValueError("identification matching requires at least one probe")
    for user_id, state_vector, residue, ordering_driver in personas:
        assembly = shared_assembly or _assembly(
            residue=residue,
            ordering_driver=ordering_driver,
        )
        conditioning = _conditioning(
            user_id=user_id,
            state_vector=state_vector,
        )
        for probe_id, sentence in probe_sentences:
            cases.append(
                ProbeCase(
                    user_id=user_id,
                    probe_id=probe_id,
                    user_input=sentence,
                    conditioning=conditioning,
                    assembly=assembly,
                )
            )
    return tuple(cases)


def build_p2_probe_cases(
    *,
    pair_id: str,
    strict_carriers: bool = True,
) -> tuple[ProbeCase, ...]:
    try:
        personas = P2_PERSONA_PAIRS[pair_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown P2 persona pair {pair_id!r}; expected one of "
            f"{sorted(P2_PERSONA_PAIRS)}"
        ) from exc
    return build_probe_cases(
        strict_carriers=strict_carriers,
        personas=personas,
        probe_sentences=P2_PROBE_SENTENCES,
    )


def _judge_materials_from_cases(cases: tuple[ProbeCase, ...]) -> tuple[JudgeMaterial, ...]:
    """Build blind-judge materials from the same owner-rendered readout.

    P3 does not carry real session-history summaries; using the owner-rendered
    state statement keeps the judge material tied to the same typed readout as
    the latent carrier and prevents a second, ad hoc persona description from
    becoming the evidence owner.
    """

    summaries: dict[str, str] = {}
    for case in cases:
        summary = case.conditioning.rendered_statement.strip()
        if not summary:
            raise ValueError(
                f"case {case.user_id}/{case.probe_id} has an empty rendered "
                "state statement; blind matching would have no candidate "
                "material."
            )
        existing = summaries.setdefault(case.user_id, summary)
        if existing != summary:
            raise ValueError(
                f"user {case.user_id!r} has inconsistent rendered state "
                "statements across probes; blind matching material must be "
                "per-user stable."
            )
    return tuple(
        JudgeMaterial(
            user_id=user_id,
            summary=summary,
            material_kind=JudgeMaterialKind.RENDERED_STATE,
        )
        for user_id, summary in sorted(summaries.items())
    )


class RecordingSynthesizer:
    """Record the actual responses while preserving the synthesizer contract."""

    def __init__(self, inner: LLMResponseSynthesizer) -> None:
        self._inner = inner
        self.responses: list[object] = []

    def synthesize(self, **kwargs: Any) -> object:
        response = self._inner.synthesize(**kwargs)
        self.responses.append(response)
        return response


def _resolve_local_weights(
    *,
    model_id: str,
    model_source: str,
    allow_download: bool,
) -> Path:
    if model_source:
        source = Path(model_source).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(
                f"--model-source is not a model directory: {source}"
            )
        return source
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "P1 requires huggingface_hub to resolve the frozen local snapshot."
        ) from exc
    resolved = snapshot_download(
        repo_id=model_id,
        local_files_only=not allow_download,
    )
    return Path(resolved).resolve()


def _fingerprint_weights(*, model_id: str, weights_root: Path) -> dict[str, object]:
    files = sorted(
        path
        for path in weights_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in WEIGHT_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"no model weight files found under {weights_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(weights_root).as_posix()
        encoded_relative = relative.encode("utf-8")
        digest.update(len(encoded_relative).to_bytes(4, "big"))
        digest.update(encoded_relative)
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return {
        "schema_version": "state-kv-substrate-fingerprint.v1",
        "model_id": model_id,
        "weights_root": str(weights_root),
        "weights_sha256": digest.hexdigest(),
        "weight_file_count": len(files),
        "weight_files": [path.relative_to(weights_root).as_posix() for path in files],
    }


def _artifact_paths(*, output: Path) -> tuple[Path, Path, Path]:
    return (
        output,
        output.with_name("transcript_identification.json"),
        output.with_name("substrate_fingerprint.json"),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _transcript_payload(
    *,
    lane: str,
    cases: tuple[ProbeCase, ...],
    responses: list[object],
    projector_id: str,
    projector_training_mode: str,
    arm_labels: tuple[str, ...],
    prefix_artifact_id: str = "",
) -> dict[str, object]:
    expected_count = len(arm_labels) * len(cases)
    if len(responses) != expected_count:
        raise RuntimeError(
            "identification transcript count mismatch: "
            f"expected {expected_count}, got {len(responses)}"
        )
    turns: list[dict[str, object]] = []
    response_index = 0
    for arm_label in arm_labels:
        for case in cases:
            response = responses[response_index]
            response_index += 1
            turns.append(
                {
                    "arm": arm_label,
                    "user": case.user_id,
                    "probe": case.probe_id,
                    "input": case.user_input,
                    "response": str(getattr(response, "text", "")),
                    "rationale_tags": list(
                        getattr(response, "rationale_tags", ())
                    ),
                }
            )
    return {
        "schema_version": "state-kv-identification-transcript.v1",
        "lane": lane,
        "personal_conditioning_projector_id": projector_id,
        "personal_conditioning_projector_training_mode": (
            projector_training_mode
        ),
        "personal_conditioning_prefix_id": prefix_artifact_id,
        "turns": turns,
    }


def _base_context() -> ResponseContext:
    return ResponseContext(
        regime_id="steady",
        regime_name="Steady",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="none",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        choices=("smoke", "p1", "p2", "p3"),
        default="smoke",
        help=(
            "synthetic wiring smoke, one frozen local Qwen runtime (p1), or "
            "the same runtime plus the prefix-KV arm G on held-out material "
            "(p2) or the hand-checked pair (p3)"
        ),
    )
    parser.add_argument(
        "--p2-pair",
        choices=tuple(P2_PERSONA_PAIRS),
        default=next(iter(P2_PERSONA_PAIRS)),
        help="held-out persona pair for the P2 lane",
    )
    parser.add_argument(
        "--output",
        default="",
        help="where to write verdict_identification.json",
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help=(
            "let the fake substrate report injection (exercises claim 2 on a "
            "fake; the verdict is still capped at insufficient_data)"
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-source",
        default="",
        help="explicit local HF snapshot directory; otherwise resolve local cache",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--personal-conditioning-scale",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--projector-artifact",
        default="",
        help=(
            "optional learned personal-conditioning projector JSON; P1 only. "
            "Omit to roll back to the fixed single-layer basis"
        ),
    )
    parser.add_argument(
        "--prefix-kv-artifact",
        default="",
        help=(
            "learned State-KV prefix generator JSON; required by the P2/P3 "
            "lanes, which add arm G and make it the candidate arm"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face to download a missing P1 model snapshot",
    )
    parser.add_argument(
        "--judge-model-id",
        default="",
        help=(
            "optional local cross-family judge model id; when set, the runner "
            "adds blind matching readouts for claims 3/4"
        ),
    )
    parser.add_argument(
        "--judge-kind",
        choices=("causal-lm", "embedding"),
        default="causal-lm",
        help="local blind judge scoring backend (defaults to causal-lm)",
    )
    parser.add_argument(
        "--judge-source",
        default="",
        help=(
            "explicit local HF snapshot directory for the judge; otherwise "
            "resolve --judge-model-id from the local cache"
        ),
    )
    parser.add_argument(
        "--judge-device",
        default="cpu",
        help="device for the local blind judge (defaults to cpu)",
    )
    parser.add_argument(
        "--judge-bootstrap-seed",
        type=int,
        default=20260726,
        help="seed for the blind-matching bootstrap confidence intervals",
    )
    args = parser.parse_args(argv)

    frozen_lane = args.lane in ("p1", "p2", "p3")
    prefix_lane = args.lane in ("p2", "p3")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if frozen_lane and args.inject:
        parser.error("--inject only applies to the smoke lane")
    if args.lane == "smoke" and args.projector_artifact:
        parser.error("--projector-artifact only applies to the frozen lanes")
    if not prefix_lane and args.prefix_kv_artifact:
        parser.error("--prefix-kv-artifact only applies to the P2/P3 lanes")
    if prefix_lane and not args.prefix_kv_artifact:
        # Arm G without an artifact could only run by falling back to another
        # carrier, which would publish a mislabelled arm.
        parser.error("--prefix-kv-artifact is required by the P2/P3 lanes")
    if frozen_lane and args.temperature != 0.0:
        parser.error(
            "the frozen lanes require --temperature 0 so C5 is deterministic; "
            "matched multi-seed sampling belongs to the blind-judge package"
        )
    if args.judge_source and not args.judge_model_id:
        parser.error("--judge-source requires --judge-model-id")
    if args.judge_model_id and not frozen_lane:
        parser.error("--judge-model-id only applies to frozen lanes")

    if args.lane == "p3":
        run_directory = "p3"
    elif args.lane == "p2":
        run_directory = f"p2-{args.p2_pair}"
    elif args.lane == "p1":
        run_directory = "p1-learned" if args.projector_artifact else "p1"
    else:
        run_directory = ""
    output = Path(args.output).expanduser() if args.output else (
        REPO_ROOT
        / "artifacts"
        / "state_kv"
        / run_directory
        / "verdict_identification.json"
    )
    output = output.resolve()
    verdict_path, transcript_path, fingerprint_path = _artifact_paths(
        output=output
    )

    if args.lane == "smoke":
        runtime: object = DeterministicFakeSubstrate(
            applies_conditioning=args.inject
        )
        substrate_kind = SubstrateEvidenceKind.TRACE_ONLY
        substrate_fingerprint = runtime.fingerprint
        fingerprint_payload: dict[str, object] = {
            "schema_version": "state-kv-substrate-fingerprint.v1",
            "model_id": runtime.model_id,
            "runtime_kind": "trace-only",
            "weights_sha256": "",
            "runtime_fingerprint": runtime.fingerprint,
        }
        cases = build_probe_cases(strict_carriers=False)
        projector_id = "trace-only"
        projector_training_mode = "none"
        prefix_artifact_id = ""
    else:
        weights_root = _resolve_local_weights(
            model_id=args.model_id,
            model_source=args.model_source,
            allow_download=args.allow_download,
        )
        fingerprint_payload = _fingerprint_weights(
            model_id=args.model_id,
            weights_root=weights_root,
        )
        projector = None
        if args.projector_artifact:
            projector_path = Path(args.projector_artifact).expanduser().resolve()
            projector = PersonalConditioningProjectorArtifact.from_json(
                projector_path.read_text(encoding="utf-8")
            )
        prefix_artifact = None
        if args.prefix_kv_artifact:
            prefix_path = Path(args.prefix_kv_artifact).expanduser().resolve()
            prefix_artifact = PrefixKVArtifact.from_json(
                prefix_path.read_text(encoding="utf-8")
            )
        runtime = TransformersOpenWeightResidualRuntime(
            model_id=args.model_id,
            pretrained_source=str(weights_root),
            device=args.device,
            hook_layer_selection="middle",
            personal_conditioning_scale=args.personal_conditioning_scale,
            personal_conditioning_projector=projector,
            personal_conditioning_prefix=prefix_artifact,
            local_files_only=True,
            runtime_origin="hf-local",
        )
        projector_id = runtime.personal_conditioning_projector_id
        projector_training_mode = (
            runtime.personal_conditioning_projector_training_mode
        )
        prefix_artifact_id = runtime.personal_conditioning_prefix_id
        fingerprint_payload.update(
            {
                "runtime_origin": runtime.runtime_origin,
                "is_frozen": runtime.is_frozen,
                "device_request": args.device,
                "personal_conditioning_scale": args.personal_conditioning_scale,
                "personal_conditioning_projector_id": projector_id,
                "personal_conditioning_projector_training_mode": (
                    projector_training_mode
                ),
                "personal_conditioning_prefix_id": prefix_artifact_id,
                "personal_conditioning_prefix_norm_cap": (
                    prefix_artifact.norm_cap
                    if prefix_artifact is not None
                    else None
                ),
            }
        )
        substrate_kind = SubstrateEvidenceKind.FROZEN_WEIGHTS
        substrate_fingerprint = (
            f"{args.model_id}@"
            f"{str(fingerprint_payload['weights_sha256'])[:16]}"
        )
        cases = (
            build_p2_probe_cases(pair_id=args.p2_pair, strict_carriers=True)
            if args.lane == "p2"
            else build_probe_cases(strict_carriers=True)
        )

    arm_labels = (
        PREFIX_IDENTIFICATION_ARM_LABELS
        if prefix_lane
        else IDENTIFICATION_ARM_LABELS
    )
    candidate_arm_label = (
        PREFIX_ARM_LABEL if prefix_lane else DEFAULT_CANDIDATE_ARM_LABEL
    )
    fingerprint_payload["identification_material"] = {
        "lane": args.lane,
        "p2_pair": args.p2_pair if args.lane == "p2" else "",
        "user_ids": sorted({case.user_id for case in cases}),
        "probe_ids": sorted({case.probe_id for case in cases}),
        "case_count": len(cases),
    }
    recording = RecordingSynthesizer(
        LLMResponseSynthesizer(
            runtime=runtime,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )
    judge = None
    if args.judge_model_id:
        judge_cls = (
            LocalEmbeddingBlindJudge
            if args.judge_kind == "embedding"
            else LocalTransformersBlindJudge
        )
        judge = judge_cls(
            judge_model_id=args.judge_model_id,
            judge_source=args.judge_source or None,
            substrate_model_id=args.model_id,
            substrate_source=str(weights_root),
            materials=_judge_materials_from_cases(cases),
            device=args.judge_device,
            local_files_only=not args.allow_download,
        )
    verdict = run_identification_smoke(
        cases=cases,
        synthesizer=recording,
        base_context=_base_context(),
        substrate_kind=substrate_kind,
        substrate_fingerprint=substrate_fingerprint,
        arm_labels=arm_labels,
        judge=judge,
        bootstrap_seed=args.judge_bootstrap_seed,
        candidate_arm_label=candidate_arm_label,
    )
    if judge is not None:
        fingerprint_payload["blind_judge"] = judge.as_json_dict()
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(verdict.to_json() + "\n", encoding="utf-8")
    _write_json(
        transcript_path,
        _transcript_payload(
            lane=args.lane,
            cases=cases,
            responses=recording.responses,
            projector_id=projector_id,
            projector_training_mode=projector_training_mode,
            arm_labels=arm_labels,
            prefix_artifact_id=prefix_artifact_id,
        ),
    )
    _write_json(fingerprint_path, fingerprint_payload)

    print(f"candidate arm = {verdict.candidate_arm_label}")
    print(f"verdict_state = {verdict.verdict_state.value}")
    for claim in verdict.claims:
        print(f"  {claim.name:38s} {claim.state.value:18s} {claim.detail}")
    print(f"  c5_grade{'':30s} {verdict.c5_grade.value:18s} {verdict.c5_detail}")
    for readout in verdict.matching:
        print(
            f"  matching {readout.arm_label}: "
            f"{readout.correct}/{readout.total} "
            f"accuracy={readout.accuracy:.3f} "
            f"CI=({readout.ci_low:.3f}, {readout.ci_high:.3f})"
        )
    for note in verdict.notes:
        print(f"  note: {note}")
    print(f"turns recorded: {len(verdict.prompt_fp_table)}")
    print(f"verdict: {verdict_path}")
    print(f"transcript: {transcript_path}")
    print(f"substrate fingerprint: {fingerprint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
