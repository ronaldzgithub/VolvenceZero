#!/usr/bin/env python3
"""Run the real State-KV pre-capture to temporal matched ablation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from volvence_zero.agent.conditioning_lineage import (
    build_conditioning_lineage_ref,
)
from volvence_zero.conditioning_bank_adapters import (
    personal_conditioning_to_bank,
)
from volvence_zero.conditioning_bank_contracts import ConditioningScope
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.state_kv_temporal_causal_evidence import (
    TemporalCausalArm,
    build_temporal_causal_verdict,
)
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    PrefixKVArtifact,
    TransformersOpenWeightResidualRuntime,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy, TemporalModule

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SOURCE_TEXT = "我刚刚差点直接放弃"
_REPAIR_STATE = (
    0.18, 0.88, 0.24, 0.38, 0.58, 0.86, 0.84, 0.72,
    0.42, 0.72, 0.28, 0.84, 0.70, 0.88, 0.52, 0.82,
)
_EXECUTE_STATE = (
    0.90, 0.16, 0.88, 0.86, 0.88, 0.12, 0.20, 0.16,
    0.86, 0.16, 0.88, 0.18, 0.92, 0.12, 0.90, 0.10,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix-kv-artifact", required=True)
    parser.add_argument("--source-text", default=DEFAULT_SOURCE_TEXT)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _conditioning(
    *,
    user_id: str,
    state_vector: tuple[float, ...],
) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=state_vector,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 3), ("relationship_state", 2)),
        source_fingerprint=hashlib.sha256(user_id.encode("utf-8")).hexdigest(),
        confidence=0.72,
        is_cold_start=False,
        description=f"temporal causal evidence state for {user_id}",
    )


def _lineage(
    *,
    conditioning: PersonalConditioningSnapshot,
    user_scope: str,
):
    bank = personal_conditioning_to_bank(
        snapshot=conditioning,
        scope=ConditioningScope(
            tenant_scope="evidence-local",
            user_scope=user_scope,
            session_scope="state-kv-temporal-causal",
        ),
    )
    return build_conditioning_lineage_ref(
        session_scope="state-kv-temporal-causal",
        banks=(bank,),
        carrier="prefix_kv",
        delivery_phase="substrate-capture",
    )


async def _capture_arm(
    *,
    label: str,
    runtime: TransformersOpenWeightResidualRuntime,
    source_text: str,
    conditioning: PersonalConditioningSnapshot | None,
    user_scope: str,
) -> TemporalCausalArm:
    lineage = (
        _lineage(conditioning=conditioning, user_scope=user_scope)
        if conditioning is not None
        else None
    )
    substrate = await OpenWeightResidualStreamSubstrateAdapter(
        runtime=runtime,
        default_source_text=source_text,
        conditioning_lineage=lineage,
        personal_conditioning=conditioning,
        personal_conditioning_carrier="prefix_kv",
    ).capture()
    temporal = await TemporalModule(
        policy=LearnedLiteTemporalPolicy()
    ).process_standalone(substrate_snapshot=substrate)
    return TemporalCausalArm(
        label=label,
        substrate=substrate,
        temporal=temporal.value,
    )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    artifact_path = Path(args.prefix_kv_artifact).expanduser().resolve()
    artifact = PrefixKVArtifact.from_json(artifact_path.read_text(encoding="utf-8"))
    model_source = args.model_source or snapshot_download(
        args.model_id,
        local_files_only=True,
    )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=args.model_id,
        pretrained_source=str(model_source),
        device=args.device,
        hook_layer_selection="all",
        personal_conditioning_prefix=artifact,
        local_files_only=True,
        runtime_origin="hf-local",
    )
    correct = _conditioning(user_id="heldout-repair", state_vector=_REPAIR_STATE)
    wrong = _conditioning(user_id="heldout-execute", state_vector=_EXECUTE_STATE)
    baseline = await _capture_arm(
        label="baseline",
        runtime=runtime,
        source_text=args.source_text,
        conditioning=None,
        user_scope="baseline",
    )
    correct_state = await _capture_arm(
        label="correct-state",
        runtime=runtime,
        source_text=args.source_text,
        conditioning=correct,
        user_scope="heldout-repair",
    )
    wrong_user = await _capture_arm(
        label="wrong-user",
        runtime=runtime,
        source_text=args.source_text,
        conditioning=wrong,
        user_scope="heldout-execute",
    )
    revoked = await _capture_arm(
        label="revoked",
        runtime=runtime,
        source_text=args.source_text,
        conditioning=None,
        user_scope="heldout-repair",
    )
    revision = Path(model_source).name
    verdict = build_temporal_causal_verdict(
        baseline=baseline,
        correct_state=correct_state,
        wrong_user=wrong_user,
        revoked=revoked,
        artifact_id=artifact.artifact_id,
        substrate_fingerprint=f"{args.model_id}@{revision}",
        source_text=args.source_text,
    )
    return verdict.as_json_dict()


def main() -> int:
    args = _parse_args()
    payload = asyncio.run(_run(args))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["gate_state"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
