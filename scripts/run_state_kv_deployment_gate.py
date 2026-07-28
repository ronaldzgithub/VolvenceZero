#!/usr/bin/env python3
"""Run State-KV deployment negative controls and publish the promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.conditioning_bank_adapters import (  # noqa: E402
    personal_conditioning_to_bank,
)
from volvence_zero.conditioning_bank_contracts import (  # noqa: E402
    ConditioningRevocationState,
    ConditioningScope,
)
from volvence_zero.agent.session_observation import (  # noqa: E402
    _personal_conditioning_delivery_from_config,
)
from volvence_zero.personal_conditioning_contracts import (  # noqa: E402
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.state_kv_deployment import (  # noqa: E402
    STATE_KV_DEPLOYMENT_ARTIFACT_ID,
    STATE_KV_DEPLOYMENT_MODEL_ID,
    STATE_KV_DEPLOYMENT_PROFILE_LABEL,
    StateKVDeploymentSafetyObservation,
    build_state_kv_deployment_report,
    validate_state_kv_deployment_runtime,
)
from volvence_zero.substrate import (  # noqa: E402
    PrefixKVArtifact,
    TransformersOpenWeightResidualRuntime,
)

DEFAULT_PROMPT = "我刚刚差点直接放弃"
_REPAIR_STATE = (
    0.18, 0.88, 0.24, 0.38, 0.58, 0.86, 0.84, 0.72,
    0.42, 0.72, 0.28, 0.84, 0.70, 0.88, 0.52, 0.82,
)
_EXECUTE_STATE = (
    0.90, 0.16, 0.88, 0.86, 0.88, 0.12, 0.20, 0.16,
    0.86, 0.16, 0.88, 0.18, 0.92, 0.12, 0.90, 0.10,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefix-kv-artifact", required=True)
    parser.add_argument("--temporal-causal", required=True)
    parser.add_argument("--judge-court", required=True)
    parser.add_argument("--generation-seed-gate", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _conditioning(
    *,
    user_id: str,
    state: tuple[float, ...],
    confidence: float = 0.72,
) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=state,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 3), ("boundary_consent", 2)),
        source_fingerprint=_sha(user_id),
        confidence=confidence,
        is_cold_start=False,
        description=f"deployment safety state for {user_id}",
    )


def _cold_start() -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(0.0 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(),
        source_fingerprint=_sha("cold-start"),
        confidence=0.0,
        is_cold_start=True,
        description="deployment safety cold start",
    )


def _generate(
    runtime: TransformersOpenWeightResidualRuntime,
    *,
    prompt: str,
    max_new_tokens: int,
    conditioning: PersonalConditioningSnapshot | None,
):
    return runtime.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        capture_residuals=False,
        personal_conditioning=conditioning,
        personal_conditioning_carrier="prefix_kv",
    )


def main() -> int:
    args = _args()
    artifact = PrefixKVArtifact.from_json(
        Path(args.prefix_kv_artifact).read_text(encoding="utf-8")
    )
    if artifact.artifact_id != STATE_KV_DEPLOYMENT_ARTIFACT_ID:
        raise ValueError(
            f"deployment artifact mismatch: {artifact.artifact_id}"
        )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=STATE_KV_DEPLOYMENT_MODEL_ID,
        pretrained_source=args.model_source,
        device=args.device,
        personal_conditioning_prefix=artifact,
        local_files_only=True,
        runtime_origin="hf-local-deployment-gate",
    )
    validate_state_kv_deployment_runtime(runtime)

    correct = _conditioning(user_id="heldout-repair", state=_REPAIR_STATE)
    wrong = _conditioning(user_id="heldout-execute", state=_EXECUTE_STATE)
    cold = _cold_start()
    zero_confidence = replace(correct, confidence=0.0)
    correct_scope = ConditioningScope(
        tenant_scope="deployment-local",
        user_scope="heldout-repair",
        session_scope="state-kv-deployment",
    )
    wrong_scope = ConditioningScope(
        tenant_scope="deployment-local",
        user_scope="heldout-execute",
        session_scope="state-kv-deployment",
    )
    revoked_bank = personal_conditioning_to_bank(
        snapshot=correct,
        scope=correct_scope,
        revocation_state=ConditioningRevocationState.REVOKED,
    )
    correct_bank = personal_conditioning_to_bank(
        snapshot=correct,
        scope=correct_scope,
    )
    wrong_bank = personal_conditioning_to_bank(
        snapshot=wrong,
        scope=wrong_scope,
    )
    if revoked_bank.is_injectable:
        raise RuntimeError("revoked conditioning bank must be inert")
    revoked_conditioning, _, _, _ = (
        _personal_conditioning_delivery_from_config(
            active_conditioning=correct,
            personal_conditioning_mode="prefix_kv",
            revocation_state=ConditioningRevocationState.REVOKED,
        )
    )

    baseline = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=None,
    )
    cold_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=cold,
    )
    zero_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=zero_confidence,
    )
    shadow_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=None,
    )
    revoked_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=revoked_conditioning,
    )
    correct_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=correct,
    )
    wrong_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=wrong,
    )
    correct_replay = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=correct,
    )
    rollback_result = _generate(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        conditioning=None,
    )

    inert_results = (
        baseline,
        cold_result,
        zero_result,
        shadow_result,
        revoked_result,
        rollback_result,
    )
    safety = StateKVDeploymentSafetyObservation(
        profile_label=STATE_KV_DEPLOYMENT_PROFILE_LABEL,
        artifact_id=artifact.artifact_id,
        model_id=runtime.model_id,
        prompt_sha256=_sha(args.prompt),
        cold_start_baseline_equal=cold_result.text == baseline.text,
        zero_confidence_baseline_equal=zero_result.text == baseline.text,
        shadow_baseline_equal=shadow_result.text == baseline.text,
        revoked_baseline_equal=revoked_result.text == baseline.text,
        rollback_baseline_equal=rollback_result.text == baseline.text,
        inert_controls_report_applied_false=all(
            not result.personal_conditioning_applied
            for result in inert_results
        ),
        active_correct_applied=correct_result.personal_conditioning_applied,
        active_wrong_user_applied=wrong_result.personal_conditioning_applied,
        active_users_diverge=correct_result.text != wrong_result.text,
        active_replay_equal=correct_result.text == correct_replay.text,
        user_cache_scopes_distinct=(
            correct_scope.cache_key_parts != wrong_scope.cache_key_parts
            and correct_bank.source_fingerprint
            != wrong_bank.source_fingerprint
        ),
        baseline_output_sha256=_sha(baseline.text),
        correct_output_sha256=_sha(correct_result.text),
        wrong_user_output_sha256=_sha(wrong_result.text),
    )
    report = build_state_kv_deployment_report(
        temporal_causal_path=args.temporal_causal,
        judge_court_path=args.judge_court,
        generation_seed_path=args.generation_seed_gate,
        safety_observation=safety,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")

    print(f"gate_state = {report.gate_state.value}")
    for claim in report.claims:
        print(f"  {claim.claim:40s} {claim.state.value:5s} {claim.detail}")
    print(f"deployment gate: {output}")
    return 0 if report.gate_state.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
