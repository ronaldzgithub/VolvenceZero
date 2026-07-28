#!/usr/bin/env python3
"""Run the State-KV cost/latency gate: Prefix-KV (arm G) vs text carrier (B').

Produces the "成本占优" half of the P3 verdict as a computed artifact. The
identification lane already showed arm G carries state without prompt bytes;
this gate measures, on the same frozen substrate and the same probe material,
whether the prefix carrier is also cheaper (attention slots) and not slower
(median wall-clock) than rendering the identical readout into the system
prompt.

Both arms run through the production ``LLMResponseSynthesizer`` wrapped in a
measuring proxy, so the recorded payloads are the exact bytes that were sent.
Prompt tokens are counted with the substrate's own tokenizer and chat
template; arm G is additionally charged its prefix slot count -- the slots
occupy K/V positions exactly like prompt tokens, and omitting them would rig
the comparison.

Usage:
    python scripts/run_state_kv_cost_gate.py \\
        --device cpu \\
        --prefix-kv-artifact artifacts/state_kv/projectors/<prefix>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in [
    Path(__file__).resolve().parent,
    *sorted((REPO_ROOT / "packages").glob("*/src")),
]:
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_identification import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _base_context,
    _fingerprint_weights,
    _resolve_local_weights,
    build_probe_cases,
)
from volvence_zero.agent.response import LLMResponseSynthesizer  # noqa: E402
from volvence_zero.state_kv_cost_gate import (  # noqa: E402
    DEFAULT_COST_ARM_LABELS,
    MeasuringRuntimeProxy,
    run_cost_gate,
)
from volvence_zero.state_kv_identification import (  # noqa: E402
    SubstrateEvidenceKind,
)
from volvence_zero.substrate import (  # noqa: E402
    PrefixKVArtifact,
    TransformersOpenWeightResidualRuntime,
)


def _chat_template_token_counter(*, weights_root: Path):
    """Count tokens exactly as the runtime encodes them.

    Loads the same tokenizer from the same local snapshot and applies the
    same chat template with ``add_generation_prompt=True``, mirroring
    ``TransformersOpenWeightResidualRuntime._build_generation_inputs``.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(weights_root), local_files_only=True
    )

    def count(chat_messages) -> int:
        payload = [
            {"role": role, "content": content}
            for role, content in chat_messages
        ]
        rendered = tokenizer.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True
        )
        # Tokenize the rendered template text directly: ``tokenize=True``
        # returns a BatchEncoding on some tokenizer versions, whose ``len``
        # is its key count, silently corrupting every measurement.
        token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        return len(token_ids)

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-source",
        default="",
        help="explicit local HF snapshot directory; otherwise resolve local cache",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--personal-conditioning-scale", type=float, default=0.08
    )
    parser.add_argument(
        "--prefix-kv-artifact",
        required=True,
        help="learned State-KV prefix generator JSON (the arm G carrier)",
    )
    parser.add_argument(
        "--latency-tolerance",
        type=float,
        default=0.10,
        help="allowed relative median-latency overhead for arm G (default 10%%)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "run the probe set this many times per arm; more repeats "
            "stabilise the latency medians on a noisy host"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face to download a missing model snapshot",
    )
    parser.add_argument(
        "--output",
        default="",
        help="where to write verdict_cost_gate.json",
    )
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not 0.0 <= args.latency_tolerance < 1.0:
        parser.error("--latency-tolerance must be in [0, 1)")

    output = Path(args.output).expanduser() if args.output else (
        REPO_ROOT / "artifacts" / "state_kv" / "cost-gate" / "verdict_cost_gate.json"
    )
    output = output.resolve()

    weights_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    fingerprint_payload = _fingerprint_weights(
        model_id=args.model_id,
        weights_root=weights_root,
    )
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
        personal_conditioning_prefix=prefix_artifact,
        local_files_only=True,
        runtime_origin="hf-local",
    )
    measuring_runtime = MeasuringRuntimeProxy(
        runtime,
        token_counter=_chat_template_token_counter(weights_root=weights_root),
    )
    synthesizer = LLMResponseSynthesizer(
        runtime=measuring_runtime,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    cases = build_probe_cases(strict_carriers=True)
    if args.repeats > 1:
        # Distinct probe ids per repeat so the per-probe comparison keys stay
        # unique; the underlying sentence and state are unchanged.
        from dataclasses import replace as _replace

        cases = tuple(
            _replace(case, probe_id=f"{case.probe_id}-r{repeat}")
            for repeat in range(args.repeats)
            for case in cases
        )

    substrate_fingerprint = (
        f"{args.model_id}@{str(fingerprint_payload['weights_sha256'])[:16]}"
    )
    verdict = run_cost_gate(
        cases=cases,
        synthesizer=synthesizer,
        measuring_runtime=measuring_runtime,
        base_context=_base_context(),
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint=substrate_fingerprint,
        prefix_slots=prefix_artifact.num_slots,
        latency_tolerance=args.latency_tolerance,
        extra_notes=(
            f"prefix_artifact_id={runtime.personal_conditioning_prefix_id}",
            f"device={args.device}",
            f"max_new_tokens={args.max_new_tokens}",
            f"repeats={args.repeats}",
            "temperature=0.0 deterministic decoding; the latency claim is "
            "per emitted token because EOS varies across arms under the "
            "fixed decode budget, so raw per-turn wall-clock would measure "
            "answer length rather than carrier cost",
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")
    fingerprint_path = output.with_name("substrate_fingerprint.json")
    fingerprint_path.write_text(
        json.dumps(fingerprint_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"gate_state = {verdict.gate_state}")
    for claim in verdict.claims:
        print(f"  {claim.name:38s} {claim.state.value:18s} {claim.detail}")
    for summary in verdict.arm_summaries:
        print(
            f"  {summary.arm_label}: turns={summary.turn_count} "
            f"median_prompt_tokens={summary.median_prompt_tokens} "
            f"prefix_slots={summary.prefix_slots} "
            f"median_latency_ms={summary.median_latency_ms:.1f}"
        )
    print(f"arms = {DEFAULT_COST_ARM_LABELS}")
    print(f"verdict: {output}")
    print(f"substrate fingerprint: {fingerprint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
