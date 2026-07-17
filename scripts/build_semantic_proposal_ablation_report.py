"""Build the LLM-proposal dependency ablation report (experiment 2 of
``docs/specs/semantic-grounding-evidence.md``).

Runs the two matched arms (``semantic-proposal-on`` vs
``semantic-proposal-off``) over the same scripted 9-slot probe cases and
writes ``semantic_proposal_ablation_report.json`` plus each arm's
grounding cross-read (experiment 1 coupling).

Lanes:

* Synthetic smoke (default) — the on arm uses the deterministic
  scripted probe runtime; validates wiring and the differential design::

      python -u scripts/build_semantic_proposal_ablation_report.py

* HF lane (real evidence) — one shared Qwen runtime for BOTH arms
  (identical residual path / substrate fingerprint), the on arm wraps
  it in ``LLMSemanticProposalRuntime``::

      python -u scripts/build_semantic_proposal_ablation_report.py \
          --substrate-mode hf --arm-runtime hf-llm

Outputs (in ``--output-dir``):

* ``semantic_proposal_ablation_report.json``
* ``semantic_grounding_report_on.json`` / ``..._off.json``  (cross-read)
* ``semantic_grounding_turns_on.json`` / ``..._off.json``
* ``semantic_proposal_ablation_manifest.json``  (sha256 sidecar)

All artifacts are non-gating reference artifacts (R12 readout-only).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from lifeform_evolution.semantic_proposal_ablation import (
    run_semantic_proposal_ablation_async,
)
from volvence_zero.agent.semantic_grounding import turn_evidence_to_payload

_MANIFEST_SCHEMA_VERSION = "semantic-proposal-ablation-manifest.v1"


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _collect_provenance() -> dict[str, object]:
    status = _git_output(("status", "--porcelain"))
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "git_branch": _git_output(("branch", "--show-current")),
        "working_tree_dirty": status not in {"", "unknown"},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _build_substrate_runtime(
    *,
    substrate_mode: str,
    model_id: str,
    device: str,
    local_files_only: bool,
):
    if substrate_mode == "hf":
        from volvence_zero.substrate import build_transformers_runtime_with_fallback

        return build_transformers_runtime_with_fallback(
            model_id=model_id,
            device=device,
            local_files_only=local_files_only,
            allow_live_substrate_mutation=False,
            fallback_to_builtin=False,
        )
    if substrate_mode == "synthetic":
        # ``None`` lets ``build_companion_lifeform`` use its default
        # synthetic semantic-surface adapter path — the same shape the
        # companion evidence harness runs on.
        return None
    raise ValueError(f"unsupported substrate_mode {substrate_mode!r}")


def _hf_llm_runtime_factory(substrate_runtime):
    """On-arm factory for the hf lane: wrap the SHARED substrate
    runtime's HF internals in ``LLMSemanticProposalRuntime`` (same
    access pattern as the vertical factory bridge)."""

    model = getattr(substrate_runtime, "_model", None)
    tokenizer = getattr(substrate_runtime, "_tokenizer", None)
    device = getattr(substrate_runtime, "_device", "cpu")
    if model is None or tokenizer is None:
        raise SystemExit(
            "--arm-runtime hf-llm requires an hf substrate runtime that "
            "exposes HF internals; got a runtime without _model/_tokenizer "
            "(synthetic fallback?). Use --substrate-mode hf."
        )
    from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
    from volvence_zero.substrate.text_generation import HFTextGenerationProvider

    provider = HFTextGenerationProvider(
        model=model, tokenizer=tokenizer, device=device
    )
    return lambda: LLMSemanticProposalRuntime(provider=provider)


async def main(
    *,
    output_dir: Path,
    substrate_mode: str,
    arm_runtime: str,
    substrate_model_id: str,
    substrate_device: str,
    substrate_local_files_only: bool,
) -> int:
    if arm_runtime == "hf-llm" and substrate_mode != "hf":
        raise SystemExit("--arm-runtime hf-llm requires --substrate-mode hf.")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _collect_provenance()

    substrate_runtime = _build_substrate_runtime(
        substrate_mode=substrate_mode,
        model_id=substrate_model_id,
        device=substrate_device,
        local_files_only=substrate_local_files_only,
    )
    if substrate_mode == "hf":
        fingerprint = f"hf:{substrate_model_id}"
        on_runtime_factory = (
            _hf_llm_runtime_factory(substrate_runtime)
            if arm_runtime == "hf-llm"
            else None
        )
    else:
        fingerprint = "synthetic:companion-semantic-surface"
        on_runtime_factory = None  # scripted probe runtime default

    print(
        f"[ablation] running both arms (substrate={substrate_mode}, "
        f"on-arm runtime={arm_runtime})",
        flush=True,
    )
    run_result = await run_semantic_proposal_ablation_async(
        on_runtime_factory=on_runtime_factory,
        substrate_runtime=substrate_runtime,
        substrate_fingerprint=fingerprint,
    )
    report = run_result.report

    artifact_paths: list[Path] = []

    grounding_refs: dict[str, dict[str, object]] = {}
    for arm_label, grounding_report, grounding_turns in (
        ("on", run_result.on_grounding_report, run_result.on_grounding_turns),
        ("off", run_result.off_grounding_report, run_result.off_grounding_turns),
    ):
        if grounding_report is None:
            continue
        turns_path = output_dir / f"semantic_grounding_turns_{arm_label}.json"
        turns_payload = turn_evidence_to_payload(grounding_turns)
        turns_payload["provenance"] = provenance
        turns_path.write_text(
            json.dumps(turns_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifact_paths.append(turns_path)

        report_path = output_dir / f"semantic_grounding_report_{arm_label}.json"
        grounding_payload = grounding_report.to_payload()
        grounding_payload["provenance"] = provenance
        report_path.write_text(
            json.dumps(grounding_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifact_paths.append(report_path)
        grounding_refs[arm_label] = _file_record(report_path)

    report_payload = report.to_payload()
    report_payload["provenance"] = provenance
    report_payload["substrate_mode"] = substrate_mode
    report_payload["grounding_report_refs"] = grounding_refs
    if substrate_mode == "synthetic":
        report_payload["evidence_tier"] = (
            "synthetic-smoke: wiring + differential-design evidence only; "
            "the dependency claim requires the hf lane with "
            "--arm-runtime hf-llm."
        )

    report_path = output_dir / "semantic_proposal_ablation_report.json"
    report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifact_paths.append(report_path)

    manifest_path = output_dir / "semantic_proposal_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "artifact_kind": "semantic_proposal_ablation_manifest",
                "artifacts": [_file_record(path) for path in artifact_paths],
                "provenance": provenance,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[ablation] verdict: {report.verdict} "
        f"(overall drop {report.overall_drop:+.3f}, proposal-channel "
        f"{report.proposal_channel_drop:+.3f}, typed-event "
        f"{report.typed_event_drop:+.3f})"
    )
    print(
        f"[ablation] on arm hit-rate {report.on_arm.overall_hit_rate:.3f} "
        f"(grounding: {report.on_arm.grounding_verdict or 'n/a'}); "
        f"off arm {report.off_arm.overall_hit_rate:.3f} "
        f"(grounding: {report.off_arm.grounding_verdict or 'n/a'})"
    )
    print(f"[ablation] report written to {report_path}")
    print(f"[ablation] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/semantic_proposal_ablation"),
    )
    parser.add_argument(
        "--substrate-mode",
        default="synthetic",
        choices=["synthetic", "hf"],
    )
    parser.add_argument(
        "--arm-runtime",
        default="scripted",
        choices=["scripted", "hf-llm"],
        help=(
            "On-arm semantic runtime: 'scripted' replays the probe "
            "ground-truth script (smoke lane); 'hf-llm' wraps the shared "
            "hf substrate in LLMSemanticProposalRuntime (evidence lane)."
        ),
    )
    parser.add_argument(
        "--substrate-model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--substrate-device", default="cpu")
    parser.add_argument("--substrate-allow-download", action="store_true")
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            main(
                output_dir=args.output_dir,
                substrate_mode=args.substrate_mode,
                arm_runtime=args.arm_runtime,
                substrate_model_id=args.substrate_model_id,
                substrate_device=args.substrate_device,
                substrate_local_files_only=not args.substrate_allow_download,
            )
        )
    )
