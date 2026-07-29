#!/usr/bin/env python3
"""Run matched full-code, rank-3, and dynamic-off State KV evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
    _fingerprint_weights,
    _resolve_local_weights,
)
from volvence_zero.agent.session import AgentSessionRunner  # noqa: E402
from volvence_zero.state_kv_control_dim_diagnostic import (  # noqa: E402
    ControlDimSample,
    build_control_dim_verdict,
)
from volvence_zero.substrate import (  # noqa: E402
    FULL_CODE_SINUSOID_CONTROL_BASIS_MODE,
    ControlBasisArtifact,
    TransformersOpenWeightResidualRuntime,
    build_sinusoid_control_basis,
    control_basis_fingerprint,
)

PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "d0",
        "The deployment changed after a rollback. What should I verify first?",
        " Verify the active version and rollback audit before proceeding.",
    ),
    (
        "d1",
        "I am overloaded and the decision is reversible. What is a careful next step?",
        " Take one reversible step, then check the outcome before expanding scope.",
    ),
    (
        "d2",
        "A collaborator broke an agreement and wants to repair it. How should we respond?",
        " Acknowledge the rupture, restate the boundary, and agree on a verifiable repair.",
    ),
    (
        "d3",
        "The evidence is mixed but the deadline is close. What protects decision quality?",
        " Separate known facts from assumptions and preserve a rollback path.",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _collect_codes(*, full_rank: int) -> tuple[dict[str, object], ...]:
    observations = []
    for probe_id, source_text, continuation_text in PROBES:
        runner = AgentSessionRunner(
            session_id=f"state-kv-control-dim:{probe_id}",
            temporal_latent_dim=full_rank,
            rare_heavy_enabled=False,
        )
        result = await runner.run_turn(source_text)
        if not result.track_z_t_codes:
            raise RuntimeError(
                f"probe {probe_id!r} published no temporal controller code"
            )
        for track, code in result.track_z_t_codes:
            if len(code) != full_rank:
                raise RuntimeError(
                    f"probe {probe_id!r} track {track!r} code width is "
                    f"{len(code)}, expected {full_rank}"
                )
            observations.append(
                {
                    "sample_id": f"{probe_id}:{track}",
                    "probe_id": probe_id,
                    "track": track,
                    "source_text": source_text,
                    "continuation_text": continuation_text,
                    "full_code": list(code),
                }
            )
    return tuple(observations)


def _score_arm(
    *,
    runtime: TransformersOpenWeightResidualRuntime,
    observations: tuple[dict[str, object], ...],
    applied_width: int,
    zero_control: bool = False,
) -> tuple[float, ...]:
    outcomes = []
    for observation in observations:
        full_code = tuple(float(value) for value in observation["full_code"])
        applied = (
            tuple(0.0 for _ in range(applied_width))
            if zero_control
            else full_code[:applied_width]
        )
        score = runtime.score_continuation(
            source_text=str(observation["source_text"]),
            continuation_text=str(observation["continuation_text"]),
            applied_control=applied,
        )
        outcomes.append(-score.mean_negative_log_likelihood)
    return tuple(outcomes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--full-rank", type=int, default=16)
    parser.add_argument(
        "--output",
        default="artifacts/state_kv/verdict_control_dim_diagnostic.json",
    )
    parser.add_argument(
        "--observation-output",
        default="artifacts/state_kv/observations_control_dim_diagnostic.json",
    )
    parser.add_argument(
        "--candidate-artifact-output",
        default="artifacts/state_kv/control_basis_full_dimension_candidate.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.full_rank <= 3:
        raise ValueError("--full-rank must be wider than the legacy rank 3")
    weights_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    model_fingerprint = _fingerprint_weights(
        model_id=args.model_id,
        weights_root=weights_root,
    )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=args.model_id,
        pretrained_source=str(weights_root),
        device=args.device,
        local_files_only=True,
        runtime_origin="hf-local",
    )
    if args.full_rank > runtime.hidden_size:
        raise ValueError(
            f"--full-rank {args.full_rank} exceeds hidden size {runtime.hidden_size}"
        )

    observations = asyncio.run(_collect_codes(full_rank=args.full_rank))
    if len(observations) < 8:
        raise RuntimeError(
            "control-dimension diagnostic requires at least eight track samples"
        )
    full_basis = build_sinusoid_control_basis(
        hidden_size=runtime.hidden_size,
        rank=args.full_rank,
    )
    rank3_basis = full_basis[:3]
    candidate = ControlBasisArtifact(
        model_id=runtime.model_id,
        hidden_size=runtime.hidden_size,
        basis=full_basis,
        layer_indices=runtime.hook_layer_indices,
        layer_gains=tuple(1.0 for _ in runtime.hook_layer_indices),
        training_mode=FULL_CODE_SINUSOID_CONTROL_BASIS_MODE,
        source_fingerprint=control_basis_fingerprint(full_basis),
        sample_count=len(observations),
        description=(
            "Unadmitted full-dimension candidate used by the matched D0 "
            "diagnostic. Production installation still requires D0 and "
            "ModificationGate.OFFLINE."
        ),
    )
    runtime.install_control_basis(
        basis=full_basis,
        provenance=f"d0-isolated:{candidate.artifact_id}",
    )
    full_outcomes = _score_arm(
        runtime=runtime,
        observations=observations,
        applied_width=args.full_rank,
    )
    runtime.install_control_basis(
        basis=rank3_basis,
        provenance="d0-isolated:matched-rank3",
    )
    rank3_outcomes = _score_arm(
        runtime=runtime,
        observations=observations,
        applied_width=3,
    )
    off_outcomes = _score_arm(
        runtime=runtime,
        observations=observations,
        applied_width=3,
        zero_control=True,
    )

    enriched = tuple(
        {
            **observation,
            "full_outcome": full_outcome,
            "rank3_outcome": rank3_outcome,
            "dynamic_off_outcome": off_outcome,
            "outcome_kind": "negative_mean_token_nll",
        }
        for observation, full_outcome, rank3_outcome, off_outcome in zip(
            observations,
            full_outcomes,
            rank3_outcomes,
            off_outcomes,
            strict=True,
        )
    )
    observation_output = (REPO_ROOT / args.observation_output).resolve()
    observation_output.parent.mkdir(parents=True, exist_ok=True)
    observation_output.write_text(
        json.dumps(
            {
                "schema_version": "state-kv-control-dim-observations.v1",
                "model": model_fingerprint,
                "candidate_artifact_id": candidate.artifact_id,
                "rank3_basis_fingerprint": control_basis_fingerprint(
                    rank3_basis
                ),
                "samples": list(enriched),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_output = (REPO_ROOT / args.candidate_artifact_output).resolve()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.write_text(candidate.to_json() + "\n", encoding="utf-8")

    verdict = build_control_dim_verdict(
        samples=tuple(
            ControlDimSample(
                sample_id=str(row["sample_id"]),
                full_code=tuple(float(value) for value in row["full_code"]),
                full_outcome=float(row["full_outcome"]),
                rank3_outcome=float(row["rank3_outcome"]),
                dynamic_off_outcome=float(row["dynamic_off_outcome"]),
            )
            for row in enriched
        ),
        artifact_id=f"state-kv-control-dim:{_sha256(observation_output)}",
        source_artifacts=(
            str(observation_output.relative_to(REPO_ROOT)),
            str(candidate_output.relative_to(REPO_ROOT)),
        ),
    )
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")
    print(f"gate_state = {verdict.gate_state}")
    print(f"bottleneck_proven = {verdict.bottleneck_proven}")
    print(f"p5d_decision = {verdict.p5d_decision}")
    print(f"output = {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
