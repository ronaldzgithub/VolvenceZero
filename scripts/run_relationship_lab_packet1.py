#!/usr/bin/env python3
"""Run Relationship Lab P1 strong-baseline and Appendable calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
    "packages/companion-ref-harness/src",
    "packages/vz-contracts/src",
    "packages/vz-memory/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from companion_ref_harness import HashingEmbedder  # noqa: E402
from companion_ref_harness.embed import (  # noqa: E402
    Embedder,
    SentenceTransformerEmbedder,
)
from huggingface_hub import snapshot_download  # noqa: E402

from lifeform_evolution.relationship_lab_baseline import (  # noqa: E402
    DEFAULT_STATELESS_MODEL_ID,
    DEFAULT_STATELESS_MODEL_SOURCE,
    HFStatelessRelationshipActionPolicy,
)
from lifeform_evolution.relationship_lab_contexts import (  # noqa: E402
    RELATIONSHIP_P1_DEFAULT_DEPTHS,
    RELATIONSHIP_P1_RAG_TOP_K,
    RelationshipP1Arm,
    build_relationship_p1_context_bundle,
    probe_relationship_p1_persisted_state,
    run_relationship_p1_console_control_probe,
)
from lifeform_evolution.relationship_lab_gate0 import (  # noqa: E402
    load_frozen_baseline_attestation,
)
from lifeform_domain_emogpt.lab import (  # noqa: E402
    load_relationship_transfer_dataset,
)
from lifeform_evolution.relationship_lab_packet1 import (  # noqa: E402
    RelationshipP1GateConfig,
    RelationshipP1RecoveryEvidence,
    assess_relationship_packet1,
    relationship_p1_prompt_path,
    reassess_relationship_packet1_report_v1,
    run_relationship_packet1_arms,
    write_relationship_packet1_artifacts,
)
from lifeform_evolution.relationship_lab_packet1b import (  # noqa: E402
    RELATIONSHIP_P1B_RAG_TOP_K,
    assess_relationship_packet1b,
    parse_relationship_evidence_scores,
    render_relationship_p1b_readout_request,
    relationship_p1b_readout_prompt_path,
    run_relationship_packet1b_arms,
    write_relationship_packet1b_artifacts,
)


_DEFAULT_GATE0_ATTESTATION = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_15b_stateless_calibration_20260819"
    / "baseline_attestation.json"
)


def _parse_int_tuple(raw: str, field_name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be comma-separated integers") from exc
    if not values:
        raise ValueError(f"{field_name} must be non-empty")
    return values


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_digest(snapshot: pathlib.Path) -> str:
    manifest = []
    for path in sorted(
        (item for item in snapshot.rglob("*") if item.is_file()),
        key=lambda item: str(item.relative_to(snapshot)),
    ):
        manifest.append(
            (
                str(path.relative_to(snapshot)),
                path.stat().st_size,
                _sha256_file(path),
            )
        )
    if not manifest:
        raise FileNotFoundError(f"embedding snapshot is empty: {snapshot}")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _FrozenCachedEmbedder:
    def __init__(
        self,
        *,
        delegate: Embedder,
        canonical_name: str,
    ) -> None:
        self._delegate = delegate
        self._canonical_name = canonical_name
        self._cache: dict[str, tuple[float, ...]] = {}

    @property
    def dim(self) -> int:
        return self._delegate.dim

    @property
    def name(self) -> str:
        return self._canonical_name

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        embedded = self._delegate.embed(text)
        self._cache[text] = embedded
        return embedded


def _build_rag_embedder(args: argparse.Namespace) -> Embedder:
    if args.rag_embedder == "hashing":
        delegate = HashingEmbedder()
        return _FrozenCachedEmbedder(
            delegate=delegate,
            canonical_name=delegate.name,
        )
    snapshot = pathlib.Path(
        snapshot_download(
            repo_id=args.rag_model_source,
            local_files_only=not args.allow_download,
        )
    )
    weights_digest = _snapshot_digest(snapshot)
    delegate = SentenceTransformerEmbedder(
        model_id=str(snapshot),
        device=(None if args.rag_device == "auto" else args.rag_device),
    )
    return _FrozenCachedEmbedder(
        delegate=delegate,
        canonical_name=(f"companion-ref-harness/sentence-transformer:{args.rag_model_source}:sha256:{weights_digest}"),
    )


def _release_embedding_runtime() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_probe_subprocess(state_root: pathlib.Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "probe-state",
            "--state-root",
            str(state_root),
        ],
        cwd=str(_REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("fresh-process recovery probe did not return an object")
    return payload


def _run_command(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"P1 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    state_root = output_dir / "state"
    rag_embedder = _build_rag_embedder(args)
    contexts = build_relationship_p1_context_bundle(
        state_root=state_root,
        rag_embedder=rag_embedder,
        background_depths=_parse_int_tuple(
            args.background_depths,
            "background_depths",
        ),
        rag_top_k=args.rag_top_k,
    )
    del rag_embedder
    _release_embedding_runtime()
    recovered_payload = _run_probe_subprocess(state_root)
    recovery = RelationshipP1RecoveryEvidence(
        expected_state_artifact_id=contexts.persisted_state.artifact_id,
        recovered_state_artifact_id=str(recovered_payload["artifact_id"]),
        fresh_process=True,
    )
    console = run_relationship_p1_console_control_probe(root=output_dir / "console_probe_state")
    gate0_baseline = load_frozen_baseline_attestation(pathlib.Path(args.gate0_baseline_attestation))
    policy = HFStatelessRelationshipActionPolicy(
        model_source=args.model_source,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=not args.allow_download,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    seed_schedule = _parse_int_tuple(args.seeds, "seeds")
    checkpoint_path = output_dir / "in_progress_decisions.jsonl"
    expected_decisions = 32 * len(seed_schedule)
    observed_decisions = 0
    with checkpoint_path.open("x", encoding="utf-8") as checkpoint:

        def observe_decision(decision: Any) -> None:
            nonlocal observed_decisions
            checkpoint.write(
                json.dumps(
                    decision.to_payload(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
            observed_decisions += 1
            print(
                f"P1 decision {observed_decisions}/{expected_decisions}: "
                f"{decision.arm.value}/{decision.scene_id} "
                f"valid={decision.valid}",
                file=sys.stderr,
                flush=True,
            )

        run = run_relationship_packet1_arms(
            policy,
            contexts=contexts,
            seed_schedule=seed_schedule,
            decision_observer=observe_decision,
        )
    report = assess_relationship_packet1(
        run=run,
        contexts=contexts,
        recovery=recovery,
        console=console,
        gate0_baseline=gate0_baseline,
        config=RelationshipP1GateConfig(
            minimum_decisions_per_arm=args.minimum_decisions_per_arm,
            minimum_steelman_accuracy=args.minimum_steelman_accuracy,
            maximum_steelman_accuracy=args.maximum_steelman_accuracy,
            minimum_steelman_pair_flip_rate=(args.minimum_steelman_pair_flip_rate),
            minimum_structured_state_pair_flip_rate=(args.minimum_structured_state_pair_flip_rate),
            maximum_rag_to_full_history_token_ratio=(args.maximum_rag_to_full_history_token_ratio),
            maximum_structured_to_full_history_token_ratio=(args.maximum_structured_to_full_history_token_ratio),
        ),
    )
    write_relationship_packet1_artifacts(
        run=run,
        report=report,
        recovery=recovery,
        console=console,
        persisted_state=contexts.persisted_state,
        contexts=contexts,
        output_dir=output_dir,
    )
    checkpoint_path.unlink()
    print(
        json.dumps(
            {
                "report": str(output_dir / "report.json"),
                "artifact_id": report.artifact_id,
                "machinery_ready": report.machinery_ready,
                "gate1_passed": report.gate1_passed,
                "arm_metrics": {arm: dict(metrics) for arm, metrics in report.arm_metrics},
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.gate1_passed else 2


def _run_p1b_command(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"P1b output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    state_root = output_dir / "state"
    rag_embedder = _build_rag_embedder(args)
    contexts = build_relationship_p1_context_bundle(
        state_root=state_root,
        rag_embedder=rag_embedder,
        background_depths=_parse_int_tuple(
            args.background_depths,
            "background_depths",
        ),
        rag_top_k=args.rag_top_k,
    )
    del rag_embedder
    _release_embedding_runtime()
    recovered_payload = _run_probe_subprocess(state_root)
    recovery = RelationshipP1RecoveryEvidence(
        expected_state_artifact_id=contexts.persisted_state.artifact_id,
        recovered_state_artifact_id=str(recovered_payload["artifact_id"]),
        fresh_process=True,
    )
    console = run_relationship_p1_console_control_probe(root=output_dir / "console_probe_state")
    gate0_baseline = load_frozen_baseline_attestation(pathlib.Path(args.gate0_baseline_attestation))
    policy = HFStatelessRelationshipActionPolicy(
        model_source=args.model_source,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=not args.allow_download,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    seed_schedule = _parse_int_tuple(args.seeds, "seeds")
    decision_checkpoint_path = output_dir / "in_progress_decisions.jsonl"
    readout_checkpoint_path = output_dir / "in_progress_readouts.jsonl"
    expected_decisions = 32 * len(seed_schedule)
    expected_readouts = 24 * len(seed_schedule)
    observed_decisions = 0
    observed_readouts = 0
    with (
        decision_checkpoint_path.open("x", encoding="utf-8") as decisions,
        readout_checkpoint_path.open("x", encoding="utf-8") as readouts,
    ):

        def observe_readout(readout: Any) -> None:
            nonlocal observed_readouts
            payload = readout.to_payload()
            payload["artifact_id"] = readout.artifact_id
            readouts.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            readouts.flush()
            os.fsync(readouts.fileno())
            observed_readouts += 1
            print(
                f"P1b readout {observed_readouts}/{expected_readouts}: "
                f"{readout.arm.value}/{readout.scene_id} "
                f"valid={readout.valid}",
                file=sys.stderr,
                flush=True,
            )

        def observe_decision(decision: Any) -> None:
            nonlocal observed_decisions
            decisions.write(
                json.dumps(
                    decision.to_payload(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            decisions.flush()
            os.fsync(decisions.fileno())
            observed_decisions += 1
            print(
                f"P1b decision {observed_decisions}/{expected_decisions}: "
                f"{decision.arm.value}/{decision.scene_id} "
                f"valid={decision.valid}",
                file=sys.stderr,
                flush=True,
            )

        run = run_relationship_packet1b_arms(
            policy,
            contexts=contexts,
            seed_schedule=seed_schedule,
            readout_observer=observe_readout,
            decision_observer=observe_decision,
        )
    p1_report = assess_relationship_packet1(
        run=run.action_run,
        contexts=contexts,
        recovery=recovery,
        console=console,
        gate0_baseline=gate0_baseline,
        config=RelationshipP1GateConfig(
            minimum_decisions_per_arm=args.minimum_decisions_per_arm,
            minimum_steelman_accuracy=args.minimum_steelman_accuracy,
            maximum_steelman_accuracy=args.maximum_steelman_accuracy,
            minimum_steelman_pair_flip_rate=(args.minimum_steelman_pair_flip_rate),
            minimum_structured_state_pair_flip_rate=(args.minimum_structured_state_pair_flip_rate),
            maximum_rag_to_full_history_token_ratio=(args.maximum_rag_to_full_history_token_ratio),
            maximum_structured_to_full_history_token_ratio=(args.maximum_structured_to_full_history_token_ratio),
        ),
    )
    report = assess_relationship_packet1b(
        run=run,
        p1_report=p1_report,
        contexts=contexts,
    )
    write_relationship_packet1_artifacts(
        run=run.action_run,
        report=p1_report,
        recovery=recovery,
        console=console,
        persisted_state=contexts.persisted_state,
        contexts=contexts,
        output_dir=output_dir,
    )
    write_relationship_packet1b_artifacts(
        run=run,
        report=report,
        output_dir=output_dir,
    )
    decision_checkpoint_path.unlink()
    readout_checkpoint_path.unlink()
    print(
        json.dumps(
            {
                "report": str(output_dir / "packet1b_report.json"),
                "artifact_id": report.artifact_id,
                "verdict": report.verdict.value,
                "gate1_passed": report.gate1_passed,
                "all_readouts_valid": report.all_readouts_valid,
                "saturated_arms": list(report.saturated_arms),
                "arm_metrics": {arm: dict(metrics) for arm, metrics in report.arm_metrics},
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.gate1_passed else 2


def _probe_command(args: argparse.Namespace) -> int:
    state = probe_relationship_p1_persisted_state(state_root=pathlib.Path(args.state_root))
    payload = state.to_payload()
    payload["artifact_id"] = state.artifact_id
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _protocol_probe_command(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="relationship-p1-protocol-") as root:
        dataset = load_relationship_transfer_dataset()
        contexts = build_relationship_p1_context_bundle(
            state_root=pathlib.Path(root),
            rag_embedder=HashingEmbedder(),
            dataset=dataset,
            background_depths=(0, 8),
        )
        observations = tuple(item for item in dataset.observations if item.scene_id == args.scene_id)
        if len(observations) != 1:
            raise KeyError(f"unknown relationship scene: {args.scene_id}")
        current_input = observations[0].current_input
        policy = HFStatelessRelationshipActionPolicy(
            model_source=args.model_source,
            model_id=args.model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            local_files_only=not args.allow_download,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        rows = []
        for arm in (
            RelationshipP1Arm.PROMPT_STEELMAN,
            RelationshipP1Arm.RAG_STEELMAN,
            RelationshipP1Arm.STRUCTURED_STATE,
        ):
            context = contexts.context(scene_id=args.scene_id, arm=arm)
            completion = policy.choose_from_messages(
                messages=(
                    {
                        "role": "system",
                        "content": relationship_p1_prompt_path(arm).read_text(encoding="utf-8").strip(),
                    },
                    {
                        "role": "user",
                        "content": context.render_user_message(current_input),
                    },
                ),
                seed=args.seed,
            )
            rows.append(
                {
                    "arm": arm.value,
                    "raw_output": completion.raw_output,
                    "valid": completion.chosen_action_id is not None,
                    "chosen_action_id": (
                        completion.chosen_action_id.value if completion.chosen_action_id is not None else None
                    ),
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                }
            )
    print(json.dumps({"scene_id": args.scene_id, "results": rows}, ensure_ascii=False))
    return 0 if all(bool(row["valid"]) for row in rows) else 2


def _p1b_protocol_probe_command(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="relationship-p1b-protocol-") as root:
        dataset = load_relationship_transfer_dataset()
        contexts = build_relationship_p1_context_bundle(
            state_root=pathlib.Path(root),
            rag_embedder=HashingEmbedder(),
            dataset=dataset,
            background_depths=(0, args.background_depth),
            rag_top_k=args.rag_top_k,
        )
        observations = tuple(item for item in dataset.observations if item.scene_id == args.scene_id)
        if len(observations) != 1:
            raise KeyError(f"unknown relationship scene: {args.scene_id}")
        observation = observations[0]
        policy = HFStatelessRelationshipActionPolicy(
            model_source=args.model_source,
            model_id=args.model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            local_files_only=not args.allow_download,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        prompt = relationship_p1b_readout_prompt_path().read_text(encoding="utf-8").strip()
        rows = []
        for arm in (
            RelationshipP1Arm.PROMPT_STEELMAN,
            RelationshipP1Arm.RAG_STEELMAN,
            RelationshipP1Arm.STRUCTURED_STATE,
        ):
            context = contexts.context(scene_id=args.scene_id, arm=arm)
            completion = policy.choose_from_messages(
                messages=(
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": render_relationship_p1b_readout_request(
                            context_text=context.context_text,
                            current_input=observation.current_input,
                        ),
                    },
                ),
                seed=args.seed,
            )
            stay_score, space_score = parse_relationship_evidence_scores(completion.raw_output)
            valid = stay_score is not None and space_score is not None
            rows.append(
                {
                    "arm": arm.value,
                    "raw_output": completion.raw_output,
                    "valid": valid,
                    "stay_score": stay_score,
                    "space_score": space_score,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                }
            )
    print(json.dumps({"scene_id": args.scene_id, "results": rows}, ensure_ascii=False))
    return 0 if all(bool(row["valid"]) for row in rows) else 2


def _reassess_report_command(args: argparse.Namespace) -> int:
    output_path = pathlib.Path(args.output_report)
    if output_path.exists():
        raise FileExistsError(f"P1 reassessment output already exists: {output_path}")
    report = reassess_relationship_packet1_report_v1(
        source_report_path=pathlib.Path(args.source_report),
        minimum_structured_state_pair_flip_rate=(args.minimum_structured_state_pair_flip_rate),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json(), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(output_path),
                "artifact_id": report.artifact_id,
                "source_report_artifact_id": report.source_report_artifact_id,
                "machinery_ready": report.machinery_ready,
                "gate1_passed": report.gate1_passed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.gate1_passed else 2


def _add_run_arguments(
    command: argparse.ArgumentParser,
    *,
    output_prefix: str,
    rag_top_k_default: int,
) -> None:
    command.add_argument("--model-source", default=DEFAULT_STATELESS_MODEL_SOURCE)
    command.add_argument("--model-id", default=DEFAULT_STATELESS_MODEL_ID)
    command.add_argument("--device", default="auto")
    command.add_argument("--torch-dtype", default="auto")
    command.add_argument("--temperature", type=float, default=0.2)
    command.add_argument("--top-p", type=float, default=0.9)
    command.add_argument("--max-new-tokens", type=int, default=48)
    command.add_argument("--seeds", default="101")
    command.add_argument(
        "--background-depths",
        default=",".join(str(item) for item in RELATIONSHIP_P1_DEFAULT_DEPTHS),
    )
    command.add_argument(
        "--rag-embedder",
        choices=("bge-m3", "hashing"),
        default="bge-m3",
    )
    command.add_argument("--rag-model-source", default="BAAI/bge-m3")
    command.add_argument("--rag-device", default="cpu")
    command.add_argument("--rag-top-k", type=int, default=rag_top_k_default)
    command.add_argument("--allow-download", action="store_true")
    command.add_argument(
        "--gate0-baseline-attestation",
        default=str(_DEFAULT_GATE0_ATTESTATION),
    )
    command.add_argument("--minimum-decisions-per-arm", type=int, default=8)
    command.add_argument("--minimum-steelman-accuracy", type=float, default=0.625)
    command.add_argument("--maximum-steelman-accuracy", type=float, default=0.875)
    command.add_argument(
        "--minimum-steelman-pair-flip-rate",
        type=float,
        default=0.5,
    )
    command.add_argument(
        "--minimum-structured-state-pair-flip-rate",
        type=float,
        default=0.5,
    )
    command.add_argument(
        "--maximum-rag-to-full-history-token-ratio",
        type=float,
        default=0.55,
    )
    command.add_argument(
        "--maximum-structured-to-full-history-token-ratio",
        type=float,
        default=0.4,
    )
    command.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "artifacts" / "relationship_lab" / f"{output_prefix}_{int(time.time())}"),
    )


def _add_protocol_probe_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model-source", default=DEFAULT_STATELESS_MODEL_SOURCE)
    command.add_argument("--model-id", default=DEFAULT_STATELESS_MODEL_ID)
    command.add_argument("--device", default="auto")
    command.add_argument("--torch-dtype", default="auto")
    command.add_argument("--temperature", type=float, default=0.2)
    command.add_argument("--top-p", type=float, default=0.9)
    command.add_argument("--max-new-tokens", type=int, default=48)
    command.add_argument("--seed", type=int, default=101)
    command.add_argument("--scene-id", default="rtv1_scene_003a")
    command.add_argument("--allow-download", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    _add_run_arguments(
        run,
        output_prefix="packet1_development",
        rag_top_k_default=RELATIONSHIP_P1_RAG_TOP_K,
    )
    p1b = subparsers.add_parser("run-p1b")
    _add_run_arguments(
        p1b,
        output_prefix="packet1b_development",
        rag_top_k_default=RELATIONSHIP_P1B_RAG_TOP_K,
    )
    probe = subparsers.add_parser("probe-state")
    probe.add_argument("--state-root", required=True)
    protocol_probe = subparsers.add_parser("probe-protocol")
    _add_protocol_probe_arguments(protocol_probe)
    p1b_protocol_probe = subparsers.add_parser("probe-p1b")
    _add_protocol_probe_arguments(p1b_protocol_probe)
    p1b_protocol_probe.add_argument("--background-depth", type=int, default=32)
    p1b_protocol_probe.add_argument(
        "--rag-top-k",
        type=int,
        default=RELATIONSHIP_P1B_RAG_TOP_K,
    )
    reassess = subparsers.add_parser("reassess-report")
    reassess.add_argument("--source-report", required=True)
    reassess.add_argument("--output-report", required=True)
    reassess.add_argument(
        "--minimum-structured-state-pair-flip-rate",
        type=float,
        default=0.5,
    )
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe-state":
        return _probe_command(args)
    if args.command == "probe-protocol":
        return _protocol_probe_command(args)
    if args.command == "probe-p1b":
        return _p1b_protocol_probe_command(args)
    if args.command == "reassess-report":
        return _reassess_report_command(args)
    if args.command == "run-p1b":
        return _run_p1b_command(args)
    return _run_command(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
