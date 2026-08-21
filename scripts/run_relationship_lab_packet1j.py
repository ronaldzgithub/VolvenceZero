#!/usr/bin/env python3
"""Prepare and execute the frozen one-shot Relationship Lab P1j v4 run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import sys
import tempfile


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/companion-ref-harness/src",
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
    "packages/vz-contracts/src",
    "packages/vz-memory/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from companion_ref_harness.embed import (  # noqa: E402
    Embedder,
    SentenceTransformerEmbedder,
)
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402
from lifeform_domain_emogpt.lab import (  # noqa: E402
    load_relationship_consumer_split_bundle,
    load_relationship_consumer_training_view,
)
from lifeform_evolution.relationship_lab_baseline import (  # noqa: E402
    HFStatelessRelationshipActionPolicy,
    frozen_model_weights_sha256,
)
from lifeform_evolution.relationship_lab_contexts import (  # noqa: E402
    RelationshipP1RagCandidateSurface,
    build_relationship_p1_context_bundle,
    load_relationship_p1_context_replay_manifest,
)
from lifeform_evolution.relationship_lab_packet1i import (  # noqa: E402
    load_relationship_p1i_calibration_protocol,
    load_relationship_p1i_calibration_report,
    load_relationship_p1i_frozen_consumer_protocol,
    validate_relationship_p1i_candidate_files,
    validate_relationship_p1i_frozen_consumer_lineage,
)
from lifeform_evolution.relationship_lab_packet1j import (  # noqa: E402
    RelationshipP1jVerdict,
    assess_relationship_p1j_qualification,
    build_relationship_p1j_checkpoint,
    execute_relationship_p1j_qualification,
    freeze_relationship_p1j_protocol,
    load_relationship_p1j_checkpoint,
    load_relationship_p1j_progress,
    load_relationship_p1j_protocol,
    load_relationship_p1j_report,
    persist_relationship_p1j_decision,
    persist_relationship_p1j_readout,
    validate_relationship_p1j_progress,
    validate_relationship_p1j_protocol_lineage,
    validate_relationship_p1j_report_lineage,
    validate_relationship_p1j_terminal_files,
    write_relationship_p1j_checkpoint,
    write_relationship_p1j_protocol,
    write_relationship_p1j_report,
)


_DEFAULT_P1I_ARTIFACT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1i_v3_training_replay_20260820"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1j_v4_one_shot_20260821"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-p1i-artifact-dir",
        default=str(_DEFAULT_P1I_ARTIFACT_DIR),
    )
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow materializing missing frozen Qwen/BGE snapshots.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Freeze v4 public inputs/context/protocol before any Qwen output, "
            "then stop."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only the already-frozen one-shot attempt.",
    )
    parser.add_argument(
        "--max-new-readouts",
        type=int,
        default=4,
        help="Maximum new Qwen outputs this process; use 0 for all remaining.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_manifest_digest(snapshot: pathlib.Path) -> str:
    manifest = tuple(
        (
            str(path.relative_to(snapshot)),
            path.stat().st_size,
            _sha256_file(path),
        )
        for path in sorted(
            (item for item in snapshot.rglob("*") if item.is_file()),
            key=lambda item: str(item.relative_to(snapshot)),
        )
    )
    if not manifest:
        raise FileNotFoundError(f"P1j snapshot is empty: {snapshot}")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _materialize_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    allow_download: bool,
) -> pathlib.Path | None:
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        if not allow_download:
            return None
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=False,
        )
    return pathlib.Path(resolved)


class _FrozenCachedEmbedder:
    def __init__(self, *, delegate: Embedder, canonical_name: str) -> None:
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
        if cached is None:
            cached = self._delegate.embed(text)
            self._cache[text] = cached
        return cached


def _release_runtime() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_frozen_consumer(source_dir: pathlib.Path):
    calibration_protocol = load_relationship_p1i_calibration_protocol()
    training_view = load_relationship_consumer_training_view()
    report = load_relationship_p1i_calibration_report(
        pathlib.Path(source_dir) / "packet1i_report.json"
    )
    consumer = load_relationship_p1i_frozen_consumer_protocol(
        pathlib.Path(source_dir) / "frozen_consumer_protocol.json"
    )
    validate_relationship_p1i_frozen_consumer_lineage(
        consumer,
        calibration_protocol=calibration_protocol,
        report=report,
        training_view=training_view,
    )
    validate_relationship_p1i_candidate_files(
        report=report,
        output_dir=pathlib.Path(source_dir),
    )
    return consumer


def _build_contexts(
    *,
    consumer,
    split_bundle,
    rag_snapshot: pathlib.Path,
    replay_manifest=None,
):
    rag_weights = _snapshot_manifest_digest(rag_snapshot)
    if rag_weights != consumer.rag_weights_sha256:
        raise ValueError("P1j BGE weights diverge from frozen consumer")
    delegate = SentenceTransformerEmbedder(model_id=str(rag_snapshot), device="cpu")
    embedder = _FrozenCachedEmbedder(
        delegate=delegate,
        canonical_name=(
            "companion-ref-harness/sentence-transformer:"
            f"{consumer.rag_model_source}:sha256:{rag_weights}"
        ),
    )
    state_root = tempfile.TemporaryDirectory(prefix="relationship-p1j-context-")
    try:
        contexts = build_relationship_p1_context_bundle(
            state_root=pathlib.Path(state_root.name),
            rag_embedder=embedder,
            dataset=split_bundle.qualification_dataset,
            background_template_package_name=consumer.training_package_name,
            background_depths=consumer.background_depths,
            rag_top_k=consumer.rag_top_k,
            rag_candidate_surface=RelationshipP1RagCandidateSurface(
                consumer.rag_candidate_surface
            ),
            rag_replay_orders=(
                () if replay_manifest is None else replay_manifest.rag_orders
            ),
        )
        if replay_manifest is not None:
            replay_manifest.validate_model_inputs(contexts)
    finally:
        del embedder
        del delegate
        state_root.cleanup()
        _release_runtime()
    return contexts


def _context_manifest_payload(contexts) -> dict[str, object]:
    return {"artifact_id": contexts.artifact_id, **contexts.to_summary_payload()}


def _prepare_attempt(
    *,
    output_dir: pathlib.Path,
    consumer,
    split_bundle,
    contexts,
    model_weights: str,
) -> tuple[object, object]:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p1j-prepare-", dir=parent) as root:
        staging = pathlib.Path(root) / "attempt"
        staging.mkdir()
        manifest_payload = _context_manifest_payload(contexts)
        manifest_path = staging / "qualification_contexts.json"
        manifest_path.write_text(
            json.dumps(
                manifest_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        protocol = freeze_relationship_p1j_protocol(
            consumer=consumer,
            split_bundle=split_bundle,
            contexts=contexts,
            context_manifest_artifact_id=contexts.artifact_id,
        )
        write_relationship_p1j_protocol(
            protocol,
            staging / "packet1j_protocol.json",
        )
        checkpoint = build_relationship_p1j_checkpoint(
            protocol=protocol,
            consumer=consumer,
            split_bundle=split_bundle,
        )
        write_relationship_p1j_checkpoint(
            checkpoint=checkpoint,
            output_dir=staging,
        )
        preflight = {
            "consumer_protocol_id": consumer.protocol_id,
            "qualification_protocol_id": protocol.protocol_id,
            "consumer_split_contract_id": (
                split_bundle.contract.contract_sha256
            ),
            "qualification_dataset_fingerprint": (
                split_bundle.qualification_dataset.dataset_fingerprint
            ),
            "context_manifest_artifact_id": contexts.artifact_id,
            "qualification_context_surface_sha256": (
                protocol.qualification_context_surface_sha256
            ),
            "model_weights_sha256": model_weights,
            "rag_weights_sha256": consumer.rag_weights_sha256,
            "qualification_public_inputs_materialized": (
                protocol.qualification_observation_count
            ),
            "qualification_qwen_outputs_observed": 0,
            "planned_qwen_outputs": protocol.planned_qwen_output_count,
            "one_shot": True,
            "ready": True,
        }
        (staging / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    return protocol, checkpoint


def _load_prepared_attempt(
    *,
    output_dir: pathlib.Path,
    consumer,
    split_bundle,
    rag_snapshot: pathlib.Path,
):
    manifest = load_relationship_p1_context_replay_manifest(
        output_dir / "qualification_contexts.json",
        dataset=split_bundle.qualification_dataset,
    )
    contexts = _build_contexts(
        consumer=consumer,
        split_bundle=split_bundle,
        rag_snapshot=rag_snapshot,
        replay_manifest=manifest,
    )
    protocol = load_relationship_p1j_protocol(
        output_dir / "packet1j_protocol.json"
    )
    validate_relationship_p1j_protocol_lineage(
        protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        context_manifest_artifact_id=manifest.artifact_id,
    )
    checkpoint = load_relationship_p1j_checkpoint(output_dir)
    expected_checkpoint = build_relationship_p1j_checkpoint(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("P1j prepared checkpoint diverges from frozen protocol")
    return protocol, checkpoint, contexts


def _terminal_exit_code(verdict: RelationshipP1jVerdict) -> int:
    return 0 if verdict is RelationshipP1jVerdict.QUALIFIED else 2


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.max_new_readouts < 0:
        raise ValueError("--max-new-readouts must be zero or positive")
    source_dir = pathlib.Path(args.source_p1i_artifact_dir)
    output_dir = pathlib.Path(args.output_dir)

    # Validate the complete P1i freeze before the evaluator materializes v4.
    consumer = _load_frozen_consumer(source_dir)
    model_snapshot = _materialize_snapshot(
        repo_id=consumer.model_source,
        revision=consumer.model_revision,
        allow_download=args.allow_download,
    )
    rag_snapshot = _materialize_snapshot(
        repo_id=consumer.rag_model_source,
        revision=None,
        allow_download=args.allow_download,
    )
    if model_snapshot is None or rag_snapshot is None:
        print(
            json.dumps(
                {
                    "consumer_protocol_id": consumer.protocol_id,
                    "model_available": model_snapshot is not None,
                    "rag_available": rag_snapshot is not None,
                    "v4_materialized": False,
                    "ready": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3
    model_weights = frozen_model_weights_sha256(model_snapshot)
    if model_weights != consumer.expected_weights_sha256:
        raise ValueError("P1j Qwen weights diverge from frozen consumer")

    # P1j is the evaluator boundary authorized to materialize the full v4 bundle.
    split_bundle = load_relationship_consumer_split_bundle()
    if (
        split_bundle.contract.contract_sha256
        != consumer.consumer_split_contract_id
        or split_bundle.qualification_dataset.dataset_fingerprint
        != consumer.qualification_dataset_fingerprint
    ):
        raise ValueError("P1j v4 split diverges from the frozen consumer")

    if output_dir.exists():
        if not args.resume and not args.prepare_only:
            raise FileExistsError(
                "P1j one-shot attempt already exists; only --resume is allowed"
            )
        protocol, checkpoint, contexts = _load_prepared_attempt(
            output_dir=output_dir,
            consumer=consumer,
            split_bundle=split_bundle,
            rag_snapshot=rag_snapshot,
        )
    else:
        if args.resume:
            raise FileNotFoundError("P1j resume cannot create a second attempt")
        contexts = _build_contexts(
            consumer=consumer,
            split_bundle=split_bundle,
            rag_snapshot=rag_snapshot,
        )
        protocol, checkpoint = _prepare_attempt(
            output_dir=output_dir,
            consumer=consumer,
            split_bundle=split_bundle,
            contexts=contexts,
            model_weights=model_weights,
        )
    print(
        json.dumps(
            {
                "stage": "prepared",
                "qualification_protocol_id": protocol.protocol_id,
                "consumer_protocol_id": consumer.protocol_id,
                "context_manifest_artifact_id": (
                    protocol.context_manifest_artifact_id
                ),
                "qualification_inputs_materialized": (
                    protocol.qualification_observation_count
                ),
                "qualification_qwen_outputs_observed_before_freeze": 0,
                "planned_qwen_outputs": protocol.planned_qwen_output_count,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0

    report_path = output_dir / "packet1j_report.json"
    if report_path.is_file():
        progress = load_relationship_p1j_progress(output_dir)
        validate_relationship_p1j_progress(
            progress,
            protocol=protocol,
            consumer=consumer,
            split_bundle=split_bundle,
            contexts=contexts,
        )
        report = load_relationship_p1j_report(report_path)
        validate_relationship_p1j_report_lineage(
            report,
            protocol=protocol,
            consumer=consumer,
            split_bundle=split_bundle,
        )
        validate_relationship_p1j_terminal_files(
            report=report,
            progress=progress,
            output_dir=output_dir,
        )
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "resumed_existing_completion": True,
                    "report_artifact_id": report.artifact_id,
                    "verdict": report.verdict.value,
                    "next_action": report.next_action,
                    "qualification_qwen_outputs": (
                        report.qualification_qwen_output_count
                    ),
                    "consumer_revision_after_qualification": False,
                },
                ensure_ascii=False,
            )
        )
        return _terminal_exit_code(report.verdict)

    progress = load_relationship_p1j_progress(output_dir)
    validate_relationship_p1j_progress(
        progress,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
    )
    policy = HFStatelessRelationshipActionPolicy(
        model_source=consumer.model_source,
        model_id=consumer.model_id,
        device=consumer.device,
        torch_dtype=consumer.torch_dtype,
        local_files_only=True,
        temperature=consumer.temperature,
        top_p=consumer.top_p,
        max_new_tokens=consumer.max_new_tokens,
    )

    def persist_readout(index, readout) -> None:
        persist_relationship_p1j_readout(
            checkpoint=checkpoint,
            output_dir=output_dir,
            index=index,
            readout=readout,
        )

    def persist_decision(index, decision) -> None:
        persist_relationship_p1j_decision(
            checkpoint=checkpoint,
            output_dir=output_dir,
            index=index,
            decision=decision,
        )
        print(
            json.dumps(
                {
                    "stage": "record_checkpointed",
                    "record_index": index,
                    "durable_outputs": index + 1,
                    "planned_outputs": protocol.planned_qwen_output_count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    execution = execute_relationship_p1j_qualification(
        policy,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        existing_progress=progress,
        max_new_readouts=(
            None if args.max_new_readouts == 0 else args.max_new_readouts
        ),
        readout_observer=persist_readout,
        decision_observer=persist_decision,
    )
    del policy
    _release_runtime()
    durable = load_relationship_p1j_progress(output_dir)
    validate_relationship_p1j_progress(
        durable,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
    )
    if (
        execution.readouts != durable.readouts
        or execution.decisions != durable.decisions
    ):
        raise RuntimeError("P1j in-memory execution diverges from durable records")
    if not durable.is_complete:
        print(
            json.dumps(
                {
                    "stage": "checkpointed",
                    "new_qwen_outputs": execution.new_qwen_outputs,
                    "durable_qwen_outputs": len(durable.readouts),
                    "planned_qwen_outputs": protocol.planned_qwen_output_count,
                    "resume_required": True,
                    "consumer_revision_after_qualification": False,
                },
                ensure_ascii=False,
            )
        )
        return 0

    report = assess_relationship_p1j_qualification(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        progress=durable,
    )
    report_path, _markdown_path = write_relationship_p1j_report(
        report=report,
        progress=durable,
        output_dir=output_dir,
    )
    loaded = load_relationship_p1j_report(report_path)
    if loaded != report:
        raise RuntimeError("P1j report round-trip changed artifact identity")
    validate_relationship_p1j_terminal_files(
        report=loaded,
        progress=durable,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(report_path),
                "report_artifact_id": report.artifact_id,
                "verdict": report.verdict.value,
                "metrics": {
                    item.arm: {
                        "valid_rate": item.valid_rate,
                        "accuracy": item.accuracy,
                        "pair_flip_rate": item.pair_flip_rate,
                    }
                    for item in report.arm_metrics
                },
                "next_action": report.next_action,
                "qualification_qwen_outputs": (
                    report.qualification_qwen_output_count
                ),
                "consumer_revision_after_qualification": False,
            },
            ensure_ascii=False,
        )
    )
    return _terminal_exit_code(report.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
