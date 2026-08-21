#!/usr/bin/env python3
"""Prepare and execute the frozen Relationship Lab P1k oracle disclosure ladder.

P1k is evaluator-only and non-competitive.  It must not run concurrently with
the P1j Qwen process: both load the same frozen 3B substrate on CPU.
"""

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
from lifeform_evolution.relationship_lab_packet1b import (  # noqa: E402
    load_relationship_packet1b_report,
)
from lifeform_evolution.relationship_lab_packet1g import (  # noqa: E402
    load_relationship_packet1g_report,
)
from lifeform_evolution.relationship_lab_packet1i import (  # noqa: E402
    load_relationship_p1i_calibration_protocol,
    load_relationship_p1i_calibration_report,
    load_relationship_p1i_frozen_consumer_protocol,
    validate_relationship_p1i_candidate_files,
    validate_relationship_p1i_frozen_consumer_lineage,
)
from lifeform_evolution.relationship_lab_packet1k import (  # noqa: E402
    RelationshipP1kVerdict,
    assess_relationship_p1k_diagnostic,
    build_relationship_p1k_checkpoint,
    execute_relationship_p1k_diagnostic,
    freeze_relationship_p1k_protocol,
    load_relationship_p1k_checkpoint,
    load_relationship_p1k_progress,
    load_relationship_p1k_protocol,
    load_relationship_p1k_report,
    persist_relationship_p1k_decision,
    persist_relationship_p1k_readout,
    validate_relationship_p1k_progress,
    validate_relationship_p1k_protocol_lineage,
    write_relationship_p1k_checkpoint,
    write_relationship_p1k_protocol,
    write_relationship_p1k_report,
)


_DEFAULT_P1I_ARTIFACT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1i_v3_training_replay_20260820"
)
_DEFAULT_P1G_ARTIFACT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1g_v3_conditioned_top4_20260820"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1k_v3_oracle_ladder_20260821"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-p1i-artifact-dir",
        default=str(_DEFAULT_P1I_ARTIFACT_DIR),
    )
    parser.add_argument(
        "--source-p1g-artifact-dir",
        default=str(_DEFAULT_P1G_ARTIFACT_DIR),
        help="Frozen P1g artifact that owns the exact v3 public context order.",
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
        help="Freeze v3 public inputs, disclosures and protocol, then stop.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only the already-frozen diagnostic attempt.",
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
        raise FileNotFoundError(f"P1k snapshot is empty: {snapshot}")
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
    return consumer, training_view


def _load_source_context_manifest(consumer, training_view, source_p1g_artifact_dir):
    source_dir = pathlib.Path(source_p1g_artifact_dir)
    p1g_report = load_relationship_packet1g_report(source_dir / "packet1g_report.json")
    p1b_report = load_relationship_packet1b_report(
        source_dir / "p1b_candidate" / "packet1b_report.json"
    )
    manifest = load_relationship_p1_context_replay_manifest(
        source_dir / "p1b_candidate" / "contexts.json",
        dataset=training_view.training_dataset,
    )
    if (
        p1g_report.artifact_id != consumer.source_p1g_report_artifact_id
        or manifest.dataset_fingerprint != consumer.training_dataset_fingerprint
        or manifest.background_templates_sha256 != consumer.background_templates_sha256
        or manifest.rag_config_sha256 != consumer.rag_config_sha256
        or p1b_report.evaluated_context_surface_sha256
        != consumer.training_context_surface_sha256
    ):
        raise ValueError("P1k source v3 context lineage diverges from frozen consumer")
    return manifest


def _build_contexts(
    *,
    consumer,
    training_view,
    rag_snapshot: pathlib.Path,
    replay_manifest,
):
    rag_weights = _snapshot_manifest_digest(rag_snapshot)
    if rag_weights != consumer.rag_weights_sha256:
        raise ValueError("P1k BGE weights diverge from frozen consumer")
    delegate = SentenceTransformerEmbedder(model_id=str(rag_snapshot), device="cpu")
    embedder = _FrozenCachedEmbedder(
        delegate=delegate,
        canonical_name=(
            "companion-ref-harness/sentence-transformer:"
            f"{consumer.rag_model_source}:sha256:{rag_weights}"
        ),
    )
    state_root = tempfile.TemporaryDirectory(prefix="relationship-p1k-context-")
    try:
        contexts = build_relationship_p1_context_bundle(
            state_root=pathlib.Path(state_root.name),
            rag_embedder=embedder,
            dataset=training_view.training_dataset,
            background_template_package_name=consumer.training_package_name,
            background_depths=consumer.background_depths,
            rag_top_k=consumer.rag_top_k,
            rag_candidate_surface=RelationshipP1RagCandidateSurface(
                consumer.rag_candidate_surface
            ),
            rag_replay_orders=replay_manifest.rag_orders,
        )
        replay_manifest.validate_model_inputs(contexts)
    finally:
        del embedder
        del delegate
        state_root.cleanup()
        _release_runtime()
    return contexts


def _prepare_attempt(
    *,
    output_dir: pathlib.Path,
    consumer,
    training_view,
    contexts,
    model_weights: str,
) -> tuple[object, object]:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p1k-prepare-", dir=parent) as root:
        staging = pathlib.Path(root) / "attempt"
        staging.mkdir()
        protocol = freeze_relationship_p1k_protocol(
            consumer=consumer,
            dataset=training_view.training_dataset,
            contexts=contexts,
            seed_schedule=consumer.seed_schedule,
        )
        if protocol.weights_sha256 != model_weights:
            raise ValueError("P1k materialized Qwen weights diverge from protocol")
        write_relationship_p1k_protocol(protocol, staging / "packet1k_protocol.json")
        checkpoint = build_relationship_p1k_checkpoint(
            protocol=protocol,
            dataset=training_view.training_dataset,
        )
        write_relationship_p1k_checkpoint(checkpoint=checkpoint, output_dir=staging)
        preflight = {
            "consumer_protocol_id": consumer.protocol_id,
            "diagnostic_protocol_id": protocol.protocol_id,
            "training_dataset_fingerprint": protocol.training_dataset_fingerprint,
            "context_surface_sha256": protocol.context_surface_sha256,
            "model_weights_sha256": model_weights,
            "planned_qwen_outputs": protocol.planned_output_count,
            "observation_count": protocol.observation_count,
            "tiers": list(protocol.tiers),
            "competitive": False,
            "ready": True,
        }
        (staging / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_dir)
    return protocol, checkpoint


def _load_prepared_attempt(
    *,
    output_dir: pathlib.Path,
    consumer,
    training_view,
    rag_snapshot: pathlib.Path,
    source_p1g_artifact_dir: pathlib.Path,
):
    replay_manifest = _load_source_context_manifest(
        consumer,
        training_view,
        source_p1g_artifact_dir,
    )
    contexts = _build_contexts(
        consumer=consumer,
        training_view=training_view,
        rag_snapshot=rag_snapshot,
        replay_manifest=replay_manifest,
    )
    protocol = load_relationship_p1k_protocol(output_dir / "packet1k_protocol.json")
    validate_relationship_p1k_protocol_lineage(
        protocol,
        consumer=consumer,
        dataset=training_view.training_dataset,
        contexts=contexts,
    )
    checkpoint = load_relationship_p1k_checkpoint(output_dir)
    expected_checkpoint = build_relationship_p1k_checkpoint(
        protocol=protocol,
        dataset=training_view.training_dataset,
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("P1k prepared checkpoint diverges from frozen protocol")
    return protocol, checkpoint, contexts


def _terminal_exit_code(verdict: RelationshipP1kVerdict) -> int:
    return 0 if verdict is not RelationshipP1kVerdict.MACHINERY_REGRESSION else 2


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.max_new_readouts < 0:
        raise ValueError("--max-new-readouts must be zero or positive")
    source_dir = pathlib.Path(args.source_p1i_artifact_dir)
    output_dir = pathlib.Path(args.output_dir)

    consumer, training_view = _load_frozen_consumer(source_dir)
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
                    "ready": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 3
    model_weights = frozen_model_weights_sha256(model_snapshot)
    if model_weights != consumer.expected_weights_sha256:
        raise ValueError("P1k Qwen weights diverge from frozen consumer")

    if output_dir.exists():
        if not args.resume and not args.prepare_only:
            raise FileExistsError(
                "P1k diagnostic attempt already exists; only --resume is allowed"
            )
        protocol, checkpoint, contexts = _load_prepared_attempt(
            output_dir=output_dir,
            consumer=consumer,
            training_view=training_view,
            rag_snapshot=rag_snapshot,
            source_p1g_artifact_dir=pathlib.Path(args.source_p1g_artifact_dir),
        )
    else:
        if args.resume:
            raise FileNotFoundError("P1k resume cannot create a second attempt")
        replay_manifest = _load_source_context_manifest(
            consumer,
            training_view,
            pathlib.Path(args.source_p1g_artifact_dir),
        )
        contexts = _build_contexts(
            consumer=consumer,
            training_view=training_view,
            rag_snapshot=rag_snapshot,
            replay_manifest=replay_manifest,
        )
        protocol, checkpoint = _prepare_attempt(
            output_dir=output_dir,
            consumer=consumer,
            training_view=training_view,
            contexts=contexts,
            model_weights=model_weights,
        )
    print(
        json.dumps(
            {
                "stage": "prepared",
                "diagnostic_protocol_id": protocol.protocol_id,
                "consumer_protocol_id": consumer.protocol_id,
                "planned_qwen_outputs": protocol.planned_output_count,
                "tiers": list(protocol.tiers),
                "competitive": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0

    report_path = output_dir / "packet1k_report.json"
    if report_path.is_file():
        progress = load_relationship_p1k_progress(output_dir)
        validate_relationship_p1k_progress(
            progress,
            protocol=protocol,
            dataset=training_view.training_dataset,
            contexts=contexts,
        )
        report = load_relationship_p1k_report(report_path)
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "resumed_existing_completion": True,
                    "report_artifact_id": report.artifact_id,
                    "verdict": report.verdict.value,
                    "next_action": report.next_action,
                },
                ensure_ascii=False,
            )
        )
        return _terminal_exit_code(report.verdict)

    progress = load_relationship_p1k_progress(output_dir)
    validate_relationship_p1k_progress(
        progress,
        protocol=protocol,
        dataset=training_view.training_dataset,
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
        persist_relationship_p1k_readout(
            checkpoint=checkpoint,
            output_dir=output_dir,
            index=index,
            readout=readout,
        )

    def persist_decision(index, decision) -> None:
        persist_relationship_p1k_decision(
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
                    "planned_outputs": protocol.planned_output_count,
                    "tier": decision.tier.value,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    execution = execute_relationship_p1k_diagnostic(
        policy,
        protocol=protocol,
        dataset=training_view.training_dataset,
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
    durable = load_relationship_p1k_progress(output_dir)
    validate_relationship_p1k_progress(
        durable,
        protocol=protocol,
        dataset=training_view.training_dataset,
        contexts=contexts,
    )
    if (
        execution.readouts != durable.readouts
        or execution.decisions != durable.decisions
    ):
        raise RuntimeError("P1k in-memory execution diverges from durable records")
    if not durable.is_complete:
        print(
            json.dumps(
                {
                    "stage": "checkpointed",
                    "new_qwen_outputs": execution.new_outputs,
                    "durable_qwen_outputs": len(durable.readouts),
                    "planned_qwen_outputs": protocol.planned_output_count,
                    "resume_required": True,
                },
                ensure_ascii=False,
            )
        )
        return 0

    report = assess_relationship_p1k_diagnostic(
        protocol=protocol,
        progress=durable,
    )
    report_path, _markdown_path = write_relationship_p1k_report(
        report=report,
        output_dir=output_dir,
    )
    loaded = load_relationship_p1k_report(report_path)
    if loaded != report:
        raise RuntimeError("P1k report round-trip changed artifact identity")
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(report_path),
                "report_artifact_id": report.artifact_id,
                "verdict": report.verdict.value,
                "next_action": report.next_action,
                "metrics": {
                    item.tier: {
                        "accuracy": item.accuracy,
                        "pair_flip_rate": item.pair_flip_rate,
                        "functional": item.functional,
                    }
                    for item in report.tier_metrics
                },
            },
            ensure_ascii=False,
        )
    )
    return _terminal_exit_code(report.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
