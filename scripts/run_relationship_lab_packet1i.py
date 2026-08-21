#!/usr/bin/env python3
"""Run P1i training-only ordinary-Qwen consumer calibration and freeze."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import sys
import tempfile
import time


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
from lifeform_evolution.relationship_lab_baseline import (  # noqa: E402
    HFStatelessRelationshipActionPolicy,
    StatelessActionCompletion,
    frozen_model_weights_sha256,
)
from lifeform_evolution.relationship_lab_contexts import (  # noqa: E402
    RelationshipP1RagCandidateSurface,
    build_relationship_p1_context_bundle,
    load_relationship_p1_context_replay_manifest,
)
from lifeform_evolution.relationship_lab_packet1i import (  # noqa: E402
    assess_relationship_p1i_calibration,
    build_relationship_p1i_candidate_checkpoint,
    finalize_relationship_p1i_candidate_checkpoint,
    freeze_relationship_p1i_consumer_protocol,
    load_relationship_p1i_calibration_protocol,
    load_relationship_p1i_candidate_artifact,
    load_relationship_p1i_candidate_checkpoint,
    load_relationship_p1i_candidate_progress,
    load_relationship_p1i_calibration_report,
    load_relationship_p1i_frozen_consumer_protocol,
    persist_relationship_p1i_decision,
    persist_relationship_p1i_readout,
    relationship_p1i_run_from_progress,
    relationship_p1i_calibration_protocol_path,
    run_relationship_p1i_candidate,
    summarize_relationship_p1i_candidate,
    validate_relationship_p1i_candidate_files,
    validate_relationship_p1i_candidate_progress,
    validate_relationship_p1i_context_lineage,
    validate_relationship_p1i_frozen_consumer_lineage,
    validate_relationship_p1i_local_lineage,
    write_relationship_p1i_candidate_checkpoint,
    write_relationship_p1i_report_and_protocol,
)
from lifeform_evolution.relationship_lab_packet1b import (  # noqa: E402
    RelationshipEvidenceReadout,
    load_relationship_packet1b_report,
)
from lifeform_evolution.relationship_lab_packet1g import (  # noqa: E402
    load_relationship_packet1g_report,
)


_DEFAULT_P1G_ARTIFACT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1g_v3_conditioned_top4_20260820"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(relationship_p1i_calibration_protocol_path()),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            _REPO_ROOT
            / "artifacts"
            / "relationship_lab"
            / f"qwen25_3b_packet1i_v3_training_{int(time.time())}"
        ),
    )
    parser.add_argument(
        "--source-p1g-artifact-dir",
        default=str(_DEFAULT_P1G_ARTIFACT_DIR),
        help="Frozen P1g artifact that owns the exact v3 context order.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow materializing missing frozen Qwen/BGE snapshots.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen lineage and complete v3 context without invoking Qwen.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output directory from durable readout checkpoints.",
    )
    parser.add_argument(
        "--max-new-readouts",
        type=int,
        default=4,
        help="Maximum new Qwen readouts per process; use 0 for an unbounded run.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
        raise FileNotFoundError(f"P1i embedding snapshot is empty: {snapshot}")
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


class _ChunkReadoutBudgetReached(RuntimeError):
    """Stop before the next model call after every completed call is durable."""


class _ReplayLimitedPolicy:
    def __init__(
        self,
        *,
        delegate,
        replay_readouts: tuple[RelationshipEvidenceReadout, ...],
        max_new_readouts: int | None,
    ) -> None:
        self._delegate = delegate
        self._replay_readouts = replay_readouts
        self._max_new_readouts = max_new_readouts
        self._call_index = 0
        self.new_readout_count = 0
        self.model_id = delegate.model_id
        self.weights_sha256 = delegate.weights_sha256
        self.generation_config_sha256 = delegate.generation_config_sha256
        self.prompt_sha256 = delegate.prompt_sha256

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        return self._delegate.choose(current_input=current_input, seed=seed)

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        if self._call_index < len(self._replay_readouts):
            readout = self._replay_readouts[self._call_index]
            if seed != readout.seed:
                raise ValueError("P1i replay seed diverges from checkpoint")
            completion = StatelessActionCompletion(
                raw_output=readout.raw_output,
                chosen_action_id=readout.compiled_action,
                prompt_tokens=readout.prompt_tokens,
                completion_tokens=readout.completion_tokens,
            )
        else:
            if (
                self._max_new_readouts is not None
                and self.new_readout_count >= self._max_new_readouts
            ):
                raise _ChunkReadoutBudgetReached
            completion = self._delegate.choose_from_messages(
                messages=messages,
                seed=seed,
            )
            self.new_readout_count += 1
        self._call_index += 1
        return completion

    def count_tokens(self, text: str) -> int:
        return self._delegate.count_tokens(text)


class _CheckpointObservers:
    def __init__(
        self,
        *,
        checkpoint,
        candidate_dir: pathlib.Path,
        existing_decisions: int,
    ) -> None:
        self._checkpoint = checkpoint
        self._candidate_dir = candidate_dir
        self._existing_decisions = existing_decisions
        self.readout_index = 0
        self.decision_index = 0

    def observe_readout(self, readout) -> None:
        persist_relationship_p1i_readout(
            checkpoint=self._checkpoint,
            candidate_dir=self._candidate_dir,
            index=self.readout_index,
            readout=readout,
        )
        self.readout_index += 1

    def observe_decision(self, decision) -> None:
        record_index = self.decision_index
        persist_relationship_p1i_decision(
            checkpoint=self._checkpoint,
            candidate_dir=self._candidate_dir,
            index=record_index,
            decision=decision,
        )
        self.decision_index += 1
        if record_index >= self._existing_decisions:
            print(
                json.dumps(
                    {
                        "stage": "record_checkpointed",
                        "candidate_id": self._checkpoint.candidate.candidate_id,
                        "record_index": record_index,
                        "durable_decisions": self.decision_index,
                        "planned_readouts": len(
                            self._checkpoint.planned_record_keys
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _load_source_context_manifest(
    protocol,
    training_view,
    source_p1g_artifact_dir: pathlib.Path,
):
    source_dir = pathlib.Path(source_p1g_artifact_dir)
    p1g_report = load_relationship_packet1g_report(
        source_dir / "packet1g_report.json"
    )
    p1b_report = load_relationship_packet1b_report(
        source_dir / "p1b_candidate" / "packet1b_report.json"
    )
    manifest = load_relationship_p1_context_replay_manifest(
        source_dir / "p1b_candidate" / "contexts.json",
        dataset=training_view.training_dataset,
    )
    mismatches = {
        "p1g_report_artifact_id": (
            p1g_report.artifact_id,
            protocol.source_p1g_report_artifact_id,
        ),
        "p1g_consumer_protocol_id": (
            p1g_report.consumer_protocol_id,
            protocol.source_p1g_consumer_protocol_id,
        ),
        "p1b_report_artifact_id": (
            p1b_report.artifact_id,
            p1g_report.p1b_report_artifact_id,
        ),
        "context_bundle_artifact_id": (
            manifest.artifact_id,
            p1b_report.context_bundle_artifact_id,
        ),
        "dataset_fingerprint": (
            manifest.dataset_fingerprint,
            protocol.training_dataset_fingerprint,
        ),
        "background_depths": (
            manifest.background_depths,
            protocol.background_depths,
        ),
        "background_templates_sha256": (
            manifest.background_templates_sha256,
            protocol.background_templates_sha256,
        ),
        "rag_config_sha256": (
            manifest.rag_config_sha256,
            protocol.rag_config_sha256,
        ),
        "evaluated_context_surface_sha256": (
            p1b_report.evaluated_context_surface_sha256,
            protocol.evaluated_context_surface_sha256,
        ),
    }
    drift = sorted(name for name, values in mismatches.items() if values[0] != values[1])
    if drift:
        raise ValueError(f"P1i frozen P1g context lineage mismatch: {drift}")
    return manifest


def _prepare_context(
    protocol,
    training_view,
    rag_snapshot: pathlib.Path,
    *,
    replay_manifest,
):
    rag_weights = _snapshot_manifest_digest(rag_snapshot)
    if rag_weights != protocol.rag_weights_sha256:
        raise ValueError("P1i BGE weights diverge from frozen protocol")
    delegate = SentenceTransformerEmbedder(model_id=str(rag_snapshot), device="cpu")
    embedder = _FrozenCachedEmbedder(
        delegate=delegate,
        canonical_name=(
            "companion-ref-harness/sentence-transformer:"
            f"{protocol.rag_model_source}:sha256:{rag_weights}"
        ),
    )
    state_root = tempfile.TemporaryDirectory(prefix="relationship-p1i-context-")
    try:
        bundle = build_relationship_p1_context_bundle(
            state_root=pathlib.Path(state_root.name),
            rag_embedder=embedder,
            dataset=training_view.training_dataset,
            background_depths=protocol.background_depths,
            rag_top_k=protocol.rag_top_k,
            rag_candidate_surface=RelationshipP1RagCandidateSurface(
                protocol.rag_candidate_surface
            ),
            rag_replay_orders=replay_manifest.rag_orders,
        )
        replay_manifest.validate_model_inputs(bundle)
        validate_relationship_p1i_context_lineage(protocol, bundle=bundle)
    finally:
        del embedder
        del delegate
        state_root.cleanup()
        _release_runtime()
    return bundle


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.max_new_readouts < 0:
        raise ValueError("--max-new-readouts must be zero or positive")
    protocol = load_relationship_p1i_calibration_protocol(pathlib.Path(args.protocol))
    training_view = validate_relationship_p1i_local_lineage(protocol)
    model_snapshot = _materialize_snapshot(
        repo_id=protocol.model_source,
        revision=protocol.model_revision,
        allow_download=args.allow_download,
    )
    rag_snapshot = _materialize_snapshot(
        repo_id=protocol.rag_model_source,
        revision=None,
        allow_download=args.allow_download,
    )
    if model_snapshot is None or rag_snapshot is None:
        print(
            json.dumps(
                {
                    "protocol_id": protocol.protocol_id,
                    "training_dataset_fingerprint": (
                        protocol.training_dataset_fingerprint
                    ),
                    "qualification_inputs_observed": 0,
                    "qualification_qwen_outputs_observed": 0,
                    "model_available": model_snapshot is not None,
                    "rag_available": rag_snapshot is not None,
                    "ready": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 3
    model_weights = frozen_model_weights_sha256(model_snapshot)
    if model_weights != protocol.expected_weights_sha256:
        raise ValueError("P1i Qwen weights diverge from frozen protocol")
    replay_manifest = _load_source_context_manifest(
        protocol,
        training_view,
        pathlib.Path(args.source_p1g_artifact_dir),
    )
    contexts = _prepare_context(
        protocol,
        training_view,
        rag_snapshot,
        replay_manifest=replay_manifest,
    )
    preflight = {
        "protocol_id": protocol.protocol_id,
        "consumer_split_contract_id": protocol.consumer_split_contract_id,
        "training_dataset_fingerprint": protocol.training_dataset_fingerprint,
        "training_context_surface_sha256": (
            protocol.training_context_surface_sha256
        ),
        "source_context_manifest_artifact_id": replay_manifest.artifact_id,
        "candidate_count": len(protocol.candidates),
        "selection_method": protocol.selection_method,
        "model_weights_sha256": model_weights,
        "rag_weights_sha256": protocol.rag_weights_sha256,
        "qualification_inputs_observed": 0,
        "qualification_qwen_outputs_observed": 0,
        "ready": True,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_dir = pathlib.Path(args.output_dir)
    preflight_path = output_dir / "preflight.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"P1i output directory already exists: {output_dir}")
        if not preflight_path.is_file():
            raise FileNotFoundError(f"P1i resume preflight is missing: {preflight_path}")
        existing_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if existing_preflight != preflight:
            raise ValueError("P1i resume preflight diverges from frozen lineage")
    else:
        if args.resume:
            raise FileNotFoundError(f"P1i resume directory is missing: {output_dir}")
        output_dir.mkdir(parents=True)
        preflight_path.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    existing_report_path = output_dir / "packet1i_report.json"
    existing_consumer_path = output_dir / "frozen_consumer_protocol.json"
    if existing_consumer_path.exists() and not existing_report_path.exists():
        raise ValueError("P1i consumer protocol exists without its calibration report")
    if existing_report_path.exists():
        report = load_relationship_p1i_calibration_report(existing_report_path)
        validate_relationship_p1i_candidate_files(
            report=report,
            output_dir=output_dir,
        )
        consumer_protocol = (
            load_relationship_p1i_frozen_consumer_protocol(existing_consumer_path)
            if existing_consumer_path.exists()
            else freeze_relationship_p1i_consumer_protocol(
                calibration_protocol=protocol,
                report=report,
                training_view=training_view,
            )
        )
        validate_relationship_p1i_frozen_consumer_lineage(
            consumer_protocol,
            calibration_protocol=protocol,
            report=report,
            training_view=training_view,
        )
        write_relationship_p1i_report_and_protocol(
            report=report,
            consumer_protocol=consumer_protocol,
            output_dir=output_dir,
        )
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "resumed_existing_completion": True,
                    "report": str(existing_report_path),
                    "report_artifact_id": report.artifact_id,
                    "frozen_consumer_protocol": str(existing_consumer_path),
                    "frozen_consumer_protocol_id": consumer_protocol.protocol_id,
                    "selected_candidate_id": report.selected_candidate_id,
                    "ranking": list(report.ranking),
                    "qualification_inputs_observed": 0,
                    "qualification_qwen_outputs_observed": 0,
                    "next_action": report.next_action,
                },
                ensure_ascii=False,
            )
        )
        return 0

    policy = HFStatelessRelationshipActionPolicy(
        model_source=protocol.model_source,
        model_id=protocol.model_id,
        device=protocol.device,
        torch_dtype=protocol.torch_dtype,
        local_files_only=True,
        temperature=protocol.temperature,
        top_p=protocol.top_p,
        max_new_tokens=protocol.max_new_tokens,
    )
    remaining_budget = (
        None if args.max_new_readouts == 0 else args.max_new_readouts
    )
    candidate_artifacts = []
    for candidate in protocol.candidates:
        candidate_dir = output_dir / f"candidate_{candidate.round_index:02d}"
        summary_path = candidate_dir / "candidate.json"
        if summary_path.is_file():
            artifact = load_relationship_p1i_candidate_artifact(candidate_dir)
            if artifact.candidate != candidate:
                raise ValueError("P1i completed candidate is out of protocol order")
            candidate_artifacts.append(artifact)
            print(
                json.dumps(
                    {
                        "stage": "candidate_reused",
                        "round_index": candidate.round_index,
                        "candidate_id": candidate.candidate_id,
                        "artifact_id": artifact.artifact_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        if remaining_budget == 0:
            print(
                json.dumps(
                    {
                        "stage": "chunk_complete",
                        "new_readouts": args.max_new_readouts,
                        "next_candidate_id": candidate.candidate_id,
                        "resume_required": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _release_runtime()
            return 0
        expected_checkpoint = build_relationship_p1i_candidate_checkpoint(
            policy,
            protocol=protocol,
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
        )
        if candidate_dir.exists():
            checkpoint = load_relationship_p1i_candidate_checkpoint(candidate_dir)
            if checkpoint != expected_checkpoint:
                raise ValueError("P1i candidate checkpoint lineage changed on resume")
        else:
            write_relationship_p1i_candidate_checkpoint(
                checkpoint=expected_checkpoint,
                candidate_dir=candidate_dir,
            )
            checkpoint = expected_checkpoint
        progress = load_relationship_p1i_candidate_progress(candidate_dir)
        validate_relationship_p1i_candidate_progress(
            progress,
            protocol=protocol,
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
        )
        print(
            json.dumps(
                {
                    "stage": "candidate_started",
                    "round_index": candidate.round_index,
                    "candidate_id": candidate.candidate_id,
                    "durable_readouts": len(progress.readouts),
                    "durable_decisions": len(progress.decisions),
                    "planned_readouts": len(checkpoint.planned_record_keys),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        limited_policy = _ReplayLimitedPolicy(
            delegate=policy,
            replay_readouts=progress.readouts,
            max_new_readouts=remaining_budget,
        )
        observers = _CheckpointObservers(
            checkpoint=checkpoint,
            candidate_dir=candidate_dir,
            existing_decisions=len(progress.decisions),
        )
        try:
            run = run_relationship_p1i_candidate(
                limited_policy,
                protocol=protocol,
                candidate=candidate,
                training_view=training_view,
                contexts=contexts,
                readout_observer=observers.observe_readout,
                decision_observer=observers.observe_decision,
            )
        except _ChunkReadoutBudgetReached:
            progress = load_relationship_p1i_candidate_progress(candidate_dir)
            validate_relationship_p1i_candidate_progress(
                progress,
                protocol=protocol,
                candidate=candidate,
                training_view=training_view,
                contexts=contexts,
            )
            print(
                json.dumps(
                    {
                        "stage": "candidate_checkpointed",
                        "round_index": candidate.round_index,
                        "candidate_id": candidate.candidate_id,
                        "new_readouts": limited_policy.new_readout_count,
                        "durable_readouts": len(progress.readouts),
                        "durable_decisions": len(progress.decisions),
                        "planned_readouts": len(checkpoint.planned_record_keys),
                        "resume_required": True,
                        "qualification_inputs_observed": 0,
                        "qualification_qwen_outputs_observed": 0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _release_runtime()
            return 0
        progress = load_relationship_p1i_candidate_progress(candidate_dir)
        validate_relationship_p1i_candidate_progress(
            progress,
            protocol=protocol,
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
        )
        durable_run = relationship_p1i_run_from_progress(
            progress,
            training_view=training_view,
        )
        if durable_run != run:
            raise RuntimeError("P1i in-memory run diverges from durable checkpoint")
        artifact = summarize_relationship_p1i_candidate(run)
        finalize_relationship_p1i_candidate_checkpoint(
            run=run,
            artifact=artifact,
            candidate_dir=candidate_dir,
        )
        candidate_artifacts.append(artifact)
        if remaining_budget is not None:
            remaining_budget -= limited_policy.new_readout_count
        print(
            json.dumps(
                {
                    "stage": "candidate_complete",
                    "round_index": candidate.round_index,
                    "candidate_id": candidate.candidate_id,
                    "artifact_id": artifact.artifact_id,
                    "selection_metrics": dict(artifact.selection_metrics),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    del policy
    _release_runtime()
    report = assess_relationship_p1i_calibration(
        protocol=protocol,
        training_view=training_view,
        candidate_artifacts=tuple(candidate_artifacts),
    )
    consumer_protocol = freeze_relationship_p1i_consumer_protocol(
        calibration_protocol=protocol,
        report=report,
        training_view=training_view,
    )
    report_path, _markdown_path, consumer_path = (
        write_relationship_p1i_report_and_protocol(
            report=report,
            consumer_protocol=consumer_protocol,
            output_dir=output_dir,
        )
    )
    loaded = load_relationship_p1i_calibration_report(report_path)
    if loaded != report:
        raise RuntimeError("P1i strict report round-trip changed artifact identity")
    loaded_consumer = load_relationship_p1i_frozen_consumer_protocol(consumer_path)
    if loaded_consumer != consumer_protocol:
        raise RuntimeError("P1i consumer round-trip changed protocol identity")
    validate_relationship_p1i_frozen_consumer_lineage(
        loaded_consumer,
        calibration_protocol=protocol,
        report=loaded,
        training_view=training_view,
    )
    validate_relationship_p1i_candidate_files(report=loaded, output_dir=output_dir)
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(report_path),
                "report_artifact_id": report.artifact_id,
                "frozen_consumer_protocol": str(consumer_path),
                "frozen_consumer_protocol_id": consumer_protocol.protocol_id,
                "selected_candidate_id": report.selected_candidate_id,
                "ranking": list(report.ranking),
                "qualification_inputs_observed": 0,
                "qualification_qwen_outputs_observed": 0,
                "next_action": report.next_action,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
