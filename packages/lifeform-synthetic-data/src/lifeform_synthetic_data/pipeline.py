"""Recoverable batch generation pipeline for unified synthetic experience."""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .canonical import canonical_json, stable_hash, write_canonical_json
from .contracts import (
    SCHEMA_VERSION,
    ArtifactRef,
    CorpusManifest,
    CorpusSplit,
    CountEntry,
    ExperienceTrajectory,
    GenerationTier,
    KeyValue,
    QualityRecord,
    QualitySeverity,
    ScenarioBlueprint,
)
from .llm import (
    BudgetLedger,
    CostLimitExceeded,
    JsonCompletionClient,
    LLMAuthenticationError,
    LLMQuotaError,
    LLMRenderError,
    RateCard,
    TokenUsage,
    estimate_upper_bound_usd,
)
from .live_through import (
    LifeformFactory,
    default_lifeform_factory,
    live_through_trajectory,
)
from .prompt_manager import build_render_prompt
from .renderer import render_trajectory
from .scenario import validate_unified_v1_package
from .storage import (
    AppendOnlyJournal,
    CompletedRecord,
    ContentAddressedStore,
    sha256_file,
    write_quarantine_record,
    write_run_config,
)
from .world import GENERATOR_VERSION, compile_structural_trajectory


def _run_default_live_worker(
    trajectory: ExperienceTrajectory,
) -> ExperienceTrajectory:
    return live_through_trajectory(
        trajectory,
        lifeform_factory=default_lifeform_factory,
    )


@dataclass(frozen=True)
class GenerationRunConfig:
    output_root: Path
    run_id: str
    created_at: str
    git_sha: str
    generation_tier: GenerationTier
    base_seed: int
    split_replicates: tuple[tuple[CorpusSplit, int], ...]
    shard_size: int = 256
    concurrency: int = 1
    max_cost_usd: float = 0.0
    max_output_tokens: int = 4096
    export_parquet: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.created_at.strip():
            raise ValueError("created_at must be non-empty")
        if not self.git_sha.strip():
            raise ValueError("git_sha must be non-empty")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if not self.split_replicates:
            raise ValueError("split_replicates must be non-empty")
        splits = tuple(item[0] for item in self.split_replicates)
        if len(splits) != len(set(splits)):
            raise ValueError("split_replicates cannot repeat a split")
        if any(count < 0 for _, count in self.split_replicates):
            raise ValueError("split replicate counts must be non-negative")
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    def replicates_for(self, split: CorpusSplit) -> int:
        entries = dict(self.split_replicates)
        if split not in entries:
            return 0
        return entries[split]


@dataclass(frozen=True)
class GenerationJob:
    ordinal: int
    blueprint: ScenarioBlueprint
    replicate_index: int
    seed: int
    trajectory_id: str
    source_trajectory: ExperienceTrajectory | None = None


@dataclass(frozen=True)
class CostEstimate:
    pending_calls: int
    conservative_upper_bound_usd: float
    max_cost_usd: float
    within_budget: bool


@dataclass(frozen=True)
class GenerationRunResult:
    run_root: Path
    planned_count: int
    completed_count: int
    resumed_count: int
    quarantined_count: int
    shard_paths: tuple[Path, ...]
    manifest_path: Path
    actual_cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class CorpusGenerationPipeline:
    def __init__(
        self,
        *,
        config: GenerationRunConfig,
        blueprints: tuple[ScenarioBlueprint, ...],
        clients: tuple[JsonCompletionClient, ...] = (),
        rate_card: RateCard | None = None,
        scenario_package_hash: str | None = None,
        lifeform_factory: LifeformFactory | None = None,
        source_trajectories: tuple[ExperienceTrajectory, ...] = (),
        structural_enricher: (Callable[[ExperienceTrajectory], ExperienceTrajectory] | None) = None,
    ) -> None:
        if not blueprints:
            raise ValueError("blueprints must be non-empty")
        if config.generation_tier is GenerationTier.RENDERED:
            if not clients:
                raise ValueError("rendered generation requires at least one client")
            if rate_card is None:
                raise ValueError("rendered generation requires an explicit rate card")
        elif clients:
            raise ValueError("LLM clients are only valid for rendered generation")
        if structural_enricher is not None and config.generation_tier is not GenerationTier.RENDERED:
            raise ValueError("structural enrichment is only valid for rendered generation")
        if source_trajectories and config.generation_tier is not GenerationTier.LIVE_THROUGH:
            raise ValueError("source trajectories are only valid for live-through generation")
        if any(trajectory.generation_tier is not GenerationTier.RENDERED for trajectory in source_trajectories):
            raise ValueError("live-through source trajectories must be rendered")
        self._config = config
        self._blueprints = tuple(sorted(blueprints, key=lambda item: item.scenario_id))
        self._source_trajectories = tuple(sorted(source_trajectories, key=lambda item: item.trajectory_id))
        self._structural_enricher = structural_enricher
        self._clients = clients
        self._rate_card = rate_card
        self._uses_default_lifeform_factory = lifeform_factory is None
        self._lifeform_factory = lifeform_factory or default_lifeform_factory
        self._scenario_package_hash = (
            scenario_package_hash if scenario_package_hash is not None else validate_unified_v1_package().package_hash
        )
        self._run_root = config.output_root / config.run_id
        self._journal = AppendOnlyJournal(self._run_root / "journal.jsonl")
        self._store = ContentAddressedStore(self._run_root)
        self._quarantine_lock = threading.Lock()
        self._budget = (
            BudgetLedger(
                max_cost_usd=config.max_cost_usd,
                rate_card=rate_card,
            )
            if rate_card is not None
            else None
        )

    def plan_jobs(self) -> tuple[GenerationJob, ...]:
        if self._source_trajectories:
            blueprints = {item.scenario_id: item for item in self._blueprints}
            jobs: list[GenerationJob] = []
            for ordinal, trajectory in enumerate(self._source_trajectories):
                blueprint = blueprints.get(trajectory.scenario_ref)
                if blueprint is None:
                    raise ValueError(f"live-through source references unknown scenario: {trajectory.scenario_ref}")
                if trajectory.split is not blueprint.split:
                    raise ValueError(f"live-through source split does not match blueprint: {trajectory.trajectory_id}")
                jobs.append(
                    GenerationJob(
                        ordinal=ordinal,
                        blueprint=blueprint,
                        replicate_index=int({item.key: item.value for item in trajectory.metadata}["replicate_index"]),
                        seed=trajectory.provenance.seed,
                        trajectory_id=trajectory.trajectory_id,
                        source_trajectory=trajectory,
                    )
                )
            return tuple(jobs)
        jobs: list[GenerationJob] = []
        ordinal = 0
        for blueprint in self._blueprints:
            replicate_count = self._config.replicates_for(blueprint.split)
            for replicate_index in range(replicate_count):
                seed = self._config.base_seed + ordinal
                trajectory_id = f"trajectory:{blueprint.scenario_id}:{replicate_index:05d}:{seed:010d}"
                jobs.append(
                    GenerationJob(
                        ordinal=ordinal,
                        blueprint=blueprint,
                        replicate_index=replicate_index,
                        seed=seed,
                        trajectory_id=trajectory_id,
                    )
                )
                ordinal += 1
        return tuple(jobs)

    def estimate_cost(
        self,
        *,
        pending_jobs: tuple[GenerationJob, ...] | None = None,
    ) -> CostEstimate:
        jobs = pending_jobs if pending_jobs is not None else self.plan_jobs()
        if self._config.generation_tier is not GenerationTier.RENDERED:
            return CostEstimate(
                pending_calls=0,
                conservative_upper_bound_usd=0.0,
                max_cost_usd=self._config.max_cost_usd,
                within_budget=True,
            )
        if self._rate_card is None:
            raise ValueError("rendered cost estimation requires a rate card")
        if self._rate_card.input_usd_per_million == 0.0 and self._rate_card.output_usd_per_million == 0.0:
            return CostEstimate(
                pending_calls=len(jobs),
                conservative_upper_bound_usd=0.0,
                max_cost_usd=self._config.max_cost_usd,
                within_budget=True,
            )
        total = 0.0
        for job in jobs:
            structural = self._compile(job)
            prompt = build_render_prompt(structural)
            total += estimate_upper_bound_usd(
                system_prompt=prompt.system_prompt,
                user_prompts=(prompt.user_prompt,),
                max_output_tokens=self._config.max_output_tokens,
                rate_card=self._rate_card,
            )
        return CostEstimate(
            pending_calls=len(jobs),
            conservative_upper_bound_usd=total,
            max_cost_usd=self._config.max_cost_usd,
            within_budget=total <= self._config.max_cost_usd + 1e-12,
        )

    def run(self) -> GenerationRunResult:
        jobs = self.plan_jobs()
        self._write_or_verify_run_config(jobs)
        completed = self._journal.completed()
        resumed_records: dict[str, CompletedRecord] = {}
        pending: list[GenerationJob] = []
        planned_ids = {job.trajectory_id for job in jobs}
        unexpected = sorted(set(completed) - planned_ids)
        if unexpected:
            raise ValueError(f"resume journal contains jobs absent from this plan: {unexpected[:5]}")
        for job in jobs:
            record = completed.get(job.trajectory_id)
            if record is None:
                pending.append(job)
                continue
            self._store.load(record)
            resumed_records[job.trajectory_id] = record
        self._restore_budget(tuple(resumed_records.values()))
        estimate = self.estimate_cost(pending_jobs=tuple(pending))
        budget_snapshot = self._budget.snapshot() if self._budget is not None else None
        settled = budget_snapshot.settled_cost_usd if budget_snapshot is not None else 0.0
        projected = settled + estimate.conservative_upper_bound_usd
        if projected > self._config.max_cost_usd + 1e-12:
            raise CostLimitExceeded(
                f"preflight upper bound ${projected:.6f} exceeds "
                f"--max-cost-usd ${self._config.max_cost_usd:.6f}; "
                f"pending_calls={estimate.pending_calls}"
            )

        newly_completed, quarantined_count = self._run_pending(tuple(pending))
        all_records = {
            **resumed_records,
            **{record.trajectory_id: record for record in newly_completed},
        }
        ordered_records = tuple(all_records[job.trajectory_id] for job in jobs if job.trajectory_id in all_records)
        shards = self._store.materialize_master_shards(
            ordered_records,
            shard_size=self._config.shard_size,
        )
        if self._config.export_parquet:
            self._export_parquet(ordered_records)
        manifest = self._build_manifest(
            jobs=jobs,
            records=ordered_records,
            shards=shards,
            quarantined_count=quarantined_count,
        )
        manifest_path = self._run_root / "run-manifest.json"
        write_canonical_json(manifest_path, manifest)
        prompt_tokens = sum(record.prompt_tokens for record in ordered_records)
        completion_tokens = sum(record.completion_tokens for record in ordered_records)
        return GenerationRunResult(
            run_root=self._run_root,
            planned_count=len(jobs),
            completed_count=len(ordered_records),
            resumed_count=len(resumed_records),
            quarantined_count=quarantined_count,
            shard_paths=shards,
            manifest_path=manifest_path,
            actual_cost_usd=sum(record.cost_usd for record in ordered_records),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _run_pending(
        self,
        jobs: tuple[GenerationJob, ...],
    ) -> tuple[tuple[CompletedRecord, ...], int]:
        if not jobs:
            return (), 0
        if self._config.generation_tier is GenerationTier.LIVE_THROUGH and self._uses_default_lifeform_factory:
            return self._run_pending_live_processes(jobs), 0
        completed: list[CompletedRecord] = []
        quarantined_count = 0
        iterator = iter(jobs)
        active: dict[concurrent.futures.Future[CompletedRecord | None], GenerationJob] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.concurrency) as executor:
            for _ in range(min(self._config.concurrency, len(jobs))):
                job = next(iterator)
                active[executor.submit(self._run_job, job)] = job
            while active:
                done, _ = concurrent.futures.wait(
                    active,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    active.pop(future)
                    try:
                        record = future.result()
                    except (
                        LLMAuthenticationError,
                        LLMQuotaError,
                        CostLimitExceeded,
                    ):
                        for pending_future in active:
                            pending_future.cancel()
                        raise
                    if record is None:
                        quarantined_count += 1
                    else:
                        completed.append(record)
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        continue
                    active[executor.submit(self._run_job, next_job)] = next_job
        return tuple(completed), quarantined_count

    def _run_pending_live_processes(
        self,
        jobs: tuple[GenerationJob, ...],
    ) -> tuple[CompletedRecord, ...]:
        completed: list[CompletedRecord] = []
        iterator = iter(jobs)
        worker_count = min(
            self._config.concurrency,
            os.cpu_count() or 1,
            len(jobs),
        )
        active: dict[
            concurrent.futures.Future[ExperienceTrajectory],
            GenerationJob,
        ] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            for _ in range(worker_count):
                job = next(iterator)
                active[
                    executor.submit(
                        _run_default_live_worker,
                        self._compile(job),
                    )
                ] = job
            while active:
                done, _ = concurrent.futures.wait(
                    active,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    active.pop(future)
                    try:
                        trajectory = future.result()
                    except BaseException:
                        for pending_future in active:
                            pending_future.cancel()
                        raise
                    record = self._store.put_trajectory(trajectory)
                    self._journal.append_completed(
                        record,
                        timestamp=self._config.created_at,
                    )
                    completed.append(record)
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        continue
                    active[
                        executor.submit(
                            _run_default_live_worker,
                            self._compile(next_job),
                        )
                    ] = next_job
        return tuple(completed)

    def _run_job(self, job: GenerationJob) -> CompletedRecord | None:
        structural = self._compile(job)
        completion_cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        trajectory = structural
        if self._config.generation_tier is GenerationTier.RENDERED:
            if self._budget is None:
                raise ValueError("rendered run has no budget ledger")
            client = self._clients[job.ordinal % len(self._clients)]
            try:
                trajectory, completion, _ = render_trajectory(
                    structural,
                    client=client,
                    budget=self._budget,
                    max_output_tokens=self._config.max_output_tokens,
                )
            except (
                LLMAuthenticationError,
                LLMQuotaError,
                CostLimitExceeded,
            ):
                raise
            except LLMRenderError as error:
                self._quarantine(job, error)
                return None
            completion_cost = completion.cost_usd
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
        elif self._config.generation_tier is GenerationTier.LIVE_THROUGH:
            trajectory = live_through_trajectory(
                structural,
                lifeform_factory=self._lifeform_factory,
            )
        stored = self._store.put_trajectory(trajectory)
        record = replace(
            stored,
            cost_usd=completion_cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._journal.append_completed(record, timestamp=self._config.created_at)
        return record

    def _compile(self, job: GenerationJob):
        if job.source_trajectory is not None:
            source = job.source_trajectory
            return replace(
                source,
                metadata=source.metadata
                + (
                    KeyValue(
                        key="source_master_trajectory_hash",
                        value=stable_hash(source),
                    ),
                    KeyValue(
                        key="source_master_run_id",
                        value=source.provenance.run_id,
                    ),
                ),
                provenance=replace(
                    source.provenance,
                    run_id=self._config.run_id,
                    created_at=self._config.created_at,
                    git_sha=self._config.git_sha,
                ),
            )
        trajectory = compile_structural_trajectory(
            job.blueprint,
            replicate_index=job.replicate_index,
            seed=job.seed,
            run_id=self._config.run_id,
            created_at=self._config.created_at,
            git_sha=self._config.git_sha,
        )
        if self._structural_enricher is not None:
            trajectory = self._structural_enricher(trajectory)
            if trajectory.generation_tier is not GenerationTier.STRUCTURAL:
                raise ValueError("structural enricher must preserve the structural tier")
        return trajectory

    def _quarantine(self, job: GenerationJob, error: LLMRenderError) -> None:
        message = str(error)
        with self._quarantine_lock:
            write_quarantine_record(
                self._run_root / "quarantine.jsonl",
                trajectory_id=job.trajectory_id,
                error_type=type(error).__name__,
                message=message,
                timestamp=self._config.created_at,
            )
        self._journal.append_quarantined(
            trajectory_id=job.trajectory_id,
            error_type=type(error).__name__,
            message=message,
            timestamp=self._config.created_at,
        )

    def _restore_budget(self, records: tuple[CompletedRecord, ...]) -> None:
        if self._budget is None:
            return
        prompt_tokens = sum(record.prompt_tokens for record in records)
        completion_tokens = sum(record.completion_tokens for record in records)
        self._budget.restore(
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            calls=len(records),
        )

    def _write_or_verify_run_config(
        self,
        jobs: tuple[GenerationJob, ...],
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._config.run_id,
            "created_at": self._config.created_at,
            "git_sha": self._config.git_sha,
            "generation_tier": self._config.generation_tier.value,
            "generator_version": GENERATOR_VERSION,
            "base_seed": self._config.base_seed,
            "split_replicates": [
                {"split": split.value, "count": count} for split, count in self._config.split_replicates
            ],
            "shard_size": self._config.shard_size,
            "max_output_tokens": self._config.max_output_tokens,
            "scenario_package_hash": self._scenario_package_hash,
            "scenario_ids": [item.scenario_id for item in self._blueprints],
            "planned_trajectory_ids_hash": _hash_strings(tuple(job.trajectory_id for job in jobs)),
            "model_ids": [client.model_id for client in self._clients],
            "rate_card": (
                {
                    "input_usd_per_million": self._rate_card.input_usd_per_million,
                    "output_usd_per_million": self._rate_card.output_usd_per_million,
                    "currency": self._rate_card.currency,
                }
                if self._rate_card is not None
                else None
            ),
        }
        if self._source_trajectories:
            payload["source_master_trajectory_hashes_hash"] = _hash_strings(
                tuple(stable_hash(item) for item in self._source_trajectories)
            )
        write_run_config(self._run_root / "run-config.json", payload)

    def _build_manifest(
        self,
        *,
        jobs: tuple[GenerationJob, ...],
        records: tuple[CompletedRecord, ...],
        shards: tuple[Path, ...],
        quarantined_count: int,
    ) -> CorpusManifest:
        record_ids = {record.trajectory_id for record in records}
        completed_jobs = tuple(job for job in jobs if job.trajectory_id in record_ids)
        split_counts = Counter(job.blueprint.split.value for job in completed_jobs)
        family_counts = Counter(job.blueprint.family for job in completed_jobs)
        artifact_refs = tuple(
            ArtifactRef(
                artifact_id=f"shard:{index:05d}",
                kind="master_jsonl_gzip",
                uri=path.relative_to(self._run_root).as_posix(),
                sha256=sha256_file(path),
                mime_type="application/x-ndjson+gzip",
                license_id="Proprietary-Synthetic-v1",
            )
            for index, path in enumerate(shards)
        )
        prompt_hashes = sorted({record.prompt_hash for record in records if record.prompt_hash is not None})
        hard_pass = quarantined_count == 0 and len(records) == len(jobs)
        quality = (
            QualityRecord(
                quality_id=f"quality:{self._config.run_id}:completion",
                check_kind="planned_trajectory_completion",
                passed=hard_pass,
                severity=(QualitySeverity.INFO if hard_pass else QualitySeverity.ERROR),
                score=(len(records) / len(jobs)) if jobs else 1.0,
                evidence_refs=("journal.jsonl", "quarantine.jsonl"),
                description=(f"completed={len(records)}, planned={len(jobs)}, quarantined={quarantined_count}"),
            ),
        )
        return CorpusManifest(
            schema_version=SCHEMA_VERSION,
            corpus_id="unified_synthetic_experience_v1",
            run_id=self._config.run_id,
            generated_at=self._config.created_at,
            generator_version=GENERATOR_VERSION,
            git_sha=self._config.git_sha,
            scenario_package_hash=self._scenario_package_hash,
            generation_tier=self._config.generation_tier,
            trajectory_count=len(records),
            split_counts=tuple(CountEntry(key=split.value, count=split_counts[split.value]) for split in CorpusSplit),
            family_counts=tuple(
                CountEntry(key=family, count=family_counts[family])
                for family in sorted({item.family for item in self._blueprints})
            ),
            model_ids=tuple(sorted({record.model_id for record in records if record.model_id})),
            prompt_hashes=tuple(
                KeyValue(key=f"render_slots_{index}", value=prompt_hash)
                for index, prompt_hash in enumerate(prompt_hashes)
            ),
            shard_refs=artifact_refs,
            quality=quality,
            description=(
                "Unified synthetic experience corpus. Generator truth, "
                "rendered text, and runtime observations remain separate."
            ),
        )

    def _export_parquet(self, records: tuple[CompletedRecord, ...]) -> Path:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Parquet export requires lifeform-synthetic-data[parquet]") from error
        rows = []
        for record in records:
            trajectory = self._store.load(record)
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "scenario_ref": trajectory.scenario_ref,
                    "split": trajectory.split.value,
                    "family": trajectory.family,
                    "generation_tier": trajectory.generation_tier.value,
                    "trajectory_hash": record.trajectory_hash,
                    "trajectory_json": canonical_json(trajectory),
                }
            )
        table = pa.Table.from_pylist(rows)
        output = self._run_root / "master" / "master.parquet"
        pq.write_table(table, output, compression="zstd")
        return output


def _hash_strings(values: tuple[str, ...]) -> str:
    import hashlib

    payload = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CorpusGenerationPipeline",
    "CostEstimate",
    "GenerationJob",
    "GenerationRunConfig",
    "GenerationRunResult",
]
