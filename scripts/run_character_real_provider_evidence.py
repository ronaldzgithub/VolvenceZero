"""Run the Zhang Wuji multi-family proof with a local real text model.

The model only receives the production action-abstraction and action-
applicability prompts. Other semantic-owner prompts are explicitly excluded
by ``ActionEvidenceOnlyTextProvider`` so this lane measures the two remaining
structured-provider dependencies without changing their owners.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

from lifeform_core import LifeformConfig
from lifeform_domain_character import (
    ActionEvidenceOnlyTextProvider,
    BehaviorFamilyPortfolioReport,
    BehaviorFamilyRoutingObservation,
    BehaviorFidelityStimulus,
    ChapterLiveThroughDriver,
    build_character_lifeform,
    build_zhang_wuji_profile,
    capture_behavior_fidelity_async,
    evaluate_behavior_family_portfolio,
    evaluate_real_provider_behavior_evidence,
    read_ledger_json,
)
from lifeform_expression import GroundedResponseSynthesizer
from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    CaseActionAbstractionPromotion,
    CaseMemoryRecord,
    build_filesystem_persistence_backend,
)
from volvence_zero.brain import BrainConfig
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    build_default_memory_store,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state.llm_runtime import (
    LLMSemanticProposalRuntime,
)
from volvence_zero.substrate import (
    build_transformers_runtime_with_fallback,
)
from volvence_zero.substrate.text_generation import (
    HFTextGenerationProvider,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEDGER = (
    _REPO_ROOT
    / "artifacts"
    / "character-live-through"
    / "zhang_wuji.reviewed_ledger.json"
)
_BOUNDARY_REFERENCE = "withhold-disclosure-until-moral-clarity"
_PROTECTION_REFERENCE = "intervene-immediately-to-protect-life"
_SELECTED_SCENES = (
    ("ch-8", 0),
    ("ch-9", 0),
    ("ch-10", 2),
    ("ch-11", 0),
    ("ch-12", 3),
    ("ch-26", 1),
    ("ch-30", 1),
)


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args,
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _provenance(
    *,
    model_id: str,
    model_path: Path,
    device: str,
) -> dict[str, object]:
    status = _git_output(("status", "--porcelain"))
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "git_branch": _git_output(("branch", "--show-current")),
        "working_tree_dirty": status not in {"", "unknown"},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "provider_kind": "huggingface-local-frozen",
        "model_id": model_id,
        "model_snapshot": model_path.name,
        "device": device,
        "temperature": 0.0,
        "network_allowed": False,
    }


def _held_out_profile():
    profile = build_zhang_wuji_profile()
    return replace(
        profile,
        signature_cases=tuple(
            case
            for case in profile.signature_cases
            if case.case_id != "protecting-bystander-from-collateral"
        ),
        strategy_priors=tuple(
            prior
            for prior in profile.strategy_priors
            if prior.rule_id != "crisis-decisive-when-bystander-at-risk"
        ),
    )


def _config(*, application_dir: Path) -> LifeformConfig:
    return LifeformConfig(
        brain_config=BrainConfig(
            application_persistence_dir=str(application_dir),
            temporal_latent_dim=4,
            final_rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.ACTIVE,
                internal_rl_runtime_segment_credit=WiringLevel.ACTIVE,
                internal_rl_batch_accumulation_size=1,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            rare_heavy_enabled=False,
        )
    )


def _case_store(*, application_dir: Path) -> ApplicationCaseMemoryStore:
    return ApplicationCaseMemoryStore(
        persistence_backend=build_filesystem_persistence_backend(
            base_dir=str(application_dir / "case_memory")
        )
    )


def _case_store_digest(*, application_dir: Path) -> str:
    store = _case_store(application_dir=application_dir)
    if not store.load_from_backend():
        raise RuntimeError("case-memory source state was not persisted")
    return hashlib.sha256(repr(store.records).encode()).hexdigest()


def _clone_case_store(
    *,
    source_application_dir: Path,
    sandbox_application_dir: Path,
) -> None:
    source = _case_store(application_dir=source_application_dir)
    if not source.load_from_backend():
        raise RuntimeError("case-memory source state was not persisted")
    sandbox = _case_store(application_dir=sandbox_application_dir)
    sandbox.restore_checkpoint(
        source.create_checkpoint(
            checkpoint_id="real-provider-routing-clone"
        )
    )
    if not sandbox.save_to_backend():
        raise RuntimeError("case-memory routing clone was not persisted")


def _held_out_stimuli() -> tuple[BehaviorFidelityStimulus, ...]:
    return (
        BehaviorFidelityStimulus(
            case_id="behavior-family:boundary:sealed-route",
            character_id="zhang-wuji",
            scene_id="real-provider-boundary",
            phase_label="mature",
            setting=(
                "At an unfamiliar council, a former ally presses you to "
                "reveal another person's entrusted act and declare final "
                "judgment immediately."
            ),
            decision_point=(
                "The evidence remains incomplete and you are not authorized "
                "to disclose the confidence. Decide what you say now."
            ),
            evidence_locator="reviewed-real-provider:boundary:v1",
        ),
        BehaviorFidelityStimulus(
            case_id="behavior-family:protection:falling-beam",
            character_id="zhang-wuji",
            scene_id="real-provider-protection",
            phase_label="mature",
            setting=(
                "Inside an unfamiliar storehouse, a support beam starts "
                "falling toward an injured porter who cannot move away."
            ),
            decision_point=(
                "The beam will strike within seconds. Decide what concrete "
                "action you take before asking how the accident happened."
            ),
            evidence_locator="reviewed-real-provider:protection:v1",
        ),
    )


def _promotion_role(
    promotion: CaseActionAbstractionPromotion,
) -> str | None:
    outcome_ids = promotion.source_outcome_ids
    if (
        any("ch-11-scene-1" in item for item in outcome_ids)
        and any("ch-12-scene-4" in item for item in outcome_ids)
    ):
        return "protection"
    if (
        any(
            "ch26-winehouse-identity-revelation" in item
            for item in outcome_ids
        )
        and any("ch-30-scene-2" in item for item in outcome_ids)
    ):
        return "boundary"
    return None


def _failure_portfolio(
    *,
    promotions: tuple[CaseActionAbstractionPromotion, ...],
    pending_family_ids: tuple[str, ...],
) -> BehaviorFamilyPortfolioReport:
    return evaluate_behavior_family_portfolio(
        suite_id="zhang-wuji-real-structured-provider-v1",
        expected_schema_ids=(
            _BOUNDARY_REFERENCE,
            _PROTECTION_REFERENCE,
        ),
        promotions=promotions,
        routing_observations=(),
        pending_family_ids=pending_family_ids,
    )


def _load_promotions(
    *,
    application_dir: Path,
) -> tuple[
    ApplicationCaseMemoryStore,
    tuple[CaseMemoryRecord, ...],
    tuple[CaseActionAbstractionPromotion, ...],
]:
    store = _case_store(application_dir=application_dir)
    if not store.load_from_backend():
        raise RuntimeError("real-provider case-memory state was not persisted")
    promotion_records = tuple(
        record
        for record in store.records
        if record.action_abstraction_promotion is not None
    )
    promotions = tuple(
        record.action_abstraction_promotion
        for record in promotion_records
        if record.action_abstraction_promotion is not None
    )
    return store, promotion_records, promotions


def _run(
    *,
    provider: ActionEvidenceOnlyTextProvider,
    workspace: Path,
) -> tuple[
    BehaviorFamilyPortfolioReport,
    dict[str, tuple[str, ...]],
    tuple[CaseActionAbstractionPromotion, ...],
]:
    ledger = read_ledger_json(_LEDGER)
    chapters = {
        chapter.chapter_id: chapter for chapter in ledger.chapters
    }
    source_application_dir = workspace / "source-application"
    memory_backend = FileSystemPersistenceBackend(
        base_dir=str(workspace / "memory")
    )
    family_ids_by_scene: dict[str, tuple[str, ...]] = {}
    for ordinal, (chapter_id, scene_index) in enumerate(_SELECTED_SCENES):
        bundle = build_character_lifeform(
            _held_out_profile(),
            config=_config(application_dir=source_application_dir),
            memory_store=build_default_memory_store(
                persistence_backend=memory_backend
            ),
            response_synthesizer=GroundedResponseSynthesizer(),
            semantic_proposal_runtime=LLMSemanticProposalRuntime(
                provider=provider
            ),
        )
        chapter = chapters[chapter_id]
        scene = replace(
            chapter.scenes[scene_index],
            canonical_action_schema=None,
        )
        report = ChapterLiveThroughDriver().run_ledger(
            ledger=replace(
                ledger,
                chapters=(
                    replace(
                        chapter,
                        scenes=(scene,),
                        semantic_events=(),
                    ),
                ),
            ),
            lifeform=bundle.lifeform,
            session_id=f"real-provider-{ordinal}-{chapter_id}",
        )
        report.require_success()
        family_ids_by_scene[scene.scene_id] = (
            report.per_scene_evidence[0].experienced_action_family_ids
        )
        print(
            f"[live-through] {ordinal + 1}/{len(_SELECTED_SCENES)} "
            f"{scene.scene_id} families={family_ids_by_scene[scene.scene_id]}",
            flush=True,
        )
        if chapter_id == "ch-12":
            (
                checkpoint_store,
                _records,
                checkpoint_promotions,
            ) = _load_promotions(
                application_dir=source_application_dir
            )
            if not any(
                _promotion_role(promotion) == "protection"
                for promotion in checkpoint_promotions
            ):
                print(
                    "[exit] protection family was not promoted; "
                    "later families and held-out routing remain untested",
                    flush=True,
                )
                return (
                    _failure_portfolio(
                        promotions=checkpoint_promotions,
                        pending_family_ids=tuple(
                            sorted(
                                {
                                    item.action_family_id
                                    for item in (
                                        checkpoint_store
                                        .pending_action_abstraction_evidence()
                                    )
                                }
                            )
                        ),
                    ),
                    family_ids_by_scene,
                    checkpoint_promotions,
                )

    store, promotion_records, promotions = _load_promotions(
        application_dir=source_application_dir
    )
    by_role = {
        role: promotion
        for promotion in promotions
        if (role := _promotion_role(promotion)) is not None
    }
    if set(by_role) != {"boundary", "protection"}:
        return (
            _failure_portfolio(
                promotions=promotions,
                pending_family_ids=tuple(
                    sorted(
                        {
                            item.action_family_id
                            for item in (
                                store.pending_action_abstraction_evidence()
                            )
                        }
                    )
                ),
            ),
            family_ids_by_scene,
            promotions,
        )

    expected_schema_ids = (
        by_role["boundary"].schema_id,
        by_role["protection"].schema_id,
    )
    source_digest = _case_store_digest(
        application_dir=source_application_dir
    )
    schema_by_case_id = {
        record.case_id: record.action_abstraction_promotion.schema_id
        for record in promotion_records
        if record.action_abstraction_promotion is not None
    }
    routing_observations = []
    for index, (stimulus, expected_schema) in enumerate(
        zip(
            _held_out_stimuli(),
            expected_schema_ids,
            strict=True,
        )
    ):
        sandbox_dir = workspace / "routing-sandboxes" / str(index)
        _clone_case_store(
            source_application_dir=source_application_dir,
            sandbox_application_dir=sandbox_dir,
        )
        sandbox_bundle = build_character_lifeform(
            _held_out_profile(),
            config=_config(application_dir=sandbox_dir),
            memory_store=build_default_memory_store(),
            response_synthesizer=GroundedResponseSynthesizer(),
            semantic_proposal_runtime=LLMSemanticProposalRuntime(
                provider=provider
            ),
        )
        capture = asyncio.run(
            capture_behavior_fidelity_async(
                stimulus=stimulus,
                lifeform=sandbox_bundle.lifeform,
                arm_id="real-structured-provider",
                source_state_sha256_before=source_digest,
                source_state_sha256_after=source_digest,
                source_state_digest_reader=lambda: _case_store_digest(
                    application_dir=source_application_dir
                ),
            )
        )
        routing_observations.append(
            BehaviorFamilyRoutingObservation(
                case_id=stimulus.case_id,
                expected_schema_id=expected_schema,
                selected_schema_id=(
                    schema_by_case_id.get(
                        capture.action_grounding_source_case_id
                    )
                    if capture.action_grounding_source_case_id is not None
                    else None
                ),
                source_state_digest_verified=(
                    capture.source_state_digest_verified
                ),
                outcome_feedback_submitted=(
                    capture.outcome_feedback_submitted
                ),
                evaluation_feedback_submitted=(
                    capture.evaluation_feedback_submitted
                ),
            )
        )
        print(
            f"[held-out] {index + 1}/2 {stimulus.case_id} "
            f"selected={routing_observations[-1].selected_schema_id!r} "
            f"expected={expected_schema!r}",
            flush=True,
        )

    portfolio = evaluate_behavior_family_portfolio(
        suite_id="zhang-wuji-real-structured-provider-v1",
        expected_schema_ids=expected_schema_ids,
        promotions=promotions,
        routing_observations=tuple(routing_observations),
        pending_family_ids=tuple(
            sorted(
                {
                    item.action_family_id
                    for item in store.pending_action_abstraction_evidence()
                }
            )
        ),
    )
    return portfolio, family_ids_by_scene, promotions


def _write_artifacts(
    *,
    output_dir: Path,
    payload: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "real_provider_behavior_evidence.json"
    report_bytes = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    report_path.write_bytes(report_bytes)
    manifest = {
        "schema_version": "real-provider-behavior-evidence-manifest.v1",
        "files": (
            {
                "path": report_path.name,
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "size_bytes": len(report_bytes),
            },
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _REPO_ROOT
            / "artifacts"
            / "character-live-through"
            / "real-provider-v1"
        ),
    )
    args = parser.parse_args()

    print(f"[provider] resolving local model {args.model_id}", flush=True)
    model_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            local_files_only=True,
        )
    )
    runtime = build_transformers_runtime_with_fallback(
        model_id=str(model_path),
        device=args.device,
        local_files_only=True,
        allow_live_substrate_mutation=False,
        fallback_to_builtin=False,
    )
    print(
        f"[provider] loaded snapshot={model_path.name} device={args.device}",
        flush=True,
    )
    raw_provider = HFTextGenerationProvider(
        model=runtime._model,
        tokenizer=runtime._tokenizer,
        device=runtime._device,
    )
    provider_id = f"hf-local:{args.model_id}@{model_path.name}"
    provider = ActionEvidenceOnlyTextProvider(
        provider=raw_provider,
        provider_id=provider_id,
    )

    with TemporaryDirectory(
        prefix="volvence-real-provider-"
    ) as workspace_name:
        portfolio, family_ids_by_scene, promotions = _run(
            provider=provider,
            workspace=Path(workspace_name),
        )
    provider_report = evaluate_real_provider_behavior_evidence(
        provider_id=provider_id,
        traces=provider.traces,
        portfolio=portfolio,
    )
    report_path, manifest_path = _write_artifacts(
        output_dir=args.output_dir,
        payload={
            "schema_version": "real-provider-behavior-evidence.v1",
            "provenance": _provenance(
                model_id=args.model_id,
                model_path=model_path,
                device=args.device,
            ),
            "provider_report": asdict(provider_report),
            "portfolio_report": asdict(portfolio),
            "provider_traces": tuple(
                asdict(trace) for trace in provider.traces
            ),
            "family_ids_by_scene": family_ids_by_scene,
            "promotions": tuple(asdict(item) for item in promotions),
        },
    )
    print(report_path)
    print(manifest_path)
    print(provider_report.claim_status)
    return 0 if provider_report.real_provider_supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
