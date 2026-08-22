#!/usr/bin/env python3
"""Freeze and execute the first-and-only P1m development qualification."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import pathlib
import sys
import tempfile


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/companion-ref-harness/src",
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
    "packages/vz-cognition/src",
    "packages/vz-contracts/src",
    "packages/vz-memory/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from companion_ref_harness.embed import SentenceTransformerEmbedder  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402
from lifeform_domain_emogpt.lab import (  # noqa: E402
    RelationshipAction,
    build_relationship_p1m_pair_plans,
    canonical_json,
    load_relationship_p1m_generation_recipe,
    load_relationship_transfer_dataset,
    run_p2_development_episode,
    sha256_json,
)
from lifeform_domain_emogpt.relationship_condition_reader import (  # noqa: E402
    PrototypeRelationshipPreferenceForecastRuntime,
)
from lifeform_evolution.relationship_lab_baseline import (  # noqa: E402
    frozen_model_weights_sha256,
)
from lifeform_evolution.relationship_lab_packet1k import (  # noqa: E402
    load_relationship_p1k_report,
)
from lifeform_evolution.relationship_lab_packet1m import (  # noqa: E402
    load_relationship_p1m_generation_attestation,
    load_relationship_p1m_generation_records,
    validate_relationship_p1m_generation_attestation_files,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (  # noqa: E402
    RelationshipP1mQualificationArm,
    RelationshipP1mQualificationDecision,
    RelationshipP1mQualificationPlan,
    RelationshipP1mQualificationProtocol,
    RelationshipP1mQwenPlanRecord,
    RelationshipP1mQwenReadout,
    RelationshipP1mStructuredPlanRecord,
    RelationshipP1mStructuredReadout,
    assess_relationship_p1m_qualification,
    build_relationship_p1m_qualification_plan,
    build_relationship_p1m_reader_artifact,
    load_relationship_p1m_qualification_plan,
    load_relationship_p1m_qualification_protocol,
    load_relationship_p1m_qualification_report,
    relationship_p1m_forced_choice_prompt_path,
    relationship_p1m_forced_choice_request_path,
    relationship_p1m_public_episode,
    validate_relationship_p1m_qualification_report_files,
    write_relationship_p1m_qualification_plan,
    write_relationship_p1m_qualification_protocol,
    write_relationship_p1m_qualification_report,
)
from lifeform_evolution.relationship_lab_packet1m_recovery import (  # noqa: E402
    load_relationship_p1m_generation_recovery_protocol,
)


_QWEN_MODEL_SOURCE = "Qwen/Qwen2.5-3B-Instruct"
_QWEN_MODEL_REVISION = "main"
_QWEN_MODEL_ID = "qwen2.5-3b-instruct-p1m-exact-choice-v1"
_QWEN_EXPECTED_WEIGHTS_SHA256 = (
    "3ccf77de3297aba6772fcb743af28b806d7b7c3e348cc7e8ad729fa98a4146cd"
)
_BGE_MODEL_SOURCE = "BAAI/bge-m3"
_BGE_MODEL_REVISION = "main"
_BGE_MODEL_ID = "bge-m3-p1m-named-condition-reader-v1"
_BGE_EXPECTED_WEIGHTS_SHA256 = (
    "d548612967dcb4d75fb51e37fcfa65f3533a248f5c1157f1e0b338e261fd4b1e"
)
_QWEN_DEVICE = "cpu"
_QWEN_TORCH_DTYPE = "bfloat16"
_SCORING_METHOD = "exact_first_assistant_token_logits_A_vs_B"

_DEFAULT_PACKAGE_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "relationship_transfer_p1m_v1_transport_r5_20260822"
)
_DEFAULT_P1K_REPORT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1k_r1_v3_diagnostic_matrix_20260821"
    / "packet1k_report.json"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1m_v1_qualification_20260822"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default=str(_DEFAULT_PACKAGE_DIR))
    parser.add_argument("--source-p1k-report", default=str(_DEFAULT_P1K_REPORT))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze all inputs, model lineage, record plans and gates, then stop.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an already-frozen first qualification attempt.",
    )
    parser.add_argument(
        "--max-new-qwen-readouts",
        type=int,
        default=8,
        help="Maximum new exact-logit readouts; 0 means all remaining.",
    )
    parser.add_argument(
        "--max-new-structured-readouts",
        type=int,
        default=8,
        help="Maximum new owner/reader readouts after Qwen completes; 0 means all.",
    )
    parser.add_argument(
        "--qwen-batch-size",
        type=int,
        default=4,
        help="Execution-only left-padded batch size; frozen to 4 after first use.",
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
        raise FileNotFoundError(f"empty frozen snapshot: {snapshot}")
    return sha256_json(manifest)


def _materialize_snapshot(
    *,
    repo_id: str,
    revision: str,
    allow_download: bool,
) -> pathlib.Path:
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        if not allow_download:
            raise
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=False,
        )
    return pathlib.Path(resolved)


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        temporary = pathlib.Path(handle.name)
    temporary.replace(target)


def _ledger_text(records: tuple[object, ...]) -> str:
    return "".join(
        canonical_json({**record.to_payload(), "artifact_id": record.artifact_id})
        + "\n"
        for record in records
    )


def _ledger_sha256(records: tuple[object, ...]) -> str:
    return hashlib.sha256(_ledger_text(records).encode("utf-8")).hexdigest()


def _write_ledger(path: pathlib.Path, records: tuple[object, ...]) -> None:
    _atomic_write_text(path, _ledger_text(records))


def _load_jsonl(path: pathlib.Path) -> tuple[object, ...]:
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"ledger contains an empty line: {path}")
    return tuple(json.loads(line) for line in lines)


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


class _FrozenEmbedder:
    def __init__(self, snapshot: pathlib.Path) -> None:
        self._delegate = SentenceTransformerEmbedder(
            model_id=str(snapshot),
            device="cpu",
        )
        self._cache: dict[str, tuple[float, ...]] = {}

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self._cache.get(text)
        if cached is None:
            cached = tuple(float(item) for item in self._delegate.embed(text))
            self._cache[text] = cached
        return cached


class _FrozenQwenInputRenderer:
    def __init__(self, snapshot: pathlib.Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.system_prompt = relationship_p1m_forced_choice_prompt_path().read_text(
            encoding="utf-8"
        ).strip()
        token_a = self.tokenizer("A", add_special_tokens=False)["input_ids"]
        token_b = self.tokenizer("B", add_special_tokens=False)["input_ids"]
        if not isinstance(token_a, list) or len(token_a) != 1:
            raise ValueError("P1m candidate A is not one tokenizer token")
        if not isinstance(token_b, list) or len(token_b) != 1:
            raise ValueError("P1m candidate B is not one tokenizer token")
        self.token_a_id = int(token_a[0])
        self.token_b_id = int(token_b[0])
        if self.token_a_id == self.token_b_id:
            raise ValueError("P1m candidate token ids collide")

    def render(self, request: str) -> tuple[str, int]:
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        return rendered, int(encoded["input_ids"].shape[-1])


class _FrozenQwenExactChoicePolicy:
    def __init__(
        self,
        *,
        snapshot: pathlib.Path,
        input_renderer: _FrozenQwenInputRenderer,
        protocol: RelationshipP1mQualificationProtocol,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        if protocol.qwen_device != _QWEN_DEVICE:
            raise ValueError("P1m exact-choice runtime is CPU-only")
        self._torch = torch
        self._renderer = input_renderer
        self._protocol = protocol
        self._model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(protocol.qwen_device)
        self._model.eval()

    def score(
        self,
        record: RelationshipP1mQwenPlanRecord,
    ) -> RelationshipP1mQwenReadout:
        return self.score_batch((record,))[0]

    def score_batch(
        self,
        records: tuple[RelationshipP1mQwenPlanRecord, ...],
    ) -> tuple[RelationshipP1mQwenReadout, ...]:
        if not records:
            raise ValueError("P1m exact-choice batch cannot be empty")
        rendered_inputs: list[str] = []
        for record in records:
            rendered, prompt_tokens = self._renderer.render(record.request_text)
            if (
                hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                != record.model_input_sha256
                or prompt_tokens != record.prompt_tokens
            ):
                raise ValueError("P1m runtime model input differs from frozen plan")
            rendered_inputs.append(rendered)
        encoded = self._renderer.tokenizer(
            rendered_inputs,
            return_tensors="pt",
            padding=True,
        )
        encoded = {
            key: value.to(self._protocol.qwen_device)
            for key, value in encoded.items()
        }
        with self._torch.inference_mode():
            output = self._model(**encoded)
        batch_logits = output.logits[:, -1].float()
        readouts: list[RelationshipP1mQwenReadout] = []
        for batch_index, record in enumerate(records):
            logits = batch_logits[batch_index]
            logit_a = float(logits[self._protocol.token_a_id].item())
            logit_b = float(logits[self._protocol.token_b_id].item())
            if not math.isfinite(logit_a) or not math.isfinite(logit_b):
                raise ValueError("P1m exact-choice logits are non-finite")
            chosen_label = (
                None
                if logit_a == logit_b
                else ("A" if logit_a > logit_b else "B")
            )
            chosen_action = (
                None
                if chosen_label is None
                else (
                    record.candidate_a
                    if chosen_label == "A"
                    else record.candidate_b
                )
            )
            readouts.append(
                RelationshipP1mQwenReadout(
                    protocol_id=self._protocol.protocol_id,
                    record_index=record.record_index,
                    arm=record.arm,
                    scene_id=record.scene_id,
                    model_input_sha256=record.model_input_sha256,
                    logit_a=logit_a,
                    logit_b=logit_b,
                    chosen_label=chosen_label,
                    chosen_action_id=chosen_action,
                    prompt_tokens=record.prompt_tokens,
                )
            )
        return tuple(readouts)


def _freeze_qwen_execution_manifest(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mQualificationProtocol,
    batch_size: int,
    completed_readouts_before_optimization: int,
) -> None:
    if batch_size != 4:
        raise ValueError("P1m Qwen execution batch size is frozen to 4")
    path = output_dir / "qwen_execution_manifest.json"
    payload = {
        "schema_version": "relationship-p1m-qwen-execution-manifest.v1",
        "protocol_id": protocol.protocol_id,
        "scoring_method": protocol.scoring_method,
        "execution_method": "left_padded_batch_forward_then_atomic_readout_batch",
        "batch_size": batch_size,
        "completed_readouts_before_optimization": completed_readouts_before_optimization,
        "semantic_consumer_changed": False,
        "model_lineage_changed": False,
        "prompt_or_request_changed": False,
        "candidate_tokens_changed": False,
        "qualification_gate_changed": False,
        "truth_attached_before_batch_readouts_durable": False,
        "reason": "Reduce CPU wall time while computing the same frozen per-input first-token logits.",
        "claim_boundary": (
            "Batch grouping is execution-only. Every rendered input hash and token "
            "count remains frozen; all readouts in a batch are durably written "
            "before evaluator truth is attached."
        ),
    }
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        observed_without_id = {
            key: value for key, value in observed.items() if key != "artifact_id"
        }
        frozen_completed = observed_without_id.get(
            "completed_readouts_before_optimization"
        )
        if not isinstance(frozen_completed, int) or frozen_completed < 0:
            raise ValueError("P1m Qwen execution manifest count is invalid")
        if completed_readouts_before_optimization < frozen_completed:
            raise ValueError("P1m Qwen readout ledger predates frozen batch execution")
        expected = {
            **payload,
            "completed_readouts_before_optimization": frozen_completed,
        }
        if (
            observed_without_id != expected
            or observed.get("artifact_id") != sha256_json(expected)
        ):
            raise ValueError("P1m Qwen execution manifest drift")
        return
    payload_with_id = {**payload, "artifact_id": sha256_json(payload)}
    _atomic_write_text(
        path,
        json.dumps(payload_with_id, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _load_source_package(package_dir: pathlib.Path):
    recipe = load_relationship_p1m_generation_recipe()
    pair_plans = build_relationship_p1m_pair_plans(recipe)
    generation_protocol = load_relationship_p1m_generation_recovery_protocol(
        package_dir / "generation_protocol.json"
    )
    generation_records = load_relationship_p1m_generation_records(
        output_dir=package_dir,
        protocol=generation_protocol,
        plans=pair_plans,
    )
    attestation = load_relationship_p1m_generation_attestation(
        package_dir / "generation_attestation.json"
    )
    validate_relationship_p1m_generation_attestation_files(
        attestation,
        output_dir=package_dir,
        protocol=generation_protocol,
        records=generation_records,
    )
    if (
        _sha256_file(package_dir / "renderer_transport.json")
        != generation_protocol.transport_id
        or _sha256_file(package_dir / "surface_seed_inventory.json")
        != generation_protocol.surface_seed_inventory_sha256
    ):
        raise ValueError("P1m package transport/inventory lineage drift")
    dataset = load_relationship_transfer_dataset(package_dir)
    if dataset.dataset_fingerprint != attestation.dataset_fingerprint:
        raise ValueError("P1m package fingerprint differs from attestation")
    return dataset, generation_protocol, generation_records, attestation


def _observations_by_scene(dataset) -> dict[str, object]:
    return {item.scene_id: item for item in dataset.observations}


def _validate_plan_against_public_dataset(
    *,
    plan: RelationshipP1mQualificationPlan,
    dataset,
    input_renderer: _FrozenQwenInputRenderer,
) -> None:
    if plan.dataset_fingerprint != dataset.dataset_fingerprint:
        raise ValueError("P1m frozen plan dataset fingerprint drift")
    observations = _observations_by_scene(dataset)
    for record in plan.qwen_records:
        observation = observations.get(record.scene_id)
        if observation is None:
            raise ValueError("P1m Qwen plan references an unknown scene")
        rendered, prompt_tokens = input_renderer.render(record.request_text)
        if (
            hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            != record.model_input_sha256
            or prompt_tokens != record.prompt_tokens
        ):
            raise ValueError("P1m frozen Qwen plan no longer reproduces")
        if set(record.ordered_history_event_ids) != {
            item.event_id for item in observation.histories
        }:
            raise ValueError("P1m Qwen plan history surface drift")
    for record in plan.structured_records:
        observation = observations.get(record.scene_id)
        if observation is None:
            raise ValueError("P1m structured plan references an unknown scene")
        episode = relationship_p1m_public_episode(observation)
        if sha256_json(episode.to_sut_sequence()) != record.public_episode_sha256:
            raise ValueError("P1m structured public episode drift")


def _build_protocol(
    *,
    frozen_at_iso: str,
    p1k_report_id: str,
    generation_protocol,
    attestation,
    dataset,
    qwen_snapshot_sha256: str,
    input_renderer: _FrozenQwenInputRenderer,
    bge_weights_sha256: str,
    reader_artifact,
    plan: RelationshipP1mQualificationPlan,
) -> RelationshipP1mQualificationProtocol:
    prompt_sha = _sha256_file(relationship_p1m_forced_choice_prompt_path())
    request_sha = _sha256_file(relationship_p1m_forced_choice_request_path())
    qwen_config_sha = sha256_json(
        {
            "device": _QWEN_DEVICE,
            "torch_dtype": _QWEN_TORCH_DTYPE,
            "scoring_method": _SCORING_METHOD,
            "candidate_tokens": ["A", "B"],
            "candidate_token_ids": [
                input_renderer.token_a_id,
                input_renderer.token_b_id,
            ],
            "prompt_sha256": prompt_sha,
            "request_template_sha256": request_sha,
            "sampling": False,
            "generation": False,
        }
    )
    return RelationshipP1mQualificationProtocol(
        frozen_at_iso=frozen_at_iso,
        source_p1k_report_artifact_id=p1k_report_id,
        source_generation_attestation_id=attestation.artifact_id,
        source_generation_protocol_id=generation_protocol.protocol_id,
        source_transport_id=generation_protocol.transport_id,
        source_seed_inventory_sha256=(
            generation_protocol.surface_seed_inventory_sha256
        ),
        package_name=dataset.package_name,
        dataset_fingerprint=dataset.dataset_fingerprint,
        pair_count=len(dataset.mirrored_pairs()),
        scene_count=len(dataset.observations),
        qwen_model_source=_QWEN_MODEL_SOURCE,
        qwen_model_revision=_QWEN_MODEL_REVISION,
        qwen_model_id=_QWEN_MODEL_ID,
        qwen_weights_sha256=_QWEN_EXPECTED_WEIGHTS_SHA256,
        qwen_snapshot_sha256=qwen_snapshot_sha256,
        qwen_device=_QWEN_DEVICE,
        qwen_torch_dtype=_QWEN_TORCH_DTYPE,
        prompt_sha256=prompt_sha,
        request_template_sha256=request_sha,
        token_a_id=input_renderer.token_a_id,
        token_b_id=input_renderer.token_b_id,
        scoring_method=_SCORING_METHOD,
        qwen_config_sha256=qwen_config_sha,
        bge_model_source=_BGE_MODEL_SOURCE,
        bge_model_revision=_BGE_MODEL_REVISION,
        bge_weights_sha256=bge_weights_sha256,
        reader_artifact=reader_artifact,
        rag_top_k=4,
        plan_artifact_id=plan.artifact_id,
        planned_qwen_readouts=len(plan.qwen_records),
        planned_structured_readouts=len(plan.structured_records),
        qualification_inputs_observed_before_freeze=0,
        qwen_outputs_observed_before_freeze=0,
        structured_outputs_observed_before_freeze=0,
        first_qualification_attempt_only=True,
        evaluation_feedback_allowed=False,
    )


def _load_qwen_readouts(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mQualificationProtocol,
    plan: RelationshipP1mQualificationPlan,
) -> tuple[RelationshipP1mQwenReadout, ...]:
    raw = _load_jsonl(output_dir / "qwen_readouts.jsonl")
    if len(raw) > len(plan.qwen_records):
        raise ValueError("P1m Qwen readout ledger exceeds plan")
    readouts = tuple(RelationshipP1mQwenReadout.from_payload(item) for item in raw)
    for index, (readout, record) in enumerate(
        zip(readouts, plan.qwen_records, strict=False)
    ):
        expected_action = (
            record.candidate_a
            if readout.chosen_label == "A"
            else record.candidate_b
            if readout.chosen_label == "B"
            else None
        )
        if (
            readout.protocol_id != protocol.protocol_id
            or readout.record_index != index
            or readout.arm is not record.arm
            or readout.scene_id != record.scene_id
            or readout.model_input_sha256 != record.model_input_sha256
            or readout.prompt_tokens != record.prompt_tokens
            or readout.chosen_action_id is not expected_action
        ):
            raise ValueError("P1m Qwen readout ledger lineage drift")
    return readouts


def _load_structured_readouts(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP1mQualificationProtocol,
    plan: RelationshipP1mQualificationPlan,
) -> tuple[RelationshipP1mStructuredReadout, ...]:
    raw = _load_jsonl(output_dir / "structured_readouts.jsonl")
    if len(raw) > len(plan.structured_records):
        raise ValueError("P1m structured readout ledger exceeds plan")
    readouts = tuple(
        RelationshipP1mStructuredReadout.from_payload(item) for item in raw
    )
    for index, (readout, record) in enumerate(
        zip(readouts, plan.structured_records, strict=False)
    ):
        if (
            readout.protocol_id != protocol.protocol_id
            or readout.record_index != index
            or readout.scene_id != record.scene_id
            or readout.reader_artifact_id != protocol.reader_artifact.artifact_id
        ):
            raise ValueError("P1m structured readout ledger lineage drift")
    return readouts


def _load_decisions(
    *,
    path: pathlib.Path,
    protocol: RelationshipP1mQualificationProtocol,
    qwen_plan: tuple[RelationshipP1mQwenPlanRecord, ...] | None = None,
    structured_plan: tuple[RelationshipP1mStructuredPlanRecord, ...] | None = None,
) -> tuple[RelationshipP1mQualificationDecision, ...]:
    raw = _load_jsonl(path)
    plan = qwen_plan if qwen_plan is not None else structured_plan
    if plan is None or len(raw) > len(plan):
        raise ValueError("P1m decision ledger plan mismatch")
    decisions = tuple(
        RelationshipP1mQualificationDecision.from_payload(item) for item in raw
    )
    for index, (decision, record) in enumerate(zip(decisions, plan, strict=False)):
        expected_arm = (
            record.arm
            if isinstance(record, RelationshipP1mQwenPlanRecord)
            else RelationshipP1mQualificationArm.STRUCTURED_STATE
        )
        if (
            decision.protocol_id != protocol.protocol_id
            or decision.record_index != index
            or decision.arm is not expected_arm
            or decision.scene_id != record.scene_id
            or decision.mirror_pair_id != record.mirror_pair_id
        ):
            raise ValueError("P1m decision ledger lineage drift")
    return decisions


def _decision_for_readout(
    *,
    protocol: RelationshipP1mQualificationProtocol,
    record,
    readout,
    dataset,
) -> RelationshipP1mQualificationDecision:
    dynamic = dataset.dynamic_for_scene(record.scene_id)
    return RelationshipP1mQualificationDecision(
        protocol_id=protocol.protocol_id,
        arm=(
            record.arm
            if isinstance(record, RelationshipP1mQwenPlanRecord)
            else RelationshipP1mQualificationArm.STRUCTURED_STATE
        ),
        record_index=record.record_index,
        scene_id=record.scene_id,
        mirror_pair_id=record.mirror_pair_id,
        readout_artifact_id=readout.artifact_id,
        chosen_action_id=(
            readout.chosen_action_id
            if isinstance(readout, RelationshipP1mQwenReadout)
            else readout.recommended_action_id
            if readout.valid
            else None
        ),
        expected_action_id=dynamic.preferred_action,
    )


def _recover_qwen_decision_gap(
    *,
    output_dir,
    protocol,
    plan,
    dataset,
    readouts,
    decisions,
    batch_size,
):
    if batch_size <= 0:
        raise ValueError("P1m Qwen recovery batch size must be positive")
    if not len(decisions) <= len(readouts) <= len(decisions) + batch_size:
        raise ValueError("P1m Qwen ledgers violate readout-before-truth order")
    while len(readouts) > len(decisions):
        index = len(decisions)
        decision = _decision_for_readout(
            protocol=protocol,
            record=plan.qwen_records[index],
            readout=readouts[index],
            dataset=dataset,
        )
        decisions = (*decisions, decision)
    if len(readouts) > 0:
        _write_ledger(output_dir / "qwen_decisions.jsonl", decisions)
    return decisions


def _recover_structured_decision_gap(
    *,
    output_dir,
    protocol,
    plan,
    dataset,
    readouts,
    decisions,
):
    if not len(decisions) <= len(readouts) <= len(decisions) + 1:
        raise ValueError("P1m structured ledgers violate readout-before-truth order")
    if len(readouts) > len(decisions):
        index = len(decisions)
        decision = _decision_for_readout(
            protocol=protocol,
            record=plan.structured_records[index],
            readout=readouts[index],
            dataset=dataset,
        )
        decisions = (*decisions, decision)
        _write_ledger(output_dir / "structured_decisions.jsonl", decisions)
    return decisions


def _structured_readout(
    *,
    protocol: RelationshipP1mQualificationProtocol,
    record: RelationshipP1mStructuredPlanRecord,
    observation,
    runtime: PrototypeRelationshipPreferenceForecastRuntime,
) -> RelationshipP1mStructuredReadout:
    episode = relationship_p1m_public_episode(observation)
    if sha256_json(episode.to_sut_sequence()) != record.public_episode_sha256:
        raise ValueError("P1m structured runtime episode differs from plan")
    run = asyncio.run(
        run_p2_development_episode(
            episode,
            forecast_runtime=runtime,
        )
    )
    condition = run.forecast.condition_readout
    if condition is None:
        raise RuntimeError("P1m named reader did not publish a condition readout")
    return RelationshipP1mStructuredReadout(
        protocol_id=protocol.protocol_id,
        record_index=record.record_index,
        scene_id=record.scene_id,
        recommended_action_id=RelationshipAction(
            run.forecast.recommended_action_id
        ),
        forecast_id=run.forecast.forecast_id,
        confidence=run.forecast.confidence,
        condition_label=condition.condition_label,
        condition_confidence=condition.confidence,
        condition_margin=condition.normalized_margin,
        condition_candidate_scores=condition.candidate_scores,
        reader_artifact_id=condition.reader_artifact_id,
        source_observation_sha256=condition.source_observation_sha256,
        persistence_payload_sha256=run.persistence_payload_sha256,
        persisted_record_count=run.persisted_record_count,
        persisted_action_outcome_count=run.persisted_action_outcome_count,
        raw_history_replayed_at_probe=run.raw_history_replayed_at_probe,
    )


def _write_report_markdown(report, *, output_dir: pathlib.Path) -> None:
    lines = [
        "# P1m first qualification",
        "",
        f"- report id: `{report.artifact_id}`",
        f"- verdict: `{report.verdict.value}`",
        f"- qualification passed: `{str(report.qualification_passed).lower()}`",
        f"- scenario versioning closed: `{str(report.scenario_versioning_closed).lower()}`",
        "",
        "| arm | valid | accuracy | accuracy Wilson lower | pair flip | flip Wilson lower |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in report.arm_metrics:
        lines.append(
            f"| {metric.arm.value} | {metric.valid_decisions}/{metric.decisions} | "
            f"{metric.accuracy:.3f} | {metric.accuracy_wilson_lower:.3f} | "
            f"{metric.pair_flip_rate:.3f} | {metric.pair_flip_wilson_lower:.3f} |"
        )
    lines.extend(("", report.claim_boundary, ""))
    _atomic_write_text(output_dir / "qualification_report.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.max_new_qwen_readouts < 0 or args.max_new_structured_readouts < 0:
        raise ValueError("readout limits must be zero or positive")
    if args.qwen_batch_size != 4:
        raise ValueError("P1m Qwen execution batch size is frozen to 4")
    package_dir = pathlib.Path(args.package_dir)
    output_dir = pathlib.Path(args.output_dir)
    report_path = output_dir / "qualification_report.json"
    if report_path.is_file():
        plan = load_relationship_p1m_qualification_plan(
            output_dir / "qualification_plan.json"
        )
        protocol = load_relationship_p1m_qualification_protocol(
            output_dir / "qualification_protocol.json"
        )
        report = load_relationship_p1m_qualification_report(report_path)
        validate_relationship_p1m_qualification_report_files(
            report,
            output_dir=output_dir,
            protocol=protocol,
            plan=plan,
        )
        execution_manifest_path = output_dir / "qwen_execution_manifest.json"
        if not execution_manifest_path.is_file():
            raise FileNotFoundError("P1m terminal execution manifest is missing")
        _freeze_qwen_execution_manifest(
            output_dir=output_dir,
            protocol=protocol,
            batch_size=args.qwen_batch_size,
            completed_readouts_before_optimization=(
                protocol.planned_qwen_readouts
            ),
        )
        print(
            json.dumps(
                {
                    "stage": "terminal",
                    "report_artifact_id": report.artifact_id,
                    "verdict": report.verdict.value,
                    "qualification_passed": report.qualification_passed,
                    "scenario_versioning_closed": (
                        report.scenario_versioning_closed
                    ),
                    "terminal_integrity_validated": True,
                },
                ensure_ascii=False,
            )
        )
        return 0

    dataset, generation_protocol, _generation_records, attestation = (
        _load_source_package(package_dir)
    )
    p1k_report = load_relationship_p1k_report(
        pathlib.Path(args.source_p1k_report)
    )
    if p1k_report.artifact_id != generation_protocol.source_p1k_report_artifact_id:
        raise ValueError("P1m qualification P1k lineage drift")

    qwen_snapshot = _materialize_snapshot(
        repo_id=_QWEN_MODEL_SOURCE,
        revision=_QWEN_MODEL_REVISION,
        allow_download=args.allow_download,
    )
    qwen_weights_sha256 = frozen_model_weights_sha256(qwen_snapshot)
    if qwen_weights_sha256 != _QWEN_EXPECTED_WEIGHTS_SHA256:
        raise ValueError("P1m Qwen weights differ from frozen 3B substrate")
    qwen_snapshot_sha256 = _snapshot_manifest_digest(qwen_snapshot)
    input_renderer = _FrozenQwenInputRenderer(qwen_snapshot)

    bge_snapshot = _materialize_snapshot(
        repo_id=_BGE_MODEL_SOURCE,
        revision=_BGE_MODEL_REVISION,
        allow_download=args.allow_download,
    )
    bge_weights_sha256 = _snapshot_manifest_digest(bge_snapshot)
    if bge_weights_sha256 != _BGE_EXPECTED_WEIGHTS_SHA256:
        raise ValueError("P1m BGE weights differ from frozen named reader")
    reader_artifact = build_relationship_p1m_reader_artifact(
        dataset,
        embedding_model_id=_BGE_MODEL_ID,
        embedding_weights_sha256=bge_weights_sha256,
    )

    protocol_path = output_dir / "qualification_protocol.json"
    plan_path = output_dir / "qualification_plan.json"
    if protocol_path.is_file() or plan_path.is_file():
        if not protocol_path.is_file() or not plan_path.is_file():
            raise ValueError("P1m qualification protocol/plan freeze is incomplete")
        if not args.resume and not args.prepare_only:
            raise FileExistsError("P1m qualification exists; use --resume")
        plan = load_relationship_p1m_qualification_plan(plan_path)
        protocol = load_relationship_p1m_qualification_protocol(protocol_path)
        _validate_plan_against_public_dataset(
            plan=plan,
            dataset=dataset,
            input_renderer=input_renderer,
        )
        expected_protocol = _build_protocol(
            frozen_at_iso=protocol.frozen_at_iso,
            p1k_report_id=p1k_report.artifact_id,
            generation_protocol=generation_protocol,
            attestation=attestation,
            dataset=dataset,
            qwen_snapshot_sha256=qwen_snapshot_sha256,
            input_renderer=input_renderer,
            bge_weights_sha256=bge_weights_sha256,
            reader_artifact=reader_artifact,
            plan=plan,
        )
        if protocol != expected_protocol:
            raise ValueError("P1m qualification frozen protocol lineage drift")
    else:
        if args.resume:
            raise FileNotFoundError("P1m --resume cannot create a qualification")
        embedder = _FrozenEmbedder(bge_snapshot)
        plan = build_relationship_p1m_qualification_plan(
            dataset,
            render_model_input=input_renderer.render,
            embed=embedder.embed,
        )
        del embedder
        _release_runtime()
        protocol = _build_protocol(
            frozen_at_iso=datetime.now(timezone.utc).isoformat(),
            p1k_report_id=p1k_report.artifact_id,
            generation_protocol=generation_protocol,
            attestation=attestation,
            dataset=dataset,
            qwen_snapshot_sha256=qwen_snapshot_sha256,
            input_renderer=input_renderer,
            bge_weights_sha256=bge_weights_sha256,
            reader_artifact=reader_artifact,
            plan=plan,
        )
        write_relationship_p1m_qualification_plan(plan, output_dir=output_dir)
        write_relationship_p1m_qualification_protocol(
            protocol,
            output_dir=output_dir,
        )

    print(
        json.dumps(
            {
                "stage": "prepared",
                "protocol_id": protocol.protocol_id,
                "plan_artifact_id": plan.artifact_id,
                "dataset_fingerprint": plan.dataset_fingerprint,
                "planned_qwen_readouts": len(plan.qwen_records),
                "planned_structured_readouts": len(plan.structured_records),
                "qualification_inputs_observed_before_freeze": 0,
                "qwen_outputs_observed_before_freeze": 0,
                "structured_outputs_observed_before_freeze": 0,
                "first_qualification_attempt_only": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0

    qwen_readouts = _load_qwen_readouts(
        output_dir=output_dir,
        protocol=protocol,
        plan=plan,
    )
    qwen_decisions = _load_decisions(
        path=output_dir / "qwen_decisions.jsonl",
        protocol=protocol,
        qwen_plan=plan.qwen_records,
    )
    qwen_decisions = _recover_qwen_decision_gap(
        output_dir=output_dir,
        protocol=protocol,
        plan=plan,
        dataset=dataset,
        readouts=qwen_readouts,
        decisions=qwen_decisions,
        batch_size=args.qwen_batch_size,
    )
    remaining_qwen = len(plan.qwen_records) - len(qwen_readouts)
    qwen_allowance = (
        remaining_qwen
        if args.max_new_qwen_readouts == 0
        else min(remaining_qwen, args.max_new_qwen_readouts)
    )
    if qwen_allowance:
        _freeze_qwen_execution_manifest(
            output_dir=output_dir,
            protocol=protocol,
            batch_size=args.qwen_batch_size,
            completed_readouts_before_optimization=len(qwen_readouts),
        )
        policy = _FrozenQwenExactChoicePolicy(
            snapshot=qwen_snapshot,
            input_renderer=input_renderer,
            protocol=protocol,
        )
        execution_end = len(qwen_readouts) + qwen_allowance
        while len(qwen_readouts) < execution_end:
            batch_start = len(qwen_readouts)
            batch_end = min(
                batch_start + args.qwen_batch_size,
                execution_end,
            )
            records = plan.qwen_records[batch_start:batch_end]
            batch_readouts = policy.score_batch(records)
            qwen_readouts = (*qwen_readouts, *batch_readouts)
            _write_ledger(output_dir / "qwen_readouts.jsonl", qwen_readouts)
            batch_decisions = tuple(
                _decision_for_readout(
                    protocol=protocol,
                    record=record,
                    readout=readout,
                    dataset=dataset,
                )
                for record, readout in zip(records, batch_readouts, strict=True)
            )
            qwen_decisions = (*qwen_decisions, *batch_decisions)
            _write_ledger(output_dir / "qwen_decisions.jsonl", qwen_decisions)
            print(
                json.dumps(
                    {
                        "stage": "qwen_readout_batch_checkpointed",
                        "record_indexes": list(range(batch_start, batch_end)),
                        "arms": [record.arm.value for record in records],
                        "scene_ids": [record.scene_id for record in records],
                        "chosen_labels": [
                            readout.chosen_label for readout in batch_readouts
                        ],
                        "durable_qwen_readouts": len(qwen_readouts),
                        "planned_qwen_readouts": len(plan.qwen_records),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del policy
        _release_runtime()
    if len(qwen_readouts) < len(plan.qwen_records):
        print(
            json.dumps(
                {
                    "stage": "qwen_partial",
                    "durable_qwen_readouts": len(qwen_readouts),
                    "remaining_qwen_readouts": (
                        len(plan.qwen_records) - len(qwen_readouts)
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0

    structured_readouts = _load_structured_readouts(
        output_dir=output_dir,
        protocol=protocol,
        plan=plan,
    )
    structured_decisions = _load_decisions(
        path=output_dir / "structured_decisions.jsonl",
        protocol=protocol,
        structured_plan=plan.structured_records,
    )
    structured_decisions = _recover_structured_decision_gap(
        output_dir=output_dir,
        protocol=protocol,
        plan=plan,
        dataset=dataset,
        readouts=structured_readouts,
        decisions=structured_decisions,
    )
    remaining_structured = len(plan.structured_records) - len(structured_readouts)
    structured_allowance = (
        remaining_structured
        if args.max_new_structured_readouts == 0
        else min(remaining_structured, args.max_new_structured_readouts)
    )
    if structured_allowance:
        embedder = _FrozenEmbedder(bge_snapshot)
        runtime = PrototypeRelationshipPreferenceForecastRuntime(
            artifact=protocol.reader_artifact,
            embedder=embedder,
        )
        observations = _observations_by_scene(dataset)
        for index in range(
            len(structured_readouts),
            len(structured_readouts) + structured_allowance,
        ):
            record = plan.structured_records[index]
            observation = observations[record.scene_id]
            readout = _structured_readout(
                protocol=protocol,
                record=record,
                observation=observation,
                runtime=runtime,
            )
            structured_readouts = (*structured_readouts, readout)
            _write_ledger(
                output_dir / "structured_readouts.jsonl",
                structured_readouts,
            )
            decision = _decision_for_readout(
                protocol=protocol,
                record=record,
                readout=readout,
                dataset=dataset,
            )
            structured_decisions = (*structured_decisions, decision)
            _write_ledger(
                output_dir / "structured_decisions.jsonl",
                structured_decisions,
            )
            print(
                json.dumps(
                    {
                        "stage": "structured_readout_checkpointed",
                        "record_index": index,
                        "scene_id": record.scene_id,
                        "condition_label": readout.condition_label,
                        "recommended_action_id": (
                            readout.recommended_action_id.value
                        ),
                        "durable_structured_readouts": len(structured_readouts),
                        "planned_structured_readouts": len(
                            plan.structured_records
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del runtime
        del embedder
        _release_runtime()
    if len(structured_readouts) < len(plan.structured_records):
        print(
            json.dumps(
                {
                    "stage": "structured_partial",
                    "durable_structured_readouts": len(structured_readouts),
                    "remaining_structured_readouts": (
                        len(plan.structured_records) - len(structured_readouts)
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0

    all_decisions = (*qwen_decisions, *structured_decisions)
    report = assess_relationship_p1m_qualification(
        protocol=protocol,
        plan=plan,
        decisions=all_decisions,
        qwen_readout_ledger_sha256=_ledger_sha256(qwen_readouts),
        structured_readout_ledger_sha256=_ledger_sha256(structured_readouts),
        qwen_decision_ledger_sha256=_ledger_sha256(qwen_decisions),
        structured_decision_ledger_sha256=_ledger_sha256(structured_decisions),
        created_at_iso=datetime.now(timezone.utc).isoformat(),
    )
    write_relationship_p1m_qualification_report(report, output_dir=output_dir)
    _write_report_markdown(report, output_dir=output_dir)
    print(
        json.dumps(
            {
                "stage": "complete",
                "report_artifact_id": report.artifact_id,
                "verdict": report.verdict.value,
                "qualification_passed": report.qualification_passed,
                "scenario_versioning_closed": report.scenario_versioning_closed,
                "metrics": {
                    item.arm.value: item.to_payload()
                    for item in report.arm_metrics
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
