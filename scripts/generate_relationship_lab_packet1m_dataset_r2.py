#!/usr/bin/env python3
"""Preflight, freeze, and run the renderer-only P1m generation recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import pathlib
import sys
import tempfile

import yaml


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
    "packages/vz-contracts/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from huggingface_hub import snapshot_download  # noqa: E402
from lifeform_domain_emogpt.lab import (  # noqa: E402
    RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
    RelationshipP1mDeterministicFieldRenderer,
    build_relationship_p1m_dataset_payloads,
    build_relationship_p1m_field_plans,
    build_relationship_p1m_manifest_payload,
    build_relationship_p1m_pair_plans,
    build_relationship_p1m_preflight_field_plans,
    build_relationship_p1m_scenes_payload,
    build_relationship_p1m_ssot_fragment,
    build_relationship_p1m_test_suite_payload,
    compose_relationship_p1m_surface_rendering,
    load_relationship_p1m_generation_recipe,
    load_relationship_p1m_renderer_transport,
    load_relationship_transfer_dataset,
    relationship_p1m_field_prompt_path,
    relationship_p1m_recipe_path,
    relationship_p1m_renderer_transport_path,
    relationship_p1m_surface_seed_inventory_path,
    validate_relationship_p1m_field_output,
    validate_relationship_p1m_transport_against_recipe,
)
from lifeform_evolution.relationship_lab_baseline import (  # noqa: E402
    frozen_model_weights_sha256,
)
from lifeform_evolution.relationship_lab_packet1k import (  # noqa: E402
    load_relationship_p1k_report,
)
from lifeform_evolution.relationship_lab_packet1m import (  # noqa: E402
    RELATIONSHIP_P1M_SOURCE_VERDICT,
    build_relationship_p1m_generation_attestation,
    load_relationship_p1m_generation_attestation,
    load_relationship_p1m_generation_records,
    persist_relationship_p1m_generation_record,
    validate_relationship_p1m_generation_attestation_files,
    write_relationship_p1m_generation_attestation,
)
from lifeform_evolution.relationship_lab_packet1m_recovery import (  # noqa: E402
    build_relationship_p1m_renderer_preflight_report,
    freeze_relationship_p1m_generation_recovery_protocol,
    load_relationship_p1m_field_batches,
    load_relationship_p1m_generation_recovery_protocol,
    load_relationship_p1m_raw_field_attempts,
    load_relationship_p1m_renderer_preflight_report,
    persist_relationship_p1m_field_batch,
    persist_relationship_p1m_raw_field_attempt,
    write_relationship_p1m_generation_recovery_protocol,
    write_relationship_p1m_renderer_preflight_report,
)


_FAILED_ATTEMPT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "relationship_transfer_p1m_v1_first_attempt_20260822"
)
_DEFAULT_INCIDENT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "relationship_transfer_p1m_v1_transport_r2_20260822"
    / "generation_recovery_authorization.json"
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
    / "relationship_transfer_p1m_v1_transport_r5_20260822"
)
_PRIOR_PACKAGES = (
    RELATIONSHIP_TRANSFER_V1_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-p1k-report", default=str(_DEFAULT_P1K_REPORT))
    parser.add_argument("--source-incident", default=str(_DEFAULT_INCIDENT))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the four-field non-scenario transport preflight and stop.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Require passing preflight, freeze zero-scenario-output protocol, stop.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only the already-frozen recovery attempt.",
    )
    parser.add_argument(
        "--max-new-renderings",
        type=int,
        default=1,
        help="New pair model calls this invocation; 0 means all remaining.",
    )
    return parser.parse_args(argv)


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


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_runtime(*, device: str, torch_dtype: str) -> tuple[str, str]:
    import torch

    resolved_device = device
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    resolved_dtype = torch_dtype
    if torch_dtype == "auto":
        resolved_dtype = "bfloat16" if resolved_device == "cpu" else "float16"
    if resolved_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("--torch-dtype must be auto, bfloat16, float16, or float32")
    return resolved_device, resolved_dtype


class _HFRelationshipP1mFieldRenderer:
    def __init__(
        self,
        *,
        snapshot: pathlib.Path,
        model_id: str,
        expected_weights_sha256: str,
        generation_config_sha256: str,
        device: str,
        torch_dtype: str,
        temperature: float,
        top_p: float,
        min_new_tokens: int,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.model_id = model_id
        self.weights_sha256 = frozen_model_weights_sha256(snapshot)
        if self.weights_sha256 != expected_weights_sha256:
            raise ValueError("P1m field renderer weights drift")
        self.generation_config_sha256 = generation_config_sha256
        self._device = device
        self._temperature = temperature
        self._top_p = top_p
        self._min_new_tokens = min_new_tokens
        self._max_new_tokens = max_new_tokens
        self._torch = torch
        self._prompt = relationship_p1m_field_prompt_path().read_text(
            encoding="utf-8"
        ).strip()
        self._tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=dtype_map[torch_dtype],
            low_cpu_mem_usage=True,
        ).to(device)
        self._model.eval()

    def render_fields(
        self,
        *,
        renderer_inputs: tuple[str, ...],
        seeds: tuple[int, ...],
    ) -> tuple[str, ...]:
        if len(renderer_inputs) != len(seeds):
            raise ValueError("P1m field inputs/seeds mismatch")
        outputs: list[str] = []
        for renderer_input, seed in zip(renderer_inputs, seeds, strict=True):
            rendered = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": renderer_input},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self._tokenizer(rendered, return_tensors="pt")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            self._torch.manual_seed(seed)
            sampling: dict[str, object] = {"do_sample": self._temperature > 0.0}
            if self._temperature > 0.0:
                sampling.update(
                    temperature=self._temperature,
                    top_p=self._top_p,
                )
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    min_new_tokens=self._min_new_tokens,
                    max_new_tokens=self._max_new_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                    **sampling,
                )
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            outputs.append(
                self._tokenizer.decode(
                    generated[0, prompt_tokens:],
                    skip_special_tokens=True,
                )[:2000]
            )
        return tuple(outputs)


def _load_renderer(
    *,
    snapshot: pathlib.Path | None,
    transport,
    weights_sha256: str,
    device: str,
    torch_dtype: str,
) -> object:
    if transport.renderer_kind == "deterministic_typed_surface_realizer":
        renderer = RelationshipP1mDeterministicFieldRenderer(transport)
        if renderer.weights_sha256 != weights_sha256:
            raise ValueError("P1m deterministic renderer contract hash drift")
        return renderer
    if snapshot is None:
        raise ValueError("P1m HF renderer requires a local model snapshot")
    return _HFRelationshipP1mFieldRenderer(
        snapshot=snapshot,
        model_id=transport.model_id,
        expected_weights_sha256=weights_sha256,
        generation_config_sha256=transport.generation_config_sha256,
        device=device,
        torch_dtype=torch_dtype,
        temperature=transport.temperature,
        top_p=transport.top_p,
        min_new_tokens=transport.min_new_tokens,
        max_new_tokens=transport.max_new_tokens,
    )


def _all_public_text(package_name: str) -> set[str]:
    dataset = load_relationship_transfer_dataset(package_name=package_name)
    return {
        text
        for observation in dataset.observations
        for text in (
            observation.current_input,
            *(
                value
                for history in observation.histories
                for value in (history.user_utterance, history.user_reaction)
            ),
        )
    }


def _assert_fresh_public_text(output_dir: pathlib.Path) -> None:
    generated = load_relationship_transfer_dataset(output_dir)
    generated_text = {
        text
        for observation in generated.observations
        for text in (
            observation.current_input,
            *(
                value
                for history in observation.histories
                for value in (history.user_utterance, history.user_reaction)
            ),
        )
    }
    prior_text = set().union(*(_all_public_text(name) for name in _PRIOR_PACKAGES))
    overlap = generated_text & prior_text
    if overlap:
        raise ValueError(f"P1m recovery public text overlaps prior data: {overlap!r}")


def _write_json(path: pathlib.Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_yaml(path: pathlib.Path, payload: object) -> None:
    _atomic_write_text(
        path,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )


def _materialize_package(
    *,
    output_dir,
    recipe,
    transport,
    pair_plans,
    protocol,
    generation_records,
) -> None:
    public, truth = build_relationship_p1m_dataset_payloads(
        recipe,
        plans=pair_plans,
        renderings=tuple(item.rendering for item in generation_records),
    )
    _atomic_write_text(
        output_dir / "generation_recipe.json",
        relationship_p1m_recipe_path().read_text(encoding="utf-8"),
    )
    _atomic_write_text(
        output_dir / "renderer_transport.json",
        relationship_p1m_renderer_transport_path().read_text(encoding="utf-8"),
    )
    _atomic_write_text(
        output_dir / "surface_seed_inventory.json",
        relationship_p1m_surface_seed_inventory_path().read_text(encoding="utf-8"),
    )
    _write_json(output_dir / "rendered_observations.json", public)
    _write_json(output_dir / "generator_truth.json", truth)
    _write_json(
        output_dir / "ssot_fragment.json",
        build_relationship_p1m_ssot_fragment(),
    )
    _write_yaml(output_dir / "manifest.yaml", build_relationship_p1m_manifest_payload())
    _write_yaml(
        output_dir / "scenes.yaml",
        build_relationship_p1m_scenes_payload(pair_plans),
    )
    _write_yaml(
        output_dir / "test_suite.yaml",
        build_relationship_p1m_test_suite_payload(),
    )
    dataset = load_relationship_transfer_dataset(output_dir)
    if len(dataset.mirrored_pairs()) != 24 or len(dataset.observations) != 48:
        raise ValueError("P1m recovery generated package size mismatch")
    _assert_fresh_public_text(output_dir)
    attestation = build_relationship_p1m_generation_attestation(
        protocol=protocol,
        records=generation_records,
        dataset_fingerprint=dataset.dataset_fingerprint,
        package_dir=output_dir,
        created_at_iso=datetime.now(timezone.utc).isoformat(),
    )
    write_relationship_p1m_generation_attestation(
        attestation,
        output_dir=output_dir,
    )
    loaded = load_relationship_p1m_generation_attestation(
        output_dir / "generation_attestation.json"
    )
    validate_relationship_p1m_generation_attestation_files(
        loaded,
        output_dir=output_dir,
        protocol=protocol,
        records=generation_records,
    )
    if transport.source_recipe_id != loaded.recipe_id:
        raise ValueError("P1m recovery attestation lost transport recipe lineage")


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.max_new_renderings < 0:
        raise ValueError("--max-new-renderings must be zero or positive")
    output_dir = pathlib.Path(args.output_dir)
    incident_path = pathlib.Path(args.source_incident)
    recipe = load_relationship_p1m_generation_recipe()
    transport = load_relationship_p1m_renderer_transport()
    validate_relationship_p1m_transport_against_recipe(transport, recipe=recipe)
    pair_plans = build_relationship_p1m_pair_plans(recipe)
    field_plans_by_pair = tuple(
        build_relationship_p1m_field_plans(transport, plan=plan)
        for plan in pair_plans
    )
    preflight_plans = build_relationship_p1m_preflight_field_plans(transport)
    source_report = load_relationship_p1k_report(pathlib.Path(args.source_p1k_report))
    if source_report.verdict.value != RELATIONSHIP_P1M_SOURCE_VERDICT:
        raise ValueError("P1m recovery requires the terminal P1k floor verdict")
    if not incident_path.is_file():
        raise FileNotFoundError("P1m recovery requires the durable v1 incident")
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    if (
        incident.get("generation_protocol_id")
        != transport.source_failed_generation_protocol_id
        or incident.get("accepted_renderings") != 0
        or incident.get("consumer_outputs") != 0
        or incident.get("qualification_started") is not False
        or incident.get("semantic_recipe_changed") is not False
        or incident.get("qualification_gate_changed") is not False
        or incident.get("authorized_transport") != transport.schema_version
    ):
        raise ValueError("P1m recovery incident does not authorize renderer-only repair")
    incident_sha256 = _sha256_file(incident_path)
    if transport.renderer_kind == "deterministic_typed_surface_realizer":
        device = "cpu"
        torch_dtype = "not_applicable"
        snapshot = None
        weights_sha256 = transport.prompt_sha256
    else:
        device, torch_dtype = _resolve_runtime(
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
        snapshot = pathlib.Path(
            snapshot_download(repo_id=transport.model_source, local_files_only=True)
        )
        weights_sha256 = frozen_model_weights_sha256(snapshot)

    preflight_path = output_dir / "renderer_preflight.json"
    if preflight_path.is_file():
        preflight = load_relationship_p1m_renderer_preflight_report(
            preflight_path,
            field_plans=preflight_plans,
        )
        if (
            preflight.transport_id != transport.transport_id
            or preflight.weights_sha256 != weights_sha256
            or preflight.runtime_device != device
            or preflight.torch_dtype != torch_dtype
        ):
            raise ValueError("P1m recovery preflight lineage drift")
    else:
        if not args.preflight_only:
            raise FileNotFoundError(
                "run --preflight-only before freezing the P1m recovery protocol"
            )
        renderer = _load_renderer(
            snapshot=snapshot,
            transport=transport,
            weights_sha256=weights_sha256,
            device=device,
            torch_dtype=torch_dtype,
        )
        raw_outputs = renderer.render_fields(
            renderer_inputs=tuple(item.renderer_input for item in preflight_plans),
            seeds=tuple(item.seed for item in preflight_plans),
        )
        _write_json(
            output_dir / "renderer_preflight_raw.json",
            {
                "transport_id": transport.transport_id,
                "surface_seed_inventory_sha256": (
                    transport.surface_seed_inventory_sha256
                ),
                "field_input_sha256": [
                    item.renderer_input_sha256 for item in preflight_plans
                ],
                "seeds": [item.seed for item in preflight_plans],
                "raw_outputs": list(raw_outputs),
                "scenario_outputs": 0,
                "consumer_outputs": 0,
            },
        )
        outputs = tuple(
            validate_relationship_p1m_field_output(raw, plan=plan)
            for raw, plan in zip(raw_outputs, preflight_plans, strict=True)
        )
        preflight = build_relationship_p1m_renderer_preflight_report(
            transport=transport,
            field_plans=preflight_plans,
            outputs=outputs,
            model_id=renderer.model_id,
            weights_sha256=renderer.weights_sha256,
            generation_config_sha256=renderer.generation_config_sha256,
            runtime_device=device,
            torch_dtype=torch_dtype,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        write_relationship_p1m_renderer_preflight_report(
            preflight,
            output_dir=output_dir,
        )
        del renderer
        _release_runtime()
    print(
        json.dumps(
            {
                "stage": "renderer_preflight_passed",
                "preflight_artifact_id": preflight.artifact_id,
                "valid_fields": len(preflight.outputs),
                "scenario_outputs": 0,
                "consumer_outputs": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.preflight_only:
        return 0

    protocol_path = output_dir / "generation_protocol.json"
    if protocol_path.is_file():
        if not args.resume and not args.prepare_only:
            raise FileExistsError("P1m recovery exists; use --resume or --prepare-only")
        protocol = load_relationship_p1m_generation_recovery_protocol(protocol_path)
        expected = freeze_relationship_p1m_generation_recovery_protocol(
            recipe=recipe,
            pair_plans=pair_plans,
            transport=transport,
            preflight=preflight,
            source_p1k_report_artifact_id=source_report.artifact_id,
            source_p1k_verdict=source_report.verdict.value,
            source_incident_sha256=incident_sha256,
            weights_sha256=weights_sha256,
            runtime_device=device,
            torch_dtype=torch_dtype,
            frozen_at_iso=protocol.frozen_at_iso,
        )
        if protocol != expected:
            raise ValueError("P1m recovery protocol lineage drift")
    else:
        if args.resume:
            raise FileNotFoundError("P1m recovery --resume cannot create a protocol")
        protocol = freeze_relationship_p1m_generation_recovery_protocol(
            recipe=recipe,
            pair_plans=pair_plans,
            transport=transport,
            preflight=preflight,
            source_p1k_report_artifact_id=source_report.artifact_id,
            source_p1k_verdict=source_report.verdict.value,
            source_incident_sha256=incident_sha256,
            weights_sha256=weights_sha256,
            runtime_device=device,
            torch_dtype=torch_dtype,
            frozen_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        write_relationship_p1m_generation_recovery_protocol(
            protocol,
            output_dir=output_dir,
        )
    raw_attempts = load_relationship_p1m_raw_field_attempts(
        output_dir=output_dir,
        protocol=protocol,
        pair_plans=pair_plans,
        field_plans_by_pair=field_plans_by_pair,
    )
    field_batches = load_relationship_p1m_field_batches(
        output_dir=output_dir,
        protocol=protocol,
        field_plans_by_pair=field_plans_by_pair,
    )
    generation_records = load_relationship_p1m_generation_records(
        output_dir=output_dir,
        protocol=protocol,
        plans=pair_plans,
    )
    if not (
        len(generation_records)
        <= len(field_batches)
        <= len(raw_attempts)
        <= len(generation_records) + 1
    ):
        raise ValueError("P1m recovery ledgers violate raw→validated→composed order")
    print(
        json.dumps(
            {
                "stage": "prepared",
                "recovery_protocol_id": protocol.protocol_id,
                "source_failed_protocol_id": protocol.source_failed_protocol_id,
                "recipe_id": protocol.recipe_id,
                "pair_plan_sha256": protocol.pair_plan_sha256,
                "accepted_scenario_renderings_before_freeze": 0,
                "consumer_outputs_before_freeze": 0,
                "semantic_recipe_changed": False,
                "qualification_gate_changed": False,
                "durable_renderings": len(generation_records),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0

    attestation_path = output_dir / "generation_attestation.json"
    if attestation_path.is_file():
        attestation = load_relationship_p1m_generation_attestation(attestation_path)
        validate_relationship_p1m_generation_attestation_files(
            attestation,
            output_dir=output_dir,
            protocol=protocol,
            records=generation_records,
        )
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "resumed_existing_completion": True,
                    "attestation_artifact_id": attestation.artifact_id,
                    "dataset_fingerprint": attestation.dataset_fingerprint,
                },
                ensure_ascii=False,
            )
        )
        return 0

    # Crash recovery never calls the model twice for one pair.
    if len(raw_attempts) > len(field_batches):
        index = len(field_batches)
        outputs = tuple(
            validate_relationship_p1m_field_output(raw, plan=plan)
            for raw, plan in zip(
                raw_attempts[index].raw_outputs,
                field_plans_by_pair[index],
                strict=True,
            )
        )
        persist_relationship_p1m_field_batch(
            output_dir=output_dir,
            protocol=protocol,
            pair_plans=pair_plans,
            field_plans_by_pair=field_plans_by_pair,
            outputs=outputs,
        )
        field_batches = load_relationship_p1m_field_batches(
            output_dir=output_dir,
            protocol=protocol,
            field_plans_by_pair=field_plans_by_pair,
        )
    if len(field_batches) > len(generation_records):
        index = len(generation_records)
        rendering = compose_relationship_p1m_surface_rendering(
            pair_plan=pair_plans[index],
            field_outputs=field_batches[index].outputs,
        )
        persist_relationship_p1m_generation_record(
            output_dir=output_dir,
            protocol=protocol,
            plans=pair_plans,
            rendering=rendering,
        )
        generation_records = load_relationship_p1m_generation_records(
            output_dir=output_dir,
            protocol=protocol,
            plans=pair_plans,
        )

    remaining = len(pair_plans) - len(raw_attempts)
    allowance = remaining if args.max_new_renderings == 0 else min(
        remaining,
        args.max_new_renderings,
    )
    if allowance:
        renderer = _load_renderer(
            snapshot=snapshot,
            transport=transport,
            weights_sha256=weights_sha256,
            device=device,
            torch_dtype=torch_dtype,
        )
        for index in range(len(raw_attempts), len(raw_attempts) + allowance):
            plans = field_plans_by_pair[index]
            raw_outputs = renderer.render_fields(
                renderer_inputs=tuple(item.renderer_input for item in plans),
                seeds=tuple(item.seed for item in plans),
            )
            persist_relationship_p1m_raw_field_attempt(
                output_dir=output_dir,
                protocol=protocol,
                pair_plans=pair_plans,
                field_plans_by_pair=field_plans_by_pair,
                raw_outputs=raw_outputs,
            )
            outputs = tuple(
                validate_relationship_p1m_field_output(raw, plan=plan)
                for raw, plan in zip(raw_outputs, plans, strict=True)
            )
            persist_relationship_p1m_field_batch(
                output_dir=output_dir,
                protocol=protocol,
                pair_plans=pair_plans,
                field_plans_by_pair=field_plans_by_pair,
                outputs=outputs,
            )
            rendering = compose_relationship_p1m_surface_rendering(
                pair_plan=pair_plans[index],
                field_outputs=outputs,
            )
            record = persist_relationship_p1m_generation_record(
                output_dir=output_dir,
                protocol=protocol,
                plans=pair_plans,
                rendering=rendering,
            )
            print(
                json.dumps(
                    {
                        "stage": "raw_validated_composed_checkpointed",
                        "record_index": record.record_index,
                        "pair_id": record.pair_id,
                        "raw_fields": len(raw_outputs),
                        "durable_renderings": record.record_index + 1,
                        "planned_renderings": len(pair_plans),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del renderer
        _release_runtime()

    generation_records = load_relationship_p1m_generation_records(
        output_dir=output_dir,
        protocol=protocol,
        plans=pair_plans,
    )
    if len(generation_records) < len(pair_plans):
        print(
            json.dumps(
                {
                    "stage": "partial",
                    "durable_renderings": len(generation_records),
                    "remaining": len(pair_plans) - len(generation_records),
                },
                ensure_ascii=False,
            )
        )
        return 0

    _materialize_package(
        output_dir=output_dir,
        recipe=recipe,
        transport=transport,
        pair_plans=pair_plans,
        protocol=protocol,
        generation_records=generation_records,
    )
    attestation = load_relationship_p1m_generation_attestation(attestation_path)
    print(
        json.dumps(
            {
                "stage": "complete",
                "attestation_artifact_id": attestation.artifact_id,
                "dataset_fingerprint": attestation.dataset_fingerprint,
                "mirrored_pairs": attestation.pair_count,
                "consumer_outputs": attestation.consumer_outputs,
                "next_action": attestation.next_action,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
