#!/usr/bin/env python3
"""Freeze and materialize the first P1m generated relationship package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import pathlib
import sys
import tempfile
from typing import Any

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
    build_relationship_p1m_dataset_payloads,
    build_relationship_p1m_manifest_payload,
    build_relationship_p1m_pair_plans,
    build_relationship_p1m_scenes_payload,
    build_relationship_p1m_ssot_fragment,
    build_relationship_p1m_test_suite_payload,
    load_relationship_p1m_generation_recipe,
    load_relationship_transfer_dataset,
    relationship_p1m_recipe_path,
    relationship_p1m_renderer_prompt_path,
    render_relationship_p1m_pair,
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
    freeze_relationship_p1m_generation_protocol,
    load_relationship_p1m_generation_attestation,
    load_relationship_p1m_generation_protocol,
    load_relationship_p1m_generation_records,
    persist_relationship_p1m_generation_record,
    validate_relationship_p1m_generation_attestation_files,
    validate_relationship_p1m_generation_protocol,
    write_relationship_p1m_generation_attestation,
    write_relationship_p1m_generation_protocol,
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
    / "relationship_transfer_p1m_v1_first_attempt_20260822"
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
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze protocol with zero renderer/consumer outputs, then stop.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only the existing first generated attempt.",
    )
    parser.add_argument(
        "--max-new-renderings",
        type=int,
        default=1,
        help="Accepted pair renderings this invocation; 0 means all remaining.",
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


class _HFRelationshipP1mSurfaceRenderer:
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
            raise ValueError("P1m renderer weights drifted after protocol freeze")
        self.generation_config_sha256 = generation_config_sha256
        self._device = device
        self._temperature = temperature
        self._top_p = top_p
        self._max_new_tokens = max_new_tokens
        self._torch = torch
        self._prompt = relationship_p1m_renderer_prompt_path().read_text(
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

    def render(self, *, renderer_input: str, seed: int) -> str:
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": renderer_input},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        self._torch.manual_seed(seed)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": True,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        with self._torch.inference_mode():
            generated = self._model.generate(**encoded, **generation_kwargs)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        completion_ids = generated[0, prompt_tokens:]
        return self._tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()[:20000]


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
        raise ValueError(
            f"P1m generated public text overlaps a prior package: {sorted(overlap)!r}"
        )


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
    output_dir: pathlib.Path,
    recipe,
    plans,
    protocol,
    records,
) -> None:
    public, truth = build_relationship_p1m_dataset_payloads(
        recipe,
        plans=plans,
        renderings=tuple(item.rendering for item in records),
    )
    _atomic_write_text(
        output_dir / "generation_recipe.json",
        relationship_p1m_recipe_path().read_text(encoding="utf-8"),
    )
    _write_json(output_dir / "rendered_observations.json", public)
    _write_json(output_dir / "generator_truth.json", truth)
    _write_json(
        output_dir / "ssot_fragment.json",
        build_relationship_p1m_ssot_fragment(),
    )
    _write_yaml(
        output_dir / "manifest.yaml",
        build_relationship_p1m_manifest_payload(),
    )
    _write_yaml(
        output_dir / "scenes.yaml",
        build_relationship_p1m_scenes_payload(plans),
    )
    _write_yaml(
        output_dir / "test_suite.yaml",
        build_relationship_p1m_test_suite_payload(),
    )
    dataset = load_relationship_transfer_dataset(output_dir)
    if len(dataset.mirrored_pairs()) != 24 or len(dataset.observations) != 48:
        raise ValueError("P1m strict loader returned the wrong generated package size")
    _assert_fresh_public_text(output_dir)
    attestation = build_relationship_p1m_generation_attestation(
        protocol=protocol,
        records=records,
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
        records=records,
    )


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
    recipe = load_relationship_p1m_generation_recipe()
    plans = build_relationship_p1m_pair_plans(recipe)
    source_report = load_relationship_p1k_report(pathlib.Path(args.source_p1k_report))
    if source_report.verdict.value != RELATIONSHIP_P1M_SOURCE_VERDICT:
        raise ValueError("P1m generation requires terminal P1k substrate-floor verdict")
    resolved_device, resolved_dtype = _resolve_runtime(
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    snapshot = pathlib.Path(
        snapshot_download(
            repo_id=recipe.renderer.model_source,
            local_files_only=True,
        )
    )
    weights_sha256 = frozen_model_weights_sha256(snapshot)

    protocol_path = output_dir / "generation_protocol.json"
    if protocol_path.is_file():
        if not args.resume and not args.prepare_only:
            raise FileExistsError(
                "P1m generation attempt exists; only --resume or --prepare-only is allowed"
            )
        protocol = load_relationship_p1m_generation_protocol(protocol_path)
        validate_relationship_p1m_generation_protocol(
            protocol,
            recipe=recipe,
            plans=plans,
            renderer_weights_sha256=weights_sha256,
        )
        if (
            protocol.runtime_device != resolved_device
            or protocol.torch_dtype != resolved_dtype
        ):
            raise ValueError("P1m renderer runtime device/dtype drift")
    else:
        if args.resume:
            raise FileNotFoundError("P1m --resume cannot create a second attempt")
        protocol = freeze_relationship_p1m_generation_protocol(
            recipe=recipe,
            plans=plans,
            source_p1k_report_artifact_id=source_report.artifact_id,
            source_p1k_verdict=source_report.verdict.value,
            renderer_weights_sha256=weights_sha256,
            runtime_device=resolved_device,
            torch_dtype=resolved_dtype,
            frozen_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        write_relationship_p1m_generation_protocol(
            protocol,
            output_dir=output_dir,
        )
    records = load_relationship_p1m_generation_records(
        output_dir=output_dir,
        protocol=protocol,
        plans=plans,
    )
    print(
        json.dumps(
            {
                "stage": "prepared",
                "generation_protocol_id": protocol.protocol_id,
                "source_p1k_report_artifact_id": source_report.artifact_id,
                "recipe_id": recipe.recipe_id,
                "renderer_model_id": protocol.renderer_model_id,
                "planned_mirrored_pairs": protocol.pair_count,
                "durable_renderings": len(records),
                "renderer_outputs_before_freeze": 0,
                "consumer_outputs_before_freeze": 0,
                "first_attempt_only": True,
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
            records=records,
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

    if len(records) < len(plans):
        renderer = _HFRelationshipP1mSurfaceRenderer(
            snapshot=snapshot,
            model_id=recipe.renderer.model_id,
            expected_weights_sha256=protocol.renderer_weights_sha256,
            generation_config_sha256=(
                protocol.renderer_generation_config_sha256
            ),
            device=resolved_device,
            torch_dtype=resolved_dtype,
            temperature=recipe.renderer.temperature,
            top_p=recipe.renderer.top_p,
            max_new_tokens=recipe.renderer.max_new_tokens,
        )
        remaining = len(plans) - len(records)
        allowance = remaining if args.max_new_renderings == 0 else min(
            remaining,
            args.max_new_renderings,
        )
        for index in range(len(records), len(records) + allowance):
            rendering = render_relationship_p1m_pair(
                renderer,
                plan=plans[index],
            )
            record = persist_relationship_p1m_generation_record(
                output_dir=output_dir,
                protocol=protocol,
                plans=plans,
                rendering=rendering,
            )
            print(
                json.dumps(
                    {
                        "stage": "rendering_checkpointed",
                        "record_index": record.record_index,
                        "pair_id": record.pair_id,
                        "attempt_index": record.rendering.attempt_index,
                        "durable_renderings": record.record_index + 1,
                        "planned_renderings": len(plans),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del renderer
        _release_runtime()

    records = load_relationship_p1m_generation_records(
        output_dir=output_dir,
        protocol=protocol,
        plans=plans,
    )
    if len(records) < len(plans):
        print(
            json.dumps(
                {
                    "stage": "partial",
                    "durable_renderings": len(records),
                    "remaining": len(plans) - len(records),
                },
                ensure_ascii=False,
            )
        )
        return 0

    _materialize_package(
        output_dir=output_dir,
        recipe=recipe,
        plans=plans,
        protocol=protocol,
        records=records,
    )
    attestation = load_relationship_p1m_generation_attestation(attestation_path)
    print(
        json.dumps(
            {
                "stage": "complete",
                "attestation_artifact_id": attestation.artifact_id,
                "dataset_fingerprint": attestation.dataset_fingerprint,
                "mirrored_pairs": attestation.pair_count,
                "scenes": attestation.scene_count,
                "consumer_outputs": attestation.consumer_outputs,
                "next_action": attestation.next_action,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
