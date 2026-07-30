#!/usr/bin/env python3
"""Bake a Relationship projector from frozen-model contrastive residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from run_state_kv_identification import (  # noqa: E402
    DEFAULT_MODEL_ID,
    _fingerprint_weights,
    _resolve_local_weights,
)
from volvence_zero.relationship_conditioning import (  # noqa: E402
    RELATIONSHIP_CONDITIONING_READOUT_LABELS,
)
from volvence_zero.substrate import (  # noqa: E402
    TransformersOpenWeightResidualRuntime,
    build_relationship_contrastive_projector_artifact,
)

DEFAULT_ANCHORS = (
    REPO_ROOT
    / "packages"
    / "vz-substrate"
    / "src"
    / "volvence_zero"
    / "substrate"
    / "prompts"
    / "relationship_conditioning_projector_anchors.json"
)


def _load_anchors(path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = set(RELATIONSHIP_CONDITIONING_READOUT_LABELS)
    if set(raw) != expected:
        raise ValueError(
            "Relationship projector anchors must match the owner readout; "
            f"missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )
    anchors: dict[str, dict[str, tuple[str, ...]]] = {}
    for label in RELATIONSHIP_CONDITIONING_READOUT_LABELS:
        item = raw[label]
        positive = tuple(str(text).strip() for text in item["positive"])
        negative = tuple(str(text).strip() for text in item["negative"])
        if not positive or not negative or not all((*positive, *negative)):
            raise ValueError(
                f"Relationship projector anchor {label!r} needs non-empty "
                "positive and negative material."
            )
        anchors[label] = {"positive": positive, "negative": negative}
    return anchors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--anchors", default=str(DEFAULT_ANCHORS))
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "artifacts"
            / "state_kv"
            / "projectors"
            / "qwen2.5-0.5b-relationship-contrastive.json"
        ),
    )
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)

    anchor_path = Path(args.anchors).expanduser().resolve()
    anchors = _load_anchors(anchor_path)
    weights_root = _resolve_local_weights(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    weights = _fingerprint_weights(
        model_id=args.model_id,
        weights_root=weights_root,
    )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=args.model_id,
        pretrained_source=str(weights_root),
        device=args.device,
        hook_layer_selection="middle",
        local_files_only=True,
        runtime_origin="hf-local",
        allow_offline_substrate_training=True,
    )
    injection_layer = runtime.hook_layer_indices[0]
    rows: dict[str, tuple[float, ...]] = {}
    sample_count = 0
    for label in RELATIONSHIP_CONDITIONING_READOUT_LABELS:
        material = anchors[label]
        positive, negative = runtime.capture_for_contrastive(
            positive_texts=material["positive"],
            negative_texts=material["negative"],
            layer_index=injection_layer,
        )
        rows[label] = tuple(
            positive_value - negative_value
            for positive_value, negative_value in zip(
                positive,
                negative,
                strict=True,
            )
        )
        sample_count += len(material["positive"]) + len(material["negative"])
        print(f"captured {label} ({sample_count} anchor texts)", flush=True)

    anchor_bytes = anchor_path.read_bytes()
    source_fingerprint = hashlib.sha256(
        (
            str(weights["weights_sha256"])
            + ":"
            + hashlib.sha256(anchor_bytes).hexdigest()
        ).encode("utf-8")
    ).hexdigest()
    artifact = build_relationship_contrastive_projector_artifact(
        model_id=args.model_id,
        hidden_size=runtime.hidden_size,
        vector_labels=RELATIONSHIP_CONDITIONING_READOUT_LABELS,
        layer_indices=runtime.hook_layer_indices,
        contrastive_rows=rows,
        source_fingerprint=source_fingerprint,
        sample_count=sample_count,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.to_json() + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "relationship-projector-bake.v1",
        "artifact_id": artifact.artifact_id,
        "projector_version": artifact.projector_version,
        "training_mode": artifact.training_mode,
        "model_id": args.model_id,
        "weights_sha256": weights["weights_sha256"],
        "anchor_sha256": hashlib.sha256(anchor_bytes).hexdigest(),
        "injection_layers": list(runtime.hook_layer_indices),
        "basis_row_norm": 1.0,
        "per_layer_gain": 1.0,
        "base_model_mutated": False,
    }
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"artifact_id={artifact.artifact_id}")
    print(f"projector_version={artifact.projector_version}")
    print(f"written: {output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
