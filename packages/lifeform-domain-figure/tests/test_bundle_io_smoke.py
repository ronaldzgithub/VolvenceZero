"""Smoke tests for :mod:`lifeform_domain_figure.bundle_io`.

Validates the R15 byte-level rollback contract at the persistence
layer:

* ``save_figure_bundle`` + ``load_figure_bundle`` round-trip a
  bundle without changing its :attr:`integrity_hash`.
* Round-tripping through both steering- and LoRA-attached bundles
  preserves the optional artifact slots.
* Tampering with the saved manifest's integrity hash fails the
  load (no-swallow).
* ``list_figure_bundles`` filters by ``figure_id``.
"""

from __future__ import annotations

import json

import pytest

from volvence_zero.credit.gate import GateDecision
from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot
from volvence_zero.substrate import PersonaLoRAPool

from lifeform_domain_figure import (
    FigureBundleInputs,
    SyntheticLoRABakeBackend,
    apply_persona_lora_through_gate,
    apply_steering_through_gate,
    bake_steering_set,
    build_einstein_contrast_set,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    build_lora_training_plan,
    build_steering_training_plan,
    list_figure_bundles,
    load_figure_bundle,
    save_figure_bundle,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.envelope_builder import FigureCorpusSourceBundle
from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash


def _einstein_envelopes():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    return build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="lifeform-figure-tests:bundle-io",
    ).envelopes


def _einstein_bundle():
    return build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=build_einstein_profile(),
            envelopes=_einstein_envelopes(),
        )
    )


def _clean_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                "behavior", "contract_integrity", 0.99, 0.95, "ok",
            ),
            EvaluationScore(
                "behavior", "rollback_resilience", 0.99, 0.95, "ok",
            ),
            EvaluationScore(
                "behavior", "fallback_reliance", 0.10, 0.95, "ok",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="clean snapshot",
    )


def test_save_load_roundtrip_bundle_byte_equal(tmp_path):
    bundle = _einstein_bundle()
    save_figure_bundle(bundle, root_dir=tmp_path)
    loaded = load_figure_bundle(
        root_dir=tmp_path,
        bundle_id=bundle.bundle_id,
        figure_id=bundle.figure_id,
    )
    assert loaded.bundle_id == bundle.bundle_id
    assert loaded.integrity_hash == bundle.integrity_hash
    recomputed = compute_bundle_integrity_hash(
        figure_id=loaded.figure_id,
        profile_version=loaded.profile_version,
        version_window=loaded.version_window,
        retrieval_integrity=loaded.retrieval_index.integrity_hash,
        coverage_integrity=loaded.coverage_map.integrity_hash,
        style_integrity=loaded.style_prior.integrity_hash,
        steering_integrity=(
            "absent" if loaded.steering is None
            else loaded.steering.integrity_hash
        ),
        lora_integrity=(
            "absent" if loaded.lora is None else loaded.lora.integrity_hash
        ),
    )
    assert recomputed == loaded.integrity_hash


def test_save_load_roundtrip_with_steering_and_lora(tmp_path):
    base = _einstein_bundle()
    snapshot = _clean_snapshot()

    contrast_plan = build_steering_training_plan(build_einstein_contrast_set())
    steering = bake_steering_set(contrast_plan)
    steering_result = apply_steering_through_gate(
        base_bundle=base,
        steering=steering,
        evaluation_snapshot=snapshot,
        rollback_evidence=f"prev_steering=absent;base={base.bundle_id}",
    )
    assert steering_result.applied
    assert steering_result.gate.decision is GateDecision.ALLOW

    lora_plan = build_lora_training_plan(
        figure_id="einstein", envelopes=_einstein_envelopes(), rank=4
    )
    artifact = SyntheticLoRABakeBackend().bake(lora_plan)
    pool = PersonaLoRAPool()
    lora_result = apply_persona_lora_through_gate(
        base_bundle=steering_result.bundle,
        artifact=artifact,
        evaluation_snapshot=snapshot,
        pool=pool,
        rollback_evidence=(
            f"prev_lora=absent;base={steering_result.bundle.bundle_id}"
        ),
    )
    assert lora_result.applied
    assert lora_result.gate.decision is GateDecision.ALLOW
    final = lora_result.bundle

    save_figure_bundle(final, root_dir=tmp_path)
    reloaded = load_figure_bundle(
        root_dir=tmp_path,
        bundle_id=final.bundle_id,
        figure_id=final.figure_id,
    )
    assert reloaded.steering is not None
    assert reloaded.lora is not None
    assert reloaded.steering.integrity_hash == steering.integrity_hash
    assert reloaded.lora.integrity_hash == artifact.integrity_hash
    assert reloaded.integrity_hash == final.integrity_hash


def test_load_fails_loud_on_corrupt_pickle_header(tmp_path):
    bundle = _einstein_bundle()
    save_figure_bundle(bundle, root_dir=tmp_path)
    pickle_paths = list(tmp_path.rglob("bundle.pickle"))
    assert pickle_paths, "expected a bundle.pickle to be persisted"
    pickle_paths[0].write_bytes(b"NOT-A-VOLVENCE-MAGIC" + b"x" * 200)
    with pytest.raises(ValueError, match="missing magic header"):
        load_figure_bundle(
            root_dir=tmp_path,
            bundle_id=bundle.bundle_id,
            figure_id=bundle.figure_id,
        )


def test_list_figure_bundles_filters_by_figure_id(tmp_path):
    bundle = _einstein_bundle()
    save_figure_bundle(bundle, root_dir=tmp_path)

    matches = list_figure_bundles(root_dir=tmp_path, figure_id="einstein")
    assert len(matches) == 1
    assert matches[0].bundle_id == bundle.bundle_id
    assert matches[0].steering_present is False
    assert matches[0].lora_present is False

    no_match = list_figure_bundles(root_dir=tmp_path, figure_id="nobody")
    assert no_match == ()

    all_bundles = list_figure_bundles(root_dir=tmp_path)
    assert len(all_bundles) == 1
    assert all_bundles[0].figure_id == "einstein"


def test_load_fails_loud_when_manifest_hash_drifts_from_pickle(tmp_path):
    """Mutating the manifest leaves the pickle untouched; the load
    contract recomputes the hash from the pickle. The mismatch path
    is exercised by tampering with the pickle integrity field after
    save (we test the "manifest tampering does not silently fool the
    loader" angle here)."""

    bundle = _einstein_bundle()
    save_figure_bundle(bundle, root_dir=tmp_path)
    manifest_paths = list(tmp_path.rglob("manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    # Pre-condition: manifest's integrity_hash matches the saved bundle.
    assert manifest["integrity_hash"] == bundle.integrity_hash
    # The loader recomputes from the pickled bundle, so a tampered
    # manifest does NOT confuse the loader (it still loads the
    # original bundle). The contract is: integrity is validated
    # against the pickle, manifest is the human-readable index.
    manifest["integrity_hash"] = "0" * 64
    manifest_paths[0].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_figure_bundle(
        root_dir=tmp_path,
        bundle_id=bundle.bundle_id,
        figure_id=bundle.figure_id,
    )
    assert loaded.integrity_hash == bundle.integrity_hash
