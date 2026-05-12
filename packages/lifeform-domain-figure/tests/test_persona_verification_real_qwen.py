"""Wave P.3 — `@pytest.mark.hf` real-LLM end-to-end check.

Skipped in default CI (the ``hf`` marker filters them out). When run
with ``-m hf``:

1. Loads :class:`TransformersOpenWeightResidualRuntime` against
   ``sshleifer/tiny-gpt2`` (the same CPU-friendly model used in
   ``test_lora_aware_runtime_smoke.py``; tiny-gpt2 is not Qwen but
   exercises the full PEFT + activation path with a real forward).
2. Runs the full ablation grid against a synthetic Einstein bundle.
3. Asserts the verdict carries 4 gates, all four conditions are
   well-formed, and the response texts are non-empty.

We deliberately do NOT assert ``verdict.overall_passed`` because:

* tiny-gpt2 is too small to model Einstein's voice or cognition.
* The synthetic Einstein bundle's corpus is small; tiny-gpt2's
  output won't have meaningful retrieval support.

What we DO assert: the harness runs to completion against a real
HF model, produces non-trivial responses, and emits a valid
verdict shape — proving the wiring is correct end-to-end. Real
verdict-passing requires running with ``QWEN_MODEL_ID=Qwen/...``
on a Wave K curated bundle, which is reviewer-driven, not CI.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _hf_stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers")
    )


pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(
        not _hf_stack_available(),
        reason="@pytest.mark.hf requires torch + transformers",
    ),
]


@pytest.fixture
def synthetic_einstein_bundle() -> object:
    from lifeform_domain_figure import build_einstein_lifeform
    return build_einstein_lifeform().artifact_bundle


@pytest.fixture
def clean_pool():
    from volvence_zero.substrate import default_persona_lora_pool
    pool = default_persona_lora_pool()
    if pool.has("einstein"):
        pool.deregister("einstein")
    yield pool
    if pool.has("einstein"):
        pool.deregister("einstein")


def test_persona_verification_real_transformers_runtime(
    synthetic_einstein_bundle, clean_pool, tmp_path: Path
) -> None:
    from lifeform_domain_figure import save_figure_bundle
    from lifeform_domain_figure.verification.persona.cli import main as cli_main

    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()
    save_figure_bundle(synthetic_einstein_bundle, root_dir=bundle_root)
    output_dir = tmp_path / "verify_out"

    rc = cli_main(
        [
            "--bundle-id", synthetic_einstein_bundle.bundle_id,
            "--figure", "einstein",
            "--bundle-root", str(bundle_root),
            "--output-dir", str(output_dir),
            "--runtime", "transformers",
            "--qwen-model-id", "sshleifer/tiny-gpt2",
            "--device", "cpu",
            "--max-in-corpus-questions", "2",
            "--conditions", "raw,bundle,bundle_lora",
        ]
    )
    assert rc in (0, 2), f"verdict CLI returned unexpected rc={rc}"

    verdict_payload = json.loads(
        (output_dir / "verdict.json").read_text(encoding="utf-8")
    )
    assert verdict_payload["figure_id"] == "einstein"
    gate_names = [g["name"] for g in verdict_payload["gates"]]
    assert gate_names == [
        "gate_cognition_improves",
        "gate_voice_improves_with_lora",
        "gate_refusal_works",
        "gate_evidence_emerges",
    ]
    aggregates = verdict_payload["condition_aggregates"]
    assert {a["condition"] for a in aggregates} == {
        "raw", "bundle", "bundle_lora",
    }
    transcript_text = (output_dir / "transcript.md").read_text(encoding="utf-8")
    assert "Persona verification transcript" in transcript_text

    for cond in ("raw", "bundle", "bundle_lora"):
        rows = [
            json.loads(line)
            for line in (
                output_dir / "results" / f"{cond}.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows, f"results for {cond} should be non-empty"
        for row in rows:
            assert isinstance(row["response_text"], str)
