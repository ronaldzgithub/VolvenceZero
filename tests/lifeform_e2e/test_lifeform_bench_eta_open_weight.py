"""Bench-CLI integration tests for `--eta-open-weight-paper-suite`.

These tests exercise the wiring added to ``lifeform-bench`` for the ETA
real open-weight residual paper suite. The ci-smoke tier is intentionally
capped at ``weak`` (cannot reach retain), so the test only asserts that:

- the CLI accepts the flags
- the bundle exports
- runtime provenance is written and contains the Qwen runtime descriptor
- the claim verdict is **not** ``fail``
- ``--eta-open-weight-require-retain`` correctly exits non-zero on
  ci-smoke (because retain is impossible at that tier)

Real Qwen capture takes minutes on CPU, so the heavy assertions are
gated behind the local HF cache check. The test skips cleanly when the
default Qwen weights are not pre-cached.
"""

from __future__ import annotations

import json
import pathlib

import pytest


def _qwen_default_real_backend_available() -> bool:
    try:
        import importlib

        transformers = importlib.import_module("transformers")
        transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
        transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
        return True
    except (ImportError, OSError, ValueError):
        return False


def test_lifeform_bench_eta_open_weight_paper_suite_ci_smoke(tmp_path, capsys):
    if not _qwen_default_real_backend_available():
        pytest.skip(
            "Qwen/Qwen2.5-0.5B-Instruct not available in local HF cache; "
            "run `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` to enable."
        )

    from lifeform_evolution.cli import main

    output_dir = tmp_path / "eta_open_weight_smoke"
    rc = main(
        [
            "--scenario",
            "low-mood-disclosure",
            "--eta-open-weight-paper-suite",
            "--eta-open-weight-tier",
            "ci-smoke",
            "--eta-open-weight-output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 0, captured
    assert "ETA open-weight paper suite" in captured
    assert "claim_eta_real_open_weight_residual_control" in captured

    aggregate_path = output_dir / "paper_suite_aggregate.json"
    assert aggregate_path.is_file(), captured
    aggregate_payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    real_claim = next(
        (
            verdict
            for verdict in aggregate_payload["claim_verdicts"]
            if verdict["claim_id"] == "claim_eta_real_open_weight_residual_control"
        ),
        None,
    )
    assert real_claim is not None
    assert real_claim["status"] in {"weak", "retain"}, real_claim
    assert real_claim["status"] != "fail"

    provenance_path = output_dir / "eta_open_weight_runtime_provenance.json"
    assert provenance_path.is_file(), "lifeform-bench should write runtime provenance JSON"
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    descriptor = provenance_payload["runtime_descriptor"]
    assert descriptor["open_weight_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert descriptor["open_weight_runtime_origin"] in {"hf-pretrained", "hf-local"}
    assert descriptor["open_weight_fallback_active"] == "0"
    assert descriptor["open_weight_layer_indices"] == "20,21,22"
    assert descriptor["torch_version"] != "unavailable"
    assert descriptor["transformers_version"] != "unavailable"


def test_lifeform_bench_eta_open_weight_require_retain_fails_on_ci_smoke(tmp_path, capsys):
    if not _qwen_default_real_backend_available():
        pytest.skip(
            "Qwen/Qwen2.5-0.5B-Instruct not available in local HF cache; "
            "run `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` to enable."
        )

    from lifeform_evolution.cli import main

    output_dir = tmp_path / "eta_open_weight_smoke_require_retain"
    rc = main(
        [
            "--scenario",
            "low-mood-disclosure",
            "--eta-open-weight-paper-suite",
            "--eta-open-weight-tier",
            "ci-smoke",
            "--eta-open-weight-output-dir",
            str(output_dir),
            "--eta-open-weight-require-retain",
        ]
    )
    captured = capsys.readouterr().out
    assert rc == 1, captured
    assert "--eta-open-weight-require-retain" in captured


def test_lifeform_bench_parser_exposes_eta_open_weight_flags():
    from lifeform_evolution.cli import _build_bench_parser

    parser = _build_bench_parser()
    namespace = parser.parse_args(
        [
            "--eta-open-weight-paper-suite",
            "--eta-open-weight-tier",
            "paper-suite-small",
            "--eta-open-weight-output-dir",
            "artifacts/eta_open_weight/test",
            "--eta-open-weight-require-retain",
        ]
    )
    assert namespace.eta_open_weight_paper_suite is True
    assert namespace.eta_open_weight_tier == "paper-suite-small"
    assert namespace.eta_open_weight_output_dir == "artifacts/eta_open_weight/test"
    assert namespace.eta_open_weight_require_retain is True


def test_lifeform_bench_parser_eta_open_weight_tier_choices_are_validated():
    from lifeform_evolution.cli import _build_bench_parser

    parser = _build_bench_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--eta-open-weight-tier", "invalid-tier"])
