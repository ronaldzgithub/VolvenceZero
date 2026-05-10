"""Smoke tests for ``scripts/external_bench/compare_ablation.py``.

Exercise verdict logic with synthetic summary JSONs so the full
debt #29 verdict-emission machinery is locked in even before any
real eqbench3 run lands.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "external_bench" / "compare_ablation.py"


@pytest.fixture(scope="module")
def script_module():
    if not SCRIPT_PATH.exists():
        pytest.skip(f"compare_ablation.py not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("compare_ablation_mod", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_ablation_mod"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic-summary helpers
# ---------------------------------------------------------------------------


def _write_summary(
    tmp_path: pathlib.Path,
    *,
    track: str,
    rubric: float | None,
    request_mode: str,
    substrate: str = "Qwen/Qwen2.5-1.5B-Instruct",
    skip_red_line: str | None = None,
    file_name: str | None = None,
) -> pathlib.Path:
    path = tmp_path / (file_name or f"{track}.summary.json")
    attestation = {
        "frozen_substrate": True,
        "no_kernel_modification": True,
        "no_benchmark_text_in_system_prompt": True,
        "no_internal_architecture_terms_in_model_card": True,
        "track": track,
        "request_mode": request_mode,
        "substrate_model_id": substrate,
        "harness": "EQ-bench/eqbench3",
    }
    if skip_red_line is not None:
        # Force one red-line to False to test the verifier's refusal path.
        attestation[skip_red_line] = False
    summary = {
        "track": track,
        "attestation": attestation,
        "rubric_average": rubric,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loader / verifier
# ---------------------------------------------------------------------------


def test_load_summary_accepts_clean_summary(script_module, tmp_path) -> None:
    path = _write_summary(tmp_path, track="companion", rubric=68.5, request_mode="lifeform")
    result = script_module._load_summary(path)  # noqa: SLF001
    assert result.name == "companion"
    assert result.rubric_average == 68.5
    assert result.request_mode == "lifeform"


def test_load_summary_refuses_when_red_line_is_false(script_module, tmp_path) -> None:
    path = _write_summary(
        tmp_path,
        track="companion",
        rubric=68.5,
        request_mode="lifeform",
        skip_red_line="frozen_substrate",
    )
    with pytest.raises(ValueError) as excinfo:
        script_module._load_summary(path)  # noqa: SLF001
    assert "not True" in str(excinfo.value)
    assert "frozen_substrate" in str(excinfo.value)


def test_load_summary_refuses_when_attestation_missing(script_module, tmp_path) -> None:
    path = tmp_path / "missing_attestation.summary.json"
    path.write_text(json.dumps({"track": "companion", "rubric_average": 68.5}), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        script_module._load_summary(path)  # noqa: SLF001
    assert "attestation" in str(excinfo.value)


def test_verify_consistent_substrate_rejects_mismatch(script_module, tmp_path) -> None:
    p1 = _write_summary(
        tmp_path, track="companion", rubric=70.0, request_mode="lifeform",
        substrate="Qwen/Qwen2.5-1.5B-Instruct", file_name="a.json",
    )
    p2 = _write_summary(
        tmp_path, track="raw", rubric=68.0, request_mode="raw",
        substrate="Qwen/Qwen2.5-7B-Instruct", file_name="b.json",
    )
    r1 = script_module._load_summary(p1)  # noqa: SLF001
    r2 = script_module._load_summary(p2)  # noqa: SLF001
    with pytest.raises(ValueError) as excinfo:
        script_module._verify_consistent_substrate([r1, r2])  # noqa: SLF001
    assert "inconsistent substrate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _build_three_track_results(script_module, tmp_path, *, companion=68.0, cold=66.0, raw=65.0):
    p_companion = _write_summary(
        tmp_path, track="companion", rubric=companion, request_mode="lifeform",
        file_name="companion.summary.json",
    )
    p_cold = _write_summary(
        tmp_path, track="companion-cold", rubric=cold, request_mode="lifeform",
        file_name="cold.summary.json",
    )
    p_raw = _write_summary(
        tmp_path, track="raw", rubric=raw, request_mode="raw",
        file_name="raw.summary.json",
    )
    results = [
        script_module._load_summary(p_companion),  # noqa: SLF001
        script_module._load_summary(p_cold),  # noqa: SLF001
        script_module._load_summary(p_raw),  # noqa: SLF001
    ]
    return results


def test_verdict_pipeline_helps_bootstrap_helps_above_threshold(script_module, tmp_path) -> None:
    results = _build_three_track_results(
        script_module, tmp_path, companion=72.0, cold=68.0, raw=65.0
    )
    verdict = script_module._build_verdict(  # noqa: SLF001
        results=results, publish_threshold=70.0
    )
    assert verdict.pipeline_delta == 7.0
    assert verdict.bootstrap_delta == 4.0
    assert verdict.go_no_go == "go"
    joined = "\n".join(verdict.recommendations)
    assert "pipeline contribution +7.00 is meaningful" in joined
    assert "trained bootstraps add +4.00" in joined


def test_verdict_pipeline_negative_recommends_diagnosis(script_module, tmp_path) -> None:
    results = _build_three_track_results(
        script_module, tmp_path, companion=60.0, cold=60.0, raw=68.0
    )
    verdict = script_module._build_verdict(  # noqa: SLF001
        results=results, publish_threshold=65.0
    )
    assert verdict.pipeline_delta == -8.0
    joined = "\n".join(verdict.recommendations)
    assert "NEGATIVE" in joined
    assert "follow-up debt" in joined


def test_verdict_below_threshold_holds(script_module, tmp_path) -> None:
    results = _build_three_track_results(
        script_module, tmp_path, companion=55.0, cold=54.0, raw=53.0
    )
    verdict = script_module._build_verdict(  # noqa: SLF001
        results=results, publish_threshold=65.0
    )
    assert verdict.go_no_go == "hold"
    joined = "\n".join(verdict.recommendations)
    assert "HOLD" in joined


def test_verdict_with_only_two_tracks_omits_unknown_delta(script_module, tmp_path) -> None:
    p_companion = _write_summary(
        tmp_path, track="companion", rubric=70.0, request_mode="lifeform",
        file_name="companion.summary.json",
    )
    p_raw = _write_summary(
        tmp_path, track="raw", rubric=65.0, request_mode="raw",
        file_name="raw.summary.json",
    )
    results = [
        script_module._load_summary(p_companion),  # noqa: SLF001
        script_module._load_summary(p_raw),  # noqa: SLF001
    ]
    verdict = script_module._build_verdict(  # noqa: SLF001
        results=results, publish_threshold=65.0
    )
    assert verdict.pipeline_delta == 5.0
    assert verdict.bootstrap_delta is None  # cold track absent


def test_verdict_with_missing_rubric_returns_insufficient_data(script_module, tmp_path) -> None:
    p_companion = _write_summary(
        tmp_path, track="companion", rubric=None, request_mode="lifeform",
        file_name="companion.summary.json",
    )
    results = [script_module._load_summary(p_companion)]  # noqa: SLF001
    verdict = script_module._build_verdict(  # noqa: SLF001
        results=results, publish_threshold=65.0
    )
    assert verdict.go_no_go == "insufficient_data"


def test_render_table_includes_threshold_and_recommendations(script_module, tmp_path) -> None:
    results = _build_three_track_results(script_module, tmp_path)
    verdict = script_module._build_verdict(  # noqa: SLF001
        results=results, publish_threshold=70.0
    )
    rendered = script_module._render_table(verdict)  # noqa: SLF001
    assert "publish_threshold: 70.00" in rendered
    assert "Recommendations:" in rendered
    assert "go/no-go" in rendered.lower()


def test_main_emits_json_when_output_passed(script_module, tmp_path) -> None:
    results = _build_three_track_results(script_module, tmp_path)
    summary_paths = [r.summary_path for r in results]
    out_path = tmp_path / "verdict.json"
    rc = script_module.main(  # noqa: SLF001 - this is the public main()
        [
            "--summaries",
            *[str(p) for p in summary_paths],
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    body = json.loads(out_path.read_text(encoding="utf-8"))
    assert "tracks" in body
    assert body["go_no_go"] in {"go", "hold", "insufficient_data"}
