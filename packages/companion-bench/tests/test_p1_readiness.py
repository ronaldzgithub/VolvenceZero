from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts" / "companion_bench"
sys.path.insert(0, str(SCRIPTS))

from p1_readiness import (  # noqa: E402
    P1_PORTS,
    P1ReadinessError,
    fingerprint_weights,
    free_p1_ports,
    require_non_qwen_models,
    write_track_fingerprints,
)


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weight_fingerprint_is_content_addressed(tmp_path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    before = fingerprint_weights("Qwen/test", tmp_path)
    second.write_bytes(b"changed")
    after = fingerprint_weights("Qwen/test", tmp_path)

    assert before.weight_file_count == 2
    assert before.weights_sha256 != after.weights_sha256


def test_track_fingerprints_are_required_for_p1(tmp_path) -> None:
    module = _load_script("assert_same_substrate.py")
    fingerprint = fingerprint_weights("Qwen/test", _write_weight(tmp_path))
    write_track_fingerprints(fingerprint, tmp_path / "out", ("raw", "volvence"))
    loaded = [
        module.fingerprint_from_file(
            track,
            tmp_path / "out" / track / "substrate_fingerprint.json",
        )
        for track in ("raw", "volvence")
    ]
    canonical = module.assert_consistent(loaded, require_weights_sha256=True)
    assert canonical.weights_sha256 == fingerprint.weights_sha256

    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps({"substrate_model_id": "Qwen/test"}),
        encoding="utf-8",
    )
    with pytest.raises(module.SubstrateMismatchError, match="weights_sha256"):
        module.assert_consistent(
            [module.fingerprint_from_file("raw", missing), loaded[1]],
            require_weights_sha256=True,
        )


def test_qwen_judge_is_rejected() -> None:
    with pytest.raises(P1ReadinessError, match="cross-family"):
        require_non_qwen_models((("arc judge", "qwen-plus"),))


def test_p1_ports_are_single_lifeform_topology() -> None:
    assert P1_PORTS == (8000, 8500, 8501, 8502, 8600)


def test_free_p1_ports_kills_listeners(monkeypatch) -> None:
    occupied_calls = {"n": 0}

    def fake_occupied(host: str = "127.0.0.1") -> tuple[int, ...]:
        occupied_calls["n"] += 1
        return (8000,) if occupied_calls["n"] <= 2 else ()

    monkeypatch.setattr("p1_readiness.occupied_p1_ports", fake_occupied)
    monkeypatch.setattr("p1_readiness._pids_listening_on_port", lambda port: (4242,))
    killed: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)

    monkeypatch.setattr("p1_readiness.os.kill", fake_kill)
    monkeypatch.setattr("p1_readiness.time.sleep", lambda _s: None)

    result = free_p1_ports(sigkill_after_s=0, wait_free_s=1, poll_interval_s=0)
    assert result == (4242,)
    assert killed


def test_roster_routes_volvence_tracks_through_vertical_query() -> None:
    import yaml

    roster = yaml.safe_load(
        (SCRIPTS / "reference_systems.same_substrate_ablation.yaml").read_text(
            encoding="utf-8"
        )
    )
    by_id = {entry["submission_id"]: entry for entry in roster["systems"]}
    expected = {
        "abl-volvence": "companion",
        "abl-volvence-cold": "companion-cold",
        "abl-pe-off": "companion-pe-drive-off",
        "abl-eta-off": "companion-eta-off",
        "abl-active-learning-off": "companion-active-learning-off",
        "abl-lora-adapter": "companion-lora-adapter",
    }
    for submission_id, vertical in expected.items():
        assert by_id[submission_id]["base_url"] == (
            f"http://127.0.0.1:8000/v1?vertical={vertical}"
        )


def test_p1_readiness_rejects_existing_scores_without_resume(tmp_path) -> None:
    runner = _load_script("run_same_substrate_ablation.py")
    (tmp_path / "scores" / "partial").mkdir(parents=True)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "companion-p1-run-manifest.v1",
                "temporal_bootstrap_sha256": "a",
                "regime_bootstrap_sha256": "b",
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(output_dir=tmp_path, resume=False)
    with pytest.raises(SystemExit, match="--resume"):
        runner._assert_real_run_ready(args, tier="p1")


def test_p1_dry_run_needs_no_live_endpoints(tmp_path) -> None:
    runner = _load_script("run_same_substrate_ablation.py")
    rc = runner.main(
        [
            "--phase",
            "p1",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0


def test_score_resume_reuses_completed_summary(monkeypatch, tmp_path) -> None:
    scorer = _load_script("score_reference_systems.py")
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        """
systems:
  - submission_id: completed
    system_name: Completed
    model_identifier: completed-model
    base_url: http://127.0.0.1:1/v1
    api_key_env: UNUSED
    leaderboard_category: bespoke
""".lstrip(),
        encoding="utf-8",
    )
    summary = tmp_path / "scores" / "completed" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"aggregate": {"final_mean": 1.0}, "arc_count": 30}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scorer.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resume unexpectedly started a submission")
        ),
    )

    rc = scorer.main(
        [
            "--roster",
            str(roster),
            "--output-dir",
            str(tmp_path / "scores"),
            "--user-sim-model",
            "openai/gpt-5-mini",
            "--user-sim-key-env",
            "UNUSED",
            "--perturn-model",
            "openai/gpt-5-mini",
            "--perturn-key-env",
            "UNUSED",
            "--arc-model",
            "anthropic/claude",
            "--arc-key-env",
            "UNUSED",
            "--resume",
        ]
    )
    assert rc == 0


def test_score_resume_rejects_incomplete_summary(monkeypatch, tmp_path) -> None:
    scorer = _load_script("score_reference_systems.py")
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        """
systems:
  - submission_id: partial
    system_name: Partial
    model_identifier: partial-model
    base_url: http://127.0.0.1:1/v1
    api_key_env: UNUSED
    leaderboard_category: bespoke
""".lstrip(),
        encoding="utf-8",
    )
    summary = tmp_path / "scores" / "partial" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"aggregate": {"final_mean": 1.0}, "arc_count": 2}),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scorer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scorer,
        "_expected_arc_count",
        lambda _args: 30,
    )

    rc = scorer.main(
        [
            "--roster",
            str(roster),
            "--output-dir",
            str(tmp_path / "scores"),
            "--user-sim-model",
            "openai/gpt-5-mini",
            "--user-sim-key-env",
            "UNUSED",
            "--perturn-model",
            "openai/gpt-5-mini",
            "--perturn-key-env",
            "UNUSED",
            "--arc-model",
            "anthropic/claude",
            "--arc-key-env",
            "UNUSED",
            "--resume",
        ]
    )
    assert rc == 0
    assert calls, "incomplete summary should trigger a rerun"


def _write_weight(root: pathlib.Path) -> pathlib.Path:
    model = root / "weights"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    return model
