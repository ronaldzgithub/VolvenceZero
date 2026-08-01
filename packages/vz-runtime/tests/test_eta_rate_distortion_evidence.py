"""Tests for the ETA rate-distortion instrument's gap detection and verdict.

These cover the two decisions that carry the retain/kill weight and can be
recomputed without an accelerator: whether a near-vertical gap exists on an
aggregate curve, and how the frozen/joint arm pair maps onto the verdict set.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from volvence_zero.agent.eta_rate_distortion_evidence import (
    GapAssessment,
    RateDistortionCurvePoint,
    RateDistortionPoint,
    adjudicate_rate_distortion,
    assess_gap,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import companion_test_plan_common as common  # noqa: E402


def _curve_point(
    *,
    arm: str,
    alpha: float,
    rate: float,
    distortion: float,
    distortion_std: float = 0.01,
    boundary_f1: float = 0.4,
) -> RateDistortionCurvePoint:
    return RateDistortionCurvePoint(
        arm=arm,
        alpha=alpha,
        rate_mean=rate,
        rate_std=0.0,
        distortion_mean=distortion,
        distortion_std=distortion_std,
        heldout_distortion_mean=distortion,
        boundary_f1_mean=boundary_f1,
        switch_frequency_mean=0.3,
        seed_count=3,
    )


def _curve(
    arm: str,
    cells: tuple[tuple[float, float, float], ...],
    *,
    distortion_std: float = 0.01,
) -> tuple[RateDistortionCurvePoint, ...]:
    return tuple(
        _curve_point(
            arm=arm,
            alpha=alpha,
            rate=rate,
            distortion=distortion,
            distortion_std=distortion_std,
        )
        for alpha, rate, distortion in cells
    )


def _gap(
    arm: str,
    *,
    detected: bool,
    boundary_f1_inside: float = 0.0,
    boundary_f1_outside: float = 0.0,
) -> GapAssessment:
    return GapAssessment(
        arm=arm,
        gap_detected=detected,
        distortion_span=1.0,
        rate_span=1.0,
        noise_scale=0.01,
        max_drop=0.9,
        max_drop_share=0.9,
        max_drop_rate_share=0.05,
        gap_low_alpha=0.03,
        gap_high_alpha=0.1,
        drop_share_threshold=0.5,
        rate_share_threshold=0.25,
        noise_multiple=2.0,
        boundary_f1_gap_region=boundary_f1_inside,
        boundary_f1_outside_gap=boundary_f1_outside,
    )


# A near-vertical segment: 90% of the distortion improvement bought with 5%
# of the rate span, between alpha=0.03 and alpha=0.1.
_VERTICAL_CELLS = (
    (0.01, 0.00, 2.00),
    (0.03, 0.10, 1.95),
    (0.10, 0.15, 1.05),
    (1.00, 1.00, 1.00),
)


def test_assess_gap_detects_the_near_vertical_segment() -> None:
    assessment = assess_gap(_curve("frozen", _VERTICAL_CELLS), arm="frozen")

    assert assessment.gap_detected is True
    assert assessment.gap_low_alpha == 0.03
    assert assessment.gap_high_alpha == 0.1
    assert assessment.max_drop_share == pytest.approx(0.9)
    assert assessment.max_drop_rate_share == pytest.approx(0.05)


def test_assess_gap_rejects_a_flat_curve() -> None:
    flat = _curve(
        "joint",
        (
            (0.01, 0.0, 1.5),
            (0.1, 0.5, 1.5),
            (1.0, 1.0, 1.5),
        ),
    )

    assessment = assess_gap(flat, arm="joint")

    assert assessment.gap_detected is False
    assert assessment.distortion_span == pytest.approx(0.0)


def test_assess_gap_rejects_a_gradual_slope_that_spends_too_much_rate() -> None:
    gradual = _curve(
        "frozen",
        (
            (0.01, 0.00, 2.00),
            (0.10, 0.33, 1.75),
            (1.00, 0.66, 1.50),
            (3.00, 1.00, 1.00),
        ),
    )

    assessment = assess_gap(gradual, arm="frozen")

    assert assessment.gap_detected is False
    assert assessment.max_drop_share >= assessment.drop_share_threshold
    assert assessment.max_drop_rate_share > assessment.rate_share_threshold


def test_assess_gap_rejects_a_drop_inside_the_cross_seed_noise_floor() -> None:
    noisy = _curve("frozen", _VERTICAL_CELLS, distortion_std=1.0)

    assessment = assess_gap(noisy, arm="frozen")

    assert assessment.gap_detected is False
    assert assessment.distortion_span <= (
        assessment.noise_multiple * assessment.noise_scale
    )


def test_assess_gap_requires_at_least_three_grid_cells() -> None:
    with pytest.raises(ValueError, match="at least three grid cells"):
        assess_gap(
            _curve("frozen", ((0.01, 0.0, 2.0), (1.0, 1.0, 1.0))),
            arm="frozen",
        )


def _two_arm_curves(
    *, separated: bool
) -> tuple[RateDistortionCurvePoint, ...]:
    frozen = _curve("frozen", _VERTICAL_CELLS, distortion_std=0.0)
    joint_cells = tuple(
        (alpha, rate, distortion + (1.0 if separated else 0.0))
        for alpha, rate, distortion in _VERTICAL_CELLS
    )
    joint = _curve("joint", joint_cells, distortion_std=0.0)
    return frozen + joint


def test_adjudication_declares_the_instrument_invalid_when_arms_match() -> None:
    result = adjudicate_rate_distortion(
        _two_arm_curves(separated=False),
        (_gap("frozen", detected=True), _gap("joint", detected=False)),
        arms=("frozen", "joint"),
    )

    assert result.arms_distinguishable is False
    assert result.verdict == "instrument-invalid"


def test_adjudication_retains_eta_when_only_the_frozen_arm_has_a_gap() -> None:
    result = adjudicate_rate_distortion(
        _two_arm_curves(separated=True),
        (
            _gap(
                "frozen",
                detected=True,
                boundary_f1_inside=0.7,
                boundary_f1_outside=0.3,
            ),
            _gap("joint", detected=False),
        ),
        arms=("frozen", "joint"),
    )

    assert result.arms_distinguishable is True
    assert result.verdict == "retain-eta"


def test_adjudication_weakens_the_verdict_when_boundary_f1_is_not_higher() -> None:
    result = adjudicate_rate_distortion(
        _two_arm_curves(separated=True),
        (
            _gap(
                "frozen",
                detected=True,
                boundary_f1_inside=0.3,
                boundary_f1_outside=0.7,
            ),
            _gap("joint", detected=False),
        ),
        arms=("frozen", "joint"),
    )

    assert result.verdict == "retain-weak"


def test_adjudication_kills_eta_when_the_frozen_arm_has_no_gap() -> None:
    result = adjudicate_rate_distortion(
        _two_arm_curves(separated=True),
        (_gap("frozen", detected=False), _gap("joint", detected=False)),
        arms=("frozen", "joint"),
    )

    assert result.verdict == "kill-eta"


def test_adjudication_is_inconclusive_when_the_joint_arm_also_gaps() -> None:
    result = adjudicate_rate_distortion(
        _two_arm_curves(separated=True),
        (_gap("frozen", detected=True), _gap("joint", detected=True)),
        arms=("frozen", "joint"),
    )

    assert result.verdict == "inconclusive-joint-arm-gap"


def test_adjudication_refuses_a_sweep_without_the_validity_control() -> None:
    frozen = _curve("frozen", _VERTICAL_CELLS, distortion_std=0.0)

    result = adjudicate_rate_distortion(
        frozen,
        (_gap("frozen", detected=True),),
        arms=("frozen",),
    )

    assert result.verdict == "incomplete-sweep"
    assert result.arms_distinguishable is False


def _report_fixture(verdict: str = "kill-eta"):
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        RATE_DISTORTION_SCHEMA_VERSION,
        RateDistortionEvidenceReport,
    )

    curves = _two_arm_curves(separated=True)
    point = RateDistortionPoint(
        arm="frozen",
        alpha=0.01,
        seed=0,
        train_rate=0.1,
        train_distortion=2.0,
        heldout_rate=0.1,
        heldout_distortion=2.1,
        baseline_train_distortion=2.5,
        baseline_heldout_distortion=2.6,
        mean_switch_probability=0.4,
        hard_switch_frequency=0.3,
        train_boundary_f1=0.4,
        heldout_boundary_f1=0.4,
        optimizer_steps=1,
        final_total_loss=2.0,
        final_grad_norm=0.1,
        wall_seconds=0.5,
    )
    return RateDistortionEvidenceReport(
        schema_version=RATE_DISTORTION_SCHEMA_VERSION,
        model_id="fixture-model",
        device="mps",
        runtime_origin="hf-pretrained",
        fallback_active=False,
        injection_layer_index=11,
        control_norm_cap=1.0,
        probe_hidden_norm=2.0,
        n_z=16,
        alpha_grid=(0.01, 0.03, 0.1, 1.0),
        seed_schedule=(0,),
        updates_per_run=1,
        learning_rate=0.02,
        substrate_learning_rate=1e-4,
        switch_threshold=0.55,
        arms=("frozen", "joint"),
        observation_protocol="partially-observable-no-remaining-route.v1",
        action_vocabulary=("alpha",),
        train_case_ids=("case-train",),
        heldout_case_ids=("case-heldout",),
        train_step_count=1,
        heldout_step_count=1,
        points=(point,),
        curves=curves,
        gaps=(_gap("frozen", detected=False), _gap("joint", detected=False)),
        arms_distinguishable=True,
        arm_separation=1.0,
        arm_separation_threshold=0.02,
        verdict=verdict,
        verdict_reason="fixture",
        description="fixture",
    )


def test_rate_distortion_cli_holds_the_shared_mps_lock_during_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_eta_rate_distortion as cli

    lock = tmp_path / "mps.lock"
    output = tmp_path / "artifact"
    observed: dict[str, object] = {}

    def fake_sweep(**_: object):
        # A second plan must not be able to take the device mid-sweep.
        try:
            with common.exclusive_mps_lock(lock, plan_id="seven-day"):
                observed["lock_held_during_sweep"] = False
        except common.MPSLockBusyError:
            observed["lock_held_during_sweep"] = True
        return _report_fixture()

    monkeypatch.setattr(cli, "run_eta_rate_distortion_evidence", fake_sweep)
    monkeypatch.setattr(
        cli,
        "require_mps",
        lambda: common.MPSAvailability("fixture", True, True, True),
    )
    monkeypatch.setattr(cli, "_maybe_plot", lambda *_a, **_k: "")
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "run_eta_rate_distortion.py",
            "--output-dir",
            str(output),
            "--mps-lock",
            str(lock),
        ),
    )

    cli.main()

    assert observed["lock_held_during_sweep"] is True
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["mps_exclusive_lock"] == str(lock)
    assert manifest["mps_attestation"]["fallback_disabled"] is True


def test_rate_distortion_cli_refuses_to_start_when_cpu_fallback_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_eta_rate_distortion as cli

    output = tmp_path / "artifact"

    def fail_require() -> None:
        raise common.MPSUnavailableError("cannot silently fall back to CPU")

    monkeypatch.setattr(cli, "require_mps", fail_require)
    monkeypatch.setattr(
        cli,
        "run_eta_rate_distortion_evidence",
        lambda **_: pytest.fail("the sweep started without an MPS attestation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "run_eta_rate_distortion.py",
            "--output-dir",
            str(output),
            "--mps-lock",
            str(tmp_path / "mps.lock"),
        ),
    )

    with pytest.raises(common.MPSUnavailableError, match="fall back to CPU"):
        cli.main()

    assert not (output / "artifact_manifest.json").exists()


def _preregistration(tmp_path: Path, **overrides: object) -> Path:
    import preregister_eta_rate_distortion as prereg

    payload = prereg.build_preregistration(
        alphas=(0.01, 0.1, 1.0),
        seeds=2,
        n_z=4,
        updates=1,
        learning_rate=0.02,
        substrate_learning_rate=1e-4,
        switch_threshold=0.55,
        model_id="fixture-model",
        device="mps",
        arms=("frozen", "joint"),
    )
    payload.update(overrides)
    target = tmp_path / "prereg.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _matching_args(**overrides: object):
    import argparse

    values = {
        "alphas": [0.01, 0.1, 1.0],
        "seeds": 2,
        "n_z": 4,
        "updates": 1,
        "learning_rate": 0.02,
        "substrate_learning_rate": 1e-4,
        "switch_threshold": 0.55,
        "model_id": "fixture-model",
        "device": "mps",
        "arms": ["frozen", "joint"],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_preregistration_accepts_the_execution_it_froze(tmp_path: Path) -> None:
    import run_eta_rate_distortion as cli

    payload = cli._validate_preregistration(
        json.loads(_preregistration(tmp_path).read_text(encoding="utf-8")),
        args=_matching_args(),
    )

    assert payload["claim_scope"] == "eta-temporal-abstraction-criterion-only"


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"alphas": [0.01, 0.1, 2.0]}, "alpha_grid"),
        ({"seeds": 3}, "seed_schedule"),
        ({"n_z": 16}, "n_z"),
        ({"updates": 40}, "updates_per_run"),
        ({"model_id": "other-model"}, "model_id"),
        ({"arms": ["frozen"]}, "arms"),
    ),
)
def test_preregistration_rejects_a_changed_sweep_variable(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    import run_eta_rate_distortion as cli

    with pytest.raises(cli.PreregistrationMismatch, match=message):
        cli._validate_preregistration(
            json.loads(_preregistration(tmp_path).read_text(encoding="utf-8")),
            args=_matching_args(**override),
        )


def test_preregistration_rejects_a_loosened_gap_threshold(
    tmp_path: Path,
) -> None:
    import run_eta_rate_distortion as cli

    path = _preregistration(
        tmp_path,
        gap_thresholds={
            "drop_share_threshold": 0.1,
            "rate_share_threshold": 0.9,
            "noise_multiple": 0.5,
        },
    )

    with pytest.raises(
        cli.PreregistrationMismatch, match="drop_share_threshold"
    ):
        cli._validate_preregistration(
            json.loads(path.read_text(encoding="utf-8")),
            args=_matching_args(),
        )


def test_preregistration_rejects_source_drift_since_freezing(
    tmp_path: Path,
) -> None:
    import run_eta_rate_distortion as cli

    path = _preregistration(
        tmp_path,
        frozen_source_files={
            "scripts/run_eta_rate_distortion.py": "0" * 64,
        },
    )

    with pytest.raises(cli.PreregistrationMismatch, match="source drift"):
        cli._validate_preregistration(
            json.loads(path.read_text(encoding="utf-8")),
            args=_matching_args(),
        )


def test_preregistration_rejects_a_foreign_schema(tmp_path: Path) -> None:
    import run_eta_rate_distortion as cli

    path = _preregistration(tmp_path, schema_version="something-else.v9")

    with pytest.raises(cli.PreregistrationMismatch, match="schema_version"):
        cli._validate_preregistration(
            json.loads(path.read_text(encoding="utf-8")),
            args=_matching_args(),
        )


def _checkpoint_cache(tmp_path: Path, *, resume: bool = False):
    import msc_prediction_checkpoint as checkpoint
    import run_eta_rate_distortion as cli

    store = checkpoint.PredictionRunCheckpointStore(
        output_dir=tmp_path / "run",
        configuration={"experiment_id": "eta-rate-distortion-criterion"},
        resume=resume,
        schema_namespace=cli.CHECKPOINT_SCHEMA_NAMESPACE,
    )
    return cli.RateDistortionCheckpointCache(store), store


def _point(*, arm: str, alpha: float, seed: int) -> RateDistortionPoint:
    return RateDistortionPoint(
        arm=arm,
        alpha=alpha,
        seed=seed,
        train_rate=0.1,
        train_distortion=1.5,
        heldout_rate=0.1,
        heldout_distortion=1.6,
        baseline_train_distortion=2.0,
        baseline_heldout_distortion=2.1,
        mean_switch_probability=0.4,
        hard_switch_frequency=0.2,
        train_boundary_f1=0.5,
        heldout_boundary_f1=0.5,
        optimizer_steps=1,
        final_total_loss=1.5,
        final_grad_norm=0.01,
        wall_seconds=0.1,
    )


def test_checkpoint_cache_round_trips_a_cell_across_processes(
    tmp_path: Path,
) -> None:
    cache, _ = _checkpoint_cache(tmp_path)
    point = _point(arm="frozen", alpha=0.03, seed=1)

    assert cache.load_point(arm="frozen", alpha=0.03, seed=1) is None
    cache.store_point(point)

    resumed_cache, _ = _checkpoint_cache(tmp_path, resume=True)
    assert resumed_cache.load_point(arm="frozen", alpha=0.03, seed=1) == point
    assert resumed_cache.load_point(arm="joint", alpha=0.03, seed=1) is None


def test_checkpoint_journal_refuses_an_existing_output_without_resume(
    tmp_path: Path,
) -> None:
    _checkpoint_cache(tmp_path)

    with pytest.raises(FileExistsError, match="immutable without --resume"):
        _checkpoint_cache(tmp_path)


def test_checkpoint_journal_namespace_is_not_the_prediction_journal(
    tmp_path: Path,
) -> None:
    _, store = _checkpoint_cache(tmp_path)

    state = json.loads(
        (store.output_dir / "run_state.json").read_text(encoding="utf-8")
    )
    assert state["schema_version"] == "eta-rate-distortion-run-state.v1"


def _stub_sweep_dependencies(
    monkeypatch: pytest.MonkeyPatch, *, executed: list[tuple[str, float, int]]
) -> None:
    """Replace the model-bound parts of the sweep with cheap stand-ins."""

    from types import SimpleNamespace

    from volvence_zero.agent import eta_rate_distortion_evidence as module

    scorer = SimpleNamespace(
        injection_layer_index=11,
        control_norm_cap=1.0,
        probe_hidden_norm=2.0,
        reset_joint_parameters=lambda: None,
        restore_and_freeze=lambda: None,
    )
    runtime = SimpleNamespace(
        model_id="fixture-model",
        runtime_origin="hf-pretrained",
        fallback_active=False,
        build_steered_action_scorer=lambda **_: scorer,
    )
    trace = SimpleNamespace(steps=(object(),))

    monkeypatch.setattr(
        module, "_build_eta_open_weight_runtime", lambda _config: runtime
    )
    monkeypatch.setattr(
        module, "_validate_eta_open_weight_runtime", lambda **_: None
    )
    monkeypatch.setattr(
        module, "build_default_eta_proof_environment", lambda: object()
    )
    monkeypatch.setattr(
        module,
        "_action_options",
        lambda _environment: (
            SimpleNamespace(action_id="alpha"),
            SimpleNamespace(action_id="beta"),
        ),
    )
    monkeypatch.setattr(
        module,
        "default_eta_proof_cases",
        lambda: (
            SimpleNamespace(case_id="train-1", split="train"),
            SimpleNamespace(case_id="heldout-1", split="heldout"),
        ),
    )
    monkeypatch.setattr(
        module, "_build_traces", lambda _cases, **_kwargs: (trace,)
    )
    monkeypatch.setattr(module, "_baseline_distortion", lambda *_a: 2.0)

    def fake_run_single(*, arm: str, alpha: float, seed: int, **_kwargs):
        executed.append((arm, alpha, seed))
        return _point(arm=arm, alpha=alpha, seed=seed)

    monkeypatch.setattr(module, "_run_single", fake_run_single)


def test_sweep_skips_cells_already_present_in_the_resume_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        run_eta_rate_distortion_evidence,
    )

    alphas = (0.01, 0.1, 1.0)
    cache, _ = _checkpoint_cache(tmp_path)
    cache.store_point(_point(arm="frozen", alpha=0.1, seed=0))

    executed: list[tuple[str, float, int]] = []
    _stub_sweep_dependencies(monkeypatch, executed=executed)

    report = run_eta_rate_distortion_evidence(
        alpha_grid=alphas,
        seed_schedule=(0,),
        n_z=4,
        updates_per_run=1,
        point_cache=cache,
    )

    assert ("frozen", 0.1, 0) not in executed
    assert len(executed) == 5
    assert len(report.points) == 6
    assert {(point.arm, point.alpha) for point in report.points} == {
        (arm, alpha) for arm in ("frozen", "joint") for alpha in alphas
    }


def test_sweep_journals_every_cell_it_computes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        run_eta_rate_distortion_evidence,
    )

    cache, store = _checkpoint_cache(tmp_path)
    _stub_sweep_dependencies(monkeypatch, executed=[])

    run_eta_rate_distortion_evidence(
        alpha_grid=(0.01, 0.1, 1.0),
        seed_schedule=(0,),
        n_z=4,
        updates_per_run=1,
        point_cache=cache,
    )

    assert len(store.immutable_file_manifest()) == 6


def test_released_mps_lock_does_not_advertise_a_live_owner(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "mps.lock"

    with common.exclusive_mps_lock(lock, plan_id="seven-day"):
        held = json.loads(lock.read_text(encoding="utf-8"))
    released = json.loads(lock.read_text(encoding="utf-8"))

    assert held["state"] == "held"
    assert released["state"] == "released"
    with common.exclusive_mps_lock(lock, plan_id="eta-rate-distortion"):
        assert json.loads(lock.read_text(encoding="utf-8"))["state"] == "held"
