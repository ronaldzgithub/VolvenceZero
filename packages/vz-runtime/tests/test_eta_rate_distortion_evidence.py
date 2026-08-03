"""Tests for the ETA rate-distortion instrument's gap detection and verdict.

These cover the two decisions that carry the retain/kill weight and can be
recomputed without an accelerator: whether a near-vertical gap exists on an
aggregate curve, and how the frozen/joint arm pair maps onto the verdict set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from volvence_zero.agent.eta_rate_distortion_evidence import (
    GapAssessment,
    RateAxisResponse,
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
        posterior_parameterization="legacy",
        rate_gating="per-step",
        gate_mode="continuous",
        corpus_origin="fixture",
        corpus_seed=0,
        corpus_objective_count=2,
        train_route_count=1,
        heldout_route_count=1,
        action_vocabulary=("alpha",),
        train_case_ids=("case-train",),
        heldout_case_ids=("case-heldout",),
        train_step_count=1,
        heldout_step_count=1,
        points=(point,),
        curves=curves,
        gaps=(_gap("frozen", detected=False), _gap("joint", detected=False)),
        rate_axis_responses=(
            RateAxisResponse(
                arm="frozen",
                spearman_alpha_rate=0.0,
                rate_span=1.0,
                rate_min=0.0,
                rate_max=1.0,
                alpha_count=4,
                description="fixture",
            ),
            RateAxisResponse(
                arm="joint",
                spearman_alpha_rate=0.0,
                rate_span=1.0,
                rate_min=0.0,
                rate_max=1.0,
                alpha_count=4,
                description="fixture",
            ),
        ),
        arms_distinguishable=True,
        arm_separation=1.0,
        arm_separation_threshold=0.02,
        verdict=verdict,
        verdict_reason="fixture",
        description="fixture",
    )


def test_v2_protocol_states_the_plan_once_and_drops_progress_leaks() -> None:
    """v2 must give source_text only at step 0 and never leak completed objectives."""

    import dataclasses

    from volvence_zero.agent import eta_rate_distortion_evidence as module
    from volvence_zero.agent.eta_proof_benchmark import (
        generate_eta_proof_corpus,
    )
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        OBSERVATION_PROTOCOL_V1,
        OBSERVATION_PROTOCOL_V2,
        OBSERVATION_PROTOCOL_V3,
        _rate_distortion_observation_bundle,
    )

    @dataclasses.dataclass(frozen=True)
    class _StubSnapshot:
        description: str = "stub"
        residual_sequence: tuple[object, ...] = ()

    corpus = generate_eta_proof_corpus(
        seed=7,
        objective_count=4,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=6,
        heldout_route_count=3,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )
    case = corpus.train_cases[0]

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            module,
            "_runtime_capture_snapshot",
            lambda **_kwargs: _StubSnapshot(),
        )
        _, v1_texts, v1_targets = _rate_distortion_observation_bundle(
            case,
            environment=corpus.environment,
            open_weight_runtime=object(),
            protocol_version=OBSERVATION_PROTOCOL_V1,
        )
        _, v2_texts, v2_targets = _rate_distortion_observation_bundle(
            case,
            environment=corpus.environment,
            open_weight_runtime=object(),
            protocol_version=OBSERVATION_PROTOCOL_V2,
        )
        _, v3_texts, v3_targets = _rate_distortion_observation_bundle(
            case,
            environment=corpus.environment,
            open_weight_runtime=object(),
            protocol_version=OBSERVATION_PROTOCOL_V3,
        )

    # Same expert supervision under both surfaces.
    assert v1_targets == v2_targets
    assert len(v2_texts) == len(v1_texts)

    # The route plan appears exactly once, only at step 0.
    assert case.source_text in v2_texts[0]
    for text in v2_texts[1:]:
        assert case.source_text not in text
        assert "Route plan" not in text

    # v2 never leaks progress via completed objectives, and never repeats the
    # per-step task-context fingerprint.
    for text in v2_texts:
        assert "Completed objectives" not in text
        assert "Task context" not in text
    # v3 makes the step-0 plan readable to the frozen substrate while keeping
    # the same locality and no-progress-leak contract.
    assert v3_targets == v2_targets
    assert v3_texts[0].startswith("Route plan: visit ")
    assert case.source_text not in " ".join(v3_texts)
    assert all("Completed objectives" not in text for text in v3_texts)
    assert all("Route plan" not in text for text in v3_texts[1:])
    # v1, by contrast, repeats the fingerprint and progress on every step.
    assert all("Task context" in text for text in v1_texts)


def test_v4_protocol_staggers_revelation_at_objective_arrivals() -> None:
    """v4: step 0 names only the first objective; each arrival at an
    objective reveals the next; corridor steps stay strictly local."""

    import dataclasses

    import pytest as _pytest

    from volvence_zero.agent import eta_rate_distortion_evidence as module
    from volvence_zero.agent.eta_proof_benchmark import (
        generate_eta_proof_corpus,
    )
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        OBSERVATION_PROTOCOL_V3,
        OBSERVATION_PROTOCOL_V4,
        _rate_distortion_observation_bundle,
    )

    @dataclasses.dataclass(frozen=True)
    class _StubSnapshot:
        description: str = "stub"
        residual_sequence: tuple[object, ...] = ()

    corpus = generate_eta_proof_corpus(
        seed=7,
        objective_count=4,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=6,
        heldout_route_count=3,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )

    with _pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            module,
            "_runtime_capture_snapshot",
            lambda **_kwargs: _StubSnapshot(),
        )
        for case in corpus.train_cases + corpus.heldout_cases:
            objectives = tuple(
                waypoint
                for waypoint in case.route_signature
                if corpus.environment.location(waypoint).is_objective
            )
            _, v3_texts, v3_targets = _rate_distortion_observation_bundle(
                case,
                environment=corpus.environment,
                open_weight_runtime=object(),
                protocol_version=OBSERVATION_PROTOCOL_V3,
            )
            _, v4_texts, v4_targets = _rate_distortion_observation_bundle(
                case,
                environment=corpus.environment,
                open_weight_runtime=object(),
                protocol_version=OBSERVATION_PROTOCOL_V4,
            )

            # Identical expert supervision and step count.
            assert v4_targets == v3_targets
            assert len(v4_texts) == len(v3_texts)

            # Step 0 reveals ONLY the first objective; later objectives are
            # absent from the step-0 text's plan prefix.
            assert v4_texts[0].startswith(
                f"Next objective: {objectives[0]}. "
            )
            step0_plan = v4_texts[0].split(" Current location: ")[0]
            for later in objectives[1:]:
                assert later not in step0_plan

            # Every non-final objective is revealed exactly once, on the
            # arrival step at its predecessor.
            for index in range(1, len(objectives)):
                reveal = (
                    f"Reached {objectives[index - 1]}. "
                    f"Next objective: {objectives[index]}. "
                )
                matching = [
                    text for text in v4_texts if text.startswith(reveal)
                ]
                assert len(matching) == 1
                assert (
                    f"Current location: {objectives[index - 1]}."
                    in matching[0]
                )

            # All other steps are strictly local (no plan text at all).
            reveal_free = [
                text
                for text in v4_texts[1:]
                if not text.startswith("Reached ")
            ]
            for text in reveal_free:
                assert text.startswith("Current location: ")
                assert "objective" not in text.lower()


def _stage2_v2_corpus():
    from volvence_zero.agent.eta_proof_benchmark import (
        generate_eta_proof_corpus,
    )

    return generate_eta_proof_corpus(
        seed=7,
        objective_count=4,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=6,
        heldout_route_count=3,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )


def test_stage2_v2_probe_label_is_a_function_of_the_visible_text() -> None:
    """Instrument-v2 core invariant: under the v4 staged-plan protocol the
    active subgoal is deterministically readable from the cumulative prefix,
    on heldout routes -- the property whose absence made the v1 instrument
    unpassable (heldout information ceiling ~0.18 vs the 0.25 gate bar)."""

    from volvence_zero.agent.eta_rate_distortion_evidence import (
        OBSERVATION_PROTOCOL_V4,
        eta_stage2_probe_rows,
    )

    corpus = _stage2_v2_corpus()
    rows, vocab = eta_stage2_probe_rows(
        corpus.heldout_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    assert rows

    labels_by_text: dict[str, set[int]] = {}
    for row in rows:
        labels_by_text.setdefault(row.observation_text, set()).add(
            row.subgoal_label
        )
    assert all(len(labels) == 1 for labels in labels_by_text.values())

    # The active subgoal was explicitly revealed somewhere in the prefix.
    for row in rows:
        revealed = f"Next objective: {vocab[row.subgoal_label]}."
        assert revealed in row.observation_text


def test_stage2_v2_probe_prefix_never_reveals_a_future_objective() -> None:
    """No leak: objectives after the active subgoal must not have been
    announced anywhere in the prefix."""

    from volvence_zero.agent.eta_rate_distortion_evidence import (
        OBSERVATION_PROTOCOL_V4,
        eta_stage2_probe_rows,
    )

    corpus = _stage2_v2_corpus()
    for case in corpus.train_cases + corpus.heldout_cases:
        rows, vocab = eta_stage2_probe_rows(
            (case,),
            environment=corpus.environment,
            protocol_version=OBSERVATION_PROTOCOL_V4,
        )
        objectives = tuple(
            waypoint
            for waypoint in case.route_signature
            if corpus.environment.location(waypoint).is_objective
        )
        for row in rows:
            active = vocab[row.subgoal_label]
            position = objectives.index(active)
            for future in objectives[position + 1 :]:
                assert (
                    f"Next objective: {future}." not in row.observation_text
                )


def test_stage2_v2_documents_drop_the_fingerprint_for_the_protocol_surface() -> None:
    """v2 documents must not carry the hash-fingerprint context line and must
    share the surface (header + step lines + actions) with the probe rows."""

    from volvence_zero.agent.eta_rate_distortion_evidence import (
        ETA_STAGE2_DOCUMENT_HEADER,
        OBSERVATION_PROTOCOL_V4,
        eta_stage2_documents,
        eta_stage2_probe_rows,
    )

    corpus = _stage2_v2_corpus()
    documents = eta_stage2_documents(
        corpus.train_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    assert len(documents) == len(corpus.train_cases)
    for case, document in zip(corpus.train_cases, documents, strict=True):
        assert case.source_text not in document
        assert document.startswith(ETA_STAGE2_DOCUMENT_HEADER)
        assert document.endswith("Episode complete.")
        assert " Action: go to " in document
        first_objective = next(
            waypoint
            for waypoint in case.route_signature
            if corpus.environment.location(waypoint).is_objective
        )
        assert f"Next objective: {first_objective}. " in document

    # Every probe-row prefix is a literal prefix of its route's document
    # (modulo the bare current-step line lacking the action suffix).
    rows, _ = eta_stage2_probe_rows(
        corpus.train_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    document_by_case = {
        case.case_id: document
        for case, document in zip(corpus.train_cases, documents, strict=True)
    }
    for row in rows:
        document = document_by_case[row.case_id]
        *past_lines, current_line = row.observation_text.split("\n")
        assert document.startswith("\n".join(past_lines))
        assert current_line in document


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


def test_rate_distortion_cli_marks_an_unpreregistered_run_as_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_eta_rate_distortion as cli

    output = tmp_path / "artifact"
    monkeypatch.setattr(
        cli, "run_eta_rate_distortion_evidence", lambda **_: _report_fixture()
    )
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
            str(tmp_path / "mps.lock"),
        ),
    )

    cli.main()

    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["preregistered"] is False
    assert manifest["verdict_authoritative"] is False
    assert manifest["claim_scope"] == "mechanism-only-smoke"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "not authoritative" in report


def test_rate_distortion_cli_binds_an_authoritative_run_to_its_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import preregister_eta_rate_distortion as prereg
    import run_eta_rate_distortion as cli

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
    preregistration = tmp_path / "prereg.json"
    preregistration.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "artifact"

    monkeypatch.setattr(
        cli, "run_eta_rate_distortion_evidence", lambda **_: _report_fixture()
    )
    monkeypatch.setattr(
        cli,
        "require_mps",
        lambda: common.MPSAvailability("fixture", True, True, True),
    )
    monkeypatch.setattr(cli, "_maybe_plot", lambda *_a, **_k: "")
    argv = (
        "run_eta_rate_distortion.py",
        "--output-dir",
        str(output),
        "--mps-lock",
        str(tmp_path / "mps.lock"),
        "--preregistration",
        str(preregistration),
        "--alphas",
        "0.01",
        "0.1",
        "1.0",
        "--seeds",
        "2",
        "--n-z",
        "4",
        "--updates",
        "1",
        "--model-id",
        "fixture-model",
    )
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["preregistered"] is True
    assert manifest["verdict_authoritative"] is True
    assert manifest["preregistration_sha256"] == hashlib.sha256(
        preregistration.read_bytes()
    ).hexdigest()

    # A run whose grid disagrees with the frozen protocol must not start.
    monkeypatch.setattr(
        sys,
        "argv",
        (
            *argv[:2],
            str(tmp_path / "artifact-2"),
            *argv[3:-1],
            "other-model",
        ),
    )
    with pytest.raises(cli.PreregistrationMismatch, match="model_id"):
        cli.main()


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
        "model_source": None,
        "device": "mps",
        "arms": ["frozen", "joint"],
        "corpus_seed": None,
        "objective_count": 8,
        "corridor_count": 2,
        "extra_edge_probability": 0.35,
        "train_routes": 200,
        "heldout_routes": 60,
        "train_lengths": [2, 3],
        "heldout_lengths": [3, 4],
        "observation_protocol": "partially-observable-no-remaining-route.v1",
        "posterior_parameterization": "legacy",
        "rate_gating": "per-step",
        "gate_mode": "continuous",
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
        (
            {"observation_protocol": "partially-observable-no-route-identity.v2"},
            "observation_protocol",
        ),
        (
            {"posterior_parameterization": "smooth"},
            "posterior_parameterization",
        ),
        (
            {"rate_gating": "switch-gated"},
            "rate_gating",
        ),
        (
            {"gate_mode": "hard-st"},
            "gate_mode",
        ),
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
    environment = SimpleNamespace(objective_locations=lambda: ("alpha",))
    monkeypatch.setattr(
        module, "build_default_eta_proof_environment", lambda: environment
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


def test_surrogate_seam_uses_injected_runtime_and_scorer_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A surrogate screen must never touch the real backend."""

    from types import SimpleNamespace

    from volvence_zero.agent import eta_rate_distortion_evidence as module
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        run_eta_rate_distortion_evidence,
    )

    executed: list[tuple[str, float, int]] = []
    _stub_sweep_dependencies(monkeypatch, executed=executed)

    # If the seam works, the real backend is never built or validated.
    def forbidden_build(_config: object) -> object:
        raise AssertionError("surrogate screen built the real backend")

    monkeypatch.setattr(
        module, "_build_eta_open_weight_runtime", forbidden_build
    )
    monkeypatch.setattr(
        module,
        "_validate_eta_open_weight_runtime",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("surrogate screen validated the real backend")
        ),
    )

    factory_calls: list[bool] = []
    surrogate_scorer = SimpleNamespace(
        injection_layer_index=3,
        control_norm_cap=1.0,
        probe_hidden_norm=2.0,
        reset_joint_parameters=lambda: None,
        restore_and_freeze=lambda: None,
    )

    def scorer_factory(*, action_options: object, joint_training: bool):
        del action_options
        factory_calls.append(joint_training)
        return surrogate_scorer

    injected_runtime = SimpleNamespace(
        model_id="surrogate-tiny",
        runtime_origin="surrogate",
        fallback_active=False,
    )

    report = run_eta_rate_distortion_evidence(
        alpha_grid=(0.01, 0.1, 1.0),
        seed_schedule=(0,),
        n_z=4,
        updates_per_run=1,
        arms=("frozen", "joint"),
        runtime=injected_runtime,
        scorer_factory=scorer_factory,
        posterior_parameterization="smooth",
    )

    assert factory_calls == [False, True]
    assert report.model_id == "surrogate-tiny"
    assert report.posterior_parameterization == "smooth"
    assert len(executed) == 6


def test_surrogate_runtime_without_factory_uses_its_own_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injecting only a tiny runtime uses that runtime's own scorer."""

    from types import SimpleNamespace

    from volvence_zero.agent import eta_rate_distortion_evidence as module
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        run_eta_rate_distortion_evidence,
    )

    executed: list[tuple[str, float, int]] = []
    _stub_sweep_dependencies(monkeypatch, executed=executed)
    monkeypatch.setattr(
        module,
        "_build_eta_open_weight_runtime",
        lambda _c: (_ for _ in ()).throw(
            AssertionError("real backend built for a surrogate run")
        ),
    )

    scorer = SimpleNamespace(
        injection_layer_index=3,
        control_norm_cap=1.0,
        probe_hidden_norm=2.0,
        reset_joint_parameters=lambda: None,
        restore_and_freeze=lambda: None,
    )
    built: list[bool] = []
    injected_runtime = SimpleNamespace(
        model_id="surrogate-tiny",
        runtime_origin="surrogate",
        fallback_active=False,
        build_steered_action_scorer=lambda **kw: (
            built.append(kw["joint_training"]) or scorer
        ),
    )

    report = run_eta_rate_distortion_evidence(
        alpha_grid=(0.01, 0.1, 1.0),
        seed_schedule=(0,),
        n_z=4,
        updates_per_run=1,
        runtime=injected_runtime,
    )

    assert built == [False, True]
    assert report.model_id == "surrogate-tiny"


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
