"""Read-only Gate 0 calibration for Relationship Lab v0.

The relationship vertical owns rendered observations, sealed dynamics,
reactive transitions, and decision sidecars.  This module only orchestrates
those frozen surfaces and publishes a verdict.  It never writes memory,
credit, PE, semantic owners, or steering state.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from lifeform_domain_emogpt.lab import (
    CandidateOutcomePrediction,
    OutcomeProbability,
    PreActionRelationshipDecision,
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    ReactiveRelationshipEnvironment,
    RelationshipAction,
    RelationshipDecisionTrace,
    RelationshipModelLineage,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
    sha256_json,
)


RELATIONSHIP_GATE0_REPORT_SCHEMA_VERSION = "relationship-lab-gate0-report.v1"
RELATIONSHIP_BASELINE_ATTESTATION_SCHEMA_VERSION = "relationship-lab-baseline-attestation.v1"


class GateCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


@dataclass(frozen=True)
class Gate0CalibrationConfig:
    samples_per_action: int = 256
    minimum_action_effect: float = 0.5
    maximum_baseline_accuracy: float = 0.85
    minimum_baseline_decisions: int = 24

    def __post_init__(self) -> None:
        if self.samples_per_action < 32:
            raise ValueError("samples_per_action must be >= 32")
        if not 0.0 < self.minimum_action_effect < 1.0:
            raise ValueError("minimum_action_effect must be in (0, 1)")
        if not 0.0 < self.maximum_baseline_accuracy < 1.0:
            raise ValueError("maximum_baseline_accuracy must be in (0, 1)")
        if self.minimum_baseline_decisions < 1:
            raise ValueError("minimum_baseline_decisions must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "samples_per_action": self.samples_per_action,
            "minimum_action_effect": self.minimum_action_effect,
            "maximum_baseline_accuracy": self.maximum_baseline_accuracy,
            "minimum_baseline_decisions": self.minimum_baseline_decisions,
        }


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def _require_iso_timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


@dataclass(frozen=True)
class FrozenBaselineAttestation:
    """Frozen stateless/raw calibration result used by the last Gate 0 tooth."""

    arm_id: str
    dataset_fingerprint: str
    model_id: str
    weights_sha256: str
    prompt_sha256: str
    generation_config_sha256: str
    seed_schedule_sha256: str
    decision_ledger_sha256: str
    evaluated_split: str
    valid_decisions: int
    correct_decisions: int
    evaluated_decisions: int
    context_tokens_total: int
    hidden_test_opened: bool
    frozen_at_iso: str
    schema_version: str = RELATIONSHIP_BASELINE_ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_BASELINE_ATTESTATION_SCHEMA_VERSION:
            raise ValueError("baseline attestation schema_version mismatch")
        if self.arm_id not in {"stateless", "raw"}:
            raise ValueError("Gate 0 baseline attestation must be stateless or raw")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("baseline model_id must be non-empty")
        for field_name, value in (
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("weights_sha256", self.weights_sha256),
            ("prompt_sha256", self.prompt_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("seed_schedule_sha256", self.seed_schedule_sha256),
            ("decision_ledger_sha256", self.decision_ledger_sha256),
        ):
            _require_sha256(value, field_name)
        if self.evaluated_split not in {"train", "validation", "calibration"}:
            raise ValueError("Gate 0 baseline must use train/validation/calibration, never heldout")
        integer_fields = (
            ("valid_decisions", self.valid_decisions),
            ("correct_decisions", self.correct_decisions),
            ("evaluated_decisions", self.evaluated_decisions),
            ("context_tokens_total", self.context_tokens_total),
        )
        for field_name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if not isinstance(self.hidden_test_opened, bool):
            raise ValueError("hidden_test_opened must be a boolean")
        if self.evaluated_decisions < 1:
            raise ValueError("evaluated_decisions must be positive")
        if not 0 <= self.valid_decisions <= self.evaluated_decisions:
            raise ValueError("valid_decisions must be within evaluated_decisions")
        if not 0 <= self.correct_decisions <= self.valid_decisions:
            raise ValueError("correct_decisions must be within valid_decisions")
        if self.context_tokens_total < 0:
            raise ValueError("context_tokens_total must be non-negative")
        if self.hidden_test_opened:
            raise ValueError("Gate 0 baseline cannot be frozen after hidden test opening")
        _require_iso_timestamp(self.frozen_at_iso, "frozen_at_iso")

    @property
    def accuracy(self) -> float:
        return self.correct_decisions / self.evaluated_decisions

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "prompt_sha256": self.prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "evaluated_split": self.evaluated_split,
            "valid_decisions": self.valid_decisions,
            "correct_decisions": self.correct_decisions,
            "evaluated_decisions": self.evaluated_decisions,
            "context_tokens_total": self.context_tokens_total,
            "hidden_test_opened": self.hidden_test_opened,
            "frozen_at_iso": self.frozen_at_iso,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "FrozenBaselineAttestation":
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise ValueError("baseline attestation must be a JSON object")
        expected = {
            "artifact_id",
            "schema_version",
            "arm_id",
            "dataset_fingerprint",
            "model_id",
            "weights_sha256",
            "prompt_sha256",
            "generation_config_sha256",
            "seed_schedule_sha256",
            "decision_ledger_sha256",
            "evaluated_split",
            "valid_decisions",
            "correct_decisions",
            "evaluated_decisions",
            "context_tokens_total",
            "hidden_test_opened",
            "frozen_at_iso",
        }
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        if missing or extra:
            raise ValueError(f"baseline attestation fields do not match schema; missing={missing}, extra={extra}")
        string_fields = (
            "artifact_id",
            "schema_version",
            "arm_id",
            "dataset_fingerprint",
            "model_id",
            "weights_sha256",
            "prompt_sha256",
            "generation_config_sha256",
            "seed_schedule_sha256",
            "decision_ledger_sha256",
            "evaluated_split",
            "frozen_at_iso",
        )
        for field_name in string_fields:
            if not isinstance(raw[field_name], str):
                raise ValueError(f"{field_name} must be a string")
        integer_fields = (
            "valid_decisions",
            "correct_decisions",
            "evaluated_decisions",
            "context_tokens_total",
        )
        for field_name in integer_fields:
            if isinstance(raw[field_name], bool) or not isinstance(raw[field_name], int):
                raise ValueError(f"{field_name} must be an integer")
        if not isinstance(raw["hidden_test_opened"], bool):
            raise ValueError("hidden_test_opened must be a boolean")
        artifact_id = raw.pop("artifact_id")
        attestation = cls(
            schema_version=raw["schema_version"],
            arm_id=raw["arm_id"],
            dataset_fingerprint=raw["dataset_fingerprint"],
            model_id=raw["model_id"],
            weights_sha256=raw["weights_sha256"],
            prompt_sha256=raw["prompt_sha256"],
            generation_config_sha256=raw["generation_config_sha256"],
            seed_schedule_sha256=raw["seed_schedule_sha256"],
            decision_ledger_sha256=raw["decision_ledger_sha256"],
            evaluated_split=raw["evaluated_split"],
            valid_decisions=raw["valid_decisions"],
            correct_decisions=raw["correct_decisions"],
            evaluated_decisions=raw["evaluated_decisions"],
            context_tokens_total=raw["context_tokens_total"],
            hidden_test_opened=raw["hidden_test_opened"],
            frozen_at_iso=raw["frozen_at_iso"],
        )
        _require_sha256(artifact_id, "artifact_id")
        if artifact_id != attestation.artifact_id:
            raise ValueError("baseline artifact_id does not match canonical payload")
        return attestation


@dataclass(frozen=True)
class Gate0Check:
    check_id: str
    status: GateCheckStatus
    summary: str
    metrics: tuple[tuple[str, bool | float | int | str], ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "summary": self.summary,
            "metrics": {key: value for key, value in self.metrics},
        }


@dataclass(frozen=True)
class RelationshipGate0Report:
    created_at_iso: str
    dataset_fingerprint: str
    package_file_hashes: tuple[tuple[str, str], ...]
    config: Gate0CalibrationConfig
    checks: tuple[Gate0Check, ...]
    machinery_ready: bool
    gate0_passed: bool
    baseline_attestation_id: str | None
    schema_version: str = RELATIONSHIP_GATE0_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_GATE0_REPORT_SCHEMA_VERSION:
            raise ValueError("Gate 0 report schema_version mismatch")
        _require_iso_timestamp(self.created_at_iso, "created_at_iso")
        _require_sha256(self.dataset_fingerprint, "dataset_fingerprint")
        for name, digest in self.package_file_hashes:
            if not name:
                raise ValueError("package file name must be non-empty")
            _require_sha256(digest, f"package_file_hashes[{name}]")
        check_ids = tuple(check.check_id for check in self.checks)
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("Gate 0 check ids must be unique")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "dataset_fingerprint": self.dataset_fingerprint,
            "package_file_hashes": {name: digest for name, digest in self.package_file_hashes},
            "config": self.config.to_payload(),
            "checks": [check.to_payload() for check in self.checks],
            "verdicts": {
                "machinery_ready": self.machinery_ready,
                "gate0_passed": self.gate0_passed,
            },
            "baseline_attestation_id": self.baseline_attestation_id,
            "claim_boundary": (
                "P0 proves instrument and evidence-contract readiness only. "
                "Gate 0 requires a frozen real-substrate stateless/raw baseline "
                "at or below the configured non-saturation ceiling, with "
                "every structured decision valid."
            ),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _package_file_hashes(root: pathlib.Path) -> tuple[tuple[str, str], ...]:
    required = (
        "manifest.yaml",
        "ssot_fragment.json",
        "scenes.yaml",
        "test_suite.yaml",
        "rendered_observations.json",
        "generator_truth.json",
        "prereg_template.json",
    )
    hashes: list[tuple[str, str]] = []
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Relationship Lab package is missing {path}")
        hashes.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(hashes)


def _positive_mass(
    prediction: CandidateOutcomePrediction,
    dataset: RelationshipTransferDataset,
) -> float:
    return math.fsum(prediction.probability_of(kind) for kind in dataset.positive_outcomes)


def _opposite_action(action: RelationshipAction) -> RelationshipAction:
    if action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE:
        return RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
    if action is RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION:
        return RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    raise ValueError("neutral_noop has no mirrored opposite")


def _mirrored_check(dataset: RelationshipTransferDataset) -> Gate0Check:
    pairs = dataset.mirrored_pairs()
    byte_identical = all(len({item[0].current_input.encode("utf-8") for item in members}) == 1 for _, members in pairs)
    opposite = all(
        {item[1].preferred_action for item in members}
        == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
        for _, members in pairs
    )
    families = {item.probe_surface_family for item in dataset.observations}
    passed = len(pairs) >= 6 and len(families) >= 4 and byte_identical and opposite
    return Gate0Check(
        check_id="mirrored_counterfactual",
        status=GateCheckStatus.PASS if passed else GateCheckStatus.FAIL,
        summary=(
            "Every mirrored pair has byte-identical current input and opposite non-noop optimal actions."
            if passed
            else "Mirrored-pair action conflict or coverage is invalid."
        ),
        metrics=(
            ("mirrored_pairs", len(pairs)),
            ("surface_families", len(families)),
            ("byte_identical", byte_identical),
            ("opposite_actions", opposite),
        ),
    )


def _reactive_effect_check(
    dataset: RelationshipTransferDataset,
    environment: ReactiveRelationshipEnvironment,
    config: Gate0CalibrationConfig,
) -> Gate0Check:
    analytic_effects: list[float] = []
    empirical_effects: list[float] = []
    action_transmission = True
    evidence_refs: set[str] = set()
    for observation in dataset.observations:
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        preferred = dynamic.preferred_action
        opposite = _opposite_action(preferred)
        preferred_distribution = environment.distribution_for(
            scene_id=observation.scene_id,
            action=preferred,
        )
        opposite_distribution = environment.distribution_for(
            scene_id=observation.scene_id,
            action=opposite,
        )
        analytic_effects.append(
            _positive_mass(preferred_distribution, dataset) - _positive_mass(opposite_distribution, dataset)
        )
        preferred_positive = 0
        opposite_positive = 0
        for seed in range(config.samples_per_action):
            preferred_outcome = environment.settle(
                scene_id=observation.scene_id,
                decision_id=f"gate0:{observation.scene_id}:{seed}",
                action=preferred,
                seed=seed,
            )
            opposite_outcome = environment.settle(
                scene_id=observation.scene_id,
                decision_id=f"gate0:{observation.scene_id}:{seed}",
                action=opposite,
                seed=seed,
            )
            action_transmission = action_transmission and (
                preferred_outcome.selected_action is preferred and opposite_outcome.selected_action is opposite
            )
            evidence_refs.add(preferred_outcome.environment_evidence_ref)
            evidence_refs.add(opposite_outcome.environment_evidence_ref)
            preferred_positive += int(preferred_outcome.typed_outcome in dataset.positive_outcomes)
            opposite_positive += int(opposite_outcome.typed_outcome in dataset.positive_outcomes)
        empirical_effects.append((preferred_positive - opposite_positive) / config.samples_per_action)
    minimum_analytic = min(analytic_effects)
    minimum_empirical = min(empirical_effects)
    passed = (
        action_transmission
        and minimum_analytic >= config.minimum_action_effect
        and minimum_empirical >= config.minimum_action_effect
    )
    return Gate0Check(
        check_id="reactive_action_effect",
        status=GateCheckStatus.PASS if passed else GateCheckStatus.FAIL,
        summary=(
            "Selected actions physically reach the environment and cause a configured minimum typed-outcome effect."
            if passed
            else "Action transmission or the reactive outcome effect is too weak."
        ),
        metrics=(
            ("action_transmission", action_transmission),
            ("minimum_analytic_effect", round(minimum_analytic, 6)),
            ("minimum_empirical_effect", round(minimum_empirical, 6)),
            ("unique_environment_evidence_refs", len(evidence_refs)),
        ),
    )


def _determinism_check(
    dataset: RelationshipTransferDataset,
    environment: ReactiveRelationshipEnvironment,
) -> Gate0Check:
    scene = dataset.observations[0]
    first = environment.settle(
        scene_id=scene.scene_id,
        decision_id="gate0-determinism",
        action=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        seed=17,
    )
    second = environment.settle(
        scene_id=scene.scene_id,
        decision_id="gate0-determinism",
        action=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        seed=17,
    )
    counterfactual = environment.settle(
        scene_id=scene.scene_id,
        decision_id="gate0-determinism",
        action=RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        seed=17,
    )
    repeat_equal = first == second
    action_changes_evidence = first.environment_evidence_ref != counterfactual.environment_evidence_ref
    passed = repeat_equal and action_changes_evidence
    return Gate0Check(
        check_id="environment_determinism",
        status=GateCheckStatus.PASS if passed else GateCheckStatus.FAIL,
        summary=(
            "Same inputs settle identically; changing the action changes the content-addressed environment evidence."
            if passed
            else "Reactive environment determinism/content addressing failed."
        ),
        metrics=(
            ("repeat_equal", repeat_equal),
            ("action_changes_evidence", action_changes_evidence),
        ),
    )


def _leakage_check(dataset: RelationshipTransferDataset) -> Gate0Check:
    try:
        dataset.assert_no_sut_truth_leakage()
    except ValueError as exc:
        return Gate0Check(
            check_id="sut_truth_leakage",
            status=GateCheckStatus.FAIL,
            summary=str(exc),
            metrics=(("leakage_count", 1),),
        )
    return Gate0Check(
        check_id="sut_truth_leakage",
        status=GateCheckStatus.PASS,
        summary=(
            "SUT payloads contain public histories only; latent ids, preferred "
            "actions, profiles, pair ids, and future outcomes are absent."
        ),
        metrics=(("leakage_count", 0),),
    )


def _uniform_predictions() -> tuple[CandidateOutcomePrediction, ...]:
    return tuple(
        CandidateOutcomePrediction(
            action_id=action,
            outcomes=tuple(
                OutcomeProbability(outcome, 1.0 / len(RELATIONSHIP_OUTCOMES)) for outcome in RELATIONSHIP_OUTCOMES
            ),
        )
        for action in RELATIONSHIP_ACTIONS
    )


def _decision_trace_check(
    dataset: RelationshipTransferDataset,
    environment: ReactiveRelationshipEnvironment,
) -> Gate0Check:
    observation = dataset.observations[0]
    dynamic = dataset.dynamic_for_scene(observation.scene_id)
    before = "2026-08-19T00:00:00+00:00"
    after = "2026-08-19T00:00:01+00:00"
    decision_id = "gate0-sidecar-smoke"
    pre_action = PreActionRelationshipDecision(
        decision_id=decision_id,
        pre_action_timestamp=before,
        candidate_predictions=_uniform_predictions(),
        chosen_action_id=RelationshipAction.NEUTRAL_NOOP,
        source_snapshot_hashes=(dataset.dataset_fingerprint,),
        lineage=RelationshipModelLineage(
            model_id="gate0-contract-fixture-not-a-baseline",
            weights_sha256=sha256_json("fixture-weights"),
            prompt_sha256=sha256_json("fixture-prompt"),
            generation_config_sha256=sha256_json("fixture-generation"),
            seed=0,
        ),
    )
    outcome = environment.settle(
        scene_id=observation.scene_id,
        decision_id=decision_id,
        action=pre_action.chosen_action_id,
        seed=0,
    )
    trace = RelationshipDecisionTrace(
        trajectory_sha256=observation.trajectory_sha256,
        user_scope_hash=observation.user_scope_hash,
        scenario_family=observation.probe_surface_family,
        surface_scene_id=observation.scene_id,
        split=dynamic.split,
        sealed_latent_dynamic_id=dynamic.dynamic_id,
        pre_action=pre_action,
        observed_typed_outcome=outcome.typed_outcome,
        outcome_observed_at=after,
        environment_evidence_ref=outcome.environment_evidence_ref,
    )
    round_trip = RelationshipDecisionTrace.from_json(trace.to_json())
    passed = round_trip == trace and round_trip.artifact_id == trace.artifact_id
    return Gate0Check(
        check_id="decision_trace_contract",
        status=GateCheckStatus.PASS if passed else GateCheckStatus.FAIL,
        summary=(
            "Pre-action bet and post-action settlement round-trip through one content-addressed frozen sidecar."
            if passed
            else "Decision sidecar round-trip/content id failed."
        ),
        metrics=(
            ("round_trip_equal", passed),
            ("trace_artifact_id", trace.artifact_id),
        ),
    )


def _baseline_check(
    *,
    dataset: RelationshipTransferDataset,
    config: Gate0CalibrationConfig,
    baseline: FrozenBaselineAttestation | None,
) -> Gate0Check:
    if baseline is None:
        return Gate0Check(
            check_id="frozen_baseline_non_saturation",
            status=GateCheckStatus.PENDING,
            summary=(
                "No frozen real-substrate stateless/raw calibration attestation "
                "was supplied; machinery may be ready, but Gate 0 cannot pass."
            ),
            metrics=(("baseline_supplied", False),),
        )
    dataset_match = baseline.dataset_fingerprint == dataset.dataset_fingerprint
    enough_decisions = baseline.evaluated_decisions >= config.minimum_baseline_decisions
    all_outputs_valid = baseline.valid_decisions == baseline.evaluated_decisions
    non_saturated = baseline.accuracy <= config.maximum_baseline_accuracy
    passed = dataset_match and enough_decisions and all_outputs_valid and non_saturated
    return Gate0Check(
        check_id="frozen_baseline_non_saturation",
        status=GateCheckStatus.PASS if passed else GateCheckStatus.FAIL,
        summary=(
            "Frozen real-substrate stateless/raw baseline is inside the configured non-saturation ceiling."
            if passed
            else "Frozen baseline is mismatched, too small, malformed, or saturated."
        ),
        metrics=(
            ("baseline_supplied", True),
            ("dataset_match", dataset_match),
            ("evaluated_decisions", baseline.evaluated_decisions),
            ("valid_decisions", baseline.valid_decisions),
            ("all_outputs_valid", all_outputs_valid),
            ("accuracy", round(baseline.accuracy, 6)),
            ("non_saturated", non_saturated),
        ),
    )


def run_relationship_gate0_calibration(
    *,
    config: Gate0CalibrationConfig | None = None,
    baseline: FrozenBaselineAttestation | None = None,
    package_root: pathlib.Path | None = None,
    created_at_iso: str | None = None,
) -> RelationshipGate0Report:
    """Run P0 machinery checks and the optional frozen-baseline tooth."""

    effective_config = config or Gate0CalibrationConfig()
    root = pathlib.Path(package_root or relationship_transfer_package_dir())
    dataset = load_relationship_transfer_dataset(root)
    environment = ReactiveRelationshipEnvironment(dataset)
    machinery_checks = (
        _mirrored_check(dataset),
        _reactive_effect_check(dataset, environment, effective_config),
        _determinism_check(dataset, environment),
        _leakage_check(dataset),
        _decision_trace_check(dataset, environment),
    )
    baseline_check = _baseline_check(
        dataset=dataset,
        config=effective_config,
        baseline=baseline,
    )
    checks = (*machinery_checks, baseline_check)
    machinery_ready = all(check.status is GateCheckStatus.PASS for check in machinery_checks)
    gate0_passed = machinery_ready and baseline_check.status is GateCheckStatus.PASS
    timestamp = created_at_iso or datetime.now(timezone.utc).isoformat()
    return RelationshipGate0Report(
        created_at_iso=timestamp,
        dataset_fingerprint=dataset.dataset_fingerprint,
        package_file_hashes=_package_file_hashes(root),
        config=effective_config,
        checks=checks,
        machinery_ready=machinery_ready,
        gate0_passed=gate0_passed,
        baseline_attestation_id=(baseline.artifact_id if baseline is not None else None),
    )


def format_relationship_gate0_report(report: RelationshipGate0Report) -> str:
    lines = [
        "# Relationship Lab Gate 0 calibration",
        "",
        f"- artifact_id: `{report.artifact_id}`",
        f"- dataset_fingerprint: `{report.dataset_fingerprint}`",
        f"- machinery_ready: **{str(report.machinery_ready).lower()}**",
        f"- gate0_passed: **{str(report.gate0_passed).lower()}**",
        "",
        "## Checks",
        "",
    ]
    for check in report.checks:
        lines.append(f"- `{check.check_id}` — **{check.status.value}**: {check.summary}")
    claim_boundary = (
        "P0 only establishes instrument and evidence-contract readiness. This "
        "report includes a frozen real-substrate baseline, but it is not formal "
        "hidden-test or four-capability evidence unless the preregistration and "
        "secret heldout are separately frozen."
        if report.baseline_attestation_id is not None
        else "P0 only establishes instrument and evidence-contract readiness. "
        "Without a frozen real-substrate stateless/raw baseline attestation, "
        "Gate 0 remains pending and no model-capability claim is allowed."
    )
    lines.extend(["", "## Claim boundary", "", claim_boundary, ""])
    return "\n".join(lines)


def write_relationship_gate0_report(
    report: RelationshipGate0Report,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "report.json"
    markdown_path = target / "report.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(
        format_relationship_gate0_report(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def load_frozen_baseline_attestation(
    path: pathlib.Path,
) -> FrozenBaselineAttestation:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return FrozenBaselineAttestation.from_json(file_path.read_text(encoding="utf-8"))


__all__ = [
    "FrozenBaselineAttestation",
    "Gate0CalibrationConfig",
    "Gate0Check",
    "GateCheckStatus",
    "RELATIONSHIP_BASELINE_ATTESTATION_SCHEMA_VERSION",
    "RELATIONSHIP_GATE0_REPORT_SCHEMA_VERSION",
    "RelationshipGate0Report",
    "format_relationship_gate0_report",
    "load_frozen_baseline_attestation",
    "run_relationship_gate0_calibration",
    "write_relationship_gate0_report",
]
