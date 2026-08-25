"""S1 admission evidence for a frozen full-width residual readout.

S1 follows the sealed ETA Stage-3 negative result and its P1 attribution.  It
does not retry ETA: it asks whether one fixed, full-width hidden layer can be
turned into an immutable active-subgoal readout with explicit lineage.  The
result is an admission gate for a later no-bias causal-steering experiment,
not a causal claim and not production wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import statistics

from volvence_zero.agent.eta_proof_benchmark import (
    ETAProbeRow,
    ETAProofCorpus,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V4,
    eta_stage2_probe_rows,
)
from volvence_zero.substrate import (
    FrozenResidualReadoutArtifact,
    OpenWeightResidualRuntime,
    SubstrateFingerprint,
    SubstrateForwardRepresentationPublisher,
    SubstrateResidualReadoutPublisher,
    SubstrateResidualReadoutSnapshot,
    fit_frozen_residual_readout,
)


ETA_S1_RESIDUAL_READOUT_SCHEMA_VERSION = "eta-s1-residual-readout-evidence.v1"


@dataclass(frozen=True)
class S1ResidualReadoutThresholds:
    min_heldout_accuracy: float = 0.80
    chance_multiple: float = 2.0
    min_late_accuracy: float = 0.50
    max_train_heldout_gap: float = 0.20


@dataclass(frozen=True)
class S1ResidualReadoutMetrics:
    accuracy: float
    chance_accuracy: float
    majority_accuracy: float
    early_accuracy: float
    late_accuracy: float
    mean_score_margin: float
    min_score_margin: float
    support: int
    early_support: int
    late_support: int


@dataclass(frozen=True)
class S1ResidualReadoutAdmission:
    admitted: bool
    condition_accuracy: bool
    condition_chance_multiple: bool
    condition_late_retention: bool
    condition_generalization_gap: bool
    failed_conditions: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class S1ResidualReadoutEvidenceReport:
    schema_version: str
    claim_scope: str
    model_id: str
    model_source: str
    device: str
    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    observation_protocol: str
    corpus_seed: int
    objective_count: int
    train_route_count: int
    heldout_route_count: int
    train_row_count: int
    heldout_row_count: int
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int
    ridge_alpha: float
    class_ids: tuple[str, ...]
    artifact_id: str
    training_snapshot_fingerprint: str
    heldout_snapshot_fingerprint: str
    thresholds: S1ResidualReadoutThresholds
    train_metrics: S1ResidualReadoutMetrics
    heldout_metrics: S1ResidualReadoutMetrics
    admission: S1ResidualReadoutAdmission
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str = ""


def _sample_id(row: ETAProbeRow) -> str:
    return f"{row.split}:{row.case_id}:step-{row.step_index}"


def _validate_rows(rows: tuple[ETAProbeRow, ...], *, split: str) -> None:
    if not rows:
        raise ValueError(f"S1 residual readout {split} rows must be non-empty")
    sample_ids = tuple(_sample_id(row) for row in rows)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"S1 residual readout {split} sample ids are not unique")
    if any(row.split != split for row in rows):
        raise ValueError(f"S1 residual readout {split} rows contain another split")


def _metrics(
    *,
    rows: tuple[ETAProbeRow, ...],
    readout: SubstrateResidualReadoutSnapshot,
    class_ids: tuple[str, ...],
) -> S1ResidualReadoutMetrics:
    if tuple(row.sample_id for row in readout.readouts) != tuple(
        _sample_id(row) for row in rows
    ):
        raise ValueError("S1 residual readout rows and published scores misalign")
    expected = tuple(class_ids[row.subgoal_label] for row in rows)
    predicted = tuple(row.predicted_class_id for row in readout.readouts)
    correct = tuple(
        prediction == target
        for prediction, target in zip(predicted, expected, strict=True)
    )
    step_values = sorted({row.step_index for row in rows})
    median_step = step_values[len(step_values) // 2]
    early_indices = tuple(
        index for index, row in enumerate(rows) if row.step_index < median_step
    )
    late_indices = tuple(
        index for index, row in enumerate(rows) if row.step_index >= median_step
    )

    def bucket_accuracy(indices: tuple[int, ...]) -> float:
        return (
            sum(int(correct[index]) for index in indices) / len(indices)
            if indices
            else 0.0
        )

    counts = tuple(expected.count(class_id) for class_id in class_ids)
    margins = tuple(row.score_margin for row in readout.readouts)
    support = len(rows)
    return S1ResidualReadoutMetrics(
        accuracy=sum(correct) / support,
        chance_accuracy=1.0 / len(class_ids),
        majority_accuracy=max(counts) / support,
        early_accuracy=bucket_accuracy(early_indices),
        late_accuracy=bucket_accuracy(late_indices),
        mean_score_margin=statistics.fmean(margins),
        min_score_margin=min(margins),
        support=support,
        early_support=len(early_indices),
        late_support=len(late_indices),
    )


def assess_s1_residual_readout(
    *,
    train_metrics: S1ResidualReadoutMetrics,
    heldout_metrics: S1ResidualReadoutMetrics,
    thresholds: S1ResidualReadoutThresholds,
) -> S1ResidualReadoutAdmission:
    if thresholds.min_heldout_accuracy <= 0.0 or thresholds.min_heldout_accuracy > 1.0:
        raise ValueError("S1 min_heldout_accuracy must be in (0, 1]")
    if thresholds.chance_multiple <= 0.0:
        raise ValueError("S1 chance_multiple must be positive")
    if thresholds.min_late_accuracy <= 0.0 or thresholds.min_late_accuracy > 1.0:
        raise ValueError("S1 min_late_accuracy must be in (0, 1]")
    if thresholds.max_train_heldout_gap < 0.0:
        raise ValueError("S1 max_train_heldout_gap must be non-negative")

    conditions = {
        "heldout-accuracy": (
            heldout_metrics.accuracy >= thresholds.min_heldout_accuracy
        ),
        "chance-multiple": (
            heldout_metrics.accuracy
            >= thresholds.chance_multiple * heldout_metrics.chance_accuracy
        ),
        "late-retention": (
            heldout_metrics.late_accuracy >= thresholds.min_late_accuracy
        ),
        "generalization-gap": (
            train_metrics.accuracy - heldout_metrics.accuracy
            <= thresholds.max_train_heldout_gap
        ),
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return S1ResidualReadoutAdmission(
        admitted=not failed,
        condition_accuracy=conditions["heldout-accuracy"],
        condition_chance_multiple=conditions["chance-multiple"],
        condition_late_retention=conditions["late-retention"],
        condition_generalization_gap=conditions["generalization-gap"],
        failed_conditions=failed,
        description=(
            "S1 frozen readout admitted for S2 axis testing."
            if not failed
            else "S1 frozen readout blocked: " + ", ".join(failed)
        ),
    )


def run_eta_s1_residual_readout(
    *,
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    model_fingerprint: SubstrateFingerprint,
    model_source: str,
    device: str,
    expected_layer_indices: tuple[int, ...] = (20,),
    expected_activation_widths: tuple[int, ...] = (896,),
    ridge_alpha: float = 1.0,
    thresholds: S1ResidualReadoutThresholds | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[S1ResidualReadoutEvidenceReport, FrozenResidualReadoutArtifact]:
    train_rows, class_ids = eta_stage2_probe_rows(
        corpus.train_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    heldout_rows, heldout_class_ids = eta_stage2_probe_rows(
        corpus.heldout_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    if class_ids != heldout_class_ids:
        raise RuntimeError("S1 class vocabulary differs between corpus splits")
    _validate_rows(train_rows, split="train")
    _validate_rows(heldout_rows, split="heldout")

    representation_publisher = SubstrateForwardRepresentationPublisher(
        runtime,
        model_fingerprint=model_fingerprint,
    )

    def capture_progress(split: str) -> Callable[[str, int, int], None]:
        def publish(sample_id: str, completed: int, total: int) -> None:
            if progress is not None and (completed == total or completed % 25 == 0):
                progress(
                    f"capture split={split} completed={completed}/{total} "
                    f"sample={sample_id}"
                )

        return publish

    train_representations = representation_publisher.publish(
        tuple((_sample_id(row), row.observation_text) for row in train_rows),
        progress=capture_progress("train"),
    )
    heldout_representations = representation_publisher.publish(
        tuple((_sample_id(row), row.observation_text) for row in heldout_rows),
        progress=capture_progress("heldout"),
    )
    for split, snapshot in (
        ("train", train_representations),
        ("heldout", heldout_representations),
    ):
        if snapshot.lineage.layer_indices != expected_layer_indices or (
            snapshot.lineage.activation_widths != expected_activation_widths
        ):
            raise RuntimeError(
                f"S1 {split} capture geometry mismatch: expected "
                f"{(expected_layer_indices, expected_activation_widths)!r}, got "
                f"{(snapshot.lineage.layer_indices, snapshot.lineage.activation_widths)!r}"
            )

    import torch

    artifact = fit_frozen_residual_readout(
        torch_module=torch,
        snapshot=train_representations,
        labels=tuple(
            (_sample_id(row), class_ids[row.subgoal_label]) for row in train_rows
        ),
        class_ids=class_ids,
        ridge_alpha=ridge_alpha,
    )
    readout_publisher = SubstrateResidualReadoutPublisher(artifact)
    train_readout = readout_publisher.publish(train_representations)
    heldout_readout = readout_publisher.publish(heldout_representations)
    train_metrics = _metrics(
        rows=train_rows,
        readout=train_readout,
        class_ids=class_ids,
    )
    heldout_metrics = _metrics(
        rows=heldout_rows,
        readout=heldout_readout,
        class_ids=class_ids,
    )
    active_thresholds = thresholds or S1ResidualReadoutThresholds()
    admission = assess_s1_residual_readout(
        train_metrics=train_metrics,
        heldout_metrics=heldout_metrics,
        thresholds=active_thresholds,
    )
    report = S1ResidualReadoutEvidenceReport(
        schema_version=ETA_S1_RESIDUAL_READOUT_SCHEMA_VERSION,
        claim_scope="s1-readout-admission-no-causal-claim",
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        model_fingerprint=model_fingerprint,
        runtime_origin=artifact.runtime_origin,
        observation_protocol=OBSERVATION_PROTOCOL_V4,
        corpus_seed=corpus.seed,
        objective_count=corpus.objective_count,
        train_route_count=corpus.train_route_count,
        heldout_route_count=corpus.heldout_route_count,
        train_row_count=len(train_rows),
        heldout_row_count=len(heldout_rows),
        layer_indices=artifact.layer_indices,
        activation_widths=artifact.activation_widths,
        representation_dim=artifact.representation_dim,
        ridge_alpha=ridge_alpha,
        class_ids=class_ids,
        artifact_id=artifact.artifact_id,
        training_snapshot_fingerprint=artifact.training_snapshot_fingerprint,
        heldout_snapshot_fingerprint=(
            heldout_representations.lineage.snapshot_fingerprint
        ),
        thresholds=active_thresholds,
        train_metrics=train_metrics,
        heldout_metrics=heldout_metrics,
        admission=admission,
        production_wiring_changed=False,
        feedback_to_learning=False,
        description=(
            "S1 full-width layer-aligned frozen residual readout; admission "
            f"for S2={admission.admitted}. No causal or production claim."
        ),
    )
    return report, artifact


__all__ = [
    "ETA_S1_RESIDUAL_READOUT_SCHEMA_VERSION",
    "S1ResidualReadoutAdmission",
    "S1ResidualReadoutEvidenceReport",
    "S1ResidualReadoutMetrics",
    "S1ResidualReadoutThresholds",
    "assess_s1_residual_readout",
    "run_eta_s1_residual_readout",
]
