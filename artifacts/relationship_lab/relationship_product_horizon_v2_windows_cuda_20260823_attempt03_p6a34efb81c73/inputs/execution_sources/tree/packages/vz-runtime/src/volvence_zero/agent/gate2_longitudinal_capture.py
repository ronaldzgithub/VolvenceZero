"""Full-width companion capture for Gate 2 v35 longitudinal readout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    _build_eta_open_weight_runtime,
    _install_learned_control_basis,
    _validate_eta_open_weight_runtime,
    eta_gate2_selector_fresh_cases,
)
from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    build_gate11_longitudinal_source_plans,
    load_gate11_longitudinal_source_records,
    validate_gate11_longitudinal_source_prefix,
)
from volvence_zero.agent.gate2_longitudinal_readout import (
    GATE2_LONGITUDINAL_INPUT_FILENAME,
    GATE2_LONGITUDINAL_INPUT_SCHEMA_VERSION,
    GATE2_LONGITUDINAL_OUTCOME_FILENAME,
    GATE2_LONGITUDINAL_OUTCOME_SCHEMA_VERSION,
    GATE2_V35_CONTROL_BASIS_FINGERPRINT,
    GATE2_V35_SELECTOR_FINGERPRINT,
    _canonical_bytes,
    _load_companion_records,
    _sha256_file,
    load_gate2_v35_selector_bundle,
)
from volvence_zero.internal_rl import (
    KernelResidualActionSelectorArtifact,
    residual_action_state_vector,
    selector_artifact_from_payload,
)
from volvence_zero.substrate import (
    OpenWeightResidualRuntime,
    SubstrateSnapshot,
    SurfaceKind,
)


GATE2_LONGITUDINAL_CAPTURE_SCHEMA_VERSION = (
    "gate2-longitudinal-v35-companion-capture.v1"
)
GATE2_LONGITUDINAL_CAPTURE_MIN_EFFECT = 0.02
GATE2_LONGITUDINAL_CAPTURE_MIN_SESSION_POSITIVE_RATE = 0.60
GATE2_LONGITUDINAL_CAPTURE_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
_T_CRITICAL_95_DF2 = 4.302652729749
_TRACK_SCALE = (0.7, 0.7, 0.7)


@dataclass(frozen=True)
class Gate2CandidateControlContract:
    controls: tuple[tuple[float, float, float], ...]
    source_sha256: str
    mapping_fingerprint: str


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


def _append_companion_row(path: Path, payload: Mapping[str, Any]) -> None:
    row = dict(payload)
    row["record_sha256"] = hashlib.sha256(
        _canonical_bytes(row)
    ).hexdigest()
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()


def load_gate2_candidate_control_contract(
    path: str | Path,
) -> Gate2CandidateControlContract:
    """Freeze the v35 candidate-index to applied-control mapping."""

    source = Path(path)
    controls_by_index: dict[int, set[tuple[float, float, float]]] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(
                f"blank v35 counterfactual row at {source}:{line_number}"
            )
        row = json.loads(line)
        candidate_index = row.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < 22
        ):
            raise ValueError(
                f"invalid v35 candidate index at {source}:{line_number}"
            )
        raw_control = row.get("applied_control")
        if not isinstance(raw_control, list) or len(raw_control) != 3:
            raise ValueError(
                f"invalid v35 applied control at {source}:{line_number}"
            )
        control = tuple(float(value) for value in raw_control)
        if any(not math.isfinite(value) for value in control):
            raise ValueError(
                f"non-finite v35 applied control at {source}:{line_number}"
            )
        controls_by_index.setdefault(candidate_index, set()).add(control)
    if set(controls_by_index) != set(range(22)):
        raise ValueError("v35 candidate control map is incomplete")
    if any(len(controls) != 1 for controls in controls_by_index.values()):
        raise ValueError("v35 candidate control map drifted across rows")
    controls = tuple(
        next(iter(controls_by_index[index])) for index in range(22)
    )
    if controls[0] != (0.0, 0.0, 0.0):
        raise ValueError("v35 candidate zero arm is not strict zero")
    mapping_fingerprint = hashlib.sha256(
        _canonical_bytes(controls)
    ).hexdigest()
    return Gate2CandidateControlContract(
        controls=controls,
        source_sha256=_sha256_file(source),
        mapping_fingerprint=mapping_fingerprint,
    )


def gate2_permutation_action_index(*, seed: int, global_index: int) -> int:
    if seed not in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        raise ValueError(f"unregistered Gate 2 longitudinal seed {seed}")
    if (
        isinstance(global_index, bool)
        or not isinstance(global_index, int)
        or global_index < 0
    ):
        raise ValueError("Gate 2 permutation index must be non-negative")
    seed_rank = GATE11_LONGITUDINAL_SOURCE_SEEDS.index(seed)
    return (global_index + seed_rank * 7) % 22


def _load_selector_model(
    selector_artifact_path: Path,
) -> KernelResidualActionSelectorArtifact:
    load_gate2_v35_selector_bundle(selector_artifact_path)
    wrapper = json.loads(selector_artifact_path.read_text(encoding="utf-8"))
    artifact = selector_artifact_from_payload(wrapper["artifact"])
    if not isinstance(artifact, KernelResidualActionSelectorArtifact):
        raise TypeError("v35 longitudinal selector must be kernel ridge")
    return artifact


def build_gate2_longitudinal_capture_runtime(
    *,
    expected_basis_fingerprint: str = (
        GATE2_V35_CONTROL_BASIS_FINGERPRINT
    ),
) -> tuple[OpenWeightResidualRuntime, dict[str, str]]:
    """Build real Qwen and deterministically restore the v35 learned basis."""

    config = ETAOpenWeightRuntimeConfig(
        device="cpu",
        activation_width=896,
        max_prefix_steps=8,
    )
    runtime = _build_eta_open_weight_runtime(config)
    _validate_eta_open_weight_runtime(runtime=runtime, config=config)
    provenance = _install_learned_control_basis(
        runtime=runtime,
        cases=eta_gate2_selector_fresh_cases(),
        open_weight_config=config,
    )
    actual_fingerprint = provenance["control_basis_fingerprint"]
    if actual_fingerprint != expected_basis_fingerprint:
        raise ValueError(
            "Gate 2 longitudinal learned basis fingerprint drift: "
            f"expected={expected_basis_fingerprint}, "
            f"actual={actual_fingerprint}"
        )
    return runtime, provenance


def _capture_snapshot(
    *,
    runtime: OpenWeightResidualRuntime,
    source_text: str,
) -> SubstrateSnapshot:
    capture = runtime.capture(source_text=source_text)
    if runtime.fallback_active:
        raise RuntimeError("Gate 2 longitudinal capture entered fallback")
    if not capture.residual_sequence or not capture.residual_activations:
        raise RuntimeError(
            "Gate 2 longitudinal capture lacks real residual sequence"
        )
    return SubstrateSnapshot(
        model_id=runtime.model_id,
        is_frozen=runtime.is_frozen,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=capture.token_logits,
        feature_surface=capture.feature_surface,
        residual_activations=capture.residual_activations,
        residual_sequence=capture.residual_sequence,
        unavailable_fields=(),
        description=(
            f"{capture.description} Gate 2 v35 longitudinal clean readout."
        ),
    )


def _strict_existing_prefix(
    rows: Mapping[str, Mapping[str, Any]],
    source_ids: tuple[str, ...],
    *,
    label: str,
) -> None:
    observed = tuple(rows)
    expected = source_ids[: len(observed)]
    if observed != expected:
        raise ValueError(
            f"Gate 2 {label} resume prefix drift: "
            f"expected={expected[:3]!r}.../{len(expected)}, "
            f"actual={observed[:3]!r}.../{len(observed)}"
        )


def capture_gate2_longitudinal_seed(
    *,
    source_root: str | Path,
    companion_root: str | Path,
    seed: int,
    runtime: OpenWeightResidualRuntime,
    selector: KernelResidualActionSelectorArtifact,
    candidate_contract: Gate2CandidateControlContract,
    max_records: int | None = None,
) -> dict[str, object]:
    """Capture/resume one immutable source seed."""

    source_seed_root = Path(source_root) / f"seed_{seed}"
    target_seed_root = Path(companion_root) / f"seed_{seed}"
    target_seed_root.mkdir(parents=True, exist_ok=True)
    source_path = source_seed_root / "transitions.jsonl"
    records = load_gate11_longitudinal_source_records(source_path)
    plans = build_gate11_longitudinal_source_plans(seed)
    validate_gate11_longitudinal_source_prefix(
        records=records,
        plans=plans,
    )
    if max_records is not None:
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records < 1
        ):
            raise ValueError("max_records must be a positive integer")
        records = records[:max_records]
    source_ids = tuple(str(record["transition_id"]) for record in records)
    input_path = target_seed_root / GATE2_LONGITUDINAL_INPUT_FILENAME
    outcome_path = target_seed_root / GATE2_LONGITUDINAL_OUTCOME_FILENAME
    inputs = _load_companion_records(input_path)
    outcomes = _load_companion_records(outcome_path)
    _strict_existing_prefix(inputs, source_ids, label="readout input")
    _strict_existing_prefix(outcomes, source_ids, label="matched outcome")
    if len(outcomes) > len(inputs):
        raise ValueError("Gate 2 outcome resume is ahead of readout input")

    for index in range(len(outcomes), len(records)):
        record = records[index]
        transition_id = str(record["transition_id"])
        source_input = record.get("input")
        if not isinstance(source_input, dict):
            raise ValueError(f"{transition_id} lacks source input")
        source_text = source_input.get("prediction_turn")
        continuation_text = source_input.get("settlement_turn")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"{transition_id} lacks prediction turn")
        if (
            not isinstance(continuation_text, str)
            or not continuation_text.strip()
        ):
            raise ValueError(f"{transition_id} lacks settlement turn")

        existing_input = inputs.get(transition_id)
        if existing_input is None:
            snapshot = _capture_snapshot(
                runtime=runtime,
                source_text=source_text,
            )
            state_features = residual_action_state_vector(snapshot)
            if len(state_features) != selector.input_dim:
                raise ValueError(
                    f"{transition_id} readout state dimension drift: "
                    f"expected={selector.input_dim}, "
                    f"actual={len(state_features)}"
                )
            action_values = selector.predict_action_values(state_features)
            selected_action_index = max(
                range(len(action_values)),
                key=lambda action_index: (
                    action_values[action_index],
                    -action_index,
                ),
            )
            input_payload = {
                "schema_version": (
                    GATE2_LONGITUDINAL_INPUT_SCHEMA_VERSION
                ),
                "transition_id": transition_id,
                "seed": seed,
                "global_index": int(record["global_index"]),
                "selector_fingerprint": GATE2_V35_SELECTOR_FINGERPRINT,
                "control_basis_fingerprint": (
                    GATE2_V35_CONTROL_BASIS_FINGERPRINT
                ),
                "state_features": list(state_features),
                "state_fingerprint": hashlib.sha256(
                    _canonical_bytes(state_features)
                ).hexdigest(),
                "predicted_action_values": list(action_values),
                "selected_action_index": selected_action_index,
                "capture_source": "real",
                "fallback_active": False,
                "substrate_mutation_applied": False,
                "runtime_origin": runtime.runtime_origin,
                "model_id": runtime.model_id,
            }
            _append_companion_row(input_path, input_payload)
            inputs[transition_id] = input_payload
        else:
            selected_action_index = int(
                existing_input["selected_action_index"]
            )
        permutation_action_index = gate2_permutation_action_index(
            seed=seed,
            global_index=int(record["global_index"]),
        )
        zero_control = candidate_contract.controls[0]
        selected_control = candidate_contract.controls[
            selected_action_index
        ]
        permutation_control = candidate_contract.controls[
            permutation_action_index
        ]
        score_by_control = {}
        for control in {
            zero_control,
            selected_control,
            permutation_control,
        }:
            score_by_control[control] = runtime.score_continuation(
                source_text=source_text,
                continuation_text=continuation_text,
                applied_control=control,
                track_scale=_TRACK_SCALE,
            ).mean_negative_log_likelihood
        zero_nll = score_by_control[zero_control]
        selected_nll = score_by_control[selected_control]
        permutation_nll = score_by_control[permutation_control]
        selected_delta = zero_nll - selected_nll
        permutation_delta = zero_nll - permutation_nll
        outcome_payload = {
            "schema_version": GATE2_LONGITUDINAL_OUTCOME_SCHEMA_VERSION,
            "transition_id": transition_id,
            "seed": seed,
            "global_index": int(record["global_index"]),
            "consumer_session_index": (
                int(record["global_index"]) // 10
            ),
            "selector_fingerprint": GATE2_V35_SELECTOR_FINGERPRINT,
            "control_basis_fingerprint": (
                GATE2_V35_CONTROL_BASIS_FINGERPRINT
            ),
            "selected_action_index": selected_action_index,
            "permutation_action_index": permutation_action_index,
            "zero_nll": zero_nll,
            "selected_nll": selected_nll,
            "permutation_nll": permutation_nll,
            "selected_realized_delta": selected_delta,
            "zero_realized_delta": 0.0,
            "permutation_null_mean": permutation_delta,
            "selector_minus_permutation": (
                selected_delta - permutation_delta
            ),
            "selected_equals_permutation": (
                selected_action_index == permutation_action_index
            ),
            "permutation_schedule": (
                "(global_index+seed_rank*7)%22"
            ),
            "outcome_chain": (
                "isolated-residual-forward->"
                "realized-continuation-nll-readout"
            ),
            "typed_pe_credit_executed": False,
            "source_fixed_outcome_reused": False,
            "substrate_mutation_applied": False,
            "source_record_sha256": record["record_sha256"],
        }
        _append_companion_row(outcome_path, outcome_payload)
        outcomes[transition_id] = outcome_payload
        if (index + 1) % 10 == 0 or index + 1 == len(records):
            _write_json(
                target_seed_root / "progress.json",
                {
                    "schema_version": (
                        GATE2_LONGITUDINAL_CAPTURE_SCHEMA_VERSION
                    ),
                    "seed": seed,
                    "source_transition_count": len(records),
                    "readout_input_count": len(inputs),
                    "matched_outcome_count": len(outcomes),
                    "last_transition_id": transition_id,
                },
            )
    return summarize_gate2_longitudinal_seed(
        seed=seed,
        source_transition_count=len(records),
        inputs=inputs,
        outcomes=outcomes,
    )


def summarize_gate2_longitudinal_seed(
    *,
    seed: int,
    source_transition_count: int,
    inputs: Mapping[str, Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    selected_minus_permutation = tuple(
        float(row["selector_minus_permutation"])
        for row in outcomes.values()
    )
    selected_minus_zero = tuple(
        float(row["selected_realized_delta"])
        for row in outcomes.values()
    )
    by_session: dict[int, list[float]] = {}
    for row in outcomes.values():
        by_session.setdefault(
            int(row["consumer_session_index"]),
            [],
        ).append(float(row["selector_minus_permutation"]))
    session_means = tuple(
        statistics.fmean(values)
        for _, values in sorted(by_session.items())
    )
    selected_actions = tuple(
        int(row["selected_action_index"]) for row in outcomes.values()
    )
    action_counts = {
        action_index: selected_actions.count(action_index)
        for action_index in sorted(set(selected_actions))
    }
    action_probabilities = tuple(
        count / len(selected_actions)
        for count in action_counts.values()
    ) if selected_actions else ()
    action_entropy = -sum(
        probability * math.log(probability)
        for probability in action_probabilities
        if probability > 0.0
    )
    complete = (
        source_transition_count >= 500
        and len(inputs) == source_transition_count
        and len(outcomes) == source_transition_count
    )
    primary_mean = (
        statistics.fmean(selected_minus_permutation)
        if selected_minus_permutation
        else 0.0
    )
    zero_mean = (
        statistics.fmean(selected_minus_zero)
        if selected_minus_zero
        else 0.0
    )
    session_positive_rate = (
        sum(value > 0.0 for value in session_means) / len(session_means)
        if session_means
        else 0.0
    )
    gates = {
        "count_at_least_500": complete,
        "selector_minus_permutation_at_least_0_02": (
            complete
            and primary_mean >= GATE2_LONGITUDINAL_CAPTURE_MIN_EFFECT
        ),
        "selector_minus_zero_at_least_0_02": (
            complete
            and zero_mean >= GATE2_LONGITUDINAL_CAPTURE_MIN_EFFECT
        ),
        "session_positive_rate_at_least_0_60": (
            complete
            and session_positive_rate
            >= GATE2_LONGITUDINAL_CAPTURE_MIN_SESSION_POSITIVE_RATE
        ),
    }
    return {
        "seed": seed,
        "source_transition_count": source_transition_count,
        "readout_input_count": len(inputs),
        "matched_outcome_count": len(outcomes),
        "consumer_session_count": len(session_means),
        "selector_minus_permutation_mean": primary_mean,
        "selector_minus_zero_mean": zero_mean,
        "session_primary_positive_rate": session_positive_rate,
        "selected_action_coverage": len(action_counts),
        "selected_action_entropy": action_entropy,
        "selected_action_counts": action_counts,
        "selected_equals_permutation_rate": (
            sum(
                bool(row["selected_equals_permutation"])
                for row in outcomes.values()
            )
            / len(outcomes)
            if outcomes
            else 0.0
        ),
        "gates": gates,
        "complete": complete,
        "single_seed_stoploss_passed": all(gates.values()),
    }


def _confidence_interval_95(
    values: Sequence[float],
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else 0.0
        return (value, value)
    mean = statistics.fmean(values)
    half_width = (
        _T_CRITICAL_95_DF2
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    return (mean - half_width, mean + half_width)


def export_gate2_longitudinal_capture_bundle(
    *,
    output_root: str | Path,
    source_root: str | Path,
    selector_artifact_path: str | Path,
    candidate_artifact_path: str | Path,
    selector_lineage: Mapping[str, object],
    candidate_contract: Gate2CandidateControlContract,
    basis_provenance: Mapping[str, str],
    source_hashes_before: Mapping[str, str],
) -> dict[str, object]:
    root = Path(output_root)
    seed_summaries = []
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        seed_root = root / f"seed_{seed}"
        source_path = Path(source_root) / f"seed_{seed}" / "transitions.jsonl"
        records = load_gate11_longitudinal_source_records(source_path)
        inputs = _load_companion_records(
            seed_root / GATE2_LONGITUDINAL_INPUT_FILENAME
        )
        outcomes = _load_companion_records(
            seed_root / GATE2_LONGITUDINAL_OUTCOME_FILENAME
        )
        seed_summaries.append(
            summarize_gate2_longitudinal_seed(
                seed=seed,
                source_transition_count=len(records),
                inputs=inputs,
                outcomes=outcomes,
            )
        )
    seed_1201 = seed_summaries[0]
    all_complete = all(
        bool(summary["complete"]) for summary in seed_summaries
    )
    all_seed_gates = all(
        bool(summary["single_seed_stoploss_passed"])
        for summary in seed_summaries
    )
    primary_means = tuple(
        float(summary["selector_minus_permutation_mean"])
        for summary in seed_summaries
        if summary["complete"]
    )
    confidence_interval = _confidence_interval_95(primary_means)
    cross_seed_ci_gate = (
        all_complete
        and confidence_interval[0]
        >= GATE2_LONGITUDINAL_CAPTURE_MIN_EFFECT
    )
    readout_supported = all_complete and all_seed_gates and cross_seed_ci_gate
    single_seed_stoploss_failed = (
        bool(seed_1201["complete"])
        and not bool(seed_1201["single_seed_stoploss_passed"])
    )
    status = (
        "longitudinal-readout-supported"
        if readout_supported
        else "single-seed-stoploss"
        if single_seed_stoploss_failed
        else "not-supported"
        if all_complete
        else "capture-in-progress"
    )
    source_hashes_after = {
        path: _sha256_file(Path(path)) for path in source_hashes_before
    }
    source_unchanged = dict(source_hashes_before) == source_hashes_after
    if not source_unchanged:
        raise RuntimeError("Gate 2 longitudinal capture mutated a frozen input")
    manifest = {
        "schema_version": GATE2_LONGITUDINAL_CAPTURE_SCHEMA_VERSION,
        "source_root": str(Path(source_root)),
        "selector_artifact_path": str(Path(selector_artifact_path)),
        "candidate_artifact_path": str(Path(candidate_artifact_path)),
        "selector_lineage": dict(selector_lineage),
        "candidate_mapping_fingerprint": (
            candidate_contract.mapping_fingerprint
        ),
        "basis_provenance": dict(basis_provenance),
        "track_scale": list(_TRACK_SCALE),
        "permutation_schedule": "(global_index+seed_rank*7)%22",
        "min_effect": GATE2_LONGITUDINAL_CAPTURE_MIN_EFFECT,
        "min_session_positive_rate": (
            GATE2_LONGITUDINAL_CAPTURE_MIN_SESSION_POSITIVE_RATE
        ),
        "required_files": list(
            GATE2_LONGITUDINAL_CAPTURE_REQUIRED_FILES
        ),
        "preregistered_plan": (
            ".cursor/plans/"
            "gate-2-longitudinal-v35-companion-capture_20260730.plan.md"
        ),
    }
    _write_json(root / "manifest.yaml", manifest)
    _write_jsonl(
        root / "predictions.jsonl",
        tuple(
            {
                "seed": summary["seed"],
                "preregistered_min_effect": (
                    GATE2_LONGITUDINAL_CAPTURE_MIN_EFFECT
                ),
                "preregistered_min_session_positive_rate": (
                    GATE2_LONGITUDINAL_CAPTURE_MIN_SESSION_POSITIVE_RATE
                ),
            }
            for summary in seed_summaries
        ),
    )
    _write_jsonl(root / "outcomes.jsonl", tuple(seed_summaries))
    _write_jsonl(
        root / "prediction_errors.jsonl",
        tuple(
            {
                "seed": summary["seed"],
                "selector_minus_permutation_mean": summary[
                    "selector_minus_permutation_mean"
                ],
                "selector_minus_zero_mean": summary[
                    "selector_minus_zero_mean"
                ],
            }
            for summary in seed_summaries
        ),
    )
    _write_jsonl(
        root / "segments.jsonl",
        tuple(
            {
                "seed": summary["seed"],
                "consumer_session_count": summary[
                    "consumer_session_count"
                ],
                "session_primary_positive_rate": summary[
                    "session_primary_positive_rate"
                ],
            }
            for summary in seed_summaries
        ),
    )
    _write_jsonl(
        root / "credit.jsonl",
        tuple(
            {
                "seed": summary["seed"],
                "credit_kind": (
                    "realized-continuation-likelihood-improvement"
                ),
                "mean": summary["selector_minus_permutation_mean"],
            }
            for summary in seed_summaries
        ),
    )
    _write_jsonl(
        root / "action_selection.jsonl",
        tuple(
            {
                "seed": summary["seed"],
                "selected_action_coverage": summary[
                    "selected_action_coverage"
                ],
                "selected_action_entropy": summary[
                    "selected_action_entropy"
                ],
                "selected_action_counts": summary[
                    "selected_action_counts"
                ],
            }
            for summary in seed_summaries
        ),
    )
    ablation_results = {
        "schema_version": GATE2_LONGITUDINAL_CAPTURE_SCHEMA_VERSION,
        "seed_summaries": seed_summaries,
        "primary_seed_means": list(primary_means),
        "primary_confidence_interval_95": list(confidence_interval),
        "cross_seed_ci_gate": cross_seed_ci_gate,
        "readout_supported": readout_supported,
    }
    _write_json(root / "ablation_results.json", ablation_results)
    verdict = {
        "schema_version": GATE2_LONGITUDINAL_CAPTURE_SCHEMA_VERSION,
        "status": status,
        "single_seed_stoploss_failed": single_seed_stoploss_failed,
        "all_seeds_complete": all_complete,
        "cross_seed_ci_gate": cross_seed_ci_gate,
        "longitudinal_readout_supported": readout_supported,
        "official_gate2_longitudinal_verdict": "not-supported",
        "inherited_gate2_evidence_level": "causal-supported",
        "promotion_allowed": False,
        "live_injection_enabled": False,
    }
    _write_json(root / "promotion_verdict.json", verdict)
    rollback = {
        "schema_version": GATE2_LONGITUDINAL_CAPTURE_SCHEMA_VERSION,
        "source_hashes_before": dict(source_hashes_before),
        "source_hashes_after": source_hashes_after,
        "source_unchanged": source_unchanged,
        "selector_updated": False,
        "substrate_weights_updated": False,
        "runtime_owner_state_written": False,
        "live_injection_enabled": False,
        "rollback": "delete this companion artifact directory",
    }
    _write_json(root / "rollback_evidence.json", rollback)
    _write_jsonl(
        root / "state_diff.jsonl",
        (
            {
                "source_unchanged": source_unchanged,
                "selector_updated": False,
                "substrate_weights_updated": False,
                "runtime_owner_state_written": False,
            },
        ),
    )
    report = (
        "# Gate 2 longitudinal v35 companion capture\n\n"
        f"- status: `{status}`\n"
        f"- official Gate 2 longitudinal verdict: `not-supported`\n"
        f"- seed 1201 complete: `{seed_1201['complete']}`\n"
        f"- seed 1201 selector−permutation: "
        f"`{seed_1201['selector_minus_permutation_mean']:.6f}`\n"
        f"- seed 1201 selector−zero: "
        f"`{seed_1201['selector_minus_zero_mean']:.6f}`\n"
        f"- seed 1201 session positive rate: "
        f"`{seed_1201['session_primary_positive_rate']:.6f}`\n"
        f"- all seeds complete: `{all_complete}`\n"
        f"- cross-seed 95% CI: `{confidence_interval}`\n\n"
        "Controls were applied only inside isolated teacher-forced scoring "
        "for a frozen substrate. No session or owner state was written.\n"
    )
    (root / "report.md").write_text(report, encoding="utf-8")
    freeze_manifest = {
        name: _sha256_file(root / name)
        for name in GATE2_LONGITUDINAL_CAPTURE_REQUIRED_FILES
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for name in (
            GATE2_LONGITUDINAL_INPUT_FILENAME,
            GATE2_LONGITUDINAL_OUTCOME_FILENAME,
        ):
            path = root / f"seed_{seed}" / name
            if path.is_file():
                freeze_manifest[str(path.relative_to(root))] = (
                    _sha256_file(path)
                )
    _write_json(root / "freeze_manifest.json", freeze_manifest)
    return verdict


def reconcile_gate2_longitudinal_readout_chain(
    *,
    output_root: str | Path,
) -> int:
    """Correct the readout provenance label and refresh row digests."""

    root = Path(output_root)
    updated_count = 0
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        path = (
            root
            / f"seed_{seed}"
            / GATE2_LONGITUDINAL_OUTCOME_FILENAME
        )
        if not path.is_file():
            continue
        rows = _load_companion_records(path)
        normalized_rows = []
        for row in rows.values():
            normalized = dict(row)
            normalized.pop("record_sha256", None)
            normalized["outcome_chain"] = (
                "isolated-residual-forward->"
                "realized-continuation-nll-readout"
            )
            normalized["typed_pe_credit_executed"] = False
            normalized["record_sha256"] = hashlib.sha256(
                _canonical_bytes(normalized)
            ).hexdigest()
            normalized_rows.append(normalized)
        _write_jsonl(path, tuple(normalized_rows))
        updated_count += len(normalized_rows)
    freeze_manifest = {
        name: _sha256_file(root / name)
        for name in GATE2_LONGITUDINAL_CAPTURE_REQUIRED_FILES
        if (root / name).is_file()
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for name in (
            GATE2_LONGITUDINAL_INPUT_FILENAME,
            GATE2_LONGITUDINAL_OUTCOME_FILENAME,
        ):
            path = root / f"seed_{seed}" / name
            if path.is_file():
                freeze_manifest[str(path.relative_to(root))] = (
                    _sha256_file(path)
                )
    _write_json(root / "freeze_manifest.json", freeze_manifest)
    return updated_count


def run_gate2_longitudinal_companion_capture(
    *,
    source_root: str | Path,
    selector_artifact_path: str | Path,
    candidate_artifact_path: str | Path,
    output_root: str | Path,
    seeds: Sequence[int],
    max_records: int | None = None,
) -> dict[str, object]:
    """Run/resume the preregistered capture with seed-1201 stop-loss."""

    requested_seeds = tuple(seeds)
    if not requested_seeds:
        raise ValueError("Gate 2 longitudinal capture requires a seed")
    if any(seed not in GATE11_LONGITUDINAL_SOURCE_SEEDS for seed in seeds):
        raise ValueError("Gate 2 longitudinal capture seed is unregistered")
    selector_path = Path(selector_artifact_path)
    candidate_path = Path(candidate_artifact_path)
    selector_lineage = load_gate2_v35_selector_bundle(selector_path)
    selector = _load_selector_model(selector_path)
    candidate_contract = load_gate2_candidate_control_contract(candidate_path)
    source_hashes_before = {
        str(selector_path): _sha256_file(selector_path),
        str(candidate_path): _sha256_file(candidate_path),
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        source_path = Path(source_root) / f"seed_{seed}" / "transitions.jsonl"
        source_hashes_before[str(source_path)] = _sha256_file(source_path)
    runtime, basis_provenance = (
        build_gate2_longitudinal_capture_runtime()
    )
    for seed in requested_seeds:
        if seed != GATE11_LONGITUDINAL_SOURCE_SEEDS[0]:
            seed_1201_root = (
                Path(output_root)
                / f"seed_{GATE11_LONGITUDINAL_SOURCE_SEEDS[0]}"
            )
            source_1201 = load_gate11_longitudinal_source_records(
                Path(source_root)
                / f"seed_{GATE11_LONGITUDINAL_SOURCE_SEEDS[0]}"
                / "transitions.jsonl"
            )
            summary_1201 = summarize_gate2_longitudinal_seed(
                seed=GATE11_LONGITUDINAL_SOURCE_SEEDS[0],
                source_transition_count=len(source_1201),
                inputs=_load_companion_records(
                    seed_1201_root / GATE2_LONGITUDINAL_INPUT_FILENAME
                ),
                outcomes=_load_companion_records(
                    seed_1201_root / GATE2_LONGITUDINAL_OUTCOME_FILENAME
                ),
            )
            if not summary_1201["single_seed_stoploss_passed"]:
                raise RuntimeError(
                    "Gate 2 seed 1201 stop-loss blocks later seeds"
                )
        summary = capture_gate2_longitudinal_seed(
            source_root=source_root,
            companion_root=output_root,
            seed=seed,
            runtime=runtime,
            selector=selector,
            candidate_contract=candidate_contract,
            max_records=max_records,
        )
        if (
            seed == GATE11_LONGITUDINAL_SOURCE_SEEDS[0]
            and summary["complete"]
            and not summary["single_seed_stoploss_passed"]
        ):
            break
    return export_gate2_longitudinal_capture_bundle(
        output_root=output_root,
        source_root=source_root,
        selector_artifact_path=selector_path,
        candidate_artifact_path=candidate_path,
        selector_lineage=selector_lineage,
        candidate_contract=candidate_contract,
        basis_provenance=basis_provenance,
        source_hashes_before=source_hashes_before,
    )
