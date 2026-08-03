from __future__ import annotations

import hashlib
import math
import statistics
from typing import Callable

import pytest

from volvence_zero.agent.seven_day_n_plus_one import (
    SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER,
    SEVEN_DAY_N_PLUS_ONE_SCHEMA_VERSION,
    SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER,
    _context_source,
    _cosine,
    _mse,
    _sha256,
    _turn_rows,
    _vector_sha256,
    build_seven_day_n_plus_one_contract,
)


@pytest.fixture
def seven_day_n_plus_one_contract() -> dict[str, object]:
    return build_seven_day_n_plus_one_contract(
        sut_model={
            "model_id": "frozen/sut",
            "weights_sha256": "a" * 64,
            "frozen": True,
            "local_files_only": True,
        },
        execution_device="mps",
    )


@pytest.fixture
def attach_n_plus_one(
    seven_day_n_plus_one_contract: dict[str, object],
) -> Callable[[dict[str, object], float], None]:
    def attach(run: dict[str, object], quality: float) -> None:
        if not -1.0 <= quality <= 1.0:
            raise ValueError("synthetic N+1 quality must be within [-1, 1]")
        rows = _turn_rows(run)
        vectors = tuple(
            (math.cos(index * 0.17), math.sin(index * 0.17))
            for index in range(35)
        )
        case_id = f"{run['scenario_id']}:seed-{run['paraphrase_seed']}"
        target_rows = [
            {
                "sample_id": f"{case_id}:user-turn-{index + 1}",
                "day_index": row[0],
                "exchange_index": row[1],
                "source_sha256": hashlib.sha256(
                    row[2].encode("utf-8")
                ).hexdigest(),
                "values_sha256": _vector_sha256(vectors[index]),
            }
            for index, row in enumerate(rows)
        ]
        context_rows = []
        for index in range(34):
            source = _context_source(rows, end_index=index)
            context_rows.append(
                {
                    "sample_id": (
                        f"{case_id}:{run['arm_label']}:context-turn-{index + 1}"
                    ),
                    "history_turns": index + 1,
                    "source_sha256": hashlib.sha256(
                        source.encode("utf-8")
                    ).hexdigest(),
                    "values_sha256": hashlib.sha256(
                        f"{run['arm_label']}:{index}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        target_fingerprint = _sha256(
            {"case_id": case_id, "target_rows": target_rows}
        )
        model_fingerprint = {
            "model_id": "frozen/sut",
            "version": "test",
            "weights_sha256": "a" * 64,
        }
        target_lineage = {
            "schema_version": "substrate-forward-representation.v1",
            "snapshot_fingerprint": target_fingerprint,
            "model_fingerprint": model_fingerprint,
            "runtime_origin": "test-frozen-runtime",
            "readout_kind": "latest-token-selected-layer-residual-l2.v1",
            "layer_indices": [1],
            "activation_widths": [2],
            "representation_dim": 2,
        }
        context_lineage = {
            **target_lineage,
            "snapshot_fingerprint": _sha256(context_rows),
        }
        angle = math.acos(quality)
        observations = []
        for source_index in range(25, 35):
            actual = vectors[source_index]
            predicted = (
                actual[0] * math.cos(angle) - actual[1] * math.sin(angle),
                actual[0] * math.sin(angle) + actual[1] * math.cos(angle),
            )
            persistence = vectors[source_index - 1]
            cosine = _cosine(predicted, actual)
            mse = _mse(predicted, actual)
            persistence_cosine = _cosine(persistence, actual)
            persistence_mse = _mse(persistence, actual)
            observations.append(
                {
                    "sample_id": f"prediction-target-turn-{source_index + 1}",
                    "target_day_index": rows[source_index][0],
                    "target_exchange_index": rows[source_index][1],
                    "history_turns": source_index,
                    "predicted_representation": list(predicted),
                    "actual_representation": list(actual),
                    "persistence_representation": list(persistence),
                    "mean_squared_error": mse,
                    "cosine_similarity": cosine,
                    "cosine_error": 1.0 - cosine,
                    "persistence_mean_squared_error": persistence_mse,
                    "persistence_cosine_similarity": persistence_cosine,
                    "target_values_sha256": target_rows[source_index][
                        "values_sha256"
                    ],
                }
            )
        cosine_values = [float(row["cosine_similarity"]) for row in observations]
        mse_values = [float(row["mean_squared_error"]) for row in observations]
        persistence_cosine_values = [
            float(row["persistence_cosine_similarity"])
            for row in observations
        ]
        persistence_mse_values = [
            float(row["persistence_mean_squared_error"])
            for row in observations
        ]
        run["n_plus_one_representation_evidence"] = {
            "schema_version": SEVEN_DAY_N_PLUS_ONE_SCHEMA_VERSION,
            "contract_sha256": _sha256(seven_day_n_plus_one_contract),
            "target_owner": SEVEN_DAY_N_PLUS_ONE_TARGET_OWNER,
            "predictor_owner": SEVEN_DAY_N_PLUS_ONE_PREDICTOR_OWNER,
            "target_arm_independent": True,
            "target_personal_conditioning_applied": False,
            "evaluation_writeback_allowed": False,
            "target_lineage": target_lineage,
            "context_lineage": context_lineage,
            "target_rows": target_rows,
            "context_rows": context_rows,
            "target_sequence_sha256": _sha256(target_rows),
            "context_sequence_sha256": _sha256(context_rows),
            "training_example_count": 24,
            "heldout_example_count": 10,
            "head_parameter_fingerprint": _sha256(
                {"arm": run["arm_label"], "quality": quality}
            ),
            "heldout_mean_cosine_similarity": statistics.fmean(cosine_values),
            "heldout_mean_cosine_error": 1.0
            - statistics.fmean(cosine_values),
            "heldout_mean_squared_error": statistics.fmean(mse_values),
            "heldout_persistence_mean_cosine_similarity": statistics.fmean(
                persistence_cosine_values
            ),
            "heldout_persistence_mean_squared_error": statistics.fmean(
                persistence_mse_values
            ),
            "observations": observations,
        }

    return attach
