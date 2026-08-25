"""Authorized outer runner for the v2 reader-qualification anchor contract.

The prediction and scoring mechanisms remain the reviewed v1 stages.  This
wrapper changes only the authority boundary: the protocol, public-anchor
receipt, and every chained integrity receipt are bound to an independently
supplied v2 execution-protocol ID.  It deliberately delegates sequencing and
final-manifest construction to the existing injectable outer core instead of
forking that execution logic.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Mapping

from . import relationship_condition_reader_qualification_execution as _v1_execution
from .relationship_condition_reader_qualification_execution_protocol_v2 import (
    relationship_condition_reader_qualification_integrity_guard_v2,
    validate_relationship_condition_reader_qualification_execution_protocol_v2,
    validate_relationship_condition_reader_qualification_public_anchor_receipt_v2,
)


def _frozen_python_executable(protocol: Mapping[str, object]) -> pathlib.Path:
    runtime_identity = protocol.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("v2 execution protocol runtime_identity must be a mapping")
    python_identity = runtime_identity.get("python")
    if not isinstance(python_identity, Mapping):
        raise ValueError("v2 execution protocol Python identity must be a mapping")
    executable_text = python_identity.get("executable")
    if not isinstance(executable_text, str) or not executable_text:
        raise ValueError("v2 execution protocol Python executable must be non-empty text")
    frozen = pathlib.Path(executable_text)
    if not frozen.is_absolute() or str(frozen) != executable_text:
        raise ValueError("v2 execution protocol Python executable is not canonical")
    resolved = frozen.resolve()
    if str(resolved) != executable_text:
        raise ValueError("v2 execution protocol Python executable does not resolve canonically")
    if not resolved.is_file():
        raise FileNotFoundError(f"frozen qualification Python executable is absent: {resolved}")
    return resolved


def execute_authorized_relationship_condition_reader_qualification_execution_v2(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    public_anchor_receipt_payload: Mapping[str, object],
    expected_public_anchor_receipt_artifact_id: str,
    repository_root: pathlib.Path,
    preflight_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
    execution_root: pathlib.Path,
    run_nonce: str,
    python_executable: pathlib.Path | None = None,
    prediction_timeout_seconds: int = 7_200,
    scorer_timeout_seconds: int = 600,
) -> Mapping[str, object]:
    """Run the existing qualification stages under v2 external authority.

    This entry point never derives either expected identity from the supplied
    payloads.  The caller must provide both the v2 protocol ID and the v2
    public-anchor receipt artifact ID through an independent channel.
    """

    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol_v2(
        execution_protocol_payload,
        expected_protocol_id=expected_execution_protocol_id,
    )
    if protocol_id != expected_execution_protocol_id:
        raise ValueError("v2 execution protocol validator returned an unexpected protocol id")

    frozen_executable = _frozen_python_executable(execution_protocol_payload)
    executable = pathlib.Path(python_executable or sys.executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"qualification Python executable is absent: {executable}")
    if str(executable) != str(frozen_executable):
        raise ValueError("qualification child Python differs from frozen runtime identity")
    for value, field_name in (
        (prediction_timeout_seconds, "prediction_timeout_seconds"),
        (scorer_timeout_seconds, "scorer_timeout_seconds"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")

    def integrity_guard_factory() -> object:
        return relationship_condition_reader_qualification_integrity_guard_v2(
            execution_protocol=execution_protocol_payload,
            expected_execution_protocol_id=protocol_id,
            repository_root=repository_root,
            bge_snapshot_root=bge_snapshot_root,
        )

    def prediction_stage(**kwargs: object) -> Mapping[str, object]:
        return _v1_execution.execute_relationship_condition_reader_qualification_prediction_stage(**kwargs)

    def scoring_stage(**kwargs: object) -> Mapping[str, object]:
        return _v1_execution.execute_relationship_condition_reader_qualification_scoring_stage(**kwargs)

    return _v1_execution._execute_authorized_qualification_with_stages(
        execution_protocol_payload=execution_protocol_payload,
        execution_protocol_raw=execution_protocol_raw,
        expected_execution_protocol_id=protocol_id,
        public_anchor_receipt_payload=public_anchor_receipt_payload,
        expected_public_anchor_receipt_artifact_id=(expected_public_anchor_receipt_artifact_id),
        repository_root=repository_root,
        preflight_root=preflight_root,
        bge_snapshot_root=bge_snapshot_root,
        execution_root=execution_root,
        run_nonce=run_nonce,
        python_executable=executable,
        prediction_timeout_seconds=prediction_timeout_seconds,
        scorer_timeout_seconds=scorer_timeout_seconds,
        anchor_validator=(validate_relationship_condition_reader_qualification_public_anchor_receipt_v2),
        integrity_guard_factory=integrity_guard_factory,
        prediction_stage=prediction_stage,
        scoring_stage=scoring_stage,
    )


__all__ = [
    "execute_authorized_relationship_condition_reader_qualification_execution_v2",
]
