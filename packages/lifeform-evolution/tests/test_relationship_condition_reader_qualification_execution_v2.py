from __future__ import annotations

import hashlib
import inspect
import pathlib
import sys
from typing import Mapping

import pytest

from lifeform_evolution import (
    relationship_condition_reader_qualification_execution_v2 as execution_v2,
)


_PROTOCOL_ID = hashlib.sha256(b"v2-protocol").hexdigest()
_RECEIPT_ID = hashlib.sha256(b"v2-anchor-receipt").hexdigest()
_RUN_NONCE = hashlib.sha256(b"v2-run-nonce").hexdigest()


def test_v2_outer_consumes_the_frozen_protocol_dependency_api() -> None:
    protocol_validator = inspect.signature(
        execution_v2.validate_relationship_condition_reader_qualification_execution_protocol_v2
    )
    assert tuple(protocol_validator.parameters) == (
        "payload",
        "expected_protocol_id",
        "repository_root",
        "preflight_root",
        "bge_snapshot_root",
    )
    assert protocol_validator.parameters["expected_protocol_id"].kind is (inspect.Parameter.KEYWORD_ONLY)

    anchor_validator = inspect.signature(
        execution_v2.validate_relationship_condition_reader_qualification_public_anchor_receipt_v2
    )
    assert tuple(anchor_validator.parameters) == (
        "receipt_payload",
        "expected_receipt_artifact_id",
        "execution_protocol_payload",
        "execution_protocol_raw",
        "expected_execution_protocol_id",
        "expected_execution_root",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in tuple(anchor_validator.parameters.values())[1:]
    )

    guard_factory = inspect.signature(execution_v2.relationship_condition_reader_qualification_integrity_guard_v2)
    assert tuple(guard_factory.parameters) == (
        "execution_protocol",
        "expected_execution_protocol_id",
        "repository_root",
        "bge_snapshot_root",
    )
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in guard_factory.parameters.values())


def _call_public_runner(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validator_result: str = _PROTOCOL_ID,
    prediction_timeout_seconds: object = 31,
    scorer_timeout_seconds: object = 17,
    python_executable: pathlib.Path | None = None,
) -> tuple[Mapping[str, object], dict[str, object], list[tuple[str, object]]]:
    protocol = {
        "schema_version": "test-v2-protocol",
        "opaque": "the wrapper must not derive authority from this payload",
        "runtime_identity": {"python": {"executable": str(pathlib.Path(sys.executable).resolve())}},
    }
    protocol_raw = b'{"opaque":"not-used-by-the-wrapper-test"}\n'
    receipt = {"artifact_id": _RECEIPT_ID}
    calls: list[tuple[str, object]] = []
    captured: dict[str, object] = {}
    fake_guard = object()

    def validate_protocol(
        payload: Mapping[str, object],
        *,
        expected_protocol_id: str,
        **kwargs: object,
    ) -> str:
        calls.append(("validate_protocol", (payload, expected_protocol_id, kwargs)))
        return validator_result

    def validate_anchor(*args: object, **kwargs: object) -> str:
        calls.append(("validate_anchor", (args, kwargs)))
        return _RECEIPT_ID

    def build_guard(**kwargs: object) -> object:
        calls.append(("build_guard", kwargs))
        return fake_guard

    def prediction_stage(**kwargs: object) -> Mapping[str, object]:
        calls.append(("prediction_stage", kwargs))
        return {"stage": "prediction"}

    def scoring_stage(**kwargs: object) -> Mapping[str, object]:
        calls.append(("scoring_stage", kwargs))
        return {"stage": "scoring"}

    def outer_core(**kwargs: object) -> Mapping[str, object]:
        captured.update(kwargs)
        assert kwargs["anchor_validator"] is validate_anchor
        assert (
            kwargs["anchor_validator"](
                kwargs["public_anchor_receipt_payload"],
                expected_receipt_artifact_id=kwargs["expected_public_anchor_receipt_artifact_id"],
                execution_protocol_payload=kwargs["execution_protocol_payload"],
                execution_protocol_raw=kwargs["execution_protocol_raw"],
                expected_execution_protocol_id=kwargs["expected_execution_protocol_id"],
                expected_execution_root=kwargs["execution_root"],
            )
            == _RECEIPT_ID
        )
        assert kwargs["integrity_guard_factory"]() is fake_guard
        assert kwargs["prediction_stage"](sentinel="prediction") == {"stage": "prediction"}
        assert kwargs["scoring_stage"](sentinel="scoring") == {"stage": "scoring"}
        return {
            "execution_protocol_id": kwargs["expected_execution_protocol_id"],
            "public_anchor_receipt_artifact_id": kwargs["expected_public_anchor_receipt_artifact_id"],
        }

    monkeypatch.setattr(
        execution_v2,
        "validate_relationship_condition_reader_qualification_execution_protocol_v2",
        validate_protocol,
    )
    monkeypatch.setattr(
        execution_v2,
        "validate_relationship_condition_reader_qualification_public_anchor_receipt_v2",
        validate_anchor,
    )
    monkeypatch.setattr(
        execution_v2,
        "relationship_condition_reader_qualification_integrity_guard_v2",
        build_guard,
    )
    monkeypatch.setattr(
        execution_v2._v1_execution,
        "execute_relationship_condition_reader_qualification_prediction_stage",
        prediction_stage,
    )
    monkeypatch.setattr(
        execution_v2._v1_execution,
        "execute_relationship_condition_reader_qualification_scoring_stage",
        scoring_stage,
    )
    monkeypatch.setattr(
        execution_v2._v1_execution,
        "_execute_authorized_qualification_with_stages",
        outer_core,
    )

    result = execution_v2.execute_authorized_relationship_condition_reader_qualification_execution_v2(
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=_PROTOCOL_ID,
        public_anchor_receipt_payload=receipt,
        expected_public_anchor_receipt_artifact_id=_RECEIPT_ID,
        repository_root=tmp_path / "repository",
        preflight_root=tmp_path / "preflight",
        bge_snapshot_root=tmp_path / "bge",
        execution_root=tmp_path / "execution",
        run_nonce=_RUN_NONCE,
        python_executable=python_executable or pathlib.Path(sys.executable),
        prediction_timeout_seconds=prediction_timeout_seconds,  # type: ignore[arg-type]
        scorer_timeout_seconds=scorer_timeout_seconds,  # type: ignore[arg-type]
    )
    return result, captured, calls


def test_v2_outer_threads_one_external_protocol_id_through_all_authority_hooks(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, captured, calls = _call_public_runner(tmp_path, monkeypatch)

    assert result == {
        "execution_protocol_id": _PROTOCOL_ID,
        "public_anchor_receipt_artifact_id": _RECEIPT_ID,
    }
    assert captured["expected_execution_protocol_id"] == _PROTOCOL_ID
    assert captured["expected_public_anchor_receipt_artifact_id"] == _RECEIPT_ID
    assert captured["prediction_timeout_seconds"] == 31
    assert captured["scorer_timeout_seconds"] == 17
    assert captured["python_executable"] == pathlib.Path(sys.executable).resolve()

    validation_call = calls[0]
    assert validation_call[0] == "validate_protocol"
    _payload, expected_protocol_id, optional_reobservations = validation_call[1]
    assert expected_protocol_id == _PROTOCOL_ID
    assert optional_reobservations == {}

    guard_call = next(value for name, value in calls if name == "build_guard")
    assert guard_call["expected_execution_protocol_id"] == _PROTOCOL_ID
    assert guard_call["execution_protocol"] is captured["execution_protocol_payload"]
    _anchor_args, anchor_kwargs = next(value for name, value in calls if name == "validate_anchor")
    assert anchor_kwargs["expected_execution_protocol_id"] == _PROTOCOL_ID
    assert anchor_kwargs["expected_receipt_artifact_id"] == _RECEIPT_ID


def test_v2_outer_rejects_a_validator_that_does_not_return_the_external_id(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        ValueError,
        match="v2 execution protocol validator returned an unexpected protocol id",
    ):
        _call_public_runner(
            tmp_path,
            monkeypatch,
            validator_result=hashlib.sha256(b"different-v2-protocol").hexdigest(),
        )


def test_v2_outer_rejects_child_python_outside_frozen_runtime_identity(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_python = tmp_path / "other-python.exe"
    other_python.write_bytes(b"not-the-frozen-python")

    with pytest.raises(
        ValueError,
        match="child Python differs from frozen runtime identity",
    ):
        _call_public_runner(
            tmp_path,
            monkeypatch,
            python_executable=other_python,
        )


@pytest.mark.parametrize(
    ("prediction_timeout_seconds", "scorer_timeout_seconds", "field_name"),
    [
        (True, 1, "prediction_timeout_seconds"),
        (0, 1, "prediction_timeout_seconds"),
        (1, False, "scorer_timeout_seconds"),
        (1, 0, "scorer_timeout_seconds"),
    ],
)
def test_v2_outer_rejects_invalid_timeouts_before_any_stage(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction_timeout_seconds: object,
    scorer_timeout_seconds: object,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _call_public_runner(
            tmp_path,
            monkeypatch,
            prediction_timeout_seconds=prediction_timeout_seconds,
            scorer_timeout_seconds=scorer_timeout_seconds,
        )
