"""Tests for P1.3 training-job policy enforcement (403 when forbidden)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import TrainingJob, TrainingJobType
from dlaas_platform_api.app import (
    CONTROL_PLANE_STORES_KEY,
    _training_policy_error,
)
from dlaas_platform_registry import ContractNotFound


@dataclass
class _Contract:
    service_contract: dict[str, Any]


class _Contracts:
    def __init__(self, contract: _Contract | None) -> None:
        self._contract = contract

    async def get(self, contract_id: str) -> _Contract:
        if self._contract is None:
            raise ContractNotFound(contract_id)
        return self._contract


class _Stores:
    def __init__(self, contract: _Contract | None) -> None:
        self.contracts = _Contracts(contract)


def _request(contract: _Contract | None) -> Any:
    app: dict[str, Any] = {}
    if contract is not None or True:
        app[CONTROL_PLANE_STORES_KEY] = _Stores(contract)

    class _Req:
        def __init__(self, app):
            self.app = app

    return _Req(app)


def _adapter_job() -> TrainingJob:
    return TrainingJob(
        job_id="j1",
        ai_id="ai_1",
        contract_id="ctr_1",
        job_type=TrainingJobType.ADAPTER_CANDIDATE,
    )


async def test_forbidden_when_adapter_training_disabled() -> None:
    contract = _Contract(
        service_contract={
            "adoption_config": {
                "training": {"allow_adapter_training": False},
                "substrate": {"allow_rare_heavy_refresh": True},
            }
        }
    )
    resp = await _training_policy_error(_request(contract), _adapter_job())
    assert isinstance(resp, web.Response)
    assert resp.status == 403


async def test_forbidden_when_rare_heavy_disabled() -> None:
    contract = _Contract(
        service_contract={
            "adoption_config": {
                "training": {"allow_adapter_training": True},
                "substrate": {"allow_rare_heavy_refresh": False},
            }
        }
    )
    resp = await _training_policy_error(_request(contract), _adapter_job())
    assert isinstance(resp, web.Response)
    assert resp.status == 403


async def test_allowed_when_both_enabled() -> None:
    contract = _Contract(
        service_contract={
            "adoption_config": {
                "training": {"allow_adapter_training": True},
                "substrate": {"allow_rare_heavy_refresh": True},
            }
        }
    )
    assert await _training_policy_error(_request(contract), _adapter_job()) is None


async def test_permissive_when_no_contract() -> None:
    # ContractNotFound -> permissive (legacy / dev behaviour preserved).
    assert await _training_policy_error(_request(None), _adapter_job()) is None
