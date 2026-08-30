from __future__ import annotations

import io
import json

import pytest

from lifeform_domain_venture import (
    FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION,
    build_foundry_public_contract,
    load_foundry_public_contract_fixture,
    load_foundry_public_contract_json_schema,
    validate_foundry_public_contract,
)
from lifeform_domain_venture.foundry_public_contract import main


def test_packaged_foundry_contract_is_content_addressed_and_exact() -> None:
    fixture = load_foundry_public_contract_fixture()
    assert fixture == build_foundry_public_contract()
    assert fixture["schema_version"] == FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION
    assert fixture["contract_id"] == (
        "venture-foundry-public-contract:" + str(fixture["content_sha256"])
    )
    assert fixture["decision_points"] == [
        "opportunity_brainstorm",
        "candidate_comparison",
        "experiment_planning",
        "product_design",
        "portfolio_review",
        "monitor_attribution",
        "stop_review",
    ]
    assert fixture["exposure_contract"]["pairing_arms"] == ["off", "shadow"]
    assert fixture["advice_contract"] == {
        "wiring_level": "shadow",
        "applied": False,
        "active_allowed": False,
        "adoption_is_learning_signal": False,
    }
    assert fixture["production_activation_contract"] == {
        "effectiveness_result": "candidate_only",
        "approval_authority": "foundry_named_human",
        "producer_activation_api_exposed": False,
        "producer_self_approval_allowed": False,
        "advice_active_allowed": False,
    }


def test_foundry_contract_rejects_hash_or_boundary_tampering() -> None:
    fixture = build_foundry_public_contract()
    with pytest.raises(ValueError, match="content_sha256"):
        validate_foundry_public_contract({**fixture, "content_sha256": "0" * 64})
    changed = json.loads(json.dumps(fixture))
    changed["advice_contract"]["applied"] = True
    with pytest.raises(ValueError, match="content_sha256"):
        validate_foundry_public_contract(changed)


def test_foundry_contract_schema_and_cli_are_read_only_and_reproducible(tmp_path) -> None:
    schema = json.loads(load_foundry_public_contract_json_schema())
    assert schema["properties"]["content_sha256"]["const"] == (
        build_foundry_public_contract()["content_sha256"]
    )
    shown = io.StringIO()
    assert main(["show"], stdout=shown) == 0
    assert json.loads(shown.getvalue()) == build_foundry_public_contract()

    fixture_path = tmp_path / "venture-foundry-contract.json"
    fixture_path.write_text(shown.getvalue(), encoding="utf-8")
    validated = io.StringIO()
    assert main(["validate", str(fixture_path)], stdout=validated) == 0
    assert json.loads(validated.getvalue()) == build_foundry_public_contract()
