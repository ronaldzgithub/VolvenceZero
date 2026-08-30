"""Content-addressed producer contract pinned by Foundry integrations.

The manifest describes only the public Venture Brain boundary. Foundry keeps
ownership of cohort assignment, effectiveness evaluation, activation review,
and every commercial or external action. This module deliberately exposes no
activation method.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TextIO

from lifeform_domain_venture.venture_brain_contracts import (
    ADVICE_SCHEMA_VERSION,
    CONTEXT_PACK_SCHEMA_VERSION,
    CONTEXT_REQUEST_SCHEMA_VERSION,
    OUTCOME_RECEIPT_SCHEMA_VERSION,
    OUTCOME_REPORT_SCHEMA_VERSION,
    VentureDecisionPoint,
    stable_content_sha256,
)


FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION = "venture-foundry-public-contract.v1"
FOUNDRY_PUBLIC_CONTRACT_ID_PREFIX = "venture-foundry-public-contract:"


def _contract_payload() -> dict[str, object]:
    return {
        "schema_version": FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION,
        "producer": "lifeform-domain-venture",
        "consumer": "foundry",
        "request_schema_version": CONTEXT_REQUEST_SCHEMA_VERSION,
        "context_pack_schema_version": CONTEXT_PACK_SCHEMA_VERSION,
        "advice_schema_version": ADVICE_SCHEMA_VERSION,
        "outcome_report_schema_version": OUTCOME_REPORT_SCHEMA_VERSION,
        "outcome_receipt_schema_version": OUTCOME_RECEIPT_SCHEMA_VERSION,
        "decision_points": [item.value for item in VentureDecisionPoint],
        "exposure_contract": {
            "owner": "foundry",
            "pairing_arms": ["off", "shadow"],
            "pairing_required": True,
            "required_exposure_fields": [
                "cohort_id",
                "arm",
                "user_id",
                "portfolio_id",
                "decision_id",
                "decision_point",
            ],
            "producer_exchange_lineage_fields": [
                "context_pack_id",
                "content_sha256",
                "advice.advice_id",
                "advice.applied",
            ],
            "producer_assigns_cohort": False,
            "producer_reports_effectiveness": False,
        },
        "advice_contract": {
            "wiring_level": "shadow",
            "applied": False,
            "active_allowed": False,
            "adoption_is_learning_signal": False,
        },
        "outcome_contract": {
            "typed_decision_field": "decision",
            "delayed_business_outcome_field": "commercial_outcome",
            "pe_eligible_pair": {
                "evidence_class": "field",
                "outcome_kind": "field_experiment_result",
            },
            "settlement_point": "next_new_context_pack",
        },
        "isolation_contract": {
            "user_scope_source": "lifeform_session_identity",
            "portfolio_scope_source": "request.portfolio_id",
            "cross_user_recall_allowed": False,
            "cross_portfolio_recall_allowed": False,
        },
        "rollback_contract": {
            "owner": "foundry",
            "fallback_mode": "off",
            "stop_consuming_context_pack_is_rollback": True,
            "service_failure_auto_activation_allowed": False,
        },
        "production_activation_contract": {
            "effectiveness_result": "candidate_only",
            "approval_authority": "foundry_named_human",
            "producer_activation_api_exposed": False,
            "producer_self_approval_allowed": False,
            "advice_active_allowed": False,
        },
    }


def build_foundry_public_contract() -> dict[str, object]:
    """Return the canonical immutable-by-value Foundry producer manifest."""

    payload = _contract_payload()
    digest = stable_content_sha256(payload)
    return {
        **payload,
        "contract_id": f"{FOUNDRY_PUBLIC_CONTRACT_ID_PREFIX}{digest}",
        "content_sha256": digest,
    }


def validate_foundry_public_contract(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate identity, digest, and every frozen v1 capability boundary."""

    if not isinstance(payload, Mapping):
        raise ValueError("Foundry public contract must be a JSON object")
    value = dict(payload)
    expected = build_foundry_public_contract()
    if set(value) != set(expected):
        unknown = sorted(set(value) - set(expected))
        missing = sorted(set(expected) - set(value))
        raise ValueError(f"Foundry public contract fields changed: unknown={unknown!r} missing={missing!r}")
    if value.get("schema_version") != FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Foundry public contract schema_version is unsupported")
    digest = value.get("content_sha256")
    if not isinstance(digest, str):
        raise ValueError("Foundry public contract content_sha256 must be text")
    digest_payload = dict(value)
    digest_payload.pop("contract_id")
    digest_payload.pop("content_sha256")
    if stable_content_sha256(digest_payload) != digest:
        raise ValueError("Foundry public contract content_sha256 does not match its payload")
    if value.get("contract_id") != f"{FOUNDRY_PUBLIC_CONTRACT_ID_PREFIX}{digest}":
        raise ValueError("Foundry public contract identity does not match its digest")
    if value != expected:
        raise ValueError("Foundry public contract v1 capability boundary changed")
    return value


def load_foundry_public_contract_fixture() -> dict[str, object]:
    """Load and validate the packaged interop fixture."""

    raw = (
        resources.files("lifeform_domain_venture")
        .joinpath("fixtures/venture_foundry_public_contract.v1.json")
        .read_text(encoding="utf-8")
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("packaged Foundry public contract fixture must be an object")
    return validate_foundry_public_contract(payload)


def load_foundry_public_contract_json_schema() -> str:
    """Return the packaged JSON Schema for independent consumer validation."""

    return (
        resources.files("lifeform_domain_venture")
        .joinpath("schemas/venture_foundry_public_contract.v1.schema.json")
        .read_text(encoding="utf-8")
    )


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Show or validate the public manifest without contacting either system."""

    parser = argparse.ArgumentParser(prog="venture-foundry-contract")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="print the canonical producer contract")
    validate_parser = subparsers.add_parser("validate", help="validate a pinned contract fixture")
    validate_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    try:
        if args.command == "show":
            payload = build_foundry_public_contract()
        else:
            loaded = json.loads(args.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("Foundry public contract fixture must be a JSON object")
            payload = validate_foundry_public_contract(loaded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(_canonical_json(payload), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FOUNDRY_PUBLIC_CONTRACT_ID_PREFIX",
    "FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION",
    "build_foundry_public_contract",
    "load_foundry_public_contract_fixture",
    "load_foundry_public_contract_json_schema",
    "main",
    "validate_foundry_public_contract",
)
