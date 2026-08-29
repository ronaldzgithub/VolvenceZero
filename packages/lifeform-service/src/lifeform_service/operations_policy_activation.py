"""Fail-closed Operations policy activation for the AutoCompany staging lane.

The evidence manifest is an integrity index, not an authorization by itself.
ACTIVE startup also requires the exact content-addressed activation receipt to
be pinned in deployment configuration.  The full benchmark, review, receipt,
and checkpoint lineage is then reconstructed before the controller is built.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from lifeform_domain_operations import (
    OPERATIONS_ACTIVATION_SCOPE,
    OPERATIONS_BENCHMARK_EVIDENCE_SCOPE,
    OperationsBrainController,
    OperationsPolicyActivationReceipt,
    OperationsPolicyBenchmarkReport,
    OperationsPolicyCheckpoint,
    OperationsPromotionReview,
    operations_policy_benchmark_preregistration,
    operations_policy_benchmark_scenario_set,
    validate_operations_policy_activation,
)
from lifeform_domain_operations.operations_brain_contracts import (
    stable_content_sha256,
)
from volvence_zero.runtime import WiringLevel


OPERATIONS_BRAIN_ENVIRONMENT_FIELD = "AUTOCOMPANY_OPERATIONS_BRAIN_ENVIRONMENT"
OPERATIONS_BRAIN_WIRING_FIELD = "AUTOCOMPANY_OPERATIONS_BRAIN_WIRING"
OPERATIONS_POLICY_BUNDLE_DIR_FIELD = "AUTOCOMPANY_OPERATIONS_POLICY_BUNDLE_DIR"
OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD = "AUTOCOMPANY_OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID"

_BUNDLE_SCHEMA_VERSION = "operations-policy-evidence-bundle.v1"
_BUNDLE_FILES = (
    "activation_receipt.json",
    "benchmark_report.json",
    "candidate_checkpoint.json",
    "preregistration.json",
    "promotion_review.json",
    "scenario_set.json",
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "activation_receipt_id",
        "activation_scope",
        "benchmark_report_id",
        "evidence_scope",
        "file_sha256",
        "production_default_changed",
        "promotion_review_id",
        "rollback_config_field",
    }
)
_ArtifactT = TypeVar("_ArtifactT")


class OperationsPolicyActivationError(ValueError):
    """The staged Operations policy bundle or runtime gate is invalid."""


@dataclass(frozen=True)
class OperationsPolicyActivationBundle:
    """Verified immutable artifacts admitted by the staging startup gate."""

    bundle_dir: Path
    candidate_checkpoint: OperationsPolicyCheckpoint
    benchmark_report: OperationsPolicyBenchmarkReport
    promotion_review: OperationsPromotionReview
    activation_receipt: OperationsPolicyActivationReceipt
    file_sha256: tuple[tuple[str, str], ...]


def _strict_fields(
    payload: Mapping[str, object],
    *,
    expected: frozenset[str],
    label: str,
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise OperationsPolicyActivationError(f"{label} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise OperationsPolicyActivationError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _read_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OperationsPolicyActivationError(f"cannot read Operations activation artifact {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationsPolicyActivationError(f"Operations activation artifact {path} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise OperationsPolicyActivationError(f"Operations activation artifact {path} must be a JSON object")
    return value, raw


def _require_text(payload: Mapping[str, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise OperationsPolicyActivationError(f"Operations activation manifest field {field} must be non-empty text")
    return value


def _require_manifest_file_hashes(
    value: object,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise OperationsPolicyActivationError("Operations activation manifest file_sha256 must be an object")
    names = frozenset(value)
    expected_names = frozenset(_BUNDLE_FILES)
    if names != expected_names:
        missing = expected_names - names
        unknown = names - expected_names
        detail = []
        if missing:
            detail.append(f"missing={','.join(sorted(missing))}")
        if unknown:
            detail.append(f"unknown={','.join(sorted(unknown))}")
        raise OperationsPolicyActivationError("Operations activation manifest file set drift: " + "; ".join(detail))
    result: list[tuple[str, str]] = []
    for name in _BUNDLE_FILES:
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OperationsPolicyActivationError(
                f"Operations activation manifest digest for {name} is not a lowercase SHA-256"
            )
        result.append((name, digest))
    return tuple(result)


def _parse_typed_artifact(
    *,
    label: str,
    parser: Callable[[Mapping[str, object]], _ArtifactT],
    payload: Mapping[str, object],
) -> _ArtifactT:
    try:
        return parser(payload)
    except (TypeError, ValueError) as exc:
        raise OperationsPolicyActivationError(f"invalid {label} in Operations activation bundle: {exc}") from exc


def load_operations_policy_activation_bundle(
    bundle_dir: str | Path,
    *,
    expected_activation_receipt_id: str,
) -> OperationsPolicyActivationBundle:
    """Load and fully revalidate one staging-only policy evidence bundle."""

    expected_receipt_id = expected_activation_receipt_id.strip()
    if not expected_receipt_id:
        raise OperationsPolicyActivationError("ACTIVE Operations startup requires an expected activation receipt id")
    root = Path(bundle_dir).expanduser()
    if not root.is_dir():
        raise OperationsPolicyActivationError(f"Operations activation bundle directory does not exist: {root}")

    manifest, _manifest_raw = _read_json_object(root / "manifest.json")
    _strict_fields(
        manifest,
        expected=_MANIFEST_FIELDS,
        label="Operations activation manifest",
    )
    if manifest["schema_version"] != _BUNDLE_SCHEMA_VERSION:
        raise OperationsPolicyActivationError("unsupported Operations activation bundle schema")
    if manifest["production_default_changed"] is not False:
        raise OperationsPolicyActivationError("Operations activation bundle must not change the production default")
    if _require_text(manifest, "activation_scope") != OPERATIONS_ACTIVATION_SCOPE:
        raise OperationsPolicyActivationError("Operations activation bundle scope is not autocompany_staging")
    if _require_text(manifest, "evidence_scope") != OPERATIONS_BENCHMARK_EVIDENCE_SCOPE:
        raise OperationsPolicyActivationError("Operations activation bundle evidence scope drift")
    if _require_text(manifest, "rollback_config_field") != OPERATIONS_BRAIN_WIRING_FIELD:
        raise OperationsPolicyActivationError("Operations activation bundle rollback field drift")
    manifest_receipt_id = _require_text(manifest, "activation_receipt_id")
    if manifest_receipt_id != expected_receipt_id:
        raise OperationsPolicyActivationError("Operations activation receipt does not match the deployment pin")

    file_hashes = _require_manifest_file_hashes(manifest["file_sha256"])
    payloads: dict[str, dict[str, object]] = {}
    for name, expected_digest in file_hashes:
        payload, raw = _read_json_object(root / name)
        observed_digest = hashlib.sha256(raw).hexdigest()
        if observed_digest != expected_digest:
            raise OperationsPolicyActivationError(f"Operations activation artifact raw SHA-256 mismatch: {name}")
        payloads[name] = payload

    candidate_checkpoint = _parse_typed_artifact(
        label="candidate checkpoint",
        parser=OperationsPolicyCheckpoint.from_json,
        payload=payloads["candidate_checkpoint.json"],
    )
    benchmark_report = _parse_typed_artifact(
        label="benchmark report",
        parser=OperationsPolicyBenchmarkReport.from_json,
        payload=payloads["benchmark_report.json"],
    )
    promotion_review = _parse_typed_artifact(
        label="promotion review",
        parser=OperationsPromotionReview.from_json,
        payload=payloads["promotion_review.json"],
    )
    activation_receipt = _parse_typed_artifact(
        label="activation receipt",
        parser=OperationsPolicyActivationReceipt.from_json,
        payload=payloads["activation_receipt.json"],
    )
    preregistration = payloads["preregistration.json"]
    scenario_set = payloads["scenario_set.json"]
    if preregistration != operations_policy_benchmark_preregistration():
        raise OperationsPolicyActivationError("Operations benchmark preregistration differs from the reviewed protocol")
    if scenario_set != operations_policy_benchmark_scenario_set():
        raise OperationsPolicyActivationError("Operations benchmark scenario set differs from the reviewed protocol")
    if stable_content_sha256(preregistration) != benchmark_report.preregistration_sha256:
        raise OperationsPolicyActivationError("Operations benchmark preregistration lineage mismatch")
    if stable_content_sha256(scenario_set) != benchmark_report.scenario_set_sha256:
        raise OperationsPolicyActivationError("Operations benchmark scenario-set lineage mismatch")
    if benchmark_report.candidate_checkpoint_sha256 != candidate_checkpoint.content_sha256:
        raise OperationsPolicyActivationError("Operations benchmark candidate checkpoint digest mismatch")
    if _require_text(manifest, "benchmark_report_id") != benchmark_report.report_id:
        raise OperationsPolicyActivationError("Operations activation manifest/report lineage mismatch")
    if _require_text(manifest, "promotion_review_id") != promotion_review.review_id:
        raise OperationsPolicyActivationError("Operations activation manifest/review lineage mismatch")
    if manifest_receipt_id != activation_receipt.activation_receipt_id:
        raise OperationsPolicyActivationError("Operations activation manifest/receipt lineage mismatch")
    try:
        validate_operations_policy_activation(
            report=benchmark_report,
            review=promotion_review,
            receipt=activation_receipt,
            candidate_checkpoint=candidate_checkpoint,
        )
    except (TypeError, ValueError) as exc:
        raise OperationsPolicyActivationError(f"Operations ModificationGate bundle validation failed: {exc}") from exc

    return OperationsPolicyActivationBundle(
        bundle_dir=root.resolve(),
        candidate_checkpoint=candidate_checkpoint,
        benchmark_report=benchmark_report,
        promotion_review=promotion_review,
        activation_receipt=activation_receipt,
        file_sha256=file_hashes,
    )


def _configured_value(
    environ: Mapping[str, str],
    field: str,
    *,
    default: str | None = None,
) -> str:
    raw = environ.get(field, default)
    if raw is None or not raw.strip():
        raise OperationsPolicyActivationError(f"{field} must be configured with a non-empty value")
    return raw.strip()


def build_operations_brain_controller_from_env(
    environ: Mapping[str, str] | None = None,
) -> OperationsBrainController:
    """Build the route controller with production-safe deployment defaults."""

    source = os.environ if environ is None else environ
    environment = _configured_value(
        source,
        OPERATIONS_BRAIN_ENVIRONMENT_FIELD,
        default="production",
    ).lower()
    if environment not in {"production", "staging"}:
        raise OperationsPolicyActivationError(f"{OPERATIONS_BRAIN_ENVIRONMENT_FIELD} must be production or staging")
    wiring_value = _configured_value(
        source,
        OPERATIONS_BRAIN_WIRING_FIELD,
        default=WiringLevel.SHADOW.value,
    ).lower()
    try:
        requested_wiring = WiringLevel(wiring_value)
    except ValueError as exc:
        raise OperationsPolicyActivationError(
            f"{OPERATIONS_BRAIN_WIRING_FIELD} must be disabled, shadow, or active"
        ) from exc

    if requested_wiring is WiringLevel.ACTIVE:
        if environment != "staging":
            raise OperationsPolicyActivationError("Operations policy ACTIVE is limited to the staging environment")
        bundle = load_operations_policy_activation_bundle(
            _configured_value(source, OPERATIONS_POLICY_BUNDLE_DIR_FIELD),
            expected_activation_receipt_id=_configured_value(
                source,
                OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD,
            ),
        )
        return OperationsBrainController(
            policy_checkpoint_seed=bundle.candidate_checkpoint,
            policy_wiring_level=WiringLevel.ACTIVE,
            activation_receipt=bundle.activation_receipt,
        )

    # DISABLED is enforced at the AutoCompany consumer boundary.  The shared
    # service route remains safe to expose, but its policy can only be SHADOW.
    return OperationsBrainController(policy_wiring_level=WiringLevel.SHADOW)


__all__ = (
    "OPERATIONS_BRAIN_ENVIRONMENT_FIELD",
    "OPERATIONS_BRAIN_WIRING_FIELD",
    "OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD",
    "OPERATIONS_POLICY_BUNDLE_DIR_FIELD",
    "OperationsPolicyActivationBundle",
    "OperationsPolicyActivationError",
    "build_operations_brain_controller_from_env",
    "load_operations_policy_activation_bundle",
)
