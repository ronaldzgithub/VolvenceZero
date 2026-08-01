"""Content-addressed build requests for the rare-heavy Common Adapter lane.

Forge may plan an immutable training run, but it never trains, evaluates,
publishes, or activates the resulting runtime bundle.  Those responsibilities
remain with the substrate pipeline and cognition-owned OFFLINE gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    SchemaStore,
    atomic_write_json,
    canonical_json,
    sha256_bytes,
    sha256_text,
    utc_now,
    utc_stamp,
)


@dataclass(frozen=True)
class RareHeavyTrainingSpec:
    common_adapter_version: str
    runtime_origin: str
    description: str
    seed: int
    target_modules: tuple[str, ...]
    hook_layers: tuple[int, ...]
    control_scale: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    max_steps: int
    state_kv_seed: int
    state_kv_states: int
    state_kv_epochs: int
    state_kv_slots: int
    state_kv_rank: int
    state_kv_norm_cap: float
    state_kv_learning_rate: float


@dataclass(frozen=True)
class RareHeavyEvaluationSpec:
    min_case_count: int = 8
    min_mean_relative_improvement: float = 0.01
    max_regression_rate: float = 0.25
    max_preservation_nll_regression: float = 0.05
    min_counterfactual_accuracy: float = 0.60


@dataclass(frozen=True)
class RareHeavyRequestResult:
    request_id: str
    request_path: Path


def create_rare_heavy_request(
    *,
    config: ForgeConfig,
    model_id: str,
    model_weights_sha256: str,
    traces_path: Path,
    control_basis_path: Path,
    held_out_path: Path,
    training: RareHeavyTrainingSpec,
    evaluation: RareHeavyEvaluationSpec | None = None,
    output_path: Path | None = None,
) -> RareHeavyRequestResult:
    """Freeze one bounded DISABLED request without starting expensive work."""

    trace_ref = _content_ref(config=config, path=traces_path, name="traces")
    trace_ref["trace_count"] = _nonempty_line_count(
        traces_path.expanduser().resolve(),
        name="traces",
    )
    inputs = {
        "traces": trace_ref,
        "control_basis": _content_ref(
            config=config,
            path=control_basis_path,
            name="control_basis",
        ),
        "held_out": _content_ref(
            config=config,
            path=held_out_path,
            name="held_out",
        ),
    }
    payload = {
        "schema_version": "forge-rare-heavy-request.v1",
        "request_id": "pending",
        "created_at": utc_now(),
        "owner": "vz-substrate",
        "requested_wiring": "DISABLED",
        "base_model": {
            "model_id": model_id,
            "weights_sha256": model_weights_sha256,
        },
        "inputs": inputs,
        "training": _jsonable_dataclass(training),
        "evaluation": _jsonable_dataclass(
            evaluation or RareHeavyEvaluationSpec()
        ),
        "training_order": ["rare-heavy", "state-kv", "offline-gate"],
        "training_decides_gate": False,
    }
    payload["request_id"] = _request_id(payload)
    SchemaStore(config.paths.forge_root / "schemas").validate(
        payload,
        "rare_heavy_request.schema.json",
    )
    destination = (
        output_path
        or config.paths.artifacts_root
        / f"forge_rare_heavy_{utc_stamp()}"
        / "request.json"
    ).expanduser().resolve()
    if not destination.is_relative_to(config.paths.artifacts_root):
        raise ForgeError("rare-heavy requests may only be written below artifacts/")
    atomic_write_json(destination, payload)
    return RareHeavyRequestResult(
        request_id=str(payload["request_id"]),
        request_path=destination,
    )


def validate_rare_heavy_request(
    *,
    config: ForgeConfig,
    request_path: Path,
) -> dict[str, object]:
    """Validate schema, identity, and every content-addressed request input."""

    from .foundation import read_json

    resolved = request_path.expanduser().resolve()
    payload = read_json(resolved)
    SchemaStore(config.paths.forge_root / "schemas").validate(
        payload,
        "rare_heavy_request.schema.json",
    )
    if payload["request_id"] != _request_id(payload):
        raise ForgeError("rare-heavy request_id does not match its payload")
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, dict):
        raise ForgeError("rare-heavy request inputs must be an object")
    for name in ("traces", "control_basis", "held_out"):
        raw_ref = raw_inputs[name]
        if not isinstance(raw_ref, dict):
            raise ForgeError(f"rare-heavy input {name} must be an object")
        locator = raw_ref["locator"]
        declared = raw_ref["sha256"]
        if not isinstance(locator, str) or not isinstance(declared, str):
            raise ForgeError(f"rare-heavy input {name} reference is malformed")
        source = Path(locator).expanduser()
        if not source.is_absolute():
            source = config.paths.repo_root / source
        try:
            actual = sha256_bytes(source.resolve().read_bytes())
        except OSError as exc:
            raise ForgeError(
                f"cannot read rare-heavy input {name}: {source}: {exc}"
            ) from exc
        if actual != declared:
            raise ForgeError(
                f"rare-heavy input {name} digest mismatch: "
                f"declared={declared}, actual={actual}"
            )
        if name == "traces":
            actual_count = _nonempty_line_count(source.resolve(), name="traces")
            if raw_ref["trace_count"] != actual_count:
                raise ForgeError(
                    "rare-heavy input traces count mismatch: "
                    f"declared={raw_ref['trace_count']}, actual={actual_count}"
                )
    return payload


def _content_ref(
    *,
    config: ForgeConfig,
    path: Path,
    name: str,
) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise ForgeError(f"cannot read rare-heavy input {name}: {resolved}: {exc}") from exc
    try:
        locator = resolved.relative_to(config.paths.repo_root).as_posix()
    except ValueError:
        locator = str(resolved)
    return {"locator": locator, "sha256": sha256_bytes(content)}


def _jsonable_dataclass(value: object) -> dict[str, object]:
    payload = asdict(value)
    return {
        name: list(item) if isinstance(item, tuple) else item
        for name, item in payload.items()
    }


def _nonempty_line_count(path: Path, *, name: str) -> int:
    try:
        count = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise ForgeError(f"cannot count rare-heavy input {name}: {path}: {exc}") from exc
    if count <= 0:
        raise ForgeError(f"rare-heavy input {name} must contain at least one record")
    return count


def _request_id(payload: dict[str, object]) -> str:
    canonical = {name: value for name, value in payload.items() if name != "request_id"}
    return f"rare-heavy:{sha256_text(canonical_json(canonical))}"


__all__ = [
    "RareHeavyEvaluationSpec",
    "RareHeavyRequestResult",
    "RareHeavyTrainingSpec",
    "create_rare_heavy_request",
    "validate_rare_heavy_request",
]
