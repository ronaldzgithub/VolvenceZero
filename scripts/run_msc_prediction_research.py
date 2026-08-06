#!/usr/bin/env python3
"""Run the real-human MSC N+1 capacity/four-arm research lane.

This runner never vendors corpus text and never routes mismatch around the PE
owner. Its default Volvence arm uses the complete service/runtime collector;
the legacy bounded-state mode remains an explicit pilot. Formal adjudication
also requires the isolated R5 temporal-capacity ladder and an immutable
post-smoke preregistration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import struct
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Protocol

from companion_bench.msc_corpus import MSCDyad, load_msc_split
from companion_bench.msc_runtime_collection import (
    MSCFullRuntimeCollectedSample,
    collect_msc_full_runtime_contexts,
    msc_runtime_scope_ids,
)
from companion_bench.prediction_research import (
    PREDICTION_ARMS,
    CapacityObservation,
    MSCDialogueTurn,
    MSCFullRuntimeContext,
    MSCNextTurnExample,
    PredictionObservation,
    SameSubstrateContextAttestation,
    TemporalCapacityObservation,
    adjudicate_capacity_ladder,
    adjudicate_prediction_experiment,
    adjudicate_temporal_capacity_ladder,
    build_msc_next_turn_examples,
    examples_fingerprint,
    render_long_context,
    render_stateless_context,
    render_summary_retrieval_context,
)
from lifeform_evolution.seven_day_companion import (
    HTTPSevenDayCompanionService,
)
from lifeform_evolution.seven_day_process_host import (
    SubprocessSevenDayServiceHost,
)
from volvence_zero.prediction import ForwardRepresentationBatch, PredictionErrorModule
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SubstrateForwardRepresentationLineage,
    SubstrateForwardRepresentationPublisher,
    SubstrateForwardRepresentationSnapshot,
    build_transformers_runtime_with_fallback,
    fingerprint_model_weight_files,
)

from companion_test_plan_common import guarded_mps_runner_entrypoint
from msc_prediction_checkpoint import PredictionRunCheckpointStore


TEMPORAL_CAPACITY_N_Z = (3, 16, 64, 256)
TEMPORAL_CAPACITY_FIXED_FORWARD_HEAD_N_Z = 3
MSC_FORMAL_PREREGISTRATION_SCHEMA_VERSION = "msc-n-plus-one-formal-prereg.v1"
MSC_PREDICTION_PLAN_ID = "msc-n-plus-one-prediction-mps.v1"


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)  # type: ignore[arg-type]
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing final result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: object) -> None:
    _write_immutable_bytes(
        path,
        (
            json.dumps(payload, indent=2, sort_keys=True, default=_jsonable)
            + "\n"
        ).encode("utf-8"),
    )


def _stable_subset(dyads: tuple[MSCDyad, ...], limit: int | None) -> tuple[MSCDyad, ...]:
    if limit is None:
        return dyads
    if limit < 1:
        raise ValueError("dyad limits must be positive")
    return tuple(
        sorted(
            dyads,
            key=lambda row: hashlib.sha256(row.dyad_id.encode("utf-8")).hexdigest(),
        )[:limit]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_formal_preregistration(
    path: Path,
    *,
    expected_run_configuration: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"MSC formal preregistration is not valid JSON: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("MSC formal preregistration root must be an object")
    if payload.get("schema_version") != MSC_FORMAL_PREREGISTRATION_SCHEMA_VERSION:
        raise ValueError("MSC formal preregistration schema is unsupported")
    if payload.get("claim_scope") != "msc-n-plus-one-pe-eta-load-bearing":
        raise ValueError("MSC formal preregistration claim_scope drift")
    if payload.get("run_configuration") != dict(expected_run_configuration):
        raise ValueError("MSC formal preregistration run configuration drift")
    if payload.get("temporal_capacity_intervention") != {
        "temporal_n_z": list(TEMPORAL_CAPACITY_N_Z),
        "fixed_forward_head_n_z": TEMPORAL_CAPACITY_FIXED_FORWARD_HEAD_N_Z,
        "all_other_factors_fixed": True,
        "execution_order": ["R3", "R4", "R5"],
    }:
        raise ValueError("MSC formal preregistration R5 intervention drift")
    if payload.get("primary_test") != {
        "comparison": "volvence-minus-long_context",
        "session_index": 5,
        "quality_min_cosine_advantage": 0.02,
        "quality_ci_lower_bound_strictly_positive": True,
        "quality_advantage_slope_strictly_positive": True,
        "scaling_min_cosine_equivalence": -0.01,
        "scaling_max_token_ratio": 0.10,
        "scaling_max_latency_ratio": 0.50,
        "stateless_and_summary_are_eligibility_only": True,
    }:
        raise ValueError("MSC formal preregistration primary test drift")
    smoke = payload.get("passed_smoke_artifact")
    if not isinstance(smoke, dict):
        raise ValueError("MSC formal preregistration lacks passed smoke artifact")
    smoke_path_value = smoke.get("path")
    smoke_sha256 = smoke.get("sha256")
    if not isinstance(smoke_path_value, str) or not isinstance(
        smoke_sha256, str
    ):
        raise ValueError("MSC formal preregistration smoke lineage is invalid")
    smoke_path = Path(smoke_path_value)
    if not smoke_path.is_absolute():
        smoke_path = resolved.parent / smoke_path
    if not smoke_path.is_file() or _sha256_file(smoke_path) != smoke_sha256:
        raise ValueError("MSC formal preregistration smoke artifact drift")
    try:
        smoke_payload = json.loads(smoke_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("MSC preregistered smoke artifact is not valid JSON") from exc
    if (
        not isinstance(smoke_payload, dict)
        or smoke_payload.get("schema_version") != "msc-r5-smoke-manifest.v1"
        or smoke_payload.get("passed") is not True
        or smoke_payload.get("formal_claim_permitted") is not False
    ):
        raise ValueError("MSC preregistered smoke artifact did not pass")
    return payload, _canonical_sha256(payload)


def _sha256_vector(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"!{len(values)}d", *values)).hexdigest()


def _source_sha256s(texts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts)


def _audit_payload(audit: object, *, msc_root: Path) -> dict[str, object]:
    payload = asdict(audit)  # type: ignore[arg-type]
    audit_path = Path(str(payload["path"])).resolve()
    try:
        payload["path"] = audit_path.relative_to(msc_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"MSC audit path {audit_path} is outside corpus root {msc_root.resolve()}"
        ) from exc
    payload["path_base"] = "msc_root"
    return payload


def _hashed_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact lineage file does not exist: {path}")
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _workspace_pythonpath(repository_root: Path) -> str:
    roots = tuple(
        str(path.resolve())
        for path in sorted((repository_root / "packages").glob("*/src"))
        if path.is_dir()
    )
    existing = os.environ.get("PYTHONPATH", "")
    return os.pathsep.join((*roots, *((existing,) if existing else ())))


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
        return port


def _acquire_output_lock(output_dir: Path) -> Any:
    """Hold a non-blocking process lock for one immutable output root."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - formal lane is POSIX/MPS
        raise RuntimeError("MSC output locking requires POSIX fcntl") from exc
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.msc-output.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"MSC output is already locked by another process: {output}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "schema_version": "msc-output-lock.v1",
                "output_dir": str(output),
                "pid": os.getpid(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


@contextmanager
def _running_msc_runtime_service(
    *,
    repository_root: Path,
    user_ids: tuple[str, ...],
    substrate_model: str,
    substrate_model_source: Path | None = None,
    substrate_device: str,
    substrate_layer_indices: tuple[int, ...],
    substrate_activation_width: int,
    substrate_max_length: int,
    substrate_weights_sha256: str,
    temporal_n_z: int,
    max_new_tokens: int,
    startup_timeout_s: float,
    evidence_profile: str = "msc-runtime-collector-v1",
    steering_bundle_path: Path | None = None,
) -> Iterator[
    tuple[
        Any,
        Mapping[str, object],
    ]
]:
    """Start one isolated service and yield a per-user HTTP client factory."""

    if not user_ids or len(set(user_ids)) != len(user_ids):
        raise ValueError("MSC runtime service users must be non-empty and unique")
    if not substrate_layer_indices:
        raise ValueError("MSC runtime service requires explicit residual layers")
    if temporal_n_z not in {3, 16, 64, 256}:
        raise ValueError("MSC runtime service temporal_n_z is not preregistered")
    if evidence_profile not in {
        "msc-runtime-collector-v1",
        "msc-steering-shadow-collector-v1",
    }:
        raise ValueError("MSC runtime service evidence profile is unsupported")
    if evidence_profile == "msc-steering-shadow-collector-v1":
        if steering_bundle_path is None or not steering_bundle_path.is_file():
            raise ValueError("MSC steering service requires an artifact bundle")
    elif steering_bundle_path is not None:
        raise ValueError("ordinary MSC runtime service cannot load steering bundle")
    port = _available_local_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="volvence-msc-runtime-") as raw_root:
        runtime_root = Path(raw_root)
        memory_root = runtime_root / "memory"
        evidence_root = runtime_root / "evidence"
        users_path = runtime_root / "alpha-users.json"
        users_path.write_text(
            json.dumps({"users": user_ids}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command_parts = [
            sys.executable,
            "-m",
            "lifeform_service.cli",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--vertical",
            "companion",
            "--substrate-mode",
            "hf-shared",
            "--substrate-model-id",
            substrate_model,
            "--substrate-device",
            substrate_device,
            "--substrate-local-files-only",
            "--substrate-layer-indices",
            *(str(index) for index in substrate_layer_indices),
            "--substrate-activation-width",
            str(substrate_activation_width),
            "--substrate-max-length",
            str(substrate_max_length),
            "--substrate-expected-weights-sha256",
            substrate_weights_sha256,
            "--alpha-enabled",
            "--alpha-users-file",
            str(users_path),
            "--memory-scope-root-dir",
            str(memory_root),
            "--evidence-root-dir",
            str(evidence_root),
            "--allow-evidence-time-override",
            "--companion-evidence-profile",
            evidence_profile,
            "--msc-temporal-n-z",
            str(temporal_n_z),
            "--max-sessions",
            "1",
            "--idle-eviction-seconds",
            "0",
            "--log-level",
            "INFO",
        ]
        if substrate_model_source is not None:
            command_parts.extend(
                ("--substrate-model-source", str(substrate_model_source))
            )
        if steering_bundle_path is not None:
            command_parts.extend(
                ("--steering-artifact-bundle", str(steering_bundle_path))
            )
        command = tuple(command_parts)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = _workspace_pythonpath(repository_root)
        environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["VZ_LIFEFORM_MAX_NEW_TOKENS"] = str(max_new_tokens)
        host_client = HTTPSevenDayCompanionService(
            base_url=base_url,
            user_id=user_ids[0],
            instance_id="msc-runtime-service-starting",
            vertical="companion",
            timeout_s=max(120.0, startup_timeout_s),
        )
        host = SubprocessSevenDayServiceHost(
            command=command,
            service=host_client,
            health_url=f"{base_url}/v1/health",
            expected_persistence_scope_sha256=hashlib.sha256(
                memory_root.resolve().as_posix().encode("utf-8")
            ).hexdigest(),
            log_dir=runtime_root / "logs",
            cwd=repository_root,
            environment=environment,
            startup_timeout_s=startup_timeout_s,
        )
        try:
            try:
                instance_id = host.start_initial()
            except (OSError, RuntimeError, TimeoutError) as exc:
                logs = tuple((runtime_root / "logs").glob("service-*.log"))
                detail = ""
                if logs:
                    detail = logs[-1].read_text(
                        encoding="utf-8", errors="replace"
                    )[-4000:]
                raise RuntimeError(
                    "MSC full-runtime service failed to start"
                    + (("\n" + detail) if detail else "")
                ) from exc
            attestation_path = (
                evidence_root / "companion_evidence_runtime_profile.json"
            )
            attestation = json.loads(
                attestation_path.read_text(encoding="utf-8")
            )
            if (
                not isinstance(attestation, dict)
                or attestation.get("profile") != evidence_profile
                or attestation.get("substrate_model_id") != substrate_model
                or attestation.get("temporal_n_z") != temporal_n_z
            ):
                raise ValueError("MSC runtime service profile attestation drift")

            def factory(user_id: str) -> HTTPSevenDayCompanionService:
                if user_id not in user_ids:
                    raise ValueError("MSC runtime service user is outside allowlist")
                return HTTPSevenDayCompanionService(
                    base_url=base_url,
                    user_id=user_id,
                    instance_id=instance_id,
                    vertical="companion",
                    timeout_s=max(120.0, startup_timeout_s),
                )

            try:
                yield factory, attestation
            except Exception as exc:
                logs = tuple((runtime_root / "logs").glob("service-*.log"))
                detail = ""
                if logs:
                    detail = logs[-1].read_text(
                        encoding="utf-8", errors="replace"
                    )[-8000:]
                raise RuntimeError(
                    "MSC full-runtime service failed during collection"
                    + (("\n" + detail) if detail else "")
                ) from exc
        finally:
            host.close()


class FrozenContextEncoder(Protocol):
    model_id: str
    device: str
    embedding_dim: int
    max_seq_length: int
    fingerprint: str
    same_substrate: bool

    def encode(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]: ...

    def token_cost(self, text: str) -> tuple[int, int]: ...


class FrozenSentenceEncoder:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_seq_length: int,
        batch_size: int,
    ) -> None:
        if max_seq_length < 8 or batch_size < 1:
            raise ValueError("encoder max_seq_length/batch_size are invalid")
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "MSC prediction research requires sentence-transformers and "
                "huggingface-hub"
            ) from exc
        snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
        self._model = SentenceTransformer(
            str(snapshot), local_files_only=True, device=device
        )
        transformer = self._model[0]
        declared_limit = int(self._model.max_seq_length)
        position_limit = int(transformer.auto_model.config.max_position_embeddings)
        self.max_seq_length = min(max_seq_length, declared_limit, position_limit)
        self._model.max_seq_length = self.max_seq_length
        self._tokenizer = self._model.tokenizer
        self._tokenizer.truncation_side = "left"
        self.batch_size = batch_size
        self.model_id = model_id
        self.device = device
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        self.same_substrate = False
        digest = hashlib.sha256()
        digest.update(model_id.encode("utf-8"))
        digest.update(str(self.max_seq_length).encode("ascii"))
        digest.update(b"recency-left-truncation")
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            relative = path.relative_to(snapshot).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(_sha256_file(path).encode("ascii"))
        self.fingerprint = digest.hexdigest()

    def encode(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        if not texts:
            return (), ()
        rows: list[tuple[float, ...]] = []
        latency_ms: list[float] = []
        for text in texts:
            started = time.perf_counter()
            values = self._model.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            latency_ms.append((time.perf_counter() - started) * 1000.0)
            rows.append(tuple(float(value) for value in values[0]))
        return (tuple(rows), tuple(latency_ms))

    def token_cost(self, text: str) -> tuple[int, int]:
        token_count = len(
            self._tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        )
        return (
            max(1, min(token_count, self.max_seq_length)),
            max(0, token_count - self.max_seq_length),
        )


class FrozenSubstrateContextEncoder:
    """Zero-truncation context encoder on the target-owning frozen substrate."""

    def __init__(
        self,
        *,
        model_id: str,
        snapshot: Path,
        model_fingerprint: SubstrateFingerprint,
        target_lineage: SubstrateForwardRepresentationLineage,
        device: str,
        activation_width: int,
        layer_indices: tuple[int, ...] | None,
    ) -> None:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MSC same-substrate context encoding requires transformers"
            ) from exc
        if target_lineage.model_fingerprint != model_fingerprint:
            raise ValueError(
                "same-substrate context encoder target/model fingerprint drift"
            )
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        context_limit = int(config.max_position_embeddings)
        if context_limit < 1:
            raise ValueError("substrate context limit must be positive")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        runtime = build_transformers_runtime_with_fallback(
            model_id=model_id,
            model_source=str(snapshot),
            device=device,
            layer_indices=layer_indices,
            hook_layer_selection="middle",
            activation_width=activation_width,
            max_length=context_limit,
            fail_on_truncation=True,
            local_files_only=True,
            fallback_mode="deny",
            runtime_mode="strict-local",
            expected_model_weights_sha256=model_fingerprint.weights_sha256,
        )
        self._publisher = SubstrateForwardRepresentationPublisher(
            runtime,
            model_fingerprint=model_fingerprint,
        )
        self._tokenizer = tokenizer
        self._target_lineage = target_lineage
        self.model_id = model_id
        self.device = device
        self.embedding_dim = target_lineage.representation_dim
        self.max_seq_length = context_limit
        self.same_substrate = True
        contract = {
            "schema_version": "msc-same-substrate-context-encoder.v1",
            "model_fingerprint": asdict(model_fingerprint),
            "readout_kind": target_lineage.readout_kind,
            "layer_indices": target_lineage.layer_indices,
            "activation_widths": target_lineage.activation_widths,
            "representation_dim": target_lineage.representation_dim,
            "context_limit": context_limit,
            "truncation_policy": "deny",
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _assert_lineage(
        self, lineage: SubstrateForwardRepresentationLineage
    ) -> None:
        expected = self._target_lineage
        comparable = (
            lineage.model_fingerprint,
            lineage.runtime_origin,
            lineage.readout_kind,
            lineage.layer_indices,
            lineage.activation_widths,
            lineage.representation_dim,
        )
        target = (
            expected.model_fingerprint,
            expected.runtime_origin,
            expected.readout_kind,
            expected.layer_indices,
            expected.activation_widths,
            expected.representation_dim,
        )
        if comparable != target:
            raise ValueError(
                "same-substrate context/target residual lineage mismatch"
            )

    def encode(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        rows: list[tuple[float, ...]] = []
        latency_ms: list[float] = []
        for index, source_text in enumerate(texts):
            self.token_cost(source_text)
            sample_id = (
                f"context:{index}:"
                f"{hashlib.sha256(source_text.encode('utf-8')).hexdigest()}"
            )
            started = time.perf_counter()
            snapshot = self._publisher.publish(((sample_id, source_text),))
            latency_ms.append((time.perf_counter() - started) * 1000.0)
            self._assert_lineage(snapshot.lineage)
            rows.append(snapshot.representations[0].values)
        return (tuple(rows), tuple(latency_ms))

    def token_cost(self, text: str) -> tuple[int, int]:
        token_count = len(
            self._tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        )
        if token_count > self.max_seq_length:
            raise ValueError(
                "MSC same-substrate context exceeds the frozen context limit: "
                f"{token_count} > {self.max_seq_length}"
            )
        return (max(1, token_count), 0)


def _unit(values: Iterable[float]) -> tuple[float, ...]:
    row = tuple(values)
    norm = math.sqrt(sum(value * value for value in row))
    if norm <= 1e-12:
        return tuple(0.0 for _ in row)
    return tuple(value / norm for value in row)


def _mean_vectors(
    vectors: tuple[tuple[float, ...], ...], *, dimension: int
) -> tuple[float, ...]:
    if not vectors:
        return tuple(0.0 for _ in range(dimension))
    return tuple(
        sum(row[index] for row in vectors) / len(vectors)
        for index in range(dimension)
    )


def _ema_vectors(
    vectors: tuple[tuple[float, ...], ...], *, dimension: int, decay: float = 0.85
) -> tuple[float, ...]:
    state = tuple(0.0 for _ in range(dimension))
    for vector in vectors:
        state = tuple(
            decay * previous + (1.0 - decay) * current
            for previous, current in zip(state, vector, strict=True)
        )
    return state


def _all_atomic_texts(
    examples_by_split: dict[str, tuple[MSCNextTurnExample, ...]]
) -> tuple[str, ...]:
    texts: dict[str, None] = {}
    for examples in examples_by_split.values():
        for example in examples:
            texts[example.target_text] = None
            for persona in example.personas:
                texts[persona] = None
            for turn in example.history:
                texts[turn.text] = None
    return tuple(texts)


def _target_atomic_texts(
    examples_by_split: dict[str, tuple[MSCNextTurnExample, ...]],
) -> tuple[str, ...]:
    texts: dict[str, None] = {}
    for examples in examples_by_split.values():
        for example in examples:
            texts[example.target_text] = None
            texts[example.latest_text] = None
    return tuple(
        sorted(
            texts,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )
    )


def _substrate_target_contract(
    *, model_id: str
) -> tuple[Path, SubstrateFingerprint]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "MSC substrate target capture requires huggingface-hub"
        ) from exc
    snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    model_fingerprint = SubstrateFingerprint(
        model_id=model_id,
        version=snapshot.name,
        weights_sha256=fingerprint_model_weight_files(snapshot),
    )
    return snapshot, model_fingerprint


def _substrate_context_limit(snapshot: Path) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "MSC substrate context limit requires transformers"
        ) from exc
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    context_limit = int(config.max_position_embeddings)
    if context_limit < 1:
        raise ValueError("MSC substrate context limit must be positive")
    return context_limit


def _build_substrate_target_publisher(
    *,
    model_id: str,
    snapshot: Path,
    model_fingerprint: SubstrateFingerprint,
    device: str,
    activation_width: int,
    layer_indices: tuple[int, ...] | None,
) -> SubstrateForwardRepresentationPublisher:
    runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        model_source=str(snapshot),
        device=device,
        layer_indices=layer_indices,
        hook_layer_selection="middle",
        activation_width=activation_width,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
    )
    return SubstrateForwardRepresentationPublisher(
        runtime,
        model_fingerprint=model_fingerprint,
    )


def _substrate_target_map(
    publisher: SubstrateForwardRepresentationPublisher,
    texts: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], SubstrateForwardRepresentationSnapshot]:
    sample_sources = tuple(
        (f"text:{hashlib.sha256(value.encode('utf-8')).hexdigest()}", value)
        for value in texts
    )
    snapshot = publisher.publish(sample_sources)
    return (
        {
            text: row.values
            for text, row in zip(texts, snapshot.representations, strict=True)
        },
        snapshot,
    )


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("MSC prediction checkpoints require numpy") from exc
    return np


def _array_vectors(
    value: object,
    *,
    rows: int,
    dimension: int | None,
    field: str,
) -> tuple[tuple[float, ...], ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(
            f"{field} checkpoint shape mismatch: expected {rows} rows, got {array.shape}"
        )
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(
            f"{field} checkpoint dimension mismatch: expected {dimension}, "
            f"got {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{field} checkpoint contains non-finite values")
    return tuple(tuple(float(item) for item in row) for row in array.tolist())


def _array_ints(value: object, *, rows: int, field: str) -> tuple[int, ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{field} checkpoint shape mismatch")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{field} checkpoint dtype must be integer")
    return tuple(int(item) for item in array.tolist())


def _array_floats(value: object, *, rows: int, field: str) -> tuple[float, ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{field} checkpoint shape mismatch")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} checkpoint contains non-finite values")
    return tuple(float(item) for item in array.tolist())


def _array_strings(value: object, *, rows: int, field: str) -> tuple[str, ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{field} checkpoint shape mismatch")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result):
        raise ValueError(f"{field} checkpoint contains an empty value")
    return result


def _array_scalar_string(value: object, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} checkpoint is missing")
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{field} checkpoint must be scalar")
    result = str(array.item())
    if not result:
        raise ValueError(f"{field} checkpoint is empty")
    return result


def _full_runtime_source_sha256s(dyad: MSCDyad) -> tuple[str, ...]:
    return _source_sha256s(
        (
            *dyad.initial_personas[0],
            *(
                utterance.text
                for session in dyad.sessions
                for utterance in session.utterances
            ),
        )
    )


def _full_runtime_checkpoint_identity(
    *,
    dyad: MSCDyad,
    examples: tuple[MSCNextTurnExample, ...],
    model_fingerprint: SubstrateFingerprint,
    layer_indices: tuple[int, ...],
    activation_width: int,
    context_limit: int,
    max_new_tokens: int,
    temporal_n_z: int,
) -> tuple[str, str, dict[str, object]]:
    scope_digest, _, _ = msc_runtime_scope_ids(dyad)
    unit = (
        f"contexts/volvence-runtime/nz-{temporal_n_z}/"
        f"{dyad.split}/{scope_digest}"
    )
    relative = (
        f"checkpoints/contexts/volvence-runtime/nz-{temporal_n_z}/"
        f"{dyad.split}-{scope_digest}.npz"
    )
    metadata = {
        "schema_version": "msc-full-runtime-dyad-checkpoint.v2",
        "split": dyad.split,
        "dyad_scope_sha256": scope_digest,
        "sample_ids": tuple(example.sample_id for example in examples),
        "example_fingerprint": examples_fingerprint(examples),
        "source_sha256": _full_runtime_source_sha256s(dyad),
        "model_fingerprint": asdict(model_fingerprint),
        "requested_layer_indices": layer_indices,
        "requested_activation_width": activation_width,
        "substrate_context_limit": context_limit,
        "generation_max_new_tokens": max_new_tokens,
        "temporal_n_z": temporal_n_z,
        "profile": "msc-runtime-collector-v1",
        "exposure_policy": "target-persona-plus-all-dialogue-turns.v1",
        "cost_accounting": "incremental-full-turn-plus-slow-loop.v1",
        "raw_text_retained": False,
        "evaluation_writeback_allowed": False,
    }
    return unit, relative, metadata


def _load_full_runtime_dyad(
    *,
    store: PredictionRunCheckpointStore,
    dyad: MSCDyad,
    examples: tuple[MSCNextTurnExample, ...],
    model_fingerprint: SubstrateFingerprint,
    layer_indices: tuple[int, ...],
    activation_width: int,
    context_limit: int,
    max_new_tokens: int,
    temporal_n_z: int,
) -> tuple[MSCFullRuntimeCollectedSample, ...] | None:
    unit, relative, metadata = _full_runtime_checkpoint_identity(
        dyad=dyad,
        examples=examples,
        model_fingerprint=model_fingerprint,
        layer_indices=layer_indices,
        activation_width=activation_width,
        context_limit=context_limit,
        max_new_tokens=max_new_tokens,
        temporal_n_z=temporal_n_z,
    )
    arrays = store.load_arrays(
        unit=unit,
        relative_path=relative,
        expected_metadata=metadata,
    )
    if arrays is None:
        return None
    rows = len(examples)
    expected_dimension = len(layer_indices) * activation_width
    vectors = _array_vectors(
        arrays.get("vectors"),
        rows=rows,
        dimension=expected_dimension,
        field="MSC full-runtime vectors",
    )
    values_sha256 = _array_strings(
        arrays.get("values_sha256"), rows=rows, field="values_sha256"
    )
    source_sha256 = _array_strings(
        arrays.get("source_sha256"), rows=rows, field="source_sha256"
    )
    active_speakers = _array_strings(
        arrays.get("active_speaker_id"),
        rows=rows,
        field="active_speaker_id",
    )
    context_input_tokens = _array_ints(
        arrays.get("context_input_tokens"),
        rows=rows,
        field="context_input_tokens",
    )
    context_output_tokens = _array_ints(
        arrays.get("context_output_tokens"),
        rows=rows,
        field="context_output_tokens",
    )
    generation_latency = _array_floats(
        arrays.get("generation_latency_ms"),
        rows=rows,
        field="generation_latency_ms",
    )
    context_latency = _array_floats(
        arrays.get("context_latency_ms"),
        rows=rows,
        field="context_latency_ms",
    )
    propagate_events = _array_ints(
        arrays.get("propagate_event_count"),
        rows=rows,
        field="propagate_event_count",
    )
    interval_input = _array_ints(
        arrays.get("interval_input_tokens"),
        rows=rows,
        field="interval_input_tokens",
    )
    interval_output = _array_ints(
        arrays.get("interval_output_tokens"),
        rows=rows,
        field="interval_output_tokens",
    )
    interval_latency = _array_floats(
        arrays.get("interval_latency_ms"),
        rows=rows,
        field="interval_latency_ms",
    )
    observation_turns = _array_ints(
        arrays.get("observation_turn_count"),
        rows=rows,
        field="observation_turn_count",
    )
    scene_boundaries = _array_ints(
        arrays.get("scene_boundary_count"),
        rows=rows,
        field="scene_boundary_count",
    )
    model_id = _array_scalar_string(arrays.get("model_id"), field="model_id")
    model_version = _array_scalar_string(
        arrays.get("model_version"), field="model_version"
    )
    weights_sha256 = _array_scalar_string(
        arrays.get("weights_sha256"), field="weights_sha256"
    )
    runtime_origin = _array_scalar_string(
        arrays.get("runtime_origin"), field="runtime_origin"
    )
    readout_kind = _array_scalar_string(
        arrays.get("readout_kind"), field="readout_kind"
    )
    slot_surface_sha256 = _array_scalar_string(
        arrays.get("runtime_slot_surface_sha256"),
        field="runtime_slot_surface_sha256",
    )
    stored_layers = _array_ints(
        arrays.get("layer_indices"),
        rows=len(layer_indices),
        field="layer_indices",
    )
    stored_widths = _array_ints(
        arrays.get("activation_widths"),
        rows=len(layer_indices),
        field="activation_widths",
    )
    temporal_values = _array_ints(
        arrays.get("temporal_n_z"), rows=1, field="temporal_n_z"
    )
    if (
        model_id != model_fingerprint.model_id
        or weights_sha256 != model_fingerprint.weights_sha256
        or stored_layers != layer_indices
        or stored_widths
        != tuple(activation_width for _ in layer_indices)
    ):
        raise ValueError("MSC full-runtime checkpoint substrate lineage drift")
    temporal_n_z = temporal_values[0]
    if temporal_n_z != metadata["temporal_n_z"]:
        raise ValueError("MSC full-runtime checkpoint temporal_n_z drift")
    return tuple(
        MSCFullRuntimeCollectedSample(
            context=MSCFullRuntimeContext(
                sample_id=example.sample_id,
                active_speaker_id=active_speakers[index],
                values=vectors[index],
                values_sha256=values_sha256[index],
                source_sha256=source_sha256[index],
                model_id=model_id,
                model_version=model_version,
                weights_sha256=weights_sha256,
                runtime_origin=runtime_origin,
                readout_kind=readout_kind,
                layer_indices=stored_layers,
                activation_widths=stored_widths,
                temporal_n_z=temporal_n_z,
                input_token_count=context_input_tokens[index],
                output_token_count=context_output_tokens[index],
                total_token_count=(
                    context_input_tokens[index] + context_output_tokens[index]
                ),
                generation_latency_ms=generation_latency[index],
                latency_ms=context_latency[index],
                propagate_event_count=propagate_events[index],
                runtime_slot_surface_sha256=slot_surface_sha256,
            ),
            interval_input_token_count=interval_input[index],
            interval_output_token_count=interval_output[index],
            interval_total_token_count=(
                interval_input[index] + interval_output[index]
            ),
            interval_latency_ms=interval_latency[index],
            observation_turn_count=observation_turns[index],
            scene_boundary_count=scene_boundaries[index],
        )
        for index, example in enumerate(examples)
    )


def _save_full_runtime_dyad(
    *,
    store: PredictionRunCheckpointStore,
    dyad: MSCDyad,
    examples: tuple[MSCNextTurnExample, ...],
    samples: tuple[MSCFullRuntimeCollectedSample, ...],
    model_fingerprint: SubstrateFingerprint,
    layer_indices: tuple[int, ...],
    activation_width: int,
    context_limit: int,
    max_new_tokens: int,
    temporal_n_z: int,
) -> None:
    if tuple(sample.sample_id for sample in samples) != tuple(
        example.sample_id for example in examples
    ):
        raise ValueError("MSC full-runtime dyad sample identity drift")
    unit, relative, metadata = _full_runtime_checkpoint_identity(
        dyad=dyad,
        examples=examples,
        model_fingerprint=model_fingerprint,
        layer_indices=layer_indices,
        activation_width=activation_width,
        context_limit=context_limit,
        max_new_tokens=max_new_tokens,
        temporal_n_z=temporal_n_z,
    )
    first = samples[0]
    if any(sample.context.temporal_n_z != temporal_n_z for sample in samples):
        raise ValueError("MSC full-runtime samples changed temporal_n_z")
    np = _require_numpy()
    store.save_arrays(
        unit=unit,
        relative_path=relative,
        metadata=metadata,
        arrays={
            "vectors": np.asarray(
                tuple(sample.context.values for sample in samples),
                dtype=np.float64,
            ),
            "values_sha256": np.asarray(
                tuple(sample.context.values_sha256 for sample in samples)
            ),
            "source_sha256": np.asarray(
                tuple(sample.context.source_sha256 for sample in samples)
            ),
            "active_speaker_id": np.asarray(
                tuple(sample.context.active_speaker_id for sample in samples)
            ),
            "context_input_tokens": np.asarray(
                tuple(sample.context.input_token_count for sample in samples),
                dtype=np.int64,
            ),
            "context_output_tokens": np.asarray(
                tuple(sample.context.output_token_count for sample in samples),
                dtype=np.int64,
            ),
            "generation_latency_ms": np.asarray(
                tuple(sample.context.generation_latency_ms for sample in samples),
                dtype=np.float64,
            ),
            "context_latency_ms": np.asarray(
                tuple(sample.context.latency_ms for sample in samples),
                dtype=np.float64,
            ),
            "propagate_event_count": np.asarray(
                tuple(sample.context.propagate_event_count for sample in samples),
                dtype=np.int64,
            ),
            "interval_input_tokens": np.asarray(
                tuple(sample.interval_input_token_count for sample in samples),
                dtype=np.int64,
            ),
            "interval_output_tokens": np.asarray(
                tuple(sample.interval_output_token_count for sample in samples),
                dtype=np.int64,
            ),
            "interval_latency_ms": np.asarray(
                tuple(sample.interval_latency_ms for sample in samples),
                dtype=np.float64,
            ),
            "observation_turn_count": np.asarray(
                tuple(sample.observation_turn_count for sample in samples),
                dtype=np.int64,
            ),
            "scene_boundary_count": np.asarray(
                tuple(sample.scene_boundary_count for sample in samples),
                dtype=np.int64,
            ),
            "model_id": np.asarray(first.context.model_id),
            "model_version": np.asarray(first.context.model_version),
            "weights_sha256": np.asarray(first.context.weights_sha256),
            "runtime_origin": np.asarray(first.context.runtime_origin),
            "readout_kind": np.asarray(first.context.readout_kind),
            "runtime_slot_surface_sha256": np.asarray(
                first.context.runtime_slot_surface_sha256
            ),
            "layer_indices": np.asarray(
                first.context.layer_indices, dtype=np.int64
            ),
            "activation_widths": np.asarray(
                first.context.activation_widths, dtype=np.int64
            ),
            "temporal_n_z": np.asarray(
                (first.context.temporal_n_z,), dtype=np.int64
            ),
        },
    )


def _prepare_full_runtime_arm(
    *,
    store: PredictionRunCheckpointStore,
    repository_root: Path,
    split_payload: dict[str, tuple[MSCDyad, ...]],
    examples_by_split: dict[str, tuple[MSCNextTurnExample, ...]],
    model_fingerprint: SubstrateFingerprint,
    layer_indices: tuple[int, ...],
    activation_width: int,
    context_limit: int,
    max_new_tokens: int,
    substrate_device: str,
    startup_timeout_s: float,
    temporal_n_z: int,
    included_splits: tuple[str, ...],
) -> tuple[
    dict[
        str,
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[float, ...],
        ],
    ],
    dict[str, object],
]:
    if temporal_n_z not in {3, 16, 64, 256}:
        raise ValueError("MSC temporal_n_z is not preregistered")
    if (
        not included_splits
        or len(set(included_splits)) != len(included_splits)
        or any(
            split not in {"train", "validation", "heldout"}
            for split in included_splits
        )
    ):
        raise ValueError("MSC runtime included_splits are invalid")
    by_dyad: dict[tuple[str, str], tuple[MSCFullRuntimeCollectedSample, ...]] = {}
    missing: list[tuple[str, MSCDyad, tuple[MSCNextTurnExample, ...]]] = []
    for split in included_splits:
        examples_for_dyad = {
            dyad.dyad_id: build_msc_next_turn_examples((dyad,))
            for dyad in split_payload[split]
        }
        for dyad in split_payload[split]:
            dyad_examples = examples_for_dyad[dyad.dyad_id]
            cached = _load_full_runtime_dyad(
                store=store,
                dyad=dyad,
                examples=dyad_examples,
                model_fingerprint=model_fingerprint,
                layer_indices=layer_indices,
                activation_width=activation_width,
                context_limit=context_limit,
                max_new_tokens=max_new_tokens,
                temporal_n_z=temporal_n_z,
            )
            if cached is None:
                missing.append((split, dyad, dyad_examples))
            else:
                by_dyad[(split, dyad.dyad_id)] = cached

    profile_unit = (
        f"contexts/volvence-runtime/nz-{temporal_n_z}/profile-attestation"
    )
    profile_path = (
        f"checkpoints/contexts/volvence-runtime/nz-{temporal_n_z}/"
        "profile-attestation.json"
    )
    cached_profile = store.load_json(
        unit=profile_unit,
        relative_path=profile_path,
    )
    if cached_profile is not None and not isinstance(cached_profile, dict):
        raise ValueError("MSC runtime profile checkpoint must be an object")
    if missing:
        user_ids = tuple(
            msc_runtime_scope_ids(dyad)[1] for _, dyad, _ in missing
        )
        with _running_msc_runtime_service(
            repository_root=repository_root,
            user_ids=user_ids,
            substrate_model=model_fingerprint.model_id,
            substrate_device=substrate_device,
            substrate_layer_indices=layer_indices,
            substrate_activation_width=activation_width,
            substrate_max_length=context_limit,
            substrate_weights_sha256=model_fingerprint.weights_sha256,
            temporal_n_z=temporal_n_z,
            max_new_tokens=max_new_tokens,
            startup_timeout_s=startup_timeout_s,
        ) as (service_factory, live_profile):
            if cached_profile is None:
                store.save_json(
                    unit=profile_unit,
                    relative_path=profile_path,
                    payload=live_profile,
                )
                cached_profile = dict(live_profile)
            elif cached_profile != live_profile:
                raise ValueError("MSC runtime service profile changed on resume")
            total = len(missing)
            for index, (split, dyad, dyad_examples) in enumerate(
                missing, start=1
            ):
                print(
                    f"[msc-runtime] {index}/{total} {split} "
                    f"scope={msc_runtime_scope_ids(dyad)[0][:12]}",
                    flush=True,
                )
                samples = collect_msc_full_runtime_contexts(
                    (dyad,), service_factory=service_factory
                )
                _save_full_runtime_dyad(
                    store=store,
                    dyad=dyad,
                    examples=dyad_examples,
                    samples=samples,
                    model_fingerprint=model_fingerprint,
                    layer_indices=layer_indices,
                    activation_width=activation_width,
                    context_limit=context_limit,
                    max_new_tokens=max_new_tokens,
                    temporal_n_z=temporal_n_z,
                )
                by_dyad[(split, dyad.dyad_id)] = samples
    if cached_profile is None:
        raise ValueError("MSC runtime contexts lack a profile attestation")

    prepared: dict[
        str,
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[float, ...],
        ],
    ] = {}
    all_samples: list[MSCFullRuntimeCollectedSample] = []
    for split in included_splits:
        samples = tuple(
            sample
            for dyad in split_payload[split]
            for sample in by_dyad[(split, dyad.dyad_id)]
        )
        if tuple(sample.sample_id for sample in samples) != tuple(
            example.sample_id for example in examples_by_split[split]
        ):
            raise ValueError("MSC full-runtime split sample order drift")
        all_samples.extend(samples)
        prepared[split] = (
            tuple(sample.context.values for sample in samples),
            tuple(sample.interval_total_token_count for sample in samples),
            tuple(0 for _ in samples),
            tuple(sample.interval_latency_ms for sample in samples),
        )
    surfaces = {
        (
            sample.context.model_id,
            sample.context.model_version,
            sample.context.weights_sha256,
            sample.context.runtime_origin,
            sample.context.readout_kind,
            sample.context.layer_indices,
            sample.context.activation_widths,
            sample.context.temporal_n_z,
            sample.context.runtime_slot_surface_sha256,
        )
        for sample in all_samples
    }
    if len(surfaces) != 1:
        raise ValueError("MSC full-runtime surface differs across checkpoints")
    surface = next(iter(surfaces))
    if surface[7] != temporal_n_z:
        raise ValueError("MSC full-runtime temporal capacity attestation drift")
    attestation = {
        "schema_version": "msc-full-runtime-collection-attestation.v2",
        "volvence_full_stack": True,
        "collector_profile_attestation": cached_profile,
        "model_id": surface[0],
        "model_version": surface[1],
        "weights_sha256": surface[2],
        "runtime_origin": surface[3],
        "readout_kind": surface[4],
        "layer_indices": surface[5],
        "activation_widths": surface[6],
        "temporal_n_z": surface[7],
        "runtime_slot_surface_sha256": surface[8],
        "sample_count": len(all_samples),
        "included_splits": included_splits,
        "dyad_count": sum(len(split_payload[split]) for split in included_splits),
        "minimum_propagate_event_count": min(
            sample.context.propagate_event_count for sample in all_samples
        ),
        "total_observation_turn_count": sum(
            sample.observation_turn_count for sample in all_samples
        ),
        "total_scene_boundary_count": sum(
            sample.scene_boundary_count for sample in all_samples
        ),
        "total_interval_token_count": sum(
            sample.interval_total_token_count for sample in all_samples
        ),
        "total_interval_latency_ms": sum(
            sample.interval_latency_ms for sample in all_samples
        ),
        "exposure_policy": "target-persona-plus-all-dialogue-turns.v1",
        "cost_accounting": "incremental-full-turn-plus-slow-loop.v1",
        "one_time_startup_cost_included": False,
        "final_post_target_teardown_cost_included": False,
        "raw_text_retained": False,
        "evaluation_writeback_allowed": False,
    }
    return prepared, attestation


def _validate_full_runtime_target_lineage(
    attestation: Mapping[str, object],
    *,
    target_lineage: SubstrateForwardRepresentationLineage,
    expected_temporal_n_z: int,
) -> None:
    expected = (
        target_lineage.model_fingerprint.model_id,
        target_lineage.model_fingerprint.weights_sha256,
        target_lineage.runtime_origin,
        target_lineage.readout_kind,
        target_lineage.layer_indices,
        target_lineage.activation_widths,
        expected_temporal_n_z,
    )
    observed_layers = attestation.get("layer_indices")
    observed_widths = attestation.get("activation_widths")
    if not isinstance(observed_layers, (list, tuple)) or not isinstance(
        observed_widths, (list, tuple)
    ):
        raise ValueError("MSC full-runtime attestation geometry is malformed")
    observed = (
        attestation.get("model_id"),
        attestation.get("weights_sha256"),
        attestation.get("runtime_origin"),
        attestation.get("readout_kind"),
        tuple(observed_layers),
        tuple(observed_widths),
        attestation.get("temporal_n_z"),
    )
    if (
        observed != expected
        or attestation.get("volvence_full_stack") is not True
        or attestation.get("raw_text_retained") is not False
        or attestation.get("evaluation_writeback_allowed") is not False
    ):
        raise ValueError(
            "MSC full-runtime context and target substrate lineage differ"
        )


def _lineage_from_payload(value: object) -> SubstrateForwardRepresentationLineage:
    if not isinstance(value, dict):
        raise ValueError("substrate target checkpoint lineage must be an object")
    model = value.get("model_fingerprint")
    if not isinstance(model, dict):
        raise ValueError("substrate target checkpoint model fingerprint is invalid")
    try:
        model_fingerprint = SubstrateFingerprint(
            model_id=str(model["model_id"]),
            version=str(model["version"]),
            weights_sha256=str(model["weights_sha256"]),
        )
        return SubstrateForwardRepresentationLineage(
            schema_version=str(value["schema_version"]),
            snapshot_fingerprint=str(value["snapshot_fingerprint"]),
            model_fingerprint=model_fingerprint,
            runtime_origin=str(value["runtime_origin"]),
            readout_kind=str(value["readout_kind"]),
            layer_indices=tuple(int(item) for item in value["layer_indices"]),
            activation_widths=tuple(
                int(item) for item in value["activation_widths"]
            ),
            representation_dim=int(value["representation_dim"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"substrate target checkpoint lineage lacks {exc.args[0]!r}"
        ) from exc


def _load_or_build_atomic_embeddings(
    *,
    store: PredictionRunCheckpointStore,
    encoder: FrozenContextEncoder,
    texts: tuple[str, ...],
) -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[int, int]],
    dict[str, float],
]:
    np = _require_numpy()
    costs = tuple(encoder.token_cost(text) for text in texts)
    metadata = {
        "schema_version": "msc-atomic-context-checkpoint.v2",
        "encoder_fingerprint": encoder.fingerprint,
        "embedding_dim": encoder.embedding_dim,
        "source_sha256": _source_sha256s(texts),
        "cost_accounting": "per-source-measured-no-hidden-truncation.v1",
        "raw_text_retained": False,
    }
    unit = "contexts/atomic"
    arrays = store.load_arrays(
        unit=unit,
        relative_path="checkpoints/contexts/atomic.npz",
        expected_metadata=metadata,
    )
    if arrays is None:
        vectors, latency_ms = encoder.encode(texts)
        if len(latency_ms) != len(texts):
            raise ValueError("atomic context latency rows are not sample-matched")
        store.save_arrays(
            unit=unit,
            relative_path="checkpoints/contexts/atomic.npz",
            metadata=metadata,
            arrays={
                "vectors": np.asarray(vectors, dtype=np.float64),
                "tokens": np.asarray(
                    tuple(cost[0] for cost in costs), dtype=np.int64
                ),
                "truncated": np.asarray(
                    tuple(cost[1] for cost in costs), dtype=np.int64
                ),
                "latency_ms": np.asarray(latency_ms, dtype=np.float64),
            },
        )
        return (
            dict(zip(texts, vectors, strict=True)),
            dict(zip(texts, costs, strict=True)),
            dict(zip(texts, latency_ms, strict=True)),
        )
    vectors = _array_vectors(
        arrays.get("vectors"),
        rows=len(texts),
        dimension=encoder.embedding_dim,
        field="atomic context vectors",
    )
    tokens = _array_ints(
        arrays.get("tokens"), rows=len(texts), field="atomic context tokens"
    )
    truncated = _array_ints(
        arrays.get("truncated"),
        rows=len(texts),
        field="atomic context truncated tokens",
    )
    latency = _array_floats(
        arrays.get("latency_ms"),
        rows=len(texts),
        field="atomic context latency",
    )
    checkpoint_costs = tuple(zip(tokens, truncated, strict=True))
    if checkpoint_costs != costs:
        raise ValueError("atomic context tokenizer cost drift")
    return (
        dict(zip(texts, vectors, strict=True)),
        dict(zip(texts, checkpoint_costs, strict=True)),
        dict(zip(texts, latency, strict=True)),
    )


def _load_or_build_substrate_targets(
    *,
    store: PredictionRunCheckpointStore,
    texts: tuple[str, ...],
    snapshot: Path,
    model_fingerprint: SubstrateFingerprint,
    device: str,
    activation_width: int,
    layer_indices: tuple[int, ...] | None,
) -> tuple[
    dict[str, tuple[float, ...]],
    SubstrateForwardRepresentationLineage,
    SubstrateFingerprint,
]:
    np = _require_numpy()
    source_hashes = _source_sha256s(texts)
    metadata = {
        "model_fingerprint": asdict(model_fingerprint),
        "requested_activation_width": activation_width,
        "requested_layer_indices": layer_indices,
        "source_sha256": source_hashes,
        "raw_text_retained": False,
    }
    unit = "targets/substrate"
    arrays = store.load_arrays(
        unit=unit,
        relative_path="checkpoints/targets/substrate.npz",
        expected_metadata=metadata,
    )
    if arrays is None:
        publisher = _build_substrate_target_publisher(
            model_id=model_fingerprint.model_id,
            snapshot=snapshot,
            model_fingerprint=model_fingerprint,
            device=device,
            activation_width=activation_width,
            layer_indices=layer_indices,
        )
        embedding_map, target_snapshot = _substrate_target_map(publisher, texts)
        vectors = tuple(embedding_map[text] for text in texts)
        value_hashes = tuple(
            row.values_sha256 for row in target_snapshot.representations
        )
        store.save_arrays(
            unit=unit,
            relative_path="checkpoints/targets/substrate.npz",
            metadata=metadata,
            arrays={
                "vectors": np.asarray(vectors, dtype=np.float64),
                "values_sha256": np.asarray(value_hashes),
                "lineage_json": np.asarray(
                    json.dumps(asdict(target_snapshot.lineage), sort_keys=True)
                ),
            },
        )
        return embedding_map, target_snapshot.lineage, model_fingerprint
    lineage_raw = arrays.get("lineage_json")
    if lineage_raw is None:
        raise ValueError("substrate target checkpoint lacks lineage_json")
    lineage = _lineage_from_payload(json.loads(str(lineage_raw.item())))
    if lineage.model_fingerprint != model_fingerprint:
        raise ValueError("substrate target checkpoint model fingerprint drift")
    vectors = _array_vectors(
        arrays.get("vectors"),
        rows=len(texts),
        dimension=lineage.representation_dim,
        field="substrate target vectors",
    )
    hashes_raw = arrays.get("values_sha256")
    if hashes_raw is None:
        raise ValueError("substrate target checkpoint lacks values_sha256")
    value_hashes = tuple(str(item) for item in hashes_raw.tolist())
    if len(value_hashes) != len(vectors) or any(
        expected != _sha256_vector(vector)
        for expected, vector in zip(value_hashes, vectors, strict=True)
    ):
        raise ValueError("substrate target checkpoint vector hash drift")
    return dict(zip(texts, vectors, strict=True)), lineage, model_fingerprint


def _load_or_build_arm_vectors(
    *,
    store: PredictionRunCheckpointStore,
    arm: str,
    split: str,
    examples: tuple[MSCNextTurnExample, ...],
    encoder: FrozenContextEncoder,
    atomic_embeddings: dict[str, tuple[float, ...]],
    atomic_costs: dict[str, tuple[int, int]],
    atomic_latency_ms: dict[str, float],
    retrieval_count: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    np = _require_numpy()
    metadata = {
        "arm": arm,
        "split": split,
        "encoder_fingerprint": encoder.fingerprint,
        "example_fingerprint": examples_fingerprint(examples),
        "sample_ids": tuple(example.sample_id for example in examples),
        "retrieval_count": retrieval_count,
        "cost_accounting": "per-sample-end-to-end-v2",
        "raw_text_retained": False,
    }
    unit = f"contexts/{arm}/{split}"
    relative_path = f"checkpoints/contexts/{arm}-{split}.npz"
    arrays = store.load_arrays(
        unit=unit,
        relative_path=relative_path,
        expected_metadata=metadata,
    )
    if arrays is None:
        prepared = _prepare_arm_vectors(
            arm=arm,
            examples=examples,
            encoder=encoder,
            atomic_embeddings=atomic_embeddings,
            atomic_costs=atomic_costs,
            atomic_latency_ms=atomic_latency_ms,
            retrieval_count=retrieval_count,
        )
        vectors, tokens, truncated, latency = prepared
        store.save_arrays(
            unit=unit,
            relative_path=relative_path,
            metadata=metadata,
            arrays={
                "vectors": np.asarray(vectors, dtype=np.float64),
                "tokens": np.asarray(tokens, dtype=np.int64),
                "truncated": np.asarray(truncated, dtype=np.int64),
                "latency_ms": np.asarray(latency, dtype=np.float64),
            },
        )
        return prepared
    return (
        _array_vectors(
            arrays.get("vectors"),
            rows=len(examples),
            dimension=encoder.embedding_dim,
            field=f"{arm}/{split} context vectors",
        ),
        _array_ints(arrays.get("tokens"), rows=len(examples), field="tokens"),
        _array_ints(
            arrays.get("truncated"), rows=len(examples), field="truncated"
        ),
        _array_floats(
            arrays.get("latency_ms"), rows=len(examples), field="latency_ms"
        ),
    )


def _retrieved_turns(
    example: MSCNextTurnExample,
    *,
    embeddings: dict[str, tuple[float, ...]],
    count: int,
) -> tuple[MSCDialogueTurn, ...]:
    query = embeddings[example.latest_text]
    candidates = tuple(example.history[:-1])
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: sum(
            left * right
            for left, right in zip(
                query, embeddings[item[1].text], strict=True
            )
        ),
        reverse=True,
    )[:count]
    return tuple(candidates[index] for index, _ in sorted(ranked))


def _volvence_state(
    example: MSCNextTurnExample,
    *,
    embeddings: dict[str, tuple[float, ...]],
    dimension: int,
) -> tuple[float, ...]:
    target_history = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
    )
    previous_sessions = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
        and turn.session_index < example.session_index
    )
    current_session = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
        and turn.session_index == example.session_index
    )
    personas = tuple(embeddings[text] for text in example.personas)
    latest = embeddings[example.latest_text]
    ema = _ema_vectors(target_history, dimension=dimension)
    persona = _mean_vectors(personas, dimension=dimension)
    consolidated = _mean_vectors(previous_sessions, dimension=dimension)
    within_session = _mean_vectors(current_session, dimension=dimension)
    return _unit(
        0.45 * latest[index]
        + 0.25 * ema[index]
        + 0.15 * persona[index]
        + 0.10 * consolidated[index]
        + 0.05 * within_session[index]
        for index in range(dimension)
    )


def _volvence_source_texts(example: MSCNextTurnExample) -> tuple[str, ...]:
    """Sources consumed while reconstructing the bounded relationship state."""

    return (
        *example.personas,
        *(
            turn.text
            for turn in example.history
            if turn.speaker == example.target_speaker
        ),
        example.latest_text,
    )


def _source_cost(
    texts: tuple[str, ...],
    *,
    atomic_costs: dict[str, tuple[int, int]],
    atomic_latency_ms: dict[str, float],
) -> tuple[int, int, float]:
    missing = tuple(
        text
        for text in texts
        if text not in atomic_costs or text not in atomic_latency_ms
    )
    if missing:
        raise ValueError(
            "MSC arm source is absent from the frozen atomic context map: "
            f"{_source_sha256s(missing)!r}"
        )
    return (
        sum(atomic_costs[text][0] for text in texts),
        sum(atomic_costs[text][1] for text in texts),
        sum(atomic_latency_ms[text] for text in texts),
    )


def _prepare_arm_vectors(
    *,
    arm: str,
    examples: tuple[MSCNextTurnExample, ...],
    encoder: FrozenContextEncoder,
    atomic_embeddings: dict[str, tuple[float, ...]],
    atomic_costs: dict[str, tuple[int, int]],
    atomic_latency_ms: dict[str, float],
    retrieval_count: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    if arm == "volvence":
        vectors: list[tuple[float, ...]] = []
        tokens: list[int] = []
        truncated: list[int] = []
        latency: list[float] = []
        for example in examples:
            started = time.perf_counter()
            vectors.append(
                _volvence_state(
                    example,
                    embeddings=atomic_embeddings,
                    dimension=encoder.embedding_dim,
                )
            )
            state_ms = (time.perf_counter() - started) * 1000.0
            source_tokens, source_truncated, source_latency = _source_cost(
                _volvence_source_texts(example),
                atomic_costs=atomic_costs,
                atomic_latency_ms=atomic_latency_ms,
            )
            tokens.append(source_tokens)
            truncated.append(source_truncated)
            latency.append(source_latency + state_ms)
        return (
            tuple(vectors),
            tuple(tokens),
            tuple(truncated),
            tuple(latency),
        )
    preprocessing_tokens = [0 for _ in examples]
    preprocessing_truncated = [0 for _ in examples]
    preprocessing_latency = [0.0 for _ in examples]
    if arm == "stateless":
        texts = tuple(render_stateless_context(example) for example in examples)
    elif arm == "long_context":
        texts = tuple(render_long_context(example) for example in examples)
    elif arm == "summary_retrieval":
        rendered: list[str] = []
        for index, example in enumerate(examples):
            started = time.perf_counter()
            candidates = tuple(example.history[:-1])
            retrieved = _retrieved_turns(
                example,
                embeddings=atomic_embeddings,
                count=retrieval_count,
            )
            retrieval_ms = (time.perf_counter() - started) * 1000.0
            source_tokens, source_truncated, source_latency = _source_cost(
                (example.latest_text, *(turn.text for turn in candidates)),
                atomic_costs=atomic_costs,
                atomic_latency_ms=atomic_latency_ms,
            )
            preprocessing_tokens[index] = source_tokens
            preprocessing_truncated[index] = source_truncated
            preprocessing_latency[index] = source_latency + retrieval_ms
            rendered.append(
                render_summary_retrieval_context(
                    example,
                    retrieved_turns=retrieved,
                )
            )
        texts = tuple(rendered)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    vectors, per_item_ms = encoder.encode(texts)
    if len(per_item_ms) != len(examples):
        raise ValueError(f"{arm} encoder latency rows are not sample-matched")
    costs = tuple(encoder.token_cost(text) for text in texts)
    return (
        vectors,
        tuple(
            cost[0] + preprocessing_tokens[index]
            for index, cost in enumerate(costs)
        ),
        tuple(
            cost[1] + preprocessing_truncated[index]
            for index, cost in enumerate(costs)
        ),
        tuple(
            per_item_ms[index] + preprocessing_latency[index]
            for index in range(len(examples))
        ),
    )


def _batches(indices: tuple[int, ...], batch_size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _make_batch(
    *,
    batch_id: str,
    examples: tuple[MSCNextTurnExample, ...],
    context_vectors: tuple[tuple[float, ...], ...],
    target_vectors: tuple[tuple[float, ...], ...],
    persistence_vectors: tuple[tuple[float, ...], ...],
    indices: tuple[int, ...],
    target_lineage: SubstrateForwardRepresentationLineage,
) -> ForwardRepresentationBatch:
    return ForwardRepresentationBatch(
        batch_id=batch_id,
        sample_ids=tuple(examples[index].sample_id for index in indices),
        context_representations=tuple(context_vectors[index] for index in indices),
        target_representations=tuple(target_vectors[index] for index in indices),
        persistence_representations=tuple(
            persistence_vectors[index] for index in indices
        ),
        history_turns=tuple(examples[index].history_turns for index in indices),
        target_lineage=target_lineage,
    )


def _train_head(
    *,
    train_examples: tuple[MSCNextTurnExample, ...],
    train_contexts: tuple[tuple[float, ...], ...],
    train_targets: tuple[tuple[float, ...], ...],
    train_persistence: tuple[tuple[float, ...], ...],
    n_z: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    target_lineage: SubstrateForwardRepresentationLineage,
) -> PredictionErrorModule:
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=len(train_contexts[0]),
        target_dim=len(train_targets[0]),
        n_z=n_z,
        seed=seed,
        learning_rate=learning_rate,
        device=device,
    )
    rng = random.Random(seed)
    for epoch in range(epochs):
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)
        for batch_index, batch_indices in enumerate(
            _batches(tuple(indices), batch_size)
        ):
            module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"train-e{epoch}-b{batch_index}",
                    examples=train_examples,
                    context_vectors=train_contexts,
                    target_vectors=train_targets,
                    persistence_vectors=train_persistence,
                    indices=batch_indices,
                    target_lineage=target_lineage,
                ),
                update=True,
            )
    return module


def _evaluate_head(
    *,
    module: PredictionErrorModule,
    examples: tuple[MSCNextTurnExample, ...],
    context_vectors: tuple[tuple[float, ...], ...],
    target_vectors: tuple[tuple[float, ...], ...],
    persistence_vectors: tuple[tuple[float, ...], ...],
    batch_size: int,
    target_lineage: SubstrateForwardRepresentationLineage,
) -> tuple[tuple[Any, ...], tuple[float, ...], str]:
    settlements = []
    latencies = []
    fingerprint = ""
    indices = tuple(range(len(examples)))
    for batch_index, batch_indices in enumerate(_batches(indices, batch_size)):
        snapshot = module.process_forward_representation_batch(
            _make_batch(
                batch_id=f"eval-b{batch_index}",
                examples=examples,
                context_vectors=context_vectors,
                target_vectors=target_vectors,
                persistence_vectors=persistence_vectors,
                indices=batch_indices,
                target_lineage=target_lineage,
            ),
            update=False,
        )
        settlements.extend(snapshot.settlements)
        latencies.extend(
            snapshot.elapsed_ms / snapshot.sample_count
            for _ in snapshot.settlements
        )
        fingerprint = snapshot.parameter_fingerprint
    return tuple(settlements), tuple(latencies), fingerprint


def _targets(
    examples: tuple[MSCNextTurnExample, ...],
    embeddings: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(embeddings[example.target_text] for example in examples)


def _persistence(
    examples: tuple[MSCNextTurnExample, ...],
    embeddings: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(embeddings[example.latest_text] for example in examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accept-noncommercial-license", action="store_true", required=True
    )
    parser.add_argument(
        "--encoder", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--context-encoder-mode",
        choices=("substrate", "legacy-sentence"),
        default="substrate",
        help=(
            "substrate is the R3 zero-truncation path; legacy-sentence is "
            "mechanism-pilot-only"
        ),
    )
    parser.add_argument(
        "--volvence-context-mode",
        choices=("full-runtime", "bounded-prototype"),
        default="full-runtime",
        help=(
            "full-runtime is the R4 service/propagate collector; "
            "bounded-prototype remains pilot-only"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--substrate-model", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--substrate-device", default="auto")
    parser.add_argument("--substrate-activation-width", type=int, default=896)
    parser.add_argument("--substrate-layer-indices", type=int, nargs="+")
    parser.add_argument("--runtime-max-new-tokens", type=int, default=16)
    parser.add_argument("--runtime-startup-timeout", type=float, default=600.0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--retrieval-count", type=int, default=4)
    parser.add_argument("--train-dyads", type=int, default=24)
    parser.add_argument("--validation-dyads", type=int, default=12)
    parser.add_argument("--heldout-dyads", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=None,
        help=(
            "Immutable formal preregistration. Without it, even complete data "
            "remains ineligible for a formal thesis claim."
        ),
    )
    parser.add_argument(
        "--emit-run-configuration",
        type=Path,
        default=None,
        help=(
            "Write the normalized base run configuration and exit before "
            "creating an output/checkpoint root. Used to freeze preregistration."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_noncommercial_license:
        raise SystemExit("MSC license acceptance flag is required")
    if (
        args.epochs < 1
        or args.retrieval_count < 1
        or args.substrate_activation_width < 1
        or args.runtime_max_new_tokens < 16
        or args.runtime_startup_timeout <= 0.0
    ):
        raise ValueError(
            "epochs/retrieval-count/substrate width/runtime limits are invalid"
        )
    seeds = tuple(args.seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("research run requires at least three unique non-negative seeds")

    output = args.output.resolve()
    msc_root = args.msc_root.resolve()
    corpus_provenance = msc_root.parent / "DOWNLOAD_PROVENANCE.json"
    if not corpus_provenance.is_file():
        raise FileNotFoundError(
            "MSC artifact capture requires DOWNLOAD_PROVENANCE.json next to the "
            f"extracted corpus root; missing {corpus_provenance}"
        )
    try:
        corpus_provenance_payload = json.loads(
            corpus_provenance.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"MSC corpus provenance is not valid JSON: {corpus_provenance}"
        ) from exc
    if not isinstance(corpus_provenance_payload, dict):
        raise ValueError("MSC corpus provenance root must be a JSON object")
    if corpus_provenance_payload.get("schema_version") != "msc-download-provenance.v1":
        raise ValueError("MSC corpus provenance schema_version is unsupported")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path(__file__).resolve(),
        repository_root / "scripts/companion_test_plan_common.py",
        repository_root / "scripts/freeze_seven_day_execution_root.py",
        repository_root / "scripts/freeze_msc_execution_root.py",
        repository_root / "scripts/msc_prediction_checkpoint.py",
        repository_root / "scripts/run_msc_prediction_test_plan.py",
        repository_root
        / "packages/companion-bench/src/companion_bench/msc_corpus.py",
        repository_root
        / "packages/companion-bench/src/companion_bench/msc_runtime_collection.py",
        repository_root
        / "packages/companion-bench/src/companion_bench/prediction_research.py",
        repository_root
        / "packages/lifeform-core/src/lifeform_core/lifeform.py",
        repository_root
        / "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/__init__.py",
        repository_root
        / "packages/lifeform-evolution/src/lifeform_evolution/seven_day_companion.py",
        repository_root
        / "packages/lifeform-evolution/src/lifeform_evolution/seven_day_process_host.py",
        repository_root
        / "packages/lifeform-expression/src/lifeform_expression/llm_synthesizer.py",
        repository_root
        / "packages/lifeform-service/src/lifeform_service/app.py",
        repository_root
        / "packages/lifeform-service/src/lifeform_service/cli.py",
        repository_root
        / "packages/lifeform-service/src/lifeform_service/companion_evidence_profile.py",
        repository_root
        / "packages/lifeform-service/src/lifeform_service/msc_runtime_collector.py",
        repository_root
        / "packages/lifeform-service/src/lifeform_service/steering_activation.py",
        repository_root
        / "packages/lifeform-service/src/lifeform_service/verticals.py",
        repository_root
        / "packages/vz-runtime/src/volvence_zero/agent/response.py",
        repository_root
        / "packages/vz-runtime/src/volvence_zero/agent/session.py",
        repository_root / "packages/vz-runtime/src/volvence_zero/brain.py",
        repository_root
        / "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
        repository_root
        / "packages/vz-cognition/src/volvence_zero/prediction/error.py",
        repository_root
        / "packages/vz-cognition/src/volvence_zero/prediction/forward_representation.py",
        repository_root
        / "packages/vz-substrate/src/volvence_zero/substrate/forward_representation.py",
        repository_root
        / "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
        repository_root
        / "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
        repository_root
        / "packages/vz-temporal/src/volvence_zero/temporal/interface.py",
    )
    source_hashes = {
        path.relative_to(repository_root).as_posix(): _sha256_file(path)
        for path in source_paths
    }
    layer_indices = (
        tuple(args.substrate_layer_indices)
        if args.substrate_layer_indices is not None
        else None
    )
    if args.volvence_context_mode == "full-runtime" and layer_indices is None:
        raise ValueError(
            "full-runtime Volvence context requires explicit substrate layers"
        )
    substrate_snapshot, frozen_substrate_fingerprint = (
        _substrate_target_contract(model_id=args.substrate_model)
    )
    substrate_context_limit = _substrate_context_limit(substrate_snapshot)
    base_run_configuration = {
        "schema_version": "msc-n-plus-one-resumable-run.v4",
        "msc_root": os.fspath(msc_root),
        "corpus_provenance_sha256": _sha256_file(corpus_provenance),
        "encoder": args.encoder,
        "context_encoder_mode": args.context_encoder_mode,
        "volvence_context_mode": args.volvence_context_mode,
        "device": args.device,
        "substrate_model": args.substrate_model,
        "substrate_device": args.substrate_device,
        "substrate_activation_width": args.substrate_activation_width,
        "substrate_layer_indices": layer_indices,
        "substrate_context_limit": substrate_context_limit,
        "substrate_weights_sha256": (
            frozen_substrate_fingerprint.weights_sha256
        ),
        "runtime_max_new_tokens": args.runtime_max_new_tokens,
        "runtime_startup_timeout": args.runtime_startup_timeout,
        "temporal_capacity_n_z": TEMPORAL_CAPACITY_N_Z,
        "temporal_capacity_fixed_forward_head_n_z": (
            TEMPORAL_CAPACITY_FIXED_FORWARD_HEAD_N_Z
        ),
        "max_seq_length": args.max_seq_length,
        "encoder_batch_size": args.encoder_batch_size,
        "head_batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "retrieval_count": args.retrieval_count,
        "train_dyads": args.train_dyads,
        "validation_dyads": args.validation_dyads,
        "heldout_dyads": args.heldout_dyads,
        "seeds": seeds,
        "source_sha256": source_hashes,
    }
    normalized_base_configuration = json.loads(
        json.dumps(base_run_configuration, sort_keys=True)
    )
    if not isinstance(normalized_base_configuration, dict):
        raise RuntimeError("MSC run configuration normalization failed")
    if args.emit_run_configuration is not None:
        if args.preregistration is not None or args.resume:
            raise ValueError(
                "--emit-run-configuration rejects preregistration and resume"
            )
        _write_json(
            args.emit_run_configuration.resolve(),
            normalized_base_configuration,
        )
        print(args.emit_run_configuration.resolve())
        return 0
    formal_preregistration = None
    formal_preregistration_sha256 = None
    if args.preregistration is not None:
        if (
            args.context_encoder_mode != "substrate"
            or args.volvence_context_mode != "full-runtime"
            or (args.train_dyads, args.validation_dyads, args.heldout_dyads)
            != (1001, 500, 501)
            or len(seeds) < 3
        ):
            raise ValueError(
                "MSC formal preregistration requires substrate/full-runtime, "
                "train=1001, validation=500, heldout=501, and at least 3 seeds"
            )
        (
            formal_preregistration,
            formal_preregistration_sha256,
        ) = _load_formal_preregistration(
            args.preregistration,
            expected_run_configuration=normalized_base_configuration,
        )
    run_configuration = normalized_base_configuration | {
        "formal_preregistration_sha256": formal_preregistration_sha256,
    }
    output_lock = _acquire_output_lock(output)
    store = PredictionRunCheckpointStore(
        output_dir=output,
        configuration=run_configuration,
        resume=args.resume,
        formal_claim_authorized=formal_preregistration is not None,
    )

    split_payload: dict[str, tuple[MSCDyad, ...]] = {}
    audits = {}
    limits = {
        "train": args.train_dyads,
        "validation": args.validation_dyads,
        "heldout": args.heldout_dyads,
    }
    for split in ("train", "validation", "heldout"):
        dyads, audit = load_msc_split(msc_root, split=split, strict=True)
        split_payload[split] = _stable_subset(dyads, limits[split])
        audits[split] = audit
    examples_by_split = {
        split: build_msc_next_turn_examples(dyads)
        for split, dyads in split_payload.items()
    }
    empty_splits = tuple(
        split for split, examples in examples_by_split.items() if not examples
    )
    if empty_splits:
        raise ValueError(f"MSC splits produced no N+1 examples: {empty_splits!r}")
    corpus_index = {
        split: {
            "official_audit": _audit_payload(audits[split], msc_root=msc_root),
            "selected_dyads": len(split_payload[split]),
            "prediction_examples": len(examples_by_split[split]),
            "example_fingerprint": examples_fingerprint(examples_by_split[split]),
        }
        for split in ("train", "validation", "heldout")
    }
    cached_corpus_index = store.load_json(
        unit="corpus/index",
        relative_path="checkpoints/corpus/index.json",
    )
    if cached_corpus_index is None:
        store.save_json(
            unit="corpus/index",
            relative_path="checkpoints/corpus/index.json",
            payload=corpus_index,
        )
    elif cached_corpus_index != corpus_index:
        raise ValueError("MSC corpus/example fingerprint drift on resume")

    target_texts = _target_atomic_texts(examples_by_split)
    target_embeddings, target_lineage, target_model_fingerprint = (
        _load_or_build_substrate_targets(
            store=store,
            texts=target_texts,
            snapshot=substrate_snapshot,
            model_fingerprint=frozen_substrate_fingerprint,
            device=args.substrate_device,
            activation_width=args.substrate_activation_width,
            layer_indices=layer_indices,
        )
    )
    if args.context_encoder_mode == "substrate":
        encoder: FrozenContextEncoder = FrozenSubstrateContextEncoder(
            model_id=args.substrate_model,
            snapshot=substrate_snapshot,
            model_fingerprint=target_model_fingerprint,
            target_lineage=target_lineage,
            device=args.substrate_device,
            activation_width=args.substrate_activation_width,
            layer_indices=layer_indices,
        )
    else:
        encoder = FrozenSentenceEncoder(
            model_id=args.encoder,
            device=args.device,
            max_seq_length=args.max_seq_length,
            batch_size=args.encoder_batch_size,
        )
    atomic_texts = _all_atomic_texts(examples_by_split)
    atomic_embeddings, atomic_costs, atomic_latency_ms = (
        _load_or_build_atomic_embeddings(
            store=store,
            encoder=encoder,
            texts=atomic_texts,
        )
    )
    targets_by_split = {
        split: _targets(examples, target_embeddings)
        for split, examples in examples_by_split.items()
    }
    persistence_by_split = {
        split: _persistence(examples, target_embeddings)
        for split, examples in examples_by_split.items()
    }

    # R3 is completed and attested before any R4 service collection begins.
    # This makes the roadmap's R3 -> R4 -> R5 convergence order observable in
    # the immutable checkpoint journal instead of merely describing all three
    # prerequisites in one terminal report.
    prepared: dict[
        tuple[str, str],
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[float, ...],
        ],
    ] = {}
    for split, examples in examples_by_split.items():
        prepared[("long_context", split)] = _load_or_build_arm_vectors(
            store=store,
            arm="long_context",
            split=split,
            examples=examples,
            encoder=encoder,
            atomic_embeddings=atomic_embeddings,
            atomic_costs=atomic_costs,
            atomic_latency_ms=atomic_latency_ms,
            retrieval_count=args.retrieval_count,
        )
    same_substrate_attestation: SameSubstrateContextAttestation | None = None
    convergence_stage_order: list[str] = []
    if encoder.same_substrate:
        long_context_tokens = tuple(
            token
            for split in ("train", "validation", "heldout")
            for token in prepared[("long_context", split)][1]
        )
        long_context_truncated = tuple(
            count
            for split in ("train", "validation", "heldout")
            for count in prepared[("long_context", split)][2]
        )
        same_substrate_attestation = SameSubstrateContextAttestation(
            context_model_id=encoder.model_id,
            target_model_id=target_model_fingerprint.model_id,
            context_weights_sha256=target_model_fingerprint.weights_sha256,
            target_weights_sha256=target_model_fingerprint.weights_sha256,
            context_readout_kind=target_lineage.readout_kind,
            target_readout_kind=target_lineage.readout_kind,
            context_layer_indices=target_lineage.layer_indices,
            target_layer_indices=target_lineage.layer_indices,
            context_activation_widths=target_lineage.activation_widths,
            target_activation_widths=target_lineage.activation_widths,
            context_limit=encoder.max_seq_length,
            maximum_observed_tokens=max(long_context_tokens),
            truncated_token_count=sum(long_context_truncated),
        )
        if not same_substrate_attestation.passed:
            raise ValueError("MSC R3 same-substrate context attestation failed")
        convergence_stage_order.append("R3")

    # R4 first captures the baseline n_z=3 complete runtime arm. Only after
    # that lineage passes do the remaining R5 capacity interventions run.
    temporal_runtime_prepared: dict[
        int,
        dict[
            str,
            tuple[
                tuple[tuple[float, ...], ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[float, ...],
            ],
        ],
    ] = {}
    temporal_runtime_attestations: dict[int, dict[str, object]] = {}
    if args.volvence_context_mode == "full-runtime":
        if convergence_stage_order != ["R3"]:
            raise ValueError("MSC R4 requires a completed R3 attestation")
        assert layer_indices is not None
        for temporal_n_z in TEMPORAL_CAPACITY_N_Z:
            if temporal_n_z != 3 and convergence_stage_order != ["R3", "R4"]:
                raise RuntimeError("MSC R5 started before R4 completed")
            prepared_capacity, attested_capacity = _prepare_full_runtime_arm(
                store=store,
                repository_root=repository_root,
                split_payload=split_payload,
                examples_by_split=examples_by_split,
                model_fingerprint=frozen_substrate_fingerprint,
                layer_indices=layer_indices,
                activation_width=args.substrate_activation_width,
                context_limit=substrate_context_limit,
                max_new_tokens=args.runtime_max_new_tokens,
                substrate_device=args.substrate_device,
                startup_timeout_s=args.runtime_startup_timeout,
                temporal_n_z=temporal_n_z,
                included_splits=("train", "validation"),
            )
            _validate_full_runtime_target_lineage(
                attested_capacity,
                target_lineage=target_lineage,
                expected_temporal_n_z=temporal_n_z,
            )
            temporal_runtime_prepared[temporal_n_z] = prepared_capacity
            temporal_runtime_attestations[temporal_n_z] = attested_capacity
            if temporal_n_z == 3:
                convergence_stage_order.append("R4")

    temporal_capacity_rows: list[TemporalCapacityObservation] = []
    temporal_capacity_verdict = None
    selected_temporal_n_z: int | None = None
    full_runtime_prepared = None
    full_runtime_attestation = None
    if temporal_runtime_prepared:
        for temporal_n_z in TEMPORAL_CAPACITY_N_Z:
            capacity_contexts = temporal_runtime_prepared[temporal_n_z]
            for seed in seeds:
                unit = f"temporal-capacity/nz-{temporal_n_z}/seed-{seed}"
                relative_path = (
                    "checkpoints/temporal-capacity/"
                    f"nz-{temporal_n_z}-seed-{seed}.json"
                )
                cached = store.load_json(
                    unit=unit,
                    relative_path=relative_path,
                )
                if cached is None:
                    module = _train_head(
                        train_examples=examples_by_split["train"],
                        train_contexts=capacity_contexts["train"][0],
                        train_targets=targets_by_split["train"],
                        train_persistence=persistence_by_split["train"],
                        n_z=TEMPORAL_CAPACITY_FIXED_FORWARD_HEAD_N_Z,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.head_batch_size,
                        learning_rate=args.learning_rate,
                        device=args.device,
                        target_lineage=target_lineage,
                    )
                    settlements, _, _ = _evaluate_head(
                        module=module,
                        examples=examples_by_split["validation"],
                        context_vectors=capacity_contexts["validation"][0],
                        target_vectors=targets_by_split["validation"],
                        persistence_vectors=(
                            persistence_by_split["validation"]
                        ),
                        batch_size=args.head_batch_size,
                        target_lineage=target_lineage,
                    )
                    if not settlements:
                        raise ValueError(
                            "MSC temporal capacity validation produced no settlements"
                        )
                    row = TemporalCapacityObservation(
                        temporal_n_z=temporal_n_z,
                        forward_head_n_z=(
                            TEMPORAL_CAPACITY_FIXED_FORWARD_HEAD_N_Z
                        ),
                        seed=seed,
                        split="validation",
                        mean_cosine_similarity=sum(
                            item.cosine_similarity for item in settlements
                        )
                        / len(settlements),
                        mean_squared_error=sum(
                            item.mean_squared_error for item in settlements
                        )
                        / len(settlements),
                        zero_norm_prediction_count=sum(
                            item.prediction_zero_norm for item in settlements
                        ),
                    )
                    store.save_json(
                        unit=unit,
                        relative_path=relative_path,
                        payload=asdict(row),
                    )
                else:
                    if not isinstance(cached, dict):
                        raise ValueError(
                            f"temporal capacity checkpoint is not an object: {unit}"
                        )
                    row = TemporalCapacityObservation(**cached)
                    if row.temporal_n_z != temporal_n_z or row.seed != seed:
                        raise ValueError(
                            f"temporal capacity checkpoint identity drift: {unit}"
                        )
                temporal_capacity_rows.append(row)
        temporal_capacity_verdict = adjudicate_temporal_capacity_ladder(
            tuple(temporal_capacity_rows),
            complete_train=(
                len(split_payload["train"])
                == audits["train"].conversation_count
            ),
            complete_validation=(
                len(split_payload["validation"])
                == audits["validation"].conversation_count
            ),
        )
        selected_temporal_n_z = (
            temporal_capacity_verdict.chosen_temporal_n_z
        )
        convergence_stage_order.append("R5")
        assert layer_indices is not None
        full_runtime_prepared, full_runtime_attestation = (
            _prepare_full_runtime_arm(
                store=store,
                repository_root=repository_root,
                split_payload=split_payload,
                examples_by_split=examples_by_split,
                model_fingerprint=frozen_substrate_fingerprint,
                layer_indices=layer_indices,
                activation_width=args.substrate_activation_width,
                context_limit=substrate_context_limit,
                max_new_tokens=args.runtime_max_new_tokens,
                substrate_device=args.substrate_device,
                startup_timeout_s=args.runtime_startup_timeout,
                temporal_n_z=selected_temporal_n_z,
                included_splits=("train", "validation", "heldout"),
            )
        )
        _validate_full_runtime_target_lineage(
            full_runtime_attestation,
            target_lineage=target_lineage,
            expected_temporal_n_z=selected_temporal_n_z,
        )

    for arm in PREDICTION_ARMS:
        if arm == "long_context":
            continue
        if arm == "volvence" and full_runtime_prepared is not None:
            for split in ("train", "validation", "heldout"):
                prepared[(arm, split)] = full_runtime_prepared[split]
            continue
        for split, examples in examples_by_split.items():
            prepared[(arm, split)] = _load_or_build_arm_vectors(
                store=store,
                arm=arm,
                split=split,
                examples=examples,
                encoder=encoder,
                atomic_embeddings=atomic_embeddings,
                atomic_costs=atomic_costs,
                atomic_latency_ms=atomic_latency_ms,
                retrieval_count=args.retrieval_count,
            )

    capacity_rows: list[CapacityObservation] = []
    for n_z in (3, 16, 64, 256):
        for seed in seeds:
            unit = f"capacity/nz-{n_z}/seed-{seed}"
            relative_path = f"checkpoints/capacity/nz-{n_z}-seed-{seed}.json"
            cached = store.load_json(unit=unit, relative_path=relative_path)
            if cached is None:
                module = _train_head(
                    train_examples=examples_by_split["train"],
                    train_contexts=prepared[("volvence", "train")][0],
                    train_targets=targets_by_split["train"],
                    train_persistence=persistence_by_split["train"],
                    n_z=n_z,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.head_batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    target_lineage=target_lineage,
                )
                settlements, _, _ = _evaluate_head(
                    module=module,
                    examples=examples_by_split["validation"],
                    context_vectors=prepared[("volvence", "validation")][0],
                    target_vectors=targets_by_split["validation"],
                    persistence_vectors=persistence_by_split["validation"],
                    batch_size=args.head_batch_size,
                    target_lineage=target_lineage,
                )
                if not settlements:
                    raise ValueError(
                        "MSC forward-head capacity validation produced no settlements"
                    )
                row = CapacityObservation(
                    forward_head_n_z=n_z,
                    seed=seed,
                    split="validation",
                    mean_cosine_similarity=sum(
                        item.cosine_similarity for item in settlements
                    )
                    / len(settlements),
                    mean_squared_error=sum(
                        item.mean_squared_error for item in settlements
                    )
                    / len(settlements),
                    zero_norm_prediction_count=sum(
                        item.prediction_zero_norm for item in settlements
                    ),
                )
                store.save_json(
                    unit=unit,
                    relative_path=relative_path,
                    payload=asdict(row),
                )
            else:
                if not isinstance(cached, dict):
                    raise ValueError(f"capacity checkpoint is not an object: {unit}")
                row = CapacityObservation(**cached)
                if row.forward_head_n_z != n_z or row.seed != seed:
                    raise ValueError(f"capacity checkpoint identity drift: {unit}")
            capacity_rows.append(row)
    capacity_verdict = adjudicate_capacity_ladder(
        tuple(capacity_rows),
        complete_train=(
            len(split_payload["train"]) == audits["train"].conversation_count
        ),
        complete_validation=(
            len(split_payload["validation"])
            == audits["validation"].conversation_count
        ),
    )

    observations: list[PredictionObservation] = []
    head_fingerprints: dict[str, str] = {}
    chosen_n_z = capacity_verdict.chosen_forward_head_n_z
    for arm in PREDICTION_ARMS:
        for seed in seeds:
            unit = f"heldout/{arm}/seed-{seed}"
            relative_path = f"checkpoints/heldout/{arm}-seed-{seed}.json"
            cached = store.load_json(unit=unit, relative_path=relative_path)
            if cached is None:
                module = _train_head(
                    train_examples=examples_by_split["train"],
                    train_contexts=prepared[(arm, "train")][0],
                    train_targets=targets_by_split["train"],
                    train_persistence=persistence_by_split["train"],
                    n_z=chosen_n_z,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.head_batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    target_lineage=target_lineage,
                )
                settlements, head_latency, fingerprint = _evaluate_head(
                    module=module,
                    examples=examples_by_split["heldout"],
                    context_vectors=prepared[(arm, "heldout")][0],
                    target_vectors=targets_by_split["heldout"],
                    persistence_vectors=persistence_by_split["heldout"],
                    batch_size=args.head_batch_size,
                    target_lineage=target_lineage,
                )
                tokens = prepared[(arm, "heldout")][1]
                truncated = prepared[(arm, "heldout")][2]
                context_latency = prepared[(arm, "heldout")][3]
                rows = tuple(
                    PredictionObservation(
                        arm=arm,
                        seed=seed,
                        sample_id=example.sample_id,
                        dyad_id=example.dyad_id,
                        session_index=example.session_index,
                        history_turns=example.history_turns,
                        cosine_similarity=settlement.cosine_similarity,
                        mean_squared_error=settlement.mean_squared_error,
                        persistence_cosine_similarity=(
                            settlement.persistence_cosine_similarity
                        ),
                        persistence_mean_squared_error=(
                            settlement.persistence_mean_squared_error
                        ),
                        context_token_count=tokens[index],
                        context_truncated_tokens=truncated[index],
                        latency_ms=context_latency[index] + head_latency[index],
                        prediction_zero_norm=(
                            settlement.prediction_zero_norm
                        ),
                    )
                    for index, (example, settlement) in enumerate(
                        zip(
                            examples_by_split["heldout"],
                            settlements,
                            strict=True,
                        )
                    )
                )
                store.save_json(
                    unit=unit,
                    relative_path=relative_path,
                    payload={
                        "arm": arm,
                        "seed": seed,
                        "selected_forward_head_n_z": chosen_n_z,
                        "target_snapshot_fingerprint": (
                            target_lineage.snapshot_fingerprint
                        ),
                        "head_fingerprint": fingerprint,
                        "observations": tuple(asdict(row) for row in rows),
                    },
                )
            else:
                if not isinstance(cached, dict):
                    raise ValueError(f"heldout checkpoint is not an object: {unit}")
                if (
                    cached.get("arm") != arm
                    or cached.get("seed") != seed
                    or cached.get("selected_forward_head_n_z") != chosen_n_z
                    or cached.get("target_snapshot_fingerprint")
                    != target_lineage.snapshot_fingerprint
                ):
                    raise ValueError(f"heldout checkpoint identity drift: {unit}")
                raw_rows = cached.get("observations")
                fingerprint = cached.get("head_fingerprint")
                if not isinstance(raw_rows, list) or not isinstance(fingerprint, str):
                    raise ValueError(f"heldout checkpoint payload is invalid: {unit}")
                rows = tuple(
                    PredictionObservation(**raw_row)
                    for raw_row in raw_rows
                    if isinstance(raw_row, dict)
                )
                if len(rows) != len(raw_rows):
                    raise ValueError(f"heldout checkpoint row is invalid: {unit}")
                if any(row.arm != arm or row.seed != seed for row in rows):
                    raise ValueError(f"heldout checkpoint row identity drift: {unit}")
            observations.extend(rows)
            head_fingerprints[f"{arm}:seed{seed}"] = fingerprint

    full_runtime_passed = bool(
        full_runtime_attestation is not None
        and full_runtime_attestation.get("volvence_full_stack") is True
        and full_runtime_attestation.get("raw_text_retained") is False
        and full_runtime_attestation.get("evaluation_writeback_allowed") is False
    )
    temporal_capacity_passed = bool(
        temporal_capacity_verdict is not None
        and temporal_capacity_verdict.evidence_level == "formal"
        and temporal_capacity_verdict.capacity_integrity_passed
    )
    prediction_verdict = adjudicate_prediction_experiment(
        tuple(observations),
        heldout_sorted_id_sha256=audits["heldout"].sorted_id_sha256,
        encoder_fingerprint=encoder.fingerprint,
        volvence_full_stack=full_runtime_passed,
        same_substrate_context=(
            same_substrate_attestation is not None
            and same_substrate_attestation.passed
        ),
        temporal_controller_capacity=temporal_capacity_passed,
        formal_preregistered=formal_preregistration is not None,
    )
    formal_experiment_executed = prediction_verdict.evidence_level == "formal"
    manifest = {
        "schema_version": "msc-n-plus-one-research.v4",
        "license_policy": "noncommercial-research-only",
        "formal_preregistration": (
            {
                "path": str(args.preregistration.resolve()),
                "sha256": formal_preregistration_sha256,
                "payload": formal_preregistration,
            }
            if args.preregistration is not None
            else None
        ),
        "corpus": corpus_index,
        "convergence_stage_order": convergence_stage_order,
        "encoder": {
            "role": (
                "R3 same-substrate zero-truncation context readout"
                if encoder.same_substrate
                else "legacy MiniLM context-only mechanism pilot"
            ),
            "model_id": encoder.model_id,
            "fingerprint": encoder.fingerprint,
            "embedding_dim": encoder.embedding_dim,
            "max_seq_length": encoder.max_seq_length,
            "device": encoder.device,
            "truncation_policy": (
                "deny" if encoder.same_substrate else "recency-left-truncation"
            ),
        },
        "same_substrate_context_attestation": (
            asdict(same_substrate_attestation)
            | {"passed": same_substrate_attestation.passed}
            if same_substrate_attestation is not None
            else None
        ),
        "full_runtime_context_attestation": full_runtime_attestation,
        "temporal_capacity_runtime_attestations": (
            temporal_runtime_attestations
        ),
        "target_representation": {
            "owner": "vz-substrate",
            "lineage": asdict(target_lineage),
            "captured_text_count": len(target_texts),
            "raw_text_retained": False,
        },
        "training": {
            "seeds": seeds,
            "forward_head_capacity_n_z": (3, 16, 64, 256),
            "selected_forward_head_n_z": chosen_n_z,
            "temporal_capacity_n_z": TEMPORAL_CAPACITY_N_Z,
            "temporal_capacity_fixed_forward_head_n_z": (
                TEMPORAL_CAPACITY_FIXED_FORWARD_HEAD_N_Z
            ),
            "selected_temporal_n_z": selected_temporal_n_z,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "head_batch_size": args.head_batch_size,
            "head_fingerprints": head_fingerprints,
        },
        "checkpointing": {
            "schema_version": "msc-prediction-run-state.v1",
            "resume_supported": True,
            "resume_requires_exact_configuration_and_source_sha": True,
            "immutable_unit_count": len(store.immutable_file_manifest()),
            "mutable_control_state": "run_state.json",
            "mutable_control_state_is_evidence": False,
            "intermediate_effect_analysis_allowed": False,
            "raw_corpus_text_retained": False,
        },
        "arms": {
            "volvence": (
                "complete service/runtime/propagate context with target-persona "
                "and all dialogue observations; incremental turn and slow-loop "
                "costs are measured"
                if full_runtime_passed
                else "PE-owned bounded recurrent relationship-state prototype; "
                "not a full runtime-stack attestation"
            ),
            "stateless": "persona plus latest partner message",
            "long_context": (
                "complete rendered relationship history on the target-owning "
                "frozen substrate; over-limit input fails and truncation is zero"
                if encoder.same_substrate
                else "legacy MiniLM recency-truncated relationship history"
            ),
            "summary_retrieval": (
                "persona summary plus frozen-encoder top-k extractive memories"
            ),
        },
        "claim_boundary": (
            "R3 same-substrate zero-truncation, R4 complete runtime collection, "
            "and R5 temporal-controller capacity are attested; the only primary "
            "comparison is Volvence versus long_context at session five."
            if temporal_capacity_passed
            else "R3, R4, and a complete R5 temporal capacity ladder are all "
            "required before formal thesis adjudication."
        ),
        "execution_status": {
            "evidence_level": prediction_verdict.evidence_level,
            "thesis_status": prediction_verdict.thesis_exit,
            "formal_experiment_executed": formal_experiment_executed,
            "formal_preregistration_source": (
                _hashed_file(args.preregistration.resolve())
                if args.preregistration is not None
                else None
            ),
            "completed_blockers": (
                "official-msc-corpus",
                "substrate-n-plus-one-target",
                *(("same-substrate-long-context-steelman",) if encoder.same_substrate else ()),
                *(("complete-volvence-runtime-arm",) if full_runtime_passed else ()),
                *(("temporal-controller-capacity-ladder",) if temporal_capacity_passed else ()),
            ),
            "remaining_blockers": (
                *(("same-substrate-long-context-steelman",) if not encoder.same_substrate else ()),
                *(("complete-volvence-runtime-arm",) if not full_runtime_passed else ()),
                *(("temporal-controller-capacity-ladder",) if not temporal_capacity_passed else ()),
            ),
        },
        "run_configuration": run_configuration,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "corpus_provenance.json", corpus_provenance_payload)
    _write_json(output / "capacity_ladder.json", capacity_verdict)
    _write_json(
        output / "temporal_capacity_ladder.json",
        temporal_capacity_verdict,
    )
    _write_json(output / "prediction_verdict.json", prediction_verdict)
    _write_immutable_bytes(
        output / "prediction_observations.jsonl",
        "".join(
            json.dumps(asdict(observation), sort_keys=True) + "\n"
            for observation in observations
        ).encode("utf-8"),
    )
    _write_immutable_bytes(
        output / "report.md",
        "\n".join(
            (
                "# MSC N+1 prediction research report",
                "",
                f"- Evidence level: `{prediction_verdict.evidence_level}`",
                f"- Thesis status: `{prediction_verdict.thesis_exit}`",
                (
                    "- Formal experiment executed: "
                    f"`{str(formal_experiment_executed).lower()}`"
                ),
                (
                    "- N+1 target owner: `vz-substrate` / "
                    f"`{target_lineage.readout_kind}`"
                ),
                (
                    "- N+1 target model: "
                    f"`{target_model_fingerprint.to_short_id()}`"
                ),
                (
                    "- R3 same-substrate context: "
                    f"`{bool(same_substrate_attestation and same_substrate_attestation.passed)}`"
                ),
                f"- R4 complete Volvence runtime: `{full_runtime_passed}`",
                f"- R5 temporal capacity: `{temporal_capacity_passed}`",
                (
                    "- Temporal-capacity exit: "
                    f"`{temporal_capacity_verdict.temporal_capacity_claim_exit}`"
                    if temporal_capacity_verdict is not None
                    else "- Temporal-capacity exit: `NOT_RUN`"
                ),
                f"- Selected temporal_n_z: `{selected_temporal_n_z}`",
                (
                    "- Context truncation policy: "
                    f"`{'deny' if encoder.same_substrate else 'legacy-left'}`"
                ),
                f"- Thesis exit: `{prediction_verdict.thesis_exit}`",
                (
                    "- Forward-head capacity exit: "
                    f"`{capacity_verdict.forward_head_claim_exit}`"
                ),
                f"- Selected PE forward_head_n_z: `{chosen_n_z}`",
                (
                    "- Longest-session cosine advantage vs long context: "
                    f"`{prediction_verdict.longest_quality_advantage:.6f}`"
                ),
                f"- Token ratio: `{prediction_verdict.longest_token_ratio:.6f}`",
                f"- Latency ratio: `{prediction_verdict.longest_latency_ratio:.6f}`",
                (
                    "- Zero-norm heldout predictions: "
                    f"`{prediction_verdict.zero_norm_prediction_count}`"
                ),
                "- Resume checkpoints: `complete`",
                "",
                "The primary test is only Volvence versus the zero-truncation ",
                "long_context control at session five. Stateless and ",
                "summary_retrieval are matched-eligibility controls only.",
                "",
            )
        ).encode("utf-8"),
    )

    current_source_hashes = {
        path.relative_to(repository_root).as_posix(): _sha256_file(path)
        for path in source_paths
    }
    if current_source_hashes != source_hashes:
        raise ValueError("MSC runner source drifted during execution")
    result_names = (
        "manifest.json",
        "corpus_provenance.json",
        "capacity_ladder.json",
        "temporal_capacity_ladder.json",
        "prediction_verdict.json",
        "prediction_observations.jsonl",
        "report.md",
    )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "msc-n-plus-one-artifact.v5",
            "evidence_level": prediction_verdict.evidence_level,
            "thesis_status": prediction_verdict.thesis_exit,
            "formal_experiment_executed": formal_experiment_executed,
            "formal_preregistration_source": (
                _hashed_file(args.preregistration.resolve())
                if args.preregistration is not None
                else None
            ),
            "corpus_provenance_source": _hashed_file(corpus_provenance),
            "source_files": {
                path.relative_to(repository_root).as_posix(): _hashed_file(path)
                for path in source_paths
            },
            "result_files": {
                name: _hashed_file(output / name) for name in result_names
            },
            "checkpoint_files": store.immutable_file_manifest(),
            "mutable_control_files_excluded": {
                "run_state.json": (
                    "resume journal only; not an effect input or evidence result"
                )
            },
            "hash_scope": (
                "artifact_manifest.json is excluded to avoid a recursive self-hash"
            ),
        },
    )
    store.mark_complete(
        formal_claim_allowed=formal_experiment_executed,
    )
    output_lock.close()
    print(output)
    print(prediction_verdict.description)
    print(capacity_verdict.description)
    if temporal_capacity_verdict is not None:
        print(temporal_capacity_verdict.description)
    return 0


def _guarded_main() -> int:
    return guarded_mps_runner_entrypoint(
        main,
        plan_id=MSC_PREDICTION_PLAN_ID,
        argv=tuple(sys.argv[1:]),
    )


if __name__ == "__main__":
    raise SystemExit(_guarded_main())
