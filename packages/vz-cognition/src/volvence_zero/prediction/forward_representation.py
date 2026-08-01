"""Offline N+1 representation forecasting owned by prediction_error.

This is the high-throughput research surface for true forward prediction.  It
does not create a runtime slot or a second mismatch owner: callers give the PE
owner immutable context/target batches, and the owner returns immutable
prediction settlements computed from the exact loss used for learning.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

from volvence_zero.substrate import SubstrateForwardRepresentationLineage


FORWARD_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION = "forward-representation-head.v2"


def _require_finite_vector(
    values: tuple[float, ...], *, field: str, expected_dim: int | None = None
) -> None:
    if not values:
        raise ValueError(f"{field} must be non-empty")
    if expected_dim is not None and len(values) != expected_dim:
        raise ValueError(
            f"{field} dimension mismatch: expected {expected_dim}, got {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} must contain only finite values")


@dataclass(frozen=True)
class ForwardRepresentationBatch:
    batch_id: str
    sample_ids: tuple[str, ...]
    context_representations: tuple[tuple[float, ...], ...]
    target_representations: tuple[tuple[float, ...], ...]
    persistence_representations: tuple[tuple[float, ...], ...]
    history_turns: tuple[int, ...]
    target_lineage: SubstrateForwardRepresentationLineage

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ValueError("forward representation batch_id must be non-empty")
        count = len(self.sample_ids)
        if count < 1:
            raise ValueError("forward representation batch must contain samples")
        if len(set(self.sample_ids)) != count:
            raise ValueError("forward representation sample_ids must be unique")
        groups = (
            self.context_representations,
            self.target_representations,
            self.persistence_representations,
            self.history_turns,
        )
        if any(len(group) != count for group in groups):
            raise ValueError(
                "forward representation batch fields must have one entry per sample"
            )
        input_dim = len(self.context_representations[0])
        target_dim = len(self.target_representations[0])
        if self.target_lineage.representation_dim != target_dim:
            raise ValueError(
                "forward representation target lineage dimension mismatch: "
                f"expected {target_dim}, got {self.target_lineage.representation_dim}"
            )
        for index in range(count):
            _require_finite_vector(
                self.context_representations[index],
                field=f"context_representations[{index}]",
                expected_dim=input_dim,
            )
            _require_finite_vector(
                self.target_representations[index],
                field=f"target_representations[{index}]",
                expected_dim=target_dim,
            )
            _require_finite_vector(
                self.persistence_representations[index],
                field=f"persistence_representations[{index}]",
                expected_dim=target_dim,
            )
            history = self.history_turns[index]
            if isinstance(history, bool) or not isinstance(history, int) or history < 1:
                raise ValueError(
                    f"history_turns[{index}] must be a positive integer; got {history!r}"
                )

    @property
    def input_dim(self) -> int:
        return len(self.context_representations[0])

    @property
    def target_dim(self) -> int:
        return len(self.target_representations[0])


@dataclass(frozen=True)
class ForwardRepresentationSettlement:
    sample_id: str
    history_turns: int
    predicted_representation: tuple[float, ...]
    actual_representation: tuple[float, ...]
    signed_error: tuple[float, ...]
    mean_squared_error: float
    cosine_similarity: float
    persistence_mean_squared_error: float
    persistence_cosine_similarity: float


@dataclass(frozen=True)
class ForwardRepresentationBatchSnapshot:
    batch_id: str
    n_z: int
    sample_count: int
    update_applied: bool
    mean_squared_error: float
    mean_cosine_similarity: float
    persistence_mean_squared_error: float
    persistence_mean_cosine_similarity: float
    mse_improvement_over_persistence: float
    elapsed_ms: float
    parameter_fingerprint: str
    target_lineage: SubstrateForwardRepresentationLineage
    settlements: tuple[ForwardRepresentationSettlement, ...]
    description: str


@dataclass(frozen=True)
class ForwardRepresentationCheckpoint:
    checkpoint_id: str
    input_dim: int
    target_dim: int
    n_z: int
    seed: int
    parameter_values: tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]
    parameter_fingerprint: str
    target_lineage: SubstrateForwardRepresentationLineage
    schema_version: str = FORWARD_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _mse(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "N+1 forward representation research requires torch"
        ) from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("forward representation requested CUDA but it is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("forward representation requested MPS but it is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "forward representation device must be auto/cpu/cuda/mps; "
            f"got {requested!r}"
        )
    return requested


class TorchForwardRepresentationHead:
    """Bounded bottleneck predictor used only through PredictionErrorModule."""

    def __init__(
        self,
        *,
        input_dim: int,
        target_dim: int,
        n_z: int,
        seed: int,
        learning_rate: float,
        device: str,
    ) -> None:
        if input_dim < 1 or target_dim < 1:
            raise ValueError("forward representation dimensions must be positive")
        if n_z < 1:
            raise ValueError("forward representation n_z must be positive")
        if seed < 0:
            raise ValueError("forward representation seed must be non-negative")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("forward representation learning_rate must be within (0, 1]")
        torch = _require_torch()
        resolved_device = _resolve_device(torch, device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, n_z),
                torch.nn.Tanh(),
                torch.nn.Linear(n_z, target_dim),
            )
        self._torch = torch
        self._model = model.to(device=resolved_device, dtype=torch.float32)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=float(learning_rate)
        )
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.n_z = n_z
        self.seed = seed
        self.device = resolved_device
        self._target_lineage: SubstrateForwardRepresentationLineage | None = None

    def process(
        self,
        batch: ForwardRepresentationBatch,
        *,
        update: bool,
    ) -> ForwardRepresentationBatchSnapshot:
        if batch.input_dim != self.input_dim or batch.target_dim != self.target_dim:
            raise ValueError(
                "forward representation batch/head dimension mismatch: "
                f"batch=({batch.input_dim},{batch.target_dim}) "
                f"head=({self.input_dim},{self.target_dim})"
            )
        if self._target_lineage is None:
            self._target_lineage = batch.target_lineage
        elif batch.target_lineage != self._target_lineage:
            raise ValueError(
                "forward representation target lineage changed after head binding"
            )
        torch = self._torch
        started = time.perf_counter()
        contexts = torch.tensor(
            batch.context_representations,
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.tensor(
            batch.target_representations,
            dtype=torch.float32,
            device=self.device,
        )
        self._model.train(update)
        predictions = self._model(contexts)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if update:
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
        prediction_rows = tuple(
            tuple(float(value) for value in row)
            for row in predictions.detach().cpu().tolist()
        )
        settlements: list[ForwardRepresentationSettlement] = []
        for index, prediction in enumerate(prediction_rows):
            target = batch.target_representations[index]
            persistence = batch.persistence_representations[index]
            settlements.append(
                ForwardRepresentationSettlement(
                    sample_id=batch.sample_ids[index],
                    history_turns=batch.history_turns[index],
                    predicted_representation=prediction,
                    actual_representation=target,
                    signed_error=tuple(
                        actual - predicted
                        for actual, predicted in zip(target, prediction, strict=True)
                    ),
                    mean_squared_error=_mse(prediction, target),
                    cosine_similarity=_cosine(prediction, target),
                    persistence_mean_squared_error=_mse(persistence, target),
                    persistence_cosine_similarity=_cosine(persistence, target),
                )
            )
        count = len(settlements)
        mean_mse = sum(item.mean_squared_error for item in settlements) / count
        persistence_mse = (
            sum(item.persistence_mean_squared_error for item in settlements) / count
        )
        mean_cosine = sum(item.cosine_similarity for item in settlements) / count
        persistence_cosine = (
            sum(item.persistence_cosine_similarity for item in settlements) / count
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        fingerprint = self.parameter_fingerprint()
        return ForwardRepresentationBatchSnapshot(
            batch_id=batch.batch_id,
            n_z=self.n_z,
            sample_count=count,
            update_applied=update,
            mean_squared_error=mean_mse,
            mean_cosine_similarity=mean_cosine,
            persistence_mean_squared_error=persistence_mse,
            persistence_mean_cosine_similarity=persistence_cosine,
            mse_improvement_over_persistence=persistence_mse - mean_mse,
            elapsed_ms=elapsed_ms,
            parameter_fingerprint=fingerprint,
            target_lineage=batch.target_lineage,
            settlements=tuple(settlements),
            description=(
                f"N+1 representation batch {batch.batch_id}: n_z={self.n_z}, "
                f"samples={count}, update={update}, mse={mean_mse:.6f}, "
                f"persistence_mse={persistence_mse:.6f}."
            ),
        )

    def _parameter_rows(
        self,
    ) -> tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]:
        rows = []
        for name, parameter in self._model.named_parameters():
            detached = parameter.detach().cpu()
            rows.append(
                (
                    name,
                    tuple(int(value) for value in detached.shape),
                    tuple(float(value) for value in detached.reshape(-1).tolist()),
                )
            )
        return tuple(rows)

    def parameter_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, shape, values in self._parameter_rows():
            digest.update(name.encode("utf-8"))
            digest.update(repr(shape).encode("ascii"))
            digest.update(repr(values).encode("ascii"))
        return digest.hexdigest()

    def export_checkpoint(self, *, checkpoint_id: str) -> ForwardRepresentationCheckpoint:
        if not checkpoint_id.strip():
            raise ValueError("forward representation checkpoint_id must be non-empty")
        if self._target_lineage is None:
            raise RuntimeError(
                "forward representation checkpoint requires a bound target lineage"
            )
        return ForwardRepresentationCheckpoint(
            checkpoint_id=checkpoint_id,
            input_dim=self.input_dim,
            target_dim=self.target_dim,
            n_z=self.n_z,
            seed=self.seed,
            parameter_values=self._parameter_rows(),
            parameter_fingerprint=self.parameter_fingerprint(),
            target_lineage=self._target_lineage,
        )

    def restore_checkpoint(self, checkpoint: ForwardRepresentationCheckpoint) -> None:
        if checkpoint.schema_version != FORWARD_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "forward representation checkpoint schema mismatch: "
                f"{checkpoint.schema_version!r}"
            )
        expected = (self.input_dim, self.target_dim, self.n_z)
        actual = (checkpoint.input_dim, checkpoint.target_dim, checkpoint.n_z)
        if actual != expected:
            raise ValueError(
                f"forward representation checkpoint geometry mismatch: "
                f"expected={expected}, got={actual}"
            )
        if checkpoint.target_lineage.representation_dim != self.target_dim:
            raise ValueError(
                "forward representation checkpoint target lineage dimension mismatch"
            )
        if (
            self._target_lineage is not None
            and self._target_lineage != checkpoint.target_lineage
        ):
            raise ValueError(
                "forward representation checkpoint target lineage mismatch"
            )
        parameter_by_name = dict(self._model.named_parameters())
        if set(parameter_by_name) != {row[0] for row in checkpoint.parameter_values}:
            raise ValueError("forward representation checkpoint parameter names mismatch")
        torch = self._torch
        with torch.no_grad():
            for name, shape, values in checkpoint.parameter_values:
                parameter = parameter_by_name[name]
                if tuple(parameter.shape) != shape or parameter.numel() != len(values):
                    raise ValueError(
                        f"forward representation checkpoint shape mismatch for {name!r}"
                    )
                tensor = torch.tensor(
                    values, dtype=parameter.dtype, device=self.device
                ).reshape(shape)
                parameter.copy_(tensor)
        if self.parameter_fingerprint() != checkpoint.parameter_fingerprint:
            raise ValueError("forward representation checkpoint fingerprint mismatch")
        self._target_lineage = checkpoint.target_lineage


__all__ = (
    "FORWARD_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION",
    "ForwardRepresentationBatch",
    "ForwardRepresentationBatchSnapshot",
    "ForwardRepresentationCheckpoint",
    "ForwardRepresentationSettlement",
)
