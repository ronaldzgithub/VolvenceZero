from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate.adapter import ResidualActivation, SubstrateSnapshot
from volvence_zero.substrate.residual_backend import (
    SubstrateDeltaAdapterLayer,
    SubstrateOnlineFastCheckpoint,
)

if TYPE_CHECKING:
    from volvence_zero.evaluation import EvaluationSnapshot
    from volvence_zero.prediction import PredictionErrorSnapshot


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float, *, limit: float = 0.12) -> float:
    return max(-limit, min(limit, value))


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _checkpoint_hash(checkpoint: SubstrateOnlineFastCheckpoint | None) -> str:
    if checkpoint is None:
        return "none"
    return hashlib.sha256(repr(checkpoint).encode("utf-8")).hexdigest()


def _credit_gate_bindings():
    from volvence_zero.credit.gate import (
        GateDecision,
        ModificationGate,
        ModificationProposal,
        evaluate_gate,
    )

    return GateDecision, ModificationGate, ModificationProposal, evaluate_gate


@dataclass(frozen=True)
class SubstrateFastMemoryCell:
    layer_index: int
    momentum: tuple[float, ...]
    error_trace: tuple[float, ...]
    write_gate: float
    stability: float
    mean_abs_activation: float
    mean_abs_delta: float
    step_count: int
    description: str


@dataclass(frozen=True)
class SubstrateFastMemoryState:
    cells: tuple[SubstrateFastMemoryCell, ...]
    aggregated_signal: tuple[float, ...]
    optimizer_state_norm: float
    parameter_change_rate: float
    step_count: int
    state_hash: str
    source_state_hash: str
    description: str


@dataclass(frozen=True)
class SubstrateSelfModSnapshot:
    recommended: bool
    desired_gate: str
    gate_preview: str
    target: str
    checkpoint_hash: str
    checkpoint: SubstrateOnlineFastCheckpoint | None
    pe_magnitude: float
    optimizer_state_norm: float
    parameter_change_rate: float
    layer_count: int
    update_count: int
    fast_memory_state: SubstrateFastMemoryState | None
    description: str


class SubstrateSelfModModule(RuntimeModule[SubstrateSelfModSnapshot]):
    slot_name = "substrate_self_mod"
    owner = "SubstrateSelfModModule"
    value_type = SubstrateSelfModSnapshot
    dependencies = ("substrate", "evaluation", "prediction_error")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        session_id: str = "runtime-session",
        wave_id: str = "wave-0",
        pe_threshold: float = 0.18,
        external_pe_magnitude: float = 0.0,
        external_pe_reward: float = 0.0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._session_id = session_id
        self._wave_id = wave_id
        self._pe_threshold = max(0.0, pe_threshold)
        self._external_pe_magnitude = max(0.0, external_pe_magnitude)
        self._external_pe_reward = external_pe_reward
        self._update_count = 0
        self._fast_memory_state: SubstrateFastMemoryState | None = None

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[SubstrateSelfModSnapshot]:
        substrate_snapshot = upstream.get("substrate")
        evaluation_snapshot = upstream.get("evaluation")
        prediction_snapshot = upstream.get("prediction_error")
        value = self._build_snapshot_value(
            substrate_snapshot=substrate_snapshot.value if substrate_snapshot is not None else None,
            evaluation_snapshot=evaluation_snapshot.value if evaluation_snapshot is not None else None,
            prediction_snapshot=prediction_snapshot.value if prediction_snapshot is not None else None,
        )
        return self.publish(value)

    async def process_standalone(self, **kwargs: Any) -> Snapshot[SubstrateSelfModSnapshot]:
        return self.publish(
            self._build_snapshot_value(
                substrate_snapshot=kwargs.get("substrate_snapshot"),
                evaluation_snapshot=kwargs.get("evaluation_snapshot"),
                prediction_snapshot=kwargs.get("prediction_snapshot"),
            )
        )

    def _build_snapshot_value(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        evaluation_snapshot: "EvaluationSnapshot | None",
        prediction_snapshot: "PredictionErrorSnapshot | None",
    ) -> SubstrateSelfModSnapshot:
        GateDecision, ModificationGate, ModificationProposal, evaluate_gate = _credit_gate_bindings()
        if substrate_snapshot is None or not substrate_snapshot.residual_activations:
            return SubstrateSelfModSnapshot(
                recommended=False,
                desired_gate=ModificationGate.ONLINE.value,
                gate_preview=GateDecision.BLOCK.value,
                target="substrate.online_fast.delta",
                checkpoint_hash="none",
                checkpoint=None,
                pe_magnitude=0.0,
                optimizer_state_norm=0.0,
                parameter_change_rate=0.0,
                layer_count=0,
                update_count=self._update_count,
                fast_memory_state=self._fast_memory_state,
                description="Substrate self-mod owner skipped because no residual activations were available.",
            )
        prediction_error = getattr(prediction_snapshot, "error", None) if prediction_snapshot is not None else None
        pe_magnitude = prediction_error.magnitude if prediction_error is not None else 0.0
        pe_magnitude = max(pe_magnitude, self._external_pe_magnitude)
        recommended = pe_magnitude >= self._pe_threshold
        checkpoint = None
        optimizer_state_norm = 0.0
        parameter_change_rate = 0.0
        layer_count = 0
        fast_memory_state = self._fast_memory_state
        if recommended:
            (
                fast_memory_state,
                adapter_layers,
                optimizer_state_norm,
                parameter_change_rate,
            ) = self._advance_fast_memory_state(
                residual_activations=substrate_snapshot.residual_activations,
                pe_magnitude=pe_magnitude,
            )
            layer_count = len(adapter_layers)
            recommended = bool(adapter_layers)
            if recommended:
                next_update_count = self._update_count + 1
                checkpoint = SubstrateOnlineFastCheckpoint(
                    checkpoint_id=f"{self._session_id}:{self._wave_id}:substrate-online-fast:{next_update_count}",
                    model_id=substrate_snapshot.model_id,
                    runtime_origin="substrate-self-mod-owner",
                    delta_scale=min(0.18, 0.03 + pe_magnitude * 0.04),
                    update_count=next_update_count,
                    source_wave_id=self._wave_id,
                    source_turn_index=next_update_count,
                    gate=ModificationGate.ONLINE.value,
                    optimizer_state_norm=optimizer_state_norm,
                    parameter_change_rate=parameter_change_rate,
                    description=(
                        f"Online-fast substrate delta proposal for {substrate_snapshot.model_id} "
                        f"layers={len(adapter_layers)} pe={pe_magnitude:.3f}."
                    ),
                    adapter_parameter_count=sum(len(layer.delta_vector) for layer in adapter_layers),
                    adapter_layers=adapter_layers,
                    fast_state_hash=fast_memory_state.state_hash,
                    source_fast_state_hash=fast_memory_state.source_state_hash,
                    fast_memory_signal=fast_memory_state.aggregated_signal,
                    optimizer_state_description=fast_memory_state.description,
                )
                self._update_count = next_update_count
                self._fast_memory_state = fast_memory_state
        gate_preview = GateDecision.ALLOW
        if (
            evaluation_snapshot is not None
            and getattr(evaluation_snapshot, "alerts", None) is not None
            and getattr(evaluation_snapshot, "turn_scores", None) is not None
        ):
            gate_preview = evaluate_gate(
                proposal=ModificationProposal(
                    target="substrate.online_fast.delta",
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="substrate-online-fast:old",
                    new_value_hash=_checkpoint_hash(checkpoint),
                    justification="Turn-time bounded substrate self-modification proposal.",
                ),
                evaluation_snapshot=evaluation_snapshot,
            )
        if not recommended:
            gate_preview = GateDecision.BLOCK if evaluation_snapshot is None else gate_preview
        return SubstrateSelfModSnapshot(
            recommended=recommended,
            desired_gate=ModificationGate.ONLINE.value,
            gate_preview=gate_preview.value,
            target="substrate.online_fast.delta",
            checkpoint_hash=_checkpoint_hash(checkpoint),
            checkpoint=checkpoint,
            pe_magnitude=pe_magnitude,
            optimizer_state_norm=optimizer_state_norm,
            parameter_change_rate=parameter_change_rate,
            layer_count=layer_count,
            update_count=self._update_count,
            fast_memory_state=fast_memory_state,
            description=(
                f"Substrate self-mod proposal recommended={recommended} "
                f"gate_preview={gate_preview.value} pe={pe_magnitude:.3f} "
                f"external_reward={self._external_pe_reward:.3f} "
                f"layers={layer_count} change_rate={parameter_change_rate:.3f}."
            ),
        )

    def _advance_fast_memory_state(
        self,
        *,
        residual_activations: tuple[ResidualActivation, ...],
        pe_magnitude: float,
    ) -> tuple[SubstrateFastMemoryState, tuple[SubstrateDeltaAdapterLayer, ...], float, float]:
        previous_cells = (
            {cell.layer_index: cell for cell in self._fast_memory_state.cells}
            if self._fast_memory_state is not None
            else {}
        )
        adapter_layers: list[SubstrateDeltaAdapterLayer] = []
        cells: list[SubstrateFastMemoryCell] = []
        optimizer_norms: list[float] = []
        parameter_change_rates: list[float] = []
        pe_scale = min(1.0, pe_magnitude / max(self._pe_threshold, 1e-6))
        for activation in residual_activations:
            if not activation.activation:
                continue
            previous_cell = previous_cells.get(activation.layer_index)
            previous_momentum = (
                previous_cell.momentum
                if previous_cell is not None
                else tuple(0.0 for _ in activation.activation)
            )
            previous_error_trace = (
                previous_cell.error_trace
                if previous_cell is not None
                else tuple(0.0 for _ in activation.activation)
            )
            mean_abs_activation = _mean_abs(activation.activation)
            write_gate = _clamp_unit(pe_scale * 0.55 + mean_abs_activation * 0.45)
            updated_momentum = tuple(
                _clamp_signed(
                    previous_momentum[index] * (0.58 + (1.0 - write_gate) * 0.25)
                    + math.tanh(value) * write_gate * 0.16
                )
                for index, value in enumerate(activation.activation)
            )
            updated_error_trace = tuple(
                _clamp_signed(
                    previous_error_trace[index] * 0.82
                    + (updated_momentum[index] - previous_momentum[index]) * 0.75,
                    limit=0.18,
                )
                for index in range(len(updated_momentum))
            )
            mean_abs_delta = _mean_abs(
                tuple(
                    updated_momentum[index] - previous_momentum[index]
                    for index in range(len(updated_momentum))
                )
            )
            stability = _clamp_unit(1.0 - min(abs(mean_abs_delta - mean_abs_activation), 1.0))
            optimizer_norms.append(_mean_abs(updated_momentum))
            parameter_change_rates.append(mean_abs_delta)
            cells.append(
                SubstrateFastMemoryCell(
                    layer_index=activation.layer_index,
                    momentum=updated_momentum,
                    error_trace=updated_error_trace,
                    write_gate=write_gate,
                    stability=stability,
                    mean_abs_activation=mean_abs_activation,
                    mean_abs_delta=mean_abs_delta,
                    step_count=(previous_cell.step_count + 1) if previous_cell is not None else 1,
                    description=(
                        f"Fast-memory layer {activation.layer_index} "
                        f"gate={write_gate:.3f} stability={stability:.3f} delta={mean_abs_delta:.3f}."
                    ),
                )
            )
            adapter_layers.append(
                SubstrateDeltaAdapterLayer(
                    layer_index=activation.layer_index,
                    delta_vector=updated_momentum,
                    mean_abs_delta=mean_abs_delta,
                    description=(
                        f"Online-fast substrate delta for layer {activation.layer_index} "
                        f"derived from explicit fast-memory state."
                    ),
                )
            )
        optimizer_state_norm = sum(optimizer_norms) / max(len(optimizer_norms), 1)
        parameter_change_rate = sum(parameter_change_rates) / max(len(parameter_change_rates), 1)
        aggregated_signal = self._fast_memory_signal(cells)
        source_state_hash = self._fast_memory_state.state_hash if self._fast_memory_state is not None else "root"
        state_payload = (
            tuple(
                (
                    cell.layer_index,
                    cell.write_gate,
                    cell.stability,
                    cell.mean_abs_activation,
                    cell.mean_abs_delta,
                    cell.step_count,
                )
                for cell in cells
            ),
            aggregated_signal,
            round(optimizer_state_norm, 6),
            round(parameter_change_rate, 6),
        )
        state_hash = hashlib.sha256(repr(state_payload).encode("utf-8")).hexdigest()
        fast_memory_state = SubstrateFastMemoryState(
            cells=tuple(cells),
            aggregated_signal=aggregated_signal,
            optimizer_state_norm=optimizer_state_norm,
            parameter_change_rate=parameter_change_rate,
            step_count=self._update_count + 1,
            state_hash=state_hash,
            source_state_hash=source_state_hash,
            description=(
                f"Fast-memory owner advanced {len(cells)} layers with "
                f"optimizer_norm={optimizer_state_norm:.3f}, change_rate={parameter_change_rate:.3f}."
            ),
        )
        return (fast_memory_state, tuple(adapter_layers), optimizer_state_norm, parameter_change_rate)

    def _fast_memory_signal(
        self,
        cells: list[SubstrateFastMemoryCell],
    ) -> tuple[float, ...]:
        if not cells:
            return ()
        return (
            _clamp_unit(sum(cell.write_gate for cell in cells) / len(cells)),
            _clamp_unit(sum(cell.stability for cell in cells) / len(cells)),
            _clamp_unit(sum(cell.mean_abs_activation for cell in cells) / len(cells)),
            _clamp_unit(sum(cell.mean_abs_delta for cell in cells) / len(cells)),
            _clamp_unit(sum(_mean_abs(cell.error_trace) for cell in cells) / len(cells)),
            _clamp_unit(sum(cell.step_count for cell in cells) / max(len(cells) * 10.0, 1.0)),
        )
