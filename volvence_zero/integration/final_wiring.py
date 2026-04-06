from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from volvence_zero.credit import CreditModule, ModificationProposal
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule
from volvence_zero.memory import MemoryModule, MemoryStore
from volvence_zero.reflection import ReflectionEngine, ReflectionModule, WritebackMode
from volvence_zero.regime import RegimeModule
from volvence_zero.runtime import EventRecorder, SlotRegistry, Snapshot, WiringLevel, propagate
from volvence_zero.substrate import SubstrateAdapter, SubstrateModule
from volvence_zero.temporal import TemporalModule, TemporalPolicy


@dataclass(frozen=True)
class FinalRolloutConfig:
    substrate: WiringLevel = WiringLevel.ACTIVE
    memory: WiringLevel = WiringLevel.ACTIVE
    dual_track: WiringLevel = WiringLevel.ACTIVE
    evaluation: WiringLevel = WiringLevel.ACTIVE
    regime: WiringLevel = WiringLevel.ACTIVE
    credit: WiringLevel = WiringLevel.ACTIVE
    reflection: WiringLevel = WiringLevel.SHADOW
    temporal: WiringLevel = WiringLevel.SHADOW
    kill_switches: frozenset[str] = frozenset()

    def level_for(self, module_name: str, default: WiringLevel) -> WiringLevel:
        if module_name in self.kill_switches:
            return WiringLevel.DISABLED
        return {
            "substrate": self.substrate,
            "memory": self.memory,
            "dual_track": self.dual_track,
            "evaluation": self.evaluation,
            "regime": self.regime,
            "credit": self.credit,
            "reflection": self.reflection,
            "temporal": self.temporal,
        }.get(module_name, default)


@dataclass(frozen=True)
class FinalAcceptanceReport:
    passed: bool
    active_slots: tuple[str, ...]
    shadow_slots: tuple[str, ...]
    disabled_slots: tuple[str, ...]
    issues: tuple[str, ...]
    recommendations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class FinalIntegrationResult:
    active_snapshots: dict[str, Snapshot[Any]]
    shadow_snapshots: dict[str, Snapshot[Any]]
    acceptance_report: FinalAcceptanceReport
    event_count: int


def build_final_runtime_modules(
    *,
    config: FinalRolloutConfig,
    substrate_adapter: SubstrateAdapter,
    memory_store: MemoryStore | None = None,
    credit_proposals: tuple[ModificationProposal, ...] = (),
    reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
    temporal_policy: TemporalPolicy | None = None,
    session_id: str = "runtime-session",
    wave_id: str = "wave-0",
) -> list[Any]:
    return [
        SubstrateModule(
            adapter=substrate_adapter,
            wiring_level=config.level_for("substrate", WiringLevel.ACTIVE),
        ),
        MemoryModule(
            store=memory_store or MemoryStore(),
            wiring_level=config.level_for("memory", WiringLevel.SHADOW),
        ),
        DualTrackModule(
            wiring_level=config.level_for("dual_track", WiringLevel.SHADOW),
        ),
        EvaluationModule(
            session_id=session_id,
            wave_id=wave_id,
            wiring_level=config.level_for("evaluation", WiringLevel.ACTIVE),
        ),
        RegimeModule(
            wiring_level=config.level_for("regime", WiringLevel.SHADOW),
        ),
        CreditModule(
            pending_proposals=credit_proposals,
            wiring_level=config.level_for("credit", WiringLevel.SHADOW),
        ),
        ReflectionModule(
            engine=ReflectionEngine(writeback_mode=reflection_mode),
            wiring_level=config.level_for("reflection", WiringLevel.DISABLED),
        ),
        TemporalModule(
            policy=temporal_policy,
            wiring_level=config.level_for("temporal", WiringLevel.SHADOW),
        ),
    ]


async def run_final_wiring_turn(
    *,
    config: FinalRolloutConfig,
    substrate_adapter: SubstrateAdapter,
    memory_store: MemoryStore | None = None,
    credit_proposals: tuple[ModificationProposal, ...] = (),
    reflection_mode: WritebackMode = WritebackMode.PROPOSAL_ONLY,
    temporal_policy: TemporalPolicy | None = None,
    session_id: str = "runtime-session",
    wave_id: str = "wave-0",
) -> FinalIntegrationResult:
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=substrate_adapter,
        memory_store=memory_store,
        credit_proposals=credit_proposals,
        reflection_mode=reflection_mode,
        temporal_policy=temporal_policy,
        session_id=session_id,
        wave_id=wave_id,
    )
    recorder = EventRecorder()
    registry = SlotRegistry()
    shadow_snapshots: dict[str, Snapshot[Any]] = {}
    active_snapshots = await propagate(
        modules,
        registry=registry,
        recorder=recorder,
        shadow_snapshots=shadow_snapshots,
        session_id=session_id,
        wave_id=wave_id,
    )
    acceptance_report = build_acceptance_report(
        config=config,
        active_snapshots=active_snapshots,
        shadow_snapshots=shadow_snapshots,
        recorder=recorder,
    )
    return FinalIntegrationResult(
        active_snapshots=active_snapshots,
        shadow_snapshots=shadow_snapshots,
        acceptance_report=acceptance_report,
        event_count=len(recorder.events),
    )


def build_acceptance_report(
    *,
    config: FinalRolloutConfig,
    active_snapshots: dict[str, Snapshot[Any]],
    shadow_snapshots: dict[str, Snapshot[Any]],
    recorder: EventRecorder,
) -> FinalAcceptanceReport:
    active_slots = tuple(sorted(active_snapshots))
    shadow_slots = tuple(sorted(shadow_snapshots))
    disabled_slots = tuple(
        sorted(
            name
            for name in ("substrate", "memory", "dual_track", "evaluation", "regime", "credit", "reflection", "temporal")
            if config.level_for(name, WiringLevel.DISABLED) is WiringLevel.DISABLED
        )
    )
    issues: list[str] = []
    recommendations: list[str] = []

    expected_active = {
        "substrate",
        "memory",
        "dual_track",
        "evaluation",
        "regime",
        "credit",
    }
    missing_active = sorted(expected_active - set(active_slots))
    if missing_active:
        issues.append(f"Missing active slots: {', '.join(missing_active)}")

    violation_count = sum(1 for event in recorder.events if event.event_type == "contract.violation")
    if violation_count:
        issues.append(f"Observed {violation_count} contract violation events during final wiring.")

    if config.reflection is not WiringLevel.DISABLED and "reflection" not in shadow_slots and "reflection" not in active_slots:
        issues.append("Reflection wiring configured but no reflection snapshot was produced.")

    if config.temporal is not WiringLevel.DISABLED and "temporal_abstraction" not in shadow_slots and "temporal_abstraction" not in active_slots:
        issues.append("Temporal wiring configured but no temporal snapshot was produced.")

    if config.reflection is WiringLevel.ACTIVE:
        recommendations.append("Keep reflection in proposal-only mode until writeback acceptance is proven.")
    if config.temporal is WiringLevel.ACTIVE:
        recommendations.append("Validate temporal active mode against rollout evidence before widening scope.")
    if not recommendations:
        recommendations.append("Core chain is wired; next step is controlled widening via rollout evidence.")

    passed = not issues
    description = (
        f"Final wiring acceptance {'passed' if passed else 'failed'} with "
        f"{len(active_slots)} active slots, {len(shadow_slots)} shadow slots, "
        f"{len(disabled_slots)} disabled slots."
    )
    return FinalAcceptanceReport(
        passed=passed,
        active_slots=active_slots,
        shadow_slots=shadow_slots,
        disabled_slots=disabled_slots,
        issues=tuple(issues),
        recommendations=tuple(recommendations),
        description=description,
    )
