from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from companion_standard import Snapshot

from lifeform_synthetic_data.canonical import stable_hash
from lifeform_synthetic_data.contracts import (
    CorpusSplit,
    GenerationTier,
)
from lifeform_synthetic_data.llm import (
    BudgetLedger,
    CostLimitExceeded,
    JsonCompletion,
    LLMAuthenticationError,
    RateCard,
    TokenUsage,
)
from lifeform_synthetic_data.pipeline import (
    CorpusGenerationPipeline,
    GenerationRunConfig,
)
from lifeform_synthetic_data.projections import load_master_run
from lifeform_synthetic_data.renderer import render_trajectory
from lifeform_synthetic_data.scenario import load_unified_v1_blueprints
from lifeform_synthetic_data.world import compile_structural_trajectory


class FakeRenderClient:
    def __init__(self, *, model_id: str = "fake-renderer") -> None:
        self._model_id = model_id
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> JsonCompletion:
        del system_prompt
        self.calls += 1
        request = json.loads(user_prompt.split("输入：\n", maxsplit=1)[1])
        slots = [
            {
                "turn_id": slot["turn_id"],
                "text": f"rendered::{slot['role']}::{index}",
            }
            for index, slot in enumerate(request["slots"])
        ]
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        return JsonCompletion(
            model_id=self.model_id,
            request_id=f"fake-request-{self.calls}",
            payload_json=json.dumps(
                {
                    "trajectory_id": request["trajectory_id"],
                    "slots": slots,
                }
            ),
            usage=usage,
            cost_usd=0.00015,
        )


class AuthenticationFailureClient(FakeRenderClient):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> JsonCompletion:
        del system_prompt, user_prompt
        self.calls += 1
        raise LLMAuthenticationError("invalid credentials")


@dataclass(frozen=True)
class FakeSnapshotValue:
    description: str
    z_t: tuple[float, ...]
    beta_t: float


class FakeLifeformSession:
    def __init__(self) -> None:
        self.turn_index = 0

    async def run_turn(self, text: str):
        self.turn_index += 1
        value = FakeSnapshotValue(
            description=f"public state after {self.turn_index}",
            z_t=(0.1, 0.2),
            beta_t=0.25,
        )
        snapshot = Snapshot(
            slot_name="temporal_abstraction",
            owner="fake_temporal_owner",
            version=self.turn_index,
            timestamp_ms=self.turn_index * 1000,
            value=value,
        )
        shadow_snapshot = Snapshot(
            slot_name="temporal_abstraction",
            owner="fake_temporal_shadow_owner",
            version=self.turn_index,
            timestamp_ms=self.turn_index * 1000,
            value=value,
        )
        return SimpleNamespace(
            active_snapshots={"temporal_abstraction": snapshot},
            shadow_snapshots={"temporal_abstraction": shadow_snapshot},
            response=SimpleNamespace(text=f"runtime::{text}"),
        )

    async def advance_tick(self, ticks: int, *, reason: str):
        del ticks, reason
        return ()

    async def end_scene(self, *, reason: str, drain_slow_loop: bool):
        del reason, drain_slow_loop
        return None


class FakeLifeform:
    def create_session(self, *, session_id: str) -> FakeLifeformSession:
        del session_id
        return FakeLifeformSession()


def _config(
    output_root: Path,
    *,
    tier: GenerationTier,
    max_cost_usd: float = 0.0,
) -> GenerationRunConfig:
    return GenerationRunConfig(
        output_root=output_root,
        run_id=f"test-{tier.value}",
        created_at="2026-07-20T00:00:00Z",
        git_sha="test-git-sha",
        generation_tier=tier,
        base_seed=17,
        split_replicates=((CorpusSplit.TRAIN, 1),),
        shard_size=1,
        concurrency=2,
        max_cost_usd=max_cost_usd,
        max_output_tokens=256,
    )


def test_world_compiler_and_renderer_preserve_generator_truth() -> None:
    blueprint = load_unified_v1_blueprints()[0]
    structural = compile_structural_trajectory(
        blueprint,
        replicate_index=0,
        seed=1,
        run_id="test-render",
        created_at="2026-07-20T00:00:00Z",
        git_sha="test",
    )
    client = FakeRenderClient()
    budget = BudgetLedger(
        max_cost_usd=10.0,
        rate_card=RateCard(1.0, 1.0),
    )

    rendered, completion, prompt = render_trajectory(
        structural,
        client=client,
        budget=budget,
        max_output_tokens=256,
    )

    assert rendered.generation_tier is GenerationTier.RENDERED
    assert rendered.truth_frames == structural.truth_frames
    assert rendered.annotations == structural.annotations
    assert rendered.provenance.prompt_hash == prompt.prompt_hash
    assert completion.usage.total_tokens == 150
    assert all(turn.text.startswith("rendered::") for session in rendered.sessions for turn in session.turns)


def test_structural_pipeline_resume_is_byte_identical(tmp_path: Path) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    config = _config(tmp_path, tier=GenerationTier.STRUCTURAL)
    first_pipeline = CorpusGenerationPipeline(
        config=config,
        blueprints=(blueprint,),
    )

    first = first_pipeline.run()
    first_bytes = first.shard_paths[0].read_bytes()
    second_pipeline = CorpusGenerationPipeline(
        config=config,
        blueprints=(blueprint,),
    )
    second = second_pipeline.run()

    assert first.completed_count == 1
    assert second.completed_count == 1
    assert second.resumed_count == 1
    assert second.shard_paths[0].read_bytes() == first_bytes


def test_rendered_pipeline_accounts_cost_and_tokens(tmp_path: Path) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    client = FakeRenderClient()
    rate_card = RateCard(1.0, 1.0)
    pipeline = CorpusGenerationPipeline(
        config=_config(
            tmp_path,
            tier=GenerationTier.RENDERED,
            max_cost_usd=10.0,
        ),
        blueprints=(blueprint,),
        clients=(client,),
        rate_card=rate_card,
    )

    result = pipeline.run()

    assert result.completed_count == 1
    assert result.quarantined_count == 0
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.actual_cost_usd == pytest.approx(0.00015)


def test_preflight_budget_stops_before_external_call(tmp_path: Path) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    client = FakeRenderClient()
    pipeline = CorpusGenerationPipeline(
        config=_config(
            tmp_path,
            tier=GenerationTier.RENDERED,
            max_cost_usd=0.000001,
        ),
        blueprints=(blueprint,),
        clients=(client,),
        rate_card=RateCard(100.0, 100.0),
    )

    with pytest.raises(CostLimitExceeded, match="preflight"):
        pipeline.run()

    assert client.calls == 0


def test_authentication_failure_is_not_quarantined_or_hidden(tmp_path: Path) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    client = AuthenticationFailureClient()
    pipeline = CorpusGenerationPipeline(
        config=_config(
            tmp_path,
            tier=GenerationTier.RENDERED,
            max_cost_usd=10.0,
        ),
        blueprints=(blueprint,),
        clients=(client,),
        rate_card=RateCard(1.0, 1.0),
    )

    with pytest.raises(LLMAuthenticationError):
        pipeline.run()

    assert not (tmp_path / "test-rendered" / "quarantine.jsonl").exists()


def test_live_through_pipeline_captures_public_snapshot_hashes(
    tmp_path: Path,
) -> None:
    blueprint = load_unified_v1_blueprints()[0]
    structural = compile_structural_trajectory(
        blueprint,
        replicate_index=0,
        seed=17,
        run_id="rendered-master",
        created_at="2026-07-20T00:00:00Z",
        git_sha="master-git-sha",
    )
    rendered, _, _ = render_trajectory(
        structural,
        client=FakeRenderClient(),
        budget=BudgetLedger(
            max_cost_usd=10.0,
            rate_card=RateCard(1.0, 1.0),
        ),
        max_output_tokens=256,
    )
    pipeline = CorpusGenerationPipeline(
        config=_config(tmp_path, tier=GenerationTier.LIVE_THROUGH),
        blueprints=(blueprint,),
        lifeform_factory=FakeLifeform,
        source_trajectories=(rendered,),
    )

    result = pipeline.run()

    assert result.completed_count == 1
    assert (result.run_root / "snapshot_sidecars").is_dir()
    trajectory = load_master_run(result.run_root)[0]
    metadata = {item.key: item.value for item in trajectory.metadata}
    assert metadata["source_master_trajectory_hash"] == stable_hash(rendered)
    assert metadata["source_master_run_id"] == "rendered-master"
    assert trajectory.provenance.model_id == "fake-renderer"
    assert all(
        turn.text.startswith("rendered::user")
        for session in trajectory.sessions
        for turn in session.turns
        if turn.role.value == "user"
    )
    assert {frame.wiring_level for frame in trajectory.snapshot_frames} == {
        "active",
        "shadow",
    }
    assert len({frame.snapshot_id for frame in trajectory.snapshot_frames}) == len(trajectory.snapshot_frames)
