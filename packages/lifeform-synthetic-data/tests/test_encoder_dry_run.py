from __future__ import annotations

import json
from pathlib import Path

from companion_encoder.model import EncoderConfig
from companion_encoder.train import TrainConfig, train

from lifeform_synthetic_data.projections import write_relationship_encoder_layout
from lifeform_synthetic_data.scenario import load_unified_v1_blueprints
from lifeform_synthetic_data.world import compile_structural_trajectory


def test_relationship_projection_trains_tiny_encoder(tmp_path: Path) -> None:
    blueprints = load_unified_v1_blueprints()
    selected = (
        next(item for item in blueprints if item.family == "relationship_continuity"),
        next(item for item in blueprints if item.family == "absence_reengagement"),
    )
    trajectories = tuple(
        compile_structural_trajectory(
            blueprint,
            replicate_index=0,
            seed=index,
            run_id="tiny-encoder-dry-run",
            created_at="2026-07-20T00:00:00Z",
            git_sha="test",
        )
        for index, blueprint in enumerate(selected)
    )
    data_root = tmp_path / "relationship-data"
    manifest_path = write_relationship_encoder_layout(
        trajectories,
        output_root=data_root,
    )

    result = train(
        data_dir=data_root,
        out_dir=tmp_path / "encoder",
        encoder_config=EncoderConfig(
            hidden_dim=8,
            num_layers=1,
            num_attention_heads=2,
            max_input_bytes=128,
            embedding_dim=8,
            dropout=0.0,
        ),
        train_config=TrainConfig(
            epochs=1,
            batch_size=32,
            seed=7,
            device="cpu",
        ),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"] == "whole_scenario_family"
    assert result.train_example_count > 0
    assert result.val_example_count > 0
    assert result.checkpoint_path.is_file()
    assert len(result.history) == 1
