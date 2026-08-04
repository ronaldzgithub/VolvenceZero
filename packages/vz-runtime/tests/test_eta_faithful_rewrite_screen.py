from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from volvence_zero.agent.eta_faithful_rewrite_screen import (  # noqa: E402
    FAITHFUL_ACTION_PROMPT_SUFFIX,
    _build_faithful_trace_bundle,
    run_eta_faithful_rewrite_screen,
)
from volvence_zero.agent.eta_proof_benchmark import (  # noqa: E402
    generate_eta_proof_corpus,
)
from volvence_zero.substrate import (  # noqa: E402
    OpenWeightRuntimeCapture,
    ResidualActivation,
)


class _ToyScorer:
    def __init__(self, *, action_options, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.injection_layer_index = 20
        self.control_norm_cap = 2.0
        self.probe_hidden_norm = 8.0
        self._index = {
            option.action_id: index for index, option in enumerate(action_options)
        }

    def action_index(self, action_id: str) -> int:
        return self._index[action_id]

    def action_nll(
        self,
        *,
        source_texts: tuple[str, ...],
        control_deltas,
        action_indices: tuple[int, ...],
    ):
        del source_texts
        targets = torch.tensor(
            [1.0 if index % 2 == 0 else -1.0 for index in action_indices],
            dtype=torch.float64,
        )
        return (
            control_deltas.to(dtype=torch.float64)[:, 0] - targets
        ).pow(2) + 0.5

    def trainable_parameters(self) -> tuple:
        return ()

    def clear_prefix_cache(self) -> None:
        return None


class _FullWidthRuntime:
    model_id = "faithful-fixture"
    runtime_origin = "fixture"
    fallback_active = False
    is_frozen = True

    def __init__(self, *, width: int) -> None:
        self.width = width
        self.scorer_prompt_suffix: str | None = None

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        digest = hashlib.sha256(source_text.encode("utf-8")).digest()
        activation = tuple(
            (digest[index % len(digest)] / 255.0) * 2.0 - 1.0
            for index in range(self.width)
        )
        return OpenWeightRuntimeCapture(
            token_logits=(0.1, 0.2),
            feature_surface=(),
            residual_activations=(
                ResidualActivation(
                    layer_index=20,
                    activation=activation,
                    step=0,
                ),
            ),
            residual_sequence=(),
            description="full-width faithful ETA fixture",
        )

    def build_steered_action_scorer(
        self, *, action_options, prompt_suffix: str, **_kwargs
    ):
        self.scorer_prompt_suffix = prompt_suffix
        return _ToyScorer(
            action_options=action_options,
            hidden_size=self.width,
        )


def _corpus():
    return generate_eta_proof_corpus(
        seed=20260802,
        objective_count=4,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=4,
        heldout_route_count=2,
        train_lengths=(2,),
        heldout_lengths=(2,),
    )


def test_faithful_trace_bundle_uses_cumulative_full_width_prefixes() -> None:
    corpus = _corpus()
    bundle = _build_faithful_trace_bundle(
        cases=corpus.train_cases[:2],
        corpus=corpus,
        runtime=_FullWidthRuntime(width=6),
        split_label="train",
        injection_layer_index=20,
        residual_width=6,
        progress=None,
    )

    assert len(bundle.traces) == 2
    for trace in bundle.traces:
        assert trace.steps[0].observation_text.startswith("Navigation episode.")
        assert all(
            step.observation_text.endswith(FAITHFUL_ACTION_PROMPT_SUFFIX)
            for step in trace.steps
        )
        assert all(
            len(step.residual_activations) == 1
            and len(step.residual_activations[0].activation) == 6
            for step in trace.steps
        )
        assert len(bundle.boundary_labels[trace.trace_id]) == len(trace.steps) - 1


def test_faithful_screen_runs_new_claim_without_bias_or_production_wiring() -> None:
    corpus = _corpus()
    runtime = _FullWidthRuntime(width=6)
    report = run_eta_faithful_rewrite_screen(
        corpus=corpus,
        runtime=runtime,
        model_source="fixture",
        device="cpu",
        screen_train_route_count=2,
        screen_heldout_route_count=1,
        alpha_grid=(0.03, 0.3, 3.0),
        primary_alpha=0.3,
        seed_schedule=(0,),
        updates_per_run=2,
        learning_rate=0.01,
        n_z=4,
        residual_width=6,
        steering_rank=2,
        scorer_max_length=128,
        max_observed_source_tokens=64,
    )

    assert len(report.points) == 3
    assert report.observation_surface == "stage2-v4-cumulative-causal-prefix"
    assert report.current_observation_mode == "learned-projection"
    assert report.steering_parameterization == "low-rank-multiplicative"
    assert report.free_bias_present is False
    assert report.substrate_trainable_parameter_count == 0
    assert report.production_wiring_changed is False
    assert report.feedback_to_learning is False
    assert runtime.scorer_prompt_suffix == ""
    assert all(point.zero_code_strict_noop for point in report.points)
    assert all(point.input_projection_parameters_changed > 0 for point in report.points)
    assert all(point.low_rank_parameters_changed > 0 for point in report.points)
