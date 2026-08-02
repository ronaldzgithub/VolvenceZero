"""Minute-level surrogate screen for the ETA smooth-posterior rate axis.

The authoritative Gate-1 sweep costs about an hour on the real 0.5B substrate.
Before paying for it, this screen runs the *same* rate-distortion machinery
(same objective, optimizer, alpha sweep, smooth posterior) against a tiny
builtin transformers runtime injected through
``run_eta_rate_distortion_evidence(runtime=..., scorer_factory=...)``. It exists
only to catch two failure modes cheaply:

- **seed variance blow-up**: if the per-alpha rate is bimodal across seeds the
  smooth reparameterization did not fix the optimization noise, and
- **non-monotone rate axis**: if ``spearman(alpha, rate)`` is not clearly
  negative the KL is still not trading information for accuracy.

A screen failure is a hard veto: do not spend real-backend time. A screen pass
is NOT authoritative evidence -- the tiny model has no linguistic competence, so
distortion values are meaningless. Only the rate-axis shape is informative here.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    LocalSubstrateRuntimeMode,
    SubstrateFallbackMode,
    _build_eta_open_weight_runtime,
    generate_eta_proof_corpus,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V2,
    run_eta_rate_distortion_evidence,
)
from volvence_zero.substrate.steered_action_scoring import (
    TransformersSteeredActionScorer,
)
from volvence_zero.temporal.metacontroller_components import (
    POSTERIOR_PARAMETERIZATION_SMOOTH,
)

_SURROGATE_HIDDEN = 16
_SURROGATE_BLOCKS = 4
_SURROGATE_INJECTION_LAYER = 1
_SURROGATE_VOCAB = 256


class _TinyBlock(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(hidden_states))


class _TinyCausalLM(torch.nn.Module):
    """A tiny real-autograd causal LM matching the steered-scorer contract."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(_SURROGATE_VOCAB, _SURROGATE_HIDDEN)
        self.blocks = torch.nn.ModuleList(
            _TinyBlock(_SURROGATE_HIDDEN) for _ in range(_SURROGATE_BLOCKS)
        )
        self.final_norm = torch.nn.LayerNorm(_SURROGATE_HIDDEN)
        self.lm_head = torch.nn.Linear(
            _SURROGATE_HIDDEN, _SURROGATE_VOCAB, bias=False
        )

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.lm_head(hidden[:, indices, :]))


class _WordTokenizer:
    """Whitespace tokenizer with stable ids; enough for the scorer contract."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self._ids: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return _SURROGATE_VOCAB

    def _token_id(self, word: str) -> int:
        if word not in self._ids:
            self._ids[word] = 1 + (len(self._ids) % (_SURROGATE_VOCAB - 1))
        return self._ids[word]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        del add_special_tokens, return_tensors, padding, truncation
        if isinstance(texts, str):
            return {"input_ids": [self._token_id(w) for w in texts.split()]}
        limit = max_length or 32
        sequences = [
            [self._token_id(word) for word in text.split()][:limit] or [1]
            for text in texts
        ]
        width = max(len(sequence) for sequence in sequences)
        return {
            "input_ids": torch.tensor(
                [
                    seq + [self.pad_token_id] * (width - len(seq))
                    for seq in sequences
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(seq) + [0] * (width - len(seq))
                    for seq in sequences
                ],
                dtype=torch.long,
            ),
        }


def _build_surrogate_runtime(*, layer_indices: tuple[int, ...], device: str):
    """A tiny builtin transformers runtime (random weights) for capture only."""

    config = ETAOpenWeightRuntimeConfig(
        runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        fallback_mode=SubstrateFallbackMode.ALLOW_BUILTIN,
        require_real_backend=False,
        layer_indices=layer_indices,
        model_dtype="float32",
        device=device,
    )
    return _build_eta_open_weight_runtime(config)


def _build_surrogate_scorer(*, action_options, joint_training, device):
    """A tiny real-autograd steered scorer decoupled from the capture runtime.

    The builtin capture runtime uses a hashing tokenizer that the steered
    scorer cannot consume, and its distortion is meaningless anyway. This
    lightweight scorer keeps the frozen/joint control seam intact so the sweep
    machinery -- and thus the smooth-posterior rate axis -- is exercised.
    """

    torch.manual_seed(20260802)
    model = _TinyCausalLM().to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return TransformersSteeredActionScorer(
        torch_module=torch,
        model=model,
        tokenizer=_WordTokenizer(),
        block_modules=tuple(model.blocks),
        final_norm_module=model.final_norm,
        injection_layer_index=_SURROGATE_INJECTION_LAYER,
        hidden_size=_SURROGATE_HIDDEN,
        device=device,
        model_id="eta-surrogate-tiny-lm",
        action_options=action_options,
        joint_training=joint_training,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Surrogate screen for the smooth-posterior rate axis; a hard veto "
            "gate before the authoritative real-backend Gate-1 run."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eta_stage1_surrogate_screen_20260802"),
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--train-routes", type=int, default=8)
    parser.add_argument("--heldout-routes", type=int, default=4)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--layer-indices", type=int, nargs="+", default=(1, 2, 3)
    )
    parser.add_argument(
        "--spearman-max",
        type=float,
        default=-0.5,
        help=(
            "Screen passes iff spearman(alpha, rate) <= this. The tiny random "
            "surrogate has meaningless distortion, so this is a gross-failure "
            "gate (catches a flat or increasing rate axis), not a precise "
            "monotonicity test."
        ),
    )
    parser.add_argument(
        "--max-seed-cv",
        type=float,
        default=0.6,
        help=(
            "Screen passes iff the max per-alpha rate CV <= this; catches the "
            "bimodal seed collapse that the legacy clamp produced."
        ),
    )
    args = parser.parse_args()

    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=8,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )
    runtime = _build_surrogate_runtime(
        layer_indices=tuple(args.layer_indices), device=args.device
    )

    def scorer_factory(*, action_options, joint_training):
        return _build_surrogate_scorer(
            action_options=action_options,
            joint_training=joint_training,
            device=args.device,
        )

    report = run_eta_rate_distortion_evidence(
        alpha_grid=tuple(args.alphas),
        seed_schedule=tuple(range(args.seeds)),
        n_z=args.n_z,
        updates_per_run=args.updates,
        arms=("frozen",),
        corpus=corpus,
        observation_protocol=OBSERVATION_PROTOCOL_V2,
        posterior_parameterization=POSTERIOR_PARAMETERIZATION_SMOOTH,
        runtime=runtime,
        scorer_factory=scorer_factory,
    )

    # Per-alpha rate spread across seeds (screening for bimodal collapse).
    rates_by_alpha: dict[float, list[float]] = {}
    for point in report.points:
        if point.arm != "frozen":
            continue
        rates_by_alpha.setdefault(point.alpha, []).append(point.train_rate)
    per_alpha = []
    max_cv = 0.0
    for alpha in sorted(rates_by_alpha):
        rates = rates_by_alpha[alpha]
        mean = statistics.fmean(rates)
        std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        cv = std / mean if mean > 1e-9 else 0.0
        max_cv = max(max_cv, cv)
        per_alpha.append(
            {
                "alpha": alpha,
                "rate_mean": mean,
                "rate_std": std,
                "rate_cv": cv,
                "rates": rates,
            }
        )

    frozen_axis = next(
        (r for r in report.rate_axis_responses if r.arm == "frozen"), None
    )
    if frozen_axis is None:
        raise RuntimeError("surrogate screen produced no frozen rate axis.")
    spearman = frozen_axis.spearman_alpha_rate
    rate_span = frozen_axis.rate_span

    monotone_ok = spearman <= args.spearman_max
    variance_ok = max_cv <= args.max_seed_cv
    screen_pass = monotone_ok and variance_ok

    result = {
        "schema_version": "eta-rate-axis-surrogate-screen.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Surrogate (tiny builtin) screen. Rate-axis shape only; distortion "
            "values are meaningless. Not authoritative."
        ),
        "runtime": {
            "model_id": report.model_id,
            "runtime_origin": report.runtime_origin,
            "fallback_active": report.fallback_active,
            "injection_layer_index": report.injection_layer_index,
        },
        "posterior_parameterization": report.posterior_parameterization,
        "observation_protocol": report.observation_protocol,
        "alpha_grid": list(args.alphas),
        "per_alpha": per_alpha,
        "spearman_alpha_rate": spearman,
        "rate_span": rate_span,
        "max_seed_cv": max_cv,
        "thresholds": {
            "spearman_max": args.spearman_max,
            "max_seed_cv": args.max_seed_cv,
        },
        "monotone_ok": monotone_ok,
        "variance_ok": variance_ok,
        "screen_pass": screen_pass,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "surrogate_screen.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {out / 'surrogate_screen.json'}")
    print(
        f"spearman(alpha,rate)={spearman:+.3f} (<= {args.spearman_max} ? "
        f"{monotone_ok}), rate_span={rate_span:.3f}"
    )
    print(f"max seed CV={max_cv:.3f} (<= {args.max_seed_cv} ? {variance_ok})")
    print(f"SCREEN: {'PASS' if screen_pass else 'FAIL'}")
    raise SystemExit(0 if screen_pass else 1)


if __name__ == "__main__":
    main()
