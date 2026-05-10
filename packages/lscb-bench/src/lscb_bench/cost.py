# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.

"""Per-submission cost telemetry (RFC §6.7).

The cost model has four addends per submission:

* Inference on the system under test (SUT).
* Per-turn rubric judge tokens.
* Arc-level judge tokens.
* Pairwise Elo (vs reference systems) tokens.

We track (a) raw token counts via the SUT / judge response usage
fields, and (b) USD cost using a configurable price-per-million-tokens
table loaded from ``lscb_bench/pricing.yaml`` or supplied directly.
The cost telemetry is included in every submission artifact so the
leaderboard can publish per-submission cost alongside score (the RFC
notes the cost as a feature, not a hidden burden).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping

from lscb_bench.arc_runner import ArcRecord


@dataclasses.dataclass(frozen=True)
class TokenPricing:
    """Per-million-token USD prices for one model.

    Both ``input`` and ``output`` are in USD per 1,000,000 tokens.
    Missing model in the price book → cost is reported as null and
    the submission report flags it; we never silently bill at $0.
    """

    model: str
    input_per_mtok: float
    output_per_mtok: float


# Reference price book (2026-Q1). Numbers here are conservative
# upper bounds; production runs override via SubmissionRunner.
_DEFAULT_PRICES: dict[str, TokenPricing] = {
    "anthropic/claude-3.7-sonnet": TokenPricing("anthropic/claude-3.7-sonnet", 3.0, 15.0),
    "anthropic/claude-opus-4.6": TokenPricing("anthropic/claude-opus-4.6", 15.0, 75.0),
    "openai/gpt-5": TokenPricing("openai/gpt-5", 5.0, 15.0),
    "openai/gpt-5-mini": TokenPricing("openai/gpt-5-mini", 0.5, 1.5),
    "google/gemini-3-pro": TokenPricing("google/gemini-3-pro", 5.0, 15.0),
    "deepseek/deepseek-v3": TokenPricing("deepseek/deepseek-v3", 0.27, 1.10),
    "qwen/qwen2.5-72b-instruct": TokenPricing("qwen/qwen2.5-72b-instruct", 0.40, 1.20),
    "meta/llama-3-70b": TokenPricing("meta/llama-3-70b", 0.40, 1.20),
    "lifeform-companion": TokenPricing("lifeform-companion", 0.0, 0.0),
    "lifeform-companion-cold": TokenPricing("lifeform-companion-cold", 0.0, 0.0),
    "lifeform-raw": TokenPricing("lifeform-raw", 0.0, 0.0),
    "fake-sut/echo": TokenPricing("fake-sut/echo", 0.0, 0.0),
    "fake/perturn": TokenPricing("fake/perturn", 0.0, 0.0),
    "fake/arc": TokenPricing("fake/arc", 0.0, 0.0),
}


def default_pricing() -> dict[str, TokenPricing]:
    """Public copy of the default price book (caller can override)."""
    return dict(_DEFAULT_PRICES)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _MutableCounter:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclasses.dataclass(frozen=True)
class CostBreakdown:
    """Read-only cost summary for one submission."""

    sut_calls: int
    sut_input_tokens: int
    sut_output_tokens: int
    sut_usd: float | None
    perturn_calls: int
    perturn_input_tokens: int
    perturn_output_tokens: int
    perturn_usd: float | None
    arc_calls: int
    arc_input_tokens: int
    arc_output_tokens: int
    arc_usd: float | None
    total_usd: float | None
    missing_models: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "sut": {
                "calls": self.sut_calls,
                "input_tokens": self.sut_input_tokens,
                "output_tokens": self.sut_output_tokens,
                "usd": self.sut_usd,
            },
            "perturn_judge": {
                "calls": self.perturn_calls,
                "input_tokens": self.perturn_input_tokens,
                "output_tokens": self.perturn_output_tokens,
                "usd": self.perturn_usd,
            },
            "arc_judge": {
                "calls": self.arc_calls,
                "input_tokens": self.arc_input_tokens,
                "output_tokens": self.arc_output_tokens,
                "usd": self.arc_usd,
            },
            "total_usd": self.total_usd,
            "missing_models": list(self.missing_models),
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Accumulate token counts for one submission and emit a breakdown."""

    def __init__(
        self,
        *,
        pricing: Mapping[str, TokenPricing] | None = None,
    ) -> None:
        self._pricing = dict(pricing) if pricing is not None else default_pricing()
        self._sut: dict[str, _MutableCounter] = {}
        self._perturn: dict[str, _MutableCounter] = {}
        self._arc: dict[str, _MutableCounter] = {}

    def record_sut(
        self,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        c = self._sut.setdefault(model, _MutableCounter())
        c.input_tokens += int(prompt_tokens or 0)
        c.output_tokens += int(completion_tokens or 0)
        c.calls += 1

    def record_perturn_judge(
        self,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        c = self._perturn.setdefault(model, _MutableCounter())
        c.input_tokens += int(prompt_tokens or 0)
        c.output_tokens += int(completion_tokens or 0)
        c.calls += 1

    def record_arc_judge(
        self,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        c = self._arc.setdefault(model, _MutableCounter())
        c.input_tokens += int(prompt_tokens or 0)
        c.output_tokens += int(completion_tokens or 0)
        c.calls += 1

    def record_arc_record(self, arc: ArcRecord) -> None:
        """Convenience: pull SUT token usage straight from an ArcRecord."""
        for s in arc.sessions:
            for t in s.turns:
                self.record_sut(
                    model=t.sut_model_id,
                    prompt_tokens=t.sut_prompt_tokens,
                    completion_tokens=t.sut_completion_tokens,
                )

    def freeze(self) -> CostBreakdown:
        missing: set[str] = set()
        sut_usd, sut_in, sut_out, sut_calls = self._aggregate(self._sut, missing)
        pt_usd, pt_in, pt_out, pt_calls = self._aggregate(self._perturn, missing)
        arc_usd, arc_in, arc_out, arc_calls = self._aggregate(self._arc, missing)

        if any(usd is None for usd in (sut_usd, pt_usd, arc_usd)):
            total = None
        else:
            total = (sut_usd or 0.0) + (pt_usd or 0.0) + (arc_usd or 0.0)
        return CostBreakdown(
            sut_calls=sut_calls,
            sut_input_tokens=sut_in,
            sut_output_tokens=sut_out,
            sut_usd=sut_usd,
            perturn_calls=pt_calls,
            perturn_input_tokens=pt_in,
            perturn_output_tokens=pt_out,
            perturn_usd=pt_usd,
            arc_calls=arc_calls,
            arc_input_tokens=arc_in,
            arc_output_tokens=arc_out,
            arc_usd=arc_usd,
            total_usd=total,
            missing_models=tuple(sorted(missing)),
        )

    def _aggregate(
        self,
        counters: dict[str, _MutableCounter],
        missing: set[str],
    ) -> tuple[float | None, int, int, int]:
        usd: float | None = 0.0
        in_total = 0
        out_total = 0
        calls_total = 0
        for model, c in counters.items():
            in_total += c.input_tokens
            out_total += c.output_tokens
            calls_total += c.calls
            price = self._pricing.get(model)
            if price is None:
                usd = None
                missing.add(model)
                continue
            if usd is not None:
                usd += (c.input_tokens / 1_000_000) * price.input_per_mtok
                usd += (c.output_tokens / 1_000_000) * price.output_per_mtok
        return usd, in_total, out_total, calls_total
