"""StylePriorInjector — L1 voice-shape enforcement.

Wraps a :class:`lifeform_domain_figure.FigureStylePrior` artifact
and exposes:

* :meth:`compute_logit_bias` — produces a ``{token_id: bias}`` dict
  suitable for the substrate runtime's ``logit_bias`` parameter on
  HuggingFace generate calls. The injector takes a tokenizer
  encode-adapter callable so it does not depend on any specific
  tokenizer implementation.
* :meth:`render_style_hint_text` — returns a short human-readable
  hint string the prompt planner can append to a system prompt to
  bias the model toward the figure's documented register without
  injecting verbatim corpus excerpts.
* :meth:`sentence_length_targets` — surfaces the percentile-based
  length hints so the planner can budget response length to the
  figure's typical shape.

The injector itself is stateless: a process-wide instance can be
shared across sessions because it holds no per-session mutable
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


_DEFAULT_TOP_WORDS_FOR_BIAS = 80
_DEFAULT_TOP_BIGRAMS_FOR_HINT = 6
_DEFAULT_LOGIT_BIAS_GAIN = 1.5
_DEFAULT_LOGIT_BIAS_CAP = 2.0


@dataclass(frozen=True)
class StylePriorInjectorConfig:
    """Tunables for :class:`StylePriorInjector`."""

    top_words_for_bias: int = _DEFAULT_TOP_WORDS_FOR_BIAS
    top_bigrams_for_hint: int = _DEFAULT_TOP_BIGRAMS_FOR_HINT
    logit_bias_gain: float = _DEFAULT_LOGIT_BIAS_GAIN
    logit_bias_cap: float = _DEFAULT_LOGIT_BIAS_CAP

    def __post_init__(self) -> None:
        if self.top_words_for_bias <= 0:
            raise ValueError(
                f"StylePriorInjectorConfig.top_words_for_bias must be > 0, "
                f"got {self.top_words_for_bias!r}"
            )
        if self.logit_bias_gain < 0.0:
            raise ValueError(
                f"StylePriorInjectorConfig.logit_bias_gain must be >= 0, "
                f"got {self.logit_bias_gain!r}"
            )
        if self.logit_bias_cap <= 0.0:
            raise ValueError(
                f"StylePriorInjectorConfig.logit_bias_cap must be > 0, "
                f"got {self.logit_bias_cap!r}"
            )


class _StylePriorLike(Protocol):
    """Duck-typed style prior contract."""

    figure_id: str
    top_words: tuple[Any, ...]
    top_bigrams: tuple[Any, ...]
    sentence_length_percentiles: tuple[tuple[str, float], ...]
    term_list: tuple[str, ...]


# Tokenizer adapter: a callable that takes a token string and returns
# the integer ids corresponding to the leading sub-tokens. Different
# tokenizers handle prefix whitespace differently so the adapter is
# the caller's choice. Returning an empty tuple is fine — the
# injector will simply skip that token.
TokenIdAdapter = Callable[[str], tuple[int, ...]]


@dataclass(frozen=True)
class StyleHint:
    """Render-side payload of style hints for the prompt planner."""

    figure_id: str
    voice_shape_text: str
    top_term_examples: tuple[str, ...]
    top_bigram_examples: tuple[str, ...]
    sentence_length_targets: tuple[tuple[str, float], ...]


class StylePriorInjector:
    """L1 enforcer that wraps a :class:`FigureStylePrior` artifact."""

    def __init__(
        self,
        style_prior: _StylePriorLike,
        *,
        config: StylePriorInjectorConfig | None = None,
    ) -> None:
        self._style_prior = style_prior
        self._config = config or StylePriorInjectorConfig()

    @property
    def style_prior(self) -> _StylePriorLike:
        return self._style_prior

    @property
    def config(self) -> StylePriorInjectorConfig:
        return self._config

    def compute_logit_bias(
        self, *, encode_token: TokenIdAdapter
    ) -> dict[int, float]:
        """Return a ``{token_id: bias}`` map for the runtime decoder.

        Bias is computed as ``min(cap, gain * frequency_rank)``
        capped by ``logit_bias_cap``. Higher-frequency words get
        larger biases so the decoder is more likely to emit them
        when otherwise indifferent. ``encode_token`` translates each
        word to integer ids; tokens that yield no ids are skipped
        (no silent fallback — the injector simply produces no bias
        for them, which is the correct behaviour).
        """

        biases: dict[int, float] = {}
        gain = self._config.logit_bias_gain
        cap = self._config.logit_bias_cap
        if gain <= 0.0:
            return biases
        top_n = self._config.top_words_for_bias
        top_words = list(self._style_prior.top_words)[:top_n]
        if not top_words:
            return biases
        max_frequency = max((entry.frequency for entry in top_words), default=0.0)
        if max_frequency <= 0.0:
            return biases
        for entry in top_words:
            ids = encode_token(entry.ngram)
            if not ids:
                continue
            scaled = gain * (entry.frequency / max_frequency)
            bias = min(cap, scaled)
            for token_id in ids:
                # Take the max bias if multiple entries map to the same id
                # (subword tokenizers can collide on common stems).
                if token_id in biases:
                    biases[token_id] = max(biases[token_id], bias)
                else:
                    biases[token_id] = bias
        return biases

    def render_style_hint_text(self) -> str:
        """Return a short hint string suitable for a system-prompt section."""

        prior = self._style_prior
        top_terms = self._top_term_strings()[:5]
        top_bigrams = self._top_bigram_strings()[: self._config.top_bigrams_for_hint]
        percentiles = dict(prior.sentence_length_percentiles)
        median = int(percentiles.get("p50", 0.0))
        long_form = int(percentiles.get("p90", 0.0))
        parts = [
            f"Voice prior for figure {prior.figure_id!r}:",
            (
                "  prefer the figure's documented vocabulary; example terms — "
                + (", ".join(top_terms) if top_terms else "(none)")
                + "."
            ),
            (
                "  characteristic phrasings include: "
                + (", ".join(top_bigrams) if top_bigrams else "(none)")
                + "."
            ),
            (
                f"  median sentence length around {median} chars; "
                f"long-form sentences up to ~{long_form} chars."
            ),
        ]
        return "\n".join(parts)

    def style_hint(self) -> StyleHint:
        """Return a structured :class:`StyleHint` for downstream planners."""

        return StyleHint(
            figure_id=self._style_prior.figure_id,
            voice_shape_text=self.render_style_hint_text(),
            top_term_examples=tuple(self._top_term_strings()[:10]),
            top_bigram_examples=tuple(
                self._top_bigram_strings()[: self._config.top_bigrams_for_hint]
            ),
            sentence_length_targets=self._style_prior.sentence_length_percentiles,
        )

    def sentence_length_targets(self) -> tuple[tuple[str, float], ...]:
        return self._style_prior.sentence_length_percentiles

    def _top_term_strings(self) -> list[str]:
        return [entry.ngram for entry in self._style_prior.top_words]

    def _top_bigram_strings(self) -> list[str]:
        return [entry.ngram for entry in self._style_prior.top_bigrams]


__all__ = [
    "StyleHint",
    "StylePriorInjector",
    "StylePriorInjectorConfig",
    "TokenIdAdapter",
]
