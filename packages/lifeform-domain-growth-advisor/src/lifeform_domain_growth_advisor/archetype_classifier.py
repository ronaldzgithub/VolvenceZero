"""ArchetypeClassifier Protocol + LLMArchetypeClassifier (debt #66).

The growth-advisor vertical encodes 5 reviewed mom archetypes (in
``profiles/cheng_laoshi.py`` ``GrowthAdvisorKnowledgeSeed`` with
``domain == "user_archetype"``):

* ``anxious``     — anxious / catastrophising
* ``comparing``   — comparing to other kids / classes
* ``standard_seeking`` — wants a clear standard / number
* ``venting``     — wants to vent without judgement
* ``product_seeking`` — directly asks for a brand / product

Without an explicit identification mechanism, downstream routing
(boundary triggers, playbook day-N selection, monthly report
``archetype_distribution``) has no signal to dispatch on. Three
candidate paths were reviewed in
``docs/specs/growth-advisor-archetype-detection.md``:

* (a) ``LLMArchetypeClassifier`` — call an LLM every N=3 turns to
  classify; **adopted** for the SHADOW phase.
* (b) keyword / regex matching — **REJECTED** by
  ``.cursor/rules/no-keyword-matching-hacks.mdc``.
* (c) learned metacontroller β_t (debt #44 SYS-1) — long-term path;
  not yet available.

This module defines the Protocol + LLM implementation. The
classifier is deliberately **stateful only at the per-end-user
``ArchetypeStateSnapshot`` level** so the kernel SSOT (R8) is the
classifier itself; downstream consumers read the snapshot.

Refs:

* docs/moving forward/growth-advisor-pilot-packet.md §2.3 G-C
* docs/specs/growth-advisor-archetype-detection.md
* docs/known-debts.md #66
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol


ARCHETYPE_IDS: tuple[str, ...] = (
    "anxious",
    "comparing",
    "standard_seeking",
    "venting",
    "product_seeking",
)


@dataclasses.dataclass(frozen=True)
class ArchetypeClassification:
    """Single classification output from a classifier turn.

    * ``primary`` — most likely archetype id.
    * ``probabilities`` — full distribution over ``ARCHETYPE_IDS``;
      sums to 1.0 (validated in ``__post_init__``).
    * ``confidence`` — how sure the classifier is about ``primary``
      (typically equal to ``probabilities[primary]``).
    * ``rationale`` — short free-text reason from the LLM (for
      audit + debugging; not used for routing).
    """

    primary: str
    probabilities: dict[str, float]
    confidence: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.primary not in ARCHETYPE_IDS:
            raise ValueError(
                f"ArchetypeClassification.primary={self.primary!r} not in "
                f"{ARCHETYPE_IDS}"
            )
        missing = set(ARCHETYPE_IDS) - set(self.probabilities)
        if missing:
            raise ValueError(
                f"ArchetypeClassification.probabilities missing keys: {sorted(missing)}"
            )
        total = sum(self.probabilities.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"ArchetypeClassification.probabilities sums to {total:.4f}; "
                f"expected ~1.0"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"ArchetypeClassification.confidence={self.confidence} "
                f"not in [0, 1]"
            )


@dataclasses.dataclass(frozen=True)
class ArchetypeStateSnapshot:
    """Per-end-user archetype state owned by the classifier.

    Downstream consumers (boundary policy / playbook routing /
    monthly report) read this snapshot; they MUST NOT re-derive
    archetype state themselves (R8 SSOT — classifier is the owner).

    * ``current`` — most recent classification.
    * ``history`` — last K classifications (for hysteresis).
    * ``stability`` — fraction of last K classifications that agree
      with ``current.primary`` (high = stable archetype, low =
      transitioning).
    """

    current: ArchetypeClassification
    history: tuple[ArchetypeClassification, ...]
    stability: float


class ArchetypeClassifier(Protocol):
    """Contract for any archetype classifier (LLM / future β_t)."""

    def classify(
        self,
        *,
        recent_user_turns: Sequence[str],
        end_user_id: str,
    ) -> ArchetypeClassification:
        """Classify the end user's current archetype based on turns."""

    def state_for(self, end_user_id: str) -> ArchetypeStateSnapshot | None:
        """Return the per-user state snapshot, or None if uninitialised."""


@dataclasses.dataclass(frozen=True)
class LLMArchetypeClassifierConfig:
    """Config for ``LLMArchetypeClassifier``.

    * ``llm_provider_label`` — e.g. ``"deepseek-v4"`` (the actual
      provider is injected via ``llm_provider`` callable).
    * ``call_every_n_turns`` — sample frequency (default 3).
    * ``history_window`` — how many past classifications to keep.
    * ``min_turns_to_classify`` — wait until this many user turns
      before first classification (avoids classifying on greeting).
    """

    llm_provider_label: str = "deepseek-v4"
    call_every_n_turns: int = 3
    history_window: int = 10
    min_turns_to_classify: int = 2


class LLMArchetypeClassifier:
    """LLM-driven classifier; SHADOW scaffold (real LLM call pending).

    The constructor takes a callable ``llm_provider(prompt: str) -> str``
    so tests can inject fakes; production wires this through
    ``lifeform-expression`` prompt machinery (per
    ``llm-prompt-centralization.mdc``).

    SHADOW: ``classify()`` returns a deterministic placeholder
    classification (uniform probabilities, primary="anxious") until
    the real LLM call lands as part of #66 ACTIVE. Tests can pass
    a fake provider that returns proper JSON to exercise the
    parsing path.
    """

    def __init__(
        self,
        *,
        llm_provider,  # noqa: ANN001 — callable[[str], str]
        config: LLMArchetypeClassifierConfig | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._config = config or LLMArchetypeClassifierConfig()
        self._state: dict[str, ArchetypeStateSnapshot] = {}
        self._turn_counter: dict[str, int] = {}

    def state_for(self, end_user_id: str) -> ArchetypeStateSnapshot | None:
        return self._state.get(end_user_id)

    def classify(
        self,
        *,
        recent_user_turns: Sequence[str],
        end_user_id: str,
    ) -> ArchetypeClassification:
        # SHADOW: real LLM call lands as #66 ACTIVE; for now return a
        # deterministic placeholder so downstream wiring doesn't
        # NoneType-crash in dev.
        if len(recent_user_turns) < self._config.min_turns_to_classify:
            classification = ArchetypeClassification(
                primary="anxious",
                probabilities={
                    "anxious": 0.20,
                    "comparing": 0.20,
                    "standard_seeking": 0.20,
                    "venting": 0.20,
                    "product_seeking": 0.20,
                },
                confidence=0.20,
                rationale="SHADOW placeholder: too few turns to classify",
            )
        else:
            try:
                raw = self._llm_provider(
                    self._build_prompt(recent_user_turns)
                )
                classification = self._parse_llm_output(raw)
            except (KeyError, ValueError) as exc:
                # Per no-swallow-errors: re-raise with context, not pass.
                raise ValueError(
                    f"LLMArchetypeClassifier failed to parse provider output: {exc}"
                ) from exc

        prior = self._state.get(end_user_id)
        new_history = ((prior.history if prior else ())
                       + (classification,))[-self._config.history_window:]
        if new_history:
            agree = sum(
                1 for c in new_history if c.primary == classification.primary
            )
            stability = agree / len(new_history)
        else:
            stability = 1.0
        self._state[end_user_id] = ArchetypeStateSnapshot(
            current=classification,
            history=new_history,
            stability=stability,
        )
        return classification

    def _build_prompt(self, recent_user_turns: Sequence[str]) -> str:
        # SHADOW: prompt template lives in lifeform-expression
        # (debt #66 G-C SSOT). Until that file lands, this method
        # builds a minimal prompt inline so the classifier exists
        # but explicitly delegates the long-form prompt to the
        # centralised template at #66 ACTIVE.
        return (
            "[SHADOW prompt — production loads from "
            "packages/lifeform-expression/src/lifeform_expression/prompts/"
            "growth_advisor_archetype_classify.txt]\n"
            f"Recent user turns: {list(recent_user_turns)!r}"
        )

    def _parse_llm_output(self, raw: str) -> ArchetypeClassification:
        # SHADOW: parse JSON dict {primary, probabilities, confidence, rationale}.
        import json

        data = json.loads(raw)
        return ArchetypeClassification(
            primary=str(data["primary"]),
            probabilities={
                k: float(v) for k, v in data["probabilities"].items()
            },
            confidence=float(data["confidence"]),
            rationale=str(data.get("rationale", "")),
        )


__all__ = (
    "ARCHETYPE_IDS",
    "ArchetypeClassification",
    "ArchetypeClassifier",
    "ArchetypeStateSnapshot",
    "LLMArchetypeClassifier",
    "LLMArchetypeClassifierConfig",
)
