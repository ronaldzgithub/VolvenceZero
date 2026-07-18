# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""The baseline columns of the G2 report (torch-free).

Release gate G2 says: the encoder ships only if it significantly beats
(a) the majority-class / global-mean trivial baseline and (b) an LLM
zero-shot labeler on the val split. Both live here so the comparison is
reproducible from the same harness that scores the encoder.

The zero-shot baseline calls any OpenAI-compatible chat endpoint (same
procurement convention as companion-trajgen llm mode). Its predictions
are ``MODEL_PREDICTION``-provenance comparison artifacts — they are never
written back into a training set (guarded by
``tests/contracts/test_companion_encoder_boundaries.py``).
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files

from companion_standard import RelationshipPhase

from companion_encoder.dataset import AnchorExample

_NEUTRAL_REGRESSION = 0.5  # scored value for unparseable zero-shot outputs


@dataclass(frozen=True)
class StatePrediction:
    """One predicted relationship state (any predictor: encoder or baseline)."""

    phase: RelationshipPhase
    trust_level: float
    continuity_level: float
    repair_pressure: float
    confidence: float
    valid: bool = True  # False when a fallback value was substituted

    @property
    def regression_values(self) -> tuple[float, float, float]:
        return (self.trust_level, self.continuity_level, self.repair_pressure)


@dataclass(frozen=True)
class MajorityBaseline:
    """Predicts the training majority phase + training-mean regressions."""

    phase: RelationshipPhase
    trust_level: float
    continuity_level: float
    repair_pressure: float

    @staticmethod
    def fit(train_examples: tuple[AnchorExample, ...]) -> "MajorityBaseline":
        if not train_examples:
            raise ValueError("cannot fit majority baseline on empty training set")
        phase_counts = Counter(example.phase for example in train_examples)
        count = len(train_examples)
        return MajorityBaseline(
            phase=phase_counts.most_common(1)[0][0],
            trust_level=sum(e.trust_level for e in train_examples) / count,
            continuity_level=sum(e.continuity_level for e in train_examples) / count,
            repair_pressure=sum(e.repair_pressure for e in train_examples) / count,
        )

    def predict(self, examples: tuple[AnchorExample, ...]) -> tuple[StatePrediction, ...]:
        prediction = StatePrediction(
            phase=self.phase,
            trust_level=self.trust_level,
            continuity_level=self.continuity_level,
            repair_pressure=self.repair_pressure,
            confidence=1.0,
        )
        return tuple(prediction for _ in examples)


def load_zero_shot_prompt() -> tuple[str, str]:
    """(system, user_template) from the versioned prompt file."""
    text = (
        files("companion_encoder").joinpath("prompts/zero_shot_labeler.md")
        .read_text(encoding="utf-8")
    )
    system_match = re.search(r"## system\n(.*?)\n## user\n", text, re.DOTALL)
    user_match = re.search(r"## user\n(.*)$", text, re.DOTALL)
    if system_match is None or user_match is None:
        raise ValueError("zero_shot_labeler.md missing '## system' / '## user' sections")
    return system_match.group(1).strip(), user_match.group(1).strip()


def parse_zero_shot_answer(raw_text: str) -> StatePrediction:
    """Parse the model's JSON answer; substitute a documented neutral
    fallback (marked ``valid=False``) when the output is unusable, so the
    baseline is scored on every example instead of silently shrinking its
    denominator."""
    fallback = StatePrediction(
        phase=RelationshipPhase.ESTABLISHED,
        trust_level=_NEUTRAL_REGRESSION,
        continuity_level=_NEUTRAL_REGRESSION,
        repair_pressure=_NEUTRAL_REGRESSION,
        confidence=0.0,
        valid=False,
    )
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match is None:
        return fallback
    try:
        data = json.loads(json_match.group(0))
        return StatePrediction(
            phase=RelationshipPhase(data["phase"]),
            trust_level=min(max(float(data["trust_level"]), 0.0), 1.0),
            continuity_level=min(max(float(data["continuity_level"]), 0.0), 1.0),
            repair_pressure=min(max(float(data["repair_pressure"]), 0.0), 1.0),
            confidence=1.0,
        )
    except (KeyError, TypeError, ValueError):
        return fallback


class OpenAICompatibleZeroShotLabeler:
    """LLM zero-shot baseline over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._system_prompt, self._user_template = load_zero_shot_prompt()

    def _complete(self, transcript: str) -> str:
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self._model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {
                            "role": "user",
                            "content": self._user_template.replace(
                                "{transcript}", transcript
                            ),
                        },
                    ],
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    def predict(self, examples: tuple[AnchorExample, ...]) -> tuple[StatePrediction, ...]:
        return tuple(
            parse_zero_shot_answer(self._complete(example.text))
            for example in examples
        )


__all__ = [
    "MajorityBaseline",
    "OpenAICompatibleZeroShotLabeler",
    "StatePrediction",
    "load_zero_shot_prompt",
    "parse_zero_shot_answer",
]
