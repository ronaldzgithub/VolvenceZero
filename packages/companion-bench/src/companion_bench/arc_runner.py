# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Multi-session arc orchestration.

For one ``(scenario, paraphrase_seed)`` pair the runner:

1. Mints a stable ``arc_id`` and per-session ``session_id`` of the
   form ``f"{arc_id}-s{n}"``.
2. Drives N sessions, each with a PRNG-determined turn count inside
   :attr:`ScenarioSpec.session_turn_range`.
3. On every turn, asks the user simulator for the next user message
   (Packet 2), then POSTs to the SUT through :class:`SUTClient`.
4. Always sends ``metadata.session_id`` so OpenAI-compat backends with
   cross-session memory can carry state forward (RFC §7.1).
5. Inserts a `[time_elapsed: D days]` system-context line at session
   boundaries — this is part of the *user* envelope, not the system
   prompt, so it does not contaminate per-turn rubric scoring.
6. Captures every turn's transcript and any telemetry headers the SUT
   returns (e.g., ``x-lifeform-pe-magnitude`` when the SUT happens to
   be us; benign no-op for any other SUT).

The runner is system-agnostic per :data:`packages/companion-bench` license:
it never imports ``volvence_zero.*`` or ``lifeform_*``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import pathlib
import random
import time
import uuid
from typing import Any

from companion_bench.spec import ScenarioSpec
from companion_bench.sut_client import SUTClient, SUTResponse
from companion_bench.user_simulator import (
    GeneratedUserTurn,
    TurnContext,
    UserSimulator,
    UtteranceClient,
)


# ---------------------------------------------------------------------------
# Transcript dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArcTurn:
    """One turn record (user + assistant pair) within an arc."""

    session_index: int
    turn_index: int
    inter_session_gap_days: int
    user_text: str
    assistant_text: str
    fsm_action: str | None
    fsm_payload: str | None
    sut_model_id: str
    sut_prompt_tokens: int | None
    sut_completion_tokens: int | None
    sut_telemetry: dict[str, str]


@dataclasses.dataclass(frozen=True)
class ArcSession:
    """Per-session container."""

    session_index: int
    session_id: str
    inter_session_gap_days: int
    turns: tuple[ArcTurn, ...]


@dataclasses.dataclass(frozen=True)
class ArcRecord:
    """Full record of one ``(scenario, paraphrase_seed)`` run."""

    arc_id: str
    scenario_id: str
    scenario_hash: str
    family: str
    paraphrase_seed: int
    submission_id: str
    sut_model_id: str
    started_at: str
    finished_at: str
    sessions: tuple[ArcSession, ...]
    user_simulator_model: str
    summary_extra: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "arc_id": self.arc_id,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "family": self.family,
            "paraphrase_seed": self.paraphrase_seed,
            "submission_id": self.submission_id,
            "sut_model_id": self.sut_model_id,
            "user_simulator_model": self.user_simulator_model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "sessions": [
                {
                    "session_index": s.session_index,
                    "session_id": s.session_id,
                    "inter_session_gap_days": s.inter_session_gap_days,
                    "turns": [dataclasses.asdict(t) for t in s.turns],
                }
                for s in self.sessions
            ],
            "summary_extra": dict(self.summary_extra),
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArcRunConfig:
    """Knobs the runner exposes to the orchestrator."""

    submission_id: str
    user_simulator_model: str
    sut_max_tokens: int | None = 512
    sut_temperature: float | None = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_arc(
    *,
    spec: ScenarioSpec,
    paraphrase_seed: int,
    sut_client: SUTClient,
    user_backend: UtteranceClient,
    config: ArcRunConfig,
    arc_id: str | None = None,
) -> ArcRecord:
    """Run one arc and return a :class:`ArcRecord` transcript.

    The ``arc_id`` is hash-derived from
    ``(submission_id, scenario_id, paraphrase_seed)`` if not supplied,
    so the same submission running the same scenario seed always uses
    the same session_ids — important for re-run reproducibility (RFC
    §7.3).
    """

    if arc_id is None:
        arc_id = _derive_arc_id(
            submission_id=config.submission_id,
            scenario_id=spec.scenario_id,
            paraphrase_seed=paraphrase_seed,
        )
    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    sim = UserSimulator(
        spec=spec,
        paraphrase_seed=paraphrase_seed,
        backend=user_backend,
    )
    arc_seed_str = f"arc|{spec.scenario_id}|{paraphrase_seed}"
    rng = random.Random(arc_seed_str)

    transcript_messages: list[dict[str, str]] = []
    sessions: list[ArcSession] = []

    for s_idx in range(1, spec.arc_length_sessions + 1):
        if s_idx == 1:
            gap_days = 0
        else:
            gap_days = spec.inter_session_gap_days[s_idx - 2]
            # Inject a user-side time-elapsed marker so the SUT sees
            # session boundaries explicitly. Kept inside the user
            # turn (not the system prompt) per RFC §7.1.
            transcript_messages = []  # session_id changes; reset history view
        session_id = f"{arc_id}-s{s_idx}"
        turn_count = _draw_turn_count(rng=rng, spec=spec)

        turns: list[ArcTurn] = []
        for t_idx in range(1, turn_count + 1):
            ctx = TurnContext(
                session_index=s_idx,
                turn_index=t_idx,
                inter_session_gap_days=gap_days if t_idx == 1 else 0,
            )
            generated = sim.next_turn(ctx)
            user_text = _maybe_add_gap_prefix(generated.text, gap_days, t_idx)
            transcript_messages.append({"role": "user", "content": user_text})

            sut_resp = sut_client.chat(
                messages=transcript_messages.copy(),
                session_id=session_id,
                user_id=None,
                max_tokens=config.sut_max_tokens,
                temperature=config.sut_temperature,
            )
            assistant_text = sut_resp.text
            transcript_messages.append({"role": "assistant", "content": assistant_text})
            sim.append_assistant(assistant_text)

            turns.append(
                ArcTurn(
                    session_index=s_idx,
                    turn_index=t_idx,
                    inter_session_gap_days=ctx.inter_session_gap_days,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    fsm_action=(
                        generated.fsm_step.action if generated.fsm_step else None
                    ),
                    fsm_payload=(
                        generated.fsm_step.payload if generated.fsm_step else None
                    ),
                    sut_model_id=sut_resp.model_id,
                    sut_prompt_tokens=sut_resp.usage_prompt_tokens,
                    sut_completion_tokens=sut_resp.usage_completion_tokens,
                    sut_telemetry=_extract_lscb_telemetry(sut_resp),
                ),
            )

        sessions.append(
            ArcSession(
                session_index=s_idx,
                session_id=session_id,
                inter_session_gap_days=gap_days,
                turns=tuple(turns),
            ),
        )

    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    from companion_bench.spec import scenario_hash as _hash

    return ArcRecord(
        arc_id=arc_id,
        scenario_id=spec.scenario_id,
        scenario_hash=_hash(spec),
        family=spec.family.value,
        paraphrase_seed=paraphrase_seed,
        submission_id=config.submission_id,
        sut_model_id=_first_model_id(sessions),
        started_at=started_at,
        finished_at=finished_at,
        sessions=tuple(sessions),
        user_simulator_model=config.user_simulator_model,
        summary_extra={
            "identity_name": sim.identity.name,
            "identity_occupation": sim.identity.occupation,
            "lexicon_version": sim.identity.lexicon_version,
            "total_turns": sum(len(s.turns) for s in sessions),
        },
    )


def write_arc_record(record: ArcRecord, out_dir: pathlib.Path | str) -> pathlib.Path:
    """Write the arc transcript JSON to disk; returns the file path."""
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / f"{record.arc_id}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record.to_json(), fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_arc_id(*, submission_id: str, scenario_id: str, paraphrase_seed: int) -> str:
    digest = hashlib.sha256(
        f"{submission_id}|{scenario_id}|{paraphrase_seed}".encode("utf-8")
    ).hexdigest()
    return f"arc-{digest[:16]}"


def _draw_turn_count(*, rng: random.Random, spec: ScenarioSpec) -> int:
    lo, hi = spec.session_turn_range
    return rng.randint(lo, hi)


def _maybe_add_gap_prefix(text: str, gap_days: int, turn_index: int) -> str:
    """Prefix the first turn of a non-S1 session with a time-elapsed marker.

    Intentionally surfaced as user-side natural language rather than a
    system metadata field so any SUT (including ones that never look
    at metadata) sees the gap explicitly.
    """
    if turn_index != 1 or gap_days <= 0:
        return text
    return f"[It has been {gap_days} day(s) since we last spoke.] {text}"


def _extract_lscb_telemetry(resp: SUTResponse) -> dict[str, str]:
    """Pull telemetry headers we care about into a flat dict.

    The runner does NOT depend on any of these; they are recorded as
    bonus metadata for downstream ablation analysis. Missing headers
    are simply absent from the result.
    """
    relevant_prefixes = ("x-lifeform-", "x-lscb-", "x-bench-")
    out: dict[str, str] = {}
    for key, value in resp.response_headers.items():
        lk = key.lower()
        if any(lk.startswith(p) for p in relevant_prefixes):
            out[lk] = value
    return out


def _first_model_id(sessions: list[ArcSession]) -> str:
    for s in sessions:
        for t in s.turns:
            return t.sut_model_id
    return "unknown"
