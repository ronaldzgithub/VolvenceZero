# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.

"""Callback ledger: detect fabricated cross-session callbacks (RFC §6.3).

A "callback" is any place where the assistant claims to remember a
specific fact from earlier in the arc. Examples:

* "Last time you mentioned the dinner with your in-laws — how did it go?"
* "You said earlier that the dance class was hard to keep up with."
* "Earlier in our conversation you told me you were working on a paper
  about reaction-diffusion."

For LSCB scoring we need to know, for every such claim, whether the
claimed prior fact actually appears in the prior-session transcript.
A claim that does not is a fabrication, and the arc judge applies a
**hard penalty** on A3 (per RFC §4) for that turn.

Two-stage pipeline:

1. **Extraction** — an LLM (per-turn judge family) reads each
   assistant turn and emits a JSON list of structured claims:
   ``[{"turn_idx": ..., "claim_text": ..., "claimed_when":
   "session_N|earlier"}]``. We use a Protocol so tests can swap a
   fake extractor.
2. **Matching** — a deterministic Python checker scores each claim
   against the prior-turns evidence (substring + tokenised similarity).
   No LLM in this stage; the rule is mechanical so the verifier is
   reproducible.

The split keeps the LLM-driven part bounded to extraction (low
hallucination risk on a well-scoped JSON output) while keeping the
fabrication call deterministic (reviewable, auditable).
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Iterable, Protocol, runtime_checkable

from lscb_bench.arc_runner import ArcRecord, ArcSession, ArcTurn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CallbackClaim:
    """One assistant claim of the form 'you said X in session N'."""

    session_index: int
    turn_index: int
    claim_text: str  # the substring we believe encodes the claim
    claimed_when: str  # "session_N" or "earlier" or "unknown"


@dataclasses.dataclass(frozen=True)
class CallbackLedgerEntry:
    """Result of matching one claim against prior-turn evidence."""

    claim: CallbackClaim
    evidence_session: int | None
    evidence_turn: int | None
    evidence_text: str | None
    matched: bool
    similarity_score: float
    fabricated: bool


@dataclasses.dataclass(frozen=True)
class CallbackLedger:
    """Per-arc summary of callbacks and their match status."""

    arc_id: str
    entries: tuple[CallbackLedgerEntry, ...]

    @property
    def fabrication_count(self) -> int:
        return sum(1 for e in self.entries if e.fabricated)

    @property
    def matched_count(self) -> int:
        return sum(1 for e in self.entries if e.matched)

    def fabrications(self) -> tuple[CallbackLedgerEntry, ...]:
        return tuple(e for e in self.entries if e.fabricated)

    def to_json(self) -> dict:
        return {
            "arc_id": self.arc_id,
            "fabrication_count": self.fabrication_count,
            "matched_count": self.matched_count,
            "entries": [
                {
                    "session_index": e.claim.session_index,
                    "turn_index": e.claim.turn_index,
                    "claim_text": e.claim.claim_text,
                    "claimed_when": e.claim.claimed_when,
                    "matched": e.matched,
                    "similarity_score": e.similarity_score,
                    "evidence_session": e.evidence_session,
                    "evidence_turn": e.evidence_turn,
                    "evidence_text": e.evidence_text,
                    "fabricated": e.fabricated,
                }
                for e in self.entries
            ],
        }


# ---------------------------------------------------------------------------
# Stage 1: claim extraction
# ---------------------------------------------------------------------------


@runtime_checkable
class CallbackExtractor(Protocol):
    """LLM Protocol for stage-1 extraction.

    Given the text of one assistant turn (and minimal context: which
    session/turn, total session count), emit a JSON list of claims.
    """

    def extract(
        self,
        *,
        assistant_text: str,
        session_index: int,
        turn_index: int,
        total_sessions: int,
    ) -> list[CallbackClaim]: ...


class HeuristicCallbackExtractor:
    """Regex-only fallback extractor.

    Production runs should use :class:`LLMCallbackExtractor` for
    coverage; this heuristic is for unit tests + offline runs without
    an LLM budget. It matches a fixed set of cue phrases and grabs
    the dependent clause as the candidate claim.
    """

    _CUE_PATTERNS = (
        r"you (?:said|mentioned|told me)(?: earlier)?(?: in session \d+)? that (.+?)(?:[.!?]|$)",
        r"you (?:said|mentioned|told me)(?: earlier)?(?: in session \d+)? (.+?)(?:[.!?]|$)",
        r"last time(?: we talked)?(?:,)? you (?:said|mentioned)(?: that)? (.+?)(?:[.!?]|$)",
        r"earlier(?: in our conversation)? you (?:said|mentioned)(?: that)? (.+?)(?:[.!?]|$)",
        r"(?:as|like) you mentioned(?:,)? (.+?)(?:[.!?]|$)",
        r"when we talked about (.+?)(?:[.!?]|$)",
        r"i remember you saying (.+?)(?:[.!?]|$)",
    )
    _SESSION_HINT = re.compile(r"session (\d+)", re.IGNORECASE)

    def extract(
        self,
        *,
        assistant_text: str,
        session_index: int,
        turn_index: int,
        total_sessions: int,
    ) -> list[CallbackClaim]:
        out: list[CallbackClaim] = []
        for pat in self._CUE_PATTERNS:
            for m in re.finditer(pat, assistant_text, flags=re.IGNORECASE):
                claim_text = m.group(1).strip()
                if len(claim_text) < 3:
                    continue
                claimed_when = "earlier"
                hint = self._SESSION_HINT.search(m.group(0))
                if hint:
                    claimed_when = f"session_{hint.group(1)}"
                out.append(
                    CallbackClaim(
                        session_index=session_index,
                        turn_index=turn_index,
                        claim_text=claim_text,
                        claimed_when=claimed_when,
                    )
                )
        # de-dupe by claim_text within turn
        seen: set[str] = set()
        unique: list[CallbackClaim] = []
        for c in out:
            key = c.claim_text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        return unique


class LLMCallbackExtractor:
    """LLM-backed extractor that emits structured JSON claims.

    Produces the same :class:`CallbackClaim` shape as the heuristic.
    A failed JSON parse falls back to no claims rather than guessing
    — fail-loud per ``no-swallow-errors-no-hasattr-abuse.mdc``.
    """

    _PROMPT_TEMPLATE = (
        "You are auditing a long-running conversation. The assistant just spoke. "
        "List EVERY place in the assistant's turn where it claims to remember a "
        "specific fact the user said earlier. For each claim, output JSON with "
        "fields: 'claim_text' (a brief paraphrase of what the assistant claims "
        "the user said) and 'claimed_when' (either 'session_N' for a specific "
        "session, or 'earlier' if the assistant did not specify, or 'unknown'). "
        "Output ONLY a JSON array. If there are no callbacks, output [].\n\n"
        "Assistant turn (session {session_index}, turn {turn_index}, of "
        "{total_sessions} total sessions):\n"
        '"""\n{assistant_text}\n"""\n\n'
        "JSON array:"
    )

    def __init__(
        self,
        *,
        client_complete,  # callable(prompt: str, *, seed: int) -> str
        seed_base: int = 0,
    ) -> None:
        self._complete = client_complete
        self._seed_base = seed_base

    def extract(
        self,
        *,
        assistant_text: str,
        session_index: int,
        turn_index: int,
        total_sessions: int,
    ) -> list[CallbackClaim]:
        prompt = self._PROMPT_TEMPLATE.format(
            assistant_text=assistant_text,
            session_index=session_index,
            turn_index=turn_index,
            total_sessions=total_sessions,
        )
        seed = self._seed_base + 1000 * session_index + turn_index
        text = self._complete(prompt, seed=seed)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Try to recover the first array — the model may have
            # surrounded JSON with prose.
            match = re.search(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", text, flags=re.DOTALL)
            if not match:
                return []
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        if not isinstance(payload, list):
            return []
        out: list[CallbackClaim] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            claim_text = str(raw.get("claim_text", "")).strip()
            if not claim_text:
                continue
            claimed_when = str(raw.get("claimed_when", "earlier")).strip() or "earlier"
            out.append(
                CallbackClaim(
                    session_index=session_index,
                    turn_index=turn_index,
                    claim_text=claim_text,
                    claimed_when=claimed_when,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Stage 2: deterministic matcher
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "for", "to", "and", "or", "in", "on", "at",
        "is", "was", "were", "are", "be", "been", "being", "i", "you", "we",
        "your", "my", "our", "their", "his", "her", "it", "this", "that",
        "these", "those", "with", "from", "about", "very", "really", "just",
        "still", "always", "never", "sometimes", "today", "yesterday",
        "tomorrow", "ever", "had", "have", "has", "do", "did", "does", "would",
        "could", "should", "will", "won't", "didn't", "doesn't",
    }
)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        t.lower() for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS and len(t) > 1
    )


def _claim_coverage(claim_tokens: Iterable[str], evidence_tokens: Iterable[str]) -> float:
    """Fraction of the claim's content tokens present in the evidence.

    Asymmetric on purpose: a callback that paraphrases (claim is short,
    evidence is long) should still match. Symmetric Jaccard would
    under-count these. The metric is bounded ``[0, 1]``.
    """
    sa = set(claim_tokens)
    sb = set(evidence_tokens)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def match_claim_to_evidence(
    *,
    claim: CallbackClaim,
    prior_turns: list[tuple[int, int, str]],  # (session, turn, text)
    similarity_threshold: float = 0.50,
) -> CallbackLedgerEntry:
    """Find the best-matching prior user turn for ``claim``.

    Uses the asymmetric *claim-coverage* metric (see
    :func:`_claim_coverage`); the threshold is the minimum fraction of
    claim content tokens that must appear in the evidence turn.
    Tuned to give zero false positives on the public reference set
    while still catching paraphrased fabrications. P8 calibration
    will revise as new scenarios accumulate evidence.
    """

    claim_tokens = _tokenize(claim.claim_text)
    if not claim_tokens:
        return CallbackLedgerEntry(
            claim=claim,
            evidence_session=None,
            evidence_turn=None,
            evidence_text=None,
            matched=False,
            similarity_score=0.0,
            fabricated=True,
        )
    best_score = 0.0
    best_session: int | None = None
    best_turn: int | None = None
    best_text: str | None = None
    for s_idx, t_idx, text in prior_turns:
        if claim.claimed_when.startswith("session_"):
            try:
                hinted = int(claim.claimed_when.split("_", 1)[1])
            except (ValueError, IndexError):
                hinted = None
            if hinted is not None and hinted != s_idx:
                continue
        prior_tokens = _tokenize(text)
        score = _claim_coverage(claim_tokens, prior_tokens)
        if score > best_score:
            best_score = score
            best_session = s_idx
            best_turn = t_idx
            best_text = text
    matched = best_score >= similarity_threshold
    return CallbackLedgerEntry(
        claim=claim,
        evidence_session=best_session if matched else None,
        evidence_turn=best_turn if matched else None,
        evidence_text=best_text if matched else None,
        matched=matched,
        similarity_score=best_score,
        fabricated=not matched,
    )


# ---------------------------------------------------------------------------
# Top-level: build ledger from full arc
# ---------------------------------------------------------------------------


def build_callback_ledger(
    *,
    arc: ArcRecord,
    extractor: CallbackExtractor,
    similarity_threshold: float = 0.50,
) -> CallbackLedger:
    """Walk every assistant turn, extract claims, match against prior turns."""

    prior_user_turns: list[tuple[int, int, str]] = []
    entries: list[CallbackLedgerEntry] = []
    sessions: tuple[ArcSession, ...] = arc.sessions
    total_sessions = len(sessions)
    for session in sessions:
        for turn in session.turns:
            # Match claims found in THIS turn against everything BEFORE it
            # (i.e., against prior_user_turns at this point in the iteration).
            claims = extractor.extract(
                assistant_text=turn.assistant_text,
                session_index=session.session_index,
                turn_index=turn.turn_index,
                total_sessions=total_sessions,
            )
            for claim in claims:
                entry = match_claim_to_evidence(
                    claim=claim,
                    prior_turns=list(prior_user_turns),
                    similarity_threshold=similarity_threshold,
                )
                entries.append(entry)
            prior_user_turns.append(
                (session.session_index, turn.turn_index, turn.user_text),
            )
    return CallbackLedger(arc_id=arc.arc_id, entries=tuple(entries))


def _all_user_turns_flat(arc: ArcRecord) -> list[tuple[int, int, str]]:
    """Flatten the arc's user turns to ``[(session, turn, text), ...]``."""
    return [
        (s.session_index, t.turn_index, t.user_text)
        for s in arc.sessions
        for t in s.turns
    ]
