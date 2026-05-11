# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Pairwise Elo rating: TrueSkill + Bradley-Terry MLE (RFC §6.5).

Per the RFC we report two pairwise rankings alongside the absolute
Companion Bench score:

* **TrueSkill**: probabilistic skill rating that handles uncertainty
  well in small-sample regimes (matches the EQ-Bench v3 pairing track).
* **Bradley-Terry MLE**: classical paired-comparison MLE, easier to
  compare across benchmarks because everyone implements it the same way.

A "match" in Companion Bench is a comparison of two systems on the same arc. The
winner is whichever has the higher final Companion Bench score on that arc; if
the two are within the documented ``tie_threshold`` (default 0.5 Companion Bench
points) we score it as a tie.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Iterable

import trueskill


@dataclasses.dataclass(frozen=True)
class PairwiseMatch:
    """One per-arc head-to-head between two systems."""

    arc_id: str
    system_a: str
    system_b: str
    score_a: float
    score_b: float


@dataclasses.dataclass(frozen=True)
class TrueSkillRating:
    system: str
    mu: float
    sigma: float
    conservative: float  # mu - 3*sigma, the leaderboard-cited number


@dataclasses.dataclass(frozen=True)
class BradleyTerryRating:
    system: str
    score: float
    rank: int


# ---------------------------------------------------------------------------
# Match construction
# ---------------------------------------------------------------------------


def derive_matches_from_arc_scores(
    *,
    by_arc: dict[str, dict[str, float]],
    tie_threshold: float = 0.5,
) -> tuple[PairwiseMatch, ...]:
    """Build a PairwiseMatch list from ``arc_id → {system: score}``.

    Every pair of systems that ran the same arc generates one match
    (head-to-head). Order within each pair is arbitrary but stable
    (sorted by system name).
    """
    matches: list[PairwiseMatch] = []
    for arc_id, score_by_system in by_arc.items():
        systems = sorted(score_by_system.keys())
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                a, b = systems[i], systems[j]
                matches.append(
                    PairwiseMatch(
                        arc_id=arc_id,
                        system_a=a,
                        system_b=b,
                        score_a=score_by_system[a],
                        score_b=score_by_system[b],
                    )
                )
    return tuple(matches)


# ---------------------------------------------------------------------------
# TrueSkill
# ---------------------------------------------------------------------------


def compute_trueskill(
    matches: Iterable[PairwiseMatch],
    *,
    tie_threshold: float = 0.5,
) -> tuple[TrueSkillRating, ...]:
    """Run TrueSkill across the match set and return per-system ratings."""

    env = trueskill.TrueSkill(draw_probability=0.10)
    env.make_as_global()
    ratings: dict[str, trueskill.Rating] = {}

    def _r(name: str) -> trueskill.Rating:
        if name not in ratings:
            ratings[name] = env.create_rating()
        return ratings[name]

    for m in matches:
        ra = _r(m.system_a)
        rb = _r(m.system_b)
        diff = m.score_a - m.score_b
        if abs(diff) <= tie_threshold:
            new_a, new_b = trueskill.rate_1vs1(ra, rb, drawn=True)
        elif diff > 0:
            new_a, new_b = trueskill.rate_1vs1(ra, rb, drawn=False)
        else:
            new_b, new_a = trueskill.rate_1vs1(rb, ra, drawn=False)
        ratings[m.system_a] = new_a
        ratings[m.system_b] = new_b

    out = [
        TrueSkillRating(
            system=name,
            mu=r.mu,
            sigma=r.sigma,
            conservative=r.mu - 3.0 * r.sigma,
        )
        for name, r in ratings.items()
    ]
    out.sort(key=lambda x: x.conservative, reverse=True)
    return tuple(out)


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


def compute_bradley_terry(
    matches: Iterable[PairwiseMatch],
    *,
    tie_threshold: float = 0.5,
    max_iter: int = 500,
    tolerance: float = 1e-6,
) -> tuple[BradleyTerryRating, ...]:
    """MLE Bradley-Terry by iterative reweighting (Hunter 2004).

    Returns ratings on the log-odds scale, normalised so the median
    rated system is 0.
    """

    wins: dict[tuple[str, str], float] = defaultdict(float)
    systems_set: set[str] = set()
    for m in matches:
        systems_set.update([m.system_a, m.system_b])
        diff = m.score_a - m.score_b
        if abs(diff) <= tie_threshold:
            wins[(m.system_a, m.system_b)] += 0.5
            wins[(m.system_b, m.system_a)] += 0.5
        elif diff > 0:
            wins[(m.system_a, m.system_b)] += 1.0
        else:
            wins[(m.system_b, m.system_a)] += 1.0

    systems = sorted(systems_set)
    if not systems:
        return ()
    n = len(systems)
    idx = {s: i for i, s in enumerate(systems)}
    # Initial ratings
    pi = [1.0] * n
    win_total = [0.0] * n
    for (a, b), w in wins.items():
        win_total[idx[a]] += w
    matchups: dict[tuple[int, int], float] = defaultdict(float)
    for (a, b), w in wins.items():
        matchups[(idx[a], idx[b])] += w
    matchup_pairs = sorted({tuple(sorted(p)) for p in matchups.keys()})

    for _ in range(max_iter):
        new_pi = [0.0] * n
        for i in range(n):
            denom = 0.0
            for (a_idx, b_idx) in matchup_pairs:
                if i not in (a_idx, b_idx):
                    continue
                other = b_idx if i == a_idx else a_idx
                games = matchups[(a_idx, b_idx)] + matchups[(b_idx, a_idx)]
                if games == 0:
                    continue
                denom += games / (pi[i] + pi[other])
            new_pi[i] = win_total[i] / denom if denom > 0 else pi[i]
        # Normalise to keep numbers stable
        avg = sum(new_pi) / n if n > 0 else 1.0
        if avg > 0:
            new_pi = [x / avg for x in new_pi]
        delta = max(abs(new_pi[i] - pi[i]) for i in range(n))
        pi = new_pi
        if delta < tolerance:
            break

    log_pi = [math.log(max(1e-9, x)) for x in pi]
    median_log = sorted(log_pi)[n // 2]
    out = [
        BradleyTerryRating(
            system=systems[i],
            score=log_pi[i] - median_log,
            rank=0,  # filled below
        )
        for i in range(n)
    ]
    out.sort(key=lambda x: x.score, reverse=True)
    return tuple(
        BradleyTerryRating(system=r.system, score=r.score, rank=i + 1)
        for i, r in enumerate(out)
    )


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EloReport:
    trueskill: tuple[TrueSkillRating, ...]
    bradley_terry: tuple[BradleyTerryRating, ...]

    def to_json(self) -> dict:
        return {
            "trueskill": [
                {"system": r.system, "mu": r.mu, "sigma": r.sigma,
                 "conservative": r.conservative}
                for r in self.trueskill
            ],
            "bradley_terry": [
                {"system": r.system, "score": r.score, "rank": r.rank}
                for r in self.bradley_terry
            ],
        }


def build_elo_report(
    *,
    by_arc: dict[str, dict[str, float]],
    tie_threshold: float = 0.5,
) -> EloReport:
    matches = derive_matches_from_arc_scores(by_arc=by_arc, tie_threshold=tie_threshold)
    return EloReport(
        trueskill=compute_trueskill(matches, tie_threshold=tie_threshold),
        bradley_terry=compute_bradley_terry(matches, tie_threshold=tie_threshold),
    )
