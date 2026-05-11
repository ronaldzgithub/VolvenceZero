# Companion Bench Governance Charter — Draft v0.1

> Status: Draft
> RFC reference: [`companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md) §11
> Last updated: 2026-05-10

This document is the operational charter for the Companion Bench working group.
v1.0 reference implementation ships before the working group is
formed; this draft is the placeholder the chair-elect ratifies on
formation. **No single organisation has more than one voting seat.**

## 1. Purpose

Companion Bench exists to evaluate long-running companion AI systems on
dimensions that single-turn benchmarks cannot probe (continuity,
repair, personalization, long absence, boundary, goal drift). The
working group exists to:

* Maintain the methodology RFC and scenario set.
* Operate the public leaderboard.
* Govern held-out scenarios (private repo + quarterly rotation).
* Curate judge model selection (RFC §6.3, §8.1).
* Ratify scoring methodology changes (RFC §6.4).

## 2. Membership

The working group has 3–5 voting member organisations. Initial
formation criteria:

* Must publish or run companion-class evaluation work.
* Must commit budgeted compute to operate one quarterly held-out run.
* Must agree to the no-self-submission-during-chair-term constraint.

Member organisations propose one voting representative each; alternates
are allowed. No member may hold more than one voting seat.

## 3. Steering committee

Once five member organisations are seated, the working group elects a
three-person steering committee. Steering committee terms are 12
months; election is held annually.

## 4. Rotating chair

The chair rotates across member organisations on a 6-month cadence.
The chair has **no submission privileges** for the duration of their
term — this is the RFC §11 anti-self-favouring constraint.

The chair's responsibilities:

* Convene quarterly meetings.
* Sign off on quarterly held-out paraphrase rotation
  (`scripts/companion_bench/generate_heldout_seeds.py`).
* Maintain the working-group public log at
  `docs/external/lscb-working-group-log.md`.
* Forward submission flags from the verifier to the working group.

## 5. Public comment

Per RFC §11, all breaking changes to scoring methodology require a
public comment period of ≥ 4 weeks. A breaking change is one of:

* Axis weight adjustment of > 0.05 on any axis.
* Removal or addition of an axis.
* Change to the `score = clip(...)` aggregation formula.
* Change to the A6 hard-cap threshold or value.
* Change to the per-turn rubric criteria set.

Comment intake: GitHub issues on `VolvenceZero/companion-bench` with the
`comment-period` label. The chair compiles a comment summary that
the working group references at the next quarterly meeting.

## 6. Held-out governance

The held-out scenario set lives in a private repo (RFC §3 P3, §8.6).
Governance constraints:

* The held-out body never enters any public log, leaderboard payload,
  PR diff, or working-group meeting minutes (only hashes).
* Quarterly paraphrase-seed rotation is performed by the chair; a
  hash-only diff is published in
  `docs/external/companion-bench-heldout-rotation-log.md`.
* The deploy key is held by exactly one person at any time. On chair
  rotation the key is rotated; the outgoing chair attests to key
  destruction.

## 7. Judge model rotation

Per RFC §6.3 and §8.1, judge models are rotated quarterly to mitigate
family bias. The working group selects:

* The per-turn judge family.
* The arc judge family (must differ from per-turn).
* The deterministic-fake judge for CI smoke.

Rotation history is logged at
`docs/external/companion-bench-judge-rotation-log.md`. The history file is
public; the chair signs each rotation entry.

## 8. Conflict of interest

A member organisation must recuse from working-group decisions about:

* Their own submission's leaderboard placement.
* Held-out scenarios that probe their system's known weaknesses.
* Judge selection for the quarter when their submission is being
  scored.

Recusal is logged in the meeting minutes.

## 9. Funding

Companion Bench is funded by member organisations' budgeted compute donations.
The working group publishes annual cost transparency at
`docs/external/lscb-funding-summary.md`. No commercial sponsorship
of leaderboard placement is permitted.

## 10. Amendment process

Amendments to this charter require:

1. Written proposal by any voting member.
2. Public comment period ≥ 2 weeks.
3. 4/5 majority vote (or unanimous if < 5 members).
4. Sign-off by the rotating chair.

## 11. Chair-elect candidates

> This section is filled in at working-group formation. v1.0 ships
> with this section blank.

| Candidate | Affiliation | Term proposed |
|---|---|---|
| _to be filled_ | _to be filled_ | _to be filled_ |
